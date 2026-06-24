# 第 11 章 路由、汇合与分支控制流

第 10 章我们把生成器接到了管线末端，得到了一条"嵌入 → 检索 → 排序 → 提示 → 生成"的笔直流水线。但真实的 RAG 系统从来不是一条直线：上传的文件可能是 PDF、也可能是 mp3，需要送进不同的转换器；同一个查询要同时跑 BM25 和向量检索，两路结果必须合成一份；LLM 输出的 JSON 校验失败，要回炉重生成而不是崩掉。这些都需要在数据流图上**分叉**与**汇合**。

Haystack 的图是静态的——`Pipeline` 在构建期就把所有组件和边固定下来（第 3 章）。它没有 `if`、没有 `for`、没有 `try`。那么条件分支、循环、map-reduce 这些控制流从哪来？答案是：**用数据流动态地激活/跳过边来表达**。Router 在运行时只往一部分 output socket 写值，下游连着的边因此"不触发"，等价于 `if/else`；Joiner 用 `Variadic` 输入把多条边收成一条，等价于扇入汇合；BranchJoiner 把校验失败的边接回上游入口，等价于 `while` 重试。本章逐一拆解 `haystack/components/routers/`、`joiners/`、`validators/` 三个目录的真实实现，讲清"图是死的，路由让它活"的设计。

## 11.1 ConditionalRouter：用 Jinja2 表达式在运行时选路

`conditional_router.py` 里的 `ConditionalRouter` 是最通用的分支组件。它的核心是一个 `routes` 列表，每个元素是一个 `Route`（`TypedDict`），由四个必填字段加一个可选字段构成：

```python
class Route(TypedDict):
    condition: str
    output: str | list[str]
    output_name: str | list[str]
    output_type: type | list[type]
    output_passthrough: NotRequired[bool]
```

- `condition` 是一个 Jinja2 字符串表达式，渲染后必须能被 `ast.literal_eval` 解析成布尔值。
- `output` 是另一个 Jinja2 表达式，决定这条路输出什么值。
- `output_name` 是这个输出在管线里的"端口名"，下游组件就连到这个名字。
- `output_type` 声明输出类型，用于建立连接时的类型检查。

**动态生成 output socket** 是这个组件最关键的机制。

- `__init__` 遍历 `routes`，把每条路的 `output_name → output_type` 收集到 `output_types` 字典里，最后调用 `component.set_output_types(self, **output_types)`（`conditional_router.py:328`）。
- 这意味着 socket 不是写死在类上的，而是**由用户传入的路由表在实例化时算出来的**——`enough_streams`、`insufficient_streams` 这种端口名都是运行配置决定的。
- 输入端同理：`_extract_variables` 用 Jinja2 环境解析 `condition` 和 `output` 模板，提取出所有变量名（`conditional_router.py:516`），减去 `optional_variables` 后用 `component.set_input_types(self, **dict.fromkeys(mandatory_input_types, Any))` 注册成必填输入。

**运行期的选路逻辑** 在 `run` 里（`conditional_router.py:404`）：

- 按 `routes` 列表顺序逐条评估 `condition`，渲染后做 `ast.literal_eval`，第一个为真的路被选中，渲染它的 `output` 并以 `{output_name: output_value}` 形式返回，然后立刻 `return`——后面的路不再看。这正是 `if/elif/else` 的语义：顺序短路、只走一条。
- 返回的字典只含被选中路的端口，其余 output socket 没有值，于是连在那些端口上的下游边就"不流动"，整条分支被跳过。
- 如果所有 `condition` 都为假，抛出 `NoRouteSelectedException`；条件本身解析出错则包成 `RouteConditionException`。

几个值得注意的设计细节：

- **沙箱**：默认用 Jinja2 的 `SandboxedEnvironment` 渲染条件，防止模板里执行任意代码；只有显式传 `unsafe=True` 才换成 `NativeEnvironment` 并打印警告（`conditional_router.py:271`、`278`）。这是把"模板可能来自不可信来源"当成安全边界来处理 [[Jinja2 Sandbox]](https://jinja.palletsprojects.com/).
- **fallback 路由**：把最后一条路的 `condition` 写成 `"{{ True }}"`，配合 `optional_variables` 把某些变量标成可选（缺失时填 `None`），就能实现"没命中任何特定条件就走默认路"的兜底模式（文档示例在 `conditional_router.py:226`）。
- **output_passthrough**：默认 `output` 走 Jinja2 渲染，但 Jinja2 会把对象转成字符串——`ParsedQuery(...)` 这类 dataclass 会被毁成它的 `repr`。设 `output_passthrough=True` 后，`output` 被当成纯变量名，直接从 `kwargs` 取值原样透传（`conditional_router.py:425`），从而保住复杂类型的身份（文档里甚至断言 `result["search_query"] is query`）。
- **类型校验**：`validate_output_type=True` 时，`run` 会用 `_output_matches_type` 递归检查输出值是否匹配声明类型（`conditional_router.py:447`、`549`），不匹配抛 `ValueError`。这个方法本身值得一读：它用 `get_origin`/`get_args` 拆解类型，对 `Sequence` 逐元素递归、对 `Mapping` 逐键值递归、对 Union（含 `X | Y` 写法）用 `any(...)`，空序列/空映射一律放行（`conditional_router.py:566`、`578`、`592`）——这是一套手写的运行期类型匹配器，因为 Python 的 `isinstance` 处理不了 `list[int]` 这种带参泛型。

**多输出与序列化**也要留意。一条路的 `output`/`output_name`/`output_type` 都允许传成等长列表，`run` 里用 `zip(..., strict=True)` 配对，从而一条 `condition` 命中后**同时点亮多个端口**（`conditional_router.py:424`）。`_validate_routes` 在 `__init__` 期就强制三者等长、且 `condition`、`output`（非 passthrough 时）必须是合法 Jinja2 模板（`conditional_router.py:488`、`492`）。序列化方面，`to_dict` 会把每个 `output_type` 用 `serialize_type` 转成字符串、把 `custom_filters` 用 `serialize_callable` 存下来（`conditional_router.py:340`、`345`），`from_dict` 再逐一 `deserialize_type`/`deserialize_callable` 还原——所以连路由表带的自定义 Jinja2 过滤器都能进 YAML（第 5 章）。

## 11.2 专用 Router：MIME、元数据、语言、LLM 分类

`ConditionalRouter` 足够通用，但常见分发场景有更省心的专用组件，它们的共同套路是：**把"分类标签"直接做成 output socket 的名字**，再在 `run` 里把输入按标签分桶。

`FileTypeRouter`（`file_type_router.py`）按 MIME 类型分发文件或 `ByteStream`。`__init__` 把每个 `mime_type` 编译成正则（支持 `audio/.*` 这样的模式），并调用 `set_output_types` 时用 `**dict.fromkeys(mime_types, ...)` 把每个 MIME 字符串变成一个 output 端口，外加固定的 `unclassified` 和 `failed` 两个端口（`file_type_router.py:97`）。`run` 里对每个源用 `_guess_mime_type` 推断类型，`pattern.fullmatch` 命中第一个就归到对应桶，没命中进 `unclassified`，文件不存在则进 `failed`（除非 `raise_on_failure=True`）。一个易被忽略的细节：当传入 `meta` 时，源会先被 `get_bytestream_from_source` 转成 `ByteStream` 并写入元数据，再做分类，方便下游转换器拿到附带信息。这是典型的"扇出"：一个输入列表被劈成多路，分别接不同的转换器，正好衔接第 6 章的多格式转换。

`MetadataRouter`（`metadata_router.py`）按文档元数据的过滤规则分发。`rules` 是 `{边名: 过滤表达式}`，`run` 里对每个文档用 `document_matches_filter`（复用第 7 章 DocumentStore 的同一套过滤语义）逐条匹配，命中就追加到对应边。注意它和 `FileTypeRouter` 的一个语义差异：`MetadataRouter` 允许一个文档**同时命中多条规则**（`current_obj_matched` 标记只要有一条命中就为真，但不 `break`），即可以"广播"到多个分支；只有一条都不命中才进 `unmatched`。

`TextLanguageRouter`（`text_language_router.py`）用 `langdetect` 检测语言，把文本送到对应语言的端口，检测不出或不在 `languages` 列表里则进 `unmatched`。注意它在 `__init__` 里已发出 `FutureWarning`，将在 3.0 移到 `langdetect-haystack` 包。`DocumentTypeRouter` 与 `FileTypeRouter` 几乎同构，区别是 MIME 来源可以是文档 `meta` 字段或 `file_path` 字段。

`LLMMessagesRouter`（`llm_messages_router.py`）更激进：它内嵌一个 `ChatGenerator`，让 LLM 自己对消息做分类（典型用于 Llama Guard 这类内容审核）。`__init__` 要求 `output_names` 与 `output_patterns` 等长非空，把每个 pattern 预编译成正则，并用 `set_output_types` 注册 `output_names + ["unmatched"]` 加一个调试用的 `chat_generator_text` 端口（`llm_messages_router.py:88`）。`run` 里先校验消息只含 user/assistant 角色，可选地前置 `system_prompt`，调内嵌生成器拿到回复文本，然后按 `output_patterns` 顺序 `pattern.search`，命中第一个就把原消息放到对应端口并 `break`，全不命中走 `for...else` 落到 `unmatched`（`llm_messages_router.py:135`）。它还实现了 `warm_up`，会把底层生成器的 `warm_up` 一并触发——把"判断走哪条边"这件事本身交给了模型。

这些专用 Router 验证了同一个设计哲学：**output socket 是动态的、由配置决定的，路由就是"只点亮其中一部分端口"**。

## 11.3 DocumentJoiner：多路文档汇合与四种 JoinMode

分叉之后要汇合。`document_joiner.py` 的 `DocumentJoiner` 是 RAG 里最常用的扇入组件——把多个检索器的结果合成一份。它的输入声明是 `documents: Variadic[list[Document]]`（`document_joiner.py:137`）。

**`Variadic` 是汇合的类型学基础**。

- 在 `core/component/types.py` 里，`Variadic` 是 `Annotated[Iterable[T], HAYSTACK_VARIADIC_ANNOTATION]`——一个带特殊标记的注解（`types.py:23`）。
- `InputSocket.__post_init__` 检测到这个标记后把 `is_lazy_variadic` 置真（`types.py:81`），管线据此允许**多条边连到同一个输入端口**，并在运行时把它们各自的值打包成一个列表传进来。所以 `run` 里第一行 `documents = list(documents)` 拿到的是"列表的列表"。
- `__post_init__` 还会"解包"两层 `get_args`，把 `Variadic[list[Document]]` 还原成内层的 `list[Document]` 供连接期做类型匹配（`types.py:101`）；如果忘了写类型参数（裸 `Variadic`）会直接抛 `ComponentError`。
- `Variadic`（lazy）会等所有上游都到齐才触发；与之对应的 `GreedyVariadic`（`types.py:31`）则收到第一个输入就立刻运行——这个区别在 11.5 节讲循环时至关重要。

四种 `JoinMode`（`document_joiner.py:19`）对应四种合并算法，在 `__init__` 里用字典映射到具体方法：

- **`concatenate`**（默认，`_concatenate`，`document_joiner.py:170`）：按 `doc.id` 去重，重复文档只保留分数最高的那个（`max(docs, key=...)`，`score=None` 当作 `-inf`）。
- **`merge`**（`_merge`，`document_joiner.py:184`）：对重复文档做加权求和。`weights` 在 `__init__` 里被归一化（除以总和，`document_joiner.py:130`），缺省则平均分。`scores_map[doc.id] += score * weight`，最后用 `dataclasses.replace` 写回新分数。
- **`reciprocal_rank_fusion`**（`_rrf` → `_reciprocal_rank_fusion`）：见下文。
- **`distribution_based_rank_fusion`**（`_distribution_based_rank_fusion`，`document_joiner.py:209`）：对每一路先做基于分布的归一化——算均值和标准差，用 `[mean-3σ, mean+3σ]` 作为缩放区间把分数压到 `[0,1]`，再走 `_concatenate` 去重。当一路内所有分数相同（`delta_score==0`）时判定该路对查询无信息量，分数置 0。

**RRF 公式**在 `utils/misc.py:156` 的 `_reciprocal_rank_fusion` 里。常量 `k = 61`——注释说明原论文建议 60，加 1 是因为 Python 列表 0-based 而论文用 1-based 排名（`misc.py:164`）。核心累加是 `scores_map[doc.id] += (weight * len(document_lists)) / (k + rank)`，即每个文档的贡献只取决于它在各路里的**排名 `rank`**而非原始分数，名次越靠前贡献越大，再跨路求和。这正是 Reciprocal Rank Fusion 的思想：不同检索器分数量纲不可比，但排名可比，用 `1/(k+rank)` 把排名转成可加的分数 [[Reciprocal Rank Fusion]](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). 这也解释了为什么 RRF 和 `merge` 都吃 `weights` 而 `concatenate`、`distribution_based_rank_fusion` 不吃。

合并完成后，`run` 统一做两件收尾（`document_joiner.py:153`）：若 `sort_by_score=True`（默认），按分数降序排序，`score=None` 当 `-inf`；若传了 `top_k`（run 参数优先于实例 `top_k`），截断到前 `k` 个。

## 11.4 其他 Joiner：Answer、List、String

汇合不止于文档。`AnswerJoiner`（`answer_joiner.py`）合并多个生成器产出的 `GeneratedAnswer | ExtractedAnswer` 列表，目前只有 `CONCATENATE` 一种模式，即 `itertools.chain.from_iterable` 拍平；同样支持 `sort_by_score`（默认 `False`）和 `top_k`。`ListJoiner`（`list_joiner.py`）更原始，把多个同型列表 `chain` 成一个扁平列表，构造时可传 `list_type_` 约束元素类型（如 `list[ChatMessage]`），不传则当 `list[Any]`——它常用在 agent 里把历史消息、工具结果、反馈累积成一条对话记录。`StringJoiner`（`string_joiner.py`）最简单，把多路 `str` 收成 `list[str]`。

这四个 Joiner 全部用 `Variadic` 输入，差别只在合并逻辑和类型声明。它们共同回答了"map 之后怎么 reduce"：Router 扇出（map）、各分支独立处理、Joiner 扇入（reduce），这就是在静态图上表达 map-reduce 的方式。

## 11.5 BranchJoiner 与校验回路：在循环里汇流

最精彩的控制流是**循环**。`branch.py` 的 `BranchJoiner` 看似是个退化的 Joiner——它接收多路同型输入，只把"第一个收到的值"原样转发（`run` 里 `kwargs["value"][0]`，且断言只能有一个值，`branch.py:127`）。它的妙处全在输入类型：`component.set_input_types(self, value=GreedyVariadic[type_])`（`branch.py:95`）。

用 `GreedyVariadic` 而非 `Variadic` 是循环能成立的关键。`GreedyVariadic` 意味着**只要任意一条边送来值就立刻运行**，不等其他边。设想一个重试回路：管线入口 → `BranchJoiner` → 生成器 → 校验器，校验器的"失败"边又接回 `BranchJoiner`。第一次运行时，入口的值到达 `BranchJoiner`，它贪婪地立刻转发给生成器；若校验失败，失败边把修正提示送回 `BranchJoiner`，它又立刻转发触发重新生成。如果用 lazy `Variadic`，`BranchJoiner` 会傻等"两条边都到齐"——但回路里第二条边只有在第一轮跑完后才可能有值，永远死锁。`GreedyVariadic` 让它"来一个转一个"，于是闭环得以驱动。这就是为什么文档说 `BranchJoiner` 用来"关闭管线里的循环"（`branch.py:21`）。

`BranchJoiner(type_)` 一次只管一种数据类型，`type_` 同时决定它接收和发出的类型（`branch.py:96`），保证回路两端类型一致。

**校验失败如何走 error 分支**由 `validators/json_schema.py` 的 `JsonSchemaValidator` 给出范本。它声明了两个 output socket：`validated` 和 `validation_error`（`json_schema.py:112`）。`run` 取最后一条消息，先用 `is_valid_json` 判断是否合法 JSON，再用 `jsonschema.validate` 校验结构：

- 通过 → 返回 `{"validated": [last_message]}`，只点亮 `validated` 端口；
- 不通过 → 捕获 `ValidationError`，用 `default_error_template` 拼出一段**给 LLM 看的修正提示**（含失败的 JSON、错误信息、JSON Path、schema），返回 `{"validation_error": [ChatMessage.from_user(recovery_prompt)]}`，只点亮 `validation_error` 端口（`json_schema.py:172`）。

注意这里**没有抛异常**——校验失败被建模成"走另一条边"而非"崩溃"。把 `validation_error` 边接回 `BranchJoiner`，再接回生成器，就构成了一个自动重试回路：LLM 输出不合规 → 校验器生成修正提示 → 回灌生成器重新生成，直到通过或管线达到运行上限。`default_error_template` 末尾甚至明确要求 LLM"只输出修正后的原始 JSON、不要 markdown、不要注释"（`json_schema.py:97`），可见这是专为机器自愈设计的。

## 11.6 为什么这样设计：静态图上的动态控制流

把本章组件放回第 4 章的执行引擎来看，设计意图就清晰了。管线是一张固定的有向图，调度器按拓扑顺序运行组件、按边传递数据。Router 和 Joiner 不改变图的结构，只改变**哪些边在本次运行中真正有数据流过**：

- Router 在运行时只往部分 output socket 写值 → 部分边"熄灭" → `if/else` 分支。
- Joiner 用 `Variadic` 让多条边汇入一个端口 → 扇入汇合 → map-reduce 的 reduce。
- `GreedyVariadic` 的 `BranchJoiner` + 回灌边 → `while` 循环 / 重试。
- Validator 把"错误"建模成一条普通输出边 → 错误处理成为数据流的一部分，而非控制流异常。

这种"用数据流的有无来表达控制流"的做法，好处是序列化友好（整张图连同路由表都能存进 YAML，第 5 章）、可视化清晰（分支汇合一眼可见）、且组件之间彻底解耦——Router 不知道下游是谁，只管点亮端口。代价是表达 `if`/`loop` 不如写代码直观，需要理解 socket 的动态生成和 `Variadic` 的语义。下一章的 Tool 与 Agent 会进一步把"模型决定走哪条边"这件事推到极致。

## 本章小结

- Haystack 的管线图是静态的，条件分支、汇合、循环都通过**运行时点亮/熄灭边**来表达，而非语言级的 `if`/`for`。
- `ConditionalRouter` 用 `routes` 列表（`condition`/`output`/`output_name`/`output_type`）驱动选路，按顺序短路评估、只走第一条命中的路，并在 `__init__` 里**动态生成 input/output socket**。
- 条件渲染默认走 Jinja2 `SandboxedEnvironment`，`unsafe=True` 才换成 `NativeEnvironment`；`output_passthrough=True` 可绕过 Jinja2 直接透传复杂对象。
- 专用 Router（`FileTypeRouter`、`MetadataRouter`、`TextLanguageRouter`、`DocumentTypeRouter`、`LLMMessagesRouter`）的共性是把分类标签做成 output 端口名，`run` 时按标签分桶，未命中归 `unclassified`/`unmatched`。
- `DocumentJoiner` 用 `Variadic[list[Document]]` 接收多路结果，提供 `concatenate`/`merge`/`reciprocal_rank_fusion`/`distribution_based_rank_fusion` 四种合并算法，统一做 `sort_by_score` 和 `top_k` 收尾。
- RRF 在 `_reciprocal_rank_fusion` 里以常量 `k=61`、按排名（而非原始分数）累加 `1/(k+rank)`，解决不同检索器分数不可比的问题。
- `Variadic`（lazy，等齐）与 `GreedyVariadic`（来一个跑一个）的区别，决定了 Joiner 是用于扇入汇合还是用于闭合循环。
- `BranchJoiner` 用 `GreedyVariadic` 转发首个到达的值，是把校验失败边/重试边接回上游、实现 agent loop 的关键枢纽。
- `JsonSchemaValidator` 把校验结果分流到 `validated` 与 `validation_error` 两个端口，失败时生成给 LLM 的修正提示而非抛异常，从而支持自动重生成回路。

## 动手实验

1. **实验一：观察动态 socket 与短路选路** — 用一个 `count` 输入构造两条路（`{{count > 5}}` → `many`、`{{count <= 5}}` → `few`），实例化 `ConditionalRouter` 后打印 `router.__haystack_output__`/`__haystack_input__`（或在管线里看连接），确认 `many`/`few` 是由 routes 算出来的；分别用 `count=10` 和 `count=3` 调 `run`，验证返回字典只含被选中的那个端口。
2. **实验二：对比四种 JoinMode** — 造两路带 `score` 的 `Document` 列表（含一两个 `id` 重复的文档），分别用 `concatenate`、`merge`（带 `weights=[0.7, 0.3]`）、`reciprocal_rank_fusion`、`distribution_based_rank_fusion` 实例化 `DocumentJoiner` 并 `run`，对照输出分数，手算一个文档的 RRF 分数 `(weight*L)/(61+rank)` 验证与代码一致。
3. **实验三：搭一个校验重试回路** — 用 `BranchJoiner(list[ChatMessage])` + 一个会先返回坏 JSON 再返回好 JSON 的假生成器 + `JsonSchemaValidator`，把 `validator.validation_error` 连回 `BranchJoiner`，运行后断言最终走到 `validated` 端口，并观察 `validation_error` 里生成的修正提示文本结构。
4. **实验四：把 `Variadic` 换成普通输入看会怎样** — 写一个最小 Joiner，先用 `Variadic[list[int]]` 让两条边连进同一端口并成功汇合；再把类型改成普通 `list[list[int]]`，尝试连第二条边到同一端口，观察 `Pipeline.connect` 是否报错，从而理解 `Variadic` 标记对"多边入一口"的必要性。

> **下一章预告**：第 12 章 工具系统 Tool 与 Toolset。我们将从"数据怎么分叉汇合"转向"模型怎么调用外部能力"——拆解 `Tool` 如何把一个 Python 函数包装成带 JSON Schema 的可调用工具、`Toolset` 如何聚合多个工具，以及生成器拿到工具结果后如何回流，为第 13 章的 Agent 闭环铺路。
