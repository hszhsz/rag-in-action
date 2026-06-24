# 第 12 章 工具系统 Tool 与 Toolset

第 11 章我们讲清了 Pipeline 如何用 `Router`、`BranchJoiner`、`ConditionalRouter` 把数据在静态图里来回拨动。但 Agent 的世界要走得更远一步：让大模型自己决定"现在该调哪个工具、传什么参数"。要做到这一点，框架必须先把"工具"抽象成一种大模型能读懂、又能被代码执行的统一形态——这正是 `haystack/tools/` 目录的职责。

本章逐文件解构 Haystack 的工具系统：核心契约 `Tool`（`tool.py`），从函数反射 schema 的 `create_tool_from_function` 与 `@tool`（`from_function.py`），把组件包成工具的 `ComponentTool`（`component_tool.py`）与把整条 Pipeline 包成工具的 `PipelineTool`（`pipeline_tool.py`），聚合去重的 `Toolset`（`toolset.py`），按需检索的大工具集 `SearchableToolset`（`searchable_toolset.py`），以及工具间共享数据的 `State`（`state.py` / `state_utils.py`）。

## 12.1 统一契约：`Tool` dataclass

`tool.py` 里的 `Tool` 是整个体系的地基。它是一个 `@dataclass`，字段恰好覆盖了 LLM function-calling 协议所需的全部信息，再加上一个可调用对象：

```python
name: str
description: str
parameters: dict[str, Any]
function: Callable
outputs_to_string: dict[str, Any] | None = None
inputs_from_state: dict[str, str] | None = None
outputs_to_state: dict[str, dict[str, Any]] | None = None
```

前三个字段（`name`/`description`/`parameters`）正是大模型准备一次调用所需的"工具规格"：名字、用途描述、以及一份描述入参的 JSON Schema [[Function calling]](https://platform.openai.com/docs/guides/function-calling)。`function` 则是真正被执行的同步可调用对象。设计的核心思想是**契约与实现解耦**：Agent 只面对 `Tool` 这个统一形态，至于背后是个 lambda、一个组件还是一整条 Pipeline，Agent 完全不关心。

`__post_init__` 在构造时做了几道关键校验，把错误前移到"工具定义时"而非"调用时"：

- 用 `inspect.iscoroutinefunction(self.function)` 拒绝 async 函数，明确"工具函数必须是同步的"。
- 用 `Draft202012Validator.check_schema(self.parameters)` 校验 `parameters` 是合法的 JSON Schema（Draft 2020-12）[[JSON Schema]](https://json-schema.org/)，不合法直接抛 `ValueError`。
- 逐项校验 `outputs_to_state`/`outputs_to_string`/`inputs_from_state` 的结构（`source` 必须是 str、`handler` 必须 callable 等）。

校验里还有一个值得注意的细节：`inputs_from_state` 与 `outputs_to_state` 引用的参数名/输出名会被对照"有效集合"检查。`_get_valid_inputs()` 默认通过 `inspect.signature(self.function)` 反射出函数全部参数名，再并上 schema 里的 `properties`；`_get_valid_outputs()` 默认返回 `None`（普通函数没有正式输出 schema，于是跳过输出校验）。子类 `ComponentTool` 会重写这两个方法，改用组件的 input/output socket 名集合——这是面向对象的多态在工具校验上的精妙运用。

执行端 `invoke(**kwargs)` 极简：调用 `self.function(**kwargs)`，任何异常都被包装成带 `tool_name` 的 `ToolInvocationError`（定义在 `errors.py`），让上层能定位是哪个工具崩了。另外有一个 `tool_spec` 属性，专门返回 `{"name", "description", "parameters"}` 三元组喂给大模型；以及一个默认空实现的 `warm_up()`，供需要建连接、加载模型的工具重写（且要求幂等，因为可能被多次调用）。

序列化方面，`to_dict()` 用 `asdict(self)` 拍平字段，再把 `function` 用 `serialize_callable` 转成可导入的字符串路径，把带 `handler` 的 `outputs_to_state`/`outputs_to_string` 也逐一序列化；`from_dict()` 反向用 `deserialize_callable` 还原。最后封装成 `{"type": <qualified_class_name>, "data": {...}}`——`type` 字段让反序列化时能找到正确的（可能是子类的）类。

## 12.2 从函数反射 schema：`create_tool_from_function` 与 `@tool`

手写一份 JSON Schema 既繁琐又容易和函数签名脱节。`from_function.py` 的 `create_tool_from_function` 用 Python 的类型注解 + docstring 自动生成它，核心思路是"借道 Pydantic"：

1. 描述（`description`）默认取自 `function.__doc__`，可显式覆盖（传空串可故意留空）。
2. 用 `inspect.signature(function)` 遍历参数，跳过三类：在 `inputs_from_state.values()` 里的（运行时从 State 注入）、类型是 `State`（含 `Optional[State]`，由 `_unwrap_optional` 判定）的、以及 `_contains_callable_type` 判定含 `Callable` 的（Pydantic 无法为其生成 JSON Schema）。
3. 缺类型注解的参数直接抛 `ValueError`——强制要求类型提示。
4. 无默认值的参数用 Ellipsis（`...`）标记为"必填"，这是 Pydantic 表达 required 的约定。
5. 若参数用 `typing.Annotated` 标注，其 `__metadata__[0]` 被取作该参数的描述。
6. 用 `pydantic.create_model(function.__name__, **fields)` 动态建模型，再 `model.model_json_schema()` 拿到 schema。

生成后还有两步收尾：`_remove_title_from_schema` 递归删掉 Pydantic 自动塞进来的冗余 `title`（注释里特意指出 Pydantic 没有编程开关能关掉它，只能事后清理，并附了 issue 链接）；再把 `Annotated` 抽出的描述写回 `schema["properties"][param]["description"]`。

`@tool` 装饰器是它的语法糖，通过两个 `@overload` 同时支持"裸用"和"带参数用"：

```python
@tool                       # 无参
def f(): ...

@tool(name="custom_name")   # 带参
def f(): ...
```

实现上，`tool(function=None)` 时返回一个 `decorator` 闭包（带参用法），否则直接 `decorator(function)`（裸用）。两条路径最终都汇入 `create_tool_from_function`。这种"一个函数 + 类型注解 = 一个 LLM 工具"的体验，是工具系统最低门槛的入口。

## 12.3 组件即工具：`ComponentTool`

Haystack 的杀手锏是已有的庞大组件生态（检索器、网络搜索、嵌入器……）。`ComponentTool` 让这些 `@component` 无需改造就能当工具用，复用既有生态而非另起炉灶。

它继承 `Tool`，构造函数先做两道守卫：非 `Component` 实例抛 `TypeError`；已经被加进某条 Pipeline 的组件（`__haystack_added_to_pipeline__` 为真）抛 `ValueError`——因为工具应当是独立可调用的，不能和别的图共享状态。

schema 来自组件的 input socket。`_create_tool_parameters_schema` 遍历 `component.__haystack_input__._sockets_dict`，对每个 socket：

- 跳过 `inputs_from_state` 映射走的、`Callable`、以及 `State` 类型的输入（与 `from_function` 同一套规则）。
- 描述优先取自 `_get_component_param_descriptions`（解析 `run` 方法 docstring 的 `:param:`，见 `parameters_schema_utils.py`），缺失则用 `f"Input '{input_name}' for the component."` 兜底。
- 必填性来自 socket 自身：`... if socket.is_mandatory else socket.default_value`。
- 类型经 `_resolve_type` 递归解析——这是把 Haystack 数据类（如 `Document`、`ChatMessage`）翻译成 Pydantic 可生成 schema 的形态的关键：dataclass 转成动态 Pydantic 模型，`list[X]`/`dict`/`Union` 递归处理，`Tool`/`Toolset` 因含 `Callable` 被替换成占位模型 `_ToolSchemaPlaceholder`/`_ToolsetSchemaPlaceholder`。

执行端的 `component_invoker` 是个闭包：它拿到 LLM 给的 `kwargs`（一堆原始 JSON 值），逐个用 `_convert_param` 把它们按 socket 声明的类型"还原"成组件期望的对象，再 `component.run(**converted_kwargs)`。`_convert_param` 的逻辑覆盖了三种情况：目标类型有 `from_dict`（含 `list[Dataclass]` 与单个 dataclass）的就调 `from_dict`；Union 类型里优先挑出带 `from_dict` 的 `list[T]` 那一支；其余交给 `TypeAdapter(param_type).validate_python` 做 Pydantic 校验。这一层"参数翻译"正是 LLM 的字符串/字典与强类型组件之间的桥梁。

名字若不显式给出，会把类名（如 `SerperDevWebSearch`）按驼峰转 snake_case 得到 `serper_dev_web_search`；描述默认取组件 docstring。`warm_up()` 被重写为转发到组件自身的 `warm_up`（带 `_is_warmed_up` 幂等保护）。`_get_valid_inputs`/`_get_valid_outputs` 重写为返回组件 input/output socket 名集合，让 `inputs_from_state`/`outputs_to_state` 的校验能精确到组件真实的输入输出。

## 12.4 把整条 Pipeline 包成工具：`PipelineTool`

`PipelineTool`（`pipeline_tool.py`）直接继承 `ComponentTool`，做法巧妙：它在构造时把传入的 `Pipeline`/`AsyncPipeline` 用 `SuperComponent(pipeline=..., input_mapping=..., output_mapping=...)` 包成一个组件，再交给父类 `ComponentTool.__init__`。也就是说——**Pipeline 先变成组件，再变成工具**，整个 schema 生成与参数翻译逻辑一行都不用重写。

`input_mapping`/`output_mapping` 决定了这条工具对外暴露哪些入参、回传哪些出参，例如示例里 `input_mapping={"query": ["embedder.text"]}` 把工具的 `query` 接到 Pipeline 内部 `embedder` 的 `text` socket，`output_mapping={"retriever.documents": "documents"}` 把检索结果重命名为 `documents` 暴露出去。`parameters_schema_utils.py` 的 `_get_component_param_descriptions` 还特意为 `_SuperComponent` 做了增强：沿 `input_mapping` 找到底层组件，把它们 `run` 方法的 docstring 拼成一段组合描述，让 LLM 看到的工具说明能反映 Pipeline 内部各环节的真实语义。序列化时 `to_dict` 存的是 `pipeline.to_dict()` 加上映射与 `is_pipeline_async` 标记，反序列化据此重建 `Pipeline` 或 `AsyncPipeline`。

## 12.5 聚合与去重：`Toolset`

`Toolset`（`toolset.py`）把一组相关工具当作一个整体来管理。它也是 `@dataclass`，唯一字段 `tools: list[Tool]`（用 `field(default_factory=list)` 初始化）。

它实现了完整的集合协议（`__iter__`/`__contains__`/`__len__`/`__getitem__`），因此"在任何期望工具列表的地方都能直接传 Toolset"——比如 `ToolInvoker` 和各类 ChatGenerator。`__contains__` 还贴心地支持按名字判断（`"tool_name" in toolset`）。

去重是核心约束。`__post_init__` 调用 `_check_duplicate_tool_names`（定义在 `tool.py`），该函数统计名字出现次数，只要有重名就抛 `ValueError`。`add()`（添加单个 Tool 或合并另一个 Toolset）与 `__add__`（`+` 运算符）在落地前都会先对"合并后的完整列表"跑一遍去重检查。为什么名字必须唯一？因为 LLM 是靠 `name` 来指名调用工具的，重名会让调用产生歧义。

`Toolset` 还为"动态加载"留了扩展位：文档明确鼓励子类（如从 MCP server、OpenAPI URL 加载工具）在 `__init__` 里做动态加载，并重写 `to_dict`/`from_dict` 去序列化"端点描述符"而非工具实例本身。`warm_up()` 默认遍历 warm 每个工具，子类可改成只 warm 一个共享连接。`__add__` 合并两个 Toolset 时返回一个内部 `_ToolsetWrapper`，它持有原始 toolsets 并据此重写迭代/计数，从而保留各自的独立配置——这对动态 toolset 很重要，避免合并后丢失各自的加载逻辑。

## 12.6 海量工具按需检索：`SearchableToolset`

工具一多，把全部定义塞进 LLM 上下文既烧 token 又干扰决策。`SearchableToolset`（`searchable_toolset.py`）用 BM25 检索解决这个问题：它不暴露全部工具，而是给 LLM 一个引导工具 `search_tools`，让模型用关键词先"找工具"，命中的工具才在下一轮迭代被加载进来。

关键机制：

- `search_threshold`（默认 `8`）。catalog 小于这个数时进入 passthrough 模式，直接暴露全部工具，不启用检索。
- `warm_up()` 里先 `warm_up_tools(self._raw_catalog)` 触发懒加载 toolset（如 `eager_connect=False` 的 MCPToolset）连接，再 `flatten_tools_or_toolsets` 拍平，然后把每个工具的 `name + description` 写进一个 `InMemoryDocumentStore`，并用 `create_tool_from_function` 生成 `search_tools` 引导工具。
- `search_tools` 内部用 `document_store.bm25_retrieval` 检索，把命中的工具 `warm_up()` 后塞进 `self._discovered_tools`。它只返回"已加载哪些工具"的简短消息（而非完整定义）以省 token。
- 真正让新工具对 LLM 可见的是 `__iter__`：Agent 每轮循环都重新迭代 toolset，非 passthrough 模式下产出"引导工具 + 已发现的工具"，于是上一轮检索到的工具自然出现在下一轮。

`SearchableToolset` 还显式禁用了 `add` 和 `__add__`（抛 `NotImplementedError`），因为它的工具集是检索驱动、运行期动态变化的，静态拼接没有意义。

## 12.7 工具间共享数据：`State`

工具不只是无状态函数，多个工具间常需要传递中间结果（检索到的文档、上下文、累积的对话）。`State`（`state.py`）就是 Agent 与其工具之间这块共享内存。

`State` 内部包一个 `_data` 字典，由 `schema` 描述。schema 每个条目形如 `{"type": SomeType, "handler": Callable[[Any, Any], Any] | None}`。构造时 `_validate_schema` 校验每个 `type` 合法（`state_utils._is_valid_type`）、`handler` 可调用，并对保留键 `messages` 强制要求 `list[ChatMessage]` 类型。构造还会自动补一个 `messages` 字段（`{"type": list[ChatMessage], "handler": merge_lists}`），让对话历史天然可累积。

`handler` 决定"写入时如何合并"。若 schema 没指定，按类型选默认：list 类型用 `merge_lists`（拼接，`state_utils.py`），其余用 `replace_values`（覆盖）。`set(key, value, handler_override=None)` 的逻辑是：取当前值，用 `handler_override or schema 里的 handler` 算出新值再写回；key 不在 schema 里直接抛 `ValueError`。`get` 返回的是 `deepcopy`，避免外部意外篡改内部状态。

这套机制正好对接 `Tool` 的 `inputs_from_state` 与 `outputs_to_state`：前者把 State 里某个键映射成工具的入参（运行时由 ToolInvoker 注入，因此这类参数在 schema 生成阶段被跳过），后者把工具的某个输出写回 State 的某个键（可带 `handler` 做自定义合并）。一进一出，工具便能"读上文、写上文"，多步骤协作得以串起来。`to_dict`/`from_dict` 借助 `_serialize_value_with_schema` 与类型/可调用序列化把整个 State 落盘再还原。

## 12.8 序列化与工具的统一搬运

跨工具/Toolset 的统一序列化收口在 `serde_utils.py`：`serialize_tools_or_toolset` 把 `Toolset`、`list[Tool | Toolset]` 或 `None` 转成可存储结构；`deserialize_tools_or_toolset_inplace` 反向就地还原，靠 `import_class_by_name` 按 `type` 字段找回 `Tool`/`ComponentTool`/`PipelineTool`/`Toolset` 等具体类。`utils.py` 的 `warm_up_tools` 与 `flatten_tools_or_toolsets` 则统一处理"单个 Tool / 单个 Toolset / 混合列表"三种形态，是 Agent 和 ToolInvoker 在运行前批量预热、拍平工具时的通用工具函数。`__init__.py` 里定义的类型别名 `ToolsType = Sequence[Tool | Toolset] | Toolset` 正是所有这些 API 接受的统一入参形态。

## 本章小结

- `Tool` 是工具系统的统一契约：`name`/`description`/`parameters`(JSON Schema)/`function` 四要素恰好覆盖 LLM function-calling 所需，外加可选的 state/string 映射。
- `__post_init__` 在定义期就用 `Draft202012Validator` 校验 schema、拒绝 async 函数、并对照有效参数/输出集合校验映射配置，把错误前移。
- `create_tool_from_function` 借 Pydantic `create_model` + `model_json_schema` 从类型注解反射 schema，`Annotated` 元数据当描述，`@tool` 是它的双 `overload` 语法糖。
- `ComponentTool` 把组件 input socket 翻译成 parameters schema，`component_invoker` 用 `_convert_param` 把 LLM 的原始参数还原成强类型再喂给 `component.run`，复用既有组件生态。
- `PipelineTool` 先用 `SuperComponent` 把 Pipeline 包成组件，再继承 `ComponentTool`，几乎零重写地把整条 Pipeline 变成一个工具。
- `Toolset` 实现集合协议、可迭代可合并，`_check_duplicate_tool_names` 强制工具名唯一，因为 LLM 靠名字指名调用；并为动态加载留了 `to_dict`/`from_dict` 扩展位。
- `SearchableToolset` 用 BM25 + 引导工具 `search_tools` 实现大工具集按需发现，靠 `__iter__` 每轮重新迭代让新发现的工具逐步可见。
- `State` 以 `schema` + `handler`（`merge_lists`/`replace_values`）管理工具间共享数据，配合 `inputs_from_state`/`outputs_to_state` 让工具能读写共享上文。
- 统一契约让 Agent 与具体工具实现彻底解耦：无论背后是函数、组件还是 Pipeline，Agent 只面对 `Tool`。

## 动手实验

1. **实验一：用 `@tool` 反射 schema** — 写一个带 `Annotated[str, "城市名"]` 参数和 `Literal` 枚举参数的 `get_weather` 函数，加 `@tool` 装饰器，打印 `get_weather.parameters`，确认 `properties` 里出现了从 `Annotated` 抽出的 `description` 和 `Literal` 生成的 `enum`，并验证 `title` 字段已被 `_remove_title_from_schema` 清除。
2. **实验二：组件即工具** — 任取一个内置组件实例化后传给 `ComponentTool`，打印 `tool.parameters`，对照组件 `__haystack_input__._sockets_dict` 的 socket 名与必填性，确认 schema 一一对应；再故意把组件先 `add_component` 进一个 Pipeline 后再创建 `ComponentTool`，观察 `__haystack_added_to_pipeline__` 触发的 `ValueError`。
3. **实验三：Toolset 去重** — 构造两个 `name` 相同的 `Tool`，分别尝试 `Toolset([a, b])` 与先 `Toolset([a])` 再 `.add(b)`，确认两条路径都抛出 `Duplicate tool names found` 的 `ValueError`；再 `to_dict()`/`from_dict()` 往返一个合法 Toolset，验证工具被正确还原。
4. **实验四：State 合并语义** — 创建 `State(schema={"docs": {"type": list[str]}, "name": {"type": str}})`，对 `docs` 连续 `set` 两次观察 `merge_lists` 的拼接效果，对 `name` 连续 `set` 两次观察 `replace_values` 的覆盖效果，再传入自定义 `handler_override` 改变某次合并行为，最后 `to_dict()` 检查 schema 与 data 的序列化结构。

> **下一章预告**：第 13 章 Agent 与 SuperComponent。我们将把本章的 `Tool`/`Toolset`/`State` 装进真正的 Agent 循环，看 `Agent` 如何驱动 ChatGenerator 与 ToolInvoker 反复"思考-调用工具-观察"，以及 `SuperComponent` 如何把一整条 Pipeline 折叠成一个可复用组件。
