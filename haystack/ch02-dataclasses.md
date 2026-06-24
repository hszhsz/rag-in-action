# 第 2 章 核心数据模型 Dataclasses

第 1 章我们看到 Haystack 用 `@component` 装饰器把任意类变成"管道节点",每个节点声明自己的输入输出类型,然后由 Pipeline 按图连接。但一个关键问题被刻意留到了本章:这些节点之间到底**传递什么**?如果每个组件各自定义私有的数据结构,那么"任意拼接"就成了空话——下游根本无法理解上游吐出来的东西。Haystack 的答案是:用一组**统一的、可序列化的 dataclass** 作为组件间的通用契约。检索器吐 `Document`,生成器吃 `ChatMessage` 吐 `ChatMessage`,流式输出走 `StreamingChunk`,二进制文件用 `ByteStream` 承载。这些类就是流经整个 Pipeline 的"血液"。

这一章我们逐个解剖这些核心数据模型。它们全部定义在 `haystack/dataclasses/` 目录下,通过 `haystack/dataclasses/__init__.py` 用 `LazyImporter` 惰性导出。理解它们的字段语义、`id` 如何由内容算哈希、`to_dict`/`from_dict` 如何支撑 Pipeline 的存档与断点,是读懂后续所有组件的前提。

## Document:RAG 的最小数据单元

`Document` 定义在 `haystack/dataclasses/document.py`,是整个检索链路的最小数据单元——文档存储里存的是它,检索器返回的是它,提示词里拼进去的也是它。它的字段非常克制:

- `id: str`——唯一标识符,默认空串,不设置时自动由内容生成;
- `content: str | None`——文本内容;
- `blob: ByteStream | None`——二进制数据(图片、音频、PDF 原始字节);
- `meta: dict[str, Any]`——自定义元数据,要求 JSON 可序列化;
- `score: float | None`——得分,通常由检索器赋值用于排序;
- `embedding: list[float] | None`——稠密向量表示;
- `sparse_embedding: SparseEmbedding | None`——稀疏向量表示。

注意 `content` 与 `blob` 在语义上是"二选一"的两种载体:文本走 `content`,非文本走 `blob`。这种分离让一个 `Document` 既能表达纯文本切片,也能表达一张图片或一段音频,而不必为多模态再造一套类。

### id 如何由内容自动算哈希

`Document` 没有把 `id` 交给用户随手填,而是在 `__post_init__` 里做了一件关键的事:

```python
def __post_init__(self) -> None:
    self.id = self.id or self._create_id()
```

只有当用户**没有显式传入** `id` 时才自动生成。生成逻辑在 `_create_id` 里,它把 `content`、`blob.data`、`blob.mime_type`、`meta`、`embedding`、`sparse_embedding` 拼成一个字符串,再做 `hashlib.sha256(...).hexdigest()`:

```python
data = f"{text}{dataframe}{blob!r}{mime_type}{meta}{embedding}{sparse_embedding}"
return hashlib.sha256(data.encode("utf-8")).hexdigest()
```

这意味着 `id` 是**内容寻址**的:内容相同的两个 `Document` 会得到相同的 `id`,这天然支持文档存储里的去重与幂等写入。源码里还保留了一个 `dataframe = None` 的局部变量,注释解释这是为了"即使 `dataframe` 字段已被移除,`id` 的计算结果也保持不变"——这是 Haystack 1.x 迁移到 2.x 的兼容包袱,作者宁愿留一个无用变量也要保证旧文档的 `id` 不漂移。

### 1.x 兼容与不可变保护

`Document` 用元类 `_BackwardCompatible` 处理历史包袱:在 `__init__` 之前拦截参数,把 1.x 里以 NumPy `ndarray` 存储的 `embedding` 转成 `list`,并丢弃 `content_type`、`id_hash_keys`、`dataframe` 这三个 `LEGACY_FIELDS`。`content` 若不是字符串则直接报错。

`Document` 还被 `@_warn_on_inplace_mutation` 装饰(定义在 `haystack/utils/dataclasses.py`)。这个装饰器包裹 `__setattr__`,一旦实例在初始化完成后被原地修改字段,就发出警告,提示应改用 `dataclasses.replace(...)`。原因很现实:同一个 `Document` 实例可能被 Pipeline 里多个组件共享引用,原地改字段会污染其他分支。这是一种"软不可变"约束——不阻止你改,但提醒你改了有风险。

### 序列化:to_dict / from_dict

`to_dict` 用 `dataclasses.asdict` 转字典,并把 `blob`、`sparse_embedding` 替换成它们各自的 `to_dict()` 结果(因为 `ByteStream` 的 `bytes` 与 `SparseEmbedding` 不是 JSON 原生类型)。它还有个 `flatten: bool = True` 参数:默认会把 `meta` 字典摊平到顶层,以兼容 1.x 的扁平结构。`from_dict` 则反向操作,并做一个聪明的"反摊平":凡是不在 `Document` 字段名(加上 `LEGACY_FIELDS`)里的键,都视为元数据塞回 `meta`;若同时传了 `meta=` 和摊平键则报错。

`__eq__` 的实现也值得一提——两个 `Document` 是否相等,直接比较 `self.to_dict() == other.to_dict()`,即"字典表示完全一致即相等"。

## SparseEmbedding:稀疏向量

`SparseEmbedding` 定义在 `haystack/dataclasses/sparse_embedding.py`,极简:`indices: list[int]` 与 `values: list[float]`,分别是非零元素的下标和值。`__post_init__` 校验两者长度一致,否则抛 `ValueError`。它服务于 SPLADE 这类稀疏检索场景,与稠密 `embedding` 并存于 `Document`,让混合检索成为可能。

## ByteStream:二进制数据的载体

`ByteStream` 定义在 `haystack/dataclasses/byte_stream.py`,是 Haystack 里所有非文本二进制的统一载体。三个字段:`data: bytes`、`meta: dict`、`mime_type: str | None`。

它提供了一组工厂与转换方法:`from_file_path` 从文件读字节(可选 `guess_mime_type=True` 用 `_guess_mime_type` 猜 MIME),`from_string` 把字符串按编码转字节,`to_file` 写回磁盘,`to_string` 解码回字符串。

序列化上有个细节:JSON 不支持 `bytes`,所以 `to_dict` 把 `data` 转成**整数列表** `list(self.data)`,`from_dict` 再用 `bytes(data["data"])` 还原。此外它还有个 `_to_trace_dict`,把二进制 `data` 替换成 `"Binary data (N bytes)"` 占位串——这样可观测性/追踪后端不会被大块二进制塞爆,这个"追踪专用序列化"的模式在 `ChatMessage` 与 `ImageContent` 里也复用了。

## ChatMessage:统一的多角色消息模型

`ChatMessage` 定义在 `haystack/dataclasses/chat_message.py`,是这一章最厚重的一个类(八百多行)。它要解决的问题是:LLM 对话里有 user、system、assistant、tool 四种角色,assistant 还可能携带工具调用与推理过程,tool 消息装的是工具执行结果——如何用**一个类**统一表达所有这些?

### ChatRole 与内容部分

角色由 `ChatRole(str, Enum)` 枚举:`USER`、`SYSTEM`、`ASSISTANT`、`TOOL`,并提供 `from_str` 做字符串到枚举的转换。

消息内容不是一个字符串,而是一个**内容部分列表** `_content: Sequence[ChatMessageContentT]`。`ChatMessageContentT` 是一个联合类型,涵盖六种内容部分:`TextContent`(纯文本)、`ToolCall`(模型发起的工具调用,含 `tool_name`/`arguments`/`id`/`extra`)、`ToolCallResult`(工具执行结果,含 `result`/`origin`/`error`)、`ImageContent`、`ReasoningContent`(模型的推理文本)、`FileContent`。这种"消息=角色+内容部分列表"的设计,让一条 assistant 消息可以同时包含推理、文本和多个工具调用。

`ChatMessage` 的字段都是带下划线前缀的私有字段(`_role`、`_content`、`_name`、`_meta`),通过只读 `@property` 暴露,意在引导用户用工厂方法构造而非直接 `ChatMessage(...)`。

### 四个 from_* 工厂方法

构造 `ChatMessage` 的正道是四个类方法:

- `from_user(text=..., content_parts=...)`——用户消息,`text` 与 `content_parts` 二选一,后者允许混入 `ImageContent`/`FileContent` 实现多模态输入,二者都不给或都给都会报错;
- `from_system(text=...)`——系统提示;
- `from_assistant(text=..., tool_calls=..., reasoning=...)`——助手消息,可同时带文本、工具调用和推理内容,内容部分按 reasoning→text→tool_calls 顺序拼装;
- `from_tool(tool_result, origin, error=False)`——工具结果消息,内部包装成一个 `ToolCallResult`,`origin` 指回触发它的那个 `ToolCall`。

### 便捷属性

由于内容是列表,直接取值会很啰嗦,所以 `ChatMessage` 提供了大量便捷属性。复数形式返回列表:`texts`、`tool_calls`、`tool_call_results`、`images`、`files`、`reasonings`;单数形式返回第一个或 `None`:`text`、`tool_call`、`tool_call_result`、`image`、`file`、`reasoning`。它们的实现都是对 `_content` 做 `isinstance` 过滤。还有 `is_from(role)` 判断消息来源角色。这套属性让上层代码可以写 `message.text`、`message.tool_calls` 而无需关心底层列表结构。

### 与 OpenAI 格式互转

`ChatMessage` 不是凭空造的,它要能喂给真实的 LLM API。`to_openai_dict_format` 把内部表示翻译成 OpenAI Chat Completions API 的字典,过程里分了三条私有路径:`_user_message_to_openai`(处理多模态 user 消息,把 `ImageContent` 转 `image_url`、`FileContent` 转 `file`)、`_tool_result_message_to_openai`(tool 结果,带 `tool_call_id`)、`_system_assistant_message_to_openai`(把 `ToolCall` 转成 OpenAI 的 `function` 结构,`arguments` 用 `json.dumps(..., ensure_ascii=False)` 序列化以保留 emoji 等字符)。这里有不少"现实妥协":OpenAI Chat Completions 不支持推理内容,所以 reasoning 被直接忽略;不支持多模态工具结果,遇到就报错并提示改用 Responses API。

反向的 `from_openai_dict_format` 先用 `_validate_openai_message` 校验,再按角色分派到对应的 `from_*` 工厂。参数 `require_tool_call_ids` 默认要求工具调用必须带 `id`(OpenAI 强制),但允许关掉以兼容"浅层" OpenAI 兼容 API。

### 序列化

`to_dict` 把 `_role` 转成字符串、把每个内容部分用 `_serialize_content_part` 序列化成带类型键的字典(如 `{"tool_call": {...}}`),其中 `TextContent` 特殊处理为扁平的 `{"text": "..."}`。`from_dict` 极其重视**跨版本兼容**:它同时认 `role`/`_role`、`content`/`_content`,认 `content` 是列表(2.9.0 起的当前格式)还是字符串(2.9.0 前的格式),逐一兜底。`_deserialize_content_part` 根据字典里的键(`text`/`image`/`tool_call`/`tool_call_result`/`reasoning`/`file`)还原对应类型,遇到非法格式抛出一段**专为 LLM 设计的详细报错**——因为 Agent 运行时可能让模型自己拼消息,详细错误能引导模型改正。

## StreamingChunk:流式生成的增量片段

`StreamingChunk` 定义在 `haystack/dataclasses/streaming_chunk.py`,承载 LLM 流式输出的一个增量片段。核心字段:

- `content: str`——这一片的文本增量;
- `index: int | None`——这一片属于哪个内容块;
- `tool_calls: list[ToolCallDelta] | None`——工具调用的增量(`ToolCallDelta` 的 `arguments` 是 JSON 片段,可逐字拼接);
- `tool_call_result`、`reasoning`——工具结果与推理的增量;
- `start: bool`——是否是某内容块的起始;
- `finish_reason: FinishReason | None`——结束原因;
- `component_info: ComponentInfo | None`——哪个组件产生了这一片。

`FinishReason` 是个 `Literal`,沿用 OpenAI 约定的 `"stop"`/`"length"`/`"tool_calls"`/`"content_filter"`,外加 Haystack 自有的 `"tool_call_results"`。

`__post_init__` 里有两条硬约束:`content`、`tool_calls`、`tool_call_result`、`reasoning` 四者**至多设一个**(否则报错),且只要设了后三者之一就**必须同时设 `index`**(因为流式拼接需要知道每片归属哪个块)。`ComponentInfo` 通过 `from_component` 从组件实例提取 `type`(模块名+类名)与 `name`(挂进 Pipeline 时的名字),让前端能知道是哪个组件在吐流。

`StreamingChunk` 同样有 `to_dict`/`from_dict`,把嵌套的 `ComponentInfo`/`ToolCallDelta`/`ToolCallResult`/`ReasoningContent` 递归序列化。把一串 chunk 按 `index` 归组、把各自的 `content` 与 `arguments` 增量拼接,就能还原出一条完整的 assistant `ChatMessage`——这正是各 Generator 内部做的事。

同文件还定义了回调类型 `StreamingCallbackT`(同步/异步两版)与 `select_streaming_callback`,后者在初始化回调与运行时回调之间择优(运行时优先),并校验同步/异步是否匹配。这把"流式片段"这个数据模型和"如何消费它"的回调契约放在了一起。

## Answer:抽取式与生成式答案

`Answer` 定义在 `haystack/dataclasses/answer.py`,本身是一个 `@runtime_checkable` 的 `Protocol`——只规定三件事:有 `data`、`query`、`meta` 字段,有 `to_dict`/`from_dict`。任何满足这个协议的类都是"答案"。

具体实现有两个:`ExtractedAnswer` 服务抽取式 Reader,持有 `query`/`score`/`data`,以及指向来源 `Document` 的引用和 `Span`(`start`/`end` 偏移)定位答案在文档/上下文里的位置;`GeneratedAnswer` 服务生成式 Generator,持有 `data`(答案文本)、`query`、引用的 `documents` 列表与 `meta`。两者的序列化都走 `haystack.core.serialization` 的 `default_to_dict`/`default_from_dict`(第 5 章会详谈这套通用序列化机制),并且会把内部的 `Document`、乃至 `meta["all_messages"]` 里的 `ChatMessage` 一并递归序列化——可见这些数据模型是层层嵌套、整体可存档的。

## 设计动机:为什么是一组统一的可序列化 dataclass

回到本章开头的问题。Haystack 选择用这一组 dataclass 作为组件间契约,背后有三层考量:

第一,**互操作**。所有检索器都吐 `Document`、所有生成器都吃/吐 `ChatMessage`,于是 A 公司的检索器能直接接 B 公司的生成器——这正是第 1 章"任意 component 可拼接"得以成立的数据基础。

第二,**可存档与断点续跑**。每个类都实现了 `to_dict`/`from_dict`,且嵌套类型(`ByteStream`、`SparseEmbedding`、`ChatMessage` 内容部分)都能递归序列化成纯 JSON 可表达的结构。这意味着 Pipeline 在任意环节的中间产物都能落盘,支撑序列化(第 5 章)与断点(`breakpoints` 模块)。

第三,**演进韧性**。`Document` 的元类兼容、`ChatMessage.from_dict` 同时认新旧字段名和新旧 `content` 格式、`_create_id` 保留无用 `dataframe` 变量,这些细节都指向同一个目标:在 1.x→2.x 的大版本迁移和持续演进中,**老数据不能失效、老 `id` 不能漂移**。再加上 `@_warn_on_inplace_mutation` 的软不可变保护,这组 dataclass 既稳定又安全地充当了管道里流动的血液。

## 本章小结

- Haystack 用 `haystack/dataclasses/` 下一组统一、可序列化的 dataclass 作为组件间契约,它们是流经 Pipeline 的"血液"。
- `Document` 是 RAG 最小数据单元,`content`(文本)与 `blob`(二进制)二选一,另有 `meta`/`score`/`embedding`/`sparse_embedding`。
- `Document.id` 不显式设置时由 `_create_id` 基于内容做 `sha256` 哈希生成,实现内容寻址与天然去重;`__post_init__` 只在未设置时才生成。
- `ByteStream` 是二进制统一载体,序列化时把 `bytes` 转整数列表绕开 JSON 限制,并有 `_to_trace_dict` 避免大 payload 进追踪后端。
- `ChatMessage` 以"角色 + 内容部分列表"统一表达 user/system/assistant/tool 四种消息,内容部分含 `TextContent`/`ToolCall`/`ToolCallResult`/`ImageContent`/`ReasoningContent`/`FileContent`。
- `ChatMessage` 用四个 `from_*` 工厂构造、用复数/单数便捷属性取值,并能与 OpenAI 格式双向互转(互转时对不支持的特性做妥协)。
- `StreamingChunk` 承载流式增量,`__post_init__` 约束四类内容至多设一个且设非文本内容时必须带 `index`,据此可按块拼回完整响应。
- `Answer` 是一个 `Protocol`,`ExtractedAnswer`(带 `Span` 定位)与 `GeneratedAnswer`(带引用文档)是其两个实现。
- 所有 dataclass 都实现 `to_dict`/`from_dict` 并递归序列化嵌套类型,这是 Pipeline 存档与断点续跑的基础。
- `@_warn_on_inplace_mutation` 提供软不可变保护,提示用 `dataclasses.replace` 替代原地修改,避免共享实例被污染。

## 动手实验

1. **实验一:验证 Document 的内容寻址** — 用相同 `content` 与 `meta` 构造两个 `Document`,打印各自的 `id`,确认完全一致;再改动其中一个的 `meta`,观察 `id` 随之变化,从而理解 `_create_id` 把哪些字段纳入了哈希。
2. **实验二:ChatMessage 的工厂与属性** — 用 `ChatMessage.from_assistant(text="...", tool_calls=[ToolCall(...)], reasoning="...")` 造一条助手消息,分别读取它的 `text`、`tool_calls`、`reasoning` 属性,再调用 `to_openai_dict_format()` 看推理内容是否被忽略、工具调用是否被转成 `function` 结构。
3. **实验三:序列化往返** — 对一个带 `blob`(用 `ByteStream.from_string` 造)的 `Document` 调 `to_dict()` 再 `from_dict()`,断言往返前后 `==` 相等;观察 `to_dict` 输出里 `blob.data` 是如何变成整数列表的。
4. **实验四:拼回流式响应** — 手工构造若干 `StreamingChunk`(部分只带 `content`、部分带 `tool_calls` 与 `index`),按 `index` 归组并拼接 `content` 字符串与 `ToolCallDelta.arguments`,还原出一条完整的 assistant 消息内容;故意同时设置 `content` 和 `reasoning`,观察 `__post_init__` 抛出的 `ValueError`。

> **下一章预告**:数据模型解决了"传什么",下一章《Pipeline 图构建与连接》将解决"怎么连"——我们会深入 `Pipeline` 如何用有向图把组件按 `connect` 串起来、输入输出 socket 如何按类型校验匹配、以及这张图在执行前是如何被验证的。
