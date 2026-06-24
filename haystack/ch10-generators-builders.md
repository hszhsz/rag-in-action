# 第 10 章 生成器与 Prompt 构建

第 9 章我们把检索器召回、重排器排序后的 `List[Document]` 拿到了手里。但文档本身不是答案——它们只是"原料"。要让大模型基于这些原料生成回答，还差两步关键拼装：第一步是把文档塞进一个 prompt 模板，渲染成给 LLM 的输入；第二步是把 LLM 的回复连同来源文档封装成结构化的 `GeneratedAnswer`。Haystack 把这两步拆成了三类组件：**Prompt 构建器**（`PromptBuilder` / `ChatPromptBuilder`）、**生成器**（`OpenAIChatGenerator` 等）、**答案构建器**（`AnswerBuilder`）。

这种"builder / generator / builder"的三段式解耦不是偶然。它背后是 Haystack 的一条设计哲学：**prompt 即数据模板，生成是纯函数，引用抽取是后处理**。模板渲染不调用网络、生成器不关心文档从哪来、答案构建器不关心模型是谁。本章逐个解构它们的真实源码，讲清"输入输出契约""流式 chunk 怎么拼""tool calling 怎么透传""引用怎么用正则抽出来"，以及最重要的——为什么这样切分职责。

## 一、ChatGenerator 的输入输出契约

打开 `haystack/components/generators/chat/openai.py`，`OpenAIChatGenerator` 的 `run` 方法签名揭示了整个 ChatGenerator 家族的统一契约：输入是 `messages: list[ChatMessage] | str`，输出经 `@component.output_types(replies=list[ChatMessage])` 声明为 `{"replies": [...]}`。也就是说，**ChatGenerator 吃一串 `ChatMessage`，吐一串 `ChatMessage`**。这与第 2 章介绍的 `ChatMessage` 数据模型严丝合缝：输入可以包含 system / user / assistant / tool 多种角色，输出则是 assistant 角色的回复。

输入容错由 `_normalize_messages`（在 `generators/utils.py`）处理：如果传进来的是裸字符串，它会被包成 `[ChatMessage.from_user(messages)]`；如果是合法的 `list[ChatMessage]` 则原样返回；否则抛 `TypeError`。这让用户既能 `run("你好")` 也能传完整对话历史，降低了上手门槛。`run` 开头还有一个 `if len(messages) == 0: return {"replies": []}` 的短路——空输入直接返回空，不浪费一次 API 调用。

`run` 之外还有一个完全对称的 `run_async`，它用 `self.async_client` 与 `await` 调用同一套 `_prepare_api_call`。同步与异步共享参数准备逻辑，只在"谁来发请求""怎么消费流"上分叉，这是 Haystack 避免逻辑重复的典型手法。

## 二、_prepare_api_call：参数合并与端点选择

真正把 `ChatMessage` 翻译成 OpenAI 请求体的是 `_prepare_api_call`。它做了几件事，每一件都值得拆开看：

**generation_kwargs 的两级合并**。`generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}`——初始化时设的参数是底，`run` 时传的参数覆盖在上。这意味着你可以在构造器里设好默认 `temperature`，再在某次调用临时调高。

**消息格式转换**。`openai_formatted_messages = [message.to_openai_dict_format() for message in messages]`，把 Haystack 自己的 `ChatMessage` 转成 OpenAI API 要的 `{"role": ..., "content": ...}` 字典。生成器不自己拼字典，而是把转换职责下放给数据模型，这样换 provider 时只需各自的 `to_*_format`。

**tools 透传**。`flatten_tools_or_toolsets(tools or self.tools)` 把工具（或 Toolset）摊平成列表，逐个取 `t.tool_spec` 包成 `{"type": "function", "function": ...}`。若 `tools_strict` 为真，还会调用 `_make_schema_strict` 递归地给 JSON schema 加上 `additionalProperties: false` 并把所有属性塞进 `required`，以满足 OpenAI 严格模式的 schema 约束 [[OpenAI Chat API]](https://platform.openai.com/docs/api-reference/chat)。

**端点的三岔路口**。这是最精妙的部分。Haystack 不直接调死 `chat.completions.create`，而是返回一个 `openai_endpoint` 字段当"提示"：
- 若设了 `response_format` 且**非流式** → 用 `parse` 端点（OpenAI 的结构化输出解析），注释明确写道 `stream` 不能传给 `chat.completions.parse`；
- 否则 → 用 `create` 端点，并带上 `stream=is_streaming`。

回到 `run`，它用 `openai_endpoint = api_args.pop("openai_endpoint")` 取出这个提示，再 `getattr(self.client.chat.completions, openai_endpoint)` 动态拿到方法对象。这种"用字符串选方法"的写法把"该走哪个端点"的决策集中在一处，`run` 本身保持干净。

## 三、流式回调与 chunk 拼装

非流式分支很简单：拿到 `ChatCompletion` 后，对每个 `choice` 调 `_convert_chat_completion_to_chat_message` 转成 `ChatMessage`。我们重点看流式。

当 `streaming_callback is not None`，`run` 走 `_handle_stream_response`。它的循环逻辑是：

```python
for chunk in chat_completion:
    chunk_delta = _convert_chat_completion_chunk_to_streaming_chunk(
        chunk=chunk, previous_chunks=chunks, component_info=component_info)
    chunks.append(chunk_delta)
    callback(chunk_delta)
return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]
```

每收到一个原始 `ChatCompletionChunk`，先转成 Haystack 统一的 `StreamingChunk`，**立即回调**给用户（实现边生成边显示），同时累积到 `chunks` 列表；流结束后再用 `_convert_streaming_chunks_to_chat_message` 把所有 chunk 缝合成一条完整的 `ChatMessage`。注意 `assert len(chunk.choices) <= 1`——流式只支持单 choice，这也是 `_prepare_api_call` 里 `if is_streaming and num_responses > 1: raise ValueError` 的由来。

`_convert_chat_completion_chunk_to_streaming_chunk` 里藏着不少细节。OpenAI 的第一个 chunk 只带 role 信息（如 `"assistant"`），此时 `index` 设为 `None`；当 `include_usage` 开启时，最后一个 chunk 的 `choices` 为空、只带 usage，函数直接返回一个 `content=""` 但带 usage 的 `StreamingChunk`。`finish_reason` 通过一个映射表归一化，特别地把 OpenAI 的 `function_call` 也映射成 `"tool_calls"`，屏蔽了新老 API 差异。

缝合逻辑 `_convert_streaming_chunks_to_chat_message`（在 `generators/utils.py`）则负责把碎片还原：
- 文本用 `"".join([chunk.content for chunk in chunks])` 直接拼接；
- **工具调用按 `index` 跨 chunk 累加**——因为流式时一个工具调用的 name 和 arguments 会分散在多个 chunk，代码用 `tool_call_data: dict[int, dict]` 按索引归集，`name`/`arguments` 字符串累加，最后 `json.loads(arguments)` 解析；若 JSON 非法则跳过并告警，提示用户开 `tools_strict`；
- `finish_reason` 取最后一个非空值；`usage` 从后往前找第一个非 `None` 的（OpenAI 在末尾空 choice chunk 返回 usage，但其它 provider 如 Qwen3 位置不同，注释明确指出了这点）；
- `completion_start_time` 记录第一个 chunk 的 `received_at`，用于测首 token 延迟。

这套拼装把"provider 的流式怪癖"统一收口在一个函数里，下游永远拿到一条规整的 `ChatMessage`。

## 四、tool calling 的端到端透传

把前面的线索串起来看 tool calling 的完整链路：用户在构造器或 `run` 传 `tools` → `_prepare_api_call` 把它们转成 OpenAI 工具定义 → API 返回工具调用 → 非流式由 `_convert_chat_completion_to_chat_message` 解析、流式由缝合函数累加 → 最终落在 `ChatMessage.from_assistant(text=..., tool_calls=tool_calls, meta=meta)` 的 `tool_calls` 字段里。

值得注意的是 `_convert_chat_completion_to_chat_message` 里对工具类型的过滤：`openai_tool_calls = [tc for tc in message.tool_calls if not isinstance(tc, ChatCompletionMessageCustomToolCall)]`，注释说明当前只支持 function 类型工具，自定义工具被显式排除。每个工具调用解析失败（`json.JSONDecodeError`）时不会让整条消息报错，而是跳过单个调用并告警——这是面向不可靠 LLM 输出的防御式设计。生成器只负责把工具调用"透传"成 `ToolCall` 对象，**真正执行工具是下游 `ToolInvoker` 的事**（见第 12 章），职责边界很清晰。

`warm_up` 方法体现了同样的解耦：它只做 `warm_up_tools(self.tools)` 且幂等（`_is_warmed_up` 标志位），把工具的预热从生成逻辑里剥离。

## 五、usage、finish_reason 与后处理

无论流式与否，`run` 返回前都会对每条 `completions` 调 `_check_finish_reason`。它检查 `meta["finish_reason"]`：若是 `"length"` 就警告"回复因 token 上限被截断，请调大 `max_completion_tokens`"；若是 `"content_filter"` 就警告内容被过滤。这种主动告警把"为什么回答不完整"这类隐性问题显式化。

usage 统计的来源在两条路径上不同：非流式直接取 `completion.usage`（经 `_serialize_object` 转 dict）；流式则如上节所述从 chunk 里捞。`_serialize_object` 是个通用工具，优先调 `model_dump()`，否则递归处理 `__dict__`/dict/list，把 OpenAI 的 pydantic 对象转成纯 dict，保证 meta 可序列化。最终 meta 里固定包含 `model`、`index`、`finish_reason`、`usage`，有需要时还带 `logprobs`。

## 六、chat 与非 chat 生成器的差异

`haystack/components/generators/openai.py` 里的 `OpenAIGenerator`（非 chat）几乎完全复用了 chat 版的内部函数——它直接从 `chat.openai` 导入 `_check_finish_reason`、`_convert_chat_completion_chunk_to_streaming_chunk`、`_convert_chat_completion_to_chat_message`。区别在于：非 chat 版输入是字符串（外加可选 `system_prompt`），输出 `replies` 是 `list[str]`。`generators/utils.py` 的 `_generators_deprecation_warning` 表明非 chat 生成器已被标记为 Haystack 3.0 将移除，官方建议统一用 ChatGenerator（它现在也支持字符串输入）。换言之，**非 chat 生成器是历史包袱，新代码应一律用 chat 版**。

## 七、ChatPromptBuilder：Jinja2 把文档渲染进消息

回到链路起点。`haystack/components/builders/chat_prompt_builder.py` 的 `ChatPromptBuilder` 负责把检索到的 `documents`、用户的 `query` 等变量渲染进 prompt 模板。它的 `template` 可以是 `list[ChatMessage]`，也可以是一个特殊字符串。

**渲染引擎**用的是 `SandboxedEnvironment`（来自 `jinja2.sandbox`），而非普通 Jinja2 环境。沙箱化是出于安全考虑：模板里可能含用户/检索内容，沙箱限制了危险属性访问 [[Jinja2 Sandbox]](https://jinja.palletsprojects.com/en/stable/sandbox/)。环境还注册了 `ChatMessageExtension`（支持 `{% message role="..." %}` 语法）和——若 `arrow` 已安装——`Jinja2TimeExtension`（支持 `{% now %}` 取当前时间，便于在 prompt 里注入日期）。

**变量推断**在 `__init__` 完成。当传了 `template` 但没显式给 `variables`，它遍历 system/user 角色的消息文本，用 `_extract_template_variables_and_assignments`（基于 Jinja2 AST 的 `meta.find_undeclared_variables`）找出所有未声明变量，再减去模板内 `{% set %}` 赋值的变量，得到真正需要外部输入的变量集。这套 AST 分析让 builder 能**自动把模板变量声明为组件输入端口**——下面这段是关键：

```python
for var in self.variables:
    if self.required_variables == "*" or var in self.required_variables:
        component.set_input_type(self, var, Any)
    else:
        component.set_input_type(self, var, Any, "")
```

必需变量声明为无默认值的输入（不连接就报错），可选变量声明为默认 `""` 的输入。这正是第 1 章 `@component` 契约的动态用法：**输入端口是在运行期按模板内容生成的**。如果有变量但 `required_variables` 没设，还会告警提醒用户显式声明必需项，避免多分支 pipeline 里因变量缺失导致的意外执行。

**运行期渲染**在 `run`：`template_variables_combined = {**kwargs, **template_variables}`，pipeline 传入的 kwargs 打底、`template_variables` 覆盖。对每条 user/system 消息，先 `_validate_variables` 校验必需变量齐全，再 `self._env.from_string(message.text).render(...)` 渲染。注意它用 `dataclasses.replace(message, _content=[TextContent(text=rendered_text)])` 生成新消息而非原地改——**避免污染原始模板**，让同一个 builder 实例可被多次安全调用。assistant/tool 角色的消息则原样透传，不做渲染。

字符串模板走 `_render_chat_messages_from_str_template`：渲染后按行 `json.loads` 解析成多条 `ChatMessage`，这条路径配合 `ChatMessageExtension` 的 `{% message %}` 块和 `templatize_part` 过滤器（用于把图片等多模态内容嵌入），但该过滤器与 `list[ChatMessage]` 模板互斥，源码用 `FILTER_NOT_ALLOWED_ERROR_MESSAGE` 显式拒绝。

`PromptBuilder`（非 chat，`prompt_builder.py`）是简化版：模板是单个字符串，输出 `prompt: str`，同样用 `SandboxedEnvironment` 和 AST 变量推断，但渲染结果直接返回字符串。两者共享变量推断与 `_validate_variables` 逻辑，只在输出类型上分叉。

## 八、AnswerBuilder：把回复封成带引用的答案

`haystack/components/builders/answer_builder.py` 是链路终点。`AnswerBuilder.run` 接收 `query`、`replies`（可以是 `list[str]` 或 `list[ChatMessage]`，兼容两类生成器）、可选的 `meta` 和 `documents`，输出 `answers: list[GeneratedAnswer]`。

**文本与元数据抽取**对 `ChatMessage` 与字符串做了分支：`extracted_reply = reply.text or "" if isinstance(reply, ChatMessage) else str(reply)`，并把消息自带的 `reply.meta` 与外部 `meta` 合并，还塞进一个 `all_messages` 记录完整回复列表。

**答案抽取**由 `_extract_answer_string` 按 `pattern` 正则完成：无 pattern 则整段作答；有 pattern 且有捕获组用 `match.group(1)`，无捕获组用 `match.group(0)`。构造器和 run 都会调 `_check_num_groups_in_regex` 确保 pattern 至多一个捕获组，否则报错——这是对"模糊正则"的护栏。

**引用抽取**是 AnswerBuilder 的精华。默认引用正则 `DEFAULT_REFERENCE_PATTERN = r"\[(\d+)\]"` 匹配 `[1]` 这样的标号；若开启 `expand_reference_ranges`，会自动切换到 `EXPANDED_REFERENCE_PATTERN = r"\[(\d+(?:[,-]\d+)*)\]"`，支持 `[6-10]` 范围引用。`_extract_reference_idxs` 用 `re.findall` 找出所有标号，范围模式下按 `,` 和 `-` 拆分展开，并**用 `num_documents` 把范围上限钳住**，防止 `[1-999999999]` 这类越界引用撑爆内存。注意引用从 `[1]` 起算，代码统一做 `int(match) - 1` 转 0 基索引。

抽出索引后，对每个被引用（或全部，取决于 `return_only_referenced_documents`）的文档，复制一份并写入 `source_index`（1 基位置）和 `referenced`（布尔）两个 meta 字段——同样用 `dataclasses.replace` 避免改原文档。越界索引会告警跳过。最终封成 `GeneratedAnswer(data=answer_string, query=query, documents=referenced_docs, meta=...)`。

为什么要单独一个 AnswerBuilder？因为"从自由文本里抽答案和引用"是个与模型无关、与检索无关的纯后处理问题。把它独立出来，意味着你换任何生成器、任何检索器，引用抽取逻辑都不用动——这正是三段式解耦的价值兑现。

## 九、FallbackChatGenerator：多 provider 容错

`haystack/components/generators/chat/fallback.py` 的 `FallbackChatGenerator` 把"多模型回退"做成了一个包装组件。它持有 `chat_generators: list[ChatGenerator]`，`run` 时按顺序逐个尝试：

```python
for idx, gen in enumerate(self.chat_generators):
    try:
        result = self._run_single_sync(gen, messages, generation_kwargs, tools, streaming_callback)
        ... # 成功就补上元数据并返回
    except Exception as e:
        failed.append(gen_name); last_error = e
```

任何异常（超时、429、401、500…注释里逐条列举）都触发回退到下一个生成器；全部失败才抛 `RuntimeError`，错误信息里带上失败列表和最后一个错误。成功时返回的 `meta` 会补上 `successful_chat_generator_index`、`successful_chat_generator_class`、`total_attempts`、`failed_chat_generators`，便于观测到底是第几个模型救了场。

它的实现处处体现"透明转发"：参数原样传给底层生成器，`run_async` 里若底层有 `run_async` 就 `await`，否则用 `asyncio.to_thread` 把同步 `run` 丢到线程池。文档里特别强调**超时完全委托给底层生成器**——Fallback 自己不实现超时，只在底层抛超时异常时接住。这是个清醒的设计取舍：与其重复造一套不可靠的超时机制，不如依赖各 provider 客户端成熟的超时实现。`warm_up` 也只是遍历底层逐个预热，自身不持有任何模型状态。

## 本章小结

- ChatGenerator 的统一契约是输入 `list[ChatMessage] | str`、输出 `{"replies": list[ChatMessage]}`，`_normalize_messages` 让裸字符串也能用。
- `_prepare_api_call` 集中处理 generation_kwargs 两级合并、消息格式转换、tools 转换与 strict schema，并用 `openai_endpoint` 字段在 `create` / `parse` 端点间动态选择。
- 流式路径每个 chunk 即时回调再累积，`_convert_streaming_chunks_to_chat_message` 负责文本拼接、工具调用按 `index` 跨 chunk 累加、usage 倒序查找、首 token 时间记录。
- tool calling 全程"透传"为 `ToolCall` 对象，生成器只解析不执行，自定义工具被显式排除，JSON 解析失败时跳过单个调用而非整体报错。
- `_check_finish_reason` 对 `length` / `content_filter` 主动告警，把回复截断这类隐性问题显式化。
- 非 chat 的 `OpenAIGenerator` 复用 chat 版内部函数，已被标记为 Haystack 3.0 移除，新代码应统一用 ChatGenerator。
- `ChatPromptBuilder` 用 `SandboxedEnvironment` 渲染，靠 Jinja2 AST 自动推断模板变量并动态声明为组件输入端口，必需/可选变量分别有无默认值。
- 渲染用 `dataclasses.replace` 生成新消息，避免污染原始模板，保证 builder 可重复安全调用。
- `AnswerBuilder` 用 `pattern` 抽答案、`reference_pattern` 抽引用，支持 `[6-10]` 范围展开并钳制上限，给文档写入 `source_index` 和 `referenced` 元字段。
- `FallbackChatGenerator` 按序尝试多个生成器、任何异常即回退，透明转发参数、超时完全委托底层，meta 里记录成功索引与失败列表。
- 三段式 builder/generator/builder 解耦的本质：prompt 是数据模板、生成是与文档无关的纯函数、引用抽取是与模型无关的后处理。

## 动手实验

1. **实验一：观察流式 chunk 的拼装** — 用 `OpenAIChatGenerator` 传入一个自定义 `streaming_callback`（参考 `generators/utils.py` 里的 `print_streaming_chunk`），在回调里打印每个 `StreamingChunk` 的 `content`、`index`、`finish_reason`、`meta.get("usage")`。运行后观察：第一个 chunk 是否只有 role、usage 是否出现在最后一个空 `choices` 的 chunk，再对照 `_convert_streaming_chunks_to_chat_message` 缝合出的最终 `ChatMessage.meta`。

2. **实验二：用 ChatPromptBuilder 把检索文档喂进 prompt** — 构造一个 `ChatPromptBuilder(template=[ChatMessage.from_system("你是问答助手"), ChatMessage.from_user("根据文档回答：{% for d in documents %}[{{loop.index}}] {{d.content}}\n{% endfor %}\n问题：{{query}}")], required_variables=["documents", "query"])`，打印 `builder.variables` 与 `builder.required_variables`，再 `run` 一次传入文档和查询，验证渲染后的消息文本，并故意漏传 `query` 观察 `_validate_variables` 抛出的异常信息。

3. **实验三：用 AnswerBuilder 抽取引用** — 仿照源码 docstring 的例子，构造三篇文档和回复 `"法国的首都是巴黎 [2]。"`，用 `AnswerBuilder(reference_pattern=r"\[(\d+)\]", return_only_referenced_documents=False)` 运行，打印每篇文档的 `meta["source_index"]` 与 `meta["referenced"]`；再把回复改成 `"参见 [1-3]。"` 并设 `expand_reference_ranges=True`，验证范围引用被正确展开。

4. **实验四：用 FallbackChatGenerator 模拟回退** — 构造两个生成器，第一个故意用错误的 `api_base_url`（必然失败），第二个为正常配置，包进 `FallbackChatGenerator([bad, good])` 后 `run`。检查返回 `meta` 里的 `successful_chat_generator_index`、`total_attempts`、`failed_chat_generators`，确认回退发生在第 0 个生成器抛异常之后。

> **下一章预告**：本章解决了"召回文档怎么变成答案"，但真实 RAG 流程往往不是一条直线——查询要先分类走不同分支、多路检索结果要汇合、条件不满足要兜底。第 11 章将解构 Haystack 的**路由、汇合与分支控制流**：`ConditionalRouter`、`BranchJoiner`、`Multiplexer` 等组件如何用纯数据组件搭出复杂的非线性 pipeline。
