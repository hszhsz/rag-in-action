# 第 13 章 Agent 系统

第 12 章我们把 Workflow 拆成了"事件驱动的步骤图":每个 `@step` 消费一种事件、产出另一种事件,`Context` 充当跨步骤的共享状态与事件总线。那一章留下了一个悬念——既然 Workflow 能编排任意异步步骤,那么"会自己思考、自己挑工具、看到结果再决定下一步"的 Agent,是不是也能用同一套机制表达出来?

答案是肯定的,而且这正是 LlamaIndex 新一代 Agent 的设计哲学。**一个 Agent,本质上就是一个预置了"推理 → 调用工具 → 观察结果 → 再推理"循环的 Workflow**。在 `llama_index/core/agent/workflow/` 目录下,`BaseWorkflowAgent` 直接同时继承 `Workflow` 和 Pydantic 的 `BaseModel`,把这个循环固化成了几个标准的 `@step`;而 `FunctionAgent`、`ReActAgent`、`CodeActAgent` 只需要填空式地实现三个抽象方法,就得到了一个完整可运行、可流式、可被多智能体编排的 Agent。本章我们就沿着源码,把这个"Agent 即 Workflow"的等式从里到外讲透,并重点落到 RAG 与 Agent 的交汇点——`QueryEngineTool`。

## Agent 即 Workflow:`BaseWorkflowAgent` 如何复用事件循环

打开 `agent/workflow/base_agent.py`,第一眼就是那个意味深长的类定义:

```python
class BaseWorkflowAgent(
    Workflow, BaseModel, PromptMixin, metaclass=BaseWorkflowAgentMeta
):
    """Base class for all agents, combining config and logic."""
```

它用 `BaseWorkflowAgentMeta`(组合了 `WorkflowMeta` 与 Pydantic 的 `ModelMetaclass`)解决了多继承的元类冲突,从而让 Agent 既是一个可校验配置的 Pydantic 模型(字段如 `name`、`description`、`system_prompt`、`tools`、`llm`、`system_prompt`),又是一个能被 `.run()` 驱动的 Workflow。配置即字段:`llm` 默认通过 `get_default_llm()` 取 `Settings.llm`(呼应第 2 章的全局 `Settings`),`tools` 字段带一个 `field_validator`,会把传入的普通 callable 自动包成 `FunctionTool.from_defaults(tool)`,并禁止使用保留名 `handoff`。

真正体现"Agent = 预置循环的 Workflow"的,是 `base_agent.py` 里那一串 `@step`。它们串成了一条闭环的事件流水线:

- `init_run`(消费 `AgentWorkflowStartEvent`)→ `_init_context` 初始化 memory、state、`max_iterations`、`num_iterations`,把用户消息写入 memory,产出 `AgentInput`;
- `setup_agent`(消费 `AgentInput`)→ 拼上 `system_prompt`,必要时用 `state_prompt` 把当前 state 注入最后一条消息,产出 `AgentSetup`;
- `run_agent_step`(消费 `AgentSetup`)→ 调 `self.take_step(...)` 拿到 `AgentOutput`,这一步正是各类 Agent 的差异所在;
- `parse_agent_output`(消费 `AgentOutput`)→ 累加 `num_iterations`,判断是否到达上限;若 `ev.tool_calls` 为空就调 `finalize` 并 `StopEvent` 收尾,否则为每个 tool call `ctx.send_event(ToolCall(...))`;
- `call_tool`(消费 `ToolCall`)→ 按名查工具、执行、包成 `ToolCallResult`;
- `aggregate_tool_results`(消费 `ToolCallResult`)→ 用 `ctx.collect_events` 等齐本轮所有结果,调 `handle_tool_call_results` 回填,再产出一个新的 `AgentInput`,**回到循环开头**。

这条 `AgentInput → AgentSetup → AgentOutput →(ToolCall → ToolCallResult)→ AgentInput` 的回路,就是 Agent 循环。它完全用第 12 章的事件机制实现:循环的"返回上一步"不是 `while` 语句,而是 `aggregate_tool_results` 再次发出 `AgentInput`。

循环的"刹车"由两处常量与字段控制:`DEFAULT_MAX_ITERATIONS = 20` 是默认最大迭代数;`early_stopping_method` 字段(`Literal["force", "generate"]`,默认 `"force"`)决定到顶后是抛 `WorkflowRuntimeError`,还是用 `DEFAULT_EARLY_STOPPING_PROMPT` 再做一次 LLM 调用生成兜底回答(`_generate_early_stopping_response`)。

`BaseWorkflowAgent` 把"差异点"收敛成三个 `@abstractmethod`:

```python
@abstractmethod
async def take_step(self, ctx, llm_input, tools, memory) -> AgentOutput: ...
@abstractmethod
async def handle_tool_call_results(self, ctx, results, memory) -> None: ...
@abstractmethod
async def finalize(self, ctx, output, memory) -> AgentOutput: ...
```

`take_step` 负责"怎么推理、怎么决定要不要调工具",`handle_tool_call_results` 负责"怎么把观察结果记回上下文",`finalize` 负责"怎么把这一轮的临时记录落进 memory"。整个循环骨架是公用的,子类只需填这三块。这就是为什么 `FunctionAgent`、`ReActAgent` 的文件都很短——骨架已在基类。

## 工具抽象:`BaseTool`、`FunctionTool` 与从签名自动生成 schema

Agent 的"手"是工具。工具侧在 `llama_index/core/tools/types.py`。`BaseTool` 继承自 `DispatcherSpanMixin`(为第 14 章的可观测性埋点),只要求两件事:一个 `metadata` 属性(`ToolMetadata`)和一个 `__call__`。`ToolMetadata` 是一个 `@dataclass`,持有 `description`、`name`、`fn_schema`(默认 `DefaultToolFnSchema`,只有一个 `input: str` 字段)、`return_direct`。它的关键方法是把 schema 转成各厂商格式:`get_parameters_dict()` 从 `fn_schema.model_json_schema()` 抽出 `type/properties/required/$defs`;`to_openai_tool()` 进一步包成 OpenAI 的 `{"type": "function", "function": {...}}` 结构,并对超过 1024 字符的 description 报错。`fn_schema_str` 则把参数 schema 序列化成 JSON 字符串——ReAct 的提示里会用到它。

`ToolOutput` 是工具的统一返回壳:它内部存的是 `blocks: List[ContentBlock]`(多模态友好),但提供 `content` 属性把所有 `TextBlock` 拼成字符串;还带 `is_error` 标志和 `raw_input`/`raw_output`。基类执行工具出错时(`base_agent.py` 的 `_call_tool`)正是构造一个 `is_error=True` 的 `ToolOutput`,而非直接抛出——让错误也能作为"观察"回到 LLM 自我修复。

`AsyncBaseTool` 在 `BaseTool` 之上加了 `call`/`acall` 抽象,`adapt_to_async_tool` 用 `BaseToolAsyncAdapter` 把同步工具透明转异步(`asyncio.to_thread`)。基类的 `_ensure_tools_are_async` 保证所有工具进循环前都是异步的。

最常用的工具是 `function_tool.py` 里的 `FunctionTool`——它把一个普通 Python 函数变成工具。核心魔法在 `from_defaults`:

- `name` 取自函数名 `fn.__name__`,`description` 默认拼成 `"{name}{签名}\n" + docstring`;
- 用 `inspect.signature` 取参数,`create_schema_from_function(...)` 从类型注解自动生成 Pydantic 的 `fn_schema`(即工具的 JSON schema);
- `extract_param_docs` 解析 docstring(支持 Sphinx `:param:`、Google、Javadoc 三种风格),把参数说明回填到 schema 字段的 `description`。

也就是说,你写一个带类型标注和 docstring 的函数,`FunctionTool` 就能自动产出 LLM function calling 所需的全部 schema,无需手写。`FunctionTool` 还有一个 RAG/Agent 协作的细节:它会用 `_is_context_param` 检测函数签名里是否有 `Context` 类型参数(`requires_context`/`ctx_param_name`),如果有,基类 `_call_tool` 会把当前 Workflow 的 `ctx` 注入进去——这让工具能读写 Agent 的共享 state,`handoff` 工具正是靠它实现的。

## `QueryEngineTool`:把 RAG 查询引擎包成一个工具

这是本章与前面整本"RAG 部分"的交汇点。第 11 章我们构建了 `QueryEngine`(检索 + 响应合成);现在 `tools/query_engine.py` 里的 `QueryEngineTool` 只做一件事——**把一个 `BaseQueryEngine` 包成 `AsyncBaseTool`**,从而让 Agent 可以"调用 RAG"作为它的一个动作:

```python
class QueryEngineTool(AsyncBaseTool):
    def __init__(self, query_engine, metadata, resolve_input_errors=True): ...

    async def acall(self, *args, **kwargs) -> ToolOutput:
        query_str = self._get_query_str(*args, **kwargs)
        response = await self._query_engine.aquery(query_str)
        return ToolOutput(
            content=str(response),
            tool_name=self.metadata.get_name(),
            raw_input={"input": query_str},
            raw_output=response,
        )
```

它的 `from_defaults` 给了默认 `DEFAULT_NAME = "query_engine_tool"` 和一段默认描述("Useful for running a natural language query against a knowledge base...")。`_get_query_str` 兼容三种入参:位置参数取第一个、kwargs 里取 `input` 键(对应 `DefaultToolFnSchema` 的 `input` 字段)、否则在 `resolve_input_errors=True` 时把整个 kwargs 转字符串兜底。注意 `raw_output` 里**保留了完整的 `response` 对象**,因此 source nodes、引用等信息没有丢,只是 `content` 给 LLM 看的是字符串。

由此,RAG 在 Agent 视角下退化成"一个能回答自然语言问题的工具"。一个常见的 Agentic RAG 模式就是:给 `FunctionAgent` 配多个 `QueryEngineTool`(每个对应一个知识库),由 LLM 自行决定查哪个、查几次、怎么综合。与之并列的还有 `retriever_tool.py` 的 `RetrieverTool`,它包的是 `BaseRetriever`(第 9 章),`acall` 里 `aretrieve` 后还会跑 `node_postprocessors`,把检出的节点用 `get_content(MetadataMode.LLM)` 拼成文本返回——区别在于 `QueryEngineTool` 返回"答案",`RetrieverTool` 返回"原始相关文档"。

## `FunctionAgent`:基于原生 function calling 的循环

`agent/workflow/function_agent.py` 里的 `FunctionAgent` 是当模型支持原生 function calling 时的首选。它的 `take_step` 第一行就硬性校验 `self.llm.metadata.is_function_calling_model`,否则报错。核心调用是把工具直接交给 LLM:

```python
return await self.llm.achat_with_tools(
    chat_history=current_llm_input,
    allow_parallel_tool_calls=self.allow_parallel_tool_calls,
    tools=tools,
)
```

`allow_parallel_tool_calls` 默认 `True`,允许一轮并行多调；`initial_tool_choice` 可强制第一轮调某个工具。拿到响应后用 `self.llm.get_tool_calls_from_response(...)` 解析出结构化的 `tool_calls`(`ToolSelection` 列表),封进 `AgentOutput` 返回。这里有一个 `scratchpad`(`scratchpad_key = "scratchpad"`):`take_step` 把 LLM 的回复 append 进 scratchpad,`handle_tool_call_results` 把每个工具结果包成 `role="tool"` 的 `ChatMessage`(带 `tool_call_id`)append 进 scratchpad,`finalize` 再 `memory.aput_messages(scratchpad)` 一次性落库并清空。换句话说,**循环中的中间步骤先攒在 scratchpad,只有收尾时才写进长期 memory**,避免半截对话污染历史。流式路径 `_get_streaming_response` 则边收边发 `AgentStream` 事件,同时实时解析 tool_calls。

`FunctionAgent` 的优雅之处:它几乎不关心"提示词长什么样",工具 schema 怎么传、tool_calls 怎么解析,全部委托给 LLM 集成层的 `achat_with_tools` / `get_tool_calls_from_response`。这是"模型懂 function calling"带来的红利。

## `ReActAgent`:用文本协议(Thought/Action/Observation)模拟工具调用

不是所有模型都支持原生 function calling。`agent/workflow/react_agent.py` 的 `ReActAgent` 用纯文本协议补上这一缺口:让 LLM 输出固定格式的 `Thought / Action / Action Input`,再由程序解析出"要调哪个工具、传什么参数"。

它的 `take_step` 不调 `achat_with_tools`,而是用普通的 `achat`/`astream_chat`,关键在输入与输出两端各加一层翻译:

- **输入端 `formatter`(`ReActChatFormatter`)**:`format` 把工具描述塞进系统提示。`get_react_tool_descriptions` 为每个工具拼出 `Tool Name / Tool Description / Tool Args`(`Tool Args` 用的就是 `metadata.fn_schema_str`),再连同 `tool_names` 填进 `system_header`(默认模板来自 `templates/system_header_template.md`,经 `prompts.py` 替换 `{context_prompt}` 得到 `REACT_CHAT_SYSTEM_HEADER` / `CONTEXT_REACT_CHAT_SYSTEM_HEADER`)。历史推理被还原成交替的 assistant(Thought/Action)与 user/tool(Observation)消息,`observation_role` 默认 `MessageRole.USER`。
- **输出端 `output_parser`(`ReActOutputParser`)**:`parse` 用正则在行首匹配 `Thought:`/`Action:`/`Answer:`。规则是"Action 优先于 Answer";命中 Action 时 `parse_action_reasoning_step` 用 `extract_tool_use` 的正则抽出 thought/action/action_input,并用 `dirtyjson` 容错解析 JSON 参数(弱模型常产出不规范 JSON);命中 Answer 时产出 `ResponseReasoningStep`(`is_done = True`);三者都没有则当作"模型直接给了答案"。

解析出的步骤是 `agent/react/types.py` 里的 `ReasoningStep` 体系:`ActionReasoningStep`(thought/action/action_input,`is_done=False`)、`ObservationReasoningStep`(observation,`is_done` 取决于 `return_direct`)、`ResponseReasoningStep`(thought/response,`is_done=True`)。`take_step` 若拿到 `ActionReasoningStep`,就自己造一个 `ToolSelection`(`tool_id` 用 `uuid4`)放进 `AgentOutput.tool_calls`——于是**对基类循环而言,ReAct 和 Function 产出的 `AgentOutput` 结构完全一致**,后续 `call_tool`/`aggregate_tool_results` 无需区分两者。

`ReActAgent` 的容错也值得一提:若 LLM 输出为空或解析 `ValueError`,`take_step` 不会崩溃,而是返回带 `retry_messages` 的 `AgentOutput`,把格式说明回灌给 LLM 让它重试(基类 `parse_agent_output` 看到 `retry_messages` 就重新组 `AgentInput`)。`handle_tool_call_results` 把工具结果包成 `ObservationReasoningStep` 追加到 `current_reasoning`;`finalize` 在末尾是 `ResponseReasoningStep` 时把整段推理写入 memory,并把响应里多余的 `Answer:` 前缀去掉。

## 两种 Agent 的取舍

源码本身给出了选择标准。`AgentWorkflow.from_tools_or_functions` 里有一行决断:

```python
agent_cls = FunctionAgent if llm.metadata.is_function_calling_model else ReActAgent
```

结论清晰:**模型支持原生 function calling 就用 `FunctionAgent`,否则退化到 `ReActAgent`**。`FunctionAgent` 把工具 schema 与解析交给模型/集成层,提示更短、解析更稳、天然支持并行调用;`ReActAgent` 用文本协议把工具能力"喂"给任意聊天模型,通用性强,代价是提示更长、解析依赖正则与 `dirtyjson` 容错、对模型遵循格式的能力更敏感。`agent/workflow/codeact_agent.py` 的 `CodeActAgent` 则是第三条路:动作不是结构化 tool call,而是 `<execute>...</execute>` 包裹的 Python 代码(`EXECUTE_TOOL_NAME = "execute"`),用 `code_execute_fn` 真正运行——适合需要组合计算的场景,但需要可信的代码沙箱。

## `AgentWorkflow`:多智能体编排与 handoff

单个 Agent 已是 Workflow;`agent/workflow/multi_agent_workflow.py` 的 `AgentWorkflow` 再上一层,把**多个 Agent 编排进同一个 Workflow**,并支持它们之间移交控制权(handoff)。它本身也是一个 `Workflow + PromptMixin`,`@step` 流水线与 `BaseWorkflowAgent` 几乎一一对应,区别在于每一步都从 `ctx.store` 取 `current_agent_name` 找到当前 agent,再委托其 `take_step`/`handle_tool_call_results`/`finalize`。

构造时要求:多 agent 时每个都必须有非默认的 `name` 与 `description`;必须指定一个 `root_agent`(单 agent 时自动取该 agent),它就是对话入口。handoff 机制是这样落地的:`_get_handoff_tool` 为当前 agent 动态生成一个名为 `handoff` 的 `FunctionTool`(`return_direct=True`),其描述由 `DEFAULT_HANDOFF_PROMPT` 填入"当前可用 agents"(被 `can_handoff_to` 过滤——只列出本 agent 允许移交的目标)。当 LLM 选择 `handoff` 工具,模块级的 `handoff(ctx, to_agent, reason)` 协程会校验目标合法性,然后 `ctx.store.set("next_agent", to_agent)`。在 `aggregate_tool_results` 里,读到 `next_agent` 就把 `current_agent_name` 切换过去;并且对 `return_direct` 的 handoff 做了特判——`if return_direct_tool.tool_name != "handoff"` 才 `StopEvent`,**handoff 不终止系统,而是带着切换后的 agent 继续循环**。这样多个专精 agent 就能像接力一样协作。`from_tools_or_functions` 还提供了"一个工具集 → 单 agent workflow"的便捷构造。

## Agent 事件流:呼应第 12 章与第 14 章

`agent/workflow/workflow_events.py` 定义了 Agent 运行时对外发出的事件,全部继承第 12 章的 `Event`/`StartEvent`:`AgentInput`、`AgentSetup`、`AgentStream`(流式增量,含 `delta`/`response`/`tool_calls`/`thinking_delta`)、`AgentStreamStructuredOutput`、`AgentOutput`(含 `response`/`tool_calls`/`retry_messages`/`structured_response`)、`ToolCall`、`ToolCallResult`、以及作为入口的 `AgentWorkflowStartEvent`。基类在多个关键节点 `ctx.write_event_to_stream(...)`:流式生成时持续吐 `AgentStream`,决策完吐 `AgentOutput`,调工具前后吐 `ToolCall` 与 `ToolCallResult`。

这正是第 12 章 streaming 能力的直接受益者——调用方用 `handler.stream_events()` 就能实时拿到 Agent 的思考增量、工具调用、工具结果,做出"边想边显示"的交互。而工具基类继承 `DispatcherSpanMixin`、事件全程贯穿,又为下一章的 instrumentation(Span/Event 可观测性)铺好了路:Agent 的每一步推理与每一次工具调用,天然都是可被观测、可被追踪的事件。

## 本章小结

- 新一代 Agent 全面建在第 12 章的 Workflow 之上:`BaseWorkflowAgent` 同时继承 `Workflow` 与 Pydantic `BaseModel`,用 `BaseWorkflowAgentMeta` 解决元类冲突,Agent 既是配置模型又是可运行 Workflow。
- Agent 循环由一组 `@step` 事件流水线构成:`init_run → setup_agent → run_agent_step → parse_agent_output →(ToolCall → call_tool → aggregate_tool_results)→` 再次 `AgentInput`,"返回循环"是再次发事件而非 `while`。
- 循环的边界由 `DEFAULT_MAX_ITERATIONS = 20` 与 `early_stopping_method`(`"force"` 抛错 / `"generate"` 兜底生成)控制;差异点收敛为 `take_step`、`handle_tool_call_results`、`finalize` 三个抽象方法。
- 工具抽象层:`BaseTool`/`ToolMetadata`/`ToolOutput` 统一接口;`FunctionTool.from_defaults` 用 `inspect.signature` + `create_schema_from_function` + docstring 解析,从函数自动生成工具名、描述与 JSON schema。
- `QueryEngineTool` 把第 11 章的 `QueryEngine` 包成 `AsyncBaseTool`,`acall` 内部 `aquery` 并返回 `ToolOutput`(`raw_output` 保留完整 response),这是 RAG 与 Agent 的交汇点;`RetrieverTool` 则返回原始相关文档。
- `FunctionAgent` 走原生 function calling:`achat_with_tools` + `get_tool_calls_from_response`,中间步骤攒在 `scratchpad`,`finalize` 时才落 memory;要求 `is_function_calling_model`。
- `ReActAgent` 用文本协议:`ReActChatFormatter` 把工具描述写进系统提示,`ReActOutputParser` 正则解析 `Thought/Action/Action Input`(`dirtyjson` 容错),产出与 Function 一致的 `AgentOutput`,并用 `retry_messages` 自我纠错。
- 选择标准由源码 `from_tools_or_functions` 给定:支持 function calling 用 `FunctionAgent`,否则用 `ReActAgent`;`CodeActAgent` 以执行代码作为动作。
- `AgentWorkflow` 编排多 agent:`root_agent` 为入口,动态生成 `handoff` 工具(受 `can_handoff_to` 约束),通过 `next_agent` 切换控制权且 handoff 不终止系统。
- Agent 全程发出 `AgentInput/AgentOutput/AgentStream/ToolCall/ToolCallResult` 等事件,既支撑流式交互,又为第 14 章可观测性奠基。

## 动手实验

1. **实验一:画出 Agent 循环的事件图** — 阅读 `/tmp/llama_explore/llama-index-core/llama_index/core/agent/workflow/base_agent.py`,找出全部带 `@step` 的方法,记录每个方法消费的事件类型与返回的事件类型(注意 `parse_agent_output` 的 `Union[StopEvent, AgentInput, ToolCall, None]`),把它们连成一张闭环图,标出"回到循环开头"的那条边来自 `aggregate_tool_results` 返回的 `AgentInput`。
2. **实验二:验证 FunctionTool 自动生成 schema** — 在 `function_tool.py` 的 `from_defaults` 中,梳理从 `fn.__name__`、`inspect.signature`、`create_schema_from_function` 到 `extract_param_docs` 的步骤;再对照 `tools/types.py` 的 `ToolMetadata.to_openai_tool` 与 `get_parameters_dict`,说明一个带类型注解和 docstring 的函数如何变成 OpenAI function 格式,并指出 1024 字符 description 的限制在哪一行触发。
3. **实验三:对比 Function 与 ReAct 的 take_step** — 并排阅读 `function_agent.py` 与 `react_agent.py` 的 `take_step`,找出前者调用 `self.llm.achat_with_tools` 而后者调用 `self.llm.achat` 的差异;再到 `react/output_parser.py` 跟踪 `extract_tool_use` 的正则与 `dirtyjson` 容错,理解 ReAct 如何把文本解析成与 Function 同构的 `AgentOutput.tool_calls`。
4. **实验四:拆解多智能体 handoff** — 在 `multi_agent_workflow.py` 中定位模块级 `handoff` 协程、`_get_handoff_tool`(注意 `return_direct=True` 与 `can_handoff_to` 过滤),以及 `aggregate_tool_results` 里 `next_agent` 的读取与 `tool_name != "handoff"` 的特判;并结合 `prompts.py` 的 `DEFAULT_HANDOFF_PROMPT` / `DEFAULT_HANDOFF_OUTPUT_PROMPT`,说明控制权移交为什么不会终止整个 Workflow。

> **下一章预告**:第 14 章将收尾于 instrumentation 可观测性——深入 `Dispatcher`/`Span`/`Event` 体系,看 LlamaIndex 如何为本章 Agent 的每一步推理与工具调用、乃至前面各章的检索与合成自动埋点;并跨越三部书归纳出可迁移的 RAG 工程原则,为整本《RAG in Action》画上句号。
