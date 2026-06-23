# 第 10 章：Agent 注册表与多模型适配

上一章我们把 RAG 对话链路从输入走到了输出，但那条链路里始终隐含着一个尚未展开的角色：当用户开启工具调用、让模型自己决定"先检索还是先画图"时，到底是谁在驱动这个"思考—行动—观察"的循环？答案是 Agent。问题在于，Langchain-Chatchat 并不假定你用的是 GPT-4 这类原生支持 function-calling 的闭源模型，它要在 ChatGLM3、Qwen 这些开源模型，以及自建平台模型上都能跑 Agent，而这些模型的工具调用能力与输出格式彼此不同：有的靠特殊 token 触发工具、有的走 ReAct 文本协议、有的支持标准的 tool-calling 接口。

于是就有了本章的主角——`agents_registry`。它是一个按 `agent_type` 分发的工厂函数：对每一种模型族，挑选合适的 prompt 模板、合适的 agent 构造器、再包进合适的 executor。读懂它，就读懂了 Chatchat 如何用一套统一的执行骨架，去适配一堆"性格各异"的开源模型。本章源码集中在 `chatchat/server/agents_registry/agents_registry.py` 与 `langchain_chatchat/agents/` 目录下。

## 一个工厂函数，九种 agent_type

整个适配体系的入口是 `agents_registry()` 函数，它的签名很说明问题：

```python
def agents_registry(
        agent_type: str,
        llm: BaseLanguageModel,
        llm_with_platform_tools: List[Dict[str, Any]] = [],
        tools: Sequence[...] = [],
        mcp_tools: Sequence[MCPStructuredTool] = [],
        callbacks: List[BaseCallbackHandler] = [],
        verbose: bool = False,
        **kwargs: Any,
):
```

函数体是一长串 `if/elif` 分支，按 `agent_type` 字符串选择不同的构造路径。把分支列全，就是这个框架支持的全部 Agent 形态：`glm3`、`qwen`、`platform-agent`、`structured-chat-agent`、`default`、`openai-functions`、`openai-tools`、`tool-calling`、`platform-knowledge-mode`。每个分支的模式高度一致，都是三步走：

1. 用 `get_prompt_template_dict("action_model", agent_type)` 取出该类型对应的 prompt 字典；
2. 调用一个 `create_prompt_*` 模板函数把字典变成 `ChatPromptTemplate`；
3. 调用一个 `create_*_agent` 构造器把 `llm + tools + prompt` 拼成 Runnable，再交给 executor。

这种"模板字典 + 模板函数 + agent 构造器 + executor"的四件套，正是适配多模型的关键抽象：模型族之间的差异被压缩进前两步（prompt 怎么写、工具怎么渲染），而执行骨架尽量复用。值得注意的是，几乎所有分支都显式设置了 `return_intermediate_steps=True`，意味着 executor 不止返回最终答案，还会把中间的每一步工具调用都保留下来——这对前文提到的流式回放和会话续接至关重要。

还有一个细节值得先点明：`agents_registry` 自己不创建 `llm`，而是接收一个已经构造好的 `BaseLanguageModel`，外加一个 `llm_with_platform_tools` 列表（平台工具的 OpenAI 工具描述）。也就是说，"模型是谁、绑没绑工具"在进入工厂之前就已确定，工厂只负责"为这个模型套上正确的 agent 外壳"。这层职责切分让同一个工厂可以被不同上游复用，无论 llm 来自本地推理服务还是自建平台。

## 五个 prompt 模板函数的差异

模型适配的第一战场在 prompt。`create_prompt_template.py` 提供了五个模板构造函数，它们都从同一份 `template` 字典里取 `SYSTEM_PROMPT` 与 `HUMAN_MESSAGE`，但拼出来的 `ChatPromptTemplate` 结构各不相同，差异恰好对应不同模型的工具调用范式：

- `create_prompt_glm3_template`：SystemMessage 只需 `tools` 变量，HumanMessage 含 `agent_scratchpad` 与 `input`，**没有**末尾的 scratchpad 占位——scratchpad 被嵌进 human 消息里，符合 ChatGLM3 的对话格式。
- `create_prompt_structured_react_template`：供 `qwen` 与 `structured-chat-agent` 共用，SystemMessage 需要 `tools` 与 `tool_names` 两个变量（ReAct 要在 prompt 里列工具名），且末尾**额外**放一个 `MessagesPlaceholder("agent_scratchpad")`。
- `create_prompt_platform_template`：SystemMessage `input_variables=[]`，不在文本里塞工具，末尾用 scratchpad 占位——标准 tool-calling 范式。
- `create_prompt_gpt_tool_template`：SystemMessage 只需 `tool_names`，供 `openai-functions` 使用。
- `create_prompt_platform_knowledge_mode_template`：SystemMessage 需要 `current_working_directory`、`tools`、`mcp_tools` 三个变量，HumanMessage 还需要 `datetime`，是变量最多、能力最全的模板。

五个函数都为 `chat_history` 声明了相同的 `input_types`（允许 AI/Human/Chat/System/Function/Tool 六类消息），并用 `MessagesPlaceholder(variable_name="chat_history", optional=True)` 把历史以可选方式注入。可见"多轮历史"是所有模型族共享的通用能力，而"工具怎么暴露给模型"才是真正需要按族定制的部分。

## glm3 与 qwen：开源模型的两种 ReAct 方言

`glm3` 分支用 `create_prompt_glm3_template` 建模板，再用 `create_structured_glm3_chat_agent` 建 agent。打开 `glm3_agent.py`，文件头第一句注释就点明了它的身世：这是从 langchain 仓库的 `glm3_agent.py` 改造而来、专门适配 ChatGLM3-6B 的版本。它的特别之处在工具渲染：`render_glm3_json` 把每个工具序列化成 ChatGLM3 能理解的 JSON 结构，字段是 `name`、`description`、`parameters`，而且会把 `tool.description` 按 `" - "` 切开只取后半段作为描述。组装出来的 agent 是一条标准的 Runnable 管道：

```python
agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_platform_tool_messages(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | PlatformToolsAgentOutputParser(instance_type="glm3")
)
```

注意末端的 `PlatformToolsAgentOutputParser(instance_type="glm3")`——同一个解析器类，靠 `instance_type` 区分模型方言。还有那个 `llm_with_stop`：`create_structured_glm3_chat_agent` 默认 `stop_sequence=True`，于是给 LLM 绑定了停止符 `["<|observation|>"]`，这是 ChatGLM3 触发"工具观察"的特殊 token，防止模型自言自语把观察结果也编出来。

`qwen` 分支结构几乎一样，但有两处耐人寻味的差异。第一，进分支第一行就是 `llm.streaming = False`，注释直白地写着 `# qwen agent not support streaming`——这是一个非常典型的"模型特定 workaround"，Qwen 的 ReAct agent 在这套实现里没法稳定流式输出，于是直接关掉。第二，`create_qwen_chat_agent` 的 `render_qwen_json` 渲染风格完全不同，它生成的是一段自然语言描述（`Call this tool to interact with the {name} API...`），停止符也换成了 Qwen 的一组：`["<|endoftext|>", "<|im_start|>", "<|im_end|>", "\nObservation:"]`。同样地，输出解析器是 `PlatformToolsAgentOutputParser(instance_type="qwen")`。两个分支并排看，就能直观感受到"同一种 ReAct 思路，两种模型方言"是什么意思：prompt 用的都是 `create_prompt_structured_react_template`，但工具怎么描述、用什么 token 收尾，全得按模型来。

## platform 系列：面向自建平台模型的标准工具调用

`platform-agent` 分支走的是另一条路。它用 `create_prompt_platform_template` 建模板，用 `create_platform_tools_agent` 建 agent。从 `create_prompt_platform_template` 的结构能看出端倪：它的 SystemMessage 不需要 `tools`/`tool_names` 变量（`input_variables=[]`），而是在消息序列末尾放了一个 `MessagesPlaceholder(variable_name="agent_scratchpad")`。这说明 platform 模型走的是标准的"工具列表通过 API 绑定、scratchpad 以消息形式回传"的 tool-calling 范式，而不是把工具描述塞进 prompt 文本的 ReAct 范式。

更进一步，`platform-knowledge-mode` 分支是 platform 体系里最完整的一个：它的模板 `create_prompt_platform_knowledge_mode_template` 的 SystemMessage 需要三个变量——`current_working_directory`、`tools`、`mcp_tools`，HumanMessage 还要 `datetime`。对应的 `create_platform_knowledge_agent` 构造时额外接收 `mcp_tools` 和 `current_working_directory`（默认 `/tmp`）。这是唯一把 MCP 工具一并纳入 agent 的分支，executor 也是唯一显式传 `mcp_tools=mcp_tools` 的（其余 platform 分支不传）。换句话说，知识模式是 Chatchat 把"本地知识库工具 + 平台内置工具 + MCP 外部工具 + 工作目录"打通的集大成形态。

## 两类 executor：PlatformToolsAgentExecutor 与原生 AgentExecutor

九个分支，executor 其实只有两种。`glm3`、`qwen`、`platform-agent`、`structured-chat-agent`、`platform-knowledge-mode` 用的是自定义的 `PlatformToolsAgentExecutor`；而 `default`、`openai-functions`、`openai-tools`、`tool-calling` 用的是 langchain 原生的 `AgentExecutor`[[AgentExecutor]](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html)。这条分界线大致对应"需要平台工具适配 vs 标准模型直接可用"。

`PlatformToolsAgentExecutor` 定义在 `all_tools_agent.py`，继承自 langchain 的 `AgentExecutor`，但重写了 `_call`、`_acall`、`_perform_agent_action` 等核心方法。在展开它的三个扩展点之前，先看一眼 `_call` 的主循环骨架，它本质上仍是经典的 agent loop：

```python
while self._should_continue(iterations, time_elapsed):
    next_step_output = self._take_next_step(...)
    if isinstance(next_step_output, AgentFinish):
        return self._return(next_step_output, intermediate_steps, ...)
    intermediate_steps.extend(next_step_output)
    ...
    iterations += 1
    time_elapsed = time.time() - start_time
```

循环由 `_should_continue` 用迭代次数与耗时双重把关，每轮 `_take_next_step` 决定"再调一个工具还是收尾"，产出 `AgentFinish` 就返回，否则把这一步追加进 `intermediate_steps` 继续。异步版 `_acall` 还多套了一层 `asyncio_timeout(self.max_execution_time)`，超时则走 `return_stopped_response` 优雅停机。骨架照搬自父类，真正的定制全在下面三处：

第一，**入参里的 intermediate_steps 续接**。`_call` 开头会从 `inputs` 里把 `intermediate_steps` 取出来，并显式从 inputs 字典里 `pop` 掉，确保它不会被当成普通输入二次传给 prompt。这正是上一章会话续接能续上中间步骤的底层支撑。

第二，**MCP 工具的特殊执行路径**。`_perform_agent_action` 里，如果 `agent_action` 是 `MCPToolAction` 实例，会按 `tool.name` 加 `tool.server_name` 双重匹配，从 `self.mcp_tools` 里找到对应工具再执行；找不到就返回一句明确的 not found 字符串而非抛异常。这把 MCP 这种"按服务器命名空间组织"的外部工具，无缝接进了 langchain 的 action 执行框架。

第三，**平台适配工具的特殊调用约定**。当 `agent_action.tool` 命中 `AdapterAllToolStructType` 枚举（如绘图、网页浏览等内置工具）时，调用 `tool.run` 传入的不是普通 `tool_input`，而是 `{"agent_action": agent_action}` 整个动作对象，并标红日志。普通工具才走常规的 `tool.run(agent_action.tool_input)`。

`_acall` 里还有一段专门处理 `DrawingToolAgentAction` 与 `WebBrowserAgentAction` 的逻辑：当本轮产出里没有任何"真实工具"（`exist_tools = 全部工具名 - {WEB_BROWSER, DRAWING_TOOL}`）参与时，遇到绘图或网页动作就直接构造 `AgentFinish` 提前返回。注释解释了原因——langchain 不输出这类适配工具的 message 信息，所以这里强制让它们的结果立即终结循环，避免 agent 继续空转。这是又一个针对"平台工具不完全符合 langchain 假设"打的补丁。

## 标准分支：openai-tools 与 tool-calling 的极简组装

对于原生支持工具调用的模型，组装反而最简洁。`openai-tools` 与 `tool-calling` 合用一个分支，prompt 直接用四条消息硬编码：`SystemMessage(FUNCTIONS_PREFIX)`、`HumanMessagePromptTemplate("{input}")`、`AIMessage(FUNCTIONS_SUFFIX)`、`MessagesPlaceholder("agent_scratchpad")`。其中 `FUNCTIONS_PREFIX` 与 `FUNCTIONS_SUFFIX` 是从 `kwargs` 里取的（`kwargs.get("FUNCTIONS_PREFIX")`），意味着这两个分支的系统提示与收尾语完全由调用方传入，工厂本身不预设任何 ReAct 文本协议——因为这类模型的工具调用走的是 API 层而非 prompt 层。区别仅在 runnable 的构造器：`openai-tools` 用 `create_openai_tools_agent`[[create_openai_tools_agent]](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.openai_tools.base.create_openai_tools_agent.html)，`tool-calling` 用 `create_tool_calling_agent`[[create_tool_calling_agent]](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html)。两者都用 `RunnableMultiActionAgent` 包一层，再交给原生 `AgentExecutor`。`openai-functions` 分支类似，只是它的 prompt 用 `create_prompt_gpt_tool_template`，且额外 `partial` 了 `tool_names`。`default` 分支最特殊：它只放一个 SystemMessage、不带任何工具占位逻辑，对应"单纯对话、不进工具循环"的退化形态，因此它虽然也调 `create_chat_agent`，但 executor 用的是原生 `AgentExecutor` 而非平台版。

## 工厂的上游：PlatformToolsRunnable 如何调用 agents_registry

工厂虽然是本章核心，但它并非凭空被调用。`platform_tools/base.py` 里的 `PlatformToolsRunnable.create_agent_executor` 是工厂最重要的上游：它把 `agents_registry` 作为参数（`agents_registry: Callable`）传进来，自己负责"准备工具、准备回调、准备 MCP 客户端"，再调用工厂拿到 executor。这段预处理逻辑揭示了工具进入 agent 前要过几道手续。

首先是工具分流。`create_agent_executor` 用 `_is_assistants_builtin_tool` 判断每个工具是不是平台内置工具（即 `type` 字段命中 `AdapterAllToolStructType`）：普通工具走 `t.copy(update={"callbacks": final_callbacks})` 复制一份并挂上回调；内置工具则走 `cls.paser_all_tools` 经 `TOOL_STRUCT_TYPE_TO_TOOL_CLASS` 注册表实例化成 `AdapterAllTool`。两类工具合并成 `temp_tools` 才传给工厂。与此并行，所有工具还会经 `_get_assistants_tool`（内部调 `convert_to_openai_tool`）转成 `llm_with_all_tools`，作为 `llm_with_platform_tools` 喂给工厂——这就解释了工厂签名里那个 `llm_with_platform_tools` 参数的来历。

其次是 MCP 客户端的同步获取。`create_agent_executor` 里 `nest_asyncio.apply()` 后，用 `loop.run_until_complete(cls.create_mcp_client(...))` 在同步上下文里启动一个 `MultiServerMCPClient` 并 `get_tools()`，再把这批 `mcp_tools` 一路传给工厂。`create_mcp_client` 还有个细节：对 `transport == "stdio"` 的连接，会把 `env` 补上 `os.environ` 与 `PYTHONHASHSEED=0`，保证子进程环境一致。

最后是入口校验：`if not isinstance(llm, ChatPlatformAI): raise ValueError`——平台 Runnable 只接受 `ChatPlatformAI` 类型的模型。这条断言把"平台 agent 必须配平台模型"这一约束钉死在了代码里。

## 共享的骨架：agent_scratchpad 的统一格式化

读到这里你可能会问：既然 `glm3`、`qwen`、`structured-chat-agent` 这些分支各自有不同的工具渲染和停止符，它们之间还剩下什么是真正共享的？答案藏在每条 Runnable 管道的第一节——`RunnablePassthrough.assign` 里那个 `agent_scratchpad`。

无论 `glm3_agent.py` 还是 `qwen_agent.py`，agent 管道的开头都是同一行：

```python
RunnablePassthrough.assign(
    agent_scratchpad=lambda x: format_to_platform_tool_messages(x["intermediate_steps"]),
)
```

也就是说，"把已经执行过的中间步骤翻译成模型能读的 scratchpad"这件事，被收敛到了同一个 `format_to_platform_tool_messages` 函数里，所有模型族共用。这是适配设计里很重要的一条原则：**差异化只发生在"如何把工具描述给模型"和"如何解析模型的回复"这两端，中间的状态累积逻辑保持统一**。配合 executor 里 `intermediate_steps` 的续接，整套机制保证了不论用哪个模型，多轮工具调用的上下文都以一致的方式被维护。

这也解释了为什么前面强调 `return_intermediate_steps=True` 如此重要：`intermediate_steps` 既是 executor 的输出、又是下一轮 agent 管道的输入，它是连接"执行"与"回放"的同一条数据。

## 设计意图回望：为什么要"按模型族适配 agent"

把九个分支看完，回过头就能理解 `agents_registry` 这套设计要解决的根本矛盾。Agent 的执行骨架——"思考、选工具、跑工具、看结果、再思考"——是模型无关的；但触发工具调用的"接口"却严重依赖具体模型。GPT 系列有原生 function-calling，平台模型有标准 tool-calling 协议，而 ChatGLM3、Qwen 这类开源模型在很多部署场景下只能靠文本里的特殊 token 和 ReAct 约定来"假装"会调工具。

如果不做适配，结果只有两种：要么只支持闭源模型、放弃本地离线部署的初衷；要么为每个模型各写一套 executor，重复且难维护。`agents_registry` 选的是第三条路——**用 `agent_type` 一个维度收敛全部差异，把差异限制在 prompt 模板与工具渲染两端，让执行骨架（`PlatformToolsAgentExecutor` / `AgentExecutor`）和状态累积（`format_to_platform_tool_messages`、`intermediate_steps`）最大化复用**。

这也是为什么本章反复出现"同一个类靠参数区分方言"的模式：`PlatformToolsAgentOutputParser(instance_type=...)`、`create_prompt_structured_react_template` 被 qwen 与 structured-chat 共用。差异被参数化，而不是被复制。对一个要在异构开源生态里跑 Agent 的离线 RAG 系统来说，这正是务实的工程选择。

## 错误即文档：末尾的 ValueError

`agents_registry` 的 `else` 分支抛出 `ValueError`，把全部支持的 `agent_type` 一字不漏地列在错误消息里：

```python
raise ValueError(
    f"Agent type {agent_type} not supported at the moment. Must be one of "
    "'tool-calling', 'openai-tools', 'openai-functions', "
    "'default','ChatGLM3','structured-chat-agent','platform-agent','qwen','glm3'"
)
```

这是一种朴素但实用的自文档化手法——使用者传错类型时，报错信息本身就是一份"支持清单"。（细看会发现这份清单里写的是 `'ChatGLM3'`，而真正命中的分支用的是 `'glm3'`，且 `platform-knowledge-mode` 也未列入，提示词与实际分发存在轻微不同步，读源码时不能只信注释和错误文案。）

## 回调驱动的状态流：AgentStatus 与流式桥接

把 agent 的执行过程"播报"出去，靠的是 `agent_callback_handler.py` 里的 `AgentExecutorAsyncIteratorCallbackHandler`。它定义了一套 `AgentStatus` 整型状态码：`chain_start=0`、`llm_start=1`、`llm_new_token=2`、`llm_end=3`、`agent_action=4`、`agent_finish=5`、`tool_require_approval=6`、`tool_start=7`、`tool_end=8`、`error=-1`、`chain_end=-999`。每个 langchain 回调钩子（`on_llm_start`、`on_tool_start`、`on_agent_action`...）都把事件序列化成带 `status` 字段的 JSON，塞进一个 `asyncio.Queue`。

这套回调与上一节的 `PlatformToolsRunnable`（`platform_tools/base.py`）配合，构成了完整的流式桥接：`PlatformToolsRunnable.invoke` 里用 `asyncio.create_task` 启动 `agent_executor.ainvoke`，然后 `async for chunk in self.callback.aiter()` 逐条消费队列，按 `status` 把 JSON 转成 `PlatformToolsLLMStatus`、`PlatformToolsAction`、`PlatformToolsActionToolStart/End`、`PlatformToolsFinish`、`PlatformToolsApprove` 等强类型对象往外 `yield`。`on_llm_new_token` 里还做了一件细活：遇到 `\nAction:`、`\nObservation:`、`<|observation|>` 这些特殊 token 时会切断，只把动作前的文本作为可见 token 推送，避免把 ReAct 的内部协议词泄露给用户。

另外 `on_chain_end` 会把 `intermediate_steps` 从 outputs 里抽出来单独存到 `self.intermediate_steps`，`invoke` 结束时再 `extend` 回 `PlatformToolsRunnable` 的 `intermediate_steps` 并追加历史消息——这就闭合了"执行—回放—续接"的整个环路。`tool_require_approval` 状态和 `ApprovalMethod`（`CLI`/`BACKEND`）则预留了工具调用前人工审批的钩子，当前 `on_tool_start` 里的审批逻辑被注释掉了，是个尚未启用的扩展点。

## 本章小结

1. `agents_registry()` 是一个按 `agent_type` 分发的工厂函数，支持 `glm3`、`qwen`、`platform-agent`、`structured-chat-agent`、`default`、`openai-functions`、`openai-tools`、`tool-calling`、`platform-knowledge-mode` 九种形态。
2. 每个分支统一遵循"模板字典 → `create_prompt_*` 模板函数 → `create_*_agent` 构造器 → executor"四件套，把模型族差异压缩进 prompt 与工具渲染两步。
3. 几乎所有分支都设置 `return_intermediate_steps=True`，为流式回放与会话续接保留完整中间步骤。
4. `glm3` 用 `render_glm3_json` 输出 JSON 工具描述、停止符 `<|observation|>`；`qwen` 用 `render_qwen_json` 输出自然语言描述、停止符含 `<|im_end|>` 等，二者是 ReAct 的两种模型方言。
5. `qwen` 分支进入时强制 `llm.streaming = False`，是显式的"模型不支持流式"的 workaround。
6. 同一个 `PlatformToolsAgentOutputParser` 靠 `instance_type`（`glm3`/`qwen`）区分模型输出格式。
7. `platform-knowledge-mode` 是唯一同时纳入本地工具、MCP 工具与工作目录的分支，是平台能力的集大成形态。
8. executor 分两类：平台适配分支用自定义 `PlatformToolsAgentExecutor`，标准模型分支用 langchain 原生 `AgentExecutor`。
9. `PlatformToolsAgentExecutor` 重写执行逻辑以支持 `intermediate_steps` 续接、`MCPToolAction` 双键匹配、`AdapterAllToolStructType` 工具的特殊调用约定，以及绘图/网页工具的提前 `AgentFinish`。
10. `else` 分支的 `ValueError` 把支持类型列进错误消息，是"错误即文档"，但其文案与实际分支名存在轻微不同步。
11. `AgentExecutorAsyncIteratorCallbackHandler` 用整型 `AgentStatus` 状态码把执行事件序列化进队列，经 `PlatformToolsRunnable` 转成强类型对象流式输出，并过滤掉 ReAct 协议词。

## 动手实验

1. **实验一：枚举支持的 agent_type** — 阅读 `agents_registry.py`，把九个分支的 `agent_type` 字符串、对应的 `create_prompt_*` 模板函数、`create_*_agent` 构造器、executor 类型整理成一张四列对照表；再特意用一个不存在的类型（如 `"foo"`）触发 `else` 分支，对比错误消息里列出的类型与实际分支名，找出 `ChatGLM3` 与 `glm3`、缺失的 `platform-knowledge-mode` 这两处不同步。
2. **实验二：对比 glm3 与 qwen 的工具渲染** — 并排阅读 `glm3_agent.py` 的 `render_glm3_json` 与 `qwen_agent.py` 的 `render_qwen_json`，用一个含两个参数的示例工具，手写出两个函数各自会生成的字符串，体会 JSON 结构与自然语言描述两种风格的差异；再记下两个 `create_*_chat_agent` 默认绑定的 `stop` 序列分别是什么。
3. **实验三：追踪 PlatformToolsAgentExecutor 的三个扩展点** — 在 `all_tools_agent.py` 的 `_perform_agent_action` 中定位 `MCPToolAction` 分支、`AdapterAllToolStructType` 分支、`InvalidTool` 兜底分支，用注释标出每条分支的触发条件与传给 `tool.run` 的参数形态有何不同；再看 `_acall` 中绘图/网页工具提前返回 `AgentFinish` 的条件 `exist_tools`，写出它是如何计算的。
4. **实验四：画出状态流转图** — 读 `agent_callback_handler.py` 的 `AgentStatus` 与各 `on_*` 钩子，再读 `platform_tools/base.py` 中 `PlatformToolsRunnable.invoke` 的 `async for` 分支，画一张从 `chain_start` 到 `chain_end` 的状态流转图，标注每个 `status` 会被转换成哪个 `PlatformTools*` 输出对象；特别关注 `on_llm_new_token` 是如何处理 `<|observation|>` 等特殊 token 的。

> **下一章预告**：本章我们看到 agent 被反复传入一个 `tools` 序列，却一直没追问这些工具从何而来、如何被发现与注册。第 11 章《工具系统 Tools Factory》将深入 Chatchat 的工具工厂，看它如何把知识库检索、搜索引擎、代码执行等能力统一封装成 agent 可调用的 `BaseTool`，以及配置如何驱动工具的动态装载。
