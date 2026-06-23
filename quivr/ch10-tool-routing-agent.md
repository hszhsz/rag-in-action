# 第 10 章 工具路由与 Agent

前面九章，我们把 Quivr 当作一个"纯 RAG"系统来读：文件被切分、向量化、检索、重排、压缩，最后交给 LLM 生成答案。但真实世界里，向量库里的文档常常不足以回答用户的问题——也许答案在今天的新闻里，也许用户根本不是在提问，而是在下达"以后都用法语回答"这样的指令。Quivr 的回答是：不要为"带工具的 agent"另起炉灶，而是把工具节点当作配置驱动的 LangGraph 图里**可选的节点加一条条件边**。当 context 足以回答，图就走 `generate_rag`；当 context 不足，图就先走 `run_tool` 调用外部工具（如 web search）补充信息，再回到生成节点。

本章钻进 `quivr_core/llm_tools/` 与 `quivr_core/rag/quivr_rag_langgraph.py`，看清三件事：工具是如何被注册、查找、包装成能产出 `Document` 的统一接口的；`tool_routing` 这条条件边如何用结构化输出判断"任务可完成性"并用 LangGraph 的 `Send` 动态分发；以及 Quivr 如何把"指令型交互"叠加在 RAG 之上，让同一张图既能回答问题，又能按对话改变自身行为。

## 工具体系：工厂、类别与 ToolWrapper

工具的入口是 `llm_tools/llm_tools.py` 里的 `LLMToolFactory.create_tool`。它的逻辑极其朴素——遍历 `TOOLS_CATEGORIES`，按工具名查找：

```python
class LLMToolFactory:
    @staticmethod
    def create_tool(tool_name: str, config: Dict[str, Any]) -> Union[ToolWrapper, Type]:
        for category, tools_class in TOOLS_CATEGORIES.items():
            if tool_name in tools_class.tools:
                return tools_class.create_tool(tool_name, config)
            elif tool_name.lower() == category and tools_class.default_tool:
                return tools_class.create_tool(tools_class.default_tool, config)
        raise ValueError(f"Tool {tool_name} is not supported.")
```

`TOOLS_CATEGORIES` 是一个把类别名映射到类别对象的字典，目前注册了两类：`WebSearchTools`（来自 `web_search_tools.py`）和 `OtherTools`（来自 `other_tools.py`）。查找有两条路径：第一条是精确匹配——若 `tool_name` 出现在某类别的 `tools` 列表里（比如 `"tavily"`），直接调用该类别的 `create_tool`；第二条是**"类别名当默认工具"**——若 `tool_name.lower()` 等于类别名（比如 `"web search"`）且该类别声明了 `default_tool`，就回退到默认工具。这意味着用户既可以精确指定 `tavily`，也可以只说"我要 web search"，系统会自动落到默认实现。两条路都走不通就抛 `ValueError`，绝不静默返回 `None`。

`TOOLS_LISTS` 则把所有类别的工具枚举展开成一个扁平字典（`{tool.value: tool ...}`），它的用途在配置校验时会用到（见本章末"工具校验"一节）。

每个类别本身是 `entity.py` 里的 `ToolsCategory`：它带 `name`、`description`、`tools`（枚举列表）、`default_tool` 和一个 `create_tool` 回调。注意构造函数里有一行 `self.name = self.name.lower()`——所以即使 `WebSearchTools` 声明 `name="Web Search"`，类别名最终是小写的 `"web search"`，这正是上面 `tool_name.lower() == category` 能匹配上的原因。

真正让工具能融入 RAG 的，是 `ToolWrapper`：

```python
class ToolWrapper:
    def __init__(self, tool: BaseTool, format_input: Callable, format_output: Callable):
        self.tool = tool
        self.format_input = format_input
        self.format_output = format_output
```

它把一个 langchain `BaseTool` 和两个适配函数捆在一起。`format_input` 把"任务字符串"转成工具需要的入参，`format_output` 把工具的原始返回转成 RAG 流程统一消费的 `List[Document]`。以 `web_search_tools.py` 里的 Tavily 为例，`format_input` 简单地把任务包成 `{"query": task}`，`format_output` 则把 Tavily 返回的每条搜索结果（含 `content` 与 `url`）映射成一个 `Document`，并把 `url` 写进 `file_name`/`original_file_name` 元数据：

```python
def format_output(response: Any) -> List[Document]:
    metadata = {"integration": "", "integration_link": ""}
    return [
        Document(
            page_content=d["content"],
            metadata={
                **metadata,
                "file_name": d["url"] if "url" in d else "",
                "original_file_name": d["url"] if "url" in d else "",
            },
        )
        for d in response
    ]
```

这个设计的精妙之处在于：**工具的产物和向量库的检索产物是同一种类型**——都是 `Document`。于是 `run_tool` 拿到的结果可以无缝喂进后续的重排、压缩与生成节点，工具不需要"特殊待遇"。

默认 web search 工具是 Tavily。`web_search_tools.py` 里 `create_tavily_tool` 从 `config` 或环境变量 `TAVILY_API_KEY` 取密钥，缺失就报错；并设了一组默认参数：`max_results=5`、`search_depth="advanced"`、`include_answer=True`[[Tavily Search API]](https://docs.tavily.com/documentation/api-reference/endpoint/search)。`WebSearchTools` 类别声明 `default_tool=WebSearchToolsList.TAVILY`，与 `config.py` 里的 `DefaultWebSearchTool.TAVILY` 一致。`OtherTools` 目前只有一个成员 `cited_answer`，且没有 `default_tool`，它走的是另一种返回（直接返回一个 langchain 工具/结构体，而非 `ToolWrapper`）。

## tool_routing：判断"可完成性"的条件边

`tool_routing`（`quivr_rag_langgraph.py` 第 544 行起）是这张图里最关键的条件分支。它的职责不是"生成答案"，而是为每个 task 决定：现有的 context 加上 chat history，是否足以把这个 task 完成？如果不够，应该选哪个工具？

它先做一个快路径——若 `tasks` 为空，直接 `Send("generate_rag", state)`。否则用 `collect_tools` 把工作流配置里的工具整理成一段人类可读的描述（"Available tools which can be activated: Tool 1 name: ... description: ..."），作为 `activated_tools` 喂给 prompt。然后对每个 task 并发地调用结构化输出：

```python
for task_id in tasks.ids:
    input = {
        "chat_history": state["chat_history"].to_list(),
        "tasks": tasks(task_id).definition,
        "context": combine_documents(tasks(task_id).docs),
        "activated_tools": validated_tools,
    }
    msg = custom_prompts[TemplatePromptName.TOOL_ROUTING_PROMPT].format(**input)
    async_jobs.append(
        (self.ainvoke_structured_output(msg, TasksCompletion), task_id)
    )
```

结构化输出的目标类是 `TasksCompletion`，它有四个字段，但真正承载决策的是两个布尔/字符串字段：

```python
class TasksCompletion(BaseModel):
    is_task_completable_reasoning: Optional[str] = Field(default=None, ...)
    is_task_completable: bool = Field(
        description="Whether the user task or question can be completed using the provided context and chat history BEFORE any tool is used.",
    )
    tool_reasoning: Optional[str] = Field(default=None, ...)
    tool: Optional[str] = Field(
        description="The tool that shall be used to complete the task.",
    )
```

注意每个决策字段都配了一个 `*_reasoning` 字段——这是一种"让模型先写理由再下结论"的轻量 chain-of-thought：模型在结构化输出里先填 `is_task_completable_reasoning`，再填 `is_task_completable`，约束它的判断有迹可循。

判断的标准写在 `prompts.py` 的 `TOOL_ROUTING_PROMPT` 里。系统消息要求模型逐个 task 考虑，判断 context 和 history 是否"包含完成该 task 所需的全部信息"；如果不够，**只能从下面列出的工具里选**最合适的一个；并反复强调三条护栏：没有工具就原样返回不选工具、没有相关工具就不选、**不得提议未列出的工具**。这层约束保证了路由不会幻想出一个不存在的工具名，从而避免下游 `create_tool` 抛 `ValueError`。

拿到所有响应后，`tool_routing` 把结果写回每个 task：`set_completion` 记录可完成性，若不可完成且模型给了工具就 `set_tool`。最后做分发决策：

```python
if tasks.has_non_completable_tasks():
    send_list.append(Send("run_tool", payload))
else:
    send_list.append(Send("generate_rag", payload))
```

只要存在任何一个不可完成的 task，整批就先去 `run_tool`；否则直接去生成。`has_non_completable_tasks` 的判断依赖 `UserTasks.non_completable_tasks` 属性——它筛出所有 `is_completable()` 为 `False` 的 task。

## Send：LangGraph 的动态分发

`tool_routing` 返回的不是一个普通的状态字典，而是 `List[Send]`。`Send` 是 LangGraph 提供的原语，用于在条件边里**动态地、map 风格地**把负载分发到下游节点，每个 `Send("node_name", payload)` 都会以指定 payload 触发一次目标节点[[LangGraph Send / map-reduce]](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)。这与传统条件边"返回一个字符串选一条边"不同——`Send` 允许在运行时根据数据决定要触发哪些节点、各自带什么状态。

在 Quivr 里，这种能力被用来实现"按需走工具"：图编译时 `tool_routing` 这个节点的条件边声明了它可能连到 `run_tool` 和 `generate_rag`（来自 `WorkflowConfig` 的 `conditional_edge.conditions`），但具体走哪条、带什么 payload，完全由运行时 `tool_routing` 返回的 `Send` 决定。`routing`/`routing_split` 同样用 `Send` 在 `edit_system_prompt` 与 `filter_history` 之间分流。可以说，`Send` 是 Quivr 把"一张静态配置的图"变成"数据驱动的动态路径"的关键齿轮。

`_add_node_edges`（第 1051 行）揭示了边是怎么从配置长出来的：若 `NodeConfig` 声明了 `edges`，就 `add_edge` 加静态边；若声明了 `conditional_edge`，就用 `getattr(self, ...)` 把字符串 `routing_function` 解析成方法（比如 `"tool_routing"`），交给 `workflow.add_conditional_edges`。这正是第 7、8 章讲过的"配置即代码"在工具路由上的延续——加一个工具节点，不需要改图的构建逻辑，只需在 YAML 里加节点和条件边。

## run_tool：执行工具并把结果变成 docs

被 `Send` 选中后，`run_tool`（第 588 行）负责真正执行工具。它只处理"不可完成且有工具"的 task：

```python
for task_id in tasks.ids:
    if not tasks(task_id).is_completable() and tasks(task_id).has_tool():
        tool = tasks(task_id).tool
        tool_wrapper = LLMToolFactory.create_tool(tool, {})
        formatted_input = tool_wrapper.format_input(tasks(task_id).definition)
        async_jobs.append((tool_wrapper.tool.ainvoke(formatted_input), task_id))
```

流程很清晰：用 `LLMToolFactory.create_tool` 实例化（此处 `config` 传空字典，密钥从环境变量取）→ `format_input` 把任务转成工具入参 → `tool.ainvoke` 异步调用工具 → 所有 task 用 `asyncio.gather` 并发等待。回来后再统一 `format_output` 转成 `Document`，并用 `filter_chunks_by_relevance` 按相关性阈值过滤，最后 `set_docs` 存回对应 task：

```python
for response, task_id in zip(responses, task_ids, strict=False):
    _docs = tool_wrapper.format_output(response)
    _docs = self.filter_chunks_by_relevance(_docs)
    tasks.set_docs(task_id, _docs)
```

这里有一处值得留意的实现细节：`tool_wrapper` 是循环里反复赋值的局部变量，`format_output` 阶段用的是最后一次创建的那个 wrapper。当所有不可完成的 task 用同一种工具（典型场景下都是 web search）时，这没有问题；但若不同 task 选了不同工具，`format_output` 的适配可能不匹配——这是当前代码的一个隐含约束。`run_tool` 完成后，task 的 `docs` 字段已经填上了工具产出的文档，状态回流到图里，后续生成节点便能像消费向量检索结果一样消费它们。注意被注释掉的 `if tool not in [...activated_tools]: raise ValueError` 那段——曾经这里要校验工具是否激活，现在改由 `tool_routing` 的 prompt 护栏来保证。

## 结构化输出与优雅降级

`tool_routing` 用的 `ainvoke_structured_output`（第 1175 行）是 Quivr 与各家 LLM 打交道时的"通用变压器"：

```python
async def ainvoke_structured_output(self, prompt, output_class):
    try:
        structured_llm = self.llm_endpoint._llm.with_structured_output(
            output_class, method="json_schema"
        )
        return await structured_llm.ainvoke(prompt)
    except openai.BadRequestError:
        structured_llm = self.llm_endpoint._llm.with_structured_output(output_class)
        return await structured_llm.ainvoke(prompt)
```

它优先用 `with_structured_output(method="json_schema")`，这是基于 JSON Schema 的严格结构化输出，能让模型直接返回符合 pydantic 模型的对象[[LangChain with_structured_output]](https://python.langchain.com/docs/how_to/structured_output/)。但并非所有模型/供应商都支持 `json_schema` 方法，于是 `try/except openai.BadRequestError` 兜底——一旦报错，就退回到不指定 method 的默认结构化输出（通常是工具调用或函数调用模式）。这种**优雅降级**让同一份代码能跑在支持与不支持 json_schema 的模型上，是 Quivr 多供应商策略（第 5 章）在 agent 层的延续。`invoke_structured_output` 是其同步版本，`routing` 节点里也能看到同样的 `try/except` 模式。

## 指令型交互：让 RAG 学会"听从指挥"

到目前为止，我们假设用户输入都是"任务"。但 `SPLIT_PROMPT`、`USER_INTENT_PROMPT`、`UPDATE_PROMPT` 这一组 prompt 揭示了 Quivr 更野心的一面：区分**指令（instruction）**与**任务（task）**。

入口在 `routing`/`routing_split` 节点。它们用 `SPLIT_PROMPT` 把用户输入拆成结构化的 `SplittedInput`：既抽取 `instructions`（让系统改变行为的指令，如"用法语回答""启用 web search"），也抽取 `task_list`（要完成的具体任务）。`SPLIT_PROMPT` 给了很具体的示例，比如把 "What is Apple? Who is its CEO?" 拆成两个独立 task。`USER_INTENT_PROMPT` 则更纯粹地做意图二分类：直接改变系统行为的输入意图是 `prompt`，其余（提问、总结、翻译……）是 `task`。

一旦识别出指令，路由会 `Send("edit_system_prompt", ...)`。`edit_system_prompt`（第 414 行）用 `UPDATE_PROMPT` 把"用户指令 + 当前系统 prompt + 可用工具 + 已激活工具"喂给模型，要求它产出更新后的系统 prompt 和工具激活决策，结构化输出到 `UpdatedPromptAndTools`：

```python
class UpdatedPromptAndTools(BaseModel):
    prompt_reasoning: Optional[str] = Field(default=None, ...)
    prompt: Optional[str] = Field(default=None, ...)
    tools_reasoning: Optional[str] = Field(default=None, ...)
    tools_to_activate: Optional[List[str]] = Field(default_factory=list, ...)
    tools_to_deactivate: Optional[List[str]] = Field(default_factory=list, ...)
```

随后 `update_active_tools` 按返回的 `tools_to_activate`/`tools_to_deactivate`，在 `validated_tools` 与 `activated_tools` 之间增删工具；`self.retrieval_config.prompt` 被替换成更新后的系统 prompt。`UPDATE_PROMPT` 里特别叮嘱："若系统 prompt 引用了某个工具，就把该工具加入已激活列表""若系统 prompt 与用户指令矛盾，删除矛盾陈述"。

这套机制的意义在于：Quivr 不止是一次性问答的 RAG，而是一个**会随对话演化行为与工具集**的助手。用户说一句"以后启用 web search"，系统就把 Tavily 加入激活工具；用户说"用法语回答"，系统就改写自己的 system prompt。RAG 与 agent 在这里真正合流。

## 工具校验：错误即文档

工具名是从配置里来的字符串，难免拼错。`config.py` 的 `WorkflowConfig.validate_available_tools`（第 574 行）在配置加载时就做校验，并把拼写错误变成有用的提示：

```python
def validate_available_tools(self):
    if self.available_tools:
        valid_tools = list(TOOLS_CATEGORIES.keys()) + list(TOOLS_LISTS.keys())
        for tool in self.available_tools:
            if tool.lower() in valid_tools:
                self.validated_tools.append(
                    LLMToolFactory.create_tool(tool, {}).tool
                )
            else:
                matches = process.extractOne(
                    tool.lower(), valid_tools, scorer=fuzz.WRatio
                )
                if matches:
                    raise ValueError(
                        f"Tool {tool} is not a valid ToolsCategory or ToolsList. Did you mean {matches[0]}?"
                    )
                ...
```

`valid_tools` 同时收集了类别名（`TOOLS_CATEGORIES.keys()`，如 `"web search"`）和具体工具名（`TOOLS_LISTS.keys()`，如 `"tavily"`），所以两种写法都合法。合法的工具会被 `LLMToolFactory.create_tool` 实例化并把底层 langchain tool 存进 `validated_tools`。一旦不匹配，就用 `rapidfuzz` 的 `process.extractOne` 做模糊匹配，给出 `Did you mean tavily?` 式的建议。这正是第 7 章反复出现的"错误即文档"理念——校验在配置阶段就发生（`WorkflowConfig.__init__` 里调用），且报错本身就引导用户写对配置，而不是等到运行时调工具才失败。

## 设计意图小结

把这些拼图合起来看，Quivr 的工具与 agent 体系有一条清晰的主线：**统一在一张配置驱动的 LangGraph 图里，用条件边和 `Send` 把"是否用工具"变成数据驱动的运行时决策**。纯 RAG 与带工具的 agent 不是两套系统，而是同一张图的不同路径——`tool_routing` 判断 context 够不够，够就 `generate_rag`，不够就先 `run_tool` 用 web search 补料再生成。工具输出与向量检索输出都是 `Document`，所以工具能无痛接入既有的重排、压缩、生成链路。而指令型交互又在此之上加了一层：系统能随对话改写自己的 prompt、增删自己的工具。配置层的模糊匹配校验则保证这一切从一开始就建立在正确的工具名之上。

## 本章小结

- `LLMToolFactory.create_tool` 按名字在 `TOOLS_CATEGORIES`（`WebSearchTools`/`OtherTools`）里查找工具，支持"精确工具名"与"类别名当默认工具"两条路径，都失败则抛 `ValueError`。
- `ToolWrapper`（`entity.py`）把 langchain `BaseTool` 与 `format_input`/`format_output` 捆在一起，让工具产物统一转成 `Document`，从而无缝接入 RAG 后续链路；默认 web search 工具是 Tavily（`max_results=5`、`search_depth="advanced"`）。
- `tool_routing` 是条件边：用 `TOOL_ROUTING_PROMPT` 加结构化输出 `TasksCompletion` 逐个 task 判断 `is_task_completable`，不可完成时从已激活工具里选 `tool`，prompt 护栏严禁选未列出的工具。
- LangGraph 的 `Send("node", payload)` 实现 map 风格的动态分发：有不可完成 task 就 `Send("run_tool", ...)`，否则 `Send("generate_rag", ...)`，把静态图变成数据驱动的路径。
- `run_tool` 对"不可完成且有工具"的 task 实例化工具、`format_input` → `ainvoke` → `format_output` → `filter_chunks_by_relevance`，用 `asyncio.gather` 并发，再 `set_docs` 回写。
- `ainvoke_structured_output` 优先用 `with_structured_output(method="json_schema")`，捕获 `openai.BadRequestError` 后回退普通结构化输出，实现对不同模型的优雅降级。
- 每个结构化决策字段都配了 `*_reasoning` 字段，约束模型"先写理由再下结论"，是一种轻量 CoT。
- `SPLIT_PROMPT`/`USER_INTENT_PROMPT` 区分指令与任务；指令走 `edit_system_prompt` + `UPDATE_PROMPT`，产出 `UpdatedPromptAndTools`，由 `update_active_tools` 增删激活工具并改写系统 prompt。
- `WorkflowConfig.validate_available_tools` 在配置加载期用 `rapidfuzz` 做模糊匹配，拼错工具名时给出 "Did you mean ...?" 提示，践行"错误即文档"。
- 整体设计意图：纯 RAG 与 agent 统一在同一张配置图里，工具节点只是可选节点加条件边，context 不足时自动补料再生成。

## 动手实验

1. **实验一：读懂可完成性判断** — 打开 `prompts.py`，通读 `TOOL_ROUTING_PROMPT` 的系统消息，逐条对照三条护栏（"没有工具就不选""没有相关工具就不选""不得提议未列出的工具"），再回到 `quivr_rag_langgraph.py` 看 `TasksCompletion` 的 `is_task_completable` 与 `tool` 字段如何被 `set_completion`/`set_tool` 写回，理解从"判断"到"决策"的完整链路。

2. **实验二：跟踪 Send 的动态分发** — 在 `tool_routing` 的 `if tasks.has_non_completable_tasks()` 分支两侧各加一行日志，构造一个向量库里没有答案的问题（如"今天的天气"）并配置启用了 web search，观察图先走 `run_tool` 再 `generate_rag`；再构造一个库里有答案的问题，确认它直接走 `generate_rag`。对照 `_add_node_edges` 理解条件边是如何从配置生成的。

3. **实验三：给 create_tool 传未知工具名** — 在一个 Python REPL 里 `from quivr_core.llm_tools.llm_tools import LLMToolFactory`，调用 `LLMToolFactory.create_tool("googel_search", {})`，观察抛出的 `ValueError: Tool googel_search is not supported.`；再到 `WorkflowConfig` 的 `available_tools` 里填一个拼错的工具名，触发 `validate_available_tools` 的 `rapidfuzz` 模糊匹配，看 "Did you mean tavily?" 提示。

4. **实验四：读 USER_INTENT_PROMPT 区分指令 vs 任务** — 阅读 `USER_INTENT_PROMPT` 与 `SPLIT_PROMPT`，分别用 "Answer in French" 和 "What is Apple? Who is its CEO?" 两个输入，推演 `SplittedInput` 应该如何填充 `instructions` 与 `task_list`；再顺着 `routing` 节点看指令如何 `Send("edit_system_prompt", ...)`，最终在 `update_active_tools` 里改变 `activated_tools`。

> **下一章预告**：工具路由让 Quivr 学会了"补料"，但答案还要一字一句流回用户面前。第 11 章《流式回答、序列化与工程原则收尾》将拆解 `answer_astream` 如何用 `astream_events` 把 LangGraph 的中间步骤与 LLM token 增量解析成 `ParsedRAGChunkResponse`，配置如何被序列化与复现，并把全书贯穿的工程原则——配置即代码、错误即文档、优雅降级、统一抽象——做一次收束。
