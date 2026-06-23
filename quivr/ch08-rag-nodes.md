# 第 8 章 RAG 节点详解

第 7 章我们看清了 Quivr 的核心引擎 `QuivrQARAGLangGraph` 如何把一份 `WorkflowConfig` 编译成一张 LangGraph 状态图：`_build_workflow` 遍历配置里的节点逐个 `add_node`，再按 `edges`/`conditional_edge` 连线，最终 `workflow.compile()` 得到可执行图。配置只是骨架，真正的血肉是图上每一个**节点函数**。本章我们就钻进 `quivr_core/rag/quivr_rag_langgraph.py`，沿着默认 RAG 图的执行顺序，把 `routing`、`filter_history`、`rewrite`、`retrieve`、`generate_rag`/`generate_chat_llm` 逐个拆开，看清一次用户提问是如何在节点之间流动、被改写、被检索、被生成的。

理解这些节点的关键，是先抓住两条主线：一是**流经全图的状态对象** `AgentState`，每个节点都是 `(state: AgentState) -> AgentState` 的纯函数，靠 `{**state, ...}` 做增量更新；二是 Quivr 特有的**多任务模型** `UserTasks`，它让"一句话里的多个问题"得以被拆开、并发检索、统一回答。把这两条主线和 `rag/prompts.py` 里集中管理的 prompt 工厂串起来，整张 RAG 图的设计意图就清晰了。

## 状态模型：AgentState 与节点契约

`AgentState` 是一个 `TypedDict`，定义了所有节点共享的"传送带"。它的字段包括 `messages`、`reasoning`、`chat_history`、`files`、`tasks`、`instructions`，以及一组面向 Zendesk 等场景的 `ticket_metadata`/`user_metadata`/`enforced_system_prompt`，还有检索过滤器 `_filter`：

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    reasoning: List[str]
    chat_history: ChatHistory
    files: str
    tasks: UserTasks
    instructions: str
    ...
    _filter: Optional[Dict[str, Any]]
```

最值得玩味的是 `messages` 字段的类型标注 `Annotated[Sequence[BaseMessage], add_messages]`。在 LangGraph 中，state 的字段可以附带一个 **reducer**：当多个节点（或并发分支）返回对同一字段的更新时，reducer 决定它们如何合并。`add_messages` 是 LangChain 官方提供的 reducer，语义是"把新消息追加到已有消息列表"，而不是覆盖——这正是对话历史能持续累积的原因 [[LangGraph: add_messages reducer]](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)。其余字段没有标注 reducer，因此遵循默认的"后写覆盖"语义。

正因为有 reducer，节点函数才能安全地返回**增量更新**。看 `filter_history` 的返回 `return {**state, "chat_history": _chat_history}`：它用 `{**state}` 摊平原状态再覆盖一个字段。这是贯穿全文件的统一写法，既保证了不丢失其他字段，又让每个节点的"副作用面"一目了然——它到底改了哪几个 key。LangGraph 节点签名都是 `(state: AgentState) -> AgentState`，这种**纯函数 + 增量字典**的契约，是整张图可组合、可测试的基础。

## 多任务模型：UserTasks 与 UserTaskEntity

Quivr 不假设"一次用户输入只对应一个检索 query"。它引入了一对数据结构来承载多任务。最小单元是 `UserTaskEntity`（Pydantic 模型）：

```python
class UserTaskEntity(BaseModel):
    id: UUID
    definition: str
    docs: List[Document] = Field(default_factory=list)
    completable: bool = Field(default=False, ...)
    tool: Optional[str] = Field(default=None, ...)
```

每个任务有自己的 `definition`（任务文本，会在 rewrite 阶段被改写）、`docs`（该任务专属的检索结果）、`completable`（能否仅凭上下文完成）和 `tool`（需要调用的工具，留待第 10 章）。它还提供 `has_tool()` 与 `is_completable()` 两个语义化判断。

容器 `UserTasks` 用一个 `{id: UserTaskEntity}` 字典管理所有任务，构造时接收一组任务文本 `task_definitions`，为每条生成 `uuid4()` 作为 key。它把"对任务的增删查"封装成一组方法与属性：`set_docs`/`set_definition`/`set_completion`/`set_tool` 按 `id` 定点修改；`has_tasks()`/`has_non_completable_tasks()` 做存在性判断；`non_completable_tasks`/`completable_tasks` 按完成状态分组；而 `definitions` 返回所有任务文本、`docs` 把所有任务的文档拼平成一个大列表（`[doc for task ... for doc in task.docs]`）。后两个属性正是 `generate_rag` 拼 prompt 时要用的。

为什么要这么设计?因为用户一句话里可能藏着多个独立问题。`rag/prompts.py` 的 `SPLIT_PROMPT` 给了最直白的例子:输入 `'What is Apple? Who is its CEO? When was it founded?'` 应当被拆成 `['What is Apple?', 'Who is the CEO of Apple?', 'When was Apple founded?']`。把它们拆成独立任务后，每个任务都能拿到自己最相关的 chunk,而不是让三个问题挤在一个 query 里互相稀释检索质量。这是 Quivr 在"检索粒度"上的一个关键设计决策。

## routing 节点：把一句话拆成指令与任务

`routing` 是图的入口分流器。它取出用户的第一条消息 `state["messages"][0].content`，套上 `SPLIT_PROMPT` 格式化成请求，再让 LLM 做**结构化输出**：

```python
msg = custom_prompts[TemplatePromptName.SPLIT_PROMPT].format(
    user_input=state["messages"][0].content,
)
try:
    structured_llm = self.llm_endpoint._llm.with_structured_output(
        SplittedInput, method="json_schema"
    )
    response = structured_llm.invoke(msg)
except openai.BadRequestError:
    structured_llm = self.llm_endpoint._llm.with_structured_output(SplittedInput)
    response = structured_llm.invoke(msg)
```

这里用 `with_structured_output(SplittedInput, method="json_schema")` 强制 LLM 吐出符合 `SplittedInput` 模式的对象;一旦后端不支持 `json_schema` 方法抛出 `openai.BadRequestError`，就**降级**到不带 method 参数的结构化输出再试一次。这种"先用更严格的模式、失败再退回宽松模式"的容错，是对接异构 LLM 供应商时的实用手法。

`SPLIT_PROMPT` 的系统提示把输出分成两部分:`instructions`(指令)与 `task_list`(任务)。指令是"让系统改变行为"的话,例如 `'Answer in French'`、`'You are an expert legal assistant'`、`'Use web search'`,会被**收敛进一个字符串**;任务则是真正要回答的问题,可被拆成多条。提示里还反复强调"不要去解决任务、只做改写"、"拆出的每个任务都必须独立自包含、脱离对话历史也能理解",并明确"若没有指令返回空串"、"若找不到任务就原样返回用户输入"。

拿到结果后,`routing` 不直接修改 state,而是返回一个 `List[Send]`——这是 LangGraph 的**动态分发**机制。若有指令就 `Send("edit_system_prompt", {"instructions": instructions})` 去更新系统提示;否则若有任务列表,就 `Send("filter_history", {"chat_history": ..., "tasks": UserTasks(response.task_list)})`,把任务列表包装成 `UserTasks` 后送进下一节点。注意此处 `instructions` 还会回退到 `self.retrieval_config.prompt`,把配置里预设的系统提示也纳入考量。

## filter_history 节点：双上限裁剪对话历史

进入正式 RAG 流程前,`filter_history` 先给对话历史"瘦身"。它的目标是只保留最近、且不超预算的若干轮对话。核心循环从最新一对消息往回遍历:

```python
for human_message, ai_message in reversed(list(chat_history.iter_pairs())):
    message_tokens = self.llm_endpoint.count_tokens(
        human_message.content
    ) + self.llm_endpoint.count_tokens(ai_message.content)
    if (
        total_tokens + message_tokens
        > self.retrieval_config.llm_config.max_context_tokens
        or total_pairs >= self.retrieval_config.max_history
    ):
        break
    _chat_history.append(human_message)
    _chat_history.append(ai_message)
    total_tokens += message_tokens
    total_pairs += 1
```

这里有几个要点。其一,历史以**对**为单位:`iter_pairs()` 把"一个 Human 消息 + 一个 AI 消息"算作一对,这与对话的自然结构一致。其二,采用**双重上限**:既要 `total_tokens` 不超过 `llm_config.max_context_tokens`,又要 `total_pairs` 不超过 `retrieval_config.max_history`(默认 10),任一条件触顶就 `break`。其三,遍历顺序是 `reversed(...)`,即**从最近往前**累加,保证在预算耗尽时优先保留最新的几轮对话。

token 计数用的是 `self.llm_endpoint.count_tokens(...)`,对 Human 与 AI 两条内容分别计数后相加。函数 docstring 里还留着一句旧注释"a token is 4 characters"和一个 `# TODO: replace with tiktoken`,说明这里曾用"4 字符约等于 1 token"的粗略估算,如今已切换到 `count_tokens` 的真实计数——这是一段能看到工程演进痕迹的代码。最后它构造一个全新的 `ChatHistory`(带新 `chat_id`、沿用原 `brain_id`)装裁剪后的历史,以 `{**state, "chat_history": _chat_history}` 返回。

## rewrite 节点：把依赖上下文的问题改写成独立问题

`rewrite` 解决一个检索领域的经典痛点:用户的后续问题往往依赖上文,比如"再多讲讲她"——直接拿这句去做向量检索几乎无效,因为它本身不含可检索的实体。该节点用 `CONDENSE_TASK_PROMPT` 把这类问题**重写成自包含的独立问题**。

`CONDENSE_TASK_PROMPT` 的系统消息说得很清楚:"给定对话历史和最新的用户任务(可能引用了历史中的上下文),改写出一个无需历史也能理解的独立任务;不要去完成它,只在需要时改写、否则原样返回"。模板里用 `MessagesPlaceholder(variable_name="chat_history")` 注入历史,再用 `User task: {task}\n Standalone task:` 收尾。

`rewrite` 的另一个亮点是**并发**。它先确定任务来源——优先用 state 里已有的 `tasks`,否则把首条消息包成单任务;然后为每个任务构造一个改写请求,用 `model.ainvoke(msg)` 生成协程,统一收进 `async_jobs`,最后用 `asyncio.gather` 一次性并发等待:

```python
async_jobs = []
for task_id in tasks.ids:
    msg = custom_prompts[TemplatePromptName.CONDENSE_TASK_PROMPT].format(
        chat_history=state["chat_history"].to_list(),
        task=tasks(task_id).definition,
    )
    model = self.llm_endpoint._llm
    async_jobs.append((model.ainvoke(msg), task_id))

responses = await asyncio.gather(*(jobs[0] for jobs in async_jobs)) if async_jobs else []
```

拿到结果后,它用 `tasks.set_definition(task_id, response.content)` 把每个任务的 `definition` 就地替换成改写后的版本,再 `{**state, "tasks": tasks}` 返回。注意改写是**写回 `definition` 本身**的,所以下游 `retrieve` 看到的就是已改写、可直接检索的 query。多任务并发改写,意味着无论用户问了几个子问题,改写耗时都接近单个问题——这是 `asyncio.gather` 带来的实质收益。

## retrieve 节点：为每个任务取回各自的文档

`retrieve` 是 RAG 的"R",但本章只关注它在图中的**位置与产物**,重排与压缩的细节留到第 9 章。它同样以任务为单位工作:对每个 `task_id`,用改写后的 `tasks(task_id).definition` 作为 query 去检索,把结果通过 `tasks.set_docs(task_id, _docs)` 绑回该任务:

```python
async_jobs = []
for task_id in tasks.ids:
    async_jobs.append(
        (compression_retriever.ainvoke(tasks(task_id).definition), task_id)
    )
responses = await asyncio.gather(*(task[0] for task in async_jobs)) if async_jobs else []
for response, task_id in zip(responses, task_ids, strict=False):
    _docs = self.filter_chunks_by_relevance(response)
    tasks.set_docs(task_id, _docs)
```

可以看到它与 `rewrite` 共享同一套"为每个任务起一个协程、`asyncio.gather` 并发、`zip` 回填"的骨架。检索管线本身是 `ContextualCompressionRetriever`(`base_retriever` 取 `k` 个候选、`reranker` 取 `top_n` 个,见第 9 章),取回后还经过 `filter_chunks_by_relevance` 按相关性阈值过滤。开头若 `not tasks.has_tasks()` 则直接 `return {**state}` 短路。最终 `{**state, "tasks": tasks}` 返回——此时每个 `UserTaskEntity.docs` 都已装满它专属的相关 chunk,为生成阶段备好弹药。

## generate 节点：RAG 生成与纯聊天的两条分支

生成阶段有两个节点,对应"有检索"与"无检索"两条路径。

`generate_rag` 是主路径。它先把所有任务的文档拼平 `docs = tasks.docs`,交给 `_build_rag_prompt_inputs` 组装 prompt 输入,再用 `RAG_ANSWER_PROMPT` 格式化、`reduce_rag_context` 控预算,最后绑定工具后 `invoke`:

```python
def generate_rag(self, state: AgentState) -> AgentState:
    tasks = state["tasks"]
    docs = tasks.docs if tasks else []
    inputs = self._build_rag_prompt_inputs(state, docs)
    prompt = custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT]
    state, inputs = self.reduce_rag_context(state, inputs, prompt)
    msg = prompt.format(**inputs)
    llm = self.bind_tools_to_llm(self.generate_rag.__name__)
    response = llm.invoke(msg)
    return {**state, "messages": [response]}
```

`_build_rag_prompt_inputs` 是理解 RAG prompt 数据来源的关键。它把六类信息装进一个字典:`context` 由 `combine_documents(docs)` 把所有 chunk 拼成一段文本(没有文档则填 `"None"`);`task` 是用户原始问题 `messages[0].content`;`rephrased_task` 是 `state["tasks"].definitions`,即所有任务改写后的版本;`custom_instructions` 取自 `retrieval_config.prompt`;`files` 来自 `state["files"]`;`chat_history` 是 `state["chat_history"].to_list()`。注意它**同时**把原始问题与改写后问题都喂给模型,让模型既知道用户字面问了什么、又拿到便于结合上下文的版本。

`generate_chat_llm` 是无检索的纯聊天分支,用于不需要查文档的对话场景。它从 `messages` 里分离出系统消息与用户消息,组装 `task`/`custom_instructions`/`chat_history`,经 `reduce_rag_context` 后用 `CHAT_LLM_PROMPT` 结构(系统消息 + 历史占位 + 用户消息)调用 LLM。`CHAT_LLM_PROMPT` 的系统提示只声明助手身份、当日日期与"若有自定义指令则遵循",模板用 `User Task: {task}\nAnswer:` 收尾——没有任何 `context` 字段,这正是它与 RAG 路径的根本区别。两个生成节点都以 `{**state, "messages": [response]}` 返回,而 `messages` 的 `add_messages` reducer 会把回答追加进对话历史。

## prompt 工厂:集中管理的只读注册表

所有 prompt 都在 `rag/prompts.py` 的工厂函数 `_define_custom_prompts()` 里集中定义。它返回一个 `dict[TemplatePromptName, BasePromptTemplate]`,用枚举 `TemplatePromptName` 做 key(`RAG_ANSWER_PROMPT`、`CONDENSE_TASK_PROMPT`、`CHAT_LLM_PROMPT`、`SPLIT_PROMPT`、`DEFAULT_DOCUMENT_PROMPT` 等)。模块加载时执行一次:

```python
_templ_registry: dict[TemplatePromptName, BasePromptTemplate] = _define_custom_prompts()
custom_prompts = types.MappingProxyType(_templ_registry)
```

`types.MappingProxyType` 把注册表包成**只读视图**,任何节点都只能 `custom_prompts[TemplatePromptName.XXX]` 读取、无法意外篡改。这是一种"prompt 即配置、集中管理、全局唯一"的设计:所有节点引用同一份 prompt 定义,改 prompt 只需改这一个文件。

`RAG_ANSWER_PROMPT` 的系统消息集中了 Quivr 的**防幻觉 prompt 工程**。几条硬约束很值得记住:"You must use ONLY the provided context to complete the task. Do not use any prior knowledge or external information, even if you are certain of the answer."(只准用提供的上下文,哪怕你确信答案也不许用先验知识);"Don't cite the source id in the answer"(别在答案里引用 source id);以及用户回答模板里的"if you cannot provide an answer using ONLY the provided context ... just answer that you don't have the answer"(找不到就直说不知道)和"if the provided context contains contradictory ... information, state so"(上下文矛盾时如实指出)。这套约束把"宁可不答、绝不编造"写进了 prompt,是 RAG 系统对抗幻觉的第一道防线。

最后,`DEFAULT_DOCUMENT_PROMPT` 决定了每个 chunk **如何拼进 context**:

```python
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="Filename: {original_file_name}\nSource: {index} \n {page_content}"
)
```

它把每个文档渲染成"Filename + Source 索引 + 正文"三段式。`combine_documents` 正是用这个模板把检索到的 chunk 逐个格式化、串成最终的 `context` 字符串。换言之,从向量库取回的原始 `Document` 要先经过这个模板"穿衣服",才会出现在喂给 LLM 的 prompt 里——文件名与来源索引也因此被一并带入,为答案的可溯源性留下线索。

## 本章小结

- `AgentState` 是流经全图的 `TypedDict`,节点签名统一为 `(state: AgentState) -> AgentState`,靠 `{**state, ...}` 做增量更新。
- `messages` 字段用 `Annotated[..., add_messages]` 标注 reducer,使新消息**追加**而非覆盖,对话历史得以累积。
- `UserTasks`/`UserTaskEntity` 构成多任务模型:一次输入可拆成多个任务,各自持有 `definition`、`docs`、`completable`、`tool`。
- `routing` 用 `SPLIT_PROMPT` + `with_structured_output` 把输入拆成 `instructions` 与 `task_list`,并用 `Send` 做动态分发,带 `json_schema` 失败降级的容错。
- `filter_history` 以"一对消息"为单位、从最近往前遍历,受 `max_context_tokens` 与 `max_history`(默认 10)双重上限约束,用 `count_tokens` 真实计数。
- `rewrite` 用 `CONDENSE_TASK_PROMPT` 把依赖上下文的问题改写成独立问题,并用 `asyncio.gather` 并发处理所有任务,写回各任务的 `definition`。
- `retrieve` 为每个任务用改写后的 query 并发检索,经相关性过滤后 `set_docs` 绑回各任务;重排压缩细节见第 9 章。
- `generate_rag` 经 `_build_rag_prompt_inputs` 同时喂入原始问题与改写问题、上下文、文件与历史,用 `RAG_ANSWER_PROMPT` 生成;`generate_chat_llm` 是无 `context` 的纯聊天分支。
- 所有 prompt 集中在 `_define_custom_prompts` 工厂,以 `TemplatePromptName` 枚举注册进只读 `MappingProxyType`,实现"prompt 即配置"。
- `RAG_ANSWER_PROMPT` 的"仅用上下文、找不到就说不知道、矛盾要指出"是核心防幻觉约束;`DEFAULT_DOCUMENT_PROMPT` 决定 chunk 以"Filename/Source/正文"三段式拼入 context。

## 动手实验

1. **实验一:验证 SPLIT_PROMPT 的多任务拆分** — 打开 `rag/prompts.py`,定位 `SPLIT_PROMPT` 的系统消息,找到 `'What is Apple? Who is its CEO? When was it founded?'` 这个示例,数清它被拆成几个任务;再回到 `routing` 函数,确认拆分结果是如何包成 `UserTasks(response.task_list)` 并通过 `Send` 送入 `filter_history` 的。
2. **实验二:跟踪 filter_history 的 token 裁剪** — 阅读 `filter_history`,在纸上模拟一段含 12 对消息的历史,设 `max_history=10`,推演循环何时因 `total_pairs >= max_history` 而 `break`;再把 `max_context_tokens` 调到极小值,观察哪条上限会先触顶,理解"双上限、最近优先"的裁剪策略。
3. **实验三:对比 rewrite 前后的 query** — 找到 `CONDENSE_TASK_PROMPT` 的系统消息与 `rewrite` 函数,以"她是谁?"(依赖上文)为例,说明改写后应变成怎样的自包含问题;再解释 `tasks.set_definition` 为何要就地覆盖 `definition`,以及这对下游 `retrieve` 的检索质量意味着什么。
4. **实验四:在 RAG_ANSWER_PROMPT 里找防幻觉约束** — 通读 `RAG_ANSWER_PROMPT` 的 `system_message_template`、`context_template` 与 `template_answer`,逐条摘出"只用提供的上下文""不引用 source id""找不到就说不知道""矛盾要指出"四条约束;再对照 `_build_rag_prompt_inputs`,确认 `context`、`task`、`rephrased_task` 分别由哪段代码产出。

> **下一章预告**:本章我们把 `retrieve` 当成黑箱,只关心它"为每个任务取回文档"的角色。下一章《重排与上下文压缩》将拧开这个黑箱——`ContextualCompressionRetriever` 如何用 `k` 个候选喂给 `top_n` 的重排器,`get_reranker` 支持哪些重排后端,`IdempotentCompressor` 这个空操作占位符的意义,以及 `reduce_rag_context` 如何在 token 预算逼近时动态裁剪 context,把检索质量与上下文预算这对矛盾调和到一起。
