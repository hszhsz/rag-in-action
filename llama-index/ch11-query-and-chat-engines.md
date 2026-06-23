# 第 11 章:查询引擎与对话引擎 QueryEngine & ChatEngine

第 9 章我们看到 retriever 把一个问题转成一批带分数的 `NodeWithScore`,第 10 章我们看到 response synthesizer 把这批节点和问题揉进 prompt、调用 LLM、生成最终答案。这两块是 RAG 的左右手,但它们各自都不是"能直接拿来回答问题"的对象——你还得手动 retrieve、手动 synthesize、手动在中间塞后处理。把"检索 → 后处理 → 合成"这条端到端链路封装成一个对象,对外只暴露一个 `query()` 方法,就是本章的第一个主角 `QueryEngine`。

但 `QueryEngine` 是无状态的:每次 `query()` 都是独立的一炮,它不知道"上一句你问了什么"。真实的问答产品是多轮对话,用户会说"那它呢"、"再详细点"。要让系统理解这种指代,就得在 `QueryEngine` 之上叠一层对话记忆——这就是第二个主角 `ChatEngine`。本章从 `BaseQueryEngine` 抽象出发,深入 `RetrieverQueryEngine` 的端到端串联(把第 9、10 章正式接起来),再过一遍可组合的高级查询引擎,最后跃迁到 `ChatEngine`,讲透"对话 = 查询 + 记忆"这件事。

## BaseQueryEngine:一个 query 方法的契约

所有查询引擎的根类是 `base/base_query_engine.py` 里的 `BaseQueryEngine`,它同时继承 `PromptMixin` 和 `DispatcherSpanMixin`。它的设计极度克制:构造函数只接收一个 `callback_manager`,核心契约是两个抽象方法 `_query` 和 `_aquery`,以及它们对外的包装 `query` / `aquery`。

`query` 是一个标准的模板方法,自己不做业务,只负责"埋点 + 归一化输入 + 转交子类":

```python
@dispatcher.span
def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
    dispatcher.event(QueryStartEvent(query=str_or_query_bundle))
    with self.callback_manager.as_trace("query"):
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        query_result = self._query(str_or_query_bundle)
    dispatcher.event(QueryEndEvent(query=str_or_query_bundle, response=query_result))
    return query_result
```

几个值得注意的设计:第一,字符串会被自动包装成 `QueryBundle`,所以下游永远拿到结构化对象(`QueryBundle` 里可以携带 `custom_embedding_strs`、`embedding` 等额外信息);第二,`query` 方法体外面包了 `@dispatcher.span` 装饰器并发出 `QueryStartEvent` / `QueryEndEvent`,同时用 `callback_manager.as_trace("query")` 开了一条 trace——这就是第 14 章 instrumentation 的接入点,任何查询引擎天然可观测;第三,`BaseQueryEngine` 还预留了 `retrieve` / `synthesize` / `asynthesize` 三个方法,但默认实现直接 `raise NotImplementedError`,提示"本引擎不支持单步拆解,请直接用 query"。这意味着"能不能单独调 retrieve"是子类自愿暴露的能力,而非根类强制。

## RetrieverQueryEngine:把第 9、10 章接起来

`query_engine/retriever_query_engine.py` 里的 `RetrieverQueryEngine` 是整个体系最核心、也是 `index.as_query_engine()` 默认返回的引擎。它的职责就一句话:把 retriever、node_postprocessors、response_synthesizer 串成一条流水线。

构造函数接收三个关键组件。如果 `response_synthesizer` 没传,它会用 `get_response_synthesizer(llm=Settings.llm, ...)` 兜底——注意这里直接读 `Settings.llm`,呼应第 2 章的全局配置;`node_postprocessors` 没传则默认空列表;并且它会把 callback_manager 注入每一个 postprocessor,保证后处理阶段也能被同一条 trace 捕获。

真正把第 9、10 章接起来的是 `_query`:

```python
@dispatcher.span
def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
    with self.callback_manager.event(
        CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
    ) as query_event:
        nodes = self.retrieve(query_bundle)
        response = self._response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
        )
        query_event.on_end(payload={EventPayload.RESPONSE: response})
    return response
```

这三行就是 RAG 的端到端骨架:`self.retrieve()` 拿节点 → `synthesize()` 生成答案。而后处理藏在 `retrieve` 里:

```python
def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    nodes = self._retriever.retrieve(query_bundle)
    return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)
```

`_apply_node_postprocessors` 是一个朴素的 for 循环,依次调用每个 postprocessor 的 `postprocess_nodes`,前一个的输出喂给后一个——所以 postprocessor 列表的顺序是有语义的(比如先做 similarity cutoff 再做 rerank,和反过来结果不同)。至此,第 9 章(retriever)→ 后处理 → 第 10 章(synthesizer)三段被一条 `_query` 串成完整链路,异步版 `_aquery` 结构完全对称,只是把每一步换成 `aretrieve` / `asynthesize`。

组装入口是类方法 `from_args`,这是工程上最常用的构造方式。它把"合成器相关的一大堆参数"集中暴露——`response_mode`(默认 `ResponseMode.COMPACT`)、`text_qa_template`、`refine_template`、`streaming`、`use_async`、`output_cls` 等——然后内部调用 `get_response_synthesizer(...)` 一次性把合成器造好,再连同 retriever、node_postprocessors 一起交给构造函数。换句话说,`from_args` 是"声明式配置一条 RAG 流水线"的便捷门面,而 `__init__` 是"组件已经造好、直接拼装"的底层入口。此外 `with_retriever` 还允许复用同一套合成器和后处理器、只替换 retriever,生成一个新引擎,这在多索引场景里很方便。

## as_query_engine 这条最常见的入口链路

值得专门拆开讲 `index.as_query_engine(...)`——绝大多数 LlamaIndex 入门代码就是这一行。索引侧的 `as_query_engine` 通常先调用自己的 `as_retriever(...)` 造出一个该索引类型对应的 retriever(向量索引就是 `VectorIndexRetriever`,呼应第 6、9 章),再把用户传入的 `node_postprocessors`、`response_mode`、`llm`、`streaming` 等参数透传给 `RetrieverQueryEngine.from_args(retriever, ...)`。

这就解释了为什么"一行 `as_query_engine` 就能问答"——它本质上是把第 6 章的索引、第 9 章的 retriever、第 10 章的 synthesizer 自动拼成了一台 `RetrieverQueryEngine`。理解这条链路的价值在于:你能在任何一层介入——换 retriever、插后处理器、改 response_mode、调 streaming,或者干脆绕过 `as_query_engine`、自己 `index.as_retriever()` 取出 retriever 再用 `from_args` 手工拼装,灵活度完全掌握在自己手里。

在 prompt 管理上,`RetrieverQueryEngine._get_prompt_modules` 把 `response_synthesizer` 注册为名为 `"response_synthesizer"` 的子模块,因此 `BaseQueryEngine` 继承自 `PromptMixin` 的 `get_prompts`/`update_prompts` 能直接穿透到合成器内部的 QA、refine 模板。这意味着改 prompt 不必拆开重建引擎,一行 `update_prompts` 就能统一改写整条流水线用到的提示词——这是"模块即 prompt 容器"设计的直接红利。

## 可组合的高级 QueryEngine

`RetrieverQueryEngine` 是最常用的一种,但 `query_engine/` 目录下还有一批"把别的 QueryEngine 当积木"的高级引擎,它们共同体现了"`_query` 是统一契约,谁都可以被嵌套"的可组合性。

**`RouterQueryEngine`(`router_query_engine.py`)** 解决"一个问题该交给哪个子引擎"。它的 `_query` 先用 `self._selector.select(self._metadatas, query_bundle)` 让选择器(基于 LLM 或 embedding)挑出最匹配的子引擎索引;如果选中多个(`result.inds` 长度大于 1),就分别 query 再用 `combine_responses` 合并,否则直接路由到单个 `selected_query_engine.query(query_bundle)`。它还会把选择结果塞进 `final_response.metadata["selector_result"]`,方便调试"到底走了哪条路"。

**`SubQuestionQueryEngine`(`sub_question_query_engine.py`)** 解决"复杂问题要拆解"。它的 `_query` 先用 `self._question_gen.generate(...)` 把原问题拆成若干子问题,每个子问题再交给对应的子引擎查(支持 `use_async` 并发),把每个 `(子问题, 答案)` 包成 `SubQuestionAnswerPair`、再 `_construct_node` 转成节点,最后把所有子答案当作节点喂给 `self._response_synthesizer.synthesize(...)` 做一次总合成。这是"分而治之 + 再聚合"的典型实现。

**`CitationQueryEngine`(`citation_query_engine.py`)** 解决"答案要带引文出处"。它会先把检索到的节点用 `SentenceSplitter` 重新按 `DEFAULT_CITATION_CHUNK_SIZE = 512`、`DEFAULT_CITATION_CHUNK_OVERLAP = 20` 切成更细的"引文块",给每块编号,再用专门的 `CITATION_QA_TEMPLATE` 提示 LLM 在答案里用 `[1]`、`[2]` 的形式标注来源。

此外还有几个轻量包装器:**`TransformQueryEngine`(`transform_query_engine.py`)** 的 `_query` 只做一件事——先 `self._query_transform.run(query_bundle, ...)` 改写查询(如 HyDE、问题分解),再把改写后的 bundle 交给内部 `self._query_engine.query(...)`;**`MultiStepQueryEngine`(`multistep_query_engine.py`)** 做多步推理式的迭代查询;**`RetryQueryEngine`(`retry_query_engine.py`)** 的 `_query` 先正常查一次,再用 `self._evaluator.evaluate_response(...)` 评估答案是否 `passing`,不通过就用 `FeedbackQueryTransformation` 改写查询、递归构造一个 `max_retries - 1` 的新引擎重试。这些引擎本身都不直接接触 retriever,而是把另一个 QueryEngine 当成被包装的内核——这正是统一 `_query` 契约带来的红利。

## 从 QueryEngine 到 ChatEngine:单轮无状态到多轮有记忆

`QueryEngine.query()` 是无状态的纯函数:同样的问题进去,同样的答案出来,它不记得历史。但对话不是这样——"它的价格是多少?"里的"它"指代上一轮提到的产品。要支持多轮,系统必须维护对话历史并在每一轮把历史考虑进去。

`chat_engine/types.py` 里的 `BaseChatEngine` 定义了这套契约。和 `QueryEngine` 只有 `query` 不同,`ChatEngine` 有一组方法:`chat` / `achat`(等待式)、`stream_chat` / `astream_chat`(流式),外加 `reset()`(清空对话状态)和 `chat_history` 属性。每个 `chat` 方法都接收当前 `message` 和可选的 `chat_history`,返回 `AgentChatResponse`(或流式版 `StreamingAgentChatResponse`)。`AgentChatResponse` 是个 dataclass,除了 `response` 文本,还带 `sources`(`List[ToolOutput]`)和 `source_nodes`,并在 `__post_init__` 里自动从 sources 抽取 source_nodes——这样对话答案也能溯源到底层检索节点。`BaseChatEngine` 还顺手提供了 `chat_repl()` / `streaming_chat_repl()` 两个交互式命令行循环,方便本地调试。

## ChatMode 全景

`types.py` 末尾的 `ChatMode` 枚举罗列了 `index.as_chat_engine(chat_mode=...)` 可选的所有策略,每个枚举值的 docstring 直接说明对应实现类:

- `SIMPLE = "simple"`:对应 `SimpleChatEngine`,纯和 LLM 聊天,完全不用知识库。
- `CONDENSE_QUESTION = "condense_question"`:对应 `CondenseQuestionChatEngine`,先把对话上下文 + 最后一句压成独立问题,再丢给 query engine。
- `CONTEXT = "context"`:对应 `ContextChatEngine`,先用用户消息检索文本,把 context 塞进 system prompt 再生成。
- `CONDENSE_PLUS_CONTEXT = "condense_plus_context"`:对应 `CondensePlusContextChatEngine`,先压缩成独立问题、再用它检索 context、再连同 prompt 和用户消息一起交给 LLM。
- `REACT = "react"` / `OPENAI = "openai"`:对应 agent 循环,docstring 明确标注 `NOTE: Deprecated and unsupported.`。
- `BEST = "best"`:根据当前 LLM 自动选最佳引擎,docstring 写明它实际落到 `condense_plus_context`。

可见知识库相关的核心策略其实是三种:CondenseQuestion、Context、CondensePlusContext。下面逐一对比它们"如何处理历史"。

## 三种对话策略:CondenseQuestion vs Context vs CondensePlusContext

**CondenseQuestionChatEngine(`chat_engine/condense_question.py`)** 的思路是"把历史融进问题本身"。它持有一个内部 `query_engine`。在 `chat()` 里,先从 memory 取历史,然后调 `_condense_question(chat_history, message)`:如果没有历史就原样返回,否则用 `messages_to_history_str` 把历史拼成字符串,套进 `DEFAULT_PROMPT`(一段"把跟进消息改写成包含全部上下文的独立问题"的模板)让 LLM 改写,得到 `condensed_question`,再用它去 `self._query_engine.query(...)`。最后把 user 和 assistant 两条消息 `put` 进 memory。它有个值得注意的工程细节:由于老的 query engine 通过类属性配置 streaming,它在调用前会临时改写 `_response_synthesizer._streaming` 标志、在 `finally` 里复位。这种引擎的特点是——它包装的是一个完整 QueryEngine,所以检索 + 合成都发生在子引擎里,历史只影响"问题怎么问"。

**ContextChatEngine(`chat_engine/context.py`)** 的思路是"把历史和 context 都塞进 prompt"。它直接持有一个 `retriever`(而非 QueryEngine)。在 `chat()` 里,先 `_get_nodes(message)` 用原始 message 检索并跑 node_postprocessors(注意:它用的是当前 message 直接检索,不做问题压缩),再从 memory 取历史,然后 `_get_response_synthesizer(chat_history)` 动态构造一个 `CompactAndRefine` 合成器——关键在于它会用 `get_prefix_messages_with_context` 把 `DEFAULT_CONTEXT_TEMPLATE`(把 `{context_str}` 包成系统提示)、system prompt、prefix messages 和聊天历史拼成一组消息。也就是说,历史是以完整 ChatMessage 列表的形式进 prompt 的,检索到的文档也作为 context 注入 system prompt,最后 `synthesizer.synthesize(message, nodes)` 生成答案并把两条消息写回 memory。

**CondensePlusContextChatEngine(`chat_engine/condense_plus_context.py`)** 是前两者的结合:先用 `DEFAULT_CONDENSE_PROMPT_TEMPLATE` 把"历史 + 跟进问题"压成独立问题(除非 `skip_condense=True`),用这个独立问题去检索拿 context(检索更精准),再用 `DEFAULT_CONTEXT_PROMPT_TEMPLATE` 把 context 注入 prompt、连同历史和用户消息交给 LLM。这就是 `BEST` 模式的默认落点:压缩保证检索质量,context 注入保证生成时既看得到文档又看得到对话。

三者的核心区别可以这样记:CondenseQuestion 让历史影响"问什么"(改写问题,检索用改写后的);Context 让历史影响"怎么答"(原样检索,历史进 prompt);CondensePlusContext 两头都管(改写后检索 + 历史进 prompt)。

## ChatMemoryBuffer:token_limit 驱动的记忆管理

对话的"记忆"由 memory 模块承载。`memory/chat_memory_buffer.py` 里的 `ChatMemoryBuffer` 是最经典的实现(文件头部 docstring 已标注 `Deprecated: Please use llama_index.core.memory.Memory instead`,但它清晰展示了记忆管理的核心算法,且仍被多处 `from_defaults` 默认使用)。

它继承 `BaseChatStoreMemory`,底层用一个 `chat_store`(默认 `SimpleChatStore`)按 `chat_store_key` 存消息列表。最关键的字段是 `token_limit`,`from_defaults` 里它的默认推导逻辑是:如果传了 llm,就取 `int(context_window * DEFAULT_TOKEN_LIMIT_RATIO)`,其中 `DEFAULT_TOKEN_LIMIT_RATIO = 0.75`;否则退到 `DEFAULT_TOKEN_LIMIT = 3000`。也就是默认把上下文窗口的 75% 留给历史。

真正的记忆截断发生在 `get()` 里。它先取出全部历史,然后用一个 while 循环从后往前保留尽量多的消息,直到累计 token(由 `tokenizer_fn` 计数)不超过 `token_limit`:

```python
while token_count > self.token_limit and message_count > 1:
    message_count -= 1
    while message_count > 1 and chat_history[-message_count].role in (
        MessageRole.TOOL, MessageRole.ASSISTANT,
    ):
        message_count -= 1
    cur_messages = chat_history[-message_count:]
    token_count = self._token_count_for_messages(cur_messages) + initial_token_count
```

注意内层 while 的细节:截断时不能让历史以 `ASSISTANT` 或 `TOOL` 消息开头(因为 tool 消息必须由前面的 assistant 消息引出,assistant 也不该是对话的起点),所以会连带多砍一条,保证返回的历史结构合法。如果连最近一条都超限,就只返回 `chat_history[-1:]`,让 LLM 自己抛"消息过长"的显式错误,而不是悄悄失败。各对话引擎在 `from_defaults` 里普遍把 token_limit 设成 `llm.metadata.context_window - 256`,给当轮的问题和答案留出余量——这就是"对话 = 查询 + 记忆"在工程上的落地:每一轮 `chat()` 都先从 memory 的 `get()` 取一段不超限的历史,查询后再 `put()` 回去,记忆既被持续累积,又被 token_limit 持续约束。

## 本章小结

- `BaseQueryEngine` 用模板方法 `query`/`aquery` 统一了"埋点 + 把 str 归一成 `QueryBundle` + 转交抽象 `_query`"的契约,`@dispatcher.span` 和 `callback_manager.as_trace` 让所有查询引擎天然可观测。
- `RetrieverQueryEngine._query` 是 RAG 端到端骨架:`retrieve()`(内含 node_postprocessors 串行后处理)→ `response_synthesizer.synthesize()`,把第 9 章检索、后处理和第 10 章合成正式接起来。
- `from_args` 是声明式配置流水线的门面,集中暴露 `response_mode`(默认 `ResponseMode.COMPACT`)、模板、streaming 等参数,内部用 `get_response_synthesizer` 一次性造好合成器。
- 高级查询引擎把别的 QueryEngine 当积木:`RouterQueryEngine` 用 selector 路由、`SubQuestionQueryEngine` 拆子问题再聚合、`CitationQueryEngine` 带引文(512/20 切块)、`Transform`/`MultiStep`/`Retry` 各做查询改写、多步与失败重试。
- `QueryEngine` 无状态;`ChatEngine`(`BaseChatEngine`)叠加对话记忆,提供 `chat`/`stream_chat`/`achat`/`astream_chat` 与 `reset`,返回带 `source_nodes` 的 `AgentChatResponse`。
- `ChatMode` 枚举映射各实现类,知识库相关核心三策略为 `CONDENSE_QUESTION`、`CONTEXT`、`CONDENSE_PLUS_CONTEXT`,`BEST` 实际落到 `condense_plus_context`,`REACT`/`OPENAI` 已废弃。
- CondenseQuestion 把历史压成独立问题再交 QueryEngine(历史影响"问什么");Context 原样检索、把历史与 context 注入 prompt(历史影响"怎么答");CondensePlusContext 两者结合。
- `ChatMemoryBuffer` 以 `token_limit` 截断历史,默认 `context_window * 0.75`(`DEFAULT_TOKEN_LIMIT_RATIO`)或 `3000`,`get()` 从后往前保留且避免历史以 ASSISTANT/TOOL 消息开头,体现"对话 = 查询 + 受限记忆"。

## 动手实验

1. **实验一:确认默认查询引擎类型** — 在 `/tmp/llama_explore` 用 Grep 搜 `def as_query_engine`,沿调用链确认 `index.as_query_engine()` 默认返回 `RetrieverQueryEngine`;再读 `retriever_query_engine.py` 的 `from_args`,记录 `response_mode` 默认值是 `ResponseMode.COMPACT`、`streaming` 默认 `False`。
2. **实验二:画出 _query 三段链路** — 读 `RetrieverQueryEngine._query` 与 `retrieve`/`_apply_node_postprocessors`,在纸上画出 `retriever.retrieve → 依次 postprocess_nodes → synthesizer.synthesize` 的数据流,并验证 node_postprocessors 是按列表顺序串行执行(前一个输出喂下一个)。
3. **实验三:对比三种对话引擎处理历史的位置** — 并排读 `condense_question.py`、`context.py`、`condense_plus_context.py` 的 `chat()`,标出各自"在哪一步用到 chat_history":CondenseQuestion 在 `_condense_question` 改写问题,Context 在 `_get_response_synthesizer` 把历史拼进 prompt 消息,CondensePlusContext 两处都有。
4. **实验四:推演 token_limit 截断** — 读 `chat_memory_buffer.py` 的 `from_defaults` 和 `get()`,手算:若 `context_window=8192` 且未显式传 token_limit,默认 token_limit 是多少(8192×0.75);再追踪 `get()` 的内层 while,解释为什么截断后历史不能以 `MessageRole.ASSISTANT` 或 `MessageRole.TOOL` 开头。

> **下一章预告**:本章的查询引擎与对话引擎仍是"固定流水线 + 嵌套包装"的组合方式,面对复杂分支、循环、并行就力不从心。第 12 章我们将进入 LlamaIndex 新一代编排引擎 Workflow:用 `@step` 声明步骤、用 `Event` 在步骤间传递数据、用 `Context` 共享状态,以事件驱动的方式取代旧的 Query Pipeline,优雅地表达任意复杂的 RAG 与 Agent 流程。
