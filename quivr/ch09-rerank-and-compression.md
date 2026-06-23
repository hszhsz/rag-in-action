# 第 9 章 重排与上下文压缩

在前几章里，我们看到 Quivr 的检索节点会从向量库里召回一批候选 chunk。但向量相似度只是一个粗糙的近似信号：它能把"大致相关"的内容捞上来，却分不清"恰好命中"和"只是字面相似"。如果把这批粗召回的结果原封不动塞给 LLM，既会引入噪声拉低答案质量，又会无谓地消耗宝贵的上下文 token 预算。本章要讲清楚 Quivr 如何把"一堆候选"逐步加工成"喂给 generate 节点的高质量 context"——这条加工流水线由重排（reranking）、阈值过滤、自适应扩展和预算压缩四个环节串成。

这套机制几乎全部集中在 `quivr_core/rag/quivr_rag_langgraph.py` 与 `quivr_core/rag/entities/config.py` 两个文件里。前者实现 `retrieve`、`dynamic_retrieve`、`get_reranker`、`filter_chunks_by_relevance`、`reduce_rag_context` 等节点逻辑，后者用 `RerankerConfig` 和 `RetrievalConfig` 把所有可调参数收拢成声明式配置。下面我们沿着数据流，从粗召回讲到精上下文。

## 为什么需要重排

`RetrievalConfig` 在 `config.py` 里把检索器一次召回的 chunk 数量定为 `k: int = 40`，注释明确写着"Number of chunks returned by the retriever"。这是一个故意偏大的值：向量近邻搜索本质是在嵌入空间里找距离最近的 40 个 chunk，召回率高但精确率低，里面难免混进只是话题接近、却不能直接回答问题的片段。

重排器（reranker）做的是第二轮、更精细的打分。与双塔（bi-encoder）的向量检索不同，reranker 通常是一个 cross-encoder：它把"查询 + 单个 chunk"拼成一对一起送进模型，让二者在每一层做充分的注意力交互，从而给出比纯向量距离更可靠的相关性分数 [[Cohere Rerank]](https://docs.cohere.com/docs/rerank-overview) [[cross-encoder vs bi-encoder]](https://www.sbert.net/examples/applications/cross-encoder/README.html)。代价是它慢、贵，无法直接对整个向量库打分——所以工程上的标准做法是"向量检索召回 top-k，再用 reranker 精排取 top-n"。`RerankerConfig` 里的 `top_n: int = 5`（注释"Number of chunks returned by the re-ranker"）正是这个 top-n：从 40 个候选中精选 5 个最相关的交给下游。

## RerankerConfig：声明式的重排配置

`config.py` 的 `RerankerConfig` 把重排相关参数集中在一处：

```python
class RerankerConfig(QuivrBaseConfig):
    supplier: DefaultRerankers | None = None
    model: str | None = None
    top_n: int = 5  # Number of chunks returned by the re-ranker
    api_key: str | None = None
    relevance_score_threshold: float | None = None
    relevance_score_key: str = "relevance_score"
```

几个字段各司其职：`supplier` 选定重排服务商，`model` 指定具体模型，`top_n` 控制精排后保留多少 chunk，`relevance_score_threshold` 是可选的相关性下限，`relevance_score_key` 则是相关性分数写进 chunk metadata 时使用的键名（默认 `"relevance_score"`）。

供应商由 `DefaultRerankers` 枚举定义，目前可用的是 Cohere 和 Jina：

```python
class DefaultRerankers(str, Enum):
    COHERE = "cohere"
    JINA = "jina"
    # MIXEDBREAD = "mixedbread-ai"

    @property
    def default_model(self) -> str:
        return {
            self.COHERE: "rerank-v3.5",
            self.JINA: "jina-reranker-v2-base-multilingual",
        }[self]
```

`default_model` 这个 property 给每个供应商配了一个默认模型：Cohere 默认 `rerank-v3.5` [[Cohere Rerank]](https://docs.cohere.com/docs/rerank-overview)，Jina 默认 `jina-reranker-v2-base-multilingual` [[Jina Reranker]](https://jina.ai/reranker/)。注释里被注掉的 `MIXEDBREAD` 则提示这套枚举设计是可扩展的——加一个供应商只需补一个枚举值和默认模型映射。

`RerankerConfig` 在构造时通过 `__init__` 调用 `validate_model()` 做两件兜底：其一，如果用户只填了 `supplier` 没填 `model`，就自动用 `self.supplier.default_model` 补上默认模型；其二，从环境变量里读 API key（变量名由 `normalize_to_env_variable_name(self.supplier)` 拼成 `{SUPPLIER}_API_KEY`），如果设了 supplier 却找不到对应 key，直接抛 `ValueError`。这意味着一旦你声明要用 Cohere/Jina，配置阶段就会强制校验密钥，避免运行到检索节点才失败。

## get_reranker：按供应商分派，无配置时安全降级

真正把配置变成可调用对象的是 `get_reranker`：

```python
def get_reranker(self, **kwargs):
    config = self.retrieval_config.reranker_config
    supplier = kwargs.pop("supplier", config.supplier)
    model = kwargs.pop("model", config.model)
    top_n = kwargs.pop("top_n", config.top_n)
    api_key = kwargs.pop("api_key", config.api_key)

    if supplier == DefaultRerankers.COHERE:
        reranker = CohereRerank(
            model=model, top_n=top_n, cohere_api_key=api_key, **kwargs
        )
    elif supplier == DefaultRerankers.JINA:
        reranker = JinaRerank(
            model=model, top_n=top_n, jina_api_key=api_key, **kwargs
        )
    else:
        reranker = IdempotentCompressor()

    return reranker
```

这个方法有两个设计要点。第一是**参数覆盖**：它先从 `reranker_config` 取默认值，再允许 `kwargs` 逐项覆盖 `supplier`/`model`/`top_n`/`api_key`。后面我们会看到 `dynamic_retrieve` 正是靠传 `top_n=...` 在不改配置的前提下动态放大精排数量。

第二是**安全降级**：当 `supplier` 是 `COHERE` 就实例化 `CohereRerank`，是 `JINA` 就实例化 `JinaRerank`，否则（包括 `supplier=None` 这个默认情况）落到 `else` 分支返回 `IdempotentCompressor()`。换句话说，如果用户没有配置任何重排器，系统不会报错，而是退回到一个"什么都不做"的压缩器。

## IdempotentCompressor：诚实的 no-op 默认

`IdempotentCompressor` 是 Quivr 处理"没有 reranker"这件事的优雅方案：

```python
class IdempotentCompressor(BaseDocumentCompressor):
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        A no-op document compressor that simply returns the documents it is given.

        This is a placeholder until a more sophisticated document compression
        algorithm is implemented.
        """
        return documents
```

它继承自 LangChain 的 `BaseDocumentCompressor`，实现了与真实 reranker 完全相同的 `compress_documents` 接口，但函数体只有一行 `return documents`——把输入原样返回。这就是"幂等"二字的含义：压缩前后文档集合不变。

这个设计的价值在于**接口一致性**：下游的 `ContextualCompressionRetriever` 只认 `BaseDocumentCompressor` 这个抽象，它不需要知道自己拿到的是真重排器还是占位器。无论有没有配置 reranker，检索流水线的代码结构完全一样，区别仅仅是"压缩"这一步是否真的过滤、重排了文档。docstring 里那句"This is a placeholder until a more sophisticated document compression algorithm is implemented"也很诚实地承认这是一个占位实现——它不假装做了压缩，只是保证管道不断。

## retrieve 节点：ContextualCompressionRetriever 组合召回与重排

`retrieve` 节点把"召回"和"重排"用 LangChain 的 `ContextualCompressionRetriever` 拼装起来：

```python
kwargs = {
    "search_kwargs": {
        "k": self.retrieval_config.k,
        "filter": _filter,
    }
}
base_retriever = self.get_retriever(**kwargs)

kwargs = {"top_n": self.retrieval_config.reranker_config.top_n}
reranker = self.get_reranker(**kwargs)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=base_retriever
)
```

`get_retriever` 内部其实就是 `self.vector_store.as_retriever(**kwargs)`（没有 vector_store 时抛 `ValueError`），它带着 `k=40` 和可选的 metadata `filter` 去做向量召回。`ContextualCompressionRetriever` 则把 `base_retriever`（粗召回）和 `base_compressor`（重排器）串成一条管道：先用 base_retriever 拿回 k 个候选，再用 base_compressor 压缩重排成 top_n 个 [[ContextualCompressionRetriever]](https://python.langchain.com/docs/how_to/contextual_compression/)。注意这里传给 `get_reranker` 的 `top_n` 就是 config 里的值，覆盖逻辑此处只是显式重申了一遍。

由于一次用户输入可能被拆成多个独立 task（见前面章节的 `routing`/`rewrite`），`retrieve` 会为每个 task 各自发起一次检索，并用 `asyncio.gather` 并发执行：

```python
async_jobs = []
for task_id in tasks.ids:
    async_jobs.append(
        (compression_retriever.ainvoke(tasks(task_id).definition), task_id)
    )
responses = await asyncio.gather(*(task[0] for task in async_jobs)) if async_jobs else []
```

每个 task 拿到自己的检索结果后，还要经过 `filter_chunks_by_relevance` 再过一道阈值，然后通过 `tasks.set_docs(task_id, _docs)` 把文档绑定到对应 task 上。

## filter_chunks_by_relevance：阈值过滤与宽容降级

重排器给每个 chunk 打了分（写进 metadata 的 `relevance_score_key`），`filter_chunks_by_relevance` 负责按阈值砍掉低分项：

```python
def filter_chunks_by_relevance(self, chunks: List[Document], **kwargs):
    config = self.retrieval_config.reranker_config
    relevance_score_threshold = kwargs.get(
        "relevance_score_threshold", config.relevance_score_threshold
    )

    if relevance_score_threshold is None:
        return chunks

    filtered_chunks = []
    for chunk in chunks:
        if config.relevance_score_key not in chunk.metadata:
            logger.warning(
                f"Relevance score key {config.relevance_score_key} not found in metadata, cannot filter chunks by relevance"
            )
            filtered_chunks.append(chunk)
        elif chunk.metadata[config.relevance_score_key] >= relevance_score_threshold:
            filtered_chunks.append(chunk)

    return filtered_chunks
```

这里有三层防御：第一，如果 `relevance_score_threshold` 是 `None`（默认就是 None），直接返回全部 chunk，不做任何过滤——阈值过滤是一项"opt-in"的能力。第二，如果某个 chunk 的 metadata 里压根没有 `relevance_score_key`（比如用的是 `IdempotentCompressor`，它不打分），就 `logger.warning` 提示一句，但**仍然保留这个 chunk**——这是"宽容降级"，宁可保留也不误删。第三，只有当 chunk 确实带分且分数 `>=` 阈值时才保留。这套逻辑让阈值过滤既能在配了真重排器时精确生效，又不会在缺分情况下把所有内容都过滤光。

## dynamic_retrieve：按需放大检索范围的自适应循环

`retrieve` 用固定的 `k` 和 `top_n` 跑一遍就结束，但 Quivr 还提供了一个更聪明的 `dynamic_retrieve` 节点。它的核心洞察是：如果一次精排后返回的相关 chunk 数**恰好等于** `top_n`，那很可能说明相关内容被 top_n 这个上限截断了——还有更多相关 chunk 没被取出来。于是它会逐步放大检索范围重试：

```python
MAX_ITERATIONS = 3
k = self.retrieval_config.k
top_n = self.retrieval_config.reranker_config.top_n
number_of_relevant_chunks = top_n
i = 1

while number_of_relevant_chunks == top_n and i <= MAX_ITERATIONS:
    top_n = self.retrieval_config.reranker_config.top_n * i
    kwargs = {"top_n": top_n}
    reranker = self.get_reranker(**kwargs)

    k = max([top_n * 2, self.retrieval_config.k])
    kwargs = {"search_kwargs": {"k": k}}
    base_retriever = self.get_retriever(**kwargs)
    ...
```

循环条件 `number_of_relevant_chunks == top_n and i <= MAX_ITERATIONS` 说明了两个停止信号：要么某次精排返回的相关 chunk 数小于 top_n（说明已经把所有相关内容捞干净了，不用再扩），要么迭代次数到了 `MAX_ITERATIONS = 3` 的上限。每一轮里，`top_n` 按 `reranker_config.top_n * i` 线性放大（第 1 轮 5、第 2 轮 10、第 3 轮 15），而召回数 `k = max([top_n * 2, self.retrieval_config.k])` 保证候选池始终至少是精排数的两倍、且不低于配置的 k=40。`get_reranker(**kwargs)` 和 `get_retriever(**kwargs)` 此处正是借助前面提到的 kwargs 覆盖能力，在不改配置对象的前提下临时调大参数。

除了"相关数不再等于 top_n"和"超迭代上限"，循环里还埋了第三个、也是最关键的刹车——上下文预算：

```python
docs = tasks.docs
if not docs:
    break

context_length = self.get_rag_context_length(state, docs)
if context_length >= self.retrieval_config.llm_config.max_context_tokens:
    logging.warning(
        f"The context length is {context_length} which is greater than "
        f"the max context tokens of {self.retrieval_config.llm_config.max_context_tokens}"
    )
    break

number_of_relevant_chunks = max(_n)
i += 1
```

一旦把当前所有文档拼进 RAG prompt 后的 token 数 `context_length` 达到或超过 `llm_config.max_context_tokens`，就立即 `break`，不再继续放大——因为再多的相关 chunk 也塞不进模型了。这让 `dynamic_retrieve` 在"尽量多捞相关内容"和"不撑爆上下文窗口"之间取得动态平衡。

## 预算守门：get_rag_context_length 与 reduce_rag_context

无论检索阶段怎么努力，最终喂给 LLM 前都要做一道 token 预算把关。`get_rag_context_length` 负责度量：

```python
def get_rag_context_length(self, state: AgentState, docs: List[Document]) -> int:
    final_inputs = self._build_rag_prompt_inputs(state, docs)
    msg = custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT].format(**final_inputs)
    return self.llm_endpoint.count_tokens(msg)
```

它把文档连同任务、聊天历史等一起填进 `RAG_ANSWER_PROMPT`，再用 `llm_endpoint.count_tokens` 算出完整 prompt 的 token 数。这就是 `dynamic_retrieve` 判断是否超限时用的那个度量。

真正动手压缩的是 `reduce_rag_context`，它在 `generate_rag` 节点调用，会循环裁剪直到 prompt 落进安全预算：

```python
MAX_ITERATIONS = 20
SECURITY_FACTOR = 0.85
...
while n > max_context_tokens * SECURITY_FACTOR:
    chat_history = inputs["chat_history"] if "chat_history" in inputs else []
    if len(chat_history) > 0:
        inputs["chat_history"] = chat_history[2:]
    elif tasks:
        longest_task_id = max(
            task_token_counts.items(), key=lambda x: x[1]["total"]
        )[0]
        if task_token_counts[longest_task_id]["docs"]:
            removed_tokens = task_token_counts[longest_task_id]["docs"].pop()
            task_token_counts[longest_task_id]["total"] -= removed_tokens
            tasks.set_docs(longest_task_id, tasks(longest_task_id).docs[:-1])
    else:
        logging.warning(...)
        break
    ...
```

这里有几个值得注意的设计。其一，目标不是 `max_context_tokens` 本身，而是 `max_context_tokens * SECURITY_FACTOR`（`SECURITY_FACTOR = 0.85`），留出 15% 的安全余量，给模型输出和 token 计数误差兜底。其二，裁剪有优先级：**先砍最老的聊天历史**（`chat_history[2:]` 一次去掉一对最早的 human/AI 消息），历史砍光了才动文档。其三，砍文档时挑 token 总量最大的那个 task（`longest_task_id`），从中弹掉最后一篇文档——优先收缩"最臃肿"的 task，尽量保住其他 task 的上下文。其四，整个循环有 `MAX_ITERATIONS = 20` 的硬上限，且在既无历史可砍又无 task 可缩时打 warning 后 break，避免死循环。每轮裁剪后都用 `combine_documents` 重新拼 context、重算 token 数 `n`，直到降到安全线以下。

至此，一次粗召回的 40 个候选，经历了 cross-encoder 重排取 top_n、相关性阈值过滤、（可选的）自适应扩展，最后被预算守门裁剪进模型窗口，变成了 `generate_rag` 节点真正引用的高质量 context。

## 本章小结

- 向量检索召回偏大（`RetrievalConfig.k = 40`）以保证召回率，再由 reranker 精排取 `top_n=5` 以保证精确率，这是"粗召回 + 精排"的标准范式。
- `RerankerConfig`（config.py）集中管理重排参数：`supplier`、`model`、`top_n=5`、`relevance_score_threshold`、`relevance_score_key="relevance_score"`，构造时自动补默认模型并强制校验 API key。
- `DefaultRerankers` 枚举支持 COHERE（默认 `rerank-v3.5`）与 JINA（默认 `jina-reranker-v2-base-multilingual`），`default_model` property 提供每个供应商的默认模型，设计上可扩展。
- `get_reranker` 按 supplier 分派 `CohereRerank`/`JinaRerank`，无配置时退回 `IdempotentCompressor`，实现"无配置安全降级"；并支持用 kwargs 覆盖 `top_n` 等参数。
- `IdempotentCompressor` 是继承 `BaseDocumentCompressor` 的 no-op 占位器，`compress_documents` 直接 `return documents`，保证检索管道在无重排器时仍然结构一致。
- `retrieve` 节点用 `ContextualCompressionRetriever` 组合 `base_retriever`（向量召回）与 `base_compressor`（reranker），并用 `asyncio.gather` 对多个 task 并发检索。
- `filter_chunks_by_relevance` 按 `relevance_score_threshold` 过滤：阈值为 None 时不过滤，缺少分数键时 warn 但保留 chunk（宽容降级），只在有分且达标时保留。
- `dynamic_retrieve` 以 `MAX_ITERATIONS=3` 自适应循环：当相关 chunk 数恰好等于 top_n 时把 top_n 线性放大、k 同步扩大重试，捞出更多相关内容。
- 自适应循环以 `context_length >= max_context_tokens` 为硬刹车，一旦上下文撑满立即停止扩展，避免塞爆模型窗口。
- `get_rag_context_length` 用 `count_tokens` 度量完整 RAG prompt 的 token 数；`reduce_rag_context` 以 `SECURITY_FACTOR=0.85` 为目标，优先砍最老历史、再砍最臃肿 task 的文档，循环上限 `MAX_ITERATIONS=20`。

## 动手实验

1. **实验一：观察 no-op 降级** — 构造一个 `RetrievalConfig`，`reranker_config` 不设 `supplier`，在 `get_reranker` 返回值处打断点或加日志，确认拿到的是 `IdempotentCompressor`；再调用其 `compress_documents` 传入若干 Document，验证输出与输入完全一致（数量、顺序、内容都不变）。
2. **实验二：启用阈值过滤** — 在 `RerankerConfig` 里设一个真实 supplier（如 Cohere）并填 `relevance_score_threshold=0.5`，跑一次 `retrieve`，在 `filter_chunks_by_relevance` 里打印每个 chunk 的 `metadata["relevance_score"]` 和过滤前后数量，观察低于 0.5 的 chunk 被剔除；再把 supplier 改成 None，确认看到"score key not found"的 warning 且 chunk 全部保留。
3. **实验三：跟踪 top_n 翻倍** — 在 `dynamic_retrieve` 的 while 循环里打印每轮的 `i`、`top_n`、`k` 和 `number_of_relevant_chunks`，用一个相关内容很多的问题触发它，观察当相关数恰好等于 top_n 时 top_n 如何按 `top_n*i` 放大、k 如何按 `max(top_n*2, 40)` 扩大，直到相关数小于 top_n 或 `i` 超过 `MAX_ITERATIONS=3` 停止。
4. **实验四：验证预算刹车** — 把 `llm_config.max_context_tokens` 调到一个很小的值（如 5000），在 `dynamic_retrieve` 里打印每轮的 `context_length`，确认一旦 `context_length >= max_context_tokens` 就立刻 break；再到 `reduce_rag_context` 里打印 `n` 与 `max_context_tokens * 0.85` 的对比，观察聊天历史和文档被逐步裁剪直到 `n` 降到安全线以下。

> **下一章预告**：精排和压缩解决了"喂什么 context"，但 Quivr 的工作流不止于检索——它还能让 LLM 在回答前自己判断"该不该用工具""用哪个工具"。第 10 章《工具路由与 Agent》将深入 `tool_routing`、`run_tool` 与 `LLMToolFactory`，看 Quivr 如何把单纯的 RAG 升级成会调用外部工具的 Agent 工作流。
