# 第 8 章：检索器与重排

知识库服务在上一章把文档切片、向量化后落入了某种向量库，但"查得到"和"查得准"之间还隔着一层。Langchain-Chatchat 把"如何从向量库取回文档"独立成了一个轻量的检索器抽象层，放在 `chatchat/server/file_rag/retrievers/` 目录下。它的职责很克制：屏蔽底层向量库（FAISS、Milvus、PG、ES……）的差异，对外只暴露一个 `get_relevant_documents(query)` 方法；同时给不同的召回策略——纯向量、向量加关键词混合、Milvus 专用——留出可替换的实现位置。

这一章我们把检索器这一层逐文件拆开：先看基类 `BaseRetrieverService` 定义的契约和 `get_Retriever` 工厂，再依次解读 `VectorstoreRetrieverService`、`EnsembleRetrieverService` 与 `MilvusVectorstoreRetrieverService` 三种策略的差异，最后落到 `reranker/reranker.py` 里的 `LangchainReranker`——一个基于 CrossEncoder 的可选后处理器，以及它在当前代码里"写好却被注释掉"的真实状态。所有论断都以源码为准。

值得先建立一个整体认知：在 RAG 的"检索—召回—精排—生成"四段流程里，本章的检索器层负责"召回"，重排器负责"精排"。Langchain-Chatchat 的设计哲学是让这两段都尽量薄、尽量可替换，把复杂度往 LangChain 标准组件（`VectorStoreRetriever`、`BM25Retriever`、`EnsembleRetriever`、`BaseDocumentCompressor`）上推，自己只写胶水代码和少量中文适配。读完本章你会发现，整个检索器层加起来不到 200 行源码，却把多后端、多策略、可选精排都覆盖了——这种克制本身就是一种值得学习的工程取向。

## 检索器基类：一个三方法的抽象契约

打开 `file_rag/retrievers/base.py`，整个基类只有 28 行，干净到几乎是一份接口声明。`BaseRetrieverService` 用 `metaclass=ABCMeta` 声明为抽象基类，构造函数 `__init__(self, **kwargs)` 不做任何实际工作，只把参数原样转发给 `do_init`：

```python
class BaseRetrieverService(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.do_init(**kwargs)

    @abstractmethod
    def do_init(self, **kwargs): pass

    @abstractmethod
    def from_vectorstore(vectorstore, top_k, score_threshold): pass

    @abstractmethod
    def get_relevant_documents(self, query: str): pass
```

三个抽象方法各司其职。`do_init` 是子类的"真正构造逻辑"，由 `__init__` 间接调用，这种"构造函数转发到 `do_init`"的写法把对象创建和初始化解耦，子类只需覆盖 `do_init` 而不必复写 `__init__`。`from_vectorstore` 是一个工厂方法，约定从一个已经建好的 `langchain.vectorstores.VectorStore`、加上 `top_k` 和 `score_threshold` 两个检索参数，构造出检索器服务实例——注意它在基类里没有 `@staticmethod` 装饰，但所有子类实现时都把它写成了静态方法。`get_relevant_documents` 是运行期唯一对外的查询入口，输入查询字符串，返回文档列表。这三个方法构成了检索器层对上层（知识库服务）的全部承诺。

为什么基类要把构造拆成 `__init__` 与 `do_init` 两段？从子类实现可以反推出意图：三个子类的 `do_init` 签名各不相同（`VectorstoreRetrieverService` 与 `EnsembleRetrieverService` 都是 `do_init(self, retriever=None, top_k=5)`），如果直接覆盖 `__init__` 就要每个子类重复 `super().__init__()` 的样板，而统一在基类 `__init__` 里用 `**kwargs` 透传给 `do_init`，子类只需关心自己真正需要的字段。这是一种典型的"模板方法"骨架：基类固定调用时序，子类填充行为细节。另一个隐含约定是：实例上始终挂着一个 `self.retriever`（真正干活的 LangChain 检索器）和一个 `self.top_k`，而 `self.vs = None` 这个字段三个子类都写了却从未真正使用——属于重构残留，读源码时不必为它寻找语义。

## 工厂与注册表：用字符串选策略

检索策略的选择不在检索器内部，而是由 `file_rag/utils.py` 里的工厂完成。它维护一张名为 `Retrivals`（注意源码里的拼写）的注册表字典，把策略名映射到对应的服务类：

```python
Retrivals = {
    "milvusvectorstore": MilvusVectorstoreRetrieverService,
    "vectorstore": VectorstoreRetrieverService,
    "ensemble": EnsembleRetrieverService,
}

def get_Retriever(type: str = "vectorstore") -> BaseRetrieverService:
    return Retrivals[type]
```

`get_Retriever` 默认返回 `"vectorstore"` 对应的类。这里有一个值得注意的细节：工厂返回的是**类本身**而非实例，所以调用方拿到类后还要接着调 `.from_vectorstore(...)` 才能得到可用的检索器。这种"工厂选类、类方法造实例"的两段式写法，让每个知识库服务可以按自己的存储特性挑选策略。

顺带一提，注册表变量名是 `Retrivals`（少了一个字母，正确拼写应为 `Retrievals`），这是源码里真实存在的拼写疏漏，工厂函数 `get_Retriever` 也保留着 R 大写、其余小写的非常规命名。这类小瑕疵无伤功能，但提醒我们读源码要以"实际标识符"为准而非"应该叫什么"。注册表只有三个条目——`"milvusvectorstore"`、`"vectorstore"`、`"ensemble"`，对应三种检索器服务类；如果未来要新增一种检索策略（比如纯 BM25、或接入外部检索 API），最小改动就是写一个继承 `BaseRetrieverService` 的新类、在 `__init__.py` 里导出、再往这张表里加一行，调用方的 `get_Retriever("新名字")` 即可启用。这正是注册表模式的价值：扩展点集中、改动面可控。

策略的选取分散在各个 `kb_service` 里，且是硬编码的而非读配置。FAISS 服务在 `faiss_kb_service.py` 的 `do_search` 中选用 `get_Retriever("ensemble")`，这意味着本地 FAISS 知识库默认走的是向量加关键词的混合检索；Milvus 服务用 `get_Retriever("milvusvectorstore")`；而 Chroma、PG、ES、Zilliz 等服务则统一调用 `get_Retriever("vectorstore")`。换句话说，"用哪种检索器"由后端向量库类型决定，开箱即用，无需用户配置。`do_search` 的 `score_threshold` 默认取自 `Settings.kb_settings.SCORE_THRESHOLD`。

这种"策略与后端绑定"的取舍有它的道理：混合检索依赖把全部文档读进内存建 BM25 索引，只有 FAISS 这种内存型向量库天然具备该能力；Milvus 走网络 RPC，再额外拉全量文档建 BM25 既不现实也不经济，于是退回纯向量；而 Chroma/PG/ES 这些有各自查询语义的后端，用最通用的 `as_retriever` 包一层最稳妥。代价是用户无法在配置层面把某个 PG 知识库切成混合检索——要换策略只能改 `kb_service` 源码。`faiss_kb_service.py` 的 `do_search` 签名为 `do_search(self, query, top_k, score_threshold=Settings.kb_settings.SCORE_THRESHOLD)`，先 `load_vector_store().acquire()` 拿到上下文管理的向量库，再造检索器、查询、返回，整个过程被 `with` 块包住以保证向量库句柄被正确释放。

## 纯向量检索：最薄的一层包装

`vectorstore.py` 里的 `VectorstoreRetrieverService` 是最简单的实现，它几乎只是 LangChain `VectorStoreRetriever` 的一层薄封装。`do_init` 接收一个 `retriever` 和 `top_k`（默认 5），把它们存到实例上；真正的构造逻辑在静态方法 `from_vectorstore` 里：

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": score_threshold, "k": top_k},
)
return VectorstoreRetrieverService(retriever=retriever, top_k=top_k)
```

它调用 LangChain 向量库标准的 `as_retriever`，把检索类型固定为 `"similarity_score_threshold"`，即"相似度+阈值过滤"：先按相似度取回，再丢弃分数低于 `score_threshold` 的结果 [[VectorStoreRetriever](https://python.langchain.com/docs/how_to/vectorstore_retriever/)]。`search_kwargs` 里同时塞进 `k`（取回数量）和 `score_threshold`。`get_relevant_documents` 则在底层检索结果上再做一次 `[: self.top_k]` 截断，作为双保险——即便底层多返回了，也只放行前 `top_k` 个。这种"既传 k 又手动切片"的冗余在三种检索器里是一致的写法。

这层封装看似多余——为什么不直接用 LangChain 的 `VectorStoreRetriever`？答案在于统一接口：上层 `kb_service` 只认 `BaseRetrieverService` 这套 `from_vectorstore` + `get_relevant_documents` 的契约，于是即便底层换成 BM25 混合或 Milvus 自定义检索，调用代码一行都不用动。把 LangChain 原生检索器再包一层，付出的是几行样板，换来的是"策略可热插拔"的一致性。这也是为什么 `do_init` 默认 `top_k=5`，而实际调用时 `from_vectorstore` 又显式传入 `top_k`——默认值只是兜底，真正生效的永远是调用方从配置 `VECTOR_SEARCH_TOP_K` 传下来的值。

## 混合检索：BM25 与向量的等权融合

`ensemble.py` 的 `EnsembleRetrieverService` 是三者中最有看点的。它把**稀疏检索**（BM25 关键词匹配）和**稠密检索**（向量相似度）两路召回结果加权融合，这正是 RAG 里常说的 hybrid search。`from_vectorstore` 里同时构造了两个检索器：

```python
faiss_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": score_threshold, "k": top_k},
)
import jieba
docs = list(vectorstore.docstore._dict.values())
bm25_retriever = BM25Retriever.from_documents(
    docs, preprocess_func=jieba.lcut_for_search,
)
bm25_retriever.k = top_k
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)
```

向量这一路与纯向量检索器完全相同。关键词这一路用的是 LangChain Community 的 `BM25Retriever`，BM25 是经典的稀疏排序函数，靠词频与逆文档频率打分，对专有名词、罕见词的精确匹配尤其有效 [[BM25Retriever](https://python.langchain.com/docs/integrations/retrievers/bm25/)]。这里有两个面向中文的工程细节值得点出：其一，`BM25Retriever.from_documents` 的文档来源是 `vectorstore.docstore._dict.values()`，即直接从 FAISS 的内存文档存储里取出**全部**文档来建 BM25 索引——这是 FAISS 才有的能力，也解释了为什么混合检索目前只在 FAISS 服务里启用；其二，`preprocess_func` 传入 `jieba.lcut_for_search`，用结巴分词的搜索引擎模式替换 BM25 默认的英文空格切词，否则中文整句会被当成一个 token 而失效。源码里还留着一条 `# TODO: 换个不用torch的实现方式` 的注释和被注释掉的 `cutword.Cutter`，说明引入 jieba 是为了规避 torch 依赖的权衡。

需要强调 `vectorstore.docstore._dict` 这个用法的"侵入性"：`_dict` 是 LangChain `InMemoryDocstore` 的私有字段（带下划线前缀），直接访问它属于绕过公开接口、读内部实现。这样写换来的是拿到全量文档的最简路径，代价是耦合了 FAISS docstore 的内部结构——一旦上游 LangChain 改了字段名，这行就会断。这也从侧面印证了为什么只有 FAISS 走混合检索：它的 docstore 恰好是这种可一次性枚举的内存字典，而 Milvus、PG、ES 的文档都在外部存储里，没有等价的"一行拿全量"的捷径。对学习者而言，这是一个观察"工程便利"与"接口稳定性"如何被权衡的真实样本。

融合由 `EnsembleRetriever` 完成，它对两路结果按 `weights=[0.5, 0.5]` 等权做 RRF（Reciprocal Rank Fusion，倒数排名融合），把各路文档按排名位置加权重排后输出 [[EnsembleRetriever](https://python.langchain.com/docs/how_to/ensemble_retriever/)]。权重是写死的 0.5/0.5，没有暴露成配置项。最后 `get_relevant_documents` 同样做 `[: self.top_k]` 截断。

为什么混合检索值得专门做一路？纯向量检索擅长"语义近似"——查"如何重启服务"也能召回"重新启动进程"的段落，但它对精确字符串、产品代号、错误码、人名这类"字面命中"反而不敏感，因为这些词在嵌入空间里未必聚得很近。BM25 恰好相反，它对字面命中极其敏感却不懂语义。两者等权融合，等于让语义召回和关键词召回互补：向量路兜住"换种说法"的查询，BM25 路兜住"必须一字不差"的查询。这也是工业界 hybrid search 成为 RAG 标配的原因。Chatchat 把它默认开在 FAISS 上、用 0.5/0.5 等权，是一个无需调参就能用的安全默认；真要针对中文长文档或代码库调优，把权重改成偏向某一路、或把 BM25 分词器换成更贴合领域的实现，都是顺理成章的扩展点。

## Milvus 专用检索器：自定义相似度阈值过滤

`milvus_vectorstore.py` 单独写了一个 `MilvusRetriever`，它继承自 LangChain 的 `VectorStoreRetriever` 并覆盖了 `_get_relevant_documents` 与异步版 `_aget_relevant_documents`。为什么要自定义？因为它要返回 `(doc, similarity)` 这样的**文档加分数二元组**，而非默认检索器只返回文档。在 `search_type == "similarity_score_threshold"` 分支里，它调用 `vectorstore.similarity_search_with_score` 拿到带分数的结果，再做两步处理：先用一段 `warnings.warn` 校验分数是否落在 0 到 1 之间，提醒"relevance scores must be between 0 and 1"；然后用 `if similarity >= score_threshold` 过滤掉低分文档，并在结果为空时再发一条警告。

这里藏着一个容易踩的语义坑。该检索器对阈值的判断是 `similarity >= score_threshold`——分数**越高越保留**；而前面纯向量检索器走的是 LangChain 内置 `similarity_score_threshold` 逻辑，配上 `SCORE_THRESHOLD: float = 2.0` 这个"越小越相关、取 2 相当于不筛选"的默认值（见 `settings.py` 注释）。两套检索器对"分数高低含义"的假设并不一致，这是 Milvus（默认返回归一化到 0–1 的相似度）与 FAISS（返回 L2 距离，越小越近）底层度量差异在检索层的真实投影，使用时要按后端区分。除 `similarity_score_threshold` 外，`MilvusRetriever` 还支持 `"similarity"` 与 `"mmr"` 两种 `search_type`，遇到未知类型会 `raise ValueError`。`MilvusVectorstoreRetrieverService.from_vectorstore` 直接 `new` 这个 `MilvusRetriever`，固定用 `similarity_score_threshold`，其余结构与纯向量检索器一致。

还有一个实现细节值得对照：在 `similarity_score_threshold` 分支里，`MilvusRetriever` 同步版 `_get_relevant_documents` 过滤时写的是 `doc for doc, similarity in ...`（只保留 doc），而异步版 `_aget_relevant_documents` 过滤时写的是 `(doc, similarity) for doc, similarity in ...`（保留二元组）。也就是说同步与异步两条路径返回的元素结构其实不完全一致——同步返回 `Document`、异步返回 `(Document, score)`。这是一处需要警惕的不对称，调用方若混用同步/异步入口，对返回值的解构假设可能落空。源码里还各自带着 `warnings.warn("Relevance scores must be between 0 and 1...")` 的越界提醒和"No relevant docs were retrieved"的空结果提醒，这两条 warning 直接照搬自 LangChain 原版实现的措辞。`mmr`（Maximal Marginal Relevance，最大边际相关）分支则调 `max_marginal_relevance_search`，在相关性与多样性之间做权衡，避免召回一堆内容雷同的文档。

## 重排：CrossEncoder 把召回结果重新打分

检索召回追求"宁多勿漏"，但喂给 LLM 的上下文有限，于是需要重排（rerank）做精排。`reranker/reranker.py` 里的 `LangchainReranker` 就是这一环，它继承自 LangChain 的 `BaseDocumentCompressor`，本质是一个"文档压缩器"——输入一批文档，输出一个更短、更相关的子集。

它的核心是一个 CrossEncoder。与双塔式的 embedding（query 和 doc 各自独立编码再算余弦）不同，CrossEncoder 把 `[query, doc]` **拼成一对**一起喂进模型，让两者在注意力层充分交互，因此排序更准，代价是无法预先缓存、必须实时逐对推理 [[CrossEncoder](https://www.sbert.net/examples/applications/cross-encoder/README.html)]。构造函数里直接实例化 `sentence_transformers.CrossEncoder`：

```python
self._model = CrossEncoder(
    model_name=model_name_or_path, max_length=max_length, device=device
)
```

`__init__` 的默认参数也透露了它的设计取向：`top_n: int = 3`（重排后只保留前 3 篇）、`device: str = "cuda"`（默认上 GPU）、`max_length: int = 1024`、`batch_size: int = 32`、`num_workers: int = 0`。注释里被废弃的注释行（`# from chatchat.configs import ...`）和文件末尾被整段注释掉的 `if USE_RERANKER:` 示例，提示默认重排模型曾是 `BAAI/bge-reranker-large` [[BGE Reranker](https://huggingface.co/BAAI/bge-reranker-large)]。

这里还能看到 Pydantic 的痕迹：`LangchainReranker` 的字段用 `top_n: int = Field()`、`_model: Any = PrivateAttr()` 等声明，因为它继承自 `BaseDocumentCompressor`（一个 Pydantic 模型）。注意 `_model` 用 `PrivateAttr` 而非普通字段——CrossEncoder 实例不是可序列化的配置数据，把它标成私有属性既能挂在对象上，又不会被 Pydantic 纳入校验和序列化范围。`__init__` 里也因此分成两步：先 `self._model = CrossEncoder(...)` 实例化模型，再 `super().__init__(top_n=..., model_name_or_path=..., ...)` 把那些"配置型"字段交给父类初始化。这种"私有属性放重对象、Field 放轻配置"的拆分，是把第三方模型对象塞进 Pydantic 模型时的常见处理手法，值得记下来作为模板。`max_length=1024` 限定了 query 与 doc 拼接后的最大 token 数，超长会被截断，这对长文档精排是个需要留意的约束。

真正的重排逻辑在 `compress_documents` 里。它先对空输入短路返回 `[]`，避免无谓的模型调用；然后把每篇文档的 `page_content` 与 query 配成 `sentence_pairs = [[query, _doc] for _doc in _docs]`，交给 `self._model.predict(...)` 一次性批量打分，`convert_to_tensor=True` 让结果以张量返回。截断逻辑很谨慎：

```python
top_k = self.top_n if self.top_n < len(results) else len(results)
values, indices = results.topk(top_k)
for value, index in zip(values, indices):
    doc = doc_list[index]
    doc.metadata["relevance_score"] = value
    final_results.append(doc)
```

`top_k` 取 `top_n` 与文档总数的较小值，避免文档不足 `top_n` 时越界；用张量的 `.topk` 直接拿到分数最高的若干篇及其原始下标，按新顺序重组文档，并把分数写回每篇文档的 `metadata["relevance_score"]`，方便上游展示或二次筛选。最终返回的就是重排并截断后的文档子集。

这里把召回与精排的分工讲透：检索器（召回）追求高召回率，通常取回较多候选（比如 `top_k` 篇）；重排器（精排）在这些候选里做更精细的相关性判断，只留 `top_n`（默认 3）篇。两者用的模型截然不同——召回用的是离线就能建好的向量索引或 BM25，查询快但排序粗；精排用 CrossEncoder 实时把 query 与每篇候选成对推理，排序细但慢。所以典型搭配是"召回多、精排少"：先用便宜的检索捞回 20 篇，再用昂贵的 CrossEncoder 精排出 3 篇喂给 LLM。`LangchainReranker` 继承 `BaseDocumentCompressor` 而非自定义接口，意味着它能直接塞进 LangChain 的 `ContextualCompressionRetriever` 等压缩管线，与检索器无缝串联——这又是一次"复用框架标准抽象"的选择。`compress_documents` 的注释里仍写着"Compress documents using Cohere's rerank API"，这是从 LangChain 官方 `CohereRerank` 模板复制改写时残留的文档字符串，实际实现走的是本地 CrossEncoder 而非 Cohere API [[CohereRerank](https://python.langchain.com/docs/integrations/document_transformers/cohere_reranker/)]。

## 一次检索的完整数据流

把前面的零件串起来，一次知识库检索在源码里的真实路径是这样的：上层（对话或 API）调用某个 `kb_service` 的 `do_search(query, top_k, score_threshold)`；`do_search` 内部 `load_vector_store().acquire()` 拿到向量库句柄，调 `get_Retriever(策略名)` 取回检索器类，再用 `.from_vectorstore(vs, top_k, score_threshold)` 现场构造出检索器实例；实例的 `get_relevant_documents(query)` 触发底层 LangChain 检索器（或混合融合），对结果做 `[: top_k]` 截断后返回。整条链路里没有任何缓存检索器实例的动作——每次查询都重新 `from_vectorstore`，对 FAISS 混合检索而言还意味着每次都从 `docstore._dict` 重建一遍 BM25 索引。这在文档量不大时可以接受，但文档规模膨胀后会成为可观的开销，是一个明确的优化点。

如果重排被接回（解开 `kb_chat.py` 的注释），数据流会在 `do_search` 返回的 `docs` 之后多一站：把 `docs` 连同 `query` 交给 `LangchainReranker.compress_documents`，CrossEncoder 逐对打分、`.topk` 取前 `top_n`、写回 `relevance_score`，最后才把精排结果拼成 `context`。也就是说"检索器 → （可选重排器）→ context 拼接 → prompt → LLM"是 Chatchat RAG 设计中预留的标准管线，本章覆盖的正是前两站。理解这条数据流，下一章读对话链路时就能把检索结果如何变成 prompt 的全过程接上。

把视野再拉远一层：本章三种检索器加一个重排器，本质上都在回答同一个问题——"给定 query，从向量库里挑出最该给 LLM 看的几段文字"。Chatchat 没有发明新算法，而是用最小的胶水代码把 LangChain 既有的检索与压缩组件按后端特性编排起来，并补上中文分词、设备复用这些本地化适配。这种"站在框架肩膀上、只写差异部分"的风格贯穿全书解构的各个模块，是离线可部署 RAG 系统在有限工程预算下的务实选择。

## 配置开关与"写好却没接上"的现状

重排相关的开关集中在 `settings.py` 的 `KBSettings` 区域。和检索直接相关的默认值有 `VECTOR_SEARCH_TOP_K: int = 3`（知识库匹配向量数量）、`SCORE_THRESHOLD: float = 2.0`（注释明确"取值范围 0–2，越小越相关，取到 2 相当于不筛选，建议 0.5 左右"）、`SEARCH_ENGINE_TOP_K: int = 3`，以及 FAISS 缓存相关的 `CACHED_VS_NUM: int = 1`、`CACHED_MEMO_VS_NUM: int = 10`。

但有一个必须如实指出的现状：**重排器在当前对话主链路里是被注释掉的**。在 `server/chat/kb_chat.py` 中，原本应当装配 `LangchainReranker` 的那段代码——读取 `Settings.kb_settings.USE_RERANKER`、`RERANKER_MODEL`、`RERANKER_MAX_LENGTH` 并对 `docs` 调用 `compress_documents`——整段以 `#` 注释存在，上方还留着一行 `# TODO： 视情况使用 API` 的说明。也就是说，`LangchainReranker` 这个类是完整可用的，但默认对话流程并没有真正触发它，召回后的 `docs` 直接拼成 `context` 喂给 LLM。这反映出项目在 v0.3 重构后，倾向于把重排能力交给"在线模型 API"而非本地加载 CrossEncoder 权重，以贴合其"轻量、可离线、少重依赖"的整体取向。读源码时识别出这类"功能就绪但未接线"的状态，比想当然地认为"配了开关就生效"重要得多。

值得顺带留意的是被注释代码里的两个细节，它们透露了原本的设计意图。其一，`reranker_model = LangchainReranker(top_n=top_k, ...)` 把重排的 `top_n` 直接设成检索的 `top_k`——意味着原意是"召回多少就精排多少、不再额外缩减数量"，重排在这里主要是**重新排序**而非**进一步截断**；其二，`device=embedding_device()` 复用了嵌入模型的设备选择函数，让重排器和嵌入模型共享 GPU/CPU 决策，避免两套设备配置打架。这两点说明原作者把重排视为嵌入流水线的自然延伸，而非独立子系统。当下若要恢复重排，最贴近原意的做法就是把这段注释解开、确认 `USE_RERANKER` 等配置项在 `KBSettings` 中存在并指向一个可用的 reranker 模型路径。

回到调参视角做个总结性提醒：在 Chatchat 当前实现下，影响检索质量最直接的两个旋钮是 `VECTOR_SEARCH_TOP_K` 和 `SCORE_THRESHOLD`。前者决定召回多少候选——调大能提高召回率但会稀释上下文、增加 LLM 噪声；后者是相关性闸门，但要切记它在 FAISS（越小越相关、默认 2.0 近乎不过滤）和 Milvus（越大越相关）两种后端上方向相反，盲目套用同一个数值会得到截然不同的过滤效果。混合检索的 0.5/0.5 权重、重排的 `top_n=3` 这些都是写死或默认值，想进一步优化就得动源码而非配置。理解了"哪些能配、哪些只能改码"，才能对这套检索器层的可调空间有准确预期。

## 本章小结

- 检索器层位于 `file_rag/retrievers/`，基类 `BaseRetrieverService` 用 `ABCMeta` 定义了 `do_init` / `from_vectorstore` / `get_relevant_documents` 三个抽象方法，构造函数把参数转发给 `do_init`。
- 工厂 `get_Retriever` 基于 `Retrivals` 注册表按字符串返回检索器**类**（非实例），默认 `"vectorstore"`，调用方再用 `from_vectorstore` 造实例。
- 策略选取在各 `kb_service` 里硬编码：FAISS 用 `"ensemble"`、Milvus 用 `"milvusvectorstore"`、Chroma/PG/ES/Zilliz 用 `"vectorstore"`，由后端类型决定。
- `VectorstoreRetrieverService` 是最薄封装，固定 `search_type="similarity_score_threshold"`，并对结果做 `[: self.top_k]` 截断。
- `EnsembleRetrieverService` 把 `BM25Retriever`（结巴 `lcut_for_search` 分词）与向量检索用 `EnsembleRetriever` 以 `weights=[0.5, 0.5]` 等权融合，BM25 文档源自 FAISS 的 `docstore._dict`。
- `MilvusRetriever` 自定义 `_get_relevant_documents`，返回带分数二元组，用 `similarity >= score_threshold`（越高越保留）过滤，与 FAISS 系"越小越相关"的语义相反。
- `LangchainReranker` 继承 `BaseDocumentCompressor`，内部用 `sentence_transformers.CrossEncoder`，把 `[query, doc]` 成对打分。
- `compress_documents` 对空输入短路、批量 `predict`、用 `.topk` 取前 `top_n`（默认 3）并写回 `metadata["relevance_score"]`。
- 检索器与重排器都未缓存实例：每次 `do_search` 都重新 `from_vectorstore`，FAISS 混合检索每次都从 `docstore._dict` 重建 BM25 索引，是文档量大时的优化点。
- 关键默认值：`VECTOR_SEARCH_TOP_K=3`、`SCORE_THRESHOLD=2.0`（越小越相关）、reranker `top_n=3`、`device="cuda"`、`max_length=1024`。
- 重要现状：`kb_chat.py` 中调用 reranker 的代码整段被注释，重排类就绪但默认未接入主对话链路，项目倾向改用在线 API 方式。

## 动手实验

1. **实验一：跟踪一次混合检索的两路召回** — 在 `ensemble.py` 的 `EnsembleRetrieverService.from_vectorstore` 里，分别在构造 `faiss_retriever` 和 `bm25_retriever` 之后打印 `docs` 长度与 `bm25_retriever.k`，再对同一中文 query 调一次 `get_relevant_documents`，观察 BM25 与向量两路各召回了什么、最终融合后顺序如何变化。

2. **实验二：验证分数阈值的语义差异** — 在 FAISS 知识库上用 `do_search` 检索，把 `SCORE_THRESHOLD` 从默认 `2.0` 调到 `0.5`，对比返回文档数量；再切到 Milvus 服务，读 `MilvusRetriever._get_relevant_documents` 里 `similarity >= score_threshold` 的分支，理解为什么同一个阈值在两种后端上含义相反，并记录各自的合理取值范围。

3. **实验三：单独跑通 CrossEncoder 重排** — 写一段脚本导入 `LangchainReranker`，用 `device="cpu"`、`top_n=3` 初始化一个本地 reranker 模型，构造若干 `Document` 与一个 query，调用 `compress_documents`，打印每篇返回文档的 `metadata["relevance_score"]`，确认输出按分数降序且最多 3 篇。

4. **实验四：把重排接回对话链路** — 在 `kb_chat.py` 中取消那段被注释的 reranker 代码（或在 `context` 拼接前手动插入 `compress_documents`），在 `compress_documents` 内部打印 rerank 前后的文档顺序，对同一问题对比"接入重排 vs 不接入"时最终 `context` 的差异与回答质量。

> **下一章预告**：检索器把文档取回、重排把它们精排之后，这些 `docs` 要被拼成 prompt 上下文、与对话历史和 LLM 串成一条可流式输出的链路。第 9 章《RAG 对话链路》将沿着 `kb_chat.py` 走完从 query 到流式回答的完整链条，看 Chatchat 如何用 LangChain 的 `ChatPromptTemplate` 与 `chain` 组织 RAG 对话。
