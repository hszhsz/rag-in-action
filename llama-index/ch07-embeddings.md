# 第 7 章 嵌入模型抽象 Embeddings

第 6 章我们看到 `VectorStoreIndex` 在构建时会把每个 `Node` 的文本送去做向量化,然后存进向量库;检索时又把查询文本变成向量,在向量空间里找最近邻。那么"把文本变成向量"这件事本身是谁干的?答案就是本章的主角:嵌入模型(Embedding Model)。

嵌入模型是把一段文本映射到一个固定维度浮点向量的组件。OpenAI 的 `text-embedding-3`、HuggingFace 上的 BGE、本地跑的 sentence-transformers,乃至 Cohere、Bedrock、Ollama……每家的调用方式、鉴权、批量上限都不一样。LlamaIndex 的做法是抽出一个统一基类 `BaseEmbedding`,把"query 向量化、text 向量化、批处理、相似度、缓存、限流、埋点"这些通用逻辑全部沉淀在基类里,只把"真正调用某家模型"这一步留给子类实现。这样几十家 provider 对上层 index、retriever 而言就是同一个接口。本章带你把 `llama-index-core/llama_index/core/base/embeddings/base.py` 这块基石读透。

## BaseEmbedding:模板方法与 query/text 双路

`BaseEmbedding` 定义在 `base.py`,它同时继承了 `TransformComponent` 与 `DispatcherSpanMixin`。最核心的设计是把"对外公开方法"和"子类实现钩子"分成两层,典型的模板方法模式。

子类必须实现的抽象钩子只有两个同步方法加它们的 async 版:

```python
@abstractmethod
def _get_query_embedding(self, query: str) -> Embedding: ...

@abstractmethod
async def _aget_query_embedding(self, query: str) -> Embedding: ...

@abstractmethod
def _get_text_embedding(self, text: str) -> Embedding: ...
```

注意一个细节:`_get_text_embedding` 是抽象的,但 `_aget_text_embedding` 不是——基类给了默认实现,直接回退到 `_get_text_embedding`。也就是说一个最简子类只要实现三个同步钩子(query、text 各一,再加 async query)即可跑起来。这里 `Embedding` 是文件顶部定义的类型别名 `Embedding = List[float]`,源码注释还留了一句 `# TODO: change to numpy array`,说明当前向量就是普通 Python `float` 列表。

对外,用户调用的是 `get_query_embedding` 和 `get_text_embedding`(及其 async 版)。它们都带 `@dispatcher.span` 装饰器,内部统一做了:发 `EmbeddingStartEvent`、开 `CallbackManager` 的 `CBEventType.EMBEDDING` 事件、(可选)走限流与缓存、调子类钩子、`event.on_end` 回填 chunk 与向量、最后发 `EmbeddingEndEvent`。子类完全不用操心埋点和缓存。

**为什么要把 query 和 text 分成两路?** 这不是冗余。很多检索型嵌入模型对"问题"和"文档"使用不同的指令前缀(instruction prefix)。基类 `get_query_embedding` 的 docstring 就明确举例:对 query 可能要拼上 `"Represent the question for retrieving supporting documents: "`,而对 document 则是 `"Represent the document for retrieval: "`。把两路拆开,子类才能在各自钩子里加不同前缀,从而拿到非对称(asymmetric)检索所需的更优向量。这正是 RAG 检索质量的一个关键细节。

## get_text_embedding_batch:批处理与 embed_batch_size 的吞吐权衡

逐条调 `get_text_embedding` 在建索引时是灾难——成千上万个 node 会打出成千上万次 API 请求。`BaseEmbedding` 因此提供了 `get_text_embedding_batch`,这也是建索引时真正被调用的路径。

它的逻辑很直白:遍历 `texts`,往 `cur_batch` 里攒,直到攒满 `self.embed_batch_size` 或到达最后一条,就"flush"一批——调 `self._get_text_embeddings(cur_batch)` 一次性算完整批,然后清空继续攒:

```python
for idx, text in queue_with_progress:
    cur_batch.append(text)
    if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
        # flush
        ...
        embeddings = self._get_text_embeddings(cur_batch)
        result_embeddings.extend(embeddings)
        ...
        cur_batch = []
```

这里 `embed_batch_size` 是基类的一个 Pydantic 字段:

```python
embed_batch_size: int = Field(
    default=DEFAULT_EMBED_BATCH_SIZE,
    description="The batch size for embedding calls.",
    gt=0,
    le=2048,
)
```

默认值 `DEFAULT_EMBED_BATCH_SIZE` 定义在 `constants.py`,值为 `10`。约束 `gt=0, le=2048` 意味着合法范围是 1 到 2048。**这个参数是吞吐与稳定性的权衡旋钮**:调大它,单次请求带更多文本,网络往返次数少、整体更快,但每个请求体更大、更容易触碰 provider 的 token/条数上限或超时;调小则更稳但更慢。默认 10 是个保守起点,生产里针对自建或高配额模型常会调高。

底层 `_get_text_embeddings` 的默认实现就是 `[self._get_text_embedding(text) for text in texts]`——纯循环。能否真正"一次请求多条"取决于子类是否重写 `_get_text_embeddings` 去调 provider 的批量 API。基类把批量切分(batching)和批量请求(batch call)解耦,正是为了让不支持批量 API 的子类也能无缝复用这套切分骨架。

异步版 `aget_text_embedding_batch` 更进一步:它把每批包成一个 coroutine,再根据 `num_workers` 决定并发策略——`num_workers > 1` 时用 `run_jobs` 控制并发度,否则退化为 `asyncio.gather`,`show_progress` 为真时还会接上 `tqdm_asyncio` 显示进度条。这样大批量向量化能在并发与限流间取得平衡。

## BaseEmbedding 也是 TransformComponent:__call__ 给 node 批量填 embedding

呼应第 1 章和第 5 章:LlamaIndex 把数据处理统一抽象成 `TransformComponent`(一个 node 进、node 出的可组合算子)。`BaseEmbedding` 继承自它,所以嵌入模型本身就是一个 transform,能直接串进 `IngestionPipeline`。

它实现的 `__call__` 是粘合点:

```python
def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
    embeddings = self.get_text_embedding_batch(
        [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
        **kwargs,
    )
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding
    return nodes
```

两点值得注意。第一,它取文本时用的是 `node.get_content(metadata_mode=MetadataMode.EMBED)`——即"用于嵌入时该暴露哪些元数据"的视图,这与第 4 章讲过的 metadata 可见性控制一脉相承:你可以让某些 metadata 参与嵌入而不参与 LLM 上下文,反之亦然。第二,它内部走的正是上一节的 `get_text_embedding_batch`,所以 pipeline 里的向量化天然享受批处理。算完后逐个把 `node.embedding` 填好就返回。`acall` 是它的异步对应,内部换成 `aget_text_embedding_batch`。

正因为有这个 `__call__`,第 6 章 `VectorStoreIndex` 才能把 embed model 当作管线的一环来调用,而不需要知道它具体是哪家 provider。

## similarity 与 SimilarityMode:点积、余弦、欧氏

向量算出来之后,检索阶段要比较两个向量有多"像"。`base.py` 顶部定义了枚举与一个模块级函数,`BaseEmbedding.similarity` 只是把调用转发给它:

```python
class SimilarityMode(str, Enum):
    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"
```

`similarity(embedding1, embedding2, mode)` 的实现:

- `EUCLIDEAN`:返回 `-||e1 - e2||`(对欧氏距离取负)。取负是为了让"越相似分数越大"的排序方向和另外两种一致——距离越小,负距离越大。
- `DOT_PRODUCT`:返回 `np.dot(e1, e2)`,纯点积,不做归一化。
- 其它(即默认 `DEFAULT` / `cosine`):返回 `dot / (||e1|| * ||e2||)`,即余弦相似度。

**何时用哪种?** 余弦是默认且最常用,它只看方向不看模长,对未归一化向量也稳健,适合大多数文本检索。点积在向量已被 L2 归一化时与余弦等价,且省一次除法、最快——很多模型输出已归一化时直接用点积。欧氏距离对绝对位置敏感,在某些聚类/距离语义场景下有用,但文本语义检索里相对少见。基类把默认设成 cosine,是对"向量是否归一化"不做假设的安全选择。

此外基类还提供 `get_agg_embedding_from_queries`:把多个 query 各自向量化后用 `agg_fn` 聚合成一个向量,默认聚合函数 `mean_agg` 就是对所有向量按轴求平均(`np.array(embeddings).mean(axis=0)`)。这在多查询改写(multi-query)后需要一个代表向量时很有用。

## Pooling:token 向量如何聚合成句向量

对本地的 transformer 编码器模型(如 BERT 系),模型直接输出的是"每个 token 一个向量"的序列,要得到整句一个向量,需要做池化(pooling)。`pooling.py` 把这一策略抽成枚举:

```python
class Pooling(str, Enum):
    CLS = "cls"
    MEAN = "mean"
```

注意:核心包里的 `Pooling` 只有 `CLS` 和 `MEAN` 两个枚举值,并没有 `LAST`。它本身可调用(`__call__`),根据成员分派到两个类方法:

- `cls_pooling`:取序列首位置向量。三维张量 `(batch, seq, dim)` 取 `array[:, 0]`,二维 `(seq, dim)` 取 `array[0]`。对应 BERT 用 `[CLS]` 位置代表整句的经典做法。
- `mean_pooling`:沿序列维求平均。三维取 `array.mean(axis=1)`,二维取 `array.mean(axis=0)`。

两者对 2D/3D 之外的 shape 都会抛 `NotImplementedError`。CLS 依赖模型在该位置训练出了好的句表示;MEAN 把所有 token 摊平,通常对未专门训练 CLS 的模型更稳。具体子类(如 HuggingFace 嵌入)会按模型选择默认池化方式。

## MockEmbedding 与 resolve_embed_model:测试与默认归一化

第 2 章讲 `Settings` 与 `resolve_embed_model` 时我们已见过归一化逻辑,这里从嵌入侧再确认一遍。

`mock_embed_model.py` 里的 `MockEmbedding` 是个零成本的假模型:它有一个必填字段 `embed_dim`,四个钩子全部返回同一个常量向量 `[0.5] * self.embed_dim`(由 `_get_vector()` 给出)。它存在的意义是单测和 token 预测——不发任何网络请求就能跑通整条 index/retrieve 链路。文件里还有 `MockMultiModalEmbedding`,继承 `MultiModalEmbedding`,额外实现 `_get_image_embedding`,用来满足 `MultiModalVectorStoreIndex` 的图像嵌入检查。

`utils.py` 的 `resolve_embed_model` 则是把各种"用户传进来的东西"归一成一个 `BaseEmbedding` 实例。其 `EmbedType = Union[BaseEmbedding, "LCEmbeddings", str]`,关键分支:

- `embed_model == "default"`:若环境变量 `IS_TESTING` 为真,返回 `MockEmbedding(embed_dim=8)`;否则尝试构造 `OpenAIEmbedding()` 并校验 `OPENAI_API_KEY`。这就是"不显式配置时默认走 OpenAI"的来源。
- 字符串以 `"clip"` 开头:走多模态 `ClipEmbedding`,冒号后可带模型名,默认 `"ViT-B/32"`。
- 其它字符串:按 `"local:模型名"` 解析,`split(":", 1)` 后第一段必须等于 `"local"`,否则报错 `embed_model must start with str 'local' or of type BaseEmbedding`;通过则构造 `HuggingFaceEmbedding`,缓存目录放在 `get_cache_dir()/models` 下。这正是第 2 章提到的 `local:` 前缀约定。
- 传入的是 LangChain 的 `Embeddings`:用 `LangchainEmbedding` 适配。
- `embed_model is None`:打印提示后回退 `MockEmbedding(embed_dim=1)`,即"显式禁用嵌入"的兜底。

最后无论走哪条分支,都会 `assert isinstance(embed_model, BaseEmbedding)` 并把 `callback_manager` 挂上去再返回。归一化的终点永远是 `BaseEmbedding`,这就是整套抽象能解耦的根本。

## 稠密 vs 稀疏:BaseSparseEmbedding 一瞥

到这里讲的都是稠密向量(dense),即固定维度、几乎每一维都非零的 `List[float]`。`base_sparse.py` 提供了另一条路:稀疏嵌入。

它的类型别名是 `SparseEmbedding = Dict[int, float]`——只记录"哪些维度有值、值是多少",其余维度默认为零,适合表达 SPLADE、BM25 风格的词权重。`BaseSparseEmbedding` 的接口与稠密版高度对称(同样有 query/text 钩子、`get_text_embedding_batch`、`embed_batch_size` 默认也是 `DEFAULT_EMBED_BATCH_SIZE`),但它继承的是 `BaseModel` 而非 `TransformComponent`,且相似度用专门的 `sparse_similarity`:只在两个字典的公共 key 上累加点积,再除以各自范数,本质是稀疏向量上的余弦。稠密擅长语义泛化,稀疏擅长精确词匹配,二者常在混合检索(hybrid search)里互补——这一点后续检索章节会再展开。

## instrumentation 埋点:可观测的向量化

最后看可观测性。`BaseEmbedding` 继承 `DispatcherSpanMixin`,模块顶部 `dispatcher = instrument.get_dispatcher(__name__)`,几乎所有对外方法都带 `@dispatcher.span`,并在调用前后分别发出 `EmbeddingStartEvent` 与 `EmbeddingEndEvent`(稀疏侧对应 `SparseEmbeddingStartEvent`/`SparseEmbeddingEndEvent`)。

有个安全细节值得点出:发 start 事件前会先 `model_dict = self.to_dict()` 再 `model_dict.pop("api_key", None)`,确保密钥不会随事件泄漏到日志或追踪系统。同时基类还叠了一层老的 `CallbackManager` 事件(`CBEventType.EMBEDDING`),`event.on_end` 里带上 `EventPayload.CHUNKS` 和 `EventPayload.EMBEDDINGS`。两套机制并存:`instrumentation` 是较新的、基于 dispatcher 的细粒度 span 体系,`CallbackManager` 是更早的回调体系,LlamaIndex 在过渡期同时保留,以兼容已有集成。第 14 章会专门讲 `instrumentation` 的全貌。

## 本章小结

- 嵌入模型负责把文本映射到向量空间,`BaseEmbedding`(`base.py`)是 LlamaIndex 统一几十家 provider 的抽象基类,`Embedding = List[float]`。
- 它用模板方法模式:子类只实现 `_get_query_embedding` / `_get_text_embedding`(及 async query)等钩子,埋点、缓存、限流、批处理全在基类。
- query 与 text 分两路,是为支持检索模型对问题与文档使用不同指令前缀的非对称嵌入。
- `get_text_embedding_batch` 按 `embed_batch_size` 切批 flush;该字段默认 `DEFAULT_EMBED_BATCH_SIZE = 10`,约束 `gt=0, le=2048`,是吞吐与稳定性的权衡旋钮。
- `BaseEmbedding` 同时是 `TransformComponent`,`__call__` 用 `MetadataMode.EMBED` 取文本、走批处理、逐个回填 `node.embedding`,从而串进 `IngestionPipeline` 与 `VectorStoreIndex`。
- `SimilarityMode` 有 `cosine`(默认)、`dot_product`、`euclidean` 三种;欧氏返回负距离以统一排序方向;`get_agg_embedding_from_queries` 默认用 `mean_agg` 聚合多查询向量。
- `Pooling` 枚举(`pooling.py`)在核心包中只有 `CLS` 与 `MEAN`,负责把 token 序列向量聚合成句向量。
- `MockEmbedding`(`embed_dim`,返回 `[0.5]*embed_dim`)用于零成本测试;`resolve_embed_model` 把 `"default"`/`"local:"`/`"clip"`/LangChain/`None` 等输入统一归一成 `BaseEmbedding`。
- `BaseSparseEmbedding`(`base_sparse.py`)用 `Dict[int, float]` 表达稀疏向量,与稠密版接口对称,适合精确词匹配与混合检索。
- 所有方法带 `@dispatcher.span` 并发 `EmbeddingStartEvent`/`EmbeddingEndEvent`,发事件前 `pop("api_key")` 防泄漏,`instrumentation` 与 `CallbackManager` 双轨并存。

## 动手实验

1. **实验一:确认默认批大小** — 在 `/tmp/llama_explore` 里 `grep "DEFAULT_EMBED_BATCH_SIZE" llama-index-core/llama_index/core/constants.py` 确认值为 `10`,再到 `base.py` 看 `embed_batch_size` 字段的 `Field(default=..., gt=0, le=2048)`,理解合法范围与默认来源。
2. **实验二:跟踪 batch flush 逻辑** — 阅读 `base.py` 中 `get_text_embedding_batch` 的 for 循环,手写演算:当 `texts` 有 25 条、`embed_batch_size=10` 时会触发几次 flush(在第几个 idx),验证最后一批不足 10 条也能因 `idx == len(texts) - 1` 被 flush。
3. **实验三:验证 MockEmbedding 与默认归一化** — 读 `mock_embed_model.py` 确认四个钩子都返回 `[0.5]*embed_dim`,再读 `utils.py` 的 `resolve_embed_model`,找出 `IS_TESTING` 下返回 `MockEmbedding(embed_dim=8)`、`embed_model is None` 时兜底 `MockEmbedding(embed_dim=1)` 这两处分支。
4. **实验四:对比三种相似度** — 在 `base.py` 的 `similarity` 函数里,对 `e1=[1,0]`、`e2=[0,1]` 与 `e1=[1,0]`、`e2=[1,0]` 两组,分别手算 `cosine`、`dot_product`、`euclidean`(负距离)的结果,体会欧氏为何取负、点积与余弦在未归一化时的差异。

> **下一章预告**:向量算出来要存到哪里?第 8 章进入存储与向量库抽象——`StorageContext` 如何统筹 `docstore`/`index_store`/`vector_store` 三类存储,以及内置的 `SimpleVectorStore` 是怎样把向量、相似度检索与持久化最小化实现的。
