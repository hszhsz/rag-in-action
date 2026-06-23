# 第 5 章 索引与向量化

文档被解析成干净文本（第 2 章）、再被切成一个个 chunk（第 3 章）、在后台管线里被异步处理（第 4 章）之后，下一步是把这些文本变成"可被检索的形态"。这件事分两层：一是**向量化**——把每段文本通过 embedding 模型编码成稠密向量；二是**索引落地**——把文本、向量、元数据一起写进底层的文档存储（doc store）。RAGFlow 在这两层都做了相当克制而务实的抽象设计：embedding 侧用一个模板方法把几十家 provider 收敛成统一接口，存储侧则用一组抽象基类把 Elasticsearch、Infinity、OpenSearch、OceanBase 这几种风格迥异的后端统一到同一套契约之下。本章就钻进 `rag/llm/embedding_model.py` 和 `common/doc_store/`、`rag/utils/` 这几处，看 RAGFlow 是怎么把"可插拔"这件事真正落到代码里的。

## 向量字段的命名约定：维度即字段名

在读任何具体实现之前，先记住一个贯穿全书后半部分的关键约定。在 `rag/svr/task_executor.py` 的向量化函数里，每条 chunk 的向量并不是写进一个固定叫 `vector` 的字段，而是写进一个**名字里带维度**的字段：

```python
vector_size = 0
for d, v in zip(docs, vts):
    vector_size = len(v)
    d["q_%d_vec" % len(v)] = v
return tk_count, vector_size
```

也就是说，一个 1024 维的向量会被存进字段 `q_1024_vec`，768 维的存进 `q_768_vec`。这个 `q_%d_vec` 的命名约定看似随意，实则是整个系统的一条隐形契约：向量的维度被直接编码进字段名，使得后续无论是建索引、写入还是检索，都能从字段名反推出向量维度，而不需要额外的元数据去记录"这个知识库用的是多少维的模型"。第 6 章会看到检索侧用正则 `q_[0-9]+_vec` 来识别向量字段，本章后面也会看到 Infinity 后端用同一个正则 `q_(?P<vector_size>\d+)_vec` 从待写入的文档里反推维度。维度即字段名，这是理解 RAGFlow 向量存储的第一把钥匙。

## Embedding 抽象：一个模板方法收敛几十家 provider

RAGFlow 要对接的大模型 provider 数量惊人——光 `rag/llm/embedding_model.py` 里定义的 embedding 类就有几十个，从 `OpenAIEmbed`、`AzureEmbed` 到各家国产模型、再到本地部署的 `BuiltinEmbed`。如果每家都从零写一遍"分批、截断、累加向量、统计 token、处理错误"，代码会迅速失控。RAGFlow 的做法是在抽象基类 `Base` 里用一个模板方法 `_batched_encode` 把这些共性全部吃掉。

`Base` 类本身极简，它只声明了三个方法的契约：`encode(texts)` 编码一批文本、`encode_queries(text)` 编码单条查询、以及共享的 `_batched_encode`。注意它的构造函数是一个纯粹的占位：

```python
class Base(ABC):
    def __init__(self, key, model_name, **kwargs):
        """
        Constructor for abstract base class.
        Parameters are accepted for interface consistency but are not stored.
        Subclasses should implement their own initialization as needed.
        """
        pass
```

注释把意图说得很清楚：基类构造函数接收参数只是为了"接口一致性"，并不存储——真正的初始化交给子类。这是一种刻意的设计选择，让所有 provider 的构造签名长得一样，工厂注册时就能统一对待。

真正的精华在 `_batched_encode`。它是"OpenAI 风格 provider 背后的共享模板"，把四件事一手包办：可选的逐条截断（截到 `truncate_to` 个 token，避免超长输入被 provider 拒绝）、批次循环（每次发 `batch_size` 条）、把每条向量累加进一个 `np.ndarray`、以及把每批的 token 数累加。它通过一个 provider 提供的闭包 `call_fn(batch) -> (embeddings, token_count)` 来真正发起请求，自己绝不碰原始响应对象：

```python
def _batched_encode(self, texts, call_fn, *, batch_size, truncate_to=None):
    if truncate_to is not None:
        texts = [truncate(t, truncate_to) for t in texts]
    vectors = []
    token_count = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            embeddings, tokens = call_fn(batch)
        except ModelException:
            raise
        except Exception as e:
            logger.exception("%s embedding request failed", type(self).__name__)
            raise EmbeddingError(f"Embedding request failed for {type(self).__name__}. Error: {e}") from e
        vectors.extend(embeddings)
        token_count += tokens
    return np.array(vectors), token_count
```

这里有两个值得停下来的设计点。其一是错误处理的分层：已经是结构化、可能可重试的 `ModelException` 被原样 `raise` 保留，而其它任意异常则被统一包进 `EmbeddingError` 并附上底层细节。注释解释了为什么不依赖更上层 `log_exception` 的隐式 raise——因为"被抛出的异常类型会随 SDK 响应形状而变"，这里宁可在一个确定的点上显式抛出。其二是顺序保障：`OpenAIEmbed._call` 在解析响应时用 `_sorted_by_index(res.data)` 按 provider 返回的 `index` 重新排序，确保输出向量与输入文本严格一一对应——这是个容易被忽略却致命的细节，向量和原文一旦错位，整个检索就全错了。

有了这个模板，每个具体 provider 就只需写一个薄薄的 `_call` 闭包加一行 `encode`。比如 `OpenAIEmbed`：

```python
def encode(self, texts: list):
    # OpenAI requires batch size <=16; 8191 is the documented per-input token ceiling.
    return self._batched_encode(texts, self._call, batch_size=16, truncate_to=8191)
```

注释里 `batch_size=16` 和 `truncate_to=8191` 这两个魔法数字都不是拍脑袋——它们分别对应 OpenAI API 的批大小上限与单条输入 token 上限。**默认值里写着对外部约束的认知**，这一点和第 1 章看到的超时配置一脉相承。

## 本地嵌入：BuiltinEmbed 与 TEI

并非所有部署都接外部 API。RAGFlow 内置了 `BuiltinEmbed`，对应第 1 章拓扑里那个 TEI（Text Embeddings Inference）服务。它的实现透露了几个工程考量。首先是单例式的类级模型缓存：模型实例存在类变量 `BuiltinEmbed._model` 上，加载时用 `BuiltinEmbed._model_lock` 这把 `threading.Lock` 保护，避免多线程重复加载同一个重量级模型：

```python
class BuiltinEmbed(Base):
    _FACTORY_NAME = "Builtin"
    MAX_TOKENS = {"Qwen/Qwen3-Embedding-0.6B": 30000, "BAAI/bge-m3": 8000, "BAAI/bge-small-en-v1.5": 500}
    _model = None
    _model_name = ""
    _max_tokens = 500
    _model_lock = threading.Lock()
```

其次，它只在环境变量 `COMPOSE_PROFILES` 里含有 `tei-` 时才真正去初始化 `HuggingFaceEmbed`——这与第 1 章的 docker compose profile 机制对应，没启用 TEI profile 就不会白白加载模型。再次，那张 `MAX_TOKENS` 表给不同内置模型配了不同的截断上限，比如 `BAAI/bge-m3` 是 8000，而表里没有的模型回退到保守的 `_max_tokens = 500`。这个 500 的兜底默认值是典型的 fail-closed：宁可截短一点也不要让超长输入打挂模型。`BuiltinEmbed.encode` 里也注明 TEI 自身会按其文档自动截断输入[[Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)]，所以这一层不重复做截断，只负责按 `batch_size = 16` 分批与 `np.vstack` 拼接。

## 文档存储抽象：一套契约，多种后端

向量算出来之后要落到 doc store。RAGFlow 在这里没有把自己绑死在 Elasticsearch 上，而是定义了一个抽象基类 `DocStoreConnection`（在 `common/doc_store/doc_store_base.py`），用 `@abstractmethod` 把所有后端必须实现的操作钉成契约。读这个基类，等于读懂了 RAGFlow 对"文档存储"这件事的全部要求：

- 生命周期类：`db_type`、`health`、`create_idx(index_name, dataset_id, vector_size, parser_id)`、`delete_idx`、`index_exist`。注意 `create_idx` 的签名里直接带着 `vector_size`——建索引时就要知道向量维度，这又回到了"维度即契约"。
- 数据操作类：`search`、`get`、`insert`、`update`、`delete`。
- 结果解析类：`get_total`、`get_doc_ids`、`get_fields`、`get_highlight`、`get_aggregation`、`sql`。

这套契约的精妙之处在于它**没有暴露任何后端特有的查询语法**。检索条件不是用 ES 的 DSL 或 SQL 写的，而是用一组与后端无关的表达式对象来描述，它们也定义在同一个文件里：`MatchTextExpr`（全文匹配）、`MatchDenseExpr`（稠密向量匹配）、`MatchSparseExpr`（稀疏向量）、`MatchTensorExpr`（张量/多向量）、以及 `FusionExpr`（融合）。比如稠密向量匹配的描述：

```python
class MatchDenseExpr:
    def __init__(self, vector_column_name, embedding_data, embedding_data_type,
                 distance_type, topn=DEFAULT_MATCH_VECTOR_TOPN, extra_options=None):
        self.vector_column_name = vector_column_name
        self.embedding_data = embedding_data
        self.embedding_data_type = embedding_data_type
        self.distance_type = distance_type
        self.topn = topn
        ...
```

调用方（第 6 章的检索层）只需用这些表达式描述"我想要什么"，而把"在某个具体后端上怎么实现"完全留给各个 `*Connection` 子类。还有一个 `OrderByExpr` 用链式调用 `.asc(field)`/`.desc(field)` 来声明排序，同样是后端无关的。这是一个把"意图"与"实现"彻底解耦的经典抽象——上层写一次，底层换引擎。

## 引擎选择：一个环境变量切换后端

具体用哪个后端，由 `common/settings.py` 在初始化时根据环境变量 `DOC_ENGINE` 决定，默认值是 `elasticsearch`：

```python
DOC_ENGINE = os.getenv('DOC_ENGINE', 'elasticsearch')
DOC_ENGINE_INFINITY = (DOC_ENGINE.lower() == "infinity")
DOC_ENGINE_OCEANBASE = (DOC_ENGINE.lower() == "oceanbase")
```

在 `init_settings()` 里，它根据这个值把全局的 `docStoreConn` 实例化成对应的连接类：

```python
if lower_case_doc_engine == "elasticsearch":
    docStoreConn = rag.utils.es_conn.ESConnection()
elif lower_case_doc_engine == "infinity":
    ...
    docStoreConn = rag.utils.infinity_conn.InfinityConnection()
elif lower_case_doc_engine == "opensearch":
    docStoreConn = rag.utils.opensearch_conn.OSConnection()
elif ...:
    docStoreConn = rag.utils.ob_conn.OBConnection()
else:
    raise Exception(f"Not supported doc engine: {DOC_ENGINE}")
```

整个系统其余部分都只跟全局的 `settings.docStoreConn` 这个抽象接口打交道，谁也不需要知道背后究竟是 ES 还是 Infinity。换引擎，对上层代码而言只是改一个环境变量。结尾那个 `raise Exception(f"Not supported doc engine: ...")` 也体现了 fail-closed：配错了引擎名直接启动失败，而不是悄悄退回某个默认行为。对象存储侧也是同样的套路，由 `STORAGE_IMPL_TYPE = os.getenv('STORAGE_IMPL', 'MINIO')` 决定用 MinIO 还是各家云存储。

索引名的生成也很简单统一，定义在 `rag/nlp/search.py`：`def index_name(uid): return f"ragflow_{uid}"`——每个租户一个以 `ragflow_` 为前缀的索引，知识库则通过 `kb_id`/`dataset_id` 在索引内部区分。

## 两套实现的差异：以 ES 与 Infinity 的 insert 为例

抽象基类规定了"做什么"，具体后端各自实现"怎么做"，而这两个"怎么做"差异巨大，恰好说明了为什么需要抽象层。

`ESConnection.insert`（在 `rag/utils/es_conn.py`）走的是 Elasticsearch 的 bulk API。它把每条文档拆成"操作行 + 数据行"两条，用文档自带的 `id` 作为 ES 的 `_id` 保证幂等，并在写入时 `refresh="wait_for"` 等待刷新可见：

```python
def insert(self, documents, index_name, knowledgebase_id=None):
    operations = []
    for d in documents:
        assert "_id" not in d
        assert "id" in d
        d_copy = copy.deepcopy(d)
        d_copy["kb_id"] = knowledgebase_id
        meta_id = d_copy.get("id", "")
        operations.append({"index": {"_index": index_name, "_id": meta_id}})
        operations.append(d_copy)
    ...
    r = self.es.bulk(index=index_name, operations=operations, refresh="wait_for", timeout="60s")
```

那两行 `assert "_id" not in d` 和 `assert "id" in d` 是把"文档必须自带业务 id、且不能预置 ES 内部 `_id`"这条不变量直接写成运行期断言——契约写进断言，是第 4 章也反复出现的风格。整个写入还包在一个 `for _ in range(ATTEMPT_TIME)` 的重试循环里，遇到 `ConnectionTimeout` 会 sleep 后 `self._connect()` 重连再试。

而 `InfinityConnection.insert`（在 `rag/utils/infinity_conn.py`）的处理则完全不同。Infinity 是强 schema 的向量数据库，写入前必须先确定向量字段及其维度。它的做法是从待写入文档的字段名里用正则反推维度——这正是前面"维度即字段名"约定发挥作用的地方：

```python
vector_size = 0
patt = re.compile(r"q_(?P<vector_size>\d+)_vec")
for k in documents[0].keys():
    m = patt.match(k)
    if m:
        vector_size = int(m.group("vector_size"))
if vector_size == 0:
    raise ValueError("Cannot infer vector size from documents")
...
self.create_idx(index_name, knowledgebase_id, vector_size, parser_id)
```

ES 是 schemaless、可以直接 bulk 灌入的，而 Infinity 需要先按推断出的维度建好带向量列的表结构。两种后端的写入逻辑几乎没有共同点，但因为它们都实现了 `DocStoreConnection.insert` 这同一个契约，上层管线（第 4 章的 `insert_chunks`）调用时完全感知不到差异。这就是抽象层的全部价值所在。

## 为什么元数据过滤要写两套

抽象做得再好，也总有"漏出来"的地方。RAGFlow 里一个诚实的例子是元数据过滤：`common/` 下同时存在 `metadata_es_filter.py` 和 `metadata_infinity_filter.py` 两份实现。既然有了统一抽象，为什么过滤逻辑还要按后端各写一套？因为元数据过滤涉及把用户的过滤条件翻译成后端原生的查询表达——ES 用的是布尔查询 DSL，Infinity 用的是类 SQL 的 filter 字符串，两者的语法、类型处理、转义规则根本无法用一套代码覆盖。

这恰恰印证了抽象的边界：当不同后端的差异是"接口形状"层面的（insert/search/delete），可以用基类契约抹平；但当差异深入到"查询语言语法"层面，强行统一反而会写出一个谁都不像的畸形中间层，不如老老实实按后端分别实现。RAGFlow 选择在前者上做强抽象、在后者上接受冗余，这是一个成熟的工程取舍。

## 本章小结

- RAGFlow 用 `q_%d_vec` 的命名约定把向量维度直接编码进字段名（如 `q_1024_vec`），形成一条贯穿写入与检索的隐形契约——"维度即字段名"，检索侧用正则 `q_[0-9]+_vec` 识别、Infinity 用 `q_(?P<vector_size>\d+)_vec` 反推维度。
- embedding 侧用抽象基类 `Base` 的模板方法 `_batched_encode` 收敛几十家 provider 的共性：逐条截断、批次循环、向量累加、token 统计、统一错误包装（`EmbeddingError`，但保留结构化的 `ModelException`）。
- `Base.__init__` 刻意只占位不存储参数，让所有 provider 构造签名一致，便于工厂统一注册；`_call` 闭包负责发请求与解析响应，并用 `_sorted_by_index` 保证向量与输入文本严格对齐。
- provider 默认值里写着对外部约束的认知，如 `OpenAIEmbed` 的 `batch_size=16`、`truncate_to=8191`。
- `BuiltinEmbed` 用类级单例 + `threading.Lock` 缓存重量级本地模型，仅在 `COMPOSE_PROFILES` 含 `tei-` 时加载，并用 `MAX_TOKENS` 表配置截断上限、未知模型回退到保守的 500（fail-closed）。
- doc store 用抽象基类 `DocStoreConnection` 把 ES/Infinity/OpenSearch/OceanBase 统一到一套契约（`create_idx`/`insert`/`search`/`delete` 等），并用后端无关的表达式对象（`MatchTextExpr`/`MatchDenseExpr`/`FusionExpr`/`OrderByExpr`）描述检索意图。
- 后端由环境变量 `DOC_ENGINE`（默认 `elasticsearch`）在 `init_settings()` 里选择并实例化全局 `docStoreConn`，配错引擎直接 `raise`（fail-closed）；索引名统一为 `ragflow_{uid}`。
- ES 的 `insert` 走 schemaless 的 bulk API、用断言固化"自带 id、不预置 `_id`"的契约；Infinity 的 `insert` 必须先推断维度再 `create_idx` 建表——两者实现迥异却共享同一契约。
- 元数据过滤刻意按后端各写一套（`metadata_es_filter.py` 与 `metadata_infinity_filter.py`），因为查询语言语法层面的差异无法被强行抽象——这是抽象边界的诚实取舍。

## 动手实验

1. **实验一：追踪"维度即字段名"的全链路** — 在 `/tmp/agent_explore` 里 `grep -rn "q_%d_vec\|q_(?P<vector_size>\|q_\[0-9\]+_vec"`，列出所有出现这个约定的位置（至少应找到 `rag/svr/task_executor.py` 写入处、`rag/utils/infinity_conn.py` 反推处、`rag/nlp/search.py` 检索识别处）。画出向量字段名从"写入"到"检索"是如何被生产和消费的。

2. **实验二：读懂模板方法的错误分层** — 阅读 `rag/llm/embedding_model.py` 的 `Base._batched_encode`，解释为什么它要 `except ModelException: raise` 单独放过一类异常、再用 `except Exception` 兜住其余并包成 `EmbeddingError`。再找到 `OpenAIEmbed._call` 里的 `_sorted_by_index`，思考：如果去掉这个排序，在什么情况下会导致向量与原文错位？

3. **实验三：验证后端可插拔** — 阅读 `common/settings.py` 的 `init_settings()` 中根据 `DOC_ENGINE` 选择 `docStoreConn` 的分支。再对照 `common/doc_store/doc_store_base.py` 里 `DocStoreConnection` 的全部 `@abstractmethod`，列一张表：每个抽象方法分别由 `es_conn.py` 和 `infinity_conn.py` 的哪个函数实现。这样你就得到了一份"契约 vs 实现"对照表。

4. **实验四：理解抽象的边界** — 对比 `common/metadata_es_filter.py` 与 `common/metadata_infinity_filter.py` 两份实现，找出它们各自把过滤条件翻译成了什么形式（ES 布尔 DSL vs Infinity filter 字符串）。再回头看 `doc_store_base.py` 里的 `MatchTextExpr`/`MatchDenseExpr` 等表达式对象，思考：为什么检索表达式可以统一抽象，而元数据过滤却要分后端写两套？把你的结论与本章"抽象边界"一节对照。

> **下一章预告**：向量与文本都进了库，真正的考验是查询阶段——如何把用户那句口语化的问题，变成一个既走向量召回又走全文召回、还能融合重排、最后逐句溯源的检索过程？下一章我们钻进 `rag/nlp` 的 `query.py`、`search.py`、`term_weight.py`，解构 RAGFlow 的多路召回与融合重排，看它如何兑现"grounded citations with reduced hallucinations"这个招牌承诺。
