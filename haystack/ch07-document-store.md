# 第 7 章 DocumentStore 抽象与内存实现

上一章我们把原始文档切成了一段段 `Document`，但切好的块要落到哪里去？检索时又如何根据关键词或向量把它们捞回来？这就是 DocumentStore（文档存储）的职责。Haystack 把"存"与"取"统一在一个协议契约之下：上层组件（Retriever、Writer）只认契约，不认具体后端，于是同一套管线既能跑内存存储，也能无缝换成 Elasticsearch、Qdrant、Weaviate 等外部向量库。

本章先解剖这份契约本身——`DocumentStore` 为什么用 `Protocol` 而非抽象基类；再解剖 `DuplicatePolicy` 去重策略与过滤 DSL 的结构；最后逐行拆开 `InMemoryDocumentStore`：它如何写入、如何做 BM25 关键词检索、如何做向量相似度检索。这是整个 Haystack 存储层最值得精读的一份单文件实现，因为它把所有抽象都落成了可运行的 Python。

## 7.1 契约设计：为什么是 Protocol 而不是 ABC

打开 `haystack/document_stores/types/protocol.py`，第一眼就能看到关键选择：

```python
from typing import Any, Protocol

class DocumentStore(Protocol):
    ...
```

`DocumentStore` 继承的是 `typing.Protocol`，而不是 `abc.ABC`。这是一个深思熟虑的架构决定。`Protocol` 描述的是"结构化类型"（structural typing），也就是俗称的鸭子类型：只要一个类长得像 `DocumentStore`（实现了同名同签名的方法），它在类型检查器眼里**就是**一个 `DocumentStore`，无需显式 `class XxxStore(DocumentStore)` 继承。Python 的 `typing.Protocol` 正是为这种结构化子类型而生的 [[typing.Protocol]](https://docs.python.org/3/library/typing.html#typing.Protocol)。

这对 Haystack 的生态至关重要。`InMemoryDocumentStore`（我们稍后会看到）的类定义是 `class InMemoryDocumentStore:`——它**根本没有继承 `DocumentStore`**。Elasticsearch、Qdrant、Pinecone 这些外部后端被拆成独立的 pip 子包，由不同团队维护。如果契约是 ABC，那么每个子包都得 `import` 核心包并继承基类，造成强耦合；而用 `Protocol`，外部库只需让自己的类"碰巧"具备 `count_documents`/`filter_documents`/`write_documents`/`delete_documents` 等方法，就能被任意 Retriever 接受。契约成了一份"约定"而非"枷锁"。

协议里每个方法体都只有 `...`（Ellipsis），它们不是实现，只是签名声明。核心契约由六个方法构成：序列化对 `to_dict`/`from_dict`，以及四个数据操作 `count_documents`/`filter_documents`/`write_documents`/`delete_documents`。

值得注意的是，`protocol.py` 中的 `DocumentStore` **没有**加 `@runtime_checkable` 装饰器。这意味着你不能用 `isinstance(store, DocumentStore)` 在运行时做检查——`Protocol` 默认只在静态类型检查（如 mypy）阶段生效。若要支持 `isinstance`，必须显式标注 `@runtime_checkable`。Haystack 这里选择只做静态契约，把运行期的"是否符合契约"交给真正调用方法时的鸭子行为去暴露。

## 7.2 写入去重：DuplicatePolicy 的四种策略

`write_documents` 的第二个参数是 `policy`，类型为 `DuplicatePolicy`。看 `haystack/document_stores/types/policy.py`，它就是一个朴素的字符串枚举：

```python
class DuplicatePolicy(Enum):
    NONE = "none"
    SKIP = "skip"
    OVERWRITE = "overwrite"
    FAIL = "fail"
```

四个取值的语义在协议 docstring 里写得很清楚：当待写入文档的 `id` 已存在时，`SKIP` 跳过不写，`OVERWRITE` 覆盖旧值，`FAIL` 抛 `DuplicateError`，而 `NONE` 是默认值，表示"具体行为由后端自己决定"。这里 `Document` 的 `id` 是内容哈希（前几章已述），因此"同 id"基本等价于"同内容"。

协议还规定了返回值语义：`write_documents` 返回**实际写入的文档数**。用 `OVERWRITE` 时该数恒等于输入长度；用 `SKIP` 时则可能小于输入长度，因为被跳过的不计入。这个细节在内存实现里有精确对应，下一节会看到。

## 7.3 过滤 DSL：comparison 与 logic 两类节点

`filter_documents` 接收的 `filters` 是一个嵌套字典，构成一套小型表达式语言（DSL）。协议 docstring 把它的语法说得很死板，只有两类节点：

- **比较节点（Comparison）**：必须含 `field`、`operator`、`value` 三个键。算子有 `==`、`!=`、`>`、`>=`、`<`、`<=`、`in`、`not in` 八种。
- **逻辑节点（Logic）**：必须含 `operator` 和 `conditions` 两个键。算子有 `AND`、`OR`、`NOT` 三种，`conditions` 是一组子节点（可继续嵌套）。

一个简单过滤就是单个比较节点 `{"field": "meta.type", "operator": "==", "value": "article"}`；复杂过滤则用逻辑节点把多个条件树状组合起来。

这套 DSL 的求值器在 `haystack/utils/filters.py` 的 `document_matches_filter` 里。它的分发逻辑很优雅：

```python
def document_matches_filter(filters, document):
    if "field" in filters:
        return _comparison_condition(condition=filters, document=document)
    return _logic_condition(condition=filters, document=document)
```

看到 `field` 键就当比较节点处理，否则当逻辑节点。逻辑算子被注册成一张表 `LOGICAL_OPERATORS = {"NOT": _not, "OR": _or, "AND": _and}`，其中 `_and` 是 `all(...)`、`_or` 是 `any(...)`、`_not` 是 `not _and(...)`。比较算子同样是一张表 `COMPARISON_OPERATORS`，把 `"=="` 映射到 `_equal`、`">"` 映射到 `_greater_than`，以此类推。这种"算子字典 + 闭包函数"的写法，让新增算子只是往字典里加一行。

`_comparison_condition` 处理字段取值时有三条分支，体现了对元数据的友好处理：若 `field` 含 `.`（如 `meta.person.name`），就按点路径逐层下钻到嵌套字典；若 `field` 不是 `Document` 的真实字段名，就当作元数据键，从 `document.meta` 里取（这是为兼容旧版未加 `meta.` 前缀的过滤器）；否则直接 `getattr`。此外，排序类算子（`>`/`>=`/`<`/`<=`）在 `_prepare_ordering_comparison` 里会尝试把字符串解析成日期再比较，并对 `None` 值统一返回 `False`，避免不可靠比较。

至于 `filter_policy.py` 里的 `FilterPolicy`（`REPLACE`/`MERGE`）和那一组 `combine_*` 函数，则是另一个层面的事：它解决的是 Retriever 在"初始化时设的过滤器"与"运行时传入的过滤器"如何合并的问题，与 DocumentStore 内部如何匹配文档无关。`MERGE` 策略会按比较/逻辑节点的四种组合形态分别走 `combine_two_comparison_filters`、`combine_two_logical_filters` 等函数把两套过滤器拼起来；`REPLACE` 则直接用运行时的覆盖初始化的。

## 7.4 InMemoryDocumentStore：进程内的全局存储

现在进入主菜 `haystack/document_stores/in_memory/document_store.py`。类注释一句话点题："Stores data in-memory. It's ephemeral and cannot be saved to disk."（数据只在内存里，是易逝的）。但它其实提供了 `save_to_disk`/`load_from_disk` 把数据序列化成 JSON 落盘的旁路。

最有意思的是它的存储不是实例属性，而是**模块级全局字典**，以 `index` 名为键：

```python
_STORAGES: dict[str, dict[str, Document]] = {}
_BM25_STATS_STORAGES: dict[str, dict[str, BM25DocumentStats]] = {}
_AVERAGE_DOC_LEN_STORAGES: dict[str, float] = {}
_FREQ_VOCAB_FOR_IDF_STORAGES: dict[str, Counter] = {}
```

构造函数里若不传 `index`，就生成一个随机 `uuid.uuid4()`。这样设计的好处写在 docstring 里：多个 `InMemoryDocumentStore` 实例若共享同一个 `index`，就能共享同一份文档——`storage` 属性只是 `_STORAGES.get(self.index, {})` 的取值代理。

除了文档本身，这里还维护了三份 BM25 统计：每篇文档的词频统计 `_bm25_attr`（值是 `BM25DocumentStats` 数据类，含 `freq_token` 词频 Counter 和 `doc_len` 文档长度）、全语料平均文档长度 `_avg_doc_len`、以及用于计算 IDF 的全局词表频次 `_freq_vocab_for_idf`。这些统计是**增量维护**的，写入和删除时实时更新，避免每次检索都重扫全量。

构造函数的几个关键默认值要记牢：

- `bm25_tokenization_regex = r"(?u)\b\w+\b"`：分词正则，匹配 Unicode 单词边界内的字母数字串。
- `bm25_algorithm = "BM25L"`：**默认算法是 BM25L**，不是常见的 Okapi。可选 `"BM25Okapi"`、`"BM25L"`、`"BM25Plus"`。
- `bm25_parameters = None`：算法参数字典，为空时各算法用各自的内置默认 `k1`/`b`/`delta`/`epsilon`。
- `embedding_similarity_function = "dot_product"`：向量相似度**默认用点积**，可选 `"cosine"`。
- `return_embedding = True`：默认在结果里带回向量。

构造时 `_dispatch_bm25()` 会按 `bm25_algorithm` 从一张表里查出对应的打分方法存进 `self.bm25_algorithm_inst`，不支持的名字直接 `ValueError`。

## 7.5 写入与删除：policy 分支与统计的增量维护

`write_documents` 的实现紧扣协议契约，又补了一条内存特有的规则——docstring 明说："If `policy` is set to `DuplicatePolicy.NONE` defaults to `DuplicatePolicy.FAIL`."

```python
if policy == DuplicatePolicy.NONE:
    policy = DuplicatePolicy.FAIL
```

也就是说，内存存储把"默认行为未定义"的 `NONE` 落实成了最保守的 `FAIL`：重复就报错。接着 `written_documents = len(documents)` 作为返回计数的起点，逐篇处理：

- 若 `policy` 不是 `OVERWRITE` 且 `id` 已存在：`FAIL` 抛 `DuplicateDocumentError`；`SKIP` 打日志、`written_documents -= 1` 后 `continue`。这正好对应了协议里"SKIP 时返回数可能小于输入"的约定。
- 即便是 `OVERWRITE`，只要 `id` 已存在，也会先 `self.delete_documents([document.id])` 把旧的删掉。原因写在注释里：BM25 统计是增量维护的，必须先回退旧文档的统计贡献，再加新文档，否则统计会被污染。

写入时的统计更新是这段代码的精髓：

```python
tokens = []
if document.content is not None:
    tokens = self._tokenize_bm25(document.content)
self.storage[document.id] = document
self._bm25_attr[document.id] = BM25DocumentStats(Counter(tokens), len(tokens))
self._freq_vocab_for_idf.update(set(tokens))
n_docs = len(self._bm25_attr)
self._avg_doc_len = (len(tokens) + self._avg_doc_len * (n_docs - 1)) / n_docs
```

注意 `_freq_vocab_for_idf.update(set(tokens))` 用的是 `set(tokens)` 而非 `tokens`——这是**文档频率**（document frequency，包含该词的文档数），不是词频，因为 IDF 要的是"多少篇文档出现过这个词"。平均长度也用旧均值乘旧文档数加新长度再除以新文档数的滚动公式更新。

`delete_documents` 是对称的逆操作：从 `storage` 删除，`pop` 出该文档的统计，用 `_freq_vocab_for_idf.subtract(Counter(freq.keys()))` 回退文档频率，并用对称的滚动公式把平均长度减回去；当删到没有文档时捕获 `ZeroDivisionError` 把均值置 0。

`filter_documents` 则直接复用 `document_matches_filter` 做列表推导过滤；若 `return_embedding` 为 `False`，会用 `dataclasses.replace(doc, embedding=None)` 返回去掉向量的副本。

## 7.6 BM25 检索：三种算法变体与打分排序

`bm25_retrieval(query, filters, top_k=10, scale_score=False)` 是关键词检索入口。BM25 是经典的概率检索打分函数，综合词频、逆文档频率和文档长度归一 [[Okapi BM25]](https://en.wikipedia.org/wiki/Okapi_BM25)。它的流程是：

1. 空 query 直接 `ValueError`。
2. 强制叠加一个内容非空过滤 `{"field": "content", "operator": "!=", "value": None}`，若用户还传了 filters，就用 `AND` 把两者并起来。这保证只对有文本的文档打分。
3. 调 `filter_documents` 取候选集；为空则返回空列表。
4. 若 `_avg_doc_len == 0`（全语料无文本），给每篇打 `0.0` 分以避免除零；否则调 `self.bm25_algorithm_inst(query, all_documents)` 真正打分。
5. 按分数降序排序取 `top_k`。
6. 把分数写进 `Document` 的 `score` 字段返回。

三种打分方法 `_score_bm25l`/`_score_bm25okapi`/`_score_bm25plus` 结构相似，都是"先算 query 各 token 的 IDF，再对每篇文档累加 IDF×TF"，区别在 IDF 公式和 TF 归一公式：

- **BM25L**（默认）：参数 `k1=1.5`、`b=0.75`、`delta=0.5`。IDF 用 `log((N+1)/(n+0.5))` 并乘 `int(n != 0)`（未出现的词 IDF 记 0）；TF 引入 `ctd = freq / (1 - b + b*doc_len/avg_len)` 后再加 `delta`，缓解 BM25 对长文档的过度惩罚。
- **BM25Okapi**：参数 `k1=1.5`、`b=0.75`、`epsilon=0.25`。IDF 用 `log((N - n + 0.5)/(n + 0.5))`，对出现过多导致 IDF 为负的词，用 `epsilon * 平均IDF` 兜底成正值。它是唯一**可能返回有意义负分**的算法。
- **BM25Plus**：参数 `k1=1.5`、`b=0.75`、`delta=1.0`。IDF 用带平滑的 `log(1 + (N - n + 0.5)/(n + 0.5))`，TF 末尾加常数 `delta` 保证每个匹配词都有正向下界贡献。

打分后还有两处收尾逻辑值得注意。其一是分数缩放：若 `scale_score=True`，用 `expit(score / BM25_SCALING_FACTOR)` 把无界分数压到 0~1，`BM25_SCALING_FACTOR = 8`（注释解释这个值是经验选的，越大压得越狠）。其二是负分过滤：代码用 `negatives_are_valid = self.bm25_algorithm == "BM25Okapi" and not scale_score` 判定——只有 Okapi 且未缩放时才保留 `score <= 0.0` 的结果，其余情况一律 `continue` 丢弃非正分文档。这是因为只有 Okapi 的负分有检索意义，其他算法的非正分是噪声。

分词由 `_tokenize_bm25` 完成：先 `text.lower()` 全部小写，再用编译好的正则 `findall`。小写化被刻意抽进这个方法里，注释说是为了把所有 BM25 预处理逻辑集中、可追踪。

## 7.7 向量检索：余弦与点积的统一实现

`embedding_retrieval(query_embedding, filters, top_k=10, scale_score=False, return_embedding=False)` 是向量检索入口：

1. 校验 `query_embedding` 是非空 float 列表。
2. 按 filters（如有）筛候选，再筛出 `embedding is not None` 的文档；若一篇带向量的都没有，告警并返回空。
3. 调 `_compute_query_embedding_similarity_scores` 算相似度。
4. 用 `zip(..., strict=True)` 把文档和分数配对，降序取 `top_k`，写入 `score` 字段返回；`return_embedding` 为假时把结果里的 `embedding` 置 `None`。

相似度计算 `_compute_query_embedding_similarity_scores` 是纯 NumPy 实现。它把 query 和所有文档向量堆成矩阵，核心是一次 `np.dot(query_embedding, document_embeddings.T)`。两种相似度的差别仅在于是否先归一化：

```python
if self.embedding_similarity_function == "cosine":
    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    document_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
    query_embedding /= np.where(query_norm == 0.0, 1.0, query_norm)
    document_embeddings /= np.where(document_norms == 0.0, 1.0, document_norms)
```

余弦相似度本质就是"先把向量归一化再做点积"，所以代码用同一条 `np.dot` 路径，只在 `cosine` 分支前对两边各除以自身 L2 范数。这里用 `np.where(norm == 0.0, 1.0, norm)` 防止零向量除零产生 NaN——遇到零范数就除以 1.0。

若向量维度不一致，`np.array` 会抛"inhomogeneous shape"，代码捕获后转成更友好的 `DocumentStoreError`，提示"所有文档应用同一模型嵌入"。

最后是缩放：`scale_score=True` 时，`dot_product` 用 `expit(score / DOT_PRODUCT_SCALING_FACTOR)`（`DOT_PRODUCT_SCALING_FACTOR = 100`）压到 0~1；`cosine` 因本就落在 -1~1，直接用 `(score + 1) / 2` 线性映射到 0~1。

## 7.8 异步镜像与可观测细节

文件后半段是一组 `*_async` 方法：`write_documents_async`、`bm25_retrieval_async`、`embedding_retrieval_async` 等。它们的实现千篇一律——`asyncio.get_running_loop().run_in_executor(self.executor, lambda: 同步方法(...))`，即把同步方法丢进一个线程池执行。线程池 `self.executor` 在构造时创建（默认单 worker），`_owns_executor` 标记是否自建，`__del__`/`shutdown` 负责回收。因为内存操作本质是 CPU 同步的，这里用线程池"伪异步"只是为了不阻塞事件循环，让内存存储能嵌进异步管线。

此外还有一批运营辅助方法不属于核心协议但很实用：`update_by_filter`/`delete_by_filter` 按过滤批量改删、`count_documents_by_filter` 按过滤计数、`get_metadata_fields_info` 推断各元数据字段类型、`get_metadata_field_min_max`/`get_metadata_field_unique_values` 做元数据聚合。它们都建立在同一个 `document_matches_filter` 之上，是契约之外针对内存后端的增值能力。

## 本章小结

- `DocumentStore` 用 `typing.Protocol` 而非 ABC 定义契约，靠结构化（鸭子）类型让外部向量库子包无需继承核心包即可接入，实现松耦合。
- 协议未加 `@runtime_checkable`，契约只在静态类型检查层面生效，运行期不支持 `isinstance` 校验。
- 核心契约是六个方法：`to_dict`/`from_dict` 加 `count_documents`/`filter_documents`/`write_documents`/`delete_documents`。
- `DuplicatePolicy` 是 `NONE`/`SKIP`/`OVERWRITE`/`FAIL` 四值枚举；`InMemoryDocumentStore` 把默认的 `NONE` 落实为 `FAIL`，`SKIP` 会让返回的写入数小于输入数。
- 过滤 DSL 只有比较节点（`field`/`operator`/`value`）和逻辑节点（`operator`/`conditions`）两类，求值器 `document_matches_filter` 用算子字典加闭包函数实现，支持点路径下钻元数据。
- `InMemoryDocumentStore` 把数据与 BM25 统计放在以 `index` 为键的模块级全局字典里，多实例可共享同一 `index`，统计随写入/删除增量维护。
- BM25 默认算法是 `BM25L`（不是 Okapi），三种变体公共参数 `k1=1.5`、`b=0.75`，分词先小写化再用正则 `(?u)\b\w+\b`；只有 `BM25Okapi` 且未缩放时才保留非正分。
- 向量检索的余弦与点积共用一条 `np.dot` 路径，余弦只是多了一步 L2 归一化，并用 `np.where` 防零向量除零。
- `scale_score` 用 `expit` 把无界分数压到 0~1，BM25 缩放因子 `8`、点积缩放因子 `100`，余弦则用 `(score+1)/2`。
- 所有 `*_async` 方法只是把同步方法丢进线程池执行，为嵌入异步管线服务。

## 动手实验

1. **实验一：验证 Protocol 的鸭子类型** — 写一个最小类，只实现 `count_documents`/`filter_documents`/`write_documents`/`delete_documents` 四个方法但不继承 `DocumentStore`；用 mypy 把它赋给一个标注为 `DocumentStore` 的变量，确认静态检查通过；再尝试 `isinstance(obj, DocumentStore)`，观察因缺少 `@runtime_checkable` 而报错，体会"结构化类型只在静态层生效"。

2. **实验二：观测 DuplicatePolicy 的写入计数** — 构造同一篇内容的两个 `Document`，分别用 `policy=DuplicatePolicy.SKIP` 和 `OVERWRITE` 调 `write_documents`，打印返回值与 `count_documents()`；再用默认 `policy`（即 `NONE`→`FAIL`）写重复 id，捕获 `DuplicateDocumentError`，验证三种策略与协议约定的返回语义一致。

3. **实验三：对比三种 BM25 算法** — 用同一组文档分别建三个 `InMemoryDocumentStore`（`bm25_algorithm` 设为 `BM25L`/`BM25Okapi`/`BM25Plus`），对同一 query 调 `bm25_retrieval(scale_score=False)`，打印每篇的 `score` 与排序；特别构造一个高频词 query 触发 Okapi 的负分，验证只有 Okapi 会保留非正分文档。

4. **实验四：余弦 vs 点积的相似度差异** — 准备几篇带 `embedding` 的文档（含一个模长很大的向量），分别用 `embedding_similarity_function="cosine"` 和 `"dot_product"` 调 `embedding_retrieval`，对比两种度量下的排序差异；再传入一个零向量文档，确认 `np.where` 防除零让余弦不产生 NaN。

> **下一章预告**：文档存进去了，但 BM25 之外的"语义检索"靠的是向量，而向量从何而来？第 8 章我们将解构嵌入器 Embedders——`SentenceTransformersDocumentEmbedder` 与 `TextEmbedder` 如何把文本批量编码成向量、`meta` 字段如何拼进待编码文本、批处理与归一化参数怎样影响最终的检索质量。
