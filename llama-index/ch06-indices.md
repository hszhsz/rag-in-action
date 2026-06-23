# 第 6 章 索引体系 Indices

第 5 章里,`IngestionPipeline` 把原始 `Document` 经过一连串 `TransformComponent` 揉成了一批 `node`——切分好、附上元数据、(如果配置了 embed_model)甚至带上了 embedding 向量。但一堆零散的 node 还不能直接服务于检索:同样是这批 node,你可以把它们扔进向量库做语义召回,也可以建一张关键词倒排表做精确匹配,还可以组织成一棵树做层次摘要。**node 是原料,Index 是把原料组织成"可被高效检索的结构"的那一步**。

这一章我们只讲"组织"。Index 关心的是"node 以什么数据结构存放、用什么元数据描述它们之间的关系",而"给定一个 query 怎么把相关 node 捞出来"是 Retriever 的职责——那是下一章的内容。LlamaIndex 把这两件事拆得很干净:每个 Index 通过工厂方法 `as_retriever()` 产出一个配套的检索器,Index 负责建结构,Retriever 负责查结构。理解了这条分界线,本章的源码就有了主线。

## `BaseIndex`:从 Document 到索引结构的统一骨架

所有索引都继承自 `llama_index/core/indices/base.py` 的 `BaseIndex`。它是一个泛型抽象基类 `BaseIndex(Generic[IS], ABC)`,其中类型变量 `IS` 被约束为 `IndexStruct` 的子类(`IS = TypeVar("IS", bound=IndexStruct)`)。也就是说,每种索引都绑定一个专属的"索引结构"类型,这个类型存放在类属性 `index_struct_cls` 上。

最常用的入口是类方法 `from_documents`。它的逻辑非常清楚:先把每篇文档的哈希写进 docstore(`docstore.set_document_hash(doc.id_, doc.hash)`,为后续增量 refresh 做准备),再调用 `run_transformations(documents, transformations, ...)` 把 `Document` 变成 node,最后用这批 node 调用构造函数:

```python
nodes = run_transformations(
    documents, transformations, show_progress=show_progress, **kwargs,
)
return cls(nodes=nodes, storage_context=storage_context, ...)
```

注意这里 `from_documents` 复用的正是第 5 章的 `run_transformations`——索引构建的"文档 → node"环节,本质上就是跑一遍摄入管线。`transformations` 若不显式传入,默认取 `Settings.transformations`,与全局配置打通。

进了构造函数 `__init__`,真正的"建索引"发生在这一段:

```python
with self._callback_manager.as_trace("index_construction"):
    if index_struct is None:
        nodes = nodes or []
        index_struct = self.build_index_from_nodes(nodes + objects, **kwargs)
    self._index_struct = index_struct
    self._storage_context.index_store.add_index_struct(self._index_struct)
```

`build_index_from_nodes` 是 `BaseIndex` 提供的模板方法,它先把 node 落进 docstore(`self._docstore.add_documents(nodes, allow_update=True)`),再委托给抽象方法 `_build_index_from_nodes`——这是每个子类必须实现的核心钩子,决定"node 到底怎么组织"。这是典型的模板方法模式:公共的"写 docstore + 注册 index_struct"由基类统一负责,差异化的"构建结构"留给子类。

构造函数还揭示了 Index 与存储的关系。`BaseIndex` 自己不持有存储,而是从 `storage_context` 里取出三套存储:

```python
self._storage_context = storage_context or StorageContext.from_defaults()
self._docstore = self._storage_context.docstore
self._vector_store = self._storage_context.vector_store
self._graph_store = self._storage_context.graph_store
```

`docstore` 存 node 全文,`vector_store` 存向量,`index_store` 存 index_struct 这份元数据。构建完成后那一行 `index_store.add_index_struct(...)` 就是把"索引结构"注册进 index_store,这正是后面持久化与还原的关键。

`BaseIndex` 还把"用 Index 干活"的两个工厂方法定下来:抽象方法 `as_retriever(**kwargs) -> BaseRetriever`(子类各自实现,返回配套检索器),以及具体方法 `as_query_engine`。后者很薄——它先调 `self.as_retriever(**kwargs)` 拿到检索器,再用 `RetrieverQueryEngine.from_args(retriever, llm=llm, **kwargs)` 包一层:

```python
retriever = self.as_retriever(**kwargs)
llm = resolve_llm(llm, ...) if llm else Settings.llm
return RetrieverQueryEngine.from_args(retriever, llm=llm, **kwargs)
```

这条 `Index → Retriever → QueryEngine` 的串联,是 LlamaIndex 整套问答链路的脊梁。此外 `as_chat_engine` 在此基础上按 `ChatMode` 进一步包成对话引擎(`CONTEXT`/`CONDENSE_QUESTION`/`CONDENSE_PLUS_CONTEXT` 等)。

除了构建,`BaseIndex` 还统一了增删改:`insert(document)` 走 `run_transformations` 再 `insert_nodes`;`delete_ref_doc(ref_doc_id)` 通过 docstore 的 `get_ref_doc_info` 拿到该文档对应的所有 `node_ids` 再批量删除;`refresh_ref_docs` 则比对文档哈希,只对"新增或内容变更"的文档做插入/更新,从而省下重复的 embedding 调用。这些方法每一步结束都会再次 `index_store.add_index_struct(self._index_struct)`,保证元数据与实际数据同步。

## `IndexStruct`:作为可持久化元数据的"索引结构"

那么被反复读写的 `index_struct` 到底是什么?答案在 `llama_index/core/data_structs/data_structs.py`。基类 `IndexStruct` 是一个 `@dataclass`,且混入了 `DataClassJsonMixin`——这意味着它天生可以序列化成 JSON。它只有两个公共字段:

```python
@dataclass
class IndexStruct(DataClassJsonMixin):
    index_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    summary: Optional[str] = None

    @classmethod
    @abstractmethod
    def get_type(cls) -> IndexStructType: ...
```

注意它**不存 node 的正文**(文件头注释明确写着 "Nodes are decoupled from the indices")。IndexStruct 只是一份"node 如何组织"的描述性元数据:哪些 node id、它们之间的父子/倒排/顺序关系。正文在 docstore 里,向量在 vector_store 里,IndexStruct 只持有"地图"。每个子类用 `get_type()` 返回自己在 `IndexStructType`(`data_structs/struct_type.py` 里的枚举)里的标识,例如 `IndexStructType.VECTOR_STORE`、`LIST`、`TREE`、`KEYWORD_TABLE`——这个标识就是后面注册表查表还原的钥匙。

不同的索引用不同的 IndexStruct 子类来表达不同的组织方式:

- `IndexDict`(`get_type → VECTOR_STORE`):核心字段 `nodes_dict: Dict[str, str]`,从"向量库里的 id"映射到"node 的 doc_id"。`add_node(node, text_id)` 把 `nodes_dict[vector_id] = node.node_id`。
- `IndexList`(`get_type → LIST`):字段就是一个 `nodes: List[str]`,按插入顺序追加 node id——最朴素的"一条线"。
- `KeywordTable`(`get_type → KEYWORD_TABLE`):字段 `table: Dict[str, Set[str]]`,关键词到 node id 集合的倒排映射,`add_node(keywords, node)` 把每个关键词都指向该 node。
- `IndexGraph`(`get_type → TREE`):字段 `all_nodes`、`root_nodes`、`node_id_to_children_ids`,完整刻画一棵树的层次关系。

可以看到,IndexStruct 就是这套设计的精华:**把"组织结构"抽象成一份与正文解耦、可 JSON 化的轻量元数据**。

## `VectorStoreIndex`:嵌入并写入向量库(重头戏)

最常用的索引是 `llama_index/core/indices/vector_store/base.py` 里的 `VectorStoreIndex`,它绑定的结构正是 `IndexDict`(`index_struct_cls = IndexDict`,且类声明为 `VectorStoreIndex(BaseIndex[IndexDict])`)。它在构造时解析出 embed_model:`self._embed_model = resolve_embed_model(embed_model or Settings.embed_model, ...)`,并有一个默认 `insert_batch_size: int = 2048` 控制批大小。

它重写了 `build_index_from_nodes`:先过滤掉没有内容的 node(`node.get_content(metadata_mode=MetadataMode.EMBED) != ""`),因为空文本嵌入无意义,然后交给 `_build_index_from_nodes`。后者根据 `use_async` 选择同步的 `_add_nodes_to_index` 或异步的 `_async_add_nodes_to_index`,二者逻辑对称。

核心是 `_add_nodes_to_index`。它用 `iter_batch(nodes, self._insert_batch_size)` 把 node 切成批,逐批做两件事——嵌入、写库:

```python
for nodes_batch in iter_batch(nodes, self._insert_batch_size):
    nodes_batch = self._get_node_with_embedding(nodes_batch, show_progress)
    new_ids = self._vector_store.add(nodes_batch, **insert_kwargs)
```

`_get_node_with_embedding` 是嵌入环节:它调用 `indices/utils.py` 的 `embed_nodes(nodes, self._embed_model, ...)`。`embed_nodes` 很聪明——只对 `node.embedding is None` 的 node 收集文本,统一调用 `embed_model.get_text_embedding_batch(texts_to_embed, ...)` 做**批量嵌入**(已带 embedding 的 node 直接复用,呼应第 5 章管线里可能已经算过的向量),最后返回 `id_to_embed_map`。`_get_node_with_embedding` 再把向量逐个 `model_copy()` 回填到 node 的 `embedding` 字段。

拿到带向量的 node 后,`self._vector_store.add(nodes_batch)` 把它们写进向量库,返回向量库分配的 `new_ids`。接下来有个值得玩味的分支:

```python
if not self._vector_store.stores_text or self._store_nodes_override:
    for node, new_id in zip(nodes_batch, new_ids):
        node_without_embedding = node.model_copy()
        node_without_embedding.embedding = None
        index_struct.add_node(node_without_embedding, text_id=new_id)
        self._docstore.add_documents([node_without_embedding], allow_update=True)
else:
    # 向量库自己存了正文,就只把 image/index 节点登记进 index_struct
    for node, new_id in zip(nodes_batch, new_ids):
        if isinstance(node, (ImageNode, IndexNode)):
            ...
```

这里体现了一条避免重复存储的工程取舍:**如果向量库本身能存正文(`stores_text` 为真),就不再把 node 重复塞进 docstore 和 index_struct**,只有当向量库不存正文、或用户显式 `store_nodes_override=True` 时,才把 node 登记进 `IndexDict.nodes_dict` 和 docstore。写入时都会先 `model_copy()` 并把 `embedding` 置空,避免向量在 docstore 里被二次保存。这也解释了为什么纯向量库后端下 `ref_doc_info` 会抛 `NotImplementedError`——正文根本不在 docstore,无从重建映射。

`VectorStoreIndex.as_retriever` 返回 `VectorIndexRetriever`,并把 `node_ids=list(self.index_struct.nodes_dict.values())` 传进去——检索器要查哪些 node,正是从 IndexStruct 这份"地图"里读出来的:

```python
return VectorIndexRetriever(
    self, node_ids=list(self.index_struct.nodes_dict.values()), ...
)
```

此外还有便捷构造 `from_vector_store(vector_store, ...)`:它要求 `vector_store.stores_text` 为真,然后以 `nodes=[]` 建一个"空壳" Index——既有数据已在向量库里,无需重新嵌入即可直接检索。

## 其它索引类型:同一骨架下的不同组织策略

`VectorStoreIndex` 是语义检索的主力,但 LlamaIndex 还提供了几种用不同 IndexStruct 表达不同检索哲学的索引,它们都共用 `BaseIndex` 骨架,差异全在 `_build_index_from_nodes` 与配套 Retriever:

- **`SummaryIndex`**(`indices/list/base.py`,旧名 `ListIndex`/`GPTListIndex`):它的 `_build_index_from_nodes` 朴素到极致——`index_struct = IndexList()` 然后 `for n in nodes: index_struct.add_node(n)`,把 node id 顺序追加进一个列表,**构建期完全不调 embedding**。它的 `as_retriever` 支持 `ListRetrieverMode` 的 `DEFAULT`(全量返回)、`EMBEDDING`、`LLM` 三种模式。适用场景:node 量不大、希望"全量喂给 LLM 做摘要/汇总"的任务。
- **`KeywordTableIndex`**(`indices/keyword_table/base.py`,基类 `BaseKeywordTableIndex`):用 `KeywordTable` 这个倒排结构。构建时对每个 node 抽关键词(默认 `max_keywords_per_chunk: int = 10`,模板 `DEFAULT_KEYWORD_EXTRACT_TEMPLATE`)再 `table[keyword].add(node_id)`。`as_retriever` 支持 `DEFAULT`(LLM 抽词)、`SIMPLE`、`RAKE` 三种关键词抽取方式。适用场景:关键词精确匹配。
- **`TreeIndex`**(`indices/tree/base.py`):用 `IndexGraph`,把 node 自底向上递归摘要成一棵树(`all_nodes`/`root_nodes`/`node_id_to_children_ids`)。适用场景:长文档的层次化摘要与自顶向下检索。
- **`DocumentSummaryIndex`**(`indices/document_summary/base.py`):用 `IndexDocumentSummary`。它为每篇文档生成一段摘要(默认查询 `DEFAULT_SUMMARY_QUERY`,默认 `embed_summaries=True` 会把摘要也嵌入),把"摘要 → 底层 node"建立映射。检索时先用摘要粗筛文档,再取其 node,兼顾召回质量与开销。

一句话对比:`VectorStoreIndex` 靠向量相似度,`KeywordTableIndex` 靠倒排词表,`SummaryIndex` 靠全量遍历,`TreeIndex` 靠层次结构,`DocumentSummaryIndex` 靠摘要粗筛——同一批 node,五种"地图"。

## 注册表与 `load_index_from_storage`:持久化还原

既然 IndexStruct 可 JSON 化、又携带 `get_type()` 标识,从磁盘还原索引就有了清晰路径。`indices/registry.py` 维护一张全局映射表 `INDEX_STRUCT_TYPE_TO_INDEX_CLASS`,把每个 `IndexStructType` 关联到对应的 Index 类:

```python
INDEX_STRUCT_TYPE_TO_INDEX_CLASS = {
    IndexStructType.TREE: TreeIndex,
    IndexStructType.LIST: SummaryIndex,
    IndexStructType.KEYWORD_TABLE: KeywordTableIndex,
    IndexStructType.VECTOR_STORE: VectorStoreIndex,
    IndexStructType.DOCUMENT_SUMMARY: DocumentSummaryIndex,
    ...
}
```

`indices/loading.py` 的 `load_indices_from_storage` 利用这张表完成还原:从 `storage_context.index_store` 里取出一个或多个 `index_struct`,对每个调 `type_ = index_struct.get_type()` 拿到类型,再 `index_cls = INDEX_STRUCT_TYPE_TO_INDEX_CLASS[type_]` 查到类,最后:

```python
index = index_cls(index_struct=index_struct, storage_context=storage_context, **kwargs)
```

注意这里走的是"传 `index_struct` 而非 `nodes`"的构造路径——回看 `BaseIndex.__init__`,当 `index_struct` 不为空时它会**跳过 `build_index_from_nodes`**,直接复用已有结构,因此还原不会重新嵌入或重建,只是把元数据装回内存。对外暴露的 `load_index_from_storage(storage_context, index_id=None)` 是它的单索引便捷封装:不传 `index_id` 时要求 index_store 里只有一个索引,否则报错要求指定。

这套机制正是第 1 章 `BaseComponent` 可序列化契约的延续:node 由 docstore 持久化,向量由 vector_store 持久化,而"它们如何组织"由可 JSON 化的 IndexStruct 持久化,三者各司其职,靠 `index_id` 与 `IndexStructType` 串回一个完整的 Index。

## 本章小结

- `BaseIndex` 是所有索引的泛型抽象基类,绑定一个 `IndexStruct` 子类(类属性 `index_struct_cls`),并用模板方法模式统一了"建/增/删/改"。
- `from_documents` 通过 `run_transformations` 把 `Document` 变成 node(复用第 5 章管线),再委托 `build_index_from_nodes` → 抽象钩子 `_build_index_from_nodes` 完成实际构建。
- `build_index_from_nodes` 在基类里负责"写 docstore + 注册 index_struct"的公共逻辑,差异化的结构构建留给子类实现。
- `IndexStruct` 是一份与 node 正文解耦、可 JSON 化的元数据,只描述"node 如何组织";`IndexDict`/`IndexList`/`KeywordTable`/`IndexGraph` 分别对应向量、列表、倒排、树四种结构。
- `VectorStoreIndex` 以 `insert_batch_size=2048` 分批,经 `embed_nodes` 的 `get_text_embedding_batch` 批量嵌入,再 `vector_store.add` 写库;已带 embedding 的 node 会被复用。
- 当向量库 `stores_text` 时,node 不再重复写入 docstore 与 IndexStruct,这是避免重复存储的工程取舍。
- 各索引共用骨架但检索哲学不同:向量相似度、倒排词表、全量遍历、层次摘要、摘要粗筛。
- `registry.py` 的 `INDEX_STRUCT_TYPE_TO_INDEX_CLASS` 把 `IndexStructType` 映射到 Index 类,`load_index_from_storage` 据此查表并以 `index_struct=...` 路径还原(跳过重建)。
- `as_retriever`(抽象)与 `as_query_engine`(`RetrieverQueryEngine.from_args` 封装)构成 `Index → Retriever → QueryEngine` 的问答主链。
- Index 只负责"组织结构",检索算法是下一章 Retriever 的职责,二者通过 `as_retriever` 衔接。

## 动手实验

1. **实验一:追踪 from_documents 的 node 化路径** — 在 `/tmp/llama_explore/llama-index-core/llama_index/core/indices/base.py` 中阅读 `from_documents`,确认它先 `docstore.set_document_hash` 再 `run_transformations`,最后以 `nodes=...` 调构造函数;对照 `__init__` 里 `index_struct is None` 分支验证 `build_index_from_nodes` 何时被触发。
2. **实验二:验证向量库存正文时的去重逻辑** — 在 `vector_store/base.py` 的 `_add_nodes_to_index` 中,定位 `if not self._vector_store.stores_text or self._store_nodes_override` 分支,梳理"向量库存正文"与"不存正文"两种情况下 node 是否写入 docstore/`IndexStruct`,并解释 `ref_doc_info` 为何在前一种情况抛 `NotImplementedError`。
3. **实验三:对比四种 IndexStruct 的字段** — 在 `data_structs/data_structs.py` 中并排阅读 `IndexDict`、`IndexList`、`KeywordTable`、`IndexGraph`,记录各自的核心字段与 `add_node` 实现,以及 `get_type()` 返回的 `IndexStructType`,理解"同一批 node、四种组织"。
4. **实验四:走一遍持久化还原链路** — 在 `indices/registry.py` 找到 `INDEX_STRUCT_TYPE_TO_INDEX_CLASS`,再到 `indices/loading.py` 的 `load_indices_from_storage`,顺着 `index_struct.get_type()` → 查表 → `index_cls(index_struct=..., storage_context=...)` 推演,并回到 `base.py` 确认传入 `index_struct` 时会跳过 `build_index_from_nodes`。

> **下一章预告**:本章里 `VectorStoreIndex` 反复调用 `embed_model.get_text_embedding_batch` 把 node 变成向量,但 embed_model 究竟是什么、批处理与相似度计算如何抽象?第 7 章我们深入 Embeddings——剖析 `BaseEmbedding` 接口、批量嵌入机制与 `similarity` 度量。
