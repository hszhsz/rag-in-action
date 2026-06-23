# 第 8 章:存储与向量库抽象

上一章我们让每个 node 都拿到了 embedding。但 embedding 只是内存里的一个浮点数组,node 的全文、metadata、关系、还有第 6 章建立起来的索引结构(`IndexStruct`),最终都要落到某个地方,才能在下次进程启动时被取回。问题随之而来:node 的全文该存哪、向量该存哪、索引元数据又该存哪?如果每种索引、每种后端各搞一套存储逻辑,代码很快就会失控。

LlamaIndex 的答案是一个统一的容器 `StorageContext`,它把四类存储——`docstore`(文档/节点存储)、`index_store`(索引结构存储)、`vector_stores`(向量库,注意是复数)、`graph_store`(图存储)——聚合在一起,并约定一组抽象接口。本章只聚焦前三类与向量库抽象:我们将看到 `docstore` 和 `index_store` 如何共享一个底层的 `BaseKVStore` 三层架构,向量库如何用 `BasePydanticVectorStore` 把 Chroma、Pinecone、Qdrant 这些差异巨大的后端收敛成 `add`/`delete`/`query` 三个动作,以及 `SimpleVectorStore` 这个纯内存默认实现是怎么用暴力相似度撑起整套契约的。

## StorageContext:四类存储的聚合容器

`StorageContext` 定义在 `storage/storage_context.py`,是一个 `@dataclass`。它的字段直白地列出了所有成员:

```python
@dataclass
class StorageContext:
    docstore: BaseDocumentStore
    index_store: BaseIndexStore
    vector_stores: Dict[str, SerializeAsAny[BasePydanticVectorStore]]
    graph_store: GraphStore
    property_graph_store: Optional[PropertyGraphStore] = None
```

值得注意的是 `vector_stores` 是一个 `Dict[str, ...]`——一个 `StorageContext` 可以挂多个向量库,通过命名空间区分。文本向量库挂在 `DEFAULT_VECTOR_STORE`(值为 `"default"`,见 `vector_stores/simple.py`)下,图片向量库则挂在 `IMAGE_VECTOR_STORE_NAMESPACE`(值为 `"image"`)下,这是为多模态预留的。`vector_store` 这个单数属性被保留为向后兼容,它直接返回 `self.vector_stores[DEFAULT_VECTOR_STORE]`。

构造入口是 `from_defaults` 类方法。当不传 `persist_dir` 时,它把每一类存储都退化为对应的 `Simple*` 内存实现:

```python
if persist_dir is None:
    docstore = docstore or SimpleDocumentStore()
    index_store = index_store or SimpleIndexStore()
    graph_store = graph_store or SimpleGraphStore()
    image_store = image_store or SimpleVectorStore()
    if vector_store:
        vector_stores = {DEFAULT_VECTOR_STORE: vector_store}
    else:
        vector_stores = vector_stores or {DEFAULT_VECTOR_STORE: SimpleVectorStore()}
    if image_store:
        vector_stores[IMAGE_VECTOR_STORE_NAMESPACE] = image_store
```

这意味着「开箱即用、零外部依赖」是默认姿态——你不配置任何数据库,索引也能跑起来,全部活在内存里。注意这里 `or` 短路的写法:只有当某个参数没传(为 `None`)时才创建默认实现,所以你完全可以只替换其中一类(比如只把向量库换成 Qdrant),其余照旧用 `Simple*`。`image_store` 默认也会被构造出来并以 `"image"` 命名追加进 `vector_stores`,这是为多模态预留的第二个命名空间。

当传入 `persist_dir` 时则走另一条分支:每个 `Simple*` 实现改调各自的 `from_persist_dir` 从磁盘加载;`property_graph_store` 的加载被包在 `try/except FileNotFoundError` 里,找不到就置 `None`,体现了对「旧版本持久化目录里没有这一项」的向后兼容考量。

## KVStore → DocStore / IndexStore 的三层架构

`docstore` 和 `index_store` 看起来是两类不同的存储,但它们底下共享同一套抽象。这就是 LlamaIndex 存储设计里最值得玩味的分层:**最底层是 `BaseKVStore`,中间层是 `KVDocumentStore`/`KVIndexStore`,最上层是面向用户的 `SimpleDocumentStore`/`SimpleIndexStore`**。

`BaseKVStore` 定义在 `storage/kvstore/types.py`,接口非常精简,就是一组带 `collection` 参数的键值操作:`put`/`get`/`get_all`/`delete`,以及批量版 `put_all`/`aput_all`。`collection` 是命名空间参数,默认值 `DEFAULT_COLLECTION = "data"`,批大小默认 `DEFAULT_BATCH_SIZE = 1`。基类的 `put_all` 默认实现里有一个细节:如果 `batch_size != 1` 就直接 `raise NotImplementedError("Batching not supported by this key-value store.")`——也就是说,真正的批量写入需要具体后端(如数据库)自己重写,内存实现只支持逐条。

内存实现 `SimpleKVStore`(`storage/kvstore/simple_kvstore.py`)继承自 `MutableMappingKVStore[dict]`,真正的数据结构是 `self._collections_mappings: Dict[str, MutableMappingT]`——一个「collection 名 → 该 collection 的字典」的嵌套字典。`_get_collection_mapping` 在 collection 不存在时用工厂函数惰性创建它。`persist` 直接把整个 `_collections_mappings` 用 `json.dumps` 写文件,`from_persist_path` 再读回来。

为什么要这样分层?因为有了统一的 KV 抽象,`docstore` 和 `index_store` 就只是「同一个 KV 后端上的不同 collection」。换一个后端(比如把 `SimpleKVStore` 换成 Redis、MongoDB 的 KV 实现),`docstore` 和 `index_store` 一起切换,逻辑零改动。这是典型的「面向接口而非实现」的依赖倒置。

`KVIndexStore`(`storage/index_store/keyval_index_store.py`)就是中间层的极简范例。它的命名空间默认 `DEFAULT_NAMESPACE = "index_store"`、collection 后缀 `DEFAULT_COLLECTION_SUFFIX = "/data"`,拼出来的真实 collection 名是 `"index_store/data"`。`add_index_struct` 把 `IndexStruct` 序列化为 json 后,以 `index_struct.index_id` 为 key 存进 KV;`get_index_struct` 在不传 `struct_id` 时有个有趣的断言 `assert len(structs) == 1`——它假设单一索引场景下只有一个结构。这正好呼应第 6 章:`index_store` 存的就是索引结构的元数据。

## DocStore 的三个 collection 与 ref_doc 反向索引

`KVDocumentStore`(`storage/docstore/keyval_docstore.py`)比 `KVIndexStore` 复杂得多,因为它要同时维护三个 collection。默认命名空间 `DEFAULT_NAMESPACE = "docstore"`,三个后缀分别是:

- `DEFAULT_COLLECTION_DATA_SUFFIX = "/data"`——存 node 全文与完整内容(`_node_collection`);
- `DEFAULT_REF_DOC_COLLECTION_SUFFIX = "/ref_doc_info"`——存「源文档 → 它产出的 node id 列表」的反向索引(`_ref_doc_collection`);
- `DEFAULT_METADATA_COLLECTION_SUFFIX = "/metadata"`——存每个 node 的 `doc_hash` 与 `ref_doc_id`(`_metadata_collection`)。

`add_documents` 的核心是 `_prepare_kv_pairs`:遍历每个 node,对每个 node 调用 `_get_kv_pairs_for_insert`,一次性产出三组 kv 对,最后分别 `put_all` 到三个 collection:

```python
self._kvstore.put_all(node_kv_pairs, collection=self._node_collection, batch_size=batch_size)
self._kvstore.put_all(metadata_kv_pairs, collection=self._metadata_collection, batch_size=batch_size)
self._kvstore.put_all(ref_doc_kv_pairs, collection=self._ref_doc_collection, batch_size=batch_size)
```

注意 node 内容只在 `store_text=True` 时才写入 `_node_collection`(`_get_kv_pairs_for_insert` 里 `if store_text:` 才生成 `node_kv_pair`);metadata collection 里固定记录 `{"doc_hash": node.hash}`,这就是配合去重用的哈希存储——第 5 章摄入管线正是靠比对 `doc_hash` 来判断文档是否变化、决定要不要重新处理。`add_documents` 还有个 `allow_update` 开关:为 `False` 时若 `document_exists` 命中就直接 `raise ValueError`,防止误覆盖。

而 `ref_doc_info` 这个反向索引是整个增量更新的底座。当 node 有 `source_node`(即它从某个源文档切分而来)时,代码会把 `node.node_id` 追加进对应 `RefDocInfo.node_ids`,并在 metadata 里记下 `ref_doc_id`。`RefDocInfo`(定义在 `storage/docstore/types.py`)就是一个简单的 `node_ids: List` 加 `metadata: Dict`。多个 node 指向同一 `ref_doc_id` 时,`_merge_ref_doc_kv_pairs` 会把它们合并成一条,确保每个源文档只有一条整合记录。

有了这层反向索引,`delete_ref_doc` 就能实现「删一个源文档 = 删它产出的所有 node」:

```python
def delete_ref_doc(self, ref_doc_id: str, raise_error: bool = True) -> None:
    ref_doc_info = self.get_ref_doc_info(ref_doc_id)
    ...
    original_node_ids = ref_doc_info.node_ids.copy()
    for doc_id in original_node_ids:
        self.delete_document(doc_id, raise_error=False)
    # 兜底:确保 ref_doc 三个 collection 的记录都被清掉
    self._kvstore.delete(ref_doc_id, collection=self._ref_doc_collection)
    self._kvstore.delete(ref_doc_id, collection=self._metadata_collection)
    self._kvstore.delete(ref_doc_id, collection=self._node_collection)
```

反过来,`delete_document` 删单个 node 时会先调 `_remove_from_ref_doc_node`,把这个 node_id 从其所属 ref_doc 的 `node_ids` 里摘掉;如果摘完后该 ref_doc 已经没有任何 node 了,就连 ref_doc 本身一起清掉。这套「正向存内容、反向存归属」的双向结构,正是第 5 章 upsert/delete 增量更新能精确定位「哪些 node 属于哪个文档」的关键。

## 向量库契约:BasePydanticVectorStore

向量库的抽象定义在 `vector_stores/types.py`。这里其实有两套并存的契约:一个是 `@runtime_checkable` 的 `VectorStore` Protocol,另一个是真正被继承使用的抽象基类 `BasePydanticVectorStore`。后者继承自 `BaseComponent`(第 1 章的可序列化基类)和 `ABC`,这意味着向量库本身也是个可序列化组件。它的三个抽象方法构成了向量库的最小契约:

- `add(nodes, **kwargs) -> List[str]`:把带 embedding 的 node 写入,返回 id 列表;
- `delete(ref_doc_id, **delete_kwargs)`:按源文档 id 删除——注意删除的粒度是 `ref_doc_id`,这又一次呼应了 docstore 的反向索引设计;
- `query(query, **kwargs) -> VectorStoreQueryResult`:执行检索。

类上还有两个类级属性:`stores_text`(向量库是否同时存储 node 全文)和 `is_embedding_query: bool = True`。`stores_text` 是个关键开关,后面讲 node↔metadata 互转时会回到它。基类还提供了 `get_nodes`/`delete_nodes`/`clear` 等带默认 `NotImplementedError` 的扩展点,以及一律包装同步方法的 `async_add`/`adelete`/`aquery` 异步版,让没实现异步的后端也能用。

查询的输入输出都是结构化对象。`VectorStoreQuery` 是个 `@dataclass`,核心字段包括 `query_embedding`、`similarity_top_k: int = 1`(默认只取 1 条)、`filters: Optional[MetadataFilters]`、`mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT`,还有 `alpha`(混合检索权重)、`mmr_threshold`、`node_ids`/`doc_ids` 等。`VectorStoreQueryResult` 则只装三样东西:`nodes`、`similarities`、`ids`。

`VectorStoreQueryMode` 是个枚举,列出了所有检索模式:`DEFAULT`、`SPARSE`、`HYBRID`、`TEXT_SEARCH`、`SEMANTIC_HYBRID`,以及三种 learner 模式 `SVM`/`LOGISTIC_REGRESSION`/`LINEAR_REGRESSION` 和最大边际相关性 `MMR`。

## MetadataFilters:统一的元数据过滤

不同向量库的过滤语法千差万别,LlamaIndex 用 `MetadataFilter`/`MetadataFilters` 把它们统一成数据结构。`MetadataFilter` 是个 Pydantic 模型,有 `key`、`value`、`operator: FilterOperator = FilterOperator.EQ` 三个字段。`value` 刻意用了 Strict 类型(`StrictInt`/`StrictFloat`/`StrictStr` 及其列表),避免 int/float/str 之间的隐式转换造成过滤歧义。

`FilterOperator` 枚举把操作符列得很全:`EQ`(`==`,默认)、`NE`、`GT`/`LT`/`GTE`/`LTE`、`IN`/`NIN`、`ANY`/`ALL`、`CONTAINS`、`TEXT_MATCH`/`TEXT_MATCH_INSENSITIVE`、`IS_EMPTY`。`MetadataFilters` 则把多个 filter 用 `condition: FilterCondition` 组合,可选 `AND`/`OR`/`NOT`,且 `filters` 列表里可以嵌套 `MetadataFilters` 实现复合条件。

这套过滤如何落地到内存实现?`vector_stores/utils.py` 里的 `build_metadata_filter_fn` 把 `MetadataFilters` 编译成一个 `Callable[[str], bool]` 的判定函数。它内部的 `_process_filter_match` 逐个操作符地实现匹配逻辑(`EQ` 比相等、`IN` 判成员、`CONTAINS` 判包含、`TEXT_MATCH` 判子串等),最外层再按 `AND`/`OR`/`NOT` 聚合各 filter 的结果。外部数据库实现则会把同样的 `MetadataFilters` 翻译成各自的查询语言,但对上层调用者来说接口完全一致。

## SimpleVectorStore:暴力 top-k 与 MMR

`SimpleVectorStore`(`vector_stores/simple.py`)是默认的纯内存实现。它的数据全装在 `SimpleVectorStoreData` 里:`embedding_dict`(node_id → 向量)、`text_id_to_ref_doc_id`(node_id → 源文档 id,供按 ref_doc 删除用)、`metadata_dict`(node_id → metadata)。注意它的类属性 `stores_text: bool = False`——`SimpleVectorStore` **不存 node 全文**,只存向量和 metadata,node 全文交给 docstore。

`add` 把每个 node 的 `get_embedding()` 写进 `embedding_dict`,把 `ref_doc_id` 写进 `text_id_to_ref_doc_id`,并用 `node_to_metadata_dict(node, remove_text=True, flat_metadata=False)` 抽出 metadata(还会 `pop("_node_content", None)` 把整节点内容剔除,因为 Simple 实现靠 docstore 还原 node)。

`query` 是暴力相似度检索的典范。它先用 `build_metadata_filter_fn` 和 `node_ids` 做预过滤,把符合条件的 (node_id, embedding) 收集起来,再按 `query.mode` 分派:learner 模式调 `get_top_k_embeddings_learner`,`MMR` 模式调 `get_top_k_mmr_embeddings`,`DEFAULT` 模式调 `get_top_k_embeddings`——后者就是对所有候选向量逐个算相似度再排序取前 k 的暴力做法。这在小数据集上完全够用,但也解释了为什么生产环境要换成专门的向量库:暴力计算是 O(N) 的。`query` 返回的 `VectorStoreQueryResult` 只填 `similarities` 和 `ids`,**不填 `nodes`**——因为它压根没存 node,node 要靠调用方拿着 ids 回 docstore 取。

## node ↔ metadata dict 互转:外部库只存 metadata

这是理解 LlamaIndex 向量库设计的最后一块拼图。外部向量库(Chroma、Pinecone 等)通常只接受「向量 + 一个扁平的 metadata 字典」,并不认识 LlamaIndex 的 `TextNode` 对象。那它们怎么在检索后还原出完整的 node?答案在 `vector_stores/utils.py` 的两个函数。

`node_to_metadata_dict` 把整个 node 序列化进 metadata:它先 `node.model_dump(mode="json")`,然后把整个 node 字典 `json.dumps` 成字符串塞进 `metadata["_node_content"]`,并记下 `metadata["_node_type"] = node.class_name()`。它还冗余地把 `ref_doc_id` 写进三个键——`metadata["document_id"]`(给 Chroma)、`metadata["doc_id"]`(给 Pinecone/Qdrant/Redis)、`metadata["ref_doc_id"]`(给 Weaviate),用注释明确标注了每个键服务的后端,这样不同后端都能用自己习惯的字段名做 metadata 过滤。

反向的 `metadata_dict_to_node` 则从 metadata 里读出 `_node_content` 和 `_node_type`,按类型分派给 `Node`/`IndexNode`/`ImageNode`/`TextNode` 的 `from_json` 反序列化回 node 对象。这套机制的妙处在于:**外部向量库只需当一个「向量 + 字典」的存储,完整的 node 结构被打包成 json 字符串藏在 metadata 里**,检索回来后靠 utils 原样还原。这就是为什么 LlamaIndex 能用同一套 node 模型适配几十种异构向量库。

## persist / from_persist_dir:整体持久化

最后回到 `StorageContext` 的整体落盘。`persist` 方法(默认目录 `DEFAULT_PERSIST_DIR = "./storage"`)把每一类存储分别写到约定文件名:docstore 写 `docstore.json`、index_store 写 `index_store.json`、graph_store 写 `graph_store.json`。向量库因为可能有多个,会按命名空间拼文件名——`f"{vector_store_name}{NAMESPACE_SEP}{vector_store_fname}"`,其中 `NAMESPACE_SEP = "__"`,所以默认向量库落盘为 `default__vector_store.json`。

加载侧,`from_defaults` 在传入 `persist_dir` 时,会对各 `Simple*` 实现调用各自的 `from_persist_dir`,向量库则用 `SimpleVectorStore.from_namespaced_persist_dir` 扫描目录下所有以 `vector_store.json` 结尾的文件,按 `__` 前缀还原出多个命名空间的向量库。

`StorageContext` 还提供了 `to_dict`/`from_dict`,但 `to_dict` 有个硬约束:只有当所有存储都是 `Simple*` 实现时才允许,否则 `raise ValueError("to_dict only available when using simple doc/index/vector stores")`。这很合理——外部数据库的数据在数据库里,没法整体 dump 成一个 dict。这种「整体序列化/反序列化」的能力,正是第 1 章 `BaseComponent` 可序列化契约在存储层的体现:每一层组件都能 `to_dict`/`from_dict`,组合起来就能让整个存储上下文落盘再原样复活。

## 本章小结

- `StorageContext` 是聚合 `docstore`/`index_store`/`vector_stores`/`graph_store` 的 `@dataclass` 容器,`vector_stores` 是 `Dict[str, ...]` 支持多命名空间(文本 `"default"`、图片 `"image"`)。
- `from_defaults` 不传 `persist_dir` 时全部退化为 `Simple*` 内存实现,实现「零外部依赖开箱即用」。
- 存储采用三层架构:`BaseKVStore`(`put`/`get`/`delete` + `collection` 命名空间)→ `KVDocumentStore`/`KVIndexStore` → `SimpleDocumentStore`/`SimpleIndexStore`,换后端时 docstore 与 index_store 一起切换。
- `SimpleKVStore` 用嵌套字典 `_collections_mappings` 存数据,`DEFAULT_COLLECTION = "data"`、`DEFAULT_BATCH_SIZE = 1`,基类批量写超过 1 会抛 `NotImplementedError`。
- `KVDocumentStore` 维护三个 collection(`/data`、`/ref_doc_info`、`/metadata`),`RefDocInfo` 反向索引(源文档 → node id 列表)是第 5 章增量 upsert/delete 的底座,`delete_ref_doc` 据此级联删除。
- metadata collection 里的 `doc_hash` 是去重的依据;`KVIndexStore` 以 `index_id` 为 key 存 `IndexStruct`,呼应第 6 章索引结构元数据。
- 向量库契约 `BasePydanticVectorStore`(继承 `BaseComponent`)的最小接口是 `add`/`delete`/`query`,删除粒度是 `ref_doc_id`;查询用 `VectorStoreQuery`(`similarity_top_k` 默认 1)输入、`VectorStoreQueryResult` 输出。
- `MetadataFilters`/`MetadataFilter`/`FilterOperator` 统一了元数据过滤,`build_metadata_filter_fn` 把它编译成内存判定函数,外部库则翻译成各自查询语言。
- `SimpleVectorStore` 的 `stores_text=False`,只存向量与 metadata,`query` 用 `get_top_k_embeddings` 暴力计算 top-k,还支持 learner 与 `MMR` 模式,结果只返回 `ids`/`similarities`。
- `node_to_metadata_dict`/`metadata_dict_to_node` 把整个 node 打包进 `_node_content` json 字符串,使外部库只需存「向量 + metadata」就能还原完整 node,并冗余写 `document_id`/`doc_id`/`ref_doc_id` 适配不同后端。
- `StorageContext.persist`/`from_defaults(persist_dir=...)` 实现整体落盘与加载,`to_dict` 仅在全 `Simple*` 时可用,呼应第 1 章 `BaseComponent` 可序列化契约。

## 动手实验

1. **实验一:观察 docstore 的三个 collection** — 在 `/tmp/llama_explore` 下用 `SimpleDocumentStore` 调 `add_documents` 写入几个带 `source_node` 的 node,然后读 `docstore._kvstore._collections_mappings` 的 keys,确认出现 `docstore/data`、`docstore/ref_doc_info`、`docstore/metadata` 三个 collection,并核对 `/metadata` 里每条都含 `doc_hash`。
2. **实验二:验证 ref_doc 级联删除** — 构造两个共享同一 `ref_doc_id` 的 node 写入 docstore,调 `get_ref_doc_info` 看 `node_ids` 是否含两个 id;再调 `delete_ref_doc(ref_doc_id)`,确认两个 node 与 ref_doc 记录都被清空,对照 `keyval_docstore.py` 的 `delete_ref_doc` 实现。
3. **实验三:跑通 SimpleVectorStore 暴力检索** — 手动构造几个带 `embedding` 的 `TextNode`,`add` 进 `SimpleVectorStore`,用不同 `query_embedding` 和 `similarity_top_k` 构造 `VectorStoreQuery` 调 `query`,观察返回的 `VectorStoreQueryResult` 只有 `ids`/`similarities` 而 `nodes` 为 None,印证 `stores_text=False`。
4. **实验四:验证 node↔metadata 互转** — 取一个 `TextNode`,调 `node_to_metadata_dict(node, remove_text=True)` 打印结果,确认存在 `_node_content`/`_node_type` 及 `document_id`/`doc_id`/`ref_doc_id` 三个冗余键;再把该 metadata 传给 `metadata_dict_to_node`,比对还原出的 node 与原 node 是否一致。

> **下一章预告**:存储与向量库解决了「node 与向量存到哪、怎么取回」,但「取回哪些」需要检索策略来决定。第 9 章我们将进入检索器 Retrievers,剖析 `BaseRetriever` 的统一接口、`VectorIndexRetriever` 如何把 `VectorStoreQuery` 派给向量库、以及 `AutoMergingRetriever`、`RecursiveRetriever` 与融合检索如何在召回阶段做出更聪明的取舍。
