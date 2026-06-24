# 第 2 章 存储抽象与四类基类

任何一个 RAG 引擎,无论检索策略多花哨,落到工程层面都要回答一个朴素问题:数据存在哪、怎么读、怎么写、怎么清。LightRAG 的回答集中在一个文件里——`base.py`。它没有给每种后端(本地 JSON、Postgres、Neo4j、Milvus、Redis……)各写一套接口,而是把整个 RAG 系统所需的全部持久化需求,收敛成四个抽象基类:KV、Vector、Graph、DocStatus。任何后端只要实现这四类中相应的 ABC,就能被引擎无差别地驱动。

本章我们逐行解构 `base.py`:先看四类基类共享的根基 `StorageNameSpace`,再看每个基类的最小数据契约,然后剖析两个最值得玩味的设计决策——`index_done_callback` 的"延迟落盘"契约,以及 `BaseGraphStorage` 那一批"默认能跑、后端可优化"的批量方法。最后看检索入口的 `QueryParam` 和文档状态机 `DocStatus`,它们是引擎与存储之间流动的两种关键数据形态。

## 一切的根:`StorageNameSpace`

四个存储基类都继承自同一个 `@dataclass` ABC——`StorageNameSpace`。它只声明三个字段,却定义了所有存储的身份与生命周期:

- `namespace: str`——这个存储实例代表哪类数据。取值来自 `namespace.py` 的 `NameSpace` 类常量,例如 `full_docs`、`text_chunks`、`llm_response_cache`(KV 类),`entities`、`relationships`、`chunks`(Vector 类),`chunk_entity_relation`(Graph 类),`doc_status`(DocStatus 类)。`namespace.py` 顶部有一句强调:`# All namespace should not be changed`——这些字符串是物理表名/集合名/文件名的来源,改了就读不到旧数据。
- `workspace: str`——多租户隔离维度,同一后端里不同 workspace 的数据互不可见。
- `global_config: dict[str, Any]`——把引擎的全局配置透传给每个存储实例,后端据此决定批大小、阈值、文件路径等。

生命周期契约由四个方法构成:

- `initialize()` / `finalize()`——默认是空实现(`pass`),给需要建连接池、打开文件句柄的后端 override。
- `index_done_callback()`——`@abstractmethod`,**强制每个后端实现**。这是"提交/落盘"的统一信号(详见下一节)。
- `drop_pending_index_ops()`——默认返回 `None`(no-op),只有缓冲写的后端需要 override。
- `drop()`——`@abstractmethod`,清空该存储的全部数据并重置到初始状态。文档字符串里明确规定了返回值格式:成功返回 `{"status": "success", "message": "data dropped"}`,失败返回 `{"status": "error", "message": "<error details>"}`,不支持返回 `{"status": "error", "message": "unsupported"}`。

把 `namespace`/`workspace`/`global_config` 提到根基类,意味着引擎在创建任何存储时都用同一套构造参数;而把 `index_done_callback`/`drop` 设为抽象方法,则保证了"提交"和"清空"这两个最容易被遗漏的操作,在每个后端都必须被认真对待。`abstractmethod` 装饰的方法若未实现,实例化时就会直接报错——这是 Python `abc` 模块提供的契约保证 [[abc — Abstract Base Classes]](https://docs.python.org/3/library/abc.html)。

值得注意的一个细节是:`StorageNameSpace` 同时用了 `@dataclass` 和继承 `ABC`。这是一种刻意的组合——`@dataclass` 自动生成带 `namespace`/`workspace`/`global_config` 三个位置参数的 `__init__`,免去每个后端手写构造器的样板;`ABC` 则把 `index_done_callback`/`drop` 锁成必须实现的契约。子类如 `BaseVectorStorage` 继续叠加自己的字段(`embedding_func` 等)时,只需再标一次 `@dataclass`,dataclass 的字段继承机制会把父类字段排在前、子类字段排在后,拼成一个完整的构造签名。这就是为什么四类基类的定义都长成 `@dataclass` + 继承 `StorageNameSpace, ABC` 的统一形状。

## "延迟落盘"契约:`index_done_callback`

如果只看接口名,`index_done_callback` 像是一个无关紧要的回调。但 `base.py` 的注释把它的设计意图写得非常清楚——它是 in-memory 后端的"提交点",也是多 worker 部署下读写可见性的分水岭。

四类基类的写方法(`upsert`、`delete`、`upsert_node`、`delete_entity`……)的文档字符串里几乎都重复着同一段 "Importance notes for in-memory storage":

> 1. Changes will be persisted to disk during the next index_done_callback
> 2. Only one process should updating the storage at a time before index_done_callback, KG-storage-log should be used to avoid data corruption

也就是说,对于把数据缓冲在进程内存里的后端,`upsert` 只是改内存,真正落盘要等到 `index_done_callback()` 被调用。这是典型的"批量提交"优化:索引一篇文档会产生大量 upsert,逐条写盘代价高昂,缓冲到一批结束再统一刷盘要快得多。

更微妙的是多 worker 可见性问题。`BaseVectorStorage.upsert` 和 `BaseKVStorage.upsert` 的注释专门加了 "Multi-worker note":像 `OpenSearchVectorDBStorage`、`OpenSearchKVStorage` 这类把写缓冲在进程内存里的后端,缓冲是 **process-local** 的。在 `lightrag-gunicorn` 这种多 worker 部署里,worker A 写入后,worker B 在 A 调用 `index_done_callback()` 之前**看不到**这些写入。注释给出的纪律是:任何依赖跨 worker "写后即读"可见性的调用方,必须显式 `await index_done_callback()` 之后再去别的 worker 读。

与之配套的是 `drop_pending_index_ops()`。它的注释解释了一个边界场景:当一批文档因内部错误中止时,缓冲区里还没刷盘的记录全都属于那些"即将被标记为 FAILED、下次重跑会完整重做"的文档,所以直接丢弃缓冲是安全的,还能避免被污染/陈旧的记录被其它在途文档误刷盘或带到下一批。立即写盘的后端保持默认的 no-op 即可。

这套契约背后是一个清晰的取舍:**性能优先用缓冲,但把缓冲的"可见性边界"显式地交给 `index_done_callback` 这一个点来管理**,而不是让每个后端各自发明同步语义。注意 `BaseGraphStorage` 的写方法注释里把第 2 条措辞改成了 "Only one process should updating the storage at a time before index_done_callback, KG-storage-log should be used to avoid data corruption"——图存储不仅要管"何时刷盘",还要管"同一时刻只能一个进程写",并点名 KG-storage-log 作为防数据损坏的手段。这正是第 8 章共享存储与并发控制要展开的伏笔:`base.py` 在接口注释里就已经埋下了对并发写一致性的约束。

## 四类基类的最小数据契约

在展开每个基类之前,先把"基类"和"命名空间"的关系理清。基类回答的是"这类数据需要哪些操作",命名空间回答的是"这个实例代表哪份数据"。同一个基类会被实例化成多个不同 `namespace` 的对象——例如 `BaseKVStorage` 既被用来存 `full_docs`(原文),也被用来存 `text_chunks`(切块)和 `llm_response_cache`(响应缓存);`BaseVectorStorage` 则分别承载 `entities`、`relationships`、`chunks` 三套向量。`namespace.py` 里的 `NameSpace` 类就是这张映射表,`is_namespace()` 辅助函数用 `endswith` 做后缀匹配,让带 workspace 前缀的实际 namespace 仍能识别出它属于哪一类。理解这一点,才能明白为什么仅四个基类就足以覆盖一个完整 RAG 引擎的存储面。

### `BaseVectorStorage`:向量检索

在 `StorageNameSpace` 之上,它增加了 `embedding_func`(必填)、`cosine_better_than_threshold`(默认 `0.2`)和 `meta_fields`(默认空集)。基类还提供两个工具方法:`_validate_embedding_func()` 在 `embedding_func` 为 `None` 时直接抛 `ValueError`,要求所有向量后端在 `__post_init__` 开头调用它;`_generate_collection_suffix()` 用模型名+维度生成表/集合后缀(如 `text_embedding_3_large_3072d`),让不同 embedding 模型的数据物理隔离。

抽象方法刻画了向量库该会的事:`query(query, top_k, query_embedding=None)`(支持传入预计算 embedding 跳过重复计算)、`upsert`、`get_by_id`/`get_by_ids`、`get_vectors_by_ids`(只返回向量本体以提高效率)、`delete`,以及两个与知识图谱联动的删除接口 `delete_entity`、`delete_entity_relation`。

### `BaseKVStorage`:键值存储

最朴素的一类,带 `embedding_func` 字段。核心方法:`get_by_id`/`get_by_ids`、`filter_keys(keys)`(返回"不存在"的键,用于增量写入前去重)、`upsert`、`delete`、`is_empty()`。它服务于 `full_docs`、`text_chunks`、`llm_response_cache` 等多种命名空间——LLM 响应缓存、原文、切块全靠它。这里 `filter_keys` 是一个容易被低估的方法:文档处理管线在写入前先用它筛掉已存在的键,从而实现幂等的增量索引——重复入库同一篇文档不会重复计算。换句话说,KV 基类虽然字段最少,却同时兼任了"通用键值表"和"去重过滤器"两个角色,这也是为什么 `DocStatusStorage` 选择继承它而非另起炉灶。

### `BaseGraphStorage`:知识图谱

最庞大的基类,类文档字符串开门见山:`All operations related to edges in graph should be undirected.`(图中所有边操作都按无向处理)。它定义了节点/边的存在性判断(`has_node`、`has_edge`)、度数计算(`node_degree`、`edge_degree`)、读取(`get_node`、`get_edge`、`get_node_edges`)、写入(`upsert_node`、`upsert_edge`)、删除(`delete_node`、`remove_nodes`、`remove_edges`),以及为可视化与检索服务的标签接口:`get_all_labels`、`get_popular_labels(limit=300)`(按度数返回最热门实体)、`search_labels(query, limit=50)`(模糊搜索)、`get_knowledge_graph(node_label, max_depth=3, max_nodes=1000)`(返回 `KnowledgeGraph` 子图,带 `is_truncated` 截断标记)。这里 `get_knowledge_graph` 的返回类型 `KnowledgeGraph` 来自 `types.py`,是一个 pydantic `BaseModel`,内含 `nodes`、`edges`、`is_truncated` 三个字段;`get_all_labels` 的注释还特意提醒:大图别用它,改用 `get_popular_labels` 或 `search_labels`。

### `DocStatusStorage`:文档处理状态

它继承自 `BaseKVStorage`——文档状态本质也是键值数据,但额外约束了一组面向"流水线监控与去重"的查询方法:`get_status_counts`/`get_all_status_counts`、`get_docs_by_status`/`get_docs_by_statuses`、`get_docs_by_track_id`、`get_docs_paginated`(带分页、排序、状态过滤)、以及三个去重查询 `get_doc_by_file_path`、`get_doc_by_file_basename`(按文件名去重)、`get_doc_by_content_hash`(按内容哈希去重)。它还提供一个静态工具 `resolve_status_filter_values`,把单状态过滤 `status_filter` 和多状态过滤 `status_filters` 归一成一个可比较的字符串集合,且 `status_filters` 优先、空列表当作不过滤。

## "默认能跑,后端可优化":批量方法的回退实现

`BaseGraphStorage` 最值得学习的工程手法,是它处理批量操作的方式。`get_nodes_batch`、`node_degrees_batch`、`edge_degrees_batch`、`get_edges_batch`、`get_nodes_edges_batch`、`upsert_nodes_batch`、`has_nodes_batch`、`upsert_edges_batch` 这一整组方法,**都不是抽象方法**,而是带有默认实现的普通方法。

它们的默认实现高度一致——就是把对应的单条方法在循环里串行调用一遍。例如 `get_nodes_batch`:

```python
async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
    result = {}
    for node_id in node_ids:
        node = await self.get_node(node_id)
        if node is not None:
            result[node_id] = node
    return result
```

每个方法的文档字符串都附了同一句话:"Default implementation fetches ... one by one. Override this method for better performance in storage backends that support batch operations."(默认逐条取,支持批量的后端请 override 以获得更好性能)。注释里多次提到 `UNWIND`——这是 Cypher 的批量展开关键字,暗示 Neo4j 这类图数据库后端会用一条 `UNWIND` 查询替换掉这里的整个循环。

这套设计的价值在于降低后端接入门槛:一个新图后端只要实现少数几个 `@abstractmethod`(`has_node`、`get_node`、`upsert_node`……),所有批量方法立刻可用、且语义正确——只是性能是串行的。等到这个后端确实成为瓶颈,再针对性地 override 批量方法换上原生批操作。**正确性默认保证,性能按需优化**,这正是抽象基类"提供合理默认 + 开放扩展点"的范本。同样的回退思路也体现在 `upsert_nodes_batch`/`upsert_edges_batch`/`has_nodes_batch` 上,它们分别回退到 `upsert_node`/`upsert_edge`/`has_node`。

## 检索入口的数据契约:`QueryParam`

如果说四类基类是"写"和"读底层数据"的契约,那 `QueryParam` 就是"查询请求"的契约。它是一个 `@dataclass`,承载一次查询的全部可调参数:

- `mode`——六种检索模式:`"local"`(聚焦上下文相关信息)、`"global"`(利用全局知识)、`"hybrid"`(本地+全局)、`"naive"`(基础向量搜索)、`"mix"`(知识图谱+向量融合)、`"bypass"`,**默认 `"mix"`**。
- 三个 token 预算字段,构成"统一 token 控制系统":`max_entity_tokens`(实体上下文预算)、`max_relation_tokens`(关系上下文预算)、`max_total_tokens`(整个查询上下文的总预算,含实体+关系+切块+系统提示)。
- 检索数量:`top_k`(local 模式代表实体数、global 模式代表关系数)、`chunk_top_k`(向量检索初步取回并在重排后保留的切块数)。
- 行为开关:`only_need_context`、`only_need_prompt`、`stream`、`enable_rerank`、`include_references`,以及 `response_type`(响应格式,默认 `"Multiple Paragraphs"`)、`user_prompt`、`conversation_history` 等。

特别值得注意的是这些默认值的来源——它们不是写死的字面量,而是在类定义时从环境变量读取,缺省时回落到 `constants.py` 的 `DEFAULT_*` 常量:

```python
top_k: int = int(os.getenv("TOP_K", str(DEFAULT_TOP_K)))
max_entity_tokens: int = int(os.getenv("MAX_ENTITY_TOKENS", str(DEFAULT_MAX_ENTITY_TOKENS)))
enable_rerank: bool = os.getenv("RERANK_BY_DEFAULT", "true").lower() == "true"
```

对照 `constants.py`,这些默认值是:`DEFAULT_TOP_K = 40`、`DEFAULT_CHUNK_TOP_K = 20`、`DEFAULT_MAX_ENTITY_TOKENS = 6000`、`DEFAULT_MAX_RELATION_TOKENS = 8000`、`DEFAULT_MAX_TOTAL_TOKENS = 30000`。`base.py` 顶部还调用了 `load_dotenv(dotenv_path=".env", override=False)`,且注释说明 OS 环境变量优先于 `.env` 文件——这让每个 LightRAG 实例都能用各自目录下的 `.env` 来定制查询默认值,而无需改代码。需要留意 dataclass 的一个性质:这些 `os.getenv` 在类被**定义/导入时**求值一次,因此运行期再改环境变量不会影响已加载的默认值 [[dataclasses — Data Classes]](https://docs.python.org/3/library/dataclasses.html)。

## 文档状态机:`DocStatus` 与 `DocProcessingStatus`

文档从入库到可检索要经历多个阶段,`DocStatus` 用一个继承自 `str` 的 `Enum` 把这些阶段固定下来。类文档字符串直接画出了流水线顺序:

> Pipeline order: PENDING -> PARSING -> ANALYZING (optional) -> PROCESSING -> PROCESSED | FAILED. PREPROCESSED is deprecated, kept for backward compatibility.

各状态的语义在枚举值旁的注释里写得很具体:`PARSING` 是 Phase 1 内容抽取(parse_native/mineru/docling),`ANALYZING` 是 Phase 2 多模态分析(VLM),`PROCESSING` 是 Phase 3 实体/关系抽取,最终走向 `PROCESSED` 或 `FAILED`。`PREPROCESSED` 已废弃,仅为向后兼容保留——新流水线用 `ANALYZING` 取而代之。让 `DocStatus` 继承 `str`,好处是枚举成员可以直接当字符串用(序列化、与数据库里存的字符串比较),这正是 `DocStatusStorage.resolve_status_filter_values` 里能直接取 `status.value` 比较的前提。

`DocProcessingStatus` 是承载单篇文档状态的 `@dataclass`,字段包括 `content_summary`(前 100 字预览)、`content_length`、`file_path`(规范化的基名,注释强调它永远是去掉 hint 的纯 basename 或哨兵值 `"unknown_source"`)、`status`、`created_at`/`updated_at`、`track_id`、`chunks_count`/`chunks_list`、`error_msg`、`content_hash`(文档内容的 MD5,与 basename 一起做去重)等。

它的状态转换逻辑藏在 `__post_init__` 里,处理的正是那个废弃状态的兼容问题:

```python
def __post_init__(self):
    if self.multimodal_processed is not None:
        if (self.multimodal_processed is False
                and self.status == DocStatus.PROCESSED):
            self.status = DocStatus.PREPROCESSED
```

也就是说:如果一条记录标记了"多模态尚未处理完"(`multimodal_processed is False`)但状态却是 `PROCESSED`,构造时就自动把它降级回 `PREPROCESSED`。`multimodal_processed` 字段用 `field(default=None, repr=False)` 声明——默认不参与 `repr()`,只供内部调试。这把"状态一致性修正"放在数据对象的构造期完成,而非散落在业务代码里,是一种把不变量收口到数据类的做法。`__post_init__` 是 dataclass 在自动生成的 `__init__` 末尾自动调用的钩子 [[dataclasses — Data Classes]](https://docs.python.org/3/library/dataclasses.html)。

## 围绕契约的辅助数据结构

`base.py` 末尾还定义了几个把"操作结果"标准化的小数据类,体现统一契约的思想延伸到了返回值层面:

- `StoragesStatus(str, Enum)`——存储集合的生命周期状态:`NOT_CREATED` / `CREATED` / `INITIALIZED` / `FINALIZED`,与前面的 `initialize`/`finalize` 契约呼应。
- `DeletionResult`——删除操作的统一返回,`status` 限定为 `"success"`/`"not_found"`/`"not_allowed"`/`"fail"`,带 `doc_id`、`message`、`status_code`(默认 200)、`file_path`。
- `QueryResult` 与 `QueryContextResult`——为"引用列表(reference list)"支持而设的统一查询结果结构。`QueryResult` 同时容纳非流式 `content` 和流式 `response_iterator`(`is_streaming` 区分),并通过 `reference_list`、`metadata` 两个 `@property` 从 `raw_data` 里便捷取值;`QueryContextResult` 则只携带 LLM 上下文字符串 `context` 与 `raw_data`。

这些类型不是 ABC,但它们和四类基类一脉相承:**用最小、明确的数据契约,把引擎与外部世界的每一处交互都规范化**——查询进来是 `QueryParam`,结果出去是 `QueryResult`,删除回来是 `DeletionResult`,状态流转是 `DocStatus`。引擎主体代码因此可以只面向这些契约编程,而把"具体怎么存、怎么算"全部下放给可替换的后端实现。

## 本章小结

- LightRAG 把 RAG 的全部持久化需求收敛为四个 ABC:`BaseKVStorage`、`BaseVectorStorage`、`BaseGraphStorage`、`DocStatusStorage`,任何后端实现相应基类即可被引擎无差别驱动。
- 四类基类共享根基 `StorageNameSpace`,它声明 `namespace`/`workspace`/`global_config` 三字段,并用 `initialize`/`finalize`/`index_done_callback`/`drop_pending_index_ops`/`drop` 定义统一生命周期;其中 `index_done_callback` 与 `drop` 是强制实现的 `@abstractmethod`。
- `namespace` 取值来自 `namespace.py` 的 `NameSpace` 常量(`full_docs`、`text_chunks`、`entities`、`chunk_entity_relation`、`doc_status` 等),且明确标注"不可更改",因为它们映射到物理表/集合/文件名。
- `index_done_callback` 是"延迟落盘"契约:in-memory 后端的 `upsert`/`delete` 只改进程内缓冲,要等该回调才刷盘;在多 worker 部署下,缓冲是 process-local 的,跨 worker 写后即读必须先 `await index_done_callback()`。
- `drop_pending_index_ops` 默认 no-op,只在批处理中止、缓冲记录注定属于将被标 FAILED 重做的文档时,由缓冲型后端 override 来安全丢弃缓冲。
- `BaseGraphStorage` 的批量方法(`get_nodes_batch`、`upsert_nodes_batch` 等)不是抽象方法,而是串行回退的默认实现,后端可用 `UNWIND` 之类原生批操作 override——"默认能跑,后端可优化"。
- `QueryParam` 用一个 dataclass 承载查询请求:六种 `mode`(默认 `mix`)、三个 token 预算字段、`top_k`/`chunk_top_k` 等,默认值在导入时从环境变量读取并回落到 `constants.py` 的 `DEFAULT_*` 常量。
- `DocStatus` 是继承 `str` 的状态机,流水线顺序为 `PENDING→PARSING→ANALYZING(可选)→PROCESSING→PROCESSED|FAILED`,`PREPROCESSED` 已废弃仅作兼容。
- `DocProcessingStatus.__post_init__` 在构造期修正状态一致性:当 `multimodal_processed is False` 而 `status` 为 `PROCESSED` 时,自动降级为 `PREPROCESSED`。
- `StoragesStatus`、`DeletionResult`、`QueryResult`/`QueryContextResult` 把生命周期、删除结果、查询结果也标准化为最小数据契约,使引擎主体只面向契约编程。

## 动手实验

1. **实验一:画出存储类继承树** — 通读 `base.py`,用纸笔或绘图工具画出从 `StorageNameSpace` 出发的继承关系:`BaseVectorStorage`、`BaseKVStorage`、`BaseGraphStorage` 直接继承它,而 `DocStatusStorage` 继承自 `BaseKVStorage`。在每个类旁标注它新增了哪些字段、哪些方法是 `@abstractmethod`、哪些带默认实现,体会"哪些是必须实现的契约,哪些是开放的优化点"。
2. **实验二:验证批量方法的回退语义** — 在 `BaseGraphStorage` 中找到 `get_nodes_batch` 与它依赖的 `get_node`,写一个最小的子类:只实现所有 `@abstractmethod`(`get_node` 用一个内存 dict 返回数据,其余返回空),不实现任何 `*_batch` 方法。然后调用 `get_nodes_batch(["a","b","c"])`,确认它能正确逐条回退、跳过不存在的节点,从而印证"默认实现保证正确性"。
3. **实验三:观察 `QueryParam` 的环境变量默认值** — 在 Python 里先 `import os; os.environ["TOP_K"]="7"`,**再**导入 `QueryParam`,打印 `QueryParam().top_k`;接着尝试导入之后再改环境变量看是否生效。结合 dataclass 字段在类定义时即求值的特性,解释为什么必须在导入前设置环境变量才会改变默认 `top_k`。
4. **实验四:复现 `DocProcessingStatus` 的状态降级** — 构造一个 `DocProcessingStatus`,把 `status` 设为 `DocStatus.PROCESSED`、`multimodal_processed` 设为 `False`,打印构造后的 `status`,确认它被 `__post_init__` 自动改成了 `DocStatus.PREPROCESSED`;再把 `multimodal_processed` 设为 `None` 或 `True` 重试,对比差异,理解这条向后兼容规则的触发条件。

> **下一章预告**:数据有了存放契约,下一步就是把原始文档切成可索引的单元。第 3 章我们进入文本切分,解构 LightRAG 如何用 i18n 感知的分隔符级联、固定/递归/向量/段落四种切块策略,把一篇文档变成 `text_chunks` 里那一条条带 `tokens`、`chunk_order_index` 的 `TextChunkSchema`。
