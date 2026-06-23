# 第 5 章：知识库服务抽象层

到第 4 章为止，文档已经被加载、被切分成 `Document` 列表，等待落地到某个向量库。可现实是残酷的：用户可能用 FAISS 做本地离线检索，可能用 Milvus 撑起生产规模，可能用 Elasticsearch 复用既有搜索基建，也可能用 PGVector、Chroma 把向量塞进现成的数据库。这些向量库的 API 风格、连接方式、ID 体系、删除语义五花八门，如果让上层的「添加文档」「检索」「删库」逻辑直接面对它们，代码会迅速腐烂成一堆 `if vs_type == ...` 的分支。

Langchain-Chatchat 给出的答案是一层薄而严谨的抽象：`KBService` 抽象基类用模板方法模式把「公共流程」和「向量库特定动作」彻底分离，`KBServiceFactory` 按配置把字符串类型映射到具体实现，`SupportedVSType` 枚举钉死可选范围。本章逐行拆解这套「统一契约 + 多实现」的骨架，看它如何让七八种向量库在同一套接口下和平共处。源码集中在 `chatchat/server/knowledge_base/kb_service/` 目录下。

## 一张枚举表钉死可选向量库

抽象层的起点是 `base.py` 顶部的 `SupportedVSType` 类。它不是 Python 的 `enum.Enum`，而是一个朴素的常量容器：

```python
class SupportedVSType:
    FAISS = "faiss"
    MILVUS = "milvus"
    DEFAULT = "default"
    ZILLIZ = "zilliz"
    PG = "pg"
    RELYT = "relyt"
    ES = "es"
    CHROMADB = "chromadb"
```

每个属性的值都是小写字符串，这个设计很关键：数据库里持久化的 `vs_type`、配置文件里写的类型名、工厂方法接收的参数，全都是这些小写字符串。用类属性而非 `Enum`，是为了让 `getattr(SupportedVSType, vector_store_type.upper())` 这种「字符串转枚举」的反射写法能直接工作——这一点在工厂里会再次出现。一共八个取值，其中 `DEFAULT` 是个特殊角色：它既不是真实向量库，又在工厂里被映射到 Milvus，后文会专门解释这个看似矛盾的设计。

## KBService:模板方法模式的教科书实现

`KBService` 继承自 `abc.ABC`，是整个知识库子系统的中枢。它的精髓在于把每个对外操作拆成两层：一个**公共方法**负责所有向量库都一样的流程（参数校验、路径处理、数据库登记、缓存刷新），一个 `do_*` **抽象钩子**负责向量库特定的脏活。子类只需要实现钩子，公共流程一行都不用碰。

构造函数 `__init__` 确立了每个服务实例的身份：

```python
def __init__(self, knowledge_base_name, kb_info=None,
             embed_model=get_default_embedding()):
    self.kb_name = knowledge_base_name
    self.kb_info = kb_info or Settings.kb_settings.KB_INFO.get(
        knowledge_base_name, f"关于{knowledge_base_name}的知识库")
    self.embed_model = embed_model
    self.kb_path = get_kb_path(self.kb_name)
    self.doc_path = get_doc_path(self.kb_name)
    self.do_init()
```

注意最后一行 `self.do_init()`——构造函数的末尾立刻回调子类的 `do_init`。这意味着「建立向量库连接」这件每家都不同的事，被钉在了实例化的同一时刻完成。`embed_model` 默认值来自 `get_default_embedding()`，`kb_info`（知识库描述）若不传则回退到 `Settings.kb_settings.KB_INFO` 配置或一句模板化的「关于 XX 的知识库」。`__repr__` 返回 `"{kb_name} @ {embed_model}"`，把知识库名和嵌入模型绑定显示——这暗示了一个事实：同名知识库换了嵌入模型就是另一套向量空间。

七个 `@abstractmethod` 划定了子类必须填的空：`do_create_kb`、`vs_type`、`do_init`、`do_drop_kb`、`do_search`、`do_add_doc`、`do_delete_doc`、`do_clear_vs`。`vs_type` 本身也是抽象方法，强制每个子类自报家门返回自己的 `SupportedVSType` 值——这个返回值会在 `create_kb`、`update_info` 时被写进数据库，成为日后 `get_service_by_name` 重建服务的依据。

## add_doc:公共流程如何包裹 do_add_doc

`add_doc` 是模板方法模式最值得细看的一例。它的签名是 `add_doc(self, kb_file, docs=[], **kwargs)`，整个流程像一条流水线：

第一道关卡是嵌入模型校验：`if not self.check_embed_model()[0]: return False`，嵌入模型不可用直接短路返回。接着判断 `docs` 是否由调用方显式传入——若传入则 `custom_docs = True`，否则调用 `kb_file.file2text()` 现场切分文档并置 `custom_docs = False`。这个 `custom_docs` 标志会一路传到数据库，标记这条文件记录是「自定义切片」还是「框架切片」。

最精妙的是 source 路径的归一化处理。每个 `doc.metadata` 都会先 `setdefault("source", kb_file.filename)`，再把绝对路径转成相对于 `self.doc_path` 的相对路径：

```python
if os.path.isabs(source):
    rel_path = Path(source).relative_to(self.doc_path)
    doc.metadata["source"] = str(rel_path.as_posix().strip("/"))
```

为什么非要相对路径？因为知识库目录在不同机器上的绝对路径必然不同，而删除文档时要靠 `source` 字段做匹配（见各子类的 `do_delete_doc`）。统一成相对路径，检索、删除、迁移才能跨环境保持一致。这段逻辑还被抽成了独立的 `get_relative_source_path` 方法，供 ES、PG 等子类的删除逻辑复用。

接下来是「先删后加」的幂等保证：`self.delete_doc(kb_file)` 先清掉同名文件的旧向量，再 `doc_infos = self.do_add_doc(docs, **kwargs)` 调用子类钩子真正写入。子类钩子约定返回 `List[Dict]`，每个元素形如 `{"id": ..., "metadata": ...}`。最后 `add_file_to_db` 把文件元信息、`custom_docs` 标志、切片数 `len(docs)` 以及 `doc_infos` 一并登记进关系型数据库。整条链路里，「向量怎么写」是子类的事，「写完要登记、写前要去重、metadata 要归一化」是基类的事——职责分得干干净净。

## search_docs 与 score:阈值过滤的两种归宿

检索的公共入口 `search_docs` 同样先做嵌入模型校验，然后把活儿丢给 `do_search`：

```python
def search_docs(self, query, top_k=Settings.kb_settings.VECTOR_SEARCH_TOP_K,
                score_threshold=Settings.kb_settings.SCORE_THRESHOLD):
    if not self.check_embed_model()[0]:
        return []
    docs = self.do_search(query, top_k, score_threshold)
    return docs
```

`top_k` 与 `score_threshold` 的默认值都来自 `Settings.kb_settings`，分别是 `VECTOR_SEARCH_TOP_K` 和 `SCORE_THRESHOLD`。值得注意的是，分数阈值过滤在这一版里有两条不同的落地路径：

一条是基类在 `base.py` 末尾提供的工具函数 `score_threshold_process`。它用 `operator.le`（小于等于）把分数高于阈值的结果滤掉，再截取前 `k` 个：

```python
def score_threshold_process(score_threshold, k, docs):
    if score_threshold is not None:
        cmp = operator.le
        docs = [(doc, sim) for doc, sim in docs if cmp(sim, score_threshold)]
    return docs[:k]
```

这里用「小于等于」而非「大于等于」是因为 FAISS/PGVector 这类用 L2 欧氏距离的库，分数越小越相似。`MilvusKBService`、`RelytKBService` 这类直接调用底层 `similarity_search_with_score` 的实现会显式调用它。另一条路径是把过滤交给检索器：FAISS、Chroma、ES 等子类不直接调 `score_threshold_process`，而是通过 `get_Retriever(...)` 拿到一个检索器服务（`ensemble`/`vectorstore`/`milvusvectorstore`），把 `top_k` 和 `score_threshold` 透传进去，由检索器内部完成阈值过滤与分数归一化。两种归宿殊途同归，都把「分数语义不统一」这个难题挡在了检索器/工具函数这一层，没让它泄漏到上层。

承载分数的数据结构是 `kb_document_model.py` 里的 `DocumentWithVSId`，它继承自 LangChain 的 `Document`，只加了两个字段：`id: str = None`（向量库里的主键）和 `score: float = 3.0`。默认分数 3.0 是个「比任何真实相似距离都大」的占位值，意味着没被检索打过分的文档会排在最后。`list_docs` 方法正是用它把数据库记录和向量库内容拼成带 ID 的完整文档：它先 `list_docs_from_db` 取出某文件或某 metadata 对应的记录，再对每条记录 `self.get_doc_by_ids([x["id"]])` 去向量库捞回正文，最后 `DocumentWithVSId(**{**doc_info.dict(), "id": x["id"]})` 把关系库的 ID 和向量库的内容合体。这条「关系库管目录、向量库管内容」的分工贯穿整个子系统。

## get_doc_by_ids:基类留白的设计

并非所有方法都用模板方法模式硬性约束子类。`get_doc_by_ids`、`del_doc_by_ids`、`update_doc_by_ids` 这组「按 ID 操作」的方法采取了更柔和的策略。`get_doc_by_ids` 在基类里直接 `return []`——一个无害的空默认，子类按需覆盖；`del_doc_by_ids` 则 `raise NotImplementedError`，逼着真要用这功能的子类必须实现。这种「软默认 + 硬报错」的混搭说明：能给出安全空实现的就给（查不到返回空列表天经地义），不能假装成功的就抛异常（删除却什么都没删是危险的静默错误）。

最值得玩味的是 `update_doc_by_ids`，它把「更新」拆成「先删后加」并内置了删除语义：

```python
def update_doc_by_ids(self, docs: Dict[str, Document]) -> bool:
    if not self.check_embed_model()[0]:
        return False
    self.del_doc_by_ids(list(docs.keys()))
    pending_docs, ids = [], []
    for _id, doc in docs.items():
        if not doc or not doc.page_content.strip():
            continue
        ids.append(_id)
        pending_docs.append(doc)
    self.do_add_doc(docs=pending_docs, ids=ids)
    return True
```

注释里写得明白：「如果对应 doc_id 的值为 None，或其 page_content 为空，则删除该文档」。实现上它先把所有传入 ID 一律删掉，再把非空文档重新加回去——于是「传入空文档」自然就等价于「只删不加」。这是一种用组合替代特判的优雅写法，把删除和更新统一进同一条代码路径。

## KBServiceFactory:配置到实现的唯一入口

有了统一契约，还需要一个地方决定「这次到底实例化哪个子类」。这就是 `KBServiceFactory`，一个纯静态方法的工厂。核心是 `get_service`：

```python
@staticmethod
def get_service(kb_name, vector_store_type, embed_model=get_default_embedding(),
                kb_info=None) -> KBService:
    if isinstance(vector_store_type, str):
        vector_store_type = getattr(SupportedVSType, vector_store_type.upper())
    params = {"knowledge_base_name": kb_name, "embed_model": embed_model,
              "kb_info": kb_info}
    if SupportedVSType.FAISS == vector_store_type:
        from ...faiss_kb_service import FaissKBService
        return FaissKBService(**params)
    elif SupportedVSType.MILVUS == vector_store_type:
        ...
```

三个设计细节值得圈出来。其一，第一行的 `getattr(SupportedVSType, vector_store_type.upper())` 把传入的小写字符串（如 `"faiss"`）转成枚举值，这正是前文说「用类属性而非 Enum」的回报——反射一行搞定。其二，每个分支里的 `import` 都是**函数内延迟导入**，而不是文件顶部统一导入。这是有意为之：Milvus、Elasticsearch、Chroma、PGVector 各自依赖一堆重量级第三方库，若在模块加载时全部 import，一个没装 `pymilvus` 的本地用户连 FAISS 都用不了。延迟导入让「你用哪个库才加载哪个库」成为可能，这对一个标榜「可离线运行」的项目至关重要。其三，所有具体服务的构造参数被统一打包进 `params` 字典，保证工厂对所有子类一视同仁。

工厂里那个耐人寻味的 `DEFAULT` 分支：

```python
elif SupportedVSType.DEFAULT == vector_store_type:
    from ...milvus_kb_service import MilvusKBService
    return MilvusKBService(**params)
```

`DEFAULT` 并没有映射到 `DefaultKBService`，而是落到了 `MilvusKBService`——也就是说，当配置里写 `default` 时，系统默认就用 Milvus 兜底。文件后面其实还有第二个 `SupportedVSType.DEFAULT` 分支才返回 `DefaultKBService(kb_name)`，但由于 `if/elif` 短路，这个分支永远走不到，是一段死代码。这种「DEFAULT 即 Milvus」的取舍透露出项目对生产场景的倾向。

工厂另外两个便捷方法把「按名字找服务」做成了一行调用。`get_service_by_name` 先 `load_kb_from_db(kb_name)` 从数据库取出该知识库注册时存的 `vs_type` 和 `embed_model`，库里没有就返回 `None`，否则用查到的类型重建服务：

```python
@staticmethod
def get_service_by_name(kb_name) -> KBService:
    _, vs_type, embed_model = load_kb_from_db(kb_name)
    if _ is None:
        return None
    return KBServiceFactory.get_service(kb_name, vs_type, embed_model)
```

这就是为什么 `create_kb` 必须把 `vs_type()` 写进数据库——它是日后无配置上下文也能精确重建服务的唯一线索。`get_default()` 则直接返回 `DEFAULT` 类型的服务，用于不关心具体库的场景。

## 同一份契约,七种落地姿势

抽象的价值要靠多实现来验证。下面横向对比几个子类如何履行同一套 `do_*` 契约，差异恰恰映射出各向量库的脾气。

`FaissKBService` 是最「重」的本地实现。它没有网络连接，而是通过 `kb_faiss_pool.load_vector_store(...)` 拿到一个线程安全的 `ThreadSafeFaiss` 句柄，所有操作都包在 `with self.load_vector_store().acquire() as vs:` 上下文里加锁。`do_add_doc` 手动 `embed_documents` 后用 `vs.add_embeddings` 写入并 `vs.save_local`；`do_delete_doc` 靠遍历 `vs.docstore._dict` 比对 `metadata["source"]`（小写化后比较）找出待删 ID；`do_search` 用的是 `ensemble` 检索器。一个细节是它的 `do_add_doc` 和 `do_delete_doc` 都检查 `kwargs.get("not_refresh_vs_cache")`，允许批量操作时跳过反复存盘——这是为了第 6、7 章会讲的批量迁移性能优化。FAISS 的「向量存磁盘」特性也让它重写了 `save_vector_store` 和 `exist_doc`（后者能区分 `"in_db"` 与 `"in_folder"`）。这正是 `base.py` 里 `save_vector_store` 默认空实现、注释写着「FAISS保存到磁盘，milvus保存到数据库」的原因。[[FAISS]](https://faiss.ai/)

`MilvusKBService` 走的是「连服务、用 collection」的路子。`_load_milvus` 用 LangChain 的 `Milvus` 包装器建连接，参数全来自 `Settings.kb_settings.kbs_config.get("milvus")`，并开启 `auto_id=True` 让 Milvus 自动生成主键。它的 ID 是整型，`get_doc_by_ids` 要把字符串 ID 转成 `int` 再拼 `pk in {...}` 表达式查询；`do_delete_doc` 不靠 source 匹配，而是先用 `list_file_num_docs_id_by_kb_name_and_file_name` 从关系库查出该文件对应的所有向量主键，再 `col.delete(expr=f"pk in {id_list}")`。这是个有意思的反差：FAISS 把删除信息藏在向量库的 metadata 里，Milvus 则依赖外部关系库做「文件名→向量 ID」的映射。`do_add_doc` 还做了 metadata 清洗，把每个值 `str(v)` 字符串化、补齐 collection 字段、剔除 text/vector 字段，以适配 Milvus 的 schema 约束。[[Milvus]](https://milvus.io/docs)

`ChromaKBService` 用 `chromadb.PersistentClient(path=self.vs_path)` 在本地持久化，`do_init` 里 `get_or_create_collection`。它的 ID 是 `str(uuid.uuid1())` 生成的字符串，`do_add_doc` 逐条 `_collection.add` 写入；`do_delete_doc` 用 `where={"source": kb_file.filepath}` 条件删除；`do_drop_kb` 等价于 `delete_collection`，而 `do_clear_vs` 干脆直接复用 `do_drop_kb`。[[Chroma]](https://docs.trychroma.com/)

`ESKBService` 的 `do_init` 最庞大，因为它要同时维护两个客户端：原生 `Elasticsearch` 客户端（`es_client_python`，做精确的 ID 查询与删除）和 LangChain 的 `ElasticsearchStore`（`self.db`，做向量检索与写入）。它用 `dense_vector` 字段存向量、`context` 字段存文本，删除时构造 `term` 查询匹配 `metadata.source.keyword`，并特意把 `track_total_hits` 设为 `True` 以拿到真实命中总数，绕开 ES 默认只返回 10 条的限制。[[Elasticsearch]](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)

`PGKBService` 把向量塞进 PostgreSQL 的 `langchain_pg_embedding` 表。它在类级别用 `sqlalchemy.create_engine(..., pool_size=10)` 建了共享连接池，`do_search` 用 `vectorstore` 检索器，距离策略指定为 `DistanceStrategy.EUCLIDEAN`；删除时直接写 SQL，用 `cmetadata::jsonb @> :cmetadata` 的 JSONB 包含运算符匹配 `source`。`do_drop_kb` 更是用裸 SQL 连删 `langchain_pg_embedding` 与 `langchain_pg_collection` 两张表的相关行。这是几个实现里唯一直接操作底层数据表的，把关系库当向量库用的味道很浓。[[pgvector]](https://github.com/pgvector/pgvector)

最后是 `DefaultKBService`，一个所有 `do_*` 方法都 `pass`、`vs_type` 返回 `"default"` 的空壳。它存在的意义不是真干活，而是给「需要一个 KBService 实例但不想真连任何向量库」的校验、占位场景兜底——契约要求子类实现全部抽象方法，它就老老实实把每个方法写成空操作，证明这套抽象的最小可实现成本极低。

## 本章小结

- `SupportedVSType` 用一组小写字符串类属性钉死了八种可选向量库类型，刻意不用 `Enum` 是为了配合工厂里 `getattr(SupportedVSType, type.upper())` 的反射式映射。
- `KBService` 是模板方法模式的范本：每个对外操作拆成「公共方法 + `do_*` 抽象钩子」，公共方法管校验、去重、metadata 归一化、数据库登记，钩子管向量库特定动作。
- 构造函数末尾立即回调 `do_init`，把「建立向量库连接」钉在实例化的同一时刻；`vs_type` 也是抽象方法，强制子类自报类型并写入数据库。
- `add_doc` 用「先 `delete_doc` 再 `do_add_doc`」保证幂等，并把绝对 source 路径归一化为相对路径，让检索、删除、迁移跨环境一致。
- 分数阈值过滤有两条归宿：基类的 `score_threshold_process`（用 `operator.le`，距离越小越相似）和各子类透传给 `get_Retriever` 检索器，二者都把分数语义不统一挡在了上层之外。
- `DocumentWithVSId` 在 `Document` 基础上加 `id` 和默认 `score=3.0`，3.0 是个比真实距离都大的占位值，让未打分文档自然排末尾。
- `KBServiceFactory.get_service` 是配置到实现的唯一入口，每个分支用函数内延迟 `import` 隔离重量级依赖，是「可离线运行」的关键工程取舍。
- `DEFAULT` 类型在工厂里被映射到 `MilvusKBService` 而非 `DefaultKBService`，后者的分支因 `if/elif` 短路成了死代码。
- `get_service_by_name` 靠数据库里登记的 `vs_type`/`embed_model` 重建服务，这是 `create_kb` 必须把 `vs_type()` 写库的原因。
- 多实现横向对比揭示各库脾气：FAISS 靠线程安全句柄与 metadata 删除、Milvus 靠外部关系库映射 ID、Chroma 用 uuid 字符串、ES 维护双客户端、PG 直接写裸 SQL，而它们对外都只暴露同一套契约。

## 动手实验

1. **实验一：给 KBService 画出模板方法调用图** — 在 `base.py` 中找出 `add_doc`、`delete_doc`、`update_doc`、`search_docs`、`create_kb`、`clear_vs`、`drop_kb` 七个公共方法，逐个标注它们各自回调了哪些 `do_*` 钩子，画一张「公共方法 → 抽象钩子」的对照表，体会哪些流程被基类收口、哪些被下放给子类。

2. **实验二：追踪一次工厂解析** — 在 `KBServiceFactory.get_service` 的入口和每个 `if/elif` 分支打印日志，分别传入 `"faiss"`、`"FAISS"`、`"milvus"`、`"default"` 四个字符串，观察 `getattr(..., upper())` 的转换结果与最终实例化的类，并验证传入 `"default"` 时返回的确实是 `MilvusKBService`、`DefaultKBService` 分支永不触发。

3. **实验三：对比两种向量库的删除语义** — 并排阅读 `FaissKBService.do_delete_doc` 与 `MilvusKBService.do_delete_doc`，找出前者靠遍历 `docstore._dict` 比对 `metadata["source"]`、后者靠 `list_file_num_docs_id_by_kb_name_and_file_name` 查外部关系库的差异，写一段话解释为什么同一个「按文件删向量」的需求会有两套截然不同的实现。

4. **实验四：实现一个最小自定义 KBService** — 仿照 `DefaultKBService`，写一个把 `do_add_doc`/`do_search` 落到 Python 内存列表的 `MemoryKBService`，让 `vs_type` 返回一个新字符串，在 `SupportedVSType` 与 `KBServiceFactory` 各加一个分支，跑通 `add_doc`/`search_docs`，亲手验证「只要填满 `do_*` 钩子就能接入新向量库」这一抽象承诺。

> **下一章预告**：本章我们看到 FAISS 用 `kb_faiss_pool.load_vector_store` 拿到线程安全句柄、各实现反复调用 `get_Embeddings(self.embed_model)` 把文本变成向量。这两件事——向量库句柄的缓存复用、嵌入模型的加载与封装——都被 Langchain-Chatchat 抽成了独立的缓存层。第 6 章《向量缓存与嵌入》将深入 `kb_cache` 目录，拆解 `ThreadSafeFaiss`、`kb_faiss_pool` 的池化与并发控制，以及嵌入函数适配器如何在多模型、多知识库间共享与隔离。
