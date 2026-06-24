# 第 7 章 可插拔存储后端

第 2 章我们拆过 LightRAG 的存储抽象：四类基类 `BaseKVStorage` / `BaseVectorStorage` / `BaseGraphStorage` / `DocStatusStorage` 各自定义了一组契约方法，业务层只针对契约编程，从不直接 import 任何具体实现。这一章要回答的是契约的另一半：当一个名字（字符串 `"NetworkXStorage"`）摆在配置里时，LightRAG 是怎样把它兑换成一个真正能跑的类对象的？校验、惰性导入、环境变量检查在什么时候发生？默认实现凭什么"开箱即用"，重型后端又凭什么"换一行配置就能横向扩展"？

整个机制的核心代码量惊人地小：注册表与校验逻辑集中在 `kg/__init__.py`（160 余行），工厂 `kg/factory.py` 只有 43 行。但它们撑起了"同一套契约、四类存储、十余种后端、从本地单文件到分布式数据库平滑迁移"的全部弹性。本章顺着"声明 → 校验 → 选型 → 默认实现 → 重型 override"的链路逐段解构。

## 一、四类存储的实现清单：`STORAGE_IMPLEMENTATIONS`

`kg/__init__.py` 顶部是一个名为 `STORAGE_IMPLEMENTATIONS` 的字典，它是整套可插拔机制的"花名册"。键是四种存储类型，值里有两个字段：`implementations`（允许的实现名清单）和 `required_methods`（该类型的契约方法名）。结构精简到可以直接抄录：

```python
STORAGE_IMPLEMENTATIONS = {
    "KV_STORAGE": {
        "implementations": ["JsonKVStorage", "RedisKVStorage", "PGKVStorage",
                            "MongoKVStorage", "OpenSearchKVStorage"],
        "required_methods": ["get_by_id", "upsert"],
    },
    "GRAPH_STORAGE": {
        "implementations": ["NetworkXStorage", "Neo4JStorage", "PGGraphStorage",
                            "MongoGraphStorage", "MemgraphStorage", "OpenSearchGraphStorage"],
        "required_methods": ["upsert_node", "upsert_edge"],
    },
    # VECTOR_STORAGE / DOC_STATUS_STORAGE 同构...
}
```

以图存储为例，`GRAPH_STORAGE` 的 `implementations` 列出了六种实现：`NetworkXStorage`、`Neo4JStorage`、`PGGraphStorage`、`MongoGraphStorage`、`MemgraphStorage`、`OpenSearchGraphStorage`；它的 `required_methods` 只有两个：`upsert_node` 和 `upsert_edge`。

向量存储 `VECTOR_STORAGE` 给了七种实现——`NanoVectorDBStorage`、`MilvusVectorDBStorage`、`PGVectorStorage`、`FaissVectorDBStorage`、`QdrantVectorDBStorage`、`MongoVectorDBStorage`、`OpenSearchVectorDBStorage`，外加一个被注释掉的 `ChromaVectorDBStorage`——`required_methods` 是 `query` 和 `upsert`。

KV 存储的契约是 `get_by_id` / `upsert`，文档状态存储的契约只有一个 `get_docs_by_status`。

这里第一个值得玩味的设计：`required_methods` 故意只列**最小契约**，而不是把基类里所有抽象方法都罗列一遍。它的目的不是穷举接口，而是给"校验"提供一个轻量探针——一个类要声明自己属于某类型，至少得实现这几个最关键的方法。注册表本身保持"声明即文档"的简洁：读这几行就能知道每类存储有哪些备选、契约的核心是什么。

## 二、模块路径映射：`STORAGES`

光有名字清单还不够，工厂得知道"`PGGraphStorage` 这个类住在哪个模块里"。这由同文件里的 `STORAGES` 字典承担——它把每个实现名映射到一个相对模块路径：

```python
STORAGES = {
    "NetworkXStorage": ".kg.networkx_impl",
    "JsonKVStorage": ".kg.json_kv_impl",
    "NanoVectorDBStorage": ".kg.nano_vector_db_impl",
    "Neo4JStorage": ".kg.neo4j_impl",
    "PGKVStorage": ".kg.postgres_impl",
    "PGVectorStorage": ".kg.postgres_impl",
    "PGGraphStorage": ".kg.postgres_impl",
    "PGDocStatusStorage": ".kg.postgres_impl",
    "FaissVectorDBStorage": ".kg.faiss_impl",
    # ...
}
```

注意一个细节：四个 Postgres 实现（`PGKVStorage` / `PGVectorStorage` / `PGGraphStorage` / `PGDocStatusStorage`）全部指向同一个 `.kg.postgres_impl` 模块；四个 Mongo 实现、四个 OpenSearch 实现同理。

这说明同一个数据库后端可以在一个模块文件里同时实现四类存储，工厂按类名 `getattr` 取出对应的类即可。`STORAGES` 把"名字 → 模块"和"名字 → 类型"两件事解耦：前者管物理位置（`STORAGES`），后者管契约归属（`STORAGE_IMPLEMENTATIONS`）。两张表各司其职，互不污染。

## 三、惰性导入工厂：`get_storage_class`

`factory.py` 里的 `get_storage_class(storage_name)` 是把字符串兑换成类的唯一入口。它的实现分两段，体现了一个非常务实的取舍。

第一段是四个 `if` 硬编码分支：`JsonKVStorage`、`NanoVectorDBStorage`、`NetworkXStorage`、`JsonDocStatusStorage`——也就是四类存储各自的默认实现——在函数体内直接 import 并返回：

```python
def get_storage_class(storage_name: str) -> Callable[..., Any]:
    if storage_name == "JsonKVStorage":
        from lightrag.kg.json_kv_impl import JsonKVStorage
        return JsonKVStorage
    if storage_name == "NanoVectorDBStorage":
        from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
        return NanoVectorDBStorage
    # NetworkXStorage / JsonDocStatusStorage 同理...

    # Fallback: 其余实现走动态导入
    import_path = STORAGES[storage_name]
    module = importlib.import_module(import_path, package="lightrag")
    return getattr(module, storage_name)
```

函数顶部的 docstring 说得很直白：这四个默认后端被直接 import，"so they always work without depending on the ``STORAGES`` registry"。即使 `STORAGES` 注册表出了问题，默认四件套也永远能加载——这是"开箱即用"承诺的兜底。

第二段是 fallback：对于其它所有实现名，从 `STORAGES[storage_name]` 取出相对路径，用 `importlib.import_module(import_path, package="lightrag")` 动态导入模块，再 `getattr(module, storage_name)` 取出类。这里 `package="lightrag"` 是关键——`STORAGES` 里的值都是 `.kg.xxx` 形式的相对路径，必须锚定到顶层 `lightrag` 包才能正确解析，docstring 专门注释了这一点。

为什么不在文件顶部一次性 import 所有实现？因为 `neo4j_impl`、`postgres_impl`、`milvus_impl` 等模块各自依赖 `neo4j`、`asyncpg`、`pymilvus` 这些第三方库，而绝大多数用户只装了默认四件套需要的 `networkx` / `nano-vectordb`。

**惰性导入让"只为你用到的后端付依赖代价"成为可能**：你选了 `NanoVectorDBStorage`，就永远不会触发 `import faiss`。这与第 2 章讲的"基类契约让业务层无感切换"是一对互补的设计——基类负责接口统一，工厂负责按需加载。

## 四、第一道前置约束：兼容性校验

选型之前，LightRAG 在 `__post_init__` 阶段做了两层"约束前置"。`lightrag.py` 里把四类存储的配置组成 `storage_configs` 列表，逐个调用两个函数：

```python
storage_configs = [
    ("KV_STORAGE", self.kv_storage),
    ("VECTOR_STORAGE", self.vector_storage),
    ("GRAPH_STORAGE", self.graph_storage),
    ("DOC_STATUS_STORAGE", self.doc_status_storage),
]
for storage_type, storage_name in storage_configs:
    verify_storage_implementation(storage_type, storage_name)
    check_storage_env_vars(storage_name)
```

第一道是 `verify_storage_implementation(storage_type, storage_name)`（定义在 `kg/__init__.py`）。它先确认 `storage_type` 是已知类型，再检查 `storage_name` 是否落在该类型的 `implementations` 名单内；不匹配就抛 `ValueError`，并把"兼容的实现有哪些"一并列进报错信息。

这道校验拦的是"把 `Neo4JStorage` 误配成向量存储"这类张冠李戴的错误——名字本身合法，但放错了类型槽位。它依赖的就是第一节那张 `STORAGE_IMPLEMENTATIONS` 名单：名单就是合法性的唯一裁判。

## 五、第二道前置约束：环境变量检查

第二道是 `check_storage_env_vars(storage_name)`（定义在 `utils.py`）。它从 `STORAGE_ENV_REQUIREMENTS` 里取出该实现声明的环境变量列表，逐个检查 `os.environ`，缺哪个就抛 `ValueError` 并报出缺失变量名：

```python
def check_storage_env_vars(storage_name: str) -> None:
    from lightrag.kg import STORAGE_ENV_REQUIREMENTS
    required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        raise ValueError(
            f"Storage implementation '{storage_name}' requires the following "
            f"environment variables: {', '.join(missing_vars)}"
        )
```

`STORAGE_ENV_REQUIREMENTS` 是 `kg/__init__.py` 里的第三张表：每个后端自己声明依赖的环境变量。默认四件套（`JsonKVStorage` / `NetworkXStorage` / `NanoVectorDBStorage` / `JsonDocStatusStorage`）以及 `FaissVectorDBStorage` 的列表都是空 `[]`——它们不需要任何外部连接信息。

而重型后端各有要求：`Neo4JStorage` 声明 `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD`，四个 Postgres 实现都声明 `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DATABASE`，四个 Mongo 实现声明 `MONGO_URI` / `MONGO_DATABASE`，`MilvusVectorDBStorage` 声明 `MILVUS_URI` / `MILVUS_DB_NAME`，`QdrantVectorDBStorage` 声明 `QDRANT_URL`，四个 OpenSearch 实现都声明 `OPENSEARCH_HOSTS`。

这两道校验的共同价值是**让错误在初始化时（fail fast）就暴露**，而不是等到第一次 upsert 连数据库时才崩。配置写错、忘了设密码——在 LightRAG 对象构造的那一刻就报清楚，不会让一个跑了半小时的摄入任务在写库阶段才失败。

声明式的环境表也让"某后端到底要配什么"有了单一可查的事实来源。这张表同样被 `tools/rebuild_vdb.py`、`tools/migrate_llm_cache.py`、`tools/clean_llm_query_cache.py` 等运维脚本复用，做迁移前的环境预检——它不只是初始化的守门员，还是整个工具链共享的"需求声明"。

校验通过后，`lightrag.py` 才调用 `get_storage_class` 真正取类，并用 `functools.partial` 把 `global_config` 预绑定到类上，得到 `key_string_value_json_storage_cls` / `vector_db_storage_cls` / `graph_storage_cls` / `doc_status_storage_cls`。后续每次实例化存储都从这些 partial 出发。配置项的默认值也印证了"开箱即用"取向：`kv_storage` 默认 `JsonKVStorage`、`vector_storage` 默认 `NanoVectorDBStorage`、`graph_storage` 默认 `NetworkXStorage`、`doc_status_storage` 默认 `JsonDocStatusStorage`。

## 六、默认实现的"零依赖即可用"

默认四件套的共同基因是：纯本地文件 + 进程内内存 + 延迟落盘，不需要任何外部服务。

### KV 与 DocStatus：本地 JSON

`JsonKVStorage`（`json_kv_impl.py`）与 `JsonDocStatusStorage`（`json_doc_status_impl.py`）把数据存成 `working_dir/[workspace/]kv_store_<namespace>.json`。它们的 `self._data` 不是普通 per-process 字典，而是 `get_namespace_data` 返回的跨进程共享代理（多进程模式下是 `multiprocessing.Manager().dict()`），任何进程的写入对其它进程立即可见；磁盘文件只在 `initialize` 时读一次、在 `index_done_callback` 时刷一次，纯为持久化服务。

`upsert` 把数据写进共享 dict 后调用 `set_all_update_flags` 标脏，由后续 `index_done_callback` 用原子的 `write_json` 落盘——这套共享内存 + 脏标志协议是第 8 章的主角，这里只点到为止。值得一提的是 `JsonDocStatusStorage.upsert` 会在写完内存后**同步**调用一次 `index_done_callback` 立即落盘，因为文档状态是摄入流水线崩溃后的恢复锚点：如果进程在内存 upsert 之后、下一批次提交之前崩了，重启时这份文档必须仍以 PENDING/PROCESSING 可见，否则就会丢任务。

### 向量库：NanoVectorDB 的延迟嵌入

`NanoVectorDBStorage`（`nano_vector_db_impl.py`）基于纯本地的 `nano-vectordb`。它在初始化时构造 `NanoVectorDB(embedding_dim, storage_file=self._client_file_name)`，整个向量库就是一个进程内对象加一个本地 JSON 文件。

它最精巧的设计是**延迟嵌入**：`upsert` 并不立刻调用嵌入模型，而是把记录以 `vector=None` 缓存进 `self._pending_upserts`：

```python
async def upsert(self, data):
    pending = [(k, {"__id__": k, "__created_at__": current_time,
                    **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields}})
               for k, v in data.items()]
    async with self._storage_lock:
        for doc_id, record in pending:
            self._pending_upserts[doc_id] = _PendingNanoDoc(record=record)
```

真正的嵌入推迟到 `index_done_callback` 里的 `_flush_pending_locked`——这样多次 upsert 同一个 id、以及大量小批 upsert 都能合并成一次嵌入，省下大量模型调用。

`query` 只搜索已经 flush 进 `self._client` 的数据：先算查询向量，再 `client.query(query=embedding, top_k=top_k, better_than_threshold=self.cosine_better_than_threshold)` 过滤；未落盘的 pending 数据要靠 `get_by_id` 这类 read-your-writes 路径才看得到。`index_done_callback` 是写者的提交点：先 reload 别的进程的最新快照、再 flush 嵌入、再用 `_save_to_disk_locked`（基于 `atomic_write`，先写 tmp 文件再 rename）原子落盘，保证读者要么看到旧文件要么看到新文件，绝不会读到撕裂的半截写入；任一步失败都向上抛，让 `_insert_done` 中止整批而不是悄悄丢向量。

### 图存储：NetworkX 的同步内存操作

`NetworkXStorage`（`networkx_impl.py`）基于 `networkx`，把整张知识图谱保存为本地 graphml 文件。`upsert_node` / `upsert_edge` 极其朴素——拿到内存里的 `nx` 图对象后直接操作：

```python
async def upsert_node(self, node_id, node_data):
    graph = await self._get_graph()
    graph.add_node(node_id, **node_data)

async def upsert_edge(self, source_node_id, target_node_id, edge_data):
    graph = await self._get_graph()
    graph.add_edge(source_node_id, target_node_id, **edge_data)
```

因为 networkx 操作是全同步的，无需复杂锁；持久化同样延迟到 `index_done_callback`。`get_knowledge_graph` 则在内存图上做裁剪并返回子图供前端可视化。三个默认实现都把"立即写内存、延迟落盘"作为统一节奏，配合单写者流水线门控保证一致性。

这就是"零依赖即可用"的全部代价：`pip install` 之后无需任何数据库、无需任何环境变量，四件套就地在 `working_dir` 里跑起来。对原型验证、单机小数据场景，这是最低摩擦的起点。

## 七、重型后端如何 override 批量方法换性能

默认实现够用但不够快——尤其图存储的批量读取。第 2 章讲过 `BaseGraphStorage` 给了一组 `*_batch` 方法的**默认串行实现**。`base.py` 里的 `get_nodes_batch` 就是一个逐个调用单体方法的循环：

```python
async def get_nodes_batch(self, node_ids):
    """Default implementation fetches nodes one by one.
    Override this method for better performance ..."""
    result = {}
    for node_id in node_ids:
        node = await self.get_node(node_id)
        if node is not None:
            result[node_id] = node
    return result
```

`node_degrees_batch`、`edge_degrees_batch`、`get_edges_batch`、`get_nodes_edges_batch`、`upsert_nodes_batch`、`upsert_edges_batch` 同样是"逐个调用单体方法"的 fallback，每个 docstring 都写着同一句话：`Override this method for better performance in storage backends that support batch operations.`

这正是"默认能跑、后端可优化"原则的落地。对 `NetworkXStorage`，批量就是内存里的一个本地 for 循环，串行成本可忽略——它的 `upsert_nodes_batch` / `upsert_edges_batch` 也只是把 `add_node` / `add_edge` 套进一个循环，省去 per-call 的协程调度开销而已。

但对 `Neo4JStorage`，逐个 `get_node` 意味着 N 次网络往返——这是性能灾难。于是 `neo4j_impl.py` 把这些批量方法**全部用单条 Cypher 的 `UNWIND` 重写**：

```python
async def get_nodes_batch(self, node_ids):
    query = f"""
    UNWIND $node_ids AS id
    MATCH (n:`{workspace_label}` {{entity_id: id}})
    RETURN n.entity_id AS entity_id, n
    """
    result = await session.run(query, node_ids=node_ids)
    # ...一次拿回所有节点
```

`node_degrees_batch` / `edge_degrees_batch` / `get_edges_batch` / `get_nodes_edges_batch` 以及 `upsert_nodes_batch` / `upsert_edges_batch` 同样各用一条带 `UNWIND` 的语句把 N 次往返压成 1 次。`postgres_impl.py` 里的 `PGGraphStorage` 同理 override 了 `upsert_nodes_batch`、`get_nodes_batch`、`node_degrees_batch`、`get_nodes_edges_batch` 等，用 SQL/AGE 的批量语法替换串行 fallback。

关键在于：**这些 override 对调用方完全透明**。第 6 章的双层检索在召回阶段会调用 `get_nodes_batch` / `node_degrees_batch` 一次性拿一批实体的属性和度数；它不知道也不关心底下是 networkx 的内存 for 循环还是 Neo4j 的一条 `UNWIND`——契约方法签名一致，行为语义一致，只是性能差几个数量级。

基类提供"一定能跑"的保底语义，让新后端可以只实现单体方法就先跑起来，再按需逐个 override 批量方法压榨性能。这种"渐进式优化"的空间，正是把批量接口设计成"有默认实现的普通方法"而非"必须实现的抽象方法"换来的。

## 八、同一套契约下的平滑迁移

把前七节串起来，就得到 LightRAG 存储层最大的工程价值：从本地单机到分布式数据库的迁移，对业务代码是零改动的。

想从默认的 NetworkX 切到 Neo4j，你只需要两步：把 `graph_storage` 配成 `"Neo4JStorage"`，再设好 `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` 三个环境变量。

之后的链路全自动：`verify_storage_implementation` 确认 `Neo4JStorage` 在 `GRAPH_STORAGE` 的名单里，`check_storage_env_vars` 确认三个变量都在，`get_storage_class` 惰性 `import neo4j_impl` 并取出类，`partial` 绑定 `global_config`——业务层拿到的还是一个满足 `BaseGraphStorage` 契约的对象。

四类存储可以各自独立选型：KV 用 Redis、向量用 Milvus、图用 Neo4j、状态用 Postgres，互不耦合，因为它们在 `STORAGE_IMPLEMENTATIONS` 里是四张独立的名单，在 `STORAGES` 里各自映射模块。注册表 + 工厂 + 校验三件套，把"换后端"这件在很多系统里牵一发动全身的事，压缩成了"改配置 + 设环境变量"。

## 本章小结

- LightRAG 把 KV / Vector / Graph / DocStatus 四类存储都做成可插拔后端，注册表与工厂代码不足 250 行却撑起十余种实现的弹性。
- `kg/__init__.py` 的 `STORAGE_IMPLEMENTATIONS` 是花名册：每类型声明 `implementations`（允许的实现名）和 `required_methods`（最小契约方法名），保持"声明即文档"。
- `STORAGES` 把实现名映射到模块路径，同一数据库的四类实现常共用一个模块文件（如四个 PG 实现都在 `postgres_impl`）；它与 `STORAGE_IMPLEMENTATIONS` 解耦物理位置与契约归属。
- `factory.py` 的 `get_storage_class` 用"默认四件套硬编码 import + 其余 `importlib` 惰性导入"两段式实现，保证默认后端永远可加载，同时让用户"只为用到的后端付第三方依赖代价"。
- 初始化时两道前置约束：`verify_storage_implementation` 查兼容性、`check_storage_env_vars` 按 `STORAGE_ENV_REQUIREMENTS` 查环境变量，缺啥立即 `ValueError`，实现 fail-fast。
- `STORAGE_ENV_REQUIREMENTS` 中默认四件套与 Faiss 的需求为空，重型后端各自声明连接所需变量；这张表还被 `tools/` 下的运维脚本复用做迁移预检。
- 默认四件套零外部依赖、纯本地文件 + 进程内内存 + 延迟落盘，开箱即用；`vector_storage` 等配置默认值即指向它们。
- `NanoVectorDBStorage` 的延迟嵌入（`upsert` 只缓存、`index_done_callback` 才 flush 嵌入）合并了重复与小批写入；落盘用 `atomic_write` 的 tmp+rename 防撕裂写。
- `BaseGraphStorage` 的 `*_batch` 方法给了串行 fallback（逐个调单体方法），docstring 明确建议后端 override。
- 重型后端如 `Neo4JStorage` 用单条 `UNWIND` Cypher 把 N 次网络往返压成 1 次，`PGGraphStorage` 用批量 SQL/AGE 同理；override 对调用方完全透明。
- 四类存储独立选型、互不耦合，迁移到分布式数据库只需"改配置 + 设环境变量"，业务层因契约统一而零改动。

## 动手实验

1. **实验一：画出选型链路** — 从 `lightrag.py` 的 `storage_configs` 循环出发，跟读 `verify_storage_implementation` → `check_storage_env_vars` → `get_storage_class` → `partial` 绑定 `global_config` 的完整调用序列，画一张"配置字符串如何变成可实例化的存储类"的流程图，标出每一步可能抛 `ValueError` 的位置。
2. **实验二：验证惰性导入** — 在只装了默认依赖的环境里，把 `vector_storage` 配成 `"FaissVectorDBStorage"` 并观察 `check_storage_env_vars` 不报错（其需求为空），再切到 `"MilvusVectorDBStorage"` 但不设 `MILVUS_URI`，确认初始化阶段抛出列出缺失变量的 `ValueError`；对照 `STORAGE_ENV_REQUIREMENTS` 解释差异。
3. **实验三：对比批量实现** — 并排阅读 `base.py` 的默认 `get_nodes_batch`（串行 for 循环）和 `neo4j_impl.py` 的 `get_nodes_batch`（`UNWIND` 单条 Cypher），数清两者各发起多少次"单体查询/网络往返"，写一段说明解释为什么 networkx 不必 override 而 Neo4j 必须 override。
4. **实验四：实测延迟嵌入** — 给 `NanoVectorDBStorage.upsert` 与 `_flush_pending_locked` 加日志，对同一个 id 连续 upsert 三次后只调一次 `index_done_callback`，确认嵌入模型只被调用一次而非三次，体会延迟嵌入合并写入的效果。

> **下一章预告**：本章默认实现反复提到的"共享内存 dict、脏标志、`index_done_callback` 提交点、单写者门控"等机制，正是 LightRAG 在多进程下保证存储一致性的基石。第 8 章《共享存储与并发控制》将深入 `shared_storage.py`，拆解 `Manager().dict()` 跨进程代理、命名空间锁、`set_all_update_flags` / `clear_all_update_flags` 的脏标志协议，以及文件型与共享内存型两套截然相反的同步语义。
