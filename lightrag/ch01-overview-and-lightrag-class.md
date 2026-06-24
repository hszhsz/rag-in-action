# 第 1 章 项目全景与 LightRAG 类

每个 RAG 引擎都有一个"总入口"——你 `import` 进来、实例化、然后往里灌文档、对它提问的那个对象。LightRAG 把这个角色压缩成了一个名字直白的类：`LightRAG`。它声明在 `lightrag/lightrag.py`，是一个挂着几十个字段的 `@dataclass`，从工作目录、四类存储后端的类名、分块参数，一直到 LLM/embedding 函数，全都以字段形式摆在台面上。读懂这个类，几乎就摸清了整个系统的可配置面与装配顺序。

本章不急着钻进检索算法或图谱构建,而是先建立"地图感":LightRAG 这个仓库是怎么组织的、`LightRAG` 类如何用 dataclass 声明配置、它在 `__post_init__` 里把哪些零件装配起来、存储的生命周期怎么走、以及为什么几乎每个 async 方法都配了一个同名去掉 `a` 前缀的同步包装。把这条主干理清,后续章节(存储抽象、分块、抽取、双层检索)都能挂到它上面。

## 项目定位与 monorepo 布局

LightRAG 出自港大数据智能实验室(HKUDS),论文为 arXiv 2410.05779,README 把它定位成"Simple and Fast Retrieval-Augmented Generation"。它的核心思路是把**知识图谱**与**向量检索**结合:摄入文档时用 LLM 抽取实体与关系构图,查询时按"低层(实体)/高层(关系)"双层检索召回,再交给 LLM 生成答案。本书解构的是 `_version.py` 中标记的 `__version__ = "1.5.4"`(同文件还有一个独立的 `__api_version__ = "0312"`,用于 API 层版本协商)。

仓库是一个 monorepo,顶层关键目录:

- `lightrag/` —— Python 主包,本部聚焦的对象。`pyproject.toml` 里 `[tool.setuptools.packages.find]` 用 `include = ["lightrag*"]` 把它(及其子包)全部纳入打包。
- `lightrag/api/` —— REST 服务子模块(`lightrag_server.py`、`routers/`、`auth.py` 等),通过 `[project.scripts]` 暴露 `lightrag-server`、`lightrag-gunicorn` 等命令行入口。
- `lightrag_webui/` —— 独立的前端工程(`package.json`、`bun.lock`),构建产物经 `[tool.setuptools.package-data]` 的 `api/webui/**/*` 打进 wheel,由 API 服务托管。
- `examples/`、`tests/`、`reproduce/`、`scripts/` —— 被 `find` 的 `exclude` 列表排除在打包之外。

值得专门点出的是**包名与导入名的分裂**:`pyproject.toml` 里 `name = "lightrag-hku"`,所以 `pip install lightrag-hku`,但代码里 `import lightrag`。版本号在打包侧也不是写死的——`dynamic = ["version"]` 配合 `[tool.setuptools.dynamic]` 的 `version = {attr = "lightrag._version.__version__"}`,让 `_version.py` 成为版本号的唯一真相源,运行时和打包共用同一份定义(这正是该文件 docstring 所说"shared by packaging and runtime code"的含义)。依赖方面,`requires-python = ">=3.10"`,核心运行时依赖里能看到 `networkx`(默认图存储)、`nano-vectordb`(默认向量库)、`tiktoken`(默认分词)、`tenacity`(重试)、`json_repair`(修复 LLM 的 JSON 输出)等;重型后端(`neo4j`、`pymilvus`、`asyncpg`、`faiss-cpu` 等)和各家 LLM SDK 则被拆进 `offline-storage`、`offline-llm`、`api` 等 optional-dependencies,默认不装。

## 包的门面:`__init__.py` 的惰性导出

打开 `lightrag/__init__.py`,会发现它的 `__all__` 只暴露五个公共名:`LightRAG`、`QueryParam`、`RoleLLMConfig`、`RoleSpec`、`ROLES`,外加 `__version__`。但它并不在模块顶层直接 `from .lightrag import ...`,而是定义了一个模块级 `__getattr__`,在首次访问这些名字时才去 `lightrag.lightrag` 里取并写回 `globals()`。

这是 PEP 562 提供的模块级 `__getattr__` 惰性导入机制 [[PEP 562 – Module __getattr__ and __dir__]](https://peps.python.org/pep-0562/)。为什么要这么绕?因为 `lightrag.py` 一被导入就会牵出一长串子模块(prompt、constants、kg、operate、pipeline……),`import lightrag` 时若立刻全部加载,会拖慢启动、也可能在只想读 `__version__` 的场景下引入不必要的副作用。惰性导出让"轻量 import 包"与"重量加载实现"解耦;同时配合 `if TYPE_CHECKING:` 块里的真实 import,静态类型检查器仍能看到完整签名。这是一个小而典型的工程取舍:**对外接口收窄到五个名字,对内加载推迟到真正使用时**。

## `LightRAG` 作为 dataclass 总入口

类定义本身就是一份配置清单。`@dataclass` 的字段把"可调项"全部声明式地列出来,默认值的取法分几类:

- **纯字面量默认**:`working_dir: str = field(default="./rag_storage")`,`kv_storage="JsonKVStorage"`、`vector_storage="NanoVectorDBStorage"`、`graph_storage="NetworkXStorage"`、`doc_status_storage="JsonDocStatusStorage"`——四类存储后端**用字符串类名而非类对象**声明,这是后续可插拔存储的关键:构造时再由工厂按名字解析成类。
- **环境变量驱动**:大量字段用 `get_env_value("TOP_K", DEFAULT_TOP_K, int)` 之类的写法,把"环境变量 → 常量默认"串成一条链。比如 `top_k`、`chunk_top_k`、`max_total_tokens`、`cosine_threshold` 都遵循"先读 env,读不到落到 `constants.py` 里的 `DEFAULT_*`"。这让同一份代码在不改源码的前提下可由 `.env` 调参。
- **`default_factory`**:对可变默认(dict/list)或需要运行时求值的项使用工厂,避免可变默认值在实例间共享的经典陷阱——这正是 dataclass 要求可变默认必须走 `default_factory` 的原因 [[Python dataclasses — mutable defaults]](https://docs.python.org/3/library/dataclasses.html#mutable-default-values)。例如 `workspace` 用 `lambda: os.getenv("WORKSPACE", "")`,`embedding_cache_config` 用工厂返回 `{"enabled": False, ...}`,`chunking_func` 用工厂返回 `chunking_by_token_size`。
- **`init=False` 的内部状态**:`embedding_token_limit`、`_addon_params`、`_addon_params_dirty`、`_entity_extraction_prompt_profile` 等字段标了 `init=False`,不进构造签名,纯作内部缓存或派生状态。
- **`InitVar`**:`addon_params: InitVar[dict[str, Any] | None] = None` 是一个只在构造期可见、不会成为实例属性的"输入变量",它会作为参数传给 `__post_init__`,在那里被规整进私有的 `_addon_params`。

模块顶层还有一行 `load_dotenv(dotenv_path=".env", override=False)`,注释强调"OS 环境变量优先于 .env",这样每个工作目录可以放各自的 `.env`,而显式设置的系统环境变量永远压得住文件。

零配置取向贯穿始终:把上面这些默认值连起来看,`LightRAG(llm_model_func=..., embedding_func=...)` 只需提供两个**必填**回调(LLM 与 embedding,二者默认 `None`,缺失会报错),其余存储、分块、检索参数全都有可用默认——开箱就是本地 JSON+NetworkX+nano-vectordb 的全本地栈。

## 三个 Mixin 的职责拆分

类头是 `class LightRAG(_RoleLLMMixin, _StorageMigrationMixin, _PipelineMixin)`,主体之外的功能被切到三个 Mixin 里,各自来自独立模块:

- `_RoleLLMMixin`(来自 `lightrag/llm_roles.py`)—— 负责"按角色隔离的 LLM 封装"。LightRAG 把不同用途的 LLM 调用(抽取、查询、合并摘要等)抽象成"角色"(`ROLES`),每个角色可有独立的函数、并发上限与超时。这块逻辑(`_rebuild_role_llm_funcs`、`_log_llm_role_config` 等)放进 Mixin,使主类不被角色细节淹没。
- `_StorageMigrationMixin`(来自 `lightrag/storage_migrations.py`)—— 负责存储格式的版本迁移。
- `_PipelineMixin`(来自 `lightrag/pipeline.py`)—— 负责文档处理管线,即 `apipeline_enqueue_documents` / `apipeline_process_enqueue_documents` 这套"入队—解析—分块—抽取—入库"的流水线方法。

这种拆分的好处很实际:`lightrag.py` 单文件已超两千行,如果把角色管理、迁移、管线全塞进来会失控。Mixin 让相关方法分文件维护,同时通过 MRO 仍以 `self.` 形式共享同一份 dataclass 字段——`LightRAG` 实例既是配置容器,也是这些方法操作的状态宿主。后续第 9 章会专门拆 `_PipelineMixin`,第 10 章会拆 `_RoleLLMMixin`。

## `__post_init__` 装配了什么

dataclass 自动生成的 `__init__` 只负责把字段赋值;真正的装配在 `__post_init__(self, addon_params)` 里(它接收的那个参数,正是前面 `InitVar` 声明的 `addon_params`)。按源码顺序,它干了这几件事:

1. **快速失败的弃用检查**:若检测到已废弃的 `ENTITY_TYPES` 环境变量,直接 `raise SystemExit` 并提示改用 prompt 模板或 `addon_params`。
2. **规整 addon_params**:`_replace_addon_params` 把传入的字典包成 `ObservableAddonParams`(一个能在被修改时回调 `_mark_addon_params_dirty` 的可观察字典),随后 `_apply_chunk_size_overlay` 把分块尺寸按"addon_params > 旧构造参数 > env"的四层优先级铺平,再 `_refresh_addon_params_cache` 解析摘要语言与实体抽取 prompt profile。
3. **处理弃用字段**:对 `log_level`、`log_file_path` 发 `DeprecationWarning`,并用 `delattr` 把它们从实例上删掉以防误用。
4. **建工作目录**:`working_dir` 不存在则 `os.makedirs` 创建。
5. **校验存储实现**:对四类存储逐一 `verify_storage_implementation` 检查兼容性,并 `check_storage_env_vars` 校验所需环境变量是否齐备——把"后端没配好"的错误前移到构造期。
6. **装配 tokenizer / ollama 信息**:`tokenizer` 为空则按 `tiktoken_model_name`(默认 `gpt-4o-mini`)建 `TiktokenTokenizer`。
7. **配置合理性告警**:对 `force_llm_summary_on_merge < 3`、`summary_context_size > max_total_tokens` 等不合理组合打 warning(不阻断)。
8. **给回调套并发/超时装饰器**:对 `rerank_model_func` 和 `embedding_func` 用 `priority_limit_async_func_call` 包一层带优先级队列与超时的限流;注意 embedding 用 `dataclasses.replace` 生成**新的** `EmbeddingFunc` 实例,而非原地改写,以免同一个 embedding 对象被多个 LightRAG 实例共享时互相污染。
9. **实例化全部存储**:用工厂 `get_storage_class` 把四个存储**类名字符串**解析成类,再 `functools.partial` 把 `global_config` 预绑定上去,最后逐一构造出十多个具体存储对象——`full_docs`、`text_chunks`、`full_entities`、`entity_chunks` 等 KV 存储,`entities_vdb`/`relationships_vdb`/`chunks_vdb` 三个向量库(各自带不同的 `meta_fields`),`chunk_entity_relation_graph` 图存储,以及 `llm_response_cache`、`doc_status`。每个都按 `NameSpace.*` 常量分配命名空间,并带上 `workspace` 做数据隔离。
10. **构建角色 LLM 状态**:校验 `role_llm_configs` 的合法角色名,为 `ROLES` 中每个角色建 `_RoleLLMState`,然后 `_rebuild_role_llm_funcs`。

装配完成后,`_storages_status` 被置为 `StoragesStatus.CREATED`。注意此时存储对象虽已实例化,但**尚未真正打开连接/加载文件**——那是下一步的事。

## `global_config`:把整个实例摊平成一份字典

`__post_init__` 里有个反复出现的关键动作:`global_config = self._build_global_config()`,然后用 `functools.partial` 把它预绑定到每个存储类上。这个 `global_config` 是 LightRAG 内部的"配置快照总线"——存储后端、抽取函数、检索函数都不直接依赖 `LightRAG` 实例,而是依赖这份扁平化的字典,从而保持解耦。

`_build_global_config` 的实现很有讲究:它先 `_ensure_addon_params_cache()` 确保派生缓存是最新的,再用 `dataclasses.asdict(self)` 把整个实例**深拷贝**成字典,然后剔除几个内部私有字段(`_addon_params`、`_addon_params_dirty` 等),把规整后的 `addon_params` 重新塞回去,最后注入两组运行时才有的内容:`role_llm_funcs`(每个角色封装后的可调用 LLM)和 `llm_cache_identities`(每个角色的缓存身份)。这里有个微妙的时序细节,源码注释专门说明:`__post_init__` 里第一次调 `_build_global_config()` 时角色状态尚未建好,所以这两组会回退成空字典/`None`——因为 `asdict` 发生在角色初始化之前。

为什么不直接传 `self`?因为 `asdict` 会把 `EmbeddingFunc` 这样的对象也转成字典,破坏其方法;所以 `__post_init__` 在拿到 `global_config` 后会手动 `global_config["embedding_func"] = original_embedding_func` 把原对象塞回去。还有一个更硬的约束:事件循环对象 `_owning_loop` **不能**进 dataclass 字段,否则 `asdict` 的深拷贝会在 Python 3.14 上对活的 loop 抛 `TypeError`——这正是它被刻意放在 `__post_init__` 里以普通属性赋值、而非声明为 dataclass 字段的原因。这些都是"把实例摊平成配置字典"这一设计选择倒逼出来的工程细节。

## 存储对象清单:一个实例到底装了多少存储

`__post_init__` 末尾一口气实例化的存储对象,值得列清楚,因为它们就是后续所有数据流的落点。按存储类别分:

- **KV 存储(`key_string_value_json_storage_cls`)**:`full_docs`(原文)、`text_chunks`(分块)、`full_entities`、`full_relations`、`entity_chunks`、`relation_chunks`,以及 `llm_response_cache`(LLM 响应缓存)。
- **向量库(`vector_db_storage_cls`)**:`entities_vdb`、`relationships_vdb`、`chunks_vdb` 三个,各自声明了不同的 `meta_fields`——例如 `entities_vdb` 带 `{"entity_name", "source_id", "content", "file_path"}`,`relationships_vdb` 带 `{"src_id", "tgt_id", ...}`,`chunks_vdb` 带 `{"full_doc_id", "content", "file_path"}`。这组 `meta_fields` 决定了向量库除向量外还冗余存哪些可直接读出的元数据。
- **图存储(`graph_storage_cls`)**:`chunk_entity_relation_graph` 一个,承载实体—关系知识图。
- **文档状态(`doc_status_storage_cls`)**:`doc_status`,跟踪每篇文档 PENDING/PROCESSED/FAILED 等处理状态。

这 12 个对象在 `initialize_storages`、`finalize_storages` 以及内部的 `_index_storages()`(返回所有需要统一 flush 的存储)中被反复以同一份清单引用。把它们摆在一起看,能直观感受到 LightRAG 的数据是**多副本、按用途分库**的:同一份 chunk 既进 `text_chunks`(KV)又进 `chunks_vdb`(向量),实体既进图又进 `entities_vdb` 与若干 KV——这种冗余正是双层检索能同时按图遍历和按向量相似召回的物理基础。

## 存储生命周期:`initialize_storages` / `finalize_storages`

`__post_init__` 只把存储对象"造出来",真正可用还需异步初始化。`initialize_storages()` 是一个 `async` 方法,只在状态为 `CREATED` 时执行:

- 记录 `self._owning_loop = asyncio.get_running_loop()`——把"存储绑定到哪个事件循环"记下来,这是同步包装跨循环安全检查的依据(见下一节)。
- 设置默认 workspace、初始化该 workspace 的 pipeline_status。
- **逐个 `await storage.initialize()`**(注释明确写"must be called one by one to prevent deadlock",顺序串行而非并发,以避免共享锁死锁)。
- 完成后状态转为 `INITIALIZED`。

`finalize_storages()` 是对称的收尾:仅在 `INITIALIZED` 时执行,逐个 `await storage.finalize()`,但用 try/except 把每个存储的关闭隔离开——**一个存储关闭失败不应阻止其它存储释放资源**,失败的会被收集并最终汇总记录。完成后状态转为 `FINALIZED`。

为什么不在构造时自动初始化?dataclass 里有个 `auto_manage_storages_states` 字段(默认 `False`),其注释已标注为弃用,并明确"LightRAG 永不在创建时自动初始化存储"。这是刻意的:存储初始化必须发生在一个**确定的事件循环**上,而构造函数是同步的,无法稳妥地决定该用哪个循环。把生命周期交给调用方显式控制(常见写法是 `rag = LightRAG(...); await rag.initialize_storages()`),换来的是对"绑定到哪个 loop"的可控性。

## "每个 async 配一个 sync 包装"的 API 设计

LightRAG 是异步优先的:真实逻辑都写在 `ainsert`、`aquery`、`aquery_data` 这些协程里,然后各配一个去掉 `a` 前缀的同步方法 `insert`、`query`、`query_data`。这些同步方法的实现高度统一,都委托给一个内部辅助函数 `_run_sync`:

```
return _run_sync(
    lambda: self.aquery(query, param, system_prompt),
    sync_name="query", async_name="aquery",
    owning_loop=self._owning_loop,
)
```

`_run_sync` 的职责不只是"跑个协程",它内置了两道防护:

- **不能在运行中的事件循环里调同步包装**:先 `asyncio.get_running_loop()` 探测,若当前线程已有运行中的 loop(例如直接在 `async def` 或 FastAPI handler 里调 `rag.query(...)`),则报错并提示改用 `await aquery(...)`——因为同步包装内部要调 `loop.run_until_complete()`,而 Python 禁止在已运行的 loop 上再嵌套运行 [[asyncio.Runner / run_until_complete]](https://docs.python.org/3/library/asyncio-runner.html)。
- **跨循环检查**:存储的共享锁绑定在 `_owning_loop` 上;若从另一个仍然存活的 loop 驱动(典型如 `loop.run_in_executor(None, rag.insert, ...)`),会报"Lock is bound to a different event loop"。`_run_sync` 在 `owning_loop` 仍开着且不等于当前要驱动的 loop 时主动报错;但如果 `owning_loop` 已关闭(常见的 `asyncio.run(initialize_rag())` 模式会在初始化后关掉那个 loop),则放行到一个全新 loop 上跑,保持向后兼容。

这套设计的取向很清楚:**同步包装是给简单单线程脚本的便利糖,不是并发入口**。源码 docstring 直白地建议:SDK 集成、异步应用、并发负载都应直接用 `a*` 协程。把"误用"在边界处显式报错并给出修复建议,胜过让它在底层抛出晦涩的锁错误。

值得一提的是 `aquery` 本身也是个兼容层:它内部调用更结构化的 `aquery_llm`,再从结果里抽出 `llm_response` 还原成"老接口只返回字符串/异步迭代器"的形态——当 `llm_response["is_streaming"]` 为真时返回 `response_iterator`,否则返回 `content` 字符串。而 `aquery_data` 走的是同一套检索逻辑但在 LLM 生成前停住,返回结构化的实体/关系/chunks/references——为不需要生成、只要召回数据的场景(如自建前端、评测)提供了直达出口。它的 docstring 还列清了六种查询模式(`local`/`global`/`hybrid`/`mix`/`naive`/`bypass`)在返回数据上的差异,这些模式我们留到第 6 章双层检索时再细拆。

## `ainsert` 与管线的关系:SDK 入口的边界

理解 `insert`/`ainsert` 还要看清它在整条摄入链里的**定位**。`ainsert` 是一个面向 SDK 的便利入口,它的源码很短:生成 `track_id`(若未提供)→ 用 `resolve_chunk_options` 把 `split_by_character`/`split_by_character_only` 等运行时参数固化成一份 `chunk_options` 快照 → 调 `apipeline_enqueue_documents` 入队 → 调 `apipeline_process_enqueue_documents` 处理,最后返回 `track_id` 供调用方查询处理状态。

关键在于它的局限,源码 docstring 写得很明白:`ainsert` **永远只用固定 token(F)分块策略**,因为它故意不传 `process_options`,因此无法选择递归字符(R)、语义向量(V)或段落语义(P)策略。更进一步,LightRAG 的**服务端/REST API 根本不调用 `ainsert`**——它直接调 `apipeline_enqueue_documents` + `apipeline_process_enqueue_documents` 并带上每文档的 `process_options` 选择器,这才是 F/R/V/P 在服务端被选定的地方。这条注释揭示了一个重要事实:**SDK 入口(`ainsert`)和服务端入口(管线两方法)走的是同一条底层管线,但 SDK 入口是被刻意"阉割"到只走 F 策略的薄封装**。想从 SDK 用 R/V/P,就得绕过 `ainsert` 直接调那两个管线方法。

仓库里还保留了 `insert_custom_chunks`/`ainsert_custom_chunks` 这对"自带分块结果"的入口(已标 deprecated),它跳过分块,直接对调用方给的 `text_chunks` 计算 `chunk-` 前缀的 MD5 id、补 token 数,然后 `asyncio.gather` 并发地把 chunks 写入向量库、抽取实体、写入 `full_docs` 与 `text_chunks`。它用 `filter_keys` 做幂等去重(已存在的文档/分块会被跳过),并在 `finally` 里统一 `_insert_done_with_cleanup` 收尾——这种"先去重、再并发写、最后统一 flush"的形态,在第 9 章管线里会以更完整的版本反复出现。

## 本章小结

- `LightRAG` 类(`lightrag/lightrag.py`)是整个系统的总入口,以 `@dataclass` 形式把几十个可配置项声明式地摆在字段层。
- 包名 `lightrag-hku`、导入名 `lightrag`,版本号由 `_version.py` 的 `__version__ = "1.5.4"` 单点定义,经 `pyproject.toml` 的 `dynamic`/`attr` 机制供打包与运行时共用。
- monorepo 含主包 `lightrag/`、API 子模块 `lightrag/api/`、前端 `lightrag_webui/`;重后端与 LLM SDK 拆进 optional-dependencies,默认只装本地全栈。
- 四类存储后端用**字符串类名**声明(`JsonKVStorage`/`NanoVectorDBStorage`/`NetworkXStorage`/`JsonDocStatusStorage`),构造时由 `get_storage_class` 工厂解析,这是可插拔存储的基础。
- 字段默认值走"env → `DEFAULT_*` 常量"链与 `default_factory`,实现零配置可用——只有 `llm_model_func`、`embedding_func` 必填。
- 三个 Mixin(`_RoleLLMMixin`/`_StorageMigrationMixin`/`_PipelineMixin`)把角色 LLM、存储迁移、文档管线从主类剥离,按文件维护又共享同一份 dataclass 状态。
- `__post_init__` 负责弃用检查、addon_params 规整、目录创建、存储校验、回调限流包装、十多个存储对象实例化与角色状态构建,完成后置 `StoragesStatus.CREATED`。
- 存储生命周期由 `initialize_storages()`(串行 `await initialize`,记录 `_owning_loop)`)与 `finalize_storages()`(隔离式逐个 `finalize`)显式管理,刻意不在构造时自动初始化。
- 每个 async 方法配一个同步包装,统一经 `_run_sync` 加上"运行中 loop 禁用"与"跨循环锁绑定"两道防护;同步包装定位为单线程脚本便利糖,并发场景应直接 `await a*`。

## 动手实验

1. **实验一:画字段分类表** — 用 `Read` 通读 `lightrag.py` 第 262 行起的 dataclass 定义,把所有字段按"字面量默认 / `get_env_value` env 驱动 / `default_factory` / `init=False` 内部状态 / `InitVar`"五类各归一栏,数一数每类各有多少个,体会"配置面"有多大。
2. **实验二:追踪一个默认值的来源** — 选 `top_k`,在 `constants.py` 里找到 `DEFAULT_TOP_K` 的值,再在 shell 里设 `TOP_K=42` 后构造 `LightRAG` 并打印 `rag.top_k`,验证 `get_env_value("TOP_K", DEFAULT_TOP_K, int)` 这条 env→常量链确实生效。
3. **实验三:制造跨循环报错** — 在一个 `async def` 里直接调用 `rag.query("...")`(同步包装),观察 `_run_sync` 抛出的"cannot be called from within a running asyncio event loop"报错;再改成 `await rag.aquery("...")` 确认正常,亲手体会同步包装的边界防护。
4. **实验四:对照生命周期状态机** — 在 `__post_init__` 末尾、`initialize_storages` 末尾、`finalize_storages` 末尾分别打印 `rag._storages_status`,观察它从 `CREATED → INITIALIZED → FINALIZED` 的迁移,确认"构造只造对象、初始化才打开存储"的分离设计。

> **下一章预告**:本章我们看到四类存储后端只是一串字符串类名,真正的类要靠工厂解析。第 2 章将深入 `lightrag/base.py` 的存储抽象,拆解 `BaseKVStorage`、`BaseVectorStorage`、`BaseGraphStorage`、`DocStatusStorage` 四类基类的契约,看看 LightRAG 如何用统一接口把 JSON、向量库、图数据库这些异构后端收编到同一套生命周期之下。
