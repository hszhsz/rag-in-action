# 第 11 章 API 服务与路由

前面十章我们一直在 LightRAG 的「内核」里打转：分块、抽取、双层检索、存储抽象、共享并发。这些能力最终要被外部世界使用,就必须有一层网络服务把它们暴露出去。LightRAG 选择了 FastAPI 作为这一层的载体,把核心 `LightRAG` 实例包装成一个可以上传文档、发起检索、操作知识图谱,甚至伪装成 Ollama 服务的 HTTP 服务器。

本章我们沿着 `lightrag/api/` 目录,逐层拆解这个服务是如何被组装起来的:从 `create_app` 工厂如何拼装应用、`config.py` 如何把命令行参数与环境变量层叠成 `global_args`,到鉴权依赖、文档/检索/图谱三组路由的设计,再到 Ollama 兼容层与 Gunicorn 多 worker 部署。我们关心的不只是「有哪些接口」,而是「为什么这么设计」——尤其是那些为了反向代理、并发安全、流式输出而埋下的细节。

一个值得提前点明的主线是:LightRAG 的 API 层并非孤立的一层 HTTP 壳,它与第 8 章的共享存储、第 9 章的处理流水线深度耦合。几乎每一个会改变系统状态的接口,背后都站着一组跨进程共享的并发标志;几乎每一处生命周期与部署模式的分叉,都是为了让「多个 worker 安全地共享同一份存储」这件事成立。带着这条线索去读,API 层的许多看似琐碎的判断就都有了归宿。

## `create_app` 工厂:一次性把应用拼装出来

服务的入口是 `lightrag/api/lightrag_server.py` 里的 `create_app(args)` 函数(从第 1200 行开始)。它不是简单地 `app = FastAPI()`,而是一个完整的装配流水线。FastAPI 官方推荐用「应用工厂」模式把构建逻辑收敛到一个函数里,便于测试与多实例创建[[Bigger Applications - Multiple Files]](https://fastapi.tiangolo.com/tutorial/bigger-applications/)。

这个函数很长,但脉络清晰,可以拆成三段来理解:前置校验与依赖准备、应用本体构造、路由与端点挂载。下面我们就按这个顺序展开。

`create_app` 首先做一连串前置检查与准备:校验前端构建产物是否存在、配置日志、调用 `load_third_party_parsers()` 加载第三方解析器、`validate_parser_routing_config()` 校验解析路由、构建 `LLMConfigCache`、校验 LLM 与 embedding 绑定是否合法、校验 SSL 证书。随后确定 API Key——`api_key = os.getenv("LIGHTRAG_API_KEY") or args.key`,即环境变量优先于命令行参数。接着创建 `doc_manager = DocumentManager(args.input_dir, workspace=args.workspace)`,把输入目录与工作区绑定。

应用本体通过一组 `app_kwargs` 构造:`title="LightRAG Server API"`、`openapi_url="/openapi.json"`,但特意把 `docs_url=None`——因为 LightRAG 自己实现了一个**离线版** Swagger UI 挂在 `/docs`,静态资源放在 `/static/swagger-ui/...`,这样内网或无外网环境也能打开交互文档。`redoc_url="/redoc"` 保留。`swagger_ui_parameters={"persistAuthorization": True, "tryItOutEnabled": True}` 让用户登录态在文档页刷新后不丢失。最关键的是 `root_path=api_prefix if api_prefix else None` 与 `lifespan=lifespan`,前者用于反向代理前缀,后者管理生命周期。

为什么要把这么多检查塞进工厂的开头?因为 LightRAG 的服务一旦启动就要长期对外提供能力,任何「配置错了但进程还能跑」的情况都比「启动即失败」更难排查。把绑定校验、SSL 校验、解析路由校验全部前置到 `create_app`,等于让所有可预见的配置错误在端口监听之前就暴露,这是「fail fast」原则在服务装配阶段的体现。同样的取舍也解释了 `docs_url=None` 这个看似多余的选择:官方默认的 `/docs` 会从公网 CDN 拉取 Swagger UI 的 JS/CSS,内网部署会白屏;LightRAG 用 `_inject_swagger_theme` 把本地静态资源拼成自定义文档页,并额外注册 `/docs/oauth2-redirect`,从而做到完全离线可用。

工厂尾部(第 2085~2091 行)才是路由真正挂载的地方,且**顺序固定**:先 `app.include_router(create_document_routes(rag, doc_manager, api_key))`,再 `create_query_routes(rag, api_key, args.top_k)`、`create_graph_routes(rag, api_key)`,最后 `ollama_api = OllamaAPI(rag, top_k=args.top_k, api_key=api_key)` 以 `prefix="/api"` 挂载。三组业务路由各自接收已经确定好的 `rag`、`api_key` 等依赖,这种「构造时注入依赖」而非「运行时全局查找」的写法,让每个路由工厂都是自包含的纯函数,测试时可以传入 mock 的 `rag`。Ollama 路由统一带 `/api` 前缀,既符合 Ollama 客户端的路径约定,也让它整体落入默认白名单 `/api/*`,与业务路由的鉴权策略自然区隔。

## `lifespan`:存储的初始化与收尾

`lifespan`(第 1267 行)是一个 async 上下文管理器,FastAPI 用它替代了旧的 `startup`/`shutdown` 事件,把「启动前准备」与「关闭后清理」写在同一个函数的 `yield` 两侧[[Lifespan Events]](https://fastapi.tiangolo.com/advanced/events/)。

`yield` 之前,它把 `app.state.background_tasks = set()` 初始化为后台任务集合,然后 `await rag.initialize_storages()` 打开所有存储后端,`await rag.check_and_migrate_data()` 执行必要的数据迁移,最后打印 "Server is ready..."。`yield` 之后(进程退出时)进入 `finally`:`await rag.finalize_storages()` 关闭存储句柄。这里有一个对多 worker 部署至关重要的分支——只有当 `"LIGHTRAG_GUNICORN_MODE" not in os.environ` 时,才调用 `finalize_share_data()`。也就是说,Uvicorn 单进程模式下由 lifespan 自己清理共享内存;而 Gunicorn 多 worker 模式下,共享数据的最终清理交给 master 进程的 `on_exit` 钩子统一完成,避免某个 worker 提前销毁其他 worker 仍在使用的共享段。

把数据迁移 `check_and_migrate_data` 放进 lifespan 启动阶段也是经过权衡的:它保证了「服务对外可用」与「数据格式已是最新」这两件事的原子性——只要 `/health` 返回 ready,就意味着迁移已完成,客户端不会在迁移半途打进来读到不一致的数据。这种「先迁移、再开张」的顺序,把存储演进的复杂度对调用方完全隐藏,也为第 12 章要讲的离线运维工具划清了边界:在线服务负责轻量的、随启动自动完成的迁移,重型的缓存重建与向量库重算则交给离线脚本。

## 反向代理与中间件:`root_path` 与前缀归一化

把服务部署在 Nginx/Traefik 之后、挂在某个子路径(如 `/rag`)上时,后端收到的请求路径与对外暴露的路径会不一致。FastAPI 通过 `root_path` 告诉应用「我被挂在哪个前缀下」,从而让生成的 OpenAPI 文档和重定向 URL 带上正确前缀[[Behind a Proxy]](https://fastapi.tiangolo.com/advanced/behind-a-proxy/)。

LightRAG 不止设置了 `root_path`,还额外加了一个自定义中间件 `_RootPathNormalizationMiddleware`,**仅当** `api_prefix` 存在时才挂载。它负责在请求进入路由匹配前,把带前缀的路径剥离/归一化,使得无论代理是否已经 strip 前缀,内部路由都能正确命中。中间件的挂载顺序是:先加 `_RootPathNormalizationMiddleware`,再加 `CORSMiddleware`。由于 Starlette 的中间件按「后进先出」包裹,这个顺序保证 CORS 处理在最外层。`CORSMiddleware` 配置了 `allow_credentials=True` 与 `expose_headers=["X-New-Token"]`——后者很关键,因为令牌自动续期会通过响应头 `X-New-Token` 下发,必须显式暴露浏览器才能读到。跨域来源由 `get_cors_origins()` 决定:读取 `global_args.cors_origins`,值为 `"*"` 时返回 `["*"]`,否则按逗号切分。

`WEBUI_PATH` 固定为 `/webui`(第 386 行),前端通过 `SmartStaticFiles` 挂载;`_normalize_api_prefix` 负责把用户传入的前缀规范化(补斜杠、去重复等)。

## 配置系统:argparse 与环境变量的层叠

服务的所有可调参数集中在 `lightrag/api/config.py`(873 行)。核心是 `parse_args()`:它用 `argparse` 定义命令行选项,而**每个选项的默认值都来自 `get_env_value`**,即「命令行 > 环境变量 > 硬编码默认」三层覆盖。常见选项如下:

- `--host`(`HOST`,默认 `0.0.0.0`)、`--port`(`PORT`,默认 `9621`)
- `--working-dir`、`--input-dir`、`--workspace`:工作目录、输入目录、工作区名
- `--workers`(`WORKERS`/`DEFAULT_WOKERS`):worker 数
- `--key`(`LIGHTRAG_API_KEY`):API Key
- `--api-prefix`(`LIGHTRAG_API_PREFIX`):反向代理前缀
- `--llm-binding`、`--embedding-binding`、`--rerank-binding`:三类模型的供应商绑定

「三层覆盖」的好处是同一份镜像能适配多种部署方式:本地开发用命令行参数最快,容器部署用环境变量最干净,而代码里的硬编码默认值保证「什么都不配也能跑起来」。`get_env_value` 还负责类型转换,把环境变量字符串解析成 int/bool 等目标类型。

`parse_args()` 还会**条件性**地追加绑定特定选项:根据 `--llm-binding`、`--embedding-binding`、`--rerank-binding` 的取值,动态注入各 provider 需要的 host/model/key 等参数。随后把存储配置(默认 `DefaultRAGStorageConfig`:KV 用 `JsonKVStorage`、向量用 `NanoVectorDBStorage`、图用 `NetworkXStorage`、文档状态用 `JsonDocStatusStorage`)、模型配置、JWT 鉴权配置(`auth_accounts`、`token_secret`、`token_expire_hours=48`、`guest_token_expire_hours=24`、`jwt_algorithm=HS256`、`token_auto_renew=True`、`token_renew_threshold=0.5`)、`whitelist_paths`(默认 `/health,/api/*`)、`max_upload_size`(默认 104857600,即 100MB)、`cors_origins`(默认 `*`)一并塞进同一个命名空间。

配置末尾有两道校验:`validate_auth_configuration` 在设置了 `AUTH_ACCOUNTS` 的同时仍沿用代码里写死的默认 `DEFAULT_TOKEN_SECRET` 时会拒绝启动——防止用户开了鉴权却忘了换密钥;`validate_bedrock_auth_configuration` 校验 AWS Bedrock 凭证。这种「配置了敏感功能就必须显式覆盖默认值」的强约束,把一类典型的安全误用(用默认密钥签发生产 JWT)直接挡在启动阶段。

`global_args` 不是一个普通对象,而是 `_GlobalArgsProxy` 实例。它实现了**懒加载**:首次访问任意属性时才触发 `initialize_config()` 解析参数。这样模块在被 import 时不会立刻解析命令行(否则会与测试、子进程冲突),真正用到配置时才计算。`update_uvicorn_mode_config()` 则在 Uvicorn 模式下强制 `global_args.workers = 1`——Uvicorn 单进程无法安全地跑多 worker 共享内存,这里直接纠正。

## 鉴权:API Key 与 JWT 的组合依赖

鉴权逻辑分布在 `auth.py` 与 `utils_api.py`。`auth.py` 里的 `AuthHandler`(模块单例 `auth_handler`)从 `global_args` 读取账户、密钥与算法,**显式拒绝 `none` 算法**以防降级攻击;把 `AUTH_ACCOUNTS` 解析为 `user:password` 列表;`verify_password` 同时支持 bcrypt 与明文;`create_token` 用 `jwt.encode` 按角色(`user`/`guest`)生成不同过期时长的 JWT;`validate_token` 校验过期并返回 username/role/metadata/exp。`TokenPayload` 用 Pydantic 建模,字段含 `sub`、`exp`、`role="user"`、`metadata={}`。

真正挂到路由上的是 `utils_api.py` 的 `get_combined_auth_dependency(api_key)`,它返回一个 `combined_dependency` 异步函数,按固定顺序判定:

1. **白名单放行**:`whitelist_patterns`(由 `whitelist_paths` 预编译,`/*` 结尾按前缀匹配)命中则直接通过。
2. **令牌校验优先**:若请求带了 token,先 `validate_token`——这样无效 token 立刻得到 401,而非滑落到 API Key 分支。校验通过后进入**自动续期**逻辑:跳过 `/health`、`/documents/paginated`、`/documents/pipeline_status` 这些高频轮询路径(续期无意义),对其余路径检查剩余有效期是否低于 `token_renew_threshold`,且距上次续期超过 60 秒(`_RENEWAL_MIN_INTERVAL` 限流),满足则签发新 token 写入 `X-New-Token` 响应头。无鉴权配置时接受 guest token,有鉴权配置时接受非 guest token。
3. **无保护放行**:既没配 auth 也没配 API Key 时全部放行。
4. **API Key 校验**:`X-API-Key` 头等于配置值则通过,否则按情况返回 401(缺凭证)或 403(Key 错误/缺失)。

这种「滑动窗口续期 + 头部下发」的设计,让前端只要持续访问就能自动延长登录态,无需感知刷新接口;而限流与跳过轮询路径,避免了高频请求把续期变成性能负担。续期阈值 `token_renew_threshold=0.5` 意味着令牌过半生命周期后才开始续签,配合 `_RENEWAL_MIN_INTERVAL=60` 秒的最小间隔,把签名计算的成本控制在每用户每分钟至多一次。这里还有一个容错细节:续期逻辑整体包在 try/except 里,源码注释明确写着「Renewal failure should not affect normal request, just log」——续期失败只记日志,绝不让一次续期异常阻断本就合法的请求。

与之配套的还有 `/login` 与 `/auth-status` 两个端点。`/login` 用 FastAPI 的 `OAuth2PasswordRequestForm` 接收表单凭证,校验通过后由 `auth_handler.create_token` 签发;若根本没配置 `AUTH_ACCOUNTS`,它会签发一个 **guest** 角色的 token,这样「未开启鉴权」的部署也能让前端拿到一个统一格式的令牌走通流程。`/auth-status` 则让前端在加载时探测服务端是否开启了鉴权,从而决定要不要显示登录页。这两个端点与 `combined_dependency` 里「无 auth 配置时接受 guest token、有 auth 配置时接受非 guest token」的判定彼此咬合,构成了一套「有无鉴权都能跑同一套前端」的优雅退化方案。

## 文档路由:上传、扫描与流水线协同

文档相关接口由 `document_routes.py` 的 `create_document_routes(rag, doc_manager, api_key)` 创建。注意它每次调用都 `new` 一个 `APIRouter`,源码注释解释了原因:若用模块级 router,多次创建应用会累积重复路由、导致 operation ID 冲突。同样的「工厂内新建 router」模式贯穿 query、graph、ollama 三组路由。

接口集合按职责可以分成四组:

- **入库类**:`POST /scan`(扫描输入目录新文件)、`POST /upload`(上传单文件)、`POST /text`、`POST /texts`(直接灌入文本)。
- **状态查询类**:`GET /pipeline_status`(流水线实时状态)、`GET ` (文档状态列表 `DocsStatusesResponse`)、`GET /track_status/{track_id}`(按追踪 ID 查进度)、`POST /paginated`(分页列文档)、`GET /status_counts`(各状态计数)。
- **清理与重试类**:`DELETE ` (清空全部)、`DELETE /delete_document`(删指定文档)、`POST /clear_cache`(清缓存)、`POST /reprocess_failed`(重跑失败文档)。
- **流水线控制类**:`POST /cancel_pipeline`(取消当前流水线)。

这套接口的形态揭示了 LightRAG 文档处理是**异步**的:上传/扫描只负责入队并立即返回一个 `track_id`,真正的解析、分块、抽取在后台流水线里跑;客户端随后靠 `track_status`、`paginated`、`status_counts` 这些查询接口轮询进度。这也解释了前面续期逻辑为何要把 `/documents/paginated`、`/documents/pipeline_status` 列入跳过清单——它们正是被前端高频轮询的状态接口。

`/scan`(第 2440 行)展示了与流水线的协同:先 `generate_track_id("scan")` 生成追踪 ID,通过 `get_namespace_data("pipeline_status", workspace=rag.workspace)` 读取共享的流水线状态;若发现 `busy`/`scanning` 为真或 `pending_enqueues>0`,直接返回 `status="scanning_skipped_pipeline_busy"` 拒绝重复扫描;否则在锁内置 `pipeline_status["scanning"]=True` 与 `["scanning_exclusive"]=True`,再用 `background_tasks.add_task(run_scanning_process, rag, doc_manager, track_id)` 把实际扫描丢到后台执行,接口立即返回。这正是第 8、9 章共享状态机在 API 层的体现:用一组并发标志(`busy`/`scanning`/`scanning_exclusive`/`pending_enqueues`)在多 worker 间协调,谁也不会重复入队同一批文件。

`/upload`(第 2563 行)的健壮性同样值得注意:先 `_reserve_enqueue_slot(rag)` 预占入队名额,`sanitize_filename` 清洗文件名,`is_supported_file` 校验扩展名,超过 `MAX_UPLOAD_SIZE` 返回 413,同名规范化 basename 冲突返回 409,文件以 1MB 分块异步流式写盘——避免大文件一次性读进内存。`DocumentManager`(第 977 行)把输入目录按 workspace 隔离,`supported_extensions` 实时从解析器注册表派生,提供 `scan_directory_for_new_files` 与 `is_supported_file`。请求体模型如 `InsertTextRequest`、`InsertTextsRequest` 均用 `ConfigDict(extra="forbid", strict=True)` 严格拒绝多余字段。

`extra="forbid"` 这个选择体现了 LightRAG 对接口契约的态度:与其默默忽略客户端拼错的字段(如把 `file_source` 写成 `fileSource`),不如直接报 422 让调用方立刻发现问题——这对一个需要被多种客户端集成的服务尤其重要。`strict=True` 则关闭 Pydantic 的隐式类型转换,把「字符串 `"5"` 自动当成整数 5」这类宽松行为关掉,保证传入类型与声明严格一致。值得对比的是,文档上传的「预占名额—校验—写盘」三段式与 `/scan` 的「读状态—置标志—丢后台」三段式,本质都是同一种模式:**任何会改变流水线状态的入口,都要先在共享状态上抢占资格,失败则快速返回,成功才真正动手**。这正是把第 8 章的并发控制原语落到 HTTP 边界的标准做法。

## 检索路由:`QueryRequest` 到 `QueryParam` 的映射与流式输出

检索接口由 `query_routes.py` 的 `create_query_routes(rag, api_key=None, top_k=60)` 创建。核心数据模型是 `QueryRequest`(Pydantic),字段覆盖了检索的全部旋钮:

- `query`:查询文本,`min_length=3`。
- `mode`:检索模式,`Literal` 限定 `local`/`global`/`hybrid`/`naive`/`mix`/`bypass`,默认 `mix`。
- `only_need_context`、`only_need_prompt`:只要上下文/只要拼好的 prompt,不真正调用 LLM。
- `top_k`(`ge=1`)、`chunk_top_k`:实体/关系召回数与 chunk 召回数。
- `max_entity_tokens`、`max_relation_tokens`、`max_total_tokens`:各部分的 token 预算上限。
- `hl_keywords`、`ll_keywords`:高/低层关键词,对应双层检索。
- `conversation_history`、`user_prompt`、`response_type`:对话历史、自定义提示与期望的回答形态。
- `enable_rerank`、`include_references`(默认 True)、`include_chunk_content`(默认 False)、`stream`:重排开关、引用与原文开关、流式开关。

配套验证器 `query_strip_after`(去空白)和 `conversation_history_role_check`(校验对话角色)在反序列化阶段就把脏数据挡掉。

请求到核心参数的桥梁是 `to_query_params(is_stream)`:它 `model_dump(exclude_none=True, exclude={"query","include_chunk_content"})` 把请求字段抽成字典,直接构造 `QueryParam(**request_data)`,再覆盖 `param.stream = is_stream`。这种「模型字段名与 `QueryParam` 字段名对齐」的设计,让新增一个检索参数只需在两边各加一个字段,无需手写映射代码。

三个端点对应三种返回形态:`POST /query`(`response_model=QueryResponse`)强制 `param.stream=False`,调用 `rag.aquery_llm`,按需把引用补上 chunk 内容;`POST /query/stream` 用 `_build_stream_generator` 逐块 yield NDJSON,以 `StreamingResponse(media_type="application/x-ndjson", ...)` 返回,响应头带 `X-Accel-Buffering: no` 关闭 Nginx 缓冲以实现真正的逐字输出;`POST /query/data`(`response_model=QueryDataResponse`)调用 `rag.aquery_data`,总是带上引用,供需要结构化检索结果的客户端使用。

NDJSON(换行分隔的 JSON)是流式场景的常见选择:每行一个独立 JSON 对象,客户端读到一行即可解析一块,无需等待整体完成[[Newline Delimited JSON]](https://github.com/ndjson/ndjson-spec)。值得一提的是 `lightrag_server.py` 里专门为 `/query/data` 注册了自定义 `RequestValidationError` 处理器:该路径校验失败时返回 400 与 `{status, message, data, metadata}` 结构,而非 FastAPI 默认的 422,方便该接口的客户端统一解析错误。

为什么要把检索拆成三个端点而不是用一个布尔开关切换?因为三者的**返回契约根本不同**:`/query` 返回的是面向人的最终答案文本(可选附引用),`/query/stream` 返回的是逐字推送的事件流,`/query/data` 返回的是面向程序的结构化检索结果(实体、关系、chunk、引用)。把它们拆开,各自声明独立的 `response_model`(`QueryResponse`/`QueryDataResponse`),OpenAPI 文档里就能给出精确的返回结构,客户端也能据此生成强类型代码;若挤在一个端点里靠参数切换,文档将无法表达「不同参数下返回不同结构」这件事。`include_references`(默认 True)与 `include_chunk_content`(默认 False)这对开关也体现了同样的克制:引用元数据通常需要,但把每个引用的原文 chunk 都塞进响应会显著放大体积,所以默认不带、按需开启。

## 图谱路由:知识图谱的 CRUD

图谱接口由 `graph_routes.py` 的 `create_graph_routes(rag, api_key=None)` 创建,可以清晰地分为只读与可写两类。

只读端点(查询图谱结构):

- `GET /graph/label/list`:列出所有标签。
- `GET /graph/label/popular`:热门标签,`limit` 限 1~1000。
- `GET /graph/label/search`:按关键字 `q` 搜标签。
- `GET /graphs`:取子图,参数 `label`、`max_depth=3`、`max_nodes=1000`。
- `GET /graph/entity/exists`:判断实体是否存在。

可写端点(实体/关系的增改删与合并):

- `POST /graph/entity/edit` → `rag.aedit_entity`
- `POST /graph/relation/edit` → `rag.aedit_relation`
- `POST /graph/entity/create` → `rag.acreate_entity`
- `POST /graph/relation/create` → `rag.acreate_relation`
- `POST /graph/entities/merge` → `rag.amerge_entities`
- `DELETE /graph/entity/delete` → `rag.adelete_by_entity`
- `DELETE /graph/relation/delete` → `rag.adelete_by_relation`

所有**写**端点在动手前都会 `await check_pipeline_busy_or_raise(rag)`(从 `document_routes` 导入)。这是个关键的一致性保护:当后台流水线正在抽取实体、写图谱时,如果同时允许用户手工编辑,二者会争抢同一份图存储,导致脏写。于是图谱写操作与文档流水线被同一组并发标志互斥——又一次复用了共享状态机。这也是把读端点与写端点分得这么清楚的现实理由:读永远安全、随时可调,写则必须先确认流水线空闲,二者的并发约束截然不同。

## Ollama 兼容层:把 RAG 伪装成一个本地模型

`ollama_api.py` 里的 `OllamaAPI` 类把 LightRAG 包装成一个 Ollama 兼容服务,挂载时带 `prefix="/api"`(故 `whitelist_paths` 默认含 `/api/*`)。这样像 Open WebUI 这类原生支持 Ollama 协议的客户端,可以零改造地把 LightRAG 当成一个「本地大模型」来对话。

它实现了 Ollama 的几个关键端点:

- `GET /version`:返回写死的 `0.9.3`,让客户端的版本探测通过。
- `GET /tags`:把自己报告成一个名为 `LIGHTRAG_MODEL` 的模型,`details` 里 `format="gguf"`、`parameter_size="13B"` 等都是占位元数据。
- `GET /ps`:返回「运行中模型」列表,同样是占位信息。
- `POST /generate`:补全接口,一律绕过 RAG 直连底层 LLM。
- `POST /chat`:对话接口,是真正接入 RAG 检索的主入口。

最有意思的是 `parse_query_mode(query)`:它从查询的**前缀**解析检索模式。`/local`、`/global`、`/naive`、`/hybrid`、`/mix`、`/bypass` 前缀分别映射到对应 `SearchMode`;`/context` 系列(如 `/localcontext`)则把 `only_need_context` 置真;还支持 `/local[use mermaid format] query` 这种方括号语法注入 `user_prompt`。这套前缀约定让用户无需改客户端 UI,仅靠在聊天框输入特殊前缀就能切换 RAG 模式。

这是一个相当巧妙的「在受限协议里偷偷夹带参数」的手法:Ollama 的 `/chat` 协议本身没有「检索模式」这个概念,但用户的消息内容是自由文本。LightRAG 于是约定用 `/前缀` 开头来携带本应放进 `QueryParam` 的旋钮,`parse_query_mode` 用一张 `mode_map` 字典做前缀匹配,再用正则 `^/([a-z]*)\[(.*?)\](.*)` 抽出方括号里的 `user_prompt`。这样任何兼容 Ollama 的现成客户端都能驱动 LightRAG 的全部检索模式,而无需为 LightRAG 定制 UI——这正是「兼容层」存在的全部意义:用别人的协议外壳,装自己的能力内核。

`/chat` 把消息列表的最后一条作为 query、其余作为 `conversation_history`,构造 `QueryParam` 后调用 `rag.aquery`;但若前缀是 `/bypass`,或检测到 Open WebUI 用于生成会话标题/关键词的特征文本(正则匹配 `\n<chat_history>\nUSER:`),则**绕过 RAG**,直接把请求转发给底层 LLM(`rag.role_llm_funcs["query"]`)——因为这些元数据生成任务不需要检索,走 RAG 反而浪费且会污染结果。流式响应同样用 `application/x-ndjson` 并带 `X-Accel-Buffering: no`;`parse_request_body` 还兼容了 `application/json` 与 `application/octet-stream` 两种 Content-Type。

值得注意的是 Ollama 层伪造了一整套 Ollama 风格的响应字段:`total_duration`、`load_duration`、`prompt_eval_count`、`eval_count`、`eval_duration` 等本是 Ollama 用来汇报推理耗时与 token 数的字段,LightRAG 用 `time.time_ns()` 计时、用 `estimate_tokens`(基于 `TiktokenTokenizer`)估算 token 数填进去。这些数字对 LightRAG 自身没有意义,但 Open WebUI 之类的客户端会读取并展示它们,所以兼容层必须「装得像」——这是协议兼容的代价:不仅要接住请求,还要把响应里客户端期待的每一个字段都填上合理的值。

## 健康检查、WebUI 与工作区隔离

除了三组业务路由,`lightrag_server.py` 还在应用上直接挂了几个支撑端点。`/health`(第 2196 行)带 `dependencies=[Depends(combined_auth)]`,返回服务状态、关键配置、流水线标志、各队列状态、版本号,以及 `server_mode`(gunicorn 或 uvicorn)。它把「这个实例当前忙不忙、跑在什么模式」一次性吐给运维和前端,是排障与监控的主入口;前面提到 `/health` 被列入续期跳过清单,正是因为它会被高频探活。

前端则通过 `SmartStaticFiles`(继承自 Starlette 的 `StaticFiles`)挂在固定的 `/webui` 路径下。它做了两件常规静态服务器不做的事:其一,在返回 `index.html` 时,把占位符 `<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->` 替换为一段注入了 `window.__LIGHTRAG_CONFIG__` 的脚本,使前端能在运行时拿到后端的 API 前缀等配置,而不必在构建期写死;其二,对 HTML 设置 `no-cache`、对 `/assets/` 下的带哈希文件名资源设置不可变长缓存,兼顾「页面永远拿最新」与「静态资源最大化缓存」。

多租户隔离靠 `get_workspace_from_request(request)`:它读取请求头 `LIGHTRAG-WORKSPACE`,把其中非字母数字字符替换为 `_` 做净化,得到一个安全的工作区名。把净化逻辑放在请求入口处,意味着无论客户端传入什么奇怪字符,落到目录名、命名空间键上的都是受控字符串,从源头杜绝了路径穿越一类风险。结合前面 `DocumentManager` 按 workspace 隔离输入目录、`pipeline_status` 按 `rag.workspace` 取命名空间数据,整套服务实现了「同一进程内多工作区互不干扰」——文档、状态、图谱都按工作区分桶。

## 多 worker 部署:Gunicorn 与 Uvicorn 的分工

`main()` 是 CLI 入口。它先做 Windows 的 `SelectorEventLoop` 修复,`initialize_config()`,然后判断:若 `"GUNICORN_CMD_ARGS" in os.environ`(说明是被 Gunicorn 拉起的子进程),直接 return,把控制权交回 Gunicorn;否则走 Uvicorn 路径,依次执行:

- `check_env_file()`:校验启动目录下的 `.env`(经 `validate_runtime_target_from_env_file`),多实例部署强烈依赖每实例独立的 `.env`。
- `check_and_install_dependencies()`:用 pipmaster 按当前绑定按需安装缺失的可选依赖。
- `freeze_support()`:multiprocessing 在打包/Windows 下的必要调用。
- `configure_logging()`:挂 `RotatingFileHandler` 与 `LightragPathFilter`。
- `update_uvicorn_mode_config()`:强制单 worker。
- `display_splash_screen()`:打印配置横幅。
- `create_app(global_args)` 与 `uvicorn.run(...)`:装配并启动。

注意 `uvicorn.run` **不传** `root_path`——反向代理前缀完全由应用内的 `root_path`/中间件处理。`check_and_install_dependencies` 这一步尤其体现了 LightRAG「核心精简、按需扩展」的取舍:基础包不强制依赖全部 provider SDK,启动时根据实际选用的绑定再去装对应库,避免一个轻量部署被几十个用不到的 SDK 拖累。

两种模式的本质差异在第 8 章讲过的共享存储:Uvicorn 单进程内,所有「共享数据」其实就是进程内的全局对象,lifespan 退出时直接 `finalize_share_data()` 即可。Gunicorn 多 worker 则通过 `LIGHTRAG_GUNICORN_MODE` 环境变量让各 worker 的 lifespan **跳过** `finalize_share_data()`,把跨进程共享内存的销毁集中到 master 的 `on_exit` 钩子,确保不会有 worker 提前拆掉别人还在用的共享段。这也是为什么扫描、入队、图谱编辑都要靠 `pipeline_status` 里的并发标志来协调——多个 worker 看到的是同一份共享状态,谁先抢到标志谁执行,其余跳过。

把这条线索串起来看,本章拆解的几乎每一个设计都在为「多 worker 共享一份存储」这个前提服务:`lifespan` 的清理分支决定了共享内存归谁销毁;`config.py` 在 Uvicorn 模式强制单 worker,避免单进程下伪多 worker 撕裂共享态;`/scan`、`/upload`、图谱写操作统一用 `pipeline_status` 标志抢占资格;`workspace` 把不同租户的状态再做一层命名空间隔离。FastAPI/Starlette 只提供了路由与中间件这层「壳」,真正决定 LightRAG 能否横向扩展的,是它如何把这层壳与底层共享存储缝合在一起。理解了这一点,也就理解了为什么 API 层的代码里到处是对 `pipeline_status`、`rag.workspace`、`LIGHTRAG_GUNICORN_MODE` 的引用——它们都是同一套并发模型在网络边界上的投影。

最后值得强调的是部署形态的选择并非随意:需要单机高吞吐、能利用多核时选 Gunicorn 多 worker,但必须接受共享内存清理交给 master、且各 worker 必须看到同一份共享存储后端(否则状态标志失效);追求部署简单、调试方便或运行在受限环境时选 Uvicorn 单进程,代价是无法靠多 worker 扩展并发。LightRAG 没有把这个选择藏起来,而是用 `server_mode` 字段在 `/health` 里如实告知,让运维一眼就能确认当前实例跑在哪种模式下——这种「把关键运行态显式暴露」的克制,正是一个面向工程师的服务该有的样子。

## 本章小结

- LightRAG 的 API 层用 `create_app(args)` 工厂一次性装配应用,把日志、解析器加载、绑定校验、SSL、`DocumentManager`、`lifespan`、中间件、路由全部收敛在一处。
- `lifespan` 在启动时 `initialize_storages` + `check_and_migrate_data`,退出时 `finalize_storages`;并通过 `LIGHTRAG_GUNICORN_MODE` 区分由谁来 `finalize_share_data`,解决多进程共享内存的清理归属。
- 应用用 `root_path` + 自定义 `_RootPathNormalizationMiddleware` 支持反向代理子路径部署;`CORSMiddleware` 显式 `expose_headers=["X-New-Token"]` 以配合令牌自动续期。
- `config.py` 用 argparse 默认值取自 `get_env_value`,实现「命令行 > 环境变量 > 默认」三层层叠;`global_args` 是懒加载代理 `_GlobalArgsProxy`,首次访问才解析配置。
- 鉴权由 `get_combined_auth_dependency` 统一为一个依赖:白名单放行 → token 优先校验(含跳过轮询路径、限流的滑动窗口续期) → 无保护放行 → API Key 校验。
- 文档路由用「工厂内新建 APIRouter」避免重复 operation ID;`/scan`、`/upload` 通过 `pipeline_status` 的 `busy`/`scanning`/`pending_enqueues` 标志与 `BackgroundTasks` 协同,杜绝重复入队。
- 检索路由的 `QueryRequest.to_query_params` 通过 `model_dump` 字段对齐直接构造 `QueryParam`;`/query`、`/query/stream`(NDJSON)、`/query/data` 分别服务整体响应、逐字流式与结构化结果。
- 图谱路由的所有写操作前置 `check_pipeline_busy_or_raise`,与文档流水线互斥,防止对图存储的脏写。
- Ollama 兼容层把 RAG 伪装成 `gguf` 模型,用查询前缀(`/local`、`/context`、`/bypass` 等)选择检索模式,并对会话元数据任务绕过 RAG 直连 LLM。
- `main()` 检测 `GUNICORN_CMD_ARGS` 决定走 Uvicorn 还是把控制权交回 Gunicorn,Uvicorn 模式被强制单 worker。
- WebUI 经 `SmartStaticFiles` 注入运行时配置并区分缓存策略,`/health` 暴露 `server_mode` 与各队列状态,`get_workspace_from_request` 按 `LIGHTRAG-WORKSPACE` 头实现工作区隔离。

## 动手实验

1. **实验一:追踪一次 `/scan` 的并发保护** — 阅读 `document_routes.py` 第 2440 行起的 `/scan` 实现,画出它读取 `pipeline_status` 后的判定流程图(`busy`/`scanning`/`pending_enqueues` 任一为真即返回 `scanning_skipped_pipeline_busy`)。然后启动服务,快速连续调用两次 `POST /documents/scan`,确认第二次返回的 `status` 字段证明了「重复扫描被跳过」。

2. **实验二:验证 `QueryRequest` 到 `QueryParam` 的字段映射** — 在 `query_routes.py` 找到 `to_query_params`,列出它 `exclude` 掉的字段(`query`、`include_chunk_content`)并解释原因;再用 `curl` 向 `/query` 发一个带 `mode`、`top_k`、`only_need_context` 的请求,然后向 `/query/stream` 发同样请求,对比前者返回完整 JSON、后者返回逐行 NDJSON(注意响应头里的 `Content-Type: application/x-ndjson` 与 `X-Accel-Buffering: no`)。

3. **实验三:观察令牌自动续期** — 在 `.env` 配置 `AUTH_ACCOUNTS` 与一个自定义 `TOKEN_SECRET`(否则 `validate_auth_configuration` 会拒绝启动),登录拿到 token。把 `utils_api.py` 里的 `token_renew_threshold` 调高(如 0.99)使续期几乎立即触发,然后带 token 访问一个非轮询路径(如 `/query`),检查响应头是否出现 `X-New-Token`;再连续请求,验证 60 秒 `_RENEWAL_MIN_INTERVAL` 限流确实抑制了重复续期。

4. **实验四:用 Ollama 前缀切换检索模式** — 阅读 `ollama_api.py` 的 `parse_query_mode`,把它支持的所有前缀(`/local`、`/global`、`/hybrid`、`/mix`、`/naive`、`/bypass`、`/context` 系列、方括号 `user_prompt` 语法)整理成一张映射表。然后通过 `POST /api/chat` 分别发送 `/local 你的问题` 与 `/bypass 你的问题`,观察前者走 `rag.aquery`、后者绕过 RAG 直连底层 LLM 的行为差异。

> **下一章预告**:第 12 章《运维工具与缓存迁移》将走进 `tools` 目录,拆解 LightRAG 提供的缓存迁移、向量库重建等运维脚本——看看当存储格式升级、embedding 模型更换、或缓存需要清理重算时,这些离线工具如何安全地搬运和重建数据,与本章在线服务形成「运行时」与「运维期」的完整闭环。
