# 第 13 章：API 服务与模型接入

前面十二章我们自底向上拆解了 Chatchat 的知识库、检索、RAG 对话链路、Agent 与工具系统。
但这些能力最终要以服务的形态对外暴露——前端 WebUI、第三方客户端、乃至别的程序，都需要一个稳定的 HTTP 接口去调用它们。
Chatchat 把这层职责放在 `chatchat/server/api_server/` 目录下，用 FastAPI 把对话、知识库、工具、MCP 以及一套 OpenAI 兼容端点统一组织起来。

本章的另一条线索是模型接入。
Chatchat 自己并不托管模型权重，它把所有推理后端（Xinference、Ollama、OneAPI、在线 OpenAI 等）抽象成「平台」，要求每个平台都暴露 OpenAI 兼容的 `/v1` 接口，然后用一份 `MODEL_PLATFORMS` 配置把它们登记进来。
上层代码只认模型名，由 `get_model_info` / `get_ChatOpenAI` / `get_OpenAIClient` 这组工具按模型名反查平台、拼出 base_url 与 api_key，再构造出 langchain 的 `ChatOpenAI` 或裸的 `openai` 客户端。
理解了这套「以 OpenAI 协议为最大公约数」的设计，就理解了 Chatchat 为何能在离线与在线、单机与多后端之间自由切换。

## FastAPI 应用的组装：create_app

整个服务的入口是 `server_app.py` 里的 `create_app`。
它做的事情非常克制：创建一个 `FastAPI(title="Langchain-Chatchat API Server", version=__version__)` 实例，调用 `MakeFastAPIOffline(app)` 让 Swagger/Redoc 的静态资源本地化（离线环境下文档页也能正常渲染），随后按需挂载中间件与路由。

跨域是可选的。
只有当 `Settings.basic_settings.OPEN_CROSS_DOMAIN` 为真时，才会 `add_middleware(CORSMiddleware, allow_origins=["*"], ...)`，把来源、方法、头部全部放开。
这是一处典型的「默认安全、按配置放开」的取舍：本地单机部署不需要跨域，对外提供服务时再显式打开。

根路径 `GET /` 被定义为一个不进入 schema 的重定向（`include_in_schema=False`），直接 `RedirectResponse(url="/docs")`，让用户打开服务地址就落到交互式文档页。

路由的挂载顺序一目了然：

```python
app.include_router(chat_router)
app.include_router(kb_router)
app.include_router(tool_router)
app.include_router(openai_router)
app.include_router(server_router)
app.include_router(mcp_router)
```

这六个 router 各自带前缀与 tags，构成了 Chatchat 的功能版图：

- `chat_router`（`/chat`）是对话；
- `kb_router`（`/knowledge_base`）是知识库管理；
- `tool_router` 是工具；
- `openai_router`（`/v1`）是 OpenAI 兼容层；
- `server_router` 是服务自身的元信息；
- `mcp_router`（`/api/v1/mcp_connections`）管理 MCP 连接。

除此之外还单独挂了一个 `POST /other/completion`（`tags=["Other"]`），以及两个静态目录：`/media` 映射到 `MEDIA_PATH`、`/img` 映射到 `IMG_DIR`，用于回传生成的媒体与项目图片。

`run_api` 是启动函数：若传入了 `ssl_keyfile` 与 `ssl_certfile` 则以 HTTPS 启动 `uvicorn`，否则普通 HTTP。
模块底部直接 `app = create_app()`，使得 `uvicorn server_app:app` 这类命令可以拿到现成实例；
`__main__` 分支则用 `argparse` 解析 `--host`（默认 `0.0.0.0`）、`--port`（默认 `7861`）等参数后调用 `run_api`。

## 对话路由：把 OpenAI 协议当成内部统一入口

`chat_routes.py` 里的 `chat_router` 用前缀 `/chat`，挂载了反馈（`/feedback`）、知识库对话（`/kb_chat`）、文件对话（`/file_chat`）几个端点。
最值得关注的是 `POST /chat/chat/completions`，它的函数签名直接接收 `OpenAIChatInput`，文档字符串明确写着「请求参数与 `openai.chat.completions.create` 一致，可以通过 `extra_body` 传入额外参数」。

也就是说，Chatchat 没有发明自己的对话请求格式，而是复用了 OpenAI 的 chat completions 结构，把项目特有的参数塞进 `extra_body`。
这个接口内部根据参数组合分流：

- 传了 `tools` 就走 Agent 对话；
- 传了 `tool_choice` 就走指定工具（`extra_body` 中带 `tool_input` 则直接调用、否则经 Agent 调用）；
- 都不传就是普通 LLM 对话。

实现上有几处细节值得点出。
其一，缺省 `max_tokens` 的兜底：`if body.max_tokens in [None, 0]: body.max_tokens = Settings.model_settings.MAX_TOKENS`，让请求方可以不关心这个参数。
其二，`extra` 字段的剥离：`extra = {**body.model_extra} or {}`，随后 `for key in list(extra): delattr(body, key)`，把 pydantic 模型里多余的额外字段摘出来单独处理，避免它们污染下游标准参数。
其三，工具名的归一化：当 `tool_choice` 或 `tools` 里传的是字符串时，用 `get_tool(name)` 查到真实工具后，转换成 OpenAI 的 `{"type": "function", "function": {...}}` 结构，并用 `get_tool_config(name)` 取出工具配置。
最后它把整理好的参数喂给 `chat(...)` 协程，连同 `conversation_id`、`metadata`、`use_mcp` 等一起传入，并在有会话 id 时通过 `add_message_to_db` 落库。

`kb_routes.py` 中也有同样风格的设计。
`POST /knowledge_base/{mode}/{param}/chat/completions` 把检索模式 `mode`（`Literal["local_kb", "temp_kb", "search_engine"]`）和知识库名 `param` 编进路径，请求体仍是 `OpenAIChatInput`，把 `top_k`、`score_threshold`、`prompt_name`、`return_direct` 等从 `body.model_extra` 中取出，转交给第 9 章见过的 `kb_chat`。
除了对话端点，`kb_router` 还以「函数挂载」的方式注册了一整套知识库管理接口：`list_knowledge_bases`、`create_knowledge_base`、`upload_docs`、`recreate_vector_store`、`search_docs` 等，并把 `kb_summary_api` 的摘要接口通过 `kb_router.include_router(summary_router)` 嵌套进来。

## OpenAI 兼容层：让外部用 OpenAI SDK 调 Chatchat

`openai_routes.py` 是本章的重头戏。
它定义了 `openai_router = APIRouter(prefix="/v1", tags=["OpenAI 兼容平台整合接口"])`，把 `/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`、`/v1/images/*`、`/v1/audio/*`（音频接口标注 `deprecated`）、`/v1/files` 等 OpenAI 标准路径悉数实现。
换言之，任何一个能配 base_url 的 OpenAI 客户端，只要把地址指向 Chatchat 的 `/v1`，就能像调 OpenAI 一样调它后面挂着的所有模型。

这一层不是简单转发，而是做了「多平台整合」。
`GET /v1/models` 会遍历 `get_config_platforms()` 返回的每个平台，对每个平台 `get_OpenAIClient(name, is_async=True)` 后 `await client.models.list()`，把各平台的模型聚合到一起，并给每条记录补上 `platform_name` 字段。
这些请求用 `asyncio.create_task` 并发发起、`asyncio.as_completed` 收集，单个平台异常时只记日志并返回空列表，不影响其它平台——这是面向「多个推理后端可能时好时坏」现实的健壮性设计。

请求转发的通道是 `openai_request(method, body, ...)`。
它把请求体 `body.model_dump(exclude_unset=True)` 成 `params`，并处理 `max_tokens == 0` 时回退到 `Settings.model_settings.MAX_TOKENS`。
若 `body.stream` 为真，就返回 `EventSourceResponse(generator())` 做 SSE 流式输出；
`generator` 支持在真正的模型 chunk 前后注入自定义的 `header` / `tail`（字符串会被包装成 `OpenAIChatOutput(content=x, object="chat.completion.chunk")`），并把 `extra_json` 里的键值 `setattr` 到每个 chunk 上。
流被用户中断时捕获 `asyncio.exceptions.CancelledError` 并优雅退出。
非流式则直接 `await method(**params)` 拿结果再 `model_dump`。
`POST /v1/chat/completions` 的实现就是在 `get_model_client(body.model)` 上下文里调 `client.chat.completions.create`。

### 重名模型的并发调度

`get_model_client` 是一个 `@asynccontextmanager`，处理了一个真实场景：同一个模型名可能被多个平台同时提供（比如两台 Xinference 都挂了同一模型）。
它用 `get_model_info(model_name=model_name, multiple=True)` 拿到所有同名模型，按 `(model_name, platform_name)` 维护一组 `asyncio.Semaphore`（每平台容量取自 `api_concurrencies`，缺省 `DEFAULT_API_CONCURRENCIES = 5`），调度策略是「优先选空闲的平台，否则选当前访问数最少的」，从而把负载摊到多个后端。
选定后 `await semaphore.acquire()`，`yield get_OpenAIClient(platform_name=selected_platform, is_async=True)`，并在 `finally` 中 `semaphore.release()`。
这等于在 Chatchat 自身这一层做了一个轻量的模型路由与限流器。

### 输出结构 OpenAIChatOutput

`api_schemas.py` 里的 `OpenAIBaseOutput`（`OpenAIChatOutput` 直接继承它）定义了流式与非流式两种 OpenAI 兼容输出的拼装逻辑。
它的 `object` 默认是 `"chat.completion.chunk"`，并自带 Chatchat 业务扩展字段：`status`（对应 `AgentStatus`）、`message_type`（默认 `MsgType.TEXT`）、`message_id`、`is_ref`（是否在前端单独的展开区显示，用于引用来源）。
`model_dump` 会按 `object` 的取值分别构造 `choices`：chunk 类型用 `delta`（含 `content` 与 `tool_calls`），completion 类型用 `message`（含 `role`/`content`/`finish_reason`/`tool_calls`）。
`model_dump_json` 固定 `ensure_ascii=False` 以正确输出中文。

输入侧的 `OpenAIChatInput` 继承 `OpenAIBaseInput`，`model` 字段默认值是 `get_default_llm()`，`temperature` 默认取 `Settings.model_settings.TEMPERATURE`，并通过基类的 `Config.extra = "allow"` 允许携带任意额外字段——这正是 `chat_router` 里能从 `model_extra` 读取 `conversation_id`、`use_mcp` 等私有参数的前提。
`extra_json` 还用 `Field(None, alias="extra_body")` 与 OpenAI SDK 的 `extra_body` 对齐。

## 模型接入：MODEL_PLATFORMS 与按模型名解析平台

模型接入的配置中枢是 `settings.py` 里的 `ApiModelSettings.MODEL_PLATFORMS`，它是一个 `List[PlatformConfig]`。
`PlatformConfig` 的关键字段包括：

- `platform_name`（平台别名）；
- `platform_type`（`Literal["xinference", "ollama", "oneapi", "fastchat", "openai", "custom openai"]`）；
- `api_base_url`（默认 `http://127.0.0.1:9997/v1`）；
- `api_key`（默认 `"EMPTY"`）、`api_proxy`、`api_concurrencies`（默认 5）、`auto_detect_model`；
- 按类型分列的模型清单 `llm_models` / `embed_models` / `text2image_models` / `image2text_models` / `rerank_models` / `speech2text_models` / `text2speech_models`，每个都可填具体列表或字面量 `"auto"`。

默认配置里登记了四个平台，恰好覆盖了典型部署形态：

- `xinference`（`auto_detect_model=True`，模型列表留空靠自动探测）；
- `ollama`（`http://127.0.0.1:11434/v1`，手填 `qwen:7b` 等）；
- `oneapi`（`http://127.0.0.1:3000/v1`，把智谱/千问/千帆/星火等在线 API 经 OneAPI 网关统一成 OpenAI 协议）；
- `openai`（`https://api.openai.com/v1`，直连在线模型）。

它们的共同点是 `api_base_url` 都以 `/v1` 结尾——Chatchat 要求每个后端都讲 OpenAI 方言，OneAPI 这类网关存在的意义正是把不讲 OpenAI 方言的厂商 API 翻译成讲。

解析逻辑分几层。
`get_config_platforms()` 把 `MODEL_PLATFORMS` 里的 pydantic 对象逐个 `model_dump()`，并以 `platform_name` 为键组成字典。
`get_config_models(model_name, model_type, platform_name)` 遍历所有平台，对每种模型类型把平台下登记的模型展开成 `{model_name: {platform_name, platform_type, model_type, api_base_url, api_key, api_proxy}}` 的扁平表；
当平台 `auto_detect_model=True` 且 `platform_type == "xinference"` 时，调用带缓存的 `detect_xf_models(xf_url)`（用 `xinference_client` 的 `RESTfulClient.list_models()` 自动拉取模型清单，缓存 60 秒、LRU 容量 10，以避免频繁打到推理服务）。
`get_model_info(model_name, ..., multiple)` 则是对外的便捷封装：`multiple=False` 时只返回第一个匹配，`multiple=True` 时返回全部同名模型——前文 `get_model_client` 的重名调度就靠它。

`get_default_llm()` / `get_default_embedding()` 体现了「配置优先、可用兜底」：
如果 `DEFAULT_LLM_MODEL`（默认 `glm4-chat`）/`DEFAULT_EMBEDDING_MODEL`（默认 `bge-m3`）确实在已配置的可用模型里就用它，否则告警并退而使用列表里的第一个。

## 从模型名到 ChatOpenAI / openai.Client

上层业务拿到的只是一个模型名，真正构造客户端的是两个工厂函数。

`get_ChatOpenAI(model_name, temperature, max_tokens, streaming, callbacks, ...)` 返回 langchain 的 `ChatOpenAI`。
它先 `get_model_info(model_name)` 反查平台信息，组装 `params`（`streaming`/`verbose`/`callbacks`/`model_name`/`temperature`/`max_tokens` 等），并剔除值为 `None` 的键以免触发 openai 的校验错误。
关键在于 base_url 的来源由 `local_wrap` 决定：

- 默认 `local_wrap=False` 时直接用平台的 `api_base_url`、`api_key`、`api_proxy`（即直连后端）；
- `local_wrap=True` 时则改用 Chatchat 自身的 `f"{api_address()}/v1"` 与 `openai_api_key="EMPTY"`，也就是绕回到本章前面那套 `/v1` 兼容层，让请求经过 Chatchat 的整合与调度再出去。

构造失败时记录异常并返回 `None`，由调用方自行决定降级或报错。

这个 `local_wrap` 开关看似不起眼，却体现了 Chatchat 的一处巧思：同一个 `ChatOpenAI` 既可以直连后端，也可以「自调用」本服务的 `/v1`。
当上层希望复用 `get_model_client` 的重名调度与限流，或希望所有出站请求都经过统一的兼容层时，就把 `local_wrap` 打开；
当只需要点对点直连某个平台时，就保持默认。
这让同一套构造逻辑覆盖了「直连」与「经本服务中转」两种拓扑，而调用方无需感知差异。

`get_OpenAIClient(platform_name, model_name, is_async)` 返回的是裸的 `openai.Client` / `openai.AsyncClient`。
它支持两种定位方式：给定 `platform_name` 时直接取平台；只给 `model_name` 时先 `get_model_info` 反查出平台名。
随后用平台的 `api_base_url` 作 `base_url`、`api_key` 作密钥；
若平台配了 `api_proxy`，还会构造带代理的 `httpx.AsyncClient`/`httpx.Client`（并把本地出口地址绑到 `0.0.0.0`）注入进去。
OpenAI 兼容路由全程用的就是这个客户端，因此它们本质上是「FastAPI 收到 OpenAI 请求 → 选平台 → 用 openai 客户端再发一遍 OpenAI 请求」的代理结构。

除了基于 `ChatOpenAI` 的路径，仓库里还有 `langchain_chatchat/chat_models/base.py` 中的 `ChatPlatformAI(BaseChatModel)`——一个自定义 chat model，配套 `get_ChatPlatformAIParams` 生成参数。
它同样以平台信息为基础构造，是 Chatchat 对接平台模型的另一条实现路径，二者共享「按模型名找平台」这一核心抽象。

## 本章小结

- `create_app` 用 FastAPI 组装服务，按需启用 CORS（受 `OPEN_CROSS_DOMAIN` 控制），根路径重定向到 `/docs`，并用 `MakeFastAPIOffline` 让文档页离线可用。
- 服务由六个 router 构成版图：`chat_router`(`/chat`)、`kb_router`(`/knowledge_base`)、`tool_router`、`openai_router`(`/v1`)、`server_router`、`mcp_router`(`/api/v1/mcp_connections`)，另有 `/other/completion` 与 `/media`、`/img` 静态目录。
- 对话与知识库接口都以 `OpenAIChatInput` 为请求体、复用 OpenAI chat completions 结构，把项目私有参数放进 `extra_body`/`model_extra`，并对 `tools`/`tool_choice` 的字符串做工具名归一化。
- `openai_router` 实现了一整套 OpenAI 兼容端点，使外部 OpenAI SDK 把 base_url 指向 `/v1` 即可调用 Chatchat 背后的全部模型。
- `GET /v1/models` 并发聚合所有平台的模型列表并补上 `platform_name`，单平台失败不影响整体。
- `get_model_client` 用按平台维护的 `asyncio.Semaphore` 对同名模型做「先空闲、后最闲」的并发调度与限流，缺省并发 `DEFAULT_API_CONCURRENCIES = 5`。
- `OpenAIChatOutput` 在 OpenAI 输出结构之上扩展了 `status`/`message_type`/`message_id`/`is_ref` 等业务字段，并按 `object` 区分流式 `delta` 与非流式 `message`。
- `MODEL_PLATFORMS` 把 Xinference/Ollama/OneAPI/OpenAI 等后端统一登记为讲 OpenAI 方言（`/v1`）的平台，OneAPI 类网关负责翻译非 OpenAI 厂商 API。
- `get_config_models`/`get_model_info` 把平台配置展开为「模型名 → 平台信息」的扁平表，Xinference 平台支持 `auto_detect_model` 自动探测（带 60 秒缓存）。
- `get_ChatOpenAI` 按模型名反查平台构造 langchain `ChatOpenAI`，并通过 `local_wrap` 在「直连后端」与「绕回本地 `/v1` 兼容层」之间切换；`get_OpenAIClient` 则构造裸 openai 客户端供兼容路由代理使用。

## 动手实验

1. **实验一：用 OpenAI SDK 直连 Chatchat** — 启动 API 服务后，用官方 `openai` Python 库把 `base_url` 设为 `http://127.0.0.1:7861/v1`、`api_key` 设为 `"EMPTY"`，先调 `client.models.list()` 观察聚合了哪些平台的模型（注意返回项里的 `platform_name`），再调 `client.chat.completions.create(model=..., messages=...)` 验证对话；对照 `openai_routes.py` 体会这条「OpenAI 进、OpenAI 出」的代理链路。
2. **实验二：追踪一次 chat/completions 的参数流** — 在 `chat_routes.py` 的 `chat_completions` 里打印 `extra = {**body.model_extra}` 与 `body.tools`，分别用「只发 messages」「带 `tools=["某工具名"]`」「带 `extra_body={"conversation_id": "x"}`」三种请求调用，观察工具名如何被归一化为 function 结构、`extra` 如何被 `delattr` 剥离并转交给 `chat(...)`。
3. **实验三：给 MODEL_PLATFORMS 新增一个平台** — 在 `model_settings.yaml` 的 `MODEL_PLATFORMS` 里添加一个 `platform_type: ollama` 的平台（`api_base_url` 指向你的 Ollama `/v1`，填入真实 `llm_models`），重启后调 `GET /v1/models` 确认新模型出现，再用该模型名调 `/v1/chat/completions`；结合 `get_config_models`/`get_model_info` 理解模型名是如何反查到这个平台的。
4. **实验四：观察重名模型的并发调度** — 在配置里让两个平台提供同一个模型名，把它们的 `api_concurrencies` 设为较小值（如 1），在 `get_model_client` 中打印 `selected_platform` 与各 `semaphore._value`，并发发起多个请求，验证「先选空闲、否则选最闲」的策略以及信号量的 `acquire`/`release` 时机。

> **下一章预告**：解构到这里，Chatchat 从配置、加载、切分、向量化、检索、RAG 对话、Agent、工具、MCP 到 API 与模型接入的全栈拼图已经完整。第 14 章我们将从工程视角收尾，回看 Chatchat 贯穿全书的几条设计主线——以 OpenAI 协议为统一抽象、配置驱动的可插拔、离线优先与渐进式增强，提炼出可迁移到你自己 RAG 系统的工程原则。
