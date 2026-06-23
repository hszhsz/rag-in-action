# 第 12 章 MCP 接入

在前面的章节里，我们把 RAGFlow 拆成了一条从文档解析、分块、向量化到检索重排的流水线，又看到 Agent 画布如何在一个工具循环里调度内置工具与大模型。但所有这些能力，到目前为止都被锁在 RAGFlow 自己的 HTTP API 和前端画布里。一个外部的 agent、一个 IDE 插件、或者别人家的编排框架，想直接调用 RAGFlow 的检索能力，就得自己去读 RAGFlow 的 REST 文档、手写鉴权、手写分页拼装。这正是 Model Context Protocol（MCP）想要消除的摩擦：它定义了一套标准化的、面向工具调用的客户端-服务端协议，让任何兼容 MCP 的宿主都能用同一套握手、列工具、调工具的流程，去消费任意 MCP server 暴露的能力 [[Model Context Protocol]](https://modelcontextprotocol.io)。

RAGFlow 在这件事上扮演了两个角色。一方面它是 **MCP server**：把知识库检索封装成一个标准 MCP 工具 `ragflow_retrieval`，让外部世界以协议方式调用；这部分代码集中在 `mcp/server/server.py`。另一方面它又是 **MCP client**：在 Agent 组件里，它能连上别人部署的 MCP server，把对方的工具拉进自己的工具循环，和内置工具一视同仁地交给大模型挑选；这部分逻辑在 `common/mcp_tool_call_conn.py` 和 `agent/component/agent_with_tools.py`。本章就沿着"对外暴露"与"对内消费"两条线，把这套标准化工具协议的接入方式逐层解构。

## 一个工具、一个进程：server 端的整体形态

打开 `mcp/server/server.py`，第一个值得注意的事实是：RAGFlow 的 MCP server 是一个**独立进程**，而不是嵌在主 API 服务里的一个路由。它在文件末尾用 `click` 定义了一个完整的命令行入口 `main()`，文件顶部的注释甚至给出了六种启动示例，第一种就是 `uv run mcp/server/server.py --host=127.0.0.1 --port=9382 --base-url=http://127.0.0.1:9380 --mode=self-host --api-key=ragflow-xxxxx`。这条命令揭示了它的基本工作方式：MCP server 监听 `9382` 端口，而它要服务的请求最终会被转发到 `--base-url` 指向的 RAGFlow 主后端 `9380`。

也就是说，MCP server 本身不直接碰向量库或数据库，它是一个**协议适配层**——把 MCP 的 `list_tools`/`call_tool` 语义翻译成对 RAGFlow REST API 的 HTTP 调用。承担这个翻译职责的是 `RAGFlowConnector` 类，它在构造时就把 base URL 拼成了 API 地址：

```python
def __init__(self, base_url: str, version="v1"):
    self.base_url = base_url
    self.version = version
    self.api_url = f"{self.base_url}/api/{self.version}"
    self._async_client = None
```

所有实际的网络调用都走它的两个私有方法 `_post` 和 `_get`，而这两个方法都有一个关键的前置判断：

```python
async def _post(self, path, json=None, stream=False, files=None, api_key: str = ""):
    if not api_key:
        return None
    client = await self._get_client()
    res = await client.post(url=self.api_url + path, json=json, headers={"Authorization": f"Bearer {api_key}"})
    return res
```

没有 `api_key` 就直接返回 `None`，不会发出任何请求。这意味着鉴权不是某个中间件的额外职责，而是被下沉成了 connector 的硬约束：每一次对 RAGFlow 后端的访问都必须带上 `Bearer <api_key>`，而这个 key 正是 RAGFlow 用户自己的 API key。MCP server 自己不持有任何越权能力，它只是个透传者，外部调用者能看到什么数据，完全取决于它提供的 key 在 RAGFlow 里有什么权限。这种设计把"多租户隔离"这个最容易出错的安全问题，直接交还给了 RAGFlow 既有的鉴权体系。

MCP 协议的服务端骨架则来自官方 SDK 的低层 `Server` 类：

```python
from mcp.server.lowlevel import Server
...
app = Server("ragflow-mcp-server", lifespan=sse_lifespan)
```

`Server` 提供了 `@app.list_tools()` 和 `@app.call_tool()` 两个装饰器，分别注册"工具发现"和"工具调用"的处理函数。RAGFlow 要做的，就是在这两个钩子里把检索能力填进去。

## 把检索封装成一个 MCP 工具

server 端只暴露了**一个**工具：`ragflow_retrieval`。这个选择本身就是一种取舍——RAGFlow 没有把"列数据集""列文档""上传"等一堆 API 都摊开成工具，而是只挑了对外部 agent 最有价值的那个动作：检索。`list_tools` 的实现把数据集列表动态拼进了工具描述里：

```python
@app.list_tools()
@with_api_key(required=True)
async def list_tools(*, connector: RAGFlowConnector, api_key: str) -> list[types.Tool]:
    dataset_description = await connector.list_datasets(api_key=api_key)

    return [
        types.Tool(
            name="ragflow_retrieval",
            description="Retrieve relevant chunks from the RAGFlow retrieve interface based on the question. ... Below is the list of all available datasets, including their descriptions and IDs:"
            + dataset_description,
            inputSchema={ ... },
        ),
    ]
```

这里有个很巧妙的细节：工具的 `description` 不是静态字符串，而是在每次 `list_tools` 时实时调用 `connector.list_datasets(api_key=api_key)`，把当前这个 key 能访问到的所有数据集（描述 + ID）逐行拼进描述文本。这样一来，调用方的大模型在看到这个工具时，就顺带看到了"有哪些知识库可选、各自是干什么的"，从而能自己决定要不要在 `dataset_ids` 里填具体的库。`list_datasets` 把结果格式化成 newline-delimited JSON：

```python
result_list = []
for data in datasets:
    d = {"description": data["description"], "id": data["id"]}
    result_list.append(json.dumps(d, ensure_ascii=False))
return "\n".join(result_list)
```

工具的入参 schema 则相当丰富，它把 RAGFlow 检索 API 的几乎所有可调参数都暴露成了标准 JSON Schema：`question`（唯一必填项，`"required": ["question"]`）、可选的 `dataset_ids` 和 `document_ids`、分页的 `page`/`page_size`、相似度门槛 `similarity_threshold`（默认 `0.2`）、向量权重 `vector_similarity_weight`（默认 `0.3`）、`keyword`、`top_k`（默认 `1024`）以及 `rerank_id`。每个字段都带了 `description`、`default`、`minimum`/`maximum`，比如 `page_size` 限定 `"maximum": 100`，`top_k` 限定 `"maximum": 1024`。这些约束不是摆设——它们会被原样传给调用方的大模型，作为它生成工具调用参数的依据，是 MCP "工具自描述"价值的直接体现。

`call_tool` 的逻辑则朴素得多：它按工具名分发，对 `ragflow_retrieval` 把 arguments 里的字段逐个取出（带上和 schema 一致的默认值，注意这里 `page_size` 默认取 `10`），转交给 connector：

```python
@app.call_tool()
@with_api_key(required=True)
async def call_tool(name: str, arguments: dict, *, connector: RAGFlowConnector, api_key: str):
    if name == "ragflow_retrieval":
        ...
        return await connector.retrieval(api_key=api_key, dataset_ids=dataset_ids, ...)
    raise ValueError(f"Tool not found: {name}")
```

真正干活的是 `connector.retrieval`，它把参数拼成 `data_json` 后 `POST` 到后端的 `/retrieval` 接口。这里有一个体现"协议层补足"思路的设计：如果调用方没给 `dataset_ids`，server 不会报错，而是先调 `resolve_dataset_ids` 把该 key 能访问的所有数据集 ID 都查出来，再去检索——也就是 schema 描述里承诺的"omit dataset_ids entirely to search across ALL available datasets"。如果连一个可访问的数据集都没有，才抛出 `No accessible datasets found.`。

## 检索结果的"二次加工"与元数据缓存

`retrieval` 拿到后端返回的 chunks 后，并没有原样吐回去，而是做了一层富化。它先调用 `_get_document_metadata_cache` 把涉及到的数据集和文档的元数据准备好，再对每个 chunk 调用 `_map_chunk_fields`：

```python
def _map_chunk_fields(self, chunk_data, dataset_cache, document_cache):
    mapped = dict(chunk_data)
    dataset_id = chunk_data.get("dataset_id") or chunk_data.get("kb_id")
    if dataset_id and dataset_id in dataset_cache:
        mapped["dataset_name"] = dataset_cache[dataset_id]["name"]
    else:
        mapped["dataset_name"] = "Unknown"
    mapped["document_name"] = chunk_data.get("document_keyword", "")
    document_id = chunk_data.get("document_id")
    if document_id and document_id in document_cache:
        mapped["document_metadata"] = document_cache[document_id]
    return mapped
```

它在保留原始字段的基础上，给每个 chunk 补上了人类可读的 `dataset_name`、`document_name`，以及完整的 `document_metadata`（含文件名、类型、大小、chunk 数、创建/更新时间等）。这背后的意图很清楚：MCP 工具的消费者通常是大模型，它需要的是"自包含、可解释"的结果，而不是一堆只有内部系统才懂的 ID。最终 `retrieval` 把所有 chunk 连同 `pagination` 和 `query_info` 打包成一个结构化对象，序列化为一段文本，包进 `types.TextContent` 返回。

为了避免每次检索都重复查数据集和文档元数据，connector 内置了两层 LRU 缓存：`_dataset_metadata_cache` 和 `_document_metadata_cache`，都用 `OrderedDict` 实现，配合 `move_to_end` 和 `popitem(last=False)` 维护 LRU 顺序。缓存上限是 `_MAX_DATASET_CACHE = 32`，TTL 是 `_CACHE_TTL = 300` 秒。一个值得玩味的细节是过期时间的计算：

```python
def _get_expiry_timestamp(self):
    offset = random.randint(-30, 30)
    return time.time() + self._CACHE_TTL + offset
```

它给每个缓存项的过期时间加了 `[-30, 30]` 秒的随机抖动，这是典型的缓存防雪崩手法——避免大批缓存在同一秒集中失效、瞬间打穿后端。工具入参里那个 `force_refresh` 字段，就是给调用方一个手动绕过缓存的开关。

## 两种传输与两种启动模式

MCP 协议本身不规定传输层，server 可以跑在 stdio、SSE 或 streamable-http 上。RAGFlow 同时支持后两种 HTTP 传输，并用两个枚举把它们固化下来：

```python
class Transport(StrEnum):
    SSE = "sse"
    STEAMABLE_HTTP = "streamable-http"
```

`create_starlette_app()` 根据 `TRANSPORT_SSE_ENABLED` 和 `TRANSPORT_STREAMABLE_HTTP_ENABLED` 两个开关，分别挂载不同的路由。SSE 模式（一种较早期的传输方式，注释里直接称之为 legacy）挂了两条路由：`GET /sse` 用来建立事件流，`Mount /messages/` 用来接收客户端回发的消息——这是 SSE 双向通信的经典拆法，因为 SSE 本身只能服务端单向推。streamable-http 模式则统一挂在 `/mcp` 上，用官方的 `StreamableHTTPSessionManager` 处理 `GET`/`POST`/`DELETE`：

```python
session_manager = StreamableHTTPSessionManager(
    app=app,
    event_store=None,
    json_response=JSON_RESPONSE,
    stateless=True,
)
```

注意 `stateless=True`，这是 streamable-http 相对 SSE 的一大优势：它不要求服务端为每个客户端维持长连接会话状态，每个请求可以独立路由，更适合放在负载均衡后面横向扩展。`json_response` 开关则决定响应是纯 JSON 还是 SSE-over-HTTP 的事件流，`main()` 里还有一处兜底逻辑——若 streamable-http 被禁用但 JSON 模式仍开着，会把 `JSON_RESPONSE` 强制置回 `False`，避免出现"开了一个用不上的开关"的矛盾配置。

另一条正交的维度是**启动模式**，由 `LaunchMode` 枚举的 `self-host` 和 `host` 区分。这两者的区别全在鉴权上，集中体现在 `with_api_key` 装饰器里：

```python
def with_api_key(required: bool = True):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ...
            api_key = HOST_API_KEY
            if MODE == LaunchMode.HOST:
                api_key = _extract_token_from_request(getattr(ctx, "request", None)) or ""
                if required and not api_key:
                    raise ValueError("RAGFlow API key or Bearer token is required.")
            return await func(*args, connector=connector, api_key=api_key, **kwargs)
        return wrapper
    return decorator
```

`self-host` 模式是单租户的：整个进程绑定一个全局 `HOST_API_KEY`（启动时通过 `--api-key` 注入，`main()` 里有硬校验 `--api-key is required when --mode is 'self-host'`），所有调用都用这同一个 key。`host` 模式则是多租户的：进程不持有任何 key，而是从**每个请求**的 header 里抽取调用者自己的 token。抽取逻辑 `_extract_token_from_headers` 写得相当宽容，既认 `Authorization: Bearer xxx`，也认 `api_key`、`x-api-key`、`Api-Key`、`X-API-Key` 等多种 header 名（连大小写和 bytes 形式都覆盖了），这正好对应了 client 端示例里两种合法的鉴权写法。`host` 模式下还会额外挂一层 `AuthMiddleware`，对 `/messages/`、`/sse`、`/mcp` 路径做前置拦截，没有 token 直接返回 `401`，把校验提前到业务逻辑之前。

这套配置在生产部署里通过 `docker/entrypoint.sh` 落地。脚本里一组 `MCP_*` 变量定义了默认值（`MCP_PORT=9382`、`MCP_MODE="self-host"`、`MCP_SCRIPT_PATH="/ragflow/mcp/server/server.py"` 等），并用三个标志变量 `MCP_TRANSPORT_SSE_FLAG`、`MCP_TRANSPORT_STREAMABLE_HTTP_FLAG`、`MCP_JSON_RESPONSE_FLAG` 控制传输开关。关键在于 MCP server **默认不启动**——`ENABLE_MCP_SERVER=0`，只有命令行传入 `--enable-mcpserver` 才会把它置 `1`，进而在脚本末尾触发 `start_mcp_server`，用上面这些参数拼出和文件头注释一致的启动命令。

## client 端：最小可用的连接样例

理解了 server 端，再看 client 端就很轻松，因为 MCP 的对称性正是它的卖点。`mcp/client/client.py` 是一个针对 SSE 传输的最小样例，整个核心只有几行：

```python
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

async with sse_client("http://localhost:9382/sse") as streams:
    async with ClientSession(streams[0], streams[1]) as session:
        await session.initialize()
        tools = await session.list_tools()
        response = await session.call_tool(name="ragflow_retrieval", arguments={"dataset_ids": [...], "document_ids": [], "question": "How to install neovim?"})
```

`sse_client` 建立到 `/sse` 的连接，返回读写两条流；`ClientSession` 包住这对流，`initialize()` 完成 MCP 握手，随后就能 `list_tools()` 发现工具、`call_tool()` 调用工具。`streamable_http_client.py` 是同一套流程的 streamable-http 版本，只是换成 `streamablehttp_client("http://localhost:9382/mcp/")`，它返回的是 `(read_stream, write_stream, _)` 三元组（第三个是会话 ID 获取器，样例里用不到）。两个文件的注释都示范了 `host` 模式下如何带 token：要么 `headers={"api_key": "ragflow-..."}`，要么 `headers={"Authorization": "Bearer ragflow-..."}`，正好和 server 端 `_extract_token_from_headers` 认的那几种 header 对上。

这两个样例本身只是演示，真正把"消费外部 MCP 工具"做成产品能力的，是下面这套会话管理。

## 把外部 MCP 工具接进 Agent 工具循环

第 9 章讲 Agent 画布时，我们看到 `AgentWithTools` 在初始化阶段会把内置工具的元数据收集成 `tool_meta`，再 `bind_tools` 给大模型。MCP 工具就是在同一个地方"插队"进来的。`agent/component/agent_with_tools.py` 在收集完内置工具后，遍历配置里的每个 MCP server：

```python
tool_idx = len(self.tools)
for mcp in self._param.mcp:
    _, mcp_server = MCPServerService.get_by_id(mcp["mcp_id"])
    custom_header = self._param.custom_header
    tool_call_session = MCPToolCallSession(mcp_server, mcp_server.variables, custom_header)
    for tnm, meta in mcp["tools"].items():
        indexed_name = f"{tnm}_{tool_idx}"
        tool_idx += 1
        self.tool_meta.append(mcp_tool_metadata_to_openai_tool(meta, function_name=indexed_name))
        self.tools[indexed_name] = MCPToolBinding(tool_call_session, tnm)
```

这里有三件事值得展开。第一，每个 MCP server 会创建一个 `MCPToolCallSession`，它就是 RAGFlow 作为 client 维持的一条到外部 server 的会话。第二，外部工具被起了一个带序号的内部名 `f"{tnm}_{tool_idx}"`，再用 `mcp_tool_metadata_to_openai_tool` 把 MCP 的工具元数据翻译成 OpenAI function-calling 的格式塞进 `tool_meta`。这个翻译函数是两套协议互通的胶水：

```python
def mcp_tool_metadata_to_openai_tool(mcp_tool, function_name=None):
    ...
    return {
        "type": "function",
        "function": {
            "name": function_name or mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
        },
    }
```

MCP 工具的 `inputSchema` 直接被当作 OpenAI 工具的 `parameters`——因为两者都基于 JSON Schema，这一步几乎是零成本的字段重命名。第三，真正的执行入口是 `MCPToolBinding`，它是个 frozen dataclass，只记两样东西：`session`（哪条会话）和 `original_name`（在对方 server 上的真实工具名）。这样大模型用内部序号名 `xxx_5` 调用时，框架能找回它对应的 session 和原始名，再发给外部 server。从这一刻起，MCP 工具和内置工具在工具循环里**完全等价**，大模型并不知道某个工具其实是远程的。

## 会话的隔离、线程模型与超时

`MCPToolCallSession` 是 client 端最有工程含量的一块，它要解决一个棘手的问题：MCP 的 client SDK 是 asyncio 异步的，但 RAGFlow 的 Agent 工具循环是同步调用 `tool_call(...)` 的。把异步会话嫁接到同步调用上，RAGFlow 的做法是**给每个会话起一条专属线程跑一个独立的事件循环**：

```python
def __init__(self, mcp_server, server_variables=None, custom_header=None):
    self.__class__._ALL_INSTANCES.add(self)
    ...
    self._queue = asyncio.Queue()
    self._close = False
    self._event_loop = asyncio.new_event_loop()
    self._thread_pool = ThreadPoolExecutor(max_workers=1)
    self._thread_pool.submit(self._event_loop.run_forever)
    asyncio.run_coroutine_threadsafe(self._mcp_server_loop(), self._event_loop)
```

构造时它就开了一个单线程的 `ThreadPoolExecutor`，让事件循环在那条线程里 `run_forever`，然后把 `_mcp_server_loop` 这个长驻协程丢进去。`_mcp_server_loop` 根据 `server_type` 选 `sse_client` 还是 `streamablehttp_client` 建连，握手时还套了 `asyncio.wait_for(..., timeout=5)`，超时就把错误信息记下来。建连成功后它进入 `_process_mcp_tasks` 循环，不停从 `self._queue` 里取任务执行。

同步侧的 `tool_call` 和 `get_tools` 则通过 `run_coroutine_threadsafe` 把协程投递到那条事件循环线程，再用 `future.result(timeout=...)` 同步等结果：

```python
@override
def tool_call(self, name, arguments, timeout=10):
    if self._close:
        return "Error: Session is closed"
    future = asyncio.run_coroutine_threadsafe(
        self._call_mcp_tool(name, arguments, request_timeout=timeout),
        self._event_loop,
    )
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        ...
        return f"Timeout calling tool '{name}' (timeout={timeout})."
    except Exception as e:
        return f"Error calling tool '{name}': {e}."
```

这条"同步调用 → 投递到专属事件循环 → 等 future"的链路里，超时被层层兜底：握手 5 秒、任务从队列取阻塞 1 秒轮询 `self._close`、`tool_call`/`get_tools` 默认 10 秒、`_call_mcp_server` 内部 8 秒。任何一层超时都不会让整个 Agent 卡死，而是返回一段可读的错误文本。这种"宁可返回错误也绝不挂起"的姿态，对一个要把外部不可控服务接进自己工具循环的系统来说至关重要——你永远不知道对面的 MCP server 会不会卡住。

工具调用结果还会经过 `_call_mcp_tool` 的归一化：如果 `result.isError` 就返回 `MCP server error: ...`，内容为空返回提示语，只有第一个内容块是 `TextContent` 时才取出 `.text`，其余类型一律降级为 `Unsupported content type`。这和 server 端只用 `TextContent` 回结果是呼应的——当前这套接入聚焦在文本类工具上。

请求参数里的 header 还做了一次模板替换，`_mcp_server_loop` 用 `string.Template(...).safe_substitute(self._server_variables)` 把配置里写成 `${VAR}` 的 header 占位符填上真实值，这样数据库里存的 MCP server 配置就能把敏感的 token 用变量引用，而不是硬编码明文。

## 优雅关闭：呼应第 1 章的 shutdown

第 1 章讲进程入口时提到过 `shutdown_all_mcp_sessions`，现在我们能把它补完整。`MCPToolCallSession` 用一个类级别的 `weakref.WeakSet` 登记了所有活着的实例：

```python
class MCPToolCallSession(ToolCallSession):
    _ALL_INSTANCES: weakref.WeakSet["MCPToolCallSession"] = weakref.WeakSet()
```

用弱引用集合而非普通集合，是为了不阻止那些已经没人用的会话被 GC 回收——它只是个"还活着的会话"的旁观登记簿，不影响对象生命周期。`shutdown_all_mcp_sessions()` 进程级钩子会把这个集合快照成列表，交给 `close_multiple_mcp_toolcall_sessions` 批量关闭：

```python
def shutdown_all_mcp_sessions():
    sessions = list(MCPToolCallSession._ALL_INSTANCES)
    if not sessions:
        logging.info("No MCPToolCallSession instances to close.")
        return
    logging.info(f"Shutting down {len(sessions)} MCPToolCallSession instances...")
    close_multiple_mcp_toolcall_sessions(sessions)
```

批量关闭函数自己另起一条临时线程跑一个新事件循环，用 `asyncio.gather(*[s.close() ...], return_exceptions=True)` 并发关掉所有会话，`return_exceptions=True` 保证一个会话关闭失败不会拖垮其他会话的清理。单个 `close()` 则负责把 `_close` 置 `True`、给队列里残留的任务塞 `CancelledError`、停掉事件循环、`shutdown(wait=True)` 等线程退出，最后把自己从 `_ALL_INSTANCES` 里摘掉。

这个钩子的调用点在 `api/ragflow_server.py` 的 `signal_handler` 里：

```python
def signal_handler(sig, frame):
    logging.info("Received interrupt signal, shutting down...")
    shutdown_all_mcp_sessions()
    stop_event.set()
    stop_event.wait(1)
    sys.exit(0)
```

进程收到中断信号时，第一件事就是收掉所有 MCP 客户端会话，再退出。这是一个容易被忽略但很重要的细节：每条会话背后都拖着一条常驻线程和一个事件循环，如果进程直接退出而不主动关闭，这些线程和它们持有的网络连接就成了悬挂资源。把 MCP 会话纳入信号处理的优雅关闭路径，正是第 1 章强调的"进程生命周期要对它创建的所有后台资源负责"这一原则在 MCP 接入上的具体落实。

## 本章小结

- RAGFlow 通过 `mcp/server/server.py` 把自身能力暴露为 MCP server，它是一个独立进程（默认端口 `9382`），本质是把 MCP 的 `list_tools`/`call_tool` 翻译成对 RAGFlow 后端 REST API 的 HTTP 调用，翻译职责由 `RAGFlowConnector` 承担。
- server 只对外暴露一个工具 `ragflow_retrieval`，其工具描述在每次 `list_tools` 时动态拼入当前 key 可访问的数据集列表，`inputSchema` 把检索 API 的全部可调参数连同默认值、上下界都暴露成标准 JSON Schema，唯一必填项是 `question`。
- 鉴权被下沉到 connector 的 `_post`/`_get`：无 `api_key` 直接拒发请求，每次调用都带 `Bearer` token，多租户隔离完全复用 RAGFlow 既有的 key 权限体系。
- 检索结果经 `_map_chunk_fields` 富化（补 `dataset_name`、`document_name`、`document_metadata`），并由 `OrderedDict` 实现的两层 LRU 缓存（上限 32、TTL 300 秒、带 `±30` 秒随机抖动防雪崩）支撑元数据查询。
- 传输层支持 SSE（legacy，`/sse` + `/messages/`）和 streamable-http（`/mcp`，`stateless=True` 利于横向扩展）两种；启动模式分 `self-host`（单租户全局 key）和 `host`（多租户、逐请求从 header 取 token，并加 `AuthMiddleware` 做 401 前置拦截）。
- MCP server 在 `docker/entrypoint.sh` 中默认关闭（`ENABLE_MCP_SERVER=0`），需 `--enable-mcpserver` 显式开启，相关 `MCP_*` 变量控制端口、模式与传输开关。
- 作为 client，`mcp/client/` 下两个最小样例演示了 `sse_client`/`streamablehttp_client` + `ClientSession` 的 `initialize → list_tools → call_tool` 标准流程，并示范了 `host` 模式下 `api_key`/`Authorization` 两种带 token 写法。
- `agent/component/agent_with_tools.py` 把外部 MCP 工具接入 Agent：每个 server 建一个 `MCPToolCallSession`，用 `mcp_tool_metadata_to_openai_tool` 将 MCP 的 `inputSchema` 转成 OpenAI function-calling 格式，外部工具与内置工具在工具循环里完全等价。
- `MCPToolCallSession` 用"每会话一线程一事件循环"的模型把异步 MCP SDK 嫁接到同步工具调用上，并在握手、队列、tool_call 各层设了多重超时，坚持"宁可返回错误文本也不挂起"。
- 所有会话由类级 `weakref.WeakSet` 登记，`shutdown_all_mcp_sessions()` 在进程 `signal_handler` 中被调用，并发优雅关闭全部会话及其线程，呼应第 1 章的进程生命周期管理原则。

## 动手实验

1. **实验一：本地起一个 MCP server 并用样例 client 调通** — 参照 `mcp/server/server.py` 文件头注释的第 1 条命令，用 `uv run mcp/server/server.py --mode=self-host --api-key=<你的 RAGFlow key>` 在 `9382` 端口起 server；再运行 `mcp/client/client.py`，观察它 `initialize → list_tools → call_tool` 的完整输出，确认返回的 `tools` 里只有 `ragflow_retrieval`，且 chunk 结果里带上了 `dataset_name`/`document_metadata`。

2. **实验二：对比两种传输的连接路径** — 分别用 `--no-transport-streamable-http-enabled` 和 `--no-transport-sse-enabled` 启动 server，各自只保留一种传输；然后跑 `client.py`（连 `/sse`）和 `streamable_http_client.py`（连 `/mcp/`），看哪个能连上、哪个报错。再在 `create_starlette_app()` 里加日志打印实际注册的 `routes`，验证 SSE 模式挂的是 `/sse`+`/messages/`、streamable-http 挂的是 `/mcp`。

3. **实验三：验证 host 模式的鉴权拦截** — 用 `--mode=host` 启动 server，先不带任何 header 跑 client，观察 `AuthMiddleware` 返回 `401`；再分别用 `headers={"api_key": ...}` 和 `headers={"Authorization": "Bearer ..."}` 重试，确认两种写法都能通过 `_extract_token_from_headers`。可以故意给一个无效 key，看 connector 因 REST API 返回非 0 code 而抛出的错误文本。

4. **实验四：追踪一次 client 端工具调用的全链路** — 在 `common/mcp_tool_call_conn.py` 的 `tool_call`、`_call_mcp_tool`、`_process_mcp_tasks` 三处加日志，构造一个配置了外部 MCP server 的 Agent 并触发一次工具调用；观察请求如何从同步 `tool_call` 经 `run_coroutine_threadsafe` 投递到专属事件循环、入 `self._queue`、被 `_process_mcp_tasks` 取出执行、再把结果回传。最后给进程发 `SIGINT`，确认 `signal_handler` 调用 `shutdown_all_mcp_sessions()` 把会话线程干净退出。

> **下一章预告**：走到这里，我们已经把 RAGFlow 从文档解析、分块、向量化、检索重排，到 GraphRAG、RAPTOR、Agent 画布、沙箱、模型接入与 MCP，整条链路逐章拆开。第 13 章《收尾：可迁移的 RAG 工程原则》不再引入新代码，而是把全书观察到的设计抉择提炼成一组跨项目可迁移的工程原则——鉴权下沉与权限复用（第 12 章 connector）、宁错勿挂的超时纪律（第 12 章会话）、进程对后台资源负责的生命周期管理（第 1、12 章 shutdown）、协议适配层与自描述工具（本章）、缓存防雪崩（本章元数据缓存）等等——并逐条回指对应章节的源码证据，让你能把 RAGFlow 的经验带到自己的 RAG 系统里。
