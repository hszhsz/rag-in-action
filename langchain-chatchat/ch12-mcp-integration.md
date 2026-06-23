# 第 12 章 MCP 集成

前面两章我们看到 Chatchat 的 Agent 体系如何在 `agents_registry` 里挑选合适的 agent 类型，又如何用 Tools Factory 把一批本地 Python 函数注册成可调用的工具。这些工具有个共同的前提：它们都活在 Chatchat 自己的进程里，能力边界等于代码仓库的边界。

可是现实中，我们常常希望让模型去操作一台远端的文件系统、查询某个内部数据库、或者调用一个由别的团队维护的服务。如果每接一个外部能力都要在 Tools Factory 里写一份适配代码，工具体系很快就会臃肿且难以维护。

MCP（Model Context Protocol）正是为了解开这个耦合而生：它把"谁提供工具"和"谁使用工具"用一套标准协议隔开。

Chatchat 在 `langchain_chatchat/agent_toolkits/mcp_kit/` 下实现了一个完整的 MCP **客户端**，能够连接外部 MCP server、列出它们暴露的工具、并把这些工具自动适配成 LangChain 的 `BaseTool`，最终汇入第 10、11 章讲过的同一条 agent 工具流水线。本章就来逐行拆解这条"外部能力接入线"，从协议客户端、工具适配、持久化、管理 API 一直追到它与 agent 体系的合流点。

## MCP 是什么：一句话与一组角色

MCP 是 Anthropic 提出的开放协议，用统一的消息格式规范"模型应用如何向外部服务请求上下文、工具与提示词"，让 LLM 应用与外部数据/工具的对接像 USB 一样即插即用 [[Model Context Protocol](https://modelcontextprotocol.io/)]。

协议里有两个核心角色：**MCP server** 负责对外暴露 tools、prompts、resources；**MCP client** 嵌在模型应用内部，负责发现并调用这些能力。Chatchat 扮演的就是 client 这一端，它不实现 server，而是专注于"把别人的 server 接进来"。

从源码可以直接读出 Chatchat 支持的两种传输方式。`mcp_kit/client.py` 顶部定义了两个 `TypedDict`。

`StdioConnection` 描述本地子进程方式，字段包括 `transport`（字面量 `"stdio"`）、`command`（要启动的可执行文件）、`args`（命令行参数列表）、`env`（环境变量字典，可为 `None`）、`encoding` 与 `encoding_error_handler`（后者是 `Literal["strict", "ignore", "replace"]`）。这种方式适合把一个本地脚本或命令行工具直接拉起来作为 MCP server。

`SSEConnection` 描述远端 HTTP 方式，字段包括 `transport`（字面量 `"sse"`）、`url`（SSE 端点地址）、`headers`（可选 HTTP 头）、`timeout`（HTTP 超时）、`sse_read_timeout`（SSE 读超时）。这种方式适合连接一个已经独立部署、对外暴露 SSE 端点的远端 server。

文件里也定下了几个默认常量：`DEFAULT_HTTP_TIMEOUT = 5`、`DEFAULT_SSE_READ_TIMEOUT = 60 * 5`（即 300 秒）、`DEFAULT_ENCODING = "utf-8"`、`DEFAULT_ENCODING_ERROR_HANDLER = "strict"`。这些注释明确标注源自 `langchain-mcp-adapters`，可见 Chatchat 是把社区适配层裁剪进了自己的代码树而非作为黑盒外部依赖——这让它可以按自己的需要修改细节（比如下文要讲的 `PATH` 注入）。底层协议对象（`ClientSession`、`StdioServerParameters`、`sse_client`、`stdio_client`）则直接来自官方的 `mcp` Python SDK。

## MultiServerMCPClient：一个客户端连多台 server

`mcp_kit/client.py` 的主角是 `MultiServerMCPClient`，顾名思义它能同时管理多台 MCP server 的连接。

它的状态由三个字典构成：`self.sessions`（server 名到 `ClientSession` 的映射）、`self.server_name_to_tools`（server 名到已加载工具列表的映射）、以及一个 `AsyncExitStack`（`self.exit_stack`）用于统一管理所有异步资源的生命周期。把 session 和工具都按 server 名分桶存放，是后面能按 server 粒度查询、路由工具调用的基础。

连接的统一入口是 `connect_to_server(server_name, *, transport="stdio", **kwargs)`，它根据 `transport` 分发到两个具体方法，并在分发前做参数校验——SSE 必须有 `url`，stdio 必须有 `command` 和 `args`，缺失就抛 `ValueError`。`transport` 取值若既不是 `"stdio"` 也不是 `"sse"`，同样抛 `ValueError`，把非法配置挡在连接动作之前。

`connect_to_server_via_stdio` 里有个值得注意的工程细节。它会先 `env = env or {}`，然后判断 `if "PATH" not in env`，把宿主进程的 `os.environ.get("PATH", "")` 注入进去。

源码注释解释了原因——像 `uvx`、`npx` 这样的执行命令依赖 `PATH` 才能被定位到，如果用户传的 `env` 里没带 `PATH`，子进程就会启动失败。

准备好 `StdioServerParameters`（封装 `command`、`args`、`env`、`encoding`、`encoding_error_handler`）后，它用 `self.exit_stack.enter_async_context(stdio_client(server_params))` 拿到 `read, write` 双向流，再包一层 `ClientSession`，最后调用 `_initialize_session_and_load_tools`。整个过程都把资源交给 exit_stack 托管，为后面的统一清理埋下伏笔。

`connect_to_server_via_sse` 的逻辑对称得多，直接用 `sse_client(url, headers, timeout, sse_read_timeout)` 建立传输流，同样交给 exit_stack 托管，再走同一个初始化方法。两条路径的差异本质上只在"如何拿到 read/write 流"这一步，之后的 session 初始化与工具加载完全共用。

两条路径都汇聚到 `_initialize_session_and_load_tools(server_name, session)`。

它先 `await session.initialize()` 完成 MCP 握手，把 session 存入 `self.sessions`，然后调用 `load_mcp_tools(server_name, session)` 拉取并适配工具，结果存进 `self.server_name_to_tools`。这一步是协议层（session）与 LangChain 工具层的衔接点——握手成功之后，外部 server 暴露了哪些工具就在此刻被一次性"快照"下来。

`MultiServerMCPClient` 还实现了异步上下文管理器协议。`__aenter__` 遍历构造时传入的 `connections` 字典，逐个 `pop("transport")` 后分发连接；一旦任何环节抛异常，立刻 `await self.exit_stack.aclose()` 回滚已建立的连接，再把异常重新抛出。

`__aexit__` 则只做一件事——`await self.exit_stack.aclose()`，把所有子进程、socket、session 一次性优雅关闭。这种设计把"多台 server、多种资源"的清理责任全部收敛到一个 exit_stack 上，避免了资源泄漏，也让"连接一半失败"这种半成品状态不会留下悬挂的子进程。

对外的查询接口也很清爽。`get_tools()` 把所有 server 的工具拍平成一个大列表返回，遍历 `self.server_name_to_tools.values()` 逐一 `extend`。

`get_tools_from_server(server_name)` 按 server 过滤；`get_tool(server_name, tool_name)` 在指定 server 的工具里按 `tool.name` 精确定位单个工具，找不到返回 `None`。这三个接口分别对应"全量取用""按 server 取用""按名取用"三种粒度，覆盖了 agent 装配工具时的不同需求。

此外还有 `session(server_name)` 用于取回某台 server 的 `ClientSession`（取不到时抛 `ValueError`），以及 `list_prompts` 与 `get_prompt` 两个方法暴露 MCP 的 prompt 能力——后者内部委托给 `mcp_kit/prompts.py` 的 `load_mcp_prompt`。可以看到，这个客户端不仅管"工具"，也把 MCP 协议里的"提示词"维度一并纳入了管理范围。

## MCPStructuredTool：把协议工具翻译成 LangChain 工具

真正完成"契约转换"的是 `mcp_kit/tools.py`。它定义了一个极简的子类：`class MCPStructuredTool(StructuredTool)`，只比 LangChain 原生的 `StructuredTool` 多了一个 `server_name: str` 字段，用来记录这个工具属于哪台 server。

这个字段后面在 agent 调度时很有用——它让框架始终知道某次工具调用该路由回哪个 session。换句话说，工具被适配成 LangChain 形态后并没有"忘记"自己的出身，这为多 server 场景下的调试和归因留了一个钩子。

转换的核心是 `convert_mcp_tool_to_langchain_tool(server_name, session, tool)`。MCP 协议里每个工具自带一份 JSON Schema（`tool.inputSchema`）描述它的入参，但 LangChain 的工具需要一个 Pydantic 模型作为 `args_schema`。

`schema_dict_to_model` 这个函数就负责把前者翻成后者：它读取 schema 的 `properties` 和 `required`，逐字段把 JSON Schema 的类型字符串映射成 Python 类型——`integer→int`、`string→str`、`number→float`、`boolean→bool`，其余一律落到 `Any`。这种"够用即可"的类型映射覆盖了绝大多数工具入参，未知类型退化为 `Any` 也保证了不会因为遇到陌生类型而直接崩溃。

这里有一处体现"防御式设计"的细节。对于 `required` 列表里的字段，函数不是简单地标记必填，而是按类型生成更严格的约束：字符串字段用 `Field(..., min_length=1, required=True, ...)`，确保不能传空串；数值和布尔字段也都带 `required=True`。

注释直白地写道，required 字段必须有真实内容，"empty or null values are strictly prohibited"。这背后的动机很实际——大模型生成工具入参时偶尔会塞进空字符串或缺省占位值，如果不在 schema 层拦住，错误就会一路传到远端 server 才暴露，调试成本高。非必填字段则统一是 `Field(None, required=False, ...)`，允许为空。

每个字段还会尽量带上 schema 里的 `description`，没有就生成一句兜底描述（如 `f"Required string parameter: {field_name}"`），这些描述最终会进入模型看到的工具签名，帮助它正确填参。最后用 Pydantic 的 `create_model(schema.get('title', 'DynamicSchema'), **model_fields)` 动态造出模型类，标题取自 schema 的 `title`，缺省为 `DynamicSchema`。

**输入契约**搞定后看**输出契约**。转换函数内部定义了一个 `async def call_tool(**arguments)` 协程，它的实现就一句关键调用——`await session.call_tool(tool.name, arguments)`，把参数透传给远端工具执行，拿回一个 `CallToolResult`，再交给 `_convert_call_tool_result` 做后处理。注意这是个 `coroutine`，意味着 MCP 工具天然是异步执行的，与 `MultiServerMCPClient` 始终保活的 session 配合，整条调用链路都跑在同一个事件循环里。

`_convert_call_tool_result` 把 MCP 返回的 `content` 列表拆成两堆：`TextContent` 归入文本列表，其余（`ImageContent`、`EmbeddedResource`，被类型别名记作 `NonTextContent`）归入非文本列表。

文本部分会被抽成纯字符串列表（`[content.text for content in text_contents]`）；如果只有一条文本，就直接返回那个字符串而非列表，省去调用方解包的麻烦。这种"单条降维"的小处理对 agent 很友好——大多数工具就返回一段文本，让它直接拿到字符串比拿到只含一个元素的列表更自然。

这里还有一个关键的错误传播逻辑：`if call_tool_result.isError:` 为真时，函数会 `raise ToolException(tool_content)`——也就是说远端工具报告的错误会被转译成 LangChain 自己的工具异常类型，从而能被 agent 的错误处理机制识别和捕获，而不是让一个隐晦的远端错误码沉默地穿过整条链路。

组装时，`MCPStructuredTool` 被这样实例化：`name=tool.name`、`description=tool.description or ""`、`args_schema=tool_input_model`、`coroutine=call_tool`，以及 `response_format="content_and_artifact"`。最后那个参数告诉 LangChain：这个工具的返回是"内容 + 附件"的二元组形式，正好对应 `_convert_call_tool_result` 区分文本与非文本内容的设计。

批量入口 `load_mcp_tools` 则先 `await session.list_tools()`，再对每个工具调用上面的转换函数，返回一个 `BaseTool` 列表。一台 server 暴露多少工具，这里就生成多少个 `MCPStructuredTool`，它们共享同一个 session 但各自绑定不同的 `tool.name`。

至于 `mcp_kit/prompts.py`，它做的是 prompt 维度的同类翻译。

`convert_mcp_prompt_message_to_langchain_message` 把 MCP 的 `PromptMessage`（仅支持 `type == "text"`）按 `role` 转成 `HumanMessage` 或 `AIMessage`，遇到不支持的角色或内容类型就抛 `ValueError`；`load_mcp_prompt` 调 `session.get_prompt` 后逐条转换。这让外部 server 提供的提示词模板也能融进 LangChain 的消息体系，与工具适配是完全对称的两条翻译路径。

## 持久化：MCP 连接配置存在哪

连接信息不能每次都写死在代码里，Chatchat 把它落进了数据库。`db/models/mcp_connection_model.py` 定义了两张表。

`MCPConnectionModel`（表名 `mcp_connection`）存单条连接配置：主键 `id` 是 32 位字符串，`server_name` 唯一且非空，`transport` 标明 `stdio` 或 `sse`，`args`（JSON 列表）、`env`（JSON 字典）、`cwd`（工作目录）、`timeout`（默认 30 秒）、`enabled`（默认 True）、`description`，以及一个关键的 `config`（JSON）字段——注释说明它存放"传输特定配置，包含 command 等字段"。

此外还有 `last_connected_at`、`connection_status`（默认 `"disconnected"`）、`error_message` 这几个运行态字段，以及自动维护的 `create_time` 和带 `onupdate=func.now()` 的 `update_time`。模型还提供了 `to_dict()` 方法把自身序列化成字典，并对可能为 None 的时间字段做了 `isoformat()` 兜底。把"静态配置"（如 transport、args）与"运行态"（如 connection_status、error_message）放在同一张表，让一条记录既是配置也是状态快照。

第二张表 `MCPProfileModel`（表名 `mcp_profile`）存的是一份**全局通用配置**：自增整型主键、`timeout`（默认 30）、`working_dir`（默认 `/tmp`）、`env_vars`（JSON，非空）。

它充当所有连接的默认基线，让用户不必为每条连接重复填写公共的环境变量。这份 profile 与 connection 表是松耦合的——profile 提供默认值，单条 connection 可以用自己的 `env`、`timeout`、`cwd` 覆盖它，体现了配置分层的思路。

`db/repository/mcp_connection_repository.py` 提供了围绕这两张表的全套仓储函数，全部用 `@with_session` 装饰器自动注入并管理数据库会话。

连接维度有 `add_mcp_connection`（用 `uuid.uuid4().hex` 生成 id，并把 `args`/`env`/`config` 的 None 兜底成空容器）、`update_mcp_connection`（逐字段判空更新，只改非 None 的字段）、按 id / 按 server_name 查询、`get_all_mcp_connections`、`get_enabled_mcp_connections`（只取 `enabled=True` 且按创建时间倒序）。

此外还有 `delete_mcp_connection`、`enable_mcp_connection`/`disable_mcp_connection`（只翻转 `enabled` 布尔位），以及 `search_mcp_connections`（按关键词在 `server_name` 和 `description` 上做 `like` 模糊匹配，带 `limit` 默认 50）。这些函数返回的都是普通 dict（而非 ORM 对象），便于直接序列化给 API 层。

Profile 维度则有 `get/create/update/reset/delete_mcp_profile`。其中 `create_mcp_profile` 会先检查是否已存在配置，存在就转去 update，以此保证全局只有一份 profile；`reset_mcp_profile` 把三个字段重置回硬编码的默认值（`timeout=30`、`working_dir="/tmp"`，以及包含 `PATH`/`PYTHONPATH`/`HOME` 的默认 `env_vars`）。这种"单例 profile + 多条 connection"的组合，正是配置系统里常见的"全局默认值 + 局部覆盖"模式。

## 管理 API：MCP 连接的增删改查

`api_server/mcp_routes.py` 把上述仓储能力包成了一组 RESTful 接口，挂在 `APIRouter(prefix="/api/v1/mcp_connections", tags=["MCP Connections"])` 下。

这里有一个值得学习的路由排序技巧：源码顶部用注释强调 "MCP Profile 相关路由 - 放在前面避免与 {connection_id} 冲突"。因为 `/profile` 和 `/{connection_id}` 都能匹配 `GET /api/v1/mcp_connections/profile`，FastAPI 按声明顺序匹配，所以把静态路径 `/profile` 系列提前声明，才不会被动态参数路由抢先吃掉——否则 `profile` 会被当成一个 `connection_id` 去查库，结果自然查不到。

连接管理这一组接口包括：`POST /`（创建，会先用 `get_mcp_connections_by_server_name` 检查重名，重名返回 400）、`GET /`（列表，带 `enabled_only` 查询参数）、`GET /{connection_id}`（详情，不存在返回 404）、`PUT /{connection_id}`（更新）、`DELETE /{connection_id}`（删除）、`POST /{connection_id}/enable` 与 `POST /{connection_id}/disable`（启停）、`POST /search`（条件搜索）、`GET /server/{server_name}` 与 `GET /enabled/list`。

每个写操作都包在 try/except 里，失败时记日志并返回带 `success=False` 的 `MCPConnectionStatusResponse` 或抛 `HTTPException`。

值得一提的是更新与启停接口都会先查一遍连接是否存在，不存在就提前返回，避免对幽灵记录做无意义的写操作；响应体里统一回带 `connection_id`，方便前端关联结果。这种"先校验存在性、再操作、统一回包"的写法在这组路由里反复出现，是一种朴素但可靠的接口约定。

Profile 那组则有 `GET/POST/PUT /profile`、`POST /profile/reset`、`DELETE /profile`。其中 `GET /profile` 在数据库无记录时会返回一份硬编码的默认配置（`timeout=30`、`working_dir="/tmp"`、默认 `env_vars`），保证前端总能拿到可用值而不必处理"配置为空"的边界情况。这些接口共同构成了 Web 端"MCP 连接管理"页面的后端支撑。

## 打通线：MCP 工具如何汇入 agent 流水线

至此各个零件都摆好了，关键问题是它们怎么串成一条线、最终让 agent 用上 MCP 工具。答案藏在 `server/chat/chat.py` 里。

Agent 对话接口 `chat` 带一个 `use_mcp: bool = Body(False, ...)` 参数，只有打开它才会激活 MCP，默认关闭意味着不开启 MCP 的对话不会额外承担连接外部 server 的开销。`create_models_chains` 函数里，代码先调 `get_enabled_mcp_connections()` 从数据库取出所有启用的连接，然后做一次**格式转换**：遍历每条记录，按 `transport` 拼出 `MultiServerMCPClient` 能识别的连接字典。

对 stdio 类型，`command` 取自 `conn["config"].get("command", conn["args"][0] if conn["args"] else "")`，`args` 取 `conn["args"][1:]`——这印证了 `config` 字段确实承载着传输层细节，同时也兼容了把命令放在 `args[0]` 的历史写法；对 sse 类型，则从 `config` 里取 `url` 和 `headers`。

转换结果存入 `mcp_connections` 字典，最后这一句 `mcp_connections=mcp_connections if use_mcp else {}` 是开关的落点：关掉 `use_mcp` 时传空字典，等于不连任何 server。这种"按需连接"的设计意味着 MCP 的成本只在真正需要时才付出，普通对话不会因为数据库里登记了一堆连接而被拖慢。

这份 `mcp_connections` 顺着 `PlatformToolsRunnable.create_agent_executor` 一路传到 `agents/platform_tools/base.py`。

那里的静态方法 `create_mcp_client` 会先给每条 stdio 连接补上 `env`（合并 `os.environ` 并设 `PYTHONHASHSEED="0"`），然后 `client = MultiServerMCPClient(connections)` 并**手动 `await client.__aenter__()`**——注释特意说明这是 "without context manager to keep session alive"，即不用 `async with`，好让 session 在 agent 整个执行期间保持存活。

为什么要保活？因为前面 `MCPStructuredTool` 的 `coroutine` 闭包捕获了 `session`，工具每次被调用都要用这个活的 session 去 `call_tool`；若用 `async with` 在 `get_tools` 返回后就关掉 session，后续工具调用就会失败。建好客户端后调 `client.get_tools()` 拿到全部 MCP 工具，作为 `mcp_tools` 参数传入 `agents_registry`。

而在 `server/agents_registry/agents_registry.py` 里，`agents_registry` 函数的签名明确接收 `mcp_tools: Sequence[MCPStructuredTool] = []`。

当 agent_type 走到 `platform-knowledge` 分支时，`mcp_tools` 同时被传给 `create_platform_knowledge_agent`（让模型在 prompt 层面"看见"这些工具）和 `PlatformToolsAgentExecutor`（让执行器在调度层面"能调用"这些工具）。一边管"知道有这个工具"，一边管"真的能执行这个工具"，二者缺一不可。这就是那条完整的打通线：

> 数据库 `mcp_connection` 表 → `get_enabled_mcp_connections()` → `chat.py` 转成连接字典 → `MultiServerMCPClient` 连接并 `load_mcp_tools` 适配 → `MCPStructuredTool` 列表 → `agents_registry` 的 `mcp_tools` 参数 → agent 与 executor。

由此，一个由外部团队维护、用任意语言写成的 MCP server，只要在 Web 界面登记一条连接、点一下启用，它暴露的每个工具就会自动出现在 Chatchat agent 的可调用工具集中，与第 11 章那些本地 Tools Factory 工具平起平坐——这正是 MCP "即插即用"承诺在 Chatchat 里的具体兑现。

## MCP 工具与本地工具的异同

把 MCP 工具放回第 10、11 章的语境里对照，能更清楚它的定位。

相同之处在于：两者最终都是 LangChain 的 `BaseTool`，都进入 `agents_registry` 的工具列表，对 agent 的调度逻辑来说完全等价，agent 并不知道也不关心某个工具是本地函数还是远端 server。这正是 LangChain 工具抽象的价值——`MCPStructuredTool` 继承自 `StructuredTool`，天然就能塞进任何吃 `BaseTool` 的地方。

不同之处则体现在三个层面。

其一是**来源与配置**：本地工具在代码里静态注册，MCP 工具来自数据库里的连接配置，可以在运行期通过 API 增删启停，无需改代码或重启。其二是**执行方式**：本地工具是同步或异步的本地调用，MCP 工具一律是异步的、跨进程或跨网络的 `session.call_tool`，因此 `MultiServerMCPClient` 必须保活 session、注入 `PATH`、处理超时。其三是**入参定义**：本地工具的 `args_schema` 由开发者用 Pydantic 写死，MCP 工具的 schema 是运行期从 server 拉来的 JSON Schema 经 `schema_dict_to_model` 动态翻译而成。

理解了这三点差异，也就理解了 `mcp_kit` 这个目录为何存在——它本质上是一层"协议适配器"，把异构、远端、动态的外部能力，归一化成 agent 习以为常的本地工具形态。

正因为这层适配做得足够干净，上游的 agent 体系几乎不需要为 MCP 做任何特殊处理：`agents_registry` 只是多了一个 `mcp_tools` 参数，其余调度、回调、错误处理逻辑全部复用既有路径。这也是判断一个集成是否优雅的朴素标准——它对既有体系的侵入越小，越说明抽象选对了位置。

## 本章小结

1. MCP（Model Context Protocol）用标准协议隔离"工具提供方"与"工具使用方"，Chatchat 扮演 MCP **client**，把外部能力即插即用地接入 agent。
2. `mcp_kit/client.py` 的 `MultiServerMCPClient` 用一个 `AsyncExitStack` 统一托管多台 server 的 session 与子进程，`__aenter__`/`__aexit__` 保证连接的优雅建立与清理。
3. 支持两种传输：`StdioConnection`（本地子进程，注入 `PATH` 以便定位 `uvx`/`npx`）与 `SSEConnection`（远端 HTTP，默认 `DEFAULT_SSE_READ_TIMEOUT=300` 秒）。
4. `_initialize_session_and_load_tools` 是协议层与工具层的衔接点：先 `session.initialize()` 握手，再 `load_mcp_tools` 拉取并适配工具。
5. `MCPStructuredTool` 继承 `StructuredTool` 并多记一个 `server_name`，用于把工具调用路由回正确的 session。
6. `schema_dict_to_model` 把 MCP 工具的 JSON Schema 翻成 Pydantic `args_schema`，对 required 字段施加 `min_length=1` 等严格约束，禁止空值。
7. 输出契约由 `_convert_call_tool_result` 落地：拆分文本/非文本内容，单条文本直接返回字符串,且 `isError` 时转译为 LangChain 的 `ToolException`。
8. 工具用 `response_format="content_and_artifact"` 注册，对应"内容 + 附件"的二元返回结构。
9. 连接配置持久化在 `mcp_connection` 表，`config`（JSON）字段承载 `command`/`url`/`headers` 等传输层细节；`mcp_profile` 表存一份全局默认配置。
10. `mcp_routes.py` 提供完整 CRUD 与启停接口，并把 `/profile` 路由声明在 `/{connection_id}` 之前以规避路由冲突。
11. 打通线：`get_enabled_mcp_connections` → `chat.py` 转换 → `MultiServerMCPClient.get_tools()` → `agents_registry(mcp_tools=...)`，使 MCP 工具与本地 Tools Factory 工具汇入同一条 agent 流水线。

## 动手实验

1. **实验一：追踪一条 MCP 工具的完整生命周期** — 从 `chat.py` 的 `create_models_chains` 出发，沿着 `mcp_connections` 字典向下读到 `platform_tools/base.py` 的 `create_mcp_client`，再到 `agents_registry` 的 `mcp_tools` 参数，画一张数据流图，标出每一步的数据形态（数据库 dict → 连接 dict → `MCPStructuredTool` → agent 工具）。
2. **实验二：验证 JSON Schema 到 Pydantic 的转换** — 在本地用 `schema_dict_to_model` 喂入一个手写 schema（含一个 required 的 string 字段和一个可选的 integer 字段），打印生成模型的字段约束，确认 required 字段带 `min_length=1` 而可选字段默认为 `None`，再试着传空串触发校验报错。
3. **实验三：走通一次连接管理 API** — 启动服务后用 `curl` 依次调用 `POST /api/v1/mcp_connections/`（创建一条 stdio 连接）、`GET /api/v1/mcp_connections/`、`POST /{id}/disable`、`GET /enabled/list`，观察启用状态如何影响 `get_enabled_mcp_connections` 的返回，并验证重名创建会返回 400。
4. **实验四：对比 stdio 与 sse 两条连接路径** — 阅读 `connect_to_server_via_stdio` 与 `connect_to_server_via_sse`，列出二者在参数、超时常量、`PATH` 注入上的差异；再结合 `chat.py` 里按 `transport` 分支拼连接字典的代码，说明 `config` 字段对两种传输分别承载了哪些键。

> **下一章预告**：MCP 让 agent 能调用外部能力，但 Chatchat 本身要先成为一个稳定的服务、并把各家大模型统一接进来。第 13 章《API 服务与模型接入》将拆解 Chatchat 的 API server 架构、OpenAI 兼容接口，以及模型平台（model platform）如何把不同厂商的 LLM、Embedding 统一抽象到一套调用契约之下。
