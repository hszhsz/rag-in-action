# 第 11 章 MCP 集成与多渠道接入：一套内核，多个外壳

前面的章节里，我们把 AnythingLLM 的 RAG 内核拆得相当透彻：从文档采集、切分、向量化，到检索重排、LLM 接入，再到 aibitat agent 与 Agent Flows 的可视化编排。但这些能力如果只能在自带的 Web 界面里被使用，价值会被极大地局限。一个成熟的 RAG 平台必须回答两个方向相反的问题：第一，如何把"别人的工具"接进来，让 agent 的能力可以无限外扩；第二，如何把"自己的能力"送出去，让同一个工作区出现在网页挂件、Telegram、浏览器扩展、移动 App 等任何用户所在的地方。

这两个问题在 AnythingLLM 里分别由两套机制承担。前者是 MCP（Model Context Protocol）集成——AnythingLLM 作为 MCP 客户端，加载外部 MCP server 暴露的工具，并把它们翻译成 agent 能直接调用的插件。后者是多渠道接入——所有渠道最终都收敛回同一套 `chats/` 内核，只是各自套上了不同的鉴权外壳。本章先用主要篇幅解构 MCP，再展开多渠道的复用与鉴权设计。

## MCP 是什么：把工具从协议层标准化

MCP 是由 Anthropic 提出的开放协议，它的核心思想是用一套标准的客户端-服务端协议，让 LLM 应用（host/client）能够发现并调用外部进程（server）暴露的工具、资源与提示模板，而不必为每一个集成单独写胶水代码 [[Model Context Protocol Specification]](https://modelcontextprotocol.io/specification)。一个 MCP server 可以是本地的子进程，也可以是远程的 HTTP 服务；客户端通过 `listTools`、`callTool` 等标准方法与之交互。这种"协议先行"的设计意味着，任何遵守 MCP 的工具——无论是查 Docker 容器、读数据库还是操作文件系统——都能即插即用。

AnythingLLM 在这件事上扮演的是 **MCP 客户端** 的角色。它依赖官方 SDK `@modelcontextprotocol/sdk`，在 `server/utils/MCP/hypervisor/index.js` 顶部直接引入了 `Client`、`StdioClientTransport`、`SSEClientTransport` 与 `StreamableHTTPClientTransport` 四个类。整个集成被组织成两层：底层的 `MCPHypervisor`（生命周期管理）和上层的 `MCPCompatibilityLayer`（工具到插件的翻译）。后者继承自前者，并且都是单例。

## 配置即真相：anythingllm_mcp_servers.json

AnythingLLM 不在数据库里存 MCP server 定义，而是用一个 JSON 文件作为唯一真相来源。`MCPHypervisor` 构造时调用私有方法 `#setupConfigFile()` 解析出路径 `mcpServerJSONPath`：开发环境（`process.env.NODE_ENV === "development"`）下指向 `storage/plugins/anythingllm_mcp_servers.json`，生产环境下则基于 `process.env.STORAGE_DIR` 拼出 `plugins/anythingllm_mcp_servers.json`。如果文件不存在，它会递归建目录并写入一个空骨架 `{ "mcpServers": {} }`。

文件的顶层结构是一个 `mcpServers` 对象，键是 server 名称，值是 server 定义。getter `mcpServerConfigs` 把这个对象 `Object.entries` 展开成 `{ name, server }[]` 数组供内部使用。一个典型的 stdio 类型条目长这样：

```json
{
  "mcpServers": {
    "docker-mcp": {
      "command": "npx",
      "args": ["-y", "docker-mcp"],
      "env": { "SOME_KEY": "value" },
      "anythingllm": {
        "autoStart": true,
        "suppressedTools": ["dangerous-tool"]
      }
    }
  }
}
```

注意 `anythingllm` 这个命名空间——它是 AnythingLLM 在标准 MCP 配置之上扩展的私有字段，承载两类平台特有的元信息。`anythingllm.autoStart` 控制该 server 在 boot 时是否自动启动；`anythingllm.suppressedTools` 是一个工具名数组，用于在不删除 server 的前提下屏蔽其中某些危险工具。这种"把平台特性塞进自定义键、不污染协议标准字段"的做法，是与外部生态保持兼容的常见手法。

源码注释毫不避讳地提醒了这套机制的风险：`MCPHypervisor` 的类注释直言 "MCP is basically arbitrary code execution"，并强调 server 定义是否正确、工具是否安全是用户自己的责任；它还特别说明该类 **不会** 检查工具的依赖，比如某个工具需要 `npx`，那么 AnythingLLM 主进程运行的上下文就必须能访问 `npx`——而这在预构建镜像里并不常见，所以可能直接跑不起来。把责任边界讲清楚，本身就是工程上的诚实。

## MCPHypervisor：管理 server 子进程与连接的生命周期

`MCPHypervisor` 维护两张核心表：`mcps` 存放当前正在运行的 server 客户端实例（键为 server 名），`mcpLoadingResults` 存放每个 server 的加载结果（`status` 为 `success` 或 `failed`，附带 `message`）。这两张表的分离让前端既能拿到在跑的 server，又能拿到启动失败的 server 及其错误原因——`servers()` 方法正是据此为管理界面拼出完整列表。

启动的入口是 `bootMCPServers()`。它先做一个幂等检查：如果 `this.mcps` 里已有 server，就打印 "MCP Servers already running, skipping boot." 直接返回，避免重复启动。否则它遍历 `mcpServerConfigs`，对每个 server 先检查 `anythingllm.autoStart` 是否被显式设为 `false`——若是则跳过并记一条 failed 结果；否则调用私有方法 `#startMCPServer({ name, server })`。每个 server 的启动都包在 try/catch 里，单个失败不会中断整批，只会把对应的 `mcpLoadingResults[name]` 标记为失败并清理掉残留的连接。

`#startMCPServer` 是真正干活的地方。它先用 `#parseServerType(server)` 推断传输类型，再用 `#validateServerDefinitionByType` 按类型校验定义，然后 `new Client({ name, version: "1.0.0" })` 创建客户端，用 `#setupServerTransport` 建立传输层，挂上 `onclose`/`onerror`/`onmessage` 三个事件监听，最后调用 `mcp.connect(transport)`。这里有一个值得注意的细节：连接被包进了 `Promise.race`，与一个 30 秒的超时 Promise 赛跑——常量直接写作 `30_000`，注释为 "30 second timeout"。一个挂死的 MCP server 不会让整个启动流程无限等待，这是把外部不可控进程纳入管理时必须有的防御。

停止与清理同样细致。`pruneMCPServer(name)` 针对单个 server：取出它的 `transport`，对 `transport._process`（stdio 类型的子进程）发送 `SIGTERM`，再调用 `transport.close()`，最后从 `mcps` 中删除并把结果标记为 "Server was stopped manually by the administrator."。`pruneMCPServers()` 则是批量版本，遍历所有 server 逐个 kill 并 close，最后把 `mcps` 和 `mcpLoadingResults` 都重置为空对象。`reloadMCPServers()` 就是先 prune 再 boot 的组合，使得管理员可以在不重启整个服务的情况下应用配置变更——这正是 `/mcp-servers/force-reload` 端点背后的能力。

环境变量的处理也藏着工程心思。私有方法 `#buildMCPServerENV(server)` 通过 `patchShellEnvironmentPath()` 拿到 shell 环境，组装出带 `PATH` 与 `NODE_PATH` 兜底值的 `baseEnv`；当检测到 `process.env.ANYTHING_LLM_RUNTIME === "docker"` 时，会针对 Docker 场景重设 `NODE_PATH` 为 `/usr/local/lib/node_modules`、`PATH` 为 `/usr/local/bin:/usr/bin:/bin`；最后再把用户在 `server.env` 里指定的变量合并进去，且用户值优先。之所以费这么大劲，是因为 stdio 类型的 MCP server 本质上是 `npx` 之类的命令行进程，找不到正确的 `PATH` 就会启动失败——跨平台、跨部署形态地保证子进程能找到可执行文件，是 stdio 传输能用的前提。

## stdio 与 SSE/streamable-http：三种传输的差异

MCPHypervisor 的类型别名把传输归纳为三种：`@typedef {'stdio' | 'http' | 'sse'} MCPServerTypes`。判定逻辑在 `#parseServerType`：如果 `server.type` 是 `"sse"`、`"streamable"` 或 `"http"`，归为 `http`；否则只要定义里有 `command` 字段就是 `stdio`，有 `url` 字段就是 `http`，都没有则兜底为 `sse`。这套"显式 type 优先、再按字段推断"的策略，让配置既能精确指定也能省略。

传输对象由 `#setupServerTransport(server, type)` 创建。stdio 走的是 `new StdioClientTransport({ command, args, ...env })`，即 AnythingLLM 自己把 MCP server 作为子进程拉起来，通过标准输入输出与之通信——这也是为什么 stdio 类型需要前面那一整套环境变量与 `PATH` 的铺垫。非 stdio 一律交给 `createHttpTransport(server)`：它先 `new URL(server.url)` 解析地址，然后按 `server.type` 分流——`"streamable"` 或 `"http"` 用 `StreamableHTTPClientTransport`，其余默认用 `SSEClientTransport`，两者都把 `server.headers` 透传到 `requestInit.headers` 里，以便携带鉴权头访问远程服务。

校验逻辑也按传输类型分叉。`#validateServerDefinitionByType` 对 http/sse/streamable 类型强制要求 `url` 存在且能被 `new URL()` 解析通过，否则抛出像 `MCP server "x": missing required "url"` 这样的明确错误；对 stdio 类型则校验 `args`（若存在）必须是数组。这种"按类型给出针对性错误信息"的写法，对一个需要用户手写 JSON 配置的功能来说，是降低排错成本的关键。

两种传输形态的取舍很清楚：stdio 适合本地工具，启动快、无需网络，但依赖宿主机有对应的运行时；HTTP/SSE 适合远程或容器化部署的工具，靠 URL 与 headers 接入、不依赖本地可执行文件，更适合在受限镜像里使用。AnythingLLM 把两者统一在同一套 `Client` 抽象之下，对上层的 `MCPCompatibilityLayer` 完全透明。

## 从 MCP 工具到 aibitat 插件：协议与 agent 的桥接

启动之后，MCP server 只是"连上了"，还需要被翻译成 agent 能理解的东西——这正是 `MCPCompatibilityLayer` 的职责。它的 `activeMCPServers()` 方法先 `await this.bootMCPServers()` 确保所有 server 已启动，然后把每个 server 名映射成 `@@mcp_${name}` 格式的字符串返回。这个 `@@mcp_` 前缀是一个关键约定：它是一个"占位标记"，告诉 agent 装配流程"这里有一组 MCP 工具，稍后展开"。

在 agent 的默认能力清单里，这一步是自然嵌入的。`server/utils/agents/defaults.js` 的 functions 数组末尾就有一行 `...(await new MCPCompatibilityLayer().activeMCPServers())`，与 agent skills、imported plugins、Agent Flows 的插件并列。也就是说，对 agent 而言，MCP server 一开始只是它函数列表里的几个 `@@mcp_xxx` 占位符，与其它能力来源没有本质区别。

真正的展开发生在 agent 装配阶段。`server/utils/agents/index.js` 与 `ephemeral.js` 在遍历待加载的函数名时，都有一段专门处理 `@@mcp_` 前缀的逻辑：当 `name.startsWith("@@mcp_")` 时，剥掉前缀得到 server 名 `mcpPluginName`，调用 `convertServerToolsToPlugins(mcpPluginName, this.aibitat)`，把返回的子插件逐个 `aibitat.use(plugin.plugin())` 挂载，同时从 `@agent` 的 functions 列表里 **移除** 那个父级 `@@mcp_` 占位、把子工具名一个个 push 进去。注释把意图说得很直白："This will replace the parent MCP server plugin with the sub-tools as child plugins so they can be called directly by the agent when invoked."——一个 server 占位符被替换成它名下所有具体工具，agent 因此能直接点名调用 `docker-mcp:list-containers` 这样的工具。

`convertServerToolsToPlugins(name)` 内部是翻译的核心。它先从 `this.mcps[name]` 取出客户端，调用 `mcp.listTools()` 拿到工具清单；接着用 `getSuppressedTools(name)` 读出该 server 的 `anythingllm.suppressedTools`，把被屏蔽的工具 `filter` 掉，并打印诸如 "N tool(s) suppressed" 的日志；若过滤后一个工具都不剩则返回 null 跳过。对每个保留下来的工具，它构造一个插件对象，插件名是 `${name}-${tool.name}`，在 `setup` 里通过 `aibitat.function(...)` 注册成 agent 函数。这里有几个细节值得留意：参数 schema 直接用 `$schema: "http://json-schema.org/draft-07/schema#"` 加上工具自带的 `tool.inputSchema` 拼成，意味着 MCP 工具的入参约束被原样转交给 agent 的函数调用框架；插件被打上 `isMCPTool: true` 标记以便区分来源。

handler 则是运行时的执行体。它在被调用时 `new MCPCompatibilityLayer()`（拿单例）取出当前的 `currentMcp`，若 server 已不在运行则抛错；否则通过 `aibitat.introspect` 把"正在执行某 MCP 工具及其参数"暴露给用户观察，再调用 `currentMcp.callTool({ name: tool.name, arguments: args })` 执行真正的工具，最后用静态方法 `returnMCPResult(result)` 把结果序列化成字符串返回。`returnMCPResult` 处理得相当稳健：它用 `WeakSet` 防循环引用、把 `bigint` 转成字符串、序列化失败时返回 `[Unserializable: ...]`——因为 MCP 工具可以返回任意类型的数据，而 agent 框架只接受字符串，这一层防御不可或缺。整个 handler 也包在 try/catch 里，工具失败会返回一句人类可读的错误而不是让 agent 崩溃。

## MCP 管理 API：把生命周期开放给管理员

所有这些能力通过 `server/endpoints/mcpServers.js` 暴露成 REST 端点，且无一例外都套了 `[validatedRequest, flexUserRoleValid([ROLES.admin])]` 两道中间件——MCP 等同于任意代码执行，因此只有 admin 角色能操作。`GET /mcp-servers/list` 调 `servers()` 返回 server 列表（含运行状态、工具清单、进程 pid、错误信息）；`GET /mcp-servers/force-reload` 调 `reloadMCPServers()` 后返回最新列表；`POST /mcp-servers/toggle` 调 `toggleServerStatus(name)` 在运行/停止之间切换；`POST /mcp-servers/delete` 调 `deleteServer(name)` 杀进程并从配置文件里抹掉条目；`POST /mcp-servers/toggle-tool` 调 `toggleToolSuppression(serverName, toolName, enabled)` 来增删 `suppressedTools`。值得一提的是，`toggleServerStatus` 与 `deleteServer` 都先用 `mcp.ping()` 判活，再决定是 prune 还是 start，避免对一个已经死掉的 server 重复操作。

至此 MCP 部分收束：配置文件是真相，Hypervisor 管生命周期与传输，CompatibilityLayer 把工具翻译成 aibitat 插件，端点把控制权交给 admin。接下来转向问题的另一面——如何把这套内核送到不同渠道。

## 一套内核，多个外壳：渠道复用的总原则

AnythingLLM 的多渠道设计有一条贯穿始终的原则：**渠道只是入口与鉴权的差异，最终都收敛回同一套 chat/RAG 内核**。无论请求从网页挂件、Telegram、浏览器扩展还是移动 App 进来，到了核心处理阶段，走的都是同一组 `server/utils/chats/` 下的逻辑——同样的向量检索、同样的 pinned docs、同样的 `compressMessages`、同样的 LLM 连接器解析。每个渠道要做的，只是把自己特有的"会话身份"映射到一个 workspace，并套上自己的鉴权门禁。理解了这一点，四个渠道就可以并排来看。

## 嵌入式挂件 embed：用 embedId + 域名白名单守门

嵌入式聊天挂件是把 AnythingLLM 塞进任意第三方网站的方式。它的端点在 `server/endpoints/embed/index.js`，核心是 `POST /embed/:embedId/stream-chat`，前置三道中间件 `[validEmbedConfig, setConnectionMeta, canRespond]`。鉴权完全围绕 `embedId` 展开：`validEmbedConfig`（在 `embedMiddleware.js` 中）用 URL 里的 `embedId` 调 `EmbedConfig.getWithWorkspace({ uuid })` 取出 embed 配置及其挂接的 workspace，挂到 `response.locals.embedConfig`；找不到就直接 404。

真正的"门禁"在 `canRespond` 里，逻辑层层递进：先看 `embed.enabled` 是否被管理员关闭（否则 503）；再做 **域名白名单** 校验——用 `request.headers.origin` 与 `EmbedConfig.parseAllowedHosts(embed)` 比对，不在名单里就 401。这里还有一处安全加固：当白名单为 null（即创建挂件时没设白名单，意味着接受任意来源）且环境变量里存在 `EMBED_REQUIRE_ALLOWLIST` 时，会把"无白名单"当作"拒绝一切"处理，逼迫挂件所有者必须显式设置允许的域名。此外还校验 `sessionId` 必须是合法 UUID、message 非空、chat_mode 合法，并用 `max_chats_per_day` 与 `max_chats_per_session` 做按天和按会话的限流（超限返回 429）。这一整套是典型的"面向公网匿名用户"的防护，因为挂件没有登录账号，只能靠 embedId、域名、会话与频率来约束。

进入处理后，端点调用 `streamChatWithForEmbed(response, embed, message, sessionId, {...})`（在 `server/utils/chats/embed.js`）。这个函数几乎是 RAG 内核的等价复刻：强制把 `automatic` 模式降级为 `chat`（注释说明 "Automatic mode is NOT valid for embeds"），按 `embed.allow_model_override`/`allow_prompt_override`/`allow_temperature_override` 三个开关决定是否接受请求里的覆盖参数，然后走 `resolveLLMConnectorForEmbed`、`getVectorDbClass()`、`performSimilaritySearch`、pinned docs、`fillSourceWindow`、`compressMessages`、流式输出，与 Web 端的 RAG 流程一一对应。唯一不同的是历史和落库走的是 `EmbedChats` 模型而非 `WorkspaceChats`，会话以 `sessionId` 而非用户身份来组织。这就是"复用内核、替换外壳"最直白的体现。

## Telegram：单用户配对鉴权 + Worker 跑 agent

Telegram 接入由 `server/utils/telegramBot/index.js` 的 `TelegramBotService` 单例承载，它不是 HTTP 端点而是一个长驻的轮询服务。`bootIfActive()` 在服务启动时从 `ExternalCommunicationConnector.get("telegram")` 读出连接器配置，用 `decryptToken` 解密存储的 bot token（token 在库里是加密的），再 `start(config)` 起轮询。

它的鉴权模型与 embed 截然不同，是一套"配对审批"机制。`#setupHandlers` 里的 `guard` 函数对每条消息先用 `isVerified(this.#config.approved_users, msg.chat.id)` 判断该 chatId 是否已被批准；未批准就发 `sendPairingRequest` 走配对流程，由管理员通过 `approvePendingUser`/`denyPendingUser` 审批。更关键的是它强制 `#assertSingleUserMode()`：一旦检测到实例处于多用户模式，就直接停掉 bot 并删除连接器——Telegram 接入被刻意限制为只在单用户实例下可用，因为 Telegram 的 chatId 无法对应到 AnythingLLM 的多用户账号体系。

消息处理同样收敛回内核，但路径上有渠道特色。`#handleMessage` 按消息类型分流：语音消息先 `transcribeAudio` 转写、图片消息走 `photoToAttachment` 转附件、文档消息用 `documentToText` 抽取文本，统一拼成 message 后交给 `#runChatJob`。`#runChatJob` 不直接在主进程里跑 chat，而是通过 `BackgroundService` 的 bree 起一个独立 Worker，加载 `jobs/handle-telegram-chat.js`，把 botToken、chatId、workspaceSlug、threadSlug 等通过 `worker.send` 传进去——把可能很重、可能很慢的 agent 推理隔离到子进程，主进程只管轮询和转发。Worker 内部最终走的是 `telegramBot/chat/stream.js` 的 `streamResponse`，其注释明确写着 "Uses the same pipeline as the web UI (RAG, parsed docs, pinned docs, etc.) and stores chats with thread_id so they appear in the AnythingLLM UI."——同一条 RAG 管线、同一张 `WorkspaceChats` 表，只是把流式 token 通过 `editMessage` 不断改写同一条 Telegram 消息来模拟打字效果（受 `STREAM_EDIT_INTERVAL` 限流、超过 `MAX_MSG_LEN` 就分条）。它甚至复用了 `AgentHandler.isAgentInvocation` 来判断是否该转入 agent 流程，并支持把 agent 的工具调用做成 Telegram 内联按钮的"审批"交互（`#handleToolApprovalRequest`）。

## 浏览器扩展：API Key 鉴权，主攻"喂数据"

浏览器扩展的端点在 `server/endpoints/browserExtension.js`，鉴权用的是 Bearer API Key。中间件 `validBrowserExtensionApiKey` 从 `Authorization` 头取出 bearer token，用 `BrowserExtensionApiKey.validate(bearerKey)` 校验；在多用户模式下还会进一步用 `apiKey.user_id` 取出 user、校验是否被 suspended，并把 user 挂到 `response.locals.user`。这使得扩展能"以某个用户的身份"操作，所有 workspace 查询都通过 `multiUserMode(response) ? Workspace.whereWithUser(user) : Workspace.where()` 做按用户隔离。

与其它渠道不同，浏览器扩展的主要用途不是聊天，而是 **把网页内容喂进工作区**。`POST /browser-extension/embed-content` 接收 `workspaceId`、`textContent`、`metadata`，用 `CollectorApi().processRawText` 把原始文本交给 collector 进程处理成文档，再 `Document.addDocuments(workspace, [...], user?.id)` 嵌入到目标 workspace——这正是第 2 章 collector 与第 6 章文档管理那套内核的复用，只是入口换成了浏览器里框选的文本。`POST /browser-extension/upload-content` 则只处理文本不绑定 workspace。此外它还有一组 `/browser-extension/api-keys` 的管理端点，用 `[validatedRequest, flexUserRoleValid([ROLES.admin, ROLES.manager])]` 保护，让管理员/经理签发和吊销扩展用的 API Key。鉴权方式（API Key）与功能定位（采集而非对话）的组合，恰好匹配"扩展运行在用户浏览器、需要长期凭据、主要做数据导入"这一场景。

## 移动端：设备 token 鉴权 + 复用 ApiChatHandler

移动端端点在 `server/endpoints/mobile/index.js`，鉴权围绕"设备 token"展开。设备先通过 `POST /mobile/register`（经 `validRegistrationToken` 校验临时注册 token）在库里创建一条记录，但需要管理员审批后才可用；之后所有请求用 `validDeviceToken` 中间件校验 `x-anythingllm-mobile-device-token` 头，确认设备存在且 `approved`，并在多用户场景下把设备关联的 user 挂到 locals。临时注册 token 是一次性的，用过即废，这降低了凭据泄露的风险。

业务逻辑集中在 `POST /mobile/send/:command` 调用的 `handleMobileCommand`，它用 command 字段做路由：`workspaces` 列工作区、`workspace-content` 拉线程与历史、`model-tag` 查模型、`reset-chat`/`new-thread` 管理会话，而真正的对话是 `stream-chat` 命令——它取出 workspace 与 thread 后，直接调用 `ApiChatHandler.streamChat({...})`，复用的是与公开 API 完全相同的 chat 处理器，落库同样进 `WorkspaceChats`/`workspace_threads`。移动端因此几乎没有自己的"聊天逻辑"，它只是 ApiChatHandler 的又一个调用方，所有的多用户隔离（`getWithUser` vs `get`、按 `user.id` 过滤 threads/chats）都顺着 user 这条线自然贯穿下来。

把四个渠道并排看，鉴权方式的分化一目了然：embed 用 embedId + 域名白名单 + 频率限制对付匿名公网用户，Telegram 用配对审批 + 强制单用户对付 IM 场景，浏览器扩展用 Bearer API Key 对付长期驻留的客户端，移动端用设备 token + 审批对付 App。但无论门禁多么不同，门后都是同一套向量检索、同一套 LLM 连接器、同一套 chat 落库——这正是"一套内核、多个外壳"在代码层面的真实形态。

## 本章小结

- MCP 集成让 AnythingLLM 作为 **MCP 客户端**，通过官方 SDK `@modelcontextprotocol/sdk` 加载外部 MCP server 暴露的工具，实现 agent 能力的无限外扩。
- MCP server 定义以 JSON 文件 `anythingllm_mcp_servers.json` 为唯一真相来源，顶层结构是 `mcpServers` 对象，平台特有的 `autoStart`、`suppressedTools` 等字段被收纳在自定义的 `anythingllm` 命名空间下，不污染协议标准字段。
- `MCPHypervisor`（单例）管理 server 的生命周期：`bootMCPServers` 幂等批量启动、`#startMCPServer` 用 `Promise.race` 加 `30_000` 毫秒超时防挂死、`pruneMCPServer`/`pruneMCPServers` 发 `SIGTERM` 清理子进程、`reloadMCPServers` 实现免重启热更新。
- 传输分三种：stdio 由 `StdioClientTransport` 把工具作为子进程拉起（依赖宿主 `PATH`，故有 `#buildMCPServerENV` 与 Docker 专项环境处理）；HTTP/SSE 由 `StreamableHTTPClientTransport` 与 `SSEClientTransport` 接入远程服务并透传 `headers` 鉴权；`#parseServerType` 按显式 `type` 优先、再按 `command`/`url` 字段推断。
- `MCPCompatibilityLayer.activeMCPServers` 把每个 server 映射成 `@@mcp_${name}` 占位符放入 agent 函数列表；agent 装配时识别 `@@mcp_` 前缀，调用 `convertServerToolsToPlugins` 将占位符替换为该 server 名下所有具体工具的子插件。
- 工具到插件的翻译保留了 MCP 工具的 `inputSchema` 作为 agent 函数参数约束，打上 `isMCPTool: true` 标记，运行时 handler 调 `callTool` 执行并用 `returnMCPResult`（WeakSet 防循环、bigint 转字符串）安全序列化任意返回值。
- MCP 管理端点全部限定 `ROLES.admin`，因为源码注释直言 MCP 等同于任意代码执行；端点覆盖 list / force-reload / toggle / delete / toggle-tool 全套生命周期操作。
- 多渠道的总原则是"一套内核、多个外壳"：embed、Telegram、浏览器扩展、移动端最终都收敛回 `server/utils/chats/` 或 `ApiChatHandler` 的同一套 RAG 管线与落库逻辑。
- 各渠道鉴权方式按场景分化：embed 用 `embedId` + `parseAllowedHosts` 域名白名单 + `max_chats_per_day/session` 限流；Telegram 用 `approved_users` 配对审批且强制单用户模式、token 加密存储；浏览器扩展用 Bearer API Key 且主攻 `processRawText` 数据采集；移动端用一次性注册 token + 设备 token + 审批。
- Telegram 把 agent 推理隔离到 bree Worker（`handle-telegram-chat.js`）执行，主进程只负责轮询与消息转发，并通过 `editMessage` 限流改写实现流式打字效果。
- embed 强制把 `automatic` 模式降级为 `chat`、历史落 `EmbedChats`；而 Telegram 与移动端复用 `WorkspaceChats` 并写 `thread_id`，使外部渠道的对话同样出现在 AnythingLLM 主界面。

## 动手实验

1. **实验一：用 stdio MCP server 跑通一次 agent 调用** — 在 `storage/plugins/anythingllm_mcp_servers.json` 里添加一个 stdio 类型条目（例如 `command: "npx"`，`args: ["-y", "<某个公开 mcp 包>"]`），重启或调 `GET /mcp-servers/force-reload`，确认 `GET /mcp-servers/list` 返回该 server `running: true` 且带工具清单；随后在工作区里发 `@agent` 消息触发 agent，观察日志里 `Attached MCP::<server>:<tool>` 的挂载行与 `Executing MCP server` 的执行行，验证 `@@mcp_` 占位符如何被展开成具体工具。

2. **实验二：观察 suppressedTools 的过滤效果** — 调 `POST /mcp-servers/toggle-tool` 把某个工具 `enabled: false`，读取配置文件确认 `anythingllm.suppressedTools` 数组里多了该工具名；再 force-reload 并查看 `convertServerToolsToPlugins` 打印的 "N tool(s) suppressed" 日志，确认被屏蔽的工具不再出现在 agent 的函数列表里；最后把它重新 enable，验证恢复。

3. **实验三：复现 embed 的域名白名单与限流门禁** — 为某工作区创建一个 embed 配置并设置 `allowed_hosts` 白名单，用不在名单内的 `Origin` 头向 `POST /embed/:embedId/stream-chat` 发请求，确认返回 401；再设置 `max_chats_per_day` 为很小的值，连续请求直到触发 429，对照 `canRespond` 中间件逐条阅读判定顺序（enabled → 白名单 → sessionId → message → 限流）。

4. **实验四：追踪一个渠道请求如何收敛回内核** — 任选 Telegram 或移动端，从入口（`#runChatJob` 或 `handleMobileCommand` 的 `stream-chat` 分支）开始,沿调用链读到 `streamResponse`/`ApiChatHandler.streamChat`，标注出它在哪一步调用了与 Web 端相同的 `performSimilaritySearch`、`compressMessages`、`WorkspaceChats.new`；再对比 embed 走的 `streamChatWithForEmbed` + `EmbedChats.new`，总结"哪些是共享内核、哪些是渠道专属外壳"。

> **下一章预告**：MCP 让能力外扩、渠道让触点外延，但越是开放，安全边界就越关键。第 12 章我们将收束全书，解构 AnythingLLM 的安全与加密设计——token 如何加密存储、密钥从何而来、敏感配置如何被保护——并在此基础上提炼贯穿整个 RAG 系统的工程原则，为《解构 RAG 系统》第二部画上句号。
