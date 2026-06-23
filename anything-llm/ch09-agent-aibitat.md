# 第 9 章 Agent 系统:aibitat 框架的运转

到目前为止,我们讨论的 AnythingLLM 还是一个"问答机器人"——用户提问,系统检索文档、拼接上下文、调用 LLM 生成回答,一来一回即结束。但当用户在输入框里敲下 `@agent 帮我查一下今天的新闻并存到记忆里`,系统会切换到一条完全不同的执行路径:不再是一次性回答,而是进入一个**可以自主调用工具、根据工具结果继续决策、循环往复直到任务完成**的循环。这个循环背后的引擎,就是 AnythingLLM 自研的轻量多智能体框架 `aibitat`。

本章解构 `aibitat` 的全貌:`@agent` 如何被识别并触发、框架由哪三类节点(`agents` / `channels` / `functions`)构成、agent loop 如何在"LLM 决策 → 工具调用 → 结果回灌 → 再决策"之间循环、插件(工具)遵循怎样的定义契约、几个代表性插件如何落地、function calling 在支持原生工具调用与不支持的 provider 之间如何分流适配,以及整条链路如何通过 WebSocket 把每一步实时推给前端。所有结论都来自 `server/utils/agents/` 目录下的真实源码。

## 一、`@agent` 如何被识别并触发

agent 的入口判定不在 `aibitat` 内部,而在更外层的 `AgentHandler`(`server/utils/agents/index.js`)。它的静态方法 `isAgentInvocation({ message, workspace, chatMode })` 决定一条普通聊天消息是否应该跳转到 agent 路径。判定逻辑分两支:第一支是显式触发,调用私有静态方法 `#isAgentCommandInvocation({ message })`,它进一步委托给 `WorkspaceAgentInvocation.parseAgents(message)`;第二支是隐式触发,当 `chatMode === "automatic"` 且 `await Workspace.supportsNativeToolCalling(workspace)` 为真时,即便用户没有打 `@agent` 也会进入 agent 模式。

`parseAgents` 的实现非常直白,位于 `server/models/workspaceAgentInvocation.js`:

```js
parseAgents: function (promptString) {
  if (!promptString.startsWith("@agent")) return [];
  return promptString.split(/\s+/).filter((v) => v.startsWith("@"));
},
```

只要消息以字面量 `@agent` 开头,就返回所有 `@` 开头的 handle 数组,数组非空即视为 agent 调用。注释里那句 `must start with @agent for now.` 暴露了设计现状:目前只有一个 workspace 级别的内置 agent handle,即 `@agent`,多 agent handle 是预留能力。

被判定为 agent 调用后,系统会通过 `WorkspaceAgentInvocation.new()` 在数据库表 `workspace_agent_invocations` 里创建一条记录,生成一个 `uuid`。这个 `uuid` 就是后续 `AgentHandler` 构造函数 `constructor({ uuid })` 的唯一参数,agent 的整个生命周期都围绕这条 invocation 记录展开。前端随后用这个 uuid 建立 WebSocket 连接,服务端实例化 `AgentHandler`,依次 `await handler.init()` → `await handler.createAIbitat({ socket })` → `handler.startAgentCluster()`,真正点火。

值得注意的是 `AgentHandler` 在启动集群前会调用 `#stripAgentCommand(message)`,用正则 `/^@agent\s*/` 把消息开头的 `@agent` 指令剥掉再喂给模型。注释解释了原因:防止模型看到 `@agent` 后产生幻觉,误以为自己是某个 agent。如果剥离后内容为空(用户只打了个 `@agent`),则默认替换为 `"Hello!"`,当作打招呼处理。

## 二、aibitat 的三类节点:agents、channels、functions

`aibitat` 的核心是 `server/utils/agents/aibitat/index.js` 里的 `AIbitat` 类。类头部的 JSDoc 把它定位为"管理多个 agent 之间对话的类",并强调它通过"agent 构成的图"来引导聊天(`Guiding the chat through a graph of agents`)。这张图由三个 `Map` 承载,它们是理解整个框架的钥匙:

```js
agents = new Map();
channels = new Map();
functions = new Map();
```

`agents` 存放参与对话的节点,每个节点有一份配置(`role`、`functions`、`interrupt` 等)。`channels` 存放"群聊"——多个 agent 组成的频道,框架会用 LLM 在频道成员里选下一个发言者。`functions` 存放所有可被调用的工具。三者通过同名方法注册:`agent(name, config)`、`channel(name, members, config)`、`function(functionConfig)`,每个方法都把配置塞进对应的 `Map` 并 `return this` 以支持链式调用。

在 AnythingLLM 的实际运行里,真正注册到 `agents` 的只有两个,定义在 `server/utils/agents/defaults.js`:`USER_AGENT`(name 为 `"USER"`)和 `WORKSPACE_AGENT`(name 为 `"@agent"`)。`USER_AGENT` 的配置里 `interrupt: "ALWAYS"`,角色描述是"I am the human monitor and oversee this chat",它代表人类监督者;`WORKSPACE_AGENT` 才是真正干活的 LLM 节点。`AgentHandler.#loadAgents()` 把这两个 agent 注册进去:

```js
this.aibitat.agent(USER_AGENT.name, userAgentDef);
this.aibitat.agent(WORKSPACE_AGENT.name, workspaceAgentDef);
```

`startAgentCluster()` 则发起首条消息,`from: USER_AGENT.name`,`to: this.channel ?? WORKSPACE_AGENT.name`。也就是说,默认场景是 `USER` 与 `@agent` 之间的一对一直接对话,`channels` 机制在内置流程里并未启用——它是为多 agent 群聊预留的更通用能力。`getAgentConfig(agent)` 在取配置时还会兜底一个默认 `role`:`"You are a helpful AI assistant."`,确保任何 agent 都有系统提示词。

`functions` 的注册则发生在插件 `setup` 阶段。每个插件通过 `aibitat.use(plugin)` 安装,`use` 方法只做一件事:`plugin.setup(this)`。插件在 `setup(aibitat)` 里调用 `aibitat.function({...})` 把自己的工具登记到 `functions` Map,key 是工具的 `name`。这就是工具"注册"的全部机制——没有装饰器、没有反射,就是往一个 Map 里塞条目。

## 三、对话调度:start、chat 与 reply

点火从 `start(message)` 开始:它先 `newMessage(message)` 把首条消息写入历史并触发 `start` 事件,然后调用 `chat({ to: message.from, from: message.to })`。注意这里 `to`/`from` 做了互换——因为接下来要让"被发送方"回复"发送方"。

`chat(route, keepAlive = true)` 是调度中枢,它先判断 `route.from` 是不是一个 channel。如果是 channel,就走群聊分支:用 `selectNext(route.from)` 选出下一个发言者,检查是否达到 `maxRounds`、是否需要中断,然后递归 `chat`。`selectNext` 的实现很有意思——它把所有可用成员的角色描述和聊天历史拼成一段提示词,让 LLM 充当"导演"返回下一个该发言的角色名(`Only return the role.`),解析失败则随机挑一个。这是典型的"LLM 驱动的多 agent 编排",但如前所述,内置流程并不走这条分支。

内置流程走的是直接消息分支:调用 `reply(route)` 拿到回复,然后判断终止条件:

```js
if (
  reply === "TERMINATE" ||
  this.hasReachedMaximumRounds(route.from, route.to)
) {
  this.terminate(route.to);
  return;
}
```

字面量 `"TERMINATE"` 是 agent 主动结束对话的暗号,`hasReachedMaximumRounds` 则以 `maxRounds`(默认 100)兜底防止无限对话。若回复是 `"INTERRUPT"` 或 agent 配置要求中断(`shouldAgentInterrupt`),则调用 `interrupt` 把控制权交还人类;否则在 `keepAlive` 为真时继续递归 `chat(newChat)`,让对话在 `USER` 与 `@agent` 之间来回流动。

`reply(route)` 是连接"对话层"与"执行层"的关键。它做了几件事:取出 `route.from` 的 agent 配置,把历史格式化成 `messages`(系统提示词 + 对话历史),通过回调 `fetchParsedFileContext()` 把解析过的文件、置顶文档实时注入到最后一条 user 消息里,再根据 agent 配置里的 `functions` 名单从 `this.functions` Map 里取出真正的工具对象:

```js
let functions = fromConfig.functions
  ?.map((name) => this.functions.get(this.#parseFunctionName(name)))
  .filter((a) => !!a);
```

`#parseFunctionName` 处理了工具命名的两种特殊形态:含 `#` 的子工具(如 `sql-agent:list-database-connections`,取 `#` 之后部分),以及 `@@` 前缀的自定义插件(去掉 `@@`)。取出工具后,如果开启了 `ToolReranker`(智能技能选择),还会按用户 prompt 对工具重排序,只把最相关的若干个喂给模型,以削减 token 占用——源码注释里给出的数据是最高可省 80% 的工具调用 token。

`reply` 的最后一步是关键的分流:它根据 `this.providerInstance.supportsAgentStreaming` 决定走 `handleAsyncExecution`(流式)还是 `handleExecution`(同步)。这正是下一节 agent loop 的两个并行实现。

## 四、agent loop:决策、调用、回灌、再决策

agent loop 的本质是一个**带深度计数的递归函数**。我们以同步版 `handleExecution(messages, functions, byAgent, depth = 0, msgUUID = null)` 为例(流式版 `handleAsyncExecution` 结构几乎一致,只是把 `complete` 换成 `stream`)。

每一轮,它先调用 provider 拿到一次补全:

```js
const completion = await this.#safeProviderCall(() =>
  this.providerInstance.complete(messages, functions)
);
```

`#safeProviderCall` 是个错误包装器,把任何 provider 异常统一转成 `APIError`,提示"The agent model failed to respond",避免 provider 报错直接把 agent 崩掉。

拿到 `completion` 后,分两种情况:

**情况一:模型决定调用工具。** 当 `completion.functionCall` 存在时,从中取出 `name` 与 `arguments`,在 `this.functions` 里查到对应工具对象 `fn`。这里有几道关键处理:

- 工具调用上限检查:`const reachedToolLimit = depth >= this.maxToolCalls;`。`maxToolCalls` 默认由静态方法 `AIbitat.defaultMaxToolCalls()` 决定,读取环境变量 `AGENT_MAX_TOOL_CALLS`,无效时回退为 `10`。达到上限时不会立刻停,而是日志与 `introspect` 提示"执行完这次工具后将生成最终回答",并在下一轮递归时把 `functions` 传成空数组 `[]`,逼模型用纯文本收尾。
- 工具不存在的兜底:若 `!fn`,不报错,而是把一条 `role: "function"`、内容为 `Function "${name}" not found. Try again.` 的消息追加进历史,递归继续,给模型自我纠错的机会。
- 真正执行:`const result = await fn.handler(args);`。执行后发送遥测 `Telemetry.sendTelemetry("agent_tool_call", { tool: name })`,并 `emitter.emit("toolCallResult", {...})` 触发事件(定时任务用它捕获执行轨迹)。

工具结果回灌是 loop 的灵魂:它把工具返回值包成一条 `role: "function"` 的消息,连同原始调用一起追加到 `messages`,然后**递归调用自己**,`depth + 1`:

```js
const newMessages = [
  ...messages,
  {
    name,
    role: "function",
    content: result,
    originalFunctionCall: completion.functionCall,
  },
];
return await this.handleExecution(
  newMessages,
  reachedToolLimit ? [] : functions,
  byAgent,
  depth + 1,
  msgUUID
);
```

`originalFunctionCall` 字段必须随结果回传——后面会看到 OpenAI provider 要靠它把工具输出映射回正确的 `call_id`。如果工具在执行中通过 `addToolAttachment` 塞了图片,`collectToolAttachments()` 会把它们作为一条新的 user 消息(`[Attached image(s) from tool result]`)注入,从而复用 provider 既有的多模态处理。

这里还有一个"直接输出"短路机制:实例字段 `skipHandleExecution`。某些工具(尤其是 Flow 执行)希望把结果原样返给用户、不再经过 LLM 二次加工、也不再触发后续工具链。当工具把 `skipHandleExecution` 置为 `true`,loop 会立即把 `result` 当作最终文本返回,并通过事件 `fullTextResponse` 推给前端,然后复位该标志。类头部的注释明确说这是"防止 tool-call chaining"的临时机制。

**情况二:模型给出纯文本回答。** 当 `completion.functionCall` 为空,说明模型认为任务已完成,loop 终止递归:发送 `usageMetrics` 事件、`flushCitations(msgUUID)` 把整轮累积的引用刷给前端、`emitChatId`,最后 `return completion?.textResponse`。这个文本一路返回到 `reply`,再到 `chat`,如果不是 `TERMINATE` 就继续与 `USER` 的对话回合。

这样,"LLM 决策 → 工具调用 → 结果回灌 → 再决策"的循环,就被优雅地实现为一个对自身的尾递归:每次递归要么因工具调用而加深一层,要么因纯文本回答而触底返回,`depth` 与 `maxToolCalls` 共同保证它不会无限下钻。

`aibitat` 还提供了 `continue(feedback, attachments)` 与 `retry()` 两个恢复入口:前者在 `interrupt` 之后凭人类反馈续接对话,后者在 `state === "error"` 时弹出错误记录重试。它们配合 `getHistory` 对 `state`(`"success"` / `"interrupt"` / `"error"`)的过滤,构成了一套有状态的对话恢复能力。

## 五、插件(工具)的定义契约

所有内置工具都登记在 `server/utils/agents/aibitat/plugins/index.js`,它把 `web-browsing`、`web-scraping`、`memory`、`rag-memory`、`sql-agent`、`filesystem`、`gmail`、`request-user-input` 等模块统一导出,并额外用 `[webBrowsing.name]: webBrowsing` 这样的计算属性建立"按 slug 取插件"的别名,方便 `defaults.js` 与 `AgentHandler` 双向查找。

一个插件本质上是一个对象,具有三个顶层字段:`name`(slug)、`startupConfig`(启动参数声明,内含 `params`)、`plugin`(返回真正可安装对象的工厂函数)。工厂函数返回的对象有一个 `setup(aibitat)` 方法,在其中调用 `aibitat.function({...})` 声明工具。以 `web-browsing.js` 为例,这份"工具契约"包含:

- `name`:工具标识,如 `"web-browsing"`,即 LLM 在 function calling 里看到的函数名。
- `description`:自然语言描述,直接决定模型何时选用它。web-browsing 的描述是"Search the internet for real-time information...",刻意强调"实时""本地没有"的语义,引导模型只在需要联网时调用。
- `examples`:few-shot 示例数组,每条含 `prompt` 与 `call`(JSON 字符串化的参数)。这些示例对原生支持工具调用的模型可有可无,但对**不支持**工具调用的模型至关重要(见第七节)。
- `parameters`:JSON Schema(`$schema: "http://json-schema.org/draft-07/schema#"`),声明入参类型、枚举、必填。这正是要交给 OpenAI 等 provider 的 `parameters` 字段。
- `handler`:`async function(args)`,工具的真正逻辑。它内部用 `this.super` 访问 `aibitat` 实例,从而能调用 `this.super.introspect(...)` 推送思考过程、`this.super.addCitation(...)` 累积引用、`this.super.handlerProps.invocation.workspace` 拿到工作区上下文。

`handler` 用普通 `function`(而非箭头函数)声明,是为了让 `this` 绑定到 `aibitat.function({...})` 传入的那个配置对象——配置对象里还可以挂载 `search`、`store`、`_serpApi` 等辅助方法,它们都通过 `this` 互相调用。这是一种把"工具状态 + 辅助方法 + handler"打包进同一个 `this` 上下文的轻量封装风格。

工具执行前还可能经过一道授权关卡。WebSocket 插件给 `aibitat` 注入了 `requestToolApproval({ skillName, payload, description })` 方法:若技能未被 `AGENT_AUTO_APPROVED_SKILLS` 自动批准、也不在 `AgentSkillWhitelist` 白名单里,就向前端发 `toolApprovalRequest` 并阻塞等待,默认 `TOOL_APPROVAL_TIMEOUT_MS`(2 分钟)超时则视为拒绝。这让"危险工具需人工确认"成为框架级能力。

## 六、代表性插件解剖

**memory(rag-memory):本地检索与长期记忆。** `memory.js` 注册的工具 `name` 是 `"rag-memory"`,描述把"搜索本地文档"与"存入长期记忆"两件事合并成一个工具,通过参数 `action` 的枚举 `["search", "store"]` 区分。`handler` 先用一个 `Deduplicator`(`this.tracker.isDuplicate`)拦截重复调用,返回"This was a duplicated call and it's output will be ignored.",防止模型在一轮里反复搜同一内容。`search` 分支解析出 workspace,拿到 LLM 连接器,调用 `getVectorDbClass().performSimilaritySearch({ namespace: workspace.slug, input, topN, rerank })`——这正是把第七章讲过的 RAG 检索能力,以工具形式暴露给 agent。检索到的 `sources` 通过 `this.super.addCitation?.(sources)` 进入引用缓冲区;若无结果,返回的文案是"...We should search the web for this information.",这等于在工具结果里给模型一个"下一步去联网"的暗示,体现了工具链的协同设计。`store` 分支则把文本作为一篇文档 `addDocumentToNamespace` 写进向量库,`docAuthor: "@agent"`,实现"让 agent 记住一件事"。

**web-browsing:多搜索引擎的统一门面。** `web-browsing.js` 是一个典型的"适配器聚合"插件。工具只暴露一个 `query` 参数,但 `search` 方法内部读取系统设置 `agent_search_provider`,用一个大 `switch` 把它映射到具体引擎方法名(`_serpApi`、`_googleSearch` 之外还有 `_bingWebSearch`、`_baiduSearch`、`_tavilySearch`、`_duckDuckGoEngine` 等十余种),默认回退到 `_duckDuckGoEngine`(它直接抓 HTML 解析,无需 API key)。每个引擎方法的模式高度一致:检查对应的 `AGENT_*_API_KEY` 环境变量是否配置,未配置则 `introspect` 提示用户去注册并返回"功能未启用"的文案;配置了就发请求、归一化成 `{ title, link, snippet }` 数组、调用 `reportSearchResultsCitations` 累积引用、`countTokens` 估算结果 token 量并 `introspect` 给前端、最后 `JSON.stringify` 返回。这种"一个工具门面 + N 个可插拔后端"的结构,让换搜索引擎只是改一个设置项,而模型看到的工具契约始终不变。

**sql-agent:父子工具的拆分。** `sql-agent/index.js` 的 `plugin` 字段不是工厂函数,而是一个**数组**:`[SqlAgentListDatabase, SqlAgentListTables, SqlAgentGetTableSchema, SqlAgentQuery]`。这是框架的另一种插件形态——一个父技能拆成多个子工具。`defaults.js` 的 `agentSkillsFromSystemSettings()` 检测到 `Array.isArray(AgentPlugins[skillName].plugin)` 时,会用 `${parent}#${child}` 命名约定逐个展开,如 `sql-agent#sql-query`,而 `AgentHandler.#attachPlugins` 再按 `name.split("#")` 把每个子工具单独 `use` 进集群。以子工具 `sql-query` 为例,它的 `description` 反复强调"read-only""must only be SELECT""reasonable LIMIT",`parameters` 要求 `database_id` 与 `sql_query`,`handler` 先按 `database_id` 找到连接配置、`getDBClient` 建连、`runQuery` 执行并把错误友好化返回。把"列库、列表、看 schema、执行查询"拆成四个原子工具,而非塞进一个万能工具,正是为了让模型能分步推理、逐步收敛到正确的 SQL。

## 七、function calling 在不同 provider 下的适配

`aibitat` 支持四十余种 provider(见 `getProviderForConfig` 那个长 `switch`),它们对 function calling 的支持程度天差地别。框架用两层机制抹平差异。

第一层是**流式与非流式的分流**。provider 基类 `ai-provider.js` 暴露一个 getter `supportsAgentStreaming`,默认返回 `false`;支持的 provider(如 `openai.js` 里覆写返回 `true`)走 `handleAsyncExecution`(调用 `provider.stream`),不支持的走 `handleExecution`(调用 `provider.complete`)。这就是 `reply` 末尾那个 `if (this.providerInstance.supportsAgentStreaming)` 分支的来源。

第二层,也是更本质的,是**原生工具调用与"无工具模拟"的分流**。以 OpenAI provider 为例,它实现了 `supportsNativeToolCalling()` 返回 `true`,在 `stream` / `complete` 里直接用 OpenAI 的 Responses API:`#formatFunctions(functions)` 把每个工具映射成 `{ type: "function", name, description, parameters, strict: false }` 交给 API 的 `tools` 字段,设置 `parallel_tool_calls: false` 强制串行;`#formatToResponsesInput` 则把内部的 `role: "function"` 消息拆成 `function_call` + `function_call_output` 两条,靠 `originalFunctionCall` 里的 `call_id` 把调用与输出配对——这正是前面强调"结果回灌时必须带上 `originalFunctionCall`"的原因。流式时,它监听 `response.output_item.added`(类型为 `function_call`)、`response.function_call_arguments.delta` 等事件,边收边拼出 `functionCall.arguments`,同时把 `toolCallInvocation` 事件推给前端展示"正在组装工具调用"。最终它把结果统一成 `aibitat` 期望的形状:`{ textResponse, functionCall: { id, name, arguments } }`。错误处理上,认证错误直接抛出,限流/服务端错误则包成 `RetryError` 交由上层重试。

对于不支持原生工具调用的模型(很多本地 / 开源模型),框架提供了 `providers/helpers/untooled.js` 里的 `UnTooled` 基类做"提示词模拟工具调用"。它的 `showcaseFunctions(functions)` 把每个工具的 `name`、`description`、`parameters.properties` 以及 `examples` 拼成一大段文本喂进系统提示——这就是为什么插件契约里的 `examples` 字段在这种场景下变得不可或缺。模型被要求输出一段 JSON,`functionCall(messages, functions)` 再用 `validFuncCall` 校验这段 JSON 是否含合法的 `name` 与 `arguments`、参数是否齐全;校验通过才算一次工具调用。`UnTooled` 还内置 `Deduplicator` 与针对 MCP 工具的冷却机制(`isMCPTool`),因为某些 MCP 工具不返回值,容易让无工具模型陷入反复调用同一函数的死循环,除非设置 `MCP_NO_COOLDOWN`。如此,无论底层模型有没有"工具调用"这项原生能力,`aibitat` 上层的 agent loop 看到的都是同一个 `{ functionCall, textResponse }` 抽象,完全无感。

## 八、WebSocket:实时交互的脉搏

agent 与一问一答聊天最大的体验差异,是它会把"内心戏"实时播报出来。这套机制由 `websocket.js` 插件提供。它在 `setup(aibitat)` 里给 `aibitat` 实例挂上几个能力:`aibitat.introspect(messageText)` 把思考过程以 `type: "statusResponse"` 推给前端(`animate: true` 让前端显示"打字"动效);`aibitat.socket.send(type, content)` 是通用发送器,注释特别提醒"type param must be set or else msg will not be shown",前端按 `type` 分发渲染。

agent loop 里那些 `this?.introspect?.(...)`、`eventHandler?.("reportStreamEvent", {...})` 调用,最终都落到这个 socket 上。具体的事件类型包括:`textResponseChunk`(流式文本增量)、`toolCallInvocation`(工具调用组装中)、`fullTextResponse`(直接输出短路时的整段结果)、`usageMetrics`(token 用量)、`citations`(由 `flushCitations` 推送的引用列表)、`chatId`、`statusResponse`(introspect)。前端据此既能渲染流式回答,也能展示"agent 正在用 web-browsing 搜索…""找到 5 条结果"这样的过程提示。

WebSocket 还承载**双向阻塞式交互**。除了前面提过的 `requestToolApproval`(工具授权),还有 `requestUserClarification({ questions, allowSkip, timeoutMs })`:它发一个 `clarificationRequest` 卡片并返回一个 Promise,挂起整个 loop 直到用户在前端填完表单、跳过、或 `CLARIFICATION_DEFAULT_TIMEOUT_MS`(2 分钟)超时。这就是 `request-user-input` 技能的底座——`defaults.js` 在开启 `agent_clarifying_questions_enabled` 时,会往系统提示词里追加一句"You MUST use the request-user-input tool... the user cannot reply to text",强制模型用工具而非纯文本提问。`onInterrupt` 则在 agent 主动中断时调用 `socket.askForFeedback`,把控制权交还人类,并识别 `WEBSOCKET_BAIL_COMMANDS`(`exit`、`/stop`、`/reset` 等)来终止会话。所有这些等待都设了超时(普通反馈是 `SOCKET_TIMEOUT_MS`,5 分钟),避免前端断线后服务端线程永久挂起。

另一个并行挂载的 `chat-history` 插件则负责把 agent 的最终回复、引用、补充问卷调查等持久化到 `workspace_chats`,它会在持久化后调用 `clearCitations` / `clearClarifyingQuestionSurveys` 清空缓冲,使下一轮干净开始。`introspect` 是给人看的过程,`chat-history` 是给数据库存的结果,两个插件一显一存,共同构成了 agent 的完整可观测闭环。

## 本章小结

1. `@agent` 的识别由 `AgentHandler.isAgentInvocation` 完成,显式触发依赖 `WorkspaceAgentInvocation.parseAgents`(消息须以字面量 `@agent` 开头),`automatic` 模式 + 原生工具调用支持可隐式触发;`#stripAgentCommand` 会在喂给模型前剥掉 `@agent` 指令以防幻觉。
2. `AIbitat` 类用 `agents`、`channels`、`functions` 三个 `Map` 描述一张"agent 图";内置流程只注册 `USER` 与 `@agent` 两个 agent,走一对一直接对话,`channels` 群聊与 `selectNext` 的 LLM 选角能力是预留的更通用机制。
3. 工具注册没有魔法:插件经 `aibitat.use(plugin)` → `plugin.setup(this)` → `aibitat.function({...})`,把工具按 `name` 塞进 `functions` Map。
4. agent loop 由 `handleExecution`(同步)/`handleAsyncExecution`(流式)的**带 `depth` 计数的递归**实现:有工具调用就回灌结果并 `depth + 1` 递归,纯文本回答则触底返回;`maxToolCalls`(默认 10,可由 `AGENT_MAX_TOOL_CALLS` 覆盖)与 `maxRounds`(默认 100)共同防止失控。
5. 结果回灌时工具输出被包成 `role: "function"` 消息并携带 `originalFunctionCall`,后者用于在 OpenAI Responses API 里用 `call_id` 把调用与输出配对。
6. `skipHandleExecution` 提供"直接输出"短路,把工具结果原样返回前端、不再触发后续工具链,主要服务于 Flow 执行。
7. 插件契约由 `name`、`description`、`examples`、`parameters`(JSON Schema)、`handler` 五要素构成;`handler` 用普通函数声明以便通过 `this.super` 访问 `aibitat`(`introspect`、`addCitation`、`handlerProps.invocation`)。
8. 插件有两种形态:单一工具(`plugin` 是工厂函数)与父子工具(`plugin` 是数组,如 `sql-agent`),后者按 `${parent}#${child}` 命名展开并逐个安装。
9. function calling 通过两层分流适配:`supportsAgentStreaming` 决定流式/同步;原生工具调用(OpenAI 等)直接用 API 的 `tools` 字段,不支持的模型用 `UnTooled` 把工具契约与 `examples` 写进提示词来模拟,上层 loop 始终面对统一的 `{ functionCall, textResponse }` 抽象。
10. `web-browsing` 用"一个门面 + 十余个可插拔搜索后端"结构,靠系统设置 `agent_search_provider` 切换,默认回退 `_duckDuckGoEngine`;`memory` 把 RAG 检索与长期记忆合进一个 `action` 枚举工具。
11. WebSocket 插件赋予 agent 实时可观测性与双向交互:`introspect`/`reportStreamEvent` 播报过程,`requestToolApproval`/`requestUserClarification` 阻塞式征求人类授权与澄清,全部带超时兜底;`chat-history` 插件负责把结果与引用持久化。

## 动手实验

1. **实验一:追踪一次完整的工具调用回灌** — 在 `server/utils/agents/aibitat/index.js` 的 `handleExecution` 里,于 `const result = await fn.handler(args);` 之后加一行临时日志,打印 `name`、`depth`、`result.slice(0, 200)`。用 `@agent 搜一下今天的科技新闻` 触发一次 agent 会话,观察控制台里 `depth` 如何从 0 递增、`functions` 在达到 `maxToolCalls` 后如何被置空,亲眼确认"回灌即递归"。

2. **实验二:写一个最小自定义工具插件** — 仿照 `web-browsing.js` 的结构,在 `plugins/` 下新建一个 `name` 为 `"current-time"` 的插件,`parameters` 留空对象,`handler` 返回 `new Date().toISOString()`,并在 `plugins/index.js` 里导出。把它加入 `default_agent_skills` 系统设置后用 `@agent 现在几点` 调用,验证从注册、契约声明到被模型选中调用的完整链路;再在 `handler` 里调用 `this.super.introspect("正在读取系统时钟…")`,观察前端是否实时出现该状态。

3. **实验三:对比原生与模拟两种 function calling** — 先用 OpenAI provider 跑一次 agent,在 `openai.js` 的 `#formatFunctions` 里打印交给 API 的 `tools` 数组;再切换到一个走 `UnTooled` 的本地模型 provider,在 `untooled.js` 的 `showcaseFunctions` 里打印拼接出的提示词文本。对照两份输出,理解同一个插件契约(尤其 `examples` 字段)在两条路径下分别被如何使用。

4. **实验四:体验直接输出短路与工具上限** — 临时把某个工具(如 `sql-query`)的 `handler` 在返回前设置 `this.super.skipHandleExecution = true;`,再触发一次会调用它的查询,观察 agent 是否把工具原始结果直接返回前端、且不再继续任何工具链;随后把环境变量 `AGENT_MAX_TOOL_CALLS` 设为 `1` 重启,用一个需要多步工具(列库→看 schema→查询)的任务测试,确认 agent 在第一次工具后即被强制以纯文本收尾。

> **下一章预告**:`@agent` 调用的工具不止于内置技能与 MCP,管理员还能在可视化画布上把若干步骤(LLM 调用、API 请求、变量传递、条件分支)拖拽编排成一条 **Agent Flow**,再以 `@@flow_<uuid>` 的形式注册成一个普通工具供 agent 调用。下一章我们就深入 `AgentFlows`,解构这套"无代码工作流"如何被定义、加载为插件、并在 `aibitat` 的 agent loop 里逐节点执行。
