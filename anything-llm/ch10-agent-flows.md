# 第 10 章 Agent Flows:无代码可视化工作流编排

上一章我们解构了 aibitat 这套多智能体内核,看到了一个 agent 如何在 `reply()` 循环里挑选工具、调用 LLM、把结果回灌进对话。aibitat 的工具体系强大却也有门槛:每一个新工具都意味着一段 JavaScript,意味着开发者要懂 `aibitat.function({...})` 的注册协议。AnythingLLM 想把这种能力交到不写代码的人手里——让运营、产品、客服也能把"先调一个外部接口、再让 LLM 总结、最后抓一个网页核对"这样的多步骤逻辑组装成一个可复用的"技能"。这就是 Agent Flows 要解决的问题:用户在前端拖拽出一串节点,后端把这串节点保存成一份声明式配置,并在 agent 运行时把整份配置伪装成一个普通的 aibitat 工具。

本章沿着源码自底向上展开:先看清 flow 在数据层面到底是什么(`server/utils/agentFlows/` 这一目录里的几个文件),再看节点类型体系 `FLOW_TYPES`,接着拆 `FlowExecutor` 如何顺序执行、如何在步骤之间累积变量,然后逐个解剖三种内置执行器,最后回答两个工程关键问题:一份 flow 怎么被封装成 agent 可以调用的 function,以及它在磁盘上以什么格式存放。整个模块代码量不大——`executor.js` 约 236 行、`index.js` 约 289 行、三个执行器加起来不到 220 行——但它把"声明式配置"与"命令式执行"之间的桥接做得相当完整,是研究"低代码编排"如何落地的一个干净样本。

## 一份 flow 到底是什么:声明式的步骤序列

理解 Agent Flows 的第一把钥匙,是接受"flow 不是代码,而是数据"这个事实。在 `server/utils/agentFlows/index.js` 顶部的 JSDoc 里,`@typedef {Object} LoadedFlow` 把一份被加载后的 flow 描述得很清楚:它有一个 `name`(flow 的名字)、一个 `uuid`(唯一标识)、以及一个 `config` 对象;`config` 里装着 `description`(描述)和最关键的 `config.steps`——一个数组,数组的每一项是一个步骤,"Each step has at least a type and config"(每个步骤至少有 `type` 和 `config` 两个字段)。

这个结构透露出整套系统的设计哲学:flow 是一个**线性的步骤序列**(`steps` 是数组而非图),每一步只声明"我是什么类型"(`type`)和"我的参数是什么"(`config`),至于"怎么执行",由后端按 `type` 分派给对应的执行器。换句话说,前端拖拽出来的画布,序列化下来不过是一个 `{ name, description, steps: [...] }` 的 JSON。这种"前端画图、后端按数组顺序跑"的设计,既让可视化编辑器实现起来简单,也让执行引擎不必关心拓扑——它只需要从头到尾遍历 `steps`。值得强调的是,虽然界面是"画布"风格的拖拽,但执行模型是严格顺序的列表,数据靠变量在步骤间传递,而不是靠节点之间画连线(后文会看到变量是如何充当这些"隐形连线"的)。

## 节点类型体系:flowTypes 里的四个常量

flow 里允许出现哪些类型的步骤,完全由 `server/utils/agentFlows/flowTypes.js` 中导出的 `FLOW_TYPES` 常量决定。这个对象是整套系统的"类型注册表",它列举了四种节点:

```js
const FLOW_TYPES = {
  START: { type: "start", description: "Initialize flow variables", ... },
  API_CALL: { type: "apiCall", description: "Make an HTTP request to an API endpoint", ... },
  LLM_INSTRUCTION: { type: "llmInstruction", description: "Process data using LLM instructions", ... },
  WEB_SCRAPING: { type: "webScraping", description: "Scrape content from a webpage", ... },
};
```

注意每个键(如 `API_CALL`)是给代码读的常量名,而每个对象里的 `type` 字段(如 `"apiCall"`)才是实际写进 flow JSON 的字面量、也是 `FlowExecutor` 用来分派的依据。这种"常量名"与"字面量"分离的写法,让代码里可以安全地引用 `FLOW_TYPES.API_CALL.type` 而不必硬编码字符串 `"apiCall"`。

`FLOW_TYPES` 不只是给类型起名,它还把每种节点的参数 schema 一并声明了出来,这一点对理解前端表单和后端校验都很关键。`START` 节点的参数只有一个 `variables`(类型 `array`,"List of variables to initialize"),负责给整个 flow 注入初始变量。`API_CALL` 节点的参数最丰富:`url`、`method`(HTTP method)、`headers`(键值对数组)、`bodyType`("Type of request body (json, form)")、`body`、`formData`,以及两个对数据流动至关重要的字段——`responseVariable`("Variable to store the response",把响应存进哪个变量)和 `directOutput`("Whether to return the response directly to the user without LLM processing",是否绕过 LLM 直接把响应返回给用户)。`API_CALL` 还附带了一个 `examples` 字段,给出一个 GET 请求的样例,这通常是为了让 LLM 在生成或校验 flow 配置时有参照。`LLM_INSTRUCTION` 节点参数很简洁:`instruction`(给 LLM 的指令)和 `resultVariable`(把处理结果存进哪个变量)。`WEB_SCRAPING` 节点则有 `url`、`resultVariable` 和同样的 `directOutput`。

把这四个节点放在一起看,可以归纳出一条贯穿全部节点的约定:几乎每个有输出的节点都提供一个"把我的产出存进某个变量"的字段(`responseVariable` 或 `resultVariable`),而 `apiCall` 与 `webScraping` 还额外提供 `directOutput`,允许它们在拿到结果后直接"短路"整个 flow。这两类字段——存变量、直出——正是 `FlowExecutor` 实现数据流动与提前返回的抓手。

## FlowExecutor:如何顺序执行并累积变量

真正把声明式配置跑起来的是 `server/utils/agentFlows/executor.js` 里的 `FlowExecutor` 类。它的构造函数初始化了几样东西:`this.variables = {}`(一个贯穿整个 flow 生命周期的变量袋)、`this.introspect`(一个用于把执行过程"自省"输出给用户的函数,默认走 `console.log`)、`this.logger`(日志函数)以及 `this.aibitat = null`(执行时由调用方注入的 aibitat 实例)。这里的 `this.variables` 是理解一切的核心:它是一个**实例级、跨步骤共享的可变状态**,后续每一步的输入解析和输出写入都围绕它进行。

入口方法是 `executeFlow(flow, initialVariables = {}, aibitat)`。它做的第一件事是初始化变量袋,而且初始化逻辑很有讲究:

```js
this.variables = {
  ...(
    flow.config.steps.find((s) => s.type === "start")?.config?.variables ||
    []
  ).reduce((acc, v) => ({ ...acc, [v.name]: v.value }), {}),
  ...initialVariables, // This will override any default values with passed-in values
};
```

这段代码先从 `steps` 里找到那个 `type === "start"` 的节点,把它声明的 `variables` 数组摊平成 `{ 名字: 默认值 }` 的形式作为底座,然后用传进来的 `initialVariables` 覆盖之。注释里写得明白:传入值会覆盖默认值。这意味着 `start` 节点定义的是"变量的默认值",而 agent 调用 flow 时真正传进来的参数会优先生效——这正是把 flow 当工具调用时参数注入的入口。

随后 `executeFlow` 把 `aibitat` 挂到实例上,并调用 `this.attachLogging(aibitat?.introspect, aibitat?.handlerProps?.log)`,把执行过程中的 `introspect`/`logger` 接到真实 agent 的自省通道上——这就是为什么 flow 跑起来时,用户能在 agent 对话里看到"Making GET request to external API..."这类实时进度。接着是整个引擎的心脏,一个朴素的 `for` 循环:

```js
for (const step of flow.config.steps) {
  try {
    const result = await this.executeStep(step);
    if (result?.directOutput) {
      directOutputResult = result.result;
      break;
    }
    results.push({ success: true, result });
  } catch (error) {
    results.push({ success: false, error: error.message });
    break;
  }
}
```

这段循环有几个值得划重点的行为。其一,**严格顺序、串行 await**:步骤一个接一个执行,前一步不完成绝不会进入下一步,这正是变量得以在步骤间安全传递的前提。其二,**遇错即停**:任何一步抛异常,都会被 `catch` 记录为 `{ success: false, error }` 并 `break`,后续步骤不再执行——flow 没有分支、没有重试,失败就是终止。其三,**directOutput 短路**:一旦某步返回的结果带 `directOutput` 标志,引擎记下它的 `result` 后立即 `break`,跳过所有剩余步骤。最后,`executeFlow` 返回一个汇总对象:`{ success, results, variables, directOutput }`,其中 `success` 由 `results.every((r) => r.success)` 计算,`variables` 是 flow 跑完后变量袋的最终快照。

## 单步执行与变量写回:executeStep

循环体里调用的 `executeStep(step)` 是单步执行的封装,它做三件事:解析输入、分派执行、写回输出。第一行 `const config = this.replaceVariables(step.config)` 先把这一步声明的 `config` 里的所有 `${变量}` 占位符替换成变量袋里的当前值(替换机制下一节细说)。然后它构造一个 `context` 对象,把 `introspect`、`variables`、`logger`、`aibitat` 打包传给执行器——这就是各个 executor 拿到 aibitat、能调 LLM、能输出进度的来源。

分派靠一个 `switch (step.type)`:`start` 类型在执行器之外被特殊处理——它遍历 `config.variables`,对每个尚未在变量袋里的变量名 `v.name` 写入 `v.value || ""`,然后把整个 `this.variables` 作为结果返回;`apiCall`、`llmInstruction`、`webScraping` 分别 await 调用 `executeApiCall`、`executeLLMInstruction`、`executeWebScraping`;遇到未知类型则 `throw new Error(`Unknown flow type: ${step.type}`)`。

执行完毕后是写回环节,这是数据流动的另一半:

```js
if (config.resultVariable || config.responseVariable) {
  const varName = config.resultVariable || config.responseVariable;
  this.variables[varName] = result;
}
if (config.directOutput) result = { directOutput: true, result };
```

如果这一步的 `config` 声明了 `resultVariable` 或 `responseVariable`,引擎就把本步结果存进变量袋里对应的键(注意 `resultVariable` 优先于 `responseVariable`)。这就是节点产出"流入"后续步骤的唯一通道:**前一步写变量,后一步在自己的 `config` 里用 `${变量}` 读出来**。最后,如果 `config.directOutput` 为真,引擎把结果包成 `{ directOutput: true, result }`——正是这个包裹让上层 `executeFlow` 的循环识别出"该短路了"。

## 变量替换:步骤间数据如何流动

如果说变量袋是数据流动的"管道",那么 `replaceVariables` 与它依赖的 `getValueFromPath` 就是"接头"。`replaceVariables(config)` 的核心是一个递归的 `deepReplace`:对字符串,用正则 `/\${([^}]+)}/g` 匹配所有 `${...}` 占位符,把括号里的内容当作变量路径交给 `getValueFromPath` 求值,求得到就替换、求不到(返回 `undefined`)就保留原样;对数组逐项递归;对对象逐字段递归。这意味着变量替换不是只作用在某一个字段上,而是**深入整个 `config` 的每一个字符串叶子**——`url`、`headers` 的 value、`body`、`instruction` 里写的 `${...}` 都会被处理。

真正有意思的是 `getValueFromPath(obj, path)`,它支持点号与方括号混合的路径语法,文档里给的例子是 `"data.items[0].name"` 和 `"response.users[2].address.city"`。它没有偷懒用 `eval` 或简单 `split('.')`,而是手写了一个小解析器:逐字符扫描路径,用 `inBrackets` 状态位区分括号内外,把 `data.items[0].name` 切成 `["data", "items", "[0]", "name"]` 这样的片段序列;遇到 `[...]` 形式的片段时,去掉引号,若内容是数字就当数组下标(还会校验 `Array.isArray(current)`),否则当对象键;若路径中途遇到非对象或键不存在,就返回 `undefined`。还有一个容错细节:函数开头有 `if (typeof obj === "string") obj = safeJsonParse(obj, {})`,也就是说如果某个变量存的是一段 JSON 字符串(比如 API 返回的原始文本),引用它的子字段时会先尝试把它解析成对象再取值;末尾则有 `return typeof current === "object" ? JSON.stringify(current) : current`,取到的若是对象会被序列化成字符串再注入。

把这两段合起来看,步骤间的数据流动就完全清楚了:`apiCall` 把响应存进 `responseVariable`(比如叫 `apiResponse`),下一个 `llmInstruction` 的 `instruction` 里写 `请总结这段数据:${apiResponse.data.items[0].title}`,执行到这一步时 `replaceVariables` 会把这段路径解析出来、把值填进去,LLM 收到的就是已经填好真实数据的指令。没有连线,只有变量名约定——这是一种以"共享命名空间"为中心的数据流模型。

## 三种内置执行器解剖

`start` 是个纯内部节点,真正"干活"的是三个独立的执行器文件,它们都遵循 `async function(config, context)` 的统一签名,从 `context` 里取 `introspect`/`logger`/`aibitat`,返回一个结果(通常是字符串或可序列化对象)。

**API 调用节点(`executors/api-call.js`)**。`executeApiCall` 从 `config` 解构出 `url`、`method`、`headers`、`body`、`bodyType`、`formData`,先把 `headers` 数组 `reduce` 成一个普通对象塞进 `requestConfig.headers`。对于 `POST`/`PUT`/`PATCH` 这三种带体的方法,它按 `bodyType` 分三路处理:`form` 时用 `URLSearchParams` 拼成 `application/x-www-form-urlencoded`;`json` 时先 `safeJsonParse(body, null)` 校验合法性再 `JSON.stringify` 回去,并设 `Content-Type: application/json`;`text` 时直接 `String(body)`;其余情况原样传。请求用原生 `fetch` 发出,`!response.ok` 时抛 `HTTP error! status: ...`,成功则把响应体读成文本后再 `safeJsonParse(text, "Failed to parse output from API call block")`——拿不到合法 JSON 时返回那段提示字符串而非崩溃。整个过程穿插着 `introspect`("Making ... request to external API..."、"Sending body to ..."、"API call completed"),让用户实时看到外呼进展。

**LLM 指令节点(`executors/llm-instruction.js`)**。`executeLLMInstruction` 解构出 `instruction` 与 `resultVariable`,关键在于它复用了 aibitat 的 provider 体系:`const provider = aibitat.getProviderForConfig(aibitat.defaultProvider)`,也就是说 flow 里的 LLM 步骤用的就是当前 agent 配置的默认模型(日志里会打印 `aibitat.defaultProvider.provider::aibitat.defaultProvider.model`)。它把 `instruction` 强制规整成字符串(对象就 `JSON.stringify`),再根据 `provider.supportsAgentStreaming` 决定走 `provider.stream(...)` 还是 `provider.complete(...)`,最后返回 `completion.textResponse`。这一节点把"让 LLM 加工一段(往往是上一步 `${变量}` 填进来的)文本"变成一个原子步骤,而它对 provider 层的复用,正是第 8 章模型接入层与第 9 章 aibitat 在 flow 场景下的自然延伸。

**网页抓取节点(`executors/web-scraping.js`)**。`executeWebScraping` 是三者里最复杂的。它在函数体内 `require` 了 `CollectorApi`、`TokenManager`、`Provider` 和 `summarizeContent`,解构出 `url`、`captureAs`(默认 `"text"`)、`enableSummarization`(默认 `true`)。它把 `captureAs === "querySelector"` 映射成抓 `html`,然后用 `new CollectorApi().getLinkContent(url, captureMode)` 去抓取——这复用了第 2 章讲过的 collector 服务;若指定了 `querySelector`,再用本地的 `parseHTMLwithSelector` 函数借助 `cheerio` 按 CSS 选择器抽取片段(命中 0 个判失败、命中 1 个取其 `html()`、命中多个则拼接)。抓到内容后,它有一段很务实的"自适应"逻辑:用 `TokenManager(...).countFromString(content)` 数 token,跟 `Provider.contextLimit(...)` 比;若 `enableSummarization` 关闭或 token 在上下文窗口内,就直接返回原文;只有当内容长到超过上下文上限时,才调用 `summarizeContent(...)` 让 LLM 先压缩再返回。这样既保住了细节(短内容原样给后续步骤),又避免长网页撑爆下游 LLM 的上下文。

## flow 如何被封装成 agent 可调用的工具

写到这里,核心机制都齐了,但还差最关键的一跃:用户拖出来的 flow 怎么变成第 9 章里 aibitat 能在对话中自动挑选并调用的那种工具?答案在 `server/utils/agentFlows/index.js` 的 `AgentFlows` 类里,尤其是 `activeFlowPlugins()` 和 `loadFlowPlugin()` 这一对方法。

`activeFlowPlugins()` 扫描所有 flow,过滤掉 `flow.active === false` 的,把其余每个 flow 映射成形如 `@@flow_{uuid}` 的字符串数组返回。这个 `@@flow_` 前缀是一种"占位插件名"约定:在 `server/utils/agents/defaults.js` 与 `ephemeral.js` 里,默认 agent 的可用函数列表会 `...AgentFlows.activeFlowPlugins()` 把这些占位符摊进去。等到 agent 真正装配工具时(`server/utils/agents/index.js` 约 582 行),代码识别出 `name.startsWith("@@flow_")`,剥出 `uuid`,调用 `AgentFlows.loadFlowPlugin(uuid, this.aibitat)` 把占位符替换成真正的插件,再用 `this.aibitat.use(plugin.plugin())` 注册进 aibitat,同时把 `@agent` 的 `functions` 列表里的占位名替换成真实工具名。

`loadFlowPlugin(uuid)` 是这场"伪装"的核心。它先 `loadFlow(uuid)`,从 `start` 节点取出 `variables` 数组,用 `sanitizeToolName(flow.name)` 把人类可读的 flow 名清洗成合法的 OpenAI 工具名(注释写明"Must match `^[a-zA-Z0-9_-]{1,64}$`":转小写、空白转下划线、去掉非法字符、合并连续下划线、截断到 64 字符,清洗后为空则回退到 `flow_{uuid}`)。然后它返回一个标准的 aibitat 插件结构,关键在 `setup(aibitat)` 里调用 `aibitat.function({...})`——把 flow 注册成一个真正的 agent function:

```js
aibitat.function({
  name: toolName,
  description: flow.config.description || `Execute agent flow: ${flow.name}`,
  parameters: {
    type: "object",
    properties: variables.reduce((acc, v) => {
      if (v.name) acc[v.name] = { type: "string", description: v.description || `Value for variable ${v.name}` };
      return acc;
    }, {}),
  },
  handler: async (args) => {
    aibitat.introspect(`Executing flow: ${flow.name}`);
    const result = await AgentFlows.executeFlow(uuid, args, aibitat);
    ...
  },
});
```

这里有几处设计值得停下来看。其一,**flow 的输入参数 schema 来自 `start` 节点的 `variables`**——每个变量被声明成一个 `string` 类型的入参,变量的 `description` 直接成为给 LLM 看的参数说明。于是 LLM 在决定调用这个工具时,会按这些参数填值,而这些值正是 `executeFlow` 里会覆盖默认值的 `initialVariables`。其二,**`handler` 把 LLM 给的 `args` 原封不动当作初始变量传给 `executeFlow(uuid, args, aibitat)`**,完成了"工具调用参数 → flow 初始变量"的对接。其三,**对 directOutput 的特殊处理**:若 `result.directOutput` 为真,handler 会设 `aibitat.skipHandleExecution = true` 并直接返回字符串化后的直出内容;在 aibitat 内核里(`aibitat/index.js` 约 1057 行)看到 `skipHandleExecution` 被消费——它会让 agent 跳过"把工具结果再喂回 LLM 加工"这一步,直接把 flow 产出当作最终回答输出给用户。这正是 `apiCall`/`webScraping` 的 `directOutput` 字段一路传导到顶层的归宿:某些 flow(比如纯粹回一段 API 数据)希望原样返回,不想被 LLM 改写。其四,失败时 handler 返回 `Flow execution failed: ...` 的可读字符串而非抛错,让 agent 能把失败信息自然地纳入对话。

至此,一个端到端的链路就闭合了:用户拖图 → 存成 flow JSON → `activeFlowPlugins()` 把它列为 `@@flow_uuid` → agent 装配时 `loadFlowPlugin` 把它注册成一个名字、描述、参数齐全的 aibitat function → LLM 在对话里像调用任何内置工具一样调用它 → `handler` 触发 `FlowExecutor` 顺序跑完各步骤 → 结果(或直出)回到对话。flow 对 LLM 完全透明,它看到的只是又一个工具。

## 存储格式:每个 flow 一份 JSON 文件

最后回答存储问题。`AgentFlows` 类用一个静态字段定下存放目录:`AgentFlows.flowsDir`,它取 `process.env.STORAGE_DIR` 下的 `plugins/agent-flows`,没有该环境变量时回退到 `process.cwd()/storage/plugins/agent-flows`。也就是说,flow 不进数据库,而是以**文件**形式落在 storage 目录里——这与 AnythingLLM 把许多"插件式资产"放在 `storage/plugins/` 下的一贯做法一致。

存取都是直白的文件操作。`saveFlow(name, config, uuid = null)` 在没有 uuid 时用 `uuidv4()` 生成一个,文件名即 `${uuid}.json`,内容是 `JSON.stringify({ ...config, name }, null, 2)`——一份缩进 2 空格、人类可读的 JSON,顶层带 `name`、`description`、`steps` 等字段。`loadFlow(uuid)` 读 `${uuid}.json` 并 `safeJsonParse`,返回 `{ name, uuid, config }`。`getAllFlows()` 直接 `readdirSync` 目录、过滤出 `.json` 文件逐个解析成 `{ uuid: config }` 映射。`listFlows()` 在此基础上输出摘要(含 `active: flow.active !== false`,即 active 字段缺省视为开启)。`deleteFlow(uuid)` 则是 `fs.rmSync` 删文件。

存储路径上还有两道安全闸值得一提。第一,所有文件路径都经过 `normalizePath` 规整,并用 `isWithin(AgentFlows.flowsDir, filePath)` 校验目标确实落在 flows 目录内——这是防止 uuid 里夹带 `../` 之类路径穿越的常规防护,`loadFlow`、`saveFlow`、`deleteFlow` 都做了这层检查。第二,`saveFlow` 在写盘前会做一次**节点类型白名单校验**:它用 `Object.values(FLOW_TYPES).map((d) => d.type)` 取出当前版本支持的全部类型,要求 `config.steps.every(step => supportedFlowTypes.includes(step.type))`,否则抛错"This flow includes unsupported blocks..."。注释解释了动机:防止把含有不支持节点(例如桌面版才有的文件写入或代码执行块)的 flow 导入到不支持这些能力的环境(如 Docker)。这道闸把"前端能画什么"与"后端能跑什么"在保存这一关就对齐了,避免运行时才发现 `Unknown flow type` 的尴尬。

flow 的对外操作由 `server/endpoints/agentFlows.js` 暴露:`/agent-flows/save` 调 `saveFlow`、`/agent-flows/list` 调 `listFlows`、`DELETE /agent-flows/:uuid` 调 `deleteFlow`、`/agent-flows/:uuid/toggle` 通过重新 `saveFlow` 翻转 `active` 字段。整套 CRUD 都建立在"一个 uuid 一份 JSON"这个朴素的存储模型之上。

## 本章小结

- Agent Flows 是 AnythingLLM 的无代码编排能力:用户拖拽出的 flow 在数据层面就是一份 `{ name, description, steps: [...] }` 的 JSON,`steps` 是一个**线性数组**,每步只声明 `type` 与 `config`,执行模型是严格顺序而非图。
- `server/utils/agentFlows/flowTypes.js` 的 `FLOW_TYPES` 是类型注册表,定义了四种节点:`start`、`apiCall`、`llmInstruction`、`webScraping`;常量名(如 `API_CALL`)给代码用,`type` 字面量(如 `"apiCall"`)才写进 JSON 并用于分派。
- `FlowExecutor`(`executor.js`)以一个实例级的 `this.variables` 变量袋为核心,`executeFlow` 用 `start` 节点的 `variables` 作默认值、被传入的 `initialVariables` 覆盖,再用一个串行 `for` 循环逐步 `await` 执行,遇错即停、遇 `directOutput` 即短路。
- 单步执行 `executeStep` 三段式:先 `replaceVariables` 解析输入占位符,再按 `step.type` 分派给执行器,最后把结果按 `resultVariable`/`responseVariable`(前者优先)写回变量袋。
- 变量替换由 `replaceVariables` 递归扫描整个 `config`、对 `${...}` 调用 `getValueFromPath` 实现;`getValueFromPath` 手写解析器支持 `data.items[0].name` 这类点号加方括号的路径,并能对存成 JSON 字符串的变量先解析再取值——步骤间数据靠"写变量、读 `${变量}`"流动,没有显式连线。
- `executeApiCall`(`api-call.js`)按 `bodyType`(json/form/text)组装请求、用 `fetch` 发出、对响应 `safeJsonParse`;失败抛 `HTTP error`。
- `executeLLMInstruction`(`llm-instruction.js`)复用 `aibitat.getProviderForConfig(aibitat.defaultProvider)`,即 agent 的默认模型,按 `supportsAgentStreaming` 走 `stream` 或 `complete`,返回 `textResponse`。
- `executeWebScraping`(`web-scraping.js`)用 `CollectorApi.getLinkContent` 抓取、可用 `cheerio` 按 `querySelector` 抽取,并按 `TokenManager` 计数与 `Provider.contextLimit` 比较,内容超长时才用 `summarizeContent` 摘要。
- flow 通过 `activeFlowPlugins()` 列为 `@@flow_{uuid}` 占位符,在 agent 装配阶段被 `loadFlowPlugin` 替换为真正的 aibitat function:工具名经 `sanitizeToolName` 清洗(`^[a-zA-Z0-9_-]{1,64}$`),参数 schema 来自 `start` 节点的 `variables`,`handler` 把 LLM 给的 `args` 当初始变量交给 `executeFlow`,对 LLM 完全表现为一个普通工具。
- `directOutput` 一路从节点配置传导到顶层:handler 检测到后设 `aibitat.skipHandleExecution = true`,让 agent 跳过对结果的二次加工,把 flow 产出原样作为最终回答。
- 存储上每个 flow 是 `STORAGE_DIR/plugins/agent-flows/{uuid}.json` 一份文件;`saveFlow`/`loadFlow`/`deleteFlow` 都用 `normalizePath` + `isWithin` 防路径穿越,`saveFlow` 还用 `FLOW_TYPES` 白名单拒绝含不支持节点的 flow。

## 动手实验

1. **实验一:画一份 flow 的最小 JSON 并手验顺序执行** — 不开界面,直接按 `LoadedFlow` 结构手写一份 JSON:一个 `start` 节点(声明变量 `topic`,默认值空)、一个 `apiCall` 节点(GET 某公开 API,`responseVariable: "apiData"`)、一个 `llmInstruction` 节点(`instruction` 写 `总结这段数据:${apiData}`)。把它存到 `storage/plugins/agent-flows/test-uuid.json`,然后写一段 Node 脚本 `require` `AgentFlows` 并调 `AgentFlows.executeFlow("test-uuid", { topic: "x" })`,观察返回对象里 `results` 的顺序与 `variables` 的最终快照,确认变量确实被逐步累积。

2. **实验二:追踪一次变量替换** — 在 `executor.js` 的 `getValueFromPath` 末尾临时加日志,打印传入的 `path` 与最终 `current`;再构造一个变量 `apiData` 存成一段含嵌套数组的 JSON 字符串,在下一步用 `${apiData.items[0].name}` 引用它。运行 flow,观察日志,验证"字符串变量被 `safeJsonParse` 解析后再按 `items[0].name` 取值"以及"取到对象时被 `JSON.stringify` 注入"这两条路径,亲手确认数据是如何跨步骤流动的。

3. **实验三:验证 directOutput 短路** — 给一个三步 flow 的第一个 `apiCall` 节点加上 `directOutput: true`,在 `executeStep` 写回逻辑与 `executeFlow` 的循环里各加一行日志,运行后确认:第一步结果被包成 `{ directOutput: true, result }`、循环立即 `break`、后两步从未执行,返回对象的 `directOutput` 字段被填充。再把它作为 agent 工具调用,观察 `handler` 是否设置了 `aibitat.skipHandleExecution = true`、最终回答是否绕过了 LLM 二次加工。

4. **实验四:把 flow 装配成 agent 工具并读懂参数 schema** — 调 `AgentFlows.loadFlowPlugin(uuid)`,打印它返回的 `name`(经 `sanitizeToolName` 清洗后的工具名)与 `plugin().setup` 内注册的 `parameters.properties`,对照你 `start` 节点里声明的 `variables`,确认每个变量名都变成了一个带 `description` 的 `string` 入参;再故意把 flow 名改成含中文与空格的字符串,观察 `sanitizeToolName` 如何清洗、以及清洗为空时如何回退到 `flow_{uuid}`。

> **下一章预告**:flow 让用户把多步逻辑封装成工具,但工具的来源还可以更开放。第 11 章我们将走进 MCP(Model Context Protocol)与多渠道接入——看 AnythingLLM 如何作为 MCP 客户端把外部 MCP server 暴露的工具动态接入 agent,以及它如何把同一套对话能力延展到 Web 嵌入、API、桌面等多个渠道,让"工具生态"和"接入入口"同时向外打开。
