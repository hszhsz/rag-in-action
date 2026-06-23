# 第 1 章 三进程运行时全景：server、collector 与 frontend

如果说 RAGFlow 是一座以 Python 为骨架、把文档理解与检索深度耦合在一起的"重型工厂",那么 AnythingLLM 选择了一条几乎相反的路线:它用 JavaScript/TypeScript 构建了一个**多进程协作**的系统,把"对话与编排"、"文档采集"、"用户界面"拆成三个彼此独立、通过 HTTP 通信的进程。理解这一章,你就握住了整本书的地图——后面每一个子系统(向量库抽象、Agent、MCP、加密)都挂在这张三进程拓扑上。

本章我们从代码仓库的物理结构出发,看清三个进程各自是什么、如何启动、又如何通过一把"会滚动更新的 RSA 钥匙"建立彼此之间的信任边界。

## 一个 monorepo,三个 `index.js`

打开仓库根目录的 `package.json`,`scripts` 字段就把这套架构的意图摊开在你面前:

```json
"dev:server": "cd server && yarn dev",
"dev:collector": "cd collector && yarn dev",
"dev:frontend": "cd frontend && yarn dev",
"dev": "npx concurrently \"yarn dev:server\" \"yarn dev:frontend\" \"yarn dev:collector\""
```

`dev` 脚本用 `concurrently` 同时拉起三个子项目,这不是巧合,而是 AnythingLLM 的核心运行模型:`server`、`collector`、`frontend` 是三个**独立的 Node/前端工程**,各有自己的 `package.json`、各自 `yarn install`。根 `package.json` 里的 `setup` 脚本印证了这一点——它依次进入三个目录分别安装依赖,再统一拷贝 `.env` 模板、跑 Prisma 初始化。

三个进程的职责边界非常清晰:

- **`server/`** 是大脑。它是一个 Express 应用(`server/index.js`),挂载了几十个 endpoint 模块(`systemEndpoints`、`workspaceEndpoints`、`chatEndpoints`、`agentWebsocket`……),负责对话编排、向量检索、Agent 调度、模型网关、权限与持久化。所有业务逻辑都在这里。
- **`collector/`** 是嘴。它同样是一个 Express 应用(`collector/index.js`),但只干一件事:把各种格式的原始文档(PDF、Office、音频、网页链接)"吃进来",转换成结构化文本,吐给 server。它默认监听 `8888` 端口。
- **`frontend/`** 是脸。一个 Vite + React 工程(`frontend/src/App.jsx`、`main.jsx`),编译成静态资源。在生产模式下,它甚至不单独跑——而是被 build 后由 server 直接静态托管。

这种"前端被 server 托管"的细节,藏在 `server/index.js` 的生产分支里:

```js
if (process.env.NODE_ENV !== "development") {
  app.use(express.static(path.resolve(__dirname, "public"), { ... }));
  app.use("/", function (_, response) {
    IndexPage.generate(response);
  });
}
```

也就是说,生产环境只有 **server 和 collector 两个常驻进程**;前端编译产物被拷进 `server/public`,由 server 兜底返回。开发环境才是三进程齐飞。

## server 的启动序列:一切从 `bootHTTP` 开始

`server/index.js` 的顶部是一长串 `require("./endpoints/...")`,把每个功能域的路由模块引进来,然后逐一注册到同一个 `apiRouter` 上,最后 `app.use("/api", apiRouter)`。这是一种朴素但清晰的模块化:一个 endpoint 文件就是一个功能域,新增功能就是新增一个 `xxxEndpoints(apiRouter)` 调用。

真正的"开机自检"发生在 `server/utils/boot/index.js` 的 `bootHTTP`(或启用 HTTPS 时的 `bootSSL`)里。注意它们在 `app.listen` 的回调中按固定顺序做了一连串初始化:

```js
app.listen(port, async () => {
  await markOnboarded();
  await setupTelemetry();
  new CommunicationKey(true);
  new EncryptionManager();
  new BackgroundService().boot();
  await eagerLoadContextWindows();
  await PushNotifications.setupPushNotificationService();
  await TelegramBotService.bootIfActive();
  console.log(`Primary server in HTTP mode listening on port ${port}`);
});
```

这段顺序值得逐行品味,因为它定义了系统"可用"的前提条件:

1. `new CommunicationKey(true)` 中那个 `true` 参数,会触发**重新生成 RSA 密钥对**——这是本章下一节的主角。每次 server 启动都滚动一次钥匙。
2. `new EncryptionManager()` 初始化对称加密管理器,用于加密落库的敏感配置(如各家模型的 API Key)。
3. `new BackgroundService().boot()` 拉起后台任务系统(文档同步、定时任务等)。
4. `eagerLoadContextWindows()` 预加载各模型的上下文窗口长度——把"某模型能塞多少 token"这种元数据提前缓存,避免对话时临时查询。

`bootSSL` 与 `bootHTTP` 的初始化体几乎逐字相同,区别只在前者用 `https.createServer` 加载证书。更值得注意的是 `bootSSL` 的 `catch` 分支:一旦证书加载失败,它不会让进程崩溃,而是打印 `[SSL BOOT FAILED]` 后 **`return bootHTTP(app, port)`**——优雅降级回 HTTP。这是一个典型的"失败时退到一个仍能工作的状态"的设计取舍。

## 为什么 collector 需要一把"会过期的钥匙"

三进程架构带来一个尖锐的安全问题:collector 监听着一个 HTTP 端口,它会去**抓取任意 URL、解析任意文件**。如果这个端口能被外部直接访问,攻击者就能让 collector 去请求内网地址(SSRF),或投喂恶意文件。AnythingLLM 的答案是:**collector 只信任 server 签过名的请求**。

这把锁的实现分布在两个同名却职责相反的类里。server 侧的 `server/utils/comKey/index.js`,其类注释把设计意图写得明明白白:

> This class generates a hashed version of some text (typically a JSON payload) using a rolling RSA key that can then be appended as a header value to do integrity checking on a payload. ... This keeps accidental misconfigurations of AnythingLLM that leave the collector port open from being abused or SSRF'd by users scraping malicious sites who have a loopback embedded in a `<script>`, for example.

它的核心是一对 2048 位 RSA 密钥,在 `#generate()` 里用 `crypto.generateKeyPairSync("rsa", { modulusLength: 2048, ... })` 生成,私钥 `ipc-priv.pem`、公钥 `ipc-pub.pem` 落在 `storage/comkey` 目录。server 侧持有**私钥**,只做一件事——`sign(textData)`,用 `RSA-SHA256` 对载荷签名,返回 hex 字符串。

collector 侧的 `collector/utils/comKey/index.js` 是它的镜像:构造函数为空,只读**公钥**,只做 `verify(signature, textData)`。两者通过共享的 `storage/comkey` 目录拿到同一对钥匙(注意 collector 侧的 `keyPath` 指向 `../../../server/storage/comkey`)。

验证的关口在 collector 的中间件 `collector/middleware/verifyIntegrity.js`。每个 `/process`、`/parse` 路由都先过它:

```js
const signature = request.header("X-Integrity");
if (!signature)
  return response.status(400).json({ msg: "Failed integrity signature check." });
const validSignedPayload = comKey.verify(signature, request.body);
if (!validSignedPayload)
  return response.status(400).json({ msg: "Failed integrity signature check." });
```

没有 `X-Integrity` 头、或签名验不过,直接 400 拒绝。这意味着:**绕过 server 直接打 collector 的请求,一律失效**。而由于 `new CommunicationKey(true)` 在每次 server 启动时重新生成密钥对,即便旧签名泄露,重启后也立刻作废——这就是注释里 "rolling key"(滚动钥匙)的含义。

不过这里有一个务必看清的开发态例外:`verifyIntegrity.js` 开头,当 `process.env.NODE_ENV === "development"` 时,它打印一行日志后**直接 `next()` 放行**,完全跳过验签。这是为了本地三进程联调方便,但也提醒部署者——生产环境绝不能跑在 `development` 模式下。

## server 如何"喊话"collector

反过来看 server 怎么调 collector。`server/utils/collectorApi/index.js` 的 `CollectorApi` 类封装了这层通信。它的构造函数 `new CommunicationKey()` 拿到签名能力,`endpoint` 拼成 `http://0.0.0.0:${port}`。端口解析逻辑 `getCollectorPort()` 体现了"边界不信任输入"的防御姿态:

```js
const port = Number(process.env.COLLECTOR_PORT || this.DEFAULT_COLLECTOR_PORT);
if (Number.isInteger(port) && port > 0 && port <= 65535) return port;
console.warn(`Invalid COLLECTOR_PORT ... Falling back to ${this.DEFAULT_COLLECTOR_PORT}.`);
return this.DEFAULT_COLLECTOR_PORT;
```

环境变量给的端口不是合法整数、或越界,就告警并回退到默认 `8888`(`DEFAULT_COLLECTOR_PORT`),而不是带着一个非法端口崩溃。`CollectorApi` 还通过 `#attachOptions()` 把运行期选项(whisper 语音转写 provider、OCR 语言、`allowAnyIp` 等)一并塞进发往 collector 的请求体里——这些选项随每个请求传递,collector 侧由 `RuntimeSettings.parseOptionsFromRequest` 解析后"持久化"在本进程内。

`CollectorApi` 还定义了 `extensionRequestTimeout = 15 * 60_000`(15 分钟)这样一个超长超时,并为此单独创建了 undici 的 `Agent`。这暗示了一个现实:抓取一个大型网站、转写一段长音频,可能耗时十几分钟,常规 HTTP 超时根本不够用。

## 数据如何在三进程间流动

把上面的拼图合起来,一份文档从上传到可被对话引用,要走这样一条路:用户在 **frontend** 选择文件 → 请求打到 **server** 的 `documentEndpoints` → server 用 `CollectorApi` 把文件名与签名(`X-Integrity` 头)发给 **collector** 的 `/process` → collector 验签通过后调 `processSingleFile`,把原始文件转成结构化文本 documents 返回 → server 再做切分、向量化、写入向量库。

聊天链路则更短:frontend ↔ server 的 WebSocket / SSE,完全不经过 collector。collector 只在"喂数据"时出场。这种"采集与对话彻底解耦"的设计,让文档处理这种重、慢、易出错的活,被隔离在一个可以独立扩缩容、独立崩溃重启的进程里,不会拖垮对话主链路。

## 本章小结

- AnythingLLM 是一个 **monorepo + 三进程**架构:`server`(Express,大脑)、`collector`(Express,文档采集,默认 `8888` 端口)、`frontend`(Vite/React,界面)。三者各有独立 `package.json` 与 `index.js`。
- 根 `package.json` 的 `dev` 脚本用 `concurrently` 同时拉起三进程;`setup` 脚本逐目录安装依赖,印证了"三个独立工程"的事实。
- **生产环境只有两个常驻进程**:前端被 build 进 `server/public`,由 server 在 `NODE_ENV !== "development"` 分支用 `express.static` 静态托管并兜底返回 `IndexPage`。
- server 的启动序列固定在 `bootHTTP`/`bootSSL`:`CommunicationKey(true)` → `EncryptionManager` → `BackgroundService` → `eagerLoadContextWindows` → 推送/Telegram。顺序即依赖。
- `bootSSL` 在证书加载失败时**降级回 `bootHTTP`**,而非崩溃——典型的 fail-soft 取舍。
- 跨进程信任靠**滚动 RSA 密钥对**:server 侧 `comKey` 持私钥只签名(`sign`),collector 侧 `comKey` 持公钥只验签(`verify`),通过共享 `storage/comkey` 目录配对。
- collector 的 `verifyPayloadIntegrity` 中间件对每个处理请求强制校验 `X-Integrity` 签名头,验不过即 400,**防止 collector 端口被 SSRF 滥用**。
- `new CommunicationKey(true)` 在每次 server 启动时重新生成密钥,使旧签名重启即失效——这是 "rolling" 的本质。
- **开发态例外**:`NODE_ENV === "development"` 时验签被跳过,直接放行——生产部署绝不能用 development 模式。
- server→collector 的 `CollectorApi` 用 `getCollectorPort()` 对端口做合法性校验(整数、`0<port<=65535`),非法则告警回退 `8888`;并设置 15 分钟超长超时应对大文件/长抓取。

## 动手实验

1. **实验一:画出三进程拓扑** — 打开根 `package.json`,逐条阅读 `scripts` 中的 `dev:server`/`dev:collector`/`dev:frontend`/`setup`,在纸上画出三个进程及它们各自的工作目录;再到 `server/index.js` 找到生产分支的 `express.static(... "public" ...)`,确认前端在生产环境是如何被 server 托管的。
2. **实验二:跟踪一次启动** — 阅读 `server/utils/boot/index.js` 的 `bootHTTP`,把 `app.listen` 回调里的初始化调用按顺序列成清单。思考:为什么 `CommunicationKey(true)` 必须在系统对外提供文档处理能力之前完成?如果把它挪到最后会发生什么?
3. **实验三:理解滚动签名** — 对照 `server/utils/comKey/index.js`(持私钥、`sign`)与 `collector/utils/comKey/index.js`(持公钥、`verify`),写出一次签名—验签的完整数据流。再找到两个文件里 `keyPath` 的定义,确认它们指向同一个 `storage/comkey` 目录。
4. **实验四:验证 SSRF 防线** — 阅读 `collector/middleware/verifyIntegrity.js`,找出 `NODE_ENV === "development"` 的放行分支。设想一个攻击场景:若生产环境误以 development 模式启动,collector 会暴露什么风险?再到 `server/utils/collectorApi/index.js` 的 `getCollectorPort()`,构造几个非法 `COLLECTOR_PORT` 值(如 `"abc"`、`99999`、`-1`),推演每个会走到哪个分支。

> **下一章预告**:server 把文件名和签名发给了 collector,接下来真正"消化"文档的活由谁来干?下一章我们钻进 `collector` 进程,解剖 `processSingleFile` 如何根据 MIME 类型分发到不同转换器,OCR、音频转写、网页抓取又是如何被统一成同一种结构化文本输出的。
