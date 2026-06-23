# 第 4 章 向量化引擎 EmbeddingEngines:从零配置 native 到多后端统一契约

上一章交出了一批大小受控、带元数据头的纯文本 chunk,这一章它们将被转换成可供相似度检索的浮点向量。向量化(embedding)是 RAG 系统的咽喉:它既决定了召回质量的上限,也是整条文档入库链路里最可能 OOM、最可能因网络抖动而失败的环节。AnythingLLM 把这件事收敛到一个目录——`server/utils/EmbeddingEngines/`,里面并排躺着十几个 provider:`native`、`openAi`、`ollama`、`localAi`、`voyageAi`、`cohere`、`gemini`、`azureOpenAi`、`liteLLM`、`mistral`、`lmstudio`、`genericOpenAi`、`openRouter`、`lemonade`。

这套设计的核心张力在于:一个开源、私有化部署优先的产品,必须保证"下载即可用、不填任何 API Key 也能跑 RAG",同时又要给愿意付费的用户提供 OpenAI、Voyage 这类托管嵌入服务。AnythingLLM 的答案是:用一个名叫 `native` 的本地引擎做兜底默认值,用一份隐式契约(`embedTextInput` / `embedChunks` / `maxConcurrentChunks` / `embeddingMaxChunkLength`)把所有 provider 拍平成可互换的零件,再用 `getEmbeddingEngineSelection` 这一个 switch 把环境变量翻译成具体实例。本章就沿着"默认引擎如何零配置开箱 → 模型如何缓存到本地 → provider 抽象契约 → 并发节流 → 各家差异 → 引擎如何被选择"这条线逐层拆开。

## 一、默认引擎 native:不填任何 Key 也能跑 RAG

打开 `server/utils/EmbeddingEngines/native/index.js`,第一行有意义的代码是类的静态字段:`static defaultModel = "Xenova/all-MiniLM-L6-v2"`。这是整个 AnythingLLM 的嵌入默认值——一个 23MB 的轻量级模型,在 `native/constants.js` 的 `SUPPORTED_NATIVE_EMBEDDING_MODELS` 里它的 `apiInfo.description` 写得很直白:"A lightweight and fast model for embedding text. The default model for AnythingLLM."

`NativeEmbedder` 之所以能"零配置",关键在于它不依赖任何远端服务,而是把 HuggingFace 上的 ONNX 模型下载到本地、用 `@xenova/transformers`([[@xenova/transformers]](https://github.com/xenova/transformers.js))在 Node 进程里直接做特征抽取。注意它对这个库的加载方式很讲究:`@xenova/transformers` 是 ESM 包,而 AnythingLLM 的 server 是 CommonJS,所以 `#fetchWithHost` 里用动态 `import("@xenova/transformers")` 把它当 ESM 异步加载进来,而不是顶层 `require`。核心调用是:

```js
pipeline("feature-extraction", this.model, {
  cache_dir: this.cacheDir,
  ...(!this.modelDownloaded ? { progress_callback: ... } : {}),
})
```

`"feature-extraction"` 是 transformers.js 的标准任务名,产出的是句向量;`cache_dir` 把模型文件落到本地缓存目录(下一节细说);`progress_callback` 只在 `!this.modelDownloaded` 时挂载,首次运行会打印 `[NativeEmbedder - Downloading model] <file> <progress>%` 的进度条,后续运行因为模型已在本地而安静跳过。构造函数末尾还有一句日志写明设计意图:"The native embedding model has never been run and will be downloaded right now. Subsequent runs will be faster. (~23MB)"——首跑下载、后续命中缓存,这就是"零配置开箱"的全部代价。

构造函数里另一处值得留意的是 `supportedModels` 的白名单语义。`getEmbeddingModel()` 读取环境变量 `EMBEDDING_MODEL_PREF`,但并不无条件信任它:

```js
getEmbeddingModel() {
  const envModel = process.env.EMBEDDING_MODEL_PREF ?? NativeEmbedder.defaultModel;
  if (NativeEmbedder.supportedModels?.[envModel]) return envModel;
  return NativeEmbedder.defaultModel;
}
```

只有落在 `SUPPORTED_NATIVE_EMBEDDING_MODELS` 这张表里的模型才会被接受,否则一律回落到 `defaultModel`。源码注释解释了为什么只支持寥寥几个:"Because we need to mirror them on the CDN so non-US users can download them."——每个 native 模型都要在 Mintplex Labs 自家 CDN 上做镜像,所以不可能开放任意 HF 模型。当前白名单里除了默认的 `Xenova/all-MiniLM-L6-v2`,还有 `Xenova/nomic-embed-text-v1`(139MB,大上下文窗口)和 `MintplexLabs/multilingual-e5-small`(487MB,支持 100+ 语言)。

## 二、模型本地缓存:STORAGE_DIR、CDN 兜底与单例 pipeline

模型缓存目录由构造函数里这段决定:

```js
this.cacheDir = path.resolve(
  process.env.STORAGE_DIR
    ? path.resolve(process.env.STORAGE_DIR, `models`)
    : path.resolve(__dirname, `../../../storage/models`)
);
this.modelPath = path.resolve(this.cacheDir, ...this.model.split("/"));
this.modelDownloaded = fs.existsSync(this.modelPath);
```

逻辑很清楚:若设置了 `STORAGE_DIR`(容器部署常用),模型缓存就落在 `$STORAGE_DIR/models`;否则回落到仓库内的 `storage/models`。`modelPath` 把形如 `Xenova/all-MiniLM-L6-v2` 的模型名按 `/` 拆开再拼成子目录,`modelDownloaded` 用 `fs.existsSync` 判断这个目录是否已存在,从而决定要不要走下载流程。构造函数还会 `if (!fs.existsSync(this.cacheDir)) fs.mkdirSync(this.cacheDir)`,为老安装补建目录。

下载本身有一道兜底机制,这是被现实逼出来的工程细节。`#fallbackHost = "https://cdn.anythingllm.com/support/models/"` 是 Mintplex Labs 托管的镜像源,注释写明:这是给那些"因 IP、VPN 等原因无法直连 HF 下载端点"的用户准备的。`embedderClient()` 的下载流程是"先试默认源(HF),失败一次性切换到 fallback CDN"——注意它刻意不是递归重试,源码注释强调 "a single fallback attempt (not recursive on purpose)",还附了引发这套设计的 GitHub issue 链接。切换到 fallback 时,`#fetchWithHost(hostOverride)` 会改写 `env.remoteHost = hostOverride` 并把 `env.remotePathTemplate` 设成 `"{model}/"`,因为 S3 风格的 CDN 不支持 HF 那套带 revision 的目录结构。

更关键的是 `pipeline` 的单例化。文件顶部有两张静态 Map:

```js
/** @type {Map<string, any>} */
static #pipelines = new Map();
/** @type {Map<string, Promise<any>>} */
static #pipelinePromises = new Map();
```

为什么要把 pipeline 缓存成进程级单例?注释给了硬核理由:"ONNX sessions cannot be freed on onnxruntime-node 1.14 (dispose() is a no-op), so we must only ever create one pipeline per model."——底层 onnxruntime 的 `dispose()` 是空操作,session 释放不掉,因此每个模型在整个进程生命周期里只能创建一次 pipeline,否则内存只增不减。`embedderClient()` 的实现就是围绕这条约束的双层缓存:`#pipelines` 缓存已就绪的 pipeline,`#pipelinePromises` 缓存"正在加载中"的 Promise,后者用来防止并发首跑时同一个模型被重复初始化(典型的 promise-deduplication 模式)。加载成功后写入 `#pipelines`,`finally` 里把 `#pipelinePromises` 那条删掉。

## 三、provider 抽象契约:embedTextInput 与 embedChunks

AnythingLLM 没有定义一个显式的 `BaseEmbedder` 抽象基类,它的契约是**隐式约定**:每个 provider 类都暴露两个嵌入方法和两个数值属性。两个方法是 `embedTextInput(textInput)` 和 `embedChunks(textChunks = [])`,两个属性是 `maxConcurrentChunks` 和 `embeddingMaxChunkLength`。只要遵守这套形状,调用方(切分管线、向量库 provider)就能把任意 embedder 当成同一个零件用。

`embedTextInput` 是"单条/少量"入口,几乎所有 provider 的实现都是同一个模板——把单条统一成数组、复用 `embedChunks`、再取第一个结果。以 `openAi/index.js` 为例:

```js
async embedTextInput(textInput) {
  const result = await this.embedChunks(
    Array.isArray(textInput) ? textInput : [textInput]
  );
  return result?.[0] || [];
}
```

`native`、`ollama`、`localAi` 的 `embedTextInput` 与此一字不差,这说明"单条嵌入"在设计上被刻意定义为"批量嵌入的退化情形",避免每个 provider 各写一套单条逻辑。`embedTextInput` 在检索时尤其有用:查询语句就是一条文本,要被嵌入成查询向量去比对库里的文档向量。`native` 在这里还多做一件事——`#applyQueryPrefix`,因为像 `nomic-embed-text-v1` 这类模型区分 `search_document:` 与 `search_query:` 前缀(见 `constants.js` 里的 `chunkPrefix`/`queryPrefix`),查询侧必须前置 `queryPrefix` 才能召回正确。

`embedChunks` 才是真正的主力,负责把一整批 chunk 编码成 `number[][]`。它的返回约定是:成功返回向量数组,任何不完整都返回 `null` 或抛错,绝不返回半截结果。这一点在 `openAi` 里写得很明确——只要 OpenAI 任一批次报错,就 abort 整个序列,注释说 "the embeddings will be incomplete"。`native` 同样,在最后 `return embeddingResults.length > 0 ? embeddingResults.flat() : null`。这种"要么全成、要么 null"的契约让上游入库逻辑可以简单地用真值判断决定是否继续写库。

两个数值属性则是契约里的"元数据"。`embeddingMaxChunkLength` 告诉切分器单次能塞多长的文本(第 3 章的 `determineMaxChunkSize` 就是拿它做上限对齐),`maxConcurrentChunks` 告诉嵌入逻辑一批最多发多少条——这正是下一节并发节流的主角。

## 四、并发节流:maxConcurrentChunks 与 toChunks 分批

所有 provider 都共享同一个分批工具 `toChunks`,定义在 `server/utils/helpers/index.js`:

```js
function toChunks(arr, size) {
  return Array.from({ length: Math.ceil(arr.length / size) }, (_v, i) =>
    arr.slice(i * size, i * size + size)
  );
}
```

它把一维数组按 `size` 切成二维批次。`size` 永远来自 provider 的 `maxConcurrentChunks`。有意思的是,各 provider 对"分好的批次怎么调度"采取了两种截然不同的策略,这恰好暴露了它们各自的瓶颈所在。

**native 的策略是严格串行 + 写盘换内存。**`native` 的 `maxConcurrentChunks` 对默认模型是 25(见 `constants.js`),`embedChunks` 用 `for (let [idx, chunk] of chunks.entries())` 一批一批 `await`,完全不并发。源码注释解释得淋漓尽致:这段代码在 t3.small(2GB RAM / 1vCPU)上反复 benchmark 过,"without careful memory management for the V8 garbage collector this function will likely result in an OOM"。`maxConcurrentChunks` 之所以钉死 25 而不是 50,是因为 "50 seems to overflow no matter what"。更激进的是它把每批结果立刻 `JSON.stringify` 后 `#writeToTempfile` 追加写到一个临时 `.tmp` 文件里(手工拼 JSON 数组的 `[`、`,`、`]`),循环里把 `output = null; data = null` 主动断引用,等全部跑完再 `fs.readFileSync` 一次性读回、`JSON.parse`、`fs.rmSync` 删文件。这是用磁盘换内存,刻意把嵌入结果"赶出" V8 堆,以扛住超过 10 万词的大文档。

**openAi / localAi 的策略是并发发车 + Promise.all 汇聚。**托管 API 的瓶颈不是内存而是网络往返,所以它们把每批包成一个 Promise 立刻 push 进 `embeddingRequests`,最后 `await Promise.all(embeddingRequests)` 一起等。`openAi` 把 `maxConcurrentChunks` 设到 500,注释解释这是为了贴近 OpenAI "~8mb" 的单次 POST 上限并尽量减少往返次数。它的错误处理很考究:每个批次的 Promise 内部 catch 错误并 `resolve({ data: [], error: e })`(而不是 reject),这样 `Promise.all` 不会被任何单点失败短路;汇聚阶段再统一过滤 errors,用 `Set` 去重错误信息后整体抛出,保证"部分失败即全盘失败"的契约。

**ollama 的策略是显式批循环 + num_ctx 注入。**`ollama/index.js` 既不像 native 那样写盘,也不像 openAi 那样全并发,而是用 `for (let i = 0; i < textChunks.length; i += this.maxConcurrentChunks)` 切片串行 `await this.client.embed(...)`。它的 `maxConcurrentChunks` 默认是 1,可由 `OLLAMA_EMBEDDING_BATCH_SIZE` 调高:

```js
this.maxConcurrentChunks = process.env.OLLAMA_EMBEDDING_BATCH_SIZE
  ? Number(process.env.OLLAMA_EMBEDDING_BATCH_SIZE)
  : 1;
```

默认 1 是保守选择,因为 Ollama 多跑在用户自己的弱机器上。它还在每次 `embed` 调用里把 `options.num_ctx` 设成 `this.embeddingMaxChunkLength`,注释写明意图:"so that the maximum context window is used and content is not truncated."——用用户配的最大 chunk 长度撑满上下文窗口,避免内容被截断。另外 `embed` 用 `input` 参数(而非 `prompt`)以拿到 `number[][]` 的批量返回,再 `data.push(...embeddings)` 摊平。

无论哪种策略,它们都在批次推进时调用同一个 `reportEmbeddingProgress(chunksProcessed, totalChunks)` 回写进度。这个函数(在 `helpers/index.js`)很巧妙地兼容了两种运行环境:若 `typeof process.send === "function"`(说明跑在子 worker 进程里)就走 IPC `process.send(event)`;否则 `require("../EmbeddingWorkerManager")` 直接 `emitProgress` 走 SSE。进度上下文从 `global.__embeddingProgress` 读取,没设置就直接 return,因此非嵌入场景调用它是零开销的。

## 五、各家差异:同一契约下的不同取舍

把几个 provider 横着摆,契约一致但细节各异,这正是抽象层价值的体现。

`voyageAi/index.js` 是个"半契约"实现:它有 `embedTextInput` 和 `embedChunks`,但底层直接委托给 `@langchain/community` 的 `VoyageEmbeddings`,`embedChunks` 就是一行 `await this.voyage.embedDocuments(textChunks)`。它没有显式 `maxConcurrentChunks`,而是把批量上限通过 `batchSize: 128` 交给 LangChain 处理,注释引用了 Voyage 官方文档说 "Voyage AI's limit per request is 128"([[Voyage AI rate limits]](https://docs.voyageai.com/docs/rate-limits))。它的 `embeddingMaxChunkLength` 不是常量而是 `#getMaxEmbeddingLength()` 一个按 `model` 名分支的 switch——`voyage-3-lite` 等返回 32000,`voyage-large-2` 系列返回 16000,`voyage-2` 返回 4000,default 4000。可见 token 上限是模型属性,这家把它建模成了查表函数。

`localAi/index.js` 复用了 OpenAI 的 SDK(`require("openai")`),只是把 `baseURL` 指向 `EMBEDDING_BASE_PATH`、`apiKey` 用 `LOCAL_AI_API_KEY ?? null`,`maxConcurrentChunks` 设 50,`embeddingMaxChunkLength` 调 `maximumChunkLength()`。它还多了个 `outputDimensions` getter,从 `EMBEDDING_OUTPUT_DIMENSIONS` 读期望维度。这说明只要某后端兼容 OpenAI 的 embeddings 接口,接入成本就极低——`genericOpenAi`、`lmstudio` 等本质都是同一招的变体。

差异还体现在 `embeddingMaxChunkLength` 的来源上,大致分三类:native 来自 `constants.js` 里每个模型写死的字段(默认模型 1000、nomic 16000、e5-small 1000,且注释强调这是"字符数而非 token 数",按 2 字符≈1 token 保守换算);openAi 是硬编码 `8_191`(对齐 OpenAI 文档的 token 上限,[[OpenAI embeddings]](https://platform.openai.com/docs/guides/embeddings));ollama/localAi 则调用共享的 `maximumChunkLength()`,它读 `EMBEDDING_MODEL_MAX_CHUNK_LENGTH`,无效时回落到 `1_000`。一个值,三种确定方式,但对上游切分器而言它们都只是同名属性。

## 六、引擎如何被选择:getEmbeddingEngineSelection 的 switch

把所有 provider 接起来的总开关是 `server/utils/helpers/index.js` 里的 `getEmbeddingEngineSelection()`。它读 `process.env.EMBEDDING_ENGINE`,用一个 switch 把字符串映射到具体类,并且每个 case 都**延迟 require**对应模块——只在真正命中时才加载该 provider 的依赖,避免把所有 SDK 一次性拉进内存:

```js
function getEmbeddingEngineSelection() {
  const { NativeEmbedder } = require("../EmbeddingEngines/native");
  const engineSelection = process.env.EMBEDDING_ENGINE;
  switch (engineSelection) {
    case "openai":
      const { OpenAiEmbedder } = require("../EmbeddingEngines/openAi");
      return new OpenAiEmbedder();
    case "ollama":
      const { OllamaEmbedder } = require("../EmbeddingEngines/ollama");
      return new OllamaEmbedder();
    // ... azure / localai / lmstudio / cohere / voyageai /
    //     litellm / mistral / generic-openai / gemini /
    //     openrouter / lemonade
    case "native":
      return new NativeEmbedder();
    default:
      return new NativeEmbedder();
  }
}
```

这里有两个关键设计:其一,`case "native"` 和 `default` 返回的都是 `new NativeEmbedder()`——这意味着**只要 `EMBEDDING_ENGINE` 没设、或设了一个不认识的值,系统都自动落到 native**。这正是"零配置开箱"在选择器层面的兑现:一个全新部署、什么环境变量都没填的实例,`getEmbeddingEngineSelection()` 会安静地给你一个本地嵌入引擎,RAG 立刻可用。其二,switch 的 case 字符串(`"openai"`、`"voyageai"`、`"generic-openai"` 等)就是 `EMBEDDING_ENGINE` 的合法取值表,与 `EmbeddingEngines/` 目录下的子目录一一对应。

这个选择器并非孤立调用。`getLLMProvider({...})` 在第 138 行就用 `const embedder = getEmbeddingEngineSelection()` 取到 embedder 再注入给 LLM provider,可见嵌入引擎和对话模型在 AnythingLLM 里是解耦的两条配置线——你可以用 Anthropic 做对话、同时用 native 做嵌入。`EmbeddingWorkerManager.js` 里还有个轻量姊妹函数 `isNativeEmbedder()`:`return !engine || engine === "native"`,它不实例化任何类,只判断当前是否走 native 路径,用来决定文档入库是该走 worker 子进程(native 那条重 CPU、易 OOM 的路)还是主进程直发 API。

## 七、native 的运行容器:embedding-worker 与单例的隔离

native 引擎的内存风险大到必须被隔离,这就是 `EmbeddingWorkerManager.js` 存在的原因。它通过 `embedFiles(slug, files, workspaceId, userId)` 把"嵌入大批文档"这件重活丢进一个独立的子进程:用 `BackgroundService` 解析出 `embedding-worker.js` 脚本路径,`bg.spawnWorker(scriptPath)` 拉起 worker,然后 `worker.send({ type: "embed", files, ... })` 把任务发过去。worker 跑的就是 native 那套写盘换内存的逻辑,即便它 OOM 崩溃,主 server 进程也不受牵连——崩了会触发 `worker.on("exit")`,在 `workerCompleted` 为 false 时补发一条 `all_complete` 带 error 的 SSE 事件通知前端。

这里能看到几个工程细节互相咬合:同一个 workspace 已有 worker 在跑时,新文件通过 `worker.send({ type: "add_files", files })` 追加而非另起进程(`runningWorkers` 这张 Map 做去重);进度通过 worker 的 IPC message 冒泡到 `emitProgress` 再走 SSE 推给浏览器,与第四节里 `reportEmbeddingProgress` 的 `process.send` 分支正好对接;`eventHistory` 缓冲事件以便 SSE 断线重连时回放。也就是说,native 引擎的"进程级 pipeline 单例"约束(第二节)和"worker 子进程隔离"(本节)是配套的:每个 worker 进程内部就一个 pipeline,进程退出连同它的 ONNX session 一起被操作系统回收,绕开了 `dispose()` 是 no-op 的死结。

## 本章小结

1. AnythingLLM 的嵌入子系统位于 `server/utils/EmbeddingEngines/`,十几个 provider 并排,默认引擎是 `native`——`NativeEmbedder.defaultModel = "Xenova/all-MiniLM-L6-v2"`,一个 23MB 的本地模型,实现"不填任何 API Key 也能跑 RAG"。
2. `native` 用动态 `import("@xenova/transformers")` 把 ESM 库加载进 CommonJS server,以 `pipeline("feature-extraction", model, { cache_dir })` 在本地进程内做特征抽取,首次运行下载、后续命中缓存。
3. 模型缓存目录由 `STORAGE_DIR` 决定(`$STORAGE_DIR/models` 或仓库内 `storage/models`),`modelDownloaded` 用 `fs.existsSync(modelPath)` 判定;下载默认走 HF,失败后一次性切到 `#fallbackHost`(`https://cdn.anythingllm.com/support/models/`),刻意非递归重试。
4. native 只接受 `SUPPORTED_NATIVE_EMBEDDING_MODELS` 白名单内的模型(默认 MiniLM、nomic、multilingual-e5-small),否则回落 `defaultModel`,原因是每个模型都要在自家 CDN 镜像。
5. 因 onnxruntime-node 1.14 的 `dispose()` 是 no-op,native 用静态 `#pipelines` / `#pipelinePromises` 两张 Map 把 pipeline 做成进程级单例并去重并发首跑。
6. provider 契约是隐式约定:`embedTextInput`(单条退化为批量)、`embedChunks`(批量、要么全成要么 null/抛错)、`maxConcurrentChunks`、`embeddingMaxChunkLength`,没有显式基类但形状一致。
7. 分批工具 `toChunks(arr, size)` 全局共享,`size` 取 `maxConcurrentChunks`;调度分三派——native 串行 + 写临时文件换内存(钉死 25,避免 OOM)、openAi/localAi 并发 Promise.all(500/50,贴 API 上限)、ollama 显式批循环(默认 1,可由 `OLLAMA_EMBEDDING_BATCH_SIZE` 调)。
8. `embeddingMaxChunkLength` 三种来源:native 来自 `constants.js` 每模型字段(字符数而非 token)、openAi 硬编码 `8_191`、ollama/localAi 调 `maximumChunkLength()`(读 `EMBEDDING_MODEL_MAX_CHUNK_LENGTH`,默认 1000)。
9. voyageAi 委托给 LangChain `VoyageEmbeddings`(`batchSize: 128`),其 `embeddingMaxChunkLength` 是按 model 名分支的 switch;localAi/genericOpenAi/lmstudio 复用 OpenAI SDK 仅改 `baseURL`,接入成本极低。
10. `getEmbeddingEngineSelection()` 按 `EMBEDDING_ENGINE` 用 switch 分发并延迟 require,`case "native"` 与 `default` 都返回 `new NativeEmbedder()`,把"零配置即 native"在选择器层面坐实;嵌入引擎与 LLM provider 解耦,可独立配置。
11. native 因内存风险被隔离进 `embedding-worker.js` 子进程(`EmbeddingWorkerManager.embedFiles`),崩溃不波及主进程,进度经 IPC→SSE 回传;`reportEmbeddingProgress` 用 `process.send` 是否存在自动区分 worker/主进程两种环境;`isNativeEmbedder()` 决定走 worker 还是主进程直发 API。

## 动手实验

1. **实验一:观察 native 首跑下载与缓存命中** — 删除或重命名 `server/storage/models` 目录,在 server 里以 `EMBEDDING_ENGINE` 不设(或设 `native`)触发一次文档嵌入,观察控制台 `[NativeEmbedder - Downloading model] ... %` 进度日志与 `(~23MB)` 提示;再跑第二次,确认这次没有下载日志、`modelDownloaded` 为 true 而直接命中 `storage/models/Xenova/all-MiniLM-L6-v2`。

2. **实验二:验证选择器的默认回落** — 在 `server` 目录用 Node `require("./utils/helpers")` 取出 `getEmbeddingEngineSelection`,分别在不设 `EMBEDDING_ENGINE`、设成 `"native"`、设成一个不存在的值(如 `"foobar"`)、设成 `"openai"`(临时给个假 `OPEN_AI_KEY`)四种情况下调用,打印返回实例的 `className`,确认前三种都得到 `NativeEmbedder`、只有第四种是 `OpenAiEmbedder`,印证 `default` 分支即 native。

3. **实验三:对比两种并发节流策略** — 通读 `native/index.js` 的 `embedChunks` 与 `openAi/index.js` 的 `embedChunks`,在纸上画出二者的数据流:native 是"`for` 串行 → 每批写 `.tmp` 文件 → 末尾一次性 `readFileSync`+`rmSync`",openAi 是"`for` 全部包成 Promise → `Promise.all` 汇聚 → `Set` 去重错误"。再把 `constants.js` 里默认模型的 `maxConcurrentChunks: 25` 改成 50,理解注释里 "50 seems to overflow no matter what" 警告的含义(不必真跑,理解内存意图即可)。

4. **实验四:追踪进度回报的双环境分支** — 在 `helpers/index.js` 里定位 `reportEmbeddingProgress`,确认它先判断 `typeof process.send === "function"` 走 IPC、否则 `require("../EmbeddingWorkerManager").emitProgress` 走 SSE;再到 `EmbeddingWorkerManager.js` 找到 `embedFiles` 里 `worker.on("message", ...)` 如何把 worker 冒泡上来的事件交给 `emitProgress`,串起"worker 内 `process.send` → 管理器 `on('message')` → `emitProgress` → 浏览器 SSE"这条完整链路。

> **下一章预告**:嵌入引擎把 chunk 变成了 `number[][]` 向量,但向量本身还需要一个能持久化、能做相似度检索的家。第 5 章我们将解构 AnythingLLM 的向量数据库抽象层,看它如何用与本章同构的"统一契约 + switch 选择器"模式把 LanceDB、Chroma、Pinecone、Qdrant、Milvus 等多种向量库拍平成可互换的 provider,以及 `embeddingMaxChunkLength`、`embedChunks` 产出的向量是如何带着 chunk 原文与元数据一起落库的。
