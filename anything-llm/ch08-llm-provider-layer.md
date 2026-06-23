# 第 8 章　LLM 模型接入层(AiProviders):一份契约接通四十余家

前面七章里,我们已经把一段用户提问从采集、切分、向量化、检索一路推到了"构造提示词"这一步。但 RAG 系统真正要回答问题,终究要把构造好的消息数组交给某个大语言模型。问题在于:AnythingLLM 并不绑定任何一家厂商。打开 `server/utils/AiProviders/` 目录,你会数到四十多个子目录——`openAi`、`anthropic`、`ollama`、`gemini`、`bedrock`、`cohere`、`deepseek`、`mistral`、`groq`、`xai`、`lmStudio`、`localAi`、`koboldCPP`、`textGenWebUI`、`liteLLM`、`openRouter`……既有云端 SaaS,也有跑在本地的开源模型,还有把多家聚合起来的路由型网关。上层的聊天流程对这些差异一无所知,它只认识一个抽象:一个"LLM provider 对象",会构造提示、会做(流式或非流式)补全、会算上下文窗口、会压缩超长消息。

本章要解构的正是这层"统一契约"。我们会先读 `server/utils/helpers/index.js` 顶部那段 `BaseLLMProvider` 的 JSDoc typedef——它是整个接入层不成文却被严格遵守的接口定义;再以 OpenAI 实现作为"标准答卷"逐方法解剖;接着看流式补全里 `handleStream` 与横切其上的 `LLMPerformanceMonitor` 如何把"统计每秒吐多少 token"这件事从每个 provider 里抽离出来;然后对比 Anthropic(非 OpenAI 协议)如何在同一契约下适配自己的消息格式与缓存;再看 Ollama 代表的本地模型在上下文窗口、超时、推理标签上的特殊处理;最后回到 `getLLMProvider` 的 switch 分发,以及 `modelMap` 与 `promptWindowLimit` 共同支撑的上下文窗口管理。

## 8.1　不成文的接口:BaseLLMProvider typedef

JavaScript 没有 interface 关键字,AnythingLLM 也没有让所有 provider 继承同一个基类。它用的是一份 JSDoc `@typedef` 把契约写在 `server/utils/helpers/index.js` 顶部的注释里。`BaseLLMProvider` 被描述为"A basic llm provider object",它列出的每一个属性,都是上层代码会直接调用的方法或字段:

```js
/**
 * @typedef {Object} BaseLLMProvider - A basic llm provider object
 * @property {string} className - Provider identifier used in logs and response metrics.
 * @property {string} model - The active model name for this provider instance.
 * @property {number} defaultTemp - Default sampling temperature (typically 0.7).
 * @property {Function} streamingEnabled - Checks if streaming is enabled for chat completions.
 * @property {Function} promptWindowLimit - Returns the token limit for the current model.
 * @property {Function} isValidChatCompletionModel - Validates if the provided model is suitable for chat completion.
 * @property {Function} constructPrompt - Constructs a formatted prompt for the chat completion request.
 * @property {getChatCompletionFunction} getChatCompletion - Gets a chat completion response.
 * @property {streamGetChatCompletionFunction} streamGetChatCompletion - Streams a chat completion response.
 * @property {Function} handleStream - Handles the streaming response.
 * @property {Function} embedTextInput - Embeds the provided text input using the specified embedder.
 * @property {Function} embedChunks - Embeds multiple chunks of text using the specified embedder.
 * @property {Function} compressMessages - Compresses chat messages to fit within the token limit.
 */
```

把这份清单翻译成工程语言:任何想接入 AnythingLLM 的 provider,实例上必须有三个字段——`className`(写日志和回填到 metrics 里的标识)、`model`(当前实例选中的模型名)、`defaultTemp`(默认采样温度,清单里直接注明"typically 0.7");还必须实现一组方法。其中 `streamingEnabled` 判断能否流式,`promptWindowLimit` 返回当前模型的 token 上限,`isValidChatCompletionModel` 校验模型是否可用于对话补全,`constructPrompt` 把系统提示、上下文、历史、用户输入拼成消息数组,`getChatCompletion` / `streamGetChatCompletion` 分别做非流式与流式补全,`handleStream` 负责把流式响应一段段写回客户端,`embedTextInput` / `embedChunks` 把嵌入能力转发给 embedder,`compressMessages` 在消息超出窗口时做压缩。

紧挨着它的还有一个更小的 `BaseLLMProviderClass` typedef,只要求一个静态方法 `promptWindowLimit(modelName)`:"Returns the token limit for the provided model." 这一条之所以要单独立出来,是因为系统有时候并不想实例化一个 provider(实例化往往要求环境变量里有 API key),只想问一句"这个模型的上下文窗口是多大"。把它声明成静态方法,就能在不持有实例的情况下查询窗口大小。后文 8.7 节会看到它的用处。

理解这层契约的关键是:`getChatCompletion` 的入参与返回值也被 typedef 钉死了。`ChatCompletionResponse` 规定返回 `{ textResponse, metrics }`,而 `ResponseMetrics` 规定 `metrics` 里要有 `prompt_tokens`、`completion_tokens`、`total_tokens`、`outputTps`、`duration` 这五个数字。也就是说,无论底层是 OpenAI 的 `usage.input_tokens`、Anthropic 的 `usage.input_tokens` 还是 Ollama 的 `prompt_eval_count`,每个 provider 都得把自己厂商方言里的用量字段,翻译成这套统一的指标命名后再交出来。契约不仅约束"有哪些方法",还约束"返回什么形状的数据"。

## 8.2　标准答卷:逐方法解剖 OpenAI 实现

`server/utils/AiProviders/openAi/index.js` 里的 `OpenAiLLM` 类是这份契约的参考实现,也是其余几十家"OpenAI 兼容"provider 模仿的模板。我们顺着 typedef 的方法清单逐一拆。

构造函数定下了整个实例的基调。它先做一个硬性前置检查:`if (!process.env.OPEN_AI_KEY) throw new Error("No OpenAI API key was set.")`——没有 API key 就直接抛错,而不是延迟到第一次请求才失败。接着 `this.className = "OpenAiLLM"`,`this.model = modelPreference || process.env.OPEN_MODEL_PREF || "gpt-4o"`,这条三段式回退很典型:优先用调用方显式传入的 `modelPreference`,其次用环境变量 `OPEN_MODEL_PREF`,最后兜底 `"gpt-4o"`。`this.defaultTemp = 0.7` 兑现了 typedef 里的约定值。embedder 则是 `this.embedder = embedder ?? new NativeEmbedder()`——如果调用方没注入嵌入引擎,就退回本地的 `NativeEmbedder`,这呼应了第 4 章嵌入引擎的设计。

构造函数里还藏着一个很重要的细节——`this.limits`:

```js
this.limits = {
  history: this.promptWindowLimit() * 0.15,
  system: this.promptWindowLimit() * 0.15,
  user: this.promptWindowLimit() * 0.7,
};
```

它把当前模型的上下文窗口按比例切成三块预算:历史占 15%、系统提示占 15%、用户输入占 70%。这个 `limits` 对象本身在补全时并不直接生效,而是 8.7 节的 `compressMessages` 用来做 token 裁剪的依据。把"窗口怎么分配"在构造时就固化下来,意味着同一个 provider 实例在整个对话生命周期里预算口径一致。

`streamingEnabled()` 的实现简洁到有点出人意料:`return "streamGetChatCompletion" in this;`。它不是返回某个布尔开关,而是用 `in` 运算符检测实例上是否存在 `streamGetChatCompletion` 这个方法。换句话说,"是否支持流式"等价于"是否实现了流式方法"——一个 provider 只要写了 `streamGetChatCompletion`,就自动被认为支持流式;反之删掉这个方法就自动降级为非流式。这是用语言特性把"能力声明"和"能力实现"绑成同一件事,杜绝了二者不一致的可能。

`promptWindowLimit` 有静态和实例两个版本,内部都委托给 modelMap:

```js
static promptWindowLimit(modelName) {
  return MODEL_MAP.get("openai", modelName) ?? 4_096;
}
promptWindowLimit() {
  return MODEL_MAP.get("openai", this.model) ?? 4_096;
}
```

注意那个 `?? 4_096`——当 modelMap 查不到对应模型时,兜底返回 4096。这个保守的默认值意味着即便模型映射缺失,系统也不会用一个虚高的窗口去撑爆请求。modelMap 的工作机制留到 8.7 节细讲。

`isValidChatCompletionModel` 体现了一个性能与正确性的权衡。它先做"短路":如果模型名(小写后)包含 `gpt`、或以 `o` 开头(对应 o 系列推理模型),就直接 `return true`。只有在不满足这两个预设特征时,才真去调 `this.openai.models.retrieve(modelName)` 远程确认。源码注释把意图写得很明白:因为模型列表是通过用户 API key 实时从 OpenAI 拉取的,所以理应是真实可用的,"we don't want to hit the OpenAI api every chat because it will get spammed and introduce latency for no reason"——不想每次聊天都打一次模型查询接口,徒增延迟又容易被限流。

`constructPrompt` 是把"散件"拼成消息数组的地方。它把系统提示与上下文用一个私有方法 `#appendContext` 拼到一起,作为 `role: "system"` 的首条消息;然后用 `formatChatHistory(chatHistory, this.#generateContent)` 摊开历史;最后追加 `role: "user"` 的当前输入。`#appendContext` 的格式是把每条检索到的上下文包进 `[CONTEXT i]:\n...\n[END CONTEXT i]` 的标记里——这种显式边界标记能帮模型分清哪段是检索结果、哪段是指令。`#generateContent` 则负责多模态:无附件时直接返回字符串,有附件时返回 `[{type:"input_text",...}, {type:"input_image", image_url:...}]` 这样的内容数组,贴合 OpenAI 新版 Responses API 的输入格式 [[OpenAI Responses API]](https://platform.openai.com/docs/api-reference/responses)。

值得专门一提的是 `#temperature` 这个私有方法。它定义了一个 `NO_TEMP_MODELS = ["o", "gpt-5"]` 前缀列表,凡是模型名以这些前缀开头的(o 系列、gpt-5 系列),一律把温度强制改成 `1`,因为这些模型不接受自定义温度,源码注释写道"OpenAI accepts temperature 1"。这种"按模型特性修正请求参数"的小补丁,在每个 provider 里都会以不同形式出现——它正是统一契约下不得不容纳的厂商个性。

非流式的 `getChatCompletion` 先用 `isValidChatCompletionModel` 把关,然后把对 `this.openai.responses.create(...)` 的调用整个交给 `LLMPerformanceMonitor.measureAsyncFunction(...)` 包裹。返回时它把 OpenAI 的 `usage` 字段翻译成统一 metrics:`prompt_tokens: usage.input_tokens || 0`、`completion_tokens: usage.output_tokens || 0`,并自己算出 `outputTps: usage.output_tokens / result.duration`,把 `duration`、`model`、`provider: this.className`、`timestamp` 一并塞进去。这正是 8.1 节强调的"翻译厂商方言"在实现层的落地。

最后是 embedding 与压缩的三个方法。`embedTextInput` 和 `embedChunks` 都只是薄薄一层转发:`return await this.embedder.embedTextInput(textInput)`,源码注释直白地称之为"Simple wrapper for dynamic embedder & normalize interface for all LLM implementations"——把 embedder 包一层,只是为了让所有 LLM 实现对外暴露同一套嵌入接口。`compressMessages` 则先 `constructPrompt`,再调用 `messageArrayCompressor(this, messageArray, rawHistory)`,这条压缩链 8.7 节再展开。

## 8.3　流式补全:handleStream 与 LLMPerformanceMonitor 的横切监控

流式补全比一次性补全复杂得多:既要逐 token 写回客户端,又要处理客户端中途断开,还要在结束时统计出"总共花了多久、每秒吐了多少 token"。AnythingLLM 把这件事拆成了两半——provider 自己负责"怎么把流写给客户端"(`handleStream`),而"怎么计时和算速率"则被抽到一个横切组件 `LLMPerformanceMonitor` 里,所有 provider 共享。

先看监控这层。`server/utils/helpers/chat/LLMPerformanceMonitor.js` 里的 `LLMPerformanceMonitor` 是个纯静态类,它内部持有一个 `tokenManager = new TokenManager()`,用于在厂商不回传 prompt token 数时自己估算。它对外提供两个核心方法。`measureAsyncFunction(func)` 给非流式调用计时:记下开始时间,`await` 掉传入的 promise,算出 `duration`——并且有个巧妙的优先级:`const duration = output?.usage?.duration ?? (end - start) / 1000;`。也就是说,如果 provider 在结果里给了更精确的 `usage.duration`(比如 Ollama 用模型自报的 `eval_duration`),就用它;否则才退回墙钟时间。

流式那侧的 `measureStream` 是本章最有意思的横切设计。它接收一个 `func`(即 provider 发起的流式请求 promise)以及 `messages`、`runPromptTokenCalculation`、`modelTag`、`provider` 等参数,`await` 出真正的流对象后,直接在这个流对象身上挂载状态与方法:

```js
const stream = await func;
stream.start = Date.now();
stream.duration = 0;
stream.metrics = {
  completion_tokens: 0,
  prompt_tokens: runPromptTokenCalculation ? this.countTokens(messages) : 0,
  total_tokens: 0,
  outputTps: 0,
  duration: 0,
  ...(modelTag ? { model: modelTag } : {}),
  ...(provider ? { provider: provider } : {}),
};
stream.endMeasurement = (reportedUsage = {}) => {
  const end = Date.now();
  const estimatedDuration = (end - stream.start) / 1000;
  stream.metrics = {
    ...stream.metrics,
    ...reportedUsage,
    duration: reportedUsage?.duration ?? estimatedDuration,
    timestamp: new Date(),
  };
  stream.metrics.total_tokens =
    stream.metrics.prompt_tokens + (stream.metrics.completion_tokens || 0);
  stream.metrics.outputTps =
    stream.metrics.completion_tokens / stream.metrics.duration;
  return stream.metrics;
};
return stream;
```

这就是"装饰器"思路的极简实现:它不另造一个包装对象,而是把 `start`、`metrics`、`endMeasurement` 直接长在原始流上,返回的还是那个可以 `for await` 的流。`runPromptTokenCalculation` 这个开关也很关键——多数 provider 把它设为 `false`(因为它们能从流的结束事件里拿到厂商回传的真实 prompt token),只有那些流里完全不带用量的 provider 才需要让监控器用 `countTokens(messages)` 自己估算。`endMeasurement` 把 provider 在流结束时收集到的 `reportedUsage` 合并进来,再统一计算 `total_tokens` 和 `outputTps`(每秒输出 token 数)。

有了这套监控,provider 的 `handleStream` 就只需专注"怎么读流、怎么写回客户端、什么时候调一次 `endMeasurement`"。以 OpenAI 为例,`handleStream(response, stream, responseProps)` 返回一个 Promise,内部 `for await (const chunk of stream)` 遍历每个事件。当 `chunk.type === "response.output_text.delta"` 时取出 `chunk.delta` 作为新 token,累加进 `fullText`,并通过 `writeResponseChunk` 以 `type: "textResponseChunk"`、`close: false` 推给客户端;当 `chunk.type === "response.completed"` 时,从 `res.usage` 里取出真实的 input/output/total tokens 填进 `usage`,再发一条 `close: true` 的收尾 chunk,然后 `stream?.endMeasurement(usage)` 触发指标结算并 `resolve(fullText)`。

最值得学习的是中途断开的处理。函数一开始就注册了 `response.on("close", handleAbort)`,而 `handleAbort` 会先 `stream?.endMeasurement(usage)`——也就是说即便用户提前关掉连接,已经生成的部分仍会被计入指标——再调 `clientAbortedHandler(resolve, fullText)` 把已生成的文本作为结果返回,而不是抛错丢弃。源码里这段被各家 provider 几乎逐字复用,注释道出了意图:"We preserve the generated text but continue as if chat was completed to preserve previously generated content." 正常完成的分支里则会 `response.removeListener("close", handleAbort)` 把这个监听器摘掉,避免重复结算。整段流程把"正常结束、异常报错、客户端中断"三条路径都收口到了"调一次 `endMeasurement` + `resolve(fullText)`"这个统一出口。

## 8.4　非 OpenAI 协议如何适配:以 Anthropic 为例

OpenAI 那套是"OpenAI 兼容"provider 的模板,但真正能证明契约威力的,是一个协议完全不同的厂商如何在同一接口下落地。`server/utils/AiProviders/anthropic/index.js` 的 `AnthropicLLM` 就是反例样本——它用的是 `@anthropic-ai/sdk`,消息格式、用量字段、流式事件类型都与 OpenAI 不一样,但对外暴露的方法签名分毫不差。

第一处差异在系统消息的位置。Anthropic 的 Messages API 把系统提示作为顶层 `system` 参数单独传,而不是混在 `messages` 数组里。所以 `getChatCompletion` 里有这样两行:`system: this.#buildSystemPrompt(systemContent)` 和 `messages: messages.slice(1)`,后面那行的注释直接写着 `// Pop off the system message`——它把 `constructPrompt` 拼出来的数组里第 0 条系统消息切掉,只把其余消息塞进 `messages`,而把切下来的系统内容提到顶层。同一个 `constructPrompt` 产出的消息数组,在 OpenAI 那边整个传进去,在 Anthropic 这边却要"掐头",这正是适配层的活儿。

第二处差异是 `max_tokens` 必填。Anthropic 的接口要求显式声明最大输出 token 数,所以 `AnthropicLLM` 在构造时就启动一个异步任务 `AnthropicLLM.fetchModelMaxTokens(this.model)` 去拉取该模型的 `max_tokens` 并缓存进 `this.maxTokens`;`fetchModelMaxTokens` 内部调 `anthropic.models.retrieve(modelName)`,失败时兜底 `4096`。每次补全前还会 `await this.assertModelMaxTokens()` 确保这个值已就绪。这与 OpenAI 实现里压根不需要 `max_tokens` 形成鲜明对比——契约只规定要有 `getChatCompletion`,至于内部要不要预取额外参数,完全由各家自理。

第三处差异是温度处理与缓存控制。Anthropic 用 `temperatureParam` 方法替代 OpenAI 的 `#temperature`:它维护一个 `noTemperatureModels`(含 `"claude-opus-4-7"`、`"claude-opus-4-8"`),命中就返回 `undefined`(即不传温度,因为这些模型会对 `temperature`/`top_p`/`top_k` 报 400 错)。此外它还实现了 OpenAI 没有的 prompt caching:`get cacheControl` 读环境变量 `ANTHROPIC_CACHE_CONTROL`,只接受 `5m` 或 `1h`,据此返回 `{ type: "ephemeral", ttl }`;`#buildSystemPrompt` 在开启缓存时把系统提示包成带 `cache_control` 字段的结构 [[Anthropic Prompt Caching]](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)。这些都是 Anthropic 特有能力,契约没要求、也不阻止——provider 可以在满足契约的前提下自由扩展。

第四处差异最能体现"同名方法、异样实现"的是 `handleStream`。OpenAI 用的是 `for await` 拉取 chunk,Anthropic 用的却是事件监听:`stream.on("error", ...)`、`stream.on("streamEvent", ...)`。在 `streamEvent` 回调里,它根据 `data.type` 分别处理——`message_start` 时取 `usage.input_tokens` 作为 prompt token,`message_delta` 时取 `output_tokens` 作为 completion token,`content_block_delta` 且 `delta.type === "text_delta"` 时才把 `data.delta.text` 当作可见文本写回。流终止的判定也不同,是 `message.type === "message_stop"` 或 `data.stop_reason === "end_turn"`。尽管事件模型截然不同,这个 `handleStream` 与 OpenAI 那个共享完全一致的"骨架":同样注册 `response.on("close", handleAbort)`,同样在每个出口调 `stream?.endMeasurement(usage)`,同样用 `writeResponseChunk` 写回、用 `resolve(fullText)` 收尾。骨架统一、填充各异,正是这层契约设计的精髓。

还有一处容易被忽略但很说明问题的差异:Anthropic 的 `promptWindowLimit` 兜底值是 `?? 100_000`,而非 OpenAI 的 `?? 4_096`;`compressMessages` 调的也是 `messageStringCompressor` 而不是 `messageArrayCompressor`(对应它把系统提示拼成字符串的处理方式)。可见连"窗口兜底"和"压缩策略"都允许逐家定制。

## 8.5　本地模型的差异:Ollama 的窗口、超时与推理标签

如果说 Anthropic 展示的是"协议不同",那 `server/utils/AiProviders/ollama/index.js` 的 `OllamaAILLM` 展示的是"运行环境不同"。本地模型没有云端那种稳定的元数据接口,模型是用户自己拉下来的,上下文窗口千差万别,响应可能极慢,还可能输出推理过程——这些都逼出了一套独特处理。

最突出的是上下文窗口的"动态发现"。云端 provider 直接查 modelMap 就行,但本地模型五花八门,modelMap 根本不可能预先收录。于是 `OllamaAILLM` 用一个静态字段 `static modelContextWindows = {}` 做缓存,并在构造时调用 `OllamaAILLM.cacheContextWindows(true)`。这个方法会 `client.list()` 列出本地所有模型,再对每个模型 `client.show(...)` 取详情,从 `model_info` 里找那个以 `.context_length` 结尾的键(不同架构的键名前缀不同),把窗口大小缓存起来;找不到就退回 `4096`,带 embedding 能力的模型则跳过。注释强调这"absolutely necessary to ensure that the context windows are correct"——本地模型的窗口只能现场问,问到了缓存一份供整个进程生命周期复用。

`promptWindowLimit` 的逻辑也因此比云端复杂得多。它综合三个来源:用户在环境变量 `OLLAMA_MODEL_TOKEN_LIMIT` 里手设的上限、`cacheContextWindows` 探测到的系统上限、以及一个硬编码的安全帽。优先级是:用户设了就取"用户值与系统值的较小者"(`Math.min(userDefinedLimit, systemDefinedLimit)`,即用户偏好优先但不得超过模型物理上限);用户没设,则返回 `Math.min(systemDefinedLimit, 16384)`。后面这个 16384 的封顶很有意思,注释解释道:防止用户没指定时就直接吃满某些模型动辄上百 K 的超大窗口——超大窗口会让本地推理慢到不可用,所以默认给个温和上限。这与 OpenAI 那种简单的 `?? 4_096` 兜底,体现了完全不同的设计顾虑。

`limits` 的初始化时机也不同。OpenAI 在构造函数里同步算好 `limits`,Ollama 却把它设为 `this.limits = null`,推迟到 `assertModelContextLimits()` 里才填,且这个 assert 要 `await OllamaAILLM.cacheContextWindows()`。注释说这是"Lazy load the limits to avoid blocking the main thread on cacheContextWindows"——因为探测窗口要发网络请求,不能让它阻塞构造。所以 `compressMessages` 一进来就先 `await this.assertModelContextLimits()`,确保压缩前 `limits` 已经备好。

针对本地推理慢的现实,还有 `applyOllamaFetch`:当环境变量里设了 `OLLAMA_RESPONSE_TIMEOUT` 且大于 5 分钟时,它用 `undici` 的 `Agent` 自定义一个更长 `headersTimeout` 的 fetch,绕开默认超时,好让"跑得很慢的机器"也能等到结果;`keepAlive` 则默认 `300` 秒,对应注释"Default 5-minute timeout for Ollama model loading",控制模型在显存里驻留多久。

最后是推理标签的处理,这在云端 provider 里基本看不到。Ollama 从 v0.9.0+ 起,把"思考过程"放在响应的独立 `thinking` 字段里。`handleStream` 因此要维护一个额外的 `reasoningText` 缓冲:当 chunk 带 `message.thinking` 时,首段前面补一个 `<think>` 起始标签写回客户端,后续推理 token 继续追加;一旦开始出现正式 `content` 而推理缓冲非空,就先补一个 `</think>` 闭合标签再写正文。也就是说,它在流里手动把模型的思考过程用 `<think>...</think>` 包起来,让前端能区分"推理"和"答案"。非流式的 `getChatCompletion` 里也有对应处理:`if (res.message.thinking) content = \`<think>${res.message.thinking}</think>${content}\``。这是本地开源推理模型(如带 thinking 的 qwen3、deepseek-r1 类)催生出的、纯属 Ollama 侧的契约扩展。

此外 Ollama 还额外实现了一个 `getModelCapabilities()`,通过 `client.show` 返回的 `capabilities` 数组判断模型是否支持 `tools`、`thinking`、`vision`,失败时返回一组 `"unknown"`。这个方法不在 `BaseLLMProvider` 契约里,但被 Agent 等上层功能用来探测本地模型能力——又一个"在契约之上自由增项"的例子。

## 8.6　provider 是怎么被选出来的:getLLMProvider 的 switch 分发

到此我们已经看了三家具体实现,但上层代码从不直接 `new OpenAiLLM()`。所有 provider 的实例化都收口在 `server/utils/helpers/index.js` 的 `getLLMProvider` 这一个工厂函数里:

```js
function getLLMProvider({ provider = null, model = null } = {}) {
  const LLMSelection = provider ?? process.env.LLM_PROVIDER ?? "openai";
  const embedder = getEmbeddingEngineSelection();

  switch (LLMSelection) {
    case "openai":
      const { OpenAiLLM } = require("../AiProviders/openAi");
      return new OpenAiLLM(embedder, model);
    case "anthropic":
      const { AnthropicLLM } = require("../AiProviders/anthropic");
      return new AnthropicLLM(embedder, model);
    case "ollama":
      const { OllamaAILLM } = require("../AiProviders/ollama");
      return new OllamaAILLM(embedder, model);
    // ... 其余三十余个 case ...
    default:
      throw new Error(
        `ENV: No valid LLM_PROVIDER value found in environment! Using ${process.env.LLM_PROVIDER}`
      );
  }
}
```

这个函数有三个值得拆解的设计点。第一是 provider 的来源优先级:`provider ?? process.env.LLM_PROVIDER ?? "openai"`——调用方显式传入优先,其次读环境变量 `LLM_PROVIDER`,最后兜底 `"openai"`。这与每个 provider 内部"模型名三段式回退"是同一种风格,层层都给默认值,让系统在配置缺失时仍能跑起来。

第二是 `require` 写在 case 内部而非文件顶部。每个分支都是"命中了才 `require`"那个 provider 模块,这是刻意的惰性加载:四十多个 provider 各自依赖不同的 SDK(`openai`、`@anthropic-ai/sdk`、`ollama`……),如果在文件顶部全量 `require`,启动时就得把所有 SDK 都加载进内存,且任何一个 SDK 的初始化副作用都会被触发。改成 case 内 `require`,就只加载用户当前真正选用的那一家。

第三是 embedder 的统一注入。函数开头 `const embedder = getEmbeddingEngineSelection()` 先按系统配置选出嵌入引擎,再作为第一个参数传给每个 provider 构造函数。这解释了为什么所有 provider 的构造函数签名都是 `constructor(embedder = null, modelPreference = null)`——签名一致,正是工厂能用同一行 `new XxxLLM(embedder, model)` 套住所有分支的前提。

`switch` 里的 case 标签本身就是一份 provider 清单:`openai`、`azure`、`anthropic`、`gemini`、`lmstudio`、`localai`、`ollama`、`togetherai`、`fireworksai`、`perplexity`、`openrouter`、`mistral`、`groq`、`koboldcpp`、`textgenwebui`、`cohere`、`litellm`、`generic-openai`、`bedrock`、`deepseek`、`apipie`、`novita`、`xai`、`nvidia-nim`、`ppio`、`moonshotai`、`cometapi`、`foundry`、`zai`、`giteeai`、`docker-model-runner`、`privatemode`、`sambanova`、`lemonade`、`minimax`、`cerebras`,再加上特殊的 `anythingllm-router`——后者并不真的实例化,而是 `throw` 一个错误,提示它必须经由 `AnythingLLMModelRouter` 类在 `stream.js` 里单独处理,不能走 `getLLMProvider` 直连。这个特例提醒我们:聚合路由型 provider 的解析路径与普通 provider 不同。

文件里还并排着一个 `getLLMProviderClass({ provider })`,结构与 `getLLMProvider` 几乎一模一样,区别是它 `return OpenAiLLM`(类本身)而非 `return new OpenAiLLM(...)`(实例)。它对应的就是 8.1 节那个只要求静态 `promptWindowLimit` 的 `BaseLLMProviderClass` 契约——当系统只想查"某模型窗口多大"而不想(也不能,因为可能没配 API key)实例化时,就用它拿到类、调静态方法。`getLLMProvider` 的 JSDoc 上还挂了一条 `@notice`,提醒优先用 `resolveProviderConnector`,因为后者会正确处理 `anythingllm-router`;`getLLMProvider` 只在确定不涉及路由时才该直接用。这层层封装,本质都是为了把"选哪家 provider"这个决策牢牢收在一处。

## 8.7　上下文窗口管理:modelMap 与 compressMessages 的协作

最后这节把前面散落的几条线收束起来:`promptWindowLimit` 返回的那个数字到底从哪来?算出来又被谁用?答案是 modelMap 提供"窗口有多大",`compressMessages` 决定"超了怎么办"。

窗口数据由 `server/utils/AiProviders/modelMap/index.js` 的 `ContextWindowFinder` 类管理,它以单例形式导出为 `MODEL_MAP`(`module.exports = { MODEL_MAP: new ContextWindowFinder() }`)。它的核心是一份"provider → 模型 → 窗口大小"的两层映射。这份映射有两个来源:静态兜底来自 `legacy.js`(`LEGACY_MODEL_MAP`,里面硬编码了诸如 `"claude-3-5-sonnet-20241022": 200000` 这样的条目);动态来源则是从 LiteLLM 的开源数据文件远程拉取——`remoteUrl` 指向 `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json`,缓存有效期 `expiryMs` 为 3 天。构造时若发现缓存过期或不存在,就在后台异步 `#pullRemoteModelMap()` 重新拉取,写进 `STORAGE_DIR/models/context-windows/` 下。

拉回来的原始数据并不能直接用,要经过 `#formatModelMap` 转换。这里有个映射表 `trackedProviders`,把 AnythingLLM 的 provider 名对应到 LiteLLM 的 provider 标签,比如 `gemini: "vertex_ai-language-models"`、`cohere: "cohere_chat"`、`zai: "vercel_ai_gateway"`。转换时按 `config.litellm_provider` 过滤出该家的模型,取 `config.max_input_tokens` 作为窗口大小,并且 `key.split("/").pop()` 把可能带 `provider/model` 前缀的键名只留最后一段。随后 `#validateModelMap` 会剔除窗口值非正数的脏条目。对外则统一通过 `get(provider, model)` 查询:provider 不存在或缓存缺失返回 `null`,只给 provider 不给 model 返回该家整张表,两者都给则返回具体窗口数(找不到也返回 `null`)。各 provider 的 `promptWindowLimit` 拿到 `null` 后,再用各自的 `??` 兜底值接住——这就解释了 8.2 节那个 `?? 4_096` 是怎么生效的。

有了窗口大小,真正的裁剪发生在 `server/utils/helpers/chat/index.js` 的 `messageArrayCompressor`(以及给字符串型补全用的 `messageStringCompressor`)。文件头部的大段注释把策略讲得很透:系统提示和历史各自"at best 15% of token capacity",用户输入则可占大头,这正对应 provider 构造时算出的 `this.limits`(15%/15%/70%)。压缩入口先做一个快速放行:

```js
const tokenBuffer = 600;
const tokenManager = new TokenManager(llm.model);
if (tokenManager.statsFrom(messages) + tokenBuffer < llm.promptWindowLimit())
  return messages;
```

也就是说,如果当前消息总 token 数加上 600 的安全缓冲仍小于窗口上限,就原样返回、根本不压缩。只有真的超了,才进入分块裁剪:先看用户输入是否 `> llm.limits.user`,是则把它当成一个"霸占整条线程的独立长提示"优先保留并裁剪;再检查系统提示是否超 `llm.limits.system`,超了就把系统提示和上下文分别按 `* 0.25` 和 `* 0.75` 的目标尺寸压缩;最后用一个滑动窗口处理历史,只放行能塞进 `llm.limits.history` 的若干轮对话,单轮过大的还会按 `Math.floor(llm.limits.history / 2.2)` 这样的目标尺寸进一步裁剪。整套逻辑的注释自陈"our is much more simple"——它没有用额外模型去做摘要式压缩,而是用 tiktoken 计数加比例预算做确定性的裁剪。这与第 7 章 RAG 聊天流程在 provider 之外的取舍互相呼应,只是这里聚焦在 provider 侧:窗口由 `promptWindowLimit` 给出,预算由 `limits` 切分,裁剪由 `compressMessages` 委托的压缩器执行,三者环环相扣。

把这条链整体看一遍:用户选了某个 provider 和模型 → `getLLMProvider` 实例化对应类 → 构造时调 `promptWindowLimit()`(走 `MODEL_MAP.get` 或本地探测)拿到窗口并算出 `limits` → 上层调 `compressMessages` 时,压缩器用这个窗口和这套预算把超长消息裁到能放进去 → 裁好的消息交给 `getChatCompletion` / `streamGetChatCompletion` 发给厂商。一条提问之所以能在不同模型间无缝切换,靠的就是这条从 modelMap 到压缩器、被统一契约串起来的流水线。

## 本章小结

1. AnythingLLM 的 LLM 接入层用一份写在 `server/utils/helpers/index.js` 顶部的 JSDoc `@typedef` `BaseLLMProvider` 作为不成文契约,而非继承基类;它规定了 `className`、`model`、`defaultTemp` 三个字段和 `streamingEnabled`、`promptWindowLimit`、`constructPrompt`、`getChatCompletion`、`streamGetChatCompletion`、`handleStream`、`compressMessages` 等一组必实现方法。
2. 契约不仅约束方法名,还约束返回数据形状:`getChatCompletion` 必须回 `{ textResponse, metrics }`,且 metrics 含 `prompt_tokens`/`completion_tokens`/`total_tokens`/`outputTps`/`duration`,每家 provider 都要把厂商方言里的用量字段翻译成这套统一命名。
3. `OpenAiLLM` 是参考实现:构造时硬校验 `OPEN_AI_KEY`、用三段式回退定模型(默认 `gpt-4o`)、按 15%/15%/70% 切出 `limits`;`streamingEnabled()` 用 `"streamGetChatCompletion" in this` 把"能力声明"与"能力实现"绑成一件事。
4. `LLMPerformanceMonitor` 是横切监控组件:`measureStream` 把 `start`/`metrics`/`endMeasurement` 直接挂到原始流对象上(极简装饰器),让 provider 的 `handleStream` 只管读流写回、在每个出口调一次 `endMeasurement` 结算 `outputTps` 与 `duration`。
5. `handleStream` 的统一骨架把"正常结束、报错、客户端中断(`response.on("close", handleAbort)`)"三条路径都收口到"`endMeasurement(usage)` + `resolve(fullText)`",中断时保留已生成文本而非丢弃。
6. `AnthropicLLM` 证明非 OpenAI 协议也能套同一契约:它把系统提示提到顶层 `system` 并 `messages.slice(1)` 掐掉首条、必须预取 `max_tokens`、用事件监听式 `stream.on("streamEvent")` 而非 `for await`,还在契约之上扩展了 prompt caching。
7. `OllamaAILLM` 体现本地模型的特殊性:用静态 `modelContextWindows` 现场探测窗口、`promptWindowLimit` 在用户值/系统值之间取 `Math.min` 并对未指定情况封顶 16384、惰性初始化 `limits`、自定义长超时 fetch,并在流里用 `<think>...</think>` 手动包裹推理 token。
8. `getLLMProvider` 是唯一的实例化工厂:provider 选择走 `provider ?? process.env.LLM_PROVIDER ?? "openai"`,每个 case 内部惰性 `require` 对应 SDK,并统一注入 `getEmbeddingEngineSelection()` 选出的 embedder;`anythingllm-router` 是特例,会抛错要求改走 `AnythingLLMModelRouter`。
9. `getLLMProviderClass` 与工厂同构但返回类而非实例,对应只要求静态 `promptWindowLimit` 的 `BaseLLMProviderClass` 契约,用于在不实例化(无需 API key)时查询窗口。
10. `ContextWindowFinder`(单例 `MODEL_MAP`)管理窗口数据:静态兜底来自 `legacy.js`,动态来源每 3 天从 LiteLLM 的 `model_prices_and_context_window.json` 远程拉取,经 `trackedProviders` 映射与 `#formatModelMap` 转换后通过 `get(provider, model)` 提供,查不到由各 provider 的 `??` 兜底值接住。
11. `messageArrayCompressor` / `messageStringCompressor` 决定超窗怎么办:先用 `tokenBuffer = 600` 的缓冲判断是否需要压缩,真超了才按 `limits` 的 15%/15%/70% 预算用 tiktoken 计数做确定性裁剪,而非用额外模型做摘要。

## 动手实验

1. **实验一:验证 streamingEnabled 的"能力即实现"语义** — 在本地 clone 的 AnythingLLM 里打开 `server/utils/AiProviders/openAi/index.js`,临时把 `streamGetChatCompletion` 方法整段注释掉,写一小段脚本 `const llm = require(...).OpenAiLLM; console.log(new llm().streamingEnabled())`(或直接读 `streamingEnabled` 源码推演),观察返回值从 `true` 变为 `false`,确认 `"streamGetChatCompletion" in this` 是如何把"删掉方法"自动等价于"关闭流式"的。
2. **实验二:给 OpenAI metrics 加一条自定义字段** — 在 `getChatCompletion` 返回的 `metrics` 对象里追加一个 `provider_raw_usage: usage` 字段,然后顺着第 7 章学到的聊天流程触发一次非流式补全,在数据库的聊天记录或日志里找到回填的 metrics,核对它确实带上了你新增的字段,体会"统一 metrics 形状"是如何贯穿到持久层的。
3. **实验三:观察 modelMap 的远程拉取与兜底** — 删除 `storage/models/context-windows/` 目录后重启 server,在日志里找到 `[ContextWindowFinder] Pulling remote model map...` 与 `Remote model map synced and cached`;再断网重启一次,观察它如何打印缓存缺失警告并让 `promptWindowLimit` 退回各 provider 的 `?? 4_096` / `?? 100_000` 兜底值。
4. **实验四:触发 compressMessages 的裁剪路径** — 把某个小窗口模型(或临时改小 `promptWindowLimit` 的兜底返回值)配为当前 provider,然后发送一段远超窗口的超长用户输入,在 `server/utils/helpers/chat/index.js` 的 `messageArrayCompressor` 里加日志打印 `tokenManager.statsFrom(messages)`、`llm.promptWindowLimit()` 和裁剪前后的消息长度,亲眼看到 `tokenBuffer = 600` 的快速放行判断失败后,system/context/history 是如何按 15%/15%/70% 预算被逐块压缩的。

> **下一章预告**:本章我们看清了"一份契约接通四十余家"的 LLM 接入层——但模型只是会"说话"的大脑,真正让 AnythingLLM 能"动手"(调用工具、搜索网页、读写文件、多步推理)的,是建立在这层 provider 之上的 Agent 系统。第 9 章我们将深入 `aibitat`,解构它如何在统一的 LLM 契约之上编排多智能体对话、注册并调度工具、管理 Agent 的回合与状态,把"补全一段文本"升级为"完成一项任务"。
