# 第 7 章 一条消息的旅程：RAG 聊天与检索主流程

前六章把 AnythingLLM 的"地基"逐层铺好了：采集器把文档变成纯文本，TextSplitter 把文本切成块，EmbeddingEngines 把块变成向量，向量数据库抽象层把向量存进命名空间，DocumentManager 与工作区把这些资产组织成可检索的单元。本章要做的，是把这些零件串起来，回答一个最朴素也最关键的问题：当用户在某个工作区里敲下一句话、按下回车，到屏幕上逐字浮现出一段带引用的回答之间，服务端究竟发生了什么。

这条主流程的全部逻辑几乎集中在一个函数里——`server/utils/chats/stream.js` 的 `streamChatWithWorkspace`。它的非流式同胞、嵌入式聊天版本只是它的简化或变体，因此本章以它为主线,沿着"模式判定 → 检索 → 上下文组装 → 调 LLM → citations 回填 → 拒答兜底"逐段拆解，把每一个分支、每一个默认值都钉在源码上。

## 入口：命令、Agent 与模型路由的三道前置闸门

`streamChatWithWorkspace` 的签名透露了它要接的所有上下文：`(response, workspace, message, chatMode = "automatic", user = null, thread = null, attachments = [])`。注意第一个参数是 Express 的 `response` 对象本身——这是流式接口的标志，函数不会 `return` 一段文本，而是边算边通过 `writeResponseChunk(response, ...)` 往 SSE 流里写数据块。

函数体一开始就生成了一个 `const uuid = uuidv4()`，这个 uuid 会贯穿整次对话的所有响应块，前端据此把流式碎片拼回同一条消息。紧接着是三道"早退闸门"，它们的共同特点是：一旦命中,RAG 主流程根本不会启动。

第一道是斜杠命令。`const updatedMessage = await grepCommand(message, user)` 把用户消息里的预置命令（preset）展开成对应 prompt，但如果整条消息本身就是一个内置命令（目前 `VALID_COMMANDS` 只有 `"/reset"`，定义在 `server/utils/chats/index.js`），就直接执行该命令并 `return`。这里有个值得玩味的细节：`grepCommand` 对内置命令用 `^(${cmd})` 做前缀匹配，而对用户预置命令用 `(?:\\b\\s|^)(${preset.command})(?:\\b\\s|$)` 做词边界替换，允许一句话里混入多个预置命令并各自替换为 prompt。

第二道是 Agent 闸门。`const isAgentChat = await grepAgents({...})`——如果消息里 `@agent` 触发了智能体流程，函数直接 `if (isAgentChat) return`，把控制权完全交给 Agent 子系统（第 9、10 章详述）。这意味着 RAG 检索与 Agent 是互斥的两条路。

第三道是模型路由。`resolveLLMConnector` 包了一层 `resolveProviderConnector`，返回 `{ connector: LLMConnector, routingMetadata, prefetchedContext, error }`。这里出现了本章一个反复登场的概念——`prefetchedContext`：模型路由在判断"该用哪个模型"时，往往已经顺手把聊天历史、pinned 文档、甚至 system prompt 取好了，于是后续流程能复用这份预取结果，避免重复查库。若路由失败,函数写出一个 `type: "abort"` 的块并结束；若路由命中了某条规则且 `routedTo.shouldNotify` 为真，还会先写一个 `type: "modelRouteNotification"` 的块告诉前端"这条消息被路由到了某模型"。

## 模式判定：chat 与 query 的根本分野

走过三道闸门，函数才真正进入 RAG 领地。`VALID_CHAT_MODE = ["automatic", "chat", "query"]` 这个常量定义在 `stream.js` 顶部，是理解整章的钥匙。

`chat` 与 `query` 是两种世界观。`chat` 模式下，文档检索只是"补充材料"——即便一条命中都没有，LLM 仍可以调用自己的通识来回答。`query` 模式则是"闭卷考试"：LLM 只能依据工作区文档作答，一旦没有任何可用上下文，就必须拒答，绝不允许它用预训练知识糊弄过去。这个分野在源码里以两处早退实现。

第一处出现得极早，在任何检索之前。函数先探明工作区的向量空间状态：

```js
const VectorDb = getVectorDbClass();
const messageLimit = workspace?.openAiHistory || 20;
const hasVectorizedSpace = await VectorDb.hasNamespace(workspace.slug);
const embeddingsCount = await VectorDb.namespaceCount(workspace.slug);

if ((!hasVectorizedSpace || embeddingsCount === 0) && chatMode === "query") {
  const textResponse =
    workspace?.queryRefusalResponse ??
    "There is no relevant information in this workspace to answer your query.";
  // ... 写出 textResponse 块，并以 include: false 落库后 return
}
```

逻辑很直白：如果是 `query` 模式，而工作区压根没有命名空间或向量数为零，那就没有"卷子"可考，直接抛出拒答文案并 `return`。注意拒答文案的取值优先级——先用工作区自定义的 `workspace?.queryRefusalResponse`（Prisma schema 里该字段是 `String?`，默认 `null`），缺省时才回落到那句硬编码英文。还要注意落库时 `include: false`：这条拒答不会被算进未来的聊天历史，免得污染后续上下文。这种"早退落库且不纳入历史"的模式，在本章会反复出现。

源码在这里留了一段坦诚的注释，点明走到下一步意味着什么："1. Chatting in 'chat' mode and may or may _not_ have embeddings；2. Chatting in 'query' mode and has at least 1 embedding。"——即 `chat` 模式无论有无向量都继续，`query` 模式则保证至少有一条向量才往下走。

`messageLimit` 也在这里确定：取 `workspace?.openAiHistory`，缺省 20。这个值同时是 schema 里 `openAiHistory Int @default(20)` 的默认，决定了后面取多少轮历史。

## 检索之一：pinned 文档全文 prepend，且不占检索额度

接下来函数声明了一组贯穿全程的累加器：`contextTexts`（喂给 LLM 的上下文文本）、`sources`（回传给前端做引用的来源）、`pinnedDocIdentifiers`（pinned 文档的去重标识）。理解这三者的关系，是理解 citations 行为的前提。

第一批进入上下文的不是向量检索结果，而是 **pinned 文档**。pinned（钉选）是 AnythingLLM 的一个重要设计：用户可以把某篇文档"钉"在工作区，让它的**全文**无条件、完整地进入每一次对话的上下文，而不必经过相似度筛选。这对小而关键的文档（一份合同、一张需要逐行核对的 CSV）极其有用。

```js
const { rawHistory, chatHistory, pinnedDocs: prefetchedPinnedDocs, parsedFiles: prefetchedParsedFiles } =
  prefetchedContext ?? (await recentChatHistory({ user, workspace, thread, messageLimit }));

const pinnedDocs =
  prefetchedPinnedDocs ??
  (await new DocumentManager({
    workspace,
    maxTokens: LLMConnector.promptWindowLimit(),
  }).pinnedDocs());

pinnedDocs.forEach((doc) => {
  const { pageContent, ...metadata } = doc;
  pinnedDocIdentifiers.push(sourceIdentifier(doc));
  contextTexts.push(doc.pageContent);
  sources.push({
    text: pageContent.slice(0, 1_000) + "...continued on in source document...",
    ...metadata,
  });
});
```

这段代码藏着三个设计意图。其一，pinned 文档的 `DocumentManager` 是用 `maxTokens: LLMConnector.promptWindowLimit()` 构造的——`server/utils/DocumentManager/index.js` 的 `pinnedDocs()` 会累加每篇文档的 `token_count_estimate`，一旦 `tokens >= this.maxTokens` 就跳过后续文档并打日志 `Token limit of ... has already been exceeded by pinned documents`。也就是说，pinned 全文虽然不参与相似度筛选，但仍受模型上下文窗口这个硬上限约束，不会无限膨胀。其二，进 `contextTexts` 的是 `doc.pageContent`（全文），但进 `sources` 的 `text` 字段只截了前 1000 字符再缀上 `"...continued on in source document..."`——前端引用卡片不需要塞下整篇文档，只要一段预览即可。其三，每篇 pinned 文档都通过 `sourceIdentifier(doc)` 算出一个标识塞进 `pinnedDocIdentifiers`，这个数组马上会被传给向量检索作为"黑名单"。

`sourceIdentifier` 定义在 `index.js`，逻辑是 `title:${title}-timestamp:${published}`，标题或时间戳缺失时退化为随机 uuid。它的注释把去重意图说得很清楚：如果你 pin 了一个 CSV，向量检索很可能又从同一文档里捞出若干块，结果同一份数据在 pinned 全文和 RAG 结果里各出现一次，"result in bad results even if the LLM was not even going to hallucinate"。所以 pinned 标识要作为过滤器传下去。

紧随 pinned 之后，函数还会取一批 **parsedFiles**（`WorkspaceParsedFiles.getContextFiles`，临时上传但未向量化的随聊文件），处理方式与 pinned 完全一致——全文进 `contextTexts`，截断预览进 `sources`。它们同样不占向量检索的名额。

## 检索之二：向量相似度搜索的参数与阈值

铺完 pinned 与 parsedFiles，才轮到真正的向量检索：

```js
const vectorSearchResults =
  embeddingsCount !== 0
    ? await VectorDb.performSimilaritySearch({
        namespace: workspace.slug,
        input: updatedMessage,
        LLMConnector,
        similarityThreshold: workspace?.similarityThreshold,
        topN: workspace?.topN,
        filterIdentifiers: pinnedDocIdentifiers,
        rerank: workspace?.vectorSearchMode === "rerank",
      })
    : { contextTexts: [], sources: [], message: null };
```

注意 `embeddingsCount !== 0` 这个守卫：`chat` 模式下完全可能在零向量的工作区聊天，这时检索整段被短路成空结果，避免了对空命名空间的无谓查询。

几个参数值得逐一对照默认值（均来自 `server/prisma/schema.prisma` 的 workspaces 模型，并在向量库实现里有同名兜底）：

- `similarityThreshold` 工作区字段默认 `0.25`（`@default(0.25)`）。在 `server/utils/vectorDbProviders/lance/index.js` 的 `similarityResponse` 与 `rerankedSimilarityResponse` 里，函数签名同样写着 `similarityThreshold = 0.25`，并用 `if (this.distanceToSimilarity(item._distance) < similarityThreshold) return` 把相似度不达标的块直接丢弃。换言之，余弦距离要先转成相似度，再与阈值比较，低于阈值的命中根本不会进上下文。
- `topN` 工作区默认 `4`（`@default(4)`），向量库实现里也是 `topN = 4`。它直接作为 LanceDB `vectorSearch(queryVector).distanceType("cosine").limit(topN)` 的 limit，决定最终取回几块。
- `filterIdentifiers: pinnedDocIdentifiers`——把上一步算好的 pinned 标识传进来，检索时若某命中块的 `sourceIdentifier` 落在此列表里就跳过，并打日志 `A source was filtered from context as it's parent document is pinned`。这正是 pinned 去重落地的地方。
- `rerank: workspace?.vectorSearchMode === "rerank"`——`vectorSearchMode` schema 默认 `"default"`，只有显式设为 `"rerank"` 才会走重排路径。重排路径有意思：它先把检索 limit 扩大到 `Math.max(10, Math.min(50, Math.ceil(totalEmbeddings * 0.1)))`（即总量的 10%，下限 10、上限 50），再用 `NativeEmbeddingReranker` 对这批候选重排，取 `topK: topN`。注释解释了这个区间的权衡："reranking is expensive and time consuming"，所以既要给重排器足够候选，又不能让它在万级向量上跑到天荒地老。

`performSimilaritySearch` 返回结构是 `{ contextTexts, sources, message }`。这里的 `message` 字段不是聊天内容，而是错误信号：`if (!!vectorSearchResults.message)` 为真时，函数写出一个 `type: "abort"` 块并 `return`——这是检索本身失败（如向量库连不上）的兜底。注意当命名空间不存在时，`performSimilaritySearch` 会返回 `message: "Invalid query - no documents found for workspace!"`，正是借这个字段把"查不到"与"查出错"统一成 abort。

检索结果的 `sources` 在落地前还经过 `curateSources`：它剥掉 `vector`、`_distance` 等内部字段，只保留 metadata 与 text。这一步保证回传前端的来源对象干净、不泄露原始向量。

## 检索之三：fillSourceWindow——给追问续上引用

向量检索之后，AnythingLLM 多了一步在很多 RAG 系统里看不到的处理——`fillSourceWindow`，定义在 `server/utils/helpers/chat/index.js`。

```js
const { fillSourceWindow } = require("../helpers/chat");
const filledSources = fillSourceWindow({
  nDocs: workspace?.topN || 4,
  searchResults: vectorSearchResults.sources,
  history: rawHistory,
  filterIdentifiers: pinnedDocIdentifiers,
});
```

它要解决的痛点，函数注释举了个再具体不过的例子：第一句"What is anythingllm?"可能命中 4 个好来源；第二句追问"Tell me some features"，由于措辞泛化，向量检索可能只命中 0~1 个相关块——在 `query` 模式下这会直接触发拒答，体验很糟。`fillSourceWindow` 的做法是：如果当前命中数不足 `nDocs`（默认仍是 topN 即 4），就**从最近的聊天历史里回填**之前用过的来源，把窗口补到目标数量。

回填的去重相当严谨。它先用 `seenChunks = new Set(searchResults.map((source) => source.id))` 记下已有块；然后倒序遍历历史（最新优先），从每条历史响应里解析出 `sources`，只接纳满足全部条件的来源：不在当前 pinned 列表里（`filterIdentifiers.includes(sourceIdentifier(source)) == false`）、带有 `score` 属性（排除"它本身就来自某个 pinned 文档"的情况）、带有 `text` 属性、且 `id` 没见过（`seenChunks.has(source.id) == false`）。一旦凑满 `nDocs` 就停。注释还特意安抚了一句：别担心这个循环在长历史里失控，因为 `history` 来自 `recentChatHistory`，本身就受 messageLimit（默认 20）约束。

回填完成后，是本章最精妙的一步——**`contextTexts` 与 `sources` 的非对称合并**：

```js
contextTexts = [...contextTexts, ...filledSources.contextTexts];
sources = [...sources, ...vectorSearchResults.sources];
```

看仔细：`contextTexts` 拼的是 `filledSources.contextTexts`（含回填的历史来源），而 `sources` 拼的却是 `vectorSearchResults.sources`（仅本轮检索）。源码用一大段注释解释了这个不对称的良苦用心：让 LLM "comprehend" 一个连贯回答需要回填的历史上下文，但前端 Citations 里**不该**出现那些回填来源——因为用户会觉得"这文档跟我这句问题没关系，为什么引它"。而历史来源既然在过去被引用过，它出现在历史里就合情合理。注释的 TLDR 一针见血："reduces GitHub issues for 'LLM citing document that has no answer in it' while keep answers highly accurate."——既保住答案质量，又不让引用看起来牛头不对马嘴。

这就是 AnythingLLM 对 citations 的核心立场：**喂给模型的上下文可以多，但展示给用户的引用要诚实**。

## query 模式的最终拒答兜底

合并完上下文，函数迎来 `query` 模式的第二道、也是最后一道拒答闸门：

```js
if (chatMode === "query" && contextTexts.length === 0) {
  const textResponse =
    workspace?.queryRefusalResponse ??
    "There is no relevant information in this workspace to answer your query.";
  // 写出 textResponse 块，并以 include: false 落库后 return
}
```

它和开头那道闸门遥相呼应，但触发条件不同：开头判断的是"工作区有没有向量"，这里判断的是"经过 pinned + 向量检索 + fillSourceWindow 三路汇集后，`contextTexts` 是不是真的一块都没有"。只有在 `query` 模式且彻底空手时才拒答——`chat` 模式即便上下文为空也照常往下走，让 LLM 用通识回答。落库依旧 `include: false`，拒答不进历史。

把两道闸门连起来看，`query` 模式的"闭卷"承诺就完整了：没向量→拒答；有向量但本轮+回填+pinned 全部落空→还是拒答；任何一路供出了哪怕一块上下文→才允许 LLM 开口。

## 上下文组装：system prompt、记忆与 cannonball 压缩

确定有内容可答之后，函数组装最终要发给 LLM 的消息数组。第一步是 system prompt：

```js
const systemPrompt =
  prefetchedContext?.systemPrompt ??
  (await chatPrompt(workspace, user, { prompt: updatedMessage, rawHistory }));
```

`chatPrompt`（`index.js`）取 `workspace?.openAiPrompt`，缺省回落到 `SystemSettings.saneDefaultSystemPrompt`——那句定义在 `server/models/systemSettings.js` 的经典提示："Given the following conversation, relevant context, and a follow up question, reply with an answer to the current question the user is asking..."。之后 `chatPrompt` 还会做两件事：用 `SystemPromptVariables.expandSystemPromptVariables` 展开提示词里的变量占位符，再用 `promptWithMemories` 把工作区记忆（启用时）追加进去——记忆的重排会用到当前 `prompt` 与 `rawHistory`，这也是为什么这两个参数要一路传进来。

第二步是真正的拼装与压缩：

```js
const messages = await LLMConnector.compressMessages(
  { systemPrompt, userPrompt: updatedMessage, contextTexts, chatHistory, attachments },
  rawHistory
);
```

`compressMessages` 背后是 `server/utils/helpers/chat/index.js` 的 `messageArrayCompressor`，AnythingLLM 给它起了个生动的名字——**cannonball（炮弹）压缩**。它的哲学写在长注释里：当 prompt 超出模型窗口时，不去调用更贵的摘要模型（那既增加延迟又依赖特定供应商），而是"在挡路的输入正中央轰开一个洞"——把超长文本按 token 从中点向两侧对称删除，中间填上 `--prompt truncated for brevity--`。这就是 `cannonball` 函数干的事。

压缩有一套明确的 token 预算（注释里的 rate-limit 比例）：system 至多约 15%、history 至多约 15%、prompt 至多约 70%。开场先有个快速通道——若 `tokenManager.statsFrom(messages) + tokenBuffer(600) < llm.promptWindowLimit()`，说明根本没超窗，直接原样返回，绝大多数用户走的就是这条路。一旦超窗，处理优先级是：

- **超大用户 prompt**（`userPromptSize > llm.limits.user`）被视为"标准独白"，直接 cannonball 到窗口的 0.8 倍并丢弃 system、history、context——因为一个占了 70% 以上窗口的 prompt 几乎注定是自成一体的。
- **system 超限**则把它按 `Context:` 切成指令与上下文两段，分别向 `llm.limits.system * 0.25`（用户指令）和 `* 0.75`（上下文）压缩，刻意偏向保留上下文而非用户那段指令。
- **history 永远被最激进地压缩**，因为它"是最不重要、最灵活的"。遍历时若加入某条历史就超 `llm.limits.history` 就停；但对最近 3 对消息（`i > 2` 才 break）会尝试 cannonball（单条上限 `Math.floor(llm.limits.history / 2.2)`）以尽量保住最近的对话。

至此 `messages` 就是一个保证能塞进窗口、且给回复留足空间的数组。值得一提，对只接受字符串补全的模型，还有个并行实现 `messageStringCompressor` 走 `llm.constructPrompt` 路径，逻辑同构。

## 调 LLM：流式与非流式两条出口

组装完毕，函数按连接器能力分流：

```js
if (LLMConnector.streamingEnabled() !== true) {
  // 非流式：一次 getChatCompletion，拿到 textResponse 后用单个 textResponseChunk 块 close
} else {
  const stream = await LLMConnector.streamGetChatCompletion(messages, {
    temperature: workspace?.openAiTemp ?? LLMConnector.defaultTemp,
    user: user,
  });
  completeText = await LLMConnector.handleStream(response, stream, { uuid, sources });
  metrics = stream.metrics;
}
```

两条出口都把 `temperature` 取为 `workspace?.openAiTemp ?? LLMConnector.defaultTemp`——工作区温度优先，缺省用连接器默认。流式出口是主路：`handleStream` 接管 `response`，把模型吐出的 token 逐块写成 `textResponseChunk` 推给前端，同时携带 `sources`，并在内部累积出 `completeText` 返回。注意这里传给 `handleStream` 的 `sources` 已经是"pinned + parsedFiles + 本轮向量来源"的合集（但不含 fillSourceWindow 回填项，呼应前文那处非对称合并）。非流式出口则用一个 `close: true` 的 `textResponseChunk` 一次性吐完。两条出口殊途同归，都为下一步落库准备好了 `completeText`、`sources` 与 `metrics`。

## 落库与收尾：citations 的最终回传

最后是持久化与收尾信号：

```js
if (completeText?.length > 0) {
  const { chat } = await WorkspaceChats.new({
    workspaceId: workspace.id,
    prompt: message,
    response: { text: completeText, sources, type: chatMode, attachments, metrics },
    threadId: thread?.id || null,
    user,
  });
  writeResponseChunk(response, {
    uuid, type: "finalizeResponseStream", close: true, error: false,
    chatId: chat.id, metrics,
  });
  return;
}
```

只有当 `completeText?.length > 0` 才落库——空回答不存。落库时存的 `prompt` 是用户**原始** `message`（不是展开命令后的 `updatedMessage`），而 `response.sources` 存的正是那份"诚实"的来源集合。这也是 `fillSourceWindow` 能在未来追问里回填的数据源头：这一轮的 sources 落进历史，下一轮 `fillSourceWindow` 倒序遍历历史时就能把它们捞回来。最后写一个 `type: "finalizeResponseStream"` 块，带上 `chatId` 与 `metrics`，前端据此结束这条消息的流式渲染。

至于"同一文档的多个块如何合并展示"，那是前端的活。`frontend/src/components/.../Citation/index.jsx` 的 `combineLikeSources` 按 `title` 把同源的多个 chunk 归并到一个引用条目下，每条记录 `chunks` 列表与 `references` 计数，最终 `Object.values(combined)` 输出；`Citations` 组件只显示前 3 个（`combined.slice(0, 3)`），其余折叠成 `+N`。所以服务端回传的 sources 允许同文档重复，去重与折叠交给前端按标题聚合——服务端只负责诚实，前端负责整洁。

## 两个变体：非流式 API 与嵌入式聊天

主流程之外，还有两个同构的变体值得一瞥，它们印证了这套设计的可复用性。

嵌入式聊天 `server/utils/chats/embed.js` 的 `streamChatWithForEmbed` 几乎是 `streamChatWithWorkspace` 的精简翻版：同样取 pinned、同样 `performSimilaritySearch`、同样 `fillSourceWindow`、同样 cannonball 压缩。差异主要在边界：嵌入场景里 `automatic` 模式无效（`if (chatMode === "automatic") chatMode = "chat"`），历史从 `EmbedChats` 而非 `WorkspaceChats` 取，且允许通过 `embed.allow_prompt_override` / `allow_temperature_override` / `allow_model_override` 在请求级覆盖工作区配置。还有一处微妙差别：嵌入版调 `handleStream` 时传的是 `sources: []`，落库时才把完整 sources 存进 `EmbedChats`——前端嵌入 widget 不在流里展示引用，但记录仍然完整。

这两个变体的存在说明：AnythingLLM 把"取上下文—组装—压缩—调模型—回填引用"抽象成了一套稳定的零件（`recentChatHistory`、`fillSourceWindow`、`compressMessages`、`sourceIdentifier`、`writeResponseChunk` 等都在多处复用），不同入口只是用不同方式拼装这些零件。这种"主流程即配置组合"的风格，正是后续章节理解 LLM 接入层与 Agent 系统的基础。

## 本章小结

1. RAG 聊天主流程集中在 `server/utils/chats/stream.js` 的 `streamChatWithWorkspace`，它是流式函数，靠 `writeResponseChunk(response, ...)` 边算边推 SSE 块，全程用一个 `uuid` 串联。
2. 进入 RAG 前有三道早退闸门：内置斜杠命令（`VALID_COMMANDS`，目前仅 `/reset`）、`@agent` 触发的 Agent 流程、以及模型路由失败；模型路由还会顺手产出可复用的 `prefetchedContext`。
3. `chatMode` 有 `automatic`/`chat`/`query` 三种（`VALID_CHAT_MODE`）；`query` 是闭卷模式，无可用上下文必须拒答，拒答文案优先取 `workspace.queryRefusalResponse`，落库时 `include: false` 不进历史。
4. 上下文按优先级分三路汇入 `contextTexts`：pinned 文档全文（不占检索额度，但受 `LLMConnector.promptWindowLimit()` 的 token 上限约束）、随聊 parsedFiles 全文、以及向量检索结果。
5. 向量检索由 `performSimilaritySearch` 执行，关键默认值来自 workspaces 模型：`similarityThreshold` 默认 `0.25`、`topN` 默认 `4`、`vectorSearchMode` 默认 `"default"`（设为 `"rerank"` 才走扩召回+重排路径）。
6. `pinnedDocIdentifiers`（由 `sourceIdentifier` 算出，形如 `title:..-timestamp:..`）作为 `filterIdentifiers` 传入检索，避免 pinned 全文与 RAG 命中重复同一文档。
7. `fillSourceWindow` 在命中不足 `topN` 时从最近历史回填来源，让 `query` 模式的追问不至于因 0 命中而拒答；回填有严格去重（`seenChunks`、需含 `score`/`text`、排除 pinned）。
8. citations 采用非对称合并：`contextTexts` 含回填来源（喂模型求连贯），`sources` 只含本轮向量来源（给用户保诚实），这是为减少"引用了无关文档"类反馈而刻意为之。
9. 上下文用 `compressMessages`/`messageArrayCompressor` 做 cannonball 压缩——超窗时从文本中点对称删除，token 预算约为 system 15%、history 15%、prompt 70%，history 被最激进压缩但保最近 3 对。
10. 调用 LLM 分流式（`handleStream` 携 sources 逐块推送）与非流式（单块 close）两条出口，温度取 `workspace.openAiTemp ?? LLMConnector.defaultTemp`；仅当 `completeText` 非空才落库，并以 `finalizeResponseStream` 块带 `chatId` 收尾。
11. 嵌入式 `streamChatWithForEmbed` 与主流程同构，差异在历史来源（`EmbedChats`）、请求级覆盖开关、`automatic` 失效、以及流里不展示引用；前端 `combineLikeSources` 按 `title` 聚合同源 chunk 并只显示前 3 条。

## 动手实验

1. **实验一：观察 chat 与 query 的拒答分野** — 建一个不含任何文档的工作区，分别用 `chat` 和 `query` 模式提同一个问题。在 `streamChatWithWorkspace` 开头那处 `if ((!hasVectorizedSpace || embeddingsCount === 0) && chatMode === "query")` 前加日志，确认 `query` 走拒答早退、`chat` 继续往下；再给工作区设一个自定义 `queryRefusalResponse`，验证拒答文案随之改变。
2. **实验二：验证 pinned 去重** — pin 一篇文档，然后就该文档内容提问。在 `lance/index.js` 的 `similarityResponse` 里 `filterIdentifiers.includes(sourceIdentifier(rest))` 处加日志，观察"A source was filtered ... parent document is pinned"是否打印；再对比 `contextTexts` 里 pinned 全文与被过滤掉的 RAG 命中，理解 `pinnedDocIdentifiers` 的作用链路。
3. **实验三：触发 fillSourceWindow 回填** — 先问一个能命中 4 块的问题，再追问一个措辞泛化、向量几乎不命中的后续问题。在 `helpers/chat/index.js` 的 `fillSourceWindow` 内 `log("Need to backfill ...")` 与 `log("Citations backfilled ...")` 处确认回填发生，并对比此时 `contextTexts` 与 `sources` 的长度差，亲眼看到非对称合并。
4. **实验四：把炮弹打穿窗口** — 构造一段超长用户 prompt（超过 `llm.limits.user`），观察 `messageArrayCompressor` 是否走"超大 prompt 独白"分支、`cannonball` 是否打印 `Cannonball results X -> Y tokens`；再换成超长 system prompt（含 `Context:` 分隔），验证它如何按 0.25/0.75 比例分段压缩。

> **下一章预告**：本章我们假定 `LLMConnector` 已经就位——它会算 `promptWindowLimit()`、暴露 `limits.user/system/history`、提供 `streamGetChatCompletion` 与 `handleStream`、还能 `embedTextInput`。但这些能力在 OpenAI、Anthropic、本地 Ollama、Azure 等几十种供应商之间是如何被抹平成同一套接口的？第 8 章将深入 **LLM 模型接入层**，解构 `resolveProviderConnector` 与各 provider 的统一抽象，看 AnythingLLM 如何用一层薄薄的连接器，把异构的模型世界收束成主流程眼中"一个 LLMConnector"。
