# 第 6 章 文档管理与工作区模型

在前五章里我们一直在向量、嵌入、切分这些"原料加工"的层面打转——文档怎么被采集、怎么被切成块、怎么被嵌入成向量、又怎么被塞进向量库。但一个真正可用的 RAG 系统还需要回答一个更上层的问题:这些文档究竟"属于谁"?谁能在对话里看到它们?哪些文档是用户特别想让模型每次都读到的,哪些只是备查的资料库?这一章我们就来解构 AnythingLLM 回答这些问题的核心抽象——工作区(Workspace)。

工作区是 AnythingLLM 里最重要的组织单元。它既是文档与向量的隔离边界(每个工作区对应向量库里一个独立的 namespace),又是一份完整的对话配置载体(温度、相似度阈值、topN、聊天模式、系统提示词等全挂在它身上)。理解了工作区,再回头看文档归属、pin 置顶、文档同步,就会发现它们都是围绕这个中心结构旋转的卫星机制。本章将沿着"工作区是什么 → 文档如何进入工作区 → 对话时怎么取文档 → pin 机制如何区别对待置顶文档 → 网页等外部源如何定时重抓"这条线索逐层展开,所有论断均来自 `server/models/` 与 `server/utils/` 下的真实源码。

## 工作区:隔离单元与配置载体

工作区的数据模型定义在 `server/models/workspace.js`。打开这个文件你会先看到一个 `Workspace` 对象,它顶部声明了两个意味深长的常量:`VALID_CHAT_MODES: ["chat", "query", "automatic"]` 和 `defaultPrompt: SystemSettings.saneDefaultSystemPrompt`。前者枚举了一个工作区可以处于的三种聊天模式,后者把系统级的"安全默认系统提示词"挂为工作区的兜底提示词。换句话说,工作区一出生就携带了一整套行为参数。

这套参数集中体现在 `writable` 数组里,它列出了可以通过通用更新接口修改的字段:`name`、`openAiTemp`、`openAiHistory`、`openAiPrompt`、`similarityThreshold`、`chatProvider`、`chatModel`、`topN`、`chatMode`、`agentProvider`、`agentModel`、`queryRefusalResponse`、`vectorSearchMode`、`router_id`。注意 `slug`、`vectorTag`、`pfpFilename` 被特意用注释 `//` 排除在外——它们存在于数据库记录里,但不允许通过常规更新改写,因为 `slug` 是工作区的稳定身份标识(后面会看到它就是向量库的 namespace),随意修改会让向量数据"找不到家"。

每个字段都配了一个验证器,集中在 `validations` 对象里,这些验证器同时也定义了字段的默认值,非常值得逐一记住,因为它们直接决定了 RAG 的检索行为:

```js
similarityThreshold: (value) => {
  if (value === null || value === undefined) return 0.25;
  const threshold = parseFloat(value);
  if (isNullOrNaN(threshold)) return 0.25;
  if (threshold < 0) return 0.0;
  if (threshold > 1) return 1.0;
  return threshold;
},
topN: (value) => {
  if (value === null || value === undefined) return 4;
  const n = parseInt(value);
  if (isNullOrNaN(n)) return 4;
  if (n < 1) return 1;
  return n;
},
```

从这里可以读出几个关键默认值:相似度阈值 `similarityThreshold` 默认 `0.25`,并被钳制在 `[0.0, 1.0]` 区间;检索条数 `topN` 默认 `4`,且不允许小于 `1`;对话历史轮数 `openAiHistory` 默认 `20`(见 `openAiHistory` 验证器返回的 `return 20`);温度 `openAiTemp` 默认 `null`(交由下游 LLM 提供方决定),且负数会被归零为 `null`。`chatMode` 的验证器规定,任何不在 `VALID_CHAT_MODES` 里的值都会回落到 `"automatic"`;而在 `new` 方法创建工作区时,代码里又硬编码了一次 `chatMode: "automatic"`,可见 automatic 是 AnythingLLM 期望的默认对话模式。`vectorSearchMode` 只接受 `"default"` 或 `"rerank"`,否则回落 `"default"`,这呼应了第 5 章里向量库是否启用重排(rerank)的开关。`queryRefusalResponse` 则是 query 模式下检索不到相关内容时返回给用户的"拒答话术",默认 `null`。

工作区创建的 `new` 方法还隐藏了一个细节:`slug` 由名称经 `slugify` 生成,而 `slugify` 方法在 `server/models/workspace.js` 里被特意扩展了一组字符映射(`"+": " plus "`、`"@": " at "`、`"."`→`" dot "` 等),并把 `:`、`~`、`(`、`)`、引号、`|` 直接抹掉。注释解释了原因:某些向量库提供方对 namespace 字符有限制,与其为每个提供方写一套归一化逻辑,不如在工作区命名这一层就把麻烦字符规范掉。如果生成的 slug 与已有工作区冲突,代码会拼上一个 8 位随机种子 `slugSeed` 重新生成,确保唯一。

工作区读取时还会即时计算两个派生字段。在 `get` 与 `getWithUser` 里,返回对象都附带了 `contextWindow`(由 `_getContextWindow` 根据工作区或环境变量的 provider/model 反查 `promptWindowLimit` 得到)和 `currentContextTokenCount`(由 `_getCurrentContextTokenCount` 统计该工作区已解析文件的 token 总量)。这说明工作区不仅是配置容器,还承担了"这个上下文窗口还剩多少额度"的运行时账本角色。

## 文档如何进入工作区:addDocuments 与向量库的联动

文档归属的真相在 `server/models/documents.js`。这里的 `Document` 对象操作的是 Prisma 的 `workspace_documents` 表——表名本身就说明了文档与工作区是多对一(更准确说是"文档引用"与工作区一一对应)的关系:同一份物理文件被加到三个工作区,就会有三条 `workspace_documents` 记录,各自带独立的 `docId`,但 `filename` 相同。这个设计后面在文档同步的"扩散"逻辑里至关重要。

核心入口是 `addDocuments(workspace, additions, userId)`。它的流程可以拆成几步。首先它通过 `getVectorDbClass()` 拿到当前配置的向量库实现(第 5 章的抽象层),然后对每个待加入的路径 `path` 调用 `fileData(path)` 从磁盘读出已经过 collector 处理的文档 JSON。读出来的对象里,`pageContent` 被解构剥离,剩下的字段作为 `metadata`,然后组装出一条文档记录:

```js
const docId = uuidv4();
const { pageContent: _pageContent, ...metadata } = data;
const newDoc = {
  docId,
  filename: path.split("/")[1],
  docpath: path,
  workspaceId: workspace.id,
  metadata: JSON.stringify(metadata),
};
```

注意这里**先嵌入,后落库**的顺序。代码紧接着调用 `VectorDb.addDocumentToNamespace(workspace.slug, { ...data, docId }, path)`,把文档连同新生成的 `docId` 推进以 `workspace.slug` 为名的向量 namespace。只有当返回 `vectorized` 为真,才会执行 `prisma.workspace_documents.create({ data: newDoc })` 写入数据库记录。如果向量化失败,文件名会被推入 `failedToEmbed`,错误进 `errors` 集合,并直接 `continue` 跳过落库。这个顺序保证了一条不变量:**数据库里存在的文档记录,其向量必定已经进了向量库**,不会出现"有记录无向量"的孤儿。

整个过程还穿插了大量进度上报。`emitProgress(workspace.slug, {...})` 在批次开始(`batch_starting`)、单文档开始(`doc_starting`)、单文档完成(`doc_complete`)、单文档失败(`doc_failed`)、全部完成(`all_complete`)等节点把状态推给前端,这就是 AnythingLLM 上传文档时那个实时进度条的数据源。代码还借助全局变量 `global.__embeddingProgress` 暴露当前正在嵌入的文件,供其他模块查询,并在结束时置 `null`。最后 `addDocuments` 会发一条 `documents_embedded_in_workspace` 遥测和一条 `workspace_documents_added` 事件日志。

删除走的是对称的 `removeDocuments(workspace, removals, userId)`。它对每个待删路径先 `get` 出文档记录,然后调 `VectorDb.deleteDocumentFromNamespace(workspace.slug, document.docId)` 从向量库删,再删 `prisma.workspace_documents` 记录,最后通过 `prisma.document_vectors.deleteMany({ where: { docId } })` 清掉文档与向量 ID 的映射。这里出现的 `document_vectors` 表正是第 5 章提过的文档↔向量映射(`server/models/vectors.js` 的 `DocumentVectors`),它在 `bulkInsert` 时把每个 `{ docId, vectorId }` 落库,使得"删除一份文档时精确删除它名下的全部向量 ID"成为可能,而不必扫描整个 namespace。本章不再展开映射细节,只需记住:`documents.js` 管"文档归属哪个工作区",`vectors.js` 管"文档拆出了哪些向量 ID",两者通过 `docId` 串联。

`Document` 对象的 `writable` 字段也很说明问题:只有 `["pinned", "watched", "lastUpdatedAt"]` 三个字段允许通过 `update` 修改。前两个正是本章后两节的主角——`pinned` 控制置顶,`watched` 控制是否纳入定时同步。`update` 方法同样会先用 `writable.includes(key)` 过滤掉非法字段才写库,这是 AnythingLLM 各模型一致的防御式风格。

## DocumentManager:对话时如何取出工作区文档

文档进了工作区、向量进了 namespace,但对话发生时模型并不会"读数据库"。负责在对话时把文档内容捞出来注入上下文的,是 `server/utils/DocumentManager/index.js` 里那个出奇精简的 `DocumentManager` 类——整个文件只有七十来行。

它的构造函数只接两个参数:`workspace` 和 `maxTokens`,后者默认 `Number.POSITIVE_INFINITY`(无上限)。它还在构造时确定了文档存储路径 `documentStoragePath`:开发环境是 `../../storage/documents`,生产环境是 `process.env.STORAGE_DIR` 下的 `documents` 目录。这个路径就是 collector 处理完文档后落地的 JSON 文件所在地。

`DocumentManager` 只暴露两个有意义的方法,而且都只服务于"置顶文档"这一件事。第一个是 `pinnedDocuments()`,它做一次数据库查询,把当前工作区里所有 `pinned: true` 的 `workspace_documents` 记录查出来:

```js
async pinnedDocuments() {
  if (!this.workspace) return [];
  const { Document } = require("../../models/documents");
  return await Document.where({
    workspaceId: Number(this.workspace.id),
    pinned: true,
  });
}
```

第二个是真正干活的 `pinnedDocs()`。它先拿到每条置顶记录的 `docpath`,逐个从磁盘 `fs.readFileSync` 读出 JSON 并 `JSON.parse`。这里有两道防御性校验:如果某个文件缺少 `pageContent` 或 `token_count_estimate` 字段,就打印日志 `Skipping document - Could not find page content or token_count_estimate in pinned source.` 并跳过;如果累计 token 数 `tokens` 已经达到或超过 `this.maxTokens`,也跳过并提示"置顶文档已超出 token 上限"。每成功读入一个文档,就把它整个 push 进 `pinnedDocs` 数组,并把 `data.token_count_estimate` 累加到 `tokens`:

```js
if (tokens >= this.maxTokens) {
  this.log(
    `Skipping document - Token limit of ${this.maxTokens} has already been exceeded by pinned documents.`
  );
  continue;
}
pinnedDocs.push(data);
tokens += data.token_count_estimate || 0;
```

这段 token 预算逻辑解释了构造时 `maxTokens` 参数的意义。回到调用方 `server/utils/chats/stream.js`,你会看到 `DocumentManager` 是这样被实例化的:`new DocumentManager({ workspace, maxTokens: LLMConnector.promptWindowLimit() })`。也就是说,置顶文档的 token 预算被设成了当前 LLM 的上下文窗口大小——这是一种朴素但有效的护栏:置顶文档可以一直往里塞,直到接近撑爆模型上下文窗口为止,后续的置顶文档会被静默丢弃而不是引发 API 报错。值得一提的是,`DocumentManager` 本身不感知"普通文档",它只关心 pinned 的那一批;普通文档的检索完全由向量库负责,二者职责清晰分离。

## pin 机制:置顶文档全文注入 vs 普通文档向量检索

理解 pin 的关键,是看清同一次对话里"置顶文档"和"普通文档"走的是两条完全不同的通路。这一点在 `server/utils/chats/stream.js`(以及结构几乎一致的 `embed.js`、`openaiCompatible.js`、`apiChatHandler.js`、`router/index.js`)的上下文组装段落里展现得淋漓尽致。

第一条通路是置顶文档,走"全文注入"。代码先取出 `pinnedDocs`(可能复用路由层预取的,否则现取),然后遍历它们:

```js
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

请注意 `contextTexts.push(doc.pageContent)` 这一行——推进上下文的是文档的**完整 `pageContent`**,而不是某个片段。也就是说,被置顶的文档会把全文原样塞进给 LLM 的上下文。与此同时,它的来源标识被推入 `pinnedDocIdentifiers` 数组,而呈现给用户的 `sources` 里只保留前 1000 个字符再缀上 `...continued on in source document...`,避免引用卡片过长。

第二条通路是普通文档,走"向量检索"。紧接着的 `VectorDb.performSimilaritySearch(...)` 才是常规 RAG 的语义检索,它使用工作区的 `similarityThreshold`、`topN`、以及 `vectorSearchMode === "rerank"` 决定是否重排。而最画龙点睛的一行是它的 `filterIdentifiers: pinnedDocIdentifiers` 参数:

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

这就是 pin 机制的精髓:**置顶文档已经全文注入了,所以要把它们从向量检索结果里排除掉**,否则同一份内容会既被全文注入、又被检索召回,造成上下文重复浪费 token。`filterIdentifiers` 把 `pinnedDocIdentifiers` 传给向量库,让相似度搜索跳过这些来源。随后的 `fillSourceWindow({ nDocs: workspace?.topN || 4, ..., filterIdentifiers: pinnedDocIdentifiers })` 在用历史对话回填来源窗口时,同样带上这份排除名单,保证回填阶段也不会把置顶文档重新捞回来。

综合起来,pin 的设计意图非常清楚:对于用户认为"每一句话都重要、必须让模型完整看到"的文档(比如一份产品规格、一段合同条款),向量检索那种"只召回最相关几个片段"的方式是有损的;pin 让用户显式声明"这份文档别走检索,直接全文给模型"。代价是它吃 token,所以 `DocumentManager.pinnedDocs()` 才需要那道以 `promptWindowLimit()` 为上限的 token 预算闸门。普通文档则继续享受向量检索"大海捞针"的效率优势。两种机制并存、互斥(通过 `filterIdentifiers` 去重),共同构成工作区的上下文。

## 文档同步:可刷新的外部源如何定时重抓

文件类文档一旦嵌入就基本静止,但网页、YouTube、Confluence、GitHub 这类外部源的内容是会变的。AnythingLLM 用一套"监视(watch)+ 定时同步"的机制来让这些活源保持新鲜,核心模型是 `server/models/documentSyncQueue.js` 的 `DocumentSyncQueue`,执行体是 `server/jobs/sync-watched-documents.js`,调度者是 `server/utils/BackgroundWorkers/index.js`。

这是一个实验特性,由 `featureKey: "experimental_live_file_sync"` 标识,`enabled()` 方法读取 `SystemSettings` 里同名标签是否为 `"enabled"` 来判断开关。能被监视的源类型由 `validFileTypes` 限定:`["link", "youtube", "confluence", "github", "gitlab", "drupalwiki"]`。具体某份文档能否被监视则由 `canWatch({ title, chunkSource })` 判断,它检查 `chunkSource` 的协议前缀——例如 `link://` 且 `title` 以 `.html` 结尾、或 `youtube://`、`confluence://`、`github://`、`gitlab://`、`drupalwiki://` 开头才返回 `true`。纯文件上传(没有可重抓的源 URL)自然无法监视。

监视的开关由 `toggleWatchStatus(documentRecord, watchStatus)` 统一入口,它内部分发到 `watch` 或 `unwatch`。`watch(document)` 做两件事:先防重——查询所有 `filename` 相同且 `watched: true` 的文档记录,若已经存在对应的队列就抛错 `Cannot watch this document again - it already has a queue set.`,避免同一份文件在多个工作区里建出重复的同步队列;然后创建一条 `document_sync_queues` 记录,设置 `staleAfterMs` 和首次 `nextSyncAt`,并通过 `Document._updateAll({ filename }, { watched: true })` 把所有同名文档记录一并标记为已监视。`unwatch` 则反向删除队列并把 `watched` 置回 `false`。这里再次体现了"同一物理文件跨工作区共享一个监视队列"的设计——同步是按 `filename` 而非按工作区粒度去重的。

"多久算过期"由 `defaultStaleAfter` 这个 getter 决定,它的逻辑很有工程味道:

```js
get defaultStaleAfter() {
  const DEFAULT_STALE_AFTER = 604800000; // 7 days in MS
  const MIN_STALE_AFTER = 3600000; // 1 hour in MS
  const envValue = Number(process.env.DOCUMENT_SYNC_STALE_AFTER_MS);
  if (isNaN(envValue) || envValue <= 0) return DEFAULT_STALE_AFTER;
  return Math.max(envValue, MIN_STALE_AFTER);
}
```

默认 7 天(`604800000` 毫秒)过期一次,可通过环境变量 `DOCUMENT_SYNC_STALE_AFTER_MS` 覆盖,但强制下限 1 小时(`3600000` 毫秒)——注释直白地写明这是为了"避免重抓太频繁压垮嵌入器"。`calcNextSync(queueRecord)` 则用 `当前时间 + staleAfterMs` 算出下一次同步时刻。

调度由后台服务驱动。`BackgroundService`(`server/utils/BackgroundWorkers/index.js`)是一个基于 Bree 的单例调度器,它在 `boot()` 时读取 `DocumentSyncQueue.enabled()` 决定是否启用同步任务,`jobs()` 方法仅在 `documentSyncEnabled` 为真时才把同步任务推入待运行列表。该任务定义在 `#documentSyncJobs` 里,名为 `sync-watched-documents`,`interval: "1hr"`——也就是每小时唤醒一次去检查有没有过期文档。注意"每小时检查"和"每文档 7 天过期"是两个独立的节奏:worker 每小时跑一遍,但只处理那些 `nextSyncAt` 已到期的文档。

真正的重抓逻辑在 `server/jobs/sync-watched-documents.js`。它先调 `DocumentSyncQueue.staleDocumentQueues()` 取出所有 `nextSyncAt <= now` 的队列(并连带 `include` 出对应的 `workspaceDoc` 及其 `workspace`),若为空就退出。接着确认 collector 在线,然后逐条处理。对每条队列,它用 `Document.parseDocumentTypeAndSource(document)` 解析出 `type` 和 `source`,若类型非法或元数据缺失就调 `unwatch` 把它永久移出监视集。随后按类型分流:`link`、`youtube` 类把 `link` 传给 collector 的 `/ext/resync-source-document` 端点;`confluence`、`github`、`gitlab`、`drupalwiki` 类则把完整的 `chunkSource` 传过去(因为这些源的重抓需要编码在 URL 里的额外参数,这也呼应了 `documents.js` 里 `_stripSource` 在展示时特意剥离 search 参数的细节)。

重抓回来的 `newContent` 有三种命运。一是抓不到内容:代码会查最近 `maxRepeatFailures`(`5`)次运行里连续失败的次数,若达到 5 次就 `unwatch` 把这个坏掉的源彻底移除,否则记一条 `failed` 运行(`DocumentSyncRun.statuses.failed`)等下个周期重试。二是内容与磁盘上 `currentDocumentData.pageContent` 完全相同:说明源没变,只更新 `lastSyncedAt`/`nextSyncAt` 并记一条 `exited` 运行(原因 `Content unchanged.`),不触碰向量库。三是内容确实变了:这时才走真正的更新——先 `deleteDocumentFromNamespace` 删掉旧向量,再 `addDocumentToNamespace` 用新内容(`pageContent: newContent`)重新嵌入(并传 `true` 表示跳过缓存、强制新建向量缓存文件),同时 `updateSourceDocument` 刷新磁盘上的文档 JSON。

最妙的是更新之后的"扩散"(代码注释称之为 bloom)。由于同一份网页可能被加进了多个工作区(多条 `filename` 相同的 `workspace_documents` 记录),代码会用 `Document.where({ id: { not: document.id }, filename: document.filename }, ..., { workspace: true })` 找出所有其他引用了同名文件的工作区,逐个对它们的 namespace 也执行"删旧向量 + 嵌新内容",并把每个受影响的 slug 收进 `workspacesModified`。这样一次外部源的变更,会一致地传播到所有引用它的工作区,最终通过 `DocumentSyncQueue.saveRun(queue.id, success, { filename, workspacesModified })` 记下一条成功运行。每次运行的结果(`unknown`/`exited`/`failed`/`success`)都由 `server/models/documentSyncRun.js` 的 `DocumentSyncRun.save` 持久化进 `document_sync_executions` 表,既是审计轨迹,也是上面"连续失败 5 次则剔除"判断的数据来源。

## 对话历史的持久化:一笔带过

最后简单交代对话历史的落点。每一轮问答由 `server/models/workspaceChats.js` 的 `WorkspaceChats.new(...)` 写入 `workspace_chats` 表,记录里带 `workspaceId`、`prompt`、序列化后的 `response`、`user_id`、`thread_id`、`api_session_id` 以及一个关键布尔字段 `include`。`include` 默认为 `true`,表示这条历史是否参与后续对话的上下文拼装;当用户"重置/清空"对话时,`markThreadHistoryInvalidV2(whereClause)` 并不真的删除记录,而是把它们的 `include` 批量改为 `false`——历史被"软作废"而非物理删除,既保留了审计数据,又把它们排除出后续上下文。读取历史时,`forWorkspaceByUser`、`forWorkspace` 等查询都带上 `include: true` 过滤,且默认 `thread_id: null`、`api_session_id: null` 以区分默认线程、子线程和 API 会话。这套设计与前面 pin、检索一起,构成了"系统提示词 + 历史 + 置顶文档全文 + 向量检索片段"这个完整上下文拼装的最后一块拼图——而它的具体编排,正是下一章的主题。

## 本章小结

- 工作区(`server/models/workspace.js` 的 `Workspace`)是 AnythingLLM 的核心组织单元,既是向量隔离边界(`slug` 即向量库 namespace),又是对话配置载体(温度、阈值、topN、聊天模式、系统提示词等)。
- 工作区的可写字段由 `writable` 数组限定,`slug`/`vectorTag`/`pfpFilename` 被注释排除以保护身份稳定;每个字段有验证器,落定了 `similarityThreshold=0.25`、`topN=4`、`openAiHistory=20`、`chatMode="automatic"`、`vectorSearchMode="default"` 等默认值。
- 文档归属由 `server/models/documents.js` 的 `Document`(`workspace_documents` 表)承载,同一物理文件加入多个工作区会生成多条 `filename` 相同、`docId` 各异的记录。
- `addDocuments` 遵循"先向量化、成功后再落库"的顺序,保证不出现"有记录无向量"的孤儿;`removeDocuments` 对称地删向量、删记录、删 `document_vectors` 映射。
- `DocumentManager`(`server/utils/DocumentManager/index.js`)在对话时只负责置顶文档:`pinnedDocs()` 从磁盘读取全文,并以构造时传入的 `maxTokens`(实际取 LLM 的 `promptWindowLimit()`)为预算上限,超限则静默跳过。
- pin 机制让置顶文档走"全文注入"(`contextTexts.push(doc.pageContent)`),普通文档走向量检索;两者通过 `filterIdentifiers: pinnedDocIdentifiers` 去重,避免同一内容既注入又召回。
- 文档同步是实验特性 `experimental_live_file_sync`,只对 `link/youtube/confluence/github/gitlab/drupalwiki` 等有可重抓源的文档生效,由 `canWatch` 按 `chunkSource` 前缀判定。
- 过期阈值 `defaultStaleAfter` 默认 7 天(`604800000` ms),可由 `DOCUMENT_SYNC_STALE_AFTER_MS` 覆盖但下限 1 小时;`watch` 按 `filename` 去重,同名文件跨工作区共享一个同步队列。
- 后台调度由 `BackgroundService`(Bree)驱动,`sync-watched-documents` 任务每小时(`interval: "1hr"`)检查一次到期队列。
- 重抓逻辑(`server/jobs/sync-watched-documents.js`)区分"抓不到/内容未变/内容变更"三种命运,内容变更时删旧向量重嵌新内容,并"扩散(bloom)"到所有引用同名文件的其他工作区;连续失败 `maxRepeatFailures=5` 次则剔除监视。
- 对话历史由 `WorkspaceChats`(`workspace_chats` 表)持久化,`include` 字段实现历史的"软作废"——清空对话只是批量置 `include=false` 而非物理删除。

## 动手实验

1. **实验一:验证 pin 的"全文注入 + 检索排除"** — 在一个工作区里上传一份较长文档,先正常提问观察回答里的 Citations 来源片段;然后在文档管理面板把它 pin 置顶,再问同样的问题。结合 `server/utils/chats/stream.js` 第 148-157 行(`contextTexts.push(doc.pageContent)`)与第 185 行(`filterIdentifiers: pinnedDocIdentifiers`)阅读,确认置顶后该文档不再出现在向量检索结果中,而是被整体注入了上下文。

2. **实验二:推演置顶文档的 token 预算闸门** — 阅读 `server/utils/DocumentManager/index.js` 的 `pinnedDocs()`,找到 `tokens >= this.maxTokens` 的跳过逻辑;再到 `stream.js` 找到 `new DocumentManager({ workspace, maxTokens: LLMConnector.promptWindowLimit() })`。假设你 pin 了三份各占模型上下文窗口 40% 的文档,推演哪几份会被注入、哪份会被打印 `Skipping document - Token limit...` 日志跳过,并说明这种"静默丢弃"相比"直接报错"的取舍。

3. **实验三:跟踪一次外部源同步的完整路径** — 阅读 `server/models/documentSyncQueue.js` 的 `defaultStaleAfter`、`canWatch`、`watch`,再读 `server/jobs/sync-watched-documents.js`。画一张流程图,标出"每小时唤醒 → `staleDocumentQueues()` 取到期项 → 按 `type` 分流重抓 → 三种 `newContent` 命运 → bloom 扩散到同名文件的其他工作区"各环节,并指出 `DocumentSyncRun` 的四种 status 分别在哪一步写入。

4. **实验四:观察对话历史的软作废** — 在工作区里进行几轮对话,然后在数据库(SQLite)里查询 `workspace_chats` 表的 `include` 字段值;接着在界面上"清空对话历史",再查一次。对照 `server/models/workspaceChats.js` 的 `markThreadHistoryInvalidV2` 确认记录并未被 `DELETE`,而是 `include` 被批量改为 `false`,并思考这种设计对审计与隐私的双重影响。

> **下一章预告**:文档归属、置顶注入、向量检索、历史拼装,这些零件已经各就各位。第 7 章《RAG 聊天与检索流程》将把它们串成一条完整的链路——从用户消息进入 `stream.js`,经历系统提示词、对话历史、置顶文档全文、向量检索片段的上下文拼装,到 query/chat/automatic 三种模式的分支、`queryRefusalResponse` 的拒答兜底,再到流式回写与 Citations 生成。我们将完整走一遍 AnythingLLM 回答一个问题时,上下文究竟是怎样一砖一瓦垒起来的。
