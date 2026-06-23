# 第 5 章 向量数据库抽象层:一套接口,十余种后端

上一章我们看到文本如何被切分、被嵌入引擎转成一串浮点数。但向量本身不会自己说话——它必须被存进某个能做近邻检索的存储里,才能在用户提问时被召回。AnythingLLM 的设计目标之一,是让用户可以在 LanceDB(默认)、Pinecone、Chroma、Milvus、Qdrant、Weaviate、PGVector、Astra、Zilliz 等十余种后端之间自由切换,而上层的聊天检索、文档管理、工作区统计等逻辑完全不必感知底层用的是哪一家。

要做到这一点,AnythingLLM 没有引入重型的 ORM 或第三方向量框架,而是用一个朴素却有效的办法:定义一份"所有 provider 都必须实现的方法清单",让每个后端各自写一个 `index.js` 去实现它,再用一个 `switch` 工厂函数按环境变量分发。本章就沿着这条主线展开——先看接口契约长什么样,再通读默认实现 LanceDB 的写入与读取全流程,然后理解 namespace 即 workspace 的隔离模型,最后对比一个云托管实现 Pinecone,看抽象在不同后端上是如何被一致地兑现的。

## 5.1 统一契约:`VectorDatabase` 基类与 `BaseVectorDatabaseProvider`

抽象层有两处"契约"定义,它们互为补充。

第一处是真正的代码基类,位于 `server/utils/vectorDbProviders/base.js`,导出一个名为 `VectorDatabase` 的类。这个类本身不做任何业务,它的构造函数里有一道防线:

```js
constructor() {
  if (this.constructor === VectorDatabase) {
    throw new Error("VectorDatabase cannot be instantiated directly");
  }
}
```

也就是说 `VectorDatabase` 是抽象基类,直接 `new VectorDatabase()` 会抛错,只能被子类继承。基类把所有 provider 应当具备的方法都列了出来,每个方法的默认实现都是 `throw new Error("Must be implemented by provider")`——这是 JavaScript 没有 `abstract` 关键字时模拟"抽象方法"的常见手法:基类给出方法签名与 JSDoc 注释作为文档,但把具体逻辑强制下放给子类。基类声明的方法包括 `connect`、`heartbeat`、`totalVectors`、`namespaceCount`、`namespace`、`hasNamespace`、`namespaceExists`、`deleteVectorsInNamespace`、`addDocumentToNamespace`、`deleteDocumentFromNamespace`、`performSimilaritySearch`、`similarityResponse`,以及两个用方括号命名的方法 `"namespace-stats"` 与 `"delete-namespace"`(方法名带连字符,所以必须用字符串字面量定义),还有 `reset` 与 `curateSources`。基类唯一给出真实实现的是 `logger`,它统一了所有 provider 的日志前缀:

```js
logger(message = null, ...args) {
  console.log(`\x1b[36m[VectorDB::${this.name}]\x1b[0m ${message}`, ...args);
}
```

其中 `this.name` 由子类的 `get name()` 提供,例如 LanceDB 返回 `"LanceDb"`、Pinecone 返回 `"Pinecone"`,于是日志里就能看到 `[VectorDB::LanceDb]` 这样的带色前缀,这是基类提供给所有后端的"免费"统一能力。

第二处契约是一份 JSDoc 类型定义,写在工厂函数所在的 `server/utils/helpers/index.js` 顶部,名为 `BaseVectorDatabaseProvider` 的 `@typedef`。它不是可执行代码,而是给 IDE 与读代码的人看的"接口形状声明":

```js
/**
 * @typedef {Object} BaseVectorDatabaseProvider
 * @property {string} name - The name of the Vector Database instance.
 * @property {Function} connect - Connects to the Vector Database client.
 * @property {Function} totalVectors - Returns the total number of vectors in the database.
 * @property {Function} namespaceCount - Returns the count of vectors in a given namespace.
 * @property {Function} similarityResponse - Performs a similarity search on a given namespace.
 * @property {Function} rerankedSimilarityResponse - ... with reranking (if supported by provider).
 * @property {Function} namespace - Retrieves the specified namespace collection.
 * @property {Function} hasNamespace - Checks if a namespace exists.
 * @property {Function} namespaceExists - Verifies if a namespace exists in the client.
 * @property {Function} deleteVectorsInNamespace - Deletes all vectors in a specified namespace.
 * @property {Function} deleteDocumentFromNamespace - Deletes a document from a specified namespace.
 * @property {Function} addDocumentToNamespace - Adds a document to a specified namespace.
 * @property {Function} performSimilaritySearch - Performs a similarity search in the namespace.
 */
```

值得注意的是,`base.js` 的基类与 `helpers/index.js` 的 typedef 几乎一一对应,但并不完全相同:typedef 里额外列出了 `rerankedSimilarityResponse`(且注明"if supported by provider"),而基类没有声明它。这正好揭示了一个事实——不是每个后端都实现了重排,LanceDB 实现了 `rerankedSimilarityResponse`,而我们稍后会看到 Pinecone 就没有这个方法。换句话说,基类是"必须实现的最小集",typedef 则是"理想接口的完整描述",两者之间的差异恰恰是各后端能力差异的注脚。

工厂函数 `getVectorDbClass` 的 JSDoc 返回类型正是 `@returns { BaseVectorDatabaseProvider}`,这就把"工厂产出的实例满足统一接口"这件事在类型层面钉死了:无论底层是哪家数据库,上层代码拿到的都是一个满足同一组方法的对象。

## 5.2 默认实现 LanceDB:连接与单例

默认后端是 LanceDB,实现在 `server/utils/vectorDbProviders/lance/index.js`,类名 `LanceDb extends VectorDatabase`。LanceDB 是一个嵌入式的向量数据库——它不需要单独起服务,数据直接以文件形式落在磁盘上,这也是它适合做"开箱即用默认值"的原因。

存储路径由 getter `uri` 决定:

```js
get uri() {
  const basePath = !!process.env.STORAGE_DIR
    ? process.env.STORAGE_DIR
    : path.resolve(__dirname, "../../../storage");
  return path.resolve(basePath, "lancedb");
}
```

即数据落在 `${STORAGE_DIR}/lancedb` 目录下,生产环境读 `STORAGE_DIR` 环境变量,开发环境回退到仓库内的 `storage` 目录。

连接逻辑 `connect()` 用了一个静态私有字段做单例缓存:

```js
class LanceDb extends VectorDatabase {
  static #connection = null;
  ...
  async connect() {
    if (!LanceDb.#connection)
      LanceDb.#connection = await lancedb.connect(this.uri);
    return { client: LanceDb.#connection };
  }
}
```

`#connection` 是类级别的静态字段(`#` 表示真正的私有),意味着整个进程只会建立一次 LanceDB 连接,后续所有 `LanceDb` 实例共享同一个连接对象。这对嵌入式数据库尤为重要——避免反复打开同一个磁盘目录。`connect()` 统一返回 `{ client }` 形状,这正是基类契约里 `connect` 承诺的 `Promise<{client: any}>`,上层只认 `client` 这个键。

## 5.3 写入全流程:`addDocumentToNamespace`

写入是整个抽象层最重的方法。我们逐段拆解 LanceDB 的 `addDocumentToNamespace(namespace, documentData, fullFilePath, skipCache)`。

方法开头先从 `documentData` 里解构出正文与文档 ID,并对空文档短路:

```js
const { pageContent, docId, ...metadata } = documentData;
if (!pageContent || pageContent.length == 0) return false;
```

剩余字段被收进 `metadata`,后面会和文本一起作为向量的元数据存下来。

**第一步:查缓存。** 在没有显式 `skipCache` 的情况下,先尝试命中向量缓存:

```js
if (!skipCache) {
  const cacheResult = await cachedVectorInformation(fullFilePath);
  if (cacheResult.exists) {
    const { client } = await this.connect();
    const { chunks } = cacheResult;
    ...
    for (const chunk of chunks) {
      chunk.forEach((chunk) => {
        const id = uuidv4();
        const { id: _id, ...metadata } = chunk.metadata;
        documentVectors.push({ docId, vectorId: id });
        submissions.push({ id: id, vector: chunk.values, ...metadata });
      });
    }
    await this.updateOrCreateCollection(client, submissions, namespace);
    await DocumentVectors.bulkInsert(documentVectors);
    return { vectorized: true, error: null };
  }
}
```

这里的关键是:如果这份文件之前已经被嵌入并缓存过(`cacheResult.exists` 为真),就完全跳过昂贵的嵌入调用,直接从缓存里取出向量值 `chunk.values` 与元数据,为每个 chunk 重新生成一个 `uuidv4()` 作为 `vectorId`,写进 LanceDB 并在关系库里登记。缓存机制是本章 5.7 节的主题,这里先记住"命中缓存就不再嵌入"。

**第二步:切分。** 如果缓存未命中,就进入真正的"嵌入新文档"路径。源码注释特意说明了为什么不用 LangChain 的 `xyz.fromDocuments`——因为那样就无法"原子地控制 namespace 以便细粒度地查找/移除文档"。AnythingLLM 选择手动控制每一步。先取嵌入引擎,再构造切分器:

```js
const EmbedderEngine = getEmbeddingEngineSelection();
const textSplitter = new TextSplitter({
  chunkSize: TextSplitter.determineMaxChunkSize(
    await SystemSettings.getValueOrFallback({ label: "text_splitter_chunk_size" }),
    EmbedderEngine?.embeddingMaxChunkLength
  ),
  chunkOverlap: await SystemSettings.getValueOrFallback(
    { label: "text_splitter_chunk_overlap" }, 20
  ),
  chunkHeaderMeta: TextSplitter.buildHeaderMeta(metadata),
  chunkPrefix: EmbedderEngine?.embeddingPrefix,
});
const textChunks = await textSplitter.splitText(pageContent);
```

切分尺寸不是写死的:它取系统设置项 `text_splitter_chunk_size`,并用 `TextSplitter.determineMaxChunkSize` 与嵌入引擎自身的 `embeddingMaxChunkLength` 做约束(嵌入模型的最大上下文不能被切片超过);重叠 `chunkOverlap` 取设置项 `text_splitter_chunk_overlap`,缺省回退值是 `20`。这印证了第 3、4 章关于切分与嵌入的内容在这里被实际调用。

**第三步:向量化。** 把切好的文本块批量送入嵌入引擎:

```js
const vectorValues = await EmbedderEngine.embedChunks(textChunks);
```

随后遍历每个向量,组装出三份并行的数据结构。注意元数据里强制塞入了 `text` 字段,源码里专门有一段醒目的注释解释原因:

```js
const vectorRecord = {
  id: uuidv4(),
  values: vector,
  // [DO NOT REMOVE]
  // LangChain will be unable to find your text if you embed manually and dont include the `text` key.
  metadata: { ...metadata, text: textChunks[i] },
};
vectors.push(vectorRecord);
submissions.push({ ...vectorRecord.metadata, id: vectorRecord.id, vector: vectorRecord.values });
documentVectors.push({ docId, vectorId: vectorRecord.id });
```

三份结构各有用途:`vectors` 用于写入缓存(保留 `values`/`metadata` 的原始嵌套结构),`submissions` 是真正要写进 LanceDB 表的扁平记录(`id`、`vector`、以及展开的元数据),`documentVectors` 是要登记进关系库的 `{ docId, vectorId }` 映射。如果 `embedChunks` 返回空,直接抛错 `"Could not embed document chunks! This document will not be recorded."`,保证不会写入半成品。

**第四步:落库 + 写缓存。** 最后把数据真正写下去:

```js
if (vectors.length > 0) {
  const chunks = [];
  for (const chunk of toChunks(vectors, 500)) chunks.push(chunk);
  const { client } = await this.connect();
  await this.updateOrCreateCollection(client, submissions, namespace);
  await storeVectorResult(chunks, fullFilePath);
}
await DocumentVectors.bulkInsert(documentVectors);
return { vectorized: true, error: null };
```

`toChunks(vectors, 500)` 把向量按每批 500 个分组——这是为缓存文件做分批,便于后续逐批回放。`updateOrCreateCollection` 把 `submissions` 写进 LanceDB 表,`storeVectorResult` 把分批后的向量写入缓存文件,`DocumentVectors.bulkInsert` 把 `{ docId, vectorId }` 映射登记进关系库。整个方法用一个大 `try/catch` 包住,任何异常都会被捕获并返回 `{ vectorized: false, error: e.message }`,而不是让异常冒泡——这让上层文档管理逻辑可以拿到结构化的成功/失败结果。

这里还藏着一个 LanceDB 特有的"建表或追加"细节,即 `updateOrCreateCollection`:

```js
async updateOrCreateCollection(client, data = [], namespace) {
  const hasNamespace = await this.hasNamespace(namespace);
  if (hasNamespace) {
    const collection = await client.openTable(namespace);
    await collection.add(data);
    return true;
  }
  await client.createTable(namespace, data);
  return true;
}
```

namespace 在 LanceDB 里就是一张"表"(table)。表不存在就 `createTable` 用第一批数据建表(LanceDB 会据此推断 schema),表已存在就 `openTable` 后 `add` 追加。这就是 namespace 在 LanceDB 上的物理落点。

## 5.4 关系库登记:`models/vectors.js`

向量值本身存在向量数据库里,但 AnythingLLM 还需要在自己的关系库(通过 Prisma)里维护一份"哪个文档对应哪些向量 ID"的台账,这就是 `server/models/vectors.js` 导出的 `DocumentVectors`。

它的字段极简,只关心两件事:`docId`(文档 ID)与 `vectorId`(向量在向量库里的 ID)。`bulkInsert` 把一批映射用 `prisma.$transaction` 原子写入 `document_vectors` 表:

```js
bulkInsert: async function (vectorRecords = []) {
  if (vectorRecords.length === 0) return;
  const inserts = [];
  vectorRecords.forEach((record) => {
    inserts.push(prisma.document_vectors.create({
      data: { docId: record.docId, vectorId: record.vectorId },
    }));
  });
  await prisma.$transaction(inserts);
  return { documentsInserted: inserts.length };
},
```

这张台账的价值在删除时体现得淋漓尽致。向量数据库本身往往只能按 ID 删除,但它不知道"这个文档由哪些向量组成"。AnythingLLM 用 `DocumentVectors.where({ docId })` 反查出某文档的所有 `vectorId`,再去向量库里精准删除——这正是 `deleteDocumentFromNamespace` 的工作方式(下一节细看)。此外 `DocumentVectors` 还提供 `deleteForWorkspace`(按工作区批量清理,它先经 `Document.forWorkspace` 拿到全部 docId)、`deleteIds`、`delete` 等方法,构成了关系库侧的完整生命周期管理。

可以说,向量数据库负责"近邻检索",关系库台账负责"文档与向量的可追溯映射",两者分工明确,缺一不可。

## 5.5 读取全流程:相似度搜索、阈值过滤与 source 返回

读取入口是 `performSimilaritySearch`,它的参数也是基类契约规定的那一组:`namespace`、`input`、`LLMConnector`、`similarityThreshold`(默认 `0.25`)、`topN`(默认 `4`)、`filterIdentifiers`,LanceDB 还额外支持一个 `rerank` 开关。

方法先做防御性校验与 namespace 存在性检查,namespace 不存在就友好返回空结果而非抛错:

```js
if (!(await this.namespaceExists(client, namespace))) {
  return {
    contextTexts: [],
    sources: [],
    message: "Invalid query - no documents found for workspace!",
  };
}
```

接着把用户问题嵌入成查询向量,再根据 `rerank` 决定走普通搜索还是带重排的搜索:

```js
const queryVector = await LLMConnector.embedTextInput(input);
const result = rerank
  ? await this.rerankedSimilarityResponse({ client, namespace, query: input, queryVector, similarityThreshold, topN, filterIdentifiers })
  : await this.similarityResponse({ client, namespace, queryVector, similarityThreshold, topN, filterIdentifiers });
```

普通路径 `similarityResponse` 是核心。它打开表,用余弦距离做向量搜索,只取 `topN` 条:

```js
const response = await collection
  .vectorSearch(queryVector)
  .distanceType("cosine")
  .limit(topN)
  .toArray();
```

然后对每条结果做三件事:**阈值过滤、置顶文档过滤、组装结果**。

```js
response.forEach((item) => {
  if (this.distanceToSimilarity(item._distance) < similarityThreshold) return;
  const { vector: _, ...rest } = item;
  if (filterIdentifiers.includes(sourceIdentifier(rest))) {
    this.logger("A source was filtered from context as it's parent document is pinned.");
    return;
  }
  result.contextTexts.push(rest.text);
  result.sourceDocuments.push({ ...rest, score: this.distanceToSimilarity(item._distance) });
  result.scores.push(this.distanceToSimilarity(item._distance));
});
```

LanceDB 返回的是"距离"(`_distance`)而非"相似度",所以需要一个换算函数 `distanceToSimilarity`:

```js
distanceToSimilarity(distance = null) {
  if (distance === null || typeof distance !== "number") return 0.0;
  if (distance >= 1.0) return 1;
  if (distance < 0) return 1 - Math.abs(distance);
  return 1 - distance;
}
```

对余弦距离而言,`相似度 = 1 - 距离`,函数还对边界做了钳制。换算后的相似度若低于 `similarityThreshold`(默认 `0.25`),这条结果就被丢弃——这是阈值过滤,目的是把"勉强沾边"的低质量片段挡在上下文之外。

第二道过滤 `filterIdentifiers` 与 `sourceIdentifier(rest)` 配合,用于剔除那些"父文档已被置顶(pinned)"的片段。置顶文档会以完整原文的方式直接进入上下文,所以它的向量片段不应再被重复召回,日志会提示 `"A source was filtered from context as it's parent document is pinned."`。

最后,通过过滤的结果被组装成统一的返回形状。`performSimilaritySearch` 把 `sourceDocuments` 映射成 `sources`,并交给 `curateSources` 做最终清洗:

```js
const sources = sourceDocuments.map((metadata, i) => {
  return { metadata: { ...metadata, text: contextTexts[i] } };
});
return { contextTexts, sources: this.curateSources(sources), message: false };
```

`curateSources` 会把 `vector`、`_distance` 等内部字段剥掉,只留下对前端有意义的元数据与文本——这样返回给上层的 source 就是干净的、可直接展示来源引用的数据结构。无论哪个后端,`performSimilaritySearch` 都承诺返回 `{ contextTexts, sources, message }` 这同一形状,这是抽象一致性的核心体现。

关于带重排的 `rerankedSimilarityResponse`,LanceDB 的策略很务实:重排很贵,所以先用 `vectorSearch` 取回比 `topN` 多的候选,数量由 `Math.max(10, Math.min(50, Math.ceil(totalEmbeddings * 0.1)))` 决定——即取总向量数的 10%,但下限 10、上限 50,以保证即便工作区有上万向量也能在合理时间内重排。候选交给 `NativeEmbeddingReranker` 重排,最终分数取 `item?.rerank_score || this.distanceToSimilarity(item._distance)`,即优先用重排分,缺省回退到相似度。阈值过滤与置顶过滤逻辑与普通路径一致。

## 5.6 namespace 即 workspace:隔离模型

抽象层里反复出现的 `namespace`,在 AnythingLLM 的产品语义里就是一个 **workspace(工作区)**。这一点在端点代码里可以直接确认:`server/endpoints/api/workspace/index.js` 里调用 `await VectorDb.namespaceCount(workspace.slug)`,以及发起检索时传入 `namespace: workspace.slug`;`server/endpoints/workspaces.js`、`admin.js` 删除工作区时调用 `VectorDb["delete-namespace"]({ namespace: slug })`(或 `workspace.slug`)。也就是说,**workspace 的 `slug` 字段被直接当作 namespace 传给向量库**。

这个映射带来了天然的隔离:每个工作区的向量都存在以其 slug 命名的独立 namespace(在 LanceDB 里是独立的表,在 Pinecone 里是独立的命名空间)里,相似度搜索永远只在某一个 namespace 内进行,因此一个工作区的文档绝不会"串味"到另一个工作区的检索结果中。统计也是按 namespace 粒度的:`namespaceCount(namespace)` 返回单个工作区的向量数(LanceDB 实现里是 `openTable(namespace)` 后 `countRows()`),而 `totalVectors()` 则遍历所有表/命名空间求和——前者服务于"这个工作区有多少嵌入",后者服务于系统级统计。

删除工作区时,`"delete-namespace"` 直接 `deleteVectorsInNamespace`,在 LanceDB 里就是一句 `client.dropTable(namespace)`,整张表连同所有向量一并消失,干净利落。这种"一个工作区一个 namespace"的设计,把多租户隔离这件事下沉到了向量库的天然能力上,上层几乎零成本。

## 5.7 provider 如何被选择:`getVectorDbClass` 工厂

抽象层的"分发中枢"是 `server/utils/helpers/index.js` 里的 `getVectorDbClass`。它是一个朴素的 `switch`,读取环境变量 `VECTOR_DB`,缺省回退到 `"lancedb"`:

```js
function getVectorDbClass(getExactly = null) {
  const vectorSelection = getExactly ?? process.env.VECTOR_DB ?? "lancedb";
  switch (vectorSelection) {
    case "pinecone": { const { Pinecone } = require("../vectorDbProviders/pinecone"); return new Pinecone(); }
    case "chroma": ... return new Chroma();
    case "chromacloud": ... return new ChromaCloud();
    case "lancedb": ... return new LanceDb();
    case "weaviate": ... return new Weaviate();
    case "qdrant": ... return new QDrant();
    case "milvus": ... return new Milvus();
    case "zilliz": ... return new Zilliz();
    case "astra": ... return new AstraDB();
    case "pgvector": ... return new PGVector();
    default:
      console.error(`...No VECTOR_DB value found in environment! Falling back to LanceDB`);
      const { LanceDb: DefaultLanceDb } = require("../vectorDbProviders/lance");
      return new DefaultLanceDb();
  }
}
```

这个函数有几个值得玩味的设计点。

其一,**默认值出现了两次**:`vectorSelection` 的 `?? "lancedb"` 确保环境变量缺失时走 `case "lancedb"`,而 `default` 分支又一次回退到 LanceDB——后者覆盖的是"`VECTOR_DB` 被设成了一个无法识别的值"的情况(比如拼错)。两道防线都指向 LanceDB,这与它"零配置默认后端"的定位完全一致,并且 `default` 分支会用红色 `[ENV ERROR]` 打印警告,提示运维配置可能有误。

其二,**`require` 是惰性的**——每个 `case` 内部才 `require` 对应的 provider 模块,而不是在文件顶部一次性全部导入。这避免了在选用 LanceDB 时还去加载 Pinecone、Milvus 等 SDK 的开销与潜在的初始化副作用,也意味着只有被选中的那个后端的依赖才需要在运行环境里就绪。

其三,**返回类型被 JSDoc 标注为 `BaseVectorDatabaseProvider`**,`getExactly` 参数则被标注为那一长串字面量联合类型。`getExactly` 的存在让调用方可以"绕过环境变量,指定一个特定后端",这在某些迁移或管理场景下有用。

上层代码因此只需写 `const VectorDb = getVectorDbClass()`,拿到的就是一个满足统一契约的实例,后续调用 `VectorDb.performSimilaritySearch(...)`、`VectorDb.addDocumentToNamespace(...)` 等等,完全不需要知道也不需要分支判断当前到底用的是哪个数据库。这就是抽象层为整个系统屏蔽复杂性的方式。

## 5.8 云托管 provider 的差异点:以 Pinecone 为对照

把 LanceDB(嵌入式)与 Pinecone(云托管)放在一起对照,最能看清"统一接口"是如何容纳"实现差异"的。Pinecone 的实现位于 `server/utils/vectorDbProviders/pinecone/index.js`,类名 `PineconeDB extends VectorDatabase`,`get name()` 返回 `"Pinecone"`。

**连接差异。** LanceDB 的 `connect` 打开本地目录并做进程级单例;Pinecone 的 `connect` 则要校验环境变量、用 API Key 实例化云客户端、定位索引并确认索引就绪:

```js
async connect() {
  if (process.env.VECTOR_DB !== "pinecone") throw new Error("Pinecone::Invalid ENV settings");
  const client = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
  const pineconeIndex = client.Index(process.env.PINECONE_INDEX);
  const { status } = await client.describeIndex(process.env.PINECONE_INDEX);
  if (!status.ready) throw new Error("Pinecone::Index not ready.");
  return { client, pineconeIndex, indexName: process.env.PINECONE_INDEX };
}
```

注意它返回的对象里除了 `client` 还有 `pineconeIndex`、`indexName`,比 LanceDB 多。这不破坏契约——契约只要求 `connect` 返回含 `client` 的对象,provider 可以在自己的内部方法间约定更多字段。

**namespace 落点差异。** 在 LanceDB,namespace 是一张表;在 Pinecone,namespace 是同一个 index 内的逻辑命名空间,通过 `pineconeIndex.namespace(namespace)` 获得。`namespaceExists` 在 LanceDB 里查 `tableNames()`,在 Pinecone 里查 `describeIndexStats()` 返回的 `namespaces` 字典是否包含该键。`totalVectors` 在 LanceDB 里遍历表累加 `countRows()`,在 Pinecone 里则对 `describeIndexStats()` 的 `namespaces` 各项 `recordCount` 求和。落点与统计手段完全不同,但对外暴露的方法名与返回语义一致。

**相似度返回差异。** Pinecone 的 `similarityResponse` 直接拿到 `match.score`(Pinecone 自带相似度分),无需像 LanceDB 那样做 `distanceToSimilarity` 距离换算——这是最直观的实现差异之一。其阈值过滤逻辑等价:`if (match.score < similarityThreshold) return;`,置顶过滤同样用 `sourceIdentifier(match.metadata)`。

**写入差异。** Pinecone 的 `addDocumentToNamespace` 整体骨架与 LanceDB 几乎逐行对应——同样先查缓存、未命中再切分嵌入、最后落库并写缓存登记台账,连那段 `[DO NOT REMOVE]` 强制写入 `text` 元数据的注释都一模一样。差异在落库动作:Pinecone 用 `pineconeNamespace.upsert([...chunk])` 上传,且分批粒度是 `toChunks(vectors, 100)`(而 LanceDB 是 500);删除时 `deleteDocumentFromNamespace` 用 `pineconeNamespace.deleteMany(batchOfVectorIds)`,分批 `toChunks(vectorIds, 1000)`。这些批量大小差异源于各家 API 的限制与最佳实践。

**能力差异。** 前面提过,Pinecone 没有实现 `rerankedSimilarityResponse`——它的 `performSimilaritySearch` 里没有 `rerank` 分支,直接调用 `similarityResponse`。这正是 typedef 注释里"if supported by provider"那句话的现实写照:统一接口划定了"大家都得有"的底线(基类强制的方法),同时给"有些后端才有"的增强能力(如重排)留了空间。

把这两个 provider 并排看,结论很清晰:**接口一致、行为一致、实现各异**。上层永远只面对 `connect`/`addDocumentToNamespace`/`performSimilaritySearch` 这组方法和它们统一的返回形状,而每家数据库的连接方式、namespace 物理形态、距离/相似度语义、批量限制、是否支持重排,都被各自的 `index.js` 妥善吸收。

## 5.9 向量缓存:避免重复嵌入的省钱机制

嵌入是要花钱(调用商业 API)或花算力(本地模型)的,而同一份文件可能被加入多个工作区,或被移除后再次加入。AnythingLLM 用一套基于文件内容路径的缓存机制,确保"一份文件只嵌入一次"。这套机制实现在 `server/utils/files/index.js`,被各 provider 共享调用。

缓存目录是 `vectorCachePath`,生产环境位于 `${STORAGE_DIR}/vector-cache`,开发环境位于仓库内 `storage/vector-cache`。

写缓存的是 `storeVectorResult(vectorData, filename)`:

```js
async function storeVectorResult(vectorData = [], filename = null) {
  if (!filename) return;
  if (!fs.existsSync(vectorCachePath)) fs.mkdirSync(vectorCachePath);
  const digest = uuidv5(filename, uuidv5.URL);
  const writeTo = path.resolve(vectorCachePath, `${digest}.json`);
  fs.writeFileSync(writeTo, JSON.stringify(vectorData), "utf8");
  return;
}
```

它用 `uuidv5(filename, uuidv5.URL)` 把文件全路径哈希成一个确定性的摘要 `digest`,以 `${digest}.json` 为名写入缓存目录。`uuidv5` 是确定性的——同一个 `filename` 永远得到同一个 `digest`,这正是缓存命中的基础。

读缓存的是 `cachedVectorInformation(filename, checkOnly)`:

```js
async function cachedVectorInformation(filename = null, checkOnly = false) {
  if (!filename) return checkOnly ? false : { exists: false, chunks: [] };
  const digest = uuidv5(filename, uuidv5.URL);
  const file = path.resolve(vectorCachePath, `${digest}.json`);
  const exists = fs.existsSync(file);
  if (checkOnly) return exists;
  if (!exists) return { exists, chunks: [] };
  console.log(`Cached vectorized results of ${filename} found! Using cached data to save on embed costs.`);
  const rawData = fs.readFileSync(file, "utf8");
  return { exists: true, chunks: JSON.parse(rawData) };
}
```

它用同样的 `uuidv5` 算法算出 `digest`,检查对应缓存文件是否存在。命中时直接读回之前存下的、已经分好批的向量数据(`chunks`),日志会打印 `"...found! Using cached data to save on embed costs."`——明确点出这套机制的动机就是"省嵌入成本"。

把这两端接回 5.3 节就完整了:`addDocumentToNamespace` 在写入新文档的成功路径末尾调用 `storeVectorResult(chunks, fullFilePath)` 把向量缓存下来;下次再加入同一文件时,开头的 `cachedVectorInformation(fullFilePath)` 命中缓存,于是跳过 `EmbedderEngine.embedChunks` 这一最昂贵的步骤,直接把缓存里的向量值写进目标 namespace。注意:**缓存的命中只跳过嵌入,不跳过落库与台账登记**——因为每次加入都要为新的 namespace 生成新的 `vectorId` 并写进关系库,这也是为什么从缓存恢复时仍然要 `uuidv4()` 生成新 ID。`skipCache` 参数则提供了一个绕过缓存、强制重新嵌入的开关,留给确实需要重算的场景。

文件侧还配套了 `purgeVectorCache`、`hasVectorCachedFiles`、`purgeEntireVectorCache` 等管理函数,用于在文件被删除或需要整体重建时清理缓存,但核心的"读—写—命中"三件套就是上面这两个函数。

## 本章小结

1. AnythingLLM 的向量库抽象由两份契约共同定义:`server/utils/vectorDbProviders/base.js` 的 `VectorDatabase` 抽象基类(构造函数禁止直接实例化,方法默认 `throw "Must be implemented by provider"`),以及 `helpers/index.js` 顶部的 `BaseVectorDatabaseProvider` JSDoc typedef。
2. 基类是"必须实现的最小集",typedef 比基类多列了 `rerankedSimilarityResponse` 并注明"if supported by provider",两者差异恰好对应各后端的能力差异(LanceDB 有重排,Pinecone 没有)。
3. 默认后端 LanceDB 实现于 `lance/index.js`,是嵌入式数据库,数据落在 `${STORAGE_DIR}/lancedb`;`connect()` 用静态私有字段 `#connection` 做进程级单例,namespace 在物理上就是一张 LanceDB 表。
4. 写入主方法 `addDocumentToNamespace` 的完整流程是:查缓存 → (未命中则)切分 → 向量化 → 组装 → 落库 + 写缓存 + 登记关系库台账,全程 `try/catch` 包裹,失败返回 `{ vectorized: false, error }`,且刻意手动控制每步而不用 LangChain 的 `fromDocuments` 以便细粒度增删。
5. 关系库台账 `models/vectors.js` 的 `DocumentVectors` 只记 `{ docId, vectorId }`,通过 `prisma.$transaction` 原子写入;它让"按文档精准删除向量"成为可能(`deleteDocumentFromNamespace` 先 `where({ docId })` 反查 vectorId 再删)。
6. 读取主方法 `performSimilaritySearch` 把问题嵌成查询向量,LanceDB 用 `vectorSearch().distanceType("cosine").limit(topN)` 检索,经 `distanceToSimilarity` 把距离换算为相似度。
7. 检索结果要过两道闸:相似度低于 `similarityThreshold`(默认 `0.25`)的被丢弃,父文档已置顶(pinned)的片段被 `filterIdentifiers` + `sourceIdentifier` 剔除;最终经 `curateSources` 清洗成统一的 `{ contextTexts, sources, message }` 返回。
8. namespace 即 workspace:端点代码直接把 `workspace.slug` 作为 namespace 传入,从而实现工作区级别的天然隔离与独立统计(`namespaceCount` 单区、`totalVectors` 全局)。
9. provider 选择由 `getVectorDbClass` 工厂的 `switch (process.env.VECTOR_DB ?? "lancedb")` 完成,惰性 `require` 仅加载被选中的后端,默认值与 `default` 分支两道防线都回退到 LanceDB。
10. 云托管 provider(以 Pinecone 为例)接口一致但实现各异:连接需 API Key 与索引就绪检查、namespace 是 index 内逻辑命名空间、`match.score` 自带相似度无需换算、批量大小不同(upsert 100 / delete 1000)、且未实现重排。
11. 向量缓存(`files/index.js` 的 `storeVectorResult` / `cachedVectorInformation`)用 `uuidv5(filename, uuidv5.URL)` 把文件路径哈希成确定性 `digest`,命中即跳过最昂贵的嵌入步骤;但仍会为新 namespace 生成新 `vectorId` 并登记台账,`skipCache` 可强制重算。

## 动手实验

1. **实验一:画出统一契约对照表** — 打开 `server/utils/vectorDbProviders/base.js` 与 `helpers/index.js` 的 `BaseVectorDatabaseProvider` typedef,逐项列出两者声明的方法;再打开 `lance/index.js` 与 `pinecone/index.js`,在表里标注每个后端是否实现了该方法。重点确认 `rerankedSimilarityResponse` 只在 LanceDB 出现,以此理解"最小集"与"理想接口"的区别。

2. **实验二:跟踪一次写入的数据流** — 在 `addDocumentToNamespace` 的四个关键点(`cachedVectorInformation` 调用后、`textSplitter.splitText` 后、`embedChunks` 后、`storeVectorResult` 前)各插入一条 `this.logger(...)` 打印数组长度,然后向一个工作区上传同一份文档两次。观察第一次会打印切分块数与嵌入数,第二次因缓存命中而跳过嵌入,据此验证 5.7 节描述的"命中缓存只省嵌入、不省落库登记"。

3. **实验三:验证 namespace 即 workspace 的隔离** — 在 `getVectorDbClass()` 拿到的实例上分别调用 `namespaceCount(workspaceA.slug)` 与 `namespaceCount(workspaceB.slug)`,再调用 `totalVectors()`。确认前两者之和(在只有两个工作区时)等于后者,并用 `tables()` 列出 LanceDB 的表名,核对它们正是各工作区的 slug。

4. **实验四:观察阈值过滤的效果** — 在调用 `performSimilaritySearch` 时把 `similarityThreshold` 从默认 `0.25` 临时调到 `0.6`,对同一个问题各检索一次,对比返回 `sources` 的数量与每条的 `score`。再把 `rerank` 设为 `true` 走 `rerankedSimilarityResponse` 路径,观察候选数量(由 `Math.max(10, Math.min(50, ...))` 决定)与最终 `score` 的来源(`rerank_score` 优先)如何变化。

> **下一章预告**:本章我们看清了向量如何被存取、namespace 如何对应工作区。但"工作区"远不止是一个向量命名空间——它还管理着文档的挂载、置顶、监听同步,以及一份份文件从 `documents` 目录到向量库的完整生命周期。第 6 章《文档管理与工作区》将沿着 `models/workspace.js`、`models/documents.js` 与文档同步队列,解构 AnythingLLM 是如何把"文件"组织成"可对话的知识库"的。
