# 第 3 章 文本切分子系统 TextSplitter

上一章我们看到文档采集器(Collector)把各种格式的原始文件压成统一的纯文本。但纯文本并不能直接喂给向量数据库——一篇上万字的文章如果整体生成一个向量,既会超出嵌入模型的输入上限,也会让"语义粒度"粗到无法检索。RAG 真正可用的前提,是把长文本切成一段段大小适中、彼此略有重叠的 chunk,再逐段嵌入入库。切分的好坏直接决定了召回质量:切得太大,单个 chunk 里混入太多无关信息,检索命中后塞进上下文反而稀释了相关内容;切得太小,语义被截断,一个完整的论点被劈成两半,任何一半都无法独立回答问题。

本章解构 AnythingLLM 的文本切分子系统,核心实现集中在一个文件里:`server/utils/TextSplitter/index.js`。这个文件不长(两百行出头),但它把"配置从哪来、默认值是多少、底层用什么算法、chunk 如何带上元数据"四件事串成了一条完整链路。我们会从切分参数的来源讲起,沿着配置注入、默认值与边界、递归字符切分算法、chunkOverlap 的作用、chunk header 元数据,一直走到它在整个 RAG 管线里被调用的位置。

## 一个类两层封装:TextSplitter 与 RecursiveSplitter

打开 `server/utils/TextSplitter/index.js`,你会看到它导出的只有一个类:文件末尾的 `module.exports.TextSplitter = TextSplitter;`。但实际上文件里定义了两个类。外层是 `TextSplitter`,负责读取配置、构造元数据头、对外暴露 `splitText`;内层是 `RecursiveSplitter`,它是对 langchain 的 `RecursiveCharacterTextSplitter` 的薄封装,真正干切分的活。

外层 `TextSplitter` 用了一个私有字段 `#splitter` 来持有内层实例。它的构造函数非常克制:

```js
constructor(config = {}) {
  this.config = config;
  this.#splitter = this.#setSplitter(config);
}
```

所有切分行为都被收敛进 `#setSplitter` 这个私有方法。这种"外层管配置与元数据、内层管算法"的分层,好处是把 langchain 的依赖隔离在 `RecursiveSplitter` 一个类里——如果将来要替换底层切分库,只需重写内层,外层调用方代码(各个向量数据库 provider)完全无感。

## 切分参数从哪来:三段式的配置注入

要理解 AnythingLLM 怎么决定 `chunkSize`,不能只看 `TextSplitter` 自己。它的构造参数是由**调用方**在向量化时现场算出来的。我们以 LanceDB 的 provider 为例,`server/utils/vectorDbProviders/lance/index.js` 在嵌入一篇新文档前会这样构造切分器:

```js
const EmbedderEngine = getEmbeddingEngineSelection();
const textSplitter = new TextSplitter({
  chunkSize: TextSplitter.determineMaxChunkSize(
    await SystemSettings.getValueOrFallback({
      label: "text_splitter_chunk_size",
    }),
    EmbedderEngine?.embeddingMaxChunkLength
  ),
  chunkOverlap: await SystemSettings.getValueOrFallback(
    { label: "text_splitter_chunk_overlap" },
    20
  ),
  chunkHeaderMeta: TextSplitter.buildHeaderMeta(metadata),
  chunkPrefix: EmbedderEngine?.embeddingPrefix,
});
```

这段代码揭示了配置注入的三个来源,层层叠加:

第一,**用户配置**来自 `SystemSettings.getValueOrFallback`,读取的 label 是 `text_splitter_chunk_size` 与 `text_splitter_chunk_overlap`。这两个键就是后台"文本切分与分块"设置面板里那两个输入框对应的持久化字段。

第二,**嵌入模型的硬限制**来自 `EmbedderEngine?.embeddingMaxChunkLength`。任何嵌入模型对单次输入的 token 数都有上限——例如 OpenAI 的 embedder 在 `server/utils/EmbeddingEngines/openAi/index.js` 里把 `this.embeddingMaxChunkLength = 8_191;` 写死,native 嵌入引擎则在 `server/utils/EmbeddingEngines/native/index.js` 里从 `this.modelInfo.embeddingMaxChunkLength` 读取。用户设的 chunkSize 可以小于这个上限,但绝不能超过,否则嵌入请求会直接失败。

第三,**模型前缀**来自 `EmbedderEngine?.embeddingPrefix`。某些嵌入模型(如 native 引擎在 `get embeddingPrefix()` 里返回的值)要求在每段文本前加特定前缀才能发挥最佳效果,这个值通过 `chunkPrefix` 传进来。

值得注意的是,所有 8 个向量数据库 provider(`astra`、`chroma`、`lance`、`qdrant`、`milvus`、`weaviate`、`pgvector`、`pinecone`)都用了完全相同的构造模式——它们都 `require("../../TextSplitter")` 并以同样的四个参数 `chunkSize`/`chunkOverlap`/`chunkHeaderMeta`/`chunkPrefix` new 出切分器。这说明切分逻辑是 RAG 管线里的一个公共前置步骤,与具体存哪个向量库无关。

## 默认值与边界:isNaN 兜底与三处 20

`TextSplitter` 在三个地方都为 `chunkSize` 和 `chunkOverlap` 准备了默认值,而且这些默认值彼此一致,体现了一种"层层兜底"的防御式设计。

最贴近算法的兜底在 `#setSplitter` 里:

```js
#setSplitter(config = {}) {
  // if (!config?.splitByFilename) {// TODO do something when specific extension is present? }
  return new RecursiveSplitter({
    chunkSize: isNaN(config?.chunkSize) ? 1_000 : Number(config?.chunkSize),
    chunkOverlap: isNaN(config?.chunkOverlap)
      ? 20
      : Number(config?.chunkOverlap),
    chunkHeader: this.stringifyHeader(),
  });
}
```

这里默认 `chunkSize` 是 `1_000`、`chunkOverlap` 是 `20`。注意它用 `isNaN` 而不是简单的"或运算兜底",这是有意为之:`isNaN(0)` 为 `false`,所以如果用户真的想把 overlap 设成 0,这个值会被尊重而不会被默认值 20 覆盖。这一点和 JSDoc 注释里 `@property {number} [config.chunkOverlap = 20]` 的声明保持一致。

第二处兜底在配置读取层。前面 lance 的代码里,`getValueOrFallback({ label: "text_splitter_chunk_overlap" }, 20)` 把 fallback 同样设为 `20`——当数据库里根本没存过这个设置时,直接返回 20。

第三处兜底在配置写入层。`server/models/systemSettings.js` 里为这两个字段挂了校验函数,它们既做合法性检查也兜底默认值:

```js
text_splitter_chunk_size: (update) => {
  try {
    if (isNullOrNaN(update)) throw new Error("Value is not a number.");
    if (Number(update) <= 0) throw new Error("Value must be non-zero.");
    const { purgeEntireVectorCache } = require("../utils/files");
    purgeEntireVectorCache();
    return Number(update);
  } catch (e) {
    console.error(
      `Failed to run validation function on text_splitter_chunk_size`,
      e.message
    );
    return 1000;
  }
},
text_splitter_chunk_overlap: (update) => {
  try {
    if (isNullOrNaN(update)) throw new Error("Value is not a number");
    if (Number(update) < 0) throw new Error("Value cannot be less than 0.");
    const { purgeEntireVectorCache } = require("../utils/files");
    purgeEntireVectorCache();
    return Number(update);
  } catch (e) {
    console.error(
      `Failed to run validation function on text_splitter_chunk_overlap`,
      e.message
    );
    return 20;
  }
},
```

两条边界规则在这里被明确写死:`chunk_size` 必须大于 0(`Number(update) <= 0` 抛错),`chunk_overlap` 不能小于 0(`Number(update) < 0` 抛错,但允许等于 0)。校验失败时分别回落到 `1000` 和 `20`,与算法层、读取层完全对齐。

这段校验里还藏着一个重要副作用:每次成功修改切分参数,都会调用 `purgeEntireVectorCache()` 清空整个向量缓存。原因不难理解——chunkSize 一旦变了,之前按旧参数切出来并缓存的 chunk 就全部作废,留着只会导致新旧切分混用,所以必须强制重新切分。`server/models/systemSettings.js` 还在 `protectedFields`/`publicFields` 之类的字段清单(第 44–46、75–76 行)里登记了 `text_splitter_chunk_size`、`text_splitter_chunk_overlap`、`max_embed_chunk_size`,让它们能被前端正常读写。

## determineMaxChunkSize:把用户意愿和模型上限对齐

`chunkSize` 不是直接用用户填的值,而是先过一道静态方法 `determineMaxChunkSize`:

```js
static determineMaxChunkSize(preferred = null, embedderLimit = 1000) {
  const prefValue = isNullOrNaN(preferred)
    ? Number(embedderLimit)
    : Number(preferred);
  const limit = Number(embedderLimit);
  if (prefValue > limit)
    console.log(
      `\x1b[43m[WARN]\x1b[0m Text splitter chunk length of ${prefValue} exceeds embedder model max of ${embedderLimit}. Will use ${embedderLimit}.`
    );
  return prefValue > limit ? limit : prefValue;
}
```

它的逻辑可以一句话概括:取用户偏好值 `preferred` 与模型上限 `embedderLimit` 中较小的那个。如果用户没设(`isNullOrNaN(preferred)` 为真),就直接采用模型上限;如果用户设的值超过了模型上限,会打印一条黄底 `[WARN]` 警告并强制截断到 `embedderLimit`。文件顶部的 `isNullOrNaN` 辅助函数同时处理 `null` 和 `NaN` 两种缺失情况,保证 `preferred` 缺省时退化到模型上限而不会算出 NaN。这道工序的意义在于:它把"用户想要的粒度"和"模型物理能承受的粒度"做了一次安全对齐,避免因为用户填了一个过大的数字导致后续整批嵌入失败。

## 底层算法:langchain 的递归字符切分

真正的切分动作落在内层 `RecursiveSplitter`。它在构造时才 `require` langchain,把依赖延迟到真正用到的那一刻:

```js
class RecursiveSplitter {
  constructor({ chunkSize, chunkOverlap, chunkHeader = null }) {
    const {
      RecursiveCharacterTextSplitter,
    } = require("@langchain/textsplitters");
    this.log(`Will split with`, {
      chunkSize,
      chunkOverlap,
      chunkHeader: chunkHeader ? `${chunkHeader?.slice(0, 50)}...` : null,
    });
    this.chunkHeader = chunkHeader;
    this.engine = new RecursiveCharacterTextSplitter({
      chunkSize,
      chunkOverlap,
    });
  }
  // ...
}
```

底层用的是 langchain 的 `RecursiveCharacterTextSplitter`,来自 `@langchain/textsplitters` 包。"递归字符切分"的核心思想是:它持有一组按优先级排列的分隔符(默认大致是段落分隔 `\n\n`、换行 `\n`、空格、最后退化到单字符),先尝试用最高优先级的分隔符切;如果切出来的片段仍然超过 `chunkSize`,就在该片段内换用下一级分隔符递归地继续切,直到每段都不超过目标大小。这样做的好处是尽量保留自然的语义边界——优先在段落和句子处断开,只有实在塞不下时才硬切。这正是注释 `// Wrapper for Langchain default RecursiveCharacterTextSplitter class.` 所说的"对 langchain 默认切分器的封装"[[RecursiveCharacterTextSplitter 官方文档]](https://js.langchain.com/docs/concepts/text_splitters/)。

需要强调:AnythingLLM 本身**没有重新实现**任何切分算法,它把"按什么粒度、保留多少重叠、在哪里断句"这些算法细节完全交给了 langchain。`#setSplitter` 里那行被注释掉的 `// if (!config?.splitByFilename) {// TODO do something when specific extension is present? }` 暴露了一个尚未落地的设想:按文件扩展名走不同切分策略(例如 Markdown、代码用专门的 splitter)。但在当前版本里,**所有文件类型都走同一条递归字符切分路径**,没有 `splitByFileExtension` 这类分支真正生效——这是阅读源码才能确认、而文档里不会写明的事实。

## chunkOverlap 的作用:让语义不在边界处断裂

`chunkOverlap` 默认 `20`,表示相邻两个 chunk 之间会共享 20 个字符(注意 langchain 这里是字符长度而非 token)的重叠内容。为什么需要重叠?因为切分边界是机械的,一个完整的句子或论点很可能恰好跨在两个 chunk 的接缝上。如果没有重叠,前一个 chunk 在句子中途戛然而止,后一个 chunk 从句子中途开始,两边都丢失了上下文,检索时谁都无法独立命中。让相邻 chunk 互相"咬合"一小段,就能让跨边界的语义在至少一个 chunk 里保持完整。

不过 20 个字符的默认重叠相对 1000 的 chunkSize 来说是很小的(2%),AnythingLLM 选择了一个偏保守、偏省存储的默认值,把更激进的重叠留给用户在设置面板里自行调大。前面提到的边界校验允许 overlap 等于 0,意味着用户可以完全关掉重叠;但 `chunkSize` 不允许为 0,因为那会让切分失去意义。

## chunk 如何带上元数据:buildHeaderMeta 与 stringifyHeader

切出来的纯文本片段如果只有正文,LLM 在引用时就无法说清"这段话出自哪里"。AnythingLLM 用一套元数据头机制解决这个问题,涉及两个方法:静态的 `buildHeaderMeta` 负责"从文档元数据里提炼要写进头部的字段",实例的 `stringifyHeader` 负责"把这些字段渲染成一段文本前缀"。

`buildHeaderMeta` 通过一张 `PLUCK_MAP` 表来挑选字段,目前只挑三项:

```js
const PLUCK_MAP = {
  title: {
    as: "sourceDocument",
    pluck: (metadata) => {
      return metadata?.title || null;
    },
  },
  published: {
    as: "published",
    pluck: (metadata) => {
      return metadata?.published || null;
    },
  },
  chunkSource: {
    as: "source",
    pluck: (metadata) => {
      const validPrefixes = ["link://", "youtube://"];
      // ...
    },
  },
};
```

也就是说,文档的 `title` 会被重命名为 `sourceDocument`,`published` 保持原名,而 `chunkSource` 被重命名为 `source`。`chunkSource` 的处理最讲究:它只接受以 `link://` 或 `youtube://` 开头的来源(`validPrefixes`),并把前缀剥掉只保留真正的 URL。源码注释把意图说得很清楚——这样当用户问"你这条信息从哪来的"时,LLM 能回答出 `https://example.com` 这样的真实出处。如果 `chunkSource` 不是字符串、为空、或没有合法前缀,`pluck` 返回 `null`,这一项就不会写进头部。整个方法在 `metadata` 为空对象时直接返回 `null`,表示"无头可加"。

挑选出来的字段经 `stringifyHeader` 渲染成最终前缀:

```js
stringifyHeader() {
  let content = "";
  if (!this.config.chunkHeaderMeta) return this.#applyPrefix(content);
  Object.entries(this.config.chunkHeaderMeta).map(([key, value]) => {
    if (!key || !value) return;
    content += `${key}: ${value}\n`;
  });

  if (!content) return this.#applyPrefix(content);
  return this.#applyPrefix(
    `<document_metadata>\n${content}</document_metadata>\n\n`
  );
}
```

最终每个 chunk 前面会被加上一段形如 `<document_metadata>\nsourceDocument: ...\npublished: ...\nsource: ...\n</document_metadata>\n\n` 的头部。这里还嵌套了私有方法 `#applyPrefix`,它负责把前面提到的嵌入模型前缀 `chunkPrefix` 拼到最前面:`if (!this.config.chunkPrefix) return text;` 没有前缀就原样返回,有则前置。这样一来,一段最终入库的 chunk 文本可能是"模型前缀 + 元数据头 + 正文"三段拼接。

头部如何真正进入每个 chunk,要看内层 `_splitText`:

```js
async _splitText(documentText) {
  if (!this.chunkHeader) return this.engine.splitText(documentText);
  const strings = await this.engine.splitText(documentText);
  const documents = await this.engine.createDocuments(strings, [], {
    chunkHeader: this.chunkHeader,
  });
  return documents
    .filter((doc) => !!doc.pageContent)
    .map((doc) => doc.pageContent);
}
```

逻辑分两条路:没有 header 时,直接返回 langchain `splitText` 的字符串数组;有 header 时,先 `splitText` 得到片段,再用 langchain 的 `createDocuments` 把 `chunkHeader` 作为 chunkHeader 注入到每个文档,最后 `filter` 掉空 `pageContent`、`map` 出纯文本数组。换句话说,元数据头是借 langchain 的 `createDocuments` 机制逐 chunk 前置进去的,而不是 AnythingLLM 自己手工拼接。外层 `splitText` 只是把调用转发给内层:`return this.#splitter._splitText(documentText);`。

## 切分在 RAG 管线里的位置

把视角拉回整条管线。在任意一个向量库 provider(以 lance 为例)的入库函数里,流程是:先 `getEmbeddingEngineSelection()` 拿到嵌入引擎 → `new TextSplitter(...)` 构造切分器 → `const textChunks = await textSplitter.splitText(pageContent);` 得到 chunk 数组 → 打印 `"Snippets created from document:"` 和 chunk 数量 → `await EmbedderEngine.embedChunks(textChunks)` 批量嵌入 → 把每个向量和它对应的 `metadata: { ...metadata, text: textChunks[i] }` 一起写入向量库。

这个顺序说明:**切分是嵌入的紧前置步骤**,夹在"文档已转成纯文本"和"逐 chunk 调用嵌入模型"之间。chunk 数组的下标 `i` 同时贯穿了"嵌入向量"和"原文文本"——`textChunks[i]` 既被嵌入,也被原样塞进向量记录的 `text` 字段,供检索命中后回填上下文。源码注释还特别提醒,这个 `text` key 绝不能省略,否则 LangChain 检索时找不到原文。至此,切分子系统的产物——一批带元数据头、大小受控、彼此略有重叠的纯文本 chunk——就正式交棒给了下一站:嵌入引擎。

## 本章小结

1. 切分实现高度集中,核心全在 `server/utils/TextSplitter/index.js`,文件只导出 `TextSplitter` 一个类,内部再分外层 `TextSplitter` 与内层 `RecursiveSplitter` 两层。
2. 切分参数有三个来源叠加:用户配置(`SystemSettings` 的 `text_splitter_chunk_size`/`text_splitter_chunk_overlap`)、嵌入模型上限(`embeddingMaxChunkLength`)、模型前缀(`embeddingPrefix`)。
3. 默认值在算法层 `#setSplitter`、读取层 `getValueOrFallback` 的 fallback、写入层 `systemSettings.js` 校验函数三处保持一致:`chunkSize` 默认 `1000`,`chunkOverlap` 默认 `20`。
4. 边界规则写死在校验函数里:`chunkSize` 必须大于 0,`chunkOverlap` 不能小于 0 但允许等于 0;用 `isNaN`/`isNullOrNaN` 兜底,刻意保留了"用户把 overlap 设为 0"的意图。
5. `determineMaxChunkSize` 取用户值与模型上限的较小者,超限时打印 `[WARN]` 并截断到 `embedderLimit`,把用户意愿与模型物理上限安全对齐。
6. 底层算法完全复用 langchain 的 `RecursiveCharacterTextSplitter`(`@langchain/textsplitters`),AnythingLLM 不自研切分算法,依赖被隔离在 `RecursiveSplitter` 一个类里且延迟 `require`。
7. 当前版本对所有文件类型走同一条递归字符切分路径,`splitByFilename`/特定扩展名分支只是 `#setSplitter` 里一行被注释掉的 TODO,尚未生效。
8. `chunkOverlap` 让相邻 chunk 咬合一小段,避免语义在机械边界处断裂;默认 20 字符偏保守省存储,可由用户调大或关闭。
9. chunk 元数据头由 `buildHeaderMeta` 经 `PLUCK_MAP` 提炼 `title→sourceDocument`、`published`、`chunkSource→source` 三项,`chunkSource` 只接受 `link://`/`youtube://` 前缀并剥离前缀保留 URL。
10. `stringifyHeader` 把元数据渲染成 `<document_metadata>...</document_metadata>` 文本头,再经 `#applyPrefix` 前置模型前缀;`_splitText` 借 langchain 的 `createDocuments` 把头部逐 chunk 注入。
11. 切分是嵌入的紧前置步骤,8 个向量库 provider 都用相同模式调用;修改切分参数会触发 `purgeEntireVectorCache()` 强制重切,chunk 与其原文 `text` 字段按下标 `i` 严格对应入库。

## 动手实验

1. **实验一:验证默认值与 isNaN 兜底** — 在 `server` 目录用 Node 直接 `require("./utils/TextSplitter")`,分别 `new TextSplitter({})`、`new TextSplitter({ chunkSize: 0 })`、`new TextSplitter({ chunkOverlap: 0 })`,观察控制台 `[RecursiveSplitter] Will split with` 日志里打印的实际 `chunkSize`/`chunkOverlap`,确认空配置回落到 1000/20,而 `chunkOverlap: 0` 被尊重而非被覆盖。

2. **实验二:观察递归字符切分与重叠** — 准备一段约 3000 字符、含多个 `\n\n` 段落的长文本,用 `chunkSize: 500, chunkOverlap: 50` 调用 `splitText`,打印每个 chunk 的长度与首尾 50 个字符,核对相邻 chunk 之间确实存在约 50 字符的重叠,并尝试把 overlap 改成 0 再对比接缝处的差异。

3. **实验三:跟踪元数据头的生成** — 构造一个 `metadata = { title: "测试文档", published: "2026-01-01", chunkSource: "link://https://example.com" }`,先单独调用 `TextSplitter.buildHeaderMeta(metadata)` 看 `PLUCK_MAP` 输出的字段改名(`sourceDocument`/`published`/`source`),再把它作为 `chunkHeaderMeta` 传入切分器,打印第一个 chunk 确认 `<document_metadata>` 头被前置;然后把 `chunkSource` 改成不带 `link://` 前缀的普通字符串,验证 `source` 项被剔除。

4. **实验四:定位切分在管线中的调用点** — 用 Grep 在 `server/utils/vectorDbProviders/` 下搜索 `new TextSplitter`,任选 `lance/index.js` 通读其入库函数,确认调用顺序是"取嵌入引擎 → 构造切分器 → `splitText` → `embedChunks` → 写库",并找到 `metadata: { ...metadata, text: textChunks[i] }` 这行,理解为什么 chunk 原文必须随向量一起存。

> **下一章预告**:切分子系统交出了一批大小受控、带元数据头的纯文本 chunk,接下来它们要被转换成可供相似度检索的向量。第 4 章我们将解构向量化引擎 `EmbeddingEngines`,看 AnythingLLM 如何统一抽象 native、OpenAI 等多种嵌入后端,`embeddingMaxChunkLength`、`embeddingPrefix`、`embedChunks` 这些本章反复出现的概念将在那里得到完整解答。
