# 第 2 章 文档采集器:把任意文件变成可嵌入的结构化文本

上一章我们俯瞰了 AnythingLLM 的三进程运行时,知道了 `collector` 是一个独立的 Express 应用,专门负责"把原始素材变成文本"。这一章我们钻进 `collector` 进程内部,看它究竟如何接收一个文件、如何按类型把它分发到正确的转换器、如何把 PDF、Word、音频、网页、Excel 这些异构素材统一压平成同一套 `document` JSON 结构,再交给 `server` 进程去切分与向量化。

为什么这一进程值得单独成章?因为 RAG 系统的检索质量,有相当一部分在"文本进来之前"就已经被决定了。一个扫描版 PDF 如果没有走 OCR,后面再强的 embedding 和 rerank 都无米可炊;一个 xlsx 如果不按 sheet 拆成多个文档,跨表语义就会被搅成一锅粥。`collector` 把"格式适配"这件脏活累活从主服务里隔离出来,既让 `server` 保持干净,也让"支持一种新文件类型"退化成"写一个新的转换器文件"。我们就沿着一份文件从进门到出门的完整生命周期,逐段拆解它的实现。

## 进门:`/process`、`/parse` 与 3GB 的体量假设

`collector` 的入口是 `collector/index.js`,它是一个标准的 Express 应用。最值得先注意的是文件体量假设:文件顶部定义了常量 `const FILE_LIMIT = "3GB";`,随后这个值被同时塞进了 `bodyParser.text`、`bodyParser.json`、`bodyParser.urlencoded` 三种解析器的 `limit` 选项。也就是说,无论上传方以何种 content-type 把内容打过来,Express 都允许单次请求体积膨胀到 3GB。这是一个透露设计意图的字面量:`collector` 默认要处理的是"大文件",音频、视频、整本 PDF 都在它的射程之内,所以它宁可放开请求体上限,也不愿意因为 body 超限而把一份合法的大文件拒之门外。

真正干活的路由有两个孪生入口:`/process` 和 `/parse`,都挂着 `[verifyPayloadIntegrity]` 中间件(签名校验我们在第 1 章已经讲过,这里只需记住:非 development 环境下,没有合法 `X-Integrity` 签名头的请求会被 400 拒绝)。两者都从请求体里取出 `filename`,然后做同一件防御性的事——路径归一化:

```js
const targetFilename = path
  .normalize(filename)
  .replace(/^(\.\.(\/|\\|$))+/, "");
```

这一行用 `path.normalize` 折叠掉 `.` 和 `..`,再用正则把开头残留的 `../` 序列剥光,目的就是防止调用方用 `../../etc/passwd` 这类相对路径穿越出工作目录。`/process` 与 `/parse` 的唯一区别在于传给 `processSingleFile` 的 `options`:`/parse` 会强制注入 `parseOnly: true`,并允许携带 `absolutePath`。`parseOnly` 这个开关后面会反复出现,它决定了"产出的文档到底是被当成正式知识库素材,还是只读一次的临时上传"。

`index.js` 里还排布着另外几个入口,构成了 `collector` 的完整 API 面:`/process-link` 处理网页链接,`/util/get-link` 只取网页文本而不落盘,`/util/convert-audio-to-wav` 单独做音频格式转换,`/process-raw-text` 接收前端直接粘贴的纯文本,以及通过 `extensions(app)` 挂载的一整组 `/ext/*` 扩展数据源路由。最后还有一个 `app.get("/accepts", ...)`,它直接把常量 `ACCEPTED_MIMES` 原样吐回给调用方,让前端能动态知道"我到底能上传什么"。所有未匹配的路由由 `app.all("*", ...)` 兜底返回 200。

一个容易被忽略但很能说明设计哲学的细节:几乎所有 catch 分支返回的都是 `status(200)` 配合 `success: false`,而不是 4xx/5xx。也就是说,处理失败在 `collector` 看来是"正常业务结果"而非"HTTP 错误"。调用方(`server`)永远拿到结构一致的 `{ filename, success, reason, documents }`,靠 `success` 字段判断成败,而不用去解析 HTTP 状态码。这种"用 body 表达业务结果、用状态码只表达传输是否到达"的约定,贯穿了整个 `collector`。

还有一处启动时的副作用值得一提:`app.listen` 的回调里会 `await wipeCollectorStorage()`。这意味着每次 `collector` 进程重启,都会把 `hotdir` 和 `storage/tmp` 里的残留文件清空(只保留 `__HOTDIR__.md` 和 `.placeholder`)。设计者在 `collector/utils/files/index.js` 的注释里说得很直白:大文件处理失败后可能留下无法删除的残骸,一次重启就把它们强制清掉。

## 分发:扩展名优先,文本兜底

请求进门后,核心调度发生在 `collector/processSingleFile/index.js` 的 `processSingleFile` 函数。它做的第一件事是把文件名解析成绝对路径:

```js
const fullFilePath = normalizePath(
  options.absolutePath || path.resolve(WATCH_DIRECTORY, targetFilename)
);
```

这里出现了 `WATCH_DIRECTORY`。它定义在 `collector/utils/constants.js` 顶部:`const WATCH_DIRECTORY = require("path").resolve(__dirname, "../hotdir");`。所谓 `hotdir`(热目录),就是 `collector` 监视的"投件口"——`server` 把待处理文件放进 `hotdir`,然后调用 `/process` 告诉 `collector` 去处理哪个文件名。除非调用方显式传了 `absolutePath`(仅供内部使用),否则文件名一律相对于 `hotdir` 解析。

紧接着是一连串防御性校验,顺序本身就是契约的体现:如果没传 `absolutePath`,就用 `isWithin(path.resolve(WATCH_DIRECTORY), fullFilePath)` 确认目标仍落在 `hotdir` 之内,防住路径穿越;命中保留文件名列表 `RESERVED_FILES = ["__HOTDIR__.md"]` 直接拒绝;文件不存在则报 `"File does not exist in upload directory."`;甚至还有一条很细的判断——`if (fullFilePath.includes(".") && !fileExtension)`,即"路径里有点号但取不出扩展名"时拒绝处理。每一条失败都返回统一的 `{ success: false, reason, documents: [] }`。

校验通过后才是真正的分发逻辑,它的核心是一张映射表 `SUPPORTED_FILETYPE_CONVERTERS`(同样在 `constants.js` 里),把扩展名映射到转换器模块的相对路径:

```js
let processFileAs = fileExtension;
if (!SUPPORTED_FILETYPE_CONVERTERS.hasOwnProperty(fileExtension)) {
  if (isTextType(fullFilePath)) {
    processFileAs = ".txt";
  } else {
    if (!options.absolutePath) trashFile(fullFilePath);
    return { success: false, reason: `...not supported...`, documents: [] };
  }
}
const FileTypeProcessor = require(SUPPORTED_FILETYPE_CONVERTERS[processFileAs]);
return await FileTypeProcessor({ fullFilePath, filename: targetFilename, options, metadata });
```

这段代码体现了一个非常实用的"扩展名优先、文本兜底"策略。第一优先级是扩展名:`SUPPORTED_FILETYPE_CONVERTERS` 把 `.pdf` 映射到 `"./convert/asPDF/index.js"`、`.docx` 映射到 `"./convert/asDocx.js"`、所有音频扩展名(`.mp3`、`.wav`、`.mp4`、`.ogg`、`.m4a`、`.webm` 等)统统映射到 `"./convert/asAudio.js"`、图片(`.png`、`.jpg`、`.jpeg`、`.webp`)映射到 `"./convert/asImage.js"`。值得玩味的是,`.txt`、`.md`、`.org`、`.adoc`、`.rst`、`.csv`、`.json`、`.html` 这八种扩展名全都指向同一个 `"./convert/asTxt.js"`——在 `collector` 眼里它们本质上都是"纯文本",差异留给后续切分阶段去关心。

当扩展名不在映射表里时,代码不会立刻放弃,而是调用 `isTextType(fullFilePath)` 做"内容嗅探"。如果嗅探认定它是文本,就打印一行黄色警告并把 `processFileAs` 改成 `".txt"`,当作纯文本走 `asTxt`;只有在嗅探也失败时,才会 `trashFile` 删掉文件并报错。这个 `isTextType` 是 `collector` 鲁棒性的关键,我们单独看它。

## 内容嗅探:`isTextType` 如何"看出"一个文件是不是文本

`isTextType` 定义在 `collector/utils/files/index.js`,它采用两级判断,先看 MIME、再看字节。

第一级是 `isKnownTextMime`,它借助 `collector/utils/files/mime.js` 里的 `MimeDetector` 类(包装了 npm 的 `mime` 库)。`MimeDetector` 维护了两份黑名单:`badMimes` 列举了 `application/octet-stream`、`application/zip`、`application/vnd.microsoft.portable-executable` 等明显是二进制的类型,特别地还把 `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`(即 xlsx)显式拉黑——代码注释直言 "XLSX are binaries and need to be handled explicitly",意思是 xlsx 绝不能被当文本嗅探蒙混过关,必须走专门的 `asXlsx`。`nonTextTypes = ["multipart", "model", "audio", "video", "font"]` 则按 MIME 大类拦截。`MimeDetector` 还做了一组有趣的 `setOverrides`:它把 `ts`、`tsx`、`py`、`go`、`sh`、`lua` 等一长串源码扩展名强制重映射成 `text/plain`,并附了注释解释为什么——`.ts` 默认会被 `mime` 库判成 `video/mp2t`(MPEG transport stream 比 TypeScript 出现得早),如果不纠正,你上传一个 TypeScript 文件就会被当成视频。这种"为了正确处理源代码而打的补丁"很能说明 `collector` 服务的真实使用场景。

第二级是 `parseableAsText`,这是在 MIME 给不出确切答案(`reason === "generic"`)时的兜底。它的实现是直接读文件头 1KB:

```js
const buffer = Buffer.alloc(1024);
const bytesRead = fs.readSync(fd, buffer, 0, 1024, 0);
const content = buffer.subarray(0, bytesRead).toString("utf8");
const nullCount = (content.match(/\0/g) || []).length;
const controlCount = (content.match(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g) || []).length;
const threshold = bytesRead * 0.1;
return nullCount + controlCount < threshold;
```

逻辑朴素而有效:把前 1KB 当 UTF-8 解码,统计空字节和控制字符的数量,只要它们占比低于 10%(`bytesRead * 0.1`),就认定这是文本。这是一种经典的"二进制 vs 文本"启发式——真正的文本几乎不含 NUL 和奇怪的控制字符,而二进制文件则遍地都是。靠这个嗅探,`collector` 不必为每一种冷门的纯文本格式都登记 MIME,就能把它们识别出来并当 `.txt` 处理。

## 出门:统一的 `document` 数据结构

无论走哪个转换器,最终都收敛到同一个 JSON 形状。我们以最简单的 `collector/processSingleFile/convert/asTxt.js` 为模板,它构造的 `data` 对象是整套体系的"标准件":

```js
const data = {
  id: v4(),
  url: "file://" + fullFilePath,
  title: metadata.title || filename,
  docAuthor: metadata.docAuthor || "Unknown",
  description: metadata.description || "Unknown",
  docSource: metadata.docSource || "a text file uploaded by the user.",
  chunkSource: metadata.chunkSource || "",
  published: createdDate(fullFilePath),
  wordCount: content.split(" ").length,
  pageContent: content,
  token_count_estimate: tokenizeString(content),
};
```

逐字段看它透露的契约:`id` 用 `uuid` 的 `v4()` 生成,保证全局唯一;`url` 统一加 `"file://"` 前缀,标明来源;`title`、`docAuthor`、`description`、`docSource`、`chunkSource` 这几个元数据字段全部走 `metadata.X || <默认值>` 的模式,意味着调用方可以覆盖,不覆盖就用兜底值;`published` 调 `createdDate` 取文件的 birthtime,取不到时返回字符串 `"unknown"`;`wordCount` 用极朴素的 `content.split(" ").length` 估算词数;最关键的 `pageContent` 装的就是抽取出来的全文;`token_count_estimate` 调 `collector/utils/tokenizer` 里的 `tokenizeString` 给出 token 估算——它内部用 `js-tiktoken` 的 `cl100k_base` 编码,并以单例模式复用 encoder,还设了 `MAX_KB_ESTIMATE` 之类的护栏防止对超长字符串做昂贵编码。

构造好 `data` 之后,所有转换器都调用 `writeToServerDocuments`(定义在 `utils/files/index.js`)把它落盘成 JSON。这个函数决定了文档去哪个目录:如果传了 `destinationOverride` 就用它;否则若 `options.parseOnly` 为真就写进 `directUploadsFolder`(direct-uploads 目录,临时上传、不进知识库);默认情况写进 `documentsFolder` 下的 `"custom-documents"` 子目录。文件名会先经过 `sanitizeFileName` 剥掉 Windows 非法字符(包括各种 Unicode 引号),再补上 `.json` 后缀。函数返回时还会补两个字段:`location`(取路径最后两段拼成相对位置,供 `server` 的 `/update-embeddings` 使用)和 `isDirectUpload`。这就是为什么所有转换器返回的 `documents[0]` 都比构造时多了这两个键。

`asDocx.js`、`asPDF/index.js`、`asAudio.js`、`asImage.js` 全都复刻了这套"加载 → 抽文本 → 空内容则失败 → 构造 data → writeToServerDocuments → 非 absolutePath 则 trashFile → 返回"的骨架。比如 `asDocx` 用 `langchain` 的 `DocxLoader` 把 docx 拆成若干 `pageContent` 再 `join("")`;`asPDF` 用本地的 `PDFLoader`(`splitPages: true`)逐页加载。它们的差别只在"用什么库把字节变成字符串",拼装与落盘完全一致。这种高度一致的模板,正是"加一种新格式只需写一个新文件"的底气来源。

一个值得单独点出的设计是 `trashFile` 的条件:几乎每个转换器在收尾时都写 `if (!options.absolutePath) trashFile(fullFilePath);`。也就是说,通过 `hotdir` 进来的文件用完即焚(已经转成 JSON 了,原件没有保留价值),但当调用方显式给了 `absolutePath`(说明这是用户自己的文件、不归 `collector` 管理),`collector` 绝不删它。这条规则在分发阶段拒绝处理时也成立——`processSingleFile` 里凡是 `trashFile` 调用都裹着同样的 `if (!options.absolutePath)` 判断。

## 特殊路径之一:PDF 的"先解析、空了再 OCR"

`collector/processSingleFile/convert/asPDF/index.js` 体现了对扫描件的优雅降级。它先用数字 PDF 解析器尝试抽文字:

```js
let docs = await pdfLoader.load();
if (docs.length === 0) {
  console.log(`[asPDF] No text content found for ${filename}. Will attempt OCR parse.`);
  docs = await new OCRLoader({
    targetLanguages: options?.ocr?.langList,
  }).ocrPDF(fullFilePath);
}
```

逻辑是:`PDFLoader.load()` 对数字版 PDF(文字是可选中的真实文本)能直接抽出内容;一旦抽出来是空的——典型就是扫描版 PDF,整页其实是图片——就退回到 `OCRLoader` 的 `ocrPDF`。这种"能不 OCR 就不 OCR"的顺序选择很务实,因为 OCR 又慢又可能引入识别错误,只在数字解析彻底失败时才动用。`asPDF` 还会从 `docs[0].metadata.pdf.info` 里尝试读取 `Creator` 和 `Title` 作为 `docAuthor` 和 `description` 的次级默认值,把 PDF 自带的元数据利用起来。

## 特殊路径之二:OCR 的并行工作池

`collector/utils/OCRLoader/index.js` 里的 `OCRLoader` 类是图片和扫描 PDF 的共同底座,它基于 `tesseract.js`。构造时它做两件事:一是用 `parseLanguages` 解析逗号分隔的语言代码(如 `"eng,deu"`),并用 `validLangs.js` 里的 `VALID_LANGUAGE_CODES` 做白名单过滤,任何无效输入都回退到 `["eng"]`;二是把 Tesseract 的模型缓存目录定位到 `STORAGE_DIR/models/tesseract`,并确保目录存在,避免 Tesseract 把训练数据下载到默认位置反复重拉。

`ocrPDF` 的实现是这一章里工程含量最高的一段。它动态 `import` 了 pinned 版本的 `pdf-parse/lib/pdf.js/v2.0.550/build/pdf.js`,用 PDF.js 逐页拿到页面;再用内置的 `PDFSharp` 类(包装 `sharp`)把每一页渲染成图片 buffer——注意它只挑 `paintJpegXObject`、`paintImageXObject`、`paintInlineImageXObject` 这几个绘图操作符对应的图像对象,并以 `targetDPI = 70` 做了降采样以控制内存和耗时。然后它创建一个 Tesseract worker 池:

```js
const NUM_WORKERS = maxWorkers ?? Math.min(os.cpus().length, 4);
const workerPool = await Promise.all(
  Array(NUM_WORKERS).fill(0).map(() =>
    createWorker(this.language, OEM.LSTM_ONLY, { cachePath: this.cacheDir })
  )
);
```

worker 数量取 CPU 核数与 4 的较小值,识别引擎固定用 `OEM.LSTM_ONLY`。整个识别按 `BATCH_SIZE`(默认 10)分批,批内用一个 `pageQueue` 配合多个 worker 做"抢任务"式并发——每个 worker 在 `while (pageQueue.length > 0)` 里 `shift` 一个页码自己处理,天然实现了负载均衡。所有结果按页码 `sort` 回正确顺序后推入 `documents`。更关键的是它用 `Promise.race([timeoutPromise, processPages()])` 给整个 OCR 任务套了 `maxExecutionTime`(默认 `300_000` 毫秒,即 5 分钟)的硬超时,超时就抛错。无论成功失败,`finally` 块都会把 `global.Image` 复位并 `terminate` 所有 worker,杜绝资源泄漏。`ocrImage` 是它的简化版:单 worker、单次识别、同样套着 5 分钟超时和 `finally` 清理。

## 特殊路径之三:音频转写

`collector/processSingleFile/convert/asAudio.js` 处理所有音频和视频扩展名。它的分发同样是一张映射表:

```js
const WHISPER_PROVIDERS = {
  openai: OpenAiWhisper,
  local: LocalWhisper,
};
const WhisperProvider = WHISPER_PROVIDERS.hasOwnProperty(options?.whisperProvider)
  ? WHISPER_PROVIDERS[options?.whisperProvider]
  : WHISPER_PROVIDERS.local;
```

也就是说,音频转写可以走 OpenAI 的 Whisper API,也可以走本地模型,默认回退到 `local`。无论哪个 provider,接口都是 `whisper.processFile(fullFilePath, filename)` 返回 `{ content, error }`。拿到转写文本后,后续的 data 拼装、`writeToServerDocuments`、`trashFile`、返回值,和前面的 `asTxt` 一模一样——又一次印证了"统一的 document 结构 + 可插拔的抽取后端"这个核心模式。`index.js` 里还单独暴露了 `/util/convert-audio-to-wav` 路由,调用 `convertAudioToWav`,处理那些需要先转码成 wav 才能喂给转写引擎的格式。

## 特殊路径之四:网页链接

网页走的是另一条独立入口 `/process-link`,落到 `collector/processLink/index.js` 的 `processLink`。它先 `validateURL` + `validURL` 校验,再调 `scrapeGenericUrl`(在 `processLink/convert/generic.js`)。这个函数的精妙之处是它会先用 `determineContentType` 探测链接到底是什么:

```js
let { contentType, processVia } = await determineContentType(link);
if (processVia === "file")
  return await processAsFile({ uri: link, saveAsDocument });
else if (processVia === "youtube")
  return await loadYouTubeTranscript({ url: link }, { parseOnly: saveAsDocument === false });
```

`determineContentType`(在 `processLink/helpers/index.js`)的逻辑是:先用 `validYoutubeVideoUrl` 判断是不是 YouTube 视频,是就走 `youtube`(去抓字幕);否则发一个带 5 秒超时的 `HEAD` 请求拿 `Content-Type`,如果返回的 MIME 既不是 `text/html`/`text/plain`、又恰好在 `ACCEPTED_MIMES` 里(比如直接指向一个 `.pdf` 的链接),就标记为 `file`,走 `processAsFile` 把它下载下来再交给 `processSingleFile` 当文件处理。换言之,"贴一个 PDF 直链"和"上传一个 PDF 文件"最终走的是同一条管线。

只有当链接确实是网页时,才走 `getPageContent` 真正抓取。它的首选方案是 `langchain` 的 `PuppeteerWebBaseLoader`,用无头 Chromium 渲染页面(`waitUntil: "networkidle2"`),抓到 `innerHTML` 后用 `htmlToMarkdown` 转成 Markdown——这一步很重要,它把 HTML 噪声压成干净的 Markdown 文本,对后续 embedding 友好得多。如果调用方传了自定义 `scraperHeaders`,它还会重写 `loader.scrape` 方法注入这些头。一旦 Puppeteer 整体失败,它会 `catch` 住并降级到普通的 `fetch` + `htmlToMarkdown` 作为最后的保底。最终保存时,`chunkSource` 字段会被设成 `` `link://${link}` ``,把来源 URL 记进文档,便于后续溯源。`/util/get-link` 则复用同一套抓取逻辑,但传 `saveAsDocument: false`,只返回文本不落盘,主要服务于 Agent 工具调用场景。

## 特殊路径之五:原始文本与扩展数据源

最轻量的入口是 `/process-raw-text`,对应 `collector/processRawText/index.js`。它直接接收前端传来的 `textContent` 和 `metadata`,空文本则返回失败,否则用一组 `METADATA_KEYS.possible` 里的函数把元数据规整好。这里有个细节挺巧:`url` 字段的生成函数会先试着把 `metadata.url` 解析成合法的 http/https URL,成功就拼成 `` `web://${url}.website` ``,失败就退化成 `` `file://${stripAndSlug(title)}.txt` ``,用一个字段同时兼容"网页来源"和"纯本地文本"两种语义。

而 `collector/extensions/index.js` 则通过 `extensions(app)` 一次性挂载了一大批 `/ext/*` 路由,这是 `collector` 接入第三方系统的扩展面:`/ext/:repo_platform-repo` 拉取 GitHub/GitLab 仓库、`/ext/youtube-transcript` 抓 YouTube 字幕、`/ext/website-depth` 做带深度的整站爬取(默认 `depth = 1`、`maxLinks = 20`)、`/ext/confluence`、`/ext/drupalwiki`、`/ext/obsidian/vault`、`/ext/paperless-ngx`,以及一个 `/ext/resync-source-document` 用于按 `type` 重新同步已有来源。这些涉及凭据的路由(如 confluence、drupalwiki、obsidian、paperless)除了 `verifyPayloadIntegrity` 还额外挂了 `setDataSigner` 中间件,把 `EncryptionWorker` 注入到 `response.locals`,用于在落盘前加密 API key 等敏感信息——这正好呼应了第 1 章讲过的加密体系。无论数据来自哪种扩展,它们最终都殊途同归,落到同一套 `document` JSON 结构上。

## 一个例外:xlsx 的"一表一文档"

绝大多数转换器都是"一个文件 → 一个文档",但 `collector/processSingleFile/convert/asXlsx.js` 是个有意思的例外,值得专门看,因为它体现了对结构化数据的额外照顾。它用 `node-xlsx` 解析工作簿,然后分两种模式:当 `options.parseOnly` 为真时,它把所有 sheet 拼成一个合并文档(标题里带上 `(Sheets: ...)` 列表);而在默认(非 parseOnly)模式下,它会为每个 sheet 单独生成一个文档,每张表先用 `convertToCSV` 转成 CSV 文本作为 `pageContent`,落到一个以文件名命名的子文件夹里。也就是说,默认情况下一个有三张表的 xlsx 会产出三个 `document`,返回的 `documents` 数组长度为 3。这背后的考量很清楚:不同 sheet 往往是不同主题的数据,拆开能让检索更精准,避免跨表内容被混在同一个嵌入向量里。这也是为什么 `processSingleFile` 的返回类型注释写的是 `documents: Object[]`——它从一开始就允许一份文件产出多份文档。

## 本章小结

- `collector` 是独立的 Express 应用,入口 `collector/index.js` 定义 `FILE_LIMIT = "3GB"` 并把它套到三种 bodyParser 上,体现了"默认处理大文件"的假设。
- `/process` 与 `/parse` 是孪生入口,都做 `path.normalize` + 剥 `../` 的路径防穿越;`/parse` 额外强制 `parseOnly: true` 并允许 `absolutePath`。
- 几乎所有失败都返回 `status(200)` 配 `success: false`,业务成败由 body 的 `success` 字段表达,而非 HTTP 状态码。
- 分发核心 `processSingleFile` 采用"扩展名优先、文本兜底"策略:先查 `SUPPORTED_FILETYPE_CONVERTERS` 映射表,查不到再用 `isTextType` 内容嗅探,认定为文本就按 `.txt` 处理,否则 `trashFile` 并报错。
- `isTextType` 两级判断:先 `MimeDetector` 查 MIME(`badMimes` 显式拉黑 xlsx,`setOverrides` 把 `.ts`/`.py` 等源码强制改成 `text/plain`),MIME 不确定时再读前 1KB 字节,按 NUL 与控制字符占比是否低于 10% 判定。
- 所有转换器产出同一套 `document` 结构:`id`(uuid v4)、`url`(`file://` 前缀)、`pageContent`(全文)、`token_count_estimate`(`js-tiktoken` 的 `cl100k_base` 估算)等,经 `writeToServerDocuments` 落盘并补上 `location` 与 `isDirectUpload`。
- `trashFile` 一律包在 `if (!options.absolutePath)` 里:`hotdir` 进来的文件用完即焚,用户自带 `absolutePath` 的文件绝不删除。
- PDF 走"先数字解析、`docs.length === 0` 时再 OCR"的降级路径;`OCRLoader.ocrPDF` 用 `tesseract.js` 的 worker 池(`Math.min(os.cpus().length, 4)` 个)分批并发识别,并用 `Promise.race` 套 5 分钟(`300_000` ms)硬超时。
- 音频经 `asAudio` 用可插拔的 `WHISPER_PROVIDERS`(`openai`/`local`,默认 `local`)转写;网页经 `scrapeGenericUrl` 先 `determineContentType`,可能转去当文件或 YouTube 字幕处理,网页则用 Puppeteer 渲染 + `htmlToMarkdown`,失败降级到 `fetch`。
- xlsx 是"一份文件多份文档"的代表:默认模式按每张 sheet 转 CSV 各产一个 `document`,以提升结构化数据的检索精度。
- `extensions(app)` 挂载 `/ext/*` 一系列扩展数据源(GitHub、YouTube、Confluence、Obsidian、Paperless 等),涉及凭据的路由额外挂 `setDataSigner` 以加密敏感信息后落盘。

## 动手实验

1. **实验一:追一份 txt 从进门到落盘** — 打开 `collector/index.js` 的 `/process` 路由,跟着 `processSingleFile`(`collector/processSingleFile/index.js`)读到它 `require(SUPPORTED_FILETYPE_CONVERTERS[processFileAs])`,再跳到 `collector/processSingleFile/convert/asTxt.js`,对照 `data` 对象逐字段确认最终写进 JSON 的结构;然后在 `collector/utils/files/index.js` 的 `writeToServerDocuments` 里确认 `location` 和 `isDirectUpload` 是如何被追加上去的。

2. **实验二:验证文本嗅探的 10% 阈值** — 阅读 `collector/utils/files/index.js` 里的 `parseableAsText`,找到 `const threshold = bytesRead * 0.1;` 这一行,把 0.1 改成一个极小值(如 0.001)或极大值(如 0.9),推演:一个含少量控制字符的日志文件在两种阈值下分别会被判成文本还是二进制?再回到 `processSingleFile`,确认被判成二进制后会走 `trashFile` 还是被当 `.txt` 处理。

3. **实验三:观察扩展名映射表** — 打开 `collector/utils/constants.js`,对照 `ACCEPTED_MIMES` 和 `SUPPORTED_FILETYPE_CONVERTERS` 两张表,数一数有多少种音频扩展名都指向 `asAudio.js`;然后尝试在 `SUPPORTED_FILETYPE_CONVERTERS` 里给一个新扩展名(比如 `.text`)加一行映射到 `"./convert/asTxt.js"`,推演加完之后上传一个 `.text` 文件会走哪条分支(命中映射 vs 走 `isTextType` 兜底)。

4. **实验四:剖析 OCR 的超时与并发** — 阅读 `collector/utils/OCRLoader/index.js` 的 `ocrPDF`,定位 `maxExecutionTime = 300_000`、`batchSize = 10` 和 `NUM_WORKERS = maxWorkers ?? Math.min(os.cpus().length, 4)` 三个参数,画出"分批 → 批内 worker 抢 `pageQueue` → 按页码 sort → `Promise.race` 套超时 → `finally` 里 terminate worker"的执行时序图,并解释为什么把结果 `sort` 放在推入 `documents` 之前是必要的。

> **下一章预告**:文件进了门、变成了带 `pageContent` 的 `document`,但一整篇文本是没法直接喂给 embedding 模型的。第 3 章我们将解构 `server` 侧的文本切分器 `TextSplitter`,看 AnythingLLM 如何按 chunk 大小与重叠把长文本切成恰到好处的片段,为向量化做好最后的准备。
