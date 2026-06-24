# 第 4 章 文档解析 Parser

上一章我们看到 Chunker 把一段文本切成可检索的片段。但 Chunker 拿到的"文本"从哪里来？现实世界的文档是 `.docx`、`.pdf`、扫描件、Markdown、`.textpack` 压缩包……要先把这些异构格式变成"带结构的纯文本",切分和检索才有得做。这一步就是解析层(Parser)的职责。

LightRAG 没有把解析写成一长串 `if suffix == ".docx": ... elif suffix == ".pdf": ...` 的判断,而是把它做成一个**可路由、可插拔**的子系统:`registry.py` 是引擎注册表(谁能解析什么),`routing.py` 是分发决策(这个文件该交给谁),`base.py` 是所有引擎都要遵守的统一契约,`native_*` 是开箱即用的本地解析,`external/` 是接外部重型后端(mineru/docling)的适配器,`plugins.py` 则是第三方扩展点。本章逐层拆开看它"做了什么"和"为什么这样设计"。

## 4.1 统一契约:`BaseParser` 与 `ParseResult`

所有引擎——本地 docx、markdown,远程 mineru、docling,以及内部的 `reuse`/`passthrough` 处理器——都实现 `base.py` 里的抽象基类 `BaseParser`。它的契约极简:一个类属性 `engine_name`,一个抽象协程 `async def parse(self, ctx: ParseContext) -> ParseResult`。这就是为什么管线层可以"对着接口编程",不必关心背后是本地库还是 HTTP 调用。

`base.py` 的设计注释点出一条关键原则:**`BaseParser` 只携带行为(behaviour),不携带能力元数据(capability metadata)**。引擎支持哪些后缀、属于哪个并发队列、需要什么端点配置,这些"能力查询"全部放进 `registry.py` 的轻量 `ParserSpec` 表里。好处是:做能力查询(比如"上传白名单允许哪些后缀")时不需要 import 任何重型解析实现,也就不会顺带把 `httpx`、`docling` 这些大依赖拉进来。

`parse()` 的入参 `ParseContext` 是个 dataclass,封装了 `rag`(LightRAG 实例)、`doc_id`、`file_path`、`content_data`,这样解析器拿到的不是一个裸 `self`。它的两个方法值得注意:`source_path()` 通过懒加载 `lightrag.pipeline` 的函数解析出磁盘源文件路径;`resolve()` 返回 `ResolvedSource(source_path, document_name, parsed_dir)`——这是每个引擎解析前共享的"前奏":规范化文档名(`normalize_document_file_path`),并推导出 `__parsed__/<base>.parsed/` 输出目录。把这段公共逻辑抽到 `ParseContext` 上,而不是每个引擎各写一遍,是减少分歧的典型做法。注意这些 import 都写在方法体内部(call-time import),目的是保持 `base.py` 模块本身 import 廉价、无循环依赖。

输出端 `ParseResult` 同样克制。它的 `to_dict()` **只发出有语义的字段**——`parse_engine` 为 `None` 不发,`parse_stage_skipped` 为 `False` 不发,`parse_warnings` 为空不发。注释解释了原因:重构前各引擎的 `parse_native`/`parse_mineru`/`parse_docling` 返回的就是这种"按需出现键"的 dict,worker 用 `.get(...)` 读取;为了让重构后的返回字节级兼容,`to_dict()` 必须保持同样的形状,绝不凭空塞 `None`/`False` 键。这是"重构不改变可观察行为"的工程纪律。

## 4.2 注册表 `registry.py`:能力即数据

`registry.py` 仿照存储层 `STORAGES` 的约定,在模块顶层维护一张字面量 `_REGISTRY: dict[str, ParserSpec]`。`ParserSpec` 是 `frozen=True` 的 dataclass,字段包括:

- `engine_name`:引擎标识/注册键。
- `impl`:`"module:Class"` 字符串,**由 `get_parser` 懒加载**——只有真正解析某文档时才 `importlib.import_module`,把重依赖的导入推迟到最后一刻。
- `suffixes`:`frozenset` 的后缀集合,是后缀能力的单一事实源(取代了旧的 `constants.PARSER_ENGINE_SUFFIX_CAPABILITIES`)。
- `user_selectable`:是否对用户开放选择(内部的 `reuse`/`passthrough` 为 `False`)。
- `queue_group` 与 `concurrency`:解析并发队列与其大小。
- `endpoint_configured` / `endpoint_requirement`:两个闭包,纯读环境变量、不发网络,用来判断外部引擎是否已配置好端点。

内置注册了六个引擎,它们清晰地划出了"本地 / 外部 / 内部"三档:

- `native`(`impl="lightrag.parser.native_dispatch:NativeParser"`):支持 `docx`/`md`/`textpack`,本地引擎。
- `legacy`:支持一长串后缀(`txt`/`pdf`/`docx`/`html`/`json`/`csv` 及大量源码扩展名),与 native 共用本地池(`queue_group=native`,因为它本地、不走网络)。
- `mineru`:支持 PDF 与 Office 文档及多种图片格式,`queue_group=mineru`,端点由 `_mineru_endpoint_configured` 闭包按 `MINERU_API_MODE`(`official`/`local`)分别检查 `MINERU_API_TOKEN` 或 `MINERU_LOCAL_ENDPOINT`。
- `docling`:支持 PDF/Office/Markdown/HTML/图片,端点由 `DOCLING_ENDPOINT` 环境变量决定。
- `reuse` 与 `passthrough`:`noop` 内部处理器,后缀集为空、`user_selectable=False`,不可被用户选择。

`get_parser(engine)` 是唯一会触发实现导入的地方:它从 `_table` 取出 spec,按 `(engine, spec.impl)` 二元组做实例缓存(键里带 `impl` 是为了:重新注册到不同实现时不会拿到陈旧的缓存实例),缓存未命中才 `importlib` 加载类并实例化。

注册表还导出几个能力查询函数,串起整个上层:

- `supported_parser_engines()`:返回所有 `user_selectable` 的引擎名。
- `available_engine_suffixes()`:**真正可用引擎**(`user_selectable` 且 `endpoint_configured()` 通过)的后缀并集,是 API 上传白名单和输入目录扫描(`DocumentManager.supported_extensions`)的单一来源。
- `engine_endpoint_configured` / `engine_endpoint_requirement`:把 spec 上的闭包暴露给路由层做门禁判断。

`available_engine_suffixes()` 的设计很有意思:默认部署(没配任何外部端点)时,它等于本地引擎(legacy ∪ native)的后缀;一旦配置了 `MINERU_LOCAL_ENDPOINT`,mineru 的图片/Office 后缀就自动加入;第三方引擎的后缀也会在通过自身端点门禁后自动并入。也就是说,**"系统能吃哪些文件"是随配置动态生长的,而不是写死在某个常量里**。

`parser_specs_snapshot()` 返回一份浅拷贝快照。注释说明了用途:管线在一个批次开始时取一次快照,贯穿队列构造、路由与解析 worker——这样即便有并发的 `register_parser` 调用,也不会在批次中途改变引擎集合,保证批次内的一致性。

## 4.3 路由 `routing.py`:这个文件交给谁解析

`routing.py` 是本章最核心也最庞大的模块(约 1234 行),负责回答"给定一个 `file_path`,该用哪个引擎、带什么处理选项和参数"。它的对外结果是 `ParserDirectives`(`frozen` dataclass):`engine`、`process_options`(纯选择器字符串 `i/t/e/!/F/R/V/P`)、`chunk_params`(切分参数,按选择器字符叠加)、`engine_params`(解析引擎参数,如 mineru 的 `page_range`/`language`,docling 的 `force_ocr`)。

路由有三个输入来源,优先级在 `resolve_parser_directives` 里写得很清楚:

1. **文件名内嵌提示(filename hint)**:形如 `report.[mineru].pdf`,正则 `_PARSER_HINT_RE = re.compile(r"\.\[([^\]]*)\](\.[^.]+)$")` 在文件名末尾匹配 `.[engine].ext`。引擎和/或选项以此为最高优先级。`_filename_hint_match` 负责把括号内的引擎、处理选项、切分参数、引擎参数都解析出来;它是非抛出的"尽力分类器",因为扫描分组、文件名规范化都需要一个 best-effort 分类。真正的摄入入口走 `resolve_parser_directives`,它会校验非法 hint 并**直接抛错**而不是悄悄回退。
2. **`LIGHTRAG_PARSER` 路由规则**:环境变量里的 `<suffix-pattern>:<engine>` 规则列表,用 `fnmatch` 按后缀匹配,第一条匹配且"可用"(`_engine_is_usable`)的规则,补齐文件名 hint 没指定的那部分。
3. **默认引擎**:`_DEFAULT_ENGINE_BY_SUFFIX` 给 `textpack` 指定 native(因为只有 native 处理它,无需 hint 或规则);其余后缀一律回落到 `PARSER_ENGINE_LEGACY`。注释特意说明 `.md` **故意不在**默认表里——它保留 legacy 默认,要走 native 得像 `.docx` 一样显式用 hint 或规则,体现了"行为变更必须显式触发"的保守姿态。

最终引擎的合成是一行短路:

```python
engine = hinted_engine or rule_engine or default_engine or PARSER_ENGINE_LEGACY
```

这一行把三级优先级压成了一个表达式:文件名 hint 最高,其次 `LIGHTRAG_PARSER` 规则,再次按后缀的默认引擎,最后兜底 legacy。短路求值天然实现了"前者命中就不看后者"的优先级语义,既好读又难写错。

`_engine_is_usable(engine, suffix, *, require_external_endpoint)` 是路由的"门禁三连",三个条件全过才算可用:

- 引擎必须在 `supported_parser_engines()` 里(用户可选)。
- 必须 `parser_engine_supports_suffix`(后缀在该引擎能力集中)。
- 若要求外部端点,则必须 `parser_engine_endpoint_configured`(端点已配)。

当文件名 hint 指定的引擎对该文件不可用(后缀不对或端点缺失),`resolve_parser_directives` 会把 `hinted_engine` 置 `None` 回退到规则解析,但**保留 hint 里的处理选项**——降级而不丢配置。

参数合并的细节也很讲究:

- `chunk_params` 按选择器字符逐个叠加,规则参数在前、文件名 hint 在后(同键 hint 胜出)。
- `engine_params` 是扁平的(每个文件只有一个最终引擎),只保留"归属引擎 == 最终引擎"的那部分参数,归属于落选引擎的参数会被丢弃。这避免了把 mineru 的参数误喂给 docling。

路由还提供 `validate_parser_routing_config`,在**启动时**集中校验 `LIGHTRAG_PARSER` 语法:模式不能为空、不能带点(用 `pdf` 不是 `*.pdf`)、引擎必须受支持、规则后缀必须落在引擎能力内、外部引擎必须配好端点、引擎参数与处理选项都要合法。把校验前移到启动期,意味着配置错误会立刻报错,而不是等到某个文档解析时才神秘失败。

最后,`resolve_stored_document_parser_engine` 处理"已存储文档行"的恢复/重试场景:`lightrag` 格式回 `reuse`(复用已解析的 sidecar),非 `pending_parse` 的已抽取内容回 `passthrough`(逐字透传),`pending_parse` 则尊重存储的引擎、缺失才回落文件名规则。它"从不抛错、从不过滤未知引擎",把回退与校验交给 parse worker,好让 worker 的告警路径真正能触发。

## 4.4 处理选项 `ProcessOptions`:解析之外的旋钮

路由解析出的 `process_options` 不是空字符串那么简单,它是一串单字符选择器(`i/t/e/!/F/R/V/P`),`routing.py` 用 `ProcessOptions` dataclass 把它解码成布尔旗标:`images`(`i`,处理图片)、`tables`(`t`)、`equations`(`e`)、`skip_kg`(`!`,跳过知识图谱抽取),外加一个 `chunking` 切分策略枚举。

切分策略选择器 `F/R/V/P` 通过 `_CHUNK_STRATEGY_KEYS` 映射到四种策略:`F`→`fixed_token`(定长 token)、`R`→`recursive_character`(递归字符)、`V`→`semantic_vector`(语义向量)、`P`→`paragraph_semantic`(段落语义)。这正是上一章 Chunker 读取的策略;**解析层只负责"选策略",真正的切分参数由 `chunk_options` 通道携带**,二者职责分离。

这里有几处工程细节值得留意。`ProcessOptions.raw` 逐字保留原始字符串(含重复和顺序)供存储/审计,而布尔旗标反映去重后的逻辑状态。`chunking_explicit` 属性专门回答"用户是否显式选了切分策略"——因为 `chunking` 字段在"显式选 F"和"没选回落默认 F"两种情况下都等于 `PROCESS_OPTION_CHUNK_FIXED`,无法区分,只能数 `raw` 里是否真有切分选择器字符。`validate_process_options` 会拒绝多个切分模式同时出现(`F` 和 `R` 不能并存),保证策略唯一。`sanitize_process_options` 则静默丢弃不支持的字符,但"不去重、不重排",把用户的规范意图原样落盘。这种"原始留存 + 逻辑去重"双轨制,是审计需求与运行需求的平衡。



LightRAG 把解析后端分成两类,体现了一个清晰的取舍。下表对比两类后端的关键差异:

| 维度 | 本地 native/legacy | 外部 mineru/docling |
| --- | --- | --- |
| 依赖 | 仅本地 Python 库 | 需部署/配置远程端点 |
| 网络 | 不走网络 | 走 HTTP |
| 并发池 | 与 native 共享本地池 | 独立队列(`max_parallel_parse_mineru` / `max_parallel_parse_docling`) |
| 质量 | 受本地库限制(无 OCR) | 版面分析、表格还原、公式识别、OCR |
| 可用条件 | 始终可用 | 端点配好才进 `available_engine_suffixes` 白名单 |

**本地 native/legacy** 零外部依赖、零网络、与 native 共享并发池,开箱即用。但解析质量受限于本地库——比如 docx 靠 `python-docx` 读 OOXML,无法处理扫描 PDF 的 OCR。

**外部 mineru/docling** 解析质量高,代价是要部署/配置端点、走 HTTP、有独立并发队列,且只在端点配置好后才进入 `available_engine_suffixes` 的白名单。MinerU 是开源的高精度 PDF 解析与文档抽取工具 [[MinerU]](https://github.com/opendatalab/MinerU)。Docling 则是 IBM 开源的文档转换库,擅长把 PDF/Office 转成结构化表示 [[Docling]](https://github.com/docling-project/docling)。

`external/mineru/client.py` 的 `MinerURawClient` 展示了外部后端的典型形态。它在构造时一次性读 `MINERU_*` 环境变量:

- `api_mode`(`official`/`local`)、端点、`api_token`。
- 轮询间隔 `MINERU_POLL_INTERVAL_SECONDS`、最大轮询次数 `MINERU_MAX_POLLS`(注释算出 600×2s≈20 分钟,大 PDF 可调高)。
- 一组解析选项 `enable_table`/`enable_formula`/`is_ocr`/`page_ranges`/`local_parse_method` 等(由 `MinerUParserOptions.from_env` 装配)。

`download_into()` 的流程是:上传文件 → 轮询任务 → 下载 zip bundle → `_normalize_raw_bundle` 整理 → `_build_and_write_manifest` 写清单。它特意捕获 `httpx.RequestError` 并重新包装成带"引擎名 + 端点 + 异常类名"的 `RuntimeError`,因为底层传输错误的原始消息(如 "All connection attempts failed")完全看不出是 MinerU 在出问题——这样 `doc_status` 的 `error_msg` 永远非空且可归因。这是外部后端必须直面的"远程调用可观测性"问题。

## 4.5 native 解析如何保留结构:docx 与 markdown

本地解析的核心模板在 `native_base.py` 的 `NativeParserBase.parse`,它把本地解析流程固定了一次,子类不必各写一遍:

1. 解析 + 校验源(`ctx.resolve` + `validate_source`)。
2. 算出 `parsed_dir` 与 `asset_dir`。
3. 预清理:`rmtree(parsed_dir)` + `mkdir` + 建 `asset_dir`,任何失败都回滚(`rmtree(..., ignore_errors=True)`)。
4. 在线程里跑 `extract()`(`asyncio.to_thread`,因为本地库是同步阻塞的)。
5. `build_ir()` 把 blocks 规范成 `IRDoc`。
6. `write_sidecar(clean_parsed_dir=False)` 落盘(asset 目录已预填,不能再清)。
7. `_persist_parsed_full_docs` 写 full_docs,`archive_source` 归档源文件。

子类只需实现 `extract`(产出 `(blocks, warnings, metadata)`)与 `build_ir` 两个钩子。如果 `extract` 返回空 blocks,模板会清理目录并抛 `ValueError`——空内容被当作失败而非静默成功。`extract` 之所以跑在线程里,是因为它要做磁盘 IO 和 CPU 密集的 XML 解析,放进线程池才不会阻塞事件循环。

`native_dispatch.py` 的 `NativeParser` 是 native 引擎的入口,它**不**子类化 `NativeParserBase`,而是按源文件后缀把整个 `parse(ctx)` 委派给具体解析器:`.md`/`.textpack` 给 `NativeMarkdownParser`,其余(默认)给 `NativeDocxParser`。注释解释了为什么这样设计:每个具体解析器要保留自己的 `sidecar_path_style`——docx 用 `basename_only`(为了字节级一致的 golden 输出),markdown 用 `with_prefix`。如果用基类只转发 `extract`/`build_ir` 钩子,这个差异就丢了。

**docx 结构提取**(`docx/parse_document.py`)的精髓是**标题层级 + heading path**。它做了两件别的库常忽略的事:

- **沿样式继承链解析标题级别**:`parse_styles_outline_levels` 直接打开 docx 的 zip 读 `word/styles.xml`,提取每个样式的 `outlineLvl`,并**沿 `basedOn` 继承链递归解析**(带 `visited` 集防环)——因为很多文档的标题级别不是直接标在段落上,而是继承自样式。`get_heading_level` 的优先级是"段落直接格式 > 样式定义",`outlineLvl` 0–8 才算真标题,9 是正文。
- **维护标题栈逐块累积**:`extract_docx_blocks` 遍历 `doc._element.body`,维护 `current_heading_stack`(`{level: heading_text}` 的 dict),每遇到一个标题就 `_flush_current_block` 把前面的段落收成**一个 heading-scoped 块**,块里带 `heading`、`parent_headings`、`level`。这意味着每个块都知道自己在文档大纲里的位置(heading path),供后续切分与检索使用。

它还处理了不少真实世界的脏数据:超长"标题"(作者设了大纲级别却用软换行 `<w:br/>` 把正文写进同一段)会在第一个软换行处切分,首行留作标题、其余降级为正文,并记 `heading_softbreak_split_count` 告警;实在切不开就整段降为正文。表格被转成 JSON 二维数组并以 `<table>` 占位符发出,`_collect_table_headers` 还收集跨页重复表头(`w:tblHeader`)。值得强调的是:native parser **只做标题驱动的结构切分**,块大小控制(长块锚点切分、表格行切分、小块合并)被刻意留给下游 P-chunker——`_flush_current_block` 的注释明说了这条职责边界。

把 `extract_docx_blocks` 这个函数包装成符合 `NativeParserBase` 契约的引擎,是 `docx/parser.py` 的 `NativeDocxParser`。它实现 `extract` 钩子:先构造 `DrawingExtractionContext`(带 `docx_path`、`blocks_output_path`、图片导出目录),`load_relationships` 加载 docx 内部的关系映射(用来定位嵌入图片),再调 `extract_docx_blocks` 拿到 `(blocks, warnings, metadata)`。`build_ir` 把 blocks 交给 `NativeDocxIRBuilder().normalize` 归一成 `IRDoc`。它的类属性 `sidecar_path_style = "basename_only"` 和 `empty_content_label = "DOCX"` 正是 4.5 节开头提到的"每个具体解析器保留自己的 path style"的落地。

`surface_warnings` 钩子展示了"告警一次、降级不崩"的处理:当文档里有段落缺 `w14:paraId`(常见于老版本 Word 存的文件),它累加 `missing_paraid_count`,每文档记一条日志(提示"用 Word 2013+ 重存以重新生成 id"),受影响的块会发出 `positions: [{"type": "paraid", "range": null}]`。解析器不会因为缺 id 而失败——这是面向真实文档"尽量产出可用结果"的务实态度。


**markdown 结构提取**(`markdown/parser.py`)的 `NativeMarkdownParser` 继承 `NativeParserBase`。它的 `extract` 支持两种输入:

- 裸 `.md`:`utf-8-sig` 解码(容忍 BOM)。
- `.textpack` 压缩包:用 `safe_extract_zip` 安全解压(限制条目数与总字节防 zip 炸弹),`_find_text_file` 按扩展名而非固定文件名定位正文,支持 `.textbundle` 目录布局。

实际解析委托给 `extract_markdown`,产出 blocks 以及 `md_tables`/`md_equations`/`md_drawings`/`md_assets` 元数据,再由 `NativeMarkdownIRBuilder` 归一成 `IRDoc`。

这里还有一处工程亮点:markdown 里的图片可能是远程 URL,`_MarkdownImageResolver` 配合一组 `_Guarded*` 的 HTTP handler 做了 **SSRF 防护**——把 socket 钉在校验过的公网地址、拒绝内网(`_host_is_public` / `_pin_socket`),这是"解析用户文档"必须警惕的安全面。下载的图片缓存在 `<file>.native_raw/` 兄弟目录,能熬过 `parsed_dir` 的 `rmtree`,在重解析时复用——这个缓存的签名(`options_signature`)还会在下载配置或源内容变化时失效。

## 4.6 解析产出的统一中间表示 IR

无论本地还是外部引擎,解析最终都收敛到同一种中间表示 `IRDoc`(`build_ir` 的返回类型),再由 `write_sidecar` 落盘。这是整个解析层最重要的"统一点":前端格式千差万别,后端切分与检索只面对一种结构。

`external/docling/ir_builder.py` 的 `DoclingIRBuilder._normalize` 把 Docling 的 JSON 遍历成与 MinerU 完全相同的累积器结构:

- 标题栈 `heading_stack` 跟踪当前层级。
- 块累积器 `cb_lines`/`cb_tables`/`cb_drawings`/`cb_equations`/`cb_parents` 暂存当前块的内容。
- `_flush_block` 在遇到新标题时收成一个 `IRBlock(content_template, heading, level, parent_headings, positions, tables, drawings, equations)`。

注释明说这是"和 `MinerUIRBuilder` 一模一样的结构,好让下游 P-chunking 和溯源(provenance)无论引擎如何都表现一致"。换句话说,无论文档是被 MinerU 还是 Docling 解析,下游拿到的块形状完全相同。

值得一提的是,`DoclingIRBuilder` 在构造时读 `DOCLING_ENGINE_VERSION` 与 `DOCLING_BBOX_ATTRIBUTES`(后者用 `env_json` 解析,解析失败回退到默认 `{"origin": "LEFTBOTTOM"}` 并告警)。这种"环境变量驱动、坏配置不崩"的小处理,在解析层里反复出现,是它面对真实部署环境的一贯姿态。

heading 级别的归一也统一了语义。`_docling_heading_level` 的映射规则是:

- `title` → level 1。
- `section_header` → `item.level + 1`(无法解析时回退 2)。
- 其余 → 0(非标题)。

这与 docx 把 `outlineLvl` 0-based 转 1-based、markdown 数 `#` 的逻辑殊途同归——**不同引擎的"标题"概念被压平到同一套 level 体系**,这正是"解析产出携带结构元数据供切分与检索使用"得以成立的前提。`IRBlock` 里还带 `positions`(bbox 锚点/页码),这些版面坐标是外部引擎相对本地引擎的额外价值,也一路传到下游。

## 4.7 插件机制 `plugins.py`:第三方引擎扩展点

`plugins.py` 提供了不改 LightRAG 源码就接入新解析引擎的能力。它基于 Python 的 entry points 机制:第三方包在 `pyproject.toml` 里声明 `lightrag.parsers` 组的入口点,指向一个**零参数可调用对象**,该对象自己调用 `register_parser` 把 `ParserSpec` 注册进表。`load_third_party_parsers(*, force=False)` 遍历 `entry_points(group="lightrag.parsers")` 逐个 `ep.load()` 并执行,进程内幂等(`_loaded` 标志,`force=True` 可重跑供测试)。

两个设计细节呼应了前面的原则:其一,插件作者应让入口点 **import 廉价**,把真正的实现导入推迟到 `ParserSpec.impl` 字符串(注册表本就懒加载),这与 4.2 节"能力即数据、实现懒加载"一脉相承。其二,**一个坏插件不能拖垮启动**——`load_third_party_parsers` 对每个插件 `try/except`,失败就连同来源(`ep.value`)记日志并跳过;内置引擎是静态注册的,永不受影响。它由 API server(`create_app`,在路由规则校验之前调用,好让 `LIGHTRAG_PARSER` 能引用第三方引擎名)和调试 CLI 各调一次。

## 本章小结

- LightRAG 把"原始文件 → 可切分文本"做成可路由、可插拔的解析层,用注册表分发取代了 `if engine == …` 链。
- `base.py` 定下统一契约:`BaseParser.parse(ctx) -> ParseResult`,行为与能力元数据分离——能力放进 `registry.py` 的 `ParserSpec`,使能力查询不触发重依赖导入。
- `ParseResult.to_dict()` 只发出有语义的字段,刻意保持与重构前各引擎返回 dict 的字节级兼容。
- `registry.py` 以模块级字面量表登记六个引擎,`impl` 字符串懒加载;`available_engine_suffixes()` 让"系统能吃哪些文件"随端点配置动态生长。
- `routing.py` 按"文件名 hint → `LIGHTRAG_PARSER` 规则 → 默认引擎 → legacy"四级优先级解析 `ParserDirectives`,门禁三连(受支持/后缀匹配/端点已配)决定引擎是否可用。
- 路由配置在启动期由 `validate_parser_routing_config` 集中校验,错误立即暴露而非延迟到解析时。
- 本地 native/legacy 开箱即用但质量受限;外部 mineru/docling 质量高但需端点、走 HTTP、有独立并发队列——这是开箱即用 vs 高质量的明确取舍。
- docx 解析读 `styles.xml` 的 `outlineLvl` 并沿 `basedOn` 继承链解析标题级别,按标题切成 heading-scoped 块并携带 `parent_headings`,表格转 JSON 占位符。
- native parser 只做标题驱动的结构切分,块大小控制刻意留给下游 P-chunker;`process_options` 用 `i/t/e/!/F/R/V/P` 选择器解耦"开关与策略选择",其中 `F/R/V/P` 经 `_CHUNK_STRATEGY_KEYS` 映射到四种切分策略,参数走独立的 `chunk_options` 通道。
- 所有引擎收敛到统一的 `IRDoc`/`IRBlock`(content/heading/level/parent_headings/positions),不同引擎的"标题"概念被压平到同一套 level 体系,供下游切分与检索一致使用。
- `plugins.py` 用 `lightrag.parsers` entry points 支持第三方引擎,入口点 import 廉价、坏插件被隔离不影响启动与内置引擎。

## 动手实验

1. **实验一:观察路由决策** — 在 `lightrag/parser/routing.py` 所在环境里,用 Python 交互式调用 `resolve_parser_directives("a.docx")`、`resolve_parser_directives("a.[native].docx")`、`resolve_parser_directives("a.[mineru].pdf", require_external_endpoint=False)`,对比返回的 `ParserDirectives.engine`,验证"文件名 hint 优先、默认回落 legacy"的优先级。
2. **实验二:能力随配置生长** — 先调 `available_engine_suffixes()` 记下默认后缀集;再 `os.environ["MINERU_LOCAL_ENDPOINT"]="http://localhost:8000"` 后重新调用,观察 `png`/`pdf` 等 mineru 后缀是否并入,理解上传白名单的动态来源。
3. **实验三:docx 标题层级** — 造一个含一级/二级标题的 `.docx`,调用 `parse_styles_outline_levels(path)` 看 styleId→outlineLvl 映射,再调 `extract_docx_blocks(path)` 打印每个块的 `heading`/`level`/`parent_headings`,确认 heading path 被正确记录。
4. **实验四:注册一个假引擎** — 写一个零参数 `register()` 函数,内部用 `ParserSpec(engine_name="myengine", impl="...", suffixes=frozenset({"foo"}))` 调 `register_parser`,然后调 `supported_parser_engines()` 与 `available_engine_suffixes()`,验证新引擎与其后缀是否自动出现在可用集合里。

> **下一章预告**:文档被解析成带 heading path 的结构化块、再切成 chunk 后,LightRAG 要从中抽取出实体与关系来构建知识图谱。第 5 章我们深入实体关系抽取——看它如何用 LLM 提示词把纯文本变成 `(实体, 关系, 实体)` 三元组,如何去重合并、如何把抽取结果写入图存储与向量库。
