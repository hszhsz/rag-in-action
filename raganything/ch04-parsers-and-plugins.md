# 第 4 章 MinerU/Docling 解析器与插件注册

第 3 章把 `Parser` 基类拆开后，我们看到的是一份"契约"：所有解析器都要把异构文档吐成统一的 content_list（`type` + `text`/`img_path`/`table_body` + `page_idx` 的字典列表），Office/文本格式则统一靠基类的 `convert_office_to_pdf`/`convert_text_to_pdf` 先降维成 PDF。本章把镜头推近到契约的两个真实实现：`MineruParser` 与 `DoclingParser`。它们对同一份 PDF 会产出语义等价的 content_list，但走的是两条截然不同的集成路线——一个 `subprocess` 调外部 CLI、一个进程内调 Python API。读完这两套实现的细节，再看文件尾部那套 `register_parser`/`get_parser` 注册机制，你就能理解 RAG-Anything 是如何用"契约层 + 注册表"做到第三方解析器即插即用的。

源码全部位于 `/home/mira/files/raganything_src/raganything/parser.py`（单文件 2749 行）。

## 两种集成路线：CLI 子进程 vs 进程内 Python API

`MineruParser` 与 `DoclingParser` 最根本的差异，是它们如何驱动底层模型。

`MineruParser` 走的是**子进程路线**。它的核心方法 `_run_mineru_command` 拼出一个 `["mineru", "-p", input, "-o", output, "-m", method]` 命令行数组，再用 `subprocess.Popen` 启动外部的 `mineru` 可执行文件 [[MinerU]](https://github.com/opendatalab/MinerU)。它不在自己的进程里加载任何深度学习模型，而是把整个解析交给 CLI 进程，事后从磁盘读回产物。

`DoclingParser` 走的是**进程内路线**。它的 `_run_docling_python` 直接 `from docling.document_converter import DocumentConverter`，在当前 Python 进程里 `converter.convert(...)`，拿到 `result.document` 后调 `export_to_dict()` 把整篇文档导成内存字典 [[Docling]](https://github.com/docling-project/docling)。类文档字符串明确写道，这是为了"避免每次调用都重新初始化 Docling 的深度学习模型"。

这两条路线各有代价，源码里的注释把取舍说得很透：

- **CLI 隔离**：MinerU 用子进程把模型、依赖、潜在的崩溃都关在另一个进程里，主进程不被 MinerU 的庞大依赖污染；坏处是每次解析都要付一次进程启动 + 模型加载的固定开销，且只能通过磁盘文件交换数据。
- **进程内更快但耦合**：Docling 把 `DocumentConverter` 实例**按配置缓存**（详见下文 `_get_converter`），使得多文档批处理时模型只加载一次；代价是 `docling` 包必须可 import，且其依赖直接进入主进程。

`check_installation` 的实现差异正是这种取舍的镜像：`MineruParser.check_installation`（1512 行）跑 `subprocess.run(["mineru", "--version"])` 探测 CLI 是否在 PATH 上；`DoclingParser.check_installation`（2112 行）则只 `from docling.document_converter import DocumentConverter` 试 import。类文档明确指出这是相对早期 CLI 实现的**行为变更**：只装了 docling CLI 而没装 Python 包的环境，现在会得到不同结果——但因为 `parse_*` 走的就是 Python API，这个 import 检查反而是更忠实的预检。

## MinerU 的子进程治理：实时流式日志与超时

`_run_mineru_command`（788 行）不是简单地 `subprocess.run` 等结果，而是做了相当工程化的进程治理，因为 MinerU 可能跑很久（首次会下载模型）。

它用 `subprocess.Popen` 拿到管道后，开两个 `daemon` 线程把 stdout/stderr 通过 `Queue` 异步抽出来，主线程在 `while process.poll() is None` 循环里实时消费。这样既能实时看到 MinerU 的进度，又不会因为管道缓冲区写满而死锁。循环之外还有一整套失败治理。归纳其关键动作：

- **流式日志分级**：stdout 行以 `[MinerU]` 前缀打 INFO 日志；stderr 行按内容含 `"warning"`/`"error"` 分级打日志，并把 error 行收集进 `error_lines`。
- **超时熔断**：若 `timeout` 非空且 `time.monotonic() - start_time` 超限，就 `process.kill()` + `process.wait()` 并抛 `TimeoutError`，错误信息直指"常见于模型下载因网络卡住"。
- **退出码校验**：进程正常结束后，若 `return_code != 0` 或 `error_lines` 非空，则抛自定义的 `MineruExecutionError`（携带 `return_code` 与 `error_msg`）。
- **缺失友好化**：`FileNotFoundError` 被翻译成"mineru command not found，请 `pip install -U 'mineru[core]'`"。
- **平台适配**：Windows 下设 `creationflags = subprocess.CREATE_NO_WINDOW` 隐藏控制台窗口。

这些都是把"调外部进程"做扎实所必须的防御。

## Windows 不安全路径的 hash 规避

MinerU 既然要把路径作为命令行参数传给外部进程，跨平台 robustness 就成了真问题。Windows 下含非 ASCII 字符、或路径片段以空格/句点结尾的路径，会让外部工具或文件系统行为异常。`MineruParser` 为此专门做了一层防御。

`_is_mineru_unsafe_windows_path`（714 行）先判断路径是否不安全：

- 非 Windows（`_IS_WINDOWS` 为 `False`）直接返回 `False`，整层逻辑在 Linux/macOS 上零开销；
- 在 Windows 上，检测路径能否 `encode("ascii")`——不能就是不安全（含中文等非 ASCII）；
- 同时检测任一路径片段或文件名主干（`path.stem`）是否以 `" "` 或 `"."` 结尾。

一旦检测到不安全，`_prepare_mineru_paths`（734 行）就启动规避：

- 用 `_mineru_safe_path_hash`（730 行）对原路径 `resolve()` 后取 MD5 前 10 位生成短 hash；
- `tempfile.mkdtemp(prefix="raganything_mineru_")` 建临时目录；
- 不安全的输入文件被 `shutil.copy2` 复制成 `input_<hash><suffix>`；
- 不安全的输出目录被替换为 `mineru_<hash>` 临时目录。

MinerU 全程只看到这些纯 ASCII 安全短路径。

解析完成后，`parse_pdf` 在 `try` 块里调 `_copy_mineru_output_tree`（766 行）把临时输出目录整棵树拷回用户真正指定的 `base_output_dir`，再在 `finally` 里调 `_cleanup_mineru_temp_dir`（782 行，`shutil.rmtree(..., ignore_errors=True)`）清掉临时目录。值得注意的是，`parse_image` 调 `_prepare_mineru_paths` 时额外传了 `hash_path=image_path`：因为图片可能已被转成临时 PNG，hash 要基于原始路径算才稳定。这层"检测→hash 短路径→拷回→清理"的设计，让同一份代码在中文路径、带空格目录的 Windows 机器上也能可靠跑通。

## `_read_output_files`：把磁盘产物读回统一 content_list

子进程路线的代价是数据只能落盘。`_read_output_files`（1034 行）负责把 MinerU 写到磁盘的产物收编回内存 content_list，这是 MinerU 兑现"契约"的关键一步。

它先按约定找 `<file_stem>.md` 和 `<file_stem>_content_list.json`。但 MinerU 2.7+ 会按 backend 把产物放进不同子目录（`pipeline→auto/`、`vlm-*→vlm/`、`hybrid-*→hybrid_auto/`），所以方法**不硬编码目录名**，而是遍历 `<output_dir>/<file_stem>/` 下的子目录、找到含 `_content_list.json` 的那个为准，扫描失败才回退到基于 `method` 的路径。这种"扫描优先、约定回退"的写法对上游目录结构变化更健壮。

读到 JSON 后做两类归一：其一是**字段别名兼容**，MinerU 2.0 把 `img_caption`/`img_footnote` 改名为 `image_caption`/`image_footnote`，代码用 `_FIELD_ALIASES` 双向补齐，让下游无论拿到哪个名字都能工作；其二是**图片路径绝对化**，把 `img_path`/`table_img_path`/`equation_img_path` 这些相对路径基于 `images_base_dir` 解析成绝对路径，并做**路径穿越安全检查**——用 `is_relative_to(resolved_base)` 确认图片确实落在产物目录内，否则清空该字段并告警。最后每个 item 还过一遍 `attach_public_media_urls` 挂上可公开访问的媒体 URL。归一完成，MinerU 的磁盘产物就变成了和 Docling 完全同构的 content_list。

## Docling 的转换器缓存与递归展平

`DoclingParser` 没有磁盘往返的负担，但它要解决的是另一个问题：Docling 的文档对象是一棵带 `$ref` 引用的树，得展平成线性 content_list。

先看缓存。`_get_converter`（1716 行）以 `(table_mode, do_tables, do_ocr, artifacts_path)` 四元组为 key 缓存 `DocumentConverter`。它用了经典的双重检查锁：先在锁外读 `_converter_cache.get(cache_key)`（CPython 里字典读是原子的，命中走无竞争快路径），未命中再 `with self._converter_cache_lock` 重新检查并构建。这保证同一个 `DoclingParser` 实例被多线程共享时，相同配置的 Docling 模型只会加载一次。构建时它还设 `generate_picture_images = True`、`images_scale = 2.0`，让 Docling 把图片字节直接嵌进导出的字典，省掉二次回读源文档。

再看展平。`parse_pdf`/`parse_office_doc`/`parse_html` 拿到 `doc_dict` 后，都从 `doc_dict["body"]` 入口调 `read_from_block_recursive`（1886 行）。这个递归函数沿 `block["children"]` 下钻：叶子节点直接转换；非叶子节点先转换自身（`body`/`groups` 这类纯容器除外），再遍历子节点。子节点用 `$ref` 字段指向同级集合，形式是 `#/<type>/<index>`（如 `#/texts/3`），函数把它 split 出 `member_type`、`member_num`，再到 `docling_content[member_type][int(member_num)]` 取出真正的块继续递归。`$ref` 格式不合法或解析失败都只告警跳过，不中断整篇解析。

真正做字典转换的是 `read_from_block`（1935 行），它按 `type` 分流：`texts` 里 `label == "formula"` 的转成 `{"type": "equation", ...}`、其余转成 `{"type": "text", "text": block["orig"]}`；`pictures` 从 `block["image"]["uri"]` 取 base64 data URI、`b64decode` 后写成 `images/image_<num>.png` 并填绝对 `img_path`；其余按 `table` 处理，把 `block["data"]` 填进 `table_body`。注意 `page_idx` 在这里是用 `cnt // 10` 估算的——Docling 的块流没有逐块页号，这是一个务实的近似。MinerU 从磁盘 JSON 读回、Docling 从内存树递归展平，路径完全不同，终点却是同一份 content_list 契约。

## 插件注册：契约层 + 注册表 = 可扩展

文件尾部那组函数，是把"统一契约"变成"开放生态"的最后一块拼图。

注册表本身极简：模块级 `_CUSTOM_PARSERS: Dict[str, type] = {}`（2479 行）一个全局字典，加上常量 `SUPPORTED_PARSERS = ("mineru", "docling", "paddleocr")`（2574 行）列出内建解析器。围绕它有一组对称 API：

- `register_parser(name, parser_class)`（2482 行）：让第三方注册自定义解析器（注释称之为 Bring-Your-Own-Parser）。它先 `_normalize_parser_name` 把名字 strip + lower 归一，强校验 `parser_class` 必须是 `Parser` 子类（否则 `TypeError`），并禁止覆盖内建名 `{"mineru", "docling", "paddleocr"}`（否则 `ValueError`）。文档字符串里给的示例正是注册一个 `MarkerParser`。
- `unregister_parser(name)`（2539 行）/`list_parsers()`（2557 行）/`get_supported_parsers()`（2577 行）：分别做注销、列出全部（内建 + 自定义）名字到类名的映射、返回 `SUPPORTED_PARSERS` 拼上自定义名的完整元组。

工厂函数 `get_parser(parser_type)`（2582 行）是所有调用方拿解析器的唯一入口：把名字归一后，`mineru`/`docling`/`paddleocr` 直接 `new` 对应类，否则查 `_CUSTOM_PARSERS`，再不命中就抛 `ValueError` 并列出全部支持的名字。这意味着只要第三方解析器满足 `Parser` 契约（实现 `parse_document`/`check_installation`），`register_parser` 之后它就能被 `get_parser` 当作一等公民取出——上层 `RAGAnything` 完全无需改动。

这套机制和 `raganything/__init__.py` 的导出是呼应的：包初始化里把 `register_parser`/`unregister_parser`/`list_parsers`/`get_supported_parsers` 重导出，且用 `if "register_parser" in globals()` 做 feature-gated 判断，仅当符号成功导入时才把它们追加进 `__all__`。换言之，注册 API 是"可选能力"，按是否可用动态决定是否对外暴露。这正是"契约层定义了什么是解析器、注册表决定了有哪些解析器"——前者保证可替换，后者保证可扩展。

## 本章小结

- `MineruParser` 与 `DoclingParser` 实现同一 content_list 契约，但前者走 `subprocess` 调外部 `mineru` CLI、后者走进程内的 Docling Python API，是两条根本不同的集成路线。
- CLI 路线（MinerU）以进程隔离换取依赖解耦与崩溃隔离，代价是进程启动开销与磁盘数据交换；Python API 路线（Docling）以 `_get_converter` 按配置缓存换取多文档低延迟，代价是依赖直接进入主进程。
- `_run_mineru_command` 用 `Popen` + 双 `daemon` 线程 + `Queue` 实时流式抽取日志、内置 `timeout` 熔断、失败抛 `MineruExecutionError`，把"调外部进程"做成了可观测可控的工程动作。
- MinerU 的 Windows 安全路径处理由 `_is_mineru_unsafe_windows_path` 检测、`_prepare_mineru_paths` 用 MD5 短 hash 建临时安全路径、`_copy_mineru_output_tree` 拷回、`_cleanup_mineru_temp_dir` 清理，保障中文/带空格路径的跨平台健壮性。
- `_read_output_files` 用"扫描子目录优先、`method` 路径回退"定位产物，做 `img_caption`/`image_caption` 双向别名兼容、图片路径绝对化与路径穿越安全检查，把磁盘 JSON 收编成统一 content_list。
- Docling 的 `read_from_block_recursive` 沿 `children` 与 `$ref`（`#/<type>/<index>`）递归遍历文档树，由 `read_from_block` 把 `texts`/`pictures`/表格分流成 content_list 条目，与 MinerU 殊途同归。
- 插件机制由 `_CUSTOM_PARSERS` 注册表 + `register_parser`/`get_parser` 工厂构成：注册时强校验 `Parser` 子类、禁止覆盖内建名，取用时统一经 `get_parser` 工厂，第三方解析器即插即用。
- `__init__.py` 用 feature-gated 的 `if "register_parser" in globals()` 决定是否把注册 API 加入 `__all__`，体现"契约层保证可替换、注册表保证可扩展"的分层设计。

## 动手实验

1. **实验一：对比两种集成路线的产出** — 准备一份带图片和表格的 PDF，分别用 `get_parser("mineru")` 和 `get_parser("docling")` 调 `parse_document`，打印两份 content_list 的 `type` 分布与长度；观察它们如何收敛到同一契约，并注意 Docling 的 `page_idx` 是 `cnt // 10` 近似而非真实页号。
2. **实验二：触发并观察 MinerU 子进程治理** — 给 `parse_pdf` 传一个很小的 `timeout`（如 `timeout=1`），观察 `_run_mineru_command` 抛出 `TimeoutError` 及其"模型下载卡住"提示；再把 `mineru` 从 PATH 移除，确认 `check_installation` 返回 `False`、解析抛出友好的安装提示。
3. **实验三：复现 Windows 安全路径规避** — 阅读 `_is_mineru_unsafe_windows_path`/`_prepare_mineru_paths`，在非 Windows 上用 Python 手动以一个含中文或以空格结尾的路径调用这两个 classmethod（注意非 Windows 会直接判定安全），再思考若把 `_IS_WINDOWS` 临时置 `True` 会走出怎样的 MD5 短 hash 临时路径分支。
4. **实验四：注册一个自定义解析器** — 按 `register_parser` 文档字符串的示例，写一个最小 `MarkerParser(Parser)`（实现 `check_installation` 与 `parse_document` 返回一条假 content_list），`register_parser("marker", MarkerParser)` 后用 `list_parsers()`/`get_supported_parsers()` 确认它已上榜，再 `get_parser("marker").parse_document(...)` 跑通；最后试图 `register_parser("mineru", ...)` 验证覆盖内建名会抛 `ValueError`。

> **下一章预告**：解析器把文档拆成了 content_list，但里面的公式、表格还只是粗糙的原始块。第 5 章将深入 RAG-Anything 的公式抽取与增强 Markdown 生成，看它如何把 `equation`/`table` 块进一步结构化、拼接成对 LLM 更友好的 Markdown 表示。
