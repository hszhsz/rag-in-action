# 第 11 章 批处理的两条流水线：BatchMixin 与 BatchParser

第 10 章里我们看到 `QueryMixin` 如何把一次问答拆解为检索、重排与生成。但在真实场景中，知识库往往不是一篇文档喂进去就完事——你手里可能是一个塞满几百份 PDF、Office 文件和图片的目录。如果还按单文件、串行的方式一篇篇 `process_document_complete`，I/O 与解析的等待时间会被白白浪费，几百份文档可能要跑一整夜。批处理要解决的正是这个工程问题：在受控的并发度下，把「找文件 → 解析 → 入库」这条流水线尽可能榨干吞吐，同时不让一两个坏文件拖垮整批。

RAG-Anything 在这一层提供了两套并存的实现：一套是挂在主类上的 `BatchMixin`（`raganything/batch.py`），它既保留了基于 `asyncio` 的旧版整文件夹处理，又封装了对新版解析器的调用；另一套是独立、可单独命令行运行的 `BatchParser`（`raganything/batch_parser.py`），它用线程池做并行解析，带进度条、超时和结构化结果。这一章我们把两套流水线拆开来读，弄清它们各自的并发模型、容错策略，以及为什么作者要让它们并存。

## 两套并发模型：asyncio 信号量 vs 线程池

理解本章最关键的一点，是这两套实现用了**两种完全不同的并发原语**。

`BatchMixin.process_folder_complete` 走的是协程路线。它先用 `asyncio.Semaphore(max_workers)` 创建一个信号量，再为每个文件 `asyncio.create_task` 包一个 `process_single_file` 协程，最后用 `asyncio.gather(*tasks, return_exceptions=True)` 一次性等待全部完成。并发上限由信号量在每个协程入口处的 `async with semaphore:` 卡住——超过 `max_workers` 个协程时，多出来的会在这里挂起等待。这套模型契合 `process_document_complete` 本身是 `async` 的事实：解析与入库过程中大量 `await`，协程在等待时让出事件循环，CPU 不被白占。

`BatchParser.process_batch` 走的是线程路线。它用 `concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)`，通过 `executor.submit` 把每个文件的 `process_single_file`（注意这是个**同步**方法）提交进去，再用 `as_completed(future_to_file, timeout=self.timeout_per_file)` 按完成顺序回收结果。因为底层解析器（如 mineru）的 `parse_document` 是同步阻塞调用，用线程池把它们摊到多个线程上才能真正并行 [[concurrent.futures — ThreadPoolExecutor]](https://docs.python.org/3/library/concurrent.futures.html)。

一句话概括设计取舍：`BatchMixin` 因为要复用异步的入库链路而选 `asyncio`；`BatchParser` 因为只做同步解析、还要当独立 CLI 用，选线程池更直接。

## process_folder_complete：旧版整文件夹流水线

`process_folder_complete` 是源码注释里标明的「ORIGINAL BATCH PROCESSING METHOD (RESTORED)」，它的职责是把一个文件夹一站式解析并入库。

它的参数几乎全部默认 `None`，然后在函数体开头从 `self.config` 兜底填充：`output_dir` 取 `config.parser_output_dir`、`parse_method` 取 `config.parse_method`、`file_extensions` 取 `config.supported_file_extensions`、`recursive` 取 `config.recursive_folder_processing`、`max_workers` 取 `config.max_concurrent_files`，`display_stats` 默认 `True`。这种「显式传参覆盖、否则回落配置」的模式贯穿整个 `BatchMixin`，调用方既能零配置开箱即用，也能逐项精调。

收集文件时，它按 `recursive` 决定 glob 模式：递归用 `**/*{file_ext}`、否则用 `*{file_ext}`，对每个扩展名调用 `folder_path_obj.glob(pattern)`。处理前还会先 `_ensure_lightrag_initialized()` 并检查 `init_result.get("success")`，失败直接抛 `RuntimeError`；文件夹不存在则抛 `FileNotFoundError`；没找到文件则记一条 warning 后直接返回。

最有意思的是子目录处理。内部协程用一个 lambda 判断文件是否在子目录（`len(file_path.relative_to(dir_path).parents) > 1`），若在子目录，就把 `output_dir` 拼成 `output_path / file_path.parent.relative_to(folder_path_obj)`，并把 `file_name` 设为相对路径。这样解析产物的目录层级会**镜像**输入文件夹结构，避免不同子目录里的同名文件互相覆盖。

## 容错：单文件失败不拖垮整批

两套流水线在容错上思路一致——把异常关进每个文件自己的边界里，绝不让它冒泡终止整批。

`process_folder_complete` 的内部协程用 `try/except Exception` 包住单文件处理，成功返回 `(True, file_path, None)`，失败记 error 日志后返回 `(False, file_path, str(e))`。`gather` 又加了 `return_exceptions=True` 作第二道防线：万一协程本身抛出未被捕获的异常，结果里会是一个 `Exception` 对象而非元组，汇总阶段用 `isinstance(result, Exception)` 区分，归入 `failed_files`（文件名标为 `"unknown"`）。

`BatchParser.process_single_file` 同样用 `try/except` 把解析包起来，失败返回 `(False, file_path, error_msg)`。而 `process_batch` 还在 `as_completed` 循环外套了一层 `try/except/finally`：如果整个回收过程出异常（比如触发 `timeout`），`except` 分支会遍历 `future_to_file`，把所有 `not future.done()` 的文件标记为失败并写入「Processing interrupted」错误；`finally` 保证进度条 `pbar.close()` 一定被调用。这是一种「批级别」的兜底，单文件兜底之上再加一层。

## BatchProcessingResult：结构化的结果对象

`BatchParser` 不返回裸字典，而是返回一个 `@dataclass` 定义的 `BatchProcessingResult`，字段包括 `successful_files`、`failed_files`、`total_files`、`processing_time`、`errors`（文件路径到错误信息的字典）、`output_dir`，以及一个默认 `False` 的 `dry_run` 标记。

它提供两个便利成员。`success_rate` 是个 `@property`，按 `len(successful_files) / total_files * 100` 算成功百分比，并对 `total_files == 0` 做了除零保护返回 `0.0`。`summary()` 方法把这些字段拼成一段多行可读文本，包含总数、成功数（带百分比）、失败数、处理耗时（保留两位小数）、输出目录和 dry-run 标志。`process_batch` 在返回前会 `self.logger.info(result.summary())`，CLI 的 `main()` 也会 `print` 它——同一个 `summary()` 既服务日志又服务终端。

## dry_run：先看会处理哪些文件，再决定跑不跑

`process_batch` 有一个 `dry_run` 参数（默认 `False`）。开启后，它在 `filter_supported_files` 选出文件之后、真正提交线程池之前就提前返回：把筛出的文件全部放进 `successful_files`、`total_files` 设为筛出数量、`processing_time` 设为 `0.0`、`dry_run=True`，**不触发任何解析**。

这个设计对大批量场景非常实用——面对几百个文件和递归目录，你往往想先确认「到底哪些文件会被处理、扩展名过滤是否符合预期」，再决定是否启动耗时的解析。CLI 里 `--dry-run` 直接暴露了这个能力，并会逐行打印「files that would be processed」。

## 文件筛选与支持的扩展名

`BatchParser.get_supported_extensions` 把解析器声明的 `OFFICE_FORMATS`、`IMAGE_FORMATS`、`TEXT_FORMATS` 三个集合做并集，再额外并上 `{".pdf"}`，返回成列表。也就是说支持范围由具体解析器决定，PDF 则是无条件支持的兜底项。

`filter_supported_files` 负责把混杂的输入（文件与目录）展开成扁平的、仅含受支持类型的文件列表。它逐个判断路径：是文件就按 `path.suffix.lower()` 比对扩展名集合，不支持则 warning；是目录则按 `recursive` 决定用 `rglob("*")`（递归）还是 `glob("*")`（仅当前层），并对每个条目再校验 `is_file()` 与扩展名；路径不存在则 warning「Path does not exist」。注意扩展名比对统一用 `.lower()`，所以 `.PDF` 和 `.pdf` 都能被识别。

`BatchMixin` 把这两个能力转交了出去：`get_supported_file_extensions` 和 `filter_supported_files`（mixin 版）内部都临时 new 一个 `BatchParser` 来委托执行，让主类无需关心扩展名细节。

## BatchMixin 对新版解析的封装

`BatchMixin` 提供了一组面向「新版 BatchParser」的方法。`process_documents_batch`（同步）和 `process_documents_batch_async`（异步）结构对称：都先用 config 兜底参数，再构造 `BatchParser(parser_type=self.config.parser, max_workers=..., show_progress=..., skip_installation_check=True)`，然后分别调用 `process_batch` 或 `process_batch_async`。

两处细节值得注意。其一，构造时固定传 `skip_installation_check=True`，源码注释解释为「Skip installation check for better UX」——避免因解析器安装检测的误报打断批处理。`BatchParser.__init__` 里这个检查即便失败也只 warning 不抛错，注释写明「the parser might still work」。其二，`process_batch_async` 并非真正的异步解析，它在 `BatchParser` 里是用 `loop.run_in_executor(None, self.process_batch, ...)` 把同步版本扔进默认线程池跑——本质是「在事件循环里不阻塞地等同步批处理跑完」，方便嵌进异步代码。

## process_documents_with_rag_batch：解析与入库的两段式编排

如果说前面的方法只管解析，`process_documents_with_rag_batch` 才是把「批量解析」和「批量入库」串起来的总指挥。它分两步：第一步调 `process_documents_batch` 解析得到 `parse_result`；第二步 `_ensure_lightrag_initialized` 后，遍历 `parse_result.successful_files`，对每个文件 `await process_document_complete` 做 RAG 入库，逐个收进 `rag_results` 字典（成功记 `{"status": "success", "processed": True}`，失败记错误信息且 `processed=False`）。源码注释坦承入库这步「could be parallelized in the future」——目前是串行的。

这个方法还是本章唯一接入回调系统的入口。它在开头通过 `getattr(self, "callback_manager", None)` 拿回调管理器，开始时 `dispatch("on_batch_start", file_count=...)`，结束时 `dispatch("on_batch_complete", ...)` 并带上 `total_files`、`successful`、`failed`、`duration_seconds`。耗时用 `time.time()` 在首尾相减得出。这套 `on_batch_start/on_batch_complete` 回调正是下一章可观测性的伏笔。最终它返回一个聚合字典，含 `parse_result`、`rag_results`、`total_processing_time` 及两个 RAG 计数。

## 独立 CLI：batch_parser.py 的 main()

`batch_parser.py` 文件末尾的 `main()` 让这个模块可以脱离主类、作为命令行工具单独使用。它用 `argparse` 暴露 `paths`（多个）、`--output/-o`（必填）、`--parser`（默认 `mineru`）、`--method`（`auto/txt/ocr`，默认 `auto`）、`--workers`（默认 `4`）、`--no-progress`、`--recursive`（默认 `True`）、`--timeout`（默认 `300`）、`--dry-run` 等参数，配置好日志后构造 `BatchParser` 跑 `process_batch`，打印 `summary()`，并按 `result.failed_files` 是否非空返回退出码 `1` 或 `0`——这让它能干净地嵌进 shell 脚本或 CI 流水线。`--parser` 的帮助文本特别说明：作为库使用时通过 `register_parser()` 注册的自定义解析器也可用，但独立 CLI 自身不做插件发现。

## 本章小结

- RAG-Anything 的批处理有两套并存实现：`BatchMixin`（`batch.py`）与独立的 `BatchParser`（`batch_parser.py`）。
- 两者用不同并发原语：`process_folder_complete` 用 `asyncio.Semaphore` + `gather` 控制协程并发；`BatchParser.process_batch` 用 `ThreadPoolExecutor` + `as_completed` 控制线程并发。
- 并发上限统一由 `max_workers` 控制，默认从 `config.max_concurrent_files` 兜底。
- 容错采用双层结构：单文件 `try/except` 把失败收成结果元组，外层（`return_exceptions=True` 或批级 `try/except/finally`）兜底，单个坏文件不会终止整批。
- `process_folder_complete` 会镜像输入文件夹的子目录结构到输出目录，避免同名文件覆盖。
- `BatchProcessingResult` 是结构化 `@dataclass`，提供 `success_rate` 属性（含除零保护）和 `summary()` 文本。
- `dry_run` 让你在不触发解析的前提下预览将被处理的文件清单。
- 支持的扩展名由解析器的 `OFFICE_FORMATS`/`IMAGE_FORMATS`/`TEXT_FORMATS` 并集加 `.pdf` 决定，比对时统一 `.lower()`。
- `process_documents_with_rag_batch` 把解析与入库编排成两段式，并通过 `callback_manager` 派发 `on_batch_start`/`on_batch_complete`，但入库步骤目前是串行的。
- `batch_parser.py` 提供独立 CLI（`main()`），按失败与否返回退出码，可嵌进脚本或 CI。

## 动手实验

1. **实验一：观察两种并发模型** — 准备一个含十余个文件的文件夹，分别调用 `process_folder_complete`（asyncio 路线）和 `process_documents_batch`（线程池路线），把 `max_workers` 从 `1` 调到 `8`，记录各自耗时，体会信号量与 `ThreadPoolExecutor` 在 I/O 密集场景下的差异。

2. **实验二：玩转 dry_run** — 直接调用 `BatchParser.process_batch(..., dry_run=True)` 或运行 CLI 的 `--dry-run`，对一个递归目录预览将处理的文件；故意混入几个不支持的扩展名（如 `.xyz`），确认它们被 `filter_supported_files` 过滤并触发 warning。

3. **实验三：制造失败验证容错** — 在批处理目录里放一个损坏的 PDF，跑 `process_documents_batch` 后打印 `result.summary()` 与 `result.errors`，确认坏文件被归入 `failed_files`、其余文件仍成功，且 `success_rate` 计算正确。

4. **实验四：跟踪批处理回调** — 给主类设置一个 `callback_manager` 并注册 `on_batch_start`/`on_batch_complete` 回调，调用 `process_documents_with_rag_batch`，在回调里打印 `file_count`、`successful`、`failed`、`duration_seconds`，观察解析与入库两段式编排的全过程。

> **下一章预告**：批处理给坏文件兜了底，但单文件解析本身的瞬时抖动、外部服务限流又该如何应对？第 12 章我们进入韧性与可观测性回调，解构 `resilience.py` 的 retry 重试与 `CircuitBreaker` 熔断器，以及 `callbacks.py` 的回调系统——看清楚本章那两个 `on_batch_*` 事件背后到底是怎样一套机制。
