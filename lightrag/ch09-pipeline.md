# 第 9 章 文档处理管线 Pipeline

前面几章把 LightRAG 的零件拆开了：第 3 章的切分、第 4 章的解析、第 5 章的实体关系抽取、第 7、8 章的存储与共享并发。
可这些零件要拼成一台能持续吞文档、跑得稳、崩了能续、还允许中途叫停的机器，靠的就是本章的主角——文档处理管线。
它的全部实现集中在 `pipeline.py`(4479 行) 的 `_PipelineMixin` 类里。
这个 Mixin 不持有任何存储字段，而是通过 MRO 复用 `LightRAG` 主类提供的 `self.full_docs`、`self.doc_status`、`self.chunks_vdb` 等句柄，把"怎么处理文档"这件事单独剥离成一个可读的模块。

LightRAG 的管线设计有三个值得反复咀嚼的工程决策。
其一，**入队与处理彻底分离**，登记文档和真正干活是两个独立入口。
其二，**一次只允许一条管线在跑**，新来的请求不另起炉灶而是"挂起合并"进正在运行的循环。
其三，**三阶段级联队列**——解析(PARSING)、多模态分析(ANALYZING)、抽取(PROCESSING)各有独立 worker 池，文档像在三道工序的流水线上逐级流动。
下面逐一拆开，并在最后回到失败、续跑与取消三件最考验工程功力的事上。

## 入队：只登记，不处理

`apipeline_enqueue_documents`(行 232) 是文档进入系统的门。
它的职责被严格限定在四步：校验/生成文档 ID、生成初始 DocStatus、过滤已处理文档、写入 `doc_status`。
它**绝不触发任何抽取**——这条边界是后面所有调度逻辑能成立的前提。

文档 ID 的生成有一套优先级（`_add_content`，行 479-503）：

- 调用方显式传 `ids` 时直接采用——这是 SDK 直插路径 `ainsert` 的标志，会强制走 `FULL_DOCS_FORMAT_RAW` 格式；
- 有已知文件来源时用规范化文件名算 `compute_mdhash_id(..., prefix="doc-")`；
- RAW 格式则用内容哈希；
- 其余情况（如 `pending_parse` 服务端上传）用 `文件名-track_id-index` 兜底。

这套规则保证了同名文件、同内容文件能被映射到稳定的 doc_id，为后续去重打基础。

去重是入队阶段的重头戏，分三层（行 505-705）：

- **批内同名去重**：本次调用内部用 `source_to_doc_id` 字典挡掉重复文件名；
- **跨批文件名去重**：`get_existing_doc_by_file_basename` 查 doc_status 里是否已有同 basename 记录；
- **内容哈希去重**：`get_existing_doc_by_content_hash` 抓住"换了文件名但正文相同"的情况。

三者分别打上 `duplicate_kind` 为 `batch_duplicate`/`filename`/`content_hash` 的标记，归入 `duplicate_attempts`。
关键设计是这段"读 doc_status 判重—写 doc_status"的临界区被 `enqueue_serialize` 这个 workspace 级锁串行化（行 634-638）。
否则两个并发上传相同内容时可能都读到"不存在"、都写 PENDING，绕过去重。
注释里说得很直白——这个锁只覆盖判重+写入，不阻塞处理循环，也不阻塞 scan 分类。

初始状态由 `_initial_doc_status`(行 586) 构造，永远是 `DocStatus.PENDING`。
它记录 `content_summary`、`content_length`、`file_path`、`track_id`，并把 `process_options` 镜像进 `metadata`，方便管理 UI 不查 `full_docs` 就能展示每篇文档的处理策略。
注意 `pending_parse` 格式此时 `content` 可为空——内容要等解析 worker 提取后才有，所以连 `content_hash` 都先不算。

入队对并发的态度很克制：它**允许在处理循环 busy 时入队**。
因为处理循环每跑完一批都会重新查 `doc_status`，并在 busy 时被设上 `request_pending` 标志（后面细说）。
只有两种状态会拒绝入队并抛 `RuntimeError`（行 333-345）：

- `scanning_exclusive`——扫描任务正在分类阶段读写 doc_status；
- `destructive_busy`——清库/删档正在丢弃存储，并发写会被静默覆盖。

`from_scan=True` 是 scan 自身入队时用来绕过第一个守卫的后门，外部调用方必须保持 False。
这条"入队不阻塞处理、处理回头补查"的契约，是后面 busy/pending 调度成立的另一半基础：正因为入队侧从不抢占处理循环，处理循环才能放心地用"跑完一批再回查"的简单模型，而不必担心漏掉运行期间新进来的文档。

## 处理主循环与"一次只跑一条管线"

`apipeline_process_enqueue_documents`(行 927) 才是真正干活的入口。
它一进来就抢 `pipeline_status` 的锁，判断 `busy` 标志（行 946-982）：

- 若 `busy=False`：先查出所有"在途"文档（`_INFLIGHT_DOC_STATUSES`，行 107），无文档就直接返回；有文档则把自己置为 `busy=True`，清空 `request_pending`、`cancellation_requested` 等标志，开始循环。
- 若 `busy=True`：**不另起管线**，只把 `request_pending` 置为 True 然后立即返回。

这里 `_INFLIGHT_DOC_STATUSES` 的定义很重要——它是 `(PROCESSING, FAILED, PENDING, PARSING, ANALYZING)` 五种状态的元组。
也就是说，崩溃残留在任意中间态的文档、以及上轮失败的 FAILED 文档，都会在新一轮被重新捞起，这是断点续跑的起点。

为什么只允许一条循环？
因为 LightRAG 的实体关系抽取与图合并会写共享存储和共享 flush 缓冲（第 8 章），多条管线并行会互相踩缓冲、重复抽取。
所以全局只允许一条循环，新来的工作通过 `request_pending` 信号"搭车"进现有循环，而不是各跑各的。

退出时的交接是这套机制最精巧的地方，集中在 `_atomic_release_busy_or_consume_pending`(行 1495)。
每批处理完，循环都调它，它在**同一个临界区**里做二选一：

- 若 `request_pending=True`：清掉该标志、返回 False，循环重新查 doc_status 继续干；
- 若为 False：把 `busy=False` 一并清掉、返回 True，循环 `break` 退出。

主循环里用 `busy_released_in_loop` 变量记录是否已在这里释放过 busy，`finally` 块据此决定要不要再清一次（行 1102-1112）。
注释解释了这个原子性要解决的竞态：一个并发入队若在"循环读到 request_pending=False"和"finally 清 busy"之间挤进来设 request_pending=True，新文档就会卡死在 PENDING。
把"观察 request_pending"和"释放 busy"塞进同一把锁，这扇窗就关上了。

## 三阶段级联队列：parse → analyze → process

真正的批处理在 `_run_pipeline_batch`(行 1143)。它搭起三层级联队列：

- **Layer 1 解析**：按 parser 的 `queue_group` 分组，每组一个 `asyncio.Queue`（`maxsize=self.queue_size_parse`），`native` 组永远存在（行 1166-1170）；
- **Layer 2 多模态分析**：单个 `q_analyze` 队列；
- **Layer 3 抽取**：单个 `q_process` 队列。

每层各起一批 worker 任务（行 1232-1241）：

- 解析 worker 数量由 `_group_concurrency`(行 1183) 按组算出——内置组用 `max_parallel_parse_<group>` 实例字段，第三方组用 ParserSpec 声明的 `concurrency`，未声明则共享 native 的额度；
- 分析 worker 起 `max_parallel_analyze` 个；
- 抽取 worker 起 `max_parallel_insert` 个。

所有 worker 共享一个 `_BatchRunContext`(行 194)。
它打包了 `pipeline_status`、锁、三组队列，以及一个 `asyncio.Semaphore(self.max_parallel_insert)` 信号量——这就是批内文档级并发的闸门，呼应第 8 章的全局并发槽。
`parser_specs` 也是在批开始时一次性快照进 ctx 的，保证一批运行期间 `register_parser` 不会改变引擎集合，做到 snapshot 一致性。

文档投递时，`_run_pipeline_batch` 按每篇文档解析引擎对应的 `queue_group`，把 `(doc_id, status_doc)` 投到正确的解析队列（行 1251-1291）。
然后 `await` 三层队列依次 `join()`（行 1293-1295）。
这里有个关键的失败隔离：取 `full_docs` 内容的 `get_by_id` 若对单篇文档失败，只把那一篇标 FAILED 并 `continue`，不让整批崩（行 1262-1279）。

`finally` 块保证所有 worker 被 `cancel()` 并 gather（行 1296-1299）。
哪怕 join 因后端故障抛异常，也绝不让 worker 变孤儿任务——孤儿会继续 drain 队列、继续往 `history_messages` 写日志，而调用方的 finally 已经清了 busy，造成"busy=False 但处理还在跑"的幻象。
批次若因内部存储错误中止，还会调 `_discard_pending_index_ops` 丢弃共享 flush 缓冲里属于已 FAILED 文档的脏记录（行 1307-1313），呼应第 2 章提到的 `drop_pending_index_ops`——让这些文档下轮重跑，而不是带着半截脏数据。

### 三个 worker 对应三个状态阶段

`_parse_worker`(行 1546) 消费解析队列。
它先做边界取消检查，把状态置 `PARSING`，按 doc 从快照里解析出真正的 parser（`engine` 参数只是队列组 ID，真正引擎按 doc 重新解析，行 1629-1647）。
然后跑 `parser.parse()`，用 `strip_control_characters` 清掉控制字符（包括会让 PostgreSQL 写入崩溃的 NUL），刷新可能被 `pending_parse→raw` 转换打补丁的 `content_hash`，再调 `_mark_duplicate_after_parse` 做解析后的内容去重（行 1782-1792）。
一个值得注意的优化：解析完成后**把正文从队列负载里弹出**（`parsed_data_w.pop("content")`，行 1822），后续 analyze/process 阶段按 doc_id 从 `full_docs` 重新读正文——避免大文档常驻三层队列内存。
失败则写 `FAILED`，且写状态本身若也失败（后端宕机）只 log 不再抛，免得 worker 自己死掉（行 1858-1865）。

`_analyze_worker`(行 1869) 是多模态阶段。
它把状态置 `ANALYZING`，调 `analyze_multimodal`(行 3247)——这一步跑 VLM 对图表/图片做描述，本章只需知道它是可选阶段。
它用三态决策记录结果（行 1944-1958）：`analyzing_stage_skipped`（无 blocks/无 i/t/e 选项跳过）、`multimodal_processed`（真正完成，盖 `analyzing_end_time`）、两者皆无（软失败，什么都不写）。
没有多模态需求时这一阶段实质是直通。
它仍会盖上 `analyzing_start_time`/`analyzing_end_time` 这类阶段耗时戳，这些字段靠 `_upsert_doc_status_transition` 的 carry-over 一路保留到 PROCESSED，事后能用来做每阶段耗时分析。

`_process_worker`(行 1998) 最薄，只是把 `(doc_id, status_doc, parsed_data)` 转交给 `process_single_document`。
但它有个硬底线：若 `process_single_document` 连自己的 FAILED 写入都抛了（说明 doc_status 后端宕了），worker **不能死**——否则会卡死 `q_process.join()`，把整条管线吊在 busy。
它转而设 `cancellation_requested + reason=internal_error` 走批次中止路径，继续 drain 队列让批次干净收尾（行 2010-2027）。

## 单文档全流程 process_single_document

`process_single_document`(行 2035) 是 PROCESSING→PROCESSED 状态机。
它先抢信号量 `ctx.semaphore`（批内并发闸），从 `full_docs` 读正文、解析 `process_options`，然后分四步走：

1. **断点续跑清理**：调 `_purge_stale_extraction_if_resuming`（下节细说）。
2. **切分派发**：按 `process_options` 是否显式指定策略二选一（行 2188-2335）。显式 F/R/V/P 走标准化文件切分器，参数来自入队时持久化的 `chunk_options` 快照；无选择子则走可被用户自定义的 `self.chunking_func` 旧式 6 参签名。这套设计让 `ainsert` 的旧调用方不受影响，又让上传路径能精确选策略。切分后还有两道保险：`enforce_chunk_token_limit_before_embedding` 硬切超长 chunk（行 2418-2440），以及 `backfill_chunk_sidecars` 回填块溯源（行 2466-2469）。
3. **两阶段持久化**：Stage 1 用 `asyncio.create_task` 并行写 `doc_status=PROCESSING`、`chunks_vdb`、`text_chunks`（行 2484-2510）；Stage 2 跑 `_process_extract_entities` 抽取实体关系——除非 `process_options` 带 `!`（`skip_kg`）则整段跳过，chunks 仍留在向量库供 naive/mix 检索（行 2516-2531）。
4. **合并与收尾**：抽取成功后调 `merge_nodes_and_edges` 合并进知识图谱（行 2570），再写 `doc_status=PROCESSED` 并 `_insert_done()`（行 2600-2616）。

整个流程在 extract 和 merge 两个阶段边界都包了 `try/except`，失败统一交给 `_finalize_doc_failure`。
而且每个阶段开头都插了 `_raise_if_cancelled`，让中途取消能尽快生效。
值得专门点出的是 merge 阶段的 `IndexFlushError`：`_insert_done` flush 的是**跨文档共享的缓冲**，无法判断是谁的记录失败，所以一旦它失败就设 `internal_error` 中止整批（行 2628-2648）。
继续干会把别的文档误标 PROCESSED 却丢数据，宁可整批回滚重来。
这与单文档失败只标自己 FAILED 的策略形成对比——抽取/合并是文档自洽的，可以隔离；而 flush 是共享的，错一个就得停全场。这条"按失败影响范围决定隔离粒度"的判断，是理解整套异常处理的钥匙。

## 进度回写与历史日志治理

整条管线对外暴露状态的唯一窗口是第 8 章介绍的共享 `pipeline_status`。
管线在每个关键节点都往里写两类信息：`latest_message`（一行最新进度，供 WebUI 顶栏滚动显示）和 `history_messages`（追加式的历史流水）。
主循环开头会把这批文档数写进 `docs`、`batchs`，把 `cur_batch` 归零（行 1069-1073），单文档处理时再每完成一篇就 `ctx.processed_count += 1` 并回写 `cur_batch`（行 2090-2093）——前端据此画进度条。

值得一提的是 `history_messages` 的内存治理。
它是 `Manager.list` 撑起的跨进程共享列表，长跑会无限膨胀。
`process_single_document` 里有一段保护：当历史超过 10000 条时，原地 `del [...:-5000]` 只留最新 5000 条（行 2106-2114）。
注意它用的是**原地切片删除**而非重新赋值——因为重新赋值会断开 `Manager.list` 的共享引用，让其他进程看不到后续追加。
这是共享状态编程里典型的"必须 in-place 修改"陷阱，第 8 章已埋过伏笔。

另外，`job_name` 由 `_format_job_name`(行 1526) 用首篇文档路径前 20 字符 + `[N files]` 拼成，让运维一眼能认出当前批次在处理哪一摞文件。

## 断点续跑与一致性校验

进程崩溃后怎么恢复？答案藏在两处。

其一，主循环捞文档用的 `_INFLIGHT_DOC_STATUSES` 包含 PARSING/ANALYZING/PROCESSING——崩溃时停在任意中间态的文档，重启后都会被重新捞起。

其二，`_validate_and_fix_document_consistency`(行 1315) 在每批处理前做双写一致性校验。
它遍历待处理文档，凡 `full_docs` 里查不到正文的，FAILED 文档保留（供人工复查），其余视为"幽灵行"直接从 doc_status 删除（行 1327-1390）。
随后把所有中断态文档（PROCESSING/PARSING/ANALYZING/FAILED）统一重置回 PENDING，清空 `error_msg`，把 metadata 收敛成只剩入队时的指令字段（`doc_status_reset_metadata`），避免上轮残留的耗时/结果字段污染 WebUI 显示（行 1399-1493）。
重置同时镜像回内存里的 `status_doc`，让 worker 把干净的 metadata 一路带下去。

真正的"清理 stale 抽取结果"在 `_purge_stale_extraction_if_resuming`(行 2668)。
当一篇文档已经有提取过的正文（lightrag 格式有 sidecar、或 raw 格式有非空 content），却又要在新 `process_options` 下重跑时，它先按 `status_doc.chunks_list` 把上轮的 chunk 和关联 KG 条目全删（`_purge_doc_chunks_and_kg`，行 2744）。
再清空 `status_doc` 上的 `chunks_list/chunks_count`，防止后续状态机写回过期 ID（行 2750-2754）。
它还会在文件名 hint 与已存 `parse_engine` 不一致时发警告——已提取内容是事实来源，想换引擎得删了重传。
这一步只在文档"确实已有提取产物"时才触发（lightrag 有 sidecar 或 raw 有非空 content），首次处理的文档会直接短路返回，不付出任何额外删除代价。

一致性的另一半在状态迁移的唯一出口 `_upsert_doc_status_transition`(行 2760)。
所有 PARSING/ANALYZING/PROCESSING/PROCESSED/FAILED 转换都走它，统一写 `content_summary`、`content_hash`、`updated_at`。
并通过 `doc_status_transition_metadata` 做 metadata 的 carry-over，保证 `process_options`、`parse_engine`、`parse_start_time` 等字段跨状态不丢、不跳变。

## 失败隔离与取消机制

失败隔离的原则是"一篇坏不连累一批"。
`_finalize_doc_failure`(行 2887) 是 extract/merge 阶段失败的统一收尾。
它记录错误（区分用户取消与内部错误两种文案）、`cancel()` 掉挂起的阶段任务、flush LLM 响应缓存（让已完成的兄弟任务的 cache_id 在重启后还在），最后写一行保留了失败 chunk 快照和耗时 metadata 的 FAILED 状态（行 2956-2978）。
解析后内容撞车则走 `_mark_duplicate_after_parse`(行 3048)，标 FAILED 并写 `[DUPLICATE:content_hash]` 摘要、删掉重复的 `full_docs` 行、归档源文件（行 3078-3119）。

取消机制是一套协作式的标志位。
`_raise_if_cancelled`(行 2795) 在锁内检查 `cancellation_requested`，有则抛 `PipelineCancelledException`，供 `process_single_document` 在各阶段边界主动检查。
`_cancellation_requested`(行 2830) 是只读版，供 worker 在取下一个队列项前判断要不要直接把它标 FAILED 跳过（`_mark_doc_cancelled_in_stage`，行 2844）。
`_cancellation_label`(行 2806) 区分"用户取消"和"内部错误取消"两种归因——后者由 `IndexFlushError` 或 process worker 未捕获异常触发，会附上驱动名和根因。
它会在主循环里以 error 级日志打出可操作的 `_internal_halt_message`(行 2818)，提示运维"解决存储问题后重启，受影响文档仍在队列里"。
这种把"用户主动叫停"和"系统被迫 halt"分开记录的设计，让事后排障能一眼看清到底是谁按的停止键。
取消的传播也分两种风格并非偶然：`process_single_document` 这类长任务在阶段边界主动 `_raise_if_cancelled` 抛异常、走 `_finalize_doc_failure` 收尾；而 worker 在取队列项前用只读的 `_cancellation_requested` 判断，直接把待处理项标 FAILED 后 `task_done()` 排空队列，好让 `q.join()` 能正常返回、批次干净落幕。
两种风格共用同一个标志位，却各自选了最适合所在位置的退出方式。

## 本章小结

- LightRAG 文档处理是一条三阶段级联流水线：PARSING→ANALYZING→PROCESSING，分别由 `_parse_worker`/`_analyze_worker`/`_process_worker` 三组 worker 驱动，对应 `DocStatus` 的三个中间态。
- 入队（`apipeline_enqueue_documents`）与处理（`apipeline_process_enqueue_documents`）彻底分离：前者只登记文档、置 PENDING、做三层去重，后者才真正干活。
- "一次只跑一条管线"靠 `busy` 标志 + `request_pending` 实现：新请求 busy 时不另起管线而是挂起合并，由 `_atomic_release_busy_or_consume_pending` 在同一把锁里原子地决定释放 busy 还是吃掉 pending，关掉退出交接的竞态窗口。
- 并发分两层：批内文档级用 `asyncio.Semaphore(max_parallel_insert)`，解析阶段按 `queue_group` 分组限流（`_group_concurrency`），呼应第 8 章的并发槽设计。
- 断点续跑靠 `_INFLIGHT_DOC_STATUSES` 把崩溃残留的中间态文档重新捞起，`_validate_and_fix_document_consistency` 清幽灵行并把中断态重置回 PENDING，`_purge_stale_extraction_if_resuming` 删掉上轮 stale 的 chunk 与 KG。
- 失败隔离让单文档失败标 FAILED 不拖垮整批，`_finalize_doc_failure` 统一收尾；批次因内部错误中止时用 `_discard_pending_index_ops` 丢弃共享 flush 缓冲里的脏记录。
- 取消机制用 `cancellation_requested` 协作式标志位，`_raise_if_cancelled`（抛异常）与 `_cancellation_requested`（只读分支）两种风格并存，且严格区分"用户取消"与"内部错误 halt"两种归因。
- 双写一致性由 `_upsert_doc_status_transition` 这个唯一迁移出口保证，配合 metadata carry-over 让关键字段跨状态不丢失。
- 大文档优化：解析后正文从队列负载弹出，下游按 doc_id 从 `full_docs` 重读，避免正文常驻三层队列内存。
- worker 的硬底线是"绝不让自己死"：状态写入失败只 log 不抛、process worker 异常转批次中止，杜绝队列 join 永久挂起导致 busy 卡死。

## 动手实验

1. **实验一：画出三阶段队列拓扑** — 打开 `pipeline.py` 的 `_run_pipeline_batch`(行 1143)，数清楚解析队列（按 `queue_group` 动态生成）、`q_analyze`、`q_process` 三层各起了多少 worker，把 `max_parallel_parse_native`、`max_parallel_analyze`、`max_parallel_insert` 三个字段找出来，画一张"文档如何在三道队列间流动"的拓扑图。
2. **实验二：复现 busy/pending 调度** — 阅读 `apipeline_process_enqueue_documents`(行 927-982) 与 `_atomic_release_busy_or_consume_pending`(行 1495)。用纸笔模拟：管线正在处理批次 A 时，一个新文档入队设了 `request_pending=True`，推演循环退出时这两段代码如何让批次 A 处理完后自动捞起新文档，而不是把它卡在 PENDING。
3. **实验三：追踪一次崩溃续跑** — 假设进程在某文档处于 `PROCESSING` 时被 kill。沿 `_INFLIGHT_DOC_STATUSES`(行 107)→`_validate_and_fix_document_consistency`(行 1315)→`_purge_stale_extraction_if_resuming`(行 2668) 走一遍，写出重启后这篇文档经历的状态变化序列，并说明上一轮写进 `chunks_vdb` 的脏 chunk 在哪一步被清掉。
4. **实验四：辨析两种取消** — 对比 `_finalize_doc_failure`(行 2887) 在 `PipelineCancelledException` 与普通异常下的不同文案分支，再看 `process_single_document` 里 `IndexFlushError` 如何设置 `cancellation_reason="internal_error"`（行 2637-2645）。说明为什么 LightRAG 要把"用户取消"和"内部存储错误 halt"在 `error_msg` 和日志里区分开，这对运维排障意味着什么。

> **下一章预告**：文档处理管线在抽取阶段反复调用 LLM 和 embedding，可这些模型究竟是怎么接进来的？第 10 章将拆解 LightRAG 的 LLM 接入层与多供应商绑定——它如何用统一的函数签名抹平 OpenAI、Ollama、Azure、Bedrock 等供应商的差异，如何做重试、缓存与并发限流，以及 `llm_response_cache` 在管线里扮演的角色。
