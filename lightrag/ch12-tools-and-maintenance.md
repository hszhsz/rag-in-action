# 第 12 章 运维工具与缓存迁移

一个 RAG 系统跑得久了，麻烦往往不在“查询”本身，而在数据。换了一个嵌入模型，向量库里旧的向量就和新的查询向量不在一个空间；把存储从 JSON 文件迁到 PostgreSQL，旧的 LLM 缓存得搬过去；WebUI 上手工编辑了一个实体，结果向量写失败、图谱和向量库悄悄漂移；Qdrant 升级了集合 schema，旧数据要做兼容。LightRAG 没有把这些一次性的、危险的、需要人盯着的操作塞进服务主流程，而是放进 `lightrag/tools/` 目录，做成一批独立的命令行工具，外加一个 `lightrag/evaluation/` 评估模块。

这一章解构这些“非主路径”代码。它们的代码量不小——`migrate_llm_cache.py` 有 1560 行、`clean_llm_query_cache.py` 1249 行、`rebuild_vdb.py` 1076 行——而且不约而同地体现出同一套运维工程取向：CLI 化、流式低内存、按批次、可中断、删除前必须确认、源数据只读、失败可重跑。读懂它们，等于看到了 LightRAG 对“数据安全”这件事的真实态度。

## `lightrag/tools/` 目录里有什么

先列清楚这个目录的内容。除了任务点名的几个大文件，目录里还有 `check_initialization.py`（启动自检）、`download_cache.py`（下载缓存）、`hash_password.py`（口令哈希）、`__init__.py`，以及三份配套的 README（`README_MIGRATE_LLM_CACHE.md`、`README_CLEAN_LLM_QUERY_CACHE.md`、`README_REBUILD_VDB.md`）和子目录 `lightrag_visualizer/`。`evaluation/` 目录里则有 `eval_rag_quality.py`、`offline_retrieval_check.py`、`sample_dataset.json`、`sample_retrieval_oracle.json` 和示例文档。

这些工具的共同入口风格是 `python -m lightrag.tools.<name>`，每个文件都在顶部用 `sys.path.insert(0, ...)` 把项目根目录加入路径、用 `load_dotenv(dotenv_path=".env", override=False)` 加载当前目录的 `.env`（注意 `override=False`，意味着 OS 环境变量优先于 `.env`），并调用 `setup_logger("lightrag", level="INFO")`。它们都不在服务进程里运行，而是“像一次服务启动那样”单独跑——这一点在 `rebuild_vdb.py` 的文档字符串里被明确强调。

## `migrate_llm_cache.py`：跨存储后端搬运 LLM 缓存

这个工具解决的是一类很具体的需求：把 `default:extract:*` 和 `default:summary:*` 这两类 LLM 响应缓存，在不同 KV 存储实现之间迁移，同时保留 workspace 隔离。支持的存储类型由常量 `STORAGE_TYPES` 列出：`JsonKVStorage`、`RedisKVStorage`、`PGKVStorage`、`MongoKVStorage`、`OpenSearchKVStorage`。

它的设计要点全在 `MigrationTool` 类里。第一，**配置自适应**。`get_workspace_for_storage` 的优先级是“存储专属 env 变量（如 `POSTGRES_WORKSPACE`，由 `WORKSPACE_ENV_MAP` 映射）> 通用 `WORKSPACE` > 空串”；`check_env_vars` 在缺变量时只 `print` 警告而非硬失败（注释明确写 “warnings only, no hard failure”），并退回去查 `config.ini`（`check_config_ini_for_storage`）。这让工具在各种部署形态下都能尽量跑起来，把真正的失败点留到 `initialize_storage` 里真正建连接时才暴露。

第二，**每种后端各写一套读取/计数/流式取数函数**，因为底层 API 完全不同：JSON 直接遍历 `storage._data`，Redis 用 `SCAN` 游标分页（`get_default_caches_redis`，带 pipeline 失败回退到逐 key get），PostgreSQL 用 `LIMIT/OFFSET` 翻页且把列名映射回缓存字段（`get_default_caches_pg`），MongoDB 用 `$regex` 加 `batch_size` 游标，OpenSearch 用 `_iter_raw_docs`。每个分页循环里都穿插 `await asyncio.sleep(0)` 让出控制权，避免阻塞事件循环。

第三，**默认走流式迁移**。`run()` 里 `setup_storage(..., use_streaming=True)` 只数总量、不一次性把数据读进内存；真正迁移由 `migrate_caches_streaming` 完成——`async for batch in self.stream_default_caches(...)` 一边从源读一边 `target_storage.upsert(batch)` 写目标，`DEFAULT_BATCH_SIZE = 1000`。文档里把它叫 “Memory-optimized mode”。`migrate_caches`（一次性载入全部）作为 Legacy 模式保留。

第四，**安全护栏**。迁移前 `count_available_storage_types` 校验至少有 2 种可用存储，否则直接报 “Migration Not Possible”；源为空则提示无可迁移并退出；目标已有数据会打印 “Migration will overwrite records with the same keys” 警告；最后 `confirm = input("\nContinue? (y/n): ")`，非 `y` 即取消。统计用 `MigrationStats` dataclass 逐批记录成功/失败/丢失记录数与错误明细，`print_migration_report` 给出成功率和前 5 条错误。值得注意的是：单批 `upsert` 失败只 `add_error` 后继续，不中断整体——这意味着迁移是“尽力而为 + 报告”，而非“全有或全无”的事务。

## `clean_llm_query_cache.py`：按模式与类型清理查询缓存

与迁移工具结构上是“孪生兄弟”，但目标相反：删除查询缓存。它处理的 key 不是 extract/summary，而是按 `QUERY_MODES = ["mix", "hybrid", "local", "global"]` 与 `CACHE_TYPES = ["query", "keywords"]` 组合出的 `mode:cache_type:*` 模式。同样支持 5 种 KV 后端，每种都有独立的 `count_query_caches_*` 计数实现。

清理策略由交互菜单给出三档：`[1]` 删全部、`[2]` 只删 `query`（保留 keywords）、`[3]` 只删 `keywords`（保留 query），对应 `cleanup_type` 的 `all`/`query`/`keywords`。`calculate_total_to_delete` 先按选项算出待删数量，若为 0 会拦住并要求重选，避免空操作。删除前同样有确认 `input("\nContinue with deletion? (y/n): ")`。

这个工具最有“运维血泪”味道的设计有两处。其一，**对 `JsonKVStorage` 的并发警告**：因为 JSON 是内存数据库、不支持多进程并发访问同一文件，所以选它时会弹出红色警告并强制二次确认 `input("\nHas LightRAG Server been shut down? (yes/no): ")`，答非 `yes` 直接取消。其二，**删除后必须置脏标志**：`delete_query_caches` 在 `async with storage._storage_lock` 里 `del storage._data[key]` 之后，紧接着 `await set_all_update_flags(storage.namespace, workspace=storage.workspace)`，代码注释写得很直白——“Without this, deletions remain in-memory only and are lost on exit”。删完还会 `index_done_callback()` 落盘，再 `count_query_caches` 复查，最后 `print_cleanup_report`。这是对第 8 章共享存储更新标志机制的一次实战应用。

## `rebuild_vdb.py`：把向量库从权威源头重建出来

这是这批工具里设计最克制、注释最考究的一个。它的世界观写在文档字符串里：**知识图谱和 `text_chunks` KV 存储才是权威数据源**，向量库只是它们的派生物。`entities_vdb <- 图节点`、`relationships_vdb <- 图边`、`chunks_vdb <- text_chunks KV`。当运行时某次向量写入失败（比如 WebUI 编辑实体时），图和向量就漂移了；或者换了嵌入模型/维度，旧向量与新查询空间不匹配。这时就 drop 掉向量库、从权威源重新嵌入。

重建函数三件套是 `rebuild_entities_vdb`、`rebuild_relationships_vdb`、`rebuild_chunks_vdb`。关键的正确性细节在于：**payload 必须和运行时的权威写入点逐字段对齐**。`rebuild_entities_vdb` 的注释写 “Payloads mirror the authoritative write point in operate._merge_nodes_then_upsert field for field”，它复用 `_truncate_vdb_content`、用 `compute_mdhash_id(entity_name, prefix="ent-")` 算 id、并在 `payloads` 里去重（`duplicates` 计数）。chunks 的重建有个微妙处理：直接枚举 `text_chunks` 命名空间下的所有 key（`enumerate_kv_keys`）而非走 `doc_status.chunks_list`，因为 `ainsert_custom_kg` 写的 chunk 没有 doc_status 记录，且 chunk id 有 `chunk-<hash>`、`{doc_id}-chunk-{order}` 等多种 schema，没有单一前缀能匹配全部。

它的安全机制比前两者更严密。第一，**两阶段提交语义**：`_drop_vdb` 先 drop，`_upsert_batch` 用 `safe_vdb_operation_with_exception` 带 `max_retries=3` 重试上插，但只把成功上插的记下为 `staged`；只有 `_flush`（调 `index_done_callback`）真正落盘成功，才把 `staged` 计入 `rebuilt`。注释解释这是为延迟嵌入后端（nano/faiss）准备的——它们在 `index_done_callback` 里才真正算向量并持久化，所以嵌入服务挂了会在 flush 阶段才暴露，处理方式和上插失败一样：记录、清零 staged、继续，**因为源从不被修改，用户可以重跑**。第二，**诊断与重建分离**：菜单 `[1]` 是只读的一致性检查 `check_vdb_consistency`（用 `make_relation_vdb_ids` 兼容历史上反序的关系 id），让用户先判断是否值得做昂贵的全量重嵌；`[2]/[3]/[4]` 才是真正的 drop+rebuild。第三，**强确认与粘性失败**：启动先 `confirm_server_stopped`（答 `yes` 才继续），重建前再 `confirm_rebuild`；`run()` 的返回值有“粘性失败”语义——一旦任何一次重建报了 batch/flush 错误，整个会话 `success = False`，注释说明这是因为“在向量库已经被 drop 之后的部分重建，绝不能对自动化看起来像一次干净的恢复”。

## `prepare_qdrant_legacy_data.py`：构造 legacy 数据以测试迁移

前几个工具是“在生产里搬数据”，这个则是“为测试造数据”。它的用途很专门：把带 `workspace_id` 的新版 Qdrant 集合（`lightrag_vdb_chunks`/`entities`/`relationships`，见 `COLLECTION_NAMESPACES`）里的数据，复制成不带 `workspace_id`、按 `{workspace}_{suffix}` 动态命名的 legacy 集合，用来验证 `setup_collection` 里的迁移逻辑。

它是这批工具中唯一用 `argparse` 提供完整 flag 的：`--workspace`（默认 `space1`）、`--types`（逗号分隔的集合类型）、`--batch-size`（默认 `DEFAULT_BATCH_SIZE = 500`）、`--dry-run`、`--clear-target`。`--dry-run` 贯穿全程——`delete_collection`、`create_legacy_collection`、`copy_collection_data` 内部都先判断 `if self.dry_run` 只打印不执行，这是“预览优先”的典范。复制逻辑用 Qdrant 的 `scroll` 加 workspace `Filter` 分批拉取，再 `new_payload.pop(WORKSPACE_ID_FIELD, None)` 去掉 workspace 字段以模拟 legacy 格式；缺 id 时用 `uuid.UUID(bytes=hashed[:16], version=4).hex` 生成确定性 id。流程上还坚持“先验证 workspace 有数据再建 legacy 集合”（`get_workspace_count` 前置检查），避免造出空集合。依赖通过 `pipmaster` 按需安装（`pm.is_installed("qdrant-client")`）。

## `lightrag_visualizer/graph_visualizer.py`：3D 知识图谱可视化

这是一个交互式桌面工具，不是数据迁移类。它读取 LightRAG 导出的 GraphML 文件，用 ModernGL + imgui_bundle 渲染成可旋转的 3D 图谱。依赖也是 `pipmaster` 懒加载：`moderngl`、`imgui_bundle`、`pyglm`、`python-louvain` 缺失时自动 `pm.install`。

核心数据流很清晰：`GraphViewer.load_graphml` 里 `nx.read_graphml(filepath)` 把图读进 NetworkX；`community.best_partition(self.graph)` 用 Louvain 算法做社区发现，`generate_colors(num_communities)` 给每个社区分配颜色；节点位置由 `nx.spring_layout` 计算布局，再封装成 `Node3D`（带 position/color/label/size）。`main()` 用 `hello_imgui` 起一个窗口标题为 “3D GraphML Viewer” 的应用，界面上有 “Load GraphML” 按钮通过 `tkinter.filedialog` 选文件。它对中英文字体分别准备了 `Geist-Regular.ttf` 和 `SmileySans-Oblique.ttf`。把可视化做成离线独立工具而非服务端点，避免了在生产服务里背上一堆图形依赖。

## `evaluation/eval_rag_quality.py`：用 RAGAS 给回答质量打分

评估模块和工具是一脉相承的“离线、CLI、对真实服务说话”。`eval_rag_quality.py` 用 RAGAS 框架对 LightRAG 的回答做四个维度打分：Faithfulness（忠实度）、Answer Relevance（答案相关性）、Context Recall（上下文召回）、Context Precision（上下文精确度）。

它的关键设计是**把被测系统当黑盒**：`generate_rag_response` 直接 `httpx` POST 到 `{rag_api_url}/query`，payload 固定 `mode="mix"`、`include_references=True`、`include_chunk_content=True`、`top_k` 取自 `EVAL_QUERY_TOP_K`（默认 10），再从返回的 `references[].content` 里抽出 chunk 作为 RAGAS 所需的 `contexts`。也就是说评估走的是真实的第 11 章 API 路由，而不是绕过去直接调内部函数——评的就是用户真正会拿到的东西。评估用的 LLM 与嵌入模型是另一套独立配置，通过 `EVAL_LLM_MODEL`、`EVAL_EMBEDDING_MODEL`、`EVAL_LLM_BINDING_HOST`、`EVAL_LLM_BINDING_API_KEY` 等环境变量注入，并设计了层层回退链（如嵌入 key 回退到 `EVAL_LLM_BINDING_API_KEY` 再回退到 `OPENAI_API_KEY`），方便把打分模型指到 vLLM/SGLang 等 OpenAI 兼容端点。

工程细节上，它用共享的 `httpx.AsyncClient`（带 `Timeout` 和 `Limits` 连接池）并发评估多个用例，对 `ConnectError`/`HTTPStatusError`/`ReadTimeout` 分别处理，结果同时落 CSV 和 JSON 到 `results/results_YYYYMMDD_HHMMSS.*`。`main()` 用 `argparse` 暴露 `--dataset/-d` 和 `--ragendpoint/-r` 两个参数，默认数据集是同目录的 `sample_dataset.json`。它甚至特意 `warnings.filterwarnings` 屏蔽了 `LangchainLLMWrapper is deprecated`——注释说明这是为了兼容所有 RAGAS 版本而刻意选用稳定 API 的取舍。

## 这些工具的共同模式

把六个文件横向看，能提炼出 LightRAG 在“运维工程”上的一致取向，这些取向本身就是值得复用的设计原则：

- **CLI 化、与服务解耦**：危险操作不进服务主流程，全部做成 `python -m lightrag.tools.*` 的独立入口，用同一套 `.env`/`config.ini` 配置，`load_dotenv(override=False)` 让 OS 变量优先。
- **流式 + 批处理**：迁移/清理/重建都按 `batch_size` 分批、用游标/scroll/LIMIT-OFFSET 分页，循环里 `asyncio.sleep(0)` 让出控制权，避免大数据量把内存或事件循环打爆。
- **源只读、可重跑**：`rebuild_vdb` 明说“sources are never modified”，迁移失败只记录不回滚源，所以工具天然幂等、失败可重试，这比脆弱的“事务回滚”更适合一次性运维。
- **删除/破坏前强确认**：清理和重建都有 `input("... (yes/no)")`，对 `JsonKVStorage` 还额外确认服务已关停。
- **预览优先（dry-run）**：`prepare_qdrant_legacy_data` 的每个写操作都被 `--dry-run` 短路，重建工具则提供只读的一致性检查模式。
- **逐批容错 + 报告**：用 dataclass（`MigrationStats`/`CleanupStats`/`_new_stats`）累计成功、失败、丢失记录与错误明细，结束打印结构化报告，并以退出码/`success` 传递整体结果给自动化。
- **依赖懒加载**：可视化和 Qdrant 工具用 `pipmaster` 按需安装重依赖，避免污染核心安装。
- **评估即黑盒回归**：`eval_rag_quality` 通过真实 `/query` API 评分，用独立的评估模型配置，把质量评估变成可重复执行的离线流程。

## 本章小结

- LightRAG 把 schema 演进、缓存格式变更、嵌入模型更换、存储后端迁移等“非主路径”操作集中在 `lightrag/tools/` 与 `lightrag/evaluation/`，做成独立 CLI 工具而非服务端点。
- `migrate_llm_cache.py` 在 5 种 KV 后端间迁移 `default:extract:*`/`default:summary:*` 缓存，默认走流式低内存模式，按 `DEFAULT_BATCH_SIZE=1000` 分批，逐批容错并产出 `MigrationStats` 报告。
- `clean_llm_query_cache.py` 按 `mode:cache_type` 组合清理查询缓存，提供 all/query/keywords 三档策略，删除后用 `set_all_update_flags` 置脏并落盘复查，对 `JsonKVStorage` 强制确认服务已关停。
- `rebuild_vdb.py` 把向量库视为图谱与 `text_chunks` KV 的派生物，drop 后逐字段对齐权威写入点重建，用 staged→rebuilt 的两阶段计数处理延迟嵌入后端，并提供只读一致性检查模式与“粘性失败”退出语义。
- `prepare_qdrant_legacy_data.py` 是唯一用完整 `argparse` flag 的工具，靠 `--dry-run` 全程预览、`scroll`+`Filter` 分批复制并剥离 `workspace_id`，专用于测试迁移逻辑。
- `lightrag_visualizer/graph_visualizer.py` 离线读取 GraphML，用 NetworkX + Louvain 社区发现 + spring 布局渲染 3D 图谱，重依赖通过 `pipmaster` 懒加载。
- `eval_rag_quality.py` 用 RAGAS 的四个指标，通过真实 `/query` API 黑盒评估回答质量，用独立的 `EVAL_*` 模型配置和层层回退链，结果落 CSV/JSON。
- 六个工具共享一套运维取向：源只读、流式分批、删除前确认、dry-run 预览、逐批容错加报告、依赖懒加载——失败可重跑优于脆弱的事务回滚。

## 动手实验

1. **实验一：追踪一条缓存 key 的迁移路径** — 在 `migrate_llm_cache.py` 里，从 `get_default_caches` 入口出发，对照 `get_default_caches_redis`、`get_default_caches_pg`、`get_default_caches_mongo` 三个实现，列出各自如何过滤出 `default:extract:`/`default:summary:` 前缀的 key、如何分页、以及如何把后端特有字段映射回统一缓存格式；用一句话总结为什么每种后端都要单独写一套。

2. **实验二：验证清理工具的“防数据丢失”链路** — 阅读 `clean_llm_query_cache.py` 的 `delete_query_caches`，找出 `del storage._data[key]` 之后调用 `set_all_update_flags` 的那几行，并解释如果删掉这一步，对 `JsonKVStorage` 会发生什么；再结合 `setup_storage` 里针对 `JsonKVStorage` 的二次确认，说明这两处共同防护的是同一类风险。

3. **实验三：理解 rebuild 的两阶段计数** — 在 `rebuild_vdb.py` 里通读 `_upsert_batch`、`_flush`、`_new_stats`，画出 `prepared → staged → rebuilt` 的状态流转，说明为什么对 nano/faiss 这类延迟嵌入后端必须等 `index_done_callback` 成功才把 `staged` 计入 `rebuilt`；再到 `run()` 末尾找到“粘性失败”逻辑，解释为何一次部分重建要让整个会话返回失败。

4. **实验四：跑通一次黑盒质量评估的请求构造** — 阅读 `eval_rag_quality.py` 的 `generate_rag_response`，写下它发往 `/query` 的完整 payload 字段（`mode`、`include_references`、`include_chunk_content`、`top_k` 等），并追踪返回的 `references[].content` 如何被展平成 RAGAS 的 `contexts`；再列出 `EVAL_LLM_*`/`EVAL_EMBEDDING_*` 的回退链，说明为什么评估模型要与被测系统的模型完全解耦。

> **下一章预告**：第 13 章将为全书收尾——回到工程视角，提炼 LightRAG 一以贯之的设计取向（权威源/派生物分层、可插拔存储、共享并发控制、配置驱动、离线运维工具），并跨越六部书横向归纳那些可迁移的 RAG 工程原则，作为全书结语。
