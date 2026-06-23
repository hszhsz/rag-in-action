# 第 7 章：知识库文档管理与数据库迁移

前面几章我们已经拆开了文档加载、中文切分、知识库服务抽象、向量缓存与嵌入。把这些零件拼起来，才构成 Langchain-Chatchat 真正对外的能力：一个知识库不只是一堆向量，它还要记得每个文件叫什么、什么时候改过、被切成了多少段、每一段对应向量库里的哪个 ID。

换句话说，一个文件从磁盘进入系统，会同时在**两个存储**里留下痕迹——向量库（FAISS/Milvus/PG 等）保存向量与原文，关系数据库（默认 SQLite 的 `info.db`）保存文件与分块的元数据。这两份记录必须始终对得上，否则就会出现"向量库里有、列表里看不到"或者"删了文件、向量还在"的幽灵数据。

本章聚焦这套"双写"机制。我们先看 API 层 `kb_doc_api.py` 如何编排上传、更新、删除、检索；再看仓储层 `knowledge_file_repository.py` 用 `with_session` 把关系库操作收进事务；最后看 `migrate.py` 里的 `folder2db`、`prune_db_docs`、`import_from_db` 等命令行工具，如何在批量构建、孤儿清理、库结构升级等场景下维持两份记录的一致。理解这一章，你就掌握了 Chatchat 知识库的数据生命周期。

## 两份记录：关系库里的三张表

数据库侧只有三个 ORM 模型，定义在 `knowledge_base_model.py` 与 `knowledge_file_model.py`。它们构成知识库、文件、分块三级元数据：

- `KnowledgeBaseModel`（表 `knowledge_base`）记录知识库本身：`kb_name`、`kb_info`（给 Agent 看的简介）、`vs_type`（向量库类型）、`embed_model`（嵌入模型名）、`file_count`（文件数量）。
- `KnowledgeFileModel`（表 `knowledge_file`）记录每个文件：除了 `file_name`、`file_ext`、`kb_name`，还有 `document_loader_name`、`text_splitter_name`、`file_version`、`file_mtime`、`file_size`、`custom_docs`、`docs_count`。
- `FileDocModel`（表 `file_doc`）是文件与向量库文档的桥梁：每条记录一个 `doc_id`（向量库里的文档 ID）加上 `meta_data`（JSON）。

此外 `knowledge_base_model.py` 还顺手定义了一个 Pydantic 的 `KnowledgeBaseSchema`（`from_attributes = True`），用于把 ORM 实例转成可序列化的响应对象。

这个分层很关键：`knowledge_file` 是"文件"粒度，`file_doc` 是"分块"粒度。一个文件被切成 N 段，就会产生 N 条 `FileDocModel`，它们共享同一个 `(kb_name, file_name)`，但各自的 `doc_id` 指向向量库里的一条向量。

`file_version` 字段尤其值得注意——它在 `add_file_to_db` 里每次重复入库都 `+1`，给同名文件的反复更新留下了版本痕迹。`docs_count` 则缓存了该文件的分块数，让 `list_files` 类查询不必去 `file_doc` 里 `count` 即可展示。

注意 `KnowledgeFileModel` 与向量库之间没有外键约束，`file_doc.doc_id` 是个普通 `String(50)`，与向量库里的真实 ID 也没有数据库层面的引用完整性。换言之，这套"双写"的一致性完全靠应用层代码自觉维护，而不是数据库帮你兜底。这正是本章要反复强调的主线：任何一处只写了一边、漏写了另一边，都会埋下幽灵数据。`FileDocModel.meta_data` 用的是 SQLAlchemy 的 `JSON` 类型，配合 `base.py` 里那个 `ensure_ascii=False` 的 `json_serializer`，让中文 metadata 直接以可读形式落库。

## with_session：把每次写入收进事务

仓储层所有函数的第一个参数都是 `session`，并戴着 `@with_session` 装饰器。`session.py` 里它的实现很短但很关键。

`with_session` 包装的函数体跑在 `session_scope()` 上下文里，正常结束就 `session.commit()`，抛异常就 `session.rollback()` 再 `raise`，最后 `finally` 里 `session.close()`。`session_scope` 本身是一个 `@contextmanager`，用 `SessionLocal()` 现造会话。

`SessionLocal` 在 `base.py` 由 `sessionmaker(autocommit=False, autoflush=False, bind=engine)` 生成，`engine` 则由 `create_engine` 连上 `Settings.basic_settings.SQLALCHEMY_DATABASE_URI`，并配了一个 `json_serializer`，用 `ensure_ascii=False` 序列化，保证 `meta_data` 里的中文不会被转义成 `\uXXXX`。这套写法是 SQLAlchemy 推荐的会话生命周期管理范式 [[Session Basics]](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)。`session.py` 里还另有 `get_db`（生成器，供 FastAPI 依赖注入用）与 `get_db0`（直接返回会话）两个变体，分别服务于不同的调用风格 [[FastAPI SQL Databases]](https://fastapi.tiangolo.com/tutorial/sql-databases/)。

有一个细节容易被忽略：`with_session` 的 wrapper 里调了一次 `session.commit()`，而 `session_scope` 退出时又会 `commit()` 一次——双重提交在 SQLAlchemy 里是幂等的（第二次没有待提交的变更）。

更值得玩味的是 `add_file_to_db` 这种"嵌套调用"：它内部又调了 `add_docs_to_db`，而后者也戴着 `@with_session`。由于 `with_session` 每次都新开一个 `SessionLocal()`，所以 `add_file_to_db` 与它调用的 `add_docs_to_db` 其实运行在**两个不同的会话**里。这意味着文件元数据与分块元数据并非严格的单一事务——若 `add_docs_to_db` 失败，已写入的 `KnowledgeFileModel` 不会自动回滚。这是该实现的一个已知折中，阅读源码时需要心里有数。

## 文件入库：add_file_to_db 的"更新或新增"

`add_file_to_db` 是双写里"写关系库"的核心。它先按 `kb_name` 找到对应的 `KnowledgeBaseModel`，然后查同名文件是否已存在（用 `ilike` 做大小写不敏感匹配）。

若 `existing_file` 存在，就走"更新"分支——刷新 `file_mtime`、`file_size`、`docs_count`、`custom_docs`，并把 `file_version += 1`，不动 `file_count`。若不存在，就 `new_file = KnowledgeFileModel(...)` 新建一条，`text_splitter_name` 缺省时回退到字符串 `"SpacyTextSplitter"`，同时把所属知识库的 `kb.file_count += 1`，再 `session.add(new_file)`。无论哪条分支，最后都调用 `add_docs_to_db` 把这一批分块（`doc_infos`，形式 `[{"id": str, "metadata": dict}, ...]`）逐条写成 `FileDocModel`。

`add_docs_to_db` 里有一行注释提醒：`doc_infos` 可能为 `None`，此时直接打印告警并返回 `False`，避免 `for d in doc_infos` 直接崩溃。这个防御性判断暗示着上游（向量库 `add_doc` 的返回值）在某些后端可能不返回 doc id，是双写链路里一处脆弱点。

文件信息从哪来？是 `KnowledgeFile` 对象（`utils.py`）提供的。它在 `__init__` 里就确定了 `filename`、`ext`、`filepath`、`document_loader_name`（由扩展名经 `get_LoaderClass` 推断）、`text_splitter_name`（取 `Settings.kb_settings.TEXT_SPLITTER_NAME`），并暴露 `get_mtime()`/`get_size()` 直接读磁盘文件属性。构造时还会校验扩展名是否在 `SUPPORTED_EXTS` 内，不支持的格式直接抛 `ValueError`——这也是 `file_to_kbfile` 用 `try/except` 包住构造、对坏文件"已跳过"的原因。

也就是说，关系库里的元数据是磁盘文件的"快照"，这正是后面 `prune` 类工具能用文件系统做基准来对账的前提。

## 上传链路：先落盘，再向量化

`upload_docs` 是上传的总入口。它先 `validate_kb_name` 防注入，再用 `KBServiceFactory.get_service_by_name` 拿到知识库服务。

第一步是 `_save_files_in_thread`——这是一个生成器，内部用 `run_in_thread_pool` 并发地把每个 `UploadFile` 写到 `get_file_path` 算出的磁盘路径。这里有个朴素但实用的去重判断：如果目标文件已存在、`override` 为假、且**磁盘文件大小恰好等于上传内容长度**，就认为是同一文件，返回 `code=404` 跳过写入。写入前还会 `os.makedirs` 补齐目录，异常则归类成 `code=500` 并记进结果。

文件落盘后，若 `to_vector_store` 为真，`upload_docs` 并不自己做向量化，而是**复用 `update_docs`**，把 `override_custom_docs=True`、`not_refresh_vs_cache=True` 透传过去，最后再由 `upload_docs` 统一调一次 `kb.save_vector_store()`。

这个"延迟刷新"的设计很重要：`not_refresh_vs_cache=True` 让中间每个文件的处理都不落盘向量库，等所有文件都进了内存再一次性持久化，对 FAISS 这种"全量写文件"的后端能省下大量重复 IO。这也是为什么贯穿全章的几乎每个 API 都带 `not_refresh_vs_cache` 参数。

## 多线程切分引擎：files2docs_in_thread 与 KnowledgeFile

值得单独拎出来讲的是切分的并发底座，因为上传、更新、重建、迁移四条链路全都共用它。

`KnowledgeFile`（`utils.py`）是"一个磁盘文件"的对象封装，它把"加载"与"切分"拆成可缓存的两级：`file2docs` 调对应 loader 把文件读成 `Document` 列表并缓存在 `self.docs`；`docs2texts` 再用 `make_text_splitter` 造出的切分器把 `Document` 切成分块，缓存在 `self.splited_docs`；`file2text` 则把两步串起来。

对 `.csv` 它跳过切分，对 `MarkdownHeaderTextSplitter` 走 `split_text`，其余走 `split_documents`，最后按 `zh_title_enhance` 决定是否做中文标题增强。`refresh` 参数控制是否绕过缓存重算——这正是上层"先在线程里 `file2text`、再把结果赋回 `splited_docs`"这种缓存复用手法的基础。

`files2docs_in_thread` 是批量并发的入口。它先把输入统一成 `KnowledgeFile`（支持 `KnowledgeFile` / `(filename, kb_name)` 元组 / dict 三种形态），组装成 kwargs 列表，再交给 `run_in_thread_pool`。

后者（`server/utils.py`）用 `ThreadPoolExecutor` 提交任务、`as_completed` 收结果，每个任务跑 `files2docs_in_thread_file2docs`——它把 `file2text` 包在 `try/except` 里，成功 yield `(True, (kb_name, filename, docs))`，失败 yield `(False, (kb_name, filename, msg))`。

这个"二元组 + 逐文件成败"的协议贯穿全章：`update_docs`、`folder2db.files2vs`、`recreate_vector_store` 都按 `if status:` 分流，成功者入库、失败者记录，从不让单个坏文件中断整批。线程安全的责任则下放给任务函数本身——`run_in_thread_pool` 的文档字符串明确要求"请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数"。

## 更新链路：多线程切分与自定义 docs 的分流

`update_docs` 是双写真正发生的地方，逻辑分三段。

第一段，它遍历 `file_names`，对每个文件先 `get_file_detail` 查关系库：如果该文件曾用过自定义 docs（`custom_docs` 为真）且本次没要求 `override_custom_docs`，就跳过，避免覆盖用户手工调过的分块；否则把不在 `docs` 字典里的文件包成 `KnowledgeFile` 收集进 `kb_files`。构造 `KnowledgeFile` 失败（如磁盘文件缺失）则记入 `failed_files`。

第二段，对 `kb_files` 调 `files2docs_in_thread`。这是切分的并发引擎（`utils.py`）：它把每个文件包装成关键字参数，丢给 `run_in_thread_pool`（`server/utils.py` 里基于 `ThreadPoolExecutor` + `as_completed` 的封装），每个线程跑 `files2docs_in_thread_file2docs` → `KnowledgeFile.file2text`，完成加载与切分。

`run_in_thread_pool` 用 `as_completed` 收结果，所以**返回顺序是完成顺序而非提交顺序**；它还在子线程异常时只 `logger.exception` 不中断，保证一个坏文件不拖垮整批。`files2docs_in_thread` 以 `(status, (kb_name, file_name, docs | error))` 的二元组 yield；入参既能是 `KnowledgeFile`，也能是 `(filename, kb_name)` 元组或带额外 kwargs 的 dict，相当灵活。

`update_docs` 拿到成功结果后，把切好的 `new_docs` 塞进 `kb_file.splited_docs`，再调 `kb.update_doc(...)`——是它最终触发向量库写入与 `add_file_to_db` 双写。注意这里巧妙利用了 `KnowledgeFile` 的缓存：先在线程里加载好 `Document`，再赋回 `splited_docs`，使得后续 `kb.update_doc` 不必重新读盘切分。

第三段，处理用户通过 `docs` 参数传入的自定义分块：把每条转成 langchain `Document`，调 `kb.update_doc(kb_file, docs=v, ...)`。这条路绕开了切分器，让调用方完全掌控分块内容。

三段都把失败收进 `failed_files` 字典返回，体现"尽力而为、逐文件报错"的批处理风格——这套 `failed_files` 约定同样出现在 `upload_docs` 与 `delete_docs` 的返回里，构成知识库写 API 统一的错误回传契约。

## 删除链路：先查存在，再删两边

`delete_docs` 对每个文件名先 `kb.exist_doc` 检查、不存在记入 `failed_files`，然后用 `KnowledgeFile` 包装调 `kb.delete_doc(kb_file, delete_content, not_refresh_vs_cache=True)`，最后统一 `save_vector_store`。

关系库侧对应的是 `delete_file_from_db`：它先删 `KnowledgeFileModel`，再调 `delete_docs_from_db` 删掉该文件的所有 `FileDocModel`，并把 `kb.file_count -= 1`。

`delete_docs_from_db` 用 `query.delete(synchronize_session=False)` 批量删除——`synchronize_session=False` 跳过会话内对象同步，适合"删完即弃"的场景，但要求调用方此后不再依赖会话里的过期对象 [[Deleting from Queries]](https://docs.sqlalchemy.org/en/20/orm/queries.html)。它在删除前还会先 `list_docs_from_db` 把待删文档查出来作为返回值，方便上层知道究竟删了哪些。

还有一个清库函数 `delete_files_from_db`，一次性删光某库的所有 `knowledge_file` 与 `file_doc` 并把 `file_count` 归零，供重建场景使用。

值得对比的是检索侧的读取路径。`search_docs` 在有 `query` 时走向量检索，把每条结果包成 `DocumentWithVSId`——这个模型（`kb_document_model.py`）继承 langchain `Document`，只多加了 `id`（向量库文档 ID）和 `score`（默认 `3.0`）两个字段，把"向量库里的身份"显式带回给前端。

若没有 `query` 只有 `file_name` 或 `metadata`，则走 `kb.list_docs` 从关系库按条件过滤，并顺手删掉 metadata 里的 `vector` 字段避免回传巨大向量。

仓储层的 `list_docs_from_db` 支持用 `ilike` 做大小写不敏感匹配，以及 `FileDocModel.meta_data[k].as_string() == str(v)` 对 JSON 字段做一级键过滤，这正是"按 metadata 检索"能落地的底层支撑。另有 `search_temp_docs` 走 `memo_faiss_pool` 直接在临时 FAISS 库里检索，服务于"文件对话"这种不落库的临时知识库。

## 路径安全与文件发现：对账的基准从哪来

prune 类工具能"拿文件系统当真理"对账，前提是有一套可靠的文件发现机制，而把磁盘暴露给网络请求又必须防住路径穿越。`utils.py` 里这几个函数共同撑起了这层地基。

`validate_kb_name` 是所有写 API 的第一道闸门——它逻辑极简，只检查知识库名里是否含 `"../"`，含则返回 `False`，调用方据此回 `code=403, msg="Don't attack me"`。`get_file_path` 是第二道闸门，也更严密：它把 `doc_path` 与目标文件名都 `Path(...).resolve()` 成绝对路径，再判断 `str(file_path).startswith(str(doc_path))`，只有目标确实落在知识库的 `content` 目录之内才返回路径，否则返回 `None`。这条 `startswith` 校验能挡住绝对路径、`..` 拼接等多种越界写入。

`list_files_from_folder` 则是"磁盘文件清单"的权威来源，也是 `folder2db`、`prune_*` 共同依赖的基准。它用 `os.scandir` 递归遍历 `content` 目录，有三个值得留意的细节：

- 其一，`is_skiped_path` 会跳过以 `temp`、`tmp`、`.`、`~$` 开头的条目，于是临时文件、隐藏文件、Office 锁文件不会被误当成知识文件。
- 其二，它显式处理符号链接（`entry.is_symlink()`），顺着 `realpath` 进入目标目录继续递归，让"把大语料软链进知识库"成为可能。
- 其三，所有路径都 `relpath` 后转成 posix 格式（`as_posix()`），与 `KnowledgeFile.__init__` 里 `Path(filename).as_posix()` 保持一致。

第三点尤其关键，因为对账靠的是 `set(db_files) - set(folder_files)` 这样的集合差，两边的文件名表示法必须逐字符相同，否则同一个文件会被误判为"只在一边存在"。

## 流式重建：recreate_vector_store 与进度回传

`kb_doc_api.py` 里还有一个特别的 API：`recreate_vector_store`。它解决的是"用户把文件直接拷进 `content` 目录、没走网络上传"的场景——此时数据库与向量库一无所知，需要从磁盘文件全量重建。

它的实现是一个 `EventSourceResponse`（SSE），内部 `output()` 是生成器。流程上：先 `get_service_by_name` 取库，取不到就用 `get_service(name, vs_type, embed_model)` 现造一个空库；`allow_empty_kb` 控制是否允许对"不在 `info.db`、或没有文档"的空库执行——这正是它能补建孤立目录的关键。

随后 `check_embed_model` 做嵌入模型探针，已存在的库先 `clear_vs()` 清空再 `create_kb()`，然后 `list_files_from_folder` 列出磁盘文件，喂给 `files2docs_in_thread` 并发切分。

每处理完一个文件就 `yield` 一段 JSON 进度（`total`/`finished`/`doc` 等），让前端能画进度条；失败文件 `yield` 一条 `code=500` 并 `logger.error` 后跳过。它还特意捕获 `asyncio.CancelledError`，在用户中断流时只打一条 warning 干净退出。

这种"边重建边回传进度"的设计 [[Server-Sent Events]](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)，是长耗时知识库操作在 Web 端的标准做法，与 `migrate.py` 里 `folder2db` 那套"`print` 到 stdout"的命令行风格形成对照。

## 迁移工具一：folder2db 的四种模式

`migrate.py` 提供的是命令行批处理能力，核心是 `folder2db`。它接收 `kb_names` 与一个 `mode`，`mode` 是 `Literal["recreate_vs", "update_in_db", "increment"]`；若 `kb_names` 为空则用 `list_kbs_from_folder()` 扫出所有库。

函数顶部定义了内部闭包 `files2vs`：对一批 `KnowledgeFile` 调 `files2docs_in_thread` 并发切分，成功的塞进 `splited_docs` 后 `kb.add_doc(..., not_refresh_vs_cache=True)`，把"切分 + 双写"打包复用。每个库处理前会先 `KBServiceFactory.get_service` 取服务、`kb.exists()` 不存在则 `create_kb()`。三种模式的差别在于**用什么作为入库文件的基准**：

- `recreate_vs`：先 `kb.clear_vs()` + `kb.create_kb()` 清空重建，再以 `list_files_from_folder(kb_name)`（磁盘目录）为基准全量重建。这是最彻底的"以文件系统为准"。
- `update_in_db`：以 `kb.list_files()`（关系库已登记的文件）为基准，用本地文件刷新它们的向量。适合切换嵌入模型或切分参数后重算，而不引入新文件。
- `increment`：把磁盘目录与数据库文件列表做集合差 `set(folder_files) - set(db_files)`，只对"磁盘有、库里没有"的新文件做增量向量化。
- 其他取值则落入 `else` 分支打印 `unsupported migrate mode`，不做任何动作。

源码里还保留着一段被注释掉的 `fill_info_only` 模式，注释解释了取消原因："现在数据库存了很多与文本切分相关的信息，单纯存储文件信息意义不大"——这正是双写设计演进的一个化石证据。

每个库处理完都打印一段统计：文件总数（`len(kb_files)`）、入库文件数（`len(result)`）、知识条目数（各文件 `docs` 长度之和）、用时，FAISS 类型还会额外打印 `kb_path`。这些 `print` 直出 stdout 而非走 logger，正符合它作为命令行迁移脚本的定位。

## 迁移工具二：prune 对账与库结构升级

`folder2db` 解决"文件进库"，`prune` 系列解决"对账清理"。

`prune_db_docs` 的语义是"删掉数据库里有、但本地文件夹已不存在的文档"——它算 `set(files_in_db) - set(files_in_folder)`，对差集里的每个文件调 `kb.delete_doc`，从而清除用户在文件浏览器里直接删文件后留下的"数据库孤儿"。

它的镜像是 `prune_folder_files`：算反方向的差集 `set(files_in_folder) - set(files_in_db)`，`os.remove(get_file_path(...))` 删掉"本地有、库里没有"的文件，用来回收磁盘空间。两者一起，把"磁盘"与"数据库"这两份文件清单拉回一致——而它们能成立，全靠前面 `list_files_from_folder` 与 `KnowledgeFile` 在文件名表示上的统一约定。

最后是结构升级场景的 `import_from_db`。当版本升级导致 `info.db` 表结构变化、但向量库无需重算时，它能从一个备份 SQLite 里把数据搬进新库。

实现上它用原生 `sqlite3` 连接备份库（`row_factory = sql.Row`），遍历 `Base.registry.mappers` 拿到所有 ORM 模型，按 `local_table.fullname` 逐表读取；备份库里不存在的表会被 `continue` 跳过。

对每行只保留模型里真实存在的列（`k in model.columns`），`create_time` 字段用 `dateutil.parser.parse` 还原成 `datetime`，再 `session.add(model.class_(**data))`。整段包在 `try/except` 里，失败打印路径与错误并返回 `False`，文档注释也写明"当前仅支持 sqlite"。

配套的还有 `create_tables`（`Base.metadata.create_all(bind=engine)`）与 `reset_tables`（先 `drop_all` 再 `create_tables`），分别用于初始化与彻底重置。值得一提的是 `migrate.py` 顶部刻意 import 了 `conversation_model`、`message_model`、`add_summary_to_db`、`create_mcp_profile` 等并不直接使用的符号——注释 `# ensure Models are imported` 道破了用意：只有这些模型类被导入、注册进 `Base.metadata`，`create_all` 才会真正为它们建表。这是 SQLAlchemy 声明式建表的一个常见陷阱。

## 本章小结

- 知识库采用"双写"：向量库存向量与原文，关系数据库（`info.db`）的三张表 `knowledge_base`/`knowledge_file`/`file_doc` 存知识库、文件、分块三级元数据；两者无外键，一致性靠应用层维护。
- `FileDocModel` 是文件与向量库文档的桥梁，每条记一个 `doc_id`；`DocumentWithVSId` 在检索结果里把这个 `id` 与 `score` 显式带回前端。
- `with_session` 装饰器统一了"开会话—执行—提交/回滚—关闭"的事务范式，`engine` 配 `json_serializer(ensure_ascii=False)` 保证 JSON 元数据里的中文不被转义。
- `with_session` 每次新开 `SessionLocal()`，因此 `add_file_to_db` 与其内部调用的 `add_docs_to_db` 处于不同会话、并非单一事务——这是一处需要注意的一致性折中。
- `add_file_to_db` 走"存在则更新并 `file_version += 1`、否则新增并 `file_count += 1`"逻辑；元数据是 `KnowledgeFile` 对磁盘文件的快照（`get_mtime`/`get_size`）。
- `upload_docs` 先并发落盘（按文件大小做朴素去重）再复用 `update_docs` 向量化，全程 `not_refresh_vs_cache=True` 延迟到最后统一 `save_vector_store`，对 FAISS 显著省 IO。
- `files2docs_in_thread` 基于 `run_in_thread_pool`（`ThreadPoolExecutor` + `as_completed`）并发切分，按完成顺序返回、单文件失败不拖垮整批，并复用 `KnowledgeFile.splited_docs` 缓存避免重复读盘。
- `update_docs` 三段式：跳过/覆盖自定义 docs、多线程切分入库、处理调用方传入的自定义 `Document`，失败逐文件收进 `failed_files`。
- 删除链路双删：`delete_file_from_db` 删 `KnowledgeFileModel` 再删全部 `FileDocModel` 并维护 `file_count`，批量删除用 `synchronize_session=False`。
- `folder2db` 的 `recreate_vs`/`update_in_db`/`increment` 三模式分别以磁盘目录、数据库列表、二者差集为基准；`prune_db_docs`/`prune_folder_files` 做双向对账，`import_from_db` 支持结构升级时无需重算的数据迁移。

## 动手实验

1. **实验一：观察双写的两条记录** — 通过 `upload_docs` 上传一个文本文件到测试知识库，然后用 `sqlite3` 打开 `info.db`，分别 `select * from knowledge_file where kb_name=...` 和 `select count(*) from file_doc where file_name=...`，确认 `knowledge_file.docs_count` 与 `file_doc` 行数一致；同名文件再传一次，观察 `file_version` 是否 `+1`。
2. **实验二：复现并验证孤儿清理** — 直接到知识库目录手动 `rm` 掉一个已入库的文件（绕过 API），调用 `search_docs`/`list_files` 观察"幽灵记录"仍在；再运行 `prune_db_docs([kb_name])`，确认 `knowledge_file` 与 `file_doc` 里对应记录被删、`file_count` 同步减少。
3. **实验三：对比 folder2db 三种模式** — 准备一个含 3 个文件的目录，先 `folder2db(mode="recreate_vs")` 全量建库；再往目录里加 1 个新文件，分别用 `increment` 与 `update_in_db` 跑一遍，打印统计行里的"入库文件数"，理解前者只处理新文件、后者只刷新库内已有文件。
4. **实验四：体会延迟刷新的代价** — 在 `update_docs` 里把传给 `kb.update_doc` 的 `not_refresh_vs_cache` 临时改为 `False`（即每个文件都立刻 `save_vector_store`），用 FAISS 后端批量更新 10 个文件并计时；改回 `True`（仅末尾刷新一次）再计时，比较两者耗时，理解为何 API 默认延迟刷新。

> **下一章预告**：文档已经稳稳地存进了向量库与关系库，下一步就是把它们检索出来。第 8 章《检索器与重排》将解构 Chatchat 如何从向量库召回候选、如何用重排模型（Reranker）对召回结果二次打分排序，以及 `score_threshold`、`top_k` 等参数如何在召回质量与召回数量之间权衡。
