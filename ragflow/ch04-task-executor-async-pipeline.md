# 第 4 章 任务执行器与异步管线

前几章我们沿着"文档进来、被理解、被切分"的路径，看清了 RAGFlow 如何把一个 PDF 或 Word 拆成结构化的 chunk。但那些解析与分块逻辑并不是被某个 HTTP 请求同步调用的——用户点下"解析"按钮后立刻得到的只是一句"已加入队列"，真正的重活发生在另一组进程里。这一章要解构的就是这组进程：后台任务执行器 `task_executor.py`。它是一个常驻的异步 worker，从 Redis 队列里捞任务、调用解析/分块/向量化、把结果写进文档存储，并一路把进度回写给前端。

把视角放到工程层面，这其实是一条典型的异步处理管线（pipeline）：生产者（API 服务）把任务塞进队列，消费者（执行器）以受控的并发把任务跑完。RAGFlow 在这条管线上做了不少耐人寻味的设计——优先级双队列、Redis Stream 消费组、未确认消息重放、多级信号量限流、线程化超时、两级进度聚合，以及一套"新旧两套实现并行跑、逐字段对账"的重构迁移方案。我们逐一拆开来看。

## 任务从哪来：优先级双队列与消费组

执行器消费的是 Redis Stream。队列名由 `common/settings.py` 的 `get_svr_queue_name` 生成，模板是 `{SVR_QUEUE_NAME}.{priority}.common`，而 `SVR_QUEUE_NAME` 在 `common/constants.py` 里被定为字符串 `"te"`（task executor 的缩写）。于是实际存在的队列就是 `te.1.common` 和 `te.0.common` 两条——priority 为 `1` 代表高优先级，`0` 代表低优先级。注意函数签名虽然带了 `suffix` 参数，注释里也列了 `resume/graphrag/raptor/mindmap` 等候选，但当前实现里 suffix 被硬编码成了 `common`，其余后缀是"预留"状态：

```python
def get_svr_queue_name(priority: int, suffix: str = "common") -> str:
    ...
    return f"{SVR_QUEUE_NAME}.{priority}.common"


def get_svr_queue_names(suffix:str):
    """Return queue names sorted by priority (high to low)."""
    return [get_svr_queue_name(priority, suffix) for priority in [1, 0]]
```

`get_svr_queue_names` 返回的列表顺序是 `[1, 0]`，也就是"高优先级在前"。这个顺序在消费侧被严格利用：执行器会先尝试从 `te.1.common` 取消息，取不到才退到 `te.0.common`。优先级在这里不是靠单队列内的排序实现的，而是用两条独立队列加"先查高优队列"的轮询顺序来近似——这是一种简单可靠、不依赖 Redis 内部排序能力的做法。

生产端在 `api/db/services/task_service.py` 里，通过 `REDIS_CONN.queue_product(settings.get_svr_queue_name(priority, "common"), message=task)` 把任务投递进去；`queue_product` 内部就是对 Redis 的 `XADD`（见 `rag/utils/redis_conn.py`）：

```python
def queue_product(self, queue, message) -> bool:
    for _ in range(3):
        try:
            payload = {"message": json.dumps(message)}
            self.REDIS.xadd(queue, payload)
            return True
        ...
    return False
```

消费侧用的是 Redis Stream 的消费组（consumer group）机制。组名在 `common/constants.py` 里写死为 `SVR_CONSUMER_GROUP_NAME = "rag_flow_svr_task_broker"`，所有执行器进程都加入这同一个组，因此一条消息只会被组内某一个 consumer 拿到，天然实现了多 worker 之间的负载分摊与去重。每个执行器实例的 consumer 名字在进程启动时拼出来，格式是 `task_executor_{TASK_TYPE}_{TE_IDX}`，其中 `TASK_TYPE` 默认 `common`、`TE_IDX` 默认 `0`，二者都能用命令行参数 `-t/--type` 与 `-i/--index` 覆盖：

```python
TASK_TYPE = args.type
TE_IDX = args.index
CONSUMER_NAME = f"task_executor_{TASK_TYPE}_{TE_IDX}"
```

`queue_consumer` 是实际取消息的地方，它封装了 `XREADGROUP`。一个值得注意的健壮性细节是：它每次取消息前都会先确认消费组存在，不存在就用 `xgroup_create(..., id="0", mkstream=True)` 现场创建，并且对 `BUSYGROUP`（组已存在）这类竞态错误做了吞掉处理。读取参数里 `count=1`、`block=5`，即每次只取一条、最多阻塞 5 毫秒，这让消费循环既能及时拿到新任务，又不会在空队列上长时间挂死。

## 未确认消息的重放：崩溃后不丢任务

执行器并不是简单地"取一条、跑一条、确认一条"。它在 `collect()` 里区分了两种来源：一种是该 consumer 名下**尚未 ack 的历史消息**，另一种才是**队列里的新消息**。前者通过 `get_unacked_iterator` 拿到：

```python
async def collect():
    ...
    svr_queue_names = settings.get_svr_queue_names(TASK_TYPE)
    redis_msg = None
    try:
        if not UNACKED_ITERATOR:
            UNACKED_ITERATOR = REDIS_CONN.get_unacked_iterator(svr_queue_names, SVR_CONSUMER_GROUP_NAME, CONSUMER_NAME)
        try:
            redis_msg = next(UNACKED_ITERATOR)
        except StopIteration:
            for svr_queue_name in svr_queue_names:
                redis_msg = REDIS_CONN.queue_consumer(svr_queue_name, SVR_CONSUMER_GROUP_NAME, CONSUMER_NAME)
                if redis_msg:
                    break
```

逻辑上是这样：进程起来后先把"上一辈子没干完的活"全部捞出来重放（用一个 `0` 起始游标遍历该 consumer 的 pending 消息，见 `get_unacked_iterator` 里 `current_min = 0` 与 `queue_consumer(..., current_min)` 的配合），等 `UNACKED_ITERATOR` 耗尽（`StopIteration`）后，才按 `[te.1.common, te.0.common]` 的顺序去消费新消息。这正是 Redis Stream 消费组相对于普通 list 队列的核心优势——消息被读取后进入 Pending Entries List，只有显式 `XACK` 才算消费完成；执行器进程若在处理途中崩溃，这条消息仍挂在它的 pending 列表里，下次重启就能被同一 consumer 重新认领。换句话说，RAGFlow 选择了"至少一次"投递语义来换取任务不丢失。

确认动作被刻意推迟到了任务真正处理完之后。在 `handle_task()` 的最末尾，无论成功、被取消还是抛异常，`finally` 之后才执行 `redis_msg.ack()`：

```python
    finally:
        if not task.get("dataflow_id", ""):
            ...
            PipelineOperationLogService.record_pipeline_operation(...)

    redis_msg.ack()
```

而 `ack()` 内部就是 `XACK`：`self.__consumer.xack(self.__queue_name, self.__group_name, self.__msg_id)`。把 ack 放在最后，是"先干完活再确认"的可靠队列范式；代价是如果任务执行到一半进程被杀，重启后会整条重跑（幂等性需要由下游写入来保证，比如 chunk 的 `id` 是用内容 + doc_id 做的 `xxhash` 哈希，重复写入会覆盖而非追加）。

`collect()` 拿到原始消息后并不会直接把它当任务用，而是回数据库取权威任务记录：普通任务走 `TaskService.get_task(msg["id"])`，memory 类型走 `TaskService.get_by_id`，几个特殊的 fake doc id（`GRAPH_RAPTOR_FAKE_DOC_ID`、`CANVAS_DEBUG_DOC_ID`）有专门分支。取回后立刻调 `has_canceled(task["id"])` 判断任务是否已被用户取消——若任务不存在或已取消，直接 `redis_msg.ack()` 丢弃、`FAILED_TASKS += 1`，不浪费算力。这一步把"队列消息"与"数据库任务状态"做了一次对齐，避免执行早已作废的任务。

## 一个文档任务的完整生命周期

把 `te.x.common` 里一条标准解析任务的旅程串起来，主线是 `do_handle_task(task)`（旧实现）或重构版的 `TaskHandler.handle_task()`（默认实现，下一节细说），二者流程基本同构。以旧实现为例，标准分块分支（`else` 分支，即非 raptor/graphrag/dataflow/memory）大致是：绑定 embedding 模型、初始化索引、`build_chunks`、`embedding`、`insert_chunks`、累加统计。

第一步是绑定向量模型并探测维度。代码先拿到 embedding 配置构造 `LLMBundle`，然后用一句 `embedding_model.encode(["ok"])` 编码一个占位词，纯粹为了量出向量维度 `vector_size`：

```python
    embedding_model = LLMBundle(task_tenant_id, embd_model_config, lang=task_language)
    vts, _ = embedding_model.encode(["ok"])
    vector_size = len(vts[0])
    ...
    init_kb(task, vector_size)
```

拿到维度后立刻 `init_kb` 建好对应维度的索引——这是因为向量字段名是带维度的（后面会看到 `q_%d_vec`），必须先知道维度才能建表/建索引。

第二步 `build_chunks` 是真正调用第 2、3 章那套解析与分块逻辑的地方。它先校验文件大小不超过 `settings.DOC_MAXIMUM_SIZE`（默认 `128 * 1024 * 1024`，即 128MB），再用全局 `FACTORY` 字典按 `parser_id` 选出对应解析模块（`naive/paper/book/table/...`），从对象存储拉二进制，最后在 `chunk_limiter` 信号量保护下，把同步的 `chunker.chunk(...)` 丢进线程池执行：

```python
    chunker = FACTORY[task["parser_id"].lower()]
    ...
    async with chunk_limiter:
        task_language = task.get("language") or "Chinese"
        cks = await thread_pool_exec(
            chunker.chunk,
            task["name"],
            binary=binary,
            from_page=task["from_page"],
            to_page=task["to_page"],
            lang=task_language,
            callback=progress_callback,
            ...
        )
```

这里有两个细节体现了"异步框架包裹同步重活"的工程取舍。其一，解析器本身是 CPU 密集的同步函数，直接在事件循环里调用会卡死整个 worker，所以用 `thread_pool_exec` 把它扔到线程池；其二，`chunk` 把 `callback=progress_callback` 透传了进去，意味着解析器在分页处理时能持续回吐进度，用户在前端看到的"Page(3~5): ..."这类细粒度进度正是从解析器内部冒出来的。

分块完成后，`build_chunks` 还会为每个 chunk 计算稳定 id 并把图片落库。id 用的是内容加 doc_id 的 64 位 xxhash：

```python
    d["id"] = xxhash.xxh64((chunk["content_with_weight"] + str(d["doc_id"])).encode("utf-8", "surrogatepass")).hexdigest()
```

内容相同的 chunk 会得到相同 id，这正是前面说的"重跑覆盖而非重复"的幂等基础。图片上传被包成一批 `asyncio.create_task` 并发执行，再 `asyncio.gather` 汇合——CPU 解析走线程池、IO 上传走协程并发，分工清晰。

第三步是向量化，进入 `embedding(docs, mdl, ...)`。它把每个 chunk 的标题与正文分别准备好，标题只编码一次再用 `np.tile` 复制到与 chunk 等量，正文按 `settings.EMBEDDING_BATCH_SIZE`（默认 16）分批编码，每批都在 `embed_limiter` 保护下走线程池。最终标题向量与正文向量按权重线性融合：

```python
    title_w = float(filename_embd_weight)
    if tts.ndim == 2 and cnts.ndim == 2 and tts.shape == cnts.shape:
        vects = title_w * tts + (1 - title_w) * cnts
    else:
        vects = cnts
    ...
    for i, d in enumerate(docs):
        v = vects[i].tolist()
        vector_size = len(v)
        d["q_%d_vec" % len(v)] = v
```

`filename_embd_weight` 默认 0.1（且对数据库存的 None 值做了兜底），即文件名标题只占一成权重、正文占九成。向量被写进 `q_{维度}_vec` 这样的动态字段——这就解释了为什么前面要先探测维度并据此 `init_kb`。融合标题的意图很实际：让"文件名里出现的关键词"也能对召回产生一点贡献，但又不至于喧宾夺主。

第四步 `insert_chunks` 把带向量的 chunk 写进文档存储。它通过 `settings.docStoreConn.insert(...)` 抽象层写入（底层可能是 Elasticsearch 或 Infinity，是第 5 章的主题），按 `settings.DOC_BULK_SIZE`（默认 4）分批，每批之间都重新检查 `has_canceled`，一旦发现取消就回滚已写入的部分（尤其对 RAPTOR 摘要 chunk 做了显式删除回滚，避免半成品被误判为已完成的 checkpoint）。每批写完还会调 `TaskService.update_chunk_ids` 把已落库的 chunk id 记到任务上。

收尾阶段累加文档级统计、回写最终进度。`do_handle_task` 用 `DocumentService.increment_chunk_num(task_doc_id, task_dataset_id, token_count, chunk_count, 0)` 把这次产出的 token 数与 chunk 数原子累加到文档记录，然后 `progress_callback(prog=1.0, msg="Task done (...)")` 标记完成。整个 `do_handle_task` 还套了一层 `@timeout(60 * 60 * 3, 1)`，即单任务硬上限 3 小时。

值得单独点出的是它的 `finally` 块：如果任务在任何环节被取消，执行器会主动把已经写进文档存储的该 doc 的 chunk 全部删掉，保证"取消"是干净的，不留半截索引：

```python
    finally:
        ...
        if has_canceled(task_id):
            try:
                exists = await thread_pool_exec(settings.docStoreConn.index_exist, search.index_name(task_tenant_id), task_dataset_id)
                if exists:
                    ret = await thread_pool_exec(settings.docStoreConn.delete, {"doc_id": task_doc_id}, search.index_name(task_tenant_id), task_dataset_id)
```

## 进度怎么回写：单任务即时写 vs. 文档级聚合

RAGFlow 的进度有两个层级，理解它们的分工是读懂这条管线的关键。

第一级是**单任务进度**，由执行器在处理过程中通过 `set_progress(task_id, ...)` 即时写入。这个函数在 `task_executor.py` 顶部定义，做了几件事：把 `has_canceled` 检查内联进来（一旦发现取消就把进度强制改成 `-1` 并在消息后追加 `[Canceled]`，最后抛 `TaskCanceledException` 让上层中断）、给消息打上时间戳与页码前缀、然后落库：

```python
def set_progress(task_id, from_page=0, to_page=-1, prog=None, msg="Processing..."):
    try:
        if prog is not None and prog < 0:
            msg = "[ERROR]" + msg
        cancel = has_canceled(task_id)
        if cancel:
            msg += " [Canceled]"
            prog = -1
        ...
        d = {"progress_msg": msg}
        if prog is not None:
            d["progress"] = prog
        TaskService.update_progress(task_id, d)
        close_connection()
        if cancel:
            raise TaskCanceledException(msg)
```

约定俗成的语义是：`prog` 为 `1.0` 表示完成，`-1` 表示失败/取消，`0~1` 之间表示进行中的百分比。各阶段会调用它推进百分比，比如 `embedding` 内部每编码完一批就 `callback(prog=0.7 + 0.2 * (i + 1) / len(cnts), ...)`，`insert_chunks` 写入阶段则推到 `0.8 + 0.1 * ...`——可以看出 0.7 之前留给解析分块、0.7~0.9 给向量化与写入、最后冲到 1.0。`do_handle_task` 在进入标准分块前用 `partial(set_progress, task_id, task_from_page, task_to_page)` 把任务 id 和页码绑定成一个 `progress_callback`，往后所有阶段共用这一个回调，消息里就自动带上了 `Page(a~b)` 前缀。这是一种很轻量的依赖注入：底层模块不需要知道任务 id，只管调 `callback`。

第二级是**文档级聚合进度**。一个文档可能被拆成多个任务（比如按页分片、或额外的 raptor/graphrag/mindmap 任务），前端要看的是"这个文档整体到哪了"。这个聚合发生在 `api/db/services/document_service.py` 的 `_sync_progress`：它把某文档名下所有任务查出来，求平均进度，并据此推导文档级状态：

```python
    for t in tsks:
        ...
        if 0 <= t.progress < 1:
            finished = False
        if t.progress == -1:
            bad += 1
        prg += t.progress if t.progress >= 0 else 0
        ...
    prg /= len(tsks)
    if finished and bad:
        prg = -1
        status = TaskStatus.FAIL.value
    elif finished:
        prg = 1
        status = TaskStatus.DONE.value
    elif not finished:
        status = TaskStatus.RUNNING.value
```

聚合规则很直白：只要有任意子任务进度落在 `[0,1)` 就算整体未完成；全部完成但有人是 `-1`（`bad`）则整篇判失败；全部完成且无失败才置为 `DONE` 并把进度钉到 1。它还把各任务的 `progress_msg` 排序后拼接展示，并在消息末尾追加"还有几个任务排在队列前面"的提示（`get_queue_length(priority)`）。这套两级设计把"执行器只管写自己那条任务的进度"和"前端看到的文档整体进度"解耦开了——执行器侧实现极简，聚合的复杂度全交给查询侧按需计算。

## 并发与限流：信号量、线程化超时与启动错峰

执行器的并发模型是单进程内的 asyncio 事件循环，靠多个信号量（`task_executor_limiter.py` 里基于 `LoopLocalSemaphore` 构造）来分别限制不同环节的并发度：

```python
MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", "5"))
MAX_CONCURRENT_CHUNK_BUILDERS = int(os.environ.get("MAX_CONCURRENT_CHUNK_BUILDERS", "1"))
MAX_CONCURRENT_MINIO = int(os.environ.get("MAX_CONCURRENT_MINIO", "10"))

task_limiter = LoopLocalSemaphore(MAX_CONCURRENT_TASKS)
chunk_limiter = LoopLocalSemaphore(MAX_CONCURRENT_CHUNK_BUILDERS)
embed_limiter = LoopLocalSemaphore(MAX_CONCURRENT_CHUNK_BUILDERS)
minio_limiter = LoopLocalSemaphore(MAX_CONCURRENT_MINIO)
kg_limiter = LoopLocalSemaphore(2)
```

这组默认值透露了设计者对各环节资源画像的判断：整体最多并发 5 个任务（`task_limiter`），但 chunk 构建与 embedding 各自只允许 1 个并发（`chunk_limiter`/`embed_limiter` 都取 `MAX_CONCURRENT_CHUNK_BUILDERS=1`）——因为它们最吃 CPU/GPU 且常依赖模型，多开反而互相挤兑；对象存储 IO 不吃算力，放到 10 个并发（`minio_limiter`）；知识图谱构建居中，给 2 个。`main()` 的主循环正是用 `task_limiter` 当闸门的：

```python
    while not stop_event.is_set():
        await task_limiter.acquire()
        t = asyncio.create_task(task_manager())
        tasks.append(t)
```

每开一个任务协程前先 `acquire`，协程在 `task_manager()` 的 `finally` 里 `task_limiter.release()`。当 5 个许可全部被占，主循环就阻塞在 `acquire()` 上，不再 `collect` 新任务——这是一个优雅的背压（backpressure）机制：执行器永远不会同时拉超过它能处理的任务量进来。

超时控制走的是 `common/connection_utils.py` 的 `timeout` 装饰器，它对同步函数和协程做了不同处理。对同步函数（如 `build_chunks` 内部的 `upload_to_minio`、`batch_encode`），它把目标函数丢进一个 daemon 线程跑，主线程用 `result_queue.get(timeout=seconds)` 等结果，超时就重试（默认 `attempts=2`）、最终抛 `TimeoutError`：

```python
def timeout(seconds=None, attempts=2, *, exception=None, on_timeout=None):
    ...
    def wrapper(*args, **kwargs):
        result_queue = queue.Queue(maxsize=1)
        def target():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                result_queue.put(e)
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        for a in range(attempts):
            ...
```

一个容易被忽略的开关是：实际的超时计时只在环境变量 `ENABLE_TIMEOUT_ASSERTION` 被设置时才生效（`result_queue.get(timeout=seconds)`），否则退化成无限等待的 `result_queue.get()`。协程版同理，未开启断言时直接 `await func(...)`、不套 `asyncio.wait_for`。这是个务实的取舍——生产环境默认不强切超时，避免误杀正在推进的长任务，需要时再用环境变量打开。各处的超时数字也分了层：`build_chunks` 整体 `@timeout(60 * 80, 1)`（80 分钟、只试 1 次），`upload_to_minio` 60 秒，`batch_encode` 60 秒，`do_handle_task` 3 小时。

最后是启动错峰。多个执行器副本同时拉起时，会同一瞬间冲击 Infinity/ES 的连接池，`main()` 因此根据 consumer 名字尾号做了递增延迟：

```python
    worker_num = int(CONSUMER_NAME.rsplit("_", 1)[-1])
    startup_delay = worker_num * 2.0 + random.uniform(0, 0.5)
    if startup_delay > 0:
        logging.info(f"Staggering startup by {startup_delay:.2f}s ...")
        await asyncio.sleep(startup_delay)
```

第 N 个 worker 多睡 `N * 2` 秒再加一点随机抖动，把连接建立的尖峰摊平到几秒里——一个很小但很贴合运维现实的细节。

## 心跳与僵尸清理：执行器的自我治理

执行器跑起来后还有一个常驻协程 `report_status()`，由 `main()` 用 `asyncio.create_task(report_status())` 拉起，负责周期性上报心跳并清理死掉的同伴。它启动时把自己注册进 Redis 集合 `TASKEXE`（`REDIS_CONN.sadd("TASKEXE", CONSUMER_NAME)`），然后在循环里用 `ZADD` 把一份 JSON 心跳写到以自己 consumer 名命名的有序集合里，分数是时间戳：

```python
    REDIS_CONN.sadd("TASKEXE", CONSUMER_NAME)
    redis_lock = RedisDistributedLock("clean_task_executor", lock_value=CONSUMER_NAME, timeout=60)
    while True:
        ...
        group_info = REDIS_CONN.queue_info(settings.get_svr_queue_name(0), SVR_CONSUMER_GROUP_NAME) or {}
        PENDING_TASKS = int(group_info.get("pending", 0))
        LAG_TASKS = int(group_info.get("lag", 0))
        ...
        REDIS_CONN.zadd(CONSUMER_NAME, heartbeat, now_ts)
        ...
        REDIS_CONN.zremrangebyscore(CONSUMER_NAME, 0, now_ts - 60 * 30)
```

心跳 payload 里塞了 ip、pid、boot 时间、当前 pending/lag 任务数、累计 done/failed 计数，以及 `CURRENT_TASKS`（当前正在处理的任务快照）。这些数据让运维面板能看清每个执行器在干什么、积压多少。`pending` 与 `lag` 直接取自 `queue_info`（即 `XINFO GROUPS` 返回的消费组信息），等于把 Redis Stream 的积压指标透出给了监控。

清理僵尸执行器的逻辑用 `RedisDistributedLock` 做互斥——任意时刻只让一个执行器承担清理职责。这个分布式锁的实现（`rag/utils/redis_conn.py`）很讲究：`acquire()` 前先用一段 Lua 脚本 `delete_if_equal` 把"自己之前持有的同名锁"清掉再抢，`release()` 同样只删值等于自己 `lock_value` 的锁，避免误删别人持有的锁：

```python
class RedisDistributedLock:
    def acquire(self):
        REDIS_CONN.delete_if_equal(self.lock_key, self.lock_value)
        return self.lock.acquire(token=self.lock_value)
    def release(self):
        REDIS_CONN.delete_if_equal(self.lock_key, self.lock_value)
```

拿到锁的那个执行器会遍历 `TASKEXE` 集合里所有同伴，对长时间没更新心跳的（超过 `WORKER_HEARTBEAT_TIMEOUT`，默认 120 秒）做清理。这是一套去中心化的自治：没有专门的协调进程，执行器们靠 Redis 锁轮流值班打扫，谁抢到锁谁负责。

## 新旧两套实现：在线对账式的重构迁移

读 `task_executor.py` 时最容易困惑的一点是：它在文件顶部就 `from rag.svr.task_executor_refactor.task_manager import TaskManager`，而且 `handle_task()` 里居然存在三条执行路径。这背后是一次正在进行中的大重构。`task_executor_refactor/task_executor_refactoring_plan.md` 把动机讲得很清楚：原始 `task_executor.py` 是约 1780 行的"上帝文件"，一个文件扛了任务消费、分块、向量化、索引、RAPTOR/GraphRAG、心跳七八种职责，充斥全局变量（`DONE_TASKS`/`FAILED_TASKS`/`CURRENT_TASKS`），与 `TaskService`/`DocumentService`/`REDIS_CONN` 紧耦合，几乎无法单元测试。

重构方案把它拆成了分层架构：业务层 `task_handler.py`、服务层（`ChunkService`/`EmbeddingService`/`RaptorService`/`DataflowService` 等）、基础设施层（`TaskContext`/`RecordingContext`/`Comparator` 等）。核心手法是依赖注入——所有服务通过构造函数接收一个 `TaskContext`，不再直接摸全局状态。`TaskContext` 用 `TaskLimiters`、`TaskCallbacks` 两个 dataclass 把限流器与回调打包进来，再加上对任务字典各字段的类型化访问器（`TaskDict` 是个 `TypedDict`）。于是原来散落各处的 `set_progress`、`has_canceled`、五个信号量，现在都成了从 `ctx` 上取的属性，服务对象因此可以被轻松 mock 和测试。

真正精彩的是它的迁移策略——不是"改完直接切"，而是**新旧并行跑、逐字段对账**。`handle_task()` 由环境变量 `TE_RUN_MODE` 控制走哪条路：

```python
    run_mode = os.environ.get("TE_RUN_MODE", "0")
    if run_mode == "1":  # dry run mode - compare
        set_recording_context(RecordingContext())
        await do_handle_task(task)  # original execution
        await TaskManager.dry_run_task(task, get_recording_context(), ...)
    elif run_mode == "0":  # use refactor-ed version
        set_recording_context(NullRecordingContext())
        await TaskManager.run_refactored_task(task, ...)
    else:  # original version
        set_recording_context(NullRecordingContext())
        await do_handle_task(task)
```

`TE_RUN_MODE` 默认 `"0"`，即生产默认已经走重构版 `run_refactored_task`；设为非 0 非 1 的值则回退到旧版 `do_handle_task` 作为后路。最有意思的是 `"1"`（dry run）模式：它**先用旧实现真跑一遍**，过程中由 `RecordingContext` 录下所有中间结果和写操作的返回值；然后用 `TaskManager.dry_run_task` 把同一个任务再用新实现跑一遍，但新实现的所有写操作（`docStoreConn.insert`、`DocumentService.increment_chunk_num` 等）不真正落库，而是被 `WriteOperationInterceptor` 拦截，回放旧实现录下的返回值；最后用 `ContextComparator` 把新旧两套中间结果逐 key 比对，打印差异报告。

`WriteOperationInterceptor` 维护了一份白名单 `ALLOWED_METHOD_NAMES`（`KnowledgebaseService.update_by_id`、`TaskService.update_chunk_ids`、`DocumentService.increment_chunk_num`、`docStoreConn.insert`、`docStoreConn.delete` 等），只有这些写方法允许被拦截回放，`intercept` 按 FIFO 从录制列表里 `pop(0)` 取值。这种设计的妙处在于：dry run 的目的是验证"新实现产出的数据是否与旧实现一致"，但又绝不能让验证过程真的污染数据库，所以写操作必须被截流、用旧值替身回放，让新实现在"幻觉中"完整跑完，再比对它在内存里算出的结果。`recording_context_manager` 还在生产 `run_refactored_task` 里用 `NullRecordingContext`（即 `_NULL_RECORDING_CONTEXT`）替换真正的录制器，避免生产环境为了对账白白消耗内存——plan.md 也把"RecordingContext 存中间结果导致内存增长"明确列为需监控的风险项。

plan.md 给出的迁移分三阶段：Phase 1（双路径并行、对账框架可用）已完成，Phase 2（默认切到重构版、旧码留作回退）也已经体现在 `TE_RUN_MODE` 默认 `"0"` 上，Phase 3（验证期后删除旧码）仍待进行。这意味着我们现在读到的 `task_executor.py` 正处在一个过渡态：旧的 `do_handle_task` 还活着，但只在 `TE_RUN_MODE` 显式指定或 dry run 对照时才被执行；日常生产流量已经走在重构后的 `TaskHandler` 上。值得一提的是，重构版 `TaskHandler.handle()` 的任务类型路由比旧版更全，除了 memory/dataflow/raptor/graphrag/mindmap，还多了 `evaluation`、`reembedding`、`clone` 等占位分支，能看出新架构是面向后续扩展铺路的。

## 本章小结

- 执行器消费的是 Redis Stream 双队列：队列名模板 `te.{priority}.common`（`SVR_QUEUE_NAME="te"`），高优 `te.1.common`、低优 `te.0.common`，`get_svr_queue_names` 以 `[1, 0]` 顺序返回，消费时先查高优队列实现近似优先级调度。
- 所有执行器共用一个 Redis 消费组 `rag_flow_svr_task_broker`，每个实例的 consumer 名为 `task_executor_{type}_{idx}`（默认 `task_executor_common_0`），由命令行 `-t`/`-i` 覆盖。
- `collect()` 先用 `get_unacked_iterator` 重放本 consumer 未 ack 的历史消息，耗尽后才消费新消息；`ack()`（即 `XACK`）被推迟到 `handle_task` 末尾，配合 Stream 的 Pending 机制实现"至少一次"投递、崩溃不丢任务。
- 标准任务生命周期为：绑定 embedding 模型并探测维度 → `init_kb` → `build_chunks`（线程池跑同步解析器、`chunk_limiter` 限并发、xxhash 算稳定 chunk id）→ `embedding`（标题正文按 `filename_embd_weight=0.1` 融合、写入 `q_{维度}_vec`）→ `insert_chunks`（按 `DOC_BULK_SIZE=4` 分批写 `docStoreConn`）→ `increment_chunk_num` 累加统计。
- 进度分两级：单任务进度由 `set_progress`/`TaskService.update_progress` 即时写入（`1.0`=完成、`-1`=失败/取消、`0~1`=百分比，且 0.7 前给解析分块、0.7~0.9 给向量化与写入）；文档级进度由 `document_service.py` 的 `_sync_progress` 对同文档所有任务求平均并推导整体状态。
- 并发用多个信号量分环节限流：整体 5 个任务（`task_limiter`），chunk 与 embedding 各只 1 个（最吃算力），MinIO 10 个，KG 2 个；主循环 `await task_limiter.acquire()` 形成背压，占满即停止拉新任务。
- 超时由 `connection_utils.timeout` 装饰器实现，同步函数走 daemon 线程 + `result_queue.get(timeout)`、协程走 `asyncio.wait_for`，但只有设置 `ENABLE_TIMEOUT_ASSERTION` 才真正计时，否则退化为无限等待；各环节超时分层（build_chunks 80 分钟、do_handle_task 3 小时、单次 IO/encode 60 秒）。
- `report_status()` 周期性用 `ZADD` 写心跳（含 ip/pid/pending/lag/当前任务快照）、注册进 `TASKEXE` 集合，并用 `RedisDistributedLock`（基于 Lua `delete_if_equal` 防误删）互斥地清理超过 `WORKER_HEARTBEAT_TIMEOUT=120` 秒未心跳的僵尸执行器。
- 任务取消是干净的：`set_progress` 内联 `has_canceled` 检查、置 `-1` 并抛 `TaskCanceledException`；`do_handle_task` 的 `finally` 会把已写入文档存储的该 doc chunk 整体删除，`insert_chunks` 还对半写入的 RAPTOR chunk 做回滚。
- 代码正处在重构过渡态：`task_executor.py` 顶部即引入重构版 `TaskManager`，`TE_RUN_MODE` 默认 `"0"` 已让生产流量走分层后的 `TaskHandler`，旧 `do_handle_task` 仅作回退或 dry-run 对照；重构以 `TaskContext` 依赖注入解耦全局状态。
- 重构独创"在线对账"迁移法（`TE_RUN_MODE=1`）：旧实现真跑并由 `RecordingContext` 录制，新实现用 `WriteOperationInterceptor`（白名单 FIFO 回放写返回值、不落库）跑一遍，再由 `ContextComparator` 逐 key 比对差异，保证重构等价。

## 动手实验

1. **实验一：画出一条任务的队列旅程** — 阅读 `common/settings.py:get_svr_queue_name`、`common/constants.py` 中的 `SVR_QUEUE_NAME`/`SVR_CONSUMER_GROUP_NAME`，以及 `api/db/services/task_service.py` 中两处 `REDIS_CONN.queue_product` 调用，画一张从"用户触发解析"到"消息进入 `te.{priority}.common`"再到"`collect()` 取出"的时序图，并标注高低优先级两条队列的消费顺序。

2. **实验二：验证"至少一次"与 ack 时机** — 在 `task_executor.py` 中定位 `handle_task()` 末尾的 `redis_msg.ack()` 与 `collect()` 里的 `get_unacked_iterator`，写一段说明：如果执行器在 `embedding` 阶段崩溃，这条消息会处于 Redis Stream 的什么状态？重启后它如何被重新认领？再结合 chunk id 的 xxhash 计算，论证为什么重跑不会产生重复 chunk。

3. **实验三：复盘进度百分比的分配** — 把 `embedding`（`0.7 + 0.2 * ...`）、`insert_chunks`（`0.8 + 0.1 * ...`）以及各阶段 `progress_callback` 调用按时间轴排成一张"进度预算表"，标出 0→1 区间分别留给了解析分块、向量化、写入哪几段；再读 `document_service.py:_sync_progress`，说明一个被切成 3 个子任务的文档，当其中 1 个失败、2 个完成时，前端看到的整体进度与状态是什么。

4. **实验四：跑通对账模式的心智模型** — 阅读 `task_executor_refactor/task_manager.py:dry_run_task`、`write_operation_interceptor.py` 的 `ALLOWED_METHOD_NAMES` 与 `intercept`，以及 plan.md 的第 7 节"Equivalence Guarantee"。用自己的话解释：为什么 dry run 要先跑旧实现录制、再让新实现回放写返回值而非真正落库？如果某个写方法不在白名单里会发生什么？并对照 `task_handler.py:handle()` 列出重构版比旧版 `do_handle_task` 多支持了哪几种 `task_type`。

> **下一章预告**：本章我们看到 chunk 的向量被写进 `q_{维度}_vec` 字段、最终通过 `settings.docStoreConn.insert` 落库，但那层 `docStoreConn` 抽象的背后究竟是什么？第 5 章《索引与向量化》将解构 RAGFlow 的 embedding 模型封装（`LLMBundle.encode`、批处理与截断）以及文档存储抽象层——它如何用同一套接口同时支撑 Elasticsearch、Infinity 与 OpenSearch 三种后端，索引名怎么按租户切分，向量字段与全文字段又是如何共存于一张索引的。
