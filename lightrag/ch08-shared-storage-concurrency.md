# 第 8 章 共享存储与并发控制

前面七章我们把 LightRAG 当作一个"在一个 Python 进程里跑的 asyncio 程序"来读：分块、抽取、写入向量库和图库，全部靠 `async/await` 串起来。但真实的部署形态远不止于此。`lightrag-server` 既可以用单进程 `uvicorn` 跑，也可以用 `lightrag-gunicorn` 拉起多个 worker 进程。一旦进入多 worker 模式，那些以内存为家的存储后端——`JsonKVStorage`、`NanoVectorDBStorage`、`NetworkXStorage`——立刻面临一个尴尬：每个 worker 都有一份自己的内存副本，A worker 刚把一个实体合并写进图里，B worker 的内存却毫不知情；两个 worker 同时去合并同一个实体名，谁都不知道对方在改，写缓冲就会互相覆盖。

`lightrag/kg/shared_storage.py`（2381 行）就是为解决这一整类问题而存在的横切层。它要同时满足两套语义：单进程下用 `asyncio.Lock` 协调协程，多进程下用 `multiprocessing.Manager` 托管的进程锁与共享字典协调进程。本章逐个解构它的五大件：统一锁 `UnifiedLock`、按 key 加锁的 `KeyedUnifiedLock`、共享数据初始化与命名空间、跨 worker 的 update flags 通知机制，以及基于租约+心跳的全局并发限流闸门。理解了这一层，你才真正明白 LightRAG 凭什么敢说"多 worker 部署是安全的"。

## 8.1 为什么需要这一层：单进程与多进程的双重身份

`shared_storage.py` 的全局状态里，最关键的一个开关是模块级变量 `_is_multiprocess`。它在 `initialize_share_data` 里被定下，之后整个模块的行为都围绕它分叉。

`initialize_share_data(workers, global_concurrency_limits)`（行 1258）是入口。注释写得很直白：当配合 Gunicorn 的 preload 特性使用时，它在 master 进程 fork 之前被调用一次，于是所有 worker 继承同一份已初始化的数据；单进程模式下则在 FastAPI 的 lifespan 里调用。函数体根据 `workers` 分叉：

- `workers > 1`：`_is_multiprocess = True`，创建 `_manager = Manager()`，然后把锁注册表、共享字典、init flags、update flags 全部建成 `_manager.dict()`，内部锁建成 `_manager.Lock()`（行 1331-1351）。
- `workers <= 1`：`_is_multiprocess = False`，`_internal_lock`/`_data_init_lock` 直接是 `asyncio.Lock()`，`_shared_dicts`/`_init_flags`/`_update_flags` 是普通 `dict`，`_async_locks = None`（行 1356-1366）。

源码里这段分叉短短几行就把两种世界观铺好了：

```python
if workers > 1:
    _is_multiprocess = True
    _manager = Manager()
    _lock_registry = _manager.dict()
    _registry_guard = _manager.RLock()
    _internal_lock = _manager.Lock()
    _shared_dicts = _manager.dict()
    _update_flags = _manager.dict()
    _async_locks = {
        "internal_lock": asyncio.Lock(),
        "graph_db_lock": asyncio.Lock(),
        "data_init_lock": asyncio.Lock(),
    }
else:
    _is_multiprocess = False
    _internal_lock = asyncio.Lock()
    _shared_dicts = {}
    _update_flags = {}
    _async_locks = None  # No need for async locks in single process mode
```

注意多进程分支里 `_async_locks` 仍是**本进程的** `asyncio.Lock`——它不是用来跨进程互斥的（那是 Manager 锁的活），而是每个 worker 内部的协程闸门，下一节会看到它的妙用。这种"一把跨进程锁 + 一把进程内协程锁"的双层结构贯穿全模块。Manager 的工作原理是启动一个独立的服务器进程托管对象，其它进程拿到的是代理对象，每次访问都是一次 IPC，这正是后面所有"省 IPC"设计的由来 [[multiprocessing — Managers]](https://docs.python.org/3/library/multiprocessing.html#managers)。

这个分叉是整层设计的根。`Manager` 启动一个独立的服务器进程，把字典和锁托管在那里，各 worker 通过代理对象（proxy）做 IPC 访问；普通 dict 则零开销但只活在本进程内。`initialize_share_data` 顶部还有一道 `if _initialized: return` 的幂等闸（行 1307）：LightRAG 的 `__post_init__` 也会无参调用它，但 master 已经初始化过，后续调用直接撞上这道闸返回，绝不会覆盖 master 设好的 `_global_concurrency_limits`（这一点注释在行 105-111 反复强调，是"读一次、之后只读"的配置）。

与之对称的是 `finalize_share_data`（行 1720）：多进程模式下先 `clear()` 掉共享字典、清空 update flags 里的 `Value` 对象，再 `_manager.shutdown()` 让 Manager 自动回收所有共享资源；单进程模式下只是把全局变量重置为 `None`。所有全局变量在末尾被统一复位，保证下一次 `initialize_share_data` 能干净重来。

## 8.2 UnifiedLock：一套 `async with` 接口，两种锁语义

如果每个调用点都要写 `if _is_multiprocess: ... else: ...`，代码会被锁判断淹没。`UnifiedLock`（行 175）把这个判断下沉到锁对象内部，对外只暴露统一的 `async with`。

它的构造参数里有两个锁：`lock`（主锁，可能是 `ProcessLock` 也可能是 `asyncio.Lock`）和 `async_lock`（辅助的协程锁）、外加一个 `is_async` 标志。`__aenter__`（行 219）的逻辑是分层的：

1. 若 `not self._is_async`（多进程）且 `async_lock` 存在，**先**用 `await self._async_lock.acquire()` 拿到本进程的协程闸门。
2. 再拿主锁：`is_async` 为真时直接 `await self._lock.acquire()`；否则调用 `_acquire_mp_lock_in_executor()`。

第二步的设计是这一层最精妙的地方。Manager 的进程锁是阻塞式的——调用 `lock.acquire()` 会卡住整个线程直到别的进程释放，时长无上界。如果在事件循环线程里直接调它，这个 worker 的所有协程都会被冻住。所以 `_acquire_mp_lock_in_executor`（行 193）把 `self._lock.acquire` 丢进默认线程池 executor，再 `await asyncio.shield(acquire_future)`：

```python
async def _acquire_mp_lock_in_executor(self) -> None:
    loop = asyncio.get_running_loop()
    acquire_future = loop.run_in_executor(None, self._lock.acquire)
    try:
        await asyncio.shield(acquire_future)
    except asyncio.CancelledError:
        def _release_orphaned_acquire(f) -> None:
            if f.cancelled() or f.exception() is not None:
                return
            try:
                self._lock.release()
            except Exception:
                pass
        acquire_future.add_done_callback(_release_orphaned_acquire)
        raise
```

注释解释了 `shield` 的用意（行 196-200）：如果协程在 executor 线程仍卡在 `acquire()` 时被取消，线程无法中断、终将拿到锁却没人释放，全进程死锁。于是用 `add_done_callback` 注册 `_release_orphaned_acquire`，一旦发现是这种"孤儿获取"就立刻释放掉。把阻塞调用搬到 executor、再用 `shield` 隔离取消传播，是 asyncio 里调用阻塞库的标准做法 [[run_in_executor]](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor)。这是把"取消安全"做到了极致。

两层锁的协作含义是：协程闸门（per-process）保证同一个进程内同一时刻最多只有一个协程去争这把进程锁，于是停在 executor 里的线程最多一个；进程锁（cross-process）再保证跨进程互斥。`__aexit__`（行 278）按相反顺序释放：先放主锁、再放协程闸门，并对释放失败做了兜底重试。`UnifiedLock` 还实现了同步的 `__enter__`/`__exit__`（行 341/370），但对 async 锁会直接抛 `RuntimeError("Use 'async with' ...")`——仅为遗留代码保留。

进程级的内部锁通过 `get_internal_lock`（行 1153）和 `get_data_init_lock`（行 1187）拿到。它们的实现都一样：单进程时 `is_async=True` 包一个 `asyncio.Lock`，多进程时 `is_async=False` 包 Manager 锁并附带 `_async_locks` 里对应的协程闸门。`get_internal_lock` 是这一层的"总管锁"，几乎所有对 `_shared_dicts`、`_update_flags`、`_init_flags` 的读写都在它的保护下进行。

## 8.3 KeyedUnifiedLock：按实体名加锁，而不是一把大锁

光有一把全局 `internal_lock` 还不够。实体合并、边写入这类操作只需要"同一个实体名互斥"，不该被一把大锁串行化掉所有实体。`KeyedUnifiedLock`（行 611）提供按 key 的细粒度锁，调用入口是 `get_storage_keyed_lock(keys, namespace, enable_logging)`（行 1175）。

它的工作方式（类 docstring 行 612-620）是：本地只保留一张 async keyed lock 表，每次 acquire 时按 `namespace:key` 组合键取（或创建）一把锁，并**每次都新建一个 `UnifiedLock`**，这样 `enable_logging` 等选项可以逐次变化。核心方法 `_get_lock_for_key`（行 704）：

1. 用 `_get_combined_key(namespace, key)` 拼组合键。
2. `_get_or_create_async_lock` 拿本进程的协程闸门（带引用计数）。
3. `_get_or_create_shared_raw_mp_lock(namespace, key)` 拿跨进程锁——多进程时返回 Manager 锁，单进程时返回 `None`。
4. 据此构造 `UnifiedLock`：多进程用 Manager 锁 + 协程闸门，单进程直接把协程闸门当主锁。

真实的调用现场就在 `utils_graph.py`：删除实体前要 `async with get_storage_keyed_lock([entity_name], namespace=f"{workspace}:GraphDB")`，注释（行 93-96）点明 doc-ingest 管线对边是按 `sorted([src, tgt])` 在同一命名空间加锁的，所以对 `[entity_name]` 加锁就已经与任何触及该实体的并发边写入互斥——同一 key 字符串共享同一把锁，这就是细粒度互斥的全部秘密。源码现场是这样的：

```python
workspace = entities_vdb.global_config.get("workspace", "")
namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
async with get_storage_keyed_lock(
    [entity_name], namespace=namespace, enable_logging=False
):
    if not await chunk_entity_relation_graph.has_node(entity_name):
        ...
```

这里有两个工程取舍值得点出：其一，命名空间带上 `workspace` 前缀，于是不同租户的同名实体不会互相阻塞；其二，锁的粒度恰好是"一个实体名"，既不像全局锁那样把所有实体串行化，也不像无锁那样放任并发覆盖——这是在吞吐与正确性之间精确拿捏的一刀。

锁的清理是这套机制的难点。无限创建锁会内存泄漏，于是引入引用计数 + 延迟回收。`_get_or_create_shared_raw_mp_lock`（行 527）在 `_registry_guard`（一把 `Manager().RLock()`）保护下操作 `_lock_registry`（key→锁）和 `_lock_registry_count`（key→引用数）：取到锁就把计数 +1，若计数曾归零且在 cleanup 待清单里则把它从清单移除（"复用一把待回收的锁")。`_release_shared_raw_mp_lock`（行 556）计数 -1，归零时把组合键写进 `_lock_cleanup_data` 记下时间戳。真正的回收交给通用函数 `_perform_lock_cleanup`（行 406）：只有当待清理项数超过 `CLEANUP_THRESHOLD`（500）、最早待清理项已超过 `CLEANUP_KEYED_LOCKS_AFTER_SECONDS`（300 秒）、且距上次清理超过 `MIN_CLEANUP_INTERVAL_SECONDS`（30 秒）三个条件同时满足，才扫一遍把过期锁从注册表里删掉。它甚至处理了系统时钟回拨（行 440）。async 锁走对称的 `_async_lock_count` / `_async_lock_cleanup_data`，由 `_release_async_lock`（行 671）维护。

`_KeyedLockContext`（行 899）是 `async with` 真正进入的上下文管理器。它有几个值得记住的细节：构造时把 keys **排序**（`self._keys = sorted(keys)`，行 912），这是经典的防死锁手段——所有协程按同一全序获取多把锁，绝不会出现 A 等 B 的锁、B 等 A 的锁。`__aenter__`（行 921）逐个 key 取锁并入栈记录 `entered`/`ref_incremented` 等状态；一旦中途抛异常（含 `CancelledError`），就 `await asyncio.shield(self._rollback_acquired_locks())` 把已拿到的锁**逆序回滚**（行 983-988），保证不泄漏。`__aexit__`（行 1061）同样把整个释放过程包在 `asyncio.shield` 里逆序释放，确保即便被取消也不漏锁。这种"获取要么全成功要么全回滚、释放绝不被取消打断"的严谨，是并发代码能长期稳定运行的基础。

## 8.4 共享数据与命名空间：单进程 dict 与多进程 Manager dict

存储后端的实际数据放在 `_shared_dicts` 里，按命名空间组织。命名空间由 `get_final_namespace(namespace, workspace)`（行 137）拼成 `workspace:namespace`，这样不同 workspace 的数据天然隔离（多租户）。

`get_namespace_data(namespace, first_init, workspace)`（行 1579）在 `get_internal_lock()` 保护下，按需在 `_shared_dicts` 里创建命名空间字典：多进程用 `_manager.dict()`，单进程用 `{}`（行 1612-1615）。它对 `pipeline_status` 命名空间有特殊保护——若不是 `first_init` 调用却发现它还没初始化，就抛 `PipelineNotInitializedError`，提醒你必须先调 `initialize_pipeline_status`。

谁来初始化某个命名空间？多 worker 下不能让每个 worker 都去读一遍磁盘文件。`try_initialize_namespace`（行 1551）就是这个"抢初始化权"的原语：在内部锁保护下，第一个把 `_init_flags[final_namespace]` 置 True 的 worker 返回 `True`（获得加载权限），其余 worker 返回 `False`（禁止从文件加载）。这保证了多 worker 共享数据下，磁盘只被读一次。

`initialize_pipeline_status`（行 1376）则初始化文档处理进度的共享状态。它先 `get_namespace_data("pipeline_status", first_init=True)`，再在内部锁里填一组字段：`busy`、`destructive_busy`、`scanning`、`pending_enqueues`、`docs`、`batchs`、`cur_batch`、`request_pending`、`latest_message`，以及一个 `history_messages`。注意 `history_messages` 在多进程下用 `_manager.list()`、单进程下用 `[]`（行 1396）——必须是共享列表对象，否则一个 worker 追加的处理日志别的 worker 看不到。这些字段的语义注释非常细，比如 `destructive_busy` 是 `busy` 的"破坏性子集"，专门挡住会 drop 存储或删输入文件的清理任务与并发入队的竞态（行 1401-1408）。第 9 章讲 Pipeline 时会用到这些字段。

这里还藏着一个跨进程一致性的小陷阱：`history_messages` 不能用普通 `[]`，否则在多进程下它只是某个 worker 本地的列表，别的 worker 看不到追加进去的处理日志。所以代码写成 `_manager.list() if _is_multiprocess else []`——多进程下它是 Manager 托管的共享列表代理。`finalize_share_data`（行 1762-1768）在收尾时也特意先把 `history_messages` 清空、再清 `_shared_dicts`，避免 Manager 关闭时残留代理对象。这种"凡是需要跨 worker 可见的容器，都必须走 Manager"的纪律，是读懂这一层所有 `if _is_multiprocess` 分支的总钥匙。

`base.py` 里 `index_done_callback` 的注释（第 2 章提过）正是这套机制的客户端视角：一个 worker 落盘后必须通过 `set_all_update_flags` 通知别人，别人才会在下次访问前重载——内存后端的"最终一致"就是靠 update flags 与 `try_initialize_namespace` 这对组合实现的。

## 8.5 update flags：通知其它 worker "你的缓存脏了"

这是多 worker in-memory 后端最核心的一致性机制。设想 worker A 完成抽取后把数据落盘（`index_done_callback`），它自己的内存是新的，但 worker B 的内存还是旧的。怎么让 B 知道"该重新从文件加载了"？答案是 update flags。

`_update_flags` 是一个 `namespace -> list[flag]` 的结构，每个 worker 在每个命名空间下持有自己的一个 flag。`get_update_flag(namespace)`（行 1443）在内部锁里：若该命名空间还没有 flags 列表就建一个（多进程 `_manager.list()`，单进程 `[]`），然后创建一个新 flag 追加进去并返回——多进程用 `_manager.Value("b", False)`（一个跨进程可见的布尔），单进程用一个临时定义的 `MutableBoolean` 类（行 1468）保持接口一致（都有 `.value`）。每个存储实例在初始化时调一次 `get_update_flag`，于是列表里有几个 flag 就对应几个 worker 的该命名空间副本。

通知的动作是 `set_all_update_flags(namespace)`（行 1478）：在内部锁里把该命名空间下**所有** flag 的 `.value` 置 `True`。写数据的那个 worker 落盘后调它，等于对所有人（包括自己）喊"数据变了"：

```python
async with get_internal_lock():
    if final_namespace not in _update_flags:
        raise ValueError(...)
    for i in range(len(_update_flags[final_namespace])):
        _update_flags[final_namespace][i].value = True
```

各 worker 在下次读取前检查自己的 flag，发现为 `True` 就重新从文件加载、然后调 `clear_all_update_flags`（行 1494）把它们清回 `False`。`get_all_update_flags_status`（行 1510）提供只读视图，按 workspace 前缀过滤——它用 `rsplit(":", 1)` 从右切，因为 workspace 本身可能含冒号（行 1531）。

这个机制的优雅在于：它不传输数据本身，只传输"脏"这一个比特。重载的代价（读文件）只在真正需要时才付，而通知本身极轻量。单进程下 flags 永远只有一个、也没人去 set，机制自然退化为无操作——同一套代码两种部署都正确。

## 8.6 全局并发闸门：租约 + 心跳 + 死进程回收

最后一件大头是跨 worker 的全局并发限流。LightRAG 要控制"同时在跑的 LLM 调用数"这类全局资源。单进程下用 `asyncio.Semaphore` 就够了，但多 worker 下各进程各有信号量，加起来会超额。`shared_storage.py` 用一套租约（lease）机制把它做成跨进程精确闸门。

配置来自 `_global_concurrency_limits`（group→上限，如 `"llm:extract"`、`"embedding"`、`"rerank"`）。`is_global_concurrency_limited(group)`（行 1907）是个纯内存、无 IPC 的同步检查；`get_global_concurrency_limit`（行 1920）返回具体上限。每个 group 的整个闸门状态存在 `concurrency_leases` 命名空间的**单个** key 下（注释行 1860-1880），结构是 `{"leases": {lease_id: {pid, updated_at, [suspect_since]}}, "waiters": {pid: {pid, wait_start, last_poll}}}`。之所以用单 key、整值替换，是因为多进程下命名空间是 Manager dict，对取出的值做原地修改不会跨进程持久化；单 key 也把一次 acquire 的 IPC 压到"一读 +（至多）一写"。

`_acquire_global_slot(group, track_wait)`（行 2095）是核心。它在该 group 的 keyed 锁保护下：

1. `_load_gate_state` 读出一份本地可变副本（行 1964，深拷贝 leases/waiters，避免别名）。
2. `_reap_gate_state`（行 1986）在本地副本上回收死掉/过期的租约，返回存活租约数。
3. 若 `in_use >= limit`：拿不到槽。`track_wait` 为真时登记/刷新本进程的 waiter 记录并返回是否"最久等待者"；否则直接返回 `None`。
4. 否则生成 `uuid4().hex` 作为 `lease_id`，写入 `{pid, updated_at: now}`，清掉自己的 waiter 记录，写回状态，返回租约。

核心几行把"读一次、按需写一次"的 IPC 预算体现得很清楚：

```python
async with get_storage_keyed_lock(group, namespace=_CONCURRENCY_LEASE_NAMESPACE):
    now = time.time()
    state = _load_gate_state(ns, group)
    in_use, changed = _reap_gate_state(state, now)
    if in_use >= limit:
        ...                      # 登记 waiter 或直接返回 None
        return None, _is_longest_live_waiter(state, pid, now)
    lease_id = uuid.uuid4().hex
    state["leases"][lease_id] = {"pid": pid, "updated_at": now}
    state["waiters"].pop(pid_key, None)
    ns[group] = state            # 整值写回，唯一一次写 IPC
    return lease_id, True
```

回收逻辑 `_reap_gate_state` 是自愈的精华。对每条租约：若 `pid` 已死（`_pid_alive` 行 1927 用 `os.kill(pid, 0)` 探测，权限错误时保守判活）立即回收；若心跳 `updated_at` 超过 `_heartbeat_ttl`（默认 20 秒，约 5 秒健康心跳的 4 倍）但进程还活着，先打上 `suspect_since` 标记、再宽限 `_suspect_grace`（默认 20 秒）才回收——这道"嫌疑宽限期"专门保护"活着但一时卡顿"的持有者，避免误杀。被标记 suspect 的租约**仍计入容量**，所以上限永不被突破。同一进程持多个租约时，活性探测被 `alive_cache` 记忆化、但绝不跨调用缓存（防 PID 复用误判，行 2007-2011）。waiter 记录在同一趟里清理：刚被回收 pid 的、已死的、或超过 `_waiter_stale_ttl`（默认 1 秒）没刷新的，都被剔除，免得"幽灵最久等待者"霸占座位（行 1996-2000）。

持有者靠 `renew_global_slots(group, lease_ids)`（行 2240）续约——它把租约整条重写、清掉 suspect 标记；但**绝不复活已被回收的租约**（重插可能超额）。释放是 `release_global_slot`（行 2224），幂等地 pop 掉租约。`reconcile_global_slots`（行 2266）单独跑一趟 reaper 返回存活数。对外的两个 acquire 入口：`try_acquire_global_slot`（行 2150，非阻塞、不登记 waiter）和 `try_acquire_global_slot_tracked`（行 2166，给轮询循环用，返回 `(lease_id, is_priority_waiter)` 实现跨进程软 FIFO 公平）。所有 acquire 都是 fail-closed 的——任何共享存储错误都返回 `None`（配 `_log_acquire_failure` 限流告警），宁可让任务继续排队也绝不因基础设施抖动而超额。

值得注意的公平性细节（行 2138-2142）：抢到槽后会把自己的 waiter 记录清掉、`wait_start` 重置，于是一个积压很重的进程每赢一次就被"降级"，在持续竞争下逼近跨进程的近似 round-robin。这种"软公平"配上 `_is_longest_live_waiter`（行 2075）的最久等待判定，让轮询者在被偏好时用最快间隔、否则有界退避，既不饿死也不让空槽闲置。

## 本章小结

- `shared_storage.py` 是 LightRAG 的并发横切层，让同一套存储代码既能在单进程 asyncio 下跑，也能在多 worker `lightrag-gunicorn` 下安全运行；分叉点是 `initialize_share_data` 里据 `workers` 设定的 `_is_multiprocess`。
- 单进程用普通 `dict` + `asyncio.Lock`，多进程用 `multiprocessing.Manager` 托管的共享 dict + 进程锁；`initialize_share_data` 的 `if _initialized: return` 幂等闸保证 master 设的配置不被 worker 覆盖。
- `UnifiedLock` 用一套 `async with` 接口适配两种锁语义；多进程下把阻塞的 Manager 锁丢进 executor 异步等待，并用 `asyncio.shield` + done-callback 处理取消时的"孤儿获取"防死锁。
- `KeyedUnifiedLock` / `get_storage_keyed_lock` 提供按 key（如实体名、`sorted([src,tgt])` 边）的细粒度互斥；同一 key 字符串共享同一把锁，避免一把大锁串行化所有实体。
- 锁的清理用引用计数 + 延迟回收：归零进 `_lock_cleanup_data`，由 `_perform_lock_cleanup` 在数量、年龄、间隔三条件齐备时批量回收，并处理时钟回拨。
- `_KeyedLockContext` 对 keys 排序以全序防死锁，获取失败逆序回滚、释放过程用 `shield` 包裹防取消漏锁。
- 命名空间以 `workspace:namespace` 组织实现多租户隔离；`try_initialize_namespace` 抢初始化权保证多 worker 下磁盘只被读一次。
- update flags 用 `_manager.Value("b", ...)`（多进程）/`MutableBoolean`（单进程）传播"缓存脏"这一比特：写者落盘后 `set_all_update_flags`，读者下次读前检查并重载、`clear_all_update_flags` 复位。
- 全局并发闸门以"单 key 整值替换"的租约表为准，`_acquire_global_slot` 在 group 锁下读一次、按需写一次完成限流，所有 acquire fail-closed。
- 租约自愈靠心跳续约 + 死进程立即回收 + 心跳过期经"嫌疑宽限期"后回收，suspect 租约仍计容量，杜绝 `kill -9`/OOM 后的永久容量泄漏。
- 抢槽后重置 waiter 实现近似 round-robin 软公平，配合最久等待判定让轮询者在偏好/退避间切换，不饿死也不让空槽闲置。

## 动手实验

1. **实验一：观察单进程与多进程分叉** — 在一个脚本里分别调用 `initialize_share_data(workers=1)` 与（新进程中）`initialize_share_data(workers=4)`，打印 `shared_storage._is_multiprocess` 和 `type(shared_storage._internal_lock)`，确认前者是 `asyncio.Lock`、后者是 Manager 锁代理；再观察重复调用第二次时被 `if _initialized: return` 拦截、配置不变。
2. **实验二：验证 keyed 锁的细粒度与排序防死锁** — 写两个并发协程，分别 `async with get_storage_keyed_lock(["A","B"])` 和 `get_storage_keyed_lock(["B","A"])`，在锁内 `sleep`；因为 `_KeyedLockContext` 把 keys 排序，两者实际都按 `["A","B"]` 顺序获取，不会死锁。再换成不同 key（`["A"]` 与 `["C"]`）确认它们能真正并发。
3. **实验三：复现 update flags 通知** — 在单进程下对某命名空间调两次 `get_update_flag` 模拟两个副本，调 `set_all_update_flags` 后用 `get_all_update_flags_status` 打印，看到两个 flag 都变 `True`；调 `clear_all_update_flags` 后再打印确认归 `False`，体会"只传一个脏比特"的轻量。
4. **实验四：模拟租约回收** — 用 `initialize_share_data(workers=2, global_concurrency_limits={"llm":1})` 后 `try_acquire_global_slot("llm")` 拿到一个租约；不续约，用 monkeypatch 把 `shared_storage._heartbeat_ttl` 和 `_suspect_grace` 调到 1 秒，`sleep` 后再调 `reconcile_global_slots("llm")`，观察过期租约被回收、容量释放；再把租约 pid 改成一个不存在的 pid，确认 `_pid_alive` 判死后立即回收。

> **下一章预告**：有了共享存储与并发闸门作地基，第 9 章《文档处理管线 Pipeline》将把这些原语用起来——看 LightRAG 如何用 `pipeline_status` 协调多 worker 的文档批处理、用 `busy`/`request_pending` 实现"一次只跑一个管线、新请求挂起合并"的调度，以及实体抽取如何在 keyed 锁与全局并发槽的双重约束下并行而不乱。
