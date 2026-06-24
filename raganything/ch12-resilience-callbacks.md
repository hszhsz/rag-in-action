# 第 12 章 韧性与可观测性回调：让流水线扛得住、看得见

第 11 章里我们看到 `BatchMixin` 与 `BatchParser` 如何把成百上千个文档塞进并发流水线。但并发越高、文档越多，撞上瞬时故障的概率就越大：LLM 接口偶发超时、嵌入服务限流、上游 HTTP 连接被重置。如果每一次抖动都直接让 `process_document_complete` 卡死或整批失败，那批处理能力越强、损失就越惨。RAG-Anything 把这类问题拆成两条正交的工程线：一条是**韧性**——用重试与断路器吸收瞬时故障；另一条是**可观测性**——用事件回调把流水线内部状态暴露出来供监控与度量。

这一章解构两个文件：`raganything/resilience.py` 提供 `retry`/`async_retry` 装饰器和 `CircuitBreaker` 状态机；`raganything/callbacks.py` 提供 `ProcessingEvent`、`ProcessingCallback`、`MetricsCallback` 与 `CallbackManager` 的发布-订阅机制。前者让流水线扛得住，后者让流水线看得见。

## 哪些异常值得重试：默认可重试集合的取舍

`resilience.py` 开头定义了一个模块级常量 `_DEFAULT_RETRYABLE`，初始值只有两个：`ConnectionError` 和 `TimeoutError`。注释写得很明确——它“刻意聚焦于网络/上游故障”。这是个重要的设计判断：本地编程错误（`TypeError`、`ValueError`、`KeyError`）和大多数 `OSError` 子类（`FileNotFoundError`、`PermissionError`）**默认不重试**。因为重试一个本地 bug 毫无意义，只会把一个确定性失败放大成 N 次确定性失败外加 N 段无谓的等待。

接着代码用两个 `try/except ImportError` 做**可选依赖增广**：如果装了 `httpx`，把 `httpx.ConnectError`、`ReadTimeout`、`WriteTimeout`、`PoolTimeout` 追加进去；如果装了 `openai`，再追加 `openai.APIConnectionError`、`APITimeoutError`、`RateLimitError`、`InternalServerError`。注意 `RateLimitError`（限流）和 `InternalServerError`（5xx）都被视为可重试——它们是典型的“稍后再来就好”的故障。这种“按已安装的库动态扩展异常集合”的写法，让 `resilience.py` 不强依赖任何一个 SDK，缺哪个就跳过哪段。

## 指数退避加抖动：`retry` 与 `async_retry`

`retry` 是个参数化装饰器工厂，关键默认值是：`max_attempts=3`、`base_delay=1.0`、`max_delay=60.0`、`exponential_base=2.0`、`jitter=True`。进入 `decorator` 之前先做参数校验：`max_attempts < 1`、`base_delay`/`max_delay` 为负、`exponential_base <= 0` 都会抛 `ValueError`，把非法配置挡在调用之前而不是运行期。

核心循环 `for attempt in range(1, max_attempts + 1)` 只捕获 `tuple(retryable_exceptions)` 列出的异常——其余异常直接穿透，不重试。每次失败后延迟按公式计算：

```python
delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
if jitter:
    delay *= 1.0 + random.uniform(0, 0.5)
```

这是经典的**指数退避**：第一次重试约 1s，第二次约 2s，第三次约 4s，并被 `max_delay` 截顶。`jitter` 在算出的延迟上再叠加 0~50% 的随机抖动，目的写在 docstring 里——避免“惊群”（thundering-herd），即多个 worker 同时撞上限流后又同时醒来再次齐射。抖动让它们的重试时刻错开，是高并发批处理下非常必要的细节[[Exponential Backoff And Jitter]](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)。

到达 `max_attempts` 时记 `logger.error` 并 `raise`（原异常向上抛）；否则记 `logger.warning` 并 `time.sleep(delay)`。如果传了 `on_retry` 回调，会在睡眠前以 `(exc, attempt, delay)` 调用它，方便外部埋点。

`async_retry` 是 `retry` 的异步孪生：结构完全对称，唯二区别是用 `await func(...)` 调用被装饰函数、用 `await asyncio.sleep(delay)` 而非阻塞式 `time.sleep`。还有一个贴心处理：`on_retry` 的返回值若是协程（`asyncio.iscoroutine(result)` 为真）会被 `await`，因此 `on_retry` 既可以是同步函数也可以是异步函数。在 asyncio 流水线里，用阻塞 `time.sleep` 重试会冻结整个事件循环——这正是为什么必须有独立的 `async_retry`。

## 断路器：从重试到“快速失败”

重试解决的是“偶尔抖一下”，但当上游**持续**故障时，盲目重试反而雪上加霜——每个请求都要把 3 次重试和指数延迟走一遍，线程/连接被大量挂起，故障级联放大。`CircuitBreaker` 就是为此而生：当失败累积到阈值就**断开电路**，后续调用立刻抛异常，不再触碰已知挂掉的下游。这是 Martin Fowler 描述的经典模式[[CircuitBreaker]](https://martinfowler.com/bliki/CircuitBreaker.html)。

构造参数：`failure_threshold=5`、`reset_timeout=60.0`、`name="default"`，以及可选的 `failure_exceptions`（默认同样回退到 `_DEFAULT_RETRYABLE`，即应用层 bug 默认不会触发断路）。内部状态用字符串表示，三态机：`"closed"`（正常放行）、`"open"`（拒绝所有调用）、`"half-open"`（放一个试探调用）。所有状态读写都在 `threading.Lock` 保护下进行，因为断路器是跨线程共享的。

三个核心方法构成状态转移：

- `record_success()`：把 `_failure_count` 清零、状态回到 `"closed"`、清除 `_trial_in_flight`。一次成功就完全康复。
- `record_failure()`：逻辑最微妙。若当前是 `"half-open"`，说明试探调用又失败了，直接把 `_failure_count` 拉到 `failure_threshold` 让它立刻重新 `"open"`；否则进入计数分支——**先判断上一次失败是否已超过 `reset_timeout`，是则把计数归零**再 `+1`。注释解释了原因：陈旧的失败不应计入下一波请求，避免一个小时前的偶发失败把现在的请求误伤。计数达到阈值则置 `"open"` 并打 `logger.warning`。
- `_acquire_permission()`：每次受保护调用前的守门人。若 `"open"` 且距上次失败已过 `reset_timeout`，先转 `"half-open"`；仍为 `"open"` 则抛 `CircuitBreakerOpen`；若 `"half-open"`，用 `_trial_in_flight` 实现**单飞（single-flight）**——只允许一个试探调用，其余并发调用同样被 `CircuitBreakerOpen` 拒绝；`"closed"` 则直接放行。

注意 `state` 这个 `@property` 也带了惰性转移逻辑：读取时若发现 `"open"` 已超时会顺手改成 `"half-open"`。状态既能被主动读，也能在守门时被动推进。

## 断路器作为装饰器：成功/上游故障/本地 bug 的三分

`CircuitBreaker` 实现了 `__call__` 和 `async_call`，分别装饰同步与异步函数。两者结构一致，包装逻辑值得逐句看：先 `_acquire_permission()` 守门，然后调用真正的函数，并按结果三分处理：

1. 成功 → `record_success()`，电路康复。
2. 抛出 `_failure_exceptions` 里的异常（上游/瞬时故障）→ `record_failure()` 后 `raise`，这类失败计入断路统计。
3. 抛出其他 `Exception`（应用 bug 或本地非瞬时错误）→ **不**计入断路统计，但仍要在 `_lock` 内把 `_trial_in_flight` 复位（仅当处于 `"half-open"`）后再 `raise`。

第 3 条是容易被忽略却很关键的健壮性细节：如果一个试探调用因本地 bug 抛异常而没有清掉 `_trial_in_flight`，那么 `"half-open"` 门会被永久占用，断路器再也无法收到新的试探调用，等于被本地 bug 锁死。这段 `except Exception` 兜底专门防止这种死锁。整体设计把“下游不稳定”和“我自己写错了”严格区分开——只有前者才该触发熔断。

## 可观测性：事件、回调与分发器

切到 `callbacks.py`。第一块是 `ProcessingEvent`，一个 `@dataclass`：字段含 `event_type`、`timestamp`（`default_factory=time.time`）、`file_path`、`doc_id`、`stage`、`details`（字典）、`duration_seconds`、`error`，并提供 `to_dict()` 序列化。它是事件日志里被记录的不可变记录。

第二块是 `ProcessingCallback` 基类，定义了一整套 `on_*` 钩子，覆盖流水线的每个阶段：解析（`on_parse_start`/`on_parse_complete`/`on_parse_error`）、文本插入（`on_text_insert_start`/`on_text_insert_complete`）、多模态（`on_multimodal_start`/`on_multimodal_item_complete`/`on_multimodal_complete`）、查询（`on_query_start`/`on_query_complete`/`on_query_error`）、文档级（`on_document_complete`/`on_document_error`）和批处理级（`on_batch_start`/`on_batch_complete`）。基类里所有方法体都是空的——未覆盖的钩子被静默忽略。每个方法都带 `**kwargs`，docstring 明说这是为了**向后兼容**：未来新增参数不会破坏已有子类。这是公共扩展点的标准防御姿势。

`MetricsCallback` 是开箱即用的子类，内部维护一个 `metrics` 字典，统计 `documents_processed`、`documents_failed`、各阶段累计耗时（`total_parse_time`/`total_insert_time`/`total_multimodal_time`/`total_query_time`）、内容块与多模态项总数、查询次数，以及一个 `errors` 列表。它只覆盖了 `*_complete` 和 `*_error` 这些“结算点”钩子做累加；`summary()` 把这些数字渲染成一份可读报表（错误最多展示前 5 条）；`reset()` 直接 `self.__init__()` 清零。

最后是 `CallbackManager`，发布-订阅的中枢。它用 `threading.RLock` 保护回调列表 `_callbacks`、事件日志 `_event_log` 和开关 `_log_events`。`register` 会先 `isinstance` 校验，非 `ProcessingCallback` 直接抛 `TypeError`，把契约违例挡在注册时。核心方法 `dispatch(event_name, **kwargs)` 有两个值得强调的设计：

- **快照分发**：在锁内拷贝出 `callbacks_snapshot = list(self._callbacks)`，然后**在锁外**遍历调用。这样回调内部就算去 `register`/`unregister` 其他回调，也不会在迭代时改动列表导致异常或死锁。
- **隔离失败**：每个回调用 `getattr(cb, event_name, None)` 取处理器，调用包在 `try/except Exception` 里，出错只 `logger.exception` 记录，绝不让某个用户回调的 bug 中断整条流水线。可观测性组件本身不应成为新的故障源。

只有当 `enable_event_log(True)` 打开时，`dispatch` 才会构造 `ProcessingEvent` 追加进 `_event_log`，避免无人订阅日志时的无谓开销。

## 本章小结

- `resilience.py` 用模块级 `_DEFAULT_RETRYABLE` 划定可重试范围，默认只含 `ConnectionError`/`TimeoutError`，并按是否安装 `httpx`/`openai` 动态增广，本地编程错误默认不重试。
- `retry`/`async_retry` 是对称的参数化装饰器，采用指数退避（`base_delay * exponential_base**(attempt-1)`，被 `max_delay` 截顶）加 0~50% 抖动以防惊群。
- `async_retry` 用 `await asyncio.sleep` 而非阻塞 `time.sleep`，并能 `await` 协程式的 `on_retry`，从而不冻结事件循环。
- `CircuitBreaker` 是 `closed`/`open`/`half-open` 三态机，`record_success`/`record_failure`/`_acquire_permission` 在 `threading.Lock` 下完成状态转移。
- 断路器用 `_trial_in_flight` 实现半开态单飞，且在本地 bug 路径上仍复位该标志，防止门被永久锁死。
- 断路器把“上游故障”（`_failure_exceptions`，计入熔断）与“本地 bug”（不计入熔断）严格区分。
- `ProcessingCallback` 用空实现的 `on_*` 钩子加 `**kwargs` 提供向后兼容的扩展点，覆盖解析/插入/多模态/查询/文档/批处理全阶段。
- `MetricsCallback` 在结算点钩子上累加指标，`summary()` 输出报表，是开箱即用的可观测样板。
- `CallbackManager.dispatch` 通过锁内快照、锁外遍历实现安全并发，并用 `try/except` 隔离单个回调的失败。

## 动手实验

1. **实验一：观察退避与抖动** — 用 `@retry(max_attempts=4, base_delay=0.5)` 装饰一个故意抛 `ConnectionError` 的函数，传入 `on_retry=lambda e,a,d: print(a, round(d,3))`，运行多次，确认 delay 大致按 0.5/1/2 翻倍且每次因抖动略有不同；再把 `jitter=False` 对比观察差异。
2. **实验二：跑通断路器三态** — 实例化 `CircuitBreaker(failure_threshold=3, reset_timeout=2)`，用 `__call__` 装饰一个抛 `TimeoutError` 的函数，连续调用 3 次触发 `open`，验证第 4 次立刻抛 `CircuitBreakerOpen`；`time.sleep(2)` 后读取 `.state` 确认转为 `half-open`，再调用一次成功的函数验证回到 `closed`。
3. **实验三：用 MetricsCallback 度量流水线** — 创建 `CallbackManager`，`register(MetricsCallback())`，手动 `dispatch("on_parse_complete", file_path="a.pdf", content_blocks=10, duration_seconds=1.2)` 和 `dispatch("on_document_complete", file_path="a.pdf")`，打印 `summary()`，确认 `total_content_blocks` 与 `documents_processed` 被正确累加。
4. **实验四：验证回调失败隔离与事件日志** — 注册一个在 `on_parse_start` 里主动 `raise` 的回调，再注册一个正常回调，`enable_event_log(True)` 后 `dispatch("on_parse_start", file_path="x")`，确认异常被 `logger.exception` 吞下、正常回调仍被调用，且 `event_log` 中记录了对应 `ProcessingEvent`。

> **下一章预告**：第 13 章将走进 `prompt_manager.py` 的多语言提示设计，看 RAG-Anything 如何按语言切换提示模板，并以此为收束，回顾全书九部贯穿始终的工程原则——从可选依赖、关注点分离到向后兼容与失败隔离，为整本《解构开源 RAG 系统》画上句点。
