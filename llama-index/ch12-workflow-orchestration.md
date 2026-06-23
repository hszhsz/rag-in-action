# 第 12 章 Workflow 事件驱动编排引擎

第 11 章里的 `QueryEngine` 与 `ChatEngine` 给了我们一条体面的"固定链路":检索、合成、对话记忆,一步接一步,职责清晰。但固定链路的代价是结构僵硬——它假设数据永远朝一个方向流。一旦 RAG/Agent 逻辑出现分支(检索为空就改写问题再查一次)、循环(ReAct 式的"思考-行动-观察"直到收敛)、并行(同时打多个子查询再汇总)、或者人类介入(把草稿丢给人审一眼再继续),线性管线就开始用 `if/else` 和回调把自己缠死。

LlamaIndex 给出的答案是用**事件驱动的 Workflow** 取代旧的 Query Pipeline,把它做成新一代编排基座。在当前这份源码里,`llama_index.core.workflow` 已经是一层薄薄的再导出壳子——例如 `llama-index-core/llama_index/core/workflow/workflow.py` 只写了 `from workflows.workflow import Workflow, WorkflowMeta`,真正的实现搬进了独立的 `workflows` 包(`llama-index-core/pyproject.toml` 中钉死为 `llama-index-workflows>=2.14.0,<3`)。本章解读的就是这个上游 `workflows` 包(2.x)的核心机制:它不靠你手工连线,而是靠 `@step` 方法的**类型注解**自动把整张 DAG 推断出来,再用一套校验在启动前把"孤儿 step""无人消费的事件"统统拦下。

## 为什么是事件驱动:从固定 DAG 到可循环的图

固定 DAG 的根本约束是"无环":节点之间只能单向连边,天然不支持"回到上一步重试"这种循环。Workflow 把这个约束彻底拿掉——它不再有"边"这个一等公民,只有 step 和它们收发的 `Event`。一个 step 产出某个事件,这个事件就会被路由到所有声明"接收该事件"的 step;后者执行后再产出新事件……如此往复,直到有人产出 `StopEvent`。

`workflows/workflow.py` 里 `Workflow` 类的 docstring 把这套模型的卖点列得很直白:"Validation of step signatures and event graph before running""Typed start/stop events""Streaming of intermediate events""Optional human-in-the-loop events""Retry policies per step""Resource injection"。因为路由是基于事件类型而非固定边,所以"循环"只是"某个 step 产出的事件又能喂回上游 step"的自然结果——图里允许有环,这正是 ReAct/反思类 Agent 需要的形态。

`Workflow.__init__` 暴露的几个旋钮也都围绕这套运行时:`timeout: float | None = 45.0`(默认 45 秒超时,`None` 关闭)、`verbose: bool = False`、`disable_validation: bool = False`、`num_concurrent_runs`、以及 `skip_graph_checks`(用集合显式跳过某些图校验)。注意默认超时是 **45.0 秒**而非常见的"无限",这是个容易踩的默认值。

## `@step` + Event 类型注解如何自动连边成图

这是整章的重头戏。你从不手工写"step A 连到 step B",你只写函数签名,引擎用**参数类型**(这个 step 吃哪种事件)和**返回类型**(它吐哪种事件)反推出数据流。

`@step` 装饰器定义在 `workflows/decorators.py`。它的关键工作在 `make_step_function` 里:先 `spec = inspect_signature(func, localns=localns)` 解析签名,再 `validate_step_signature(spec)` 校验,然后把结果塞进一个 `StepConfig` dataclass。`StepConfig` 的字段就是这个 step 的全部"连边信息":`accepted_events`(接收的事件类型列表)、`return_types`(返回的事件类型列表)、`context_parameter`(是否要注入 `Context`)、`num_workers`、`retry_policy`、`resources` 等。

事件类型从哪来?看 `workflows/utils.py` 的 `inspect_signature`:它遍历参数,用 `_get_param_types` 解析每个参数的注解;遇到 `Union`/`Optional`/`UnionType` 会展开并剔除 `None`(`get_origin(typ) in (Union, Optional, UnionType)` 时取 `[t for t in get_args(typ) if t is not type(None)]`)。返回类型同理由 `_get_return_types` 解析。于是一个这样的方法:

```python
@step
async def start(self, ev: StartEvent) -> StopEvent:
    return StopEvent(result="done")
```

就被登记为"接收 `StartEvent`、产出 `StopEvent`"。引擎把所有 step 的 `accepted_events` 和 `return_types` 求并、对齐,就得到了完整的事件图——无需任何显式连线。

签名合法性由 `validate_step_signature` 守门(`workflows/utils.py`):若 `accepted_events` 为空,报 `"Step signature must have at least one parameter annotated as type Event"`;若返回类型未标注,报 `"Return types of workflows step functions must be annotated with their type."`。换句话说,**类型注解不是可选的文档,而是连边的唯一依据**,漏标直接在装饰阶段就炸。

`@step` 既能装饰实例方法,也能装饰自由函数。自由函数必须显式传 `workflow=MyWorkflow`,否则 `_apply_step_decorator` 会抛 `"To decorate {name} please pass a workflow class to the @step decorator."`;它内部通过 `is_free_function(func.__qualname__)` 判断,并调用 `workflow.add_step(func)` 注册。`Workflow.add_step`(`workflows/workflow.py`)会拒绝重名 step,并自增 `_step_functions_version`——这个版本号后面用于校验缓存失效。`num_workers` 默认 4,且 `_apply_step_decorator` 强制它是大于 0 的整数,否则报 `"num_workers must be an integer greater than 0"`。

## StartEvent / StopEvent:起止的隐式约定

Workflow 没有"入口函数"和"出口函数"的概念,起止完全由两个特殊事件约定(都定义在 `workflows/events.py`,均继承自 `Event`):

- `StartEvent`——"Implicit entry event sent to kick off a `Workflow.run()`"。`Workflow.run(**kwargs)` 不显式造事件,而是把 kwargs 喂给推断出来的 `StartEvent` 子类(`_get_start_event_instance` 里 `self._start_event_class(**kwargs)`)。你也可以直接传 `start_event=MyStartEvent(...)` 实例。
- `StopEvent`——终止信号,它的 `result` 属性承载整个 run 的返回值。基类把 `result` 存在私有的 `_result`(`__init__(self, result=None, ...)` 实际写入 `_result`),`result` property 再读回来;若用了自定义 `StopEvent` 子类,run 的结果就是那个事件实例本身。

起止事件不是手工指定的,而是**从所有 step 的签名里推断**。`Workflow.__init__` 调用 `_ensure_start_event_class` 与 `_ensure_stop_event_class`(在 `workflows/representation/validate.py`)完成这件事,并固化了两条不变量:

- `_ensure_start_event_class`:扫描所有 step 的 `accepted_events`,必须恰好有**一种** `StartEvent` 子类被某个 step 接收。一个都没有 → `"At least one Event of type StartEvent must be received by any step."`;多于一种 → `"Only one type of StartEvent is allowed per workflow."`。
- `_ensure_stop_event_class`:同理扫描返回类型,锁定唯一的 `StopEvent` 子类。

`workflows/events.py` 里还有几类专用终止/信号事件值得记住:`WorkflowTimedOutEvent`(超时,带 `timeout` 与 `active_steps`)、`WorkflowCancelledEvent`、`WorkflowFailedEvent`(step 重试耗尽后永久失败,带 `step_name`/`exception`/`attempts`),以及人类介入用的 `InputRequiredEvent` 和 `HumanResponseEvent`——后两者是 HITL 的标准搭档:某 step 返回 `InputRequiredEvent` 暂停,调用方从事件流里取到它、回灌一个 `HumanResponseEvent`,另一个接收 `HumanResponseEvent` 的 step 接力继续。

## Context:跨步共享状态与 `collect_events` 实现 fan-in

事件只携带"这一跳"的数据;真正的跨 step 共享状态住在 `Context`(`workflows/context/context.py`)。step 只要在签名里加一个 `ctx: Context` 参数,`inspect_signature` 就会把它识别成 `context_parameter` 并在运行时注入。

状态读写走 `ctx.store`——一个类型化的 `StateStore`(`workflows/context/state_store.py`),协议里定义了 `async get(self, path, default=...)` 和 `async set(self, path, value)`,支持点分路径(dot-separated paths)做嵌套读写,还有 `get_state`/`set_state`/`clear` 和事务式的 `edit_state()`。未初始化时默认用 `DictState` 这个类 dict 的状态模型。

而最能体现"事件驱动"威力的是 **`collect_events`**,它把多事件汇聚(fan-in / join)收成一行代码。签名是 `collect_events(self, ev, expected: list[type[Event]], buffer_id=None)`(`workflows/context/context.py`)。它的语义在 docstring 里写得很清楚:把进来的 `ev` 塞进缓冲区,**只有当 `expected` 列出的每种事件都集齐了**才返回一个按 `expected` 顺序排好的列表,否则返回 `None`。配合典型写法:

```python
@step
async def synthesize(
    self, ctx: Context, ev: QueryEvent | RetrieveEvent
) -> StopEvent | None:
    events = ctx.collect_events(ev, [QueryEvent, RetrieveEvent])
    if events is None:
        return None
    query_ev, retrieve_ev = events
    ...
```

这个 step 会被 `QueryEvent` 和 `RetrieveEvent` 各触发一次,前几次 `collect_events` 返回 `None`(直接 `return None` 让这次执行"哑火"),直到两个事件都到齐才真正往下走——这就是用事件实现的 join。注意返回类型必须包含 `None`(`StopEvent | None`),因为 `_get_param_types` 会剔除 `None`,所以"返回 `None`"在图里是合法的"无产出"分支。

与之对应的是**发散(fan-out)**:`ctx.send_event(message, step=None)`(`workflows/context/context.py`)能主动把事件投进队列。`step` 省略时广播给所有 step,不接收该类型的 step 自动忽略;指定 `step` 时,目标 step 必须接受该事件类型,否则抛 `WorkflowRuntimeError`。典型用法是在循环里 `for i in range(10): ctx.send_event(WorkerEvent(msg=i))` 撒出十个并行任务,再用 `collect_events` 收回来。装饰器层面还内建了更声明式的 fan-in:`StepConfig` 里的 `collect_params`(多个事件参数的异构 join)和 `collection_param`/`collection_policy`(单个 `list[E]` 参数,配合 `workflows/collect.py` 里的 `Collect`/`All`/`Take` 选择何时释放——`Take(n)` 表示集到第 n 个就触发)。

此外还有 `ctx.wait_for_event(event_type, ...)`(默认 `timeout=2000` 秒),它通过抛出内部控制流异常把 step 挂起、事件到了再**重放整个 step**——所以官方提醒把它放在 step 顶部、且让它前面的代码可重复执行。

## 事件流 streaming:把中间事件实时吐给调用方

`Workflow.run()` 不直接返回结果,而是返回一个 `WorkflowHandler`(`workflows/handler.py`),它是个 `Awaitable[RunResultT]`:`await handler` 拿最终结果,`async for ev in handler.stream_events()` 拿中间事件流。

中间事件怎么进流?step 内部调用 `ctx.write_event_to_stream(ev)`(`workflows/context/context.py`),把任意事件推进流式队列;`Context.stream_events()` 是个 `AsyncGenerator[Event, None]`,逐个 yield 这些被发布的事件。`WorkflowHandler.stream_events` 内部就是 `async for ev in self.ctx.stream_events()` 的转发。这条流是 RAG/Agent 做"打字机式输出"和"实时展示思考步骤"的基础设施;`InputRequiredEvent` 之所以能驱动 HITL,正是因为它会被自动写入事件流让调用方看到。

`workflows/events.py` 里还有一类 `InternalDispatchEvent`(及其子类 `WorkflowIdleEvent`、`StepStateChanged`、`UnhandledEvent`),默认不暴露给用户,但 `stream_events(expose_internal=True)` 能把它们也放出来,用于观测 step 启停、worker 占用、workflow 何时进入 idle(等待外部输入)等内部状态。

## 校验机制:孤儿 step 与无人消费的事件,启动即报错

这一节呼应全书"错误即文档 / 不变量固化"的透镜。Workflow 把"图是不是接得通"做成了**启动前的强校验**,而不是等运行到一半才崩。

入口是 `Workflow.run()` 里第一件事就调 `self._validate()`(`workflows/workflow.py`)。除非构造时 `disable_validation=True`,否则它会跑 `_validate_workflow`,核心是 `workflows/representation/validate.py` 的 `validate_graph`,它先 `build_step_graph` 建邻接表,再跑三项检查(可被 `skip_graph_checks` 跳过):

1. **reachability(可达性 / 孤儿 step)**:从输入事件种子(`StartEvent` + `HumanResponseEvent` 子类 + `catch_error` 处理器)做正向 DFS,凡是不在 `forward_reachable` 里的 step 就是孤儿,报 `"Unreachable steps: ..."`,并附 hint `"Steps must be reachable from StartEvent or HumanResponseEvent."`。
2. **terminal_event(无人消费的事件)**:遍历所有事件类型,若某事件没有任何 step 消费,且它**不是** `StopEvent` 或 `InputRequiredEvent`(这两类才允许做终点),就报 `"Events produced but never consumed: ..."`,hint 是 `"Only StopEvent and InputRequiredEvent may be terminal."`。
3. **dead_end(死胡同)**:从输出事件(`StopEvent`/`InputRequiredEvent`)反向 DFS,凡是会产出事件却到不了任何出口的 step,报 `"Dead-end steps: ..."`。

这三条把"画错的图"在 `run()` 落地前就翻译成了人话错误。校验结果还带缓存:`_validated_version` 记录被校验过的 `_step_functions_version`,只要没 `add_step()` 改图,后续 `run()` 直接复用结果不重复校验(显式调 `workflow.validate()` 则强制重跑)。

## retry_policy / resource / Context 序列化:可靠性三件套

**重试**:`@step(retry_policy=...)` 接一个满足 `RetryPolicy` 协议的对象(`workflows/retry_policy.py`),协议核心方法是 `next(elapsed_time, attempts, error, *, seed=None) -> float | None`——返回"下次重试前等多少秒",返回 `None` 表示放弃。包里提供了组合式的 `retry_policy(retry=..., wait=..., stop=...)`(可用 `retry_if_exception_type`、`wait_exponential`、`stop_after_attempt` 等积木拼装),以及现成的 `ConstantDelayRetryPolicy(delay=5, maximum_attempts=10)`。step 内还能用 `ctx.retry_info()` 拿到当前重试号、距首次失败的秒数和上一个异常。当所有重试耗尽,可由 `@catch_error` 处理器兜底:它必须接收 `StepFailedEvent`(否则装饰阶段就报错),`for_steps` 限定覆盖范围、`max_recoveries`(默认 1)限定每条 lineage 的兜底预算。

**资源注入**:用 `typing.Annotated[T, Resource(factory, cache=True)]` 在 step 参数上声明依赖(`workflows/resource.py` 的 `Resource()`)。`factory` 可同步可异步,`cache=True`(默认)时同一资源跨 step 复用;由 `ResourceManager` 统一管理生命周期与解析缓存,并能检测循环依赖(`validate(validate_resources=True)`)。这让"一个跨多步共享的 `Memory` 对象"之类依赖以声明方式注入,而不是塞进全局变量。

**Context 序列化 / 断点续跑**:`Context.to_dict(serializer=None)`(`workflows/context/context.py`)把全局状态、事件队列、各种缓冲区、accepted events、broker log 和 running 标志统统序列化成可 JSON 编码的 dict;`Context.from_dict(workflow, data, serializer=None)` 再还原。序列化器在 `workflows/context/serializers.py`:默认 `JsonSerializer`,另有 `PickleSerializer`(`llama_index.core` 把它别名为向后兼容的 `JsonPickleSerializer`)。配合起来就能做断点续跑——把 `ctx.to_dict()` 落库,下次 `workflow.run(ctx=Context.from_dict(...))` 接着跑。

最后,把图画出来辅助调试的 `draw_all_possible_flows` 仍在 `llama-index-core/.../workflow/drawing.py`,但已标 `@deprecated`,建议改装 `llama-index-utils-workflow` 后从 `llama_index.utils.workflow` 导入——它内部同样靠 `get_steps_from_class` 读出每个 step 的 `StepConfig`,把推断出的事件图用 `pyvis` 渲染成 HTML。

## 并发模型:每个 step 的 worker 与全局并发上限

事件驱动天生适合并发:多个事件可以同时在队列里等待,引擎并行调度。粒度由两层旋钮控制。第一层是 step 级的 `num_workers`(`@step(num_workers=4)`,默认 4,存进 `StepConfig.num_workers`):它决定同一个 step 最多能有多少个并发执行的 worker 在跑——当你用 `send_event` 撒出十个 `WorkerEvent`,接收它的 step 若 `num_workers=4`,就最多四个并行处理、其余排队。这正是 fan-out 真正"并行"而非"伪并行"的来源。第二层是 workflow 级的 `num_concurrent_runs`(`Workflow.__init__` 参数),限制同一个 workflow 实例上同时进行的 `run()` 调用数,用来给整个编排器加一个全局闸门。

值得一提的是,`_apply_step_decorator` 在 step 装饰时就强校验 `num_workers` 必须是大于 0 的整数,而 `@catch_error` 处理器走 `make_step_function(..., num_workers=1)`——错误兜底天然不需要并发。`StepStateChanged` 这个内部事件(`workflows/events.py`)里带的 `worker_id` 字段,正是用来在事件流里观测"哪个 step 在哪个 worker 上跑"的。

## WorkflowHandler:await、stream 与优雅取消

`Workflow.run()` 返回的 `WorkflowHandler`(`workflows/handler.py`)是把"未来的结果"和"现在的事件流"打包成一个对象。它声明为 `class WorkflowHandler(Awaitable[RunResultT])`,所以 `result = await handler` 直接拿最终值(内部 `_await_result` / `_wait_for_result` 等待那个 `StopEvent`),而 `handler.stream_events()` 转发 `ctx.stream_events()` 给你逐事件消费。两种用法可以并行:一边 `async for` 流式展示,一边 `await` 收尾。

取消有两条路径,源码给出了明确的优先级与弃用关系。`handler.cancel()` 是"硬取消"——它走运行时兼容 shim 调 `abort()` 并 `cancel()` 掉结果任务,但若运行时不支持就抛 `NotImplementedError` 并提示改用 `await handler.cancel_run()`;`cancel_run(*, timeout=5.0)` 才是推荐的"优雅取消":它向底层 context 发信号、令其抛出 `WorkflowCancelledByUser`(`workflows/errors.py`),由 workflow 捕获后干净收尾,并用 `asyncio.wait_for(..., timeout=timeout)` 兜底等待。调用方还能通过 `handler.send_event(event, step=None)` 从外部把事件灌进正在运行的 workflow——这是 HITL 回灌 `HumanResponseEvent` 的标准入口。

## 本章小结

- 当 RAG/Agent 逻辑出现分支、循环、并行、人类介入时,第 11 章的固定链路不够用;LlamaIndex 用事件驱动的 `Workflow`(上游 `workflows` 包,`llama-index-core` 里 `llama-index-workflows>=2.14.0,<3`)取代旧 Query Pipeline 作为编排基座。
- Workflow 没有显式"边":step 产出事件,事件按类型路由到所有接收它的 step,允许有环,因此天然支持循环与重试。
- `@step`(`workflows/decorators.py`)靠 `inspect_signature` 解析参数/返回类型并存入 `StepConfig`;**类型注解是自动连边的唯一依据**,漏标参数或返回类型在装饰阶段即报错。
- 起止由特殊事件约定:`StartEvent` 是 `run()` 的隐式入口、kwargs 用来构造它;`StopEvent.result` 承载结果;两者的具体子类由 `_ensure_start_event_class`/`_ensure_stop_event_class` 从签名推断,且各自必须唯一。
- `Context` 提供跨 step 状态(`ctx.store` 的 `async get/set` 支持点分路径),`collect_events(ev, expected)` 集齐多事件才返回、实现 join/fan-in,`send_event` 与 `Collect/Take` 实现 fan-out。
- 事件流 streaming 由 `ctx.write_event_to_stream` 发布、`handler.stream_events()` 消费,是打字机输出与 HITL(`InputRequiredEvent`/`HumanResponseEvent`)的基础。
- 启动前强校验(`validate_graph`)固化三条不变量:孤儿 step(reachability)、无人消费的事件(terminal_event,只有 `StopEvent`/`InputRequiredEvent` 可做终点)、死胡同(dead_end),把画错的图翻译成人话错误。
- 可靠性三件套:`retry_policy`(`RetryPolicy.next` 协议 + `ConstantDelayRetryPolicy` 等)与 `@catch_error` 兜底、`Resource` 依赖注入、`Context.to_dict/from_dict` 序列化支持断点续跑。
- 默认值要记牢:`Workflow(timeout=45.0)`、`@step(num_workers=4)`、`wait_for_event(timeout=2000)`、`ConstantDelayRetryPolicy(delay=5, maximum_attempts=10)`。

## 动手实验

1. **实验一:把 wheel 解出来读真实实现** — 由于 `/tmp/llama_explore/llama-index-core/llama_index/core/workflow/*.py` 全是 `from workflows... import` 的再导出壳子,先 `pip3 download llama-index-workflows --no-deps` 再 `unzip` 该 wheel;在 `workflows/workflow.py` 里定位 `Workflow.__init__` 确认 `timeout` 默认 `45.0`,在 `workflows/decorators.py` 的 `step()` 确认 `num_workers=4`。
2. **实验二:验证类型注解如何连边** — 读 `workflows/utils.py` 的 `inspect_signature` 与 `_get_param_types`,确认 `Union/Optional/UnionType` 会被展开、`None` 会被剔除;再读 `make_step_function`(`workflows/decorators.py`)看 `accepted_events`/`return_types` 如何落进 `StepConfig`,理解"返回 `StopEvent | None` 中的 `None` 为何是合法无产出分支"。
3. **实验三:复盘三项图校验** — 打开 `workflows/representation/validate.py` 的 `validate_graph`,逐字找出 reachability、terminal_event、dead_end 三段对应的报错串(`"Unreachable steps:"`、`"Events produced but never consumed:"`、`"Dead-end steps:"`),再回到 `_ensure_start_event_class`/`_ensure_stop_event_class` 确认"StartEvent/StopEvent 必须唯一"两条不变量的错误文案。
4. **实验四:追踪 fan-in 与断点续跑** — 在 `workflows/context/context.py` 阅读 `collect_events` 的 docstring 与返回 `None` 的缓冲逻辑,对照 `workflows/collect.py` 的 `All`/`Take(n)` 理解释放时机;再读 `Context.to_dict`/`from_dict` 与 `workflows/context/serializers.py` 的 `JsonSerializer`/`PickleSerializer`,串起"序列化 Context → 落库 → `run(ctx=...)` 续跑"的完整链路。

> **下一章预告**:有了事件驱动的 Workflow 这块基座,第 13 章将拆解 LlamaIndex 的 Agent 系统——基于 Workflow 实现的 `FunctionAgent` 与 `ReActAgent` 如何把"工具调用"编排成思考-行动-观察的循环,以及 `AgentWorkflow` 如何把多个智能体协同成一个多 Agent 系统。
