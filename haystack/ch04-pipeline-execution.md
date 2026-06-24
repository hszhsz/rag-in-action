# 第 4 章 Pipeline 执行引擎与调度

第 3 章我们看到 `Pipeline.add_component` 与 `connect` 如何把组件与连接沉淀进一张 `networkx.MultiDiGraph`，每条边都带着 `from_socket`/`to_socket` 两端的 socket 元数据。图建好之后，真正困难的问题才浮出水面：用户调用 `run` 给了一堆输入，引擎该先跑谁、后跑谁？某个组件需要的输入还没齐怎么办？如果图里有环（cycle）——比如一个带条件路由的自我纠错循环——又怎么避免死循环？

本章解构 `haystack/core/pipeline/pipeline.py` 中同步 `Pipeline.run` 的调度主逻辑，配合 `component_checks.py` 的就绪判定与 `base.py` 的调度辅助函数，把"图怎么跑起来"讲透。Haystack 在这里做了一个关键的设计选择：它**不是**一次性算好静态拓扑序然后顺序执行，而是采用**数据驱动调度**——每一步都根据"当前哪些组件的输入已就绪"来挑下一个可运行的组件。正是这个选择，让 Haystack 能原生支持循环。

## 从用户输入到内部输入状态

`run` 的入口在 `pipeline.py` 的 `run` 方法。撇开断点（breakpoint）与快照（snapshot）恢复的分支，正常一次执行先做三件准备工作。

第一步，归一化用户输入。`run` 调用 `self._prepare_component_input_data(data)`。它支持两种写法：嵌套式 `{"retriever": {"query": ...}}`，以及扁平式 `{"query": ...}`。判定逻辑在 `base.py` 的 `_prepare_component_input_data` 里——`is_nested_component_input` 检查所有 value 是否都是 `dict`；若不是嵌套式，就遍历 `self.inputs()` 拿到每个组件的输入槽，把扁平 kwarg 分发到所有拥有同名输入槽的组件上，没匹配上的进 `unresolved_kwargs` 并打 warning。注意它对每个输入值都做了 `_deepcopy_with_exceptions`，避免同一份可变对象被多个组件共享而互相污染。

第二步，确定确定性的执行集合。`run` 用 `ordered_component_names = sorted(self.graph.nodes.keys())` 按名字排序所有组件，并据此初始化 `component_visits = dict.fromkeys(ordered_component_names, 0)`。按名字排序而非插入顺序，是为了让调度结果与组件加入图的先后无关、可复现。`component_visits` 这个计数器贯穿全程，既用于触发判定，也用于 `max_runs_per_component` 防死循环。

第三步，把用户输入转成内部格式。`inputs = self._convert_to_internal_format(pipeline_inputs=data)`。这是理解整个调度的关键数据结构。`_convert_to_internal_format` 把 `{'retriever': {'query': 'Who lives in Paris?'}}` 转成：

```
{'retriever': {'query': [{'sender': None, 'value': 'Who lives in Paris?'}]}}
```

每个 socket 对应的不是一个值，而是一个**输入记录列表**，每条记录带 `sender`（来源组件名，`None` 表示来自管线外的用户输入）和 `value`。这个"列表 + sender 标记"的设计有两层用意：variadic socket 天然要收集多个来源的输入；而 `sender is None` 这个标记，正是后面区分"用户触发"与"上游触发"、以及决定输入消费后是否保留的依据。

## 优先级：可运行性的五个档位

准备完毕后，`run` 调用 `priority_queue = self._fill_queue(...)` 建立优先级队列，然后进入 `while True` 主循环。调度的核心是给每个组件算一个**优先级**，定义在 `base.py` 的 `ComponentPriority`（一个 `IntEnum`，数值越小越优先）：

```
HIGHEST = 1
READY = 2
DEFER = 3
DEFER_LAST = 4
BLOCKED = 5
```

`_calculate_priority` 决定一个组件落到哪一档：

- 若 `can_component_run(comp, comp_inputs)` 为假 → `BLOCKED`（必需输入没齐，或没收到任何触发）。
- 若 `is_any_greedy_socket_ready` 且 `are_all_sockets_ready`（含可选输入）→ `HIGHEST`：有 greedy socket 已满且所有 socket 都齐，立刻跑。
- 若 `all_predecessors_executed`（所有前驱都执行过了）→ `READY`：不必再等，输入到此为止。
- 若 `are_all_lazy_variadic_sockets_resolved` → `DEFER`：lazy variadic 的最终状态已能确定，可以推迟但仍可运行。
- 否则 → `DEFER_LAST`：还可能有更多输入到来，尽量往后放。

这套分档把"该不该跑"与"现在跑还是等等再跑"分开了。`HIGHEST`/`READY` 是确信现在就该跑；`DEFER`/`DEFER_LAST` 是"能跑但最好让先确定的组件先跑"，留给汇合点（多输入组件）一个等待窗口；`BLOCKED` 则是当前跑不了。

底层队列是 `utils.py` 的 `FIFOPriorityQueue`，用 `heapq` 维护 `(priority, count, item)` 三元组。`count` 来自 `itertools.count()`，保证**同优先级按入队顺序出队**（FIFO），让调度对同档组件也保持确定性。它提供 `get()`（空队列返回 `None` 而非抛异常）、`peek()`、`pop()`。

## can_component_run：两道闸门

`component_checks.py` 的 `can_component_run` 是就绪判定的总入口，注释说得很清楚：一个组件要跑，必须过两道闸门——**收齐了所有必需输入**，且**收到了触发**。

```python
received_all_mandatory_inputs = are_all_sockets_ready(component, inputs, only_check_mandatory=True)
received_trigger = has_any_trigger(component, inputs)
return received_all_mandatory_inputs and received_trigger
```

为什么"收齐必需输入"还不够、非要再加一道"触发"？看 `has_any_trigger` 的三种触发源：(1) 某个前驱给它喂了输入（`any_predecessors_provided_input`）；(2) 用户从管线外给了输入且 `visits == 0`（`has_user_input and component["visits"] == 0`）；(3) 该组件不接收任何管线内输入且 `visits == 0`（`can_not_receive_inputs_from_pipeline and visits == 0`，即纯入口组件）。注释强调：触发只能让一个组件执行**一次**——因为输入在执行前会被消费删除，用户输入与"无输入入口"触发都受 `visits == 0` 约束。这道闸门正是循环不失控的基础：避免一个本应只在被上游驱动时才跑的组件，仅凭残留的默认值或用户输入被反复唤醒。

`are_all_sockets_ready` 负责"输入是否齐"。它先挑出要检查的 socket：只查必需输入时取 `socket.is_mandatory` 的；否则取 `is_mandatory or len(socket.senders)`（必需的，或有连线的）。对每个 socket，若 `has_socket_received_all_inputs` 为真，**或者**它是 lazy variadic 且 `any_socket_input_received`（收到了任意一个输入），就算"已填满"。最终要求 `filled_sockets == expected_sockets` 才返回真。`is_mandatory` 本身定义在 `InputSocket`：`self.default_value == _empty`——没有默认值的就是必需的；有默认值的非必需 socket 不影响这道闸门。

## 四类 socket 的"满"是不同的

`has_socket_received_all_inputs` 是 socket 语义的精华，它对四类 socket 给出不同的"满"定义：

- 列表为空 → 直接 `False`，没收到任何输入。
- **greedy variadic**（`is_variadic and is_greedy`）：只要有**任意一个**非 `_NO_OUTPUT_PRODUCED` 的输入就算满。greedy 的语义是"来一个就够、不等齐"，对应 `GreedyVariadic` 类型注解。
- **lazy variadic**（`is_lazy_variadic`）：要 `has_lazy_variadic_socket_received_all_inputs`——即所有 `socket.senders` 都已产出（`expected_senders == actual_senders`）才算满。lazy 的语义是"尽量等齐所有来源"。
- **普通 socket**（非 variadic）：只看 `socket_inputs[0]["value"] is not _NO_OUTPUT_PRODUCED`，第一条记录有值即满。

这里反复出现的 `_NO_OUTPUT_PRODUCED`，在 `component_checks.py` 顶部定义为 `_NO_OUTPUT_PRODUCED = _empty`（来自 `InputSocket` 的同一个哨兵）。它表示"这个前驱跑过了，但**没有**给这个 socket 产出值"——这是区分"前驱还没跑"和"前驱跑了但走了别的分支"的关键，后面分发输出时还会再见到它。`InputSocket.is_variadic` 是属性，定义为 `is_greedy or is_lazy_variadic`，二者在 `__post_init__` 里根据 `Variadic`/`GreedyVariadic` 注解推导。

## 主循环：挑、消费、运行、分发

回到 `run` 的 `while True`。每一轮做这几件事：

1. **挑候选**：`candidate = self._get_next_runnable_component(priority_queue, component_visits)`。它 `get()` 出队首，并在此处检查 `comp["visits"] > self._max_runs_per_component`（默认 `100`，见 `PipelineBase.__init__`），超限就抛 `PipelineMaxComponentRuns`。返回 `None` 则跳出循环。

2. **处理 BLOCKED**：若队首优先级是 `BLOCKED`，说明没有可运行组件了。此时调用 `_is_pipeline_possibly_blocked` 做启发式判断——若管线声明了输出却一个都没产出，就调用 `_find_components_blocking_pipeline` 找出"最可能卡住管线"的组件并打 warning，然后 `break`。`_find_components_blocking_pipeline` 的策略很实用：从队列里挑出仍有残留 inputs 的组件，再按 `component_visits` 取访问次数最少的，最后按拓扑序排序，给用户一个"它在等永远不会到的输入"的提示。

3. **平局裁决**：若队首是 `DEFER`/`DEFER_LAST` 且队里还有其他组件，调用 `_tiebreak_waiting_components`。它把所有同档（DEFER 与 DEFER_LAST 视为同档）组件取出，用 `_topological_sort()` 的位置加名字小写做 key 排序，选出拓扑上最靠前的先跑。`_topological_sort` 对 DAG 用 `networkx.lexicographical_topological_sort` [[NetworkX]](https://networkx.org/)；对含环的图则先用 `networkx.condensation` 把强连通分量缩点，让同一个环里的组件拥有相同优先级、再按名字裁决。

4. **消费输入**：`component_inputs = self._consume_component_inputs(...)`。这一步把组件要用的输入从全局 `inputs` 里"取走并删除"。对 greedy socket 只取第一个值并记入 `greedy_inputs_to_remove`（连用户输入也删，否则 greedy 用户输入会让管线无限跑）；对 lazy variadic，按 `wrap_input_in_list` 决定是保留为列表还是 `itertools.chain` 展平一层；普通 socket 只取第一个值。消费后做"剪枝"：只保留 `sender is None`（用户输入）且不属于已消费 greedy 的记录写回 `inputs[component_name]`。随后 `_add_missing_input_defaults` 用 socket 的 `default_value` 补齐缺失的可选输入（variadic 的默认值会被包成单元素列表）。

5. **运行组件**：`_run_component` 在一个 tracing span 内对输入做 `_deepcopy_with_exceptions` 后调 `instance.run(**inputs_copy)`，然后 `component_visits[component_name] += 1`，并校验输出是 `Mapping`、键与声明的输出 socket 一致（`_validate_component_output_keys`）。任何普通异常都被包成 `PipelineRuntimeError`。

6. **分发输出**：`self._write_component_outputs(...)` 把输出按 `cached_receivers` 分发到下游 socket。

## 输出分发与"没产出"的传播

`_write_component_outputs` 遍历该组件的所有接收者 `(receiver_name, sender_socket, receiver_socket, conversion_strategy)`。关键一行：

```python
value = component_outputs.get(sender_socket.name, _NO_OUTPUT_PRODUCED)
```

如果组件这次没有给某个输出 socket 产出值（典型如条件路由器只走了一个分支），就往下游写入 `_NO_OUTPUT_PRODUCED` 而不是直接不写。这一步至关重要：它让下游能区分"上游还没跑"和"上游跑了但这条路没走通"。lazy variadic 的接收槽用 `_write_to_lazy_variadic_socket` **追加**记录（因为要收集多来源）；其余用 `_write_to_standard_socket` 覆盖写（仅当有新值或当前为空时才覆盖）。若连线带 `conversion_strategy`，还会先做类型转换。

分发完后，`_write_component_outputs` 返回"未被任何下游消费的输出"——这些就是要进入管线最终结果的 leaf 输出。`run` 据此填充 `pipeline_outputs[component_name]`；若组件名在 `include_outputs_from` 集合里，则连被下游消费的输出也一并保留并返回。这解释了 `run` 的默认返回语义：只含 leaf 组件（无出边）的输出，除非显式 `include_outputs_from` 索要中间结果。

## 队列何时重算：stale 检查与循环

主循环末尾有一句：`if self._is_queue_stale(priority_queue): priority_queue = self._fill_queue(...)`。`_is_queue_stale` 的判据是：队列为空，或 `priority_queue.peek()[0] > ComponentPriority.READY`。换言之，只要队首还是 `HIGHEST`/`READY`，就继续沿用旧队列直接跑下一个；一旦队首跌到 `DEFER` 及以下，说明优先级可能已因刚才的输出分发而改变，必须重算整张队列的优先级。

这正是数据驱动调度支持循环的机制：组件运行后把新输出写回全局 `inputs`，下一次 `_fill_queue`/`_calculate_priority` 会重新评估每个组件的就绪状态。环里的组件因为又收到了新输入、`visits` 也还没到上限，会再次变成可运行——于是循环自然地转起来，直到某个分支不再产出输入（路由器选择退出分支），或 `visits` 触及 `max_runs_per_component`。

为什么不直接用静态拓扑排序？因为拓扑序的前提是 DAG，无法表达环；而 RAG 里大量出现"生成—校验—不通过则重试"这类反馈回路。Haystack 把"下一个跑谁"的决定权交给运行时的数据状态，而非编译期的图结构，代价是每步都要重算就绪性，换来的是对循环、条件分支、多输入汇合的统一表达。`max_runs_per_component`（默认 `100`）则是这套灵活机制的安全阀：任何组件被重复调度超过该上限即抛 `PipelineMaxComponentRuns`，把"逻辑错误导致的无限循环"从挂死变成一个明确的报错。

## 本章小结

- `Pipeline.run` 采用**数据驱动调度**而非静态拓扑排序：每一步根据全局 `inputs` 的当前状态重新评估哪些组件可运行，从而原生支持环、条件分支与多输入汇合。
- 用户输入经 `_prepare_component_input_data` 归一化（支持嵌套式与扁平式），再由 `_convert_to_internal_format` 转成 `{组件: {socket: [{'sender', 'value'}]}}` 的内部状态，`sender is None` 标记用户输入。
- 调度优先级 `ComponentPriority` 分五档 `HIGHEST/READY/DEFER/DEFER_LAST/BLOCKED`，由 `_calculate_priority` 依据就绪与等待情况给出；底层 `FIFOPriorityQueue` 用 `heapq + count` 保证同档 FIFO 与确定性。
- `can_component_run` 设两道闸门：`are_all_sockets_ready`（必需输入齐）与 `has_any_trigger`（来自前驱、用户或纯入口三种触发之一，后两者受 `visits == 0` 约束）。
- `has_socket_received_all_inputs` 对四类 socket 区别对待：普通看第一条记录、greedy variadic 来一个即满、lazy variadic 要等齐所有 `senders`、空列表恒为未满。
- 哨兵 `_NO_OUTPUT_PRODUCED`（即 `_empty`）让系统区分"前驱未跑"与"前驱跑了但该分支无产出"；`_write_component_outputs` 对没产出的 socket 也会写入该哨兵以传播这一信息。
- `_consume_component_inputs` 在运行前"取走并删除"输入，greedy 输入连用户来源也删，避免无限触发；运行后只剪枝保留 `sender is None` 的非 greedy 记录。
- `_is_queue_stale` 在队首跌出 `READY` 时触发整队优先级重算，这是循环得以推进的关键；`max_runs_per_component`（默认 `100`）是防死循环的安全阀。
- 当只剩 `BLOCKED` 组件时，`_is_pipeline_possibly_blocked` 与 `_find_components_blocking_pipeline` 给出"谁在等永不到来的输入"的诊断 warning。

## 动手实验

1. **实验一：观察内部输入格式** — 在本地搭一个两组件管线（如 retriever → prompt_builder），用 monkeypatch 或子类化在 `Pipeline._convert_to_internal_format` 返回后打印结果，确认每个 socket 的值都是带 `sender`/`value` 的记录列表，且用户输入的 `sender` 为 `None`。
2. **实验二：复现 max_runs 安全阀** — 用 `Pipeline(max_runs_per_component=3)` 构造一个永不退出的自环（例如一个总把输出连回自己输入的路由组件），运行后捕获 `PipelineMaxComponentRuns`，验证 `_get_next_runnable_component` 在 `visits > max` 时抛错，体会安全阀的作用。
3. **实验三：对比 greedy 与 lazy variadic** — 搭两个汇合组件，分别用 `Variadic[int]`（lazy）和 `GreedyVariadic[int]`（greedy）接收两个上游。让其中一个上游延后产出，打印各自被调度的时机，验证 greedy 收到第一个输入就跑（`HIGHEST`），而 lazy 会 `DEFER` 等齐所有 `senders`。
4. **实验四：触发 Pipeline Blocked 诊断** — 故意把某组件的一个必需输入 socket 不接任何上游也不在 `run` 里提供，运行管线后观察 `_find_components_blocking_pipeline` 打出的 warning，确认它点名了"在等永不到来输入"的组件并按拓扑序列出。

> **下一章预告**：调度引擎能把图跑起来，但一条管线如何被存盘、跨进程复现、甚至在中途暂停后再恢复？第 5 章将解构 Haystack 的序列化体系——`to_dict`/`from_dict`、YAML 持久化，以及本章频频出现却被我们刻意略过的 breakpoint 与 `PipelineSnapshot` 断点恢复机制。
