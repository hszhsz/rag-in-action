# 第 3 章 Pipeline 图构建与连接

上一章我们解构了 Haystack 的核心数据模型——`Document`、`ChatMessage` 这些在组件之间流动的数据，以及每个组件通过 `@component` 契约声明的 `InputSocket` 与 `OutputSocket`。组件本身只是一个孤立的"积木块"：它知道自己需要什么输入、能产出什么输出，但并不知道数据从哪来、要送到哪去。真正把这些积木拼成一条可运行的链路，是 `Pipeline` 的职责。

本章聚焦"拼装"这件事：`add_component` 如何把一个组件注册成图上的节点、socket 元数据如何挂到节点上；`connect("a.out", "b.in")` 如何解析两端的 socket、做类型兼容校验（包括 `Optional`、`Variadic`、`Union` 的特殊处理）、连接成功后如何回写 socket 的 `senders`/`receivers`；以及图上 socket 的连接状态如何决定整条 Pipeline 的入口与出口。更重要的是，我们要说清 Haystack 为什么坚持在"构建期"就做完这些静态校验，而不是等到运行期才报错。本章涉及的核心源码是 `core/pipeline/base.py`、`core/component/sockets.py`、`core/component/types.py` 与 `core/type_utils.py`。

## Pipeline 持有的是一张 MultiDiGraph

打开 `PipelineBase.__init__`，会看到 Pipeline 的"骨架"其实就是一张 networkx 的有向多重图：

```python
self.graph = networkx.MultiDiGraph()
self._max_runs_per_component = max_runs_per_component
self._connection_type_validation = connection_type_validation
```

选用 `MultiDiGraph` 而不是普通 `DiGraph` 是有意为之[[NetworkX]](https://networkx.org/documentation/stable/reference/classes/multidigraph.html)。两个组件之间完全可能存在不止一条连接——比如 `a` 的两个不同输出分别接到 `b` 的两个不同输入，这就需要在同一对节点间允许多条平行边，而 `DiGraph` 做不到。`MultiDiGraph` 用 `key` 区分平行边，Haystack 在 `add_edge` 时正是用 `f"{sender_socket.name}/{receiver_socket.name}"` 作为边的 `key`，从而保证每条 socket 级别的连接对应一条独立的边。

构造函数另外两个字段也值得留意：`_max_runs_per_component`（默认 `100`）是给第 4 章执行引擎防循环用的护栏；`_connection_type_validation`（默认 `True`）是一个全局开关，决定 `connect` 时是否真的做类型校验——关掉它，所有类型都会被当成兼容，这给"我知道自己在做什么"的高级用户留了后门。

## add_component：把组件登记成节点

`add_component(name, instance)` 负责把一个组件实例登记进图。它在真正 `add_node` 之前先跑了一串"准入校验"，每一条都对应一类真实的踩坑场景：

- 名字唯一：`if name in self.graph.nodes` 则抛 `ValueError`，因为后续所有连接都靠名字寻址。
- 保留名 `_debug`：这是执行期调试输出预留的名字，被占用会冲突。
- 名字不能含 `.`：因为 `connect` 用 `.` 分隔"组件名.socket名"，名字里再有点号就无法解析。
- 必须是组件：`if not isinstance(instance, Component)` 抛 `PipelineValidationError`，错误信息直接提示"是不是忘了加 `@component` 装饰器"。
- 不能跨 Pipeline 共享：通过 `getattr(instance, "__haystack_added_to_pipeline__", None)` 判断该实例是否已被别的 Pipeline 收编，是则抛 `PipelineError`。

通过校验后，它给实例打两个标记，再把节点加进图：

```python
setattr(instance, "__haystack_added_to_pipeline__", self)
setattr(instance, "__component_name__", name)
...
self.graph.add_node(
    name,
    instance=instance,
    input_sockets=instance.__haystack_input__._sockets_dict,
    output_sockets=instance.__haystack_output__._sockets_dict,
    visits=0,
)
```

这里是理解全章的关键。节点上挂了四个属性：`instance` 是组件本体；`input_sockets` 与 `output_sockets` 直接引用了组件在 `@component` 阶段就建立好的 `_sockets_dict`（来自 `Sockets` 对象，见 `sockets.py`）；`visits` 记录运行次数，初始为 `0`。

注意 `input_sockets`/`output_sockets` 是对组件内部字典的**引用**而非拷贝。这意味着后面 `connect` 改写某个 socket 的 `senders`/`receivers` 列表时，改的就是组件实例自己持有的那份 socket 对象。组件与图共享同一份 socket 状态，这正是 `__haystack_added_to_pipeline__` 标记要禁止实例跨 Pipeline 共享的根本原因——一份 socket 状态没法同时属于两张图。`remove_component` 在删节点时也必须手动把每个 `socket.senders = []`、`socket.receivers = []` 复位，并清掉 `__haystack_added_to_pipeline__`，才能让实例干净地回归"自由身"。

## socket 的三种身份：mandatory、variadic、greedy

要看懂连接逻辑，得先搞清 socket 自身携带的元数据。`InputSocket` 是个 `dataclass`（见 `types.py`），关键字段与派生属性：

- `default_value`：默认值，未设置时是哨兵类 `_empty`。
- `is_mandatory`：派生属性，`self.default_value == _empty` 即为强制输入。
- `is_lazy_variadic` / `is_greedy`：在 `__post_init__` 里根据类型的 `__metadata__` 注解推断，`is_variadic` 是两者的"或"。
- `senders`：连到此输入的组件名列表，初始为空。
- `wrap_input_in_list`：是否在传值前把输入包成一个 list，仅对 lazy variadic 有意义。

`Variadic` 与 `GreedyVariadic` 在 `types.py` 里定义为带特殊注解字符串的 `Annotated` 别名（`HAYSTACK_VARIADIC_ANNOTATION` 与 `HAYSTACK_GREEDY_VARIADIC_ANNOTATION`）[[typing]](https://docs.python.org/3/library/typing.html#typing.Annotated)。`__post_init__` 检测到这两个注解后，会做一件容易被忽略的事：把 socket 的 `type` 从 `Variadic[int]` 这种"容器类型"**解包**回内层的 `int`。它连续两次调用 `get_args`——第一次从 `Variadic` 取出 `Iterable[int]`，第二次从中取出 `int`——之所以如此是因为 Pipeline 给 variadic socket 实际喂的总是一个列表，但做类型兼容判断时我们关心的是元素类型 `int`，而非容器。若 variadic 写成了裸 `Variadic` 没带类型参数，这里会抛 `ComponentError`。

`OutputSocket` 要简单得多，只有 `name`、`type`、`receivers` 三个字段，对应"我这个输出被哪些组件接收"。

## connect：一次连接的完整生命周期

`connect(sender, receiver)` 是本章最厚重的方法（函数头部甚至挂了 `# noqa: PLR0915 PLR0912 C901` 来压制圈复杂度告警）。我们顺着它的执行流走一遍。

**第一步，解析连接字符串。** `parse_connect_string`（在 `utils.py`）按第一个 `.` 把 `"a.out"` 拆成 `("a", "out")`，没有 `.` 时 socket 名为 `None`，表示"让 Pipeline 自己挑 socket"。随后一道直接的护栏：`if sender_component_name == receiver_component_name` 就抛 `PipelineConnectError("Connecting a Component to itself is not supported.")`——禁止组件自连。

**第二步，取出两端节点的 socket 字典。** 从图节点上读 `output_sockets` 和 `input_sockets`，若组件名不在图里则抛 `ValueError`。这里有个细节：如果 sender 一个输出 socket 都没有（`if not sender_sockets`），会抛 `PipelineConnectError` 并提醒"是不是忘了用 `@component.output_types` 声明输出类型"——这是新手最常见的错误之一。

**第三步，定位候选 socket。** 如果连接字符串里显式给了 socket 名，就直接 `get` 出来，取不到则报错并把所有可选 socket 连同其类型名（`_type_name`）列出来方便排查。如果没给 socket 名，则把该端**所有** socket 都作为候选：

```python
sender_socket_candidates = [sender_socket] if sender_socket else list(sender_sockets.values())
receiver_socket_candidates = [receiver_socket] if receiver_socket else list(receiver_sockets.values())
```

**第四步，笛卡尔积式地枚举所有类型兼容的连接。** 这是连接的核心算法：

```python
for sender_sock, receiver_sock in itertools.product(sender_socket_candidates, receiver_socket_candidates):
    is_compat, conversion_strategy = _types_are_compatible(
        sender_sock.type, receiver_sock.type, self._connection_type_validation
    )
    if is_compat:
        possible_connections.append((sender_sock, receiver_sock, conversion_strategy))
```

`_types_are_compatible`（在 `type_utils.py`）返回一个二元组 `(是否兼容, 转换策略)`，下一节单独剖析。这里要理解的是 Haystack 处理"歧义"的策略层次：

- 若兼容连接多于一个且开启了类型校验，**优先保留严格匹配**（`conversion_strategy is None` 的那些），过滤掉需要类型转换的候选。这是为了向后兼容——早期 Pipeline 不支持类型转换，严格匹配必须优先。
- 若恰好只有一个兼容连接，直接采用。
- 若仍有多个兼容连接，尝试**按 socket 同名配对**（`in_sock.name == out_sock.name`）；只有当同名匹配恰好唯一时才接受，否则抛 `PipelineConnectError` 要求用户显式写明 `connect('a.具体输出', 'b.具体输入')`。
- 若一个兼容连接都没有，抛 `PipelineConnectError`，并附上 `_connections_status` 生成的、列出两端所有 socket 类型与占用状态的诊断文本。

**第五步，回写 socket 状态并落图。** 确认 `sender_socket` 与 `receiver_socket` 后，先判重——

```python
if receiver_component_name in sender_socket.receivers and sender_component_name in receiver_socket.senders:
    return self
```

已经连过的同一对 socket 不会重复加边，直接返回 `self`（`connect` 返回 Pipeline 本身，因此可以链式调用 `.connect(...).connect(...)`）。接着若接收端 socket 已有 sender（`if receiver_socket.senders`），说明要往同一个输入接第二个来源，此时调用 `_make_socket_auto_variadic` 尝试把它"升格"为 lazy variadic（下文详述）。最后两行回写状态、一行落图：

```python
sender_socket.receivers.append(receiver_component_name)
receiver_socket.senders.append(sender_component_name)
self.graph.add_edge(
    sender_component_name, receiver_component_name,
    key=f"{sender_socket.name}/{receiver_socket.name}",
    conn_type=_type_name(sender_socket.type),
    from_socket=sender_socket, to_socket=receiver_socket,
    mandatory=receiver_socket.is_mandatory,
    conversion_strategy=conversion_strategy,
)
```

边上挂的元数据很丰富：`from_socket`/`to_socket` 是两端 socket 对象、`conn_type` 是可读的类型名、`mandatory` 标记接收端是否强制、`conversion_strategy` 记录了执行期需要做的类型转换。这些信息全部在构建期算好并固化到边上，执行引擎运行时直接读取，不必再做任何类型推断。

## 类型兼容判断：strict 优先，转换兜底

`_types_are_compatible(sender, receiver, type_validation=True)` 是连接校验的大脑。它的逻辑分三层：

1. 若 `type_validation` 为 `False`（即全局开关关闭），直接返回 `(True, None)`——一切皆兼容。
2. 否则先试 `_strict_types_are_compatible`，通过则返回 `(True, None)`，没有转换策略。
3. 严格匹配失败再试 `_get_conversion_strategy`，若找到可行转换则返回 `(True, 策略)`，否则 `(False, None)`。

`_strict_types_are_compatible` 是真正判定"子类型关系"的地方，它处理了几类典型情况：

- `sender == receiver` 或 `receiver is Any` → 兼容；接收端是 `Any` 等于"什么都收"。
- `sender is Any` 而接收端不是 → **不兼容**。这是个刻意设计：`Any` 输出接到具体类型输入是不安全的，因此被拒。
- 普通类用 `issubclass` 判断子类关系；typing 泛型用 `issubclass` 会抛 `TypeError`，被 `except` 接住转入泛型分支。
- `Optional`/`Union` 的处理依托 `_safe_get_origin`，它把 `types.UnionType`（即 `X | Y` 新语法）统一归一成 `typing.Union`，从而新旧两种联合类型写法走同一套逻辑。当 sender 不是 Union 而 receiver 是 Union 时，只要 sender 与 receiver 的**任一**成员兼容即可（`any(...)`）。
- 泛型参数逐一递归比较，并对 `Callable` 单独处理（`_check_callable_compatibility`：返回类型协变、参数列表等长且逐位兼容）。

`Optional[X]` 本质是 `Union[X, None]`，所以它天然落入 Union 分支；`_type_name` 还会把"恰好两成员且含 `NoneType`"的 Union 显示成 `Optional[X]`，让报错信息更易读。

转换策略 `ConversionStrategy` 是 Haystack 较新的能力：当类型不严格相等但语义可转换时，系统记录一个策略，执行期由 `_convert_value` 实际转换。例如 `ChatMessage → str`、`str → ChatMessage`、把单值包成单元素 list（`WRAP`）、把单元素 list 拆成单值（`UNWRAP`，且为避免静默丢数据被限制在 `str`/`ChatMessage` 且只接受单元素列表）等。这套机制让"类型差一点点但完全可以自动桥接"的连接不必再写一个手工转换组件，同时把转换动作显式记录在边上而非藏在运行时。

## 多 sender 与自动 variadic 升格

`_make_socket_auto_variadic` 处理"多个上游接同一个下游输入"的场景。设计原则写在文档串里：只有当接收 socket 满足"已有 sender、尚非 variadic、类型是 `list` / `Optional[list]` / 多个 list 的 union"时，才能就地升格为 lazy variadic：

```python
if receiver_socket.is_variadic:
    return receiver_socket
receiver_origin = _safe_get_origin(receiver_socket.type)
if receiver_origin == Union:
    non_none_args = [a for a in get_args(receiver_socket.type) if a is not type(None)]
    if len(non_none_args) == 1:
        receiver_origin = _safe_get_origin(non_none_args[0])
    elif all(_safe_get_origin(arg) == list for arg in non_none_args):
        receiver_origin = list
if receiver_origin == list:
    receiver_socket.is_lazy_variadic = True
    receiver_socket.wrap_input_in_list = False
    return receiver_socket
raise error_type(...)
```

升格成功时同时把 `wrap_input_in_list` 设为 `False`，因为每个 sender 输出的本就是 list，不需要再多包一层。若类型不是 list 家族，则抛错（`connect` 路径里 `error_type` 是 `PipelineConnectError`），明确告知"该输入已被某 sender 占用，要接多个来源必须是 list 类型"。值得注意的是，这个方法在 `validate_input` 里也被复用了一次（`error_type` 改为 `ValueError`）：当用户既给某 socket 连了线、又在 `run` 时手动塞值，同样触发"多来源"检查。`connect` 的 docstring 也提醒：多 sender 汇入同一 list socket 时，结果在 `Pipeline` 下按 sender 组件名字母序排列，而 `AsyncPipeline` 不保证顺序。

## socket 状态决定入口与出口

构建期没有显式的"入口节点""出口节点"概念，Pipeline 的入口与出口完全由 socket 的连接状态**推导**出来。`descriptions.py` 里两个函数承担此事：

`find_pipeline_inputs` 收集每个组件中"可作为外部输入"的 input socket——条件是 `socket.is_variadic or (include_connected_sockets or not socket.senders)`。换言之，没有 sender 的输入 socket（无人喂它，必须从外部给）天然是 Pipeline 入口；variadic socket 即便已经连了线也仍然列入，因为它还能从外部接收额外输入。`PipelineBase.inputs()` 在此基础上做了一处微调：已连线的强制 variadic socket 被视作"非强制"（`is_mandatory and not socket.senders`），因为它的必填需求已被内部连接满足。

`find_pipeline_outputs` 对称：收集 `not socket.receivers` 的 output socket——没有下游接收的输出，自然就是 Pipeline 对外暴露的结果。

这套"由状态推导拓扑"的设计很优雅：你不需要告诉 Pipeline 谁是头、谁是尾，只要把组件连起来，悬空的输入即入口、悬空的输出即出口。第 4 章的执行引擎正是据此找到从哪些节点起跑、把哪些节点的产出收集为最终结果。

## 为什么把校验全压在构建期

通读 `connect` 会发现一个明确的取向：类型匹配、socket 存在性、自连、重复连接、多 sender 合法性，全部在 `connect` 调用的那一刻校验完毕，错就立刻抛 `PipelineConnectError`/`ValueError`。这是典型的"fail fast"——把错误暴露在离根因最近的地方。

设想另一种实现：构建期不校验，等到 `run` 时数据真的流过去才发现类型对不上。那时报错栈深埋在执行调度里，用户面对的是一个跑到一半才崩的 Pipeline，且可能已经产生了副作用（调过 LLM、写过数据库）。Haystack 选择在静态构建期就把图校验成"结构上一定能跑通"的形态，把昂贵的运行期留给真正的业务逻辑。代价是 `connect` 这个方法相当复杂，但这种复杂度是一次性付出、长期收益的——它换来的是每一条连接在落入 networkx 图之前都已被证明类型自洽，执行引擎可以放心地只管调度、不管校验。

## 本章小结

- Pipeline 的底层结构是 `networkx.MultiDiGraph`，用 `MultiDiGraph` 是为了允许两个组件间存在多条 socket 级别的平行边，每条边以 `"发送socket名/接收socket名"` 为 `key`。
- `add_component` 在登记节点前做名字唯一、保留名、点号、组件类型、跨 Pipeline 共享等准入校验，节点上挂载 `instance`、`input_sockets`、`output_sockets`、`visits` 四个属性。
- 节点上的 socket 字典是对组件内部 `_sockets_dict` 的引用而非拷贝，因此 `connect` 改写 socket 的 `senders`/`receivers` 直接作用于组件实例，这也是禁止实例跨 Pipeline 共享的根本原因。
- `InputSocket` 通过 `is_mandatory`（无默认值即强制）、`is_lazy_variadic`/`is_greedy`（由 `Annotated` 注解推断）刻画身份，variadic socket 在 `__post_init__` 中会把类型解包回元素类型。
- `connect` 用 `itertools.product` 枚举两端候选 socket 的所有兼容组合，按"严格匹配优先于可转换匹配、唯一匹配优先、同名匹配兜底"的层次消解歧义。
- 类型兼容由 `_types_are_compatible` 判定：先 `_strict_types_are_compatible` 严格判子类型，失败再用 `_get_conversion_strategy` 找转换策略；`_safe_get_origin` 把 `X | Y` 与 `typing.Union` 归一处理 `Optional`/`Union`。
- `Any` 输出连具体类型输入会被拒绝，而具体类型连 `Any` 输入被接受，体现"宽进严出"的安全取向。
- 多个 sender 接入同一 list 类型输入时，`_make_socket_auto_variadic` 会就地把 socket 升格为 lazy variadic 并将 `wrap_input_in_list` 置为 `False`。
- Pipeline 的入口/出口不是显式声明的，而是由 `find_pipeline_inputs`/`find_pipeline_outputs` 根据"输入无 sender""输出无 receiver"从 socket 状态推导而来。
- 所有结构性校验都集中在构建期完成（fail fast），落图后的连接已被证明类型自洽，运行期只负责调度而不再做类型推断。

## 动手实验

1. **实验一：观察节点元数据** — 创建两个最小组件，`pipeline.add_component("a", A())` 后，用 `pipeline.graph.nodes["a"]` 打印出该节点的字典，确认其中含有 `instance`、`input_sockets`、`output_sockets`、`visits` 四个键，并验证 `input_sockets` 与 `A().__haystack_input__._sockets_dict` 是同一个对象（用 `is` 比较）。

2. **实验二：触发各类连接报错** — 依次构造四种错误连接并捕获异常：连一个不存在的 socket 名、连两个类型不匹配的 socket、把组件连向自身、对同一对 socket 重复 `connect` 两次。观察前三者分别抛出的 `PipelineConnectError`/`ValueError` 文本，并确认第四次重复连接不会抛错而是静默返回（图中边数不增）。

3. **实验三：验证类型兼容矩阵** — 直接调用 `from haystack.core.type_utils import _types_are_compatible`，逐一测试 `(str, str)`、`(str, Any)`、`(Any, str)`、`(int, Optional[int])`、`(ChatMessage, str)` 五组，打印返回的 `(is_compatible, conversion_strategy)`，对照本章结论解释为何 `(Any, str)` 返回 `False` 而 `(ChatMessage, str)` 返回一个转换策略。

4. **实验四：观察自动 variadic 升格** — 准备一个输入类型为 `list[Document]` 的接收组件和两个产出 `list[Document]` 的发送组件，先后把两个发送端都连到同一接收 socket，连接后打印该 `InputSocket` 的 `is_lazy_variadic` 与 `wrap_input_in_list`，确认它已从 `False`/`True` 升格为 `True`/`False`；再换一个 `str` 类型的接收 socket 重复同样操作，确认第二次连接抛出 `PipelineConnectError`。

> **下一章预告**：图已经构建并校验完毕，但它还只是一张静态的连接图。第 4 章 Pipeline 执行引擎与调度将解构 Haystack 如何遍历这张图——如何用 `ComponentPriority` 与 `FIFOPriorityQueue` 决定组件运行顺序、如何处理 variadic 输入的就绪判定、如何防止无限循环，把一张图真正"跑"起来。
