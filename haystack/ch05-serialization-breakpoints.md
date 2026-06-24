# 第 5 章 序列化、YAML 与断点

第 4 章我们看清了执行引擎如何按拓扑顺序调度组件、用 `component_visits` 控制循环、把 socket 上的数据搬来搬去。但执行引擎处理的是「活着的」对象图——`Pipeline.graph` 里挂着一个个真实的组件实例。本章要回答的是一个互补的问题：怎样把这张活的对象图压成一份**纯数据**（dict / YAML / JSON），再在另一台机器、另一个进程里**原样重建**回来？以及在执行到一半时，怎样把「跑到哪了、手里拿着什么」拍成一张快照存盘，事后从断点续跑？

这两件事看似无关，实则共用同一条设计主线：Haystack 坚持「可序列化 = 纯基本类型」。一旦整张 Pipeline、整份运行状态都能落成纯数据，它就天然可以版本化（进 git）、可部署（推到 deepset Cloud）、可调试（断点存盘后离线检查）。这一章我们就沿着 `to_dict` → `from_dict` → YAML marshaller → 断点 snapshot 把这条线走通。

## 单组件序列化：type + init_parameters

整张图的序列化最终落到每个组件上，入口是 `haystack/core/serialization.py` 里的 `component_to_dict(obj, name)`。它的逻辑只有两条分支：如果组件自带 `to_dict` 方法就调它，否则走默认实现 `default_to_dict`。

默认实现非常机械：它用 `inspect.signature(obj.__init__).parameters` 拿到构造函数的形参列表，跳过 `args`/`kwargs`，然后对每个形参 `param_name` 执行 `getattr(obj, param_name)`——也就是假设「构造参数被原样存进了同名实例属性」。这是 Haystack 对组件作者的一条隐性契约：你在 `__init__` 里收到 `top_k`，就该 `self.top_k = top_k`。一旦没存，`getattr` 抛 `AttributeError`，`component_to_dict` 会检查该参数有没有默认值：有就退回默认值，没有就抛 `SerializationError`，并在错误信息里直接给出修复建议——`self.{param_name} = {param_name}` 或者自定义 `to_dict`。

收集完所有 `init_parameters` 后，交给 `default_to_dict(obj, **init_parameters)`。它产出的结构是全书最重要的一个数据形状：

```python
{"type": generate_qualified_class_name(type(obj)), "init_parameters": {...}}
```

`generate_qualified_class_name` 就是 `f"{cls.__module__}.{cls.__name__}"`——一个**全限定类名**字符串。这个字符串是反序列化时的「寻址坐标」。另外 `default_to_dict` 还做了一层贴心处理：遍历 `init_parameters`，凡是某个值自带 `to_dict` 方法的（且非 `None`），就自动调用它的 `to_dict()` 递归序列化。这意味着嵌套对象（比如一个传进来的子组件、一个配置对象）不需要组件作者手动展开。

序列化的最后一道关卡是 `_validate_component_to_dict_output`。它递归检查产出 dict 里的每一个值，只允许 `str/int/float/bool/list/dict/set/tuple/None` 这几种类型，dict 的 key 还必须是字符串，否则抛 `SerializationError`。这条校验把「纯基本类型」这条铁律落到了实处——任何残留的复杂对象都会在序列化阶段就被拦下，而不是等到 YAML dump 时才神秘失败。

## 反序列化：用全限定类名反查类

反向过程在 `default_from_dict(cls, data)`。它先做两道校验：`data` 里必须有 `type` 字段，且 `data["type"]` 必须等于 `generate_qualified_class_name(cls)`——也就是「你声称是 A 类的数据，就不能拿去构造 B 类」，否则抛 `DeserializationError`。

接着是最巧妙的一段：遍历 `init_parameters`，对每个「值是 dict 且带 `type` 键」的参数做自动嵌套反序列化。它分三种情况：

- `type == "env_var"`：识别为序列化后的 `Secret`，调 `Secret.from_dict`；
- 命中 `_is_serialized_component_device`（`type` 为 `"single"`/`"multiple"` 且结构匹配）：识别为 `ComponentDevice`，调它的 `from_dict`；
- `type` 是个含 `.` 的字符串（看起来像全限定类名）：调 `import_class_by_name(type_value)` 把类导进来，有 `from_dict` 就用，否则递归 `default_from_dict`。

最后 `cls(**init_params)` 把参数喂回构造函数，组件就重建出来了。这里能看到 `Secret` 与 `ComponentDevice` 这两个高频类型被特判——前者是密钥的环境变量引用（序列化时绝不写明文，只写「去读哪个环境变量」），后者是设备映射，它们都值得为「免手写嵌套 `from_dict`」开一个后门。

`import_class_by_name` 是整套反序列化的「DNS」。它把全限定名 `rsplit(".", 1)` 拆成模块路径与类名，用 `thread_safe_import` 导模块（线程安全是因为 Python 的 `importlib` 在多线程下并不可靠），再 `getattr` 取类。导入失败统一包成 `ImportError`，提示「Could not import class」。

值得注意的是，组件级反序列化的入口 `component_from_dict` 还接受一个 `DeserializationCallbacks`，其中 `component_pre_init` 会在组件 `__init__` 之前被调用，并允许回调**修改 `init_params`**。实现上它通过 `_hook_component_init` 这个上下文管理器把钩子挂上去。这给了部署平台一个口子——比如在初始化前注入凭据、改写模型名。

## 整图序列化：components + connections + metadata

单组件搞定后，整图就是「把每个节点序列化、把每条边记下来」。在 `haystack/core/pipeline/base.py` 的 `to_dict` 里：

- 遍历 `self.graph.nodes(data="instance")`，对每个组件调 `component_to_dict`，汇成 `components` 字典（key 是组件名）；
- 遍历 `self.graph.edges.data()`，每条边取 `from_socket.name` 与 `to_socket.name`，拼成 `{"sender": "comp.out", "receiver": "comp2.in"}` 放进 `connections` 列表；
- 再带上 `metadata`、`max_runs_per_component`、`connection_type_validation`。

注意这里不存图的拓扑结构本身——`connections` 列表里那些 `"组件名.socket名"` 字符串**就是**重建图所需的全部信息。第 3 章讲过的连接校验逻辑会在重建时重新跑一遍。

`from_dict` 是镜像操作，但比 `to_dict` 多了一层「找类」的健壮性处理。它先按 `metadata`/`max_runs_per_component`/`connection_type_validation` 建一个空 Pipeline，然后逐个组件重建。重建前先查 `component.registry`（第 1 章讲过的全局组件注册表，`@component` 装饰器会在类定义时把 `f"{module}.{name}"` 写进去）。如果 `type` 不在 registry 里，它会**主动尝试 import 对应模块**——因为组件只有被 import 过、装饰器才执行过、才会进 registry。import 后再查一次，仍找不到就报错并打印当前已注册组件清单。找到类后调 `component_from_dict` 重建实例，失败时会把（截断到 1000 字符的）组件数据一并打进错误信息，方便定位是哪个字段坏了。所有组件加完，再按 `connections` 列表逐条 `pipe.connect(sender, receiver)` 重连边。

## YAML / JSON：marshaller 这一薄层

`to_dict`/`from_dict` 产出和消费的都是 dict。要变成可存盘的字符串文本，还需要一个编解码器。Pipeline 暴露了四个方法：`dumps`（返回字符串）、`dump`（写文件对象）、`loads`（从字符串读）、`load`（从文件对象读），它们都接受一个 `marshaller` 参数，默认是 `DEFAULT_MARSHALLER = YamlMarshaller()`。

marshaller 的接口由 `haystack/marshal/protocol.py` 用一个 `Protocol` 定义，只有两个方法 `marshal`/`unmarshal`。这是个典型的鸭子类型抽象——想换成 JSON marshaller，只要实现这两个方法即可，`dumps` 完全不必改。

默认实现 `YamlMarshaller`（`haystack/marshal/yaml.py`）基于 PyYAML 的 `SafeLoader`/`SafeDumper`。`SafeDumper` 默认拒绝序列化任意 Python 对象，这正好和「纯基本类型」铁律一致。但有一个例外被特意打开了：Python 的 `tuple`。代码定义了 `YamlDumper.represent_tuple` 把元组写成 `tag:yaml.org,2002:python/tuple`，又定义 `YamlLoader.construct_python_tuple` 反向构造。为什么单独照顾元组？因为前面 `_validate_component_to_dict_output` 允许 `tuple`，而 PyYAML 的 safe 模式默认会把元组当不安全类型拒绝——不补这一手，合法的序列化数据反而 dump 不出去。

marshaller 还把 PyYAML 的异常翻译成更友好的报错：`marshal` 捕获 `RepresenterError` 抛 `TypeError`，提示「确保所有组件只序列化基本 Python 类型」；`unmarshal` 捕获 `ConstructorError` 同样给出对应提示。PyYAML 的 safe load/dump 机制本身就是为「不执行任意代码」而设计的 [[PyYAML]](https://pyyaml.org/)。

## callable 与 type 的序列化

有些组件的构造参数不是普通数据，而是**函数**或**类型**——比如自定义的回调、路由的 `output_type`。这类东西怎么落成字符串？答案在两个工具函数里。

`serialize_callable`（`haystack/utils/callable_serialization.py`）把可调用对象序列化成它的导入全路径 `f"{module.__name__}.{qualname}"`。它显式拒绝三类不可重建的对象并抛 `SerializationError`：实例方法（首参是 `self`）、lambda（`__qualname__` 含 `<lambda>`）、嵌套函数（含 `<locals>`）——道理很直白，这三类都没有稳定的导入路径，存了也找不回来。反向的 `deserialize_callable` 颇有韧性：它把路径按 `.` 切开，从最长前缀往短了试 import，找到能 import 的模块后再逐段 `getattr`；还专门处理了 classmethod/staticmethod（取 `__func__`）和被 `@tool` 装饰器换成 `Tool` 对象的情况（取 `.function`）。

`serialize_type`（`haystack/utils/type_serialization.py`）则负责把类型对象转字符串，包括 `typing` 里的特殊形态：`NoneType` 写成 `"None"`，`UnionType` 拆成 `" | ".join(...)` 递归序列化每个分支。它和 `import_class_by_name` 共享同一个 `thread_safe_import`，保证多线程导入安全。这两类工具让「函数指针」「类型注解」也能塞进纯数据流，是路由、工具这些高级组件能被序列化的前提。

## 断点：把运行中途的状态拍成快照

序列化的终极用途之一是**断点调试**。第 4 章讲的执行引擎是「一口气跑完」的；断点机制让你能在指定组件被访问到第 N 次时停下来，把当时的整份状态存盘。

断点的数据模型在 `haystack/dataclasses/breakpoints.py`，全是 `@dataclass(frozen=True)` 的不可变值对象：

- `Breakpoint`：最基础的断点，字段是 `component_name`、`visit_count`（访问到第几次才触发）、`snapshot_file_path`（存哪）。`to_dict` 直接用 `asdict`。
- `ToolBreakpoint(Breakpoint)`：针对 Agent 内部某个工具的断点，多一个 `tool_name`；为 `None` 时对所有工具生效。
- `AgentBreakpoint`：绑定到某个 Agent 组件（`agent_name`），内含一个 `Breakpoint` 或 `ToolBreakpoint`。它在 `__post_init__` 里强制约束：普通 `Breakpoint` 的 `component_name` 必须是 `"chat_generator"`，`ToolBreakpoint` 的必须是 `"tool_invoker"`——因为 Agent 内部就这两个固定子组件（下一部 Agent 章节会细讲）。

承载状态的是另外三个 dataclass：`PipelineState`（`inputs` + `component_visits` + `pipeline_outputs`）、`AgentSnapshot`（Agent 子组件的输入与访问次数）、以及总容器 `PipelineSnapshot`（`original_input_data`、`ordered_component_names`、`pipeline_state`、`break_point`、可选 `agent_snapshot`、`timestamp`、`include_outputs_from`）。`PipelineSnapshot.__post_init__` 还会校验一致性：`component_visits` 里的组件集合必须和 `ordered_component_names` 完全相等，不然抛 `ValueError`——防止存进来一份自相矛盾的快照。

## 保存与恢复快照

保存逻辑在 `haystack/core/pipeline/breakpoint.py`。`_create_pipeline_snapshot` 在断点触发时被调用，它把当前 `inputs` 和触发组件的 `component_inputs` 合并，经 `_transform_json_structure` 整形（这一步把执行引擎内部 `{"sender": ..., "value": ...}` 的信封结构剥掉，只留 value），再交给 `_serialize_with_field_fallback` 序列化，最后组装成一个 `PipelineSnapshot`。

`_serialize_with_field_fallback` 是这套机制里最体现工程取舍的一段：它先尝试整体序列化 `_serialize_value_with_schema(_deepcopy_with_exceptions(payload))`；一旦失败，且 payload 是 dict，就**逐字段重试**，只丢掉那些序列化不了的字段，并对每个被丢弃字段打 warning。设计意图很务实——快照是给人调试/续跑用的，「丢几个不可序列化字段但保住大部分可恢复状态」远比「一个字段坏掉整份快照报废」有用。即便全部字段都失败，它也返回一个结构合法的空对象 payload，让下游 `_deserialize_value_with_schema` 不至于在裸 `{}` 上抛错。

真正落盘的是 `_save_pipeline_snapshot`。它的默认行为有两个开关：其一，如果传了 `snapshot_callback`，就把快照交给回调而不写文件（方便存数据库或推远端服务）；其二，文件保存默认是**关闭**的，必须把环境变量 `HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED` 设成 `"true"` 或 `"1"` 才启用（见 `_is_snapshot_save_enabled`）。文件名按 `{agent_name?}_{component_name}_{visit_nr}_{timestamp}.json` 生成，内容是 `json.dump(pipeline_snapshot.to_dict(), ...)`。注意快照存的是 **JSON** 而非 YAML——因为它要保存的是带 schema 的运行时值，JSON 的工具链对此更顺手。

恢复一侧，`load_pipeline_snapshot(file_path)` 读 JSON、调 `PipelineSnapshot.from_dict` 还原对象，并对各类错误（文件不存在、JSON 损坏、值非法）给出明确异常。`_validate_pipeline_snapshot_against_pipeline` 则在续跑前把快照和目标 Pipeline 对账：`ordered_component_names`、`original_input_data` 的 key、`component_visits` 的 key 都必须是当前 Pipeline 里真实存在的组件，否则抛 `PipelineInvalidPipelineSnapshotError`。对账通过后它会打印「从某组件、第几次访问处恢复」，执行引擎据此把 `component_visits` 与 `inputs` 恢复到断点那一刻，继续往下跑。这正是「纯数据可序列化」这条铁律最终兑现的价值：状态既能存，就能在异机异时被完整重放。

## 本章小结

- 单组件序列化的核心数据形状是 `{"type": 全限定类名, "init_parameters": {...}}`，`default_to_dict` 依赖「构造参数被存进同名实例属性」这条隐性契约。
- `_validate_component_to_dict_output` 把「只允许基本 Python 类型」这条铁律落到强制校验上，复杂对象在序列化阶段即被拦截。
- 反序列化靠 `import_class_by_name` 用全限定名反查并 import 类；`default_from_dict` 还会自动递归处理 `Secret`、`ComponentDevice` 以及任何带全限定 `type` 的嵌套 dict。
- 整图 `to_dict` 只存 `components` + `connections`（`"组件名.socket名"` 字符串）+ `metadata`，`from_dict` 会主动 import 模块以填充 `component.registry` 再重建并重连。
- YAML marshaller 基于 PyYAML 的 safe 模式，唯一开的口子是 `tuple` 的自定义 tag，与序列化校验允许 tuple 保持一致。
- `serialize_callable`/`serialize_type` 把函数与类型转成可导入路径，并显式拒绝 lambda、嵌套函数、实例方法这些无稳定路径的对象。
- 断点数据模型 `Breakpoint`/`ToolBreakpoint`/`AgentBreakpoint` 均为 frozen dataclass，`AgentBreakpoint` 强制 `chat_generator`/`tool_invoker` 两种子组件约束。
- `_serialize_with_field_fallback` 采用「整体失败则逐字段降级」策略，宁丢个别字段也要保住大部分可恢复状态。
- 快照保存默认关闭，需 `HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED` 显式启用，存 JSON；恢复前用 `_validate_pipeline_snapshot_against_pipeline` 与目标 Pipeline 对账。

## 动手实验

1. **实验一：观察 type 与 init_parameters** — 在 Python 里实例化任意一个内置组件（如某个 Retriever 或 PromptBuilder），调用 `component_to_dict(instance, "x")`，打印结果。确认 `type` 是 `module.ClassName` 全限定名，`init_parameters` 里的值全是基本类型；再故意构造一个不把参数存进同名属性的小组件，观察 `SerializationError` 的修复建议。
2. **实验二：YAML 往返与 tuple** — 用 `Pipeline.dumps()` 把一个两组件已连接的 Pipeline 打成 YAML 字符串并 `print`，观察 `components`/`connections`/`metadata` 三段结构；再用 `Pipeline.loads(yaml_str)` 还原，比较 `to_dict()` 前后是否一致。额外把一个含 `tuple` 的字段塞进去，看 YAML 里出现的 `python/tuple` tag。
3. **实验三：跨进程重建与 registry** — 把上一步的 YAML 存盘，在一个**全新的 Python 进程**里只 `import haystack` 然后 `Pipeline.loads(open(...).read())`。先不 import 任何具体组件，观察 `from_dict` 如何根据 `type` 字符串自动 import 模块、填充 `component.registry` 并成功重建；再把 YAML 里某个 `type` 改成不存在的类名，阅读报错里打印的已注册组件清单。
4. **实验四：断点存盘与续跑** — 设置环境变量 `HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED=true`，给某个组件配一个 `Breakpoint(component_name=..., snapshot_file_path=...)` 运行 Pipeline 直到触发，找到生成的 `..._{visit}_{timestamp}.json`。用 `load_pipeline_snapshot` 读回它，打印 `pipeline_state.component_visits` 与 `original_input_data`，再故意改一个组件名后调 `_validate_pipeline_snapshot_against_pipeline`，观察 `PipelineInvalidPipelineSnapshotError`。

> **下一章预告**：本章把整张 Pipeline 与运行状态压成了纯数据。但 Pipeline 真正要处理的是文档——下一章我们进入 Haystack 的数据预处理层，解构 Converter 如何把 PDF / HTML / Markdown 等异构格式转成统一的 `Document`，以及 DocumentSplitter / DocumentCleaner 如何按句子、词数、Token 等策略切分文本，为后续嵌入与检索备料。
