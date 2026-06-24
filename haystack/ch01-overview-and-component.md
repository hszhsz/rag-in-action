# 第 1 章 框架全景与 @component 契约

前七部我们解构的系统——RAGFlow、AnythingLLM、LlamaIndex、Quivr、Chatchat、LightRAG、GraphRAG——大多围绕"RAG"这一明确目标搭建：它们各自有偏好的解析、切分、检索、生成路径，框架结构在很大程度上服务于这条主链路。从本部开始,我们要面对一个气质不同的对象:`deepset-ai/haystack`(本书解构版本 `2.31.0-rc0`)。它不预设"你一定在做 RAG",而是给你一套乐高式的积木与拼接规则——把模型、向量库、转换器、检索器、生成器统统抽象成同构的"组件"(component),再用一张有向图(Pipeline)把它们连起来。`pyproject.toml` 里的项目描述就是这套定位的官方表述:`"LLM framework to build customizable, production-ready LLM applications. Connect components (models, vector DBs, file converters) to pipelines or agents that can interact with your data."`

正因为它通用,理解 Haystack 的第一把钥匙不是某个 RAG 算法,而是它的**组件契约**:`@component` 装饰器到底对一个类做了什么、`run()` 方法如何声明类型化的输入/输出 socket、为什么 `__init__` 必须轻量而把重活留给 `warm_up()`、组件如何 `to_dict`/`from_dict` 序列化。这一章我们就钻进 `haystack/core/component/` 这几个文件,把契约逐字读透。读懂了它,后面 Pipeline 的连接校验、调度、YAML 序列化才会一通百通。

## Haystack 是什么:一切皆 component

打开 `haystack/__init__.py`,顶层导出的东西出奇地克制:`component`、`Pipeline`、`AsyncPipeline`、`SuperComponent`/`super_component`、几个数据类(`Answer`、`Document`、`ExtractedAnswer`、`GeneratedAnswer`)、以及序列化工具 `default_to_dict`/`default_from_dict`。这份 `__all__` 清单本身就透露了框架的世界观:**核心抽象只有两个——component 和 Pipeline**,其余都是数据与工具。

与前七部对比一句:RAGFlow/Chatchat 等更像"开箱即用的 RAG 应用",LlamaIndex/Quivr 围绕检索增强提供高层 API,而 Haystack 更像一个**通用的、可组合可序列化的 LLM 编排框架**——它不替你决定 pipeline 长什么样,而是保证任何符合契约的组件都能被发现、连接、校验、保存与重建。换句话说,Haystack 卖的不是"一条 RAG 流水线",而是"造流水线的规则"。

`haystack/__init__.py` 里还有两行容易被忽略却很说明问题的代码:`haystack.logging.configure_logging()` 与 `haystack.tracing.auto_enable_tracing()`。注释明确写着,前者在未安装 `structlog` 时是 no-op,后者在没有 `opentelemetry` 或 `ddtrace` 时也是 no-op。可观测性是"可选增强而非强依赖"——这种"核心精简、能力靠可选依赖叠加"的取舍贯穿全框架。

## 仓库结构总览:主包 haystack/ 下的子包分工

用 `ls` 看主包 `haystack/` 下的子目录,职责一目了然:

- `core/`:框架内核。`core/component/` 定义组件契约,`core/pipeline/` 是图构建与执行引擎,`core/serialization.py` 提供 `default_to_dict`/`default_from_dict`,`core/super_component/` 实现把子 Pipeline 封装成单个组件的 `SuperComponent`。
- `components/`:开箱即用的具体组件(转换器、嵌入器、检索器、生成器、路由器等),它们都遵循 `core/component` 的契约。
- `dataclasses/`:核心数据模型(`Document`、`Answer` 等),即第 2 章的主角。
- `document_stores/`:文档存储抽象与内存实现。
- `tools/`:工具系统(`Tool`、`Toolset`),供 Agent 与函数调用使用。
- `tracing/`、`telemetry/`、`evaluation/`:可观测、遥测与评估。
- `marshal/`:序列化格式(如 YAML)的封送处理。
- `testing/`:测试工具,其中 `testing/sample_components/` 收录了一批最小可运行的样例组件(`add_value.py`、`accumulate.py`、`joiner.py` 等),是体会契约落地的最佳读物。
- `utils/`、`human_in_the_loop/`、`data/`:辅助工具、人在回路与内置数据。

本章只深挖 `core/component/` 三个文件:`component.py`(644 行,契约的真理之源)、`sockets.py`(143 行)、`types.py`(137 行)。

## @component 契约:装饰器到底做了什么

`component.py` 顶部的模块 docstring 自称是"the source of truth for components contract",值得逐条对照。它规定:所有组件类必须被 `@component` 装饰;可选地实现 `__init__`、`warm_up`;必须实现 `run`。装饰器实例 `component` 是模块末尾的一行 `component = _Component()`——它是 `_Component` 类的单例,既是装饰器又是命名空间(`component.output_types`、`component.set_input_types` 都是它的方法)。

`@component` 的核心逻辑在 `_Component._component(cls)` 里。它做了三件硬核的事:

第一,**校验并重建类**。先检查 `if not hasattr(cls, "run")`,没有 `run()` 就抛 `ComponentError(f"{cls.__name__} must have a 'run()' method...")`——这是"必须实现 run"契约的强制点。然后用 `typing.new_class(cls.__name__, cls.__bases__, {"metaclass": ComponentMeta}, copy_class_namespace)` **重新创建一个使用 `ComponentMeta` 元类的同名类**。注释解释为什么不直接改 `__class__`:为了让语言服务器和类型检查器正确理解类的类型,必须显式重定义类。`copy_class_namespace` 回调把原类 `__dict__` 里除 `__dict__`/`__weakref__` 外的全部成员复制过来。

第二,**注册到全局 registry**。`class_path = f"{new_cls.__module__}.{new_cls.__name__}"`,然后 `self.registry[class_path] = new_cls`。这个 `registry`(在 `_Component.__init__` 里初始化为空 dict)是反序列化的关键:`from_dict` 时凭"全限定类名"字符串就能找回类对象。docstring 里也注释了一个边角案例——在 notebook 反复执行 cell 会导致同名组件重复注册,框架对此只打 debug 日志、不报错。

第三,**注入统一的 `__repr__`**。`new_cls.__repr__ = _component_repr`,让所有组件打印时都展示组件名与输入/输出 socket 列表。

真正"生成 socket、包装实例"的活,并不在 `_component` 里,而在元类 `ComponentMeta.__call__` 中——它在每次**实例化**组件时触发。

## socket 从哪来:ComponentMeta.__call__ 的实例化钩子

`ComponentMeta.__call__(cls, *args, **kwargs)` 的注释说得很清楚:"This method is called when clients instantiate a Component and runs before `__new__` and `__init__`."也就是说,每 `MyComponent(...)` 一次,这段代码就跑一次。它先(在有 pre-init hook 时)把位置参数转成关键字参数、调用钩子,再 `super().__call__(...)` 真正构造实例。实例造好后,它做四件事把"裸实例"升级成"合格组件":

1. 检测异步:`has_async_run = hasattr(instance, "run_async")`,若存在 `run_async` 但不是协程函数则报错,并把结果记到 `instance.__haystack_supports_async__`。
2. `ComponentMeta._parse_and_set_input_sockets(cls, instance)`:解析 `run` 的签名生成输入 socket。
3. `ComponentMeta._parse_and_set_output_sockets(instance)`:从被装饰 `run` 上缓存的输出规格生成输出 socket。
4. `instance.__haystack_added_to_pipeline__ = None`:标记"尚未归属任何 Pipeline"。注释解释:一个组件不能同时被多个 Pipeline 使用,这个标志用于添加时检查。

**输入 socket 的生成**(`_parse_and_set_input_sockets`)用 `inspect.signature(method)` 遍历 `run` 的参数,跳过 `self` 与 `*args`/`**kwargs`,为每个参数建一个 `InputSocket(name=..., type=annotation)`;若参数有默认值,则把默认值塞进 `default_value`。类型注解优先取 `inspect.signature` 的结果,但如果是字符串(开启了 postponed evaluation of annotations 时),就回退到 `typing.get_type_hints` 解析。这意味着:**`run` 方法的参数名 + 类型注解,直接定义了组件的输入接口**。如果同时定义了 `run` 和 `run_async`,框架会强制二者的参数签名完全一致,否则用 `_compare_run_methods_signatures` 生成详细差异并抛错。

**输出 socket 的生成**(`_parse_and_set_output_sockets`)则依赖 `@component.output_types` 装饰器:它读 `instance.run._output_types_cache`,据此构造 `Sockets(instance, deepcopy(output_types_cache), OutputSocket)`。注意这里 `deepcopy`——注释强调要把缓存内容的所有权从"类方法"转移给"实例",这样同一个类的不同实例不会共享 socket 数据。

## 声明输出:@component.output_types 与 set_output_types

输出类型不能像输入那样从签名推断(Python 函数不声明返回 dict 的键),所以 Haystack 提供两条声明途径。

第一条是装饰器 `@component.output_types(result=int)`。看 `output_types` 的实现:它是一个**装饰器工厂**,返回的 `output_types_decorator` 在类创建时执行,先校验"只能用在 `run`/`run_async` 上"(否则报错),然后 `setattr(run_method, "_output_types_cache", {name: OutputSocket(name=name, type=type_) ...})`——把输出规格临时挂在方法对象上,等实例化时由元类取走。`sample_components/add_value.py` 里的 `AddFixedValue` 就是范本:`@component.output_types(result=int)` 修饰 `run`,返回 `{"result": value + add}`,键名 `result` 必须与声明一致。

第二条是 `component.set_output_types(self, output_1=int, output_2=str)`,用于"不想用装饰器"的场景,在 `__init__` 里调用即可。它会先检查 `run`/`run_async` 是否已带装饰器,若已带则报错(两条途径不能并用),否则直接给 `instance.__haystack_output__` 赋一组 `OutputSocket`。

输入侧也有对应的动态声明:`component.set_input_types(self, value_1=str, value_2=str)` 与单个版的 `set_input_type`。它们的前置条件很硬——`_component_run_has_kwargs(...)` 必须为真,即 `run` 方法签名里必须有 `**kwargs`,否则抛"Cannot set input types on a component that doesn't have a kwargs parameter in the 'run' method"。设计意图是:当输入是动态的、无法静态写死在签名里时(比如运行时才知道有哪些字段),用 `**kwargs` 接住,再用 `set_input_types` 补声明类型。docstring 还说明若 `run` 同时显式列了参数,显式参数优先级更高。

## 组件生命周期:轻量 __init__ + warm_up 重初始化

契约 docstring 对 `__init__` 的措辞非常重:"The `__init__` must be **extremely lightweight**, because it's a frequent operation during the construction and validation of the pipeline."为什么频繁?因为构建与校验 Pipeline 的过程中组件可能被反复构造。所以"加载模型、连后端"这类重活**不该**放在 `__init__`,而要放进可选的 `warm_up(self)`。

`warm_up` 的契约也写得很实在:它由 Pipeline 在图执行**之前**调用;而且"Pipeline will not keep track of which components it called `warm_up()` on",所以**实现者必须自己防止重复初始化**(常见写法是一个 `if self._model is None:` 的幂等守卫)。这条职责划分——构造期轻、执行期重——是 Haystack 能把"构建/校验图"和"真正跑模型"两个阶段解耦的前提。

关于 `init_parameters`:契约说组件可以在 `__init__` 里设 `self.init_parameters = {同样的入参}` 来声明"需要被持久化的状态",这些值会在 Pipeline 加载时回灌给新实例的 `__init__`。但关键的一句是:"by default the `@component` decorator saves the arguments automatically"——默认不必手写,只有当组件想覆盖默认行为时才手动设 `init_parameters`。无论哪种方式,契约都强调:**这里的值必须 JSON 可序列化**,需要的话自己先序列化成字符串。

## 序列化契约:to_dict / from_dict 与 default_to_dict / default_from_dict

为什么 `init` 参数必须是"基础"类型?答案在 `core/serialization.py`。`default_to_dict(obj, **init_parameters)` 把组件序列化成固定结构:`{"type": generate_qualified_class_name(type(obj)), "init_parameters": ...}`,其中 `type` 是全限定类名(与 registry 的 key 同构)。它还有个贴心行为:`init_parameters` 里凡是带 `to_dict()` 方法的值,会被自动递归序列化。

更值得注意的是 `component_to_dict`:当组件**没有**自定义 `to_dict` 时,它用 `inspect.signature(obj.__init__)` 遍历构造参数,逐个 `getattr(obj, param_name)` 取值——这隐含一条约定:组件应当把 init 参数原样存为同名实例属性(`self.add = add`),否则取不到会报 `SerializationError`。序列化完还会跑 `_validate_component_to_dict_output`,用 `is_allowed_type` 递归校验输出里只有 `str/int/float/bool/list/dict/set/tuple/None`,出现别的类型就抛错。这正是契约里"init 参数必须 JSON 可序列化"的强制落地点。

反序列化由 `default_from_dict(cls, data)` 负责:先校验 `"type"` 字段存在且与 `cls` 全限定名匹配(不匹配抛 `DeserializationError`),再对 `init_parameters` 做"智能还原"——`type == "env_var"` 的当作 `Secret` 还原,形如 `ComponentDevice` 的还原成设备对象,凡是带点号的全限定类名且目标类有 `from_dict` 的就 `import_class_by_name` 动态导入再递归还原,最后 `cls(**init_params)` 重建实例。

当默认行为不够用时,组件可以自定义 `to_dict`/`from_dict`。`sample_components/accumulate.py` 的 `Accumulate` 是教科书式范例:它的 `function` 参数是个 callable,不可直接 JSON 化,于是 `to_dict` 把函数序列化成 `"module.func_name"` 字符串,`from_dict` 再用 `import_module` + `getattr` 把它还原回来。这正呼应了契约 docstring 的建议:"accept either a string import path or the callable itself... store `"module_path.symbol_name"` and load it via `importlib`",并点名 `Accumulate` 作为参考实现。

## Sockets 与类型:连接校验的基础

输入/输出接口被建模为 `Sockets` 对象(`sockets.py`)。`Sockets.__init__` 接收 `component`、`sockets_dict`、`sockets_io_type` 三个参数,其中 `sockets_io_type`(`InputSocket` 或 `OutputSocket`)主要用于 `__repr__` 决定打印 "Inputs:" 还是 "Outputs:"。它把 socket 既存进内部 `_sockets_dict`,又 `self.__dict__.update(...)`,于是可以用 `sockets.question` 这样的属性访问单个 socket。

单个 socket 的数据结构在 `types.py`。`InputSocket` 是个 dataclass,字段含 `name`、`type`、`default_value`(默认 `_empty`)、`senders`(连进来的上游组件列表)等;`_empty` 是个专门的哨兵类,用来区分"没有默认值(必填)"与"默认值恰好是 None"。两个派生属性很关键:`is_mandatory` 即 `default_value == _empty`,`is_variadic` 即 `is_greedy or is_lazy_variadic`。`OutputSocket` 更简单,含 `name`、`type`、`receivers`(下游接收者列表)。`senders`/`receivers` 这两个列表就是 Pipeline 把组件连成图时填充的"边",socket 既是接口声明,也是图的端点。

`types.py` 还定义了两种变长输入标记:`Variadic` 与 `GreedyVariadic`,二者都是 `Annotated[Iterable[T], 某常量]`。区别在常量:`Variadic` 标 `HAYSTACK_VARIADIC_ANNOTATION`,`GreedyVariadic` 标 `HAYSTACK_GREEDY_VARIADIC_ANNOTATION`。`InputSocket.__post_init__` 通过检查 `self.type.__metadata__[0]` 是哪一个常量来设 `is_lazy_variadic`/`is_greedy`,并把内层真实类型"解包"出来(`Variadic[str]` 最终把 `self.type` 还原成 `str`),好让后续连接校验拿真实类型而不是 `Annotated` 外壳比对。注释点明二者的运行期差异:lazy variadic 会等齐所有上游输入再跑,而 greedy variadic 收到第一个输入就立刻运行。`sample_components/joiner.py` 的 `StringJoiner` 用 `run(self, input_str: Variadic[str])` 把多个上游的字符串拼成一个——这就是一个 socket 接多条连接的典型用法。

## 设计动机:为什么是"类型化 socket + 装饰器契约"

把上面的机制连起来看,Haystack 的设计意图就清楚了。**用装饰器统一契约**,使任何第三方类只要装上 `@component`、写好 `run`,就被自动注册、生成 socket、获得统一 repr,从而"被框架发现并接纳"——这是可组合性的来源。**用类型化 socket 描述接口**,使 Pipeline 在连接两个组件时,可以拿上游 `OutputSocket.type` 与下游 `InputSocket.type` 做静态匹配,在图构建阶段就发现接错线,而不是等运行时炸——这是可静态校验连接的来源。**用 `to_dict`/`from_dict` + 全限定类名 registry**,使整张图能被序列化成纯数据(进而 YAML 化)、再原样重建——这是可序列化、可部署的来源。而把 `__init__` 与 `warm_up` 拆开、强制 init 参数 JSON 可序列化,正是为了让"构建/校验图"这个会被频繁、廉价地反复执行的阶段,与"加载大模型"这个昂贵阶段彻底解耦。三者合一,才撑得起 `pyproject.toml` 里"customizable, production-ready"的承诺。Haystack 官方文档也把组件契约作为入门第一课展开讲解 [[Haystack Documentation]](https://docs.haystack.deepset.ai/)。

## 本章小结

- Haystack(`2.31.0-rc0`)是一个通用、可组合、可序列化的 LLM 编排框架,核心抽象只有两个:component 与 Pipeline;相比前七部更像"造流水线的乐高规则"而非现成 RAG 应用。
- `haystack/__init__.py` 的 `__all__` 极其克制,只导出 `component`、`Pipeline`/`AsyncPipeline`、`SuperComponent`、几个数据类与 `default_to_dict`/`default_from_dict`;logging 与 tracing 默认是 no-op,体现"核心精简、能力靠可选依赖叠加"。
- `@component` 即 `_Component()` 单例;`_component()` 做三件事:校验 `run` 存在、用 `ComponentMeta` 元类重建同名类、把全限定类名注册进 `registry` 供反序列化查回,并注入统一 `__repr__`。
- socket 在**实例化**时由 `ComponentMeta.__call__` 生成:输入 socket 从 `run` 签名(参数名 + 类型注解 + 默认值)解析,输出 socket 从 `@component.output_types` 缓存的 `_output_types_cache` 解析,并 `deepcopy` 以隔离实例。
- 输出类型必须显式声明,两条途径:装饰器 `@component.output_types(...)` 或 `set_output_types(...)`;输入也可动态声明 `set_input_types(...)`,但要求 `run` 带 `**kwargs`,否则报错。
- 生命周期遵循"轻 `__init__` + 重 `warm_up`"——构造期会被频繁反复执行故必须廉价,`warm_up` 由 Pipeline 在执行前调用且不防重,实现者须自行幂等。
- 序列化契约:`default_to_dict` 产出 `{"type": 全限定类名, "init_parameters": ...}`,`_validate_component_to_dict_output` 强制只允许基础类型;`default_from_dict` 校验 `type` 并智能还原 `Secret`/`ComponentDevice`/带 `from_dict` 的嵌套对象。
- `init_parameters` 默认由装饰器自动保存,必须 JSON 可序列化;不可序列化的 callable 应序列化成 `"module.symbol"` 字符串再用 `importlib` 还原,`Accumulate` 是官方参考实现。
- `InputSocket`/`OutputSocket` 是 dataclass,用 `_empty` 哨兵区分"必填"与"默认 None";`senders`/`receivers` 是 Pipeline 连接时填充的图的边;`Variadic`/`GreedyVariadic` 标记变长输入,后者收到首个输入即运行。
- 设计动机一句话:类型化 socket 带来可静态校验的连接,装饰器契约带来可组合性,`to_dict`/`from_dict` + registry 带来可序列化与可部署,`__init__`/`warm_up` 拆分带来构建期与执行期的解耦。

## 动手实验

1. **实验一:给裸实例做体检** — 在 REPL 中 `from haystack.testing.sample_components.add_value import AddFixedValue`,实例化 `c = AddFixedValue(add=5)`,打印 `c`(观察被注入的 `_component_repr` 输出的 Inputs/Outputs),再 `print(c.__haystack_input__)`、`print(c.__haystack_output__)`、`print(c.__haystack_added_to_pipeline__)`,对照本章理解元类在实例化时挂上的这几个 `__haystack_*__` 属性。
2. **实验二:序列化往返** — 对上面的 `c` 调用 `from haystack import default_to_dict`;先 `c.to_dict()` 看默认序列化结构(确认含 `type` 全限定名与 `init_parameters`),再用 `default_from_dict(AddFixedValue, c.to_dict())` 重建实例并验证 `add` 值一致;然后故意把 `init_parameters` 里塞一个不可序列化对象,观察 `_validate_component_to_dict_output` 抛出的 `SerializationError`。
3. **实验三:违反契约触发报错** — 写一个类只有普通方法没有 `run`,套上 `@component`,确认抛出 "must have a 'run()' method";再写一个 `run(self, **kwargs)` 但在 `__init__` 里调用 `component.set_input_types(self, x=int)` 的组件,正常工作后,把 `**kwargs` 删掉重试,观察 "Cannot set input types on a component that doesn't have a kwargs parameter" 报错。
4. **实验四:读懂 Variadic 解包** — 实例化 `from haystack.testing.sample_components.joiner import StringJoiner`,打印其 `__haystack_input__`,确认 `input_str` 这个 socket 的 `type` 已被 `InputSocket.__post_init__` 从 `Variadic[str]` 解包成 `str`、且 `is_variadic` 为 `True`;再对照 `StringListJoiner` 的 `Variadic[list[str]]`,理解为何 `__post_init__` 要调用两次 `get_args` 才能拿到内层类型。

> **下一章预告**:第 2 章《核心数据模型 Dataclasses》将钻进 `haystack/dataclasses/`,逐字解构 `Document`、`ChatMessage`、`Answer`、`ByteStream`、`StreamingChunk` 这些在组件间流动的数据载体——看它们如何承载内容、元数据、嵌入与多模态,以及为何每个都精心实现了 `to_dict`/`from_dict` 来满足上一章讲的序列化契约。
