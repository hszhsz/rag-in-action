# 第 9 章 Agent 与编排画布

前面八章里，RAGFlow 的每一条主线——文档解析、分块、索引、检索、重排、GraphRAG、RAPTOR——都是"一条流水线一件事"，输入输出固定，调度路径写死在代码里。但真实业务往往要把这些能力拼起来：先分类用户意图，再决定走检索还是直接回答，检索结果可能需要循环精炼，某些场景还要调外部工具甚至嵌套子 Agent。把"怎么拼"从代码里抽出来、交给一份可编辑的图（graph），就是 RAGFlow `agent/` 模块要解决的问题。

本章解构这套编排系统。它的核心是一个叫 Canvas（画布）的运行时：前端用拖拽生成一份 DSL（JSON 描述的有向图），后端把 DSL 加载成一张组件图，再按一种"路径展开式"的调度器逐个执行组件，并在组件之间用一套统一的变量引用语法传递数据。我们会从数据模型讲到调度循环，再深入 LLM、categorize、iteration/loop、agent_with_tools 这几个最能体现"可编排 Agent"设计的组件，最后看流式输出、状态管理和 DSL 迁移机制。

## 9.1 DSL 数据模型：图就是一个字典

整个画布的数据结构在 `agent/canvas.py` 顶部的 `Graph` 类文档字符串里写得很清楚。一份 DSL 是一个 JSON 对象，最重要的字段是 `components`——它是一个以组件 id 为键的字典，每个组件描述自己的类型、参数，以及与其它组件的连接关系：

```python
dsl = {
    "components": {
        "begin": {
            "obj":{
                "component_name": "Begin",
                "params": {},
            },
            "downstream": ["answer_0"],
            "upstream": [],
        },
        "retrieval_0": {
            "obj": {
                "component_name": "Retrieval",
                "params": {}
            },
            "downstream": ["generate_0"],
            "upstream": ["answer_0"],
        },
        ...
    },
    "history": [],
    "path": ["begin"],
    "retrieval": {"chunks": [], "doc_aggs": []},
    "globals": {
        "sys.query": "",
        "sys.user_id": tenant_id,
        "sys.conversation_turns": 0,
        "sys.files": []
    }
}
```

这里有几个值得注意的设计。第一，连线不是单独的"边"列表，而是直接内嵌在每个组件里的 `downstream` 和 `upstream` 两个数组——也就是说图的邻接关系是双向冗余存储的，调度时只需要看 `downstream`，但反向的 `upstream` 在做引用检查和拓扑判断时也用得上。第二，`obj` 字段在 DSL 里是纯数据（`component_name` + `params`），但加载之后会被原地替换成真正的组件对象，这个"同字段两种含义"的技巧贯穿整个 Canvas 的生命周期。第三，`path` 是一个组件 id 的序列，它既是初始执行路径，也是运行时被不断追加、记录"实际走过哪些组件"的执行轨迹——后面会看到调度器就是围着这个 `path` 转的。

除了 `components`，DSL 顶层还有几个全局状态字段：`history` 存对话历史，`retrieval` 存检索引用，`globals` 存以 `sys.` 和 `env.` 为前缀的全局变量。`Canvas.__init__` 给 `globals` 设了一份默认值，包含 `sys.query`、`sys.user_id`、`sys.conversation_turns`、`sys.files`、`sys.history` 和 `sys.date`：

```python
self.globals = {
    "sys.query": "",
    "sys.user_id": tenant_id,
    "sys.conversation_turns": 0,
    "sys.files": [],
    "sys.history": [],
    "sys.date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
}
```

这些 `sys.*` 变量是整张画布共享的"系统输入区"，任何组件都能通过引用语法读到当前用户问题、历史轮数、上传文件等上下文，而不必靠连线一级一级传。这是把"全局上下文"和"局部数据流"分离的典型做法。

## 9.2 从 DSL 到组件图：load 的两段式实例化

`Graph.__init__` 接收 DSL 字符串、租户 id、任务 id，做的第一件事是把它交给 `normalize_chunker_dsl` 做版本迁移（9.9 节细讲），然后调用 `load()`。`load` 是真正把"数据"变成"对象"的地方，逻辑非常直接：

```python
def load(self):
    self.components = self.dsl["components"]
    cpn_nms = set([])
    for k, cpn in self.components.items():
        cpn_nms.add(cpn["obj"]["component_name"])
        param = component_class(cpn["obj"]["component_name"] + "Param")()
        cpn["obj"]["params"]["custom_header"] = self.custom_header
        param.update(cpn["obj"]["params"])
        try:
            param.check()
        except Exception as e:
            raise ValueError(self.get_component_name(k) + f": {e}")

        cpn["obj"] = component_class(cpn["obj"]["component_name"])(self, k, param)

    self.path = self.dsl["path"]
```

这里的关键是 `component_class`，它定义在 `agent/component/__init__.py`：每个组件类型 `Xxx` 对应两个类——参数类 `XxxParam` 和组件类 `Xxx`。`load` 对每个 DSL 节点：先按 `component_name + "Param"` 取出参数类并实例化，用 `param.update()` 把 DSL 里的 `params` 灌进去，再调 `param.check()` 校验合法性；校验通过后，用 `component_name` 取出组件类，传入 `(self, k, param)` 构造组件对象，原地替换掉 `cpn["obj"]`。从此 `cpn["obj"]` 不再是 JSON，而是一个 `ComponentBase` 子类实例，持有对 Canvas（`self`）、自身 id（`k`）和参数的引用。

`component_class` 的查找本身体现了组件注册表的设计。它不维护一张显式的字典，而是按模块名顺序动态导入：

```python
def component_class(class_name):
    for module_name in ["agent.component", "agent.tools", "rag.flow"]:
        try:
            return getattr(importlib.import_module(module_name), class_name)
        except Exception:
            pass
    assert False, f"Can't import {class_name}"
```

而 `agent/component/__init__.py` 在导入时会用 `_import_submodules()` 遍历目录下所有 `.py` 文件（跳过 `__` 开头和 `base` 开头的），把每个模块里"定义在本模块、不以下划线开头"的类全部抓进包的命名空间。这意味着新增一个组件只要在 `agent/component/` 下放一个文件、定义好 `XxxParam` 和 `Xxx` 两个类，无需改任何注册代码就能被画布识别——这是典型的"约定优于配置"插件机制。注意 `component_class` 还会去 `agent.tools` 和 `rag.flow` 里找，所以工具类（如 `Retrieval`、`Tavily`）和分块流组件也都能作为画布组件被实例化。

## 9.3 组件契约：ComponentBase 的输入、输出与 invoke

所有组件继承自 `agent/component/base.py` 里的 `ComponentBase`。它定义了组件与调度器之间的契约。组件的实际逻辑写在 `_invoke`（同步）或 `_invoke_async`（异步）里，但调度器从不直接调它们，而是调统一的包装方法 `invoke` / `invoke_async`：

```python
def invoke(self, **kwargs) -> dict[str, Any]:
    self.set_output("_created_time", time.perf_counter())
    try:
        self._invoke(**kwargs)
    except Exception as e:
        if self.get_exception_default_value():
            self.set_exception_default_value()
        else:
            self.set_output("_ERROR", str(e))
        logging.exception(e)
    self._param.debug_inputs = {}
    self.set_output("_elapsed_time", time.perf_counter() - self.output("_created_time"))
    return self.output()
```

这个包装层做了三件横切的事：记录组件开始时间 `_created_time`、捕获异常并写入 `_ERROR` 输出（或回退到用户配置的默认值）、记录耗时 `_elapsed_time`。组件作者只管在 `_invoke` 里产出业务输出，错误处理、计时、异常兜底都被基类托管了。`invoke_async` 是它的异步孪生：优先调协程版 `_invoke_async`，否则把同步 `_invoke` 丢进线程池执行（`thread_pool_exec`），保证调度器无论组件是同步还是异步都能用同一套 `await` 语义。

输入输出都不是普通属性，而是存在 `_param.inputs` 和 `_param.outputs` 两个字典里，每个值都包了一层 `{"value": ..., "type": ...}`：

```python
def output(self, var_nm: str = None) -> Union[dict[str, Any], Any]:
    if var_nm:
        return self._param.outputs.get(var_nm, {}).get("value", "")
    return {k: o.get("value") for k, o in self._param.outputs.items()}

def set_output(self, key: str, value: Any):
    if key not in self._param.outputs:
        self._param.outputs[key] = {"value": None, "type": str(type(value))}
    self._param.outputs[key]["value"] = value
```

之所以包一层，是因为画布要把组件的"输出 schema"暴露给前端（让用户在连线时能选某组件的某个输出字段），而不只是运行时的值。`reset` 方法把所有 `outputs`/`inputs` 的 `value` 清空但保留键，正是为了在多轮对话之间保留 schema、只重置数据。

最巧妙的是 `get_input`，它是组件读取上游数据的统一入口。它遍历组件的输入元素（`get_input_elements`，默认就是 `_param.inputs`），对每个值判断它是不是一个变量引用：

```python
def get_input(self, key: str = None) -> Union[Any, dict[str, Any]]:
    ...
    for var, o in self.get_input_elements().items():
        v = self.get_param(var)
        if v is None:
            continue
        if isinstance(v, str) and self._canvas.is_reff(v):
            self.set_input_value(var, self._canvas.get_variable_value(v))
        elif isinstance(v, str) and re.search(self.variable_ref_patt, v):
            elements = self.get_input_elements_from_text(v)
            kv = {k: e.get('value', '') for k, e in elements.items()}
            self.set_input_value(var, self.string_format(v, kv))
        else:
            self.set_input_value(var, v)
        res[var] = self.get_input_value(var)
    return res
```

也就是说，输入参数可以是三种形态：一个纯变量引用（整体替换为引用值）、一段含有 `{...}` 占位符的模板字符串（逐个替换后拼回字符串）、或一个普通字面量（原样使用）。这套机制让组件之间的数据流既能传结构化对象，也能传嵌着引用的文本模板，灵活性极高。

## 9.4 变量引用：跨组件取数的统一语法

把组件接起来的"胶水"是变量引用语法。`ComponentBase` 里定义了正则 `variable_ref_patt`：

```python
variable_ref_patt = r"\{* *\{([a-zA-Z:0-9]+@[A-Za-z0-9_.-]+|sys\.[A-Za-z0-9_.]+|env\.[A-Za-z0-9_.]+)\} *\}*"
```

它能匹配三类引用：`组件id@字段名`（取某组件的某个输出）、`sys.xxx`（取系统全局变量）、`env.xxx`（取环境变量）。解析的核心是 `Graph.get_variable_value`：

```python
def get_variable_value(self, exp: str) -> Any:
    exp = exp.strip("{").strip("}").strip(" ").strip("{").strip("}")
    if exp.find("@") < 0:
        return self.globals[exp]
    cpn_id, var_nm = exp.split("@")
    cpn = self.get_component(cpn_id)
    if not cpn:
        raise Exception(f"Can't find variable: '{cpn_id}@{var_nm}'")
    parts = var_nm.split(".", 1)
    root_key = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    root_val = cpn["obj"].output(root_key)
    if not rest:
        return root_val
    return self.get_variable_param_value(root_val, rest)
```

如果引用里没有 `@`，就当成全局变量直接从 `self.globals` 取；否则按 `@` 拆出组件 id 和字段名，调那个组件的 `output(root_key)`。字段名还支持点路径（如 `agent_0@structured.title`），由 `get_variable_param_value` 沿着 dict/list/对象逐层下钻——遇到字符串还会尝试 `json.loads` 再继续，这让"输出里嵌了一段 JSON 字符串"也能被点路径穿透。

与之对称的是 `get_value_with_variable`，它处理"一整段文本里散落多个引用"的情况，用正则 finditer 把每个 `{...}` 替换成对应的值。这里有个细节：如果引用到的值是一个 `partial`（也就是 9.7 节要讲的流式生成器），它会先把流跑完、拼成完整字符串再替换。这保证了"在文本模板里引用一个流式 LLM 输出"也能得到正确结果，只是会牺牲流式性。`set_variable_value` 则是反向操作，支持把值写回某组件的输出或全局变量，循环组件就是靠它在每轮迭代里更新累加变量。

## 9.5 调度器：以 path 为中心的"展开式"执行

最值得细读的是 `Canvas.run`——它是一个异步生成器，既是调度器又是事件流的源头。它的调度模型不是经典的拓扑排序，而是一种"边执行边展开 path"的流式 BFS。理解它的关键是抓住一个不变量：`self.path` 是一个不断追加的组件 id 序列，调度器维护一个游标 `idx`，每一轮处理 `path[idx:len(path)]` 这一批组件，执行完后根据每个组件的"下一步"把后继组件追加到 `path` 末尾，再把 `idx` 推进到本轮末尾。只要 `path` 还在变长，循环就继续。

`run` 开头先把系统输入灌进 `globals`：把 `query`、`user_id`、`files` 等 kwargs 写进 `sys.*`，文件还会异步解析（`get_files_async`），并把 `sys.conversation_turns` 自增。然后正式进入调度主循环：

```python
idx = len(self.path) - 1
...
while idx < len(self.path):
    to = len(self.path)
    for i in range(idx, to):
        yield decorate("node_started", {...})
    await _run_batch(idx, to)
    to = len(self.path)
    # post-processing of components invocation
    for i in range(idx, to):
        ...
    idx = to
```

每轮先为 `[idx, to)` 这一批组件各发一个 `node_started` 事件，然后 `_run_batch` 并发执行这一批，最后逐个做后处理决定后继。`_run_batch` 用一个 `asyncio.Semaphore` 限制并发（上限取线程池的 `_max_workers`，默认 5），同一批里的组件是真正并行跑的。它还有一个很重要的"依赖未就绪则跳过"逻辑：

```python
for _, ele in cpn.get_input_elements().items():
    if isinstance(ele, dict) and ele.get("_cpn_id") and ele.get("_cpn_id") not in self.path[:i] and self.path[0].lower().find("userfillup") < 0:
        self.path.pop(i)
        t -= 1
        break
else:
    call_kwargs = cpn.get_input()
    task_fn = cpn.invoke
    i += 1
```

意思是：如果某组件的输入引用了一个还没在 `path[:i]` 里出现过（即还没执行）的组件，就把它从 `path` 里 pop 掉本轮不执行——它会在依赖被满足后通过别的连线被重新追加。这是一种轻量的"数据依赖门控"，避免组件读到尚未生成的上游输出。

后处理段是整个调度的灵魂，它决定每个组件执行完后把谁追加进 `path`。核心是这段分支：

```python
if cpn_obj.component_name.lower() in ("iterationitem","loopitem") and cpn_obj.end():
    iter = cpn_obj.get_parent()
    yield _node_finished(iter)
    _extend_path(self.get_component(cpn["parent_id"])["downstream"])
elif cpn_obj.component_name.lower() in ["categorize", "switch"]:
    _extend_path(cpn_obj.output("_next"))
elif cpn_obj.component_name.lower() in ("iteration", "loop"):
    _append_path(cpn_obj.get_start())
elif cpn_obj.component_name.lower() == "exitloop" and cpn_obj.get_parent().component_name.lower() == "loop":
    _extend_path(self.get_component(cpn["parent_id"])["downstream"])
elif not cpn["downstream"] and cpn_obj.get_parent():
    _append_path(cpn_obj.get_parent().get_start())
else:
    _extend_path(cpn["downstream"])
```

这六个分支把所有控制流编码进了"该往 path 里追加什么"这一个动作里：普通组件追加自己的 `downstream`；分支组件（categorize/switch）追加它运行时算出的 `_next`（只走选中的那一支）；循环容器（iteration/loop）追加它的起始子组件 `get_start()`；循环体子项结束时（`end()` 为真）追加容器的 `downstream`（跳出循环）；循环体没结束时，子组件链的末端（`downstream` 为空且有 parent）会回到 `get_start()` 形成回环。整张图的所有调度语义——顺序、分支、循环、嵌套——全部归约到对 `path` 的增长控制上，这是一个相当精炼的设计。

调度过程中还埋着取消检查：`run`、`_run_batch` 都会调 `is_canceled()`（查 Redis 里的 `{task_id}-cancel` 键），命中则抛 `TaskCanceledException`。错误处理则靠组件的 `exception_handler()`：如果组件出错且配置了 `exception_goto`，就把跳转目标 extend 进 path 走异常分支；如果配了 `default_value`，就发一条兜底消息；否则把错误记到 `self.error` 终止整个工作流。

## 9.6 分支组件：categorize 与 switch 如何决定走向

`categorize`（语义分类）和 `switch`（条件分支）是画布里两种分支方式，它们都通过设置 `_next` 输出告诉调度器走哪一支，但判断依据完全不同。

`switch` 是纯规则判断，定义在 `agent/component/switch.py`。它的参数是一组 `conditions`，每个条件含若干 `items`（每项是 `cpn_id`、`operator`、`value`）和一个 `logical_operator`（and/or）。`_invoke` 逐条求值：

```python
for cond in self._param.conditions:
    res = []
    for item in cond["items"]:
        cpn_v = self._canvas.get_variable_value(item["cpn_id"])
        ...
        res.append(self.process_operator(cpn_v, item["operator"], operatee))
        if cond["logical_operator"] != "and" and any(res):
            self.set_output("_next", cond["to"])
            return
    if res and all(res):
        self.set_output("_next", cond["to"])
        return
self.set_output("_next", self._param.end_cpn_ids)
```

`process_operator` 实现了 `contains`、`start with`、`empty`、`=`、`≠`、`>`、`≥` 等一整套操作符，对数字会先尝试 `float()` 转换再比较。所有条件都不命中时，落到 `end_cpn_ids`（即 ELSE 分支）。逻辑很经典：or 时任一命中即走，and 时全部命中才走，否则继续看下一条。

`categorize` 则是用 LLM 做意图分类，定义在 `agent/component/categorize.py`，它继承自 `LLM`。它的 `category_description` 配置了若干类别，每个类别有 `description`、`examples` 和目标 `to`。`update_prompt` 会把这些类别和示例拼成一段 few-shot 的系统提示，要求模型"只返回类别名"。`_invoke_async` 把用户问题发给模型后，并不盲信模型输出，而是统计每个类别名在回答里出现的次数，取出现最多的那个：

```python
category_counts = {}
for c in self._param.category_description.keys():
    count = ans.lower().count(c.lower())
    category_counts[c] = count

cpn_ids = list(self._param.category_description.items())[-1][1]["to"]
max_category = list(self._param.category_description.keys())[-1]
if any(category_counts.values()):
    max_category = max(category_counts.items(), key=lambda x: x[1])[0]
    cpn_ids = self._param.category_description[max_category]["to"]

self.set_output("category_name", max_category)
self.set_output("_next", cpn_ids)
```

这个"计数取最大"的兜底很务实：即使模型啰嗦地解释了一通，只要它提到目标类别名最多，就能正确路由；如果一个类别名都没提到，则默认落到最后一个类别。两个组件最终都把决策写进 `_next`，调度器只认 `_next`，这就是 9.5 里"分支组件追加 `_next`"那一行分支的来源。

## 9.7 LLM 组件与流式输出

`agent/component/llm.py` 里的 `LLM` 是最复杂的单体组件，也是 categorize、agent_with_tools 的基类。它的参数 `LLMParam` 涵盖了 `llm_id`、`sys_prompt`、`prompts`、采样参数、`cite`、`output_structure` 等。构造时就根据 `llm_id` 查出模型类型、建好 `LLMBundle`。

`_invoke_async` 的逻辑分几条路径，每条都对应一种产出形态。第一条是结构化输出：如果配置了带 `properties` 的 `structured` schema，就把 schema 拼进提示，用 `json_repair.loads` 解析模型回答，失败则重试（最多 `max_retries + 1` 次），成功后写进 `structured` 输出。第二条、也是最关键的一条，是流式分流：

```python
downstreams = self._canvas.get_component(self._id)["downstream"] if self._canvas.get_component(self._id) else []
ex = self.exception_handler()
if any([self._canvas.get_component_obj(cid).component_name.lower() == "message" for cid in downstreams]) and not (
    ex and ex["goto"]
):
    self.set_output("content", partial(self._stream_output_async, prompt, deepcopy(msg)))
    return
```

这里的设计意图非常精妙：**如果 LLM 组件的下游是 Message 组件，它就不立即生成完整答案，而是把 `content` 输出设成一个 `partial`——一个还没执行的流式协程**。真正的逐 token 生成被推迟到 Message 组件去消费。如果下游不是 Message（比如下游还要拿这段文本做别的处理），才走非流式路径，一次性 `_generate_async` 拿到完整回答写进 `content`。这就是为什么 9.4 节里 `get_variable_value` 要专门处理"引用到 partial 时跑完流"的情况——`content` 可能是字符串，也可能是个待跑的流。

`_prepare_prompt_variables` 负责把提示里的变量引用都解析成实际值，还会从输入里抽取 base64 图片（`data:image/` 开头），如果检测到图片且模型支持图生文，会切换到 IMAGE2TEXT 模型——多模态支持就藏在这一步。它还会处理引用 citation：当 `cite` 为真且当前有检索引用时，把 `citation_prompt` 追加进系统提示，让模型在回答里带上引文标记。

流式输出的消费在 `Canvas.run` 的后处理里。当遇到 Message 组件且它的 `content` 是 partial 时，调度器把这个流跑起来，逐块 yield `message` 事件：

```python
if isinstance(cpn_obj.output("content"), partial):
    _m = ""
    buff_m = ""
    stream = cpn_obj.output("content")()
    async def _process_stream(m):
        ...
        if m == "<think>":
            return decorate("message", {"content": "", "start_to_think": True})
        elif m == "</think>":
            return decorate("message", {"content": "", "end_to_think": True})
        buff_m += m
        _m += m
        if len(buff_m) > 16:
            ev = decorate("message", {"content": m, "audio_binary": self.tts(tts_mdl, buff_m)})
            buff_m = ""
            return ev
        return decorate("message", {"content": m})
    ...
    cpn_obj.set_output("content", _m)
```

可以看到流式输出顺带做了两件事：识别 `<think>`/`</think>` 标记发出"开始思考/结束思考"事件（前端可据此渲染思维链），以及当缓冲超过 16 字符时调用 `tts()` 合成语音（`auto_play` 开启时），实现边出文字边出语音。流跑完后把累积的完整文本写回 `content`，并发一个 `message_end` 事件，其中带上引用（reference）和附件信息。

整个 `run` 的事件序列——`workflow_started`、`node_started`、`message`、`message_end`、`node_finished`、`workflow_finished`——构成了一套完整的 SSE 风格事件协议，前端据此实时渲染每个节点的状态、流式文本和最终输出。`node_finished` 里带着该组件的输入快照、输出、耗时、错误，是可观测性的关键。

## 9.8 循环结构：iteration、loop 与容器/子项的拆分

画布支持两种循环，都采用"容器组件 + 子项组件"的拆分设计。`Iteration`（遍历数组）配 `IterationItem`，`Loop`（条件/计数循环）配 `LoopItem`。容器负责准备数据和暴露起始子组件，子项负责推进游标和判断终止。

以 `iteration` 为例。`Iteration._invoke` 只做一件事：把 `items_ref` 引用的数组取出来校验是不是 list。它的 `get_start()` 在所有组件里找 `parent_id` 等于自己、且类型是 `IterationItem` 的那个子组件并返回——这就是循环体的入口：

```python
def get_start(self):
    for cid in self._canvas.components.keys():
        if self._canvas.get_component(cid)["obj"].component_name.lower() != "iterationitem":
            continue
        if self._canvas.get_component(cid)["parent_id"] == self._id:
            return cid
```

`IterationItem` 维护一个 `self._idx` 游标。每次 `_invoke` 取出 `arr[self._idx]` 写进 `item`/`result`/`index` 三个输出（`result` 是为兼容旧 DSL 保留的别名），然后 `_idx += 1`；当 `_idx >= len(arr)` 时把 `_idx` 设为 -1，`end()` 返回 True 表示循环结束：

```python
current_item = arr[self._idx]
self.set_output("item", current_item)
self.set_output("result", current_item)
self.set_output("index", self._idx)
self._idx += 1
```

回到调度器就能看懂循环是怎么转的。当容器组件执行完，9.5 节那段分支会 `_append_path(cpn_obj.get_start())` 把子项追加进 path；子项执行后，循环体内的组件依次执行；当循环体最末端组件（`downstream` 为空且有 parent）执行完，分支 `_append_path(cpn_obj.get_parent().get_start())` 又把子项追加回来——于是子项被反复执行，`_idx` 不断推进；直到子项 `end()` 为真，分支 `_extend_path(...downstream)` 把容器的下游追加进来，跳出循环。整个循环没有任何显式的 for/while，全靠 path 的回环增长实现，这与 9.5 的统一调度模型一脉相承。

`IterationItem` 还有个 `output_collation` 方法处理"循环结果收集"：每轮开始时（`_idx > 0`）它遍历同属一个容器的兄弟组件，把标了 `ref` 的输出从子项的某字段追加到容器的累加数组里，从而把每轮的产出聚合成一个列表供循环外使用。

`Loop` 则更偏"条件循环"。它的参数有 `loop_variables`（循环变量初值）、`loop_termination_condition`（终止条件）、`maximum_loop_count`（最大次数）。`LoopItem.end()` 既看 `_idx` 是否超过最大次数，也用 `evaluate_condition` 求值终止条件（支持 string/bool/number/list/dict 各类型的操作符），按 `logical_operator` 做 and/or 聚合，命中即把 `_idx` 设 -1 结束。此外还有独立的 `ExitLoop` 组件，它本身 `_invoke` 什么都不做，但调度器对它特判：当 `ExitLoop` 的 parent 是 Loop 时，直接 `_extend_path` 到 Loop 的下游，相当于循环里的 break。

## 9.9 工具调用与子 Agent：agent_with_tools

`agent/component/agent_with_tools.py` 的 `Agent` 组件是"可编排 Agent"的集大成者，它同时继承 `LLM` 和 `ToolBase`，既是一个能调模型的组件，又是一个能挂工具、还能被别的 Agent 当成工具来调的节点。

构造函数里它把配置的工具逐个实例化。`_load_tool_obj` 用我们熟悉的 `component_class` 把每个工具的 `component_name` 实例化成组件对象，并给它一个带 `-->` 分隔的层级 id（如 `agent_0-->TavilySearch`），这个命名约定让嵌套调用的轨迹能被还原：

```python
for idx, cpn in enumerate(self._param.tools):
    cpn = self._load_tool_obj(cpn)
    original_name = cpn.get_meta()["function"]["name"]
    indexed_name = f"{original_name}_{idx}"
    self.tools[indexed_name] = cpn
```

每个工具都通过 `get_meta()` 暴露一份 OpenAI function-calling 风格的 schema（来自 `agent/tools/base.py` 的 `ToolParamBase.get_meta`），这些 meta 被收集进 `self.tool_meta`。除了本地工具，它还支持 MCP（Model Context Protocol）服务器：按 `mcp_id` 取出服务器配置，建 `MCPToolCallSession`，把远端工具也包成同样的 binding 加进 `self.tools`。最后把所有工具的 meta 通过 `chat_mdl.bind_tools` 绑给模型，并用 `LLMToolPluginCallSession` 作为统一的调用会话——模型决定调哪个工具时，会话负责真正执行并回调 `tool_use_callback` 记录轨迹。

`Agent._invoke_async` 的入口处理很能体现"子 Agent"语义：当它被当成工具调用时，会收到 `user_prompt`、`reasoning`、`context` 三个参数（这正是 `AgentParam.meta` 里声明的工具参数），把它们拼成一段结构化提示作为子任务指令。如果这个 Agent 没挂任何工具，它就退化成一个纯 LLM 组件直接调 `LLM._invoke_async`；只有挂了工具，才走带工具循环的路径。同样地，它也支持"下游是 Message 则流式"的分流——`stream_output_with_tools_async` 既要边调工具边出文本，还要在多轮对话过长时先用 `full_question` 把上下文压缩成一个独立问题，并在最后按需补上 citation 和工具产物（图片/下载链接）的 markdown。

`AgentParam` 里的 `max_rounds = 5` 限制了工具调用的最大轮数，避免模型陷入无限调用工具的循环。整个工具机制的执行细节（尤其是 `code_exec` 这类需要隔离运行用户代码的工具）涉及沙箱，留到下一章展开；本章只需理解：工具就是一种实现了 `invoke`/`get_meta` 契约的组件，Agent 组件把它们包装成模型可调用的 function，调用结果再回流进对话。

## 9.10 DSL 迁移、模板与状态管理的收尾

最后看三个支撑性机制。第一是 DSL 迁移。`agent/dsl_migration.py` 的 `normalize_chunker_dsl` 在每次加载 DSL 时都会跑一遍，它把历史版本里的组件名做重命名映射：

```python
COMPONENT_RENAMES = {
    "Splitter": "TokenChunker",
    "HierarchicalMerger": "TitleChunker",
    "PDFGenerator": "DocGenerator",
}
```

它不只是改 `component_name`，还要把所有引用这些旧组件 id 的地方一并改掉——包括组件字典的键、`downstream`/`upstream`/`parent_id`、`path`、前端 `graph` 里的 nodes/edges，以及散落在各字段里的 `组件id@字段` 变量引用（用 `VARIABLE_REF_PATTERN` 正则改写）。设计意图很明确：用户存量的工作流是宝贵资产，组件改名不能让老画布失效；迁移做成一个"读时归一化"的纯函数，既不改业务参数、也不持久化（注释里写明 "Accept legacy DSL on read, but keep the in-memory canvas in the latest schema"），保证向后兼容的同时让内存里始终是最新 schema。

第二是模板库。`agent/templates/` 下有 25 个预置工作流 JSON，覆盖了 `deep_research.json`、`text2sql_data_expert.json`、`trip_planner.json`、`stock_market_research_assistant.json`、`web_search_assistant.json` 等典型 Agent 场景，以及一系列 `ingestion_pipeline_*.json` 分块流水线模板。它们本质上就是填好了 `components`/`graph` 的完整 DSL，用户可以一键创建后再改——这是降低编排门槛的产品手段。

第三是状态管理与多轮对话。`Canvas` 把会话状态分成几层：`globals` 存系统级共享变量，`history` 存对话历史，`retrieval` 存检索引用，`memory` 存长期记忆，`variables` 存用户定义的 `env.*` 变量。`reset` 方法区分对待这几层——`sys.*` 变量按类型重置为空值，`env.*` 变量则从用户定义里恢复初值。每轮 `run` 结束时把助手回答 append 进 `history` 和 `sys.history`，下一轮组件就能通过 `get_history(window_size)` 读到带窗口限制的上下文。还有一个 `userfillup` 机制：当 path 里出现需要用户补充输入的组件时，`run` 会发出 `user_inputs` 事件并提前 return，等用户填完再从断点继续——这让画布能支持"中途暂停、等待人工输入"的交互式工作流。`__str__` 方法把整个 Canvas（含运行后的 path、history、retrieval）序列化回 DSL JSON，用于持久化当前会话状态，下次加载就能恢复现场。

## 本章小结

1. RAGFlow 的 Agent 编排核心是 `Canvas`（继承自 `Graph`），它把前端拖拽生成的 DSL（一份 JSON）加载成组件图并执行；DSL 的 `components` 字段以组件 id 为键，每个节点内嵌 `downstream`/`upstream` 连线和 `obj`（类型+参数）。
2. `Graph.load` 是两段式实例化：先按 `component_name + "Param"` 建参数对象并 `check()` 校验，再按 `component_name` 建组件对象原地替换 `obj`；`component_class` 按 `agent.component`、`agent.tools`、`rag.flow` 三个模块顺序动态查找类。
3. 组件注册采用"约定优于配置"：`agent/component/__init__.py` 自动扫描目录下所有模块抓取类，新增组件只需放一个定义了 `XxxParam`/`Xxx` 的文件，无需改注册代码。
4. `ComponentBase` 定义统一契约：业务逻辑写在 `_invoke`/`_invoke_async`，调度器只调包装层 `invoke`/`invoke_async`，由它统一记录 `_created_time`/`_elapsed_time`、捕获异常写 `_ERROR`；输入输出存在 `_param.inputs`/`outputs` 字典里并包一层 `{value, type}` 以便暴露 schema。
5. 跨组件取数靠变量引用语法 `组件id@字段`、`sys.*`、`env.*`，由 `get_variable_value` 解析（支持点路径下钻、JSON 字符串穿透）；输入参数可以是纯引用、含 `{...}` 占位符的模板、或字面量三种形态。
6. 调度器 `Canvas.run` 是异步生成器，采用"以 path 为中心的展开式 BFS"：维护游标 `idx`，每轮并发执行 `path[idx:to]`（信号量限并发，默认 5），后处理根据组件类型把后继追加进 path——所有控制流（顺序/分支/循环/嵌套）都归约为对 path 增长的控制。
7. 分支由 `_next` 输出驱动：`switch` 用规则操作符（`process_operator` 支持 contains/=/>/≥ 等）判断，`categorize` 用 LLM 分类并以"类别名计数取最大"兜底；调度器只认 `_next`，只走选中的那一支。
8. LLM 组件的关键设计是"下游是 Message 则把 `content` 设成 partial 流式协程，推迟到 Message 消费"；流式消费时顺带识别 `<think>`/`</think>` 思维链标记、按缓冲合成 TTS，并发出完整的 `node_started`/`message`/`message_end`/`node_finished` 事件协议。
9. 循环采用"容器 + 子项"拆分：`Iteration`/`IterationItem`（遍历数组）与 `Loop`/`LoopItem`（条件/计数循环）；容器的 `get_start()` 暴露循环体入口，子项维护 `_idx` 游标并以 `end()` 判终止，循环靠 path 回环增长实现，无显式 for/while；`ExitLoop` 被特判为 break。
10. `Agent` 组件同时继承 `LLM` 和 `ToolBase`，把本地工具和 MCP 远端工具都包成 OpenAI function schema 绑给模型，`max_rounds=5` 限制工具调用轮数；它还能接收 `user_prompt`/`reasoning`/`context` 作为子任务指令，从而被别的 Agent 当工具调用（id 用 `-->` 记录层级轨迹）。
11. `normalize_chunker_dsl` 在读时做版本迁移（如 `Splitter`→`TokenChunker`），同步改写所有 id、连线、path、graph 与变量引用，保证存量工作流向后兼容而不改业务参数；`agent/templates/` 下 25 个预置 DSL 模板降低编排门槛；`userfillup` 机制支持中途暂停等待人工输入。

## 动手实验

1. **实验一：画出一份最小 DSL 的执行 path** — 参照 `agent/canvas.py` 顶部 `Graph` 文档字符串里的示例 DSL，手工构造一个 `begin → categorize → (两个 message 分支)` 的 JSON，对照 9.5 节调度器后处理的六个分支，逐步推演 `self.path` 在每一轮如何增长、`idx` 如何推进，确认 categorize 命中某类别后只有对应那条 message 被追加进 path。

2. **实验二：跟踪一次流式输出的分流决策** — 阅读 `agent/component/llm.py` 的 `_invoke_async`，找到判断"下游是否有 Message 组件"那段，分别构造"LLM 下游是 Message"和"LLM 下游是另一个 LLM"两种连线，说明前者为何把 `content` 设成 `partial(self._stream_output_async, ...)` 而后者走 `_generate_async` 一次性返回；再回到 `Canvas.run` 的后处理，指出 partial 是在哪一步被真正消费成 `message` 事件的。

3. **实验三：对比 switch 与 categorize 的分支判定** — 在 `agent/component/switch.py` 的 `process_operator` 里挑三个操作符（如 `contains`、`>`、`empty`），写出各自返回 True 的输入例子；再读 `agent/component/categorize.py` 的 `_invoke_async`，解释"统计类别名出现次数取最大"这段兜底逻辑在模型回答啰嗦或完全没提类别名时分别会路由到哪里，体会规则分支与语义分支的取舍。

4. **实验四：还原一次 iteration 循环的 path 回环** — 结合 `agent/component/iteration.py`、`iterationitem.py` 和 `Canvas.run` 的后处理分支，针对一个 `items_ref` 指向 3 元素数组的迭代，逐轮写出 `IterationItem._idx` 的值、每轮被追加进 path 的组件，以及 `end()` 在第几轮返回 True 从而 `_extend_path` 到容器 `downstream` 跳出循环；额外说明 `output_collation` 如何把每轮结果聚合成数组。

> **下一章预告**：本章把工具当作"实现了 invoke 契约的组件"一笔带过，但像 `code_exec` 这类要运行用户提交代码的工具，必须在隔离环境里执行才安全。第 10 章《沙箱与代码执行》将解构 `agent/sandbox` 的隔离执行机制——代码如何被打包送进受限运行时、资源与权限怎样被约束、执行结果又如何安全地回流进画布。
