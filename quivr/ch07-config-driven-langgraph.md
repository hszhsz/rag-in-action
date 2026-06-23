# 第 7 章 配置驱动的 LangGraph 工作流

在大多数 RAG 框架里，「检索—重写—生成」这条流水线是写死在代码里的：函数 A 调函数 B，函数 B 调函数 C，拓扑固定，想加一个节点就得改引擎代码。Quivr 走了另一条路——它把整条 RAG 流程抽象成**一组配置对象**，再在运行时把这些配置「翻译」成一张 LangGraph 的 `StateGraph`。换句话说，图的节点、边、条件分支统统来自配置，而不是硬编码的调用链。

这一章的主角就是「配置驱动建图」这个设计。我们会从最底层的枚举 `SpecialEdges` 看起，逐层拼出 `ConditionalEdgeConfig`、`NodeConfig`、`WorkflowConfig`，再看默认的五节点 RAG 图长什么样，最后钻进 `quivr_rag_langgraph.py` 的 `create_graph` / `_build_workflow` / `_add_node_edges` 三步曲，看引擎是如何把一串字符串配置变成一张可执行的有向图的。理解了这一章，你就理解了 Quivr 区别于「硬编码 RAG」的最核心一招。本章只讲「图怎么搭起来」，每个节点内部具体干了什么，留到第 8 章细说。

## 核心命题：RAG 是一张从配置拼出来的图

先把结论摆在最前面。Quivr 引擎在回答问题时，并不会去执行一段固定的 Python 流程，而是读取 `RetrievalConfig.workflow_config.nodes`——一个 `List[NodeConfig]`——然后逐个节点地往一张空的 `StateGraph` 里 `add_node`、`add_edge`。整张图的形状，完全由这串 `NodeConfig` 决定。

这带来的直接好处是：普通 RAG、Zendesk 工单问答、带工具调用的 agent，可以共用同一套引擎代码，只是喂进去不同的节点列表。引擎代码一行不改，流程却可以天差地别。这就是「配置驱动」四个字的分量。

为了让配置和实现既解耦又能对齐，Quivr 立了一条关键约定：**节点名即方法名**。配置里写 `name="rewrite"`，引擎建图时就 `getattr(self, "rewrite")` 把同名方法挂上去。配置层只管拓扑，实现层只管逻辑，二者靠「同名」这根线缝在一起。后面我们会反复看到这条约定如何贯穿始终。

## SpecialEdges：把 START/END 字符串翻译成 langgraph 常量

配置是用 YAML / 字典写的，里面只能写字符串。但 LangGraph 的图入口和出口是两个特殊常量 `START` 和 `END`（从 `langgraph.graph` 导入）。如何把配置里的字符串 `"START"` 对接到真正的常量？答案是 `config.py` 顶部那个小小的枚举：

```python
class SpecialEdges(str, Enum):
    start = "START"
    end = "END"
```

它继承自 `str`，所以 `SpecialEdges.start == "START"` 成立，可以直接和配置里的字符串比较。各个配置类都有一个 `resolve_special_edges` 系列方法，专门负责在初始化时把字符串 `"START"`/`"END"` 替换成 langgraph 的真常量 `START`/`END`。这是一种「边界翻译」：配置层用人类可读的字符串，越过边界进入引擎层时，自动转成框架认识的对象。

## ConditionalEdgeConfig：条件分支的声明式描述

普通的边是「从 A 直接到 B」，但 RAG 里常常需要「根据某个判断决定走 B 还是走 C」——比如「如果还需要更多工具就回到 retrieve，否则去 generate」。LangGraph 用 `add_conditional_edges` 表达这种分支，需要一个**路由函数**外加一个**条件映射**。`ConditionalEdgeConfig` 就是这种条件边的声明式描述：

```python
class ConditionalEdgeConfig(QuivrBaseConfig):
    routing_function: str
    conditions: Union[list, Dict[Hashable, str]]
```

注意 `routing_function` 是个**字符串**，不是函数对象——配置里没法写函数，只能写函数名，留到建图时再用 `getattr` 取出真正的方法。`conditions` 则描述「路由函数返回某个值时，跳到哪个节点」。它的 `__init__` 会调用 `resolve_special_edges()`，把 `conditions` 里出现的 `"END"`/`"START"` 字符串替换成 `END`/`START` 常量，无论 `conditions` 是字典还是列表都能处理。这样一来，条件边里也能优雅地连到图的终点。

## NodeConfig：一个节点的完整声明

`NodeConfig` 是整套配置体系的基本积木，它描述一个图节点的全部信息：

```python
class NodeConfig(QuivrBaseConfig):
    name: str
    description: str | None = None
    edges: List[str] | None = None
    conditional_edge: ConditionalEdgeConfig | None = None
    tools: List[Dict[str, Any]] | None = None
    instantiated_tools: List[BaseTool | Type] | None = None
```

它的 `__init__` 在 `super().__init__` 之后做了两件事：

```python
def __init__(self, **data):
    super().__init__(**data)
    self._instantiate_tools()
    self.resolve_special_edges_in_name_and_edges()
```

第一件是 `_instantiate_tools()`：如果节点声明了 `tools`（一串工具配置字典），就逐个交给 `LLMToolFactory.create_tool(tool_config.pop("name"), tool_config)` 实例化，结果存进 `instantiated_tools`。也就是说，工具在节点对象创建的那一刻就被实例化好了，运行时直接拿来用。

第二件是 `resolve_special_edges_in_name_and_edges()`：把节点自己的 `name` 以及 `edges` 列表里出现的 `"START"`/`"END"` 翻译成常量。这就是为什么默认配置里能直接写 `NodeConfig(name=START, ...)`——`START` 本身和字符串 `"START"` 在比较时是等价的，翻译后统一成 langgraph 常量。

这里有一条隐含的设计规则要记牢：**一个节点要么用 `edges`（普通边，可有多条），要么用 `conditional_edge`（条件分支），二者必有其一**。后面 `_add_node_edges` 会强制校验这一点，没有任何边的节点会直接报错。

## DefaultWorkflow：默认的五节点 RAG 图

把积木拼起来，就得到了 Quivr 开箱即用的默认 RAG 图。它定义在 `DefaultWorkflow` 枚举的 `nodes` 属性里：

```python
class DefaultWorkflow(str, Enum):
    RAG = "rag"

    @property
    def nodes(self) -> List[NodeConfig]:
        workflows = {
            self.RAG: [
                NodeConfig(name=START, edges=["filter_history"]),
                NodeConfig(name="filter_history", edges=["rewrite"]),
                NodeConfig(name="rewrite", edges=["retrieve"]),
                NodeConfig(name="retrieve", edges=["generate_rag"]),
                NodeConfig(name="generate_rag", edges=[END]),
            ]
        }
        return workflows[self]
```

这五个 `NodeConfig` 串成一条线性链，画出来就是：

```
START ──▶ filter_history ──▶ rewrite ──▶ retrieve ──▶ generate_rag ──▶ END
```

每个节点的职责（细节留给第 8 章）大致是：

- `filter_history`：根据 `max_history`（默认 10）裁剪对话历史，控制送入模型的上下文长度。
- `rewrite`：把用户当前问题结合历史改写成更适合检索的查询。
- `retrieve`：用改写后的查询去向量库召回相关文档块（数量受 `k`，默认 40 控制）。
- `generate_rag`：把召回的上下文喂给 LLM，生成最终回答。

注意这五个节点里，除了首尾两个特殊节点，中间三个的 `name`——`filter_history` / `rewrite` / `retrieve` / `generate_rag`——全都对应着引擎类里的同名方法。这就是「节点名即方法名」约定的具体落地。

## WorkflowConfig：图的容器与守门人

`WorkflowConfig` 把一串节点装进一个工作流，同时充当「守门人」，在初始化时做两道校验：

```python
class WorkflowConfig(QuivrBaseConfig):
    name: str | None = None
    nodes: List[NodeConfig] = []
    available_tools: List[str] | None = None
    validated_tools: List[BaseTool | Type] = []
    activated_tools: List[BaseTool | Type] = []

    def __init__(self, **data):
        super().__init__(**data)
        self.check_first_node_is_start()
        self.validate_available_tools()
```

第一道是 `check_first_node_is_start`：如果节点列表非空且第一个节点的 `name` 不是 `START`，直接抛 `ValueError(f"The first node should be a {SpecialEdges.start} node")`。这是一种「错误即文档」——它强制你的图必须从 `START` 出发，把约定变成了硬性约束，配置写错了立刻在加载阶段失败，而不是在运行时给出莫名其妙的行为。

第二道是 `validate_available_tools`：它把 `available_tools` 里声明的每个工具名（小写化后）拿去和 `list(TOOLS_CATEGORIES.keys()) + list(TOOLS_LISTS.keys())` 比对。这两个表来自 `llm_tools/llm_tools.py`——`TOOLS_CATEGORIES` 是工具类别（如 `WebSearchTools`、`OtherTools`），`TOOLS_LISTS` 是把每个类别下的具体工具枚举展开后的注册表。命中合法工具，就用 `LLMToolFactory.create_tool(tool, {}).tool` 实例化后存进 `validated_tools`；没命中，则用 `rapidfuzz` 做模糊匹配给出贴心建议：

```python
matches = process.extractOne(tool.lower(), valid_tools, scorer=fuzz.WRatio)
if matches:
    raise ValueError(
        f"Tool {tool} is not a valid ToolsCategory or ToolsList. Did you mean {matches[0]}?"
    )
```

这就是为什么你把 `tavily` 拼成 `tavyli` 时，会得到一句「Did you mean tavily?」的报错。`rapidfuzz` 的 `WRatio` 是一种加权模糊比分算法，能在长度不一、词序不同的字符串之间给出合理的相似度 [[rapidfuzz]](https://github.com/rapidfuzz/RapidFuzz)。`WorkflowConfig` 还有一个辅助方法 `get_node_tools(node_name)`，按节点名取回该节点预先实例化好的 `instantiated_tools`。

## 动态建图三步曲

配置就绪后，真正把它「编译」成可执行图的，是 `quivr_rag_langgraph.py` 里三个紧凑的方法。

第一步是 `create_graph`，整个流程的总入口：

```python
def create_graph(self):
    workflow = StateGraph(AgentState)
    self.final_nodes = []

    self._build_workflow(workflow)

    return workflow.compile()
```

它新建一张以 `AgentState` 为状态类型的 `StateGraph`（`StateGraph` 是 LangGraph 中以共享状态在节点间流转的核心图结构 [[LangGraph StateGraph]](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)），清空 `final_nodes`（用来记录哪些节点通向 `END`），交给 `_build_workflow` 填充，最后 `compile()` 成可运行的图返回。注意 `build_chain` 里对 `self.graph` 做了缓存——图只建一次，之后复用。

第二步是 `_build_workflow`，分两轮遍历节点列表：

```python
def _build_workflow(self, workflow: StateGraph):
    for node in self.retrieval_config.workflow_config.nodes:
        if node.name not in [START, END]:
            workflow.add_node(node.name, getattr(self, node.name))

    for node in self.retrieval_config.workflow_config.nodes:
        self._add_node_edges(workflow, node)
```

第一轮**只加节点**：跳过 `START`/`END` 这两个 langgraph 内置的虚拟端点，对其余每个节点执行 `workflow.add_node(node.name, getattr(self, node.name))`。这一行是整个配置驱动设计的灵魂——`getattr(self, node.name)` 把配置里的字符串节点名，解析成引擎类上的同名方法对象。配置说「这里有个叫 rewrite 的节点」，引擎就去找自己身上的 `rewrite` 方法挂上去。配置与实现，就靠这一个 `getattr` 对齐。第二轮再统一加边，确保所有节点都已存在后再连线，避免「边指向尚未注册的节点」的问题。

第三步是 `_add_node_edges`，逐节点决定怎么连边：

```python
def _add_node_edges(self, workflow: StateGraph, node: NodeConfig):
    if node.edges:
        for edge in node.edges:
            workflow.add_edge(node.name, edge)
            if edge == END:
                self.final_nodes.append(node.name)
    elif node.conditional_edge:
        routing_function = getattr(self, node.conditional_edge.routing_function)
        workflow.add_conditional_edges(
            node.name, routing_function, node.conditional_edge.conditions
        )
        if END in node.conditional_edge.conditions:
            self.final_nodes.append(node.name)
    else:
        raise ValueError("Node should have at least one edge or conditional_edge")
```

逻辑分三支：节点有 `edges`，就为每条边调 `workflow.add_edge`，若某条边指向 `END`，把该节点记进 `final_nodes`；节点有 `conditional_edge`，就再次用 `getattr` 把字符串 `routing_function` 解析成真方法，调 `workflow.add_conditional_edges` 接入条件分支（`add_conditional_edges` 让图按路由函数的返回值动态选择下一节点 [[LangGraph 条件边]](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph.add_conditional_edges)），若 `conditions` 里含 `END` 同样记入 `final_nodes`；两者都没有，则抛 `ValueError("Node should have at least one edge or conditional_edge")`。这正是前面提到的「一个节点必须有边」规则的执行点——又一处「错误即文档」。

`final_nodes` 这份名单在后续的流式响应里有用：引擎据此判断「答案是不是从某个终点节点流出来的」，从而在正确的时机提取文档和最终内容。

## RetrievalConfig：把一切组装进一次提问

最后，所有配置汇聚到 `RetrievalConfig`——这就是调用 `ask` 时传入引擎的总配置对象：

```python
class RetrievalConfig(QuivrBaseConfig):
    reranker_config: RerankerConfig = RerankerConfig()
    llm_config: LLMEndpointConfig = LLMEndpointConfig()
    max_history: int = 10
    max_files: int = 20
    k: int = 40  # Number of chunks returned by the retriever
    prompt: str | None = None
    workflow_config: WorkflowConfig = WorkflowConfig(nodes=DefaultWorkflow.RAG.nodes)
```

它把重排器配置、LLM 配置、历史上限 `max_history=10`、文件上限 `max_files=20`、检索块数 `k=40`，以及最关键的 `workflow_config` 组合在一起。默认的 `workflow_config` 正是用 `DefaultWorkflow.RAG.nodes` 构造的，也就是我们前面画的那条五节点线性链。它的 `__init__` 还会 `self.llm_config.set_api_key(force_reset=True)`，确保从环境变量里重新读取 API key。

到这里闭环就清楚了：用户给一份 `RetrievalConfig`（或接受默认值）→ 里面的 `workflow_config.nodes` 描述了图的形状 → 引擎 `create_graph` 时按节点名 `getattr` 出方法、按边配置连线 → 编译成 `StateGraph` → 执行回答。想换一套流程？只要换一份 `nodes` 列表，引擎代码原封不动。这就是 Quivr 把 RAG「数据化」的全部魔法。

## 本章小结

- Quivr 的 RAG 流程不是硬编码的调用链，而是运行时从一组 `NodeConfig` 动态拼出来的 LangGraph `StateGraph`，图的拓扑来自配置。
- `SpecialEdges` 枚举（`start="START"` / `end="END"`）配合各类的 `resolve_special_edges` 方法，把配置里的字符串翻译成 langgraph 的 `START`/`END` 常量。
- `ConditionalEdgeConfig` 用字符串 `routing_function` + `conditions` 映射声明条件分支，路由函数留到建图时 `getattr` 解析。
- `NodeConfig` 是基本积木，`__init__` 会 `_instantiate_tools()`（经 `LLMToolFactory.create_tool`）并 `resolve_special_edges_in_name_and_edges()`；一个节点要么有 `edges`、要么有 `conditional_edge`。
- 默认 RAG 图是五节点线性链：`START → filter_history → rewrite → retrieve → generate_rag → END`，定义在 `DefaultWorkflow.RAG.nodes`。
- `WorkflowConfig` 在初始化时充当守门人：`check_first_node_is_start` 强制首节点为 `START`（错误即文档），`validate_available_tools` 用 `rapidfuzz` 做模糊匹配并给出「Did you mean」建议。
- 动态建图三步曲：`create_graph` 建 `StateGraph` 并 `compile`；`_build_workflow` 第一轮 `add_node`、第二轮加边；`_add_node_edges` 区分普通边 / 条件边 / 报错。
- 「节点名即方法名」是核心约定，靠 `getattr(self, node.name)` 把配置字符串映射到引擎类方法，实现配置与实现的解耦又对齐。
- `_add_node_edges` 对没有任何边的节点抛 `ValueError`，并用 `final_nodes` 记录通向 `END` 的终点节点。
- `RetrievalConfig` 把 reranker、llm、`max_history=10`、`max_files=20`、`k=40` 与 `workflow_config` 组装成一次提问的总配置，默认工作流即默认 RAG 图。
- 配置驱动的最大价值：普通 RAG、Zendesk、带工具的 agent 可用不同节点组合，而引擎代码保持不变。

## 动手实验

1. **实验一：打印默认图的节点** — 在 Python 里 `from quivr_core.rag.entities.config import DefaultWorkflow`，执行 `for n in DefaultWorkflow.RAG.nodes: print(n.name, n.edges)`，确认输出正是 `START → filter_history → rewrite → retrieve → generate_rag → END` 这条线性链，并观察 `START`/`END` 已被解析成 langgraph 常量。
2. **实验二：自定义 NodeConfig 列表换图** — 手写一个去掉 `rewrite` 节点的四节点 `List[NodeConfig]`（记得把 `retrieve` 的上游边改成接 `filter_history`），用它构造 `WorkflowConfig(nodes=...)` 再塞进 `RetrievalConfig`，对照默认图体会「换配置即换流程、引擎不变」。
3. **实验三：触发首节点校验** — 故意把节点列表的第一个写成 `NodeConfig(name="filter_history", edges=["rewrite"])`（不以 `START` 开头），用它构造 `WorkflowConfig`，观察 `check_first_node_is_start` 抛出的 `ValueError("The first node should be a START node")`。
4. **实验四：体验 fuzzy 工具建议** — 构造 `WorkflowConfig(nodes=DefaultWorkflow.RAG.nodes, available_tools=["tavyli"])`（把 `tavily` 拼错），观察 `validate_available_tools` 借 `rapidfuzz` 抛出的「Did you mean tavily?」报错，再换成正确名字验证它能进入 `validated_tools`。

> **下一章预告**：这一章我们看清了图是怎么从配置「拼」出来的，但每个节点内部到底做了什么——`filter_history` 如何裁剪历史、`rewrite` 如何改写查询、`retrieve` 如何召回、`generate_rag` 如何拼上下文并调用 LLM——还是一个黑盒。第 8 章「RAG 节点详解」将逐个钻进这些方法，看清这条线性链上每一步真实的数据流转。
