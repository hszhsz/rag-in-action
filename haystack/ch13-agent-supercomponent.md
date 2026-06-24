# 第 13 章 Agent 与 SuperComponent

第 12 章我们把"工具"抽象成了 `Tool`/`Toolset`/`State` 三件套，让大模型能读懂、代码能执行。但工具本身是静态的——真正让它们活起来的，是一个会反复"思考、调用工具、观察结果、再思考"的循环。这正是 `haystack/components/agents/agent.py` 里 `Agent` 组件的职责：它把 ChatGenerator 和 ToolInvoker 串成一个 ReAct 风格的迭代回路，用 `State` 当工作记忆，用退出条件决定何时收手。

本章先逐段解构 `Agent` 的构造、循环主体与退出逻辑，再剖析它的工作记忆 `State`（`state/state.py` 与 `state/state_utils.py`），最后转向另一条工程主线——`SuperComponent`（`core/super_component/super_component.py`），看 Haystack 如何把一整条 Pipeline 折叠成一个可复用的单一组件。这两者一个让"图里跑循环"，一个让"循环可被当成图的一个节点"，恰好构成 Haystack 组合能力的闭环。

## 13.1 Agent 是一个组件：构造期的契约校验

`Agent` 用 `@component` 装饰（`agent.py` L103），意味着它本身就是一个普通组件，可以被 `add_component` 进任意 Pipeline。它的核心依赖在 `__init__` 里注入：一个必填的 `chat_generator`，一个可选的 `tools`（`ToolsType`），以及 `system_prompt`/`user_prompt`/`exit_conditions`/`state_schema`/`max_agent_steps` 等行为参数（L211-L226）。

构造期做了三道关键校验，把错误前移到"组装时"而非"运行时"：

- **ChatGenerator 必须支持工具**。它用 `inspect.signature(chat_generator.run)` 反射 `run` 方法签名，检查是否有 `tools` 参数（L262-L263）；若传了工具但生成器不支持，直接抛 `TypeError`，并明确提示"工具要传给 Agent，不要传给 chat_generator"。
- **退出条件必须合法**。有效退出条件集合是 `["text"]` 加上所有工具的名字（L270）；若 `exit_conditions` 里出现不在此集合的项，抛 `ValueError`。`"text"` 表示"模型给出纯文本回复就停"，工具名表示"某工具被成功调用就停"。
- **状态 schema 必须合法**。若提供了 `state_schema`，先用 `_validate_schema` 校验（L281-L282）。

构造的第二步是**把状态 schema 翻译成组件的输入输出类型**。这是 Agent 作为组件最精妙的地方（L302-L311）：它默认输出 `last_message: ChatMessage`，再遍历 `state_schema` 的每个键，把它同时登记为输出类型（`set_output_types`），并在该键不属于 `run` 方法固有参数时登记为输入类型（`set_input_type`）。这意味着**你在 `state_schema` 里声明的每个状态键，都会自动变成 Agent 这个组件的输入与输出 socket**——于是 Agent 能像普通组件那样在 Pipeline 里被连线。

构造的第三步是**按需创建子组件**。如果有工具，就实例化一个 `ToolInvoker`（L326-L332），把 `raise_on_tool_invocation_failure` 透传进去；如果没有工具，且当前类恰好是 `Agent`（而非子类），就打一条 warning，提示"没有工具时 Agent 退化成一个 ChatGenerator"（L333-L337）。

## 13.2 Prompt 模板与变量注册

Agent 支持两种 prompt：`system_prompt` 与 `user_prompt`，都可以是 Jinja2 模板。处理逻辑分布在构造期与 `_register_prompt_variables`（L343）里。

`user_prompt` 一旦提供，就总会创建一个 `ChatPromptBuilder`（L314-L316）。`system_prompt` 则更克制——只有当它含有 Jinja2 的 message 语法时才创建对应的 builder，判断依据是一个预编译的正则 `_JINJA2_CHAT_TEMPLATE_RE = re.compile(r"\{%\s*message\s")`（L63、L318-L320）。纯字符串的 system_prompt 不需要 builder，运行时直接 `ChatMessage.from_system` 即可（L569）。

`_register_prompt_variables` 负责把模板里的变量登记成 Agent 的输入，并做冲突检查（L378-L393）：如果某个模板变量名与 `state_schema` 的键重名，抛 `ValueError`；如果与 `run` 方法的固有参数重名，也抛 `ValueError`。这保证了运行时 kwargs 的归属不会有歧义——一个名字要么是状态键、要么是 run 参数、要么是 prompt 变量，三者互斥。是否把变量登记为"必填"则取决于 `required_variables` 是 `"*"` 还是显式列表。

## 13.3 执行前的准备：`_initialize_fresh_execution`

`run` 真正干活前，会先把一次运行所需的全部上下文打包成一个 `_ExecutionContext` dataclass（L72-L100）——它持有 `state`、`component_visits`（记录每个子组件被访问几次）、`chat_generator_inputs`、`tool_invoker_inputs`、步数 `counter`，以及若干断点/确认策略相关字段。

`_initialize_fresh_execution`（L489）负责构建全新运行的上下文，几步值得留意：

1. **渲染并拼接 prompt**。先用 `user_prompt` builder 渲染出用户消息，强制要求"恰好渲染出一条 user 角色消息"，否则抛错（L532-L540），渲染结果**追加**到 `messages` 末尾。再处理 system_prompt：若含 Jinja2 就走 builder，否则直接 `from_system`，渲染结果**前插**到 `messages` 开头（L543-L569）。如果最终所有消息都是 system 角色，会打一条 warning（L571-L572）。
2. **初始化 State**。从 kwargs 里挑出属于 `state_schema` 的键作为初始数据，构造 `State`，再把拼好的 `messages` 写进去（L574-L576）。
3. **选择回调与工具**。用 `select_streaming_callback` 合并初始化期与运行期的流式回调（L578-L580）；用 `_select_tools` 决定本轮工具（L582）。
4. **组装子组件入参**。把工具与回调分别塞进 `tool_invoker_inputs` 和 `generator_inputs`（仅当生成器支持工具时才给它传 tools，L585-L591）。

`_select_tools`（L607）支持三种 `tools` 形态：`None`（用初始化工具）、字符串列表（按名字从初始工具里筛选，非法名字抛错）、`Tool`/`Toolset` 对象。这让同一个 Agent 实例可以按运行临时收窄工具集——例如某次只允许它用只读工具。

## 13.4 循环主体：思考-调用-观察

`run`（L726）的核心是一个 `while exe_context.counter < self.max_agent_steps` 循环（L809）。`max_agent_steps` 默认 100，是防止 Agent 无限打转的硬上限；一旦触顶就打 warning 并返回当前状态（L953-L957）。每一轮的逻辑如下：

**第一步：调用 ChatGenerator。** 正常情况下，它通过 `Pipeline._run_component` 调用 chat_generator，传入当前 `state.data["messages"]` 加上预备好的生成器入参（L817-L827）。注意这里复用了 Pipeline 的组件执行器，因此 Agent 的子组件调用能纳入同一套 span 追踪与断点机制。返回的 `replies` 写回 state 的 `messages`（L857-L858）。有一个例外分支：从 ToolBreakpoint 恢复时 `skip_chat_generator` 为真，会跳过生成器、直接取最后一条消息进入工具调用（L811-L814）。

**第二步：判断是否该退出（text 条件）。** 取最后一条消息，若**没有 ToolInvoker**，或者**模型给出的是一条非空的、不含任何 tool_call 的 assistant 文本消息**，就步数加一并 `break`（L864-L872）。注释特意说明：之所以要求"非空 assistant 文本"，是为了防止一条既无工具调用又无文本的无效响应触发误退出。

**第三步：执行工具。** 若没退出，说明模型发起了工具调用。先经过确认策略 `_process_confirmation_strategies`（人类介入的钩子，L887-L893），再通过 `Pipeline._run_component` 调用 `tool_invoker`，传入待执行的工具调用消息与 `state`（L898-L909）。ToolInvoker 返回 `tool_messages` 和更新后的 `state`，二者都写回执行上下文（L941-L943）。

**第四步：判断是否该退出（工具条件）。** 若 `exit_conditions` 不是默认的 `["text"]`，调用 `_check_exit_conditions` 检查是否命中了某个工具退出条件（L946-L948）。

整个循环外层包了一个 `_create_agent_span`（L470）创建的追踪 span，把 `max_steps`/`tools`/`exit_conditions`/`state_schema` 当 tag 记录，循环结束时再记录输出与 `steps_taken`（L958-L959）。最终返回 `state.data` 的拷贝，并附上 `last_message`（L961-L964）。

`run_async`（L966）是 `run` 的镜像异步版：逻辑结构完全一致，只是把 `Pipeline._run_component` 换成 `AsyncPipeline._run_component_async`，把确认策略换成 `_process_confirmation_strategies_async`。这种"同步/异步双实现"的重复在 Haystack 里很常见——以代码冗余换取两条路径各自的清晰与可控。

## 13.5 退出条件的精确语义：`_check_exit_conditions`

`_check_exit_conditions`（L1209）决定"工具退出"是否成立，它的语义比想象中严格。它遍历所有 LLM 消息里的 `tool_calls`，只关注名字落在 `exit_conditions` 里的调用（L1224-L1228）。对每个命中的退出工具，它进一步检查该工具的执行结果有没有报错——通过比对 `tool_messages` 里 `tool_call_result.origin.tool_name` 与该调用的工具名（L1231-L1236）。

最终的返回值是 `bool(matched_exit_conditions) and not has_errors`（L1245）：**必须至少命中一个退出工具，且这些退出工具都没报错，才会退出。** 这背后的设计意图很清晰：一个被设为退出条件的工具如果调用失败了，Agent 不应该就此停下当作"成功完成"，而应该把错误反馈给模型、让它再试一轮。注释也强调"每个工具调用都会被检查，所以并行工具调用的顺序无关紧要"。

## 13.6 工作记忆：`State` 与合并语义

Agent 的工作记忆是 `State`（`state/state.py`）。它本质是一个被 `schema` 约束的字典容器，每个 schema 条目带 `type`（期望类型）和可选的 `handler`（合并函数）（L88-L94）。Agent 在构造时会确保 schema 里一定有一个 `messages` 键，类型 `list[ChatMessage]`、handler 为 `merge_lists`（`agent.py` L287-L290；`State.__init__` 也做同样兜底，L128-L129）。

`State` 最核心的设计是 **`set` 时按 handler 合并而非简单覆盖**（L153-L173）。handler 的默认选择由类型决定（L132-L141）：list 类型用 `merge_lists`，其余用 `replace_values`。这两个函数定义在 `state_utils.py`：`merge_lists`（L60）把新旧值都规整成 list 再拼接，None 当空列表处理；`replace_values`（L78）直接返回新值。`set` 还支持 `handler_override` 临时改变某次合并行为——Agent 在用确认策略改写聊天历史时就用 `handler_override=replace_values` 强制覆盖而非追加（`agent.py` L893）。

这套合并语义正是工具间共享上下文的关键：多个工具可以往同一个 list 状态键里不断追加文档，而标量状态键则被最新值覆盖。`get` 用 `deepcopy` 返回（L151），避免调用方意外篡改内部数据。

`State` 还实现了完整的序列化：`to_dict` 把 schema 用 `_schema_to_dict` 序列化（类型用 `serialize_type`、handler 用 `serialize_callable`，L17-L33），data 用 `_serialize_value_with_schema`；`from_dict` 反向还原（L191-L207）。这让 Agent 的整个运行状态能被快照、断点、恢复——也就是 13.4 里 `skip_chat_generator` 那条恢复路径的底座。

`_validate_schema`（L56）里有一处专门针对 `messages` 的硬约束：它必须是 list 类型，且元素必须是 `ChatMessage` 的子类（L73-L79），否则抛错。这保证了无论用户怎么自定义 schema，Agent 的消息记录字段始终类型正确。

## 13.7 SuperComponent：把 Pipeline 折叠成组件

如果说 Agent 让"图里能跑循环"，那 `SuperComponent`（`core/super_component/super_component.py`）解决的是相反方向的问题：**把一整条 Pipeline 当成单个组件复用**。第 12 章的 `PipelineTool` 正是靠它先把 Pipeline 包成组件、再包成工具。

实现分两层：内部基类 `_SuperComponent`（L36）和对外的 `SuperComponent`（L400）。构造时它接受一个 `pipeline` 加可选的 `input_mapping`/`output_mapping`（L37-L42）。核心思想是 **socket 重映射**：把内部 Pipeline 各组件的输入/输出 socket，映射成 SuperComponent 这个外层组件统一、简洁的输入/输出名。

构造期的工作是"对齐类型"：

- **输入侧**（L73-L85）：先拿到 `pipeline.inputs()`，若没传 `input_mapping` 就用 `_create_input_mapping` 自动生成（按 socket 名聚合所有组件，L251-L268）；再用 `_validate_input_mapping` 校验路径合法（组件与 socket 都存在，L171-L201）；最后用 `_resolve_input_types_from_mapping` 推断类型——当一个外层输入映射到多个内部 socket 时，用 `_is_compatible` 求它们的公共类型，类型冲突就抛 `InvalidMappingTypeError`，只要有一个 socket 必填则整个外层输入必填（L203-L249）。
- **输出侧**（L87-L99）：同理，注意它取的是 `pipeline.outputs(include_components_with_connected_outputs=True)`，也就是连"已被连线消费的中间输出"也能映射出来。`_create_output_mapping` 在自动模式下若发现多个组件产出同名 socket，会抛冲突错误，要求用户显式提供 mapping（L316-L338）。

运行期 `run`（L109）只做三件事：过滤掉值为 `_delegate_default` 的入参（这个哨兵对象表示"该默认值交给内部组件自己填"，定义在 `utils.py` L12），用 `_map_explicit_inputs` 把外层输入散布到内部各组件（L340-L361），调 `pipeline.run`，再用 `_map_explicit_outputs` 把内部输出收拢成外层输出（L363-L378）。`include_outputs_from` 由 `_get_include_outputs_from` 从 output_mapping 反推，确保被映射的中间组件输出会被 Pipeline 暴露出来（L128-L130）。`run_async`（L132）则要求内部 pipeline 是 `AsyncPipeline`，否则抛 `TypeError`。

类型兼容性检查 `_is_compatible`（`utils.py` L16）本身也值得一提：它先用 `_unwrap_all` 剥掉 Optional 和 Variadic 包装，再做对称的双向匹配——`Any` 与任何类型兼容、Union 取交集、泛型逐参数递归比较（L34-L135）。这套逻辑与 Pipeline 连线时的类型校验同源，保证 SuperComponent 的对外契约与内部 socket 真正对得上。

## 13.8 `@super_component` 装饰器与序列化

直接用 `SuperComponent(pipeline=..., input_mapping=..., output_mapping=...)` 是命令式写法。`@super_component` 装饰器（L561）提供了声明式写法：你写一个普通类，在 `__init__` 里构建好 `self.pipeline`（以及可选的 `self.input_mapping`/`self.output_mapping`），装饰器就帮你把它变成一个 SuperComponent。

它的实现是一套元编程：用 `functools.wraps` 包一个 `init_wrapper`，先调原始 `__init__` 建好 pipeline，校验 `pipeline` 属性存在，再调 `_SuperComponent.__init__` 完成 socket 映射初始化（L581-L595）；然后用 `types.new_class` 动态创建一个继承自 `_SuperComponent` 加原类基类的新类，拷贝原类命名空间（L601-L618），最后套上 `@component`（L626）。保留原 `__init__` 签名是为了 IDE 与文档工具能正确显示参数（L597-L598）。

序列化方面，`_to_super_component_dict`（L380）把内部 pipeline 整体 `to_dict`，连同原始（未经自动补全的）mapping 与 `is_pipeline_async` 标记一起存下，并把 `type` 字段强制写成 `SuperComponent` 的限定名（L395）。`from_dict`（L475）反向：根据 `is_pipeline_async` 选 `Pipeline` 还是 `AsyncPipeline` 反序列化内部 pipeline，再还原 SuperComponent。这样无论你用了装饰器还是命令式，存盘后都收口成统一的 `SuperComponent` 形态。此外它还转发了 `show`/`draw`（L490、L524），让你能直接画出内部 pipeline 的结构图——SuperComponent 对外是黑盒，对调试者仍是白盒。

## 本章小结

- `Agent` 本身是个 `@component`，构造期就校验"ChatGenerator 支持工具""退出条件合法""state schema 合法"，把错误前移到组装时。
- `state_schema` 的每个键会被自动登记为 Agent 组件的输入与输出 socket，这让 Agent 能像普通组件一样在 Pipeline 里连线复用。
- `run` 的主循环是 ReAct 风格的"调用 ChatGenerator → 判断 text 退出 → 经确认策略后调用 ToolInvoker → 判断工具退出"，受 `max_agent_steps` 硬上限保护。
- 循环复用 `Pipeline._run_component` 执行子组件，从而共享 span 追踪与断点机制；`run_async` 是其逐段镜像的异步实现。
- `_check_exit_conditions` 的语义是"命中退出工具且该工具未报错才退出"，失败的退出工具不会让 Agent 误判为完成。
- `State` 用 `schema`+`handler` 管理工作记忆，list 默认 `merge_lists`（拼接）、其余默认 `replace_values`（覆盖），并支持 `handler_override` 临时改变合并行为。
- `State` 完整可序列化，是 Agent 快照、断点与恢复（`skip_chat_generator` 路径）的底座；`messages` 键有"必须是 `list[ChatMessage]`"的硬约束。
- `SuperComponent` 通过 input/output socket 重映射把整条 Pipeline 折叠成单个组件，构造期用 `_is_compatible` 对齐多对一映射的公共类型。
- `@super_component` 装饰器用 `new_class` 元编程把声明式类转成 SuperComponent，序列化统一收口成 `SuperComponent` 形态并转发 `show`/`draw`。
- Agent 让"图里能跑循环"，SuperComponent 让"循环可被当成图的节点"，二者共同闭合了 Haystack 的组合能力。

## 动手实验

1. **实验一：观察 Agent 的输入输出 socket** — 用一个支持工具的 ChatGenerator 加自定义 `state_schema={"documents": {"type": list[str]}}` 构造一个 `Agent`，打印它的 `__haystack_input__` 与 `__haystack_output__`，确认 `documents` 同时出现在输入与输出 socket 里，而 `last_message` 只出现在输出里，对照 13.1 节描述验证 socket 自动登记逻辑。
2. **实验二：退出条件语义** — 给 Agent 配两个工具并把其中一个设为 `exit_conditions`，分别构造"该退出工具成功返回"与"该退出工具抛错"两条 `llm_messages`/`tool_messages`，直接调用 `_check_exit_conditions`，确认前者返回 `True`、后者返回 `False`，复现"失败的退出工具不触发退出"的语义。
3. **实验三：State 的合并与覆盖** — 创建 `State(schema={"docs": {"type": list[str]}, "title": {"type": str}})`，对 `docs` 连续 `set(["a"])`、`set(["b"])` 观察 `merge_lists` 拼成 `["a","b"]`，对 `title` 连续 `set` 两次观察 `replace_values` 的覆盖；再对 `docs` 传 `handler_override=replace_values` 验证临时覆盖生效，最后 `to_dict()` 检查序列化结构。
4. **实验四：SuperComponent 折叠 Pipeline** — 按 13.7 的示例搭一条 retriever→prompt_builder→llm 的 Pipeline，用 `SuperComponent(input_mapping={"query": ["retriever.query", "prompt_builder.query"]}, output_mapping={"llm.replies": "replies"})` 包起来，打印外层组件的输入输出 socket 确认只剩 `query` 与 `replies`，再 `to_dict()`/`from_dict()` 往返一次验证内部 pipeline 与 mapping 被正确还原。

> **下一章预告**：第 14 章 评估、可观测性与工程原则收尾。我们将看 Haystack 如何用评估组件给 RAG 与 Agent 打分，`tracing` 模块如何把本章反复出现的 span 串成可观测的执行链路，并回顾全书贯穿的契约优先、类型驱动、错误前移等工程原则，为整本书画上句号。
