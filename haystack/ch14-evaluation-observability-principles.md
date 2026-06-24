# 第 14 章 评估、可观测性与工程原则收尾

前面十三章我们把 Haystack 当成一台拆开外壳的机器，逐个看清了它的零件：`@component` 契约、可序列化的 `Pipeline`、数据驱动的调度器、`DocumentStore` 协议、检索重排链、生成器与工具系统、`Agent` 与 `SuperComponent`。但一个工程框架真正成熟的标志，不在于它能不能跑通一条流水线，而在于它能不能回答两个更难的问题：**这条流水线产出的答案到底好不好**，以及**它在生产环境里到底发生了什么**。这两个问题分别由评估（evaluation）与可观测性（observability）两套子系统承担，它们都不在"主链路"上，却决定了一个 RAG 系统能否被持续改进、被信任、被运维。

本章先拆解 Haystack 的评估模块——从纯算法指标到 LLM 裁判，再拆解它的追踪与遥测设计，看它如何在"看见内部状态"与"保护用户隐私"之间划线。最后，作为全书的终章，我们把贯穿八个开源 RAG 系统的工程原则收束成一份可迁移的清单。

## 评估的两条路线：可计算指标与 LLM 裁判

Haystack 把评估器统一放在 `components/evaluators/` 目录下，并且——这是关键设计——**每个评估器本身也是一个 `@component`**。这意味着评估不是框架外挂的脚本，而是可以拼进 `Pipeline`、可序列化、可与检索和生成共用一套数据契约的一等公民。评估器分成两条泾渭分明的路线。

第一条是**纯算法指标**，不依赖任何模型，输入真值与预测就能算出确定性分数。检索质量这一侧有三个经典指标各成一个组件：`DocumentRecallEvaluator` 衡量召回，它在 `document_recall.py` 里用 `RecallMode` 枚举区分两种语义——`SINGLE_HIT`（只要命中任一真值文档就记 1 分，个体分非 0 即 1）与 `MULTI_HIT`（按命中比例给分）；`DocumentMRREvaluator` 计算平均倒数排名，核心一行是 `reciprocal_rank = 1 / (rank + 1)`，命中越靠前分越高；`DocumentNDCGEvaluator` 计算归一化折损累计增益，用 `dcg / idcg` 把排序质量压到 0–1 区间，代码里 `log2(i + 2)` 的 `i + 2` 注释明确解释是因为 `i` 从 0 开始计数。答案质量这一侧则有 `SASEvaluator`（语义答案相似度），它用 Hugging Face 上的 Bi-Encoder 或 Cross-Encoder 计算预测答案与真值的语义相似度，默认模型是多语种的 `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`，并通过 `LazyImport` 把 `sentence-transformers` 设为可选依赖——不装就不报错，用到才检查。还有最朴素的 `AnswerExactMatchEvaluator` 做字符串精确匹配。

第二条路线是 **LLM 裁判**，用一个大模型来评判另一个大模型的输出，专门应对"无标准答案"的主观质量。基类是 `LLMEvaluator`，它接受 `instructions`（评判标准的自然语言描述）、`inputs`/`outputs`（声明输入字段与期望输出字段的类型）、`examples`（few-shot 示例）。它默认的裁判是一个 `OpenAIChatGenerator`，而且被强制设为 JSON 模式——`generation_kwargs` 里写死了 `{"response_format": {"type": "json_object"}, "seed": 42}`。这里有两个值得停下来看的设计决策：`json_object` 保证模型必须吐出可解析的结构化结果，让"裁判意见"能被程序消费而不是当成自由文本；`seed=42` 则尽量固定采样，让同一份输入在多次评估间保持可复现——评估如果每次结果都飘，就失去了作为基线的意义。`LLMEvaluator` 还在 `__init__` 阶段调用 `validate_init_parameters` 校验 `inputs`/`outputs` 的结构，并用 `component.set_input_types` 把声明的输入字段动态注册成组件的输入端口，让一个通用裁判能适配任意评判任务。

在 `LLMEvaluator` 之上，Haystack 预制了两个面向 RAG 的专用裁判。`FaithfulnessEvaluator`（忠实度）检查生成答案能否从给定上下文中推断出来：它让 LLM 先把答案拆成多条独立陈述（statements），再逐条判断能否被上下文支撑，最终分数是"可被支撑的陈述比例"。它的 `_DEFAULT_EXAMPLES` 里藏着精心设计的负例——把"柏林是德国首都"这类能从上下文推出的陈述记 1 分，把"巴黎是法国首都"这类上下文里根本没提的记 0 分——用 few-shot 把"忠实"这个抽象概念锚定成可操作的判据。另一个 `ContextRelevanceEvaluator`（上下文相关性）则评判检索到的上下文与问题是否相关，定位的是检索环节而非生成环节的质量。两个裁判都通过 `deserialize_chatgenerator_inplace` 支持把内部的 ChatGenerator 一并序列化，保证整个评估流水线可存盘可复现。

## 评估结果的承载：EvaluationRunResult

算出分数只是第一步，怎么组织、对比、导出这些分数才是评估能否融入工程流程的关键。`evaluation/eval_run_result.py` 里的 `EvaluationRunResult` 承担这个职责。它的构造函数接受三样东西：`run_name`（这次评估的名字）、`inputs`（每列输入的取值列表）、`results`（每个指标的 `score` 聚合分与 `individual_scores` 个体分列表）。

这个类最值得学习的是它的**防御式校验**。构造时它会检查所有输入列长度一致（`len({len(lst) for lst in inputs.values()}) != 1` 就报错），检查每个指标都同时带 `score` 与 `individual_scores`，还检查个体分的数量与输入样本数严格相等——一旦评估数据出现"行数对不齐"这种最隐蔽也最致命的错误，它在入口就拦下，而不是等到画图或导表时才崩。这正是第 7 章 `DocumentStore` 那条"在边界处校验数据契约"原则在评估场景的复现。

在输出侧，`EvaluationRunResult` 提供三种报告。`aggregated_report` 给出每个指标的聚合分总览；`detailed_report` 把每条输入样本与它在各指标上的得分拼成一张明细表；`comparative_detailed_report` 则接受另一个 `EvaluationRunResult`，把两次评估并排对比——它会用 `run_name` 给列名加前缀避免冲突，还会在两次 run 同名或输入列不一致时打 warning 而非直接报错，体现"对比是探索性操作，应宽容而非苛刻"的取舍。三种报告统一通过 `_handle_output` 分发到三种格式：`json`（返回 dict）、`df`（返回 pandas `DataFrame`，且 pandas 也是 `LazyImport` 的可选依赖）、`csv`（写文件）。`_write_to_csv` 把 `PermissionError`、`OSError` 和兜底 `Exception` 分别捕获并返回可读的错误字符串而不是抛出——又一次"导出失败不该让整个评估白跑"的工程克制。

## 可观测性的抽象：Tracer 与 Span 双接口

评估回答"好不好"，追踪回答"发生了什么"。Haystack 的追踪系统在 `tracing/tracer.py` 里用两个抽象基类搭骨架：`Span` 代表一次被instrument的操作，`Tracer` 负责创建并提交 span。`Span` 接口要求实现 `set_tag`（打单个标签），并附赠 `set_tags`（批量）、`raw_span`（拿到底层 span 对象）、`get_correlation_data_for_logs`（把追踪与日志关联）等默认方法。`Tracer` 接口则要求实现 `trace`（一个 `@contextlib.contextmanager`，进入即开 span、退出即关）与 `current_span`。

这套设计里最精巧的是 **`ProxyTracer` 代理模式**。全局只有一个 `tracer: ProxyTracer = ProxyTracer(provided_tracer=NullTracer())` 实例，它内部持有一个 `actual_tracer`。用户代码到处 `from haystack.tracing import tracer` 直接引用这个对象，而启用/切换/禁用追踪只需替换 `tracer.actual_tracer`——`enable_tracing` 换成真实 tracer，`disable_tracing` 换回 `NullTracer`。源码里那段注释把动机说得很白：用代理是为了"在不改变全局 tracer 实例的前提下启停追踪"，否则一旦用户在各模块里直接 import 了这个对象，就只能去每个模块里 monkey-patch。这是"用一层间接解决引用扩散"的经典手法。

默认值同样意味深长：开箱即用的 `actual_tracer` 是 `NullTracer`，它的 `trace` 直接 `yield NullSpan()`，`NullSpan.set_tag` 是空操作。也就是说**追踪默认完全关闭、零开销**，框架不会因为内置了可观测性能力就强行给每个用户增加运行负担——可观测性是"按需点亮"而非"默认常亮"。`auto_enable_tracing()` 在模块加载末尾被调用，按 `HAYSTACK_AUTO_TRACE_ENABLED` 环境变量决定是否自动探测：它先试 `_auto_configured_opentelemetry_tracer()`（巧妙地通过"开一个 span 看它是不是 `NonRecordingSpan`"来判断 OpenTelemetry 是否真的配置好了），再退而试 `_auto_configured_datadog_tracer()`（检查 `ddtrace` 的 `tracer.enabled`）。两个探测都包在 `try/except ImportError` 里——没装对应 SDK 就静默跳过，绝不因为可选后端缺失而污染用户的导入过程。

## 隐私优先：content tracing 的默认关闭

追踪一条 RAG 流水线时，最有诊断价值的信息恰恰是最敏感的：用户的查询原文、检索到的文档内容、模型生成的答案。Haystack 在这里做了一个明确的隐私设计——把"内容"与"元数据"分成两类标签。普通的 `set_tag` 记录组件名、访问次数这类元数据；而记录敏感内容必须走专门的 `set_content_tag`，它的实现只有一句要害：`if tracer.is_content_tracing_enabled: self.set_tag(key, value)`。

`is_content_tracing_enabled` 在 `ProxyTracer.__init__` 里从环境变量读取：`os.getenv(HAYSTACK_CONTENT_TRACING_ENABLED_ENV_VAR, "false").lower() == "true"`——**默认 `false`**。换句话说，即便你打开了追踪，查询和文档内容也不会被记录，除非你显式地把 `HAYSTACK_CONTENT_TRACING_ENABLED` 设成 `true`。这是"隐私默认开"（privacy by default）的标准做法：让最安全的配置成为不需要任何操作就能得到的配置，把记录敏感数据这件事变成一个需要用户主动承担责任的决定。

`tracing/logging_tracer.py` 里的 `LoggingTracer` 是这套抽象最轻量的落地实现，主要供本地调试用。它在 `trace` 的 `finally` 块里——无论成功还是抛异常都执行——把 `operation_name` 和所有 tag 用 `logger.debug` 打出来，还支持 `tags_color_strings` 给不同标签套 ANSI 颜色码（配合常量 `RESET_COLOR = "\033[0m"`），方便在终端里一眼区分组件输入输出。它的 `current_span` 直接返回 `None`，坦白自己"不存储 span"——一个刻意做薄的实现，把复杂度留给 OpenTelemetry/Datadog 这类生产级后端。

## 把追踪织进流水线：pipeline.run 的 span

抽象再好，也要在主链路上真正被调用才有意义。`core/pipeline/pipeline.py` 的 `run` 方法把整条流水线的执行包在一个名为 `haystack.pipeline.run` 的 span 里，给它打上 `input_data`、`output_data`、`metadata`、`max_runs_per_component` 等标签。每个组件执行时，`_create_component_span` 再开一个子 span，记录组件的访问次数（`_COMPONENT_VISITS` 标签），并用 `span.set_content_tag(_COMPONENT_INPUT, ...)` 和 `set_content_tag(_COMPONENT_OUTPUT, ...)` 记录组件的输入输出——注意这里用的是 `set_content_tag` 而非 `set_tag`，正好接上了上一节的隐私开关：组件级的数据流默认不落盘。

一个容易被忽略却很重要的细节：流水线在把输入交给 tracer 之前会先 `deepcopy`。因为组件 `run` 可能就地修改传入的对象，如果直接把引用交给追踪后端，记录下来的"输入"可能已经被后续执行污染。深拷贝保证 span 里留下的是执行那一刻的真实快照。这与第 5 章讲的断点机制（`_save_pipeline_snapshot`）共享同一种世界观——**要让事后能复盘，就必须在事发当刻冻结状态**，无论是为了调试快照还是为了追踪标签。

与追踪互补的是遥测。`telemetry/_telemetry.py` 里的 `Telemetry` 类向 PostHog 上报匿名使用统计，帮 deepset 团队了解哪些功能真正被用到。它的设计处处体现"遥测绝不能拖累或拖垮产品"的克制：用户 id 是写在 `~/.haystack/config.yaml` 里的一个随机 `uuid4`，不含任何身份信息；`MIN_SECONDS_BETWEEN_EVENTS = 60` 限流，同一条流水线至多每 60 秒上报一次；通过 `HAYSTACK_TELEMETRY_ENABLED` 环境变量可一键退出；最关键的是 `send_telemetry` 装饰器把整个上报包在 `try/except` 里，注释直说 "Never let telemetry break things"——遥测失败只在 debug 级别记一行日志，绝不向上抛。`pipeline_running` 上报时只收集每个组件的类名和 `_get_telemetry_data()` 返回的结构化字段，不碰任何用户数据。这又是一次"非核心功能必须对核心链路无害"的工程表达。

## 本章小结

- Haystack 把**评估器也做成 `@component`**，让评估能拼进流水线、可序列化、与主链路共用数据契约，而不是框架外的一次性脚本。
- 评估分两条路线：纯算法指标（`DocumentRecallEvaluator` 的 `SINGLE_HIT`/`MULTI_HIT`、`DocumentMRREvaluator` 的 `1/(rank+1)`、`DocumentNDCGEvaluator` 的 `dcg/idcg`、`SASEvaluator` 的语义相似度）给出确定性分数；LLM 裁判（`LLMEvaluator` 及其子类）应对无标准答案的主观质量。
- `LLMEvaluator` 默认把裁判设为 JSON 模式并写死 `seed=42`，用结构化输出与固定采样换取评估结果的可消费性与可复现性。
- `FaithfulnessEvaluator` 用"拆陈述—逐条核验—算支撑比例"把抽象的"忠实"变成可操作判据，并用精心设计的正负 `_DEFAULT_EXAMPLES` 锚定标准。
- `EvaluationRunResult` 在构造时严格校验输入等长、个体分与样本数对齐，把最隐蔽的"行数错位"挡在入口，并提供 aggregated/detailed/comparative 三种报告与 json/csv/df 三种格式。
- 追踪系统用 `Tracer`/`Span` 双抽象 + `ProxyTracer` 代理模式，实现"不改全局引用即可启停追踪"，默认装的是零开销的 `NullTracer`。
- `set_content_tag` 受 `HAYSTACK_CONTENT_TRACING_ENABLED`（默认 `false`）控制，把查询、文档、答案这类敏感内容的记录设为隐私默认关闭、需显式授权。
- `pipeline.run` 把执行包进 `haystack.pipeline.run` span 与组件子 span，对内容用 `set_content_tag`，并在交给 tracer 前 `deepcopy` 输入以冻结真实快照。
- 遥测用随机 uuid、60 秒限流、环境变量退出、`try/except` 全包裹（"Never let telemetry break things"）实现"对核心链路完全无害"的非核心功能。

## 动手实验

1. **实验一：算法指标对比** — 用 `DocumentRecallEvaluator` 分别以 `SINGLE_HIT` 和 `MULTI_HIT` 两种 `RecallMode` 评同一组检索结果，观察个体分从"非 0 即 1"变为"命中比例"，再叠加 `DocumentMRREvaluator` 与 `DocumentNDCGEvaluator`，体会三个指标对"排序质量"的不同侧重。
2. **实验二：LLM 裁判与可复现性** — 用 `FaithfulnessEvaluator` 评一条故意答非所问的样本（如把 Python 作者答成 George Lucas），看它如何拆陈述并打 0 分；然后把 `LLMEvaluator` 的 `seed` 改掉重跑多次，对比有无固定 seed 时评分的稳定性差异。
3. **实验三：点亮可观测性** — 用 `enable_tracing(LoggingTracer(tags_color_strings=...))` 打开本地追踪跑一条 RAG 流水线，先在 `HAYSTACK_CONTENT_TRACING_ENABLED` 未设时观察日志里只有元数据、没有查询和文档内容，再把它设成 `true` 重跑，确认 `set_content_tag` 此时才输出敏感内容。
4. **实验四：评估报告导出** — 构造一个 `EvaluationRunResult` 并故意让某个指标的 `individual_scores` 长度与输入对不齐，验证它在构造阶段就抛错；改正后用 `comparative_detailed_report` 对比两次 run，分别导出 json/csv/df 三种格式，观察列名前缀与同名 run 的 warning 行为。

> **全书结语**：八部书走到这里，我们从 RAGFlow 的深度文档解析出发，穿过 AnythingLLM 的工作区隔离、LlamaIndex 的索引抽象、Quivr 的极简内核、Langchain-Chatchat 的本地化栈、LightRAG 的轻量图谱、GraphRAG 的社区摘要，最后停在 Haystack 的可组合编排。八套系统风格迥异，但拆到底层，它们用代码反复确认了同一组可迁移的工程原则——**默认值就是产品对用户的隐性承诺**（Haystack 的 BM25L、`seed=42`、追踪默认关闭，都是把"最该有的行为"设成"不操作就能得到的行为"）；**约束只收紧、越界即报错**（从 `DocumentStore` 的 `DuplicatePolicy` 到 `EvaluationRunResult` 的等长校验，都在边界处把错误拦在入口）；**统一的最小数据契约是可组合性的地基**（`Document`/`ChatMessage` 之于 Haystack，正如标准化的 chunk 之于其余七套系统）；**横切关注点要织进基座而非散落各处**（追踪、遥测、序列化都被收进 `@component` 与 `Pipeline` 的统一层）；**把编排从代码里抽成数据**（可序列化的 DAG 让流水线能存盘、能 diff、能在断点续跑）；**复用生态要快，但必须在边界保留一层契约**（Tracer 对 OpenTelemetry/Datadog 的代理、评估器对 sentence-transformers 的 LazyImport，都在拥抱第三方的同时守住自己的接口）；以及**给数据分层、让错误不致命**（权威源与可重算的派生层分开，遥测失败、导出失败、可选依赖缺失都被降级为日志而非崩溃）。这些原则不属于任何单一框架，它们是"把一个能跑的 demo 变成一个能被信任、被运维、被持续改进的系统"所必须付出的代价。读源码的终点从来不是记住某个类怎么写，而是看清这些反复出现的取舍——当你下一次设计自己的 RAG 系统时，愿这八部书拆开的不只是别人的代码，更是你做每一个默认值、每一条边界、每一层抽象时的判断力。
