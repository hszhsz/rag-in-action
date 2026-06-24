# 第 10 章 索引流水线编排 Index Pipeline

前面九章我们逐个拆开了索引侧的零件:`load_input_documents` 把原始文档读入标准表,`create_base_text_units` 切出文本单元,`extract_graph` / `extract_graph_nlp` 抽出实体与关系,`finalize_graph` 把图终化,`extract_covariates` 抽协变量,`create_communities` 跑 Leiden 社区检测,`create_community_reports` 生成社区报告,`generate_text_embeddings` 把各类文本灌进向量库。这些 workflow 各自独立、互不知道彼此的存在,只对着输入表读、对着输出表写。本章要回答的问题是:谁把这一串零件按正确的顺序拼成一条完整的流水线?谁负责逐个执行、收集耗时、捕获错误、把进度回调出去?

答案是 GraphRAG 的索引编排层——一个出奇简洁的设计。它没有引入重量级的 DAG 调度器,而是把"流水线"理解为一份「workflow 名字的有序列表」,把每个 workflow 统一成同一个函数签名,再用一个工厂按 `IndexingMethod` 取出对应的名字序列、逐个 `await`。这套"编排即数据"的思路,是本章想让你看懂的核心。

## 一、统一的 workflow 契约

整条流水线之所以能像搭积木一样自由拼装,前提是每个零件长得一模一样。这个契约定义在 `index/typing/workflow.py`。每个 workflow 都是一个异步函数,签名固定为 `Callable[[GraphRagConfig, PipelineRunContext], Awaitable[WorkflowFunctionOutput]]`,源码里用类型别名 `WorkflowFunction` 表示。也就是说,无论是读文档的 `load_input_documents` 还是清状态的 `update_clean_state`,它们对外都长这样:

```python
async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    ...
```

返回值 `WorkflowFunctionOutput` 是个只有两个字段的 `dataclass`:`result`(任意类型,文档注释明确说"只用于下游日志",真正的产物要写进 storage)和 `stop`(布尔,默认 `False`,置真时让流水线在本步之后停下)。注意这里的设计取舍:workflow 之间**不通过返回值传递数据**,而是各自把正式产物写进共享的存储。这正是为什么 `result` 的注释会强调"we use it only for logging downstream"——它只是个观测窗口,不是数据通道。

`load_input_documents.py` 是这个契约最干净的范例:它的 `run_workflow` 通过 `context.input_storage` 和 `config.input` 建出 `InputReader`,把文档逐行写进 `context.output_table_provider.open("documents")` 打开的表,顺手把行数记到 `context.stats.num_documents`,最后只返回前 5 行作为采样 `WorkflowFunctionOutput(result=sample)`。真正的数据落在了 storage,返回的只是一瞥。

`load_update_documents.py` 则展示了 `stop` 的用法:当它 diff 出"没有新文档"时,直接 `return WorkflowFunctionOutput(result=None, stop=True)`,让整条更新流水线优雅退出,不必白跑后面所有步骤。

## 二、PipelineRunContext:贯穿全程的上下文

既然 workflow 之间靠存储而非返回值通信,那"存储"从哪来?答案是第二个入参 `context`。`index/typing/context.py` 里的 `PipelineRunContext` 是一个 `dataclass`,把一次运行所需的全部公共设施聚成一束:

- `stats: PipelineRunStats`——运行统计,后面讲 profiling 时细说;
- `input_storage` / `output_storage: Storage`——输入读取与长期输出存储,注释写明"写到这里的东西会落到 storage provider";
- `output_table_provider: TableProvider`——读写输出表(parquet 等)的统一抽象;
- `previous_table_provider: TableProvider | None`——只在更新模式下有值,指向上一轮的产物,用于增量 diff;
- `cache: Cache`——读历史 LLM 响应的缓存;
- `callbacks: WorkflowCallbacks`——生命周期回调;
- `state: PipelineState`——一个"任意属性袋",注释直言用于运行时状态、持久化预计算或实验性特性。

这个对象由 `index/run/utils.py` 的 `create_run_context` 构造,它的贴心之处在于每个参数都有默认兜底:不传 storage 就用 `MemoryStorage()`,不传 table provider 就包一层 `ParquetTableProvider`,不传 cache 就用 `MemoryCache()`,不传 callbacks 就用 `NoopWorkflowCallbacks()`,不传 stats 就 `PipelineRunStats()`。这让单元测试可以零配置地拉起一个上下文跑单个 workflow,而生产路径再注入真实的 blob/cosmos provider。

## 三、PipelineFactory:把流水线当数据来注册

编排的核心数据结构在 `index/workflows/factory.py`。`PipelineFactory` 是一个全用 `@classmethod` 操作的类级注册表,持有两张表:

```python
class PipelineFactory:
    workflows: ClassVar[dict[str, WorkflowFunction]] = {}
    pipelines: ClassVar[dict[str, list[str]]] = {}
```

`workflows` 把 workflow 名映射到具体函数,`pipelines` 把"方法名"映射到「workflow 名的有序列表」。注意:流水线本身只是一个 `list[str]`——名字的列表,而非函数对象的列表。这就是"编排即数据"的字面体现:一条流水线就是一份可序列化、可比较、可重排的字符串清单。

注册分两边。函数侧由 `register` / `register_all` 完成,`index/workflows/__init__.py` 在导入时一次性调用 `PipelineFactory.register_all({...})`,把全部 22 个内置 workflow 名(`"load_input_documents"`、`"create_base_text_units"`、`"extract_graph"`……一直到各 `update_*`)登记进 `workflows` 表。流水线侧由 `register_pipeline` 完成,工厂文件末尾把四种方法的序列写死:

```python
_standard_workflows = [
    "create_base_text_units", "create_final_documents", "extract_graph",
    "finalize_graph", "extract_covariates", "create_communities",
    "create_final_text_units", "create_community_reports",
    "generate_text_embeddings",
]
PipelineFactory.register_pipeline(
    IndexingMethod.Standard, ["load_input_documents", *_standard_workflows]
)
```

`config/enums.py` 里的 `IndexingMethod` 枚举给出四个取值:`Standard`(`"standard"`,全程 LLM 构图)、`Fast`(`"fast"`,NLP 构图 + LLM 摘要)、`StandardUpdate`(`"standard-update"`)、`FastUpdate`(`"fast-update"`)。对照工厂里的注册可以看出方法之间的差异恰好被压缩成"列表怎么排":

- `standard` 与 `fast` 的分叉,只是把 `extract_graph` 换成 `extract_graph_nlp` 并多插一个 `prune_graph`,社区报告也从 `create_community_reports` 换成 `create_community_reports_text`;
- 两个 `*-update` 方法则是把开头的加载步骤换成 `load_update_documents`,把标准/快速序列整段复用,再在末尾追加一整段 `_update_workflows`。

取序逻辑在 `create_pipeline`:

```python
workflows = config.workflows or cls.pipelines.get(method, [])
return Pipeline([(name, cls.workflows[name]) for name in workflows])
```

这一行有两个细节值得品。第一,`config.workflows or ...`——如果用户在配置里显式给了 `workflows` 列表,就直接用它,完全绕过内置方法。这意味着高级用户可以在 YAML 里自定义任意 workflow 顺序,而无需改一行代码;`register` 也对外开放,第三方可以注册自己的 workflow 名再塞进列表。第二,直到这一刻名字才被换成函数对象——`(name, cls.workflows[name])` 组成 `Workflow` 二元组,再喂给 `Pipeline`。

## 四、Pipeline:一个生成器薄包装

`index/typing/pipeline.py` 里的 `Pipeline` 简单到几乎透明,它只包着一个 `list[Workflow]`(即 `tuple[str, WorkflowFunction]` 的列表):

- `run()` 是个生成器,`yield from self.workflows`,逐个吐出 `(name, function)`;
- `names()` 返回所有 workflow 名,供回调与日志使用;
- `remove(name)` 按名字过滤掉某个 workflow。

`remove` 不是摆设。回到 `run_pipeline.py`,当调用方直接传入了 `input_documents` 这个 DataFrame 时,代码会把文档直接写进存储,然后 `pipeline.remove("load_input_documents")`(更新模式则移除 `"load_update_documents"`)——既然文档已经就位,加载步骤就没必要再跑。这又一次印证了"流水线是可编辑的数据"这一立场:运行前临时增删一步,只是改一个列表。

## 五、run_pipeline:逐个 await 的执行循环

执行入口是 `index/run/run_pipeline.py` 的 `run_pipeline`,它是个返回 `AsyncIterable[PipelineRunResult]` 的异步生成器。它先做"装配":用 `config` 建出 input/output storage、`output_table_provider`、`cache`,再从 `output_storage.get("context.json")` 尝试加载上一轮持久化的 `state`(以支持有状态的 workflow),把可选的 `additional_context` 合并进去。

接着按是否更新分两路构造 `context`。标准路用 `output_storage` 作输出;更新路则要复杂些:它建一个带时间戳的 `update_storage` 子目录(`time.strftime("%Y%m%d-%H%M%S")`),在其下分出 `delta`(新子集索引)与 `previous`(旧索引备份)两个 table provider,先 `_copy_previous_output` 把当前输出整体备份到 `previous`,把 `update_timestamp` 写进 state,再用 `delta_storage` / `delta_table_provider` / `previous_table_provider` 拼出更新专用的 `context`。无论哪一路,最终都委托给内部的 `_run_pipeline` 并把它逐条 `yield` 出来。

真正的执行循环在 `_run_pipeline`,逻辑紧凑得可以整段读:

```python
for name, workflow_function in pipeline.run():
    last_workflow = name
    context.callbacks.workflow_start(name, None)
    with WorkflowProfiler() as profiler:
        result = await workflow_function(config, context)
    context.callbacks.workflow_end(name, result)
    yield PipelineRunResult(
        workflow=name, result=result.result, state=context.state, error=None
    )
    context.stats.workflows[name] = profiler.metrics
    await _dump_stats_json(context)
    if result.stop:
        logger.info("Halting pipeline at workflow request")
        break
```

可以拆成几个动作:对每个 workflow,先发 `workflow_start` 回调;在 `WorkflowProfiler` 上下文里 `await` 它,拿到 `WorkflowFunctionOutput`;发 `workflow_end`;把这一步包成 `PipelineRunResult` 流式 `yield` 出去(调用方可以边跑边收结果,而不必等整条流水线结束);把本步的 profiler 指标记进 `context.stats.workflows[name]` 并落盘 `stats.json`;最后检查 `result.stop`,为真则 `break`。循环外,把 `total_runtime` 算出来,再做一次 stats 与 context 落盘。

`PipelineRunResult`(`index/typing/pipeline_run_result.py`)就是流水线对外暴露的"一步结果":`workflow`(名字)、`result`(那一瞥日志值)、`state`(当前累积的运行状态)、`error`(异常或 `None`)。

## 六、错误处理与可观测性

整段循环被一个 `try/except` 包住,这就是流水线的错误边界。`last_workflow` 在每步开头被刷新,一旦任何 workflow 抛异常,`except` 块会 `logger.exception` 记录"是哪个 workflow 挂了",然后 `yield` 一个 `error` 字段非空的 `PipelineRunResult`——注意它是 yield 而非 raise,所以异常被转成了数据,顺着同一个异步流交给上层判断。上层 `api/index.py` 的 `build_index` 收到后,会检查 `output.error is not None`,触发 `pipeline_error` 回调并记日志,但不会让整个进程崩溃。这种"错误即结果"的处理方式,让长跑的索引任务即便某步失败也能干净地报告并收尾。

可观测性由三条线支撑。其一是 **callbacks**:`callbacks/workflow_callbacks.py` 定义的 `WorkflowCallbacks` 是个 `Protocol`,声明了 `pipeline_start` / `pipeline_end` / `workflow_start` / `workflow_end` / `progress` / `pipeline_error` 六个钩子;默认实现是 noop,使用方只需实现关心的几个(比如 CLI 进度条监听 `progress`)。`utils.py` 的 `create_callback_chain` 还能把多个回调用 `WorkflowCallbacksManager` 串成一条链,广播给所有监听者。

其二是 **profiling**:`run/profiling.py` 的 `WorkflowProfiler` 是个上下文管理器,`__enter__` 里 `tracemalloc.start()` 并记起始时间,`__exit__` 里算出 `overall`(墙钟耗时)、`peak_memory_bytes`(峰值内存)、`memory_delta_bytes`(净增内存)与 `tracemalloc_overhead_bytes`,封进 `WorkflowMetrics`。每个 workflow 都被它包裹,因此 `stats.json` 里能看到逐步的耗时与内存画像。

其三是 **stats 落盘**:`PipelineRunStats`(`index/typing/stats.py`)汇总 `total_runtime`、`num_documents`、`update_documents`、`input_load_time` 以及 `workflows: dict[str, WorkflowMetrics]`。`_dump_stats_json` 用 `asdict` 把它序列化进 `stats.json`,`_dump_context_json` 则把 `state` 写进 `context.json`(写前临时摘掉体积不可控的 `additional_context`)。这两份文件既是事后分析的依据,也是更新模式下"加载上一轮 state"的来源。

## 七、增量更新流水线

更新模式不是另起炉灶,而是把同一套 workflow 复用到极致。前面已看到 `_update_workflows` 这一段:`update_final_documents`、`update_entities_relationships`、`update_text_units`、`update_covariates`、`update_communities`、`update_community_reports`、`update_text_embeddings`、`update_clean_state`。`standard-update` 的完整序列正是 `["load_update_documents", *_standard_workflows, *_update_workflows]`——先用 `load_update_documents` 只加载新增/变更文档,再把标准索引整段对增量子集跑一遍(产物落在 `delta`),最后用 `update_*` 这组步骤把 `delta` 的结果与 `previous` 的旧索引合并。

这条链路里有几处呼应前文的设计。`load_update_documents` 依赖 `context.previous_table_provider`(没有就直接报错),它调 `get_delta_docs` 做 diff,只有 `new_inputs` 才往下传;若无新文档便 `stop=True` 提前收场。收尾的 `update_clean_state` 则把 `context.state` 中所有以 `"incremental_update_"` 开头的临时键删掉,避免状态污染下一轮。这套"复用主链 + 追加合并步骤"的做法,意味着任何对标准索引 workflow 的改进都会自动惠及增量更新,无需在两处重复维护。

方法名的拼装发生在 API 层。`build_index` 收到 `method` 与 `is_update_run` 后,经 `_get_method` 把它们合成最终字符串:`f"{m}-update" if is_update_run else m`,于是 `standard` + 更新标志就变成 `"standard-update"`,正好命中工厂注册的键。CLI 侧 `cli/index.py` 的 `index_cli`(`is_update_run=False`)与 `update_cli`(`is_update_run=True`)只是这套参数的两个入口。

## 八、设计动机回顾

把视角拉远,这一层的精巧在于它的"少"。它没有发明 DAG、没有引入条件分支引擎,而是基于一个观察:索引就是一串固定顺序的步骤。于是它把流水线降维成数据(`list[str]`),把执行降维成一个 `for` 循环加 `await`,把零件之间的耦合从"互相调用"降维成"共用一个 context、各写各的存储"。这三个降维带来的红利是显著的:新增一个 workflow 只需写一个符合契约的 `run_workflow` 并 `register` 进去,再把名字插进某条 pipeline 列表,零侵入;调整索引顺序只需改列表;高级用户用 `config.workflows` 就能完全自定义;而 profiling、callbacks、错误兜底这些横切关注点全部集中在 `_run_pipeline` 一处,所有 workflow 自动获得统一的可观测性,无需各自重复埋点。这是一套用约定换灵活、用简单换可扩展的编排哲学。

## 本章小结

- 每个 workflow 都遵守统一契约:`async def run_workflow(config, context) -> WorkflowFunctionOutput`,类型别名为 `WorkflowFunction`;`WorkflowFunctionOutput` 只有 `result` 与 `stop` 两字段,数据靠存储传递、返回值仅供日志。
- `PipelineRunContext` 把 stats、input/output storage、table provider、previous provider、cache、callbacks、state 聚成一束,由 `create_run_context` 构造并对每个参数提供内存兜底实现。
- `PipelineFactory` 持有 `workflows`(名→函数)与 `pipelines`(方法→名字列表)两张类级注册表,`register_all` 在包导入时登记 22 个内置 workflow,`register_pipeline` 写死四种方法的序列。
- 流水线本身是 `list[str]`——"编排即数据";`create_pipeline` 优先用 `config.workflows`,否则按 `IndexingMethod` 取序,最后才把名字换成函数对象组成 `Pipeline`。
- `Pipeline` 是生成器薄包装,`run()` 逐个 `yield` `(name, function)`,`remove()` 支持运行前临时删步(如已直接传入文档时移除 `load_input_documents`)。
- `_run_pipeline` 用一个 `for` 循环逐个 `await` workflow,流式 `yield` `PipelineRunResult`,逐步发 `workflow_start`/`workflow_end` 回调、记 profiler 指标、落盘 `stats.json`,遇 `result.stop` 即 break。
- 错误被 `try/except` 转成 `error` 字段非空的 `PipelineRunResult` 顺流交出,而非直接抛出,`build_index` 据此报告但不崩溃。
- 可观测性由三线支撑:`WorkflowCallbacks` 协议(六个钩子 + 可串成链)、`WorkflowProfiler`(tracemalloc 记耗时与内存)、`PipelineRunStats` 落盘 `stats.json` 与 `context.json`。
- 增量更新复用主链:`load_update_documents` + 标准/快速序列(产物入 `delta`)+ `_update_workflows` 合并旧索引(`previous`),`_get_method` 把 `standard` 与更新标志拼成 `"standard-update"` 命中工厂。
- 整层设计的核心动机:把固定步骤抽象成可注册、可重排、可观测的列表式流水线,横切关注点集中一处,新增 workflow 与自定义顺序均零成本。

## 动手实验

1. **实验一:把流水线打印出来** — 在 Python 里 `from graphrag.index.workflows.factory import PipelineFactory`,先 `import graphrag.index.workflows` 触发注册,然后打印 `PipelineFactory.pipelines["standard"]` 与 `["fast"]`,逐项对比两者差异(`extract_graph` vs `extract_graph_nlp`、是否有 `prune_graph`、`create_community_reports` vs `..._text`),验证"方法差异即列表差异"。
2. **实验二:注册一个自定义 workflow** — 写一个符合契约的 `async def run_workflow(config, context)`,里面只 `context.callbacks` 打个日志并返回 `WorkflowFunctionOutput(result="hello")`,用 `PipelineFactory.register("my_probe", run_workflow)` 登记,再 `register_pipeline("probe", ["load_input_documents", "my_probe"])`,用 `create_pipeline(config, "probe")` 跑通,确认零侵入接入。
3. **实验三:读懂 stats.json** — 对一个小语料跑一次 `standard` 索引,打开输出目录的 `stats.json`,找到 `workflows` 下每个 workflow 的 `overall` 与 `peak_memory_bytes`,排序出最耗时与最吃内存的步骤,并对照 `WorkflowProfiler.__exit__` 解释这些数字是如何采集的。
4. **实验四:触发 stop 提前退出** — 在一个已索引完成的目录上跑更新模式(`is_update_run=True`)且不新增任何文档,观察 `load_update_documents` 因 `len(output) == 0` 返回 `stop=True`,确认 `_run_pipeline` 在该步后 `break`、后续 `update_*` 步骤未执行;再阅读 `update_clean_state` 理解 `incremental_update_` 前缀键的清理时机。

> **下一章预告**:索引侧到此收官——我们已经有了实体、关系、社区、报告与各类向量。从下一章起进入查询侧,《Local Search 局部检索》将拆解 `local_search` 与 `mixed_context`:GraphRAG 如何从用户问题出发,在实体图上做局部扩散,把文本单元、关系、社区报告与协变量混合成上下文,喂给 LLM 给出有出处的回答。
