# 第 1 章 项目全景与 monorepo 架构

普通的向量 RAG 做的是一件很朴素的事:把文档切片、嵌入成向量、按相似度召回、塞进提示词。它擅长回答"文档里某一段说了什么",但面对"把这整批文档里反复出现的几股势力梳理出来"这类需要**跨文档全局归纳**的问题就力不从心——因为相似度检索只会捞回与问题字面最接近的几个片段,看不到全貌。GraphRAG 的根本不同在于:它在索引阶段就用 LLM 把非结构化文本**转写成一张知识图谱**(实体、关系、声明),再用社区检测算法把图谱切成层级化的社区,并为每个社区预先生成"社区报告"。查询时既可以像普通 RAG 那样做局部检索,也可以把一批社区报告做 map-reduce 式的全局归纳。GraphRAG 官方把自己定位为"a data pipeline and transformation suite",强调它本质是一条把非结构化文本抽取成结构化数据的数据管线,这一点在 `README.md` 的 Overview 里写得很清楚。

本章是整部 GraphRAG 解构的地图。我们先弄清这个项目"长什么样":它是一个由 8 个包组成的 uv workspace monorepo,版本停在 `3.1.0`;再看它对外暴露的两层入口——给人用的 CLI(`graphrag init/index/query/update/prompt-tune`)和给程序调用的 api 层(`build_index` 与四个 `*_search` 函数);最后把从"一篇文档"到"一个回答"之间的完整数据流弧线画出来,后续每一章都是在这条弧线上的某一段做放大。读完本章,你应该能在脑子里建立起整个仓库的目录索引,知道任何一个功能大概落在哪个包、哪个目录。

## GraphRAG 是什么:从文本到图谱再到社区

要理解后面所有章节,先记住 GraphRAG 索引产物的形态。它不是一个单一的向量索引,而是一组结构化数据表,在源码里以 `data_model/` 下的一组数据类定义:`entity.py`(实体)、`relationship.py`(关系)、`covariate.py`(协变量/声明)、`community.py`(社区)、`community_report.py`(社区报告)、`text_unit.py`(文本单元)、`document.py`(文档)。这些类大多继承自 `data_model/named.py` 的 `Named` 和 `data_model/identified.py` 的 `Identified`,共享 id/short_id/title 这类基础字段。

这套数据模型直接对应 GraphRAG 与普通向量 RAG 的差异:普通 RAG 只有"文本块 + 向量",而 GraphRAG 在文本块之上又叠了一层"实体—关系—社区"的图结构。社区(`community.py`)是用 Leiden 算法对实体图做层级聚类得到的,每一层社区对应不同的抽象粒度——这也是为什么 CLI 的 `query` 命令有一个 `--community-level` 选项,其帮助文字明确写着"Leiden hierarchy level from which to load community reports. Higher values represent smaller communities"(`cli/main.py`)。Leiden 是一种在 Louvain 基础上改进、保证社区内部连通性的社区检测算法 [[From Louvain to Leiden: guaranteeing well-connected communities]](https://www.nature.com/articles/s41598-019-41695-z)。

`data_model/` 里还有几个值得一提的辅助文件,它们透露了这套模型是怎么在"内存对象"与"磁盘 parquet 表"之间往返的:`schemas.py` 集中定义各张表的列名约定,`data_reader.py` 与 `row_transformers.py` 负责把 DataFrame 的行转换成数据类实例,`types.py`、`dfs.py` 提供类型与 DataFrame 工具。换句话说,GraphRAG 全程以 pandas DataFrame 作为索引各阶段之间的"通用数据总线"——这一点和主包 `dependencies` 里把 `pandas`、`pyarrow`、`networkx` 都列为硬依赖是吻合的:pandas/pyarrow 负责表与 parquet,networkx 负责内存里的图结构。理解了这条"DataFrame + parquet"的主线,后面读任何一个索引工作流时都能预期它的输入输出长什么样。

## 8 包 monorepo:一个 uv workspace 的拆分逻辑

仓库根目录的 `pyproject.toml` 里,`[project].name` 是 `graphrag-monorepo`,version 是 `0.0.0`,并标了 `classifiers = ["Private :: Do Not Upload"]`——这个根包本身不发布,它只是个工作区容器。真正的拆分声明在 `[tool.uv.workspace]`:

```toml
[tool.uv.workspace]
members = ["packages/*"]
```

也就是说 `packages/` 下每个目录都是工作区成员。配合 `[tool.uv.sources]` 把 `graphrag-chunking`、`graphrag-common`、`graphrag-input`、`graphrag-storage`、`graphrag-cache`、`graphrag-vectors`、`graphrag-llm` 全部标记为 `{ workspace = true }`,这意味着主包依赖这些子包时,uv 会直接用本地工作区里的源码,而不是去 PyPI 拉取。`[tool.uv]` 里还有 `package = false`,表明根目录不作为一个可安装包来构建。

`packages/` 下一共是 1 个主包 + 7 个功能子包,每个子包的 `pyproject.toml` 里 description 把职责说得很直白:

- `graphrag`(主包):整个系统的门面与编排层。它聚合所有子包,放置 CLI、api、index 管线、query 引擎、config、data_model、prompts、prompt_tune 等。它的 `pyproject.toml` 里 `dependencies` 把七个子包以 `==3.1.0` 精确钉死,并声明了 `[project.scripts]` 的命令入口 `graphrag = "graphrag.cli.main:app"`。
- `graphrag-common`:description 为 "Common utilities and types for GraphRAG",是最底层的公共工具与类型,被其它包共享。
- `graphrag-input`:description 为 "Input document loading utilities for GraphRAG",负责把原始文档(txt/csv/json 等)读进来,是数据流弧线的起点。
- `graphrag-chunking`:description 为 "Chunking utilities for GraphRAG",文本切分工具,把文档切成 text unit。
- `graphrag-storage`:description 为 "GraphRAG storage package",存储后端抽象(本地文件、Azure Blob、CosmosDB 等);3.1.0 的更新点正是"Native CosmosTableProvider",所以这个包目录里还附了一份 `COSMOS_TABLE_PROVIDER_DESIGN.md`。
- `graphrag-cache`:description 为 "GraphRAG cache package",LLM 调用缓存,避免重复索引时反复花钱调用模型。它对外暴露的 `CacheType` 枚举被 `cli/index.py` 用来在 `--no-cache` 时把缓存切成 `CacheType.Noop`。
- `graphrag-vectors`:description 为 "GraphRAG vector store package",向量库抽象,存放实体描述、文本单元、社区报告的嵌入。
- `graphrag-llm`:description 为 "GraphRAG LLM package",LLM 与嵌入模型的接入层。

把切分、输入、存储、缓存、向量、LLM 各自独立成包,带来的工程收益很直接:这些都是有多种后端实现、容易单独演进、且可能被复用的能力。例如向量库可以换 LanceDB/Azure AI Search,存储可以换本地/Blob/Cosmos,缓存可以开关——把它们从主包里剥离,主包的 index 与 query 逻辑就只依赖抽象接口,不被具体后端绑死。这是后续若干章(输入、切分、存储、向量、LLM)各自成章的物理基础。

七个子包之间也存在清晰的依赖层次。根 `pyproject.toml` 的 `[tool.pyright].include` 把类型检查范围列为 `graphrag`、`graphrag-common`、`graphrag-storage`、`graphrag-cache`、`graphrag-llm` 五处(加 `tests`),侧面反映了哪些包是核心受检对象。从职责上看,`graphrag-common` 处在最底层,被几乎所有其它包共享;`graphrag-storage` 与 `graphrag-cache` 提供持久化与缓存的抽象,被索引管线频繁使用;`graphrag-input`、`graphrag-chunking` 服务于索引的前段;`graphrag-llm`、`graphrag-vectors` 同时服务于索引(嵌入)与查询(检索)。主包 `graphrag` 站在最顶端,把这些能力编排成一条完整管线。这种"底层公共 → 中层能力 → 顶层编排"的分层,使得任何一个能力包都可以在不触碰主包逻辑的前提下被替换或扩展。

## 版本 3.1.0 与打包方式

`CHANGELOG.md` 顶部第一条就是 `## 3.1.0`,说明当前我们解构的就是这个版本;它的两条变更分别是 CosmosTableProvider 的原生实现和 litellm 依赖更新。主包 `packages/graphrag/pyproject.toml` 里 `version = "3.1.0"`,并且注释提醒"Maintainers: do not change the version here manually"——版本号是由 `semversioner` 工具自动写入的。

根 `pyproject.toml` 的 `[tool.poe.tasks.release]` 揭示了版本号是如何在 8 个包之间保持同步的:release 任务序列里有一长串 `_semversioner_update_*_toml_version` 任务,逐个把每个子包 `pyproject.toml` 的 `project.version` 更新成 `semversioner current-version` 的输出。这就是为什么所有子包 version 都精确等于 `3.1.0`。

构建方面,主包用 hatchling 作为 build backend(`[build-system].requires = ["hatchling..."]`)。要求的 Python 版本是 `>=3.11,<3.14`。日常开发任务也都收敛在 `[tool.poe.tasks]` 里,比如 `index = "python -m graphrag index"`、`query = "python -m graphrag query"`、`init`、`update`、`prompt_tune`,以及分层的测试任务 `test_unit`/`test_integration`/`test_smoke`/`test_verbs`/`test_notebook`——这套 poe 任务表也是阅读源码时的索引。

## CLI 命令树:`cli/main.py` 如何用 typer 组织五个子命令

用户面的入口是 `packages/graphrag/graphrag/cli/main.py`。它用 `typer` 构建命令树:

```python
app = typer.Typer(
    help="GraphRAG: A graph-based retrieval-augmented generation (RAG) system.",
    no_args_is_help=True,
)
```

`no_args_is_help=True` 意味着不带参数直接敲 `graphrag` 会打印帮助。typer 是基于 Click 构建的现代 CLI 框架,用类型注解声明参数 [[Typer documentation]](https://typer.tiangolo.com/)。文件里用 `@app.command(...)` 注册了五个子命令,与 `pyproject.toml` 的 poe 任务一一对应:

- `@app.command("init")` → `_initialize_cli`:生成默认配置文件,内部 `from graphrag.cli.initialize import initialize_project_at`。
- `@app.command("index")` → `_index_cli`:构建知识图谱索引,委托给 `cli/index.py` 的 `index_cli`。它带 `--method`(类型是 `IndexingMethod` 枚举)、`--verbose`、`--dry-run`、`--cache/--no-cache`、`--skip-validation`。
- `@app.command("update")` → `_update_cli`:增量更新已有索引,委托给 `cli/index.py` 的 `update_cli`,把结果写到 `update_output` 目录。
- `@app.command("prompt-tune")` → `_prompt_tune_cli`:用你自己的数据自动生成定制提示词(auto templating),委托给 `cli/prompt_tune.py` 的 `prompt_tune`,这是唯一一个直接用 `asyncio.get_event_loop().run_until_complete(...)` 跑异步协程的命令。
- `@app.command("query")` → `_query_cli`:对索引发起查询。

值得注意的一个工程细节:每个命令的真正实现都是在函数体内才 `import`(如 `from graphrag.cli.index import index_cli`),而不是在文件顶部。这是典型的延迟导入,目的是让 `graphrag --help` 这类轻量调用不必加载整条索引/查询管线的重型依赖,启动更快。

`query` 命令的核心是一个 `match method:` 分发,把四种 `SearchMethod` 路由到 `cli/query.py` 的四个函数:`SearchMethod.LOCAL` → `run_local_search`,`SearchMethod.GLOBAL` → `run_global_search`,`SearchMethod.DRIFT` → `run_drift_search`,`SearchMethod.BASIC` → `run_basic_search`,落不到任一分支就 `raise ValueError(INVALID_METHOD_ERROR)`。`--method` 的默认值是 `SearchMethod.GLOBAL.value`,所以不指定方法时默认走全局搜索。`init` 命令的 `--model` 与 `--embedding` 默认值取自 `config/defaults.py` 的 `DEFAULT_COMPLETION_MODEL` 与 `DEFAULT_EMBEDDING_MODEL`,并且这两个选项带 `prompt=...`——初始化时如果没传会交互式询问。

## api 层:作为库被调用的两组入口

CLI 之下还有一层更纯粹的编程接口,放在 `packages/graphrag/graphrag/api/`。`api/__init__.py` 用 `__all__` 明确导出了三组能力:索引(`build_index`)、查询(八个函数:`global_search`/`local_search`/`drift_search`/`basic_search` 各自加一个 `*_streaming` 流式版本)、以及提示词调优(`generate_indexing_prompts`、`DocSelectionType`)。这个 `__init__` 的 docstring 也诚实地标注了"WARNING: This API is under development...Backwards compatibility is not guaranteed",提示它仍在演进。

`api/index.py` 的 `build_index` 是整个索引的程序化入口。它的签名是 `async def build_index(config, method=IndexingMethod.Standard, is_update_run=False, callbacks=None, additional_context=None, verbose=False, input_documents=None)`。它的执行骨架很清晰:先 `init_loggers` 初始化日志;把 callbacks 包成 `create_callback_chain` 或退化为 `NoopWorkflowCallbacks`;通过 `_get_method` 把 method 与 `is_update_run` 合成最终的管线名(更新时拼成 `f"{m}-update"`);调用 `PipelineFactory.create_pipeline(config, method)` 构造管线;然后 `async for output in run_pipeline(...)` 逐个工作流地驱动整条流水线,把每个 `PipelineRunResult` 收进 `outputs` 并通过 callbacks 上报 pipeline_start/error/end。注意 `input_documents` 参数允许调用方绕过文件加载、直接传入一个 `pd.DataFrame` 当输入文档。

CLI 与 api 的衔接在 `cli/index.py` 看得很直观:`index_cli` 先 `load_config(root_dir=root_dir)` 把磁盘上的 `settings.yaml` 读成 `GraphRagConfig`,再调 `_run_index`;后者做完日志初始化、缓存开关(`config.cache.type = CacheType.Noop`)、`validate_config_names` 预检、dry-run 早退之后,用 `asyncio.run(api.build_index(...))` 真正跑索引,并依据是否有 error 决定 `sys.exit` 的码。换句话说,**CLI 只是个壳,真正的逻辑全在 api 与其下游**——这正是为什么把 GraphRAG 当库嵌进自己的服务里也很自然:直接调 `api.build_index` 和 `api.global_search` 即可。

## 管线工厂:四种索引方法是怎么被"组装"出来的

`build_index` 调的 `PipelineFactory.create_pipeline` 来自 `index/workflows/factory.py`,这是理解索引数据流的关键。`PipelineFactory` 用两个类级字典持有注册表:`workflows`(工作流名 → 函数)和 `pipelines`(管线名 → 工作流名列表)。`create_pipeline` 的逻辑是:优先用 `config.workflows`(用户在配置里显式指定的工作流列表),否则按 method 从 `pipelines` 里取预设列表,再按名字逐个查出函数,组装成一个 `Pipeline`。

文件底部静态注册了四套预设管线,精确对应 `IndexingMethod` 枚举的四个取值。标准管线 `IndexingMethod.Standard` 的工作流序列是:

```python
_standard_workflows = [
    "create_base_text_units",
    "create_final_documents",
    "extract_graph",
    "finalize_graph",
    "extract_covariates",
    "create_communities",
    "create_final_text_units",
    "create_community_reports",
    "generate_text_embeddings",
]
```

再在前面拼上 `"load_input_documents"`。这串名字几乎就是后面索引部分各章的目录:切分(`create_base_text_units`)→ 图谱抽取(`extract_graph`)→ 图谱终化(`finalize_graph`)→ 协变量/声明抽取(`extract_covariates`)→ 社区检测(`create_communities`)→ 社区报告(`create_community_reports`)→ 文本嵌入(`generate_text_embeddings`)。`IndexingMethod.Fast` 是它的"省钱版":用 `extract_graph_nlp`(NLP 抽取实体而非全程 LLM)加 `prune_graph`,社区报告也换成 `create_community_reports_text`,对应 `IndexingMethod` 枚举注释里说的"using NLP for graph construction and language model for summarization"。两个 update 变体则在标准/快速序列之后追加一串 `update_*` 工作流并以 `load_update_documents` 开头,实现增量索引。

## 整体数据流弧线:从一篇文档到一个回答

把上面的零件串起来,GraphRAG 的端到端弧线是这样的。

**索引侧(写)**:`graphrag index` → `load_config` 得到 `GraphRagConfig` → `api.build_index` → `PipelineFactory` 组装工作流序列 → `run_pipeline` 逐个执行。数据形态依次变化:原始文档被 `load_input_documents`(由 `graphrag-input` 支撑)读入 → `create_base_text_units`(由 `graphrag-chunking` 支撑)切成文本单元 → `extract_graph` 用 LLM 从每个文本单元里抽出实体与关系,`finalize_graph` 合并去重成完整图 → `extract_covariates` 可选地抽取声明 → `create_communities` 用 Leiden 把实体图聚成层级社区 → `create_community_reports` 为每个社区让 LLM 写一份摘要报告 → `generate_text_embeddings`(由 `graphrag-llm` 与 `graphrag-vectors` 支撑)为实体描述、文本单元、社区报告生成向量。最终落地为一组 parquet 表(entities、relationships、communities、community_reports、text_units、documents)加向量库,中间结果可被 `graphrag-cache` 缓存。

**查询侧(读)**:`graphrag query --method ...` → `cli/query.py` 的 `run_*_search` → `api/query.py` 的对应函数。`api/query.py` 顶部的 import 已经透露了查询要读哪些产物:它从 `query/indexer_adapters.py` 引入 `read_indexer_entities`、`read_indexer_relationships`、`read_indexer_reports`、`read_indexer_communities`、`read_indexer_text_units`、`read_indexer_covariates`、`read_indexer_report_embeddings`——也就是把索引侧落地的那些 parquet 表读回内存,再交给 `query/factory.py` 的 `get_global_search_engine` 等构造对应的搜索引擎。四种检索模式各有侧重:

- **local**:从问题相关的实体出发,沿图召回邻居实体、关系、文本单元、社区报告,适合"关于某个具体实体"的问题。
- **global**:把某一层(`--community-level`)的社区报告做 map-reduce 归纳,适合需要全局视角的总结性问题;还有 `--dynamic-community-selection` 动态选社区的变体。
- **drift**:结合全局与局部,先用社区报告定位再向下钻取(对应 `prompts/query/drift_search_system_prompt.py` 里的 local/reduce 两段提示词)。
- **basic**:退化成普通向量 RAG,直接对文本单元做相似度检索(对应 `BASIC_SEARCH_SYSTEM_PROMPT`),作为不需要图能力时的兜底。

支撑这条弧线的还有几个横切组件:`config/`(下一章主角 `GraphRagConfig`,由 `load_config.py` 加载、`defaults.py` 提供默认值)、`prompts/`(分 `index/` 与 `query/` 两组提示词,`cli/initialize.py` 在 init 时会把它们写到项目的 `prompts/` 目录供用户改写)、`callbacks/`(工作流与查询的生命周期回调,如 `ConsoleWorkflowCallbacks`、`NoopWorkflowCallbacks`)、`logger/`(`standard_logging.py` 的 `init_loggers`,被 index 与 query 共用)。这些就是后续各章会逐一展开的"骨架"。

## 本章小结

- GraphRAG 的根本差异:索引阶段用 LLM 把文本转写成知识图谱,用 Leiden 算法做层级社区检测并预生成社区报告,从而支持普通向量 RAG 做不到的全局归纳类问答;它自我定位为一条"data pipeline and transformation suite"。
- 仓库是 uv workspace monorepo:根 `pyproject.toml` 的 `[tool.uv.workspace].members = ["packages/*"]`,根包 `graphrag-monorepo` 本身不发布(`Private :: Do Not Upload`、`package = false`)。
- 8 个包:主包 `graphrag` 做门面与编排,7 个功能子包 `graphrag-common`/`graphrag-input`/`graphrag-chunking`/`graphrag-storage`/`graphrag-cache`/`graphrag-vectors`/`graphrag-llm` 各管一项可替换、可复用的能力,通过 `[tool.uv.sources]` 的 `workspace = true` 本地引用。
- 版本统一为 `3.1.0`,由 `semversioner` 通过 `[tool.poe.tasks.release]` 里的一串 `update-toml` 任务同步到所有子包;主包用 hatchling 构建,要求 Python `>=3.11,<3.14`。
- CLI 在 `cli/main.py` 用 `typer` 注册 `init/index/update/prompt-tune/query` 五个子命令,均采用延迟 import,真正逻辑下沉到 `cli/*.py` 再到 api 层。
- api 层在 `api/__init__.py` 导出 `build_index` 与四组 `*_search`(各含 streaming 版本);`build_index` 通过 `PipelineFactory.create_pipeline` + `run_pipeline` 驱动工作流序列。
- `index/workflows/factory.py` 静态注册了 Standard/Fast/StandardUpdate/FastUpdate 四套管线,工作流名列表几乎就是索引部分的章节目录。
- 端到端弧线:文档→切分→图谱抽取→图谱终化→协变量→社区检测→社区报告→嵌入(写);查询时由 `indexer_adapters` 读回 parquet 产物,经 `query/factory.py` 分发到 local/global/drift/basic 四种检索(读)。

## 动手实验

1. **实验一:画出真实的工作流依赖图** — 打开 `packages/graphrag/graphrag/index/workflows/factory.py`,把 `_standard_workflows`、`_fast_workflows`、`_update_workflows` 三个列表抄下来,逐一比对它们的差异(Fast 用 `extract_graph_nlp`+`prune_graph` 替代 `extract_graph`,用 `create_community_reports_text` 替代 `create_community_reports`)。画一张表标出每个工作流名后续会在哪一章被解构。

2. **实验二:验证 CLI 到 api 的调用链** — 从 `cli/main.py` 的 `@app.command("index")` 出发,沿 `index_cli`(`cli/index.py`)→ `_run_index` → `asyncio.run(api.build_index(...))`(`api/index.py`)逐跳跟读,把每一跳传递的参数(`root_dir`、`method`、`cache` 等)记下来,确认 `--no-cache` 是如何变成 `config.cache.type = CacheType.Noop` 的。

3. **实验三:确认 8 个包与版本同步机制** — 用 `ls packages/` 列出全部成员,逐个打开各子包 `pyproject.toml` 的 `description` 与 `version`,确认都为 `3.1.0`;再读根 `pyproject.toml` 的 `[tool.poe.tasks.release]` 序列,解释 `_semversioner_update_*_toml_version` 是怎样保证 8 个包版本一致的。

4. **实验四:为查询的四种模式建立索引** — 打开 `cli/main.py` 的 `_query_cli`,记录 `match method:` 里四个 `SearchMethod` 分别路由到 `cli/query.py` 的哪个 `run_*_search`;再对照 `api/query.py` 顶部从 `query/indexer_adapters.py` 导入的 `read_indexer_*` 函数,列出查询阶段需要从磁盘读回哪些 parquet 产物,并标注每个产物由索引侧哪个工作流生成。

> **下一章预告**:索引和查询的每一步都从 `GraphRagConfig` 取参数。第 2 章将解构配置系统——`config/load_config.py` 如何把 `settings.yaml` 与 `.env` 叠加加载、`config/defaults.py` 如何提供默认值、`config/models/` 下的 pydantic 模型如何把一份 YAML 校验成强类型的 `GraphRagConfig`。
