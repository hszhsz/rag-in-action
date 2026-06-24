# 第 2 章 配置系统 GraphRagConfig

上一章我们俯瞰了 GraphRAG 的 monorepo 结构,看到核心包 `graphrag` 依赖一组 `graphrag_*` 子包(`graphrag_llm`、`graphrag_storage`、`graphrag_vectors`、`graphrag_input` 等)。这一章我们钻进 `graphrag/config/` 目录,解构整个系统的"控制面板":一个用户在 `settings.yaml` 里写下、最终被解析成 `GraphRagConfig` pydantic 对象、再被索引流水线与查询引擎层层读取的分层配置树。

GraphRAG 是一条把"原始文档"变成"实体图 + 社区报告 + 向量索引"的长流水线,每一步(切分、抽图、摘要、聚类、生成报告、四种检索)都有一堆可调旋钮。配置系统的设计目标因此非常明确:用 pydantic 把这些旋钮组织成有类型、有默认值、可嵌套的模型,让用户只需在 yaml 里覆盖少数关键项(主要是模型与 API key),其余全部由代码里成百的 `DEFAULT_*` 常量兜底。下面从模型分层、默认值、加载流程、枚举、模型配置、`graphrag init` 模板五个角度逐一拆开。

## 为什么用 pydantic 分层配置模型

配置的顶层是 `GraphRagConfig`,定义在 `graphrag/config/models/graph_rag_config.py`,它继承自 pydantic 的 `BaseModel`(`class GraphRagConfig(BaseModel)`)。选择 pydantic 而不是裸 dict 或 dataclass,带来三个直接收益:

第一是**类型校验**。每个字段都用 `Field(...)` 声明类型与默认值,例如 `concurrent_requests: int`、`async_mode: AsyncType`。当用户在 yaml 里把整数写成字符串、或拼错枚举值时,pydantic 在实例化阶段就抛 `ValidationError`,而不是等到流水线跑到一半才崩。pydantic 的核心定位正是"数据校验与设置管理"[[Pydantic]](https://docs.pydantic.dev/latest/)。

第二是**嵌套组合**。`GraphRagConfig` 的字段本身又是别的 `BaseModel`:`chunking: ChunkingConfig`、`extract_graph: ExtractGraphConfig`、`community_reports: CommunityReportsConfig`、`local_search: LocalSearchConfig` 等等。yaml 里的嵌套字典会被自动递归构造成对应的子模型对象,形成一棵配置树。

第三是**声明式默认值**。子模型里几乎每个 `Field` 的 `default=` 都指向一个集中常量,而不是就地写魔数,这把"默认值"从"模型结构"中解耦了出来(下一节详述)。

值得注意的是,GraphRAG 把"配置模型"和"默认值"用了**两套不同的机制**:模型用 pydantic `BaseModel`,而默认值容器用标准库 `@dataclass`(见 `defaults.py` 顶部 `from dataclasses import dataclass, field`)。dataclass 轻量、无校验开销,正适合做纯数据的默认值载体;pydantic 重校验,适合做面向用户输入的边界。

## `GraphRagConfig` 顶层模型如何聚合各子配置

打开 `graph_rag_config.py`,`GraphRagConfig` 的字段几乎就是 GraphRAG 全流程的目录:

- 模型相关:`completion_models: dict[str, ModelConfig]`、`embedding_models: dict[str, ModelConfig]`、`concurrent_requests: int`、`async_mode: AsyncType`。注意模型是**字典**而非单个对象——用户可以注册多个命名模型(如 `default_completion_model`),各工作流再用 `completion_model_id` 引用其一。
- 输入输出:`input: InputConfig`、`input_storage`/`output_storage`/`update_output_storage: StorageConfig`、`cache: CacheConfig`、`reporting: ReportingConfig`、`table_provider: TableProviderConfig`、`vector_store: VectorStoreConfig`。这些类型大多来自子包(`graphrag_input`、`graphrag_storage`、`graphrag_vectors`、`graphrag_cache`),配置树跨越了 monorepo 的包边界。
- 索引工作流:`chunking`、`embed_text`、`extract_graph`、`extract_graph_nlp`、`summarize_descriptions`、`prune_graph`、`cluster_graph`、`extract_claims`、`community_reports`、`snapshots`。
- 查询方法:`local_search`、`global_search`、`drift_search`、`basic_search`,对应第 11–13 章要讲的四种检索。
- `workflows: list[str] | None`:一个可选的工作流名列表,注释写明"This always overrides any built-in workflow methods",即用户可以手动指定执行顺序覆盖内置编排。

除了字段,`GraphRagConfig` 还挂了一组**校验与解析方法**,通过 pydantic 的 `@model_validator(mode="after")` 装饰的 `_validate_model` 在实例化后统一触发:

```
@model_validator(mode="after")
def _validate_model(self):
    self._validate_input_base_dir()
    self._validate_reporting_base_dir()
    self._validate_output_base_dir()
    self._validate_update_output_storage_base_dir()
    self._validate_vector_store()
    return self
```

这些方法做两件事。一是**校验**:例如 `_validate_input_base_dir` 在 `input_storage.type == StorageType.File` 时,若 `base_dir` 为空就抛 `ValueError`,并提示"Please rerun `graphrag init`"。二是**规范化副作用**:同一方法把相对路径用 `Path(...).resolve()` 转成绝对路径写回字段。`_validate_vector_store` 更进一步,它会遍历 `all_embeddings`(来自 `embeddings.py` 的 `entity_description`、`community_full_content`、`text_unit_text` 三种),为每一种在 `vector_store.index_schema` 里补上一条 `IndexSchema`,用默认 `vector_size` 兜底——这是"动态默认值":必须在拿到用户配置后才能算出来的默认。

此外两个便捷方法 `get_completion_model_config(model_id)` 与 `get_embedding_model_config(model_id)` 封装了"按 ID 取模型配置"的逻辑,找不到时抛带提示的 `ValueError`,让下游各工作流不必各自处理缺失情况。

还有一个微妙但关键的"两种系统互转"细节:`cache` 字段的默认值写法是 `CacheConfig(**asdict(graphrag_config_defaults.cache))`。`graphrag_config_defaults.cache` 是 `CacheDefaults` 这个 dataclass,而 `CacheConfig` 来自 `graphrag_cache` 子包(是 pydantic 模型),`dataclasses.asdict` 把前者递归拆成 dict 再喂给后者构造。这正是"dataclass 管默认、pydantic 管校验"两套机制在边界上的衔接方式。

## `defaults.py`:成百的 `DEFAULT_*` 常量承载"开箱即用"

`defaults.py` 是整个配置系统的"取值中心",有近 400 行。它分两层:

**顶层模块常量**,如 `DEFAULT_COMPLETION_MODEL = "gpt-4.1"`、`DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"`、`DEFAULT_MODEL_PROVIDER = "openai"`、`DEFAULT_COMPLETION_MODEL_ID = "default_completion_model"`、`ENCODING_MODEL = "o200k_base"`、`DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]`。这些值直接暴露了 GraphRAG 的"产品取向":默认锚定 OpenAI 生态、默认 tokenizer 用 `o200k_base`、默认抽取四类实体。

**分组 dataclass**,每个索引/查询步骤一个,如 `ChunkingDefaults`(`size=1200, overlap=100`)、`ClusterGraphDefaults`(`max_cluster_size=10, seed=0xDEADBEEF`)、`LocalSearchDefaults`(`text_unit_prop=0.5, community_prop=0.15, top_k_entities=10`)、`GlobalSearchDefaults`、`DriftSearchDefaults`、`CommunityReportDefaults`(`max_length=2000, max_input_length=8000`)等。这些数字就是 GraphRAG 论文与工程实践沉淀下来的"经验值"。

所有分组最终汇聚到 `GraphRagConfigDefaults` 这一个 dataclass,它用 `field(default_factory=...)` 把每个子默认对象作为字段持有,并在模块末尾实例化成单例:

```
vector_store_defaults = VectorStoreDefaults()
graphrag_config_defaults = GraphRagConfigDefaults()
```

于是,每个 pydantic 子模型的 `Field(default=...)` 都从这个单例取值。例如 `LocalSearchConfig.text_unit_prop` 的默认就是 `graphrag_config_defaults.local_search.text_unit_prop`。这种"模型只声明结构、默认值集中在 `defaults.py`"的拆分,意味着调整一个产品默认只需改一处常量,所有引用它的模型字段同步生效,且默认值可被打印、测试、复用(`init_content.py` 正是直接读它来生成模板)。

注意几个用 `ClassVar` 标注的默认(如 `StorageDefaults` 的子类 `InputDefaults.type`、`ReportingDefaults.type`、`VectorStoreDefaults.type`),它们是类级常量而非实例字段,表示"这个维度不开放给用户在该层逐实例改"。还有继承复用:`InputStorageDefaults`、`CacheStorageDefaults`、`OutputStorageDefaults`、`UpdateOutputStorageDefaults` 都继承自 `StorageDefaults`,只覆写 `base_dir`(分别为 `"input"`、`"cache"`、`"output"`、`"update_output"`),避免重复声明一整套存储字段。

几个默认值还透露了 GraphRAG 的工程取向,值得单独点出:`ExtractClaimsDefaults.enabled = False`——协变量(claims)抽取**默认关闭**,因为它会额外增加一轮 LLM 调用;`SnapshotsDefaults` 的 `embeddings`/`graphml`/`raw_graph` 全为 `False`——中间产物快照默认不落盘,省磁盘;`ExtractGraphNLPDefaults.async_mode = AsyncType.Threaded`、`concurrent_requests = 25`——fast 路线的 NLP 抽取默认用线程并发;`PruneGraphDefaults` 给出一组图剪枝阈值(`min_node_freq=2`、`min_edge_weight_pct=40.0`、`remove_ego_nodes=True`),`TextAnalyzerDefaults` 则携带一整套 spaCy 词性/语法规则(`noun_phrase_grammars` 字典、`exclude_pos_tags` 等)作为 fast 路线名词短语抽取的默认知识。这些"默认即最佳实践"的取值,是把领域经验编码进配置层的典型体现。

## `load_config.py`:yaml + .env + 环境变量插值的解析顺序

`graphrag/config/load_config.py` 出乎意料地薄——只有一个 `load_config(root_dir, cli_overrides)`,它直接委托给 `graphrag_common.config` 的同名 `load_config`,把 `GraphRagConfig` 当作 `config_initializer` 传进去:

```
return lc(
    config_initializer=GraphRagConfig,
    config_path=root_dir,
    overrides=cli_overrides,
)
```

真正的解析逻辑在 `graphrag-common/graphrag_common/config/load_config.py`。读完这段代码,可以厘清五个数据源的**合并顺序**:

1. **定位配置文件**:`_get_config_file_path` 在 `root_dir` 下按 `["settings.yaml", "settings.yml", "settings.json"]` 的顺序找第一个存在的文件;若传入的本身是文件则直接用;都找不到抛 `FileNotFoundError`。
2. **读原始文本**:`config_path.read_text(encoding="utf-8")`,此时还是带 `${VAR}` 占位符的纯文本。
3. **加载 .env**:若 `load_dot_env_file` 为真(默认),用 `python-dotenv` 的 `load_dotenv` 把配置文件**同目录**下的 `.env` 注入 `os.environ`。
4. **环境变量插值**:`_parse_env_variables` 用标准库 `string.Template(text).substitute(os.environ)` 把文本里的 `${GRAPHRAG_API_KEY}` 之类替换成真实环境变量值;**缺失变量直接抛 `ConfigParsingError`**(由 `KeyError` 转译),而不是静默留空——这避免了把字面量 `${...}` 当成 API key 发出去。
5. **解析为 dict**:按文件后缀选 `_parse_yaml`(`yaml.safe_load`)或 `_parse_json`。
6. **合并 CLI 覆盖**:若传入 `overrides`,用 `_recursive_merge_dicts` 把它**递归深合并**进 yaml 解析出的 dict——同名嵌套字典逐层合并而非整体替换,这正是 `load_config` docstring 里那个 `{'output': {'base_dir': 'override_value'}}` 例子能精准只改一个叶子的原因。
7. **切换工作目录**:`set_cwd` 默认为真时 `os.chdir(config_path.parent)`,使 yaml 里的相对路径(如 `prompts/extract_graph.txt`)以项目根为基准解析。
8. **实例化**:`config_initializer(**config_data)`,即 `GraphRagConfig(**config_data)`,触发前述全部 pydantic 校验。

所以优先级从低到高是:**子模型默认值 < .env/环境变量插值进 yaml 文本 < settings.yaml 显式值 < CLI overrides**。环境变量在这里不是覆盖某个配置字段,而是先做**文本级插值**,这是它和 CLI overrides(字典级深合并)在机制上的根本区别。

## `enums.py`:用枚举把字符串选项收紧成有限集合

`enums.py` 用 `Enum` 把若干"只能取有限几个字符串值"的选项固化下来,避免用户拼错:

- `ReportingType(str, Enum)`:`file` / `blob`,日志/报告输出去向。
- `AsyncType(str, Enum)`:`asyncio` / `threaded`,即 `async_mode` 字段的取值,默认 `Threaded`。
- `SearchMethod(Enum)`:`LOCAL` / `GLOBAL` / `DRIFT` / `BASIC`,对应四种查询。
- `IndexingMethod(str, Enum)`:`standard` / `fast` / `standard-update` / `fast-update`——`standard` 用 LLM 抽图,`fast` 用 NLP 抽图加 LLM 摘要,两个 `*-update` 是增量更新,这条枚举的 docstring 直接道出了 GraphRAG 两条索引路线的设计取舍。
- `NounPhraseExtractorType(str, Enum)`:`regex_english` / `syntactic_parser` / `cfg`,fast 路线下名词短语抽取器的三种实现。
- `ModularityMetric(str, Enum)`:`graph` / `lcc` / `weighted_components`,社区检测的模块度度量。

多数枚举继承 `str`(`class X(str, Enum)`),这样枚举成员本身就是字符串,既能直接和 yaml 里的字符串值比较、又能被 pydantic 无缝校验。其它存储/缓存/向量库类型的枚举(`StorageType`、`CacheType`、`VectorStoreType`、`ChunkerType`、`InputType`、`AuthMethod`)则定义在各自子包里,`defaults.py` 顶部把它们 import 进来当默认值用,体现了"配置语义分散在各功能包、由 config 层聚合"的 monorepo 风格。

## `errors.py`:把"配置缺失"翻译成可操作的人话

`errors.py` 定义了四个继承 `ValueError` 的异常:`ApiKeyMissingError`、`AzureApiBaseMissingError`、`AzureApiVersionMissingError`、`ConflictingSettingsError`。它们的共性是**错误信息里都带补救动作**,例如 `ApiKeyMissingError` 拼出 `"API Key is required for {llm_type} ... Please rerun \`graphrag init\` and set the API_KEY."`。这种把"是什么错 + 怎么修"写进异常消息的做法,和前面各 `_validate_*` 方法里的提示一脉相承,降低了配置出错时的排查成本。

## language_model 配置:`ModelConfig` 如何声明 provider、auth 与限流

`completion_models` 与 `embedding_models` 字典的值类型是 `ModelConfig`,来自 `graphrag_llm/config/model_config.py`。它是单个语言模型的完整声明,关键字段:

- `type: str`,默认 `LLMProviderType.LiteLLM`——底层走 LiteLLM 统一多家供应商。
- `model_provider: str`(必填,如 `openai`/`azure`)与 `model: str`(必填,如 `gpt-4.1`)。
- 鉴权:`api_base` / `api_version` / `api_key`,以及 `auth_method: AuthMethod`(默认 `ApiKey`,另有 `AzureManagedIdentity`)、`azure_deployment_name`。
- 行为:`call_args`(透传给底层 API 的关键字参数)、`retry: RetryConfig`、`rate_limit: RateLimitConfig`、`metrics: MetricsConfig`、`mock_responses`(测试用)。

`ModelConfig` 设了 `model_config = ConfigDict(extra="allow")`,允许 yaml 里写出模型未显式声明的字段——为自定义 LLM provider 留了扩展口。它的 `@model_validator` 里 `_validate_lite_llm_config` 编码了一组互斥规则:`azure` provider 必须给 `api_base`;非 `azure` 不能给 `azure_deployment_name`;用 `AzureManagedIdentity` 时不能再设 `api_key`,反之 `auth_method=api_key` 时必须有 `api_key`。这些规则把"配置组合是否自洽"的判断提前到了加载期。

限流由 `RateLimitConfig` 承载:`type` 默认 `sliding_window`,加上 `period_in_seconds`、`requests_per_period`、`tokens_per_period`,其校验器要求滑动窗口下至少指定 RPM 或 TPM 之一且为正整数。配合 `GraphRagConfig.concurrent_requests`(默认 25),GraphRAG 在并发与限流两个维度上对大批量 LLM 调用做了控制。

## `graphrag init` 生成的初始配置模板

新用户怎么得到第一份 `settings.yaml`?答案在 `init_content.py` 与 CLI 的 `initialize.py`。`init_content.py` 用一个 f-string 常量 `INIT_YAML` 拼出模板,且**直接内插 `graphrag_config_defaults` 的值**——例如 `chunking.size` 那一行写的是 `{graphrag_config_defaults.chunking.size}`,保证模板里展示的默认与代码默认永远一致,不会漂移。模板把 `model` 与 `api_key` 留成占位符:`model: <DEFAULT_COMPLETION_MODEL>`、`api_key: ${{GRAPHRAG_API_KEY}}`;另有 `INIT_DOTENV = "GRAPHRAG_API_KEY=<API_KEY>\n"`。

`cli/initialize.py` 的 `initialize_project_at(path, force, model, embedding_model)` 把模板落盘:创建根目录与 `input/` 子目录;用 `INIT_YAML.replace("<DEFAULT_COMPLETION_MODEL>", model)`(注意是 `replace` 而非 `format`,因为要保留 `${GRAPHRAG_API_KEY}` 占位符给后续 .env 插值)写出 `settings.yaml`;写出 `.env`;再把十多个内置 prompt 模板(`GRAPH_EXTRACTION_PROMPT`、`COMMUNITY_REPORT_PROMPT`、`LOCAL_SEARCH_SYSTEM_PROMPT` 等)逐一写进 `prompts/` 目录,文件名与 yaml 里 `prompt:` 字段引用的路径一一对应。这样 init 之后,用户只要在 `.env` 填上真实 key 就能跑通——"开箱即用"在此闭环。

值得回味的是,模板里只列了**少数关键项**(模型、切分、存储路径、各步骤的 prompt 路径与 model id),而绝大多数旋钮(各种 token 上限、`top_k`、并发数)都不出现在生成的 yaml 里,完全靠 `defaults.py` 兜底。这正是分层默认设计的最终用户体验:配置文件短小、可读,深度调参时再按需添加字段。

## prompt 的"路径即配置":`resolved_prompts` 的惰性读取

配置里大量出现 `prompt`/`graph_prompt`/`map_prompt` 这类字段,它们的类型都是 `str | None`,默认值都是 `None`(见各 `*Defaults`)。这里有个一致的设计模式:**配置里存的是 prompt 文件的路径,而不是 prompt 内容本身**。以 `ExtractGraphConfig` 为例,它有一个 `resolved_prompts()` 方法:

```
def resolved_prompts(self) -> ExtractGraphPrompts:
    return ExtractGraphPrompts(
        extraction_prompt=Path(self.prompt).read_text(encoding="utf-8")
        if self.prompt
        else GRAPH_EXTRACTION_PROMPT,
    )
```

逻辑是:若用户在 yaml 里给了 `prompt` 路径,就在用时 `Path(...).read_text()` 读盘;否则回退到代码内置的 `GRAPH_EXTRACTION_PROMPT` 常量。`CommunityReportsConfig.resolved_prompts()` 同理同时处理 `graph_prompt` 与 `text_prompt`。把读盘做成方法而非字段,意味着配置实例化时不碰文件系统(配置可以纯内存构造、可测试),只有真正要用 prompt 时才解析——既支持"用户自定义 prompt 文件"又保证"零配置也能跑"。`graphrag init` 生成的模板恰好把这些 `prompt:` 字段填成 `prompts/*.txt` 的相对路径,与落盘的 prompt 文件对应。

## 本章小结

- `GraphRagConfig`(`graph_rag_config.py`)是顶层 pydantic `BaseModel`,字段几乎覆盖 GraphRAG 全流程:模型、输入输出存储、各索引步骤、四种查询方法。
- 配置系统用了两套机制:面向用户输入的边界用 pydantic `BaseModel` 做校验,内部默认值容器用标准库 `@dataclass`,二者职责分离。
- `defaults.py` 是默认值中心,顶层常量(`DEFAULT_COMPLETION_MODEL="gpt-4.1"` 等)与分组 dataclass 汇聚成单例 `graphrag_config_defaults`,所有子模型的 `Field(default=...)` 都从它取值。
- `load_config` 仅是薄封装,真正解析在 `graphrag_common`:按 yaml/yml/json 顺序定位文件、加载同目录 `.env`、用 `string.Template` 做环境变量插值(缺失即报错)、解析成 dict、再用 `_recursive_merge_dicts` 深合并 CLI overrides。
- 优先级:子模型默认 < 环境变量插值 < settings.yaml 显式值 < CLI overrides;环境变量是文本级插值,CLI overrides 是字典级深合并,机制不同。
- `@model_validator(mode="after")` 触发的 `_validate_*` 方法既校验(缺 `base_dir` 报错)又规范化(相对路径 `resolve()`、为 `all_embeddings` 补 `IndexSchema` 动态默认)。
- `enums.py` 用 `str, Enum` 把选项收紧成有限集合(`AsyncType`、`SearchMethod`、`IndexingMethod`、`NounPhraseExtractorType` 等),配合 pydantic 防拼写错误。
- 语言模型用 `ModelConfig`(`extra="allow"`)声明,默认走 LiteLLM,内置 Azure/ApiKey 互斥校验;`RateLimitConfig` 与 `concurrent_requests` 共同控制 LLM 调用压力。
- `graphrag init` 经 `INIT_YAML`(内插默认值)与 `initialize_project_at` 生成 `settings.yaml` + `.env` + `prompts/`,模板只暴露少数关键项,其余靠默认兜底。
- prompt 字段存的是文件路径而非内容,`resolved_prompts()` 惰性读盘、缺省回退到内置常量(如 `GRAPH_EXTRACTION_PROMPT`),使配置可纯内存构造、零配置也能跑。

## 动手实验

1. **实验一:画出配置树** — 用 Read 通读 `graph_rag_config.py`,把 `GraphRagConfig` 的每个 `Field` 字段名与其类型(尤其哪些是子 `BaseModel`、哪些来自子包如 `InputConfig`/`StorageConfig`)列成一张树状清单,标注哪些字段是 `dict`(如 `completion_models`)、哪些是 `list | None`(如 `workflows`),理解配置树的拓扑。

2. **实验二:追一个默认值的来回** — 选定 `chunking.size`,在 `defaults.py` 找到 `ChunkingDefaults.size = 1200`,确认它如何经 `GraphRagConfigDefaults.chunking` 与单例 `graphrag_config_defaults` 暴露;再到 `graph_rag_config.py` 看 `ChunkingConfig` 默认怎么引用它;最后在 `init_content.py` 找到 `{graphrag_config_defaults.chunking.size}` 那行,验证"改一处常量、三处同步"。

3. **实验三:复现环境变量插值与报错** — 阅读 `graphrag_common/config/load_config.py` 的 `_parse_env_variables`,在纸上推演:当 `settings.yaml` 含 `api_key: ${GRAPHRAG_API_KEY}` 而环境中**未**设该变量时,`Template(...).substitute` 会抛 `KeyError`、被转译为哪个异常类(`ConfigParsingError`);再推演设了变量时替换后的文本,理解为何插值发生在 yaml 解析之前。

4. **实验四:体会校验器的副作用** — 阅读 `GraphRagConfig._validate_vector_store` 与 `_validate_input_base_dir`,说明前者如何遍历 `embeddings.py` 的 `all_embeddings` 为缺失项补 `IndexSchema`(动态默认),后者如何在 `StorageType.File` 下把 `base_dir` 用 `Path(...).resolve()` 写回绝对路径;思考"为什么这类逻辑放在 `mode="after"` 的验证器里,而不是字段默认值里"。

> **下一章预告**：配置就位后,流水线的第一站是把磁盘上的文档读进内存。第 3 章《输入加载 Input》将解构 `graphrag_input` 包:`InputConfig` 如何声明 csv/text/json/jsonl 输入类型,加载器怎样把文件读成统一的文档表,以及 `id_column`/`title_column`/`text_column` 等字段如何映射原始数据。
