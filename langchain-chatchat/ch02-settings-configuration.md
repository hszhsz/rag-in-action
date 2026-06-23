# 第 2 章：配置系统 Settings

一个能够离线部署、对接多种模型平台、又要兼顾知识库、Agent 工具与提示词模板的 RAG 应用，配置项的数量天然就会膨胀。Langchain-Chatchat 没有把这些参数散落在代码各处，而是用一套基于 `pydantic-settings` 的分层配置子系统把它们收编起来。

具体而言：每一类关注点对应一个 `BaseFileSettings` 子类，每个子类对应一个 yaml 文件，所有子类再由一个 `SettingsContainer` 聚合成全局单例 `Settings`。读懂这套机制，是读懂整个项目的前提——后续章节几乎每一处行为，最终都能回溯到这里的某个默认值。

本章解构 `chatchat/pydantic_settings_file.py` 与 `chatchat/settings.py` 两个文件。前者是配置基础设施：自定义的 settings 基类、yaml 模板生成、文件变更后的自动重载；后者是配置内容本身：从服务器端口到模型平台清单，从切分块大小到 Agent 工具开关。

我们会看到，这套设计的核心取向是「默认即可用，但默认偏保守」——开箱跑得起来，又不会在用户没明确开启的情况下擅自联网或写库。本章还会顺带牵出 `utils.py` 里日志配置与 `log_verbose` 的联动，作为「配置即时生效」的一个具体例证。

## 配置基础设施：BaseFileSettings 与 yaml 文件源

所有子配置的共同基类是 `pydantic_settings_file.py` 中的 `BaseFileSettings`，它继承自 `pydantic-settings` 的 `BaseSettings`。`pydantic-settings` 是 pydantic 官方推出的配置管理库，其特色是允许把配置来源（环境变量、`.env`、yaml、init 参数等）声明为有优先级的「源」并自动合并 [[Settings Management - pydantic-settings]](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)。

`BaseFileSettings` 用 `model_config = SettingsConfigDict(...)` 设置了几个全局行为：`use_attribute_docstrings=True` 让字段下方的文档字符串被 pydantic 采集为 schema 描述（这正是后面生成 yaml 注释的素材来源），`extra="ignore"` 表示忽略 yaml 中多余的键，再加上 `yaml_file_encoding` 与 `env_file_encoding` 统一为 utf-8。

这套基类的价值在于「约定下沉」：所有子配置无须各自重复声明编码、docstring 采集、源优先级等通用行为，只要继承 `BaseFileSettings` 就自动获得一致的加载语义，子类只需关心自己有哪些字段、绑定哪个文件。

真正决定「配置从哪里来」的，是被重写的 `settings_customise_sources` 类方法：

```python
return init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls)
```

这个返回元组的顺序就是优先级——靠前的源覆盖靠后的源。也就是说，`init_settings`（代码里实例化时传入的参数）最高，其次是进程环境变量 `env_settings`，再次是 `.env` 文件，最后才是 yaml 文件。

这一点很关键：yaml 文件给出基线值，而部署时若想临时覆盖某个端口或开关，用环境变量即可，无须改文件。这种优先级安排让「持久化基线」与「临时覆盖」各司其职，特别适合容器化部署——镜像里打包好 yaml，运行时用环境变量按环境差异微调。

每个子类各自通过 `model_config` 指定自己的 yaml 文件路径，例如 `BasicSettings` 是 `CHATCHAT_ROOT / "basic_settings.yaml"`，`KBSettings` 是 `kb_settings.yaml`，`ApiModelSettings` 是 `model_settings.yaml`，`ToolSettings` 是 `tool_settings.yaml`，`PromptSettings` 是 `prompt_settings.yaml`。一类关注点一个文件，用户改配置时不必在一个巨型文件里翻找。

这里还有一个容易被忽略却影响行为的细节：`ToolSettings` 与 `PromptSettings` 的 `model_config` 把 `extra` 覆写成了 `"allow"`（而基类默认是 `"ignore"`）。这意味着工具与提示词配置允许出现 schema 未声明的额外字段而不被丢弃——因为工具种类会随插件扩展，提示词模板也常需要用户自定义键，放宽 `extra` 给了二者更大的弹性。同时这两个类还额外声明了 `json_file`，使它们既能导出 yaml 也能导出 json。

另外要区分两个基类。`BaseFileSettings` 是「会绑定文件、会作为配置源」的顶层配置；而 `MyBaseModel` 是供嵌套使用的普通模型基类，它用的是 `ConfigDict` 而非 `SettingsConfigDict`，且 `extra="allow"`。被 `MODEL_PLATFORMS` 列表嵌套的 `PlatformConfig` 继承的正是 `MyBaseModel`——它本身不绑定文件，只是作为父配置中的一个结构化子项存在。两个基类都开了 `use_attribute_docstrings=True`，确保嵌套层级的字段注释同样能被模板生成器采集。

## yaml 模板生成：从 pydantic 模型反向产出带注释的配置

光有「读 yaml」还不够，项目还要能「写 yaml」——第一次部署时根据默认值生成一份带中文注释的配置模板供用户编辑。这件事由 `YamlTemplate` 类承担。

它的工作流是：先用 `model_obj.model_dump(**self.dump_kwds)` 把模型实例 dump 成 dict，再借助 `ruamel.yaml`（而非标准 `pyyaml`）加载成可携带注释的 `CommentedBase` 对象。选择 `ruamel.yaml` 的原因正在于它能保留并插入注释——`import_yaml()` 里专门设置了 `block_seq_indent`、`map_indent`、`sequence_dash_offset` 等缩进参数，让输出排版整齐。

注释从哪来？`get_class_comment` 读取 `model_cls.model_json_schema().get("description")`，也就是类的 docstring；`get_field_comment` 读取字段 schema 里的 `description`，若字段是枚举还会追加一行「可选值：...」。

由于基类开启了 `use_attribute_docstrings=True`，源码里写在字段下方的那些三引号说明（如 `"""是否开启日志详细信息"""`）会被自动转成 yaml 注释。换句话说，源码注释即用户文档，二者天然同步——开发者更新字段说明时，生成的配置模板注释也跟着更新，杜绝了「代码改了文档没改」的常见割裂。

`create_yaml_template` 还支持嵌套：通过 `SubModelComment`（一个 `TypedDict`）描述子模型如何展开注释，内部递归函数 `_set_subfield_comment` 逐层下钻。

`SettingsContainer.createl_all_templates` 调用时就为 `MODEL_PLATFORMS` 传入了 `{"model_obj": PlatformConfig(), "is_entire_comment": True}`，使得平台列表项也带上字段级注释。生成的模板最终可由 `write_to` 写盘，而该流程运行 `python -m chatchat.settings` 即可触发（文件末尾的 `__main__` 块调用了 `createl_all_templates`）。`BaseFileSettings.create_template_file` 还支持 `file_format="json"`，对配了 `json_file` 的 `ToolSettings`、`PromptSettings` 可导出 json 版本。

## 自动重载：文件改了就重新加载

配置文件改了要不要重启？`BaseFileSettings` 在 `model_post_init` 里把 `self._auto_reload = True` 作为默认，并暴露 `auto_reload` 属性（带 setter，可在运行时关掉）。配合文件顶部的缓存机制，配置可以在文件变更后自动刷新。

关键在 `_cached_settings` 函数，它被 `memoization` 的 `@cached` 装饰，`max_size=1`、`thread_safe=True`，并指定了 `custom_key_maker=_lazy_load_key`。`_lazy_load_key` 的逻辑是：对每个可能的配置文件（`env_file`、`json_file`、`yaml_file`、`toml_file`），若文件存在且非空，就取它的修改时间 `int(os.path.getmtime(file))` 作为缓存键的一部分。

一旦文件被改动，mtime 变化，缓存键随之变化，缓存失效，`_cached_settings` 内部就会执行 `settings.__init__()` 重新从文件加载。这是一种巧妙的「以文件 mtime 当版本号」的失效策略——不需要轮询、不需要文件监听线程，只在真正读取配置时顺带比对一次时间戳，开销极低。

把这个机制接到全局对象上的是 `settings_property`：它返回一个 `property`，每次访问都走 `_cached_settings(settings)`。于是 `SettingsContainer` 里的 `basic_settings`、`kb_settings` 等都被声明为 `settings_property(BasicSettings())`，访问 `Settings.basic_settings` 时既享受缓存、又能在文件变更后拿到最新值。

`BasicSettings` 的 docstring 里也点明了边界：「除 log_verbose/HTTPX_DEFAULT_TIMEOUT 修改后即时生效，其它配置项修改后都需要重启服务器才能生效」——即时生效靠的就是这套重载，而那些只在启动时读取一次的配置自然无法热更新。读者应理解：自动重载只保证「再次读取时拿到新值」，至于一个值改了能否真正影响行为，还取决于消费方是每次现读、还是启动时读一次后缓存在自己手里。

## 全局单例：SettingsContainer 与 Settings

所有子配置最终汇聚到文件末尾的 `SettingsContainer` 类。它把 `CHATCHAT_ROOT` 作为类属性暴露，又把五个子配置声明为类级别的 `settings_property`：

```python
basic_settings: BasicSettings = settings_property(BasicSettings())
kb_settings: KBSettings = settings_property(KBSettings())
model_settings: ApiModelSettings = settings_property(ApiModelSettings())
tool_settings: ToolSettings = settings_property(ToolSettings())
prompt_settings: PromptSettings = settings_property(PromptSettings())
```

模块最底部一行 `Settings = SettingsContainer()` 实例化出全局唯一对象，整个项目都以 `from chatchat.settings import Settings` 的方式取用配置，再以 `Settings.model_settings.DEFAULT_LLM_MODEL` 这样的点路径读到具体值。这是一种「单一入口」设计：配置的读取面收敛到一个对象，谁要改默认值，只需顺着 `Settings.<子配置>.<字段>` 反查即可。

容器还提供两个管理方法。`createl_all_templates` 逐个调用子配置的 `create_template_file(write_file=True)` 生成模板文件，其中 `model_settings` 那次特意传入 `sub_comments` 让 `MODEL_PLATFORMS` 展开为带注释的完整结构。

`set_auto_reload` 则一次性开关所有子配置的 `auto_reload`，方便在某些不希望热重载的场景（如批量测试、需要稳定快照的压测）下统一关闭。这两个方法把「生成模板」和「控制重载」这两类横切操作收敛到容器一层，避免了在各子配置上重复调用。

紧随实例化之后，模块执行了 `nltk.data.path.append(str(Settings.basic_settings.NLTK_DATA_PATH))`——把配置中约定的 nltk 数据路径注册进 nltk 的搜索路径。这说明配置对象在模块 import 阶段就被实地消费了，而非等到运行时才生效，也再次印证了路径常量「随包发行的资源」与「用户数据」分治的设计。

## BasicSettings：服务器、路径与目录约定

`BasicSettings` 管的是最基础的服务器与路径配置。它的字段揭示了几条默认取向。

服务监听上，`DEFAULT_BIND_HOST` 在非 Windows 平台默认 `"0.0.0.0"`，Windows 上默认 `"127.0.0.1"`（因为 Windows 下 WEBUI 自动弹浏览器访问 `0.0.0.0` 会失败）。这种按平台分支的默认值，反映出项目对跨平台部署细节的考量。

`API_SERVER` 默认 `{"host": DEFAULT_BIND_HOST, "port": 7861, "public_host": "127.0.0.1", "public_port": 7861}`，其中 `public_host`/`public_port` 专门用于生成公网可访问链接；`WEBUI_SERVER` 默认端口 `8501`（Streamlit 的惯用端口）。

`OPEN_CROSS_DOMAIN` 默认 `False`，跨域默认关闭，体现保守取向。`HTTPX_DEFAULT_TIMEOUT` 默认 `300` 秒，给模型加载和慢对话留足余量。这两项与 `log_verbose` 一样，属于「无须重启即生效」的少数配置。

路径方面，一切以 `CHATCHAT_ROOT` 为锚。它在文件顶层被定义为 `Path(os.environ.get("CHATCHAT_ROOT", ".")).resolve()`——优先读环境变量，未设置则退化为当前目录。围绕它，一组 `@cached_property` 派生出目录树：`DATA_PATH = CHATCHAT_ROOT / "data"`，`LOG_PATH = DATA_PATH / "logs"`，`MEDIA_PATH = DATA_PATH / "media"`，`BASE_TEMP_DIR = DATA_PATH / "temp"`（用于文件对话，并在属性内顺手创建了 `openai_files` 子目录）。

而 `PACKAGE_ROOT` 指向 `Path(__file__).parent`（代码包目录），`IMG_DIR`、`NLTK_DATA_PATH` 都挂在它下面——这区分了「用户数据」与「随包发行的资源」两类路径。把这些路径写成 `@cached_property` 而非普通字段，意味着它们是从 `CHATCHAT_ROOT` 计算出来的派生量，不会被写入 yaml 模板，也不会被用户误改成互相矛盾的值。

知识库存储路径 `KB_ROOT_PATH` 默认 `CHATCHAT_ROOT / "data/knowledge_base"`，数据库则默认走 sqlite：`SQLALCHEMY_DATABASE_URI` 默认 `"sqlite:///" + str(CHATCHAT_ROOT / "data/knowledge_base/info.db")`。docstring 也提示：用 sqlite 改 `DB_ROOT_PATH` 即可，换其它数据库则直接改 `SQLALCHEMY_DATABASE_URI`。默认用 sqlite 而非 MySQL/PostgreSQL，同样是为了零依赖启动。

最后，`make_dirs` 方法统一创建 `DATA_PATH`、`MEDIA_PATH`、`LOG_PATH`、`BASE_TEMP_DIR` 以及 `media` 下的 `image/audio/video` 子目录和 `KB_ROOT_PATH`，把目录约定一次性落地——避免运行时因目录缺失而报错。

值得注意的是 `version: str = __version__`，它把生成模板时的代码版本写进配置。docstring 明确说若版本不一致建议重建模板——这是一种防止配置 schema 漂移的自检手段。

当代码升级、新增或删除了字段，旧 yaml 与新代码的 schema 就可能错位；把版本号写进文件，等于给配置打了个时间戳，方便用户在升级后判断是否需要重新生成模板，再把自定义值迁移回去。

## ModelSettings：MODEL_PLATFORMS 的多平台模型声明

模型相关配置集中在 `ApiModelSettings`（其 yaml 文件为 `model_settings.yaml`）。两个最常被引用的默认值是 `DEFAULT_LLM_MODEL: str = "glm4-chat"` 和 `DEFAULT_EMBEDDING_MODEL: str = "bge-m3"`——分别是默认对话模型与默认嵌入模型，选型偏向中文场景。

其它通用参数包括 `HISTORY_LEN: int = 3`（默认历史轮数）、`TEMPERATURE: float = 0.7`、`MAX_TOKENS` 默认 `None`（不填则用模型自身上限）。`SUPPORT_AGENT_MODELS` 列出了一批已知支持 Agent 的模型名，如 `chatglm3-6b`、`glm-4`、`openai-api`、`Qwen-2`、`gpt-4o` 等——这份白名单决定了哪些模型可以进入 Agent 链路。还有一个 `Agent_MODEL` 字段默认留空，docstring 说明不指定就沿用 `DEFAULT_LLM_MODEL`，指定后则锁定 Agent 内部使用的模型。

`LLM_MODEL_CONFIG` 是按「角色」组织的模型参数表，键有 `preprocess_model`（意图识别，`temperature` 仅 0.05）、`llm_model`（主对话，`temperature` 0.9）、`action_model`（Agent 动作，`temperature` 0.01、`prompt_name` 为 `"ChatGLM3"`）、`postprocess_model`、`image_model`（默认 `sd-turbo`）。每个角色还各自带有 `max_tokens`、`history_len`、`prompt_name`、`callbacks` 等子参数。

每个角色的 `model` 字段留空，docstring 说明留空则自动回落到 `DEFAULT_LLM_MODEL`——同一套模型可以扮演不同角色，而温度等参数按角色定制（识别要稳、对话要活、动作要准）。

这种「一个模型多种人格」的设计避免了为每个环节都配一个独立模型的部署负担，同时又通过温度差异让各环节表现出契合其职责的随机性。意图识别几乎不需要发散（0.05），动作生成要尽量确定（0.01），而主对话则允许较大的创造性（0.9）。

真正体现「多平台」的是 `MODEL_PLATFORMS: t.List[PlatformConfig]`。`PlatformConfig` 继承自 `MyBaseModel`（注意它用的是 `MyBaseModel` 而非 `BaseFileSettings`，因为它是被嵌套的子模型，`extra="allow"` 容许额外字段）。每个平台声明 `platform_name`、`platform_type`（受限于 `Literal["xinference", "ollama", "oneapi", "fastchat", "openai", "custom openai"]`）、`api_base_url`、`api_key`、`api_proxy`、`api_concurrencies`（单模型最大并发，默认 5），再按模型类型分别列出 `llm_models`、`embed_models`、`text2image_models`、`image2text_models`、`rerank_models`、`speech2text_models`、`text2speech_models`。这些列表字段类型是 `Union[Literal["auto"], List[str]]`，配合 `auto_detect_model` 开关——设为 `True` 时可自动探测平台上可用的模型，无须手工维护清单。

这种「按模型类型分桶 + 可选自动探测」的结构，把异构平台的能力差异抹平成统一的声明：无论底层是本地的 Xinference 还是云端的 OpenAI，对上层而言都只是「某平台提供了哪些 llm/embed/rerank 模型」。

`api_proxy` 字段则为需要走代理才能访问的境外平台留了口子，`api_concurrencies` 又给每个平台单独设了并发上限，使得不同算力规格的平台可以各自限流，互不拖累。这正是「平台无关」抽象的好处——上层调度逻辑面对的是同构的 `PlatformConfig` 列表，而非一堆各有脾气的具体 SDK。

默认的 `MODEL_PLATFORMS` 预置了四个平台。第一个是 `xinference`（`http://127.0.0.1:9997/v1`，且 `auto_detect_model=True`，作为离线部署的首选，它能托管 LLM、嵌入、重排、图像、语音等多种模型）。

第二个是 `ollama`（`http://127.0.0.1:11434/v1`，预填了 `qwen:7b`、`qwen2:7b` 等本地 LLM 与 `quentinz/bge-large-zh-v1.5` 嵌入模型）。第三个是 `oneapi`（`http://127.0.0.1:3000/v1`，作为聚合层接入智谱、千问、千帆、星火等国内云端 API）。第四个是 `openai`（`https://api.openai.com/v1`，列出 `gpt-4o`、`gpt-3.5-turbo` 与 `text-embedding-3-small/large`）。

值得玩味的是：除 openai 外，默认地址全部指向本地回环 `127.0.0.1`，呼应「可离线」的定位——用户开箱即对接本机的 Xinference 或 Ollama，而非外部云端，无网络环境也能跑通整条 RAG 链路。

## KBSettings 与 ToolSettings：检索参数与工具开关

`KBSettings`（`kb_settings.yaml`）管知识库行为。`DEFAULT_KNOWLEDGE_BASE: str = "samples"`、`DEFAULT_VS_TYPE` 默认 `"faiss"`（向量库类型受 `Literal` 约束，可选 faiss/milvus/zilliz/pg/es/relyt/chromadb）。

FAISS 作为默认，正是因为它纯本地、无须额外服务，最契合开箱即用；而其它选项如 milvus、es 等则面向需要更大规模或更强检索能力的部署，留待用户按需切换。

文本切分相关的 `CHUNK_SIZE: int = 750`、`OVERLAP_SIZE: int = 150` 是后续文档处理章节会反复用到的基线——750 字一段、相邻段重叠 150 字，是中文 RAG 较常见的折中。检索侧 `VECTOR_SEARCH_TOP_K: int = 3`、`SCORE_THRESHOLD: float = 2.0`——docstring 特意解释 SCORE 越小相关度越高、取 2.0 相当于不筛选、建议设到 0.5 左右，说明默认值刻意放宽以免新手「检索不到东西」。

这种「宁可多召回、不要漏召回」的默认取向，与前面服务跨域默认关闭、工具默认关闭形成有趣的对照：涉及安全与副作用的开关一律保守关闭，而涉及检索体验的阈值则一律放宽——两类默认背后是同一个目标，即让新手开箱就能得到「能用」的结果。

`TEXT_SPLITTER_NAME` 默认 `"ChineseRecursiveTextSplitter"`，再次印证中文优先。`kbs_config` 与 `text_splitter_dict` 则用嵌套字典声明了各向量库连接参数与各切分器的 tokenizer 来源；其中 `MarkdownHeaderTextSplitter` 还预设了 `#` 到 `####` 四级标题的切分规则。还有面向缓存的 `CACHED_VS_NUM: int = 1` 与 `CACHED_MEMO_VS_NUM: int = 10`（后者用于文件对话的临时向量库），以及搜索引擎相关的 `DEFAULT_SEARCH_ENGINE: "duckduckgo"`、`SEARCH_ENGINE_TOP_K: int = 3`。

源码里还留有若干 `# TODO` 注释（如 `VECTOR_SEARCH_TOP_K` 与 tool 配置项重复、`KB_INFO` 是否还有必要），坦诚记录了配置项的历史包袱，也提醒读者：分层配置在演进中难免出现职责交叠，这本身是一手的工程史料。

`ToolSettings`（`tool_settings.yaml`，并额外指定了 `json_file` 与 `extra="allow"`）管 Agent 工具。这里最一致的设计是：几乎每个工具配置 dict 都以 `"use": False` 开头——`search_local_knowledgebase`、`search_internet`、`arxiv`、`weather_check`、`calculate`、`text2sql`、`amap`、`url_reader`、`text2promql`、`text2images` 等无一例外默认关闭。

这是典型的安全默认：联网搜索、执行 SQL、调用外部 API 这类有副作用或需要密钥的能力，必须用户显式开启。以 `text2sql` 为例，它的配置里 `read_only` 默认 `False` 但 docstring 用大段文字反复提醒生产环境的只读与权限风险（建议主从架构、从数据库层面收回写权限等），体现了对「大模型生成 SQL」这类高危操作的审慎态度。

`search_internet` 的默认引擎是 `"duckduckgo"`，并内置了 bing/metaphor/searx 等多引擎的配置骨架，开关一开即可切换。把工具开关、密钥、引擎选择全部沉到配置里，意味着同一份代码可以在「纯本地问答」与「联网 Agent」两种形态间切换，而切换的代价仅仅是改几个 `use` 标志位。

至于 `PromptSettings`（`prompt_settings.yaml`），它集中存放各角色的提示词模板——`preprocess_model`（意图识别）、`llm_model`、`rag`、`action_model`、`postprocess_model`。

类 docstring 说明：除 Agent 模板用 f-string 外，其余均用 jinja2 格式，因此模板里能看到 `{{context}}`、`{{question}}` 这类 jinja2 占位符。把提示词也纳入配置体系，意味着用户无须改代码即可调整问答风格、约束措辞——这对需要本地化、合规化输出的场景尤为重要。其中 `action_model` 还按 `glm3`、`qwen`、`structured-chat-agent` 等不同 Agent 范式预置了成套的 `SYSTEM_PROMPT`/`HUMAN_MESSAGE`，可见提示词同样是「按模型/范式分桶」组织的。

## 日志配置：log_verbose 如何贯穿全局

配置与日志的连接点在 `utils.py`。`build_logger` 基于 `loguru` 构造彩色日志器，被 `memoization` 的 `@cached` 装饰以避免重复添加 handler 导致日志重复输出（源码注释直言「默认每调用一次 build_logger 就会添加一次 handler，导致 chatchat.log 里重复输出」）。日志文件默认落在 `Settings.basic_settings.LOG_PATH` 下——配置里的路径约定在此被消费。

过滤函数 `_filter_logs` 直接读取 `Settings.basic_settings.log_verbose`：当其为 `False` 时，等级不高于 10 的 debug 日志被丢弃，等级为 40 的错误日志会被抹去 traceback。这正是前面所说「`log_verbose` 修改后即时生效」的实现——它每次记录日志都现读配置，而不缓存。`get_config_dict` 则为标准 `logging` 提供一份带 `RotatingFileHandler`（按 `maxBytes`/`backupCount` 轮转）的配置字典，供需要走 `logging` 体系的子系统使用。

## 本章小结

- 配置子系统建立在 `pydantic-settings` 之上，基类 `BaseFileSettings` 通过重写 `settings_customise_sources` 确立优先级：init 参数 > 环境变量 > `.env` > yaml 文件。
- 每类关注点对应一个子类与一个 yaml 文件：`BasicSettings`/`basic_settings.yaml`、`KBSettings`/`kb_settings.yaml`、`ApiModelSettings`/`model_settings.yaml`、`ToolSettings`/`tool_settings.yaml`、`PromptSettings`/`prompt_settings.yaml`。
- `YamlTemplate` 借助 `ruamel.yaml` 把 pydantic 模型反向生成带注释的配置模板，注释直接取自字段 docstring（`use_attribute_docstrings=True`），实现「源码注释即用户文档」。
- 自动重载由 `_cached_settings` + `_lazy_load_key` 实现：以配置文件 mtime 作为缓存键，文件一变即重新 `__init__` 加载；`settings_property` 把该机制接到全局 `Settings` 对象上。
- 路径约定以 `CHATCHAT_ROOT`（环境变量优先）为锚，派生出 `DATA_PATH`、`LOG_PATH`、`MEDIA_PATH`、`BASE_TEMP_DIR`、`KB_ROOT_PATH`，并用 `make_dirs` 统一创建。
- 模型默认值偏中文场景：`DEFAULT_LLM_MODEL="glm4-chat"`、`DEFAULT_EMBEDDING_MODEL="bge-m3"`、`TEXT_SPLITTER_NAME="ChineseRecursiveTextSplitter"`。
- `MODEL_PLATFORMS` 用 `PlatformConfig` 列表声明多平台，默认预置 xinference/ollama/oneapi/openai 且地址全部指向本地回环，呼应「可离线」定位；`auto_detect_model` 与 `"auto"` 字面量支持模型自动探测。
- `LLM_MODEL_CONFIG` 按角色（preprocess/llm/action/postprocess/image）定制温度等参数，`model` 留空则回落到 `DEFAULT_LLM_MODEL`。
- 检索默认偏宽松：`CHUNK_SIZE=750`、`OVERLAP_SIZE=150`、`VECTOR_SEARCH_TOP_K=3`、`SCORE_THRESHOLD=2.0`（约等于不筛选）。
- `ToolSettings` 中几乎所有工具默认 `"use": False`，联网/写库/外部 API 必须显式开启，体现安全默认取向。
- 日志通过 `log_verbose` 即时生效（`_filter_logs` 每次现读配置），日志路径复用 `BasicSettings.LOG_PATH`。

## 动手实验

1. **实验一：追踪一个默认值的来龙去脉** — 在 `settings.py` 中定位 `DEFAULT_EMBEDDING_MODEL`，确认其默认值为 `"bge-m3"`；再用 Grep 在仓库中搜索该字段被哪些模块读取（`Settings.model_settings.DEFAULT_EMBEDDING_MODEL`），画出「默认值 → 消费方」的引用链。

2. **实验二：验证配置源优先级** — 设置环境变量（如 `HTTPX_DEFAULT_TIMEOUT`），再在 `basic_settings.yaml` 中给同一字段写一个不同值，用 Python 打印 `Settings.basic_settings.HTTPX_DEFAULT_TIMEOUT`，观察环境变量是否覆盖了 yaml，从而印证 `settings_customise_sources` 返回元组的顺序。

3. **实验三：触发自动重载** — 启动后读取一次 `Settings.basic_settings.log_verbose`，然后手动修改 `basic_settings.yaml` 把它改为 `True` 并保存，再次读取该属性，结合 `_lazy_load_key` 用 mtime 做缓存键的逻辑，解释为何无须重启即可拿到新值。

4. **实验四：生成配置模板并阅读注释** — 运行 `python -m chatchat.settings`（触发 `createl_all_templates`），查看生成的 `model_settings.yaml`，找到 `MODEL_PLATFORMS` 部分，确认其中的字段注释正是 `PlatformConfig` 字段下方的 docstring，验证 `YamlTemplate` 的注释采集机制。

> **下一章预告**：配置系统为我们交代了 `CHUNK_SIZE`、`TEXT_SPLITTER_NAME`、`PDF_OCR_THRESHOLD` 这些参数的默认值，但真正把一篇 PDF、一份 Markdown、一个网页变成可入库的文本块，靠的是文档加载器与切分器。第 3 章将解构 Langchain-Chatchat 的文档加载子系统，看它如何在 LangChain 的 `Document` 抽象之上，把各类异构文件统一收编进知识库的入口。
