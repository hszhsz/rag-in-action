# 第 14 章 LLM、缓存、存储与工程原则收尾

前十三章,我们沿着 GraphRAG 的数据流把它从外到内拆了一整遍:从八包 uv-workspace monorepo 与 `GraphRagConfig`,到输入加载与文本切分,到用 LLM 把每个 text unit 抽成实体与关系、合并描述、终化与剪枝图谱,到 Leiden 层次社区检测与自底向上的社区报告,到向量嵌入与协变量,到 `PipelineFactory` 把这一长串 workflow 编排成可观测的索引流水线;再到查询侧的 local / global / drift / basic 四种搜索模式。但有一层始终在所有这些之下默默支撑:每一次 LLM 调用、每一次缓存命中、每一次 parquet 落盘,都依赖三个独立的基础设施子包——`graphrag-llm`、`graphrag-cache`、`graphrag-storage`。它们不出现在数据流的任何一个箭头上,却是整条流水线能跑起来的地基。

本章分三半。先解构这三个基础设施包,看 GraphRAG 如何用统一工厂 + 可组合中间件把"调模型、存数据、缓存结果"这些横切关注点收口;再看 `prompt_tune` 如何用 LLM 反过来给 LLM 生成领域定制的 prompt;最后退到最高处,把 GraphRAG 自己的设计取向提炼成几条透镜,并跨越 RAGFlow、AnythingLLM、LlamaIndex、Quivr、Langchain-Chatchat、LightRAG、GraphRAG 七套系统,归纳那些与语言、框架都无关的可迁移工程原则,为整部《RAG in Action》收尾。

## `graphrag-llm`:LiteLLM 单一底座 + 可组合中间件管线

GraphRAG 没有为每家模型供应商写一套适配类,而是把所有 LLM 与 embedding 调用都收敛到 `graphrag-llm` 一个包,底层统一压在 LiteLLM 上。契约由 `completion/completion.py` 的 `LLMCompletion`(ABC)与 `embedding/embedding.py` 的 `LLMEmbedding`(ABC)定义:前者声明 `completion(/, **kwargs)` 与 `async completion_async(...)`,返回 `LLMCompletionResponse` 或流式的 `Iterator[LLMCompletionChunk]`,并自带 `completion_thread_pool`/`completion_batch` 等批量方法;后者对称地给出 `embedding`/`embedding_async`。真正干活的实现是 `lite_llm_completion.py` 的 `LiteLLMCompletion` 与 `lite_llm_embedding.py` 的 `LiteLLMEmbedding`,内部直接调 `litellm.completion`/`acompletion`、`litellm.embedding`/`aembedding`,把 `model` 拼成 `f"{provider}/{model}"` 这一种 LiteLLM 认得的形式,Azure 托管身份则注入 `azure_ad_token_provider`。这与 LightRAG 用一个函数签名收口、AnythingLLM 用 `BaseLLMProvider` 类契约是同族思路的第三种答案:**用一个底层 SDK(LiteLLM)吃掉所有供应商差异,自己只在其上包一层薄契约**。LiteLLM 本身就是为「用统一 OpenAI 格式调上百家模型」而生的库[[LiteLLM]](https://docs.litellm.ai/)。

这个包最值得学的设计是 `middleware/with_middleware_pipeline.py` 的**中间件管线**。限流、重试、缓存、metrics 这些横切关注点没有散落进每个 provider,而是被组织成一串可选的中间件,在构造 `LiteLLMCompletion` 时由 `with_middleware_pipeline(...)` 从外到内层层包裹。文档描述的逻辑顺序是 request_count → cache → retries → rate_limiting → metrics → errors_for_testing,其中一个关键细节是**重试在限流的外层**:一次被重试的请求会重新去排限流的队,而不是绕过限流直接重发——这避免了重试本身把下游打爆。`with_cache.py` 对 streaming 与 mock 调用不缓存,命中时直接返回反序列化的 response 并把 `cached_responses` 计进 metrics;`create_cache_key.py` 里硬编码了 `_CACHE_VERSION=4`(注释记录了 fnllm v2 → litellm v3 → graphrag-llm v4 的演进),并在算 key 时显式排除 `api_key`、`timeout`、`stream` 等不影响语义的参数,保证相同语义的调用命中同一缓存。

实例化由 `completion/completion_factory.py` 的 `create_completion(model_config, ...)` 负责:它按 `model_config.type`(`LLMProviderType.LiteLLM` 或 `MockLLM`)分发,并按需把 rate_limiter、retrier、metrics_store 注入进去。限流实现是 `rate_limit/sliding_window_rate_limiter.py` 的 `SlidingWindowRateLimiter`——用请求队列与 token 队列两个滑动窗口、`period_in_seconds` 默认 `60`,还用 `_stagger = period / requests_per_period` 做错峰,避免请求在窗口边界扎堆。重试是 `retry/exponential_retry.py` 的 `ExponentialRetry`,默认 `max_retries=7`(注释点明 2^7=128s 的上限)、`base_delay=2.0`、`jitter=True`。`MockLLMCompletion` 则循环吐 `model_config.mock_responses`,是测试与离线开发的基石——这和 LlamaIndex 的 `MockLLM`、Quivr 的兜底是同一类"让系统在没有真实模型时也能跑通管线"的设计。

## `graphrag-cache` 与 `graphrag-storage`:三层解耦的持久化地基

缓存与存储是两个独立的小包,但它们的关系是分层的:**缓存复用存储**。`graphrag-storage` 的 `storage.py` 定义 `Storage`(ABC),契约是一组以字符串为 key 的方法——`async get/set/has/delete/clear`、同步的 `find(file_pattern)` 与 `keys()`、`child(name)` 派生子存储、`get_creation_date(key)`。后端由 `storage_type.py` 的 `StorageType` 枚举给出:`File`、`Memory`、`AzureBlob`、`AzureCosmos`,经 `storage_factory.py` 的 `create_storage(config)` 按 `config.type` 分发,命中不了就报错并列出已注册类型。默认的 `FileStorage` 把 `base_dir` resolve 成绝对路径并自动 mkdir,`child` 直接返回 `FileStorage(base_dir/name)`;`MemoryStorage` 干脆继承 `FileStorage` 只把 base_dir 置空——用继承复用了绝大部分文件逻辑。包里还有一个 `tables/` 子模块,用 `ParquetTable`/`CsvTable`/`CosmosTable` 与 `TableProviderFactory` 专门承载前面各章反复出现的 parquet 产物表。

`graphrag-cache` 则把"缓存"建在"存储"之上。`cache.py` 的 `Cache`(ABC)契约是全异步的 `get/set/has/delete/clear` 加 `child(name)`;`cache_type.py` 给出 `Json`/`Memory`/`Noop` 三种;`json_cache.py` 的 `JsonCache` 不自己管文件,而是**委托一个 `graphrag_storage.Storage`**:`set` 时存 `{"result": value, **debug_data}`、`get` 时读回 `["result"]`,遇到 `JSONDecodeError`/`UnicodeDecodeError` 这类坏数据就删键返 None——脏缓存自愈而非崩溃。key 由 `cache_key.py` 的 `create_cache_key` = `graphrag_common.hasher.hash_data` 算出。这一层 `Cache → Storage` 的委托,意味着换缓存后端和换存储后端是两个正交的决定:把存储从本地文件换成 Azure Blob,JsonCache 不用改一行。**把缓存、存储、表三件事拆成三层各自收紧的契约,是 GraphRAG 数据安全感的来源——任何一层的后端都能独立替换。**

## `prompt_tune`:用 LLM 给 LLM 写 prompt

GraphRAG 的图谱抽取质量高度依赖 prompt 里的领域设定:要抽哪些实体类型、用什么口吻的 persona、给几个什么样的示例。手写这些既费力又难跨领域迁移,于是 `api/prompt_tune.py` 的 `generate_indexing_prompts(...)` 把这件事自动化了。它的流程是一条"采样 → 探测 → 生成 → 组装"的链:先用 `prompt_tune/loader/input.py` 的 `load_docs_in_chunks` 从用户真实语料里采样文本块(选样方式有 `TOP`/`RANDOM`/`AUTO`,`AUTO` 会嵌入 `n_subset_max` 默认 300 个样本再取离质心最近的 `k` 默认 15 个),然后让 LLM 依次 `generate_domain`(推断领域)、`detect_language`(探测语言)、`generate_persona`(生成抽取者人设)、`generate_entity_types`(发现该领域该抽哪些实体类型)、`generate_entity_relationship_examples`(造少样本示例),最后用 `create_extract_graph_prompt` 等把这些产物组装成三份定制 prompt 文件:`extract_graph.txt`、`summarize_descriptions.txt`、`community_report_graph.txt`。

这是一个很妙的自举:**系统用它即将运行的同一个 LLM,先去理解用户的语料,再为这份语料量身定做后续抽取要用的 prompt。** 默认值都收在 `prompt_tune/defaults.py`(`K=15`、`MAX_TOKEN_COUNT=2000`、`N_SUBSET_MAX=300`、`PROMPT_TUNING_MODEL_ID="default_completion_model"`),组装 prompt 时还会用 tokenizer 把示例截到 `max_token_count` 以内、并对 `{`/`}` 转义以防后续 `str.format` 崩溃——又是一处"把现实工程约束想透了"的细节。

## 一、GraphRAG 淬炼出的工程取向

### 取向一:索引是一条可注册、可重排的 workflow 流水线

第 10 章读到 `PipelineFactory` 把 `load_input_documents`→`create_base_text_units`→`extract_graph`→`finalize_graph`→`create_communities`→`create_community_reports`→`generate_text_embeddings` 这一长串步骤,每个都实现成统一签名的 `run_workflow(config, context)`,再由不同的 `IndexingMethod` 取不同的 workflow 名序列拼成 pipeline。**把"一串固定步骤"抽象成可注册、可重排、可观测的数据,新增一个 workflow 只需 register 一下、插进序列里**,这正是"编排即数据"在索引侧的彻底落地,与 Quivr 用 `NodeConfig` 反射建图、LlamaIndex 用 `@step` 自动连边殊途同归。

### 取向二:图是知识的骨架,检索是图上的多种投影

GraphRAG 与朴素向量 RAG 最根本的分野,是它把知识组织成实体-关系图与层次社区,再围绕这张图给出四种检索投影:local 走具体实体的混合上下文、global 用社区报告做 map-reduce、drift 先全局起步再局部深挖、basic 退化成纯向量基线。**同一份图谱产物,被四种 search 引擎从四个角度读取,覆盖从细节到概览到多跳推理到基线的不同问答需求。** 这与 LightRAG 的双层检索是同源思路,但 GraphRAG 把投影的种类做得更丰富,且每种都有独立的 context_builder 与 prompt。

### 取向三:横切关注点收口进中间件与工厂

限流、重试、缓存、metrics 没有散落进每个模型调用点,而是被 `with_middleware_pipeline` 串成一条可组合管线;存储、缓存、向量库、LLM 都各有一个 `Factory[T]` + `StrEnum` type + `register_*`/`create_*` 的同构注册机制。**横切逻辑下沉一层,上层就少一份重复与遗漏**——`extract_graph`、`summarize_communities` 这些业务 workflow 完全不必关心自己调的模型有没有重试、结果有没有缓存。这与 LightRAG 把并发收口进 `shared_storage.py`、Chatchat 把线程安全收口进 `ThreadSafeObject` 是同一种智慧。

### 取向四:数据分层,权威产物与派生索引分离

GraphRAG 的索引产物是一组 parquet 表(entities、relationships、communities、community_reports、text_units),向量库里的嵌入只是这些表里文本字段的派生物。查询侧通过 `indexer_adapters` 从 parquet 读回数据再构造内存对象——**parquet 表是权威源,向量索引可以从它重算**。这正是 LightRAG 在第六部补上的那条透镜在 GraphRAG 里的回响:想清楚什么是权威源、什么是可重建的派生物,错误就从"灾难"降级成"重跑一次"。

## 二、全书结语:七套系统淬炼出的可迁移原则

至此,《RAG in Action》解构了七个开源 RAG 系统。把它们放在一起,形态差异巨大——

| 系统 | 语言 | 形态 | 最鲜明的一招 |
|---|---|---|---|
| RAGFlow | Python | 深度文档理解的 RAG 引擎 | DeepDoc 视觉版面分析 + 模板化分块 |
| AnythingLLM | JavaScript | 三进程的桌面/服务端产品 | provider 插拔 + Agent Flows 可视化编排 |
| LlamaIndex | Python | 可插拔的 RAG 框架 | 子包化 provider + Workflow 事件驱动 |
| Quivr | Python | 配置拼装 LangGraph 的薄库 | 配置驱动拓扑,换 `NodeConfig` 即换流程 |
| Langchain-Chatchat | Python | 面向中文的离线 RAG+Agent 应用 | 统一抽象吃掉生态碎片 + 中文开箱即用 |
| LightRAG | Python | 以知识图谱为核心的 graph-based RAG | 实体/关系双层检索 + 权威源/派生物分层 |
| GraphRAG | Python | 微软出品的 graph-based RAG 引擎 | 层次社区 + 四种检索投影 + workflow 流水线 |

七套系统,四种语言,七种产品形态——从重型文档理解引擎,到桌面产品,到框架,到薄库,到离线中文应用,到两套各有侧重的图谱 RAG。但当你把七套源码逐块拆开对照,会浮现出几条与语言、框架都无关的可迁移工程原则。它们不是套模板,而是从读过的代码里长出来的。

**原则一:默认值是产品的隐性主张。** 不确定时倒向"能直接跑"和"更安全"的一侧,但把真正需要用户表达意图的地方显式留空。GraphRAG 的 `init` 生成一份带保守默认的 `settings.yaml`、却要求用户填 API key;LightRAG 默认装本地全栈、却把 `llm_model_func`/`embedding_func` 设为仅有的两个必填项;Chatchat 的中文默认模型与 `init` 一键部署、Quivr 的 `default_llm()`、LlamaIndex 的 `MockLLM` 兜底、RAGFlow 各处保守的 `DEFAULT_*` 常量——七套系统都在用默认值替没有时间做配置的用户做决定,区别只在"兜底"还是"必填"。

**原则二:约束只能收紧,越界即报错。** 合并配置、对齐能力、加载后端时结果只会更严。GraphRAG 的 `ModelConfig` 校验"Azure 必须有 api_base、非 api_key 认证时禁带 api_key"、`storage_factory` 对未注册类型 fail-fast;LightRAG 的 `check_storage_env_vars` 缺变量即 `ValueError`;Quivr 把 token 上限下调到模型真实上限、低于下限直接抛错;Chatchat 对未知 `agent_type` 抛错;RAGFlow、AnythingLLM 在配置层同样把不合法状态在生效前拍死。

**原则三:统一的最小数据契约是可组合性的地基。** 让流水线尽量只流动一种东西。GraphRAG 把每个索引步骤收敛成 `run_workflow(config, context)`、把存储/缓存/LLM 各收敛成一个 ABC + 工厂、把表统一成 parquet;LightRAG 把持久化收敛成四个 ABC、把 LLM 收敛成一个函数签名;Chatchat 的所有 loader 都吐 langchain `Document`;Quivr 的 LangGraph 节点签名都是 `(state) -> state`;LlamaIndex 的 `TransformComponent` node 进 node 出;AnythingLLM 用 `BaseVectorDatabaseProvider` 统一数十家库。**先定义收得足够紧的最小契约,再让一切实现去满足它。**

**原则四:横切关注点要织进底层,而非散落各处。** GraphRAG 把限流/重试/缓存/metrics 收口进 `with_middleware_pipeline` 一条中间件管线;LightRAG 把单/多进程并发差异收口进 `shared_storage.py`;Chatchat 把 FAISS 线程安全收口进 `ThreadSafeObject`/`CachePool`;Quivr 把流式追踪收口进 `answer_astream`;LlamaIndex 把 instrumentation 沉到底层;RAGFlow 把进度回写与限流收进 task executor。**横切逻辑下沉一层,上层就少一份重复与遗漏。**

**原则五:把编排从代码里抽出来变成数据。** 当一个系统的"流程"开始频繁变化,最可持续的做法是把流程本身变成可配置的数据。GraphRAG 用 `PipelineFactory` 按 `IndexingMethod` 取 workflow 名序列建索引流水线、用 `register_*` 注册各类后端;Quivr 的 `_build_workflow` 从 `NodeConfig` 反射建图、RAGFlow 的 `agent/canvas` DSL、AnythingLLM 的 Agent Flows、LlamaIndex 的 `@step` 自动连边、Chatchat 的 `agent_type` 与 `MODEL_PLATFORMS` 配置分发、LightRAG 用字符串类名声明存储后端——七套系统不约而同地走向"流程即数据",只是抽象程度与落地形态各异。

**原则六:复用生态要快,但要在边界留一层契约。** 七套系统没有一个从零造轮子,它们都站在 LangChain / LlamaIndex / LiteLLM / 各家模型与向量库 / NetworkX / graspologic 的肩上。但走得快的代价是上游一变你就碎一地,所以每个成熟系统都在自己与上游之间留了一层收紧的契约——GraphRAG 用 LiteLLM 单一底座收口所有供应商、再用 `LLMCompletion` ABC 与中间件包一层;LightRAG 用一个函数签名收口 LLM、用四个 ABC 收口存储;Chatchat 的 `KBService` 与 `regist_tool` monkeypatch、Quivr 的 `LLMEndpoint.from_config`、LlamaIndex 的子包化、AnythingLLM 的 provider 插拔。**复用生态可以走得很快,但要在自己和上游之间留一层收紧的契约,否则上游一变你就碎一地。**

**原则七:把数据分层,让错误不致命。** 想清楚"什么是权威源、什么是可重算的派生物",并保证源只读、派生可重建,错误就从"灾难"降级成"重跑一次"。GraphRAG 的 parquet 表是权威源、向量索引是可从它重算的派生物;LightRAG 把知识图谱 + `text_chunks` KV 设为权威源、三个向量库为派生物,`rebuild_vdb.py` 据此重建;RAPTOR 的树状摘要是 chunk 的派生、RAGFlow 的索引是解析结果的派生、LlamaIndex 的 `VectorStoreIndex` 可从 docstore 重建、Chatchat 的向量库与 SQLite 双写中 SQLite 是源。**源只读、派生可重建,是一套系统在面对换模型、写失败、schema 演进时仍然从容的底气。**

## 本章小结

- GraphRAG 把 LLM/embedding 调用收敛进 `graphrag-llm`,底层统一压在 LiteLLM 上,用 `LLMCompletion`/`LLMEmbedding` 两个 ABC + `create_completion`/`create_embedding` 工厂按 `model_config.type` 分发。
- 限流、重试、缓存、metrics 被组织成 `with_middleware_pipeline` 的可组合中间件,且重试在限流外层,使被重试的请求重新排队限流而非绕过。
- `SlidingWindowRateLimiter` 用请求/token 双滑动窗口 + `_stagger` 错峰;`ExponentialRetry` 默认 `max_retries=7`、`base_delay=2.0`、带 jitter;`MockLLMCompletion` 支撑离线与测试。
- `graphrag-storage` 的 `Storage` ABC(get/set/has/delete/find/child)经 `create_storage` 分发 File/Memory/AzureBlob/AzureCosmos,`tables/` 子模块承载 parquet 产物表。
- `graphrag-cache` 的 `Cache` 建在 `Storage` 之上:`JsonCache` 委托一个 storage 读写,遇坏数据删键自愈;缓存 key 用 `hash_data`,LLM 层另有 `_CACHE_VERSION=4` 做失效控制——换缓存后端与换存储后端是正交决定。
- `prompt_tune` 用即将运行的同一个 LLM 先理解用户语料(采样 → domain/language/persona/entity_types → 示例),再组装出 `extract_graph.txt` 等三份领域定制 prompt,是一次优雅的自举。
- GraphRAG 的工程取向:索引是可注册可重排的 workflow 流水线、图是知识骨架而检索是其多种投影、横切关注点收口进中间件与工厂、parquet 权威产物与向量派生索引分层。
- 跨七部书的可迁移原则:默认值是产品主张、约束只能收紧、统一最小数据契约、横切关注点下沉、编排即数据、复用生态留契约、把数据分层让错误不致命。

## 动手实验

1. **实验一:画出一次 LLM 调用穿过的中间件层** — 打开 `graphrag-llm` 的 `middleware/with_middleware_pipeline.py`,按它包裹的顺序画出 request_count → cache → retries → rate_limiting → metrics 的洋葱图;再到 `with_cache.py` 找出"streaming 与 mock 不缓存"那几行,解释为什么这两种情况必须跳过缓存;最后说明"重试在限流外层"如何防止重试把下游打爆。

2. **实验二:验证缓存与存储的正交性** — 在 `graphrag-cache/json_cache.py` 里定位 `JsonCache` 如何委托 `graphrag_storage.Storage` 读写,再到 `cache_factory.py`/`storage_factory.py` 看两个工厂如何各自按 `type` 分发。动手把缓存后端与存储后端分别换一种(如 Json+Memory、Json+File),确认 `JsonCache` 一行不改即可工作,体会三层解耦的价值。

3. **实验三:跑通一次 prompt 自动调优** — 用一份你自己的小语料运行 `graphrag prompt-tune`(或直接调 `api/prompt_tune.py::generate_indexing_prompts`),把 `selection_method` 在 `RANDOM` 与 `AUTO` 间切换,对比生成的 `extract_graph.txt` 里 persona 与 entity_types 的差异;再把 `--domain`/`--language` 手动指定,观察哪些生成步骤被跳过,理解"自举"链上每一环的作用。

4. **实验四:为七条原则各找一处七部书的铁证** — 回到本书七个项目目录,为结语的七条可迁移原则各定位一处确切的文件/常量/分支(例如:原则三找 GraphRAG 的 `run_workflow` 统一签名与 LightRAG 四个 ABC;原则四找 GraphRAG `with_middleware_pipeline` 与 Chatchat `ThreadSafeObject`;原则七找 GraphRAG 的 parquet→向量派生与 LightRAG `rebuild_vdb.py`)。把"原则—证据"列成一张跨七项目对照表,作为你解构下一个 RAG/Agent 系统时的 checklist。

> **全书结语**:从第一部 RAGFlow 的深度文档理解,到第二部 AnythingLLM 的三进程产品,到第三部 LlamaIndex 的可插拔框架,到第四部 Quivr 用配置拼装 LangGraph 的薄库,到第五部 Langchain-Chatchat 面向中文的离线应用,到第六部 LightRAG 以知识图谱为核心的轻量 graph-based RAG,再到这第七部 GraphRAG 用层次社区与四种检索投影把图谱 RAG 做到工业级——七种系统、七种取舍,一以贯之的方法只有一个:**不停在"一般怎么做",而是钻进源码读懂"到底怎么实现"**。当你能把任意一个 RAG 引擎的每一块积木、每一处咬合都拆开看过——它的默认值替你做了什么决定、它的契约把什么收紧、它的横切逻辑沉在哪里、它把哪些数据当作可重算的派生物——你也就握住了按需拼出、甚至从零写出属于自己的 RAG/Agent 系统的能力。七部书到此结束,但你的解构之路,才刚刚开始。愿这套透镜,陪你走向下一个值得拆开的代码库。
