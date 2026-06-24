# 第 9 章 向量嵌入与协变量 Embeddings & Covariates

第 8 章结束时,GraphRAG 已经把原始文档炼成了一套完整的知识结构:实体、关系、社区层级,以及每个社区的自然语言报告。这些都是「文本」——实体有 `description`,文本单元有 `text`,社区报告有 `full_content`。但要让这些文本能被检索召回,光有文字还不够。Local Search 要根据用户问题找到最相关的实体,Basic Search 要按语义找到最相关的文本片段,这些都依赖一件事:把文本变成向量,存进向量库,以便做近似最近邻(ANN)检索。这就是本章的第一个主角——`generate_text_embeddings` 工作流。

本章的第二个主角是协变量(covariates),在 GraphRAG 里具体表现为「声明 / 断言」(claims):由 LLM 从文本中抽取出来的结构化事实,诸如「公司 A 在 2022 年因围标被处罚」这类带主体、客体、类型、状态、时间的断言。它是一个默认关闭的可选增强步骤,理解它为什么可选、它补足了什么,正好能帮我们看清 GraphRAG 索引设计中「核心」与「增量」的边界。

## 哪些文本会被嵌入:三个 embedding name

GraphRAG 并不是把所有列都拿去嵌入,而是用一组明确命名的常量来圈定可嵌入字段。`config/embeddings.py` 只定义了三个名字:`entity_description_embedding = "entity_description"`、`community_full_content_embedding = "community_full_content"`、`text_unit_text_embedding = "text_unit_text"`,并把它们汇成 `all_embeddings` 集合与 `default_embeddings` 列表。换言之,默认情况下这三类文本都会被嵌入。

这三个名字到具体「嵌哪张表的哪一列」的映射,写在工作流文件 `generate_text_embeddings.py` 的 `EMBEDDING_FIELDS` 字典里。每个条目是一个 `EmbeddingFieldConfig` 数据类,记录 `name`、`table_name`、`embed_column`,以及一个可选的 `row_transform`:

- `text_unit_text` → 表 `text_units` 的 `text` 列,无变换。
- `entity_description` → 表 `entities` 的 `title_description` 列,并挂了一个 `transform_entity_row_for_embedding` 行变换。
- `community_full_content` → 表 `community_reports` 的 `full_content` 列,无变换。

实体那行的变换值得展开。实体表里其实没有 `title_description` 这一列,它是变换函数现造的:`transform_entity_row_for_embedding` 把 `title` 和 `description` 拼成 `f"{title}:{description}"` 写进 `row["title_description"]`。这样嵌入向量里既包含实体名字本身,又包含它的描述,检索时无论用户问的是实体名还是描述性语义都更容易命中——一个很朴素却很有效的细节。

是否嵌入某个字段,最终由配置决定。`generate_text_embeddings` 函数读取 `config.embed_text.names`,逐个 name 去 `EMBEDDING_FIELDS` 查映射。`names` 默认就是 `default_embeddings`(三者全开),但用户完全可以在配置里只保留 `text_unit_text`——比如只打算用 Basic Search 时,就没必要为实体和社区报告付嵌入成本。这种「按 name 选择性嵌入」是 GraphRAG 控制索引开销的重要旋钮。

## generate_text_embeddings 工作流的执行骨架

工作流入口 `run_workflow` 先通过 `config.get_embedding_model_config` 拿到嵌入模型配置,再用 `graphrag_llm` 的 `create_embedding` 创建模型实例。这里传入了 `cache=context.cache.child(config.embed_text.model_instance_name)`,默认实例名是 `"text_embedding"`(见 `EmbedTextDefaults.model_instance_name`)——嵌入结果会按这个名字分区缓存,重复运行时命中缓存可以省下重复调用嵌入 API 的钱。模型自带的 `tokenizer` 也被取出,后面分批时要靠它数 token。

真正干活的是 `generate_text_embeddings`。它对 `embedded_fields` 里的每个 name 做四件事:

1. 检查源表是否存在。如果配置要嵌入某字段但 `table_provider` 里没有对应表,只打一条 warning 并 `continue` 跳过,而不是报错中断。这让流水线对「只跑了部分工作流」的中间状态保持健壮。
2. 创建向量库实例:`create_vector_store(config.vector_store, config.vector_store.index_schema[field_config.name])`,然后 `vector_store.connect()`。注意第二个参数——每个 embedding name 在配置里都有一份独立的 `index_schema`,意味着三类文本各写各的索引(各自的 `index_name`)。
3. 用 `AsyncExitStack` 打开输入表(流式读取,带上行变换),如果 `config.snapshots.embeddings` 打开,还会额外打开一张 `embeddings.{name}` 输出表用于落盘快照。
4. 调用 `embed_text` 操作完成嵌入与写入,返回处理行数。

整个工作流以「表」为单位流式处理,不会一次性把全表读进内存,这是 3.x 版本流式 Table 抽象带来的内存友好特性。

## embed_text 操作:缓冲、并发与 token 限制

`index/operations/embed_text/` 目录里有两个核心文件:`embed_text.py`(对接流式表与向量库)和 `run_embed_text.py`(纯粹的嵌入执行逻辑)。

`embed_text` 函数采用「缓冲—刷写」模式。它先 `vector_store.create_index()` 建好索引,再异步遍历输入表,把每行的 `id` 和待嵌入文本塞进 `buffer`。缓冲区大小不是 `batch_size`,而是 `flush_size = batch_size * num_threads`——这个设计很关键:每次刷写时要产出足够多的 API 批次,才能把 `num_threads` 个并发槽全部喂满。其中 `batch_size` 默认 `16`,`num_threads` 取 `config.concurrent_requests`。当缓冲满或遍历结束,调用 `_flush_embedding_buffer`。

`_flush_embedding_buffer` 把缓冲里的文本和 id 拆成两个列表,交给 `run_embed_text` 得到向量,再为每个非空向量构造 `VectorStoreDocument`(只含 `id` 和 `vector`),批量 `vector_store.load_documents(documents)`。`None` 向量会被跳过并计数告警。如果开了快照,还会把 `{"id", "embedding"}` 逐行写进输出表。

真正的 token 处理在 `run_embed_text.py`,它分三步精细化处理:

- **切片(_prepare_embed_texts)**:每条输入文本先用 `split_text_on_tokens` 按 `batch_max_tokens`(默认 `8191`)切成若干 snippet,`chunk_overlap=100`。为什么要切?因为单条文本(尤其社区报告 `full_content`)可能超过嵌入模型的 token 上限。函数同时记录每条原始文本被切成了几片(`sizes`)。
- **组批(_create_text_batches)**:把所有 snippet 重新打包成批,单批受双重约束——不超过 `max_batch_size` 条,且累计 token 不超过 `max_batch_tokens`。代码注释直接引用了 Azure OpenAI 的限制:每请求最多约 8191 token。
- **并发执行(_execute)**:用 `asyncio.Semaphore(num_threads)` 限流,为每个批次起一个协程,`asyncio.gather` 汇总,期间用 `progress_ticker` 报进度。

最精妙的是 `_reconstitute_embeddings`:一条被切成多片的长文本会得到多个 snippet 向量,函数把它们 `np.average` 求均值再 L2 归一化(`average / np.linalg.norm(average)`),还原成「一条原文 = 一个向量」。被切成 0 片的(空文本)返回 `None`,1 片的直接取用,多片的才走平均归一。这就是 GraphRAG 处理超长文本嵌入的标准做法——平均池化而非截断,尽量保留全文语义。

## graphrag-vectors 包:向量库抽象契约

GraphRAG 把向量库独立成了 `packages/graphrag-vectors` 这个单独的包,核心是 `vector_store.py` 里的抽象基类 `VectorStore(ABC)`。这是一份清晰的数据访问契约,值得逐条看。

基类构造函数统一了字段约定:`id_field`(默认 `"id"`)、`vector_field`(默认 `"vector"`)、`create_date_field`、`update_date_field`、`vector_size`(默认 `3072`,正是 OpenAI `text-embedding-3-large` 的维度),以及一个 `fields` 字典声明附加列及其类型。它还内置了时间戳处理:把用户声明为 `"date"` 类型的字段「炸开」(explode)成可过滤的组件字段,并自动注册内建时间戳字段——这让向量库不止能按向量检索,还能按创建/更新时间过滤。

抽象方法划定了所有后端必须实现的能力:`connect`、`create_index`、`load_documents`、`similarity_search_by_vector`、`search_by_id`、`count`、`remove`、`update`。基类还提供了两个默认实现:`insert` 单条插入直接委托给 `load_documents`;`similarity_search_by_text` 先用传入的 `text_embedder` 把查询文本嵌成向量,再转调 `similarity_search_by_vector`——这正是检索阶段「文本进、文档出」的入口。

两个数据类承载数据流:`VectorStoreDocument` 是写入单元(`id` / `vector` / `data` / 时间戳);`VectorStoreSearchResult` 是检索返回单元,包成 `document` + `score`,且文档注释明确 `score` 在 -1 到 1 之间、越大越相似。这套契约把「向量怎么存、怎么查」标准化,上层的 `embed_text` 和后续各检索器只认契约、不认后端。

## 默认后端 LanceDB 与可插拔工厂

GraphRAG 默认用 LanceDB,实现见 `lancedb.py` 的 `LanceDBVectorStore`。它构造时接收 `db_uri`(默认 `"lancedb"`),`connect` 就是 `lancedb.connect(self.db_uri)`——一个嵌入式、基于本地文件的向量库,无需独立服务,开箱即用,非常适合本地索引场景 [[LanceDB 文档]](https://lancedb.github.io/lancedb/)。

它的几个实现细节体现了 LanceDB 用 Arrow/Parquet 的特点:`create_index` 先用一个全 0 的 dummy 向量和占位列建好 schema(因为 Arrow 表必须先有 schema),建完 `IVF_FLAT` 索引后再把 dummy 行删掉。`load_documents` 把整批文档拼成一张 `pa.table` 一次性 `add`,向量列用 `FixedSizeListArray`。检索时 `similarity_search_by_vector` 走 `.search(...).limit(k)`,并把 LanceDB 返回的 `_distance` 转成相似度 `score = 1 - abs(_distance)`。它还实现了 `_compile_filter`,把统一的 `FilterExpr`(`Condition` / `AndExpr` / `OrExpr` / `NotExpr`)编译成 LanceDB 的 SQL WHERE 子句,支持 `prefilter=True` 的预过滤检索。

后端可替换由 `vector_store_factory.py` 保证。`VectorStoreType` 枚举列了三种:`LanceDB`、`AzureAISearch`、`CosmosDB`。`create_vector_store` 先看 `config.type` 是否已注册,没注册就按 `match` 惰性导入对应实现并 `register_vector_store`——惰性加载意味着不用 Azure 的人不必装 Azure 依赖。`AzureAISearchVectorStore`(`azure_ai_search.py`)是托管的云端搜索后端,还额外支持混合检索;`CosmosDBVectorStore` 则对接 Azure Cosmos DB。工厂最后把基础 `VectorStoreConfig` 和具体 `IndexSchema` 两份配置合并成一个 dict 传给初始化器。用户也能用 `register_vector_store` 注册自定义后端——这是典型的开放扩展点。

## extract_covariates:从文本抽取结构化声明

协变量工作流入口 `extract_covariates.py` 的 `run_workflow` 第一行判断就是 `if config.extract_claims.enabled:`,而 `ExtractClaimsDefaults.enabled = False`——这是全书少有的「默认关闭」工作流。关闭时它什么都不做直接返回。开启后,它从 `DataReader` 读出全部 `text_units`,创建 completion 模型,解析 prompt,调用 `extract_covariates` 抽取,最后把结果 DataFrame 写成 `covariates` 表。

抽取的语义在 prompt 模板 `EXTRACT_CLAIMS_PROMPT`(`prompts/index/extract_claims.py`)里定义得非常具体。它要求 LLM 分两步:先按 entity specification 抽出命名实体,再为每个实体抽取与「claim description」匹配的声明。每条声明要产出 8 个字段:Subject、Object(未知用 `**NONE**`)、Claim Type、Claim Status(`TRUE`/`FALSE`/`SUSPECTED`)、Claim Description、起止日期(ISO-8601)、Source Text(原文引语列表)。输出格式是用 `<|>` 分隔字段、`##` 分隔多条声明的元组串,结尾打 `<|COMPLETE|>`。prompt 里还自带了「公司围标 / 个人腐败嫌疑」两个示例做 few-shot。

底层执行器是 `ClaimExtractor`(`claim_extractor.py`)。它逐文本调 `_process_document`:把文本、`claim_description`、`entity_specs` 填进 prompt 发给模型,然后进入 gleaning(拾遗)循环——若 `max_gleanings > 0`(默认 `1`),就反复追加 `CONTINUE_PROMPT` 让模型补抽漏掉的实体,中途还用 `LOOP_PROMPT` 问模型「是否还有(Y/N)」,模型答非 `Y` 就提前退出。这套机制和第 5 章图谱抽取的 gleaning 一脉相承:用多轮对话对抗单轮抽取的召回不足。

`_parse_claim_tuples` 负责把模型返回的元组串解析回字段:去掉 `<|COMPLETE|>`,按 `##` 切成多条,每条按 `<|>` 切成 8 列,依次映射到 `subject_id`/`object_id`/`type`/`status`/`start_date`/`end_date`/`description`/`source_text`。这些被包成 `typing.py` 里的 `Covariate` 数据类。`_clean_claim` 还会用 `resolved_entities` 做一次实体对齐,把声明里的主客体替换成已解析的规范实体名。最后 `extract_covariates`(operation)给每行补上 `uuid4` 的 `id` 和递增的 `human_readable_id`,并裁剪到 `COVARIATES_FINAL_COLUMNS` 规定的列集(含 `covariate_type`、`subject_id`、`object_id`、`status`、`start_date`、`end_date`、`source_text`、`text_unit_id` 等),与 text unit 通过 `text_unit_id` 关联。

## 设计动机:核心是嵌入,增量是声明

为什么嵌入是核心、声明是可选?把两者放在一起看就清楚了。

嵌入是几乎所有检索模式的地基:Local Search 靠实体描述向量找相关实体,Basic Search 靠文本单元向量找相关片段,社区报告向量服务于按语义定位社区。没有这一步,GraphRAG 就退化成只能做结构遍历而不能做语义召回。所以它默认全开,并在工程上下足功夫——缓存、流式、并发限流、超长文本平均池化、可换后端——目标是「又快又省又稳地把文本变向量」。

声明则是事实性增强。图谱描述的是「谁和谁有关系」,声明补充的是「关于某实体有哪些可被核验的断言及其状态/时间」。它对调查类、合规类、事实核查类问题特别有价值,但代价不小:要对每个 text unit 多跑一轮(还带 gleaning 多轮)LLM 调用,token 成本可观,且并非所有应用都需要。于是 GraphRAG 把它设成 `enabled = False` 的选项,让用户按需付费——这正体现了 GraphRAG 索引设计一以贯之的取舍哲学:把昂贵且非普适的能力做成可关闭的增量步骤,把普适的基础能力做成默认且优化到位的核心步骤。

## 本章小结

- GraphRAG 用三个命名常量 `entity_description`、`community_full_content`、`text_unit_text` 圈定可嵌入字段,`config/embeddings.py` 的 `default_embeddings` 默认三者全开。
- `EMBEDDING_FIELDS` 把每个 name 映射到具体表列;实体走 `transform_entity_row_for_embedding` 把 `title` 和 `description` 拼成 `title_description` 再嵌入。
- 嵌入对象由 `config.embed_text.names` 控制,可只嵌入部分字段以节省成本;源表缺失时跳过而非报错。
- `embed_text` 用 `flush_size = batch_size * num_threads` 的缓冲—刷写模式,把缓冲喂满并发槽;默认 `batch_size=16`、`batch_max_tokens=8191`。
- `run_embed_text` 先按 token 切片(overlap 100)、再按条数与 token 双约束组批、用 `Semaphore` 并发,超长文本走 `np.average` + L2 归一化的平均池化还原成单向量。
- `graphrag-vectors` 包以 `VectorStore(ABC)` 定义统一契约(`load_documents`、`similarity_search_by_vector` 等),`VectorStoreDocument`/`VectorStoreSearchResult` 为读写单元。
- 默认后端是嵌入式的 LanceDB(`IVF_FLAT` 索引、Arrow 批写、`_distance` 转 `score`),工厂支持惰性加载 Azure AI Search、CosmosDB,并可 `register_vector_store` 自定义后端。
- 协变量抽取默认关闭(`enabled=False`);开启后 `ClaimExtractor` 用元组格式抽取主体/客体/类型/状态/起止日期/描述/原文,并支持 gleaning 多轮拾遗。
- 声明结果裁剪到 `COVARIATES_FINAL_COLUMNS`,通过 `text_unit_id` 与文本单元关联,补充图谱所缺的事实性断言。
- 设计上嵌入是普适地基所以默认开且重度优化,声明是昂贵的事实增强所以做成可选,体现 GraphRAG「核心默认、增量可关」的取舍。

## 动手实验

1. **实验一:验证选择性嵌入** — 把配置里的 `embed_text.names` 改成只含 `text_unit_text`,重跑 `generate_text_embeddings` 工作流,观察日志中 `Embedding the following fields` 只剩一项、向量库里只生成 text unit 索引,确认其余两类被跳过。
2. **实验二:追踪超长文本的平均池化** — 在 `run_embed_text._reconstitute_embeddings` 里打印每条原文的 `size`,构造一条远超 `batch_max_tokens` 的文本,验证它被切成多片、最终被 `np.average` 平均并归一化成一个向量,而短文本 `size==1` 直接取用。
3. **实验三:替换向量库后端** — 阅读 `vector_store_factory.create_vector_store` 的 `match` 分支,把 `config.vector_store.type` 改为 `azure_ai_search`,观察工厂如何惰性导入 `AzureAISearchVectorStore`;再尝试用 `register_vector_store` 注册一个打印日志的自定义 `VectorStore` 子类。
4. **实验四:开启并解析声明抽取** — 把 `extract_claims.enabled` 设为 `true`,跑 `extract_covariates`,打开生成的 `covariates` 表,对照 `_parse_claim_tuples` 的字段顺序检查 `subject_id`/`object_id`/`type`/`status`/`start_date`/`end_date`/`description`/`source_text` 是否对齐,并尝试调大 `max_gleanings` 观察召回变化与 token 消耗。

> **下一章预告**:我们已经逐一拆完了 GraphRAG 的所有索引步骤——切分、抽取、摘要、社区检测、报告、嵌入、声明。但这些工作流是被谁、按什么顺序、如何串起来跑的?第 10 章《索引流水线编排》将深入 `index/run`、`run_pipeline` 与 `workflows/factory`,看 GraphRAG 如何把一组松耦合的工作流编排成一条可恢复、可增量、可观测的完整索引流水线。
