# 第 13 章 DRIFT 与 Basic Search

第 11 章我们看到 Local Search 如何以实体为锚点向下深挖、把图谱邻域和原文块拼成局部上下文；第 12 章则看到 Global Search 如何 map-reduce 全量社区报告、回答"整个语料在讲什么"这类宏观问题。两者各有所长也各有盲区:Local 善深而不善广,容易在选错入口实体时彻底跑偏;Global 善广而不善深,拿不到具体段落级证据。本章要解构的 DRIFT(Dynamic Reasoning and Inference with Flexible Traversal)正是为弥合这条裂缝而生——它先用 global 风格的社区报告起步,拿到一个初始答案和一批 follow-up 问题,再用 local search 逐个深挖这些 follow-up,在一棵动态生长的推理树上反复迭代,最后汇总成答案。

与这套精巧机制形成鲜明对照的,是同处 `structured_search/` 目录的 Basic Search——它不碰图谱,直接对 `text_units` 向量库做相似检索、拼上下文、生成答案,是最朴素的向量 RAG。把它放进 GraphRAG,正是要给前三种图谱检索提供一个"朴素基线"。本章末尾我们再回到 `api/query.py`,看这四种模式如何被统一收口对外暴露。

## DRIFT 的设计意图:先全局起步,再局部深挖

DRIFT 的核心思路写在 `DRIFTPrimer` 的类文档字符串里:"Perform initial query decomposition using global guidance from information in community reports"。它把一次问答拆成两个阶段:

- **Primer(引子)阶段**:用社区报告这种 global 层面的概览信息,对原始 query 做一次"初始分解"——既产出一个中间答案(intermediate answer),也产出一批 follow-up 问题指明深挖方向。
- **迭代(action)阶段**:把每个 follow-up 当成一次 local search,逐个回答;每次回答又可能派生出新的 follow-up,推理树随之生长;到达深度上限后停止,汇总成最终答案。

这套"广度起步 + 深度迭代"的结构,使 DRIFT 同时具备 Global 的视野和 Local 的细节。代码层面,它由三个文件分工:`primer.py`(引子)、`action.py`(单步动作)、`state.py`(树/图状态),由 `search.py` 的 `DRIFTSearch.search` 主循环串起来。

## Primer:用 HyDE 选报告,用社区报告生成初始 follow-ups

Primer 阶段的入口是 `DRIFTSearchContextBuilder.build_context`。它做的第一件事不是直接检索,而是构造一个 `PrimerQueryProcessor` 来"扩写"查询。`PrimerQueryProcessor.expand_query` 的实现颇有意思:它用 `secrets.choice(self.reports).full_content` 随机抽一份社区报告做模板,然后让 LLM"为查询生成一个假设性答案,并模仿该模板的结构"。这正是 HyDE(Hypothetical Document Embeddings)的思路——不直接拿短查询去匹配长文档,而是先生成一段"假想答案"再去匹配,让查询向量和文档向量落在更接近的语义空间里 [[Precise Zero-Shot Dense Retrieval without Relevance Labels]](https://arxiv.org/abs/2212.10496)。prompt 里特意叮嘱"不要引入原查询中没有的新命名实体",避免假想答案带偏方向。

拿到这段假想答案的嵌入后,`build_context` 用向量化余弦相似度(`np.dot` / `np.linalg.norm`)把它和所有社区报告的 `full_content_embedding` 比一遍,用 `report_df.nlargest(self.config.drift_k_followups, "similarity")` 取出最相似的前若干份报告,只保留 `short_id`、`community_id`、`full_content` 三列返回。注意这里复用了 `drift_k_followups`(默认 `20`)作为 top-k 报告数。

接着 `DRIFTPrimer.search` 接手。它先用 `split_reports` 把这批报告按 `primer_folds`(默认 `5`)切成几折,对每一折并发调用 `decompose_query`。`decompose_query` 用 `DRIFT_PRIMER_PROMPT` 拼 prompt,并通过 `response_format=PrimerResponse` 强制 LLM 输出结构化 JSON。`PrimerResponse` 这个 pydantic 模型规定了三个字段:`intermediate_answer`(要求恰好 2000 字符、markdown 格式)、`score`(0–100 的自评分)、`follow_up_queries`(要求"至少生成五个好的 follow-up 问题")。多折并发的设计,等价于让模型从语料的不同切片各提一组 follow-up,提升初始问题的覆盖面。

回到 `DRIFTSearch.search`,这些分折结果被 `_process_primer_results` 汇总:把各折的 `intermediate_answer` 用 `\n\n` 拼接、把所有 `follow_up_queries` 摊平成一个大列表、把 `score` 取平均,然后调 `DriftAction.from_primer_response` 打包成第一个 `DriftAction`,作为推理树的根节点入树。这里有两处显式的健壮性检查:若 primer 没产出任何 intermediate answer 或任何 follow-up,直接抛 `RuntimeError`,因为没有 follow-up 就没法启动后续迭代。

## DriftAction:推理树上的一个节点

`action.py` 里的 `DriftAction` 是推理树的基本单元,把"LLM 产出的动作字符串"封装成结构化对象。它持有 `query`(本节点要回答的问题)、`answer`(回答,对应 intermediate answer)、`score`、以及 `follow_ups`(派生出的子动作列表)。`is_complete` 属性以 `answer is not None` 判断该节点是否已被回答。

它的核心是异步方法 `search`。一个未完成的 action 调用它时,会把自己的 `query` 喂给传入的 LocalSearch 引擎,并附带两个关键参数:`drift_query=global_query`(原始全局查询)和 `k_followups`。这正是 DRIFT 把 local search "改造"成会产 follow-up 的搜索的关键——回看第 11 章的 `LocalSearch.search`,当 kwargs 里含 `drift_query` 时,它会用 `global_query` 和 `followups` 去格式化 `DRIFT_LOCAL_SYSTEM_PROMPT`,而这个 prompt 末尾要求 LLM 输出 JSON 三元组 `{'response', 'score', 'follow_up_queries'}`,并明确"基于你的回答,围绕整体研究问题提出至多 {followups} 个 follow-up"。

拿到 local search 的 JSON 结果后,`DriftAction.search` 用 `try_parse_json_object` 容错解析(失败不抛异常,而是返回空结果让 `-inf` 分数自然把它排到末尾),依次 `pop` 出 `response` 写入 `self.answer`、`score` 写入 `self.score`、`follow_up_queries` 写入 `self.follow_ups`,并把 token 计数累加进 `self.metadata`。这样一个 action 执行完,就在自己身上挂上了答案、分数,以及一批新的子问题。

`DriftAction` 还实现了 `__hash__`(基于 query)和 `__eq__`,文档里写明"假设查询唯一",这是为了让它能直接作为 networkx 图的节点。

## QueryState:用 networkx 管理这棵推理树

`state.py` 的 `QueryState` 用一张 `nx.MultiDiGraph` 承载整棵推理树/图。它提供几组操作:

- `add_action` 把一个 action 作为节点加入图;`relate_actions` 在父子 action 之间加一条带权边;`add_all_follow_ups` 把某个 action 的所有 follow-up(可能是字符串,会被包成 `DriftAction(query=...)`)入图并连边。
- `find_incomplete_actions` 返回所有 `is_complete` 为假的节点;`rank_incomplete_actions` 对它们排序——若提供 scorer 则按分数降序,否则用 `random.shuffle` 打乱顺序返回。当前主循环未传 scorer,所以走的是随机化路径,避免每轮都固定选同一批节点。
- `serialize` 把图序列化成 `{nodes, edges}`,给每个节点分配整数 id、把节点的 `context_data` 按 query 聚成上下文,供最终 reduce 和返回使用。`action_token_ct` 则遍历所有节点累加 token 用量。

用图(而非简单的列表或栈)来管理状态,带来的好处是 follow-up 之间的派生关系被显式记录成边,既能序列化复盘整棵推理路径,也为将来按图结构剪枝/打分留出空间。

## search.py:迭代主循环与 reduce 收尾

`DRIFTSearch.__init__` 在构造时就内置了一个 LocalSearch 实例:`init_local_search` 用 DRIFT 配置里的一堆 `local_search_*` 参数(如 `local_search_text_unit_prop` 默认 `0.9`、`local_search_community_prop` 默认 `0.1`、`local_search_top_k_mapped_entities` 默认 `10`)拼出 local 上下文参数和模型参数,并强制 `response_format_json_object=True`,确保每次 local 回答都是可解析的 JSON。

主循环 `search` 的骨架是:

1. **冷启动**:若 `query_state.graph` 为空,先跑 primer——`build_context` 选报告、`primer.search` 分解查询、`_process_primer_results` 打包成根 action 入树并展开其 follow-ups。
2. **迭代**:`epochs` 从 0 计起,只要 `epochs < self.context_builder.config.n_depth`(默认 `n_depth = 3`)就继续。每轮先 `rank_incomplete_actions` 取出未完成节点,空了就 break;否则截取前 `drift_k_followups`(默认 `20`)个,调 `_search_step` 用 `tqdm_asyncio.gather` 并发执行这批 action 的 `search`;执行完把每个 action 及其新生 follow-ups 写回 state,`epochs += 1`。
3. **汇总**:循环结束后统计 token,`query_state.serialize(include_context=True)` 导出节点/边/上下文。若 `reduce=True`,用 `DRIFT_REDUCE_PROMPT` 把所有节点的 `answer` 收成一段综合回答(`_reduce_response`);`reduce_temperature` 默认 `0`。

这里几个参数共同框定了 DRIFT 的"预算":`n_depth` 控制迭代轮数(深度),`drift_k_followups` 控制每轮处理的 follow-up 上限(宽度),`primer_folds` 控制初始问题的多样性。三者一起决定了这次搜索要烧多少次 LLM 调用——DRIFT 是四种模式里最"贵"的,因为 primer、每个 action、最终 reduce 都各自要调模型。

流式版本 `stream_search` 是个聪明的复用:它先以 `reduce=False` 跑完整个迭代拿到中间结果,再单独调 `_reduce_response_streaming` 把最终 reduce 这一步以流式吐出。换句话说,迭代过程对用户不可见,真正流式的只有收尾的综合答案。

## Basic Search:作为对照基线的朴素向量 RAG

`basic_search/` 目录只有两个文件,代码量和上面那套树状机制形成强烈反差。`basic_context.py` 的注释一句话点题:"Implementation of a generic RAG algorithm (vector search on raw text chunks)"。

`BasicSearchContext.build_context` 的逻辑就是教科书式向量 RAG:用 `text_unit_embeddings.similarity_search_by_text` 对查询做相似检索取前 `k`(默认 `10`)个 text unit,按 token 预算 `max_context_tokens`(默认 `12_000`)逐条填进上下文直到填满,组装成一张 CSV 表返回。它完全不碰实体、关系、社区报告,只对原文块的向量库做一次检索——没有图、没有迭代、没有 follow-up。

`BasicSearch.search` 同样直白:`build_context` 拼上下文,用 `BASIC_SEARCH_SYSTEM_PROMPT` 格式化 system prompt,调一次模型(以 stream 方式聚合)产出答案,封装成 `SearchResult`。整条链路只有一次检索、一次生成。

它的存在意义恰恰在于"朴素"。GraphRAG 的全部价值主张是"图谱结构能带来超越朴素向量检索的回答质量",而要证明这一点,就必须有一个不走图的基线作对照。Basic Search 就是这个基线:当你想知道"对某个问题,图谱到底比纯向量 RAG 好多少",或者当语料尚未建好图谱、只想先跑通最简单的检索时,它就是答案。

## api/query.py:四种模式统一收口

`api/query.py` 是查询能力对外的唯一门面,它把 local / global / drift / basic 四种模式统一成同构的一组函数:每种模式都有一个 `xxx_search`(返回完整答案 + 上下文)和一个 `xxx_search_streaming`(返回异步生成器)变体,签名风格一致——都接收 `GraphRagConfig`、相关的 parquet DataFrame、`community_level`、`response_type`、`query`、`callbacks`。

分发的关键在于:api 层只负责"读数据 + 装配引擎 + 调用",真正的引擎构造下放给 `query/factory.py`。以 `drift_search_streaming` 为例,它从 `config.vector_store` 拿到实体描述嵌入库和社区 `full_content` 嵌入库,用 `read_indexer_*` 系列把 parquet 还原成数据模型对象,载入 prompt,然后调 `get_drift_search_engine` 拿到一个 `DRIFTSearch`,最后 `return search_engine.stream_search(query=query)`。`basic_search_streaming` 同理,只是它只需要 text unit 嵌入库,调的是 `get_basic_search_engine`。

值得注意的是,`drift_search` 和 `basic_search` 的"非流式"版本其实是构建在流式之上的——`drift_search` 内部 `async for chunk in drift_search_streaming(...)` 把流拼成 `full_response`,并通过挂一个 `NoopQueryCallbacks` 的 `on_context` 回调把上下文捞出来。这种"流式为本、非流式为皮"的设计,保证了两种调用路径行为一致,也避免维护两套逻辑。

`factory.py` 里的 `get_drift_search_engine` / `get_basic_search_engine` 则各自按 `config.drift_search` / `config.basic_search` 取出对应的 completion / embedding 模型配置,`create_completion` / `create_embedding` 实例化模型,再把 context builder 和参数装进对应的 Search 类。四个 `get_*_search_engine` 是 api 与各 Search 实现之间的统一装配点——上层只认 `mode`,具体接哪个引擎、用哪套 config 子段、塞哪个 prompt,都封在 factory 里。

## 四种模式:从细节到概览到多跳到基线

把全书第十一到十三章的四种检索放在一起看,GraphRAG 的设计动机就清晰了——它不是用一种检索打天下,而是用四种模式覆盖四类问答需求:

- **Local Search**:以实体为入口向下深挖,回答"关于某个具体事物/人/概念的细节",善深。
- **Global Search**:map-reduce 全量社区报告,回答"整个语料的主题、趋势、概览",善广。
- **DRIFT Search**:primer 用 global 起步定方向,再用 local 逐个深挖 follow-up,在推理树上迭代,回答需要多跳推理、既要视野又要细节的复杂问题。
- **Basic Search**:不走图的纯向量 RAG,作为衡量前三者增益的对照基线,也是最轻量的兜底。

这四种模式共享同一套索引产物(实体、关系、社区报告、text units、嵌入库),却以不同方式组合它们——这正是 GraphRAG"一次建图、多种查询"价值的最终兑现。

## 本章小结

- DRIFT 的设计意图是融合 Local 的"深"与 Global 的"广":先全局起步拿到方向与 follow-ups,再局部深挖逐个回答,在动态推理树上迭代扩展。
- Primer 阶段用 HyDE 思路(`PrimerQueryProcessor.expand_query` 随机抽社区报告做模板生成假想答案)选出 top-k 报告,再用 `DRIFT_PRIMER_PROMPT` + `PrimerResponse` 结构化输出,产出 intermediate answer、score 和至少五个 follow-up。
- `DriftAction` 是推理树节点,其 `search` 把自身 query 交给带 `drift_query` 的 LocalSearch,从 JSON 结果中解析出 answer / score / 新 follow-ups,实现"回答即派生"。
- `QueryState` 用 `nx.MultiDiGraph` 显式记录 action 间的派生关系,支持排序、序列化与 token 统计;当前主循环无 scorer,走 `random.shuffle` 选节点。
- 主循环受三个预算参数框定:`n_depth`(默认 `3`,迭代深度)、`drift_k_followups`(默认 `20`,每轮宽度与 top-k 报告数)、`primer_folds`(默认 `5`,初始问题多样性)。
- DRIFT 收尾用 `DRIFT_REDUCE_PROMPT` 把所有节点 answer 汇总成综合回答;`stream_search` 先非流式跑完迭代,只把最终 reduce 这一步流式吐出。
- Basic Search 是纯向量 RAG 基线:`similarity_search_by_text` 取前 `k`(默认 `10`)个 text unit、按 `max_context_tokens`(默认 `12_000`)拼上下文、一次生成,完全不碰图谱。
- Basic 的价值在于作为对照基线,量化图谱检索相对朴素向量检索的增益,也是语料未建图时的轻量兜底。
- `api/query.py` 把四种模式统一成 `xxx_search` + `xxx_search_streaming` 同构函数,非流式版构建在流式之上,真正的引擎装配下放给 `factory.py` 的 `get_*_search_engine`。
- 四种模式共享同一套索引产物,以不同组合方式覆盖"细节 / 概览 / 多跳 / 基线"四类需求,兑现"一次建图、多种查询"。

## 动手实验

1. **实验一:复盘 DRIFT 推理树** — 在 `DRIFTSearch.search` 的迭代循环里(`while epochs < n_depth`)打印每轮 `rank_incomplete_actions()` 返回的 query 列表和 `epochs`,再调用 `query_state.serialize()` 把 `{nodes, edges}` 落盘成 JSON。用一个需要多跳的问题跑一次 drift search,观察推理树如何从根 action 逐层展开,follow-up 数量随深度如何变化。
2. **实验二:调节深度与宽度预算** — 在 settings 中把 `drift_search.n_depth` 从默认 `3` 改为 `1` 和 `5`,把 `drift_k_followups` 从 `20` 改为 `5`,分别跑同一查询,对比 `SearchResult` 里的 `llm_calls`、`prompt_tokens` 与最终答案质量,体会"深度 × 宽度"如何线性放大 LLM 成本。
3. **实验三:验证 HyDE 的作用** — 临时修改 `PrimerQueryProcessor.expand_query`,让它直接 `return query, token_ct`(跳过假想答案生成),对比改前改后 `build_context` 选出的 top-k 社区报告 `short_id` 是否不同,直观感受 HyDE 扩写对报告召回的影响。
4. **实验四:Basic 与 Local/Global/DRIFT 横评** — 对同一组问题(一类问细节、一类问概览、一类需多跳)分别用 `basic_search`、`local_search`、`global_search`、`drift_search` 四个 api 跑一遍,把答案和 `llm_calls` 并排记录成表格,验证每种模式在自己擅长的问题类型上的相对优势与成本差异。

> **下一章预告**:四种检索模式背后,是一整套被反复复用的基础设施。第 14 章《LLM、缓存、存储与工程原则收尾》将下潜到 `graphrag-llm`、`graphrag-cache`、`graphrag-storage` 三个独立包,看模型调用、提示缓存与多后端存储如何被抽象成可替换组件,再走一遍 `prompt_tune` 的自动调优流程,最后跨越全书七部,做一次贯穿数据采集、切分、图谱、检索的工程原则收尾与全书结语。
