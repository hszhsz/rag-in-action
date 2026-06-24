# 第 11 章 Local Search 局部检索

从第 3 章到第 10 章,我们一路把原始文档碾成了一座知识图谱:文本被切成 text unit,实体与关系被抽取、摘要、终化,Leiden 算法把图切成层级社区,LLM 又为每个社区写了报告,最后所有产物连同向量被落盘成一组 parquet 文件。索引侧的工作到此为止——它把语料变成了"可被检索的结构"。从本章起我们切换到查询侧:用户敲下一个问题,系统如何从这堆 parquet 里捞回正确的上下文、塞进一次 LLM 调用、生成有出处的答案?

GraphRAG 给出了四种查询模式(local、global、drift、basic),本章解构其中最贴近"传统 RAG 直觉"的一种——Local Search。它擅长回答**与具体实体相关的细节型问题**:"X 公司和 Y 公司是什么关系""某人参与了哪些事件"。它的思路是先把查询映射到图上几个最相关的"种子实体",再围着这些实体把多种来源的上下文(实体描述、关系、所属社区报告、关联原文、协变量)按 token 预算拼成一个上下文窗口,交给 LLM 作答。这与下一章的 Global Search 形成鲜明对比:后者面对"整个语料讲了什么"这类全局概览问题,走的是社区报告 map-reduce 的路子。

## 入口与契约:BaseSearch 与 SearchResult

所有查询模式共享一个抽象基类 `BaseSearch`,定义在 `base.py`。它是个泛型类 `BaseSearch(ABC, Generic[T])`,`T` 被约束为四种 context builder 之一(`GlobalContextBuilder`、`LocalContextBuilder`、`DRIFTContextBuilder`、`BasicContextBuilder`)。构造时它持有三样核心依赖:`model`(一个 `LLMCompletion`)、`context_builder`(负责拼上下文)、`tokenizer`(默认取 `model.tokenizer`),外加两组参数字典 `model_params` 与 `context_builder_params`。基类声明了两个抽象协程方法:`search`(一次性返回完整结果)与 `stream_search`(流式生成),逼着每个子类都同时支持阻塞与流式两种调用方式。

返回值统一为 `SearchResult` dataclass。它不只装答案文本,还把整条链路的可观测信息一并带回:`response`(答案)、`context_data`(结构化的上下文表,`dict[str, pd.DataFrame]`)、`context_text`(真正塞进窗口的文本)、`completion_time`,以及 token 与调用计数(`llm_calls`、`prompt_tokens`、`output_tokens`)还有它们的分类明细(`llm_calls_categories` 等)。这种"答案 + 上下文 + 计量"三合一的契约,让上层(CLI、API、评测脚本)既能展示答案,又能回溯"这个答案到底基于哪些记录"。

`LocalSearch(BaseSearch[LocalContextBuilder])` 在 `local_search/search.py` 里实现这个契约。它在基类之上额外接收 `system_prompt`(默认 `LOCAL_SEARCH_SYSTEM_PROMPT`)、`response_type`(默认 `"multiple paragraphs"`,控制答案长度与格式)和一组 `callbacks`(`QueryCallbacks`,用于流式 token 与上下文回调)。

## search 的主流程:先建上下文,再生成答案

`LocalSearch.search` 的骨架非常清爽,可以拆成"建上下文 → 拼 prompt → 流式生成 → 打包结果"四步。

第一步,调用 `self.context_builder.build_context(query=query, conversation_history=conversation_history, **kwargs, **self.context_builder_params)`,把全部上下文拼装工作委托给 builder,拿回一个 `ContextBuilderResult`。注意它把 builder 自身可能产生的 `llm_calls`、`prompt_tokens`、`output_tokens` 记到 `"build_context"` 这个分类里——对 local search 而言 builder 不调 LLM,这些值是 0,但这套记账框架是四种模式通用的。

第二步,把 `context_result.context_chunks`(拼好的上下文文本)和 `response_type` 填进 system prompt。这里有个细节:如果 `kwargs` 里带了 `"drift_query"`(说明是被 DRIFT Search 当作子过程调用的),还会额外填入 `global_query` 和 `followups` 两个槽——这正是第 13 章 DRIFT 复用 local search 的接口痕迹。普通调用走 `else` 分支,只填 `context_data` 与 `response_type`。

第三步,用 `CompletionMessagesBuilder` 把 system prompt 作为系统消息、原始 `query` 作为用户消息组装起来,然后**强制以 `stream=True` 发起 `completion_async`**。即便是非流式的 `search`,内部也是流式拉取、累加进 `full_response`,并对每个 chunk 回调 `callback.on_llm_new_token`。这样设计的好处是:阻塞与流式两条路径共用同一套底层调用,差别只在外层是否 `yield`。

第四步,统计 token(用 `self.tokenizer.encode` 分别数 prompt 与 response),回调 `on_context` 把结构化上下文交给上层,最后组装 `SearchResult` 返回。整个 `try` 块外裹了异常兜底:一旦生成出错,返回一个 `response=""` 但仍带着 `context_data` 的 `SearchResult`,而不是直接抛栈——保证调用方总能拿到"至少检索到了什么"。

`stream_search` 是 `search` 的精简流式版:同样先 `build_context`、拼 prompt、回调 `on_context`,然后 `async for chunk in response` 里直接 `yield response_text`。它不做 token 记账,把答案的流式输出交给消费端。

## map_query_to_entities:从查询找到种子实体

整个 local search 的"图入口"是 `map_query_to_entities`(在 `context_builder/entity_extraction.py`)。它的职责是:给定一句自然语言查询,返回图上**语义最相关的若干实体**作为后续上下文聚合的种子。

机制是向量相似检索。它对 `text_embedding_vectorstore` 调用 `similarity_search_by_text`,临时用 `text_embedder` 把查询编码成向量,在**实体描述向量库**里检索。关键参数是 `k`(默认 `10`)和 `oversample_scaler`(默认 `2`),实际取 `k * oversample_scaler` 个结果——多捞一倍是为了给后续"排除指定实体"留余量。命中的向量文档再通过 `get_entity_by_id`(当 `embedding_vectorstore_key == EntityVectorStoreKey.ID`)或 `get_entity_by_key` 回填成完整的 `Entity` 对象。

这里有两条容易忽略的分支。其一,当 `query == ""`(空查询)时,它不走向量检索,而是按 `entity.rank` 降序取前 `k` 个——即"图上最重要的实体"。其二,函数支持 `include_entity_names` 与 `exclude_entity_names`:排除名单先把匹配结果过滤掉,包含名单则通过 `get_entity_by_name` 强行补进去,且**置于返回列表最前**(`return included_entities + matched_entities`)。这给了调用方手动锚定实体的能力。

值得强调的是,`embedding_vectorstore_key` 默认是 `EntityVectorStoreKey.ID`——工厂里两处注释都提醒:如果你的向量库是用实体 title 当 id 存的,要改成 `EntityVectorStoreKey.TITLE`。这是一个真实存在的部署陷阱。

## LocalSearchMixedContext.build_context:混合上下文的预算分配

`LocalSearchMixedContext`(在 `local_search/mixed_context.py`)是本章的核心。它的命名里 "Mixed" 三个字母点出了设计哲学:**单一来源不足以回答细节问题**——光有实体描述太干瘪,光有原文又丢了图结构,所以要把多种来源混合起来。构造时它把传入的 entities、community_reports、text_units、relationships 都建成 `id -> 对象` 的字典以便 O(1) 查找,covariates 则是 `dict[str, list[Covariate]]`。

`build_context` 的参数表就是一张"预算分配表"。最关键的几个:`max_context_tokens`(默认 `8000`,但工厂从配置注入,默认值为 `12_000`)、`text_unit_prop`(原文占比,签名默认 `0.5`)、`community_prop`(社区报告占比,签名默认 `0.25`)、`top_k_mapped_entities`(种子实体数,默认 `10`)、`top_k_relationships`(每个实体的关系数,默认 `10`)。函数开头有一道硬校验:`community_prop + text_unit_prop > 1` 直接 `raise ValueError`——因为剩下的 `local_prop = 1 - community_prop - text_unit_prop` 必须非负,这部分预算留给实体-关系-协变量上下文。

> 注意配置层与签名默认值并不一致。`config/defaults.py` 里 local search 的实际默认是 `text_unit_prop = 0.5`、`community_prop = 0.15`、`top_k_entities = 10`、`top_k_relationships = 10`、`conversation_history_max_turns = 5`、`max_context_tokens = 12_000`;工厂 `get_local_search_engine` 把这些从 `config.local_search` 读出来塞进 `context_builder_params`,运行时以配置为准。

整个 `build_context` 的执行顺序与预算扣减逻辑如下:

1. **并入对话历史的查询**:若有 `conversation_history`,先取最近 `conversation_history_max_turns` 轮的用户提问,拼到当前 `query` 后面再去做实体映射——这样追问"那他呢"也能命中正确实体。
2. **映射种子实体**:调用 `map_query_to_entities`,`k=top_k_mapped_entities`。
3. **对话历史上下文**:若有历史,调用 `conversation_history.build_context` 生成一段历史表,**先放进最终上下文,并从 `max_context_tokens` 里扣掉它占用的 token**——历史是"硬开销",不参与比例分配。
4. **社区上下文**:预算 `int(max_context_tokens * community_prop)`,由 `_build_community_context` 填充。
5. **本地上下文**(实体+关系+协变量):预算 `int(max_context_tokens * local_prop)`,由 `_build_local_context` 填充。
6. **原文上下文**:预算 `int(max_context_tokens * text_unit_prop)`,由 `_build_text_unit_context` 填充。

各部分以 `"\n\n".join(final_context)` 拼成 `context_chunks`,结构化记录合并进 `final_context_data`,一并装进 `ContextBuilderResult` 返回。比例参数让运维可以按场景调拨预算:文档细节问得多就调高 `text_unit_prop`,需要背景概览就调高 `community_prop`。

## 三路上下文如何各自裁剪

**社区报告**(`_build_community_context`):它不直接做向量检索,而是统计种子实体的 `community_ids`——一个社区被越多种子实体命中,就越相关。把命中数临时写进 `community.attributes["matches"]`,按 `(matches, rank)` 降序排序后用完即删,再交给通用的 `build_community_context` 按 `single_batch=True` 填到预算上限。这一步把"局部实体"和"宏观社区视角"连了起来:你问某个具体实体,系统会顺带把它所属社区的报告摘要拉进来。

**实体-关系-协变量**(`_build_local_context`):先用 `build_entity_context` 把种子实体本身的 id、title、description、rank 排成 Entities 表。然后**逐个实体增量添加**关系与协变量:每加入一个实体就重算一次 `build_relationship_context` 和 `build_covariates_context`,一旦累计 token 超过 `max_context_tokens` 就 `logger.warning` 并回退到上一个稳定状态(`break`)。关系的筛选 `_filter_relationships` 很讲究优先级:先收"网内关系"(两端都是种子实体),再收"网外关系"(一端是种子、一端是外部实体),网外还优先保留"被多个种子实体共享"的桥接关系——这恰好捕捉了图检索最有价值的部分:实体间的连接。

**原文 text unit**(`_build_text_unit_context`):遍历种子实体的 `text_unit_ids` 收集去重后的原文片段,并用 `count_relationships` 数出每个片段承载了多少条相关关系,再按 `(entity_order, -num_relationships)` 排序——即"种子实体越靠前、承载关系越多的原文越优先"。最后用 `build_text_unit_context` 按预算填充(注意这里 `shuffle_data=False`,保持排序结果)。这部分把抽象的图节点重新落回到"原始证据文本",是答案可溯源的基础。

## 对话历史的并入

对话历史由 `ConversationHistory` 类承载(`context_builder/conversation_history.py`)。它把交替的 user/assistant 消息组织成 `QATurn`(一个用户问 + 若干助手答),并提供两个被 local search 用到的方法:`get_user_turns(n)` 取最近 n 轮用户提问(用于扩充查询),`build_context(...)` 把历史渲染成一张带表头的 CSV 文本(用于塞进上下文)。

工厂里把 `conversation_history_user_turns_only` 固定为 `True`,意味着默认**只把用户的历史提问**带进上下文,不带助手的历史回答——这是个有意的取舍:历史答案往往冗长且可能含幻觉,只保留用户意图既省 token 又更干净。`build_context` 里还有 `recency_bias` 开关,但 mixed context 调用时传的是 `recency_bias=False`,配合 `max_qa_turns` 截断,保留的是最早的若干轮。

## 数据从哪来:indexer_adapters 把 parquet 读回内存

把上面这些对象喂给 builder 之前,得先从索引产物里把它们读回内存,这层桥接就是 `query/indexer_adapters.py`。它的每个 `read_indexer_*` 函数都接收一个或多个 parquet 读出的 `DataFrame`,做一轮类型适配、列改名、关联汇总,产出 query 侧的领域对象列表:

- `read_indexer_entities(entities, communities, community_level)`:把实体表与社区表 join,按 `community_level` 过滤,把每个实体归属的多个社区 id 收成集合,最终用 `degree` 当 rank、`description_embedding` 当向量列,读成 `list[Entity]`。
- `read_indexer_relationships`:用 `human_readable_id` 当 short id、`combined_degree` 当 rank,读成 `list[Relationship]`。
- `read_indexer_reports`:把社区报告与社区表按层级 roll-up——非动态选择时,只保留"实体所属的最大社区层级"对应的报告,避免同一实体的多层报告重复。
- `read_indexer_text_units` / `read_indexer_covariates`:分别读回原文片段与协变量(claims)。

文件开头那句注释很坦诚:这些做类型适配、改名、汇总的代码"最终应该消失",理想状态是直接读进对象模型。这提醒我们:query 与 index 之间的这层 adapter 是历史包袱形成的兼容层,而非精心设计的 API 边界——读源码时要把它当作"胶水"看待,不要误以为它承载了核心语义。

## 工厂装配:get_local_search_engine

`query/factory.py` 的 `get_local_search_engine` 把上述零件拼成一台可运行的引擎。它从 `config.local_search` 读出模型 id,分别 `create_completion` 和 `create_embedding` 建出对话模型与嵌入模型(local search 同时需要两者:嵌入模型用于查询→实体映射,对话模型用于生成答案),再 new 一个 `LocalSearchMixedContext` 注入全部数据与向量库 `description_embedding_store`,最后把配置里的比例参数打包进 `context_builder_params`。

这里能读出几个工厂层面的固定决策:`include_entity_rank=True` 和 `include_relationship_weight=True`(让 LLM 看到实体重要度与关系权重),`include_community_rank=False`,`return_candidate_context=False`(生产环境不返回"未入窗的候选集",那是调试/评测才需要的)。这些布尔开关在 builder 里都有对应处理分支,工厂只是替生产场景钦定了一组默认。

## 本章小结

- Local Search 面向"具体实体相关的细节型问题",通过"种子实体 → 混合上下文 → 单次 LLM 作答"完成查询,与下一章 Global Search 的全局概览定位互补。
- 所有查询模式共享 `BaseSearch` 抽象与 `SearchResult` 契约,后者把答案、结构化上下文与 token 计量三合一返回,便于溯源与可观测。
- `LocalSearch.search` 即便非流式也强制 `stream=True` 拉取,阻塞与流式两条路径共用底层调用;异常时返回空答案但保留上下文,不抛栈。
- `map_query_to_entities` 用查询向量在实体描述向量库里检索,`k * oversample_scaler` 过采样,支持空查询按 rank 兜底、include/exclude 手动锚定。
- `LocalSearchMixedContext.build_context` 是核心:用 `text_unit_prop`、`community_prop` 及推导出的 `local_prop` 把 token 预算切给三路上下文,并有 `community_prop + text_unit_prop > 1` 的硬校验。
- 三路裁剪各有策略——社区按"种子实体命中数 + rank"排序、本地上下文逐实体增量添加并在超预算时回退、原文按"实体序 + 承载关系数"排序。
- 对话历史既扩充查询(`get_user_turns`)又作为硬开销先扣预算;工厂默认 `conversation_history_user_turns_only=True`,只带用户提问。
- `indexer_adapters.py` 是 parquet 产物到 query 领域对象的胶水层,做类型适配与层级 roll-up,源码注释明言它"理应消失"。
- `get_local_search_engine` 同时装配嵌入模型(查询→实体)与对话模型(生成),并把比例与开关从 `config.local_search` 注入 `context_builder_params`,生产默认关闭候选上下文返回。
- 混合上下文的设计动机是"单一来源不足以回答细节问题",而预算占比让上下文构成可按场景在线调拨。

## 动手实验

1. **实验一:追踪一次 build_context 的预算分配** — 在 `mixed_context.py` 的 `build_context` 里,于计算 `community_tokens`、`local_tokens`、`text_unit_tokens` 后各加一行 `logger.info`,用一份小型索引产物跑一次 `LocalSearch.search`,观察三路实际分到的 token,并验证它们之和是否接近 `max_context_tokens` 减去对话历史开销。

2. **实验二:调比例看上下文变化** — 把 `context_builder_params` 里的 `text_unit_prop` 从 `0.5` 改成 `0.9`、`community_prop` 改成 `0.05`,对同一查询分别运行,对比 `SearchResult.context_text` 里 `-----Sources-----` 与 `-----Reports-----` 段落的长度变化,体会预算如何左右答案的证据构成。

3. **实验三:验证空查询兜底** — 直接调用 `map_query_to_entities`,传入 `query=""`,确认它跳过向量检索、改按 `entity.rank` 降序返回前 `k` 个实体;再传一个正常查询并打印命中实体的 title,与向量库里的 top-k 对照。

4. **实验四:观察种子实体如何牵出社区与原文** — 给一个会命中若干实体的查询,在 `_build_community_context` 与 `_build_text_unit_context` 入口打印 `selected_entities` 的 title 列表,再打印最终入窗的社区 id 与 text unit id,梳理出"实体 → 社区报告 / 原文片段"的具体牵连路径。

> **下一章预告**:第 12 章《Global Search 全局检索》。Local Search 从具体实体出发回答细节;当问题是"整个语料到底讲了什么"时,就轮到 `GlobalSearch` 登场。它放弃种子实体,改以 `GlobalCommunityContext` 把社区报告分批喂给 LLM,先 map(每批独立打分作答)、再 reduce(汇总成终答),用 map-reduce 撑起全局概览能力——我们将逐行拆解这条与 local 截然不同的检索路径。
