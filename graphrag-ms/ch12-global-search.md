# 第 12 章 Global Search 全局检索

上一章的 Local Search 擅长回答"细节型"问题:它从查询出发,锚定具体实体,沿着关系、文本单元、协变量逐层向外扩展,把"局部邻域"塞进一个上下文窗口让 LLM 作答。但有一类问题它天然回答不好——"这份语料整体在讲什么主题?""贯穿全书的核心冲突有哪些?""所有事件归纳起来呈现什么趋势?"这类"全局概览/主题归纳"型问题,答案并不藏在某几个实体的邻域里,而是分散在整个图谱的方方面面。要回答它们,必须先有一种"俯瞰全局"的素材。

GraphRAG 的答案是:不直接看实体,而是看**社区报告**。第 7、8 章已经把语料压缩成了 Leiden 层级社区,并由 LLM 为每个社区写好了一篇结构化报告。Global Search 正是建立在这批报告之上的检索模式:它把所有社区报告分批喂给多个并行的 LLM 调用(map 阶段),让每个调用从报告里抽出带重要性评分的"要点",再把所有要点按分数汇总、裁剪,交给最后一次 LLM 调用合成连贯答案(reduce 阶段)。这就是 GraphRAG 论文标题所说的"From Local to Global"——用经典的 map-reduce 把"读不完的全局"拆成"读得完的局部之和"。本章逐行解构这套机制,源码集中在 `query/structured_search/global_search/` 与其上下文构建器、两个系统提示词中。

## 12.1 为什么必须 map-reduce

设计动机要先讲清楚,否则后面的代码会显得繁琐。Local Search 之所以能"一锅烩",是因为它只取与查询相关的一小撮实体,上下文天然有界。但 Global Search 面对的是**全部**社区报告——一个中等规模语料经过多层 Leiden 划分后,社区报告数量轻易上千。把它们全部塞进单个上下文窗口既不可能(超出 token 预算),也不划算(LLM 在超长上下文里会"迷失中间")。

于是 GraphRAG 采用 map-reduce:

- **map**:把社区报告切成多个"批"(每批受 `max_context_tokens` 约束),对每一批独立、并行地调一次 LLM,让它从这一批报告里抽出若干**要点(points)**,每个要点带一个 0–100 的**重要性评分(score)**。这一步是"分而治之",每个 map 调用只看一小撮报告,上下文永远可控,且彼此独立、可并发。
- **reduce**:把所有 map 调用产出的要点收拢到一起,按 score 降序排序,裁剪到 token 预算内,再用一次 LLM 调用把这些高分要点合成为最终答案。这一步是"汇总归纳"。

`GlobalSearch.search` 的 docstring 把这两步写得很直白:"Step 1: Run parallel LLM calls on communities' short summaries to generate answer for each batch""Step 2: Combine the answers from step 2 to generate the final answer"。注意它说的是 "short summaries"——map 阶段喂的是社区报告的摘要或正文,而非具体实体描述,这正是 Global 与 Local 的根本分野。

## 12.2 GlobalCommunityContext:把报告组织成批

map-reduce 的第一步发生在上下文构建器 `GlobalCommunityContext`(`global_search/community_context.py`)。它实现 `GlobalContextBuilder` 接口,核心方法 `build_context` 返回一个 `ContextBuilderResult`,其中 `context_chunks` 是一个**字符串列表**——每个元素就是一"批"报告,后续会对应一个 map 调用。

`build_context` 的关键调用是 `build_community_context`(来自共享的 `context_builder/community_context.py`),它把 `community_reports` 渲染成一张以 `|` 分隔的表格,表头由 `_get_header` 生成,包含 `id`、`title`、可选的 `occurrence weight`(社区权重)、`summary` 或 `content`(由 `use_community_summary` 决定),以及可选的 `rank`(社区评分)。真正决定"分批"的是这段循环:

```python
for report in selected_reports:
    new_context_text, new_context = _report_context_text(report, attributes)
    new_tokens = tokenizer.num_tokens(new_context_text)
    if batch_tokens + new_tokens > max_context_tokens:
        _cut_batch()
        if single_batch:
            break
        _init_batch()
    batch_text += new_context_text
    batch_tokens += new_tokens
    batch_records.append(new_context)
```

每当当前批的 token 数加上下一篇报告会超过 `max_context_tokens`,就 `_cut_batch()` 落盘当前批、`_init_batch()` 开新批。Global Search 走的是 `single_batch=False` 分支(在 `GlobalCommunityContext.build_context` 里硬编码),所以会持续切出多个批;Local Search 复用同一函数但传 `single_batch=True`,因而只取第一批。这一处 `single_batch` 标志,就是"全局"与"局部"在同一段代码里分道扬镳的开关。

几个值得留意的工程细节:

- **社区权重(occurrence weight)**:若传入了 `entities`,`_compute_community_weights` 会把"社区内实体所关联的文本单元去重计数"作为权重,并可 `normalize_community_weight` 按最大值归一化。这个权重连同 `rank` 一起,在 `_rank_report_context` 里用 `sort_values(..., ascending=False)` 对批内报告排序——重要的报告排在前面,即便后续被 token 预算截断,也优先保留高价值内容。
- **shuffle_data**:`build_context` 默认 `shuffle_data=True`,会用固定 `random_state`(默认 `86`)对报告先洗牌再分批。这是为了避免报告原始顺序(往往按社区 id 或层级)给 map 引入系统性偏差,让每批的内容分布更均匀;固定随机种子又保证了可复现。
- **min_community_rank**:`_is_included` 用 `report.rank >= min_community_rank` 过滤掉评分过低的报告,从源头裁掉噪声。

`factory.py` 的 `get_global_search_engine` 通过 `context_builder_params` 把这些参数钉死成一套默认值:`use_community_summary=False`(即喂正文 `full_content` 而非摘要)、`shuffle_data=True`、`include_community_rank=True`、`include_community_weight=True`、`community_weight_name="occurrence weight"`,`max_context_tokens` 取自 `gs_config.max_context_tokens`(默认 `12_000`)。

## 12.3 map 阶段:并行抽取带评分的要点

回到 `GlobalSearch`。拿到 `context_result.context_chunks`(一批批报告文本)后,`search` 用 `asyncio.gather` 对每一批并发调用 `_map_response_single_batch`:

```python
map_responses = await asyncio.gather(*[
    self._map_response_single_batch(
        context_data=data, query=query, max_length=self.map_max_length, **self.map_llm_params,
    )
    for data in context_result.context_chunks
])
```

并发度由构造函数里的 `asyncio.Semaphore(concurrent_coroutines)` 控制,`factory.py` 把它设为 `config.concurrent_requests`,默认信号量在 `__init__` 里是 `32`。每个 `_map_response_single_batch` 内部用 `async with self.semaphore` 限流,避免一次性把上千批全部打到模型上。

每次 map 调用做三件事:

1. 用 `MAP_SYSTEM_PROMPT.format(context_data=..., max_length=...)` 渲染系统提示,把这一批报告表格塞进 `{context_data}` 占位符,再以 `query` 作为 user message。
2. 调 `model.completion_async(..., response_format_json_object=True)`,强制模型输出 JSON。
3. 用 `_parse_search_response` 把 JSON 解析成结构化要点列表。

`MAP_SYSTEM_PROMPT`(`prompts/query/global_search_map_system_prompt.py`)是整套机制的灵魂。它要求模型"Generate a response consisting of a list of key points",并明确每个要点必须含两个元素:一段 `Description`(综合描述),以及一个 `Importance Score`——"An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0."。提示词还规定了严格的 JSON 模板:

```json
{
    "points": [
        {"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}
    ]
}
```

注意 `description` 里嵌入了 `[Data: Reports (report ids)]` 的引用格式,且要求"Do not list more than 5 record ids in a single reference",超出则列前 5 个再加 `+more`。这套引用约定从 map 一直贯穿到 reduce 和最终答案,使 Global Search 的结论可追溯到具体社区报告。

`_parse_search_response` 的实现很务实:先用 `try_parse_json_object` 容错解析(应对模型偶尔输出脏 JSON),取出 `points` 字段,再把每个元素归一化成 `{"answer": element["description"], "score": int(element["score"])}`。任何缺字段、非列表、解析失败的情况,统一退化为 `[{"answer": "", "score": 0}]`——score 0 的要点会在 reduce 阶段被过滤,等于"这一批没贡献"。`_map_response_single_batch` 外层还包了一个 `try/except`:即便整批调用抛异常,也返回一个 score 0 的占位结果,保证一批失败不拖垮整个 gather。这种"局部失败不致命"的设计,是 map-reduce 在工程上能跑稳的前提。

每个 map 调用都返回一个 `SearchResult`,记录了 `llm_calls=1`、`prompt_tokens`、`output_tokens`(都用 `self.tokenizer.encode` 实测 token 数)。`search` 随后把各批的统计 `sum` 起来,分门别类存进 `llm_calls["map"]`、`prompt_tokens["map"]`。

## 12.4 reduce 阶段:汇总、排序、裁剪、合成

map 产出一堆零散要点后,`_reduce_response` 负责收口。它的逻辑分四步:

**第一步:收拢要点。** 遍历所有 `map_responses`,把每个要点连同它来自第几个 map(`analyst` 下标)打包:

```python
key_points.append({"analyst": index, "answer": element["answer"], "score": element["score"]})
```

这里把每个 map 调用拟人化成一位"分析师(analyst)",reduce 提示词里也以"synthesizing perspectives from multiple analysts"的口吻组织——这是个巧妙的拟人化框架,让 LLM 更自然地"综合多方观点"。

**第二步:过滤与排序。** 丢掉 `score > 0` 不成立的要点;若过滤后一个不剩且未开启 `allow_general_knowledge`,直接返回固定文案 `NO_DATA_ANSWER`("I am sorry but I am unable to answer this question given the provided data."),并打日志提示可开 `allow_general_knowledge` 让模型补充通识(代价是幻觉风险)。否则按 `score` 降序 `sorted(..., reverse=True)`。

**第三步:按预算裁剪。** 这是 reduce 真正"省 token"的地方。它把排好序的要点逐条格式化:

```python
formatted_response_data = [f"----Analyst {point['analyst'] + 1}----",
                           f"Importance Score: {point['score']}", point["answer"]]
```

逐条累加 token,一旦 `total_tokens + ... > self.max_data_tokens` 就 `break`。`max_data_tokens` 来自 `gs_config.data_max_tokens`(默认 `12_000`)。因为要点已按分数降序,裁剪等价于"只保留最重要的那批要点"——高分要点优先进入最终 prompt,低分的被预算挤掉。

**第四步:合成答案。** 把裁剪后的 `text_data` 填进 `REDUCE_SYSTEM_PROMPT`,连同 `response_type`(如 "multiple paragraphs")与 `max_length` 一起渲染,以 `query` 作 user message,流式调用 LLM。`REDUCE_SYSTEM_PROMPT`(`prompts/query/global_search_reduce_system_prompt.py`)特意告诉模型:"the analysts' reports provided below are ranked in the descending order of importance",并要求"remove all irrelevant information... merge the cleaned information into a comprehensive answer",同时"preserve all the data references previously included"但"do not mention the roles of multiple analysts"——即保留 `[Data: Reports (...)]` 引用,但不要在最终答案里暴露"分析师"这个内部脚手架。

reduce 是流式的:它对 `completion_async(stream=True)` 的 chunk 逐块累加,并对每个 chunk 触发 `callback.on_llm_new_token`,这样上层可以边生成边展示。最终 `_reduce_response` 返回一个 `SearchResult`,包含完整答案、`text_data` 作为 `context_data`、以及 reduce 这一次调用的 token 统计。

## 12.5 流式接口与统计口径

`GlobalSearch` 提供两条对外路径。`search` 返回完整的 `GlobalSearchResult`——它在 `SearchResult` 基础上扩展了 `map_responses`(每个 map 的原始结果)、`reduce_context_data`、`reduce_context_text`,便于调试与可观测。它把三段开销(`build_context`、`map`、`reduce`)分别记进 `llm_calls_categories`、`prompt_tokens_categories`、`output_tokens_categories`,再用 `sum(...values())` 汇总成总数。这种"既给总账又给分项"的统计口径,让使用者一眼看清成本花在哪——通常 map 阶段调用次数最多、最烧 token,是 Global Search 成本的大头。

另一条是 `stream_search`:它同样先 `build_context`、再并发 map,但 reduce 走 `_stream_reduce_response` 这个异步生成器,逐 token `yield` 出去。两段 reduce 逻辑(`_reduce_response` 与 `_stream_reduce_response`)的排序、过滤、裁剪代码几乎重复,差别仅在返回方式——这是为了让流式与非流式各自保持简单直白,代价是一点重复。`stream_search` 还在 map 前后触发 `on_map_response_start`/`on_map_response_end`/`on_context` 回调,把 map-reduce 的进度透出给前端。

## 12.6 动态社区选择:用 rating 预筛降本

把"全部"社区报告都跑一遍 map,在大语料上是笔不小的开销。GraphRAG 因此提供了可选的**动态社区选择(dynamic community selection)**,实现于 `DynamicCommunitySelection`(`context_builder/dynamic_community_selection.py`)。`GlobalCommunityContext` 在 `dynamic_community_selection=True` 时会在 `build_community_context` 之前先调 `self.dynamic_community_selection.select(query)`,用筛选后的报告子集替换全集。

它的核心思路是**沿社区层级自顶向下剪枝**。`select` 从 level 0 的根社区出发(`self.starting_communities = self.levels["0"]`),对队列里每个社区调 `rate_relevancy`,让 LLM 用 `RATE_QUERY` 提示词给"该社区报告与查询的相关性"打一个 0–5 的分(`rate_prompt.py`)。逻辑是:

- 评分 `>= threshold`(默认 `1`)的社区视为相关,加入结果集,并把它的**子社区**入队继续评分——即只在"看起来相关"的分支上才深入下钻;
- 若 `keep_parent=False`(默认),一旦子社区被判相关,就把父社区从结果集里 `discard` 掉,因为更细粒度的子报告通常更贴题;
- 若某一层评下来一个相关的都没有,且层级未超过 `max_level`(默认 `2`),则把下一整层的报告补入队列继续试,避免过早剪空。

`rate_relevancy` 还支持 `num_repeats`(默认 `1`):对同一报告重复评分多次,用 `np.unique` 取众数作为最终 rating,降低单次评分的随机性。整个过程的 `llm_calls`/`prompt_tokens`/`output_tokens` 会回传并计入 `build_context` 的统计——动态选择本身要花 LLM 调用(评分),但它换来的是 map 阶段报告数量的大幅下降。是否值得,取决于语料规模与查询特性,这正是它做成开关、默认关闭的原因。

## 12.7 在 factory 里组装

`get_global_search_engine` 是把上述零件拼成可用引擎的总装线。它从 `config.global_search` 取参,`create_completion` 建模型,构造 `GlobalCommunityContext`(注入 reports、communities、entities,以及动态选择开关与 kwargs),再实例化 `GlobalSearch`。几个钉死的默认值值得记住:`max_data_tokens=gs_config.data_max_tokens`、`map_max_length=gs_config.map_max_length`(默认 `1000` 词)、`reduce_max_length=gs_config.reduce_max_length`(默认 `2000` 词)、`allow_general_knowledge=False`(默认不允许补通识,优先忠于数据)、`json_mode=False`(注意:map 调用内部仍硬编码 `response_format_json_object=True`,因此 map 始终输出 JSON;`json_mode` 这里只影响 `map_llm_params` 是否塞 `response_format`)。

当 `dynamic_community_selection=True` 时,factory 还会从 `gs_config` 抽出 `dynamic_search_keep_parent`、`dynamic_search_num_repeats`、`dynamic_search_use_summary`、`dynamic_search_threshold`、`dynamic_search_max_level` 等参数,连同模型与 tokenizer 一并塞进 `dynamic_community_selection_kwargs`,传给 `GlobalCommunityContext`。

## 本章小结

- Global Search 回答"全局概览/主题归纳"型问题,与第 11 章 Local 的"细节型"互补;它的素材是**社区报告**而非具体实体邻域。
- 全局问题的答案分散在整个图谱,单个上下文窗口装不下全部报告,因此采用经典 **map-reduce**:分批 map 抽要点,汇总 reduce 合成答案。
- `GlobalCommunityContext` 用 `build_community_context`(`single_batch=False`)把报告按 `max_context_tokens` 切成多个批;同一函数在 Local 侧用 `single_batch=True` 只取一批,一个开关分出全局与局部。
- 报告在分批前会先 `shuffle_data`(固定 `random_state=86` 保证可复现),批内按 `occurrence weight` 与 `rank` 降序排序,让高价值报告优先保留。
- map 阶段对每批并发调 LLM(`asyncio.Semaphore` 限流),用 `MAP_SYSTEM_PROMPT` 让模型输出 `points`,每个要点带 0–100 的重要性 `score`,并要求 `[Data: Reports (...)]` 引用。
- reduce 阶段把所有要点按 `score` 降序排序,逐条累加裁剪到 `max_data_tokens`,再用 `REDUCE_SYSTEM_PROMPT` 把高分要点合成连贯答案,流式输出。
- map 的每个调用被拟人化为"analyst",reduce 提示词以"综合多方分析师观点"组织,但要求最终答案不暴露这一内部脚手架。
- 容错贯穿始终:JSON 解析失败、单批异常都退化为 score 0 的占位结果,保证局部失败不拖垮整体;全员 score 0 且未开通识时返回固定 `NO_DATA_ANSWER`。
- 可选的动态社区选择用 `rate_relevancy` 沿社区层级自顶向下评分剪枝,只对相关分支下钻,以少量评分调用换取 map 报告数量的下降。
- `GlobalSearchResult` 把 `build_context`/`map`/`reduce` 三段的 `llm_calls`、`prompt_tokens`、`output_tokens` 分项统计再汇总,成本透明,便于优化。

## 动手实验

1. **实验一:观察分批边界** — 在 `build_community_context` 的分批循环里临时打印每次 `_cut_batch()` 时的 `batch_tokens` 与批内报告数,跑一次 Global Search,确认每批都逼近但不超过 `max_context_tokens`(默认 `12_000`),并数一数总共切出多少批——这就是 map 调用的次数。

2. **实验二:量化 map-reduce 的成本结构** — 调用 `GlobalSearch.search` 后打印返回的 `GlobalSearchResult.llm_calls_categories` 与 `prompt_tokens_categories`,对比 `build_context`、`map`、`reduce` 三项的占比,验证 map 阶段通常是调用次数与 token 的大头,并据此判断动态社区选择是否值得开启。

3. **实验三:验证评分排序与裁剪** — 把 `max_data_tokens` 临时调到一个很小的值(如 `500`),在 `_reduce_response` 裁剪循环里打印被保留与被 `break` 丢弃的要点及其 `score`,确认保留下来的都是高分要点,直观感受"按重要性裁剪到预算"。

4. **实验四:开关动态社区选择** — 分别以 `dynamic_community_selection=False` 与 `True` 运行同一查询,记录两次的 map 调用次数(报告批数)与总 token,并在 `DynamicCommunitySelection.select` 末尾打印 `ratings` 分布,观察评分剪枝筛掉了多少社区、相关报告数下降了多少。

> **下一章预告**:Global 与 Local 各执一端,但都还是"一锤子买卖"。第 13 章我们走进 `drift_search`——它用 primer 起手、action 迭代、state 管理状态,把 Local 与 Global 的优点编织进一个多轮渐进的检索循环;同时一并解构最轻量的 `basic_search`,看看不依赖图谱时 GraphRAG 如何退化成一个朴素的向量 RAG。
