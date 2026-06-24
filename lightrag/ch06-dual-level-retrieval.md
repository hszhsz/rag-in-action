# 第 6 章 双层检索与查询模式

前面五章我们把"写入侧"拆完了：文档被解析、切分、抽取成实体与关系，灌进了图存储、向量库与 KV 存储。本章进入"读取侧"——也是 LightRAG 名字里"Light"与"RAG"真正交汇的地方。它的核心主张只有一句：**一个问题往往同时包含"具体的"与"抽象的"两层信息需求**。比如"特斯拉 2023 年在中国的销量受什么政策影响"，"特斯拉""中国"是具体实体（low-level），"政策影响""销量趋势"则是抽象主题（high-level）。LightRAG 用 LLM 把 query 拆成两组关键词，分别驱动两条检索路径——从实体出发的 local 检索、从关系出发的 global 检索——再把结果融合，这就是"双层检索"（dual-level retrieval）。

这一章我们顺着 `operate.py` 的真实调用链走一遍：`kg_query` 如何抽关键词、如何分发到六种 mode、`_perform_kg_search` 如何并发检索、`_apply_token_truncation` 与 `_build_context_str` 如何在统一 token 预算下拼装出最终交给 LLM 的上下文。读完你应当能回答："为什么 local 从节点查、global 从边查""mix 模式凭什么把三路结果揉到一起还不会爆 token"。

需要先建立一个心智模型：LightRAG 的查询侧是一条**四阶段流水线**，由 `_build_query_context`（第 5024 行）串起，注释里写得明明白白——`Search → Truncate → Merge chunks → Build LLM context`。第一阶段做纯检索（不截断、不格式化），第二阶段按 token 预算砍实体与关系，第三阶段才去拉这些幸存实体/关系对应的原文 chunk，第四阶段把一切拼成字符串。把这四步记在脑子里，后面所有函数都能对号入座。这种"检索与格式化彻底分离"的分层，也是 LightRAG 能用同一套 `_build_context_str` 服务 local/global/hybrid/mix 四种 mode 的原因。

## 六种 query mode：一张分发表

入口是 `QueryParam`（`base.py` 第 83 行）。它的 `mode` 字段用 `Literal["local", "global", "hybrid", "naive", "mix", "bypass"]` 约束取值，**默认值是 `mix`**。注意默认不是 `hybrid` 也不是 `naive`，而是图谱+向量全开的 `mix`——这透露了作者的偏好：默认就要把知识图谱和原始 chunk 一起喂给 LLM。

分发逻辑不在 `operate.py`，而在 `lightrag.py` 的查询入口（约第 2368 行起）。它把六种 mode 归成三类处理函数：

- `local` / `global` / `hybrid` / `mix` → 走 `kg_query`（`operate.py` 第 3786 行），这是知识图谱查询主流程。
- `naive` → 走 `naive_query`（`operate.py` 第 5740 行），纯向量检索，完全不碰图谱。
- `bypass` → 不检索，直接把 query 丢给 LLM（`lightrag.py` 第 2391 行）。

`bypass` 最简单：它用 `role_llm_funcs["query"]` 直接调用模型，`history_messages` 带上对话历史，`enable_cot=True`，返回的 `data` 是空字典，`metadata` 也是空——它是一个"退化为普通对话"的逃生舱，用于明确不需要检索的场景。下面我们把重心放在 `kg_query` 与 `naive_query` 上。

可以把这张分发表当成本章的地图：`bypass` 是检索量为零的极端，`naive` 是只有向量检索的另一端，中间四种 mode 才是真正发生"双层检索"的地带。理解了两端，中间的复杂度才有参照系。

`QueryParam` 上与检索强相关的旋钮还有：`top_k`（local 下表示取多少实体、global 下表示取多少关系）、`chunk_top_k`（最终保留多少 chunk）、三个 token 预算 `max_entity_tokens` / `max_relation_tokens` / `max_total_tokens`、可外部注入的 `hl_keywords` / `ll_keywords`，以及 `enable_rerank`（默认由环境变量 `RERANK_BY_DEFAULT` 决定，未设则为 `true`）。

`local` / `global` / `hybrid` / `mix` 这四种走 `kg_query` 的 mode，区别仅在于"喂哪层关键词、走哪条检索路径"，而抽词、截断、拼装、调 LLM 的下半程完全共用——这正是 LightRAG 把六种 mode 收敛到三个入口函数的底气。

## 第一步：把 query 拆成高/低两层关键词

`kg_query` 拿到 query 后第一件事就是抽关键词（第 3837 行）：

```python
hl_keywords, ll_keywords = await get_keywords_from_query(
    query, query_param, global_config, hashing_kv
)
```

`get_keywords_from_query`（第 4001 行）有个短路：如果调用方已经在 `QueryParam` 里塞了 `hl_keywords` 或 `ll_keywords`，就直接返回，**不再调用 LLM**。这给了高级用户绕过 LLM 抽词、自己指定关键词的能力。否则交给 `extract_keywords_only`（第 4156 行）。

`extract_keywords_only` 的做法是用 `PROMPTS["keywords_extraction"]`（`prompt.py` 第 484 行）让 LLM 一次性吐出两组词。这个 prompt 的设计很有讲究：它明确区分 `high_level_keywords`（"overarching concepts or themes"，捕捉用户核心意图、主题领域）和 `low_level_keywords`（"specific entities or details"，具体实体、专有名词、产品名）。它强制要求输出是纯 JSON（"first character must be `{{`"），并对"hello""asdfghjkl"这类无意义 query 规定返回两个空数组。语言由 `{language}` 控制，但专有名词保留原文。调用时还带上了 `response_format={"type": "json_object"}`（第 4219 行），双保险。

LLM 的回复要经过 `_parse_keywords_payload`（第 4095 行）这道"健壮性闸门"。它处理了现实中各种供应商不老实的情况：回复可能是 pydantic 对象（`model_dump()`）、dict、或字符串；字符串还可能被包在 ```` ```json ```` 代码块里（`_strip_markdown_code_fence`，第 4082 行），或夹带 `<think>` 标签（`remove_think_tags`）。先 `json.loads` 严格解析，失败再退化到 `json_repair.loads` 并打 warning。最后用 `_normalize_keyword_list`（第 4033 行）把每个字段规整成干净的字符串列表——纯字符串会按 `[\n,;]+` 切分，列表则逐项保留（避免把含逗号的合法短语拆碎）。

抽完词后，`kg_query` 有一段值得注意的空值兜底（第 3845-3854 行）：

- `local`/`hybrid`/`mix` 缺 `ll_keywords` 会告警，`global`/`hybrid`/`mix` 缺 `hl_keywords` 会告警；
- **两组都空**时，如果 query 短于 50 字符，就强行把整条 query 当作 `ll_keywords`（"Forced low_level_keywords to origin query"）；否则直接返回 `fail_response`。

这个"短 query 降级为低层关键词"的设计很务实：短问题往往就是一个实体名，抽不出词时直接拿原文做实体向量检索，胜过两手空空。

值得停下来想一想"为什么要让 LLM 抽两层词，而不是直接拿整条 query 去做向量检索"。原因藏在向量检索的局限里：

- 整条 query 的 embedding 是"语义平均"，对一个既问实体又问主题的复合问题，平均向量谁都不像，召回往往两边都浅。
- 把 query 拆成两组关键词，等于做了一次**意图分解**——低层词去命中向量库里那些"实体名/关系描述"的稠密向量，命中率明显更高；高层词去命中"关系/主题"的语义向量，能找到那些跨实体、需要归纳的边。
- 抽词这一步本身被缓存（`extract_keywords_only` 第 4178 行用 `compute_args_hash` 把 mode、文本、语言纳入键），同一问题重复查询时不会反复烧 LLM。

换句话说，关键词抽取是用一次廉价的 LLM 调用，换来后续两条检索路径各自更精准的召回——这是 LightRAG"双层"二字的前提。

## 第二步：双层检索的核心——`_perform_kg_search`

抽好的关键词被拼成逗号分隔的字符串，连同 query 一起传给 `_build_query_context`（第 5024 行），它是一条四阶段流水线，注释写得很清楚：**Search → Truncate → Merge chunks → Build LLM context**。第一阶段就是 `_perform_kg_search`（第 4315 行），这是双层检索的发动机。

它先做一件聪明的事：**批量预计算 embedding**（第 4363 行起）。根据当前 mode，最多需要三种文本的向量——`query`（给 chunk 向量检索用）、`ll_keywords`（给实体向量库用）、`hl_keywords`（给关系向量库用）。它把"本次真正会用到"的文本收集进 `texts_to_embed`，一次 `actual_embedding_func` 调用全部算出来，避免 2-3 次串行 API 往返。`need_ll` 仅在 `mode in ("local","hybrid","mix")` 且有低层词时为真，`need_hl` 同理对应 `("global","hybrid","mix")`——这套布尔门控精确表达了"哪层词喂哪条路径"。

然后是 mode 分发（第 4400 行起），这是双层检索的骨架：

- **`local`**：调 `_get_node_data(ll_keywords, ...)`——从**实体**出发。
- **`global`**：调 `_get_edge_data(hl_keywords, ...)`——从**关系**出发。
- **`hybrid` / `mix`**：两条都走，有低层词就 `_get_node_data`，有高层词就 `_get_edge_data`。
- 额外地，**`mix`** 模式若提供了 `chunks_vdb`，再调一次 `_get_vector_context`（第 4258 行）拉一批纯向量 chunk，并在 `chunk_tracking` 里标记来源为 `"C"`（Chunk）。

`_get_vector_context` 本身很轻：它用 `search_top_k = chunk_top_k or top_k` 去 `chunks_vdb.query` 拉 chunk，只保留含 `content` 字段的结果，给每条打上 `source_type="vector"` 和 `chunk_id`，**不做 rerank、不做截断**（这两件事统一留给下游 `process_chunks_unified`）。它读取了 `cosine_better_than_threshold`——向量库层面的相似度门槛，低于阈值的根本不会返回。这个"检索层只管召回、加工层统一处理"的职责切分，正是前面说的四阶段流水线第一阶段"纯检索"原则的体现。值得强调 `hybrid` 与 `mix` 的唯一区别就在这里：两者都走双层图谱检索，但只有 `mix` 会额外调 `_get_vector_context` 注入一路朴素向量 chunk——所以 `mix` 是 `hybrid` 的超集。

这就解释了 mode 的语义：`local`=低层单层，`global`=高层单层，`hybrid`=双层图谱，`mix`=双层图谱再叠加朴素向量。

`chunk_tracking` 这个数据结构在 `mix` 模式里第一次显出价值。它是 `chunk_id → {source, frequency, order}` 的字典，记录每个 chunk 的来源标记：`mix` 里向量检索来的 chunk 标 `"C"`、frequency 恒为 1、order 是它在向量结果中的名次（第 4448 行）。后续从实体来的 chunk 会标 `"E"`、从关系来的标 `"R"`。到了 `_build_context_str` 末尾（第 4978 行起），它会把每个最终入选 chunk 的来源拼成一行日志，格式形如 `E5/2 R2/1 C1/1`（来源+频次/次序）。这行看似不起眼的日志，是调试"我的答案到底是图谱撑起来的还是向量撑起来的"的关键抓手——它让原本黑盒的融合过程变得可观测。

### local 路径：实体 → 边 → text unit

`_get_node_data`（第 5144 行）拿低层关键词去 `entities_vdb.query` 做实体向量检索，取回 `top_k` 个最相似实体。然后用 `get_nodes_batch` + `node_degrees_batch` **并发**（`asyncio.gather`）批量取节点属性与度数，组装出 `node_datas`（每个带 `entity_name`、`rank`=度数、`created_at`）。接着调 `_find_most_related_edges_from_entities`（第 5204 行）：以这些实体为锚点，用 `get_nodes_edges_batch` 取出它们所有的边，去重后批量取边属性与边度数，最后按 `(rank, weight)` 降序排序。注释点明：**实体按余弦相似度排序，关系按 rank+weight 排序**——两套排序准则，分别贴合"语义相关"和"图谱重要性"。

最终的 text unit 由 `_find_related_text_unit_from_entities`（第 5260 行）产出，但它在第三阶段 `_merge_all_chunks` 里才被调用。它的逻辑是：每个实体的 `source_id` 字段（用 `GRAPH_FIELD_SEP` 分隔）记录了它来自哪些 chunk，函数把这些 chunk 收集、按出现次数计数并去重（保留位置靠前的实体带来的 chunk），再根据 `kg_chunk_pick_method`（`WEIGHT` 或 `VECTOR`）选出 `related_chunk_number` 个。

这里的 `kg_chunk_pick_method`（第 5302 行读自 `global_config`）是一个容易被忽略却影响很大的开关，它决定"一个实体关联了几十个 chunk 时，挑哪几个给 LLM"：

- `WEIGHT`：用 `pick_by_weighted_polling` 做线性梯度加权轮询——出现次数越多（被越多实体共享）的 chunk 越优先。注释（第 5381 行）点明，当 rerank 关闭时，这种方式"更多地把纯 KG 相关 chunk 投给 LLM"，偏向图谱结构。
- `VECTOR`：用 query 的 embedding 对候选 chunk 做余弦相似度排序（第 5343 行），挑最像 query 的。注释（第 5342 行）说这让结果"偏向朴素检索的取向"，即更贴近 query 字面语义。

也就是说，同一条 local/global 检索，挑 chunk 的口味可以在"结构优先"和"语义优先"之间切换——这是 LightRAG 把图谱检索与向量检索哲学融合得最细的地方之一。

### global 路径：关系 → 实体 → text unit

`_get_edge_data`（第 5419 行）是 local 的镜像：拿高层关键词去 `relationships_vdb.query` 做**关系向量检索**，取回 `top_k` 条边，用 `get_edges_batch` 批量补全边属性，**保持向量相似度顺序**（不像 local 那样重排）。然后 `_find_most_related_entities_from_relationships`（第 5478 行）把这些边的两端实体 `src_id`/`tgt_id` 去重收集出来，批量取节点属性（这里不需要度数）。对应的 chunk 由 `_find_related_text_unit_from_relations`（第 5511 行）从边的 `source_id` 里抽取，逻辑与实体侧对称，还会额外用 `entity_chunks` 去重，避免与 local 路径拿到的 chunk 重复。

为什么 local 重排而 global 不重排？因为 local 是"先有实体再扩散到边"，边是二跳产物，需要 rank+weight 把图谱里更核心的关系顶上来；而 global 直接对关系做向量检索，向量相似度本身就是排序依据，无需再动。

把 local 与 global 并排，可以看到它们是一对严格镜像，连函数命名都对仗：

- local 入口 `_get_node_data`（查节点），global 入口 `_get_edge_data`（查边）；
- local 用 `_find_most_related_edges_from_entities`（实体→边），global 用 `_find_most_related_entities_from_relationships`（关系→实体）；
- local 的 chunk 来自 `_find_related_text_unit_from_entities`，global 的来自 `_find_related_text_unit_from_relations`；
- 二者最终都落到"text unit / chunk"——因为不管从实体还是关系出发，LLM 真正要读的还是原文片段，图谱只是导航。

这种对称不是巧合，而是双层检索的结构性要求：low-level 与 high-level 必须是两条独立、等价、可融合的管线，hybrid/mix 才能把它们无缝拼起来。

### round-robin 融合：让两层结果交替出场

`hybrid`/`mix` 同时拿到 local 和 global 两套实体、两套关系后，`_perform_kg_search` 用**轮转合并**（round-robin，第 4456-4510 行）把它们交织起来：循环 `i`，每轮先放 local 的第 `i` 个、再放 global 的第 `i` 个，用 `seen_entities`/`seen_relations` 去重（关系用排序后的端点对做 key）。

这个细节比"先全部 local 再全部 global"高明：轮转保证了即使后面要做 token 截断，被砍掉的也是两层各自的尾部，而不会出现"low-level 全保留、high-level 全被砍"的偏科。融合后返回 `final_entities`、`final_relations`、`vector_chunks`、`chunk_tracking` 和复用的 `query_embedding`。

合并完成后，`_build_query_context`（第 5059 行）还有一处早退判断：若 `final_entities` 和 `final_relations` 都为空，对非 `mix` 模式直接返回 `None`（让上层回 `fail_response`）；唯独 `mix` 模式会再看一眼 `chunk_tracking`——只要向量检索还捞到了 chunk，就继续往下走。这再次印证 `mix` 的兜底性：哪怕图谱一无所获，纯向量那一路仍能撑起一次回答。

## 第三步：统一 token 预算控制

检索拿到的原始数据可能很多，但 LLM 的上下文窗口有限。LightRAG 用一套**三级 token 预算**把它压进窗口，对应 `QueryParam` 的三个字段。

第一级在 `_apply_token_truncation`（第 4525 行）。它把实体和关系分别格式化成精简的 dict（实体保留 `entity`/`type`/`description`，关系保留 `entity1`/`entity2`/`description`），然后调 `truncate_list_by_token_size`：对实体用 `max_entity_tokens` 截断、对关系用 `max_relation_tokens` 截断。一个细节是**计算 token 时会临时剔除 `file_path` 和 `created_at`**（第 4618 行起）——这两个字段会进最终 JSON，但不参与"该保留几条"的预算计算，避免元数据挤占有效信息的额度。截断后，函数再回头从原始数据里筛出"幸存"的实体/关系（`filtered_entities`/`filtered_relations`），供下一阶段取 chunk。

第二级、第三级在 `_build_context_str`（第 4839 行）。它做的是**动态预算分配**：先算出系统 prompt 模板的 token（`PROMPTS["rag_response"]` 填空后）、KG 上下文（实体+关系 JSON）的 token、query 的 token，再留 200 token 的 `buffer_tokens`（给 reference list 和安全余量），然后：

```python
available_chunk_tokens = max_total_tokens - (
    sys_prompt_tokens + kg_context_tokens + query_tokens + buffer_tokens
)
```

也就是说，**chunk 能用的额度 = 总预算减去其它一切已确定占用**。这个"先固定 KG、剩下全给 chunk"的策略，体现了 LightRAG 把图谱视为骨架、chunk 视为可伸缩血肉的取舍。算出 `available_chunk_tokens` 后调 `process_chunks_unified`（在 `utils.py` 第 4335 行）对 chunk 做最终处理。

这套三级预算的层次感值得品味。`max_entity_tokens` 与 `max_relation_tokens` 是**局部硬上限**，在第一级就把实体、关系各自封顶，谁也不能无限膨胀；`max_total_tokens` 是**全局总闸**，在第三级动态计算剩余额度。两者配合的效果是：实体和关系永远占用一个可预测的固定区间，而 chunk 自动填满剩下的空间——预算多就多给原文、预算紧就先保结构。`truncate_list_by_token_size` 的截断是"整条保留或整条丢弃"，不会把一个实体的描述截一半，保证喂给 LLM 的每条 JSON 都是完整的。这也是为什么第一级计算 token 时要剔除 `file_path`/`created_at`：它们是必带的元数据噪声，不该参与"还能塞几条"的核心决策。

## 第四步：chunk 的统一处理——rerank 在这里

`process_chunks_unified`（`utils.py`）是所有路径（kg 与 naive）chunk 的最后一道工序，顺序是固定的五步：

1. **rerank**（第 4362 行）：若 `enable_rerank` 为真且有 query，调 `apply_rerank_if_enabled`，`top_n` 取 `chunk_top_k`。rerank 发生在 token 截断**之前**——先用重排模型把最相关的 chunk 顶上来，再截断，保证砍掉的是真正不相关的。
2. **按分数过滤**：用 `min_rerank_score`（默认 0.5）过滤掉低分 chunk。这是 rerank 的副产品，能直接剔除噪声。
3. **`chunk_top_k` 限流**：超过 `chunk_top_k` 就硬截到前 N 条。
4. **token 截断**：用上一步传入的 `chunk_token_limit`（即 `available_chunk_tokens`）做最终 `truncate_list_by_token_size`。
5. **打编号**：给每个 chunk 加 `id`（`DC1`、`DC2`…）。

把 rerank 放在统一管线里、且置于截断之前，是一个关键工程决策：它意味着无论 local/global/mix/naive 哪条路径，chunk 的质量门槛和预算控制都走同一套逻辑，不会出现某条路径"漏 rerank"。

这五步的排列顺序本身就是一份"如何在有限窗口里塞进最优 chunk"的方法论：先 rerank（让最相关的浮上来）→ 再按分数过滤（砍掉噪声）→ 再 `chunk_top_k` 限流（控制数量上界）→ 再 token 截断（控制体积上界）→ 最后编号。任何两步顺序对调都会变差：若先截断再 rerank，截掉的可能恰是最相关的；若先限流再过滤，可能把名额浪费在低分 chunk 上。LightRAG 把这套顺序固化在一个函数里，等于把"检索质量"这件事做成了不可绕过的公共设施。

## 第五步：拼装 `kg_query_context` 交给 LLM

回到 `_build_context_str`，chunk 处理完后调 `generate_reference_list_from_chunks` 生成引用列表（每个 chunk 关联一个 `reference_id` 和 `file_path`），再把三部分填进 `PROMPTS["kg_query_context"]`（`prompt.py` 第 442 行）模板：

```
Knowledge Graph Data (Entity):       ```json {entities_str} ```
Knowledge Graph Data (Relationship): ```json {relations_str} ```
Document Chunks:                     ```json {text_chunks_str} ```
Reference Document List:             {reference_list_str}
```

实体和关系以 JSON 行的形式（每行一个 `json.dumps`）呈现，chunk 带上 `reference_id` 和可选的 `content_headings`（章节路径，由 `_attach_content_headings` 第 4696 行回填，让 LLM 知道每段文字在原文档的层级位置）。reference list 是 `[reference_id] file_path` 的清单。

`_attach_content_headings` 这步看似边角，却体现了对"chunk 失去上下文"问题的细致处理。它的 docstring 解释了原委：向量库里的 chunk 不存 `heading` 字段，round-robin 合并又把实体/关系侧的 heading 丢掉了，所以这里要按 `chunk_id` 回 `text_chunks_db` 查一次，把父级标题路径（如 `Section 1 → Subsection 1.2`）补回去。而且它在 token 截断**之前**回填（`naive_query` 第 5794 行同理），让 heading 也计入预算——因为 heading 是要进最终 JSON 给 LLM 读的，不能白占额度还不算账。每级标题有字符上限、整条路径再用 `DEFAULT_MAX_SECTION_CONTEXT_TOKENS` 兜底，避免深层目录把一个 chunk 的元数据撑爆。

这套上下文最终塞进 `PROMPTS["rag_response"]`（第 334 行）的 `{context_data}` 占位符。这个回答 prompt 强约束 LLM：只能用 Context 里的信息（"DO NOT invent"），要在文末生成 `### References` 引用段落，每条引用必须真实支撑正文，最多 5 条，引用格式 `* [n] Document Title`。这就是 LightRAG 的 **citation（引证）支持**——它不是事后拼接，而是把 reference 机制写进了 prompt 契约里，靠 `reference_id` 把 chunk 与最终引用绑定。

回到 `kg_query` 主流程（第 3877 行起）：拿到 context 后，根据 `QueryParam` 决定返回什么——`only_need_context` 只返回上下文字符串、`only_need_prompt` 返回拼好的完整 prompt、否则真正调 LLM。它还接了一层 query 缓存：用 `compute_args_hash` 把 mode、query、各种 token 预算、两组关键词、`enable_rerank` 等都纳入哈希键（第 3911 行），命中则复用历史回答。流式与非流式分别返回 `QueryResult` 的 `response_iterator` 或 `content`。

这个缓存键的选取很考究，值得单独点出：它不只哈希 query 文本，还把 `mode`、`response_type`、`top_k`、`chunk_top_k`、三个 token 预算、`hl_keywords_str`、`ll_keywords_str`、`user_prompt`、`enable_rerank`、`enable_content_headings` 全部纳入。换句话说，**任何会改变检索行为或输出形态的参数变化，都会让缓存自动失效**。这避免了一个经典的缓存陷阱——用户把 `mode` 从 `naive` 改成 `mix` 却拿到上一次 naive 的旧答案。同样，非流式回复返回前还会做一轮清洗（第 3980 行），把可能被模型回显的 `sys_prompt`、`query`、`<system>` 标签等剥掉，保证 `content` 是干净的答案。

## naive_query：作为对照的朴素基线

`naive_query`（第 5740 行）是理解前面所有复杂度的最佳对照组——它把图谱全部砍掉。流程极短：`_get_vector_context` 直接拿 query 做 chunk 向量检索（`search_top_k = chunk_top_k or top_k`），可选地 `_attach_content_headings` 回填章节路径，然后算 `available_chunk_tokens`（这里没有 KG 占用，公式只减 `sys_prompt_tokens + query_tokens + buffer_tokens`），同样走 `process_chunks_unified`（含 rerank、过滤、截断），再用 `PROMPTS["naive_query_context"]`（第 469 行）拼上下文——这个模板只有 Document Chunks 和 Reference List，没有任何实体/关系区块。回答 prompt 用 `PROMPTS["naive_rag_response"]`（第 388 行），与 `rag_response` 几乎一样，只是把"Knowledge Graph"措辞去掉了。

`naive_query` 还有一个与 `kg_query` 一致的"无结果即早退"约定（第 5786 行）：若 `_get_vector_context` 一条 chunk 都没召回，直接返回 `None`，由上层转成 `fail_response`。这种"宁可明确说没有，也不硬编一个空上下文给 LLM"的姿态，贯穿了 LightRAG 整个查询侧——`_build_query_context`、`_build_context_str`、`naive_query` 三处都有对应的早退分支，共同保证了"没有证据就不回答"的诚实性。

把 `naive` 和 `mix` 并排看，双层检索的价值就具象了：naive 只有"语义相似的原文片段"，而 mix 额外提供了"结构化的实体定义 + 关系三元组"，让 LLM 既能看到事实碎片、又能看到它们之间的连接骨架。

## 把整条链路串起来

至此可以把 `mix` 模式（默认 mode）的端到端链路完整复述一遍，作为全章的脉络收束：

1. `kg_query` 调 `get_keywords_from_query` → `extract_keywords_only`，用一次 LLM 把 query 拆成 `hl_keywords` 与 `ll_keywords`（命中缓存则跳过）。
2. `_build_query_context` 进入四阶段流水线。阶段一 `_perform_kg_search` 批量预算 embedding，对低层词走 `_get_node_data`（实体路径）、高层词走 `_get_edge_data`（关系路径），`mix` 再叠加 `_get_vector_context`，最后 round-robin 融合。
3. 阶段二 `_apply_token_truncation` 按 `max_entity_tokens`/`max_relation_tokens` 砍实体与关系。
4. 阶段三 `_merge_all_chunks` 拉取幸存实体/关系对应的 chunk，与向量 chunk 合并。
5. 阶段四 `_build_context_str` 动态算出 chunk 额度，经 `process_chunks_unified`（rerank→过滤→限流→截断）后，填进 `kg_query_context` 模板。
6. `kg_query` 把上下文塞进 `rag_response`，查缓存或调 LLM，返回 `QueryResult`。

每一步的输入都是上一步的输出，参数全程由 `QueryParam` 一个对象贯穿——这就是 LightRAG 查询侧的全貌。

还有一个工程层面的对照点：`naive_query` 和 `kg_query` 虽然检索逻辑天差地别，却共享了下游的两大部件——`process_chunks_unified`（rerank/过滤/截断）和 `generate_reference_list_from_chunks`（引用列表生成）。这意味着无论走哪条路径，chunk 的质量门槛、token 控制、citation 机制都是一致的。这种"上游分叉、下游收敛"的结构，让 LightRAG 在支持六种 mode 的同时，把"如何把 chunk 喂好、如何引证"这件事只实现了一遍。读者在扩展自己的检索策略时，也只需产出一批带 `content`/`chunk_id`/`file_path` 的 chunk，就能免费复用这套下游管线。

## 本章小结

- LightRAG 的核心思想是**双层检索**：用 LLM 把 query 拆成 `high_level_keywords`（抽象主题）和 `low_level_keywords`（具体实体），分别驱动两条检索路径。
- 六种 mode 归为三类分发：`local/global/hybrid/mix` → `kg_query`，`naive` → `naive_query`，`bypass` → 直连 LLM；`QueryParam.mode` 默认是 `mix`。
- `local` 从**实体**出发（`_get_node_data`：实体向量检索→找相关边→找 text unit，实体按相似度、关系按 rank+weight 排序）；`global` 从**关系**出发（`_get_edge_data`：关系向量检索→找相关实体→找 text unit，保持向量顺序）。
- `_perform_kg_search` 用一次批量 embedding 预计算服务多条路径，并用 round-robin 轮转合并 local/global 结果，保证截断时两层不偏科。
- 关键词抽取由 `extract_keywords_only` + `_parse_keywords_payload` 完成，后者对代码块、`<think>` 标签、JSON 损坏做了多重容错；短 query 抽不出词时降级为把原文当低层关键词。
- 统一 token 预算分三级：`max_entity_tokens`/`max_relation_tokens` 在 `_apply_token_truncation` 截断实体与关系，`max_total_tokens` 在 `_build_context_str` 里"先固定 KG、剩余全给 chunk"地动态分配。
- rerank 集中在 `process_chunks_unified`，位于 token 截断**之前**，并配合 `min_rerank_score`（默认 0.5）过滤与 `chunk_top_k` 限流，所有路径共用同一管线。
- 最终上下文按 `PROMPTS["kg_query_context"]` 拼成"实体 JSON + 关系 JSON + chunks + reference list"，citation 机制通过 `reference_id` 与 `rag_response` prompt 契约落地。
- `naive_query` 是去掉图谱的对照基线，仅向量检索 + `naive_query_context`，凸显双层检索"事实碎片 + 结构骨架"的增量价值。

## 动手实验

下面四个实验都建议在 `only_need_context=True` 或加临时 `logger` 打印的前提下进行，这样能直接观察中间产物而不必每次都烧 LLM 生成最终答案；跑完记得把临时打印移除。

1. **实验一：观察两层关键词的真实拆分** — 在 `extract_keywords_only`（`operate.py` 第 4222 行）解析出 `hl_keywords`/`ll_keywords` 后加一行打印，用一个"具体+抽象"的复合问题（如"苹果公司近年来的研发投入趋势")发起 `mix` 查询，确认低层词落到实体（"苹果公司"）、高层词落到主题（"研发投入趋势"）。再换一个 50 字以内的纯实体问题，观察"两层皆空时降级为低层关键词"的兜底分支是否触发。
2. **实验二：对比 local / global / mix 的检索产物** — 同一问题分别用三种 mode 调用，设 `only_need_context=True`，对比返回的 `kg_query_context`：观察 local 的实体区块更丰富、global 的关系区块更丰富、mix 两者兼具且多出向量 chunk（`chunk_tracking` 标记 `C`）。顺便在日志里找 `Final chunks S+F/O` 这一行，对照 `E/R/C` 来源标记理解融合比例。
3. **实验三：压测 token 预算分配** — 把 `QueryParam.max_total_tokens` 调到很小（如 2000），在 `_build_context_str`（第 4916 行）打印 `available_chunk_tokens`，观察 KG 上下文吃满后留给 chunk 的额度如何被压缩甚至变负，理解三级预算的优先级；再单独调小 `max_entity_tokens`，看实体区块是否先被砍。
4. **实验四：开关 rerank 看顺序与过滤** — 分别设 `enable_rerank=True/False` 跑同一 `naive` 查询，在 `process_chunks_unified`（`utils.py` 第 4362、4374 行）打印 rerank 前后的 chunk 顺序与 `min_rerank_score` 过滤掉的数量，验证 rerank 确实在截断前改变了 chunk 排序，并确认低于 0.5 分的 chunk 被剔除。

> **下一章预告**：本章我们假设图存储、向量库、KV 存储"就在那里"可被检索调用。第 7 章将揭开这层抽象，深入 LightRAG 的**可插拔存储后端**——同一套 `BaseGraphStorage`/`BaseVectorStorage`/`BaseKVStorage` 接口如何对接 NetworkX、Neo4j、Postgres、Milvus、Faiss 等十余种实现，以及存储工厂如何按配置动态绑定。
