# 第 8 章 社区报告生成 Community Reports

第 7 章里，Leiden 算法把实体图切成了一棵层次社区树：每个社区记录了自己的 `level`、`parent`、`children`，以及隶属其下的实体集合。但社区本身只是一组节点编号，对人和对检索器都不可读。GraphRAG 真正用于"全局问答"的语料，是接下来这一步产出的——把每个社区翻译成一份**结构化的自然语言报告**，含标题、摘要、重要性评分与若干条洞察。这一章我们就来解构 `create_community_reports` 工作流。

社区报告是整条 GraphRAG 索引管线里最"重"的 LLM 环节之一：它要为每个层级、每个社区各调用一次大模型。难点不在于"调一次模型生成一段话"，而在于两件事：其一，如何把社区内的实体、关系、协变量按重要性拼装成一段受 token 预算约束的上下文；其二，当一个父社区的局部上下文塞不下时，如何用已经生成好的子社区报告来**替代**原始细节。后者是 GraphRAG 能在任意规模图上"自底向上"完成报告生成的关键，本章会重点讲清楚。

## 工作流入口与依赖装配

工作流定义在 `index/workflows/create_community_reports.py`。`run_workflow` 通过 `DataReader` 读入三张主表——`relationships()`、`entities()`、`communities()，并在 `config.extract_claims.enabled` 且输出表里存在 `covariates` 时附加读入 `claims`（即协变量）。随后它从配置中取出补全模型、tokenizer 与提示词，调用核心函数 `create_community_reports`，把结果写回输出表 `community_reports`。

提示词来自 `config.community_reports.resolved_prompts()`，注意这里取的是 `prompts.graph_prompt`——基于图结构的提示词；文本变体取的则是 `prompts.text_prompt`。两个关键预算量也在此传入：`max_input_length`（上下文最大 token 数，默认 `8000`）与 `max_length`（报告最大长度，默认 `2000`，对应函数参数 `max_report_length`）。这两个默认值定义在 `config/defaults.py` 的 `CommunityReportDefaults` 中。

`create_community_reports` 本身是一段清晰的流水线：先 `explode_communities` 把社区成员展开成节点，再 `_prep_nodes`/`_prep_edges`/`_prep_claims` 给每行造出结构化的 details 字段，然后 `build_local_context` 拼装局部上下文，交给 `summarize_communities` 逐层生成，最后用 `finalize_community_reports` 终化为标准输出表。

## 把社区成员展开成带细节的节点

`explode_communities`（`explode_communities.py`）做的是一次关系连接：把 `communities` 表按 `entity_ids` 列 `explode` 成一行一个实体，再与 `entities` 表按 `id == entity_ids` 左连接，最后用 `nodes[COMMUNITY_ID] != -1` 过滤掉未归属任何社区的游离节点（`-1` 是哨兵值）。结果是一张"实体 × 所属社区"的长表，同一实体可能因属于不同 `level` 的社区而出现多次。

`_prep_nodes` 与 `_prep_edges` 紧接着为每行物化一个字典列：节点的 `NODE_DETAILS` 收纳 `SHORT_ID`、`TITLE`、`DESCRIPTION`、`NODE_DEGREE`；边的 `EDGE_DETAILS` 收纳 `SHORT_ID`、源、目标、`DESCRIPTION`、`EDGE_DEGREE`。缺失的描述统一用 `"No Description"` 兜底。`SHORT_ID`（即 `human_readable_id`）很重要——它就是提示词 Grounding Rules 里要求 LLM 引用的"record id"，比如 `[Data: Entities (5, 7); Relationships (23)]`。若启用协变量，`_prep_claims` 同理造出 `CLAIM_DETAILS`，含主语、类型、状态、描述。

## 局部上下文拼装：按度数排序、按预算截断

真正把零散字段拼成一段文本的，是 `graph_context/context_builder.py` 的 `build_local_context`。它逐 `level` 调用 `_prepare_reports_at_level`：先取出该层的节点集合 `nodes_set`，只保留两端都落在集合内的边；把边按源、目标聚合回节点行，再按 `COMMUNITY_ID` 把所有节点的 `ALL_CONTEXT`（节点 details + 边 details + 可选 claim details）聚成列表。每个社区拿到一份待排序的上下文字典列表后，交给 `parallel_sort_context_batch`，逐社区调用 `sort_context` 生成最终的 `CONTEXT_STRING`，同时算出 `CONTEXT_SIZE` 与一个布尔标志 `CONTEXT_EXCEED_FLAG`（`CONTEXT_SIZE > max_context_tokens`），后者标记这个社区是否"超预算"。

`sort_context`（`graph_context/sort_context.py`）的逻辑值得细看，它体现了 GraphRAG 选材的优先级。它先把所有边按"度数降序、ID 升序"排序——`key=lambda x: (-edge_degree, edge_id)`，即优先纳入连接最密集的关系，因为高度数的关系往往串起社区里最核心的实体。然后它**增量地**逐条加入边，并把边两端的节点与相关 claim 顺带带入，每加一条就用 `_get_context_string` 重新拼出 CSV 文本（分 `-----Entities-----`、`-----Claims-----`、`-----Relationships-----` 三段），一旦 `tokenizer.num_tokens(...) > max_context_tokens` 就 `break`。这种"加到撑为止"的贪心策略保证了在固定预算下尽量多地纳入最重要的信息；若一条都加不下，函数末尾仍会返回一个尽力而为的字符串而非空串。

## 层级化生成与子社区报告替代机制

`summarize_communities`（`summarize_communities.py`）是层级生成的总调度。它的关键设计是**自底向上**：`get_levels`（`utils.py`）用 `sorted(levels, reverse=True)` 把层级**降序**排列，所以最先处理的是数值最大的 level，也就是 Leiden 树最细粒度的叶子社区，最后才到最粗的根社区。这个顺序是替代机制能成立的前提——粗粒度社区生成时，它的细粒度子社区报告必须已经存在。

调度过程是这样的：先用 `communities.explode("children")` 构造 `community_hierarchy`（三列：`community`、`level`、`sub_community`）。然后对每个 level 调用传入的 `level_context_builder`（图路径下是 `build_level_context`），传入**当前已生成的全部 reports**（`pd.DataFrame(reports)`）；该函数返回这一层每个社区最终要喂给 LLM 的上下文。随后用 `derive_from_rows` 并发地对每行调用 `run_generate → _generate_report → run_extractor`，把生成的报告 `extend` 进 `reports`，供更上一层使用。

替代机制集中在 `build_level_context`。它把本层社区按 `CONTEXT_EXCEED_FLAG` 分成"合规"（`valid_context_df`）与"超预算"（`invalid_context_df`）两组。合规组直接用其局部上下文。超预算组则尝试**用子社区报告替换原始局部细节**：

- 若此时还没有任何报告（`report_df` 为空，发生在最细的底层，那里没有子社区），就退化为 `_sort_and_trim_context`——重新排序并裁剪局部上下文硬塞进预算，把 `CONTEXT_EXCEED_FLAG` 置回 `False`。
- 否则，`_get_subcontext_df(level + 1, ...)` 取出下一层（即子社区）的局部上下文与它们已生成的 `FULL_CONTENT` 报告，`_get_community_df` 把每个超预算父社区对应的所有子社区上下文聚到一起，调用 `_build_mixed_context` 生成混合上下文。极少数仍无法替代的记录（`remaining_df`）再走一次 `_sort_and_trim_context` 兜底。

混合的细节在 `build_mixed_context.py`。它把候选子社区按 `CONTEXT_SIZE` **降序**排列——从最大的子社区开始替换，因为替换大块最能腾出空间。它维护两个集合：`substitute_reports`（用报告全文替代）与 `final_local_contexts`（无报告的子社区只能退回用其局部上下文）。每替换一个子社区报告，就把"剩余子社区的局部上下文 + 已替代的报告"重新 `sort_context` 一遍，一旦总 token 数落入 `max_context_tokens` 就停止，`exceeded_limit` 置 `False`。如果连"全部子社区都用报告替代"还超预算，最后的兜底是只往里塞 `substitute_reports` 的 CSV，加到放不下为止。这样无论社区多大，父社区总能拿到一段在预算内、且以子社区摘要为主干的上下文——这正是层次摘要"压缩信息向上传递"的体现。

## 结构化报告：JSON schema、解析与容错

单个社区的报告由 `CommunityReportsExtractor`（`community_reports_extractor.py`）生成。它把上下文文本与 `max_report_length` 填进提示词的 `{input_text}`、`{max_report_length}` 占位符（常量 `INPUT_TEXT_KEY`、`MAX_LENGTH_KEY`），然后以 `response_format=CommunityReportResponse` 调用 `completion_async`——即强制 LLM 按 Pydantic schema 输出 JSON。

`CommunityReportResponse` 规定了五个字段：`title`、`summary`、`findings`（每条含 `summary` + `explanation` 的 `FindingModel` 列表）、`rating`（0–10 的浮点重要性评分）、`rating_explanation`。提示词 `prompts/index/community_report.py` 的 `COMMUNITY_REPORT_PROMPT` 把这套 schema 连同 Grounding Rules、一个完整示例都写死在模板里：报告要求 5–10 条洞察，每条引用必须形如 `[Data: Entities (5, 7); Relationships (23)]` 且单次引用不超过 5 个 id（多则加 `+more`）；`IMPACT SEVERITY RATING` 被明确定义为"社区重要性的打分"，这正是后续全局检索排序的依据。

容错分两层。`CommunityReportsExtractor.__call__` 包裹 try/except，任何异常都记日志并通过 `on_error` 回调上报，返回 `structured_output=None`。上层 `run_extractor` 拿到结果后，若 `report is None` 就告警并返回 `None`，并整体再包一层 try/except；这些 `None` 在 `summarize_communities` 里被 `[lr for lr in local_reports if lr is not None]` 过滤掉。也就是说，**单个社区生成失败不会拖垮整层**，只是少一份报告。成功时则组装成 `CommunityReport` TypedDict，把 `report.rating` 存为 `rank`、`report.model_dump_json(indent=4)` 存为 `full_content_json`，并用 `_get_text_output` 把标题/摘要/各 finding 渲染成一段 Markdown 存入 `full_content`。

## 终化输出表

`finalize_community_reports.py` 把生成的报告与 `communities` 表按 `community` 左连接，补回 `parent`、`children`、`size`、`period` 等共享字段，将 `community` 转为整型并作为 `human_readable_id`，再用 `gen_sha512_hash(row, ["full_content"])` 基于报告全文生成稳定 `id`。最终只保留 `COMMUNITY_REPORTS_FINAL_COLUMNS` 中的列：`id`、`human_readable_id`、`community`、`level`、`parent`、`children`、`title`、`summary`、`full_content`、`rating`、`explanation`（即 `rating_explanation`）、`findings`、`full_content_json`、`period`、`size`。这张表就是第 11、12 章里 Local/Global Search 直接消费的产物。

## text 变体：基于 text unit 而非图结构

`create_community_reports_text.py` 是同一目标的另一条路径。它不读 `relationships`，而是读 `text_units`，并改用 `text_unit_context/context_builder.py` 的 `build_local_context`/`build_level_context`。差异在上下文从哪里来：图路径用"实体+关系+协变量"拼上下文，文本路径用**社区成员实体所引用的原始文本块**拼上下文。

`prep_text_units.py` 先算出每个 text unit 在社区内的"实体度数和"（把该文本块涉及的、属于此社区的节点的 `NODE_DEGREE` 相加，记为 `ENTITY_DEGREE`），文本路径的 `sort_context.py` 再据此**按 `ENTITY_DEGREE` 降序**纳入文本块、按预算截断，拼成 `-----SOURCES-----`（文本块）与 `----REPORTS-----`（子社区报告替代）两段。其层级替代机制与图路径同构，同样复用 `build_mixed_context`。

两条路径的取舍很直接：图路径让报告聚焦于实体间的结构性关系，引用粒度细到单个实体/关系 id，是默认与典型选择；文本路径则把原文片段直接喂给模型，保留更多上下文细节与措辞，适合关系抽取不完整、或更看重原文证据的场景。二者输出的 `community_reports` 表 schema 完全一致，下游检索无需区分。

## 设计动机回顾

把这一章串起来看，社区报告生成的几个设计决策都服务于"全局问答"这个最终目标。报告是 Global Search **map 阶段的语料**：全局问题无法靠向量召回几个文本块回答，于是 GraphRAG 预先把全图浓缩成一批分层报告，查询时对相关报告并行"map"出局部答案再"reduce"汇总。`rating` 字段让检索能按社区重要性排序、优先纳入高影响社区。而层级结构让同一个问题能在不同粒度上聚合——粗粒度报告给全局轮廓，细粒度报告给具体细节，子社区替代机制则保证这种层级摘要在任意图规模下都能在 token 预算内完成。

## 本章小结

- `create_community_reports` 工作流为每个层级、每个社区生成一份结构化报告，输入是社区层级树 + 实体 + 关系（+ 可选协变量），输出写入 `community_reports` 表。
- `explode_communities` 把社区成员展开为带 `NODE_DETAILS`/`EDGE_DETAILS`/`CLAIM_DETAILS` 的节点行，`SHORT_ID` 即提示词要求 LLM 引用的 record id。
- 局部上下文由 `sort_context` 按"边度数降序"贪心增量拼装，受 `max_input_length`（默认 `8000`）约束，超出即 `break`，并以 `CONTEXT_EXCEED_FLAG` 标记超预算社区。
- 生成顺序自底向上：`get_levels` 把 level 降序排列，先处理最细粒度叶子社区，确保父社区生成时子社区报告已就绪。
- 替代机制是核心：父社区上下文超预算时，`build_level_context` + `build_mixed_context` 从最大的子社区开始，用其已生成的 `full_content` 报告替代原始局部细节，直至落入预算。
- 结构化输出由 `CommunityReportResponse` 强制约束，含 `title`、`summary`、`rating`（0–10 重要性）、`rating_explanation`、`findings`（5–10 条），用 `response_format` 走 JSON 模式生成。
- 容错是分层的：单个社区生成异常返回 `None` 并被过滤，不影响同层其它社区与整体管线。
- `finalize_community_reports` 用全文 SHA-512 生成稳定 `id`，补回 `parent`/`children`/`size`/`period`，输出固定列集供下游检索消费。
- `create_community_reports_text` 是基于 text unit 的变体，用原始文本块（按 `ENTITY_DEGREE` 排序）替代图结构上下文，schema 与图路径一致。
- `rating` 与层级结构共同服务于 Global Search 的 map-reduce：按重要性排序、按粒度聚合，是全局问答的语料基础。

## 动手实验

1. **实验一：追踪一份报告的诞生** — 在本地数据上运行索引到 `create_community_reports` 阶段，导出 `community_reports` 表，挑一个 `level` 最大（最细）的社区，对照 `explode_communities` → `sort_context` → `CommunityReportsExtractor` 的链路，手工还原它的 `CONTEXT_STRING` 大致包含哪些实体和关系，再读它的 `full_content` 验证 LLM 是否真的引用了这些 record id。

2. **实验二：触发替代机制** — 把 `config.community_reports.max_input_length` 调到一个很小的值（如 `1500`），重跑工作流，观察粗粒度社区的报告。在 `build_mixed_context` 里加日志打印 `substitute_reports` 与 `final_local_contexts` 的规模，确认父社区确实开始用子社区报告全文替代原始细节，并体会"从最大子社区开始替换"的效果。

3. **实验三：对比图路径与文本路径** — 在同一份数据上分别跑 `create_community_reports` 与 `create_community_reports_text`，对同一个 `community` 比较两份报告的 `summary` 与 `findings`。归纳：文本路径是否保留了更多原文措辞？图路径的引用是否更结构化（实体/关系 id）？据此判断你的语料更适合哪条路径。

4. **实验四：解析容错演练** — 临时修改 `community_report.py` 提示词，把 Return JSON 的格式示例删掉一部分制造解析失败，重跑并观察日志中 `error generating community report` / `No report found for community` 的出现，确认失败社区被 `summarize_communities` 的 `None` 过滤掉、其它社区报告照常产出，体会"局部失败不拖垮整层"的容错设计。

> **下一章预告**：报告与图谱都已就绪，但检索还需要"向量"这把钥匙。第 9 章《向量嵌入与协变量》将解构 `generate_text_embeddings` 如何为实体描述、文本块、社区报告批量生成嵌入并写入 `graphrag-vectors` 向量库，以及 `extract_covariates` 如何从文本中抽取协变量（claims）——也就是本章里那个可选的 `CLAIM_DETAILS` 究竟从何而来。
