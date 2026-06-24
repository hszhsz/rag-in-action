# 第 6 章 描述摘要与图谱终化 Summarize & Finalize

第 5 章我们看到 `extract_graph` 如何让 LLM 从每个文本单元里抽出一批实体与关系。但抽取是逐文本单元进行的:同一个实体「Microsoft」可能在十个文本单元里各被描述一次,于是它在抽取结果里挂着十条零散、彼此重复又互相矛盾的 `description`;同一对节点之间的关系也是如此。这样的图能用,但很「毛糙」——节点描述支离破碎,边里塞着大量低权重噪声,节点度数还没算出来,连稳定的 `id` 都没有。

本章解构两件把毛糙图打磨成可用图的事。一是 `summarize_descriptions`:用 LLM 把同一实体/关系的多条描述合并成一段连贯摘要,并解决超长输入的分批问题。二是「终化」(finalize)与「剪枝」(prune):`finalize_graph` 给节点算度数、补 `human_readable_id` 和 `id`、对齐最终列结构;`prune_graph` 按度数、频次、边权阈值砍掉噪声节点和弱边,为下游的社区检测瘦身。我们也会顺带看一眼不用 LLM、纯 NLP 共现建图的 `build_noun_graph` 路径,作为抽取链路的对照。

## 一、为什么要单独有「摘要」这一步

抽取产物里,实体表每行的 `description` 是一个**列表**,关系表也是。`extract_graph` 工作流在 `get_summarized_entities_relationships` 里调用 `summarize_descriptions`,把列表压成一段文本,再 `merge` 回实体/关系表(`extract_graph.py` 中先 `drop(columns=["description"])` 再 merge 摘要列)。换句话说,摘要不是可选的美化,而是抽取链路的收尾环节:它把「多条原始描述」这种中间形态,转换成下游嵌入、社区报告都能直接消费的「单段描述」。

把它拆成独立操作而非塞进抽取里,有两个工程上的好处。其一是**可缓存、可重跑**:摘要走独立的模型实例 `model_instance_name`(默认 `"summarize_descriptions"`),缓存按这个名字分区(`extract_graph.py` 里 `cache=context.cache.child(...)`),抽取改了不必重摘,反之亦然。其二是**关注点分离**:抽取关心「从文本里找出图」,摘要关心「把图里的描述写好」,两者用不同的 prompt、不同的长度预算。

## 二、summarize_descriptions:多条描述如何合并

入口 `summarize_descriptions(entities_df, relationships_df, ...)` 同时处理节点和边。它对每个节点构造一个协程:

```python
node_futures = [
    do_summarize_descriptions(
        str(row.title),
        sorted(set(row.description)),   # 去重 + 排序
        ticker, semaphore,
    )
    for row in nodes.itertuples(index=False)
]
node_results = await asyncio.gather(*node_futures)
```

注意 `sorted(set(row.description))`:多条描述先**去重再排序**。去重避免重复内容浪费 token,排序则让相同输入集合产生稳定的 prompt(从而缓存命中、结果可复现)。边的处理对称,只是 `id` 是 `(source, target)` 二元组。并发由 `asyncio.Semaphore(num_threads)` 限流,每完成一个就 `ticker(1)` 推进进度条。

真正的合并逻辑在 `SummarizeExtractor`(`description_summary_extractor.py`)。它的 `__call__` 有一个关键的「短路」分支:

```python
if len(descriptions) == 0:
    result = ""
elif len(descriptions) == 1:
    result = descriptions[0]
else:
    result = await self._summarize_descriptions(id, descriptions)
```

**只有当一个实体/关系拥有多于一条描述时,才会真正调用 LLM。** 单条描述直接原样返回,零条返回空串。这是显著的成本优化:大量只出现一次的长尾实体根本不触发摘要请求。

## 三、超长输入:分批滚动摘要

当描述很多、加起来超过模型输入窗口时,不能一股脑塞给 LLM。`_summarize_descriptions` 用一个「累积—触发—重置」的循环解决这个问题。先算出可用 token 预算——总预算减去 prompt 本身占用:

```python
usable_tokens = self._max_input_tokens - self._tokenizer.num_tokens(
    self._summarization_prompt
)
```

然后逐条把描述塞进 `descriptions_collected`,每塞一条就从 `usable_tokens` 里扣掉它的 token 数。触发摘要的条件有两个:

```python
if (usable_tokens < 0 and len(descriptions_collected) > 1) or (
    i == len(descriptions) - 1
):
    result = await self._summarize_descriptions_with_llm(
        sorted_id, descriptions_collected
    )
    if i != len(descriptions) - 1:
        descriptions_collected = [result]
        usable_tokens = (
            self._max_input_tokens
            - self._tokenizer.num_tokens(self._summarization_prompt)
            - self._tokenizer.num_tokens(result)
        )
```

读懂这段是本节重点。**第一个条件**:缓冲区装满(`usable_tokens < 0`)且至少攒了两条,就先摘要一批。**第二个条件**:遍历到最后一条,无论缓冲是否满都要做一次收尾摘要。关键在触发后那个 `if i != len(descriptions) - 1` 分支——如果后面还有描述,就把刚得到的摘要 `result` 当作**新一轮的第一条描述**重新入列,并把预算重置为「总预算减去 prompt 减去这段已摘要文本」。

这就是**滚动归并**(rolling/iterative summarization):每批的摘要带着进入下一批,继续和剩余描述一起被摘要,直到所有描述都消化完。它保证无论一个实体有多少条描述,都能在固定的输入窗口内逐步收敛成一段;代价是描述特别多时会发起多次串行 LLM 调用。注意 `len(descriptions_collected) > 1` 这个保护:它防止单独一条超长描述把 `usable_tokens` 打成负数后陷入「装一条就触发、再装一条又触发」的退化——至少要两条才肯把一批送出去。

## 四、摘要 prompt 与长度上限

实际发往模型的内容由 `_summarize_descriptions_with_llm` 拼装,把三个占位符填进 prompt:`entity_name`、`description_list`(都用 `json.dumps(..., ensure_ascii=False)` 序列化,保证中文等非 ASCII 字符不被转义)、以及 `max_length`。默认 prompt 是 `SUMMARIZE_PROMPT`(`prompts/index/summarize_descriptions.py`),核心指令是:把所有描述拼成一段全面的描述、确保涵盖每条信息、**遇到矛盾要消解成一段连贯文本**、用第三人称书写并带上实体名,最后「Limit the final description length to {max_length} words」。

长度上限分两类,默认值在 `config/defaults.py`:`max_length`(摘要输出词数上限)默认 `500`,以 prompt 文字形式约束模型;`max_input_tokens`(输入预算)默认 `4000`,以代码硬切分批保证不超窗。前者是「软约束」(靠 LLM 听话),后者是「硬约束」(靠分批循环兜底)。配置项在 `SummarizeDescriptionsConfig` 里暴露,`resolved_prompts()` 支持用户用自定义 prompt 文件覆盖默认模板。

## 五、finalize_graph:把图「终化」成数据模型

抽取+摘要得到的实体/关系表还缺几样东西:节点度数、稳定主键、对外可读的序号。`finalize_graph` 工作流补齐这些。它打开 `entities`、`relationships` 两张输出表,先流式扫一遍关系算出**度数表**:

```python
async for row in relationships_table:
    lo, hi = sorted((row["source"], row["target"]))
    if (lo, hi) not in seen:
        seen.add((lo, hi))
        degree[lo] += 1
        degree[hi] += 1
```

`_build_degree_map` 把每条边归一化成无向对 `(lo, hi)` 并去重,再给两端各加一度——这正是 NetworkX 无向图的度定义,但全程不物化 DataFrame,而是流式累加进 `Counter`。这与独立操作 `compute_degree`(`graphs/compute_degree.py`)的语义一致(后者也先 `min/max` 归一、`drop_duplicates` 再用 `value_counts` 相加)。

拿到 `degree_map` 后,`finalize_entities` 与 `finalize_relationships` 各自流式回写。实体侧按 `title` 去重,给每行补上 `degree`、自增的 `human_readable_id`、以及 `str(uuid4())` 作为 `id`,最后按 `ENTITIES_FINAL_COLUMNS` 投影出最终列序。关系侧按 `(source, target)` 去重,补上 `combined_degree = degree[source] + degree[target]`(两端度数之和,衡量边的「重要性」)、`human_readable_id`、`id`,按 `RELATIONSHIPS_FINAL_COLUMNS` 投影。

两套最终列在 `data_model/schemas.py` 里写死:实体是 `[id, human_readable_id, title, type, description, text_unit_ids, frequency, degree]`,关系是 `[id, human_readable_id, source, target, description, weight, combined_degree, text_unit_ids]`。**这里值得点明:本版本的 finalize 只做度数与主键补齐,没有图嵌入(node2vec)与二维布局(UMAP)坐标**——这些早期版本里用于可视化的字段在当前 3.1.0 代码中并不在终化产物里,最终列结构里也没有坐标列。理解这点能避免按旧资料的想象去找不存在的字段。

两种 `id` 各有分工:`uuid4` 生成的 `id` 是全局唯一主键,供跨表关联(社区、文本单元、嵌入都用它指向实体);`human_readable_id` 是从 0 递增的小整数,供 prompt、报告里给人/给 LLM 引用时短小好读。可选地,若 `config.snapshots.graphml` 打开,工作流还会把关系导出成 GraphML 快照便于外部工具查看。

## 六、prune_graph:按阈值剪枝降噪

抽取出的图往往「又大又脏」:海量只出现一两次的长尾节点、大量低权重的偶然共现边。这些噪声会让后续的 Leiden 社区检测既慢又分不清主次。`prune_graph` 操作(`operations/prune_graph.py`)按一组统计阈值做减法,顺序很讲究:

1. **算度**:`compute_degree` 得到度数表,并把孤立实体(度为 0)也补进 `degree_map`,确保阈值在与原来相同的节点总体上计算。
2. **去 ego 节点**:若 `remove_ego_nodes`(默认 `True`),移除度数最高的那个节点——它往往是「公司/系统」这种连接一切、几乎没有区分度的超级节点。
3. **按度剪**:度数 `< min_node_degree`(默认 `1`)的节点删掉;若设了 `max_node_degree_std`,用 `mean + std_trim * std` 算出上界,删掉度数过高的节点。
4. **按频次剪**:在度剪幸存的实体上,频次 `< min_node_freq`(默认 `2`)的删掉;若设了 `max_node_freq_std` 同理删掉频次过高的。注释特意说明频次阈值是在「度剪之后存活的节点集合」上算的,以复现 NetworkX 逐步删点的行为。
5. **过滤边**:只保留两端都存活的边;再按 `min_edge_weight_pct`(默认 `40`)取边权的百分位数作下界,砍掉权重低于该百分位的弱边——这是个**相对阈值**,自动适应不同语料的权重分布。
6. **可选 LCC**:若 `lcc_only`,只保留最大连通分量,丢弃边缘碎片子图。

默认配置(`min_node_freq=2`、`min_node_degree=1`、`min_edge_weight_pct=40`、`remove_ego_nodes=True`)的总体倾向是:删掉只出现一次的偶发实体、删掉度数最高的「万能」节点、砍掉权重最低的 40% 弱边。工作流 `prune_graph.py` 在剪完后做了硬校验——如果实体或关系被剪空,直接 `raise ValueError`,因为空图无法支撑后续流程。

剪枝对下游的意义不止「省钱」。社区检测的质量高度依赖图的信噪比:保留长尾节点会制造大量琐碎的单点社区,保留弱边会让本该分开的社区被偶然共现「粘」在一起。先剪枝再聚类,等于先把图的骨架显影出来,Leiden 才能在干净的结构上找到有意义的社区层级。

## 七、build_noun_graph:不用 LLM 的对照路径

抽取实体关系不一定非要 LLM。GraphRAG 还提供 `extract_graph_nlp` 工作流,底层是 `build_noun_graph`——纯 NLP、零 LLM 调用。它的逻辑朴素而有效:对每个文本单元用名词短语抽取器(spaCy/正则/句法解析,见 `np_extractors/` 下的 `regex_extractor`、`syntactic_parsing_extractor` 等,由 `factory.create_noun_phrase_extractor` 按配置选择)提取名词短语作为节点,记录每个短语出现在哪些文本单元(`title_to_ids`),节点的 `frequency` 就是它出现的文本单元数。

边则来自**共现**:`_extract_edges` 把同一文本单元里出现的名词短语两两配对(`combinations(sorted(set(titles)), 2)`),边的 `weight` 是共现的文本单元数;若 `normalize_edge_weights` 打开,再用 `calculate_pmi_edge_weights`(点互信息)归一化权重,抑制高频词带来的虚假强连接。由于抽取是 CPU 密集的 spaCy/TextBlob 运算,在 GIL 下多线程无益,`_extract_nodes` 干脆顺序处理、靠缓存(按文本+分析器哈希)跳过重复工作。

这条路径产出的节点/边表与 LLM 路径**列结构兼容**(都有 `title/frequency/text_unit_ids` 和 `source/target/weight/text_unit_ids`),因此后续的 summarize、finalize、prune 可以无缝复用。它适合预算敏感、语料规模大、或对实体语义精度要求不高的场景:速度快、可离线、确定性强,代价是缺少 LLM 那种「理解语义、归并同义实体、写出连贯描述」的能力。两条路径并存,正体现了 GraphRAG 把「建图」与「打磨图」解耦的设计——无论图从哪来,打磨流程是同一套。

## 本章小结

- 抽取产物里实体/关系的 `description` 是多条零散描述的列表,`summarize_descriptions` 用 LLM 把它们合并成一段连贯摘要,是抽取链路的收尾环节而非可选美化。
- `SummarizeExtractor.__call__` 做成本短路:零条返回空串、单条原样返回,**只有多于一条描述才真正调用 LLM**,长尾实体不触发请求。
- 描述在入列前 `sorted(set(...))` 去重排序,既省 token 又让相同输入产生稳定 prompt 以利缓存与复现。
- 超长输入用「累积—触发—重置」的滚动归并:按 `max_input_tokens`(默认 `4000`)分批,每批摘要带入下一批继续归并,直至收敛;`max_length`(默认 `500` 词)是写进 prompt 的软约束。
- `finalize_graph` 流式算节点度数(无向去重)、补 `degree`、`human_readable_id`、`uuid4` 的 `id`,关系侧补 `combined_degree`,并按 `ENTITIES/RELATIONSHIPS_FINAL_COLUMNS` 对齐最终列。
- 本版本终化产物只含度数与主键,**不含 node2vec 嵌入或 UMAP 布局坐标**,最终列结构里没有坐标列。
- `prune_graph` 按固定顺序剪枝:去 ego 节点→按度剪→按频次剪→边两端存活过滤→`min_edge_weight_pct` 百分位剪弱边→可选 LCC;默认 `min_node_freq=2`、`remove_ego_nodes=True`、`min_edge_weight_pct=40`。
- 剪枝提升图的信噪比,直接决定下游 Leiden 社区检测的质量与规模;剪空会 `raise ValueError`。
- `build_noun_graph` 是不用 LLM 的对照路径:名词短语作节点、文本单元内共现作边、PMI 归一化权重,列结构与 LLM 路径兼容,可复用同一套 summarize/finalize/prune。

## 动手实验

1. **实验一:观察摘要的成本短路** — 在 `description_summary_extractor.py` 的 `__call__` 三个分支各加一行日志,标明命中「空/单条/多条」哪一支;跑一遍小语料索引,统计真正发起 LLM 调用(走 `_summarize_descriptions`)的实体占比,直观感受长尾实体被短路掉的比例。

2. **实验二:复现滚动归并的分批边界** — 把 `max_input_tokens` 临时调到一个很小的值(如 `300`),给某个实体人为塞入 6~8 条较长描述,在 `_summarize_descriptions` 循环里打印每次触发摘要时的 `len(descriptions_collected)` 和 `usable_tokens`,确认「装满即摘、摘要回填、最后收尾」的三类触发都被走到。

3. **实验三:调剪枝阈值看图规模** — 分别用 `min_node_freq=1`(几乎不剪)和 `min_node_freq=5`、`min_edge_weight_pct=70` 各跑一次 `prune_graph`,对比剪枝前后实体数与关系数,并观察 `remove_ego_nodes` 开关对最大度节点的影响,体会阈值如何改变下游社区检测的输入规模。

4. **实验四:对照 LLM 与 NLP 两条建图路径** — 同一份语料分别走 `extract_graph` 与 `extract_graph_nlp`,导出两套实体/关系表,对比节点数量、描述质量(NLP 路径节点是名词短语、无 LLM 描述)与边权分布;再验证两套产物都能被同一个 `finalize_graph` 正确终化,理解「建图可换、打磨不变」的解耦设计。

> **下一章预告**:图谱已经摘要、终化、剪枝完毕,接下来要在这张干净的图上「发现社区」。第 7 章《社区检测 Leiden》将解构 `create_communities` 工作流与 `hierarchical_leiden` 层次聚类,看 GraphRAG 如何把实体图切分成多层级的社区结构,为社区报告生成铺路。
