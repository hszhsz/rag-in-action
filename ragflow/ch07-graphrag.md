# 第 7 章 GraphRAG 知识图谱

到这里为止，我们解构的每一种检索路径都建立在一个朴素假设之上：答案藏在某一个 chunk 里，只要把那个 chunk 找出来排在前面就够了。可现实中有大量问题并非如此——"这家公司的三位联合创始人后来分别去了哪里"、"这条供应链上下游一共牵扯了多少家机构"、"整篇报告围绕哪几个主题展开"——这类问题的答案散落在十几个段落里，没有任何单个 chunk 同时包含全部线索。向量检索擅长找"语义相近的片段"，却不擅长"沿着实体之间的关系把碎片拼起来"。RAGFlow 用 GraphRAG 来补这块短板：先把文档读成一张实体-关系图，再围绕图做社区聚类和摘要，检索时不只匹配文本，而是顺着图结构把相关实体、关系、社区报告一并召回。

本章解构 `rag/graphrag/` 这棵目录树。我们会沿着一条文档真实经过的路径走一遍：GraphRAG 何时被任务执行器触发、`Extractor` 如何用 LLM 把一段文本榨成实体和关系、子图怎样合并进全局图并算出 PageRank、`EntityResolution` 怎样把"同一个东西的不同写法"合并掉、Leiden 算法如何把图切成社区并由 LLM 写成报告，最后落到 `KGSearch` 看图结构在查询时到底怎样增强召回。本章不追求把每个函数都讲到，而是挑出最能体现"图增强"价值的几处深入。

## 一、GraphRAG 在管线里的位置：一个 KB 级别的任务

GraphRAG 不是普通文档解析流程的一部分。普通解析是 doc 级别的——一篇文档进来、切块、向量化、入库。而图必须是 KB（知识库）级别的，因为实体和关系要跨文档汇聚才有意义。所以 RAGFlow 把它做成了一种独立的任务类型。在 `rag/svr/task_executor.py` 里，任务分发表把 `"graphrag"` 映射到流水线阶段 `PipelineTaskType.GRAPH_RAG`：

```python
TASK_TYPE_TO_PIPELINE_TASK_TYPE = {
    "dataflow": PipelineTaskType.PARSE,
    "raptor": PipelineTaskType.RAPTOR,
    "graphrag": PipelineTaskType.GRAPH_RAG,
    "mindmap": PipelineTaskType.MINDMAP,
    "memory": PipelineTaskType.MEMORY,
}
```

当 `task_type == "graphrag"` 时，执行器先取出该 KB 的 `parser_config`，如果用户没开 `use_graphrag`，它会就地补上一份默认配置——默认实体类型正是 `organization`、`person` 等几类。随后它读出两个关键开关并在 `kg_limiter` 限流器的保护下懒加载真正的入口函数：

```python
graphrag_conf = kb_parser_config.get("graphrag", {})
...
with_resolution = graphrag_conf.get("resolution", False)
with_community = graphrag_conf.get("community", False)
async with kg_limiter:
    from rag.graphrag.general.index import run_graphrag_for_kb  # Lazy load, save around 2s
    result = await run_graphrag_for_kb(
        row=task,
        doc_ids=task.get("doc_ids", []),
        ...
        with_resolution=with_resolution,
        with_community=with_community,
    )
```

这段代码透露了三层设计意图。其一，实体消歧（resolution）和社区报告（community）是两个可独立开关的可选阶段，默认都关——它们都极耗 LLM 调用，把决定权交给用户。其二，懒加载注释明说"省下约 2 秒"，因为 `rag/graphrag/general/index.py` 顶部 import 了 networkx、graspologic 等重型依赖，普通解析任务不该为它们付启动开销。其三，整个 GraphRAG 任务以 KB 为单位，函数名 `run_graphrag_for_kb` 直白地宣告了它的粒度。

## 二、三套抽取器与"为什么有 light、general、ner 三种实现"

读 `rag/graphrag/general/index.py` 的 `_select_extractor` 会发现一个容易被忽略的细节：RAGFlow 不是只有一种抽取器，而是三种，由配置里的 `method` 字段挑选：

```python
def _select_extractor(graphrag_config: dict):
    method = graphrag_config.get("method", "light")
    if method == "general":
        return GeneralKGExt
    if method == "ner":
        return NerKGExt
    return LightKGExt
```

注意默认值是 `"light"`。三套实现对应三种工程取舍。`general`（`rag/graphrag/general/graph_extractor.py`）是微软 GraphRAG 的原始路线 [[Microsoft GraphRAG]](https://github.com/microsoft/graphrag)——prompt 长、含完整的少样本示例，抽取质量高但 token 开销大。`light`（`rag/graphrag/light/graph_extractor.py`）借鉴 LightRAG [[LightRAG]](https://github.com/HKUDS/LightRAG)，prompt 更精简、示例数量可配（默认 `example_number=2`），是质量与成本之间的折中，因此被选作默认。`ner`（`rag/graphrag/ner/`）则走完全不同的路：用 spaCy 做命名实体识别，抽取本身不依赖 LLM。这一点在 `load_doc_chunks` 里留下了痕迹——NER 模式直接返回原始 chunk 列表，不做"按 token 批量合并"：

```python
# For NER-based extractionm, no need to batch extract entity and relation
if _select_extractor_type(graphrag_config) == "ner":
    return contents
```

为什么 light/general 要先合并 chunk？因为 LLM 抽取是按"大块文本"调用的，把若干小 chunk 拼到接近 `batch_chunk_token_size`（默认 `DEFAULT_GRAPHRAG_BATCH_CHUNK_TOKEN_SIZE = 4096`，受 `MIN_...=512` 与 `MAX_...=8196` 约束）才发给模型，可以摊薄每次调用的固定开销，也让模型在更完整的上下文里判断实体关系。NER 不用 LLM，自然没有这个动机。

尽管入口不同，三套抽取器都继承自同一个基类 `Extractor`（`rag/graphrag/general/extractor.py`）。基类把"并发跑所有 chunk、汇总节点边、合并同名实体、做关系摘要"这套骨架固定下来，子类只需实现 `_process_single_content`——即"一个 chunk 怎样变成若干条 record"。这正是模板方法模式：流程不变，差异只在抽取那一步。

## 三、实体与关系抽取：把文本榨成带分隔符的 record

`general` 抽取器的抽取 prompt（`GRAPH_EXTRACTION_PROMPT`，在 `rag/graphrag/general/graph_prompt.py`）规定了一种极其规整的输出格式。它要求模型把每个实体写成

```
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>
```

把每条关系写成

```
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
```

记录之间用 `record_delimiter` 分隔，整个输出以 `completion_delimiter` 收尾。这些分隔符在 `graph_extractor.py` 里有确定的默认字面量：`DEFAULT_TUPLE_DELIMITER = "<|>"`、`DEFAULT_RECORD_DELIMITER = "##"`、`DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"`。用奇怪的符号而非逗号、换行做分隔，是为了降低与正文内容冲突的概率——实体描述里出现 `<|>` 的可能性远低于出现逗号。

抽取不是一次问完就算数。`_process_single_content` 实现了 GraphRAG 论文里的 "gleaning"（拾遗）机制：第一轮抽完之后，再用 `CONTINUE_PROMPT` 追问"还有没有漏掉的"，最多追问 `ENTITY_EXTRACTION_MAX_GLEANINGS = 2` 次；每追问一轮前，还会用 `LOOP_PROMPT` 让模型判断是否值得继续：

```python
for i in range(self._max_gleanings):
    history.append({"role": "user", "content": CONTINUE_PROMPT})
    ...
    response = await self._async_chat("", history, {}, task_id)
    results += response or ""
    if i >= self._max_gleanings - 1:
        break
    history.append({"role": "assistant", "content": response})
    history.append({"role": "user", "content": LOOP_PROMPT})
    continuation = await self._async_chat("", history, {}, task_id)
    if continuation != "Y":
        break
```

这里有个很讲究的细节藏在 `graph_extractor.py` 的构造函数里：它用 tiktoken 把 "YES" 和 "NO" 编码成 token，给这两个 token 加上 `logit_bias: 100` 并把 `max_tokens` 限为 1，构造出 `_loop_args`，逼模型只在 YES/NO 之间二选一地回答"是否继续"。这是把开放式问题压成单 token 判定的经典技巧，省 token 也好解析。

模型吐回的长字符串怎样变成结构化数据？`split_string_by_multi_markers`（`rag/graphrag/utils.py`）先按 record/completion 分隔符切开，再用正则 `\((.*)\)` 抠出每条括号内的内容，最后交给基类的 `_entities_and_relations`。后者对每条 record 调 `handle_single_entity_extraction` 或 `handle_single_relationship_extraction` 解析字段。这两个函数定义在 `utils.py`，做了几件关键的规范化：实体名一律大写（`entity_name.upper()`）、关系强度若不是合法浮点数就回退成 `1.0`、关系的两端实体名排序后存（`pair = sorted([source.upper(), target.upper()])`）。把无向边的两端排序是个看似琐碎实则要紧的决定——它保证 `A->B` 和 `B->A` 落到同一个 key 上，后续合并才不会把同一条边算成两条。

## 四、合并同名实体与关系：从多条 record 到一个节点

同一个实体往往在多个 chunk 里被抽到，描述各不相同。基类 `__call__` 把所有 chunk 的抽取结果汇总后，对每个实体名并发调用 `_merge_nodes`：

```python
entity_type = sorted(
    Counter([dp["entity_type"] for dp in entities]).items(),
    key=lambda x: x[1], reverse=True,
)[0][0]
description = GRAPH_FIELD_SEP.join(sorted(set([dp["description"] for dp in entities])))
already_source_ids = flat_uniq_list(entities, "source_id")
description = await self._handle_entity_relation_summary(entity_name, description, task_id=task_id)
```

实体类型用"投票"决定——出现次数最多的那个 type 胜出，这样某个 chunk 偶尔把 person 误判成 organization 不会污染结果。描述则是把所有去重后的描述用 `GRAPH_FIELD_SEP`（`"<SEP>"`）拼起来。关系合并（`_merge_edges`）类似，但权重是相加的——同一对实体被多个 chunk 反复关联，意味着这条关系更可信，权重越高。

描述拼多了会变得又长又啰嗦，于是有了 `_handle_entity_relation_summary`。它的触发条件很克制：

```python
summary_max_tokens = 512
use_description = truncate(description, summary_max_tokens)
description_list = use_description.split(GRAPH_FIELD_SEP)
if len(description_list) <= 12:
    return use_description
```

只有当一个实体/关系积累了超过 12 段描述时，才真正动用 LLM 调 `SUMMARIZE_DESCRIPTIONS_PROMPT` 去做摘要压缩；否则直接截断返回。这是典型的"省钱优先"设计——大多数实体描述并没多到需要专门摘要，对它们调 LLM 纯属浪费。

## 五、子图生成、合并与 PageRank：图怎样落地存储

抽取得到的实体列表和关系列表，在 `generate_subgraph`（`index.py`）里被装进一张 networkx 的 `nx.Graph`。值得注意的是它对关系做了完整性校验——如果一条边引用的实体不在节点集合里，这条边会被丢弃并计数：

```python
if not subgraph.has_node(rel["src_id"]) or not subgraph.has_node(rel["tgt_id"]):
    ignored_rels += 1
    continue
```

子图建好后用 `nx.node_link_data` 序列化成 JSON，以 `knowledge_graph_kwd: "subgraph"` 写进文档存储。这一步同时是检查点：`run_graphrag_for_kb` 在抽取前会先调 `load_subgraph_from_store` 查这篇 doc 是否已有子图，命中就跳过 LLM 抽取。对一个动辄几十次 LLM 调用的文档来说，这种"断点续跑"能力在任务被取消或崩溃后重启时省下大量重复成本。

各 doc 的子图要合并进 KB 的全局图。合并发生在一把 Redis 分布式锁 `RedisDistributedLock(f"graphrag_task_{kb_id}", ...)` 的保护下——因为多个 worker 可能并发处理同一 KB 的不同文档，全局图是临界资源。`merge_subgraph` 调 `graph_merge`（`utils.py`）做实际合并：同名节点的描述用 `GRAPH_FIELD_SEP` 拼接、同一条边的权重相加，然后重新计算每个节点的度并写进 `rank` 属性。合并完成后做一件对检索至关重要的事：

```python
pr = nx.pagerank(new_graph)
for node_name, pagerank in pr.items():
    new_graph.nodes[node_name]["pagerank"] = pagerank
```

整张图上跑一遍 PageRank，把每个实体的重要性算出来存进节点。这个 `pagerank` 后面会成为检索排序里 `P(E)`（实体先验概率）的代理。最后 `set_graph`（`utils.py`）把图拆成几类 chunk 写库：整图存为 `knowledge_graph_kwd: "graph"`，每篇 doc 的子图存为 `"subgraph"`，每个节点存为 `"entity"` 并附带向量、每条边存为 `"relation"`。`graph_node_to_chunk` 在写实体 chunk 时，把 `pagerank` 存成 `rank_flt`，把 `n_neighbor` 算出的 n 跳邻居路径存成 `n_hop_with_weight`——注释明说这两者会被 `rag/graphrag/search.py` 读回去做排序和关系增强。换句话说，检索时需要的图结构信息，在写入阶段就预计算好了。

## 六、实体消歧：把"同一个东西的不同写法"合并掉

LLM 抽实体有个顽疾：同一实体会以不同写法出现——"IBM" 与 "I.B.M."、"北京大学" 与 "北大"。如果不处理，图上会有一堆本该是一个节点的孪生节点，关系被稀释、社区被打散。`EntityResolution`（`rag/graphrag/entity_resolution.py`）专门解决这件事。

它的第一步不是直接问 LLM，而是先做廉价的候选筛选。所有节点按 `entity_type` 分桶，只在同类型内两两配对，再用 `is_similarity` 过一遍字符相似度：

```python
candidate_resolution[k] = [(a, b) for a, b in itertools.combinations(v, 2)
    if (a in subgraph_nodes or b in subgraph_nodes) and self.is_similarity(a, b)]
```

`is_similarity` 的逻辑很有意思，它对中英文分别处理。英文用编辑距离，阈值是较短串长度的一半（`editdistance.eval(a, b) <= min(len(a), len(b)) // 2`）；中文则用字符集合的交集比例，要求 `len(a & b)*1./max_l >= 0.8`。更妙的是它有一道"数字守门"——`_has_digit_in_2gram_diff` 检查两个名字的 2-gram 差异里是否含数字，含则直接判不相似。这是为了避免把 "GPT-3" 和 "GPT-4"、"第一季度" 和 "第二季度" 这类只差一个数字、却是不同实体的名字误合并。条件里那个 `a in subgraph_nodes or b in subgraph_nodes` 也有讲究：只有涉及本轮新加入节点的候选对才需要消歧，纯粹由老节点构成的对在之前已经处理过了。

筛出候选对后才轮到 LLM。候选对按类型批量打包成自然语言问句（"Question 1: name of person A is ..., name of person B is ..."），用 `ENTITY_RESOLUTION_PROMPT`（`rag/graphrag/entity_resolution_prompt.py`）让模型逐题回答 Yes/No，再由 `_process_results` 用三套分隔符正则把答案解析回 `(index, "yes")` 列表。被判为"同一个"的实体对，通过一张临时图 `connect_graph` 求连通分量——这一步把传递关系也合并进来：若 A=B、B=C，则 A、B、C 会落在同一个连通分量里一起合并：

```python
connect_graph = nx.Graph()
connect_graph.add_edges_from(resolution_result)
for sub_connect_graph in nx.connected_components(connect_graph):
    merging_nodes = list(sub_connect_graph)
    tasks.append(asyncio.create_task(limited_merge_nodes(graph, merging_nodes, change)))
```

真正的节点合并由基类的 `_merge_graph_nodes` 完成，它把多个节点的描述、source_id、邻边都并到第一个节点上，并把被吞掉的节点和重定向的边记进 `GraphChange`，以便 `set_graph` 精确地只删改受影响的部分。整个消歧过程支持检查点：每批候选对的结果用 `resolution_checkpoint_key` 哈希成稳定 key 存进 Redis（见 `rag/graphrag/checkpoints.py`），重跑时能"回放"已解析的批次而不重复调 LLM。消歧结束后再重算一遍全图 PageRank，因为节点合并改变了图结构。

## 七、社区检测与社区报告：Leiden 把图切块，LLM 写摘要

图建好了，但回答"整篇报告围绕哪几个主题"这种宏观问题，光有实体和关系还不够——需要把紧密相连的实体聚成"社区"，再为每个社区写一份概览。这正是 GraphRAG 区别于普通向量检索的核心价值。

聚类用的是 Leiden 算法 [[Leiden algorithm]](https://www.nature.com/articles/s41598-019-41695-z)，实现在 `rag/graphrag/general/leiden.py`，底层调 graspologic 的 `hierarchical_leiden`。`run` 函数默认 `max_cluster_size=12`、`use_lcc=True`（只在最大连通分量上聚类），随机种子固定为 `0xDEADBEEF` 以保证可复现：

```python
community_mapping = hierarchical_leiden(
    graph, max_cluster_size=max_cluster_size, random_seed=seed
)
for partition in community_mapping:
    results[partition.level] = results.get(partition.level, {})
    results[partition.level][partition.node] = partition.cluster
```

注意 `hierarchical_leiden` 是分层的——它返回多个 level，每一层是更粗或更细的社区划分。`run` 把结果整理成 `{level: {community_id: {"weight": ..., "nodes": [...]}}}`，社区权重由其成员节点的 `rank`（度）乘 `weight` 累加并归一化，等于用"这个社区里实体有多重要"来给社区排序。

聚出社区后，`CommunityReportsExtractor`（`rag/graphrag/general/community_reports_extractor.py`）为每个不少于 2 个节点的社区生成报告。它把社区内的实体和它们之间的关系分别整理成 pandas DataFrame、转成 CSV 填进 `COMMUNITY_REPORT_PROMPT`，让 LLM 产出一份结构化 JSON。生成后还做严格的 schema 校验，字段类型不符就丢弃：

```python
if not dict_has_keys_with_types(response, [
            ("title", str), ("summary", str), ("findings", list),
            ("rating", float), ("rating_explanation", str),
        ]):
    return
```

每份报告反过来把自己的 `title` 写回图上对应实体的 `communities` 属性（`add_community_info2graph`），这样检索时能从实体反查它属于哪些社区。报告最终被组装成 chunk 入库，`knowledge_graph_kwd: "community_report"`，`entities_kwd` 存社区成员实体、`weight_flt` 存社区权重——这两个字段决定了检索时社区报告如何被命中和排序。

这里还藏着一处工程上的稳健性设计。社区 chunk 的 id 用 `(kb_id, 社区 title)` 派生而非随机生成，并且入库采用"先插入新报告、再删除陈旧报告"的顺序。注释解释了原因：这样即便在插入中途崩溃，旧的整套报告也仍然完整，绝不会出现"删一半"的中间态。两个可选阶段——消歧和社区——各自跑完后会用 `set_phase_marker`（`rag/graphrag/phase_markers.py`）在 Redis 打一个 KB 级别、7 天 TTL 的完成标记；而一旦有新文档被合并进图，`clear_phase_markers` 会清掉这些标记，因为图变了、之前的消歧和社区结果就过时了。这套"阶段标记 + 检查点"的组合，让一个可能跑几小时的 GraphRAG 任务在被取消或崩溃后能从最近的安全点续上，而不是从头再来。

## 八、图检索：查询时如何利用实体图

前面所有努力都是为了这一刻。`KGSearch`（`rag/graphrag/search.py`）继承自普通检索的 `Dealer`，但它的 `retrieval` 走的是一条与向量检索完全不同的路。

第一步是查询改写。`query_rewrite` 先从库里取出"类型到样例实体"的映射（`get_entity_type2samples`），连同问题一起塞进 `minirag_query2kwd` prompt（`rag/graphrag/query_analyze_prompt.py`，借鉴 MiniRAG [[MiniRAG]](https://github.com/HKUDS/MiniRAG)），让 LLM 输出两样东西：`answer_type_keywords`（答案应该是什么类型，比如问"何时"就指向 DATE，最多 3 个）和 `entities_from_query`（问题里提到的具体实体，最多取 5 个）。这一步把自然语言问题翻译成了"在图上该找什么类型、什么实体"。

第二步是多路召回。RAGFlow 同时从三个角度命中图：

```python
ents_from_query = self.get_relevant_ents_by_keywords(ents, filters, idxnms, kb_ids, emb_mdl, ent_sim_threshold)
ents_from_types = self.get_relevant_ents_by_types(ty_kwds, filters, idxnms, kb_ids, 10000)
rels_from_txt = self.get_relevant_relations_by_txt(qst, filters, idxnms, kb_ids, emb_mdl, rel_sim_threshold)
```

第一路按问题里的实体做向量检索命中实体节点；第二路按答案类型直接捞出该类型下 PageRank 最高的实体（按 `rank_flt` 降序）；第三路对整个问题做向量检索命中关系边。三路各取所需，互为补充。

第三步是顺图扩散，这是图检索最有"图味"的部分。对每个命中的实体，读它预存的 `n_hop_ents`（即写入阶段算好的 n 跳邻居路径），沿路径把多跳之外的关系也拉进候选，并按跳数衰减地累加相似度：

```python
for nbr in nhops:
    path = nbr["path"]; wts = nbr["weights"]
    for i in range(len(path) - 1):
        f, t = path[i], path[i + 1]
        nhop_pathes[(f, t)]["sim"] = ... + ent["sim"] / (2 + i)
        nhop_pathes[(f, t)]["pagerank"] = max(nhop_pathes[(f, t)].get("pagerank", 0), wts[i])
```

`ent["sim"] / (2 + i)` 意味着离命中实体越远的边贡献越小。这就是向量检索做不到的事：即使某条关系的文本本身与问题不相似，只要它紧挨着一个相关实体，也能被召回。

第四步是打分排序。代码注释写明了排序模型 `P(E|Q) => P(E) * P(Q|E) => pagerank * sim`——一个实体的最终得分是它的 PageRank（先验重要性）乘以与查询的相似度（似然）。其间还有若干信号融合：被答案类型命中的实体相似度翻倍（`ents_from_query[ent]["sim"] *= 2`），两端落在答案类型实体里的关系得到加权。最后实体和关系各按 `sim * pagerank` 排序，分别取 `ent_topn`（默认 6）、`rel_topn`（默认 6）。

第五步是社区报告召回，把宏观视角补回来。`_community_retrieval_` 拿排序后的 top 实体当过滤条件，从 `community_report` 里捞出 `entities_kwd` 与这些实体相交、按 `weight_flt` 降序的社区报告（默认 `comm_topn=1`）。最终 `retrieval` 把实体表、关系表、社区报告拼成一段 CSV/Markdown 文本，包装成一个 `docnm_kwd` 为 "Related content in Knowledge Graph" 的虚拟 chunk 返回。它会和普通向量检索的结果一起，作为上下文喂给生成模型——这就是图增强检索最终落地的形态：不替换向量检索，而是给它补上一块"结构化、跨文档、带宏观摘要"的上下文。

## 本章小结

1. GraphRAG 是 KB 级别的独立任务类型（`PipelineTaskType.GRAPH_RAG`），入口为 `run_graphrag_for_kb`，因重型依赖而懒加载；消歧与社区两阶段默认关闭，由 `resolution`/`community` 配置开关控制。
2. 抽取器有 light、general、ner 三套，由 `method` 配置选择，默认 `light`；三者共享基类 `Extractor` 的并发抽取与合并骨架，只在 `_process_single_content` 上有差异。NER 模式不调 LLM 抽取，因而跳过按 token 合并 chunk 的步骤。
3. LLM 抽取用 `<|>`、`##`、`<|COMPLETE|>` 等专用分隔符规整输出，并通过 gleaning（最多 `ENTITY_EXTRACTION_MAX_GLEANINGS = 2` 轮拾遗，配 YES/NO 的 logit_bias 判定）尽量榨干实体。
4. 同名实体合并时类型按投票决定、描述用 `<SEP>` 拼接、关系权重相加；只有描述段数超过 12 才调 LLM 做摘要压缩，否则截断返回，以省 token。
5. 子图先写库（`knowledge_graph_kwd: "subgraph"`）兼作检查点，再在 Redis 分布式锁保护下用 `graph_merge` 合并进全局图，合并后跑 `nx.pagerank` 把实体重要性预存进节点。
6. `set_graph` 把图拆成 graph/subgraph/entity/relation 多类 chunk 落库，写入阶段就把 `pagerank`（存为 `rank_flt`）和 n 跳邻居路径（存为 `n_hop_with_weight`）预计算好供检索读回。
7. `EntityResolution` 先用类型分桶 + 字符相似度（英文编辑距离、中文字符交集，并用"数字守门"避免误合 GPT-3/GPT-4）做廉价候选筛选，再由 LLM 逐题判定，最后用连通分量做传递性合并。
8. 社区检测用分层 Leiden（`max_cluster_size=12`、固定种子 `0xDEADBEEF`），社区报告由 LLM 生成结构化 JSON 并做 schema 校验，报告 id 由 `(kb_id, title)` 派生、采用"先插后删"保证崩溃安全。
9. 阶段标记（`phase_markers`）与检查点（`checkpoints`）配合，让长耗时任务能在取消/崩溃后从安全点续跑；新文档合并会清除标记使消歧/社区结果失效重算。
10. `KGSearch` 检索分五步：查询改写出类型与实体、三路多角度召回、沿 n 跳邻居扩散关系、按 `pagerank * sim` 的 `P(E|Q)` 模型排序、再召回相关社区报告，最终拼成一个虚拟 chunk 补充给生成模型。
11. 图检索的根本价值在于能召回"文本本身不相似但结构上相邻"的关系，以及提供跨文档的社区级宏观摘要——这是单纯向量检索无法做到的。

## 动手实验

1. **实验一：观察抽取器的分隔符与 gleaning** — 打开 `rag/graphrag/general/graph_extractor.py`，确认 `DEFAULT_TUPLE_DELIMITER`、`DEFAULT_RECORD_DELIMITER`、`DEFAULT_COMPLETION_DELIMITER` 三个常量值；再到 `_process_single_content` 里跟读 gleaning 循环，结合 `ENTITY_EXTRACTION_MAX_GLEANINGS` 的值，数清楚最坏情况下抽取一个 chunk 会发起几次 LLM 调用（首轮 + 拾遗 + loop 判定）。

2. **实验二：手算实体相似度判定** — 阅读 `rag/graphrag/entity_resolution.py` 的 `is_similarity` 与 `_has_digit_in_2gram_diff`，分别代入英文对 ("IBM", "I.B.M.")、数字对 ("GPT-3", "GPT-4")、中文对 ("北京大学", "北大") 三组，按代码逻辑推演它们各自是否会被判为候选相似对，并解释"数字守门"为何能拦下第二组。

3. **实验三：复现社区聚类的可复现性** — 在 `rag/graphrag/general/leiden.py` 的 `_compute_leiden_communities` 里定位随机种子 `0xDEADBEEF` 与 `max_cluster_size` 默认值；用一张小的 networkx 图（10～20 个节点）两次调用 `leiden.run`，验证两次社区划分完全一致，再把种子改掉观察结果是否变化，体会固定种子对结果稳定性的意义。

4. **实验四：拆解图检索的排序公式** — 在 `rag/graphrag/search.py` 的 `retrieval` 里找到注释 `P(E|Q) => P(E) * P(Q|E) => pagerank * sim` 与最终 `sorted(..., key=lambda x: x[1]["sim"] * x[1]["pagerank"])`；构造一个假想场景：实体 A 的 `pagerank=0.1, sim=0.9`，实体 B 的 `pagerank=0.5, sim=0.3`，算出二者最终得分并判断谁排前面，再说明 n 跳衰减项 `sim / (2 + i)` 如何影响远端关系的权重。

> **下一章预告**：第 8 章《RAPTOR 树状摘要》。如果说 GraphRAG 用"实体-关系图"组织跨文档的结构信息，RAPTOR 则用另一种思路应对长文档——它对 chunk 做层次聚类、对每簇递归生成摘要，自底向上长出一棵摘要树，让检索可以在不同抽象层级上命中。下一章我们将解构 `rag/raptor.py`，看它如何用聚类与递归摘要把扁平的 chunk 列表变成可在多个粒度上检索的树。
