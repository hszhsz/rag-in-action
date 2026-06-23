# 第 9 章 检索器 Retrievers

第 8 章我们已经看到,向量库被抽象成一个能"按 query 取回相似 node"的存储层:你把一个查询向量丢进去,它吐回一批 `NodeWithScore`。但用户真正面对的不是 `vector_store.query`,而是 `index.as_retriever()` 返回的一个对象。这中间隔着的就是本章的主角——Retriever。

如果说 Index 决定了数据"以什么结构组织"(扁平向量、层级父子、知识图谱),那么 Retriever 决定了"在这套结构上怎么把相关 node 取出来"。同一个 `VectorStoreIndex` 既可以用最朴素的 top-k 相似度检索,也可以套上自动合并、递归跳转、多路融合等更复杂的策略。LlamaIndex 把这些策略全部收敛到 `BaseRetriever` 这一个抽象之下:不论内部多复杂,对外都只暴露一个 `retrieve(query)` 方法,返回 `List[NodeWithScore]`。本章我们从基类的模板方法读起,逐个拆解 `VectorIndexRetriever`、`AutoMergingRetriever`、`RecursiveRetriever`、`QueryFusionRetriever`,最后扫一眼 router/transform 以及检索后处理在整条链路里的位置。

## BaseRetriever:统一抽象与模板方法

所有检索器都继承自 `base_retriever.py` 里的 `BaseRetriever`。它的对外入口是 `retrieve`,而真正的检索逻辑留给子类实现的抽象方法 `_retrieve`。这是典型的模板方法模式:基类把"标准动作"固定下来,子类只填"检索算法"这一块空白。

`retrieve` 做的标准动作有四件事:

```python
@dispatcher.span
def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
    self._check_callback_manager()
    dispatcher.event(RetrievalStartEvent(str_or_query_bundle=str_or_query_bundle))
    if isinstance(str_or_query_bundle, str):
        query_bundle = QueryBundle(str_or_query_bundle)
    else:
        query_bundle = str_or_query_bundle
    with self.callback_manager.as_trace("query"):
        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: query_bundle.query_str},
        ) as retrieve_event:
            nodes = self._retrieve(query_bundle)
            nodes = self._handle_recursive_retrieval(query_bundle, nodes)
            retrieve_event.on_end(payload={EventPayload.NODES: nodes})
    dispatcher.event(RetrievalEndEvent(str_or_query_bundle=str_or_query_bundle, nodes=nodes))
    return nodes
```

第一,把字符串归一化为 `QueryBundle`——这意味着子类的 `_retrieve` 永远只需处理 `QueryBundle`,不用关心调用方传的是裸字符串还是已经带了 embedding 的查询包。第二,用 `callback_manager.event(CBEventType.RETRIEVE, ...)` 包住检索过程,并在 `instrumentation` 层抛出 `RetrievalStartEvent` / `RetrievalEndEvent`(`retrieve` 本身带 `@dispatcher.span` 装饰),这套埋点让上层观测系统能记录每次检索的 query 与命中 node。第三,调用子类的 `_retrieve` 拿到原始结果。第四,把结果交给 `_handle_recursive_retrieval` 做递归 object 处理后返回。

`_aretrieve` 是 `retrieve` 的异步孪生,结构完全对称。值得注意的是 `_retrieve` 是 `@abstractmethod`,而 `_aretrieve` 默认实现就是同步退化——`return self._retrieve(query_bundle)`,因此一个新检索器最少只需实现同步版即可跑通。

## 递归 object 处理:让检索结果可以"再展开"

`_handle_recursive_retrieval` 是基类提供的一个隐形增强:它遍历 `_retrieve` 返回的每个 node,如果某个 node 是 `IndexNode`(回顾第 1 章,`IndexNode` 是指向"另一个对象"的占位 node),就尝试把它指向的对象取出来并展开。

对象从哪来?代码里是 `obj = node.obj or self.object_map.get(node.index_id, None)`。`object_map` 在 `__init__` 里建立:如果传入了 `objects` 列表,就构造成 `{obj.index_id: obj.obj for obj in objects}`。找到对象后交给 `_retrieve_from_object` 按类型分派:

```python
if isinstance(obj, NodeWithScore):
    return [obj]
elif isinstance(obj, BaseNode):
    return [NodeWithScore(node=obj, score=score)]
elif isinstance(obj, BaseQueryEngine):
    response = obj.query(query_bundle)
    return [NodeWithScore(node=TextNode(text=str(response), ...), score=score)]
elif isinstance(obj, BaseRetriever):
    return obj.retrieve(query_bundle)
else:
    raise ValueError(f"Object {obj} is not retrievable.")
```

也就是说,检索命中的 `IndexNode` 可以背后挂着一个完整的 query engine 或另一个 retriever,命中即触发"二次检索/二次问答"。最后 `_handle_recursive_retrieval` 还用一个 `seen` 集合按 `node_id` 去重。这给了上层极大的组合自由度——你可以把"摘要 node"放进向量库,命中后再钻到它对应的细节文档里。

## VectorIndexRetriever:把 top-k 与 filters 翻译成 VectorStoreQuery

最常用的检索器是 `indices/vector_store/retrievers/retriever.py` 里的 `VectorIndexRetriever`。它的职责非常聚焦:把用户友好的参数(`similarity_top_k`、`filters`、查询模式)翻译成向量库能懂的 `VectorStoreQuery`,再把返回结果还原成 `NodeWithScore`。

`similarity_top_k` 默认值是 `DEFAULT_SIMILARITY_TOP_K`,在 `constants.py` 里定义为 `2`——这是个相当保守的默认值,生产里通常需要调大。`_retrieve` 先判断当前查询模式是否需要 embedding(`_needs_embedding`:`TEXT_SEARCH` 与 `SPARSE` 模式不需要),需要的话用 `embed_model.get_agg_embedding_from_queries` 把 query 文本编码成向量,然后调 `_get_nodes_with_embeddings`。

核心翻译动作在 `_build_vector_store_query`:

```python
return VectorStoreQuery(
    query_embedding=query_bundle_with_embeddings.embedding,
    similarity_top_k=self._similarity_top_k,
    node_ids=self._node_ids,
    doc_ids=self._doc_ids,
    query_str=query_bundle_with_embeddings.query_str,
    mode=self._vector_store_query_mode,
    alpha=self._alpha,
    filters=self._filters,
    sparse_top_k=self._sparse_top_k,
    hybrid_top_k=self._hybrid_top_k,
)
```

可以看到 `filters`(类型是第 8 章见过的 `MetadataFilters`)、`alpha`(混合检索时稀疏/稠密的权重)、`sparse_top_k`/`hybrid_top_k` 全都原样透传给向量库。检索器自己不实现过滤逻辑,而是把它声明式地下推到底层存储,谁离数据近谁来执行,这是典型的"下推优化"思路。

拿到 `VectorStoreQueryResult` 后还有一步关键处理。有些向量库只存向量不存正文,`_determine_nodes_to_fetch` 会判断哪些 node 需要回 docstore 补全:若结果带 `nodes` 字段,则挑出非 `TEXT` 类型的 node 去 docstore 取;若只带 `ids`,则全部回查。补全后由 `_convert_nodes_to_scored_nodes` 把 `similarities` 数组逐一对应到每个 node,封装成 `NodeWithScore`。这解释了为什么向量库可以"只存索引"——检索器会负责把缺失的内容从 docstore 拼回来。

## AutoMergingRetriever:上溯合并,把碎片拼回大块

`auto_merging_retriever.py` 里的 `AutoMergingRetriever` 是本章的重头戏,它正面呼应第 4 章的 `HierarchicalNodeParser` 与第 1 章的父子(PARENT/CHILD)关系图。它的动机是:层级切分把一篇文档切成了大块(父)套小块(子),向量检索命中的往往是若干分散的小叶子;如果一个父节点下的子节点被大面积命中,与其把这些碎片分别塞给 LLM,不如直接"上溯"到父节点,给 LLM 一段更完整、更连贯的上下文。

它包了一个 `vector_retriever`(就是上面的 `VectorIndexRetriever`)和一个 `storage_context`,构造参数里最关键的是 `simple_ratio_thresh: float = 0.5`。`_retrieve` 先用向量检索器取回叶子 node,然后反复调 `_try_merging` 直到不再变化:

```python
def _retrieve(self, query_bundle):
    initial_nodes = self._vector_retriever.retrieve(query_bundle)
    cur_nodes, is_changed = self._try_merging(initial_nodes)
    while is_changed:
        cur_nodes, is_changed = self._try_merging(cur_nodes)
    cur_nodes.sort(key=lambda x: x.get_score(), reverse=True)
    return cur_nodes
```

`_try_merging` 分两步:先 `_fill_in_nodes`,再 `_get_parents_and_merge`。`_fill_in_nodes` 处理的是"相邻补洞"——如果排序后相邻两个 node 在原文里本是前后相接(`cur_node.next_node == nodes[idx+1].node.prev_node`)却中间漏了一个,就从 docstore 把中间那个 node 取回来填上,分数取两侧平均。

真正的"合并"在 `_get_parents_and_merge`。它先按 `parent_node` 把命中的子 node 分组,统计每个父节点"被命中了几个子节点"(`parent_cur_children`)以及"总共有几个子节点"(`parent_num_children`,取自父节点的 `child_nodes`),然后算比例并判断:

```python
ratio = len(parent_cur_children) / parent_num_children
if ratio > self._simple_ratio_thresh:
    node_ids_to_delete.update({n.node.node_id for n in parent_cur_children})
    avg_score = sum([n.get_score() or 0.0 for n in parent_cur_children]) / len(parent_cur_children)
    parent_node_with_score = NodeWithScore(node=parent_node, score=avg_score)
    nodes_to_add[parent_node_id] = parent_node_with_score
```

这就是合并阈值的全部逻辑:命中子节点占父节点全部子节点的比例**严格大于** `_simple_ratio_thresh`(默认 `0.5`,即过半)时,就把这一批子 node 从结果里删掉,换成它们的父 node,父 node 的分数取被命中子节点的平均分。比例不到阈值则保持碎片不动。删旧增新后,只要 `node_ids_to_delete` 非空就标记 `is_changed=True`,外层循环会带着新结果再跑一遍——于是合并可以逐层向上传播:子合成父,父若又满足阈值还能再合成祖父。最终按分数降序返回。注意分母里有个细节,`parent_num_children = len(parent_child_nodes) if parent_child_nodes else 1`,父节点若没有记录子节点列表则按 1 计,避免除零。

## RecursiveRetriever:顺着 IndexNode 跳转的组合检索

`recursive_retriever.py` 的 `RecursiveRetriever` 把"递归展开"做成了一个独立的、可显式编排的检索器。基类的 `_handle_recursive_retrieval` 是隐式的一层展开,而 `RecursiveRetriever` 则维护一张图:`retriever_dict`(id→retriever)、`query_engine_dict`(id→query engine)、`node_dict`(id→node),以及一个 `root_id` 作为入口,构造时会校验 `root_id` 必须存在于 `retriever_dict`。

检索从 `_retrieve_rec(query_bundle, query_id=None)` 开始,`query_id` 为空时落到 `root_id`。`_get_object` 按 id 在三个字典里依次查找拿到对象,再分类型处理:是 `BaseNode` 就直接包成结果;是 `BaseRetriever` 就调它的 `retrieve`,然后把结果交给 `_query_retrieved_nodes`;是 `BaseQueryEngine` 就调 `query`,用 `DEFAULT_QUERY_RESPONSE_TMPL`("Query: {query_str}\nResponse: {response}")把问答结果格式化成一个 `TextNode`,并把 `sub_resp.source_nodes` 作为 additional_nodes 带出。

`_query_retrieved_nodes` 是递归的关键:遍历检索回来的 node,凡是 `IndexNode` 就取它的 `index_id` 作为新的 `query_id` 再次调 `_retrieve_rec` 钻进去;普通 `TextNode` 则原样保留。它还在进入前先按 `index_id` 去重一批指向同一索引的 IndexNode,事后用 `_deduplicate_nodes`(按 `node.id_` 保留首次出现)再去一次重。`RecursiveRetriever` 与 `AutoMergingRetriever` 形成对照:后者沿父子关系"向上合并",前者沿 IndexNode 链接"向下/横向跳转",二者都建立在第 1 章的 node 关系模型之上。

## QueryFusionRetriever:多路检索与 RRF 融合

`fusion_retriever.py` 的 `QueryFusionRetriever` 解决两个维度的"多":多个检索器、多个改写后的 query。它接收一个 `retrievers` 列表,默认每个等权(`[1.0 / len(retrievers)] * len(retrievers)`),也可传 `retriever_weights` 自定义后归一化。`num_queries` 默认 `4`:当大于 1 时,`_get_queries` 会用 `QUERY_GEN_PROMPT` 调 LLM 把原始 query 扩写成多条相关查询(生成 `num_queries - 1` 条,加上原始 query 凑齐),从而提高召回多样性。

`_retrieve` 把每个 query × 每个 retriever 的笛卡尔积都跑一遍(同步走 `_run_sync_queries`,异步走 `_run_nested_async_queries`),结果存进一个以 `(query_str, retriever_idx)` 为键的字典。融合模式由 `FUSION_MODES` 枚举决定,共四种:

- `RECIPROCAL_RANK`(`"reciprocal_rerank"`):倒数排名融合;
- `RELATIVE_SCORE`(`"relative_score"`):相对分数融合;
- `DIST_BASED_SCORE`(`"dist_based_score"`):基于分布的分数融合;
- `SIMPLE`(`"simple"`,默认):简单按原始分数取最大值去重重排。

其中 RRF(Reciprocal Rank Fusion)最经典。`_reciprocal_rerank_fusion` 的核心是:对每个结果列表按分数降序排序,然后给每个 node 累加 `1.0 / (rank + k)`,其中 `k = 60.0`(代码注释指明原论文取 k=60 效果最好):

```python
k = 60.0
for nodes_with_scores in results.values():
    for rank, node_with_score in enumerate(
        sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
    ):
        hash = node_with_score.node.hash
        hash_to_node[hash] = node_with_score
        fused_scores[hash] = fused_scores.get(hash, 0.0) + 1.0 / (rank + k)
```

RRF 只看名次不看原始分数的绝对量级,因此天然能把"分数量纲不一致"的多个检索器(比如向量相似度与 BM25)公平地融合到一起——这正是它在混合检索里被广泛采用的原因。`_relative_score_fusion` 则相反,它对每个结果集做 MinMax 归一化(`dist_based=True` 时用均值 ± 3 倍标准差定边界),再乘以该检索器的权重并除以 `num_queries`,最后按 node 的 `hash` 累加去重。无论哪种模式,最终都用 `[: self.similarity_top_k]` 截断返回。

## router 与 transform:一瞥两个组合器

`router_retriever.py` 的 `RouterRetriever` 走的是"选择"而非"融合"的路子。它持有一个 `BaseSelector` 和一组包装成 `RetrieverTool` 的候选检索器(工具壳是为了把每个检索器的 metadata 暴露给 selector)。`_retrieve` 时让 selector 根据 query 与各候选的 metadata 选出一个或多个(`select_multi` 控制),只在被选中的检索器上执行,多选时把结果按 `node_id` 合并去重。适合"这个库管财报、那个库管法务"这类需要先路由再检索的场景。

`transform_retriever.py` 的 `TransformRetriever` 更轻:它包一个底层 retriever 和一个 `BaseQueryTransform`,`_retrieve` 里先 `self._query_transform.run(query_bundle, ...)` 改写查询,再把改写后的 `QueryBundle` 交给底层检索器。HyDE、step-back 之类的查询改写策略都可以通过它接入,而无需修改底层检索逻辑。

## 检索后处理与 rerank:在链路里的位置

检索器吐出 `List[NodeWithScore]` 之后,通常还会经过一层"后处理"再进入响应合成。`postprocessor/` 目录就是这层:`sbert_rerank.py` 的 `SentenceTransformerRerank` 用交叉编码器对候选 node 重新打分并只保留 `top_n`(默认 `2`);`llm_rerank.py`、`rankGPT_rerank.py`、`structured_llm_rerank.py` 则用 LLM 来重排;`node_recency.py` 按时间新近度调整,`optimizer.py`/`pii.py` 分别做内容压缩与隐私脱敏。

其中 `metadata_replacement.py` 的 `MetadataReplacementPostProcessor` 直接呼应第 4 章的 SentenceWindow:它接收一个 `target_metadata_key`,在后处理阶段把每个 node 的正文替换成该 metadata key 里存的"窗口上下文"——检索时用单句精确命中,合成时却用其前后窗口供 LLM 阅读。这正说明了完整链路的次序:**检索(Retriever)→ 后处理/重排(NodePostprocessor)→ 响应合成(ResponseSynthesizer)**。检索负责"召回得全",rerank 负责"排序得准",后处理还能"改写得合适",三者各司其职,最后才把精选 node 交给下一章的合成器去生成答案。

## 本章小结

- `BaseRetriever` 用模板方法把检索统一为 `retrieve` → 抽象 `_retrieve` → `List[NodeWithScore]`,子类只需实现检索算法,字符串归一化、callback 与 instrumentation 埋点全在基类完成。
- `_retrieve` 是 `@abstractmethod`,`_aretrieve` 默认退化为同步实现,因此最小可用检索器只需写一个同步方法。
- `_handle_recursive_retrieval` 与 `_retrieve_from_object` 让命中的 `IndexNode` 能背后挂 retriever 或 query engine 并被二次展开,结果按 `node_id` 去重。
- `VectorIndexRetriever` 把 `similarity_top_k`(默认 `DEFAULT_SIMILARITY_TOP_K = 2`)、`filters`、`alpha` 等声明式下推给 `VectorStoreQuery`,并在结果只含向量时回 docstore 补全正文。
- `AutoMergingRetriever` 以 `simple_ratio_thresh`(默认 `0.5`)为阈值,当某父节点被命中的子节点比例严格大于阈值时上溯合并为父 node,合并可逐层向上传播,呼应 `HierarchicalNodeParser`。
- `RecursiveRetriever` 维护 `retriever_dict`/`query_engine_dict`/`node_dict` 与 `root_id`,沿 `IndexNode` 的 `index_id` 递归跳转,实现可显式编排的组合检索。
- `QueryFusionRetriever` 支持四种 `FUSION_MODES`,RRF 用 `1.0 / (rank + k)`、`k = 60.0` 只按名次融合多路结果,天然兼容不同量纲的检索器;`num_queries` 默认 `4` 会用 LLM 扩写查询。
- `RouterRetriever` 靠 selector 在候选间做路由选择,`TransformRetriever` 在检索前改写 query,二者都是组合器而非新检索算法。
- 检索之后的链路是"检索 → rerank/后处理 → 响应合成",`postprocessor/` 里的 `SentenceTransformerRerank`(默认 `top_n=2`)、`MetadataReplacementPostProcessor` 等承担排序与内容改写。

## 动手实验

1. **实验一:确认默认 top-k 的来源** — 打开 `/tmp/llama_explore/llama-index-core/llama_index/core/constants.py` 找到 `DEFAULT_SIMILARITY_TOP_K = 2`,再到 `indices/vector_store/retrievers/retriever.py` 的 `VectorIndexRetriever.__init__` 与 `_build_vector_store_query`,跟踪 `similarity_top_k` 是如何原样填进 `VectorStoreQuery` 的,理解为什么不调大默认只会取回 2 个 node。
2. **实验二:推演自动合并阈值** — 读 `retrievers/auto_merging_retriever.py` 的 `_get_parents_and_merge`,假设某父节点共有 4 个子节点、本轮命中了 3 个,手算 `ratio = 3/4 = 0.75`,验证它确实大于默认 `simple_ratio_thresh=0.5` 而触发合并;再把命中数改成 2,确认 `ratio=0.5` **不** 大于阈值因而不合并(注意是严格大于)。
3. **实验三:核对 RRF 常量与公式** — 在 `retrievers/fusion_retriever.py` 的 `_reciprocal_rerank_fusion` 里找到 `k = 60.0` 与 `fused_scores[hash] += 1.0 / (rank + k)`,对照 `FUSION_MODES` 枚举的四个取值,说明为什么 RRF 只依赖名次而不依赖原始分数,适合融合向量与 BM25 这类不同量纲的检索器。
4. **实验四:画出递归跳转链路** — 阅读 `retrievers/recursive_retriever.py` 的 `_retrieve_rec`、`_query_retrieved_nodes` 与 `_get_object`,从 `root_id` 出发,画出"命中一个 `IndexNode` → 用其 `index_id` 取出新对象 → 若是 retriever 再次检索/若是 query engine 用 `DEFAULT_QUERY_RESPONSE_TMPL` 包装"的完整跳转路径,并标出两处去重发生的位置。

> **下一章预告**:检索器把最相关的一批 node 召回并排好序之后,真正的"回答"还没生成。第 10 章我们进入响应合成 ResponseSynthesizers,看 LlamaIndex 如何把这些 node 塞进 prompt 交给 LLM,以及 refine、compact、tree_summarize 等模式在"上下文塞不下"时各自如何分批、迭代与汇总。
