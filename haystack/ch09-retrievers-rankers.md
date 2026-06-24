# 第 9 章 检索器与重排器

第 8 章我们看清了嵌入器如何把文本变成向量：`SentenceTransformersTextEmbedder` 把查询字符串编码成 `embedding`，`SentenceTransformersDocumentEmbedder` 把每篇 `Document` 编码后写回 `doc.embedding`。但向量本身不会回答问题——必须有人拿着它去仓库里"找回"最像的文档。这就是检索器（Retriever）的职责：召回。召回追求快而全，会带回一批候选；而候选里夹杂噪声，于是又需要重排器（Ranker）做精排，追求准而细。

本章把这两类组件一起解构。我们会看到一个反复出现的设计：检索器几乎不自己算相似度，它只是 `DocumentStore` 检索能力的"组件外壳"——持有 store 引用，把 `top_k`、`filters`、`scale_score` 原样委托下去；而重排器恰恰相反，它自己持有模型或排序逻辑，对召回结果重新打分。理解了这个分工，就理解了"召回快而粗、精排准而慢"为什么要分离。

## 检索器是 store 检索能力的"组件外壳"

打开 `haystack/components/retrievers/in_memory/bm25_retriever.py`，`InMemoryBM25Retriever.run` 的核心只有一行：

```python
docs = self.document_store.bm25_retrieval(query=query, filters=filters, top_k=top_k, scale_score=scale_score)
return {"documents": docs}
```

检索器没有实现 BM25 算法，真正的打分发生在 `InMemoryDocumentStore.bm25_retrieval`（第 7 章解构过）。检索器做的全部是"参数协调 + 委托 + 包装输出"。这正是 Haystack 把"检索能力"放在 store、把"组件契约"放在 retriever 的分层结果：同一个 store 可以被 BM25 检索器与 Embedding 检索器共用，而 retriever 只负责成为流水线里一个合法的 `@component`。

构造器里有几处值得注意的硬约束。`__init__` 第一件事是 `isinstance(document_store, InMemoryDocumentStore)` 检查，否则抛 `TypeError`——`InMemory*` 系列检索器与内存 store 强绑定。接着 `if top_k <= 0` 抛 `ValueError`。默认值是 `top_k: int = 10`、`scale_score: bool = False`、`filter_policy: FilterPolicy = FilterPolicy.REPLACE`。

`filter_policy` 决定运行时 `filters` 与初始化 `filters` 如何合并，逻辑在 `run` 开头：

```python
if self.filter_policy == FilterPolicy.MERGE and filters:
    filters = {**(self.filters or {}), **filters}
else:
    filters = filters or self.filters
```

`REPLACE`（默认）让运行时 filters 覆盖构造时的；`MERGE` 则把两者字典合并，运行时键优先。`top_k`/`scale_score` 同理：运行时传 `None` 就回落到构造时的实例属性。这种"运行时可覆盖、否则用默认"的模式贯穿所有检索器。

异步路径 `run_async` 与同步几乎一字不差，唯一区别是委托给 `bm25_retrieval_async` 并 `await`。Haystack 的检索器都成对提供同步/异步实现，便于在异步流水线里直接复用。

### to_dict 如何序列化内嵌的 store

检索器持有一个 store 实例，序列化时这个 store 必须一起被写出去，否则 `from_dict` 出来的检索器就成了空壳。看 `to_dict`：

```python
return default_to_dict(
    self,
    document_store=self.document_store,
    filters=self.filters,
    top_k=self.top_k,
    scale_score=self.scale_score,
    filter_policy=self.filter_policy.value,
)
```

`default_to_dict` 会递归调用 `self.document_store.to_dict()`，把内嵌 store 的完整配置嵌进检索器的 init 参数里（第 5 章序列化机制）。注意 `filter_policy` 被存成 `.value` 这个字符串枚举值；反序列化时 `from_dict` 里有 `init_params["filter_policy"] = FilterPolicy.from_str(...)` 把它还原成枚举。这是枚举字段序列化的标准处理。

## BM25 vs Embedding：输入差异是关键分水岭

`InMemoryBM25Retriever.run` 的第一个参数是 `query: str`——关键词检索吃的是原始查询字符串，store 内部用 BM25 算法做词频匹配。

而 `in_memory/embedding_retriever.py` 里 `InMemoryEmbeddingRetriever.run` 的第一个参数是 `query_embedding: list[float]`——向量检索吃的是已经编码好的查询向量，store 用 `embedding_retrieval` 做向量近邻搜索。这个签名差异决定了流水线接线方式：

- BM25 检索器：上游直接喂查询字符串，无需嵌入器；
- Embedding 检索器：上游必须先接一个 `TextEmbedder` 把查询编码成 `embedding`，再把 `embedding` 连到检索器的 `query_embedding` 输入。

Embedding 检索器多了一个 `return_embedding: bool = False` 参数（BM25 没有），控制召回的文档是否带回向量本体。默认 `False` 是为了省内存——多数下游只用 `content`，不需要向量。其余 `filters`/`top_k`/`scale_score`/`filter_policy` 的处理与 BM25 检索器完全一致。

这就是召回的两种基本路线：关键词召回快、可解释、对精确术语友好；向量召回懂语义、能跨表述匹配。实践中常把两者并行召回再合并（混合检索），但在组件层面它们就是两个独立的检索器外壳。

### FilterRetriever：只筛不排

`filter_retriever.py` 的 `FilterRetriever` 是最朴素的检索器，`run` 一行带过：

```python
return {"documents": self.document_store.filter_documents(filters=filters or self.filters)}
```

它不算相似度、不接受 `query`、也没有 `top_k`，纯粹按元数据条件 `filter_documents` 取出所有匹配文档。它持有的是通用 `DocumentStore`（不限内存），适合"取出某语言/某来源的全部文档"这类元数据驱动的场景。

### TextEmbeddingRetriever / MultiQueryTextRetriever：组合式外壳

`text_embedding_retriever.py` 的 `TextEmbeddingRetriever` 更进一步，它内部把一个 `TextEmbedder`（编码查询）和一个 `EmbeddingRetriever`（向量召回）打包成一个组件：输入一段文本，内部先编码再召回，省去用户手动连线。`multi_query_text_retriever.py` 的 `MultiQueryTextRetriever` 则持有一个文本检索器（`TextRetriever` 协议），用 `ThreadPoolExecutor` 把多条查询并行打到检索器上，再 `_deduplicate_documents` 去重、按分数排序合并——常与查询扩展（`QueryExpander`）配合。它们都不实现检索算法，仍是更高层的"外壳"。

## AutoMergingRetriever：按层级把碎片合并回父块

`auto_merging_retriever.py` 的 `AutoMergingRetriever` 是一类特殊的"后处理检索器"：它的 `run` 输入不是查询，而是上游召回的叶子文档 `documents: list[Document]`。它的前提是文档库由 `HierarchicalDocumentSplitter` 切成了树状层级——每个叶子块在 `meta` 里记着 `__parent_id`、`__level`、`__block_size`，父块记着 `__children_ids`。

思路很直白：如果同一个父块下被召回的子块足够多，那么"整段父块"往往比零散子块更有信息量，于是把子块合并成父块返回。`_check_valid_documents` 先校验叶子文档带齐 `__parent_id`/`__level`/`__block_size`，缺一即抛 `ValueError`。核心在 `_try_merge_level`：

```python
score = len(child_docs) / len(parent_doc.meta["__children_ids"])
if score > self.threshold:
    merged_docs.append(parent_doc)   # 合并成父块
else:
    docs_to_return.extend(child_docs) # 子块各自保留
```

合并比例 = 命中子块数 / 父块的子块总数，超过 `threshold`（构造时校验 `0 < threshold < 1`，默认 `0.5`）就合并。最妙的是递归：合并出的父块本身又有自己的父块，于是 `_try_merge_level(merged_docs, docs_to_return)` 继续向上一层尝试，直到 `if not merged_docs` 这一层没有任何合并发生才停。结果可能是不同层级文档的混合。`_get_parent_doc` 通过 `filter_documents({"field": "id", "operator": "==", "value": parent_id})` 回库取父块，这也是为什么文档里说它只支持 AstraDB/Elasticsearch/OpenSearch/PGVector/Qdrant 等能做精确 id 过滤的 store。

## SentenceWindowRetriever：按窗口扩展上下文

`sentence_window_retriever.py` 的 `SentenceWindowRetriever` 解决另一个问题：召回的块太短，缺上下文。它同样接在普通检索器之后，输入 `retrieved_documents`，对每篇命中文档去库里捞它前后相邻的若干块。

它依赖两个元数据字段：`source_id`（同一原文的所有块共享）和 `split_id`（块在原文中的序号），可通过 `source_id_meta_field`/`split_id_meta_field` 改名。窗口大小 `window_size: int = 3` 表示前后各取多少块。扩展的实质是构造一个范围过滤，见 `_build_filter_conditions`：

```python
min_before = split_id - window_size
max_after = split_id + window_size
conditions = [
    {"field": f"meta.{self.split_id_meta_field}", "operator": ">=", "value": min_before},
    {"field": f"meta.{self.split_id_meta_field}", "operator": "<=", "value": max_after},
    *source_id_filters,
]
return {"operator": "AND", "conditions": conditions}
```

也就是"同一 `source_id` 且 `split_id` 落在 `[split_id-window, split_id+window]` 区间内"的块全取出。取回后 `merge_documents_text` 把这些块拼成连续文本：它按 `split_idx_start` 排序，并用 `start = max(start, last_idx_end)` 消除相邻块之间的重叠字符，避免拼接出重复内容。输出有两个口：`context_windows`（拼接后的字符串列表）和 `context_documents`（按 `split_idx_start` 排序的上下文文档）。

`raise_on_missing_meta_fields: bool = True` 控制容错：为 `True` 时缺字段直接抛错；为 `False` 时 `_retrieve_context_for_document` 会 `logger.warning` 并退化为只返回原文档本身。AutoMerging 是"向上合并到父块"，SentenceWindow 是"向两侧平铺取邻居"——一个走层级树，一个走线性序号，但都是用元数据做"上下文扩展"。

## 重排器：精排为什么用 cross-encoder

召回带回的候选可能有几十上百条，分数也未必精准（BM25 的词频、向量的余弦都是粗略代理）。重排器对 `query` 和每篇文档做更细致的相关性判断。最典型的是 cross-encoder。

看 `sentence_transformers_similarity.py` 的 `SentenceTransformersSimilarityRanker`。它默认模型 `model: str | Path = "cross-encoder/ms-marco-MiniLM-L-6-v2"`，`warm_up` 时构造 `CrossEncoder`（来自 `sentence-transformers`）。cross-encoder 与召回阶段的双塔嵌入有本质区别：嵌入是把 query 和 doc 各自独立编码成向量再算距离；cross-encoder 是把 query 和 doc **拼成一对一起喂进模型**，让注意力机制充分交互，再直接输出一个相关性分数。这更准，但每个 query-doc 对都要过一次模型，所以慢——这正是"只能对召回后的少量候选做精排"的原因。Sentence-Transformers 的 CrossEncoder 是这类精排模型的标准实现。

`run` 的流程：先 `_deduplicate_documents` 按 id 去重（保留最高分），再给每篇文档拼上 `meta_fields_to_embed` 指定的元数据和前后缀，构造 `prepared_documents`，然后：

```python
activation_fn = Sigmoid() if scale_score else Identity()
ranking_result = self._cross_encoder.rank(
    query=prepared_query,
    documents=prepared_documents,
    batch_size=self.batch_size,
    activation_fn=activation_fn,
    convert_to_numpy=True,
    return_documents=False,
)
```

`scale_score: bool = True`（默认）会用 `Sigmoid` 把原始 logit 压到 `[0,1]`，否则用 `Identity` 保留原始 logit。`rank` 返回带 `corpus_id`/`score` 的结果，代码用 `replace(deduplicated_documents[index], score=score)` 把分数写回文档（`replace` 来自 `dataclasses`，生成新副本不改原对象），最后 `score_threshold` 过滤低分、`[:top_k]` 截断。`query_prefix`/`query_suffix`/`document_prefix`/`document_suffix` 是为 `bge`、`qwen` 这类要求加指令前后缀的重排模型准备的。注意这个类的 `__init__` 会发 `FutureWarning`，提示它将在 3.0 迁移到 `sentence-transformers-haystack` 包。

`transformers_similarity.py` 的 `TransformersSimilarityRanker` 是更底层的等价实现（被标记为 legacy）：它直接用 `AutoModelForSequenceClassification` + `AutoTokenizer`，在 `run` 里手写 `DataLoader` 分批前向，`model(**features).logits.squeeze(dim=1)` 取分数，再 `torch.sigmoid(similarity_scores * calibration_factor)` 做校准缩放（多了一个 `calibration_factor: float | None = 1.0` 参数），最后 `torch.sort(..., descending=True)` 排序。两者算的是同一件事，但前者借 `CrossEncoder` 封装、后者裸用 transformers。`warm_up` 把重模型加载推迟到流水线启动时，避免构造组件就吃显存——这是所有带模型的重排器的共同模式。

`llm_ranker.py` 的 `LLMRanker` 走另一条路：它不算相似度，而是把 `query` 和编号后的文档列表塞进 `DEFAULT_PROMPT_TEMPLATE`，让一个 `ChatGenerator`（默认 `OpenAIChatGenerator`，`gpt-4.1-mini`，`temperature: 0.0`）以严格 JSON schema 返回相关文档的 `index` 顺序。用大模型直接判定相关性，最灵活也最贵。

## 不靠模型的重排：位置、元数据与去冗

重排不一定要 cross-encoder。Haystack 还提供几种纯规则/纯几何的重排器。

`lost_in_the_middle.py` 的 `LostInTheMiddleRanker` 处理的是 LLM 的"中间遗忘"问题：大模型对超长上下文里**首尾**的信息更敏感，中间的容易被忽略。这一现象出自 "Lost in the Middle: How Language Models Use Long Contexts" 论文。所以这个重排器假定文档**已被前序组件按相关性排好序**，它不需要 `query`，只重新摆放位置——把最相关的放首尾、最不相关的塞中间。核心是 `insert` 的位置计算：

```python
insertion_index = len(lost_in_the_middle_indices) // 2 + len(lost_in_the_middle_indices) % 2
lost_in_the_middle_indices.insert(insertion_index, doc_idx)
```

按相关性依次取文档，交替插到中间位置的两侧，最终形成"最相关在两端、最弱在中间"的排布。它还支持 `word_count_threshold`：累计词数超阈值就停止纳入后续文档，控制喂给 LLM 的上下文总长。它通常是构建 prompt 前的最后一个组件。

`meta_field.py` 的 `MetaFieldRanker` 按某个元数据字段排序（如 `rating`、日期），而不是相关性。它的精巧在于不是简单替换排序，而是用 `weight: float = 1.0` 把"上游相关性排名"和"元数据排名"融合：`weight=0` 完全用上游、`weight=1` 完全用元数据、`0.5` 各占一半。融合算法由 `ranking_mode` 决定，默认 `reciprocal_rank_fusion`（RRF），`_calculate_rrf` 里 `1 / (k + rank)`、`k=61`（论文建议 60，因 Python 0 基索引 +1）；另一种 `linear_score` 用线性缩放。`meta_value_type` 可把字符串元数据解析成 `float`/`int`/`date` 再排序，`missing_meta` 控制缺字段的文档是 `drop`/`top`/`bottom`。

`meta_field_grouping_ranker.py` 的 `MetaFieldGroupingRanker` 不排相关性，而是按 `group_by`（可选 `subgroup_by`）把同组文档聚到一起、组内可按 `sort_docs_by` 排序，输出一个分组有序的扁平列表——让同主题内容相邻，便于 LLM 处理。

`sentence_transformers_diversity.py` 的 `SentenceTransformersDiversityRanker` 解决冗余：召回的 top 文档常常彼此高度相似，浪费上下文。它用一个 `SentenceTransformer` 编码 query 和文档，提供两种策略。`greedy_diversity_order`（`_greedy_diversity_order`）先选与 query 最像的，之后每步选"与已选集合平均相似度最低"的文档，逐步逼近最大多样性。`maximum_margin_relevance`（MMR，`_maximum_margin_relevance`）则用 `lambda_threshold: float = 0.5` 在相关性与多样性间权衡：

```python
mmr_score = lambda_threshold * relevance_score - (1 - lambda_threshold) * diversity_score
```

`lambda` 越接近 1 越重相关性、越接近 0 越重多样性，`_check_lambda_threshold` 强制它落在 `[0,1]`。这两种都是几何去冗，不依赖 cross-encoder 打分。

## 本章小结

- 检索器是 store 检索能力的"组件外壳"：`InMemoryBM25Retriever`/`InMemoryEmbeddingRetriever` 的 `run` 核心只是把 `top_k`/`filters`/`scale_score` 委托给 store 的 `bm25_retrieval`/`embedding_retrieval`，真正的算法在 store 里。
- BM25 检索器吃 `query: str`（关键词），Embedding 检索器吃 `query_embedding: list[float]`（向量），这个输入差异决定了流水线是否需要先接一个 `TextEmbedder`。
- `filter_policy` 用 `REPLACE`（默认覆盖）或 `MERGE`（字典合并）协调构造时与运行时 `filters`；`top_k`/`scale_score` 运行时传 `None` 则回落到实例默认；`to_dict` 会递归把内嵌 store 一起序列化，枚举存成 `.value`。
- `AutoMergingRetriever` 按层级树工作：命中子块比例超 `threshold`（默认 `0.5`）就 `_try_merge_level` 递归合并回父块，依赖 `__parent_id`/`__children_ids` 等元数据与精确 id 过滤。
- `SentenceWindowRetriever` 按线性序号扩展：用 `split_id±window_size` 的范围过滤捞邻居块，`merge_documents_text` 借 `split_idx_start` 去重拼接成连续上下文。
- cross-encoder 精排（`SentenceTransformersSimilarityRanker`/`TransformersSimilarityRanker`）把 query-doc 拼成一对一起过模型，比双塔召回更准但更慢，故只对召回后的少量候选做精排；`scale_score`+`Sigmoid` 把 logit 压到 `[0,1]`。
- 所有带模型的重排器都用 `warm_up` 把重模型加载延迟到启动时，`run` 内会先 `_deduplicate_documents` 去重，再用 `dataclasses.replace` 把新分数写回文档副本。
- 不靠模型的重排：`LostInTheMiddleRanker` 做首尾位置重排（应对中间遗忘），`MetaFieldRanker` 用 RRF/线性融合按元数据排序，`MetaFieldGroupingRanker` 按字段分组聚合，`SentenceTransformersDiversityRanker` 用贪心多样性或 MMR 去冗。
- 召回与精排分离的本质：召回快而粗（覆盖率优先），精排准而慢（精度优先），两段流水线各取所长。

## 动手实验

1. **实验一：观察委托链路** — 在本地用 `InMemoryDocumentStore` 写入若干 `Document`，构造 `InMemoryBM25Retriever(doc_store, top_k=3)`，在 `run` 委托行 `self.document_store.bm25_retrieval(...)` 处打断点，确认检索器自身不算分、分数来自 store；再把 `filter_policy` 改成 `MERGE`，运行时传一个 `filters`，验证它与构造时 filters 被字典合并。
2. **实验二：BM25 vs Embedding 输入差异** — 同一批文档分别建 BM25 与 Embedding 两条召回流水线。Embedding 路线必须先接 `SentenceTransformersTextEmbedder` 把查询转成 `query_embedding` 再连到检索器；故意把字符串直接喂给 `InMemoryEmbeddingRetriever.run` 看报错，体会两者签名 `query` 与 `query_embedding` 的根本区别。
3. **实验三：上下文扩展对比** — 用 `HierarchicalDocumentSplitter` 造层级文档接 `AutoMergingRetriever(threshold=0.5)`，再用普通 `DocumentSplitter` 造线性块接 `SentenceWindowRetriever(window_size=2)`。分别只命中 2 个相邻子块，对比前者是否合并成父块、后者输出的 `context_windows` 是否拼出连续无重叠文本。
4. **实验四：精排与去冗叠加** — 对同一召回结果先后接 `SentenceTransformersSimilarityRanker`（cross-encoder 精排）和 `SentenceTransformersDiversityRanker(strategy="maximum_margin_relevance", lambda_threshold=0.3)`，打印两步后的文档顺序与 `score`；把 `lambda_threshold` 调到 `0.8` 再跑一次，观察 MMR 从"重多样性"切到"重相关性"时结果如何变化。

> **下一章预告**：召回与精排把最相关的上下文备齐之后，下一步就是把它们和用户问题组织成提示词、交给大模型生成答案。第 10 章《生成器与 Prompt 构建》将解构 `PromptBuilder` 的 Jinja2 模板渲染、`OpenAIGenerator`/`ChatGenerator` 的调用契约与流式输出，看 Haystack 如何把检索结果织进一次 LLM 调用。
