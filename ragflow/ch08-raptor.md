# 第 8 章 RAPTOR 树状摘要

前面几章里，文档被切成一个个 chunk，向量化后平铺地写进检索库。这种"只有叶子层"的结构在短问答里足够用，但当问题需要跨越整篇长文的高层语义——"这份年报整体表达了什么趋势"、"这一章作者的核心论点是什么"——单个 chunk 往往只携带局部细节，检索召回的也只是碎片，缺少能直接回答宏观问题的"概括性"内容。RAPTOR（Recursive Abstractive Processing for Tree-Organized Retrieval，递归抽象处理树状组织检索）就是为了补上这个缺口：它把叶子 chunk 自底向上做层次聚类，每个簇交给 LLM 生成一段摘要，摘要本身又作为新一层"节点"继续聚类、继续摘要，直到收敛，最终在原始 chunk 之上长出一棵多层摘要树 [[RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval]](https://arxiv.org/abs/2401.18059)。检索时，这些不同抽象层级的节点和叶子 chunk 一起被索引，于是一次查询既能命中细节，也能命中高层概括。

RAGFlow 的 RAPTOR 实现集中在 `rag/raptor.py`，配置与触发逻辑分散在 `api/utils/validation_utils.py`、`api/utils/api_utils.py` 与 `rag/svr/task_executor.py`，接入文档管线的服务层在 `rag/svr/task_executor_refactor/raptor_service.py`，而判定与写回工具在 `rag/utils/raptor_utils.py`。本章按"何时触发 → 加载哪些 chunk → 核心算法（降维、聚类、最优簇数、递归摘要）→ 随机性与终止条件 → 摘要节点如何写回检索库"的顺序逐层拆开，并顺带看一眼 RAGFlow 在经典 RAPTOR 之外另起炉灶的 Psi 合并树。

## 何时触发：默认关闭的可选增强

RAPTOR 不是默认行为。在 `api/utils/validation_utils.py` 的 `RaptorConfig` 里，开关 `use_raptor` 的默认值就是 `False`：

```python
class RaptorConfig(Base):
    """Dataset parser configuration for RAPTOR summary generation."""

    use_raptor: Annotated[bool, Field(default=False)]
    prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
        Field(
            default="Please summarize the following paragraphs. Be careful with the numbers, do not make things up. Paragraphs as following:\n      {cluster_content}\nThe above is the content you need to summarize."
        ),
    ]
    max_token: Annotated[int, Field(default=256, ge=1, le=2048)]
    threshold: Annotated[float, Field(default=0.1, ge=0.0, le=1.0)]
    max_cluster: Annotated[int, Field(default=64, ge=1, le=1024)]
    random_seed: Annotated[int, Field(default=0, ge=0)]
    scope: Annotated[Literal["file", "dataset"], Field(default="file")]
    clustering_method: Annotated[Literal["gmm", "ahc"], Field(default="gmm")]
    tree_builder: Annotated[Literal["raptor", "psi"], Field(default="raptor")]
    auto_disable_for_structured_data: Annotated[bool, Field(default=True)]
    ext: Annotated[dict, Field(default={})]
```

这段 Pydantic 模型把整章后面要用到的关键旋钮一次性摆了出来：摘要提示词 `prompt`（注意它强制包含 `{cluster_content}` 占位符，且要求模型"小心数字，不要编造"），单条摘要的 token 上限 `max_token`（默认 256，区间 1~2048），GMM 软分配的概率阈值 `threshold`（默认 0.1），最大簇数 `max_cluster`（默认 64），保证可复现的随机种子 `random_seed`（默认 0），处理范围 `scope`（`file` 逐文档或 `dataset` 全库合并），聚类方法 `clustering_method`（`gmm` 或 `ahc`）以及树构建器 `tree_builder`（`raptor` 经典递归或 `psi` 合并树）。在 `api/utils/api_utils.py` 的默认 parser 配置中，只有 `naive`（通用切分）模板把 `raptor.use_raptor` 设为 `True`，而 `qa`、`manual`、`paper`、`book`、`laws`、`presentation` 等模板统统显式写成 `use_raptor: False`，`knowledge_graph` 模板也关闭。这说明 RAGFlow 把 RAPTOR 定位成"长篇连续文本"的增强，对问答对、表格、简历这类结构化或短文档并不默认开启。

真正调度 RAPTOR 的是 `rag/svr/task_executor.py`。当任务类型为 `raptor` 时，它先按 dataset 取出 `parser_config`，如果发现 `use_raptor` 还没打开，就地补一份默认 RAPTOR 配置并写回 `KnowledgebaseService`：

```python
if task_type == "raptor":
    ...
    kb_parser_config = kb.parser_config
    if not kb_parser_config.get("raptor", {}).get("use_raptor", False):
        kb_parser_config.update(
            {
                "raptor": {
                    "use_raptor": True,
                    "prompt": "Please summarize the following paragraphs. ...",
                    "max_token": 256,
                    "threshold": 0.1,
                    "max_cluster": 64,
                    "random_seed": 0,
                    "scope": "file",
                    "clustering_method": "gmm",
                    "tree_builder": "raptor",
                },
            }
        )
```

随后它绑定一个 CHAT 类型的 LLM（摘要要用到对话模型），在 `kg_limiter` 这个并发信号量的保护下调用 `run_raptor_for_kb`。值得注意的是，RAPTOR 任务并不绑在单篇文档上，而是用一个虚拟文档 ID `GRAPH_RAPTOR_FAKE_DOC_ID` 代表整个数据集层级的产物，并通过 `queue_raptor_o_graphrag_tasks` 入队——也就是说，它和 GraphRAG 共享一套"数据集级后处理任务"的调度骨架，这也是为什么 `api/db/__init__.py` 把 `PipelineTaskType.RAPTOR` 列进 `PIPELINE_SPECIAL_PROGRESS_FREEZE_TASK_TYPES`，让它在进度条上单独冻结一段。

## 加载哪些 chunk：复用既有向量，绝不重新切分

RAPTOR 的输入不是原始文件，而是已经切好、已经向量化的叶子 chunk。`RaptorService.run_raptor_for_kb`（在 `raptor_service.py`，与 `task_executor.py` 里的同名函数实现一致）先按向量维度拼出字段名 `vctr_nm = "q_%d_vec" % vector_size`，然后从检索库里把每个 chunk 的 `content_with_weight`（正文）和对应向量字段一起拉出来。`_load_doc_chunks` 里有一处很关键的防御：

```python
for d in settings.retriever.chunk_list(
        doc_id, ctx.tenant_id, [str(ctx.kb_id)],
        fields=fields,
        sort_by_position=True
):
    if vctr_nm not in d or d[vctr_nm] is None:
        skipped_chunks += 1
        logging.warning(f"RAPTOR: Chunk missing vector field '{vctr_nm}' in doc {doc_id}, skipping")
        continue
    chunks.append((d["content_with_weight"], np.array(d[vctr_nm])))
```

凡是缺少当前 embedding 维度向量的 chunk 都被跳过。这是因为 RAPTOR 的聚类完全建立在向量空间上，如果某文档是用旧的 embedding 模型解析的、维度对不上 `vctr_nm`，强行混进来会破坏距离度量。`sort_by_position=True` 则保证 chunk 按原文位置顺序进入，让"层"在概念上仍对应连续的文档区域。最终每个 chunk 被表达成 `(text, embedding)` 二元组的列表，这就是后面所有聚类、摘要操作的统一数据结构。

`scope` 决定加载粒度。`file` 模式下 `_run_file_level_raptor` 逐文档调用 `_load_doc_chunks`，每篇文档独立长出自己的摘要树；`dataset` 模式下 `_run_dataset_level_raptor` 用 `_load_all_doc_chunks` 把全库所有文档的 chunk 倒进同一个列表，跨文档聚类，产物挂在 `GRAPH_RAPTOR_FAKE_DOC_ID` 名下。两种模式之间还有迁移逻辑：服务层会通过 `_get_raptor_chunk_methods` 检查目标文档是否已有某种 tree_builder 的 RAPTOR chunk（这是"断点续跑"式的 checkpoint，已经算过的不重复算），并在切换 file/dataset 范围或更换 tree_builder 时，把过时的旧摘要排进 `cleanup_raptor_chunks` 待删队列，等新摘要成功写入后再清理。在跑 RAPTOR 之前，`should_skip_raptor`（`rag/utils/raptor_utils.py`）还会对 Excel/CSV（`STRUCTURED_EXTENSIONS`）以及用 table 解析器或开了 `html4excel` 的表格型 PDF 自动跳过——只要 `auto_disable_for_structured_data` 没被显式关掉，对这些结构化数据做语义聚类摘要意义不大，干脆不做。

## 核心算法：UMAP 降维 + GMM 聚类 + LLM 递归摘要

经典 RAPTOR 的主循环在 `RecursiveAbstractiveProcessing4TreeOrganizedRetrieval.__call__` 里。它维护一个不断增长的 `chunks` 列表和一个 `layers` 列表，后者记录每一层在 `chunks` 中的 `[start, end)` 边界。初始时 `layers = [(0, len(chunks))]`，`start, end = 0, len(chunks)`，然后进入 `while end - start > 1` 的循环：每轮只对当前层（`chunks[start:end]`）的向量做聚类与摘要，把生成的摘要 append 到 `chunks` 尾部，构成上一层，再把 `start = end; end = len(chunks)` 推进到新层，直到某层只剩一个节点或收敛。

每一层的第一步是降维。直接在高维 embedding（动辄上千维）上跑高斯混合是灾难性的——维度灾难会让 BIC 失真、协方差矩阵难以稳定估计。所以 RAGFlow 先用 UMAP 把向量压到低维：

```python
n_neighbors = int((len(embeddings) - 1) ** 0.8)
reduced_embeddings = umap.UMAP(
    n_neighbors=max(2, n_neighbors),
    n_components=min(12, len(embeddings) - 2),
    metric="cosine",
).fit_transform(embeddings)
```

这里有三个设计细节值得拆。`n_neighbors` 取 `(N-1) ** 0.8`，随节点数次线性增长，节点越多看的邻域越大，但不至于像线性那样在大数据集上把全局结构抹平；目标维度 `n_components` 封顶 12，并保证不超过 `len(embeddings) - 2`，对小簇做兜底；`metric="cosine"` 与文本 embedding 习惯用的余弦相似度一致 [[UMAP: Uniform Manifold Approximation and Projection]](https://arxiv.org/abs/1802.03426)。当本层只剩两个节点时，循环直接走 `await summarize([start, start + 1])` 把它俩合成一段摘要，跳过降维聚类——两个点没有"聚类"可言。

降维后默认走 GMM 软聚类。簇数不是写死的，而是由 `_get_optimal_clusters` 用贝叶斯信息准则（BIC）自动选：

```python
def _get_optimal_clusters(self, embeddings, random_state, task_id=""):
    """Choose the GMM cluster count with the lowest BIC score."""
    max_clusters = min(self._max_cluster, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        self._check_task_canceled(task_id, "get optimal clusters")
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters
```

它从 1 到 `min(max_cluster, N)` 逐个拟合 GMM，记录每个簇数的 BIC，取 BIC 最低者作为最优簇数。BIC 在似然里扣掉了模型复杂度惩罚项，因此能在"拟合得更好"和"簇太多过拟合"之间自动权衡，避免把每个 chunk 都拆成单独一簇 [[Gaussian mixture models — scikit-learn]](https://scikit-learn.org/stable/modules/mixture.html)。选定簇数后，真正分配标签用的是 GMM 的软概率：

```python
n_clusters = self._get_optimal_clusters(reduced_embeddings, random_state, task_id=task_id)
if n_clusters == 1:
    lbls = [0 for _ in range(len(reduced_embeddings))]
else:
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(reduced_embeddings)
    probs = gm.predict_proba(reduced_embeddings)
    lbls = [np.where(prob > self._threshold)[0] for prob in probs]
    lbls = [lbl[0] if isinstance(lbl, np.ndarray) else lbl for lbl in lbls]
```

`predict_proba` 给出每个点属于各簇的后验概率，`np.where(prob > self._threshold)` 取出所有概率超过阈值（默认 0.1）的簇——这正是 RAPTOR 论文里"软聚类、一个节点可属于多个簇"的思想：一段文字在语义上可能横跨多个主题。不过 RAGFlow 这一版随后用 `lbl[0]` 只保留首个超阈值的簇，把软标签收敛成硬标签来组织树结构，阈值在这里更多是过滤极低概率归属的噪声。每个簇对应的 chunk 下标 `ck_idx` 收集好后，为每个簇创建一个 `summarize` 协程并发执行。

摘要的核心在 `_summarize_texts`。它要解决一个现实约束：一个簇可能包含很多 chunk，全拼起来会超过 LLM 上下文窗口。代码先按比例给每条 chunk 分配可用 token：

```python
len_per_chunk = int((self._llm_model.max_length - self._max_token) / len(texts))
cluster_content = "\n".join([truncate(t, max(1, len_per_chunk)) for t in texts])
```

用 LLM 的总上下文 `max_length` 减去预留给输出的 `max_token`，再除以簇内 chunk 数，得到每条 chunk 的截断长度，逐条 `truncate` 后用换行拼接，填进 prompt 的 `{cluster_content}`。调用时 `max_tokens` 取 `max(self._max_token, 512)`（注释 `# fix issue: #10235` 表明这是为修一个截断过早的问题而设的下限），保证摘要不会被强行压到太短。摘要文本拿到后立即 `_embedding_encode` 算出它自己的向量，返回 `(cnt, embds)`——这一步至关重要：新生成的摘要必须带着自己的 embedding，才能作为节点参与上一层的聚类。整个 LLM 调用与 embedding 都包了缓存（`get_llm_cache`/`get_embed_cache`）和最多 3 次重试，并在 `_chat` 里用正则 `re.sub(r"^.*</think>", "", ...)` 剥掉推理模型可能吐出的思维链。

## 鲁棒性：错误熔断、取消检查与终止条件

长文档跑完整棵 RAPTOR 树要发几十甚至上百次 LLM 请求，任何一次都可能失败。RAGFlow 用一个错误计数器做熔断。`__init__` 里 `self._max_errors = max(1, max_errors)`（由环境变量 `RAPTOR_MAX_ERRORS` 控制，默认 3），`_summarize_texts` 捕获异常时累加 `self._error_count`，单簇失败只是跳过并回调一条警告，但一旦累计到阈值就抛 `RuntimeError` 整体中止：

```python
except Exception as exc:
    self._error_count += 1
    warn_msg = f"[RAPTOR] Skip cluster ({len(texts)} chunks) due to error: {exc}"
    logging.warning(warn_msg)
    if callback:
        callback(msg=warn_msg)
    if self._error_count >= self._max_errors:
        raise RuntimeError(f"RAPTOR aborted after {self._error_count} errors. Last error: {exc}") from exc
    return None
```

这种"允许零星失败、不允许系统性失败"的策略，避免了因为某个簇文本异常就丢掉整棵树，也避免了在模型彻底不可用时还死磕几百次。与之配套的是密布全流程的取消检查 `_check_task_canceled`：在选最优簇数、聚类前、LLM 调用前、embedding 前都调用 `has_canceled(task_id)`，一旦用户取消任务就抛 `TaskCanceledException`。`_chat` 和 `_summarize_texts` 还分别套了 `@timeout(60 * 20)`，单次摘要最长 20 分钟。

终止条件有几重。最外层是 `while end - start > 1`——当某一层收敛到只剩一个节点（树根），循环自然结束。但代码不假定每层都一定会产出摘要：每轮结束都检查 `produced = len(chunks) - end`，如果某层一个摘要都没生成（比如全部簇都因错误被跳过），就 `logging.warning(...)` 后 `break`，防止死循环：

```python
produced = len(chunks) - end
assert produced <= n_clusters, "{} vs. {}".format(produced, n_clusters)
if produced < n_clusters:
    logging.warning("RAPTOR layer produced %d/%d cluster summaries; skipped %d cluster(s) due to errors", ...)
if produced == 0:
    logging.warning("RAPTOR layer produced no summaries; stopping materialization")
    break
layers.append((end, len(chunks)))
```

随机性则被 `random_seed` 牢牢钉住。`__call__` 接收 `random_state` 参数（来自配置的 `random_seed`，默认 0），并一路透传给 `_get_optimal_clusters` 和正式拟合的 `GaussianMixture(..., random_state=random_state)`。GMM 的初始化（k-means++ 初值、EM 迭代起点）本质上是随机的，固定种子保证同样的输入每次都聚出同样的簇、生成同样结构的树，这对调试、回归测试和"重跑结果一致"的用户预期都很重要。

## 另一条路：AHC 聚类与 Psi 合并树

除了默认的 GMM，`clustering_method` 还可以选 `ahc`（凝聚层次聚类）。`_get_clusters_ahc` 先用 Ward linkage 做一棵完整的树状图（`distance_threshold=0` 让它合并到底），然后看合并距离序列的"最大间隙"来决定切几刀：

```python
distances = full_clust.distances_
if len(distances) > 1:
    gaps = np.diff(distances)
    max_gap_idx = int(np.argmax(gaps))
    n_clusters = max(1, min(n - max_gap_idx - 1, self._max_cluster))
```

直觉是：树状图里相邻两次合并的距离突然变大的地方，往往就是"自然簇"之间的分界。选好簇数后 `_adjust_tree_nodes` 还会做最多 5 轮"重分配到最近质心"的微调，让边界更干净。

更激进的是 `tree_builder="psi"` 这条线。Psi 不走"降维-聚类-摘要"的统计路线，而是构造一棵自底向上的合并树：`_rank_leaf_pairs` 把所有叶子两两按余弦相似度排序，`_PsiUnionFind` 这个定制并查集按相似度从高到低逐对合并，相似的 chunk 先并到一起，形成确定性的二叉合并结构。为了控制 N² 的对排序开销，`_split_psi_buckets` 在节点数超过 `psi_bucket_size`（默认 1024）时先用类 k-means 的余弦中心法把节点分桶，桶内做精确排序，桶根再合并；`_rebalance_psi_tree` 则保证任何节点的扇出不超过 `max_cluster`。建好结构后 `_build_psi_layers` 用 `_psi_layers` 按节点高度分层，自底向上对每个内部节点的子节点文本调用 `_summarize_texts`。Psi 的好处是完全确定、对相似度更敏感，是 RAGFlow 在经典 RAPTOR 之外提供的可选树形态，二者共用同一套摘要与写回逻辑。

## 摘要节点如何写回检索库

跑完 `raptor(...)` 拿到的是 `(processed_chunks, layers)`：`processed_chunks` 里前 `original_length` 个是原始叶子 chunk，从 `original_length` 往后全是新生成的摘要节点。`_generate_raptor`（`raptor_service.py`）只把新生成的部分组装成可写入检索库的文档：

```python
doc = {
    "doc_id": doc_id,
    "kb_id": [str(ctx.kb_id)],
    "docnm_kwd": effective_doc_name,
    "title_tks": rag_tokenizer.tokenize(effective_doc_name),
    "raptor_kwd": "raptor",
    "extra": {"raptor_method": tree_builder},
}
...
for idx, (content, vctr) in enumerate(processed_chunks[original_length:], start=original_length):
    d = copy.deepcopy(doc)
    d["id"] = make_raptor_summary_chunk_id(content, doc_id)
    d["create_time"] = str(datetime.now()).replace("T", " ")[:19]
    d["create_timestamp_flt"] = datetime.now().timestamp()
    d[vctr_nm] = vctr.tolist()
    d["content_with_weight"] = content
    d["content_ltks"] = rag_tokenizer.tokenize(content)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
    d["raptor_layer_int"] = chunk_layer.get(idx, 1)
    res.append(d)
    tk_count += num_tokens_from_string(content)
```

这里有三处标记决定了摘要节点的"身份"。`raptor_kwd` 固定写 `"raptor"`，是判定一条 chunk 是否为 RAPTOR 摘要的核心标记——`collect_raptor_chunk_ids`、`delete_raptor_chunks`、checkpoint 检查全靠它过滤。`extra.raptor_method` 记录是哪个 tree_builder（`raptor` 或 `psi`）生成的，支持后续在两种构建器之间迁移、清理旧产物。`raptor_layer_int` 标注该节点在树里的层号，由前面 `layers` 边界反推出来的 `chunk_layer` 映射给出（叶子之上从 1 开始），默认兜底为 1。

ID 的生成方式 `make_raptor_summary_chunk_id` 也有讲究：

```python
def make_raptor_summary_chunk_id(content: str, doc_id: str) -> str:
    """Build the stable ID used for generated RAPTOR summary chunks."""
    return xxhash.xxh64((content + str(doc_id)).encode("utf-8")).hexdigest()
```

它用内容加 doc_id 的 xxh64 哈希作为稳定 ID，意味着同样的摘要内容在同一文档下会得到同样的 ID，天然具备幂等性——重跑时相同内容不会重复堆积。同时摘要正文经过 `rag_tokenizer.tokenize` 和 `fine_grained_tokenize` 切成 `content_ltks` 与 `content_sm_ltks`，与普通 chunk 完全一样地参与全文检索；它也带 `q_%d_vec` 向量参与向量检索。换句话说，RAPTOR 摘要节点写回后，在检索阶段和叶子 chunk 是一等公民、平起平坐，并不需要检索器特殊处理——一次查询自然会在叶子层的细节和摘要层的概括之间按相关度竞争召回。如果数据集开了 PageRank（`ctx.pagerank`），摘要节点也会带上 `PAGERANK_FLD` 权重。

最后是事务式的清理与回滚。新摘要成功写入后，`task_executor.py` 才遍历 `raptor_cleanup_chunks` 调 `delete_raptor_chunks` 删除过时摘要（迁移 tree_builder 或切换 file/dataset 范围时产生）；而插入过程若被取消，`insert_chunks` 会专门挑出已写入的、`raptor_kwd == "raptor"` 的 chunk 回滚删除，避免下次 checkpoint 把"插了一半"的残缺树误判成"已完成"。这一前一后的对称设计，让 RAPTOR 这种重量级、易中断的后处理任务在反复重跑下仍能保持检索库的一致性。

## 本章小结

- RAPTOR 解决"只有叶子层、缺跨 chunk 高层语义"的问题：自底向上层次聚类 + 递归摘要，在原始 chunk 之上长出多层摘要树，让宏观问题也能被直接召回。
- 它默认关闭（`RaptorConfig.use_raptor=False`），仅 `naive` 模板默认开启；`api/utils/api_utils.py` 对 qa/manual/paper/book/laws 等模板显式关闭，定位为长篇连续文本的可选增强。
- 输入是已切分、已向量化的叶子 chunk（`content_with_weight` + `q_%d_vec`），缺当前维度向量的 chunk 直接跳过；`scope` 决定 file 逐文档或 dataset 全库聚类，产物挂在 `GRAPH_RAPTOR_FAKE_DOC_ID` 下。
- 每层先用 `umap.UMAP`（`metric="cosine"`、`n_components` 封顶 12、`n_neighbors=(N-1)**0.8`）降维，化解高维下 GMM 失真的问题。
- 最优簇数由 `_get_optimal_clusters` 遍历 1~`max_cluster` 取 BIC 最低者决定；GMM 用 `predict_proba` 配 `threshold`（默认 0.1）做软分配，再收敛为硬标签组织树。
- 摘要在 `_summarize_texts` 中按 `(max_length - max_token)/簇大小` 给每条 chunk 分配截断 token，调用时 `max_tokens=max(max_token, 512)`，生成后立即重新 embedding 以参与上一层聚类。
- 随机性由 `random_seed`（默认 0）透传给所有 `GaussianMixture` 的 `random_state`，保证聚类与树结构可复现。
- 鲁棒性靠 `RAPTOR_MAX_ERRORS`（默认 3）熔断、密集的 `_check_task_canceled` 取消检查和 `@timeout(60*20)` 超时；某层零产出则 `break` 防死循环。
- 除默认 GMM，还可选 `ahc`（Ward + 最大间隙切簇 + 质心微调）以及 `psi` 合并树（按余弦相似度并查集合并、分桶控制复杂度、再平衡扇出）。
- 摘要节点写回时带 `raptor_kwd="raptor"`、`extra.raptor_method`、`raptor_layer_int` 三个标记，ID 用内容+doc_id 的 xxh64 哈希保证幂等，并正常 tokenize/向量化，检索时与叶子 chunk 平等竞争召回。
- 清理与回滚成对存在：新摘要成功后才删旧摘要，取消时回滚已写入的 raptor chunk，避免 checkpoint 误判半成品。

## 动手实验

1. **实验一：观察默认配置与触发开关** — 打开 `api/utils/validation_utils.py` 的 `RaptorConfig` 与 `api/utils/api_utils.py` 的模板默认配置，列出哪些 parser 模板默认开启 RAPTOR、哪些关闭；再到 `rag/svr/task_executor.py` 的 `if task_type == "raptor"` 分支，确认当配置里 `use_raptor` 为假时系统如何就地补全默认配置并写回 KB。

2. **实验二：手算一层的聚类与 token 预算** — 假设某层有 50 个 chunk、LLM `max_length=8192`、`max_token=256`，按 `__call__` 里的公式算出 `n_neighbors=int(49**0.8)` 和 `n_components=min(12, 48)` 的取值；再设某簇含 8 个 chunk，按 `_summarize_texts` 的 `len_per_chunk=int((8192-256)/8)` 算出每条 chunk 的截断长度，理解 token 预算如何随簇大小收缩。

3. **实验三：验证随机种子的可复现性** — 在 `_get_optimal_clusters` 与 `__call__` 的正式拟合处定位所有 `random_state=random_state` 的传入点，解释为什么固定 `random_seed=0` 能让同一份输入每次聚出相同的簇；再设想把种子改成时间戳会带来什么后果（树结构每次不同、摘要 ID 漂移、checkpoint 失效）。

4. **实验四：追踪一个摘要节点的写回与清理** — 从 `_generate_raptor` 出发，跟踪一个摘要 `(content, vctr)` 如何被赋予 `make_raptor_summary_chunk_id` 的 ID、`raptor_kwd`/`raptor_layer_int` 标记并 tokenize；再到 `rag/utils/raptor_utils.py` 的 `collect_raptor_chunk_ids` 和 `delete_raptor_chunks`，说明系统如何凭这些标记识别、迁移与删除过时的 RAPTOR 摘要，以及取消任务时为何要专门回滚 `raptor_kwd=="raptor"` 的 chunk。

> **下一章预告**：摘要树和知识图谱都是"离线后处理"产物，而真正把检索、生成、工具调用串成一条可编排流水线的，是 RAGFlow 的 Agent 层。第 9 章《Agent 与编排画布》将解构 `agent/canvas.py` 的画布运行时与 `agent/component` 下各类组件，看 RAGFlow 如何用 DAG/状态机把一个个算子拼成可视化、可回放、可分支的智能工作流。
