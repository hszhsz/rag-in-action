# 第 6 章 检索与融合重排

前面几章我们看着一篇文档如何被解析、切块、加权、向量化，最后落进文档引擎（ES / Infinity / OceanBase）。索引建好之后，真正决定 RAG 体验的是另一半：用户敲下一个问题，系统怎么把它变成一组检索表达式，怎么同时跑向量召回和全文召回，又怎么把两路结果融合、重排，最终挑出几段最相关的原文喂给大模型。这一章我们就钻进 `rag/nlp/search.py` 与 `rag/nlp/query.py`，把这条「问题进、引用出」的链路逐段拆开。

RAGFlow 的检索不是「把问题丢给向量库取 top-k」这么简单。它在查询侧做了相当重的工程：分词、去停用词、term weighting、同义词扩展、短语加权，再加上多路融合与可选的 cross-encoder 精排，最后还有一个招牌特性——`insert_citations`，把生成的答案逐句对齐回原文片段，做出可溯源的引用。下面我们顺着代码的执行顺序往下走。

## 一、入口：retrieval 与 search 的分工

对外的检索入口是 `Dealer.retrieval`（`rag/nlp/search.py`）。它的签名就把这一章要讲的几个核心旋钮都摆了出来：

```python
async def retrieval(
        self,
        question,
        embd_mdl,
        tenant_ids,
        kb_ids,
        page,
        page_size,
        similarity_threshold=0.2,
        vector_similarity_weight=0.3,
        top=1024,
        doc_ids=None,
        aggs=True,
        rerank_mdl=None,
        highlight=False,
        rank_feature: dict | None = {PAGERANK_FLD: 10},
        trace_id=None,
):
```

这里有几个默认值值得记住：相似度阈值 `similarity_threshold=0.2`，向量相似度权重 `vector_similarity_weight=0.3`（也就是说默认全文权重占 0.7），候选池上限 `top=1024`，以及一个默认就开着的排序特征 `rank_feature={PAGERANK_FLD: 10}`。`rerank_mdl` 默认是 `None`，意味着 cross-encoder 精排是按需开启的可选项，而不是默认路径。

`retrieval` 自己不直接拼检索查询，它先算出一个「候选窗口」`RERANK_LIMIT`，再调 `Dealer.search` 把候选捞回来，然后才在应用侧做重排和分页。窗口大小由 `_rerank_window` 决定，这是一段很容易被忽略但设计得很讲究的逻辑：

```python
@staticmethod
def _rerank_window(page_size: int, top: int = 0) -> int:
    if page_size <= 1:
        return min(30, top) if top > 0 else 30
    window = math.ceil(64 / page_size) * page_size
    if top > 0:
        window = min(window, math.ceil(top / page_size) * page_size)
    return window
```

它的目标是凑出一个大约 64 条的「重排友好」候选池（`math.ceil(64 / page_size) * page_size`），但又必须是 `page_size` 的整数倍。为什么必须整数倍？因为 `retrieval` 用同一个值既当后端分块抓取的块大小，又当从重排后块里切单页的模数：

```python
req = {
    ...
    "page": global_offset // RERANK_LIMIT + 1,
    "size": RERANK_LIMIT,
    ...
}
...
begin = global_offset % RERANK_LIMIT
end = begin + page_size
page_idx = valid_idx[begin:end]
```

`global_offset // RERANK_LIMIT` 决定抓第几块，`global_offset % RERANK_LIMIT` 决定页在块内的起点。两者要对齐，窗口就必须是页大小的整数倍，否则深翻页会悄悄丢结果、返回短页。源码注释里把这个不变量写得很明白。这是典型的「拿候选池做重排、但又要支持分页」会遇到的工程坑，RAGFlow 用一个统一窗口把它兜住了。

## 二、从问题到检索表达式：query.py 的查询构造

`Dealer.search` 在有 `question` 时会调用 `self.qryr.question(qst, min_match=0.3)`，这里的 `qryr` 是 `query.FulltextQueryer`（`rag/nlp/query.py`）。`question` 方法是整个查询侧的核心，它把一句自然语言问题翻译成一个带权重、带同义词、带短语约束的全文检索表达式（`MatchTextExpr`）。

第一步是规范化。`add_space_between_eng_zh` 在中英文边界插空格，避免「GPT模型」这种粘连的 token 被错误切分；接着用一条很长的正则把各种标点和 Infinity 词法器的 ESCAPABLE 字符（`[\x20()^"'~*?:\\]`）替换成空格，并做繁简转换与全角转半角：

```python
txt = re.sub(
    r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>*~'\"\\]+",
    " ",
    rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(txt.lower())),
).strip()
```

这一步的意图是防御性的：用户的问题里随便一个引号、问号或括号，如果原样进了 Infinity 的查询语法，词法器就会把它当成特殊算子而报错。所以查询构造的第一件事就是把这些字符清干净。

第二步是 `rmWWW`——去掉疑问词和虚词。这个函数定义在 `common/query_base.py`，用三条正则分别处理中文疑问词（「怎么办」「什么样的」「哪家」「是不是」……）、英文 wh- 疑问词（`what/who/how/which/where/why`）和英文虚词冠词介词（`is/are/the/a/an/of/to`……）。它的设计很务实：如果剥完之后 `txt` 空了，就回退到原文 `otxt`，避免把一句纯疑问词的问题清成空串。去掉这些词的目的，是不让「如何」「什么」这种几乎在每篇文档里都出现的高频词稀释掉真正承载信息的关键词。

第三步才是分词与加权，而且 RAGFlow 在这里走了中英文两条不同的分支，判断依据是 `is_chinese`（`common/query_base.py`，按非纯英文 token 的比例是否 ≥ 0.7 来判定）。

英文分支里，先用 `rag_tokenizer.tokenize` 切词，再用 `self.tw.weights(tks, preprocess=False)` 给每个 token 算权重（term weighting 下一节讲），然后为每个词查同义词，把同义词以四分之一的权重加进去：

```python
syn = ["\"{}\"^{:.4f}".format(s, w / 4.) for s in syn if s.strip()]
```

最后还会把相邻两个词组成短语，以两倍权重加进查询：

```python
q.append(
    '"%s %s"^%.4f'
    % (tks_w[i - 1][0], tks_w[i][0], max(tks_w[i - 1][1], tks_w[i][1]) * 2)
)
```

短语加权的意图是奖励「词序也对上」的文档——同时命中两个相邻词且顺序一致，往往比分散命中两个词更相关，所以给它额外的 boost。

中文分支更复杂。它用 `self.tw.split(txt)` 先做粗粒度切分（最多取前 256 段），对每段再算权重并查同义词。对每个 token，如果满足 `need_fine_grained_tokenize`（长度 ≥ 3 且不是纯字母数字），还会做一次细粒度分词 `fine_grained_tokenize`，并把细粒度子词以宽松的近邻短语形式加进查询：

```python
if sm:
    tk = f'{tk} OR "%s" OR ("%s"~2)^0.5' % (" ".join(sm), " ".join(sm))
```

这里 `"..."~2` 是 slop=2 的近邻短语，`^0.5` 是降权——细粒度子词命中是个弱信号，给它半权。整段 token 拼好后还会追加一个整体的近邻短语 `("%s"~2)^1.5`，并把同义词以 `^0.7` 权重接在 `OR` 后面：

```python
if syns and tms:
    tms = f"({tms})^5 OR ({syns})^0.7"
```

主词组给 `^5`、同义词给 `^0.7`，这一组权重清楚地表达了设计意图：原词命中是强信号，同义词命中只是兜底。最终所有段以 `OR` 连成一个大查询，连同 `minimum_should_match=min_match` 一起包进 `MatchTextExpr`：

```python
return MatchTextExpr(
    self.query_fields, query, 100,
    {"minimum_should_match": min_match, "original_query": original_query}
), keywords
```

值得注意的是 `query_fields`——它不是在单一字段上检索，而是跨多个字段并带字段级 boost：

```python
self.query_fields = [
    "title_tks^10", "title_sm_tks^5", "important_kwd^30", "important_tks^20",
    "question_tks^20", "content_ltks^2", "content_sm_ltks",
]
```

命中标题（`^10`）、命中人工/抽取的关键词（`important_kwd^30`）、命中切块关联的问题（`question_tks^20`）都比命中正文（`content_ltks^2`）权重高得多。这套字段 boost 把「文档结构信息」编码进了打分里：出现在标题或关键词里的命中，比埋在正文里的命中更可能代表主题相关。

## 三、term weighting：为什么有的词更重要

上一节反复出现的「权重」来自 `rag/nlp/term_weight.py` 的 `Dealer.weights`。这是 RAGFlow 全文检索打分的灵魂——它决定了同样命中两个词，哪个词应该贡献更多分数。核心公式在 `weights` 里：

```python
idf1 = np.array([idf(freq(t), 10000000) for t in tks])
idf2 = np.array([idf(df(t), 1000000000) for t in tks])
wts = (0.3 * idf1 + 0.7 * idf2) * \
      np.array([ner(t) * postag(t) for t in tks])
```

权重是两路 IDF 的加权和（词频表 `freq` 给的 IDF 占 0.3、文档频率表 `df` 给的 IDF 占 0.7），再乘以两个语言学因子 `ner(t) * postag(t)`。`idf` 函数本身是经典 IDF 平滑形式 `math.log10(10 + (N - s + 0.5) / (s + 0.5))`，越罕见的词 IDF 越大——这正是 BM25 一族打分里 IDF 项的思想 [[Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)]，词越稀有越能区分文档。

两个语言学乘子才是 RAGFlow 的特色。`ner(t)` 根据命名实体类型给乘子，机构名/地名/学校/股票（`corp/loca/sch/stock`）给 3，毒性词给 2，而一两个字母的短串给 0.01（几乎抹掉，因为这种短串基本没有区分度）：

```python
m = {"toxic": 2, "func": 1, "corp": 3, "loca": 3, "sch": 3, "stock": 3, "firstnm": 1}
```

`postag(t)` 按词性给乘子：地名时间名词（`ns/nt`）给 3，普通名词给 2，而代词连词副词（`r/c/d`）只给 0.3：

```python
if t in set(["r", "c", "d"]):
    return 0.3
if t in set(["ns", "nt"]):
    return 3
if t in set(["n"]):
    return 2
```

把这两个乘子和 IDF 相乘，效果就是：一个稀有的机构名能拿到很高的权重，而一个常见的代词或连词即使没被停用词过滤掉，权重也被压得很低。最后 `weights` 会做归一化 `[(t, s / S) for t, s in tw]`，让一组词的权重和为 1，这样不同长度的查询之间权重才可比。

`term_weight.Dealer` 在初始化时从 `rag/res` 目录加载 `ner.json`（命名实体表）和 `term.freq`（词频表），这两份资源文件是这套 term weighting 的数据底座。还有一个停用词集合 `stop_words`（「请问」「您」「什么」「怎么」「相关」……）在 `pretoken` 阶段就把虚词滤掉。

term weighting 不止用在查询构造，它在重排阶段也是核心：`token_similarity` 用 `to_dict` 把一组 token 转成「带权重的词袋」，并且在相邻词之间额外造一个 bigram 项（`d[t+_t] += max(c, _c) * 0.6`），让词序也参与相似度计算。这就是为什么 RAGFlow 的文本相似度不是简单的 Jaccard，而是 term-weighted 的重叠度。

## 四、多路召回与融合：向量 + 全文怎么合到一起

回到 `Dealer.search`。当 `emb_mdl` 不为空时，它会同时构造文本匹配和向量匹配，并用一个 `FusionExpr` 把两路融合：

```python
matchDense = await self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))
...
fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
matchExprs = [matchText, matchDense, fusionExpr]
```

`get_vector` 把问题编码成查询向量，列名是 `f"q_{len(embedding_data)}_vec"`（按维度命名，和索引侧对齐），距离度量是 `cosine`，并带一个 `similarity` 阈值（默认 0.1）。注意融合权重这里写死成 `"0.05,0.95"`——这是引擎侧（ES / Infinity / OceanBase）融合时用的权重，文本 0.05、向量 0.95，向量在引擎侧融合中占绝对主导。

这套融合落到不同引擎有不同实现。在 Infinity 路径（`rag/utils/infinity_conn.py`），`weighted_sum` 会把归一化方式强制设成 `atan`，因为默认的 `minmax` 会让排最后的文档拿到 0 分：

```python
if matchExpr.method == "weighted_sum":
    # The default is "minmax" which gives a zero score for the last doc.
    matchExpr.fusion_params["normalize"] = "atan"
```

在 ES 路径（`rag/utils/es_conn.py`），融合是通过把全文 `bool_query` 的整体 boost 设成 `1 - vector_similarity_weight`、向量走 `knn` 子句来实现的，两路分数由 ES 自己叠加。这里要分清两个权重：`search` 里写死的 `"0.05,0.95"` 是引擎侧融合权重，而 `retrieval` 签名里的 `vector_similarity_weight=0.3` 是应用侧重排权重，二者作用在不同阶段，不要混淆。

还有一个鲁棒性细节：如果第一次检索 `total == 0`，`search` 会自动降低门槛重试——把 `min_match` 从 0.3 降到 0.1、把向量 `similarity` 放宽到 0.17 再搜一次：

```python
if total == 0:
    ...
    matchText, _ = self.qryr.question(qst, min_match=0.1)
    matchDense.extra_options["similarity"] = 0.17
    res = await thread_pool_exec(self.dataStore.search, ...)
```

这是为了避免一个稍微刁钻的问题直接召回空集——宁可放宽召回也不要返回「什么都没找到」。

值得一提的是，RAGFlow 在主检索路径用的是 `weighted_sum` 加权融合，而不是另一类常见的 RRF（Reciprocal Rank Fusion，按排名倒数相加 [[Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)]）。加权融合保留了分数的绝对量级，这一点对后面要用 `similarity_threshold` 做阈值截断很关键——如果用 RRF，分数变成纯排名信号，阈值就失去了语义。

## 五、应用侧重排：三条路径与权重混合

引擎把候选捞回来后，`retrieval` 会在应用侧再算一遍相似度并重排。它先把 `term_similarity_weight = 1 - vector_similarity_weight` 算出来，然后按是否有外部 reranker、用的是哪个引擎，走三条不同路径。

如果配置了 `rerank_mdl`，走 `rerank_by_model`，这是 cross-encoder 精排：

```python
sim, tsim, vsim = self.rerank_by_model(
    rerank_mdl, sres, question,
    term_similarity_weight, vector_similarity_weight,
    rank_feature=rank_feature,
)
```

`rerank_by_model`（`rag/nlp/search.py`）把每个候选块的正文、标题、关键词拼成文档串，交给 `rerank_mdl.similarity(query, docs)` 打分，再和本地的 `token_similarity` 按权重混合：

```python
vtsim, _ = rerank_mdl.similarity(query, docs)
...
return tkweight * np.array(tksim) + vtweight * vtsim + rank_fea, tksim, vtsim
```

关键在于 `rerank_mdl.similarity` 返回的分数被保证落在 `[0, 1]`。这个契约写在 `rag/llm/rerank_model.py` 的 `Base.similarity` 里：每个 provider 只实现 `_compute_rank`，统一的 `_normalize_rank` 负责归一化——已经在 `[0,1]` 的（Cohere/Jina/Voyage 这类校准过的相关性分）原样返回，超出范围的（比如 NVIDIA 那种无界 logits）才做 min-max 映射，单候选或无区分度的批次则直接 clip：

```python
if min_rank >= 0.0 and max_rank <= 1.0:
    return rank
span = max_rank - min_rank
if span < 1e-3:
    return np.clip(rank, 0.0, 1.0)
return (rank - min_rank) / span
```

为什么要这么较真？因为下游是把 reranker 分数和 token 相似度按固定权重线性混合的，如果某个 provider 吐出无界负数 logits，不归一化就会把一个真正相关的块拖到纯关键词命中之下，把最终排序搞坏。RAGFlow 用一个统一的归一化层把所有 reranker 拉到同一把尺子上。

如果没有外部 reranker，走哪条路取决于引擎。Infinity 因为在融合前已经把每一路分数归一化了，应用侧直接拿引擎返回的 `_score` 当最终分，不再重排：

```python
if settings.DOC_ENGINE_INFINITY:
    sim = [sres.field[id].get("_score", 0.0) for id in sres.ids]
```

ES 路径则更精细：主检索时为了不把 chunk 向量传回应用层，ES 路径不取向量；要拿干净的余弦分，`retrieval` 会发起第二次「只做 KNN」的调用 `_knn_scores`，按候选 id 过滤、让 ES 在引擎内算好余弦分返回，再和本地 term 相似度按用户权重混合：

```python
knn_scores = await self._knn_scores(sres, idx_names, kb_ids)
sim, tsim, vsim = self.rerank_with_knn(
    sres, question, knn_scores,
    term_similarity_weight, vector_similarity_weight,
    rank_feature=rank_feature,
)
```

`rerank_with_knn` 里的 term 相似度计算藏着一个细节——它把不同字段的 token 按重要性重复拼接：

```python
tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
```

标题重复 2 次、关键词重复 5 次、关联问题重复 6 次。重复就是变相加权：在词袋相似度里，一个词出现得越多对相似度的贡献越大，所以这等于在重排阶段再次强调结构化字段。最终分数是三项相加：`sim = tkweight * tksim + vtweight * vtsim + rank_fea`，其中 `rank_fea` 来自 `_rank_feature_scores`，它把 PageRank（`PAGERANK_FLD`，默认权重 10）和标签特征（`TAG_FLD`，乘以 10 后叠加）也算进最终分。这意味着 RAGFlow 的排序信号不止「文本 + 向量」，还有文档级的 PageRank 权威度和标签匹配度。

OceanBase 是第三条路，它仍然在结果里带回 chunk 向量，所以走历史的本地 `rerank`，用 `hybrid_similarity` 在应用侧算余弦：

```python
return np.array(sims[0]) * vtweight + np.array(tksim) * tkweight, tksim, sims[0]
```

三条路殊途同归，输出都是 `(sim, tsim, vsim)` 三元组——综合分、term 分、向量分。

## 六、阈值截断、分页与去重

拿到综合分 `sim` 后，`retrieval` 用稳定排序得到顺序，再按阈值过滤：

```python
sorted_idx = np.argsort(sim_np * -1, kind='stable')
post_threshold = 0.0 if vector_similarity_weight <= 0 else similarity_threshold
valid_idx = [int(i) for i in sorted_idx if sim_np[i] >= post_threshold]
```

两个细节。第一，用 `kind='stable'` 是为了在分数打平时保证确定性顺序，避免同一查询两次结果抖动。第二，当 `vector_similarity_weight <= 0`（纯关键词模式）时，`similarity_threshold` 会被强制设成 0——因为纯 term 分和向量余弦不在一个量纲上，对纯 term 分套向量阈值没有意义，所以这种情况下不截断，注释里也说明了这点。

过滤后再用第二节讲的 `begin/end` 从候选块里切出当前页。每个入选块被组装成一个字典，把综合分、向量分、term 分都暴露出来：

```python
"similarity": float(sim_np[i]),
"vector_similarity": float(vsim[i]),
"term_similarity": float(tsim[i]),
```

把三个分数都返回，方便上层做调试、展示或二次过滤——用户能看到一段被召回到底是因为语义近还是关键词命中。同时 `doc_aggs` 会统计每篇文档贡献了多少候选块，按数量降序排，给前端做「来源文档分布」用。

## 七、招牌特性：insert_citations 的逐句溯源

RAGFlow 最有辨识度的能力是 grounded citations——生成的答案里每句话后面挂上 `[ID:n]`，指向它依据的原文块。这个逻辑在 `insert_citations`（`rag/nlp/search.py`）。

它先把答案按句子边界切分。切分正则相当国际化，除了中文标点还覆盖了阿拉伯语标点（`، ؛ ؟ ۔`），并且对代码块（` ``` `）做了保护，不会把代码中间切断。切完后只保留长度 ≥ 5 的片段参与对齐（太短的句子没有足够语义）：

```python
for i, t in enumerate(pieces):
    if len(t) < 5:
        continue
    idx.append(i)
    pieces_.append(t)
```

然后对每个答案句子，调用 `hybrid_similarity` 同时算它和每个原文块的向量相似度与 term 相似度，混合权重是 `tkweight=0.1, vtweight=0.9`——溯源时向量语义占绝对主导，因为答案是模型改写过的，字面重合度往往不高，得靠语义匹配。匹配用了一个逐步放宽的阈值循环：

```python
thr = 0.63
while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
    for i, a in enumerate(pieces_):
        sim, tksim, vtsim = self.qryr.hybrid_similarity(...)
        mx = np.max(sim) * 0.99
        if mx < thr:
            continue
        cites[idx[i]] = list(set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
    thr *= 0.8
```

阈值从 0.63 起步，每轮乘 0.8 往下降，直到找到至少一条引用或降到 0.3 以下。这种「先严后宽」的策略保证：能高置信对齐就用高标准，实在对不上才放宽，避免硬塞低质量引用。每句最多挂 4 个引用（`[:4]`），并用 `mx = np.max(sim) * 0.99` 把每句的引用限定在「接近该句最佳匹配」的块上，不让弱相关块也被引上。最后把 `[ID:c]` 拼回答案对应位置，并用 `seted` 去重避免同一块被重复引用。

这里还有一个工程化的配套：主检索路径为了性能默认不把 chunk 向量传回应用层，但溯源需要向量。于是 `insert_citations` 的调用方会用 `fetch_chunk_vectors` 按需把指定 chunk 的向量单独拉回来，把「向量留在引擎」和「溯源要算相似度」这对矛盾解开。

## 本章小结

- 检索入口 `Dealer.retrieval` 的默认旋钮是 `similarity_threshold=0.2`、`vector_similarity_weight=0.3`（即全文权重 0.7）、候选池 `top=1024`，cross-encoder 精排 `rerank_mdl` 默认关闭，PageRank 排序特征默认开启（权重 10）。
- `_rerank_window` 把候选池凑到约 64 条且必须是 `page_size` 整数倍，用同一个值同时做块抓取和块内分页，保证深翻页不丢结果。
- `query.FulltextQueryer.question` 把问题翻译成多字段、带权重、带同义词、带短语约束的 `MatchTextExpr`，并对中英文走不同分支；`query_fields` 用字段级 boost（标题 `^10`、关键词 `^30`）把文档结构编码进打分。
- `rmWWW` 剥疑问词和虚词、`sub_special_char` 转义引擎词法器特殊字符，是查询构造里两道关键的防御与降噪步骤。
- term weighting（`term_weight.weights`）= 两路 IDF 加权和 × 命名实体乘子 × 词性乘子，再归一化；它让稀有实体词权重高、代词连词权重低，是全文打分的核心。
- 多路召回用 `FusionExpr("weighted_sum", ...)` 做加权融合而非 RRF，引擎侧融合权重写死为文本 0.05、向量 0.95；应用侧重排权重 `vector_similarity_weight` 是另一回事，作用在不同阶段。
- 重排分三条路径：有 reranker 走 `rerank_by_model`，ES 无 reranker 走第二次 KNN-only 调用的 `rerank_with_knn`，OceanBase 走带向量的本地 `rerank`，Infinity 直接用引擎归一化后的 `_score`。
- 所有 reranker 的输出由 `RerankModel.Base.similarity` 统一归一化到 `[0,1]`，防止无界 logits（如 NVIDIA）破坏线性混合后的排序。
- 重排时把标题/关键词/关联问题 token 重复拼接（`*2/*5/*6`）变相加权，并叠加 PageRank 和标签特征分。
- 阈值截断用稳定排序保证确定性；纯关键词模式（向量权重为 0）会自动放弃相似度阈值，因为量纲不可比。
- `insert_citations` 用 0.1/0.9 的 term/向量权重、从 0.63 起逐步放宽的阈值，把答案逐句对齐回原文块，每句最多 4 个引用，实现可溯源的 grounded citations。

## 动手实验

1. **实验一：观察查询构造的展开** — 在本地起一个 Python 交互环境，`from rag.nlp.query import FulltextQueryer`，构造实例后分别对一句中文问题（如「RAGFlow 怎么做混合检索」）和一句英文问题调用 `question(...)`，打印返回的 `MatchTextExpr.matching_text` 与 `keywords`，对照本章第二节，观察 `rmWWW` 去掉了哪些疑问词、同义词以什么权重（`/4` 或 `^0.7`）被加进来、相邻词的短语 boost 是否出现。

2. **实验二：拆解 term weighting** — 直接 `from rag.nlp.term_weight import Dealer`，对一组混合 token（如 `["北京", "如何", "transformer", "的"]`）调用 `weights(tks, preprocess=False)`，打印每个词的归一化权重；再去 `term_weight.py` 的 `ner`/`postag` 里改一个乘子（比如把名词从 2 改成 4），重新计算，验证语言学乘子如何放大或压低某类词的权重。

3. **实验三：扫描融合与重排权重** — 用 Grep 在 `rag/` 下搜 `weighted_sum`、`vector_similarity_weight`、`rerank_with_knn`、`_knn_scores`，画出从 `search` 的引擎侧融合权重 `"0.05,0.95"` 到 `retrieval` 的应用侧 `vector_similarity_weight=0.3` 的完整数据流；对照 `es_conn.py`/`infinity_conn.py` 确认两个引擎分别在哪一步、用什么机制把两路分数合到一起。

4. **实验四：复现引用对齐逻辑** — 阅读 `insert_citations`，准备一段假答案和两三个原文块（chunks 及对应向量），手动调用它，观察 `thr` 从 0.63 逐步乘 0.8 下降的过程中哪一轮产生了引用、每句被挂上了几个 `[ID:n]`；再把 `tkweight/vtweight` 从 0.1/0.9 改成 0.9/0.1，对比纯 term 主导时引用结果如何变化，体会为什么溯源默认让向量主导。

> **下一章预告**：多路召回 + 融合重排解决的是「在已有 chunk 集合里找最相关片段」，但当答案需要跨多个文档、多跳推理时，扁平的向量/全文检索就力不从心了。第 7 章《GraphRAG 知识图谱》将解构 `rag/graphrag` 目录，看 RAGFlow 如何从文档里抽取实体与关系、增量构建知识图谱、并通过社区检测与社区报告（community report）把图谱压缩成可检索的层次化摘要，为复杂问答提供结构化的全局视角。
