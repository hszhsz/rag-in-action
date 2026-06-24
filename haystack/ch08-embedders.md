# 第 8 章 嵌入器 Embedders

上一章我们解构了 `DocumentStore`：它是向量的归宿，`InMemoryDocumentStore` 把 `Document.embedding` 收进内存、`write_documents` 落库、检索时再算相似度。但有个前提一直被我们跳过——文档进库之前，那个 `embedding` 向量是谁算出来的？答案就是本章的主角：嵌入器（Embedder）。嵌入器把文本（或图像）转换成稠密/稀疏向量，是检索召回质量的源头。没有它，`DocumentStore` 里存的只是一堆没有几何关系的字符串。

Haystack 把嵌入能力拆成两类组件，这是本章最重要的设计观察：`TextEmbedder` 负责把**单个查询字符串**变成向量，输出 `embedding`；`DocumentEmbedder` 负责把**一批 `Document`** 变成向量，并把向量写回每个 `Document.embedding`，输出 `documents`。这种"双形态"贯穿 OpenAI、Sentence Transformers、Azure、Hugging Face 等所有 provider。本章我们就从这对孪生组件入手，一路读到模型缓存后端、批处理、元数据拼接与稀疏/多模态嵌入。

## 为什么要分 TextEmbedder 与 DocumentEmbedder

最直接的理由是**输入/输出契约不同**。在 `openai_text_embedder.py` 里，`OpenAITextEmbedder.run` 的签名是 `run(self, text: str)`，输出类型声明为 `@component.output_types(embedding=list[float], meta=dict[str, Any])`；而 `openai_document_embedder.py` 里 `OpenAIDocumentEmbedder.run` 的签名是 `run(self, documents: list[Document])`，输出为 `@component.output_types(documents=list[Document], meta=dict[str, Any])`。回顾第 3 章，Pipeline 的连边是按 socket 的名字和类型对齐的：查询路径上 `TextEmbedder.embedding` 接到检索器的 `query_embedding`，而索引路径上 `DocumentEmbedder.documents` 接到 `DocumentWriter.documents`。两条路径的数据形态天然不同，强行用一个组件反而会让 Pipeline 连线含糊。

第二个理由是**语义路径的差异**。查询和文档在很多嵌入模型里需要不同的处理：E5、bge 这类模型要求 query 前面加 `query:` 这样的指令前缀，而文档加 `passage:`。Haystack 通过 `prefix`/`suffix` 参数把这件事交给使用者，两类组件各自独立配置，互不干扰。

第三个理由很务实：两类组件还附带类型守卫。`OpenAITextEmbedder._prepare_input` 一上来就检查 `if not isinstance(text, str)`，并抛出带有引导语的 `TypeError`——"In case you want to embed a list of Documents, please use the OpenAIDocumentEmbedder."。反过来 `OpenAIDocumentEmbedder.run` 检查 `isinstance(documents, list)` 且 `documents[0]` 是 `Document`，否则提示去用 `OpenAITextEmbedder`。这种"用错了直接告诉你该用哪个"的设计，把双形态的边界做成了可读的报错。

## DocumentEmbedder 如何把向量写回 Document

`DocumentEmbedder` 最关键的动作是"算完向量塞回文档"。在 `OpenAIDocumentEmbedder.run` 里，算完 `doc_ids_to_embeddings` 字典后，它遍历原始文档：

```python
new_documents = []
for doc in documents:
    if doc.id in doc_ids_to_embeddings:
        new_documents.append(replace(doc, embedding=doc_ids_to_embeddings[doc.id]))
    else:
        new_documents.append(replace(doc))
```

注意它用的是 `dataclasses.replace`，而不是原地修改 `doc.embedding = ...`。这呼应了第 2 章 `Document` 作为不可变 dataclass 的设计——组件不应该偷偷篡改上游传下来的对象，而是产出新对象。`SentenceTransformersDocumentEmbedder.run` 也是同样的 `replace(doc, embedding=emb)` 模式，并用 `zip(documents, embeddings, strict=True)` 保证文档数和向量数严格对齐（`strict=True` 在数量不匹配时会直接报错，避免静默错位）。

另一个细节是按 `doc.id` 而非按顺序回填。OpenAI 版用 `texts_to_embed` 这个 `dict[str, str]`（键是 `doc.id`），即使某个 batch 因为 `APIError` 被跳过，没拿到向量的文档也只是 `replace(doc)`（保持 `embedding=None`），不会把别人的向量错配上去。

## warm_up：轻 init 与重加载的生命周期分离

OpenAI 这类 API embedder 的 `__init__` 里就把 `OpenAI`/`AsyncOpenAI` 客户端建好了（创建 HTTP 客户端成本低）。但本地模型完全不同——加载一个 Sentence Transformers 模型可能要几百 MB、几秒到几十秒。如果放在 `__init__` 里，那么"构建一个 Pipeline"就会变成"下载并加载所有模型"，开发体验极差。

Haystack 的解法是 `warm_up` 生命周期钩子。`SentenceTransformersTextEmbedder.__init__` 只是把一堆配置存成属性，并把 `self.embedding_backend` 初始化为 `None`：

```python
self.embedding_backend: _SentenceTransformersEmbeddingBackend | None = None
```

真正的模型加载推迟到 `warm_up`：

```python
def warm_up(self) -> None:
    if self.embedding_backend is None:
        self.embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(...)
```

回顾第 4 章，Pipeline 在 `run` 之前会统一调用所有组件的 `warm_up`。这样设计的好处是：构建 Pipeline、序列化、连线校验全都廉价（无需模型在场），只有真正要跑推理时才付出加载代价。`run` 里还有一道兜底——`if self.embedding_backend is None: self.warm_up()`，即便你单独调用组件、忘了 warm_up，也能正常工作。

`warm_up` 末尾还有个小补丁：`if self.tokenizer_kwargs and self.tokenizer_kwargs.get("model_max_length")` 时会把 `model.max_seq_length` 覆盖成指定值——因为 Sentence Transformers 的 `max_seq_length` 不在构造函数参数里，必须模型加载后再设。

## backend 工厂：模型实例的进程内缓存

如果一个 Pipeline 里既有 `SentenceTransformersTextEmbedder` 又有 `SentenceTransformersDocumentEmbedder`，而且用同一个模型，应该加载几次？答案是**一次**。秘密在 `backends/sentence_transformers_backend.py` 的 `_SentenceTransformersEmbeddingBackendFactory`。

工厂用一个类级字典 `_instances: dict[str, "_SentenceTransformersEmbeddingBackend"] = {}` 做缓存。`get_embedding_backend` 把所有影响模型身份的参数（`model`、`device`、`auth_token`、`trust_remote_code`、`revision`、`truncate_dim`、`backend` 等）打包成 `cache_params`，再用 `json.dumps(cache_params, sort_keys=True, default=str)` 算出一个稳定的 `embedding_backend_id` 当缓存键：

```python
if embedding_backend_id in _SentenceTransformersEmbeddingBackendFactory._instances:
    return _SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id]
```

命中就复用，未命中才 `new` 一个 `_SentenceTransformersEmbeddingBackend` 并存入。`sort_keys=True` 保证字典顺序不影响键，`default=str` 让 `Secret` 这类非 JSON 原生对象也能序列化进键里。这就是为什么参数完全相同的两个组件会共享同一份模型权重，省下重复加载的内存与时间。

`_SentenceTransformersEmbeddingBackend` 本身很薄：构造时实例化一个 `SentenceTransformer`，`embed` 方法就是 `return self.model.encode(data, **kwargs).tolist()`——把 numpy/tensor 转成纯 Python list，方便下游 JSON 化。所有 provider 无关的细节（`batch_size`、`show_progress_bar`、`normalize_embeddings`、`precision`）都通过 `**kwargs` 透传给 `encode`。Sentence-Transformers 的 `encode` 接口正是这套语义 [[Sentence-Transformers]](https://www.sbert.net/)。

## 元数据拼接：meta_fields_to_embed 与 embedding_separator

只嵌正文往往不够——标题、作者、章节这些元数据也携带语义。`DocumentEmbedder` 用 `meta_fields_to_embed` 和 `embedding_separator` 把元数据拼进待嵌文本。看 `OpenAIDocumentEmbedder._prepare_texts_to_embed`：

```python
meta_values_to_embed = [
    str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
]
texts_to_embed[doc.id] = (
    self.prefix + self.embedding_separator.join(meta_values_to_embed + [doc.content or ""]) + self.suffix
)
```

逻辑是：从 `doc.meta` 里挑出 `meta_fields_to_embed` 列出的字段（跳过缺失或 `None` 的），`str()` 化后用 `embedding_separator`（默认 `"\n"`）拼接，正文 `doc.content` 接在最后，最后整体套上 `prefix`/`suffix`。`SentenceTransformersDocumentEmbedder.run` 里是同一段逻辑的内联版本。注意 `doc.content or ""`——空内容的文档也不会崩，只嵌元数据。

`TextEmbedder` 没有这套机制，因为查询通常没有元数据。它的拼接极简，比如 `OpenAITextEmbedder._prepare_input` 只有 `text_to_embed = self.prefix + text + self.suffix`。这又是双形态差异的一个具体体现。

## 批处理、进度条与失败容忍

文档动辄成千上万，必须分批送进模型/API。`OpenAIDocumentEmbedder` 默认 `batch_size: int = 32`，`_embed_batch` 用 `more_itertools.batched` 切片，并套上 `tqdm(..., disable=not self.progress_bar, desc="Calculating embeddings")`——`progress_bar` 默认 `True`，想静默就传 `False`。

OpenAI 版的批处理还有一层失败容忍。每个 batch 的 `client.embeddings.create` 包在 `try/except APIError` 里，出错时记日志并根据 `raise_on_failure`（默认 `False`）决定是抛出还是 `continue` 跳过这一批。这是"宁可少几篇也别让整个索引任务挂掉"的取舍，特别适合大规模离线索引。同时它会把每批的 `usage.prompt_tokens` 和 `total_tokens` 累加进 `meta["usage"]`，方便你统计花了多少 token。

Sentence Transformers 版没有逐批 try/except——因为本地推理不像网络 API 那样会零星超时，它把整个 `texts_to_embed` 列表一次性交给 `embedding_backend.embed`，由 Sentence Transformers 内部按 `batch_size` 分批。两种 provider 因为故障模型不同，批处理策略也不同。

## Secret：API key 的脱敏与可序列化

API key 绝不能明文写进序列化结果，否则把 Pipeline 存成 YAML 就泄密了。Haystack 用 `Secret` 抽象解决这点。`OpenAITextEmbedder` 的默认值是 `api_key: Secret = Secret.from_env_var("OPENAI_API_KEY")`，用时通过 `api_key.resolve_value()` 取出真实值喂给客户端，而 `to_dict` 里序列化的是 `self.api_key`（`Secret` 对象本身，它只记录"从哪个环境变量取"而非明文）。

Sentence Transformers 版的 token 默认是 `Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)`——支持多个候选环境变量，`strict=False` 表示拿不到也不报错（公开模型不需要 token）。Azure 版更进一步，在 `azure_text_embedder.py` 里 `api_key` 和 `azure_ad_token` 都可为 `None`，但 `__init__` 会校验 `if api_key is None and azure_ad_token is None: raise ValueError`——两种鉴权方式至少得有一种。这种把"配置约束"前移到构造期的写法，让错误尽早暴露。

## normalize_embeddings、truncate_dim 与 precision

几个影响向量本身的参数值得单独说。`normalize_embeddings`（Sentence Transformers 版默认 `False`）开启后对向量做 L2 归一化，使范数为 1——这样点积就等价于余弦相似度，配合内积检索很方便。OpenAI 版没有这个开关，因为 OpenAI 的 embedding 已经是归一化的。

`truncate_dim`（默认 `None`）是给 Matryoshka 表示学习模型用的：把向量截断到更低维度以省存储/算力。源码注释明确警告，如果模型没用 Matryoshka 训练过，截断会显著掉点。注意它被打进了 backend 工厂的 `cache_params`，所以不同 `truncate_dim` 会得到不同的缓存模型实例。

`precision`（默认 `"float32"`，可选 `"int8"`/`"uint8"`/`"binary"`/`"ubinary"`）控制量化精度：非 float32 都是量化嵌入，体积更小、计算更快，但精度下降，适合大语料语义搜索。而 OpenAI/Azure 走 API 的 `dimensions` 参数则是另一个维度控制——只对 `text-embedding-3` 及更新模型有效，`_prepare_input` 里仅当 `self.dimensions is not None` 才加进请求 `kwargs`。

## 稀疏嵌入与多模态：双形态范式的延伸

Haystack 2.31 已经把同一套双形态范式推广到稀疏向量和图像。`sentence_transformers_sparse_text_embedder.py` 里的 `SentenceTransformersSparseTextEmbedder` 默认模型是 `prithivida/Splade_PP_en_v2`，输出类型是 `@component.output_types(sparse_embedding=SparseEmbedding)`——不再是 `list[float]`，而是第 2 章见过的 `SparseEmbedding`（`indices` + `values`）。它的 backend 在 `sentence_transformers_sparse_backend.py`，用的是 `SparseEncoder` 而非 `SentenceTransformer`，`embed` 里以 `convert_to_sparse_tensor=True` 拿到稀疏张量，再 `coalesce()` 后抽出列索引和值构造 `SparseEmbedding`。它同样有独立的 `_SentenceTransformersSparseEmbeddingBackendFactory` 做缓存，结构与稠密版如出一辙。

多模态方面，`image/sentence_transformers_doc_image_embedder.py` 的 `SentenceTransformersDocumentImageEmbedder` 默认 `model="sentence-transformers/clip-ViT-B-32"`，从 `doc.meta` 的 `file_path_meta_field`（默认 `"file_path"`）读取图片路径，PDF 会先用 `_batch_convert_pdf_pages_to_images` 转成图，再喂给同一个 `_SentenceTransformersEmbeddingBackend`——注意它复用了稠密文本版的 backend，因为 CLIP 把图文嵌入到了同一向量空间。它在回填时还往 meta 里写了 `"embedding_source": {"type": "image", ...}` 以便事后审计向量来自图还是文。它只有 `DocumentEmbedder` 形态，没有对应的 image TextEmbedder。

## 远程 API provider：HF 与 Azure 的差异

`hugging_face_api_text_embedder.py` 展示了另一种 API 抽象。`HuggingFaceAPITextEmbedder` 用 `api_type`（`SERVERLESS_INFERENCE_API`/`INFERENCE_ENDPOINTS`/`TEXT_EMBEDDINGS_INFERENCE`）区分三种后端，`__init__` 里据此校验 `api_params` 里该有 `model` 还是 `url`。它的 `truncate`/`normalize` 参数仅对 TEI/Inference Endpoints 生效——`_prepare_input` 里若是 Serverless 就把它们置 `None` 并打 warning。`run` 里调 `self._client.feature_extraction`，还对返回的 numpy 形状做了校验（`ndim > 2` 或 `(2, ...)` 但首维不为 1 都报错），最后 `flatten().tolist()`。

值得一提：这些 Sentence Transformers / HF API 组件的 `__init__` 都带 `warnings.warn(..., FutureWarning)`，提示它们将在 Haystack 3.0 迁出核心包到独立 integration 包。这反映了 Haystack 的演进策略——核心只保留契约和少数 provider，其余下沉到 `haystack_integrations`。

## 本章小结

- Haystack 把嵌入拆成 `TextEmbedder`（嵌 query 字符串，输出 `embedding`）与 `DocumentEmbedder`（嵌 `List[Document]`，回填 `Document.embedding`，输出 `documents`）两类，因为查询路径与索引路径在 Pipeline 里的数据形态和语义处理本就不同。
- `DocumentEmbedder` 用 `dataclasses.replace(doc, embedding=...)` 产出新文档而非原地改，并按 `doc.id` 回填、用 `zip(..., strict=True)` 防错位，呼应 `Document` 的不可变设计。
- API embedder（OpenAI/Azure）在 `__init__` 就建好客户端；本地模型 embedder 把 `__init__` 做轻，模型加载推迟到 `warm_up`，让 Pipeline 构建与序列化保持廉价。
- `_SentenceTransformersEmbeddingBackendFactory` 用参数 JSON 串作缓存键，让相同配置的多个组件共享同一份模型权重，避免重复加载。
- `meta_fields_to_embed` + `embedding_separator` 把选定元数据拼到正文前一起嵌入；`prefix`/`suffix` 给 E5/bge 这类需要指令前缀的模型留了口子。
- `batch_size`/`progress_bar` 管批处理与可视化；OpenAI 版还用 `raise_on_failure`（默认 `False`）做逐批失败容忍并累加 token `usage`。
- `Secret` 让 API key 以"环境变量引用"形式序列化而非明文，`resolve_value()` 仅在运行期取真值；Azure 版强制 api_key 与 ad_token 至少其一。
- `normalize_embeddings`、`truncate_dim`、`precision`（量化）控制向量本身，其中 `truncate_dim` 进了 backend 缓存键；API 侧用 `dimensions` 做维度裁剪。
- 双形态范式已延伸到稀疏（`SparseEncoder` 输出 `SparseEmbedding`）与多模态（CLIP 图文同空间，复用稠密 backend），并通过 `FutureWarning` 预告 3.0 把多数 provider 迁出核心包。

## 动手实验

1. **实验一：观察双形态的输入/输出契约** — 在源码中分别打开 `openai_text_embedder.py` 与 `openai_document_embedder.py`，对照 `run` 的签名与 `@component.output_types` 装饰器，记录 `text:str→embedding` 与 `documents:list[Document]→documents` 的差异；再把 `text` 传给 `OpenAIDocumentEmbedder` 触发 `TypeError`，阅读它给出的"请改用 OpenAITextEmbedder"引导语。

2. **实验二：验证 backend 缓存的复用** — 用相同 `model` 各构造一个 `SentenceTransformersTextEmbedder` 和 `SentenceTransformersDocumentEmbedder`，都调 `warm_up()` 后比较两者的 `embedding_backend` 是否为同一对象（`is`）；再改其中一个的 `truncate_dim` 后重试，观察缓存键变化导致加载出第二个实例。

3. **实验三：元数据拼接的效果** — 构造一个带 `meta={"title": "...", "author": "..."}` 的 `Document`，分别在 `SentenceTransformersDocumentEmbedder` 中设置 `meta_fields_to_embed=["title"]` 和不设置，打印 `_prepare_texts_to_embed`（或内联逻辑）拼出的待嵌文本，确认 `embedding_separator` 与 `prefix`/`suffix` 的位置。

4. **实验四：稀疏与稠密的输出对照** — 同一句话分别用 `SentenceTransformersTextEmbedder` 和 `SentenceTransformersSparseTextEmbedder` 嵌入，打印前者的 `embedding`（稠密 `list[float]`）与后者的 `sparse_embedding`（`SparseEmbedding` 的 `indices`/`values`），结合 `sentence_transformers_sparse_backend.py` 中 `coalesce()` 抽索引的逻辑理解稀疏向量的结构。

> **下一章预告**：向量算好、文档进库之后，真正的检索时刻到了。第 9 章 检索器与重排器将解构 Haystack 如何用 `query_embedding` 从 `DocumentStore` 召回候选、稠密/稀疏/混合检索器的分工，以及重排器（Ranker）如何对召回结果二次打分以提升 Top-K 质量。
