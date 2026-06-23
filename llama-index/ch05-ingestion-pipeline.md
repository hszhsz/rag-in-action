# 第 5 章:摄入管线 IngestionPipeline

第 4 章我们看到,`NodeParser` 本质上是一个 `TransformComponent`——它接收一批 node,吐出另一批 node(把 `Document` 切成更小的 `TextNode`)。但现实世界的摄入流程从来不止"切分"一步:真实的 pipeline 通常要把**解析切分 → 元数据抽取 → 向量嵌入**这几个阶段串起来,每个阶段都是一个 node 进、node 出的 `TransformComponent`。如果手写脚本依次调用它们,你很快会遇到三个工程问题:同一篇文档反复跑一遍就要重新算一遍 embedding,改了一句话却要把整库重嵌入,以及无法把这套配置序列化下来跨进程/远程执行。

LlamaIndex 用 `IngestionPipeline`(`llama_index/core/ingestion/pipeline.py`)统一回答这三个问题。它把一串 `transformations` 按顺序施加到 node 上,用节点的 `hash` 配合 docstore 做去重与增量更新,用 `IngestionCache` 跳过已算过的变换,并通过 `ConfigurableTransformations` 把每种变换枚举化以便序列化与远程化。本章逐一拆解这些机制。

## 为何需要管线:统一的 node 进、node 出接口

`IngestionPipeline` 是一个 `BaseModel`,核心字段就是 `transformations: List[TransformComponent]`。它的设计前提呼应了第 1 章对 `TransformComponent` 的约定:每个变换都是可调用对象,签名上"接收 `Sequence[BaseNode]`、返回 `Sequence[BaseNode]`"。正因为接口统一,pipeline 才能把任意多个阶段无脑串成一条链。

施加变换的核心逻辑落在模块级函数 `run_transformations` 上:

```python
for transform in transformations_with_progress:
    if cache is not None:
        hash = get_transformation_hash(nodes, transform)
        cached_nodes = cache.get(hash, collection=cache_collection)
        if cached_nodes is not None:
            nodes = cached_nodes
        else:
            nodes = transform(nodes, **kwargs)
            cache.put(hash, nodes, collection=cache_collection)
    else:
        nodes = transform(nodes, **kwargs)
```

注意 `nodes = transform(nodes)` 这种"上一步输出即下一步输入"的串联写法——这正是统一接口带来的红利。异步版本 `arun_transformations` 结构完全对称,只是把 `transform(nodes)` 换成 `await transform.acall(nodes)`。

若用户不传 `transformations`,`__init__` 会调用 `_get_default_transformations()`,默认值是 `[SentenceSplitter(), Settings.embed_model]`——即"句子切分 + 全局嵌入模型",直接呼应第 2 章的 `Settings` 全局配置。pipeline 的 `name`/`project_name` 默认取 `DEFAULT_PIPELINE_NAME`(`"default"`)与 `DEFAULT_PROJECT_NAME`(`"Default"`),来自 `llama_index/core/constants.py`。

## `run` 的执行骨架

`run`(及其异步孪生 `arun`)是整章的主干。把 `pipeline.py` 的 `run` 方法拆开看,它依次做了五件事:

1. **准备输入**:`_prepare_inputs(documents, nodes)` 把方法参数里的 `documents`/`nodes`、构造时传入的 `self.documents`,以及 `self.readers` 读出来的内容统统拼成 `input_nodes`。这里 `Document` 被当作 node 直接 `+=` 进列表——因为 `Document` 也是 `BaseNode` 的子类。
2. **决定去重策略**:读取 `self.docstore_strategy`,并在缺少 `vector_store` 时做降级(见下一节)。
3. **去重 / 增量过滤**:根据 docstore 与 vector_store 是否存在,调用 `_handle_upserts` 或 `_handle_duplicates`,得到真正需要计算的 `nodes_to_run`;若没有 docstore,则 `nodes_to_run = input_nodes`(全部都跑)。
4. **施加变换**:串行走 `run_transformations`,或在 `num_workers > 1` 时走多进程(见并行小节)。
5. **落库**:若有 `vector_store`,把 `embedding is not None` 的 node 批量 `add` 进去;若有 `docstore`,调用 `_update_docstore` 写回 node 与 hash。

一个关键设计细节:写入 vector store 时会先过滤 `nodes_with_embeddings = [n for n in nodes if n.embedding is not None]`。也就是说,只有 transformations 链里确实包含了嵌入模型、给 node 填上了 `embedding`,这些 node 才会进向量库——pipeline 不强制要求嵌入,但向量入库以 embedding 存在为前提。

## hash 去重的基石:`TextNode.hash`

整套去重机制建立在节点 `hash` 之上。回看 `schema.py`,`TextNode.hash` 的定义极其朴素:

```python
@property
def hash(self) -> str:
    doc_identity = str(self.text) + str(self.metadata)
    return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())
```

即 **`sha256(text + metadata)`**。这意味着内容或元数据任意一处改变,hash 就变;两个内容与元数据完全相同的 node,hash 必然相同。`Document` 节点的 `hash` 还会进一步把 `data`/`path`/`url` 等来源信息纳入摘要(`schema.py` 中 `sha256(self.data)`、`sha256(str(self.path)...)` 等),但思路一致:用一个稳定的内容指纹代表"这个 node 是否变过"。

正因为 hash 完全由内容决定,pipeline 才能用它当"这条记录要不要重新计算"的判据——这是后面去重、增量、缓存三套机制共同的地基。

## DocstoreStrategy:去重与增量更新的三种语义

`DocstoreStrategy` 是一个字符串枚举,定义了在 docstore 持久化场景下如何对待"已经见过的文档":

```python
class DocstoreStrategy(str, Enum):
    UPSERTS = "upserts"
    DUPLICATES_ONLY = "duplicates_only"
    UPSERTS_AND_DELETE = "upserts_and_delete"
```

默认值是 `DocstoreStrategy.UPSERTS`。三者的差异全部体现在 `run` 的分支与 `_handle_duplicates`/`_handle_upserts` 两个方法里。

**`DUPLICATES_ONLY`——只去重,不更新。** 走 `_handle_duplicates`:取出 docstore 里所有 `existing_hashes`,逐个 node 判断 `node.hash` 是否已存在(同时用 `current_hashes` 防止同一批内部的重复)。只有 hash 从没见过的 node 才会被 `set_document_hash` 记下并加入 `nodes_to_run`。它**只认 hash**,不关心 `id`,因此一旦文档内容变了(hash 变了),旧版本会被当成全新文档一起留存——它无法表达"更新"。

**`UPSERTS`——按文档 id 做新增或替换。** 走 `_handle_upserts`,以 `ref_doc_id`(没有则退回 `node.id_`)为键:

```python
existing_hash = self.docstore.get_document_hash(ref_doc_id)
if not existing_hash:
    deduped_nodes_to_run[ref_doc_id] = node          # 新文档,加入
elif existing_hash and existing_hash != node.hash:    # 同 id 但内容变了
    self.docstore.delete_ref_doc(ref_doc_id, raise_error=False)
    if self.vector_store is not None:
        self.vector_store.delete(ref_doc_id)
    deduped_nodes_to_run[ref_doc_id] = node          # 删旧 + 加新
else:
    continue                                          # 同 id 同 hash,跳过
```

这就是 upsert 的精髓:**同 id 不存在则插入;同 id 但 hash 变了则先从 docstore 与 vector_store 删除旧版本再插入新版本;同 id 同 hash 则原样跳过**。它能正确处理"文档被编辑后重新摄入"的场景,而 `DUPLICATES_ONLY` 做不到。

**`UPSERTS_AND_DELETE`——upsert 之上再做"清理已删除文档"。** 它复用 `_handle_upserts` 的全部逻辑,额外在末尾计算 `doc_ids_to_delete = existing_doc_ids_before - doc_ids_from_nodes`:凡是 docstore 里有、但本次输入里**没出现**的文档 id,统统从 docstore 与 vector_store 删除。这让 docstore 与"本次输入的全集"严格对齐,适合"每次都把数据源全量喂进来、要求库内容跟数据源一一对应"的同步任务。

写回阶段由 `_update_docstore` 收尾:`UPSERTS` 与 `UPSERTS_AND_DELETE` 会 `set_document_hashes` 再 `add_documents`(既存内容又存 hash 指纹);`DUPLICATES_ONLY` 只 `add_documents`。

## 缺少 vector_store 时的策略降级

`run` 里有一段值得注意的防御逻辑:如果设置了 `docstore` 但没有 `vector_store`,而策略又是 `UPSERTS` 或 `UPSERTS_AND_DELETE`,pipeline 会发 `UserWarning` 并把**本次运行**的 `effective_strategy` 降级为 `DUPLICATES_ONLY`:

```python
warnings.warn(
    f"docstore_strategy='{self.docstore_strategy.value}' requires a vector store "
    "to apply upsert/delete semantics; falling back to 'duplicates_only' for this run. "
    "pipeline.docstore_strategy is unchanged.",
    UserWarning, stacklevel=3,
)
effective_strategy = DocstoreStrategy.DUPLICATES_ONLY
```

原因很直接:upsert/delete 的语义包含"删除向量库里的旧向量",没有 vector_store 这个动作无从落地,继续按 upsert 处理只会给出不一致的假象。注释也强调 `pipeline.docstore_strategy is unchanged`——只降级当前这一跑,不改对象状态。后续真正的去重分支里用的就是 `effective_strategy` 而非原始字段。

## IngestionCache:用 hash 跳过重复计算

去重解决的是"哪些 node 要进入变换链",缓存解决的是"进入变换链的 node,某一步变换能不能不重算"。两者正交。`IngestionCache`(`cache.py`)是一个薄封装,默认 `collection` 为 `DEFAULT_CACHE_NAME`(`"llama_cache"`),底层默认是 `SimpleCache`(即 `SimpleKVStore`)。

缓存键由 `get_transformation_hash(nodes, transformation)` 计算,它把**当前这批 node 的全文内容**(`get_content(metadata_mode=MetadataMode.ALL)` 拼起来)与**该变换序列化后的字符串**(`transformation.to_dict()`)一起做 sha256:

```python
nodes_str = "".join([str(node.get_content(metadata_mode=MetadataMode.ALL)) for node in nodes])
transform_string = remove_unstable_values(str(transformation.to_dict()))
return sha256((nodes_str + transform_string).encode("utf-8")).hexdigest()
```

这里 `remove_unstable_values` 用正则 `r"<[\w\s_\. ]+ at 0x[a-z0-9]+>"` 把诸如 `<function test_fn at 0x...>` 这类**带内存地址的不稳定表示**抹掉,否则同一份配置每次进程重启都会算出不同的 hash,缓存永远命中不了。

命中逻辑就在 `run_transformations` 里:对每一步变换先算 `hash`,`cache.get(hash)` 命中则直接用缓存结果替换 `nodes`,跳过这一步计算;未命中才真正执行 `transform(nodes)` 并 `cache.put`。因此缓存键同时绑定"输入内容"与"变换配置"——只要这两者中任意一个变了,缓存自然失效、重新计算,语义上始终正确。`cache.put` 存的是 `doc_to_json(node)` 的序列化结果,`get` 时用 `json_to_doc` 还原。

需要时可用 `disable_cache=True` 整体关闭缓存;`run` 中传给 `run_transformations` 的就是 `self.cache if not self.disable_cache else None`。

## 并行执行:多进程切批

当数据量大、变换(尤其 embedding)CPU 密集时,`run` 支持 `num_workers > 1` 走多进程。它先把 `num_workers` 收敛到不超过 `multiprocessing.cpu_count()`(超了发 warning 并下调),再用 `_node_batcher` 把 `nodes_to_run` 切成约 `num_workers` 个批次:

```python
batch_size = max(1, int(len(nodes) / num_batches))
for i in range(0, len(nodes), batch_size):
    yield nodes[i : i + batch_size]
```

同步路径用 `multiprocessing.get_context("spawn").Pool(num_workers)` 配合 `p.starmap(_run_transformations_worker, ...)`;异步路径(`arun`)则用 `ProcessPoolExecutor` + `loop.run_in_executor` + `asyncio.gather`。两个 worker 都返回 `(nodes, cache_entries)` 二元组。

这里有个不得不处理的子进程缓存合并问题:子进程在自己的内存里 `cache.put` 的条目,主进程看不到。所以当缓存后端是内存型(`isinstance(cache.cache, MutableMappingKVStore)`)时,worker 会把自己这批的 `cache.cache.get_all(...)` 一并带回,主进程在收集结果时逐条 `self.cache.cache.put(...)` 合并回主缓存。注释明确指出:外部后端(写穿到共享存储)无需 merge,只有内存后端才需要这一步回填。

## ConfigurableTransformations:可序列化与远程化

最后一块拼图是 `transformations.py` 里的 `ConfigurableTransformations`。`IngestionPipeline` 之所以能被持久化、上传到 LlamaCloud 远程执行,前提是它的每一个变换都能被**识别、序列化、再重建**。但 `transformations` 字段里装的是任意 `TransformComponent` 实例,光有实例无法跨进程描述"这是哪类变换"。

`ConfigurableTransformations` 是一个动态构建的枚举(由 `build_configurable_transformation_enum()` 生成,赋值在模块级)。它把系统支持的每一类变换登记为枚举成员:节点解析器一侧固定登记了 `CodeSplitter`、`SentenceSplitter`、`TokenTextSplitter`、`HTMLNodeParser`、`MarkdownNodeParser`、`JSONNodeParser`、`SimpleFileNodeParser`、`MarkdownElementNodeParser`;嵌入一侧则用 `try/except (ImportError, ValidationError)` **按可用性条件登记**——只有 `llama-index-embeddings-openai` 等可选包确实装了,`OpenAIEmbedding`、`CohereEmbedding`、`GeminiEmbedding` 等成员才会被加进枚举。这避免了未安装的依赖把整个枚举构建拖垮。

每个成员的值是 `ConfigurableTransformation`,携带 `name`、`transformation_category`(`NODE_PARSER` 或 `EMBEDDING`,各自声明了输入/输出类型——节点解析是"Documents → Nodes",嵌入是"Nodes → Nodes")以及 `component_type`。`from_component(component)` 反查实例对应哪个枚举成员,`build_configured_transformation` 则把实例包成带泛型的 `ConfiguredTransformation[T]`(其 `component` 字段用 `SerializeAsAny[T]` 标注,确保子类字段被完整序列化)。这套"实例 ↔ 枚举类型 ↔ 可序列化包装"的三角,正是 pipeline 能离开本地进程、被配置化与远程化的工程动机。

至于数据的两端:`data_sources.py` 用 `DataSource` 描述一类数据来源(`component_type` 指向某个 `BaseComponent`/reader),`data_sinks.py` 用 `DataSink` 描述一类数据落点(`component_type` 指向某个 `BasePydanticVectorStore`),二者各自也有同构的 `ConfigurableComponent` 枚举,与 transformations 共用同一套"枚举化以便配置/远程化"的思路。

## 本章小结

- `IngestionPipeline`(`pipeline.py`)把一串 `TransformComponent` 按 node 进、node 出的统一接口串成可复用流水线,默认变换是 `[SentenceSplitter(), Settings.embed_model]`。
- 核心施加逻辑在模块级 `run_transformations`/`arun_transformations`:用 `nodes = transform(nodes)` 串联,异步版改用 `await transform.acall(nodes)`。
- 去重与增量的基石是 `TextNode.hash = sha256(text + metadata)`——内容指纹决定一个 node 是否"变过"。
- `DocstoreStrategy` 有三种语义:`DUPLICATES_ONLY` 只按 hash 去重不更新;`UPSERTS` 按 `ref_doc_id` 新增或"删旧加新"(默认值);`UPSERTS_AND_DELETE` 在 upsert 基础上删除输入中已不存在的文档,使库与输入全集对齐。
- 当有 docstore 但无 vector_store 时,`run` 会发 `UserWarning` 把本次 `effective_strategy` 降级为 `DUPLICATES_ONLY`,因为删除旧向量无处落地;对象字段本身不变。
- `IngestionCache`(`cache.py`,默认 collection `"llama_cache"`)用 `get_transformation_hash`(node 全文 + 变换 `to_dict()` 的 sha256,并经 `remove_unstable_values` 抹掉内存地址)作键跳过已算变换。
- 去重(哪些 node 入链)与缓存(入链 node 的某步能否不重算)是正交的两套机制。
- 并行通过 `num_workers` 切批多进程实现,内存型缓存后端需把子进程的 cache 条目回填主进程,外部后端写穿则免合并。
- `ConfigurableTransformations` 把每类变换枚举化(嵌入类按依赖可用性条件登记),配合 `ConfiguredTransformation[T]` 的 `SerializeAsAny` 实现序列化与远程化。

## 动手实验

1. **实验一:验证 hash 驱动去重** — 在 `/tmp/llama_explore` 打开 `pipeline.py` 的 `_handle_duplicates` 与 `schema.py` 的 `TextNode.hash`,构造两个 `text` 与 `metadata` 完全相同的 `TextNode`,打印 `.hash` 确认相等;再改其中一个的 `metadata`,确认 hash 变化,从而理解"内容指纹"如何决定去重判定。
2. **实验二:对比三种 DocstoreStrategy** — 阅读 `_handle_upserts` 与 `_handle_duplicates`,在纸上推演"先摄入文档 A,再把 A 的正文改一句重新摄入"在 `DUPLICATES_ONLY`、`UPSERTS`、`UPSERTS_AND_DELETE` 下分别会保留/删除哪些版本,并在源码里定位每一步对应的 `delete_ref_doc`/`delete_document` 调用。
3. **实验三:追踪缓存命中路径** — 在 `cache.py` 的 `get_transformation_hash`(实际位于 `pipeline.py`)与 `run_transformations` 的命中分支里加 print/断点,跑一个含两步变换的 pipeline 两次,观察第二次是否走 `cached_nodes is not None` 分支跳过计算;再把某步变换的参数改掉,确认缓存键变化导致重算。
4. **实验四:观察可用性条件登记** — 阅读 `transformations.py` 的 `build_configurable_transformation_enum`,在未安装 `llama-index-embeddings-openai` 的环境里打印 `ConfigurableTransformations` 的成员列表,确认 `OPENAI_EMBEDDING` 因 `except (ImportError, ValidationError)` 而缺席,理解条件登记如何避免可选依赖拖垮枚举构建。

> **下一章预告**:摄入管线把原始文档变成了一批带 embedding 的 node,但"一堆 node"还不是"可检索结构"。第 6 章进入索引体系 Indices,看 `VectorStoreIndex` 等如何把 node 组织成可被检索器高效查询的结构。
