# 第 6 章 嵌入与向量库

前几章我们把原始文件切成了 chunk。但 chunk 本身只是文本片段，机器无法直接"理解"它们之间的相似关系。要让 Brain 能够"按语义检索"，需要两件事：一是**嵌入器（Embedder）**，把每个 chunk 编码成高维向量；二是**向量库（Vector Store）**，把这些向量建成可以快速近邻检索的索引。Quivr 把这两件事封装在极简的几个函数里，默认走"本地、零外部服务"的路线，让你 `pip install` 完就能跑起来。

本章我们钻进 `quivr_core/brain/` 下的真实源码，看清三条主线：默认嵌入器与默认向量库是怎么搭出来的（`brain_defaults.py`）、检索接口 `asearch` 如何把 LangChain 的原始返回包装成 Quivr 自己的 `SearchResult`（`brain.py` + `models.py`）、以及 Brain 如何通过 `save`/`load` 把整个知识库持久化到磁盘并安全复原（`brain.py` + `serialization.py`）。这一章是把"切好的 chunk"变成"可检索知识库"的闭环，也是后面 RAG 图 retrieve 节点的地基。

## 默认嵌入器：可选依赖的优雅降级

打开 `brain/brain_defaults.py`，`default_embedder()` 短到只有十行，却体现了 Quivr 对"可选依赖"的处理哲学：

```python
def default_embedder() -> Embeddings:
    try:
        from langchain_openai import OpenAIEmbeddings

        logger.debug("Loaded OpenAIEmbeddings as default LLM for brain")
        embedder = OpenAIEmbeddings()
        return embedder
    except ImportError as e:
        raise ImportError(
            "Please provide a valid Embedder or install quivr-core['base'] package for using the defaultone."
        ) from e
```

注意两个细节。第一，`OpenAIEmbeddings()` **不传任何参数**——它会从环境变量 `OPENAI_API_KEY` 读取密钥、使用默认模型（即 `text-embedding-ada-002`/`text-embedding-3-*` 系列，具体由 `langchain_openai` 版本决定）[[LangChain OpenAIEmbeddings]](https://python.langchain.com/docs/integrations/text_embedding/openai/)。这意味着默认配置下，用户只需设好环境变量就能用，无需在代码里写任何嵌入参数。

第二，`from langchain_openai import OpenAIEmbeddings` 是**延迟 import**（放在函数体内而非模块顶部）。`langchain_openai` 属于可选依赖：如果用户没装它，import 会抛 `ImportError`，被 `except` 捕获后转译成一条友好提示——"请提供合法的 Embedder，或安装 `quivr-core['base']` 套件来使用默认的"。这是一种优雅降级：核心包保持轻量，只有真正想用默认 OpenAI 嵌入的人才需要装重依赖。同样的模式在 `default_llm()` 里也复用了一遍。

返回类型标注为 `langchain_core.embeddings.Embeddings`，这是 LangChain 的嵌入抽象基类。也就是说，**默认实现是 OpenAI，但接口是开放的**——你完全可以把 `from_files` 的 `embedder=` 参数换成任意符合 `Embeddings` 协议的对象（比如本地的 HuggingFace 嵌入），Brain 的其余逻辑无需改动。

## 默认向量库：为什么是 FAISS

紧挨着的 `build_default_vectordb()` 负责把 chunk 列表建成向量索引：

```python
async def build_default_vectordb(
    docs: list[Document], embedder: Embeddings
) -> VectorStore:
    try:
        from langchain_community.vectorstores import FAISS

        logger.debug("Using Faiss-CPU as vector store.")
        if len(docs) > 0:
            vector_db = await FAISS.afrom_documents(documents=docs, embedding=embedder)
            return vector_db
        else:
            raise ValueError("can't initialize brain without documents")

    except ImportError as e:
        raise ImportError(
            "Please provide a valid vector store or install quivr-core['base'] package for using the default one."
        ) from e
```

默认向量库是 **Faiss-CPU**（通过 `langchain_community.vectorstores.FAISS` 引入），同样走延迟 import + `ImportError` 友好提示的模式。为什么默认选 FAISS 而不是 PGVector、Chroma 这类需要起服务的方案？因为 Quivr Core 的定位是一个**库（library）**：FAISS 是纯本地、内存内的索引，不依赖任何外部数据库或网络服务，`pip install` 即用[[FAISS 官方]](https://github.com/facebookresearch/faiss)。这让"开箱即跑"成为可能，也契合"先在本地把 RAG 跑通，再按需替换"的渐进式体验。

构建用的是 `FAISS.afrom_documents(documents=docs, embedding=embedder)`——异步版本的工厂方法[[LangChain FAISS]](https://python.langchain.com/docs/integrations/vectorstores/faiss/)。它内部会调用 `embedder` 把每个 `Document` 的文本编码成向量，再建成 FAISS 索引。源码上方有一条作者 `@aminediro` 留下的 TODO，坦言"嵌入调用通常不是对所有文档并发的，而是逐个等待"——这是个已知的性能优化点，提示我们默认路径下嵌入吞吐可能受 API 串行调用限制。

最关键的一行防御是 `if len(docs) > 0` 的检查：**docs 为空时直接抛 `ValueError("can't initialize brain without documents")`**。没有文档就无从建索引，与其让 FAISS 抛一个晦涩的内部错误，不如在入口处给出明确语义。这条约束在 `afrom_files` 和 `afrom_langchain_documents` 两条创建路径里都会触发，因为它们在 `vector_db is None` 时都会调到 `build_default_vectordb`：

```python
# afrom_files / afrom_langchain_documents 内部
if vector_db is None:
    vector_db = await build_default_vectordb(docs, embedder)
else:
    await vector_db.aadd_documents(docs)
```

可以看到，如果你**显式传入**了自己的 `vector_db`，Quivr 走的是 `aadd_documents` 把新 chunk 追加进去；只有在不传（默认）时才会触发 FAISS 的从零构建。这给了"自带向量库"和"用默认 FAISS"两种姿势。

## asearch：检索接口与候选池

向量库建好之后，对外的检索入口是 `Brain.asearch`（`brain/brain.py`）：

```python
async def asearch(
    self,
    query: str | Document,
    n_results: int = 5,
    filter: Callable | Dict[str, Any] | None = None,
    fetch_n_neighbors: int = 20,
) -> list[SearchResult]:
    if not self.vector_db:
        raise ValueError("No vector db configured for this brain")

    result = await self.vector_db.asimilarity_search_with_score(
        query, k=n_results, filter=filter, fetch_k=fetch_n_neighbors
    )

    return [SearchResult(chunk=d, distance=s) for d, s in result]
```

它把请求转发给 LangChain VectorStore 的 `asimilarity_search_with_score`，这个方法返回 `(Document, score)` 元组列表——既给文档又给相似度分数。几个参数值得拆开看：

- `n_results=5`：默认返回 5 条结果，映射到底层的 `k`。
- `fetch_n_neighbors=20`：默认先取 20 个候选，映射到底层的 `fetch_k`。

这里 `fetch_k > k`（20 > 5）不是随意设的。`fetch_k` 是**候选池**的大小：向量库先捞回 20 个最近邻作为候选，再在这个池子上做过滤（`filter`）或 MMR 之类的多样性重排，最终只吐出 `k` 个。如果候选池太小、过滤又很激进，可能根本凑不齐 `k` 条；把 `fetch_k` 放大到 `k` 的数倍，是为了给"过滤/重排留出余量"的常见工程做法[[LangChain FAISS]](https://python.langchain.com/docs/integrations/vectorstores/faiss/)。

`query` 类型是 `str | Document`，既支持纯文本查询，也支持传一个文档对象做"以文搜文"。开头的 `if not self.vector_db` 防御保证：在一个没有向量库的 Brain 上检索会得到明确的 `ValueError("No vector db configured for this brain")`，而不是 `None` 上的属性错误。

## SearchResult：Brain 对外的检索结果类型

`asearch` 没有把 LangChain 原始的 `(Document, score)` 元组直接抛给调用方，而是包成了 Quivr 自己定义的 `SearchResult`（`rag/entities/models.py`）：

```python
class SearchResult(BaseModel):
    chunk: Document
    distance: float
```

字段很克制：`chunk` 是命中的 LangChain `Document`（含 `page_content` 和 `metadata`），`distance` 是相似度/距离分数。`asearch` 末尾的列表推导 `[SearchResult(chunk=d, distance=s) for d, s in result]` 就是在做这层封装。

为什么要多包一层？这是一个**边界稳定性**的设计：`SearchResult` 是 Quivr 对外承诺的稳定契约，即便将来底层换了向量库、`asimilarity_search_with_score` 的返回结构有变，调用方拿到的仍然是结构一致的 `SearchResult`。注意字段名叫 `distance`——在不同向量库里这个分数的"方向"语义可能不同（有的是距离越小越相似，有的是相似度越大越相似），使用时要结合具体向量库理解。

## 序列化的边界：save 只认 FAISS + OpenAI

Brain 能不能存到磁盘、下次直接加载复用，而不必每次重新嵌入所有文件？能，但有边界。`Brain.save`（`brain/brain.py`）的核心逻辑是一连串 `isinstance` 类型守卫：

```python
from langchain_community.vectorstores import FAISS

if isinstance(self.vector_db, FAISS):
    vectordb_path = os.path.join(brain_path, "vector_store")
    os.makedirs(vectordb_path, exist_ok=True)
    self.vector_db.save_local(folder_path=vectordb_path)
    vector_store = FAISSConfig(vectordb_folder_path=vectordb_path)
else:
    raise Exception("can't serialize other vector stores for now")

if isinstance(self.embedder, OpenAIEmbeddings):
    embedder_config = EmbedderConfig(
        config=self.embedder.dict(exclude={"openai_api_key"})
    )
else:
    raise Exception("can't serialize embedder other than openai for now")
```

两条硬约束写得很直白：**向量库只有是 `FAISS` 才能存**（用 `FAISS.save_local` 落到 `vector_store/` 子目录，再生成一个 `FAISSConfig(vectordb_folder_path=...)`）；**嵌入器只有是 `OpenAIEmbeddings` 才能存**（用 `self.embedder.dict(...)` 序列化成 `EmbedderConfig`，并显式 `exclude={"openai_api_key"}`——绝不把密钥写进配置文件）。其余类型一律抛异常。这说明当前 `save`/`load` 的持久化能力刻意收窄到了"默认路线"上。

存储部分类似，按 `LocalStorage`/`TransparentStorage` 分别生成对应的 storage_config。最后把一切打包成 `BrainSerialized`：

```python
bserialized = BrainSerialized(
    id=self.id,
    name=self.name,
    chat_history=self.chat_history.get_chat_history(),
    llm_config=self.llm.get_config(),
    vectordb_config=vector_store,
    embedding_config=embedder_config,
    storage_config=storage_config,
)
with open(os.path.join(brain_path, "config.json"), "w") as f:
    f.write(bserialized.model_dump_json())
```

也就是说，一个保存下来的 Brain 在磁盘上长这样：一个 `brain_{id}/` 目录，里面有 `config.json`（元信息 + 各种 config）和 `vector_store/`（FAISS 的索引文件）。`config.json` 把对话历史、向量库配置、存储配置、LLM 配置、嵌入配置一站式记录下来。

## BrainSerialized 与判别式联合

`brain/serialization.py` 用 Pydantic 定义了这些配置模型。最值得注意的是 `BrainSerialized.vectordb_config` 用了**判别式联合（discriminated union）**：

```python
class FAISSConfig(BaseModel):
    vectordb_type: Literal["faiss"] = "faiss"
    vectordb_folder_path: str

class PGVectorConfig(BaseModel):
    vectordb_type: Literal["pgvector"] = "pgvector"
    pg_url: str
    pg_user: str
    pg_psswd: SecretStr
    table_name: str
    vector_dim: int

class BrainSerialized(BaseModel):
    ...
    vectordb_config: Union[FAISSConfig, PGVectorConfig] = Field(
        ..., discriminator="vectordb_type"
    )
```

`Field(..., discriminator="vectordb_type")` 告诉 Pydantic：反序列化时**看 `vectordb_type` 这个字段的值来决定用哪个模型**——值是 `"faiss"` 就解析成 `FAISSConfig`，是 `"pgvector"` 就解析成 `PGVectorConfig`[[Pydantic discriminated unions]](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions)。这是一种为扩展预留的结构：配置层面 PGVector 已经定义好了（连接串、表名、向量维度俱全），但**当前 `save`/`load` 的运行逻辑只实现了 FAISS 分支**。换句话说，数据模型走在了实现前面，给未来接入 PGVector 留了平滑的扩展口。`storage_config` 同样用 `storage_type` 做判别式联合。`EmbedderConfig` 则用 `embedder_type: Literal["openai_embedding"]` 标识类型，`config` 字段存放序列化后的嵌入参数字典。

## load：还原 Brain 与 allow_dangerous_deserialization

加载是保存的逆过程，`Brain.load` 从 `config.json` 反序列化回 `BrainSerialized`，再逐段重建：

```python
with open(os.path.join(folder_path, "config.json"), "r") as f:
    bserialized = BrainSerialized.model_validate_json(f.read())
...
if bserialized.embedding_config.embedder_type == "openai_embedding":
    from langchain_openai import OpenAIEmbeddings
    embedder = OpenAIEmbeddings(**bserialized.embedding_config.config)
...
if bserialized.vectordb_config.vectordb_type == "faiss":
    from langchain_community.vectorstores import FAISS
    vector_db = FAISS.load_local(
        folder_path=bserialized.vectordb_config.vectordb_folder_path,
        embeddings=embedder,
        allow_dangerous_deserialization=True,
    )
```

注意嵌入器是用 `OpenAIEmbeddings(**...config)` 重建的——由于保存时排除了 `openai_api_key`，加载后仍然要靠环境变量提供密钥。

最需要警惕的是 `FAISS.load_local(..., allow_dangerous_deserialization=True)`。FAISS 的 `save_local` 会用 **pickle** 序列化部分元数据（docstore、index-to-id 映射），而 pickle 反序列化能执行任意代码——所以 LangChain 默认禁止加载，必须显式把 `allow_dangerous_deserialization` 置为 `True` 才放行[[LangChain FAISS]](https://python.langchain.com/docs/integrations/vectorstores/faiss/)。Quivr 在这里硬编码了 `True`，其安全前提是：**你只加载自己用 `save` 存出来的、可信的文件**。如果加载来源不明的 brain 目录，等于把任意代码执行的风险引进来。这是把"持久化便利"与"反序列化安全"权衡后的取舍，使用时务必心里有数。

`load` 末尾用还原出的 `embedder`、`vector_db`、`llm`、`storage` 重新 `cls(...)` 出一个 Brain 实例。至此整个闭环跑通：嵌入器把 chunk 编成向量、FAISS 建索引、`asearch` 做检索、`save`/`load` 让一个建好的 Brain 可以持久化复用，不必每次重新嵌入。

## 本章小结

- `default_embedder()`（`brain_defaults.py`）默认返回**无参的 `OpenAIEmbeddings()`**，靠环境变量 `OPENAI_API_KEY` 工作；延迟 import + `ImportError` 友好提示是"可选依赖优雅降级"的典型写法。
- `build_default_vectordb()` 默认用 `langchain_community.vectorstores.FAISS`（Faiss-CPU），通过 `FAISS.afrom_documents(documents=docs, embedding=embedder)` 异步构建；选 FAISS 是因为它本地、零外部服务，契合"库"的定位。
- docs 为空会抛 `ValueError("can't initialize brain without documents")`，在 `afrom_files`/`afrom_langchain_documents` 的默认路径上都会触发。
- `asearch` 转发到 `asimilarity_search_with_score(query, k=n_results, filter=..., fetch_k=fetch_n_neighbors)`，默认 `n_results=5`、`fetch_n_neighbors=20`；`fetch_k > k` 是为过滤/重排留候选池余量。
- `SearchResult`（`models.py`）只有 `chunk: Document` 和 `distance: float` 两个字段，是 Brain 对外稳定的检索结果契约。
- `save` 用 `isinstance` 守卫，**只支持 FAISS 向量库和 OpenAIEmbeddings 嵌入器**，其余抛异常；序列化嵌入器时 `exclude={"openai_api_key"}` 不落盘密钥。
- 保存产物为 `brain_{id}/config.json`（`BrainSerialized`：chat_history / vectordb_config / storage_config / llm_config / embedding_config）+ `vector_store/`（FAISS 索引）。
- `vectordb_config` 用 Pydantic 判别式联合（`FAISSConfig` vs `PGVectorConfig`，按 `vectordb_type` 判别），数据模型已为 PGVector 预留，但 `save`/`load` 当前只实现 FAISS 分支。
- `load` 用 `FAISS.load_local(..., allow_dangerous_deserialization=True)`，背后是 pickle 反序列化的代码执行风险，安全前提是只加载自己存的可信文件。
- 嵌入 → 建索引 → 检索 → 持久化，构成"chunk → 可检索知识库"的闭环，为后续 RAG 图的 retrieve 节点提供 `vector_store`。

## 动手实验

1. **实验一：建库后检索** — 用 `Brain.from_files(name="demo", file_paths=[...])` 建一个 Brain，然后 `results = await brain.asearch("你的问题")`，遍历 `results` 打印 `result.chunk.page_content` 和 `result.distance`，观察默认 `n_results=5` 是否正好返回 5 条。
2. **实验二：看 config.json 结构** — 调 `await brain.save("./out")`，到生成的 `out/brain_<id>/` 目录下打开 `config.json`，确认其中有 `vectordb_config`（`vectordb_type="faiss"`）、`embedding_config`，并验证 `vector_store/` 子目录里有 FAISS 索引文件、且 config 里**没有**明文 API key。
3. **实验三：空 docs 触发 ValueError** — 直接调 `await build_default_vectordb([], embedder)`（或用一个会被全部跳过的空文件走 `afrom_files`），确认抛出 `ValueError("can't initialize brain without documents")`，体会入口处防御的语义价值。
4. **实验四：观察 allow_dangerous_deserialization** — 用 `Brain.load("./out/brain_<id>")` 重新加载上面保存的 Brain，再次 `asearch` 验证结果一致；然后到 `brain.py` 的 `load` 里把 `allow_dangerous_deserialization=True` 临时改成 `False`，重新加载，观察 LangChain 抛出的安全报错，理解这个 flag 的含义与风险边界。

> **下一章预告**：到这里 Brain 已经能"切分 → 嵌入 → 检索 → 持久化"。但真正的"问答"还没出现——从查询到答案，中间隐藏着改写、检索、重排、生成等一连串步骤。第 7 章《配置驱动的 LangGraph 工作流》将拆开 Quivr 的 RAG 配置体系，看它如何用一份声明式配置驱动一张 LangGraph 状态图，把这些步骤编排成一条可定制的工作流。
