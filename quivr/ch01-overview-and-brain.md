# 第 1 章 库定位与 Brain 全景

Quivr 是一个把"第二大脑"产品化的开源项目，而本书第四部要解构的并不是它的产品外壳，而是藏在 `core/quivr_core/` 目录下那个真正负责检索增强生成（RAG）的引擎——`quivr-core`。它是一个可以 `pip install` 的纯 Python 库，README 用一句话给自己定位："The RAG of Quivr.com"，根目录 README 则称它为 "the brain of Quivr.com"。换句话说，整个 Quivr 产品的智能内核，就是这个库。读懂它，就读懂了 Quivr 怎么把一堆文件变成可以提问的知识库。

本章是整部 Quivr 解构的"地图"。我们会先确认这个库在技术栈上站在哪里，再聚焦于它的顶层编排类 `Brain`——从构造函数、工厂方法、默认依赖，到文件处理、向量检索、流式问答、序列化保存，逐个机制对照源码看清楚。`Brain` 把存储、解析、嵌入、向量库、LLM、RAG 图编排成一条流水线，它就是后续每一章的总枢纽。读完本章，你会对"一个文件如何被吃进 Brain、又如何被检索出来回答问题"有一张完整的全景图，后面的章节只是把这张图的每个方框放大。

## quivr-core 的库定位与技术栈

`quivr-core` 把自己定位成一个轻量、可嵌入的库：根目录 README 写道 "Simply install quivr-core and add it to your project. You can now ingest your files and ask questions." 它的安装方式就是 `pip install quivr-core`（见 `core/README.md`），不依赖任何 Web 服务即可在脚本里直接跑起来。这与上层的 Quivr 产品（带前端、数据库、API）形成分层：产品是壳，`quivr-core` 是芯 [[Quivr GitHub]](https://github.com/QuivrHQ/quivr)。

从 `brain/brain.py` 顶部的导入可以一眼看出它的技术底座。它大量依赖 LangChain 生态：

```python
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
```

也就是说，Quivr 没有自己重造"文档/嵌入/向量库"这些抽象，而是直接复用 LangChain 的标准接口 [[LangChain]](https://python.langchain.com/)。这是一个重要的设计取舍：复用 `VectorStore`/`Embeddings` 这类协议，意味着任何符合 LangChain 接口的向量库或嵌入模型都能被插进 Brain，而不必为 Quivr 单独适配。

而真正的 RAG 流程编排，则交给基于 LangGraph 的 `QuivrQARAGLangGraph`（`from quivr_core.rag.quivr_rag_langgraph import QuivrQARAGLangGraph`），这是后面 RAG 章节的主角 [[LangGraph]](https://langchain-ai.github.io/langgraph/)。源码里用到的 `Self` 类型注解（`from typing import ... Self`）也透露了语言要求：`Self` 是较新 Python 才进入 `typing` 的特性，配合项目对现代类型语法（`UUID | None` 这种 `X | None` 联合写法）的广泛使用，决定了它需要较新的 Python（3.10+）。读这个库时，建议先把环境备好，否则连导入都会报语法错误。

## Brain：顶层编排类与构造

`Brain` 类的文档字符串把它的职责讲得很清楚：一个 Brain 是"a collection of knowledge one wants to retrieve information from"，它负责把文件存进 storage、处理文件抽取文本与元数据、把结果存进向量库（默认 FAISS）、建立索引，并用 Quivr 的工作流做检索增强生成。这四件事——存、解析、向量化、问答——正是后续四章的脉络。

构造函数 `__init__` 用了全关键字参数（注意签名里的 `*`），把所有依赖显式注入：

```python
def __init__(
    self,
    *,
    name: str,
    llm: LLMEndpoint,
    id: UUID | None = None,
    vector_db: VectorStore | None = None,
    embedder: Embeddings | None = None,
    storage: StorageBase | None = None,
    workspace_id: UUID | None = None,
    chat_id: UUID | None = None,
):
```

这里有一个值得注意的设计：除了 `name` 和 `llm`，其余依赖都允许为 `None`。这说明 `Brain` 本身只是一个"装配体"，它不强制你怎么造每个零件，零件的默认值由工厂方法负责填充（见下一节）。构造的最后几行把依赖挂到实例上，并调用 `self._chats = self._init_chats()` 初始化对话历史。`_init_chats()` 的实现很简单：用 `uuid4()` 生成一个 `chat_id`，建一个 `ChatHistory(chat_id=chat_id, brain_id=self.id)`，放进字典返回；随后 `self.default_chat = list(self._chats.values())[0]` 取第一个作为默认对话。`chat_history` 是个 `@property`，直接返回 `self.default_chat`。这意味着每个新建的 Brain 天然带着一个空的默认会话，问答时若不显式传 `chat_history` 就会用它。

`Brain` 还提供了几个自省入口。`info()` 把自身打包成 `BrainInfo`（见 `brain/info.py`），里面聚合了 `ChatHistoryInfo`、`LLMInfo`、`StorageInfo` 三个 `@dataclass`：`ChatHistoryInfo` 记录 `nb_chats`/`current_default_chat`/`current_chat_history_length`，`LLMInfo` 记录 `model`/`llm_base_url`/`temperature`/`max_tokens`/`supports_function_calling`，`StorageInfo` 记录 `storage_type`/`n_files`。`__repr__` 用 `PrettyPrinter` 直接 pformat `self.info()`，而 `print_info()` 用 `rich` 的 `Tree` 和 `Panel` 把这些信息渲染成一棵带 emoji 的树（`BrainInfo.to_tree()` 里能看到 "📊 Brain Information"、"🧠 Brain Name" 等节点）。这套自省能力让 Brain 在交互式调试时非常直观——你拿到一个 Brain，敲一下 `print_info()` 就能看清它挂着什么模型、几个文件、几轮对话。

另外要留意一个"待实现"的缺口：`add_file` 方法目前只有一句 `raise NotImplementedError`，注释写着"add it to storage / add it to vectorstore"。这说明当前版本一旦用工厂方法建好 Brain，并没有提供增量追加单个文件的官方路径，向量库的内容在创建时一次性灌入。这是阅读源码时应当心里有数的边界。

## 工厂方法与"默认即可用"的依赖

`Brain` 真正面向用户的入口是几个工厂类方法：`afrom_files`（异步）、`from_files`（同步包装）、`afrom_langchain_documents`。它们体现了一个核心设计哲学——默认即可用。以 `afrom_files` 为例，它的签名直接把默认依赖写进参数：`storage: StorageBase = TransparentStorage()`，而 `llm`、`embedder`、`vector_db` 默认为 `None`，在函数体里按需补齐：

```python
if llm is None:
    llm = default_llm()
if embedder is None:
    embedder = default_embedder()
```

这三个默认值来自 `brain/brain_defaults.py`：`default_llm()` 返回 `LLMEndpoint.from_config(LLMEndpointConfig(supplier=DefaultModelSuppliers.OPENAI, model="gpt-4o"))`，即默认 LLM 是 OpenAI 的 `gpt-4o`；`default_embedder()` 返回一个无参数的 `OpenAIEmbeddings()`；而向量库由 `build_default_vectordb(docs, embedder)` 用 `FAISS.afrom_documents(...)` 构建，日志里那句 "Using Faiss-CPU as vector store." 直接说明了默认就是 FAISS-CPU。三个函数都包了 `try/except ImportError`，缺包时抛出引导性的报错，提示安装 `quivr-core['base']`。

`afrom_files` 的主体流程是一条清晰的流水线：先 `uuid4()` 生成 `brain_id`，再遍历 `file_paths`，对每个路径 `file = await load_qfile(brain_id, path)` 并 `await storage.upload_file(file)` 上传到存储；接着 `docs = await process_files(storage=storage, skip_file_error=skip_file_error, **processor_kwargs)` 把文件解析成 `list[Document]`；最后若没传 `vector_db` 就 `await build_default_vectordb(docs, embedder)`，否则 `await vector_db.aadd_documents(docs)`，再用这些零件构造并返回 `Brain` 实例。`from_files` 则只是用 `loop.run_until_complete(...)` 把 `afrom_files` 同步化的薄封装。`afrom_langchain_documents` 跳过了 storage 上传与解析两步，直接拿现成的 `list[Document]` 进向量库，适合已经有 LangChain 文档的场景。

值得提醒的是 `storage: StorageBase = TransparentStorage()` 这种"可变默认值"写法在 Python 里通常是反模式，因为默认实例会在所有调用间共享；这是阅读源码时值得留意的一个细节。

还有一个贯穿 `Brain` API 的设计模式：同步/异步双轨。库的底层一律是 `async`（`afrom_files`、`asearch`、`ask_streaming`、`aask`、`save`），而面向不熟悉 asyncio 的用户，又用 `loop = asyncio.get_event_loop()` + `loop.run_until_complete(...)` 提供了同步包装（`from_files`、`ask`）。这样既保留了异步并发的能力（例如批量嵌入），又降低了"写个脚本试一下"的门槛。读源码时可以把同步方法看作异步方法的一层薄壳，真正的逻辑都在 `a` 开头的那个里。

## process_files：从文件到 Document 的总闸

`process_files` 是一个模块级异步函数（不在类内），是 Brain 解析层的总闸门，但它本身很薄——真正的解析逻辑被推给各个 processor，这正是后续"文件抽象与解析"章节的引子。它的核心循环是：

```python
for file in await storage.get_files():
    if file.file_extension:
        processor_cls = get_processor_class(file.file_extension)
        processor = processor_cls(**processor_kwargs)
        docs = await processor.process_file(file)
        knowledge.extend(docs)
```

可以看到三层解耦：从 `storage.get_files()` 拿文件、用 `get_processor_class(file.file_extension)`（来自 `quivr_core.processor.registry`）按扩展名查注册表取出 processor 类、再由 processor 的 `process_file` 把文件变成 `list[Document]`。`skip_file_error` 参数控制容错：当文件没有扩展名、或 `get_processor_class` 抛 `KeyError`（无对应 processor）时，若 `skip_file_error=True` 就 `continue` 跳过，否则抛 `ValueError`/`Exception`。这里"按扩展名查注册表"的机制，是 Quivr 支持多种文件格式的关键，第 2、3 章会深入。

## asearch：向量检索的最小入口

`asearch` 是 Brain 暴露的纯检索接口（不经过 LLM 生成），它直接把活儿转交给 LangChain 向量库。核心一行是：

```python
result = await self.vector_db.asimilarity_search_with_score(
    query, k=n_results, filter=filter, fetch_k=fetch_n_neighbors
)
return [SearchResult(chunk=d, distance=s) for d, s in result]
```

它调用向量库的 `asimilarity_search_with_score`，返回 `(Document, score)` 的列表，再逐一包成 `SearchResult(chunk=d, distance=s)`（`SearchResult` 来自 `quivr_core.rag.entities.models`）。默认参数 `n_results: int = 5`（即 `k=5`）和 `fetch_n_neighbors: int = 20`（即 `fetch_k=20`）值得记住：`fetch_k` 通常是先取回的候选数量、`k` 是最终返回数量，这种"先粗取 20、再精选 5"的设定是 FAISS 检索的常见做法 [[FAISS]](https://faiss.ai/)。方法开头还有一道防护：`if not self.vector_db: raise ValueError("No vector db configured for this brain")`，避免在没有向量库的 Brain 上检索。

## ask_streaming：流式问答的总入口

如果说 `asearch` 是"只检索"，那么 `ask_streaming` 就是 Brain 完整的 RAG 问答入口，也是后续 RAG 章节真正的总门。它先决定用哪个 LLM：如果传入的 `retrieval_config.llm_config` 与 Brain 自带 LLM 的配置不同，就用 `LLMEndpoint.from_config(...)` 覆盖；若没传 `retrieval_config`，则用 `RetrievalConfig(llm_config=self.llm.get_config())` 兜底。随后它实例化 RAG 引擎：

```python
rag_instance = QuivrQARAGLangGraph(
    retrieval_config=retrieval_config, llm=llm, vector_store=self.vector_db
)
```

这一行就是 Brain 与 LangGraph RAG 图的接缝——`Brain` 把"配置 + LLM + 向量库"三样交给 `QuivrQARAGLangGraph`，之后的检索、组装上下文、调用模型全在那张图里完成。问答通过 `async for response in rag_instance.answer_astream(...)` 流式产出，期间逐块 `yield`（仅当 `not response.last_chunk` 时 yield 中间块），并把每块的 `answer` 累加进 `full_answer`。流结束后，它把这一轮对话追加进历史：`chat_history.append(HumanMessage(content=question))` 和 `chat_history.append(AIMessage(content=full_answer))`，最后再 yield 一次完整 `response`。

值得注意的是可观测性接入：方法里构造了一个 `LangchainMetadata`，把 `run_id`、`workspace_id`、`chat_id` 分别映射为 `langfuse_trace_id`、`langfuse_user_id`、`langfuse_session_id` 传进 RAG 图。这说明 Quivr 内建了对 Langfuse 追踪的支持，便于线上观测每一次问答 [[Langfuse]](https://langfuse.com/)。此外还有两个同步/聚合包装：`aask` 把流式结果累加成一个 `ParsedRAGResponse`（拼接 `answer` 并保留 `metadata`），`ask` 再用 `run_until_complete` 把 `aask` 同步化。三者形成"流式底座 + 聚合 + 同步"的三层 API。

## save / load：序列化的边界与限制

`Brain` 通过 `save`/`load` 实现持久化，序列化的契约定义在 `brain/serialization.py` 的 `BrainSerialized`（pydantic 模型）里。它聚合了 `id`、`name`、`chat_history`、`vectordb_config`、`storage_config`、`llm_config`、`embedding_config`。其中 `vectordb_config` 与 `storage_config` 都用了 pydantic 的 `discriminator` 字段做联合类型分发（如 `Field(..., discriminator="vectordb_type")`），所以反序列化时能根据 `vectordb_type`/`storage_type` 自动选对子类。

`save` 是异步方法，会在 `folder_path` 下建一个 `brain_{self.id}` 目录，把配置写进 `config.json`（`f.write(bserialized.model_dump_json())`）。但它的实现暴露了明确的能力边界：向量库这里 `if isinstance(self.vector_db, FAISS):` 才调用 `self.vector_db.save_local(...)`，否则直接 `raise Exception("can't serialize other vector stores for now")`；embedder 也只支持 OpenAI——`if isinstance(self.embedder, OpenAIEmbeddings):` 才序列化，且 `self.embedder.dict(exclude={"openai_api_key"})` 刻意排除了 API key，避免把密钥落盘。storage 仅支持 `LocalStorage` 与 `TransparentStorage` 两种。

`load` 是同步类方法，读出 `config.json` 后用 `BrainSerialized.model_validate_json(...)` 解析，再按 `storage_type`/`embedder_type`/`vectordb_type` 逐个还原。FAISS 的加载这样写：

```python
vector_db = FAISS.load_local(
    folder_path=bserialized.vectordb_config.vectordb_folder_path,
    embeddings=embedder,
    allow_dangerous_deserialization=True,
)
```

`allow_dangerous_deserialization=True` 是个安全敏感点：FAISS 的本地存储用 pickle，反序列化任意 pickle 文件有代码执行风险，因此 LangChain 要求显式开启这个开关。Quivr 直接硬编码为 `True`，意味着只应加载你自己保存的、可信的 Brain 目录 [[FAISS]](https://faiss.ai/)。综合来看，当前版本的"保存/加载"是有意收窄到"OpenAI embedder + FAISS"这一条最常用路径上的。

## 本章小结

- `quivr-core` 是一个可 `pip install quivr-core` 的纯 Python 库（见 `core/README.md`），是 Quivr 产品的 RAG 内核，被根 README 称为 "the brain of Quivr.com"。
- 它重度构建在 LangChain（`langchain_core` 的 `Document`/`Embeddings`/`VectorStore`/消息类型）与 LangGraph（`QuivrQARAGLangGraph`）之上，并用到了需要较新 Python 的 `typing.Self`，对应 3.10+ 的要求。
- `Brain.__init__` 用全关键字参数注入 `name/llm/vector_db/embedder/storage/id/workspace_id/chat_id`，除 `name` 和 `llm` 外其余可为 `None`，体现"装配体"定位（见 `brain.py`）。
- `_init_chats()` 用 `uuid4()` 建一个默认 `ChatHistory` 并设为 `default_chat`，`chat_history` 属性直接返回它，使每个 Brain 天生带一个空会话。
- 工厂方法 `afrom_files`/`from_files`/`afrom_langchain_documents` 实现"默认即可用"：默认 `storage=TransparentStorage()`、`llm=default_llm()`（`gpt-4o`）、`embedder=default_embedder()`（`OpenAIEmbeddings`）、向量库走 `build_default_vectordb`（FAISS-CPU），均定义在 `brain_defaults.py`。
- `process_files` 通过 `storage.get_files()` 取文件、`get_processor_class(file.file_extension)` 按扩展名查注册表、`processor.process_file(file)` 解析成 `list[Document]`，并由 `skip_file_error` 控制容错。
- `asearch` 调用 `vector_db.asimilarity_search_with_score(query, k=n_results, filter=filter, fetch_k=fetch_n_neighbors)`，默认 `n_results=5`、`fetch_n_neighbors=20`，结果包成 `SearchResult(chunk, distance)`。
- `ask_streaming` 构建 `RetrievalConfig`、实例化 `QuivrQARAGLangGraph(retrieval_config, llm, vector_store)`、用 `answer_astream` 流式产出，并把 `HumanMessage`/`AIMessage` 追加进对话历史；`aask`/`ask` 是其聚合与同步包装。
- 问答会构造 `LangchainMetadata`，把 `run_id`/`workspace_id`/`chat_id` 映射为 Langfuse 的 trace/user/session id，内建可观测性。
- `save`/`load` 以 `BrainSerialized` 写入/读取 `config.json`，FAISS 用 `save_local`/`load_local`（`allow_dangerous_deserialization=True`），当前仅支持 OpenAI embedder + FAISS 序列化，且刻意排除 `openai_api_key`。

## 动手实验

1. **实验一：用默认依赖跑通一个最小 Brain** — 在 `/home/mira/files/quivr_src/core/quivr_core/` 阅读 `brain/brain.py` 的 `afrom_files` 与 `brain_defaults.py`，确认默认 `llm=default_llm()`（`gpt-4o`）、`embedder=default_embedder()`、`storage=TransparentStorage()`。然后写一段脚本调用 `Brain.from_files(name="demo", file_paths=[...])` 喂入一个本地 txt，运行后调用 `brain.print_info()`，对照 `brain/info.py` 的 `BrainInfo.to_tree()` 验证输出的树形结构里出现了 ID、Brain Name、Files、Chats、LLM 五个节点。

2. **实验二：观察检索参数 k 与 fetch_k 的作用** — 在 `brain.py` 中定位 `asearch`，确认它把 `n_results` 映射成 `k`、`fetch_n_neighbors` 映射成 `fetch_k`。用上一个实验建好的 Brain 调用 `await brain.asearch("你的问题", n_results=2, fetch_n_neighbors=10)`，打印返回的 `SearchResult` 数量与每个 `result.distance`，验证返回数量等于 `n_results`，并思考为何 `fetch_k` 通常应大于 `k`。

3. **实验三：跟踪一次文件如何变成 Document** — 阅读 `brain.py` 顶部的 `process_files` 函数，沿着 `storage.get_files()` → `get_processor_class(file.file_extension)` → `processor.process_file(file)` 这条链路，到 `quivr_core/processor/registry.py` 里查看 `get_processor_class` 如何按扩展名返回 processor。给 `afrom_files` 传一个没有对应 processor 的扩展名文件，分别在 `skip_file_error=True` 和 `False` 下运行，观察是被跳过还是抛出 `Exception`。

4. **实验四：保存与加载 Brain 并验证序列化边界** — 阅读 `brain.py` 的 `save`/`load` 和 `brain/serialization.py` 的 `BrainSerialized`。对一个用默认依赖（OpenAI embedder + FAISS）建好的 Brain 调用 `await brain.save("/tmp/mybrain")`，检查生成的 `brain_<id>/config.json` 与 `vector_store/` 目录，确认 `config.json` 中没有 `openai_api_key`；再用 `Brain.load("/tmp/mybrain/brain_<id>")` 还原并 `print_info()`。最后阅读 `save` 中针对非 FAISS、非 OpenAIEmbeddings 抛出的 `raise Exception("can't serialize ...")` 分支，理解当前序列化的能力边界。

> **下一章预告**：本章我们把 Brain 当作"总指挥"鸟瞰了一遍，但它发出的第一道命令——`storage.get_files()`——背后到底是什么？第 2 章「文件抽象与 Storage 存储层」将钻进 `QuivrFile`、`StorageBase` 与 `TransparentStorage`/`LocalStorage`，看清一个文件从被 `upload_file` 吃进来，到带着扩展名与元数据交给解析器之前，经历了怎样的抽象与落地。
