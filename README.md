# RAG in Action：解构开源 RAG 系统

这是一本基于**真实源码**逐章解构开源 RAG（检索增强生成）系统实现原理的技术书籍。目标读者是希望深入理解 RAG 引擎内部机制的工程师——不是停留在"RAG 一般怎么做"的科普层面，而是钻进具体项目的源码，读懂它**到底是怎么实现的**。

## 写作原则

- **零虚构**：每一个技术论断都来自实际读过的源码文件，引用真实的文件名、函数名、常量与默认值。拿不准就回去读代码。
- **以源码为证**：宁可少讲一个机制，也不写一个没读过源码的机制。
- **面向工程师**：解释"它做了什么"，更解释"为什么这样设计"，且意图推断要有源码支撑。
- **逐章可验证**：每章配 4 个动手实验，引导读者亲自在源码里验证结论。

## 第一部 RAGFlow

第一个解构对象是 [RAGFlow](https://github.com/infiniflow/ragflow)——基于深度文档理解（deep document understanding）的开源 RAG 引擎。章节弧线按"运行时由外到内、由数据摄入到检索生成"展开：

| 章 | 主题 | 解构焦点 |
|---|---|---|
| 第 1 章 | 项目全景与运行时入口 | monorepo 结构、双进程模型、启动引导、依赖基础设施 |
| 第 2 章 | 深度文档理解（DeepDoc） | PDF/Office/视觉解析器、OCR、版面分析 |
| 第 3 章 | 分块模板系统 | `rag/app` 下按文档类型分块的模板化策略 |
| 第 4 章 | 任务执行器与异步管线 | `task_executor`、消息队列、进度更新 |
| 第 5 章 | 索引与向量化 | embedding、文档存储抽象、ES/Infinity/OpenSearch |
| 第 6 章 | 检索与融合重排 | `rag/nlp` 查询、多路召回、term weighting |
| 第 7 章 | GraphRAG 知识图谱 | 实体抽取、图构建、社区报告 |
| 第 8 章 | RAPTOR 树状摘要 | 层次聚类与摘要 |
| 第 9 章 | Agent 与编排画布 | `agent/canvas`、组件、DSL |
| 第 10 章 | 沙箱与代码执行 | `agent/sandbox`、gVisor 隔离 |
| 第 11 章 | 模型接入层 | `rag/llm`、LiteLLM、provider 抽象 |
| 第 12 章 | MCP 接入 | `mcp` client/server |
| 第 13 章 | 收尾：可迁移的 RAG 工程原则 | 跨项目归纳的设计透镜 |

> 实际章节会随源码阅读深入而裁剪调整，不为凑数硬塞。

RAGFlow 各章节文件位于 [`ragflow/`](./ragflow/)：

1. [第 1 章 项目全景与运行时入口](./ragflow/ch01-overview-and-runtime-entry.md)
2. [第 2 章 深度文档理解（DeepDoc）](./ragflow/ch02-deepdoc-document-understanding.md)
3. [第 3 章 分块模板系统](./ragflow/ch03-chunking-templates.md)
4. [第 4 章 任务执行器与异步管线](./ragflow/ch04-task-executor-async-pipeline.md)
5. [第 5 章 索引与向量化](./ragflow/ch05-indexing-and-embedding.md)
6. [第 6 章 检索与融合重排](./ragflow/ch06-retrieval-and-rerank.md)
7. [第 7 章 GraphRAG 知识图谱](./ragflow/ch07-graphrag.md)
8. [第 8 章 RAPTOR 树状摘要](./ragflow/ch08-raptor.md)
9. [第 9 章 Agent 与编排画布](./ragflow/ch09-agent-canvas.md)
10. [第 10 章 沙箱与代码执行](./ragflow/ch10-sandbox.md)
11. [第 11 章 模型接入层](./ragflow/ch11-model-access-layer.md)
12. [第 12 章 MCP 接入](./ragflow/ch12-mcp.md)
13. [第 13 章 收尾：可迁移的 RAG 工程原则](./ragflow/ch13-engineering-principles.md)

## 第二部 AnythingLLM

第二个解构对象是 [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)——用 JavaScript/TypeScript 构建的全栈、多进程、自托管 RAG + Agent 平台。与 RAGFlow 的 Python 重型单体不同，它把系统拆成 `server`、`collector`、`frontend` 三个独立进程，并以高度抽象的 provider 插拔机制支持数十家模型、十余种向量库。章节弧线同样由外到内、由数据摄入到对话编排展开：

| 章 | 主题 | 解构焦点 |
|---|---|---|
| 第 1 章 | 三进程运行时全景 | monorepo、server/collector/frontend、`comKey` RSA 滚动签名跨进程信任 |
| 第 2 章 | 文档采集器 Collector | `processSingleFile`、MIME 分发、OCR、音频转写、网页抓取 |
| 第 3 章 | 文本切分 TextSplitter | 递归字符切分、chunkSize/Overlap、元数据头注入 |
| 第 4 章 | 向量化引擎 EmbeddingEngines | native 零配置引擎、provider 抽象、`maxConcurrentChunks` 节流 |
| 第 5 章 | 向量数据库抽象层 | `BaseVectorDatabaseProvider` 契约、LanceDB 默认实现、namespace 隔离 |
| 第 6 章 | 文档管理与工作区 | Workspace 配置、pin 全文注入、文档同步 |
| 第 7 章 | RAG 聊天与检索流程 | chat/query 模式、相似度阈值、citations、上下文压缩 |
| 第 8 章 | LLM 模型接入层 | `BaseLLMProvider` 契约、流式 `LLMPerformanceMonitor`、modelMap |
| 第 9 章 | Agent 系统 aibitat | `@agent` 触发、agent loop、插件契约、function calling |
| 第 10 章 | Agent Flows 可视化编排 | 声明式 steps、变量传递、内置执行器 |
| 第 11 章 | MCP 与多渠道接入 | MCP 客户端、Hypervisor、embed/Telegram/扩展/移动端 |
| 第 12 章 | 安全加密与工程原则收尾 | 两套密钥混合加密、七条可迁移工程透镜 |

AnythingLLM 各章节文件位于 [`anything-llm/`](./anything-llm/)：

1. [第 1 章 三进程运行时全景](./anything-llm/ch01-three-process-runtime.md)
2. [第 2 章 文档采集器 Collector](./anything-llm/ch02-collector-document-ingestion.md)
3. [第 3 章 文本切分 TextSplitter](./anything-llm/ch03-text-splitter.md)
4. [第 4 章 向量化引擎 EmbeddingEngines](./anything-llm/ch04-embedding-engines.md)
5. [第 5 章 向量数据库抽象层](./anything-llm/ch05-vectordb-abstraction.md)
6. [第 6 章 文档管理与工作区](./anything-llm/ch06-document-and-workspace.md)
7. [第 7 章 RAG 聊天与检索流程](./anything-llm/ch07-rag-chat-pipeline.md)
8. [第 8 章 LLM 模型接入层](./anything-llm/ch08-llm-provider-layer.md)
9. [第 9 章 Agent 系统 aibitat](./anything-llm/ch09-agent-aibitat.md)
10. [第 10 章 Agent Flows 可视化编排](./anything-llm/ch10-agent-flows.md)
11. [第 11 章 MCP 与多渠道接入](./anything-llm/ch11-mcp-and-channels.md)
12. [第 12 章 安全加密与工程原则收尾](./anything-llm/ch12-security-and-principles.md)

## 第三部 LlamaIndex

第三个解构对象是 [LlamaIndex](https://github.com/run-llama/llama_index)——与前两部的"产品"定位不同,它是一个**框架**:一个 namespace package 组织的 monorepo,把核心抽象(`llama-index-core`)与每一个具体 provider/向量库/reader(`llama-index-integrations` 下可独立 `pip install` 的小包)彻底解耦。它不替你做决定,而是提供一套可插拔的积木与组装规则。章节弧线沿数据流由上而下展开:一切从 `Node` 这个统一数据模型出发,经加载、切分、摄入、嵌入、存储、索引、检索、合成,直到 Workflow 与 Agent 的事件驱动编排:

| 章 | 主题 | 解构焦点 |
|---|---|---|
| 第 1 章 | 框架全景与核心数据模型 | namespace monorepo、`schema.py`、`BaseComponent`/`TransformComponent`/`BaseNode`、关系图、`MetadataMode` |
| 第 2 章 | Settings 全局配置与依赖解析 | `Settings` 单例、懒加载、`resolve_llm`/`resolve_embed_model`、全局默认与局部覆盖 |
| 第 3 章 | 数据加载 Readers | `BaseReader`、`SimpleDirectoryReader`、fsspec 抽象、文件元数据注入 |
| 第 4 章 | 节点解析与切分 NodeParser | `TransformComponent` 接口、`SentenceSplitter` 回退链、关系建立、`HierarchicalNodeParser` 父子层级 |
| 第 5 章 | 摄入管线 IngestionPipeline | 变换串链、`hash` 去重与 `DocstoreStrategy` 增量更新、`IngestionCache`、并行 |
| 第 6 章 | 索引体系 Indices | `BaseIndex` 模板方法、`IndexStruct`、`VectorStoreIndex`、持久化还原 |
| 第 7 章 | 嵌入模型抽象 Embeddings | `BaseEmbedding`、批处理、`SimilarityMode`、Pooling、`MockEmbedding` |
| 第 8 章 | 存储与向量库抽象 | `StorageContext`、KVStore→DocStore/IndexStore 三层、`ref_doc` 反向索引、向量库契约 |
| 第 9 章 | 检索器 Retrievers | `BaseRetriever`、`VectorIndexRetriever`、`AutoMergingRetriever`、`QueryFusionRetriever`(RRF) |
| 第 10 章 | 响应合成 ResponseSynthesizers | `ResponseMode` 谱系、`Refine`/`CompactAndRefine`/`TreeSummarize`、token 预算权衡 |
| 第 11 章 | 查询引擎与对话引擎 | `RetrieverQueryEngine` 端到端串联、`ChatMode`、`ChatMemoryBuffer` 记忆管理 |
| 第 12 章 | Workflow 事件驱动编排引擎 | `@step` + Event 自动连边、`Context` 与 `collect_events`、图校验、断点续跑 |
| 第 13 章 | Agent 系统 | Agent 即 Workflow、`FunctionAgent`/`ReActAgent`、工具抽象、`AgentWorkflow` 多智能体 handoff |
| 第 14 章 | 可观测性与工程原则收尾 | `Dispatcher`/`BaseEvent`/`BaseSpan` 埋点体系、跨三部书归纳的七条可迁移工程原则 |

LlamaIndex 各章节文件位于 [`llama-index/`](./llama-index/)：

1. [第 1 章 框架全景与核心数据模型](./llama-index/ch01-framework-overview-and-data-model.md)
2. [第 2 章 Settings 全局配置与依赖解析](./llama-index/ch02-settings-and-dependency-resolution.md)
3. [第 3 章 数据加载 Readers](./llama-index/ch03-readers-data-loading.md)
4. [第 4 章 节点解析与切分 NodeParser](./llama-index/ch04-node-parser-and-chunking.md)
5. [第 5 章 摄入管线 IngestionPipeline](./llama-index/ch05-ingestion-pipeline.md)
6. [第 6 章 索引体系 Indices](./llama-index/ch06-indices.md)
7. [第 7 章 嵌入模型抽象 Embeddings](./llama-index/ch07-embeddings.md)
8. [第 8 章 存储与向量库抽象](./llama-index/ch08-storage-and-vector-stores.md)
9. [第 9 章 检索器 Retrievers](./llama-index/ch09-retrievers.md)
10. [第 10 章 响应合成 ResponseSynthesizers](./llama-index/ch10-response-synthesizers.md)
11. [第 11 章 查询引擎与对话引擎](./llama-index/ch11-query-and-chat-engines.md)
12. [第 12 章 Workflow 事件驱动编排引擎](./llama-index/ch12-workflow-orchestration.md)
13. [第 13 章 Agent 系统](./llama-index/ch13-agent-system.md)
14. [第 14 章 可观测性与工程原则收尾](./llama-index/ch14-instrumentation-and-engineering-principles.md)

## 第四部 Quivr

第四个解构对象是 [Quivr](https://github.com/QuivrHQ/quivr)——准确说是它的核心库 `quivr-core`（Quivr.com 的"大脑"，Python 3.10+）。与前三部都不同，它是一个把 **LangChain + LangGraph 当地基**、自己只写薄薄一层"编排与策略"胶水的库。它最鲜明的一招，是把 RAG 流程做成**配置驱动的 LangGraph**：图的拓扑（节点、边、条件边）不是写死的函数调用链，而是运行时从一组 `NodeConfig` 用 `getattr(self, node.name)` 拼装出来的——同一套引擎代码，换一组节点配置就能从标准 RAG 变成带 web search 的 agent、变成 zendesk 客服助手。章节弧线沿数据流由外到内展开：从 `Brain` 总入口，到文件存储、解析切分、模型与向量库两大外部能力，最后是配置驱动的 LangGraph 把检索、重排、工具、生成、流式输出编排成一条可按场景重组的问答流水线：

| 章 | 主题 | 解构焦点 |
|---|---|---|
| 第 1 章 | 库定位与 Brain 全景 | `quivr-core` 定位、`Brain` 总指挥、`afrom_files` 默认装配、零配置即可用 |
| 第 2 章 | 文件抽象与 Storage 存储层 | `QuivrFile` `__slots__`/sha1 指纹、`StorageBase` ABC、Transparent/Local 双实现 |
| 第 3 章 | Processor 注册表与多格式解析 | `MappingProxyType` 只读注册表、优先级堆、惰性导入、`ProcessorBase` 模板方法 |
| 第 4 章 | 切分策略与 SplitterConfig | `SplitterConfig` 默认值、手写递归字符切分、Tika/tiktoken 切分路径 |
| 第 5 章 | LLMEndpoint 与多供应商模型配置 | `LLMEndpoint.from_config` 工厂、`SecretStr` 守卫、`LLMTokenizer` LRU 缓存、能力探测 |
| 第 6 章 | 嵌入与向量库 | `default_embedder`、`build_default_vectordb`、FAISS 默认、`save`/`load` 守卫 |
| 第 7 章 | 配置驱动的 LangGraph 工作流 | `NodeConfig`/`WorkflowConfig`、`_build_workflow` 反射建图、`final_nodes`、首节点校验 |
| 第 8 章 | RAG 节点详解 | `AgentState` 契约、`filter_history`/`rewrite`/`retrieve`/`generate` 节点、增量状态更新 |
| 第 9 章 | 重排与上下文压缩 | `get_reranker` 回退 `IdempotentCompressor`、`ContextualCompressionRetriever`、`dynamic_retrieve` 自适应 |
| 第 10 章 | 工具路由与 Agent | `tool_routing` + `Send` 分派、`LLMToolFactory`、`ToolWrapper` 契约、模糊匹配纠错 |
| 第 11 章 | 流式回答、序列化与工程原则收尾 | `answer_astream` 订阅 `astream_events`、`reduce_rag_context` 预算退让、`BrainSerialized` 判别式联合、跨四部书的七条可迁移原则 |

Quivr 各章节文件位于 [`quivr/`](./quivr/)：

1. [第 1 章 库定位与 Brain 全景](./quivr/ch01-overview-and-brain.md)
2. [第 2 章 文件抽象与 Storage 存储层](./quivr/ch02-file-and-storage.md)
3. [第 3 章 Processor 注册表与多格式解析](./quivr/ch03-processor-registry.md)
4. [第 4 章 切分策略与 SplitterConfig](./quivr/ch04-splitter-config.md)
5. [第 5 章 LLMEndpoint 与多供应商模型配置](./quivr/ch05-llm-endpoint.md)
6. [第 6 章 嵌入与向量库](./quivr/ch06-embedding-and-vectordb.md)
7. [第 7 章 配置驱动的 LangGraph 工作流](./quivr/ch07-config-driven-langgraph.md)
8. [第 8 章 RAG 节点详解](./quivr/ch08-rag-nodes.md)
9. [第 9 章 重排与上下文压缩](./quivr/ch09-rerank-and-compression.md)
10. [第 10 章 工具路由与 Agent](./quivr/ch10-tool-routing-agent.md)
11. [第 11 章 流式回答、序列化与工程原则收尾](./quivr/ch11-streaming-serialization-and-principles.md)

## 第五部 Langchain-Chatchat

第五个解构对象是 [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)——一个**面向中文、可离线部署的 RAG 与 Agent 应用**。与前四部不同，它的取舍处处指向"开箱即用"与"吃掉生态碎片"：从默认中文模型与一键 `init` 部署，到用 `KBService` 一层抽象统一七种向量库、用 `MODEL_PLATFORMS` 统一四类推理后端、用 `agents_registry` 的 `agent_type` 统一多模型族，它反复用一层薄契约把异构的 LangChain 生态收口。技术栈是 Python + FastAPI + Streamlit 多进程，深度复用 LangChain。章节弧线沿数据流由外到内展开：从多进程启动与分层配置，到中文 loader/splitter，到知识库服务与向量缓存，再到 RAG 对话、Agent 注册表、工具系统与 MCP，最后是统一的 API 与模型接入层：

| 章 | 主题 | 解构焦点 |
|---|---|---|
| 第 1 章 | 项目全景与多进程启动 | monorepo/libs 结构、`spawn` + `Manager().Event()`、FastAPI/Streamlit 双进程、Click CLI |
| 第 2 章 | 配置系统 Settings | `pydantic-settings` 分层、yaml 持久化与热重载、`MODEL_PLATFORMS`、默认值取向 |
| 第 3 章 | 文档加载器 | `LOADER_DICT` 扩展名路由、RapidOCR loaders、`get_ocr` paddle→onnx 回退、统一产出 `Document` |
| 第 4 章 | 中文文本切分 | `ChineseRecursiveTextSplitter` 标点感知、`zh_title_enhance`、`make_text_splitter` 工厂 |
| 第 5 章 | 知识库服务抽象层 | `KBService` 模板方法、`SupportedVSType` + 工厂、七种向量库同一契约 |
| 第 6 章 | 向量缓存与嵌入 | `ThreadSafeObject`/`CachePool` LRU、`KBFaissPool` 双层锁、多后端 embedding 批处理 |
| 第 7 章 | 知识库文档管理与迁移 | 向量库+SQLite 双写一致性、`folder2db`/`prune`、`with_session` 事务、多线程切分 |
| 第 8 章 | 检索器与重排 | `get_Retriever` 工厂、vectorstore/ensemble(BM25+向量)/milvus、CrossEncoder 重排 |
| 第 9 章 | RAG 对话链路 | `kb_chat` 检索→拼 context→流式生成、`return_direct` 逃生通道、异步流式三件套 |
| 第 10 章 | Agent 注册表与多模型适配 | `agents_registry` 按 `agent_type` 分发、`PlatformToolsAgentExecutor`、模型族 workaround |
| 第 11 章 | 工具系统 Tools Factory | `regist_tool` 自动注册、`BaseTool` monkeypatch、`return_direct` 工具、`BaseToolOutput` |
| 第 12 章 | MCP 集成 | `MultiServerMCPClient`、`MCPStructuredTool` 契约转换、连接持久化、与 agent 打通线 |
| 第 13 章 | API 服务与模型接入 | `create_app` 路由组装、OpenAI 兼容层、`MODEL_PLATFORMS` 统一异构推理后端 |
| 第 14 章 | 工程原则收尾与全书结语 | Chatchat 设计取向、跨五部书归纳的七条可迁移工程原则、全书结语 |

Langchain-Chatchat 各章节文件位于 [`langchain-chatchat/`](./langchain-chatchat/)：

1. [第 1 章 项目全景与多进程启动](./langchain-chatchat/ch01-overview-and-startup.md)
2. [第 2 章 配置系统 Settings](./langchain-chatchat/ch02-settings-configuration.md)
3. [第 3 章 文档加载器](./langchain-chatchat/ch03-document-loaders.md)
4. [第 4 章 中文文本切分](./langchain-chatchat/ch04-chinese-text-splitting.md)
5. [第 5 章 知识库服务抽象层](./langchain-chatchat/ch05-kb-service-abstraction.md)
6. [第 6 章 向量缓存与嵌入](./langchain-chatchat/ch06-vector-cache-and-embeddings.md)
7. [第 7 章 知识库文档管理与迁移](./langchain-chatchat/ch07-doc-management-and-migration.md)
8. [第 8 章 检索器与重排](./langchain-chatchat/ch08-retrievers-and-rerank.md)
9. [第 9 章 RAG 对话链路](./langchain-chatchat/ch09-rag-chat-pipeline.md)
10. [第 10 章 Agent 注册表与多模型适配](./langchain-chatchat/ch10-agent-registry.md)
11. [第 11 章 工具系统 Tools Factory](./langchain-chatchat/ch11-tools-factory.md)
12. [第 12 章 MCP 集成](./langchain-chatchat/ch12-mcp-integration.md)
13. [第 13 章 API 服务与模型接入](./langchain-chatchat/ch13-api-server-and-model-access.md)
14. [第 14 章 工程原则收尾与全书结语](./langchain-chatchat/ch14-engineering-principles-and-epilogue.md)

## 第六部 LightRAG

第六个解构对象是 [LightRAG](https://github.com/HKUDS/LightRAG)——HKUDS 出品、强调"简单且快"的 **graph-based RAG** 系统（pip `lightrag-hku`，Python 3.10，论文 arXiv 2410.05779）。与前五部最根本的分野，是它把**知识图谱放在系统正中心**：用 LLM 把每个 chunk 抽成实体与关系三元组、合并成一张图，检索则是图上的 low-level（具体实体）与 high-level（抽象关系）双层投影，默认 `mix` 模式再融合朴素向量召回。它还有两处鲜明取舍：把全部持久化收敛成四个存储 ABC、十余种后端可插拔，且明确"知识图谱 + `text_chunks` KV 是权威源、三个向量库只是可重算的派生物"；以及用一层 `shared_storage` 横切，让同一套代码既能单进程 asyncio 跑、又能多 worker `lightrag-gunicorn` 安全运行。章节弧线由外到内、由数据摄入到检索生成展开：

| 章 | 主题 | 解构焦点 |
|---|---|---|
| 第 1 章 | 项目全景与 LightRAG 类 | `LightRAG` dataclass 总入口、monorepo、版本单点、三 Mixin、存储生命周期 |
| 第 2 章 | 存储抽象与四类基类 | `base.py` 四个 ABC、`StorageNameSpace`、`index_done_callback` 延迟落盘契约、`QueryParam`/`DocStatus` |
| 第 3 章 | 文本切分 Chunker | token 切分、段落语义切分、chunk 元数据 |
| 第 4 章 | 文档解析 Parser | `parser/` 路由、docx/markdown、docling/mineru 外部解析器 |
| 第 5 章 | 实体关系抽取 | `operate.py` `extract_entities`、tuple 分隔符约定、跨 chunk 合并 `_merge_nodes/edges_then_upsert` |
| 第 6 章 | 双层检索与查询模式 | `kg_query`、low/high-level 关键词、六种 mode、向量+图融合 |
| 第 7 章 | 可插拔存储后端 | `STORAGE_IMPLEMENTATIONS` 花名册、`get_storage_class` 工厂、惰性导入、批操作 override |
| 第 8 章 | 共享存储与并发控制 | `shared_storage.py`、`UnifiedLock`/`KeyedUnifiedLock`、update flags、全局并发租约 |
| 第 9 章 | 文档处理管线 Pipeline | 三段级联队列、enqueue/process 分离、断点续跑、失败隔离、双写一致 |
| 第 10 章 | LLM 接入与多供应商绑定 | 函数式契约、`openai_complete_if_cache`、`binding_options` 配置即数据、rerank、多模态 |
| 第 11 章 | API 服务与路由 | `create_app` 工厂、lifespan、鉴权、文档/查询/图谱路由、Ollama 兼容层、多 worker |
| 第 12 章 | 运维工具与缓存迁移 | `tools/` 缓存迁移、`rebuild_vdb` 权威源重建、可视化、RAGAS 评估 |
| 第 13 章 | 工程原则收尾与全书结语 | LightRAG 设计取向、跨六部书归纳的七条可迁移工程原则、全书结语 |

LightRAG 各章节文件位于 [`lightrag/`](./lightrag/)：

1. [第 1 章 项目全景与 LightRAG 类](./lightrag/ch01-overview-and-lightrag-class.md)
2. [第 2 章 存储抽象与四类基类](./lightrag/ch02-storage-abstraction-base.md)
3. [第 3 章 文本切分 Chunker](./lightrag/ch03-chunking.md)
4. [第 4 章 文档解析 Parser](./lightrag/ch04-document-parsing.md)
5. [第 5 章 实体关系抽取](./lightrag/ch05-entity-relation-extraction.md)
6. [第 6 章 双层检索与查询模式](./lightrag/ch06-dual-level-retrieval.md)
7. [第 7 章 可插拔存储后端](./lightrag/ch07-storage-backends.md)
8. [第 8 章 共享存储与并发控制](./lightrag/ch08-shared-storage-concurrency.md)
9. [第 9 章 文档处理管线 Pipeline](./lightrag/ch09-pipeline.md)
10. [第 10 章 LLM 接入与多供应商绑定](./lightrag/ch10-llm-bindings.md)
11. [第 11 章 API 服务与路由](./lightrag/ch11-api-server.md)
12. [第 12 章 运维工具与缓存迁移](./lightrag/ch12-tools-and-maintenance.md)
13. [第 13 章 工程原则收尾与全书结语](./lightrag/ch13-engineering-principles.md)

## 第七部 GraphRAG

第七个解构对象是 [GraphRAG](https://github.com/microsoft/graphrag)——微软出品、版本 3.1.0 的工业级 **graph-based RAG** 引擎。它和第六部 LightRAG 同属"以知识图谱为核心"的路线,却把图谱 RAG 做到了另一个工程量级:用一个八包的 uv-workspace monorepo 把切分、输入、LLM、缓存、存储、向量库都拆成可独立 `pip install` 的子包;索引侧把"文档 → text unit → 实体关系抽取 → 图谱终化与剪枝 → Leiden 层次社区检测 → 社区报告 → 向量嵌入"组织成一条由 `PipelineFactory` 注册、可重排、可观测的 workflow 流水线;查询侧则围绕同一份 parquet 图谱产物给出 local(细节)、global(map-reduce 概览)、drift(全局起步+局部深挖)、basic(纯向量基线)四种检索投影。它最鲜明的取舍:把横切关注点(限流/重试/缓存/metrics)收口进 `graphrag-llm` 的可组合中间件管线、把 LLM 统一压在 LiteLLM 单一底座上,并把 parquet 表作为权威源、向量索引作为可重算的派生物。章节弧线沿数据流由外到内、由索引到查询展开:

| 章 | 主题 | 解构焦点 |
|---|---|---|
| 第 1 章 | 项目全景与 monorepo 架构 | 八包 uv-workspace、子包解耦、版本单点、索引/查询两侧分野 |
| 第 2 章 | 配置系统 GraphRagConfig | `load_config`、Pydantic models、`defaults`、env 插值、`init` 脚手架 |
| 第 3 章 | 输入加载 Input | `graphrag-input` 包、csv/json/jsonl reader、factory、文档哈希 |
| 第 4 章 | 文本切分 Chunking | `graphrag-chunking` 包、token/sentence chunker、注入式编解码、字符偏移 |
| 第 5 章 | 图谱抽取 Extract Graph | LLM 实体/关系抽取、分隔符约定、gleaning 多轮补抽、跨片段合并溯源 |
| 第 6 章 | 描述摘要与图谱终化 | `summarize_descriptions` 滚动归并、`finalize_graph` 补字段、`prune_graph` 剪枝 |
| 第 7 章 | 社区检测 Leiden | `create_communities`、graspologic `hierarchical_leiden`、层次社区树、确定性 |
| 第 8 章 | 社区报告生成 | 自底向上层级生成、子社区报告替代机制、结构化 JSON 输出、text 变体 |
| 第 9 章 | 向量嵌入与协变量 | `generate_text_embeddings`、`graphrag-vectors` 包、LanceDB、`extract_covariates` claims |
| 第 10 章 | 索引流水线编排 | `PipelineFactory`、`run_pipeline`、统一 `run_workflow` 契约、`IndexingMethod`、增量更新 |
| 第 11 章 | Local Search 局部检索 | `LocalSearchMixedContext`、种子实体映射、混合上下文与 token 预算分配 |
| 第 12 章 | Global Search 全局检索 | `GlobalCommunityContext`、map-reduce、key points 评分、动态社区选择 |
| 第 13 章 | DRIFT 与 Basic Search | primer/action/state 推理树、纯向量基线、`api/query` 四模式统一收口 |
| 第 14 章 | LLM/缓存/存储与工程原则收尾 | `graphrag-llm` 中间件管线、`graphrag-cache`/`graphrag-storage` 三层解耦、`prompt_tune`、跨七部书工程原则与全书结语 |

GraphRAG 各章节文件位于 [`graphrag-ms/`](./graphrag-ms/)：

1. [第 1 章 项目全景与 monorepo 架构](./graphrag-ms/ch01-overview-and-monorepo.md)
2. [第 2 章 配置系统 GraphRagConfig](./graphrag-ms/ch02-configuration.md)
3. [第 3 章 输入加载 Input](./graphrag-ms/ch03-input-loading.md)
4. [第 4 章 文本切分 Chunking](./graphrag-ms/ch04-chunking.md)
5. [第 5 章 图谱抽取 Extract Graph](./graphrag-ms/ch05-extract-graph.md)
6. [第 6 章 描述摘要与图谱终化](./graphrag-ms/ch06-summarize-finalize.md)
7. [第 7 章 社区检测 Leiden](./graphrag-ms/ch07-community-detection.md)
8. [第 8 章 社区报告生成](./graphrag-ms/ch08-community-reports.md)
9. [第 9 章 向量嵌入与协变量](./graphrag-ms/ch09-embeddings-covariates.md)
10. [第 10 章 索引流水线编排](./graphrag-ms/ch10-index-pipeline.md)
11. [第 11 章 Local Search 局部检索](./graphrag-ms/ch11-local-search.md)
12. [第 12 章 Global Search 全局检索](./graphrag-ms/ch12-global-search.md)
13. [第 13 章 DRIFT 与 Basic Search](./graphrag-ms/ch13-drift-basic-search.md)
14. [第 14 章 LLM/缓存/存储与工程原则收尾](./graphrag-ms/ch14-infrastructure-and-principles.md)

## 第八部 Haystack

第八个解构对象是 [Haystack](https://github.com/deepset-ai/haystack)——deepset 出品、版本 2.31.0-rc0 的可组合 LLM 编排框架。如果说前七部各自代表一种"做完整 RAG 产品"的路线，Haystack 走的是另一条路：它不预设某种 RAG 形态，而是把一切都做成可拼装的乐高积木——每个能力都是一个声明了类型化输入输出端口、实现 `run()` 返回 dict、可 `to_dict`/`from_dict` 序列化的 `@component`；组件按数据契约连成一张可序列化的有向图 `Pipeline`；调度器用优先级队列数据驱动地决定谁先跑；`DocumentStore` 用 Protocol 而非继承定义存储契约；工具、Agent、SuperComponent 在同一套契约上层层复合。它最鲜明的取舍：用统一的最小数据契约（`Document`/`ChatMessage`）换可组合性，用轻量 `__init__` + 重量 `warm_up()` 分离配置与加载，把编排从代码抽成可存盘、可断点续跑的数据，并把追踪、遥测、评估这些横切关注点织进基座。章节弧线沿"契约 → 数据模型 → 流水线构建与执行 → 组件家族 → Agent → 评估收尾"由内核到外围展开：

| 章 | 主题 | 解构焦点 |
|---|---|---|
| 第 1 章 | 框架全景与 @component 契约 | 乐高式编排哲学、`@component` 装饰器、输入输出 socket、`run()` dict 契约 |
| 第 2 章 | 核心数据模型 Dataclasses | `Document`/`ChatMessage`/`Answer`/`ByteStream`、统一最小数据契约、序列化 |
| 第 3 章 | Pipeline 图构建与连接 | `MultiDiGraph`、`add_component`/`connect`、socket 类型兼容校验 |
| 第 4 章 | Pipeline 执行引擎与调度 | `_run_component`、`component_checks`、`FIFOPriorityQueue`、`ComponentPriority.BLOCKED` |
| 第 5 章 | 序列化、YAML 与断点 | `component_to_dict`/`from_dict`、`import_class_by_name`、`YamlMarshaller`、snapshot 断点 |
| 第 6 章 | 文档转换与预处理切分 | converters 统一契约、`DocumentSplitter`/`Recursive`/`Hierarchical`、`DocumentCleaner` |
| 第 7 章 | DocumentStore 抽象与内存实现 | `DocumentStore` Protocol、`DuplicatePolicy`、filter DSL、InMemory BM25（BM25L 默认） |
| 第 8 章 | 嵌入器 Embedders | Text vs Document 嵌入器、`warm_up`、backend 缓存、`meta_fields_to_embed` |
| 第 9 章 | 检索器与重排器 | 检索器作为 store 包装、AutoMerging/SentenceWindow、cross-encoder、RRF（k=61） |
| 第 10 章 | 生成器与 Prompt 构建 | `OpenAIChatGenerator`、流式、工具调用、`ChatPromptBuilder` Jinja2 沙箱、`AnswerBuilder` |
| 第 11 章 | 路由、汇合与分支控制流 | `ConditionalRouter`、`DocumentJoiner`（RRF/DBSF）、`BranchJoiner`、`JsonSchemaValidator` |
| 第 12 章 | 工具系统 Tool 与 Toolset | `Tool` dataclass、`ComponentTool`/`PipelineTool`、`Toolset`、`State`（merge/replace） |
| 第 13 章 | Agent 与 SuperComponent | Agent think-act 循环、`exit_conditions`、`State`、SuperComponent socket 重映射 |
| 第 14 章 | 评估、可观测性与工程原则收尾 | `EvaluationRunResult`、`LLMEvaluator`/Faithfulness、`Tracer`/`ProxyTracer` 代理、隐私默认关闭、PostHog 遥测、跨八部书工程原则与全书结语 |

Haystack 各章节文件位于 [`haystack/`](./haystack/)：

1. [第 1 章 框架全景与 @component 契约](./haystack/ch01-overview-and-component.md)
2. [第 2 章 核心数据模型 Dataclasses](./haystack/ch02-dataclasses.md)
3. [第 3 章 Pipeline 图构建与连接](./haystack/ch03-pipeline-build.md)
4. [第 4 章 Pipeline 执行引擎与调度](./haystack/ch04-pipeline-execution.md)
5. [第 5 章 序列化、YAML 与断点](./haystack/ch05-serialization-breakpoints.md)
6. [第 6 章 文档转换与预处理切分](./haystack/ch06-converters-preprocessors.md)
7. [第 7 章 DocumentStore 抽象与内存实现](./haystack/ch07-document-store.md)
8. [第 8 章 嵌入器 Embedders](./haystack/ch08-embedders.md)
9. [第 9 章 检索器与重排器](./haystack/ch09-retrievers-rankers.md)
10. [第 10 章 生成器与 Prompt 构建](./haystack/ch10-generators-builders.md)
11. [第 11 章 路由、汇合与分支控制流](./haystack/ch11-routers-joiners.md)
12. [第 12 章 工具系统 Tool 与 Toolset](./haystack/ch12-tools.md)
13. [第 13 章 Agent 与 SuperComponent](./haystack/ch13-agent-supercomponent.md)
14. [第 14 章 评估、可观测性与工程原则收尾](./haystack/ch14-evaluation-observability-principles.md)

## 第九部 RAG-Anything

第九个解构对象是 [RAG-Anything](https://github.com/HKUDS/RAG-Anything)——HKUDS 出品、版本 1.3.1 的多模态 RAG 框架。它和前八部最大的不同在于：它不重新发明知识图谱，而是**站在第六部 LightRAG 的肩上**，把 RAG 的能力边界从纯文本扩展到图片、表格、公式、Office 文档等多模态内容。核心类 `RAGAnything` 是一个用 Mixin 组合而成的 `@dataclass`（`class RAGAnything(QueryMixin, ProcessorMixin, BatchMixin)`），整条流水线是：多模态文档经可插拔解析器（MinerU/Docling/PaddleOCR）解析成统一的 `content_list` 数据契约，分流为纯文本（送进 LightRAG 插入）与多模态项（交给图像/表格/公式/通用四件套处理器，生成描述并作为实体插入知识图谱）。它最鲜明的工程取舍：用 `content_list` 这一最小数据契约抹平不同解析器的差异，用注入式的 `llm_model_func`/`vision_model_func`/`embedding_func` 解耦模型选型，用 `inspect.signature` 反射与多级 JSON 回退应对版本漂移与不可靠的 LLM 输出，用重试/断路器/回调把韧性与可观测性织进基座。章节弧线沿"主类与配置 → 解析器 → 处理流水线 → 多模态处理器 → 查询 → 批处理 → 韧性 → 提示收尾"由内核向外围展开：

| 章 | 主题 | 解构焦点 |
|---|---|---|
| 第 1 章 | 项目全景与 RAGAnything 主类 | Mixin 组合 `@dataclass`、站在 LightRAG 之上、`_ensure_lightrag_initialized`、生命周期与 `close` |
| 第 2 章 | 配置系统 RAGAnythingConfig | 环境变量驱动 `field(default=get_env_value(...))`、多模态开关、`__post_init__` 兼容旧字段 |
| 第 3 章 | 解析器抽象与 Parser 基类 | `Parser` 基类、`content_list` 数据契约、Office/文本→PDF、`_download_file`/`_unique_output_dir`、LibreOffice |
| 第 4 章 | MinerU/Docling 解析器与插件注册 | MinerU 子进程 CLI vs Docling Python API、Windows 安全路径 MD5、`_CUSTOM_PARSERS` 注册表 |
| 第 5 章 | 公式抽取与增强 Markdown | OMML→LaTeX AST 访问者 `_HANDLERS`、`enrich_content_list_with_docx_equations`、多后端优雅降级 |
| 第 6 章 | 文档处理流水线 ProcessorMixin | `process_document_complete` 七步、解析缓存、内容寻址 doc_id、`_convert_to_lightrag_chunks_type_aware` |
| 第 7 章 | 上下文提取 ContextExtractor | `ContextConfig`/`ContextExtractor`、page vs chunk 模式、token 截断、三种 content_source 适配 |
| 第 8 章 | 多模态处理器四件套 | `BaseModalProcessor` 模板方法、4 级 JSON 回退、图像/表格/公式/通用、`_create_entity_and_chunk` 入图 |
| 第 9 章 | 内容工具与文本插入 | `separate_content`、章节栈 `extract_section_path`、`validate_image_file`、`inspect.signature` 版本兼容 |
| 第 10 章 | 查询系统 QueryMixin | `query`/`aquery`（mode=mix 委托 LightRAG）、`query_with_multimodal`、`[VLM_IMAGE_n]` 占位、缓存键 |
| 第 11 章 | 批处理 BatchMixin 与 BatchParser | asyncio.Semaphore vs ThreadPoolExecutor 两种并发、`BatchProcessingResult`、dry_run 预览、双层容错 |
| 第 12 章 | 韧性与可观测性回调 | `retry`/`async_retry` 指数退避、`CircuitBreaker` 三态机、`ProcessingEvent`/`CallbackManager`/`MetricsCallback` |
| 第 13 章 | 多语言提示与工程原则收尾 | `PromptRegistry` 原子快照、`set_prompt_language` 惰性加载与回退、纯数据语言包、跨九部工程原则与全书结语 |

RAG-Anything 各章节文件位于 [`raganything/`](./raganything/)：

1. [第 1 章 项目全景与 RAGAnything 主类](./raganything/ch01-overview-and-raganything.md)
2. [第 2 章 配置系统 RAGAnythingConfig](./raganything/ch02-configuration.md)
3. [第 3 章 解析器抽象与 Parser 基类](./raganything/ch03-parser-abstraction.md)
4. [第 4 章 MinerU/Docling 解析器与插件注册](./raganything/ch04-parsers-and-plugins.md)
5. [第 5 章 公式抽取与增强 Markdown](./raganything/ch05-equations-and-markdown.md)
6. [第 6 章 文档处理流水线 ProcessorMixin](./raganything/ch06-processor-pipeline.md)
7. [第 7 章 上下文提取 ContextExtractor](./raganything/ch07-context-extractor.md)
8. [第 8 章 多模态处理器四件套](./raganything/ch08-modal-processors.md)
9. [第 9 章 内容工具与文本插入](./raganything/ch09-content-utils.md)
10. [第 10 章 查询系统 QueryMixin](./raganything/ch10-query-mixin.md)
11. [第 11 章 批处理 BatchMixin 与 BatchParser](./raganything/ch11-batch-processing.md)
12. [第 12 章 韧性与可观测性回调](./raganything/ch12-resilience-callbacks.md)
13. [第 13 章 多语言提示与工程原则收尾](./raganything/ch13-prompts-and-principles.md)
