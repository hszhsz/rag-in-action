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
