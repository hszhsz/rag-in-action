# 第 1 章 框架全景与核心数据模型:一切从 Node 开始

如果说前两部解构的 RAGFlow 与 AnythingLLM 都是"开箱即用的产品"——你装上、配好、就能对话,那么这一部的主角 LlamaIndex 是另一种东西:它是一个**框架**。它不替你做决定,而是给你一套积木和组装规则,让你按需拼出自己的 RAG/Agent 系统。这种定位差异,决定了我们解构它的方式也要变——不再追"一个请求怎么走完全程",而是追"这套积木的形状是怎么定义的,它们如何严丝合缝地拼在一起"。

而这一切积木的"公共接口",就是它的数据模型。在 LlamaIndex 里,无论你加载的是 PDF、网页还是数据库行,最终都会被规整成同一种东西——`Node`。读懂了 `Node` 及其相关的几个基类,你就握住了理解整个框架的钥匙:后面所有的 reader、parser、index、retriever,本质上都是在生产、变换或消费 `Node`。本章我们就从仓库的物理结构出发,一路下钻到 `schema.py` 这个定义了全部核心数据结构的文件。

## 一个 monorepo,三层包

打开仓库根目录,会看到 LlamaIndex 把自己拆成了几个并列的顶层目录:`llama-index-core`、`llama-index-integrations`、`llama-index-instrumentation`、`llama-index-utils`,外加开发工具 `llama-dev`。根 `pyproject.toml` 里 `name = "llama-index"`、`version = "0.14.22"`、`namespace_packages = true`——最后这个 `namespace_packages` 是关键信号:它声明 `llama_index` 是一个**命名空间包**,允许把同一个顶层包名拆散到几十个独立分发的子包里。

这种拆分背后是一个明确的工程取舍:**核心抽象与具体集成解耦**。`llama-index-core` 只放与任何外部服务无关的抽象基类和默认实现;而每一个具体的 LLM provider、向量库、reader,都被切成 `llama-index-integrations` 下一个能独立安装的小包。你想用 OpenAI 就 `pip install llama-index-llms-openai`,不想用就不装,核心包不会被几百个第三方 SDK 拖累。这与 AnythingLLM 把所有 provider 塞进一个 server 进程的做法形成鲜明对照——后者是产品,追求开箱即用;前者是框架,追求可插拔与依赖最小化。

我们这一部聚焦 `llama-index-core/llama_index/core` 目录,因为框架的灵魂全在这里。`ls` 一下,你会看到一张 RAG 流水线的"零件目录":`readers`(加载)、`node_parser`(切分)、`ingestion`(摄入管线)、`indices`(索引)、`embeddings`(嵌入)、`storage` 与 `vector_stores`(存储)、`retrievers`(检索)、`response_synthesizers`(响应合成)、`query_engine` 与 `chat_engine`(查询/对话)、`agent` 与 `workflow`(智能体与编排)、`instrumentation`(可观测性)。本书第二部的章节弧线,基本就是沿这条数据流从上往下走一遍。而所有这些零件咬合的接口,定义在根目录下那个 1492 行的 `schema.py` 里。

## BaseComponent:可序列化是与生俱来的能力

`schema.py` 的第一个基类是 `BaseComponent`,它继承自 pydantic 的 `BaseModel`。这个选择本身就是一个设计宣言:框架里几乎所有对象都是 pydantic 模型,意味着它们天生具备字段校验、JSON 序列化、schema 自省的能力。

`BaseComponent` 在 pydantic 之上又加了一层"身份与持久化"约定。它定义了一个 `class_name()` 类方法,默认返回 `"base_component"`,子类各自覆写——这是一个**类型标签**,序列化时会被写进 `to_dict()` 的输出里(`data["class_name"] = self.class_name()`),反序列化时据此还原成正确的子类。它还实现了 `to_dict`/`to_json`/`from_dict`/`from_json` 这组对称方法。值得注意的是它对 pickle 的处理——`__getstate__` 里会主动剔除无法序列化的字段(比如带 `private attributes` 的部分),并对老版本 pydantic 的兼容做了细致处理。一句话:在 LlamaIndex 里,"能被存下来、再读回来"不是某个模块的特性,而是从最底层基类就保证的全局契约。这条契约支撑了后面所有 docstore、index_store 的持久化能力。

紧挨着的是 `TransformComponent`,它同时继承 `BaseComponent` 和 `DispatcherSpanMixin`(后者埋下了可观测性的钩子,留到第 14 章再说)。它的核心只有一个抽象方法:

```python
@abstractmethod
def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
    """Transform nodes."""

async def acall(self, nodes, **kwargs):
    return self.__call__(nodes, **kwargs)
```

请记住这个签名:**输入一串 `BaseNode`,输出一串 `BaseNode`**。这是整个框架最重要的"形状"。节点解析器、元数据抽取器、嵌入器……几乎所有处理步骤都是 `TransformComponent`,都遵守"node 进、node 出"的统一接口。正因如此,第 5 章的 `IngestionPipeline` 才能把它们像乐高一样串成一条链——因为每一节的接口都一模一样。这是一个教科书级的"统一变换接口"设计。

## BaseNode:可被检索的最小单元

终于来到主角。`BaseNode` 继承 `BaseComponent`,它的 `model_config` 设了 `populate_by_name=True` 和 `validate_assignment=True`——后者意味着**每次给字段赋值都会重新校验**,这对保证 node 始终处于合法状态很关键。

它的字段定义把一个"可被检索的知识单元"需要什么,讲得清清楚楚:

- `id_`:默认用 `uuid.uuid4()` 生成的全局唯一 ID。
- `embedding`:这个节点的向量表示,默认 `None`(在被嵌入之前)。
- `metadata`:一个扁平字典。源码注释特意强调了它的三重用途——"作为展示给 LLM 的上下文文本的一部分注入;作为生成 embedding 的文本的一部分注入;被向量库用于 metadata 过滤"。注意它有个 `alias="extra_info"`,这是为了向后兼容老 API。
- `excluded_embed_metadata_keys` 与 `excluded_llm_metadata_keys`:两份"黑名单",分别控制哪些 metadata 字段**不**进入嵌入文本、哪些**不**进入给 LLM 的文本。
- `relationships`:一个从 `NodeRelationship` 到 `RelatedNodeInfo` 的映射。

最后这个 `relationships` 字段,是 LlamaIndex 区别于"一堆扁平 chunk"的精髓。`NodeRelationship` 是个枚举,定义了五种关系:`SOURCE`(指向源文档)、`PREVIOUS`/`NEXT`(同文档内的前后节点)、`PARENT`/`CHILD`(父子层级)。这意味着切碎后的 node 并不是孤立的碎片,而是一张**带方向的图**:每个 chunk 都知道自己来自哪篇文档、前后邻居是谁、上层归属是谁。`BaseNode` 为此提供了 `source_node`、`prev_node`、`next_node`、`parent_node`、`child_nodes` 一组 property,从 `relationships` 字典里安全地取出对应关系——而且取的时候带类型校验,比如 `child_nodes` 要求关系值必须是 list,否则直接 `raise ValueError`。第 9 章的 `AutoMergingRetriever`(命中子节点时自动上溯合并到父节点)、recursive retriever,全都建立在这张关系图之上。

### metadata 的三种"视图":一份数据,因受众而异

`BaseNode` 里有个极其精巧的设计,体现在 `MetadataMode` 枚举上,它有四个值:`ALL`、`EMBED`、`LLM`、`NONE`。配合 `get_metadata_str(mode)` 方法,同一个节点的 metadata 会根据"谁来看"而呈现不同内容:

```python
usable_metadata_keys = set(self.metadata.keys())
if mode == MetadataMode.LLM:
    for key in self.excluded_llm_metadata_keys:
        usable_metadata_keys.discard(key)
elif mode == MetadataMode.EMBED:
    for key in self.excluded_embed_metadata_keys:
        usable_metadata_keys.discard(key)
```

为什么要这么设计?因为同一份 metadata,对不同消费者价值不同。比如一个 `file_name` 字段,你可能希望它参与嵌入(便于按文件名语义检索),但不希望它出现在喂给 LLM 的上下文里(浪费 token 还可能干扰)。`EMBED` 模式生成嵌入文本时排除一份黑名单,`LLM` 模式生成提示文本时排除另一份。这种"一份数据、多种视图"的设计,让用户能精细地控制每个字段在 RAG 链路不同环节的可见性——这正是第 14 章会归纳的"留结构去内容"原则的一个变体:结构(metadata 字典)始终完整,内容(对特定受众的可见字段)按需裁剪。

## TextNode 与 Document:具体的两端

`BaseNode` 是抽象的,真正被实例化的是它的子类。最常用的是 `TextNode`,源码里它的注释写着 "Provided for backward compatibility"——它在新版里其实是 `Node` 的兼容形态,但仍是事实上的工作主力。它补上了 `text`(文本内容)、`mimetype`(默认 `"text/plain"`)、`start_char_idx`/`end_char_idx`(这个 chunk 在原文里的字符起止位置)等字段。

`TextNode` 的 `hash` property 很值得一看:

```python
@property
def hash(self) -> str:
    doc_identity = str(self.text) + str(self.metadata)
    return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())
```

它把文本和 metadata 拼起来算 SHA-256。这个 hash 不是装饰——它是第 5 章摄入管线做**去重与增量更新**的依据:内容没变,hash 就不变,管线就能跳过重复嵌入。`get_content(metadata_mode)` 则把 metadata 字符串按 `text_template` 模板和正文拼成最终内容,呼应了上一节讲的"视图"机制。

`Document` 是 `TextNode` 的另一端——它代表"一整篇尚未切分的原始文档"(`get_type()` 返回 `ObjectType.DOCUMENT`)。reader 读进来的是 `Document`,parser 切出来的是一堆 `TextNode`,二者通过 `relationships` 里的 `SOURCE` 关系连接。还有 `ImageNode`(继承 `TextNode`,加上 `image`/`image_path`/`image_url`,并在构造时按扩展名猜 mimetype)和 `IndexNode`(指向另一个对象或索引,是 recursive retrieval 的基石),共同构成了多模态与组合检索的类型基础。

`as_related_node_info()` 这个方法把任意 node 转成轻量的 `RelatedNodeInfo`(只含 `node_id`、`node_type`、`metadata`、`hash`)——这就是关系图里"指针"的实现:节点之间互相引用时,存的不是整个对象,而是这个轻量指针。

## 数据流的全景:node 的一生

把这些拼起来,LlamaIndex 处理数据的主线就清晰了:Reader 把外部数据读成 `Document`(第 3 章)→ NodeParser 这个 `TransformComponent` 把 `Document` 切成带 `relationships` 的 `TextNode` 序列(第 4 章)→ IngestionPipeline 把若干 `TransformComponent` 串成链,顺带用 hash 去重(第 5 章)→ embed_model 给每个 node 填上 `embedding`(第 7 章)→ 存进 VectorStore 与 DocStore(第 8 章)→ Index 建立可检索结构(第 6 章)→ Retriever 按 query 取回相关 node(第 9 章)→ ResponseSynthesizer 把 node 内容塞进 prompt 让 LLM 生成答案(第 10 章)。整条链上流动的,始终是同一种 `Node`。理解了这一点,后面每一章你都会有"原来如此"的踏实感。

## 本章小结

- LlamaIndex 是**框架**而非产品:它提供可插拔的积木与组装规则,与前两部解构的 RAGFlow、AnythingLLM 的"开箱即用"定位根本不同。
- 仓库是 **namespace package 的 monorepo**(`pyproject.toml` 里 `namespace_packages = true`):`llama-index-core` 放与外部无关的抽象,`llama-index-integrations` 把每个具体 provider/向量库/reader 切成可独立 `pip install` 的小包,实现"核心抽象与具体集成解耦"。
- 全部核心数据结构定义在 `llama-index-core/llama_index/core/schema.py`(约 1492 行)。
- `BaseComponent` 继承 pydantic `BaseModel`,用 `class_name()` 做类型标签,提供 `to_dict`/`from_dict` 等对称序列化方法——"可持久化"是最底层就保证的全局契约。
- `TransformComponent` 的唯一抽象方法签名是 **`Sequence[BaseNode] -> Sequence[BaseNode]`**(node 进、node 出),这个统一接口让所有处理步骤可被串成管线。
- `BaseNode` 是"可被检索的最小单元":带 `id_`、`embedding`、`metadata`、关系图等字段;`validate_assignment=True` 保证赋值即校验。
- `relationships` + `NodeRelationship`(`SOURCE`/`PREVIOUS`/`NEXT`/`PARENT`/`CHILD`)让切碎的 node 构成一张**带方向的关系图**,是 auto-merging、recursive retrieval 的基础;取关系的 property 带类型校验,非法即 `raise ValueError`。
- `MetadataMode`(`ALL`/`EMBED`/`LLM`/`NONE`)配合两份 `excluded_*_metadata_keys` 黑名单,实现"一份 metadata、多种视图":同字段可对嵌入可见、对 LLM 隐藏。
- `TextNode` 的 `hash` 用 `sha256(text + metadata)` 计算,是摄入管线**去重与增量更新**的依据;`Document` 是未切分的原始文档端,通过 `SOURCE` 关系与切出的 node 相连。
- node 的一生:Document → NodeParser 切分 → IngestionPipeline 去重 → 嵌入 → 存储 → 索引 → 检索 → 合成,全链路流动的是同一种 `Node`。

## 动手实验

1. **实验一:确认命名空间包结构** — 打开根 `pyproject.toml`,找到 `namespace_packages = true`;再 `ls llama-index-integrations/` 下随便进一个集成包(如某个 llms 包)看它自己的 `pyproject.toml`。对比它和 `llama-index-core` 的依赖差异,说明为什么"用 OpenAI 才装 OpenAI 包"在这种结构下成为可能。
2. **实验二:亲手读懂 Node 的关系图** — 在 `schema.py` 里定位 `NodeRelationship` 枚举与 `BaseNode` 的 `source_node`/`prev_node`/`child_nodes` 这几个 property。重点看 `child_nodes` 里"关系值必须是 list,否则 raise ValueError"的校验逻辑,思考为什么 SOURCE/PREV/NEXT 是单值而 CHILD 是 list。
3. **实验三:验证 metadata 的三种视图** — 阅读 `BaseNode.get_metadata_str` 的实现,手动推演:给一个 node 设 `metadata={"file_name": "a.pdf", "author": "x"}`、`excluded_llm_metadata_keys=["file_name"]`,分别在 `MetadataMode.EMBED` 和 `MetadataMode.LLM` 下,输出的 metadata 字符串各包含哪些键?
4. **实验四:追踪 hash 的用途** — 在 `schema.py` 里读 `TextNode.hash` 的计算方式,然后用 Grep 在 `llama-index-core` 里搜索 `.hash` 的调用点(重点看 `ingestion/` 目录)。找出至少一处它被用来判断"节点是否已存在/已变更"的地方,记下文件名与函数名。

> **下一章预告**:积木的形状定义好了,但用户怎么告诉框架"我要用哪个 LLM、哪个嵌入模型"?下一章我们解剖 `settings.py` 里那个全局 `Settings` 单例——它如何用懒加载和 `resolve_*` 函数,把"全局默认"与"局部覆盖"两种依赖注入方式优雅地统一起来。
