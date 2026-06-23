# 第 4 章 节点解析与切分 NodeParser

上一章里,各类 `Reader` 把原始文件读进内存,统一吐出 `Document` 对象——一篇 PDF、一个网页、一封邮件,都被装进了一个带 `text` 和 `metadata` 的容器。但 `Document` 有一个致命问题:它太大了。一篇几万字的文档,既塞不进嵌入模型有限的上下文窗口,也无法做到"检索到的就是相关的"——你不希望为了回答一个问题而把整篇文档丢给 LLM,而是希望命中其中真正相关的那几段。

于是就有了本章的主角:`NodeParser`。它的职责是把一个大 `Document` 切成若干个尺寸合适、彼此带有关系链的 `TextNode`,让每个节点都成为可独立嵌入、可独立检索、又能在需要时还原回上下文的最小单元。切分看似简单(无非是按长度截断),但要在"不切碎句子""保留重叠上下文""按 token 而非字符计长""为节点建立来源与前后关系"这些约束之间取得平衡,工程含量并不低。本章顺着 LlamaIndex 的源码,把这套切分骨架拆开来看。

## NodeParser 作为 TransformComponent:node 进,node 出

打开 `node_parser/interface.py`,第一眼要看的是基类签名:`class NodeParser(TransformComponent, ABC)`。它继承自 `TransformComponent`——这正是第 5 章摄入管线能把切分、元数据抽取、嵌入等步骤串成一条链的根本原因:所有变换组件都共享同一个调用约定。

这个约定就是 `__call__`:

```python
def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[BaseNode]:
    return self.get_nodes_from_documents(nodes, **kwargs)
```

注意参数名是 `nodes`、类型是 `Sequence[BaseNode]`,而内部转手调用的却是 `get_nodes_from_documents`。这种"node 进、node 出"的统一签名是刻意设计的:在管线里,上游产出什么、下游就接什么,`Document` 本身也是 `BaseNode` 的子类,所以一个解析器既能吃 `Document`,也能吃别的解析器产出的 `TextNode`(`HierarchicalNodeParser` 的多级切分正是靠这点,后文会讲)。`NodeParser` 还提供了 `acall`/`aget_nodes_from_documents` 的异步版本,默认 `_aparse_nodes` 直接转调同步的 `_parse_nodes`。

基类用四个 pydantic `Field` 定义了所有解析器共享的配置:`include_metadata`(默认 `True`)、`include_prev_next_rel`(默认 `True`)、`callback_manager`,以及 `id_func`(默认 `default_id_func`)。前两个布尔开关后面会反复出现,它们直接决定切分后要不要把父文档的元数据下放、要不要建立前后节点关系。

## 抽象方法 `_parse_nodes` 与编排方法 `get_nodes_from_documents`

`NodeParser` 把"怎么切"留给子类,自己只负责编排。核心是一个抽象方法:

```python
@abstractmethod
def _parse_nodes(self, nodes, show_progress=False, **kwargs) -> List[BaseNode]: ...
```

任何具体解析器都必须实现 `_parse_nodes`,它的任务就是把输入节点切成输出节点。而对外暴露的 `get_nodes_from_documents` 是一层固定的编排壳:

```python
def get_nodes_from_documents(self, documents, show_progress=False, **kwargs):
    doc_id_to_document = {doc.id_: doc for doc in documents}
    with self.callback_manager.event(CBEventType.NODE_PARSING, ...) as event:
        nodes = self._parse_nodes(documents, show_progress=show_progress, **kwargs)
        nodes = self._postprocess_parsed_nodes(nodes, doc_id_to_document)
        event.on_end({EventPayload.NODES: nodes})
    return nodes
```

这里有两个值得记住的点。第一,整个过程被包在一个 `CBEventType.NODE_PARSING` 回调事件里,这是第 14 章 instrumentation 能观测到"切分耗时、切出多少节点"的钩子来源。第二,切分(`_parse_nodes`)和后处理(`_postprocess_parsed_nodes`)是分离的:子类只管"把文本切开",而"建立关系、回填元数据、计算字符偏移"这些通用动作统一由基类的 `_postprocess_parsed_nodes` 完成。这种分工让每个具体解析器都能保持精简。

## `_postprocess_parsed_nodes`:关系、元数据与字符偏移的统一回填

`_postprocess_parsed_nodes` 是基类里逻辑最密集的一段,它遍历所有切出来的节点,做三件事。

第一件是**计算字符偏移**。它在父文档原文里用 `parent_doc.text.find(node_content, search_start)` 定位每个节点的起止位置,写入 `node.start_char_idx` / `node.end_char_idx`。这里有个细节:它维护了一个 `doc_search_positions` 字典逐文档记录上次搜索的起点,并且每次只把搜索位置推进到 `start_char_idx + 1`(而非节点末尾)。源码注释解释了原因——这样做是为了让**有重叠的相邻 chunk** 也能被正确定位,如果按末尾推进,重叠部分就会被跳过而找不到。

第二件是**回填元数据**。当 `include_metadata` 为真时,它执行 `node.metadata = {**parent_doc.metadata, **node.metadata}`,把父文档的元数据合并进节点,且以节点自身的值为优先。这就是为什么切出来的每个 chunk 都能继承诸如 `file_name`、`page_label` 这类来源信息。

第三件是**建立前后关系**。当 `include_prev_next_rel` 为真时,它在循环里检查相邻节点是否同源(`nodes[i-1].source_node.node_id == node.source_node.node_id`),只有共享同一个 `source_node` 才会建立 `NodeRelationship.PREVIOUS` / `NodeRelationship.NEXT`。这个同源判断很关键:它保证了来自不同文档的节点不会被错误地串成前后链。

## 切分的通用骨架:TextSplitter 的 split → build_nodes

`interface.py` 里还定义了一条专门处理纯文本切分的支线:`class TextSplitter(NodeParser)`。它在 `NodeParser` 之上只新增了一个抽象方法 `split_text(self, text: str) -> List[str]`——也就是说,文本类解析器只需回答"给我一段文本,你切成哪几段字符串"这一个问题,剩下的全由基类兜底。

`TextSplitter` 自己实现了 `_parse_nodes`,把这层抽象落地:

```python
def _parse_nodes(self, nodes, show_progress=False, **kwargs):
    all_nodes = []
    for node in get_tqdm_iterable(nodes, show_progress, "Parsing nodes"):
        splits = self.split_text(node.get_content())
        all_nodes.extend(build_nodes_from_splits(splits, node, id_func=self.id_func))
    return all_nodes
```

这就是整个切分体系的通用骨架:**对每个输入节点,先 `split_text` 得到字符串列表,再交给 `build_nodes_from_splits` 构造成 `TextNode`**。具体的切分算法(按句、按 token、按段落)只是 `split_text` 的不同实现,而把字符串变成节点的活儿是共用的。

在 `TextSplitter` 之上还有一层 `MetadataAwareTextSplitter`。它的存在是为了解决一个工程现实:嵌入时元数据也会被拼进文本一起送给模型,所以如果不把元数据占用的 token 预算扣掉,真正的正文就可能超出 chunk 尺寸。它新增抽象方法 `split_text_metadata_aware(text, metadata_str)`,并在 `_get_metadata_str` 里取 `MetadataMode.EMBED` 和 `MetadataMode.LLM` 两种模式中**更长的那个**元数据串来预留空间,确保两种用途都不会溢出。`SentenceSplitter` 和 `TokenTextSplitter` 都继承自这一层。

## `build_nodes_from_splits`:把字符串列表变成带 SOURCE 关系的 TextNode

`node_utils.py` 里的 `build_nodes_from_splits` 是所有解析器共用的"装配车间"。它的核心逻辑是把每个文本片段包成一个 `TextNode`,并统一挂上来源关系:

```python
relationships = {NodeRelationship.SOURCE: ref_doc.as_related_node_info()}
for i, text_chunk in enumerate(text_splits):
    node = TextNode(id_=id_func(i, document), text=text_chunk,
                    embedding=document.embedding, ...,
                    relationships=relationships)
    nodes.append(node)
```

这里有两处工程考量值得点出。其一,`relationships` 字典是在循环**外**只构造一次的。源码注释写得很直白:对文档调用 `as_related_node_info()` 会重新计算整篇文本和元数据的 hash,代价不小;如果放进循环里每个节点都算一遍,就"terrible"了——所以只算一次,所有兄弟节点共享同一个 SOURCE 引用。其二,它会按文档类型分派:`ImageDocument` 造 `ImageNode`、`Document` 和 `TextNode` 造 `TextNode`,并把 `excluded_embed_metadata_keys`、`metadata_template`、`text_template` 等格式化配置原样从父对象继承下来,保证切出来的节点和原文档在元数据呈现上保持一致。

节点 ID 由 `id_func(i, document)` 生成,默认实现 `default_id_func` 极简:`return str(uuid.uuid4())`——也就是说默认每个节点拿一个随机 UUID。需要可复现 ID(例如基于内容 hash 做去重)的场景,可以传入自定义 `id_func` 覆盖。

注意 `build_nodes_from_splits` 这里只挂了 SOURCE 关系;PREV/NEXT 是前面讲过的 `_postprocess_parsed_nodes` 在后处理阶段补的。两阶段分工于是闭环:装配车间负责"出生证"(来源),后处理负责"族谱"(前后邻里)。

## SentenceSplitter:分隔符回退链与按 token 的重叠合并

`text/sentence.py` 的 `SentenceSplitter` 是最常用的解析器,它的设计目标写在类注释里:**尽量保持句子和段落完整**,相比纯按 token 切的 `TokenTextSplitter`,更少出现把句子切到一半的"断头"节点。

先看默认值:`chunk_size` 默认取 `DEFAULT_CHUNK_SIZE`(在 `constants.py` 中为 `1024` tokens),`chunk_overlap` 默认取本文件定义的 `SENTENCE_CHUNK_OVERLAP = 200`。构造函数里有一道前置校验:如果 `chunk_overlap > chunk_size` 直接抛 `ValueError`——重叠比块还大显然是配置错误。

它的切分分两步:`_split_text` 里先 `splits = self._split(text, chunk_size)` 把文本拆成不超尺寸的细粒度片段,再 `chunks = self._merge(splits, chunk_size)` 把细片段合并回接近 chunk_size 的块。

`_split` 体现的是**分隔符回退链**。它先按粗粒度分隔符切,切不动了再换更细的:`_split_fns` 依次是 `split_by_sep(paragraph_separator)`(段落分隔符默认 `"\n\n\n"`)和句子分词器 `_chunking_tokenizer_fn`(默认基于 nltk 的 punkt);若还有超尺寸片段,再走 `_sub_sentence_split_fns`——依次是 `split_by_regex(secondary_chunking_regex)`、`split_by_sep(separator)`(词分隔符默认空格 `" "`)、`split_by_char()`(逐字符)。其中那条次级正则 `CHUNKING_REGEX = "[^,.;。?!]+[,.;。?!]?|[,.;。?!]"` 同时考虑了中英文标点,所以中文文本也能在子句层级被切开。`_get_splits_by_fns` 的策略是:逐个尝试分隔函数,**一旦某个函数能切出多于一个片段就采用它**,实现"能用粗的就不用细的"的逐级回退。`_split` 是递归的:对仍然超尺寸的片段继续往下一级回退切,直到每片都 `token_size <= chunk_size`。

`_merge` 体现的是**重叠合并**。它不断把片段塞进当前块,直到再加一个就超 `chunk_size` 就 `close_chunk()` 收口。收口时的精髓在于制造重叠:它从上一个块的尾部往前回填片段进新块,条件是 `cur_chunk_len + last_chunk[last_index][1] <= self.chunk_overlap`——也就是说,新块开头会带上一块结尾、累计不超过 `chunk_overlap` token 的内容。所有长度判断都走 `_token_size`(即 `len(self._tokenizer(text))`),用的是 token 而非字符,这也是 chunk_size 单位是 token 的原因。最后 `_postprocess_chunks` 会 `strip` 掉空白并丢弃纯空白块。

## chunk_size / chunk_overlap 的工程权衡

这两个参数是 RAG 调优里最先要碰的旋钮,源码默认值背后是有取舍的。`chunk_size` 太大,单个节点信息冗杂、检索精度下降、且逼近嵌入模型上下文上限;太小,语义被切碎、一个完整论点散落在多个节点里,检索召回反而变差。`1024` tokens 是一个对通用文档比较稳妥的折中。

`chunk_overlap` 的意义是"接缝处的语义保险":一个句子若不幸落在两个块的边界上,重叠能保证它至少完整出现在其中一个块里,避免边界信息丢失。但重叠是有成本的——重叠越大,节点数量越多、存储与嵌入计算越贵、检索时还可能返回内容高度重复的相邻块。`SentenceSplitter` 把默认重叠定在 `200`(而通用基线 `DEFAULT_CHUNK_OVERLAP` 仅 `20`),正是因为它面向"保句子完整"的场景,愿意用更大的重叠换接缝处的连贯。`_merge` 里"先收口、再从上一块尾部回填不超过 overlap 的内容"这套机制,把这个权衡精确地落到了 token 级别。

## TokenTextSplitter:更朴素的按 token 直切

`text/token.py` 的 `TokenTextSplitter` 是更朴素的实现,类注释直说它"看 word tokens"。和 `SentenceSplitter` 比有两点不同:一是默认 `chunk_overlap` 用的是 `DEFAULT_CHUNK_OVERLAP`(`20`)而非 200,因为它不刻意保句子完整;二是它的回退链更短——`_split_fns` 是 `[split_by_sep(sep) for sep in all_seps] + [split_by_char()]`,其中 `all_seps` 是主分隔符(默认空格)加 `backup_separators`(默认 `["\n"]`),没有句子分词器这一档。

它的 `split_text_metadata_aware` 还多扣了一个常量:`metadata_len = len(self._tokenizer(metadata_str)) + DEFAULT_METADATA_FORMAT_LEN`,其中 `DEFAULT_METADATA_FORMAT_LEN = 2` 是为元数据格式化(拼接符)预留的固定 token。`_merge` 的重叠逻辑也更直白:超尺寸时收口,然后从当前块头部不断 `pop(0)`,直到 `cur_len <= self.chunk_overlap` 且新片段能放下为止。当文档结构无关紧要、只想要稳定均匀的 token 块时,它比 `SentenceSplitter` 更轻量可控。

## SentenceWindowNodeParser:切小检索,还原大上下文

`text/sentence_window.py` 的 `SentenceWindowNodeParser` 实现了一个非常实用的检索技巧:**用句子粒度做嵌入和检索,但在节点元数据里存好它周围的窗口,命中后再还原成大上下文喂给 LLM**。

它的切分极细——`sentence_splitter` 默认就是 `split_by_sentence_tokenizer()`,即每个句子一个节点。关键在 `build_window_nodes_from_documents` 里给每个节点贴的窗口:

```python
window_nodes = nodes[max(0, i - self.window_size) : min(i + self.window_size + 1, len(nodes))]
node.metadata[self.window_metadata_key] = " ".join([n.text for n in window_nodes])
node.metadata[self.original_text_metadata_key] = node.text
```

`window_size` 默认 `DEFAULT_WINDOW_SIZE = 3`,即取当前句左右各 3 句、连同自己拼成窗口,存进元数据键 `window`(`DEFAULT_WINDOW_METADATA_KEY`);同时把本句原文存进 `original_text`(`DEFAULT_OG_TEXT_METADATA_KEY`)。最妙的一步是它随即把这两个键加进 `excluded_embed_metadata_keys` 和 `excluded_llm_metadata_keys`——**窗口和原文不参与嵌入计算**,所以嵌入向量代表的仍是那一句话(检索精度高),而窗口只是挂在旁边等检索命中后由专门的后处理器(`MetadataReplacementPostProcessor`,属后续章节范围)取出来替换正文,实现"小向量、大上下文"。这是对 chunk_size 两难的一种巧解:不靠折中尺寸,而是把"检索单元"和"喂给 LLM 的单元"解耦。

## HierarchicalNodeParser:父子层级,为 auto-merging 铺垫

`relational/hierarchical.py` 的 `HierarchicalNodeParser` 把同一篇文档按**多档 chunk 尺寸**切成一棵层级树。`from_defaults` 里 `chunk_sizes` 的默认值是 `[2048, 512, 128]`——顶层大块、中层中块、底层小块,每一档都生成一组 `SentenceSplitter`(存进 `node_parser_map`,id 形如 `chunk_size_2048`)。

层级构建在 `_recursively_get_nodes_from_nodes` 里递归完成:先用当前 level 的解析器把节点切成子节点,然后——

```python
if level > 0:
    for sub_node in cur_sub_nodes:
        _add_parent_child_relationship(parent_node=node, child_node=sub_node)
```

注意 `level > 0` 这道判断:第 0 层是直接切原始 `Document`,不该把"文档→大块"也建成父子关系,所以只有从第 1 层起才挂 `NodeRelationship.PARENT` / `NodeRelationship.CHILD`。`_add_parent_child_relationship` 是双向的:给父节点的 CHILD 列表追加孩子、给子节点写上 PARENT 引用。最终所有层级的节点被**拍平进一个 list** 返回(`sub_nodes + sub_sub_nodes`),而不是嵌套结构——树形关系完全靠节点间的 PARENT/CHILD 关系来表达。

文件里还附带了几个遍历辅助函数:`get_leaf_nodes`(没有 CHILD 关系的节点,即最小块)、`get_root_nodes`(没有 PARENT 关系的顶层块)、`get_child_nodes`、`get_deeper_nodes`。这套父子结构正是第 9 章 `AutoMergingRetriever` 的地基:检索时命中一堆叶子小块,如果同一个父块下的孩子被命中得足够多,就"向上合并"成父块返回,从而用小块的检索精度换大块的上下文完整度。本章先把层级是怎么切出来、关系是怎么挂上的讲清楚,合并逻辑留到检索章节。

## 按文件结构切分:MarkdownNodeParser 与 JSONNodeParser

除了按长度切,LlamaIndex 还提供按文件原生结构切的解析器。`file/markdown.py` 的 `MarkdownNodeParser` 不看 token 数,而是按 Markdown 标题切:它逐行扫描,用正则 `^(#+)\s(.*)` 识别标题,维护一个 `header_stack` 标题栈,遇到新标题就把前一段收成一个节点,并把当前所在的标题路径(用 `header_path_separator`,默认 `/`)写进元数据 `header_path`。它还特意用 `code_block` 标志位跳过代码块内以 ` ``` ` 包裹的内容,避免把代码里的 `#` 误判成标题。

`file/json.py` 的 `JSONNodeParser` 则按 JSON 的结构切分。两者都和文本切分器共用同一套出口——最终都调 `build_nodes_from_splits` 把片段装成 `TextNode`,因而同样自动获得 SOURCE 关系、元数据继承和(经基类后处理的)PREV/NEXT 关系。这再次印证了本章反复出现的主线:**切分算法可以五花八门,但"字符串列表 → 带关系的 TextNode"这道出口是统一的。**

## 本章小结

- `NodeParser` 继承自 `TransformComponent`,统一的 `__call__`(node 进 node 出)使它能无缝嵌入第 5 章的摄入管线;`Document` 本身也是 `BaseNode`,所以解析器既能吃文档也能吃节点。
- 基类把"怎么切"(抽象方法 `_parse_nodes`)和"通用后处理"(`_postprocess_parsed_nodes`)分离;`get_nodes_from_documents` 是固定编排壳,并包在 `CBEventType.NODE_PARSING` 回调事件里供观测。
- `_postprocess_parsed_nodes` 统一回填三件事:用 `text.find` 计算 `start_char_idx`/`end_char_idx`(只推进 1 位以支持重叠 chunk 定位)、按 `include_metadata` 下放父文档元数据、按 `include_prev_next_rel` 在同源节点间建立 PREV/NEXT。
- 切分通用骨架是 `split_text → build_nodes_from_splits`;`build_nodes_from_splits` 在循环外只算一次 SOURCE 关系(避免重复 hash),并按文档类型分派 `TextNode`/`ImageNode`,节点 ID 默认由 `default_id_func` 返回随机 UUID。
- `SentenceSplitter` 默认 `chunk_size=1024`、`chunk_overlap=200`,通过 `_split` 的分隔符回退链(段落→句子→次级正则→词→字符)尽量保句完整,再由 `_merge` 在 token 级别制造不超过 overlap 的重叠。
- `chunk_size` 与 `chunk_overlap` 是精度、上下文完整度与成本之间的权衡;`MetadataAwareTextSplitter` 会预扣元数据 token 预算,避免正文溢出。
- `TokenTextSplitter` 是更朴素的按 token 直切,默认 overlap 仅 `20`,回退链更短(无句子分词器档),并额外预留 `DEFAULT_METADATA_FORMAT_LEN = 2` 个格式化 token。
- `SentenceWindowNodeParser` 按句切(默认 `window_size=3`),把左右窗口存进 `window` 元数据但排除出嵌入,实现"小向量检索、大上下文还原"的解耦。
- `HierarchicalNodeParser` 按 `chunk_sizes`(默认 `[2048, 512, 128]`)递归切出父子层级,仅在 `level > 0` 时挂 PARENT/CHILD,拍平返回,为第 9 章 `AutoMergingRetriever` 铺垫。
- `MarkdownNodeParser`/`JSONNodeParser` 按文件原生结构切分,但同样收口到 `build_nodes_from_splits`,共享统一的节点装配出口。

## 动手实验

1. **实验一:确认 chunk 默认值的双重来源** — 在 `/tmp/llama_explore/llama-index-core/llama_index/core/constants.py` 里 Grep `DEFAULT_CHUNK`,确认 `DEFAULT_CHUNK_SIZE = 1024`、`DEFAULT_CHUNK_OVERLAP = 20`;再打开 `node_parser/text/sentence.py` 找到 `SENTENCE_CHUNK_OVERLAP = 200`,推演为什么 `SentenceSplitter` 的实际默认重叠是 200 而不是 20。
2. **实验二:追踪 SOURCE 关系只算一次的优化** — 在 `node_parser/node_utils.py` 的 `build_nodes_from_splits` 中定位 `relationships = {NodeRelationship.SOURCE: ref_doc.as_related_node_info()}` 这一行,确认它在 `for` 循环之外;再读紧邻的三行注释,说清把它放进循环为什么会"terrible"。
3. **实验三:验证 PREV/NEXT 的同源判断** — 在 `node_parser/interface.py` 的 `_postprocess_parsed_nodes` 里找到建立 `NodeRelationship.PREVIOUS`/`NEXT` 的两段 `if`,定位 `nodes[i-1].source_node.node_id == node.source_node.node_id` 这个条件,推演:若一次 `get_nodes_from_documents` 传入两篇不同文档,为什么跨文档的相邻节点不会被错误串联。
4. **实验四:数清 Hierarchical 的层级与关系挂载条件** — 在 `node_parser/relational/hierarchical.py` 的 `from_defaults` 里确认 `chunk_sizes` 默认 `[2048, 512, 128]`;再到 `_recursively_get_nodes_from_nodes` 找到 `if level > 0:` 那段 `_add_parent_child_relationship` 调用,解释为什么第 0 层(切原始 Document)不建立父子关系。

> **下一章预告**:本章的 `NodeParser` 只是众多 `TransformComponent` 中的一员。第 5 章将进入 `IngestionPipeline`——把节点解析、元数据抽取、嵌入等多个 `TransformComponent` 串成一条可复用的链,并引入文档 hash 去重与缓存,让重复摄入同一份数据时能跳过已算过的步骤。
