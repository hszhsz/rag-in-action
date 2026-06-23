# 第 4 章 切分策略与 SplitterConfig

上一章我们看到 Processor 注册表如何把各种格式的文件「解析」成一段长文本（`Document.page_content`）。但 RAG 真正消费的从来不是整篇文档，而是一批**带 metadata 的小块（chunk）**——它们才是被嵌入、被检索、被塞进 LLM 上下文窗口的最小单位。本章聚焦「chunk 的诞生」：一段长文本如何被切成多块，配置如何控制粒度，以及切分之后基类又做了哪些统一加工，最终产出可溯源的 `Document`。

Quivr 把切分的「数值配置」与「切分实现」刻意解耦。数值集中在一个极简的 `SplitterConfig` 里（只有 `chunk_size` 和 `chunk_overlap` 两个字段），而怎么用这两个数字则由各个 Processor 自行决定——有的按字符切，有的按 token 切。这种解耦带来了一个微妙但重要的现象：**同一份配置，在不同实现里语义并不相同**。理解这一点，是用好 Quivr 切分体系的关键。

## SplitterConfig：两个数字，按字符计

切分配置的全部内容都在 `processor/splitter.py`，短到可以整段引用：

```python
class SplitterConfig(BaseModel):
    """
    This class is used to configure the chunking of the documents.

    Chunk size is the number of characters in the chunk.
    Chunk overlap is the number of characters that the chunk will overlap with the previous chunk.
    """

    chunk_size: int = 400
    chunk_overlap: int = 100
```

它是一个 pydantic `BaseModel`，只有两个带默认值的整型字段。文档字符串明确写道：`chunk_size` 是「块里的字符数」，`chunk_overlap` 是「与上一块重叠的字符数」——注意这里官方注释的口径是**按字符（characters）**计量，这正是后面要讨论的「语义随实现而变」的源头。

默认值 `chunk_size=400`、`chunk_overlap=100` 透露出设计取向：这是一个**保守的小块**策略。400 字符大约相当于几句话到一小段；100 字符的 overlap 恰好是块大小的 25%。较小的块让检索更聚焦、命中更精确，避免一个块里混入太多无关内容稀释相似度；而 25% 的重叠则保证相邻块之间的上下文不会被一刀切断——一个跨越块边界的句子或概念，会同时出现在前后两块里，降低「答案恰好被切在缝隙处」的风险。这是检索质量与冗余存储之间的一个经典折中。

由于 `SplitterConfig` 就是普通 pydantic 模型，它可以被 `model_dump()` 序列化进 metadata（`SimpleTxtProcessor.processor_metadata` 里就把 `self.splitter_config.model_dump()` 原样写入），也可以从 YAML/字典反序列化进更大的配置树——这一点我们在本章末尾讲 `ParserConfig` 时再展开。

## 实现一：字符级手写递归切分

最朴素、也最能看清「按字符切」本意的实现，是 `simple_txt_processor.py` 里的自由函数 `recursive_character_splitter`：

```python
def recursive_character_splitter(
    doc: Document, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    assert chunk_overlap < chunk_size, "chunk_overlap is greater than chunk_size"

    if len(doc.page_content) <= chunk_size:
        return [doc]

    chunk = Document(page_content=doc.page_content[:chunk_size], metadata=doc.metadata)
    remaining = Document(
        page_content=doc.page_content[chunk_size - chunk_overlap :],
        metadata=doc.metadata,
    )

    return [chunk] + recursive_character_splitter(remaining, chunk_size, chunk_overlap)
```

这段代码值得逐行读，因为它把切分逻辑剥到了最裸：

1. 开头一句 `assert chunk_overlap < chunk_size`——这是一个硬约束。如果重叠不小于块大小，下面递归时 `remaining` 的起点 `chunk_size - chunk_overlap` 将不前进甚至倒退，导致无限递归。断言把这个非法配置在第一时间拦下。
2. **递归出口**：若 `len(doc.page_content) <= chunk_size`，内容本身就装得下一块，直接原样返回 `[doc]`，不再切分。
3. **取首块**：`doc.page_content[:chunk_size]` 截前 `chunk_size` 个字符为第一块，metadata 沿用原文档。
4. **算余下**：`remaining` 的起点是 `chunk_size - chunk_overlap`。也就是说下一块不是从第一块的末尾接着开始，而是**回退 `chunk_overlap` 个字符**，于是相邻两块自然产生了重叠区。以默认值算，每块覆盖 400 字符，下一块从第 300 字符处起步，重叠 100 字符。
5. **递归拼接**：`[chunk] + recursive_character_splitter(remaining, ...)`，对剩余部分重复同样的过程，直到余下内容小于一块为止。

这个实现的特点是**纯字符、零语义**：它不认识单词边界、句子边界或段落，纯粹按下标切。优点是简单、可预测、无外部依赖；代价是它会**切碎单词**——第 400 个字符很可能落在某个单词中间，于是块的首尾常出现半截词。对 `SimpleTxtProcessor` 处理的纯文本场景而言，这种粗放往往可以接受。

在 `SimpleTxtProcessor.process_file_inner` 里，它读完整个文件、包成一个 `Document`，然后调用上面的函数，把 `splitter_config.chunk_size` 和 `splitter_config.chunk_overlap` 原样传进去。这里 400/100 的含义就是**字符**——与 `SplitterConfig` 的文档字符串完全一致。

## 实现二：token 级的 RecursiveCharacterTextSplitter

另一条路线出现在 `tika_processor.py`（以及 `default.py`、`megaparse_processor.py`）。`TikaProcessor` 把 PDF 交给 Tika 服务器解析成纯文本后，用的是 LangChain 的切分器：

```python
self.enc = tiktoken.get_encoding("cl100k_base")
...
self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=splitter_config.chunk_size,
    chunk_overlap=splitter_config.chunk_overlap,
)
```

关键在 `from_tiktoken_encoder` 这个构造方式 [[LangChain RecursiveCharacterTextSplitter]](https://python.langchain.com/docs/how_to/recursive_text_splitter/)。它让 `RecursiveCharacterTextSplitter` 在度量块大小时不再数字符，而是用 tiktoken 的 `cl100k_base` 编码 [[tiktoken]](https://github.com/openai/tiktoken) 数 **token**。`cl100k_base` 正是 GPT-3.5/GPT-4 系列所用的 BPE 编码 [[OpenAI cl100k_base]](https://platform.openai.com/docs/guides/text-generation)。于是这里传进去的 `chunk_size=400` 含义变成了「每块约 400 个 token」，`chunk_overlap=100` 是「约 100 个 token 的重叠」。

与手写递归相比，LangChain 的 `RecursiveCharacterTextSplitter` 还更「聪明」：它会按一组分隔符（段落、换行、空格……）优先在自然边界处下刀，尽量不把句子拦腰斩断，只有在某段实在塞不进一块时才退而求其次硬切。这比 `recursive_character_splitter` 的纯下标切分对单词/句子友好得多。

切完之后，`TikaProcessor` 还给每个 chunk 写了一条 metadata：

```python
for doc in docs:
    doc.metadata = {"chunk_size": len(self.enc.encode(doc.page_content))}
```

也就是用同一个 `cl100k_base` 编码器把块内容 encode 一遍，记录这一块**实际的 token 数**。`MegaparseProcessor` 与 `default.py` 里动态构建的各类 Processor（`CSVProcessor`、`DOCXProcessor`、`MarkdownProcessor` 等）走的也是同一套：`RecursiveCharacterTextSplitter.from_tiktoken_encoder` + 给每块写 `chunk_size`（token 数）的 metadata。

## 同一份配置，两种语义

把两条实现并排看，一个值得读者警惕的事实浮现出来：**`SplitterConfig` 里那个 `chunk_size=400`，在 `SimpleTxtProcessor` 里是 400 个字符，在 `TikaProcessor`/`MegaparseProcessor`/`default.py` 里却是约 400 个 token**。

这两者不是一回事。对英文文本，一个 token 平均约对应 4 个字符，所以 400 token 的块往往比 400 字符的块**长好几倍**；对中文，token 与字的换算又是另一套比例。换句话说，你改一个数字，落到不同 Processor 上得到的块物理长度可能差出一个数量级。

这就是「配置语义随实现而变」的典型案例。`SplitterConfig` 的文档字符串只说了字符口径，这与 token 口径的多数实现并不一致——文档和代码在这里有一处需要读者自行脑补的张力。设计上更贴近 LLM 的其实是 token 口径：因为模型的上下文窗口、计费、截断都是按 token 算的，用 token 度量块大小能直接对齐「一个块占多少上下文预算」。理解了这层语义差异，在切换 Processor 或调参时就不会被表面相同的数字误导。

## chunk 后处理：基类统一注入溯源 metadata

切分只是「chunk 诞生」的前半程。无论哪个 Processor，它们的 `process_file_inner` 产出的都还只是「内容被切开、metadata 很简陋」的半成品。真正把半成品变成可溯源 `Document` 的，是基类 `ProcessorBase.process_file`（`processor_base.py`）——这是模板方法模式：子类负责解析与切分（`process_file_inner`），基类负责统一的善后加工。

核心是这段对每个 chunk 的循环：

```python
for idx, doc in enumerate(docs.chunks, start=1):
    if "original_file_name" in doc.metadata:
        doc.page_content = f"Filename: {doc.metadata['original_file_name']} Content: {doc.page_content}"
    doc.page_content = doc.page_content.replace(" ", "")
    doc.page_content = doc.page_content.encode("utf-8", "replace").decode("utf-8")
    doc.metadata = {
        "chunk_index": idx,
        "quivr_core_version": qvr_version,
        "language": detect_language(
            text=doc.page_content.replace("\\n", " ").replace("\n", " "),
            low_memory=True,
        ).value,
        **file.metadata,
        **doc.metadata,
        **self.processor_metadata,
    }
```

逐项拆解这套统一加工：

- **`chunk_index`**：`enumerate(..., start=1)`，块序号**从 1 开始**而非 0，记录每块在原文中的先后次序，是检索结果回溯原文位置的关键。
- **内容增强与清洗**：若 metadata 带 `original_file_name`，会把文件名拼到内容前面（`Filename: ... Content: ...`），让块自身携带来源线索；随后 `replace(" ", "")` 去掉空字节（NUL，很多向量库/数据库不接受），再用 `encode("utf-8", "replace").decode("utf-8")` 把非法字节替换掉，保证内容是干净的 UTF-8。
- **`quivr_core_version`**：通过 `importlib.metadata.version("quivr-core")` 取当前库版本（取不到则记为 `"dev"`），写进每块，便于日后排查「这批数据是哪个版本切的」。
- **`language`**：调用 `detect_language` 对块内容做语言探测（`low_memory=True`），把探测结果的 `.value` 存下来。注意它先把字面的 `\\n` 和真实换行 `\n` 都替换成空格再检测，避免换行符干扰语言识别。
- **metadata 合并顺序**：最后用字典展开把多来源 metadata 合到一起，顺序是 `基类字段 → file.metadata → doc.metadata → self.processor_metadata`。由于后写覆盖先写，这等于规定了**优先级**：Processor 自己的 metadata 优先级最高，其次是切分时附在块上的 metadata，再次是文件级 metadata，基类注入的 `chunk_index` 等通用字段优先级最低（会被同名键覆盖）。

走完这一轮，每个 chunk 都从「一段裸文本」升级成了「内容干净、带序号、带版本、带语言、带完整来源」的 `Document`。可以说切分决定了「切多碎」，而基类的后处理决定了「每块能不能被追溯、能不能被安全入库」——两者合起来才是 chunk 完整的诞生流程。

## ParserConfig / MegaparseConfig：切分配置如何挂进配置树

最后看 `SplitterConfig` 在整个配置体系里的位置。`rag/entities/config.py` 里：

```python
class ParserConfig(QuivrBaseConfig):
    splitter_config: SplitterConfig = SplitterConfig()
    megaparse_config: MegaparseConfig = MegaparseConfig()
```

`ParserConfig` 把「怎么切」（`splitter_config`）和「怎么解析」（`megaparse_config`）组合在一起，构成「解析阶段」的完整配置。其中 `MegaparseConfig`（定义在 `config.py`）描述的是 Megaparse 解析器的行为，字段如 `method`（默认 `ParserType.UNSTRUCTURED`）、`strategy`（默认 `StrategyEnum.FAST`）、`check_table`、`model_name`（默认 `"gpt-4o"`）等——它管的是「把文件变成文本」，与 `SplitterConfig` 管的「把文本切成块」恰好互补。

再往外，`ParserConfig` 又被 `IngestionConfig` 包住（`parser_config: ParserConfig`），`IngestionConfig` 再和 `RetrievalConfig` 一起组成顶层的 `AssistantConfig`。于是配置形成一棵树：

```
AssistantConfig
├── ingestion_config: IngestionConfig
│   └── parser_config: ParserConfig
│       ├── splitter_config: SplitterConfig   ← chunk_size / chunk_overlap
│       └── megaparse_config: MegaparseConfig
└── retrieval_config: RetrievalConfig
```

这棵树意味着：用户只要在一份 YAML/字典里写好 `splitter_config`，它就能沿着 `AssistantConfig → IngestionConfig → ParserConfig` 一路注入到摄入流程，最终被传给某个 Processor 的构造函数。切分配置因此不是散落在代码里的魔法数字，而是配置驱动体系里一个可声明、可序列化、可覆盖的节点。摄入配置体系的全貌（`IngestionConfig`、`AssistantConfig` 如何驱动 brain 构建）会在后续章节展开，这里点到为止。

## 本章小结

- `SplitterConfig`（`splitter.py`）是个极简 pydantic 模型，只有 `chunk_size: int = 400` 和 `chunk_overlap: int = 100` 两个字段，官方文档口径为「按字符」。
- 默认 400/100 是保守小块策略：小块让检索更聚焦，25% 的 overlap 维持相邻块上下文连续，避免答案被切在缝隙处。
- `recursive_character_splitter`（`simple_txt_processor.py`）是纯字符手写递归：先 `assert chunk_overlap < chunk_size` 防无限递归，内容装得下就原样返回，否则截前 `chunk_size` 字符为一块、从 `chunk_size - chunk_overlap` 处递归剩余。
- 字符级切分零语义、会切碎单词，但简单可预测、无外部依赖。
- `TikaProcessor`/`MegaparseProcessor`/`default.py` 用 `RecursiveCharacterTextSplitter.from_tiktoken_encoder` 按 `cl100k_base` 的 token 数切分，更贴近 LLM 上下文预算，且会按自然边界优先下刀。
- token 级实现还给每块写 `chunk_size` metadata，记录用 `cl100k_base` encode 后的实际 token 数。
- **同一份 `SplitterConfig` 数值在两种实现里语义不同**：`SimpleTxtProcessor` 里是字符数，token 实现里是 token 数，物理长度可能差一个数量级——这是「配置语义随实现而变」。
- 基类 `ProcessorBase.process_file`（模板方法）统一给每块注入 `chunk_index`（从 1 开始）、`quivr_core_version`、`detect_language` 探测的 `language`，并清洗 NUL/非法 UTF-8。
- metadata 合并顺序为 `基类字段 → file.metadata → doc.metadata → processor_metadata`，后者覆盖前者，决定字段优先级。
- `ParserConfig`（`config.py`）组合 `splitter_config` 与 `megaparse_config`，并经 `IngestionConfig`→`AssistantConfig` 挂进配置树，使切分配置可声明、可序列化、可注入。

## 动手实验

1. **实验一：观察字符递归切分的块数** — 用 `recursive_character_splitter` 处理一段约 1200 字符的文本，分别设 `chunk_size=400/chunk_overlap=100` 与 `chunk_size=200/chunk_overlap=50`，打印 `len(result)` 与每块首尾字符，验证步进为 `chunk_size - chunk_overlap` 且相邻块确有重叠。
2. **实验二：触发 assert** — 调用 `recursive_character_splitter(doc, chunk_size=100, chunk_overlap=100)`，观察 `chunk_overlap is greater than chunk_size` 断言如何在第一时间拦下非法配置；再把 overlap 改成 99 看是否正常返回。
3. **实验三：对比字符切分 vs token 切分** — 对同一段含长单词的英文文本，一边用 `recursive_character_splitter`，一边用 `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=100)`，对比两者块数、块的物理字符长度，并用 `tiktoken.get_encoding("cl100k_base").encode` 数每块 token 数，直观感受「字符 vs token」的语义差异。
4. **实验四：验证基类 metadata 注入** — 走完整的 `ProcessorBase.process_file` 处理一份带 `original_file_name` 的文件，打印每个 chunk 的 metadata，确认 `chunk_index` 从 1 递增、`quivr_core_version` 与 `language` 已写入，且 `processor_metadata` 的同名键覆盖了基类默认值。

> **下一章预告**：chunk 诞生后要么被嵌入入库，要么被喂给大模型生成答案。下一章《LLMEndpoint 与多供应商模型配置》将钻进 `llm/` 目录，看 Quivr 如何用统一的 `LLMEndpoint` 抽象屏蔽 OpenAI、Anthropic、Azure 等多家供应商的差异，以及 `LLMEndpointConfig` 怎样配置模型、API Key 与上下文窗口。
