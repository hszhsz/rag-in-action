# 第 6 章 文档转换与预处理切分

第 5 章我们解构了 Pipeline 的序列化与断点：一张计算图如何被存成 YAML、又如何从断点恢复。但图里流动的"血液"——`Document`——是从哪里来的？真实世界的语料是 PDF、Word、Markdown、CSV、HTML、JSON，它们必须先被"翻译"成 Haystack 唯一认识的 `Document` 对象，再被切成长度可控、语义完整的 chunk，才能进入嵌入与索引。这一章就钻进 `haystack/components/converters/` 和 `haystack/components/preprocessors/` 两个目录，看 Haystack 如何用一套统一契约把"万物转 Document"，又如何用四种切分单位、overlap 与清洗规则把长文档变成可索引的片段。

我们关心的不只是"做了什么"，更是"为什么这样设计"：统一的 `Document` 契约让上下游彻底解耦，转换器换一个、切分器换一个，Pipeline 的连线不动；overlap 则是为了保护跨边界的语义不被切碎。

## 转换器的统一输入契约

打开任意一个转换器——`PyPDFToDocument`、`TextFileToDocument`、`MarkdownToDocument`、`HTMLToDocument`、`CSVToDocument`、`JSONConverter`——它们的 `run` 方法签名几乎一模一样：

```python
def run(self, sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None) -> dict[str, list[Document]]:
```

这就是 Haystack 转换层的"宪法"：输入永远是 `sources`（一个可以混装字符串路径、`Path` 对象、`ByteStream` 的列表）加上可选的 `meta`，输出永远是 `{"documents": [...]}`。这种刻意的一致性，正是让 `FileTypeRouter` → 各转换器 → `DocumentJoiner` 能自由拼接的前提。

三种异构的输入类型由 `haystack/components/converters/utils.py` 里的 `get_bytestream_from_source` 统一归一：

```python
def get_bytestream_from_source(source, guess_mime_type=False) -> ByteStream:
    if isinstance(source, ByteStream):
        return source
    if isinstance(source, (str, Path)):
        bs = ByteStream.from_file_path(Path(source), guess_mime_type=guess_mime_type)
        bs.meta["file_path"] = str(source)
        return bs
    raise ValueError(f"Unsupported source type {type(source)}")
```

它把一切收敛成 `ByteStream`：如果本来就是 `ByteStream` 就原样返回（这让"内存里的字节"和"磁盘上的文件"对转换器一视同仁，便于流式与测试）；如果是路径，则读盘并把 `file_path` 塞进 `ByteStream.meta`。这一行 `bs.meta["file_path"] = str(source)` 看似不起眼，却是后续所有转换器能记录"这个 Document 来自哪个文件"的源头。

### meta 的归一与三层合并

`meta` 参数允许三种形态：`None`、单个 dict、dict 列表。`normalize_metadata` 把它们统一成"与 sources 等长的 dict 列表"：

```python
def normalize_metadata(meta, sources_count) -> list[dict[str, Any]]:
    if meta is None:
        return [{}] * sources_count
    if isinstance(meta, dict):
        return [meta] * sources_count
    if isinstance(meta, list):
        if sources_count != len(meta):
            raise ValueError("The length of the metadata list must match the number of sources.")
        return meta
    ...
```

单 dict 会被广播到所有源，列表则要求长度严格匹配（否则报错）。归一之后，`run` 里用 `zip(sources, meta_list, strict=True)` 逐对处理——`strict=True` 确保两边长度对齐时才迭代，这是 Python 3.10+ 防止隐性截断的写法。

每个转换器在产出 Document 前都会做一次三层 meta 合并，以 `PyPDFToDocument` 为例：

```python
merged_metadata = {**bytestream.meta, **metadata}
if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
    merged_metadata["file_path"] = os.path.basename(file_path)
document = Document(content=text, meta=merged_metadata)
```

合并顺序是"先 `ByteStream` 自带的 meta，再用户传入的 `metadata` 覆盖"——用户显式传的优先级更高。`store_full_path` 默认 `False`，于是 `file_path` 会被 `os.path.basename` 截成纯文件名，避免把本地绝对路径（可能含用户名等隐私）写进索引。`MarkdownToDocument` 在中间还多插了一层 `frontmatter`：`{**bytestream.meta, **frontmatter, **metadata}`，把 YAML 头解析进 meta。`JSONConverter` 则插入 `extra_meta`：`{**bytestream.meta, **metadata, **extra_meta}`。三个转换器的合并次序略有差异，但骨架完全一致——这正是"统一契约 + 局部扩展"的体现。

### 健壮性：跳过坏文件而非崩溃

转换器对单个源的失败一律采取"记日志、`continue`、不中断整批"的策略。`get_bytestream_from_source` 抛错、解码失败、`PdfReader` 解析失败，都会 `logger.warning(...)` 后跳过当前源。PDF 转换还专门处理"提取出空文本"的情形：`text.strip() == ""` 时会告警但仍产出一个空 Document。这种设计让一批上千个文件中的几个坏文件不会拖垮整个索引任务。

## 各格式转换器的"翻译"细节

虽然契约统一，每种格式的"翻译"逻辑各有讲究：

- **`TextFileToDocument`**：最简单，`bytestream.data.decode(encoding)`。编码优先级是"`ByteStream.meta` 里的 `encoding` > 构造参数 `encoding`（默认 `utf-8`）"，让上游可以按文件指定编码。
- **`PyPDFToDocument`**：用 pypdf 的 `PdfReader` 逐页 `extract_text`，关键一行是 `return "\f".join(texts)`——**用换页符 `\f` 连接各页**。这个 `\f` 不是随意选的：它正是下游 `DocumentSplitter` 识别"页"的依据，把"页码"信息以字符形式编码进了 content。支持 `PLAIN` 与实验性 `LAYOUT` 两种提取模式 [[pypdf]](https://pypdf.readthedocs.io/)。
- **`MarkdownToDocument`**：用 `markdown-it-py` + `mdit_plain` 的 `RendererPlain` 把 Markdown 渲染成纯文本；`extract_frontmatter=True` 时用正则 `_FRONTMATTER_PATTERN` 抓取文件开头的 `--- ... ---` YAML 块，`yaml.safe_load` 解析后并入 meta，并从 content 里切掉。
- **`HTMLToDocument`**：委托给 `trafilatura.extract` 做正文抽取（自动剥离导航、广告等噪声），`extraction_kwargs` 可控制输出格式、是否保留表格/链接。
- **`CSVToDocument`**：两种模式。默认 `file` 模式把整张 CSV 原文当成一个 Document 的 content；`row` 模式用 `csv.DictReader` 把每一行变成一个 Document，`content_column` 指定哪一列做 content，其余列进 meta（遇到与 `file_path`/`encoding` 等键冲突时自动加 `csv_` 前缀去重），并写入 `row_number`。
- **`JSONConverter`**：用 jq 表达式 `jq_schema` 过滤、`content_key` 指定 content 字段、`extra_meta_fields`（可为 `"*"`）抽取附加 meta，一个 JSON 文件可拆出多个 Document。

### `OutputAdapter`：不止于格式转换

`output_adapter.py` 里的 `OutputAdapter` 是个特别的"转换器"——它转换的不是文件，而是组件间流动的数据形状。它用 Jinja 模板（如 `{{ documents[0].content }}`）把一个组件的输出改造成下一个组件需要的形状，并在构造时从模板里反推出输入名（`_extract_template_variables_and_assignments`）。安全上默认走 `SandboxedEnvironment` + `StrictUndefined`，只有显式 `unsafe=True` 才用 `NativeEnvironment` 放开任意代码执行——这是连接异构组件的"胶水"，体现了 Haystack"用模板而非硬编码适配"的解耦思路。

### `MultiFileConverter`：一站式多格式入口

`multi_file_converter.py` 把上述转换器封装成一个 `@super_component`：内部建了一条小 Pipeline，先用 `FileTypeRouter` 按 MIME 类型分流，再连到对应转换器，最后用 `DocumentJoiner` 汇合。它支持 CSV/DOCX/HTML/JSON/MD/TEXT/PDF/PPTX/XLSX 九种类型，`ConverterMimeType` 枚举把 MIME 串集中管理。注意它对 DOCX 用 `link_format="markdown"`、对 HTML 用 `output_format="markdown"`，统一倾向于 Markdown 风格输出。输出映射里除了 `documents`，还透出 `unclassified` 和 `failed`，让调用方能拿到"没认出类型"和"转换失败"的源——这正是统一契约带来的可组合性红利。

## DocumentSplitter：四种单位的字符级切分

转换完拿到的往往是一篇长文，必须切成 chunk。核心组件是 `document_splitter.py` 的 `DocumentSplitter`，它的默认参数透露了设计取向：

```python
def __init__(self, split_by="word", split_length=200, split_overlap=0,
             split_threshold=0, ..., skip_empty_documents=True):
```

默认按 `word`、每段 200 个单位、不重叠。`split_by` 支持 `word`/`period`/`page`/`passage`/`line`/`sentence`/`function` 七种。前五种本质都是"按某个分隔字符切"，由模块级常量映射决定：

```python
_CHARACTER_SPLIT_BY_MAPPING = {"page": "\f", "passage": "\n\n", "period": ".", "word": " ", "line": "\n"}
```

注意 `page` 对应的正是转换器埋下的 `\f`——转换层与切分层在这里完成了一次隐式的"接力"。

`_split_document` 是分派中枢：`sentence` 或 `respect_sentence_boundary` 走 NLTK 句子路径；`function` 走自定义函数；其余走 `_split_by_character`。后者先按分隔符 `split`，再把分隔符补回除最后一段外的每一段（保证拼接后内容无损）：

```python
units = doc.content.split(split_at)
for i in range(len(units) - 1):
    units[i] += split_at
```

### `_concatenate_units`：滑动窗口与 split_threshold

切成最小单位后，用 `more_itertools.windowed` 做滑动窗口聚合：

```python
segments = windowed(elements, n=split_length, step=split_length - split_overlap)
```

窗口大小 `split_length`、步长 `split_length - split_overlap`——这是 overlap 的实现核心：步长小于窗口，相邻 chunk 就会重叠 `split_overlap` 个单位。`split_threshold` 则防止产生过小的尾段：当某个窗口的单位数少于阈值且已有前序 split 时，它会被并进上一段（`text_splits[-1] += txt`）而不是单独成段。

页码追踪藏在循环里：处理过的单位里数 `\f` 的个数累加到 `cur_page`（`page` 模式则直接按单位数计）。这样即使切分单位不是"页"，每个 chunk 也能知道自己起始于第几页。

### meta 的传承：source_id、page_number、_split_overlap

`_split_by_character` 在产出前把原文 ID 写进 meta：`metadata["source_id"] = doc.id`。这是 chunk 回溯到原始 Document 的关键钩子。随后 `_create_docs_from_splits` 给每个 chunk 补上 `page_number`、`split_id`（第几段）、`split_idx_start`（在原文中的字符起始位）：

```python
copied_meta["page_number"] = splits_pages[i]
copied_meta["split_id"] = i
copied_meta["split_idx_start"] = split_idx
```

当 `split_overlap > 0` 时，还会调用 `_add_split_overlap_information`，在相邻两个 chunk 的 `_split_overlap` 字段里互相记录"我和你重叠了哪一段字符区间"（`{"doc_id": ..., "range": (start, end)}`）。它先算出重叠区间，再用 `current_doc.content.startswith(overlapping_str)` 校验确实重叠才记录。这份元数据让下游（如检索后重组上下文）能精确还原 chunk 之间的拼接关系——这正是 overlap"保护跨边界语义"这一意图在数据结构上的落地。

### 句子边界：respect_sentence_boundary 与 NLTK

纯按 word 切会在句子中间切断。`respect_sentence_boundary=True`（仅对 `split_by="word"` 生效，否则被强制关掉并告警）时，先用 NLTK 句子分词器把文本切成句子，再用 `_concatenate_sentences_based_on_word_amount` 按"累计词数不超过 `split_length`"的方式聚合句子，且永远在句子边界处断开。overlap 通过 `_number_of_sentences_to_keep` 计算"下一段要保留上一段末尾几句"，并刻意跳过第一句（`sentences[1:]`），避免下一段与上一段开头完全雷同。

NLTK 路径需要 `warm_up()` 懒加载 `SentenceSplitter`（基于 `nltk` 的 PunktSentenceTokenizer），`sentence_tokenizer.py` 里 `ISO639_TO_NLTK` 把语言代码映射到 NLTK 的语言名，并用 `CustomPunktLanguageVars` 改写正则以"保留切句时的空白"——这样 chunk 拼回去能严丝合缝 [[NLTK]](https://www.nltk.org/)。

## RecursiveDocumentSplitter：分隔符回退链

`recursive_splitter.py` 的 `RecursiveDocumentSplitter` 解决"一刀切单位太死板"的问题。它的默认分隔符是一条有序回退链：

```python
self.separators = separators if separators else ["\n\n", "sentence", "\n", " "]
```

含义是：**优先用最"粗"的边界切，切不动再换更细的**。`_chunk_text` 的逻辑是：若整段已 `<= split_length` 直接返回；否则依次尝试每个分隔符，`"sentence"` 走 NLTK 句子分词，其余用 `re.split` 并把分隔符用捕获组保留回片段。对每个分隔符切出的片段，能塞进当前 chunk 就塞，塞不下就收尾当前 chunk；若单个片段仍超长，就递归用下一个分隔符切它，直到用尽分隔符——此时调 `_fall_back_to_fixed_chunking` 做按 word/char/token 的硬切兜底。

它的长度单位 `split_unit` 比 `DocumentSplitter` 更丰富，支持 `word`/`char`/`token`，其中 `token` 用 tiktoken 的 `o200k_base` 编码计数（`warm_up` 时懒加载）。这让它能直接对齐"模型上下文窗口的 token 预算"。overlap 由 `_apply_overlap` 实现，逻辑比滑动窗口复杂得多：它要在加入上一段尾部 overlap 后重新检查是否超长、超长就再切并把溢出推给下一段，直到收敛。产出的 meta 用 `parent_id`（而非 `source_id`）记录父文档，并同样维护 `split_id`/`split_idx_start`/`_split_overlap`/`page_number`。

## HierarchicalDocumentSplitter：构建多粒度树

`hierarchical_document_splitter.py` 的 `HierarchicalDocumentSplitter` 把同一篇文档切成多种粒度，组成一棵树：根是原文，叶是最小块，中间层互为父子。它接收一组 `block_sizes`（如 `{3, 2}`），内部为每个 block size 建一个 `DocumentSplitter`：

```python
self.splitters[block_size] = DocumentSplitter(
    split_length=block_size, split_overlap=self.split_overlap, split_by=self.split_by)
```

`build_hierarchy_from_doc` 从大块到小块逐层切分，给每个节点写上 `__level`、`__block_size`、`__parent_id`、`__children_ids` 四个层级 meta。如果某层只切出一个块（切不动），就跳过不再下沉。这种结构服务于"父子检索"——先用小块精确命中，再回溯到父块补全上下文，是它复用 `DocumentSplitter` 而非另起炉灶的典型组合式设计。

## DocumentCleaner：清洗规则与执行顺序

切分前后常需清洗。`document_cleaner.py` 的 `DocumentCleaner` 把若干规则按**固定顺序**串起来执行，顺序本身就是设计：

```python
if self.unicode_normalization: text = self._normalize_unicode(...)
if self.ascii_only:            text = self._ascii_only(text)
if self.remove_extra_whitespaces: text = self._remove_extra_whitespaces(text)
if self.remove_empty_lines:    text = self._remove_empty_lines(text)
if self.remove_substrings:     text = self._remove_substrings(...)
if self.remove_regex:          text = self._remove_regex(...)
if self.replace_regexes:       text = self._replace_regexes(...)
if self.remove_repeated_substrings: text = self._remove_repeated_substrings(text)
if self.strip_whitespaces:     text = text.strip()
```

Unicode 归一与 ASCII 化排在最前（先统一字符形态再做模式匹配，避免重音字符漏匹配）。默认开启 `remove_empty_lines` 与 `remove_extra_whitespaces`：前者按 `\f` 分页后逐行剔除空行；后者用 `re.sub(r"\s\s+", " ", text).strip()` 压缩连续空白。所有按页操作都以 `\f` 分割再 `\f` 拼回——再次印证 `\f` 是 Haystack 里贯穿转换、切分、清洗的"页"通用语。

最有意思的是 `remove_repeated_substrings`（默认关）：它要去掉每页都重复出现的页眉页脚。`_find_and_remove_header_footer` 取每页前/后 300 字符，用 `_find_longest_common_ngram` 找所有页共有的最长 n-gram（n 在 3~30 之间），用集合求交集（`reduce(set.intersection, ...)`）定位重复串再替换掉。文档注释也诚实地标注了局限：它靠精确匹配，能去掉"Copyright 2019 by XXX"这类固定页脚，但识别不了"Page 3 of 4"这种带变量的页码。清洗后默认生成新 ID（`keep_id=False` 时 `id=""` 触发 Document 重新计算 ID），保留 `blob`/`score`/`embedding` 等其余字段。

## 本章小结

- Haystack 转换层有一条统一"宪法"：`run(sources, meta) -> {"documents": [...]}`，让任意转换器在 Pipeline 里可自由替换、拼接。
- `get_bytestream_from_source` 把 `str`/`Path`/`ByteStream` 三类输入归一为 `ByteStream`，并把 `file_path` 写进 meta，是"文件来源"可追溯的起点。
- `normalize_metadata` 把 `None`/dict/list 三种 meta 形态统一成等长列表，配合 `zip(..., strict=True)` 逐源处理；产出时做"ByteStream.meta → 用户 meta"的覆盖式合并，并默认用 `basename` 隐去绝对路径。
- 各格式转换器逻辑各异（pypdf 逐页 `\f` 连接、Markdown 渲染+frontmatter、trafilatura 抽正文、CSV 的 file/row 双模式、JSON 的 jq 过滤），但都遵循同一契约，且对坏文件"告警跳过不中断"。
- `DocumentSplitter` 用 `_CHARACTER_SPLIT_BY_MAPPING` 把 page/passage/period/word/line 映射到分隔字符，`more_itertools.windowed` 以窗口=`split_length`、步长=`split_length - split_overlap` 实现 overlap，`split_threshold` 防止过小尾段。
- 切分会写入 `source_id`/`page_number`/`split_id`/`split_idx_start`/`_split_overlap`，让 chunk 能回溯原文、定位页码与重叠区间——overlap 由此既保护语义又可被精确还原。
- `respect_sentence_boundary` 借 NLTK 句子分词在句子边界处断开，`CustomPunktLanguageVars` 保留切句空白以便无损拼接。
- `RecursiveDocumentSplitter` 用有序分隔符回退链（默认 `["\n\n", "sentence", "\n", " "]`）由粗到细递归切分，支持 word/char/token 三种长度单位（token 走 tiktoken），用 `parent_id` 记录父文档。
- `HierarchicalDocumentSplitter` 复用多个 `DocumentSplitter` 把文档切成多粒度树，写 `__level`/`__parent_id`/`__children_ids`，服务父子检索。
- `DocumentCleaner` 按固定顺序执行清洗（Unicode 归一最先、页眉页脚去重靠最长公共 n-gram），所有按页操作以 `\f` 分割，`\f` 是贯穿转换/切分/清洗的"页"通用语。

## 动手实验

1. **实验一：验证 overlap 与回溯 meta** — 构造一个长字符串 Document，用 `DocumentSplitter(split_by="word", split_length=10, split_overlap=3)` 切分，打印每个 chunk 的 `meta` 中 `source_id`、`split_id`、`split_idx_start` 与 `_split_overlap`，确认相邻 chunk 的重叠区间互相记录、且 `step = split_length - split_overlap` 成立。
2. **实验二：page 单位与 `\f` 接力** — 手工拼一个含多个 `\f` 的字符串当 content（模拟 PDF 多页），分别用 `split_by="page"` 和 `split_by="word"` 切分，对比两者产出的 `page_number`，理解 `_concatenate_units` 里 `count("\f")` 如何累加页码。
3. **实验三：递归回退链对比** — 同一段含 `\n\n`、句子和长无分隔串的文本，分别用 `DocumentSplitter(split_by="word", split_length=20)` 和 `RecursiveDocumentSplitter(split_length=20, separators=["\n\n", "sentence", "\n", " "])` 切分，观察后者如何由粗到细回退、并在用尽分隔符时触发 `_fall_back_to_fixed_chunking`。
4. **实验四：清洗顺序与页眉页脚去重** — 造一个三页（用 `\f` 分隔）、每页都带相同页脚的字符串，用 `DocumentCleaner(remove_repeated_substrings=True)` 清洗，确认页脚被 `_find_longest_common_ngram` 识别并移除；再加入"Page 1 of 3"式页脚，验证它无法被去除，体会该启发式的边界。

> **下一章预告**：转换与切分产出的 `Document` 最终要落到某个仓库里才能被检索。第 7 章我们将解构 `DocumentStore` 抽象——它定义了写入、读取、过滤、删除的统一协议，并深入 `InMemoryDocumentStore` 的内存实现，看 BM25 索引、过滤表达式与 embedding 检索在最朴素的后端里是如何工作的，为后续接入向量数据库打下地基。
