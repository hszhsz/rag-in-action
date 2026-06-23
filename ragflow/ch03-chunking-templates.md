# 第 3 章 分块模板系统

上一章我们看到 DeepDoc 如何把一份 PDF、DOCX 或 Excel 还原成结构化的文本与表格序列。但解析只是把文档"读懂"，真正决定检索质量的，是接下来这一步：怎样把这些文本切成一个个适合喂给向量模型和倒排索引的 chunk。RAGFlow 在官网上把自己的招牌特性概括为 "Template-based chunking, intelligent and explainable"——按文档类型选择不同的分块模板。本章就来解构这套模板系统：它为什么不用一刀切的固定窗口，而是为论文、书籍、法律条文、问答对、表格各自准备一套切分逻辑；这些模板共享哪些底层机制，又在哪里产生分歧。

我们的主线是 `rag/app/` 目录。每一个文件就是一个"模板"：`naive.py`、`book.py`、`paper.py`、`laws.py`、`qa.py`、`table.py`、`one.py`、`manual.py`、`presentation.py`、`resume.py`、`picture.py`、`audio.py`、`email.py`、`tag.py`。它们对外暴露同一个约定——一个 `chunk(filename, binary, ..., callback, **kwargs)` 函数，返回一组待索引的字典。这种"同名函数、不同实现"的设计，正是模板化的工程落点。读完本章，你会明白为什么 RAGFlow 把分块做成一组可插拔的策略，而不是一个带参数的通用函数。

## 一、模板是怎样被选中的：FACTORY 与统一契约

模板系统的入口在 `rag/svr/task_executor.py`。所有模板模块在文件头部被一次性导入，然后注册进一张名为 `FACTORY` 的字典：

```python
from rag.app import laws, paper, presentation, manual, qa, table, book, resume, picture, naive, one, audio, email, tag

FACTORY = {
    "general": naive,
    ParserType.NAIVE.value: naive,
    ParserType.PAPER.value: paper,
    ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,
    ParserType.LAWS.value: laws,
    ParserType.QA.value: qa,
    ParserType.TABLE.value: table,
    ParserType.RESUME.value: resume,
    ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,
    ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email,
    ParserType.KG.value: naive,
    ParserType.TAG.value: tag,
}
```

键来自 `common/constants.py` 里的 `ParserType` 枚举（`StrEnum`），它的成员值就是这些朴素的字符串：`"naive"`、`"paper"`、`"book"`、`"qa"`、`"table"`、`"laws"`、`"one"` 等。注意几个细节：`"general"` 和 `ParserType.NAIVE.value`（即 `"naive"`）都指向 `naive` 模块——通用模板既是默认值也是兜底；而 `ParserType.KG.value`（知识图谱）同样复用 `naive`，说明在切分这一层，KG 并不需要专门的切法，它的特殊性体现在后续阶段。

选中逻辑就一行。`task_executor.py` 在处理任务时写道：

```python
chunker = FACTORY[task["parser_id"].lower()]
```

`task["parser_id"]` 是用户在知识库配置里为该文档选定的解析方式，小写后直接作为字典键取出对应模块，随后调用它的 `chunk(...)`。这意味着"换模板"对调度层是零成本的——只要新模块实现了同名的 `chunk` 函数并注册进 `FACTORY`，整个管线无需改动。这就是模板化最直接的好处：把"按文档类型切分"的多样性，收敛成一组遵守同一契约的可替换实现。

那么这个契约的产物长什么样？所有模板最终都返回一个 `list[dict]`，每个字典是一个 chunk 的索引文档。它的字段在 `rag/nlp/__init__.py` 的 `tokenize` 函数里被填充：

```python
def tokenize(d, txt, eng):
    from . import rag_tokenizer
    d["content_with_weight"] = txt
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", txt)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
```

`content_with_weight` 是 chunk 的原始正文（保留 HTML 表格标签，用于展示），`content_ltks` 是经过分词的检索字段（先用正则把 `<table>`、`<td>` 等标签替换成空格再分词），`content_sm_ltks` 则是更细粒度的二次分词。每个 chunk 字典里还会带上文档级元数据，比如 `docnm_kwd`（文件名）、`title_tks`（标题分词）。这就是所有模板"殊途同归"的终点：无论怎么切，最后都要被 `tokenize` 包装成统一的索引结构。理解了这个契约，下面看各模板如何在切的过程中各显神通。

## 二、通用模板 `naive.py`：最常用，也最完整

`naive.py` 是默认模板，也是最复杂的一个（一千两百多行）。它的核心函数 `chunk` 的 docstring 一句话讲清了它的哲学：

```python
"""
Supported file formats are docx, pdf, excel, txt.
This method apply the naive ways to chunk files.
Successive text will be sliced into pieces using 'delimiter'.
Next, these successive pieces are merge into chunks whose token number is no more than 'Max token number'.
"""
```

注意这里的"naive"不是贬义，而是指它的策略是通用的、不依赖文档语义结构的：先用分隔符把连续文本切成小片，再把这些小片合并成不超过最大 token 数的 chunk。这正好对应两个关键参数。`chunk` 函数开头给出了默认配置：

```python
parser_config = kwargs.get("parser_config", {"chunk_token_num": 512, "delimiter": "\n!?。；！？", "layout_recognize": "DeepDOC", "analyze_hyperlink": True})
```

`chunk_token_num` 默认 512（控制 chunk 大小上限），`delimiter` 默认 `"\n!?。；！？"`（中英文断句标点加换行）。`layout_recognize` 决定用哪个解析后端，`analyze_hyperlink` 控制是否抓取并递归切分文档里的超链接内容。

### 2.1 按扩展名分流到不同解析器

`chunk` 用一长串 `re.search(r"\.xxx$", filename, re.IGNORECASE)` 判断文件类型，每种类型走不同的解析路径。这是模板内部的"二级分流"：DOCX 走 `Docx()` 类（`naive.py` 内定义，逐段读取 docx body，把段落、图片、表格转成 `(text, image, table)` 三元组）；PDF 根据 `layout_recognize` 从一张 `PARSERS` 字典里选后端：

```python
PARSERS = {
    "deepdoc": by_deepdoc,
    "mineru": by_mineru,
    "docling": by_docling,
    "opendataloader": by_opendataloader,
    "tcadp parser": by_tcadp,
    "paddleocr": by_paddleocr,
    "plaintext": by_plaintext,  # default
}
```

这是一张比 `FACTORY` 更下沉的工厂表：模板决定"怎么切"，而 `PARSERS` 决定"PDF 怎么读"。Excel 走 `ExcelParser`，可选 `html4excel`（把表格转 HTML）；纯文本和代码文件（`.txt/.py/.js/.java/...`）走 `TxtParser`；Markdown 走 `naive.py` 内的 `Markdown` 类（继承 `MarkdownParser`）；HTML、EPUB、JSON、`.doc`（用 tika）各有分支。值得一提的是 Excel 与几种外部 OCR 后端（`mineru`、`docling`、`paddleocr` 等）解析出的内容会被强制设置 `parser_config["chunk_token_num"] = 0`——因为这些后端已经按版面输出了语义完整的块，再按 token 数合并反而破坏结构。这是"解析质量越高，越不需要二次切分"的一个体现。

### 2.2 切分与合并的核心：`naive_merge`

除 DOCX 和 Markdown 有专门的合并路径外，绝大多数格式最终都汇入 `rag/nlp/__init__.py` 的 `naive_merge`：

```python
chunks = naive_merge(sections, int(parser_config.get("chunk_token_num", 128)), parser_config.get("delimiter", "\n!?。；！？"), overlapped_percent)
```

`naive_merge` 接收的是 `(text, position)` 对组成的 sections，输出合并后的字符串列表。它的核心是内部的 `add_chunk`：

```python
def add_chunk(t, pos):
    nonlocal cks, tk_nums, delimiter
    tnum = num_tokens_from_string(t)
    if not pos:
        pos = ""
    if tnum < 8:
        pos = ""
    # Ensure that the length of the merged chunk does not exceed chunk_token_num
    if cks[-1] == "" or tk_nums[-1] > chunk_token_num * (100 - overlapped_percent) / 100.:
        if cks:
            overlapped = RAGFlowPdfParser.remove_tag(cks[-1])
            t = overlapped[int(len(overlapped) * (100 - overlapped_percent) / 100.):] + t
        if t.find(pos) < 0:
            t += pos
        cks.append(t)
        tk_nums.append(tnum)
    else:
        if cks[-1].find(pos) < 0:
            t += pos
        cks[-1] += t
        tk_nums[-1] += tnum
```

逻辑是一个贪心的滑动累加器：维护一个当前 chunk 列表 `cks` 和对应的 token 计数 `tk_nums`。每来一段文本，先算它的 token 数；如果当前 chunk 已经超过 `chunk_token_num * (100 - overlapped_percent) / 100`（即留出 overlap 余量后的阈值），就开一个新 chunk，否则追加到当前 chunk。这就是 docstring 里"merge into chunks whose token number is no more than Max token number"的实现。两个细节很能说明设计意图：其一，当 `tnum < 8`（极短片段）时把 `pos`（位置标记，用于 PDF 回溯原图）清空，避免给碎句记位置；其二，开新 chunk 时会把上一个 chunk 的尾部按 `overlapped_percent` 比例拷贝过来作为前缀（`overlapped[int(len(overlapped) * (100 - overlapped_percent) / 100.):]`），这就是 chunk 之间的重叠（overlap），用来缓解检索时语义被切断的问题。

`naive_merge` 还支持一种"自定义分隔符"模式。如果 `delimiter` 里出现了用反引号包裹的串（如 `` `##` ``、`` `---` ``），代码会优先按这些自定义分隔符硬切，每段直接成块，不再做 token 合并：

```python
custom_delimiters = [m.group(1) for m in re.finditer(r"`([^`]+)`", delimiter)]
has_custom = bool(custom_delimiters)
if has_custom:
    custom_pattern = "|".join(re.escape(t) for t in sorted(set(custom_delimiters), key=len, reverse=True))
    ...
    return cks
```

这给用户一个逃生口：当文档有明确的人工分节标记时，可以绕开 token 合并、严格按标记切。`naive_merge_with_images` 是它的孪生版本，逻辑一致，只是在合并文本的同时用 `concat_img` 把对应的图片纵向拼接，保证图文不分家。

### 2.3 父子分块：`children_delimiter` 与 `split_with_pattern`

`naive.py` 还有一个容易被忽略但很有代表性的机制——父子 chunk。`chunk` 开头解析了 `children_delimiter`：

```python
child_deli = (parser_config.get("children_delimiter") or "").encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
cust_child_deli = re.findall(r"`([^`]+)`", child_deli)
child_deli = "|".join(re.sub(r"`([^`]+)`", "", child_deli))
```

当配置了子分隔符，`tokenize_chunks` 在收尾时不会把整块直接索引，而是调用 `split_with_pattern` 把母块再切成多个子块，并在每个子块上记下完整母块文本 `mom_with_weight`：

```python
if child_delimiters_pattern:
    d["mom_with_weight"] = ck
    res.extend(split_with_pattern(d, child_delimiters_pattern, ck, eng))
    continue
tokenize(d, ck, eng)
res.append(d)
```

这是一种"小块召回、大块上下文"的折中：用更细的子块提高向量检索命中率，同时保留母块作为生成时的完整上下文。它体现了 RAGFlow 在"chunk 该多大"这个永恒矛盾上的工程取舍——既不想块太大稀释相关性，也不想块太小丢失语境。

### 2.4 递归处理嵌入文件与超链接

`chunk` 函数还有一个体现"通用"野心的设计：它会递归地处理文档里的嵌入文件和超链接。函数用 `is_root`（默认 `True`）标记当前是否为顶层调用。顶层调用时，先用 `extract_embed_file(binary)` 抽出 docx/pdf 内嵌的附件，对每个附件再次调用 `chunk(..., is_root=False, ...)` 并把结果并入；当 `analyze_hyperlink` 打开时，对文中超链接拉取 HTML 后同样递归切分。`is_root=False` 既防止无限递归地再去抽嵌入文件，也避免重复抓链接。一份文档因此可能产出来自正文、附件、外链三个来源的 chunk，全部拍平成一个列表返回。

## 三、专用模板的分野：用语义结构代替 token 窗口

`naive` 的策略本质是"无视语义结构、按 token 凑块"。专用模板的价值正在于反过来——它们利用文档类型特有的结构来决定切分边界。下面挑四个最能体现差异的模板。

### 3.1 `paper.py`：按论文章节聚合，摘要独立成块

论文有清晰的结构：标题、作者、摘要、正文分节。`paper.py` 的 `Pdf` 解析类专门做这件事——它在 `__call__` 里用正则 `_begin` 识别 "introduction/abstract/摘要/引言/keywords/..." 这些标志词来定位标题之后、摘要之前的作者区，并把 `title`、`authors`、`abstract`、`sections`、`tables` 拆开返回。`chunk` 函数对摘要做了特殊处理：

```python
if paper["abstract"]:
    d = copy.deepcopy(doc)
    txt = pdf_parser.remove_tag(paper["abstract"])
    d["important_kwd"] = ["abstract", "总结", "概括", "summary", "summarize"]
    d["important_tks"] = " ".join(d["important_kwd"])
    ...
    tokenize(d, txt, eng)
    res.append(d)
```

正如 docstring 所说 "The abstract of the paper will be sliced as an entire chunk, and will not be sliced partly"——摘要整段成块，绝不切碎，还被打上 `important_kwd` 关键词标签提升检索权重。正文则按章节聚合，靠的是两个通用工具 `bullets_category` 和 `title_frequency`：

```python
bull = bullets_category([txt for txt, _ in sorted_sections])
most_level, levels = title_frequency(bull, sorted_sections)
...
for i, lvl in enumerate(levels):
    if lvl <= most_level and i > 0 and lvl != levels[i - 1]:
        sid += 1
    sec_ids.append(sid)
```

`bullets_category`（在 `rag/nlp/__init__.py`）会拿 `BULLET_PATTERN` 这张正则表去匹配各段开头，统计哪一套编号体系命中最多，从而判断文档用的是中文"第 X 章/节/条"、阿拉伯数字"1.2.3"、英文"PART ONE / Chapter / Section"还是 Markdown 的 `#` 标题。`title_frequency` 则找出"最常见的标题层级"作为切分支点（pivot），把相邻同层级标题之间的内容聚成一个 chunk。于是论文的每个 chunk 大致对应一个小节，而不是机械的 512 token。

### 3.2 `book.py`：层级合并 `hierarchical_merge`

书籍更长、层级更深。`book.py` 在拿到 sections 后，先用 `make_colon_as_title` 把以冒号结尾的行升格为标题、用 `remove_contents_table` 去掉目录页，再判断编号体系并走层级合并：

```python
make_colon_as_title(sections)
bull = bullets_category([t for t in random_choices([t for t, _ in sections], k=100)])
if bull >= 0:
    chunks = ["\n".join(ck) for ck in hierarchical_merge(bull, sections, 5)]
else:
    ...
    chunks = naive_merge(sections, parser_config.get("chunk_token_num", 256), ...)
```

注意这里的回退设计：只有当 `bullets_category` 识别出了某套编号体系（`bull >= 0`）时才用 `hierarchical_merge`，否则降级回 `naive_merge`。`hierarchical_merge`（`rag/nlp/__init__.py`）把每段按 `BULLET_PATTERN` 归类到不同层级的桶里（`levels` 列表，深度由参数 `depth` 控制，这里传 5），然后对每个高层级标题用二分查找 `binary_search` 把它名下的子层级内容挂接起来，组装成一个保留完整标题路径的 chunk，最后还有一段把不超过 218 token 的相邻小块继续合并的逻辑。其意图很明确：让一个 chunk 既包含"第 X 章 > 第 Y 节"这样的层级语境，又不至于太大。这是对书籍这种深层级文档的针对性优化，naive 的扁平合并做不到。

### 3.3 `laws.py`：树形合并 `tree_merge` 与"第 X 条"

法律文本结构性最强——条文、款项、编章节井然。`laws.py` 对 DOCX 直接用内部 `Docx` 类，借助 `docx_question_level` 读出每段的标题层级，再用 `Node.build_tree` 构造一棵层级树，最后 `get_tree()` 拍平成 chunk。对 PDF/TXT 等则走 `tree_merge`：

```python
remove_contents_table(sections, eng)
make_colon_as_title(sections)
bull = bullets_category(sections)
res = tree_merge(bull, sections, 2)
```

`tree_merge`（`rag/nlp/__init__.py`）和 `hierarchical_merge` 思路相近，但它用 `Node` 树显式建模层级、并把表格用一个哨兵层级 `10**6` 挂到所在小节下面。法律模板对"第 X 条"还有专门照顾：`not_title` 函数里写着 `if re.match(r"第[零一二三四五六七八九十百0-9]+条", txt): return False`——即"第 X 条"永远被当作标题，不会因为它短或含标点就被误判成正文。这正是 `BULLET_PATTERN` 第一套模式里 `r"第[零一二三四五六七八九十百0-9]+条"` 的用武之地。法律检索讲究"按条召回"，这套以条文为切分单元的逻辑直接服务于该需求。

### 3.4 `qa.py` 与 `table.py`：连 chunk 单位都换了

前面几个模板还在切"文本块"，`qa.py` 和 `table.py` 干脆改变了 chunk 的语义单位。

`qa.py` 把每一对问答当作一个 chunk。它的 docstring 直说："Every pair of Q&A will be treated as a chunk"。Excel 走 `Excel` 类，逐行取前两个非空单元格当 `(q, a)`；TXT/CSV 按制表符或逗号分两列，遇到不成对的行就把内容续接到上一个答案里；Markdown 用 `mdQuestionLevel`（数 `#` 的个数）维护一个问题栈，把标题层级当问题、正文当答案；PDF 则靠 `qbullets_category` 识别问答型项目符号，识别不出来就直接 `raise ValueError("Unable to recognize Q&A structure.")`。每对问答经 `beAdoc` 系列函数包装时，问题进检索字段、答案进展示字段——检索按问题匹配，返回时给出答案，这是 FAQ 类知识库的理想形态。

`table.py` 则把每一行表格记录当作一个 chunk，docstring 写明 "Every row in table will be treated as a chunk"。它做得相当精细：用 `column_data_type` 逐列推断数据类型（int/float/datetime/bool/text），据此把列值写进带类型后缀的字段（`fields_map = {"text": "_tks", "int": "_long", "keyword": "_kwd", "float": "_flt", "datetime": "_dt", "bool": "_kwd"}`），还用拼音库 `Pinyin` 把中文列名转成可作字段名的 ASCII；表头支持多级合并单元格解析。每行最终既生成可读的 `content_with_weight`（形如 `- 列名: 值`），又把结构化字段存入 `chunk_data`（Infinity/OceanBase 引擎）或展开成 ES 字段。这让结构化表格既能做语义检索，又能做带过滤的结构化查询。

### 3.5 `one.py`：另一个极端——整篇即一块

作为对照，`one.py` 走向了 naive 的反面。它的 docstring 是 "One file forms a chunk which maintains original text order"。无论文档多长，它把所有 sections 用换行拼起来，只产出一个 chunk：

```python
doc = {"docnm_kwd": filename, "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))}
doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
tokenize(doc, "\n".join(sections), eng)
return [doc]
```

`one` 适合短文档或需要整体语境的场景。它与 `naive`、`qa`、`table` 一起，构成了一个"chunk 粒度光谱"：从"整篇一块"（one）、"按语义结构成块"（paper/book/laws）、"按 token 凑块"（naive），到"每行/每对成块"（table/qa）。同一份契约下，粒度可以差出几个数量级——这正是模板化要解决的问题。

## 本章小结

- RAGFlow 的"模板化分块"在工程上落地为 `rag/app/` 下一组实现同名 `chunk(...)` 函数的模块，由 `task_executor.py` 中的 `FACTORY` 字典按 `parser_id`（`ParserType` 枚举值）映射选中，调度层换模板零成本。
- `FACTORY` 中 `"general"` 与 `"naive"` 都指向 `naive` 模块，`ParserType.KG` 也复用 `naive`，说明知识图谱在切分层不需要专门策略。
- 所有模板殊途同归：最终都经 `rag/nlp/__init__.py` 的 `tokenize` 写入 `content_with_weight`（原文）、`content_ltks`（分词检索字段）、`content_sm_ltks`（细粒度分词），统一成同一索引结构。
- 通用模板 `naive.py` 的默认参数是 `chunk_token_num=512`、`delimiter="\n!?。；！？"`，核心算法 `naive_merge` 是一个贪心累加器：按 token 上限把连续小片合并成块，并支持 `overlapped_percent` 控制块间重叠。
- `naive.py` 还支持反引号自定义分隔符（命中即硬切、不再合并）、`children_delimiter` 父子分块（子块召回 + 母块 `mom_with_weight` 上下文）、以及对嵌入文件和超链接的递归切分（用 `is_root` 防止无限递归）。
- 高质量解析后端（mineru/docling/paddleocr/Excel 的 html4excel 等）会被强制设 `chunk_token_num=0`，避免对已按版面成块的内容做破坏性二次合并。
- 专用模板用语义结构取代 token 窗口：`paper.py` 把摘要整段成块并打 `important_kwd` 权重、正文按 `bullets_category` + `title_frequency` 找到的标题层级聚合。
- `book.py` 在识别出编号体系时用 `hierarchical_merge` 做层级合并（否则回退 `naive_merge`），`laws.py` 用 `tree_merge`/`Node.build_tree` 建层级树，并通过 `not_title` 强制把"第 X 条"识别为标题。
- 编号体系识别集中在 `BULLET_PATTERN`：覆盖中文"第 X 编/章/节/条"、阿拉伯数字"1.2.3"、英文"PART/Chapter/Section/Article"、Markdown `#` 等五套模式。
- `qa.py` 把每对问答当一个 chunk（问题进检索字段、答案进展示字段），`table.py` 把每行记录当一个 chunk 并按 `column_data_type` 推断类型存入带后缀的结构化字段，`one.py` 则整篇成一块——共同构成跨数量级的 chunk 粒度光谱。

## 动手实验

1. **实验一：观察 naive 的 token 合并** — 打开 `rag/nlp/__init__.py` 的 `naive_merge`，把 `chunk_token_num` 改成一个很小的值（如 32）在脑中或用一段构造文本走查 `add_chunk` 的分支，确认"超阈值开新块、未超则追加"的判定式 `tk_nums[-1] > chunk_token_num * (100 - overlapped_percent) / 100.`，再把 `overlapped_percent` 设为 20，观察新块前缀如何从上一块尾部拷贝出重叠区。

2. **实验二：追一份 PDF 的解析后端选择** — 在 `rag/app/naive.py` 的 `chunk` 中找到 PDF 分支，沿着 `layout_recognize` → `normalize_layout_recognizer` → `PARSERS.get(name, by_plaintext)` 读一遍，列出 `PARSERS` 七个后端各自对应的函数；再确认当 `name in ["tcadp","docling","mineru","paddleocr","opendataloader"]` 时 `chunk_token_num` 被置 0 的那行，理解"高质量解析免二次合并"的意图。

3. **实验三：对比四种合并策略** — 并排阅读 `naive_merge`、`hierarchical_merge`、`tree_merge`（均在 `rag/nlp/__init__.py`）和 `book.py`/`laws.py` 里对它们的调用点，用一句话各自概括切分单位（token 窗口 / 层级桶 / 层级树），并解释为什么 `book.py` 要在 `bull >= 0` 时才用 `hierarchical_merge`、否则回退 `naive_merge`。

4. **实验四：验证编号体系识别** — 阅读 `rag/nlp/__init__.py` 顶部的 `BULLET_PATTERN` 五套模式，挑几行示例文本（如"第三条 …""1.2 …""Chapter II …""## 标题"），在脑中跑一遍 `bullets_category` 的命中计数，预测它会返回哪一套模式的下标；再到 `laws.py` 的 `not_title` 里确认"第 X 条"为何永远被当作标题。

> **下一章预告**：模板把文档切成了 chunk，但这一切是在什么时候、由谁触发执行的？下一章《任务执行器与异步管线》将解构 `rag/svr/task_executor.py`——看 RAGFlow 如何用后台异步任务驱动整个文档处理流程，从取文件、调用本章的 `chunk` 函数，到嵌入与写入索引，以及 `chunk_limiter` 这类并发闸门如何保护系统。
