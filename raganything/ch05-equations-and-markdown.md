# 第 5 章 公式抽取与增强 Markdown

前几章我们看到 RAG-Anything 如何调度 MinerU、Docling 等上游解析器把文档拆成 `content_list`。但解析器并非无懈可击：当一份 DOCX 被转成 PDF 再交给 MinerU，或被 Docling 的原生管线吞下时，嵌在正文里的数学公式常常被栅格化成图片，或干脆替换成占位符——结构化的数学内容就这样从下游 RAG 管线里消失了。`omml_extractor.py` 文件头的注释把这个问题说得很直白：Word 把公式存为 `word/document.xml` 里的 `<m:oMath>` 元素，而「inline math is typically rasterized to an image or replaced with a placeholder, so the structured math content is lost to the downstream RAG pipeline」。

本章拆解 RAG-Anything 为弥补这个上游盲区而做的两块针对性增强：一是 `omml_extractor.py`，一个零依赖、纯标准库的工具，直接读 DOCX 的 XML，把 Office Math Markup Language（OMML）树递归翻译成 LaTeX，再补回 `content_list`；二是 `enhanced_markdown.py`，一个多后端、可优雅降级的 Markdown → PDF 转换器。两者都体现了同一种工程哲学：不假设外部依赖一定可靠，用最小代价补齐别人漏掉的那一块。

## 为什么要绕过解析器直接读 OMML

OMML 是 OOXML（ECMA-376）标准里定义的数学标记，Word 用 `<m:oMath>`（行内）和 `<m:oMathPara>`（独立成段的显示公式）承载它[[Office Open XML Math]](https://www.ecma-international.org/publications-and-standards/standards/ecma-376/)。它本质是一棵描述数学结构的 XML 树——分式有 `<m:f>`，上标有 `<m:sSup>`，根号有 `<m:rad>`。这些结构信息一旦经过「DOCX → PDF → 视觉解析」的链路就会丢失，因为 PDF 里只剩像素，解析器再聪明也只能 OCR 出近似结果。

RAG-Anything 的选择是釜底抽薪：DOCX 本身就是个 ZIP 包，公式的结构化定义一直好端端地躺在 `word/document.xml` 里。与其等解析器把图片 OCR 回 LaTeX，不如自己打开 ZIP、定位每个 `<m:oMath>`、把它翻译成 LaTeX。注释还点明了这个转换的目标取向：产出的 LaTeX「is intended to be searchable and human-readable rather than typographically perfect」，对未处理的结构「falls back to the concatenated text content of the element rather than failing」。一句话——RAG 的首要目标是召回（recall），不是像素级渲染。这个「recall over correctness」的契约贯穿整个模块。

## `extract_omml_equations`：把 DOCX 当 ZIP 读

入口函数 `extract_omml_equations` 接收一个 `.docx` 路径，返回按文档顺序排列的公式字典列表。它的实现非常克制：

- 用 `zipfile.ZipFile` 以只读方式打开 DOCX，读取 `word/document.xml`。文件不存在抛 `FileNotFoundError`；不是合法 ZIP 抛 `ValueError`；缺 `word/document.xml`（说明不是合法 DOCX）也抛 `ValueError`。
- 用 `ET.fromstring` 解析 XML，解析失败同样转成 `ValueError`。
- 用 `root.iter(_M + "oMath")` 遍历整棵树。注释特意说明 `iter()` 的「walks the entire tree in document order」特性正是想要的：结果列表的顺序与 Word 屏幕上的渲染顺序一致。

这里 `_M` 是预先拼好的命名空间前缀 `"{" + NS["m"] + "}"`，其中 `NS["m"]` 指向 OMML 的命名空间 URI。模块顶部把 `NS` 写成一个字典，同时声明了 `m`（math）和 `w`（wordprocessingml）两个命名空间，并据此派生 `_M` 与 `_W`。注释解释了为什么不直接用前缀字符串匹配：「we use them to construct fully qualified tag names so that lookups are robust to arbitrary namespace prefix choices in the source XML」——不同工具生成的 DOCX 可能用 `m:` 也可能用别的前缀，但命名空间 URI 是稳定的，用全限定标签名查找才可靠。

有一段防御性逻辑值得一提：函数会跳过「嵌套在另一个 `<m:oMath>` 内部的 `<m:oMath>`」。注释承认 Word 实际上不会这样嵌套（`Word does not actually produce`），但手工改过的 XML 可能会，所以用一段双层遍历检测 `parent_is_omath` 来防止重复抽取。每个公式最终翻译为一个字典，含 `index`（零基序号）、`text`（LaTeX）、`text_format`（恒为 `"latex"`）、`raw_omml`（原始 XML 字符串，留给想自带转换器的调用方）。翻译过程套在 `try/except` 里：一旦 `omml_to_latex` 抛异常，就记一条 warning 并退回 `"".join(elem.itertext())`——再次兑现「失败也要给出可搜索文本」的承诺。

## `omml_to_latex` 与 `_convert`：按节点类型分发的树翻译器

公式翻译的核心是 `omml_to_latex`，它只是对私有函数 `_convert` 的一层薄封装。`_convert` 是整个模块的发动机，采用的是经典的 **AST visitor / 递归下降** 模式——和编译器遍历语法树的思路如出一辙：

```python
def _convert(element):
    tag = _local_name(element.tag)
    handler = _HANDLERS.get(tag)
    if handler is not None:
        return handler(element)
    # 通用兜底：拼接子节点转换结果，再加自身文本
    ...
```

`_local_name` 把全限定标签名（如 `{...}oMath`）里的命名空间剥掉，只留本地名 `oMath`，再拿它去查 `_HANDLERS` 这张「本地名 → handler 函数」的字典。命中就交给对应的 `_h_*` handler，每个 handler 负责一种 OMML 结构并自行递归处理它的子节点；没命中就走通用兜底——拼接所有子节点的转换结果加上自身文本。这种「按节点类型查表分发」的设计让新增一种数学结构变得极简单：写一个 `_h_xxx` 函数、在 `_HANDLERS` 里加一行映射即可，无需改动调度逻辑。

辅助函数也都围绕这个模式服务：`_escape_text` 把 Word 爱用的 Unicode 数学符号（查 `_SYMBOL_TO_LATEX` 表，如 `×` → `\times`、`∞` → `\infty`）换成 LaTeX 命令，查不到就原样透传，保证 `\alpha` 和 `α` 都仍可搜索；`_first_child` / `_children_by_tag` 用全限定名取子元素；`_convert_children` 遍历子节点递归——它还专门容忍 `None` 输入，注释解释这是为了应对畸形 DOCX（缺 `m:num` 的分式、缺 `m:e` 的根号），返回 `""` 而不是抛异常，继续守住 recall 契约。

## 几个具体 handler 的翻译规则

把抽象的分发器落到实处，看几个 handler 怎么把 OMML 结构译成 LaTeX：

- **分式 `_h_fraction`**（标签 `f`）：取 `m:num`（分子）和 `m:den`（分母），拼成 `\frac{<分子>}{<分母>}`。
- **上标 `_h_superscript`**（`sSup`）：取基 `m:e` 和上标 `m:sup`，拼成 `{<base>}^{<sup>}`；对称地 `_h_subscript`（`sSub`）产出 `{<base>}_{<sub>}`，`_h_sub_superscript`（`sSubSup`）产出 `{<base>}_{<sub>}^{<sup>}`。还有 `_h_pre_sub_superscript`（`sPre`，前置上下标）会把脚标放到基的左侧：`{}_{<sub>}^{<sup>}{<base>}`。
- **根号 `_h_radical`**（`rad`）：取次数 `m:deg` 和被开方 `m:e`。若 `deg` 存在且非空，产出 `\sqrt[<deg>]{<base>}`（如立方根），否则退化为 `\sqrt{<base>}`。
- **大算符 `_h_nary`**（`nary`，求和、积分等）：运算符字符存在 `m:naryPr/m:chr/@m:val`。注释引用 ECMA-376 §22.1.2.74 说明——`m:chr` 缺省时运算符默认为积分号，所以代码把 `op` 初始化为 `\int`。拿到 `val` 后去查 `_NARY_OPERATORS` 表（`∑` → `\sum`、`∏` → `\prod`、`∫` → `\int` 等十几个）；查不到时**保留原始 Unicode 字符**而非硬塞 `\int`，注释强调这样能让冷僻算符（`\bigsqcup` 之类）对下游 RAG 依然可搜索。随后按需附上 `_{<sub>}`、`^{<sup>}` 和算符主体。
- **括号 `_h_delimiter`**（`d`）：开/闭/分隔字符分别取自 `m:dPr` 的 `begChr`/`endChr`/`sepChr`，缺省为 `(`、`)`、`,`。多个 `m:e` 子项用分隔符连接，外侧字符再过一道 `_delim_to_latex` 映射——它把 `{` 转义成 `\{`、`⟨` 转成 `\langle`、`⌊` 转成 `\lfloor`，空字符则映射成 `.`（对应 LaTeX 的不可见定界符）。
- **矩阵 `_h_matrix`**（`m`）：遍历每个 `m:mr`（行），行内每个 `m:e`（单元格）用 ` & ` 连接，各行用 ` \\ ` 连接，最后包进 `\begin{matrix} ... \end{matrix}`。

此外还有 `_h_bar`（`m:bar`，按 `pos` 产出 `\overline`/`\underline`）、`_h_acc`（`m:acc`，按重音字符查表产出 `\hat`/`\tilde`/`\vec` 等，缺省 `\hat`）、`_h_function`（`m:func`，已知函数名如 `sin`/`log`/`lim` 译成 `\sin{...}` 控制序列，否则退为 `name(arg)`）、`_h_group_chr`（识别 `⏞`/`⏟` 产出 `\overbrace`/`\underbrace`）等。`_HANDLERS` 末尾还把 `e`/`num`/`den`/`sub`/`sup` 等纯容器映射到 `_h_pass_through`（直接递归子节点），把「容器」和「有自身 LaTeX 语义的结构」清晰分开。

## `enrich_content_list_with_docx_equations`：把公式补回 content_list

抽出来的 LaTeX 还要回到主管线才有意义，这就是 `enrich_content_list_with_docx_equations` 的职责：把从 DOCX 抽取的公式合并进解析器已经产出的 `content_list`。它的设计有几个明确取舍：

- **不可变**：函数不修改入参，用 `list(content_list)` 复制后追加，返回新列表。
- **去重**：默认 `deduplicate_existing_equations=True`，先扫描已有的 `type == "equation"` 块收集其 `text` 进 `existing_equations` 集合，抽取出的 LaTeX 若已存在就跳过。注释坦白了局限——它按 `text` 字段精确比较，因此匹配不到解析器发出的占位符（如 `"[FORMULA]"`），也匹配不到 MinerU/Docling 通过图像 OCR 产生的 LaTeX，可能出现同一公式两份的情况；需要更严格的合并策略时可关掉去重再后处理。
- **页码是检索提示而非精确定位**：原始 DOCX 段落不带页码，所以所有抽取的公式都被**追加到列表尾部**，`page_idx` 复制自最后一个块（空列表则为 0）。注释反复说明这对检索足够（公式变成可索引的 LaTeX 文本），但不适合视觉重建；需要精确位置的调用方可借 `raw_omml` 字段把公式拼回正确位置。

追加的块结构固定为 `{"type": "equation", "img_path": "", "text": <latex>, "text_format": "latex", "page_idx": ..., "_source": "omml_extractor"}`，其中 `_source` 标记来源便于后续区分。函数最后记一条 info 日志，报告「补进了 N/M 个公式」。

## `enhanced_markdown.py`：可选依赖 + 运行时探测 + 优雅降级

第二块增强把视角从「抽取」转向「输出」：`enhanced_markdown.py` 把 Markdown 转成 PDF，用于生成排版美观的文档产物。它最值得讲的不是转换本身，而是处理外部依赖的方式——一种「可选依赖 + 运行时探测 + 优雅降级」的范式。

文件顶部用三段 `try/import ... except ImportError` 分别探测 `markdown`、`weasyprint`、`pandoc`，把结果记成 `MARKDOWN_AVAILABLE`、`WEASYPRINT_AVAILABLE`、`PANDOC_AVAILABLE` 三个布尔常量。这意味着这些库**全是可选的**：装哪个用哪个，一个都没装也不会在 import 阶段崩溃。`pandoc` 的探测尤其细致——它不直接 import，而是用 `importlib.util.find_spec("pandoc")` 检查模块是否存在，注释说明这只为检测、并不真正使用该 Python 模块。

`MarkdownConfig` 是个 `@dataclass`，集中管理样式（`page_size`、`margin`、`font_size`、`line_height`）、内容（`include_toc`、`syntax_highlighting`、`image_max_width`）、输出（`output_format`、`output_dir`）和高级选项（`custom_css`、`metadata`）。`EnhancedMarkdownConverter` 在 `__init__` 里调用 `_check_backends` 把可用后端探测一遍并记日志。

`_check_backends` 比 import 探测更进一步：除了汇总三个常量，它还用 `subprocess.run(["pandoc", "--version"], ...)` **真正去跑系统命令**，看 pandoc 可执行文件是否装在系统上，结果记成 `pandoc_system`。捕获 `CalledProcessError` 和 `FileNotFoundError` 两种失败。这把「Python 包级 pandoc」和「系统命令行 pandoc」区分得很清楚——后者才是真正用来转 PDF 的。

## 两个后端与统一入口

具体转换有两个后端实现：

- `convert_with_weasyprint`：先用 `_process_markdown_content` 把 Markdown 转成带内联 CSS 的完整 HTML 文档（启用 `tables`、`fenced_code`、`codehilite`、`toc`、`attr_list`、`footnotes` 等扩展，CSS 来自 `config.custom_css` 或 `_get_default_css` 内置样式），再交给 WeasyPrint 的 `HTML(string=...).write_pdf(output_path)` 渲染。WeasyPrint 专长于 HTML/CSS 排版[[WeasyPrint]](https://weasyprint.org/)。后端不可用时抛 `RuntimeError` 提示安装。
- `convert_with_pandoc`：把 Markdown 写进临时文件，构造 `pandoc ... --pdf-engine=wkhtmltopdf --standalone --toc --number-sections` 命令，用 `subprocess.run(..., timeout=60)` 调用系统 pandoc[[Pandoc]](https://pandoc.org/)。`finally` 块负责清理临时文件。Pandoc 擅长复杂文档。

统一入口是 `convert_markdown_to_pdf`，`method="auto"` 时调用 `_get_recommended_backend` 自动择优；也接受显式的 `"weasyprint"`/`"pandoc"`/`"pandoc_system"`。`convert_file_to_pdf` 在它之上加了文件 IO：读入 Markdown（先试 UTF-8，失败再轮流试 `gbk`/`latin-1`/`cp1252`，全失败抛 `RuntimeError`），未指定输出路径时用 `with_suffix(".pdf")` 推导。

降级策略集中在 `_get_recommended_backend`：**优先 `pandoc_system`（系统 pandoc 可用就返回 `"pandoc"`），其次 `weasyprint`，两者都没有就返回 `"none"`**。这就是「优雅降级」——运行时按实际可用性二选一，缺依赖不报错而是退而求其次。`get_backend_info` 把可用后端、推荐后端和关键配置打包返回，方便诊断；模块底部的 `main()` 还提供了 `--info` 命令行选项，直接打印各后端是否可用。

## 本章小结

- DOCX → PDF → 视觉解析的链路会丢失嵌入公式的结构信息，`omml_extractor.py` 通过直接读 `word/document.xml` 里的 `<m:oMath>` 绕过这个上游盲区，是「针对性增强」的典型。
- 整个模块奉行「recall over correctness」契约：未处理结构退回纯文本、畸形输入返回 `""` 而非抛异常，优先保证可搜索性而非排版精度。
- `extract_omml_equations` 把 DOCX 当 ZIP 读，用 `root.iter` 保证文档顺序，并带有跳过嵌套 `oMath` 的防御逻辑。
- `omml_to_latex`/`_convert` 是按节点本地名查 `_HANDLERS` 表分发的递归下降翻译器，是 AST visitor 模式的工程化体现，扩展一种结构只需加一个 `_h_*` 函数和一行映射。
- 各 handler 覆盖分式、上下标、根号、大算符、括号、矩阵、重音等结构；`_h_nary` 对未知算符保留原 Unicode、`_delim_to_latex` 处理多种定界符，都是为可搜索性服务。
- `enrich_content_list_with_docx_equations` 不可变地把公式追加进 `content_list`，去重按 `text` 精确比较，`page_idx` 仅作检索提示。
- `enhanced_markdown.py` 用三段 `try/import` 把 `markdown`/`weasyprint`/`pandoc` 全做成可选依赖，import 阶段绝不崩溃。
- `_check_backends` 进一步用 `subprocess` 真正探测系统 pandoc，区分「Python 包」与「系统命令」两种可用性。
- 转换器提供 WeasyPrint 与 Pandoc 两个后端，`convert_markdown_to_pdf` 统一入口、`convert_file_to_pdf` 加文件 IO 与多编码兜底。
- `_get_recommended_backend` 按「系统 pandoc → weasyprint → none」顺序优雅降级，缺依赖时退而求其次而非报错。

## 动手实验

1. **实验一：跑通 OMML 抽取** — 准备一份含分式、求和与根号的 `.docx`，调用 `extract_omml_equations(path)`，打印每条结果的 `index`、`text`、`raw_omml`，对照公式核对 `\frac`/`\sum`/`\sqrt` 是否正确，观察文档顺序是否与 Word 一致。
2. **实验二：给翻译器加一种结构** — 在 `omml_extractor.py` 里实现一个新 handler（例如处理 `m:eqArr` 方程组），写好 `_h_eqarr` 后在 `_HANDLERS` 字典加一行映射，构造对应 OMML 片段验证分发是否命中，体会「查表分发」扩展的低成本。
3. **实验三：观察 content_list 富化与去重** — 构造一个已含某公式 `equation` 块的 `content_list`，分别以 `deduplicate_existing_equations=True/False` 调用 `enrich_content_list_with_docx_equations`，对比返回列表长度与 `_source` 标记，验证去重按 `text` 精确比较的边界（如改动一个空格即视为不同）。
4. **实验四：探测后端与降级** — 实例化 `EnhancedMarkdownConverter`，调用 `get_backend_info()` 查看 `available_backends` 与 `recommended_backend`；卸载或重命名 `pandoc` 可执行文件后重新实例化，观察 `_check_backends` 中 `pandoc_system` 翻转、`_get_recommended_backend` 自动降级到 `weasyprint` 或 `none`。

> **下一章预告**：抽取与增强终究是流水线上的工序。第 6 章我们走进 `ProcessorMixin`——RAG-Anything 的文档处理流水线主干，看它如何把解析、内容分类、多模态处理与本章的公式富化串成一条可复用的处理链。
