# 第 2 章 深度文档理解（DeepDoc）

上一章我们把 RAGFlow 的运行时拓扑摊开，知道了文档处理发生在那个"面向文档的后台任务执行器"里。本章要钻进这条流水线的第一道、也是决定后面所有检索质量上限的一道工序：**把一份原始文件变成结构化文本**。这件事远没有"调个 PDF 库 `extract_text()`"那么简单——真实世界里的文档有双栏排版的论文、被扫描成图片的合同、跨页的财务报表、嵌着公式和插图的教材，以及那种字体编码已经坏掉、复制出来全是乱码的老 PDF。RAGFlow 把处理这些"脏文档"的能力集中在 `deepdoc/` 模块里，并起了个很有野心的名字：DeepDoc。它的官方定位写在 `deepdoc/README.md` 第一句：面对来自各领域、各种格式、又带着多样检索需求的文档，"准确的分析成为一项极具挑战的任务，DeepDoc 正是为此而生"。

DeepDoc 由两半组成，`deepdoc/README.md` 说得很直白："There are 2 parts in DeepDoc so far: vision and parser." 一半是 `deepdoc/vision`——用计算机视觉的方式"像人一样"看文档：OCR 识字、版面分析（layout recognition）判断哪块是标题哪块是表格、表格结构识别（TSR）还原单元格网格；另一半是 `deepdoc/parser`——按文件格式分派的解析器，把视觉结果或原生文本组织成下游能用的数据结构。本章会先讲清楚 parser 层的整体分派设计，然后用最大的篇幅解构 PDF 解析这条主线（OCR、版面、表格如何协作），再回头看视觉模型各自怎么工作，最后对比 Office/HTML/Markdown 这些"原生格式"在解析哲学上和 PDF 的根本差异。

## parser 层：按格式分派的一组解析器

打开 `deepdoc/parser/__init__.py`，整个 parser 层的对外契约一目了然。它把每个格式的解析器类用一个更短的别名导出：

```python
from .docx_parser import RAGFlowDocxParser as DocxParser
from .excel_parser import RAGFlowExcelParser as ExcelParser
from .html_parser import RAGFlowHtmlParser as HtmlParser
from .json_parser import RAGFlowJsonParser as JsonParser
from .markdown_parser import RAGFlowMarkdownParser as MarkdownParser
from .pdf_parser import PlainParser
from .pdf_parser import RAGFlowPdfParser as PdfParser
from .ppt_parser import RAGFlowPptParser as PptParser
from .txt_parser import RAGFlowTxtParser as TxtParser
```

这里有几个值得停下来看的设计决定。第一，每个格式一个独立类，彼此不继承一个公共基类——它们的相似只体现在"都实现了 `__call__`"这个鸭子类型约定上，而不是靠抽象基类强约束。这其实反映了一个事实：不同格式的解析在本质上差异巨大，PDF 要走视觉流水线，DOCX 直接读 XML，强行抽象出公共父类反而会得到一堆空方法。第二，`pdf_parser.py` 一个文件里导出了两个类——`PdfParser`（即 `RAGFlowPdfParser`）和 `PlainParser`。后者是一条"快速通道"：当文档本身是规整的文字版 PDF、不需要昂贵的 OCR 与版面分析时，可以直接用 `pypdf` 抽文字。第三，DeepDoc 还有一个没在 `__init__.py` 里、但藏在 `pdf_parser.py` 末尾的 `VisionParser`——它把整页 PDF 渲染成图片，交给多模态大模型（VLM）直接"看图说话"。这三条 PDF 路径（视觉流水线 / 纯文本 / VLM）的存在，本身就说明了 DeepDoc 的设计哲学：没有银弹，按文档的实际状况选最划算的那条路。

`deepdoc/vision/__init__.py` 则导出了视觉层的核心积木：`OCR`、`Recognizer`、`LayoutRecognizer`（实际指向 `LayoutRecognizer4YOLOv10`）、`AscendLayoutRecognizer` 和 `TableStructureRecognizer`。注意这一行别名——`from .layout_recognizer import LayoutRecognizer4YOLOv10 as LayoutRecognizer`——它告诉我们 RAGFlow 当前默认的版面识别模型是 YOLOv10 系。

## PDF 解析主线：六道工序的流水线

PDF 解析是 DeepDoc 里最复杂的部分，`deepdoc/README.md` 也承认"由于 PDF 的灵活性，最复杂的就是 PDF parser"。`RAGFlowPdfParser.__call__` 把整条流水线浓缩在十来行里，读懂它就读懂了主干：

```python
def __call__(self, fnm, need_image=True, zoomin=3, return_html=False, auto_rotate_tables=None):
    if auto_rotate_tables is None:
        auto_rotate_tables = os.getenv("TABLE_AUTO_ROTATE", "true").lower() in ("true", "1", "yes")

    self.outlines = extract_pdf_outlines(fnm)
    self.__images__(fnm, zoomin)
    self._layouts_rec(zoomin)
    self._table_transformer_job(zoomin, auto_rotate=auto_rotate_tables)
    self._text_merge()
    self._concat_downward()
    self._filter_forpages()
    tbls = self._extract_table_figure(need_image, zoomin, return_html, False)
    return self.__filterout_scraps(deepcopy(self.boxes), zoomin), tbls
```

这八个步骤可以归并成六道概念上的工序：渲染与取字（`__images__`）、版面识别（`_layouts_rec`）、表格结构识别（`_table_transformer_job`）、文本合并与阅读顺序重排（`_text_merge` + `_concat_downward`）、目录页过滤（`_filter_forpages`）、表格与插图抽取（`_extract_table_figure`），最后 `__filterout_scraps` 把零碎噪声清掉、给每行打上位置标签后输出。返回值是一个二元组：第一项是带位置标签的正文文本，第二项是表格/插图列表。下面逐道拆。

### 工序零：渲染、取字，以及"乱码检测"

`__images__` 做两件事：用 `pdfplumber` 把每一页按 `resolution=72 * zoomin` 渲染成图片（默认 `zoomin=3`，即 216 DPI），同时尝试用 `page.dedupe_chars().chars` 把 PDF 内嵌的字符层抽出来。注意这里用了一个进程级的锁 `LOCK_KEY_pdfplumber`——`pdfplumber` 底层不是线程安全的，RAGFlow 把它包在 `sys.modules` 里的全局锁中串行化，这是个很容易被忽略但在多 worker 环境下必须有的细节。

真正体现"深度理解"的是它对**乱码的主动防御**。很多老旧 PDF（尤其早期的中文标准文档）内嵌了自定义字体，把 CJK 字形映射到了 ASCII 码位或私有区（PUA），直接抽出来的文字层是垃圾。`__images__` 在拿到字符层后会逐页做两类检测：

```python
# Strategy 1: PUA / CID garbling
if self._is_garbled_text(sample_text, threshold=0.3):
    ...
    self.page_chars[pi] = []
    continue
# Strategy 2: font-encoding garbling (CJK mapped to ASCII)
if self._is_garbled_by_font_encoding(page_ch):
    ...
    self.page_chars[pi] = []
```

第一种策略 `_is_garbled_char` 把落在 Unicode 私有区（`0xE000`–`0xF8FF` 等区段）、替换字符 `0xFFFD`、以及 `unicodedata.category` 为 `Cn`/`Cs` 的字符判为乱码，再加上正则 `_CID_PATTERN`（匹配 `(cid:123)` 这种 pdfminer 没映射成功的占位符）；当一页里乱码比例超过阈值就清空该页字符层。第二种策略 `_is_garbled_by_font_encoding` 更精巧：它统计"来自子集嵌入字体（字体名带 `^[A-Z0-9]{2,6}\+` 前缀）的字符比例"，如果子集字体占比超过 0.3、CJK 字符比例低于 0.05、而 ASCII 标点符号比例又高于 0.4，就判定这是"CJK 被映射成 ASCII"的坏编码页。

判定为乱码后，代码做的不是报错，而是**把该页的 `page_chars` 清空**。清空的后果是：后续 `__ocr` 找不到可信的字符层，就会自动走 OCR 重新识字。这是一种"优雅降级"——好 PDF 直接用内嵌文字层（快且准），坏 PDF 自动回退到 OCR（慢但能用）。同样的逻辑在更细的 box 粒度上也存在：`__ocr` 里若一个文本框内的字符乱码占比 `garbled_count / total_count >= 0.5`，就把 `b["text"] = ""` 清空，交给 OCR 兜底。

`__images__` 里还有一处工程上的稳健性设计：函数末尾若 `len(self.boxes) == 0 and zoomin < 9`，会递归调用 `self.__images__(fnm, zoomin * 3, ...)`——也就是说如果一遍下来什么都没识别出来，它会把渲染分辨率翻三倍再试一次，最多试到 `zoomin=9`（648 DPI）。

### 工序一：OCR 与字符层的对齐

`__ocr` 是渲染图与字符层的"缝合点"。它先用 `self.ocr.detect` 在整页图上检测出所有文本框，把框按 `Recognizer.sort_Y_firstly` 以"先上后下、再左到右"的顺序排好，然后把 `pdfplumber` 抽出的每个字符 `c` 用 `Recognizer.find_overlapped` 匹配进对应的检测框：

```python
for c in chars:
    ii = Recognizer.find_overlapped(c, bxs)
    if ii is None:
        self.lefted_chars.append(c)
        continue
    ch = c["bottom"] - c["top"]
    bh = bxs[ii]["bottom"] - bxs[ii]["top"]
    if abs(ch - bh) / max(ch, bh) >= 0.7 and c["text"] != " ":
        self.lefted_chars.append(c)
        continue
    bxs[ii]["chars"].append(c)
```

这个设计的精妙在于：**检测框来自 OCR 模型（定位准），文字内容优先来自 PDF 字符层（识别准）**。两者各取所长——既不盲目相信字符层的版面（PDF 字符的坐标常常乱序），也不浪费已有的高质量文字去硬跑 OCR。只有当某个框里压根没有可用字符（`if not b["text"]`），才把这个框裁出来 `boxes_to_reg.append(b)`，最后一批 `self.ocr.recognize_batch` 走真正的 OCR 识别。这是"能不 OCR 就不 OCR"的成本意识。

值得一提的是合并字符时对空格的处理：`if c["text"] == " " and b["text"]` 时，只有当上一个字符是字母数字或常见标点（`re.match(r"[0-9a-zA-Zа-яА-Я,.?;:!%%]", ...)`）才补一个空格——这是为了避免在中文之间塞进无意义的空格。

### 工序二：版面识别给每个框贴标签

`_layouts_rec` 调用 `self.layouter`（即 `LayoutRecognizer`），把每页图和每页 OCR 框喂进去，结果是给每个文本框打上 `layout_type` 和 `layoutno`。版面识别模型能区分的标签写在 `LayoutRecognizer.labels` 里，共十类，覆盖了 `deepdoc/README.md` 列举的全部基本版面组件：

```python
labels = [
    "_background_", "Text", "Title", "Figure", "Figure caption",
    "Table", "Table caption", "Header", "Footer", "Reference", "Equation",
]
```

版面识别不只是"贴标签"这么简单，它还顺手做了**垃圾过滤**。`LayoutRecognizer.__init__` 里定义了 `self.garbage_layouts = ["footer", "header", "reference"]`，在 `__call__` 的 `findLayout` 里，落入页眉/页脚/参考文献区域的框，如果不满足"页脚出现在页面下方 90% 以外"或"页眉出现在页面上方 10% 以内"这类保护条件，就会被收进 `garbages` 并从结果里弹出。最后还有一层去重式过滤：

```python
ocr_res = [b for b in ocr_res if b["text"].strip() not in garbag_set]
```

`garbag_set` 只收录那些"在多个页面重复出现超过一次"的页眉页脚文本（`if c > 1`）——这是个很聪明的判据，因为真正的页眉页脚必然在多页重复，而正文不会。`findLayout` 的调用顺序也是有意安排的：`["footer", "header", "reference", "figure caption", "table caption", "title", "table", "text", "figure", "equation"]`，先处理要丢弃的，再处理 caption 和 title，最后才是正文与图表，避免标签互相抢占。

这里还藏着一个细节：版面识别把 `equation`（公式）统一改写成 `figure` 来处理——`bxs[i]["layout_type"] = lts_[ii]["type"] if lts_[ii]["type"] != "equation" else "figure"`。也就是说公式在 DeepDoc 里被当成图片来裁剪保留，而不是去尝试识别 LaTeX，这是个务实的取舍。

### 工序三：表格结构识别与"表格自动旋转"

`_table_transformer_job` 是 PDF 流水线里最有"深度文档理解"味道的一步。它先从 `page_layout` 里挑出所有 `type == "table"` 的区域，每个区域加 `MARGIN = 10` 的边距裁成图片，然后做一件普通解析器绝不会做的事——**自动判定并校正表格的旋转方向**。`deepdoc/README.md` 专门为这个功能写了一节，它由环境变量 `TABLE_AUTO_ROTATE` 控制、默认开启。

实现在 `_evaluate_table_orientation`：它把表格图分别旋转 0°、90°、180°、270°，对每个角度跑一次 OCR，用识别置信度算一个综合分：

```python
combined_score = avg_score * (1 + 0.1 * min(total_regions, 50) / 50)
```

也就是说"平均置信度更高、识别出的文本区域更多"的方向得分更高。但选择非 0° 方向有一道很保守的闸门：

```python
if best_angle != 0 and score_0 is not None:
    if not (best_score - score_0 > 0.2 and score_0 < 0.8):
        best_angle = 0
        best_img = table_img
        best_score = score_0
```

只有当某个旋转角的得分比 0° 高出 0.2 以上、**且** 0° 本身的得分低于 0.8 时，才会真正采用旋转。这条规则的设计意图很清楚：避免误判——一张本来就摆正的表格，绝不能因为旋转后某个角度偶然得分略高就被转歪。

确定方向后，TSR 模型 `self.tbl_det`（`TableStructureRecognizer`）在校正后的图上识别表格的行、列、表头、跨页单元格等结构；如果表格被转过，`_ocr_rotated_tables` 还会在旋转后的图上重新 OCR，并通过 `_map_rotated_point` 把识别出的文字坐标"逆旋转"映射回原始页面坐标系。最终，`_table_transformer_job` 给落在表格区域内的每个文本框贴上 `R`（行号）、`H`（表头）、`C`（列号）、`SP`（跨单元格）等标记——这些标记是后面 `construct_table` 还原表格语义的依据。

### 工序四：文本合并与阅读顺序

OCR 出来的是一堆零散的小框，要拼成可读的段落，靠的是 `_text_merge` 和 `_concat_downward`。`_text_merge` 先调 `_assign_column` 用 KMeans 聚类判断每页有几栏（`max_try = min(4, len(bxs))`，用 `silhouette_score` 选最优的列数 `best_k`，再用多数投票决定全局列数），然后在**同一栏、同一版面块**内把垂直距离小于行高 1/3 的相邻框横向合并——这正确处理了双栏论文这类排版：不会把左栏的句子和右栏的句子错误地连在一起。

跨行的纵向拼接交给一个机器学习模型来判断。`_concat_downward`（在带 bbox 的解析路径里被 `_naive_vertical_merge` 等配合使用）会为每一对上下相邻的框 `_updown_concat_features(up, down)` 抽出三十多维特征——包括两框是否同行（`up.get("R") == down.get("R")`）、垂直间距与行高之比、上一框结尾是否是句末标点（`re.search(r"([。？！；!?;+)）]|[a-z]\.)$", up["text"])`）、下一框开头是否是项目符号（`_match_proj`）等等，喂给一个 XGBoost 模型 `self.updown_cnt_mdl` 预测它们是否该拼成一段：

```python
fea = self._updown_concat_features(up, down)
if self.updown_cnt_mdl.predict(xgb.DMatrix([fea]))[0] <= 0.5:
    i += 1
    continue
```

这个模型从 HuggingFace 仓库 `InfiniFlow/text_concat_xgb_v1.0` 下载（文件名 `updown_concat_xgb.model`），而且 `__init__` 里特意把它固定在 CPU 上：`self.updown_cnt_mdl.set_param({"device": "cpu"})`，注释说明"xgboost model is very small; using CPU explicitly"——把宝贵的 GPU 留给 OCR 和版面模型。用一个轻量学习模型来做"段落拼接"这种规则难以穷尽的判断，是 DeepDoc 区别于普通解析器的又一个深度信号。

`_match_proj` 和 `proj_match` 维护着一大串中文章节编号的正则（`第[零一二三四...]+章`、`[0-9]+\.[0-9]+`、`⚫•➢` 等），用来识别"这是一个新的层级标题/列表项"，从而避免把不该连的项目错误地黏成一段。

### 工序五：目录页过滤与零碎清理

`_filter_forpages` 专门处理目录页。它匹配 `(contents|目录|目次|table of contents|致谢|acknowledge)` 这样的标题，找到后用目录条目的前缀去后续页里匹配并删除整段目录；另外它还检测带连点符（`··`）的页，若一页里这种行超过 3 个，就把整页判为目录并删掉。目录对检索是纯噪声，删掉是对的。

最后 `__filterout_scraps` 把剩下的框组织成最终文本。它的核心判据 `usefull` 决定一个框是否"够分量"：有 `layout_type` 的、宽度超过页宽 1/3 的、或高度超过该页平均行高的框才算有用。组完一组行后，只有当"是项目标题"或"平均宽度占页宽 0.35 以上、或绝对宽度大于 200"时才保留，否则当作零碎丢弃。保留下来的每一行都用 `_line_tag` 拼上一个位置标签：

```python
return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format(
    "-".join([str(p) for p in pn]), bx["x0"], bx["x1"], top, bott)
```

这个 `@@页号\tx0\tx1\ttop\tbottom##` 格式贯穿整个 RAGFlow——它把每段文本的物理位置编码进字符串，下游可以用 `remove_tag`（`re.sub(r"@@[\t0-9.-]+?##", "", txt)`）剥掉标签得到纯文本，也可以用 `extract_positions` 反解出坐标、再用 `crop` 方法把对应区域从原页图裁出来。RAGFlow 在前端"溯源高亮"——点一段检索结果跳到 PDF 原文位置——靠的就是这套位置标签。

## 表格的"语义化"：从单元格网格到自然语言句子

表格是 DeepDoc 最见功力的地方。普通解析器抽出表格往往是一堆失去行列关系的散字，而 DeepDoc 的目标是把表格还原成"LLM 能读懂的句子"。`deepdoc/README.md` 说得明确："Along with TSR, we also reassemble the content into sentences which could be well comprehended by LLM."

`TableStructureRecognizer.labels` 定义了 TSR 模型识别的六类结构：`table`、`table column`、`table row`、`table column header`、`table projected row header`、`table spanning cell`。`construct_table` 把这些结构和前面贴了 `R`/`C` 标记的文本框组装成二维网格 `tbl`：先按行号 `R` 用 `sort_R_firstly` 聚行，再按列号 `C` 用 `sort_C_firstly` 聚列（跨页表格则改用 `sort_X_firstly`，因为跨页时列号不连续），最后 `tbl[b["rn"]][b["cn"]].append(b)` 把每个框塞进对应单元格。它还有专门的"单值列/单值行回收"逻辑（`if len(rows) >= 4` 和 `if len(cols) >= 4` 两段），把那种"只有一个孤立单元格"的伪列合并到相邻列里，修正 TSR 的过分割错误。

组好网格后有两种输出。`return_html=True` 时走 `__html_table`，吐出标准 `<table>`，带 `<caption>`、`<th>`、以及从 `arr[0].get("colspan")`/`rowspan` 还原的合并单元格。而更有意思的是 `return_html=False` 时的 `__desc_table`——它把表格"翻译成句子"。它先识别表头行，把每个数据单元格和它对应的表头用连接词拼起来：中文用"的"或"："，英文用 `" for "`，并在句末标注来源：

```python
if cap:
    if is_english:
        from_ = " in "
    else:
        from_ = "来自"
    row_txt = [t + f"\t——{from_}“{cap}”" for t in row_txt]
```

于是一张"地区 / 销售额"的表会被还原成类似"地区：华东；销售额：1200——来自'2023年区域销售表'"这样的句子。这个设计的意图非常清楚：嵌入模型和大模型对自然语言句子的理解远好于对裸表格的理解，把行列关系显式写进句子，检索时"华东的销售额"这样的提问才能命中。

`is_caption` 用一组正则识别表格/图片标题（`[图表]+[ 0-9:：]{2,}`、`(?i)Table\s+\d+` 等），`_extract_table_figure` 则负责把 caption 和最近的表格/图片配对——它用"y 距离平方加 x 距离平方"作为距离度量，把标题归给最近的那张表或图。跨页表格的合并也在这里：若相邻两段表格分属相邻页、且垂直间距没超过 23 倍行高、上一段又不是被 caption/title 隔断的，就把它们拼成一张表。

## 视觉模型：OCR、版面、TSR 如何落地

DeepDoc 的视觉层全部基于 ONNX 模型在本地推理，不依赖云端 API。`Recognizer.__init__` 用 `load_model` 把 `.onnx` 文件加载成 `onnxruntime` 会话，并把已加载模型缓存在全局字典 `loaded_models` 里避免重复加载。`load_model` 会探测 CUDA 是否可用来决定用 `CUDAExecutionProvider` 还是 CPU，并通过 `OCR_INTRA_OP_NUM_THREADS`、`OCR_GPU_MEM_LIMIT_MB` 等环境变量控制线程数和显存上限——这些都是为了在多 worker 共享 GPU 的部署里不至于互相挤爆显存。模型文件优先从本地 `rag/res/deepdoc` 读，找不到就用 `snapshot_download` 从 HuggingFace 仓库 `InfiniFlow/deepdoc` 拉。

OCR 类 `OCR` 把识别拆成检测（`TextDetector`）和识别（`TextRecognizer`）两个子模型，这是 PaddleOCR 系的经典两阶段架构。`self.drop_score = 0.5` 是它的置信度阈值——低于 0.5 的识别结果直接丢弃。它有一个很细致的方向自纠：`get_rotate_crop_image` 在裁出文本框后，如果框的高宽比 `>= 1.5`（疑似竖排或被转了 90°），会把图按原向、顺时针 90°、逆时针 90° 各识别一遍，选置信度最高的那个方向——这是在单个文本框粒度上的旋转纠正，和前面表格级的 `_evaluate_table_orientation` 是同一思路的不同尺度。

版面识别的默认实现 `LayoutRecognizer4YOLOv10` 是个 YOLOv10 检测器。它的 `preprocess` 用经典的 letterbox 方式把图等比缩放并用 `(114, 114, 114)` 灰边填充到模型输入尺寸；`postprocess` 里硬编码了一个很低的阈值 `thr = 0.08`，再按类别分别做 NMS（`nms(class_boxes, class_scores, 0.45)`）。低检测阈值加上后续 `layouts_cleanup` 和按面积重叠 `find_overlapped_with_threshold(thr=0.4)` 的二次筛选，体现了"宁可先多召回、再靠规则收敛"的策略。RAGFlow 还提供了 `AscendLayoutRecognizer`，用华为昇腾 `.om` 模型和 `InferSession` 推理，由环境变量 `LAYOUT_RECOGNIZER_TYPE` 在 `onnx` 和 `ascend` 之间切换——这是为国产 NPU 部署留的口子。

`Recognizer` 基类提供的几个静态几何工具是整条流水线反复调用的基石：`sort_Y_firstly`/`sort_X_firstly` 带容差排序（同一行内的轻微高低差不算换行），`overlapped_area` 算两框的重叠面积比，`find_overlapped_with_threshold` 找重叠超过阈值的框，`layouts_cleanup` 清理同类型重叠的版面块。理解 DeepDoc，很大程度上就是理解这些"框与框之间的几何关系判定"。

## 原生格式：DOCX、PPTX、Excel、HTML、Markdown 的不同哲学

PDF 走的是"渲染成图、用视觉模型还原"的重路径，而 Office 和标记语言格式则完全相反——它们的结构信息本就保存在文件里，直接读取即可。这组对比恰好说明了 DeepDoc"按格式选最划算路径"的原则。

`RAGFlowDocxParser.__call__` 直接用 `python-docx` 遍历 `self.doc.paragraphs`，把每段文字连同它的样式名 `p.style.name` 一起收集——样式名（如 `Heading 1`）保留了标题层级信息，比 PDF 还得靠模型猜标题省事得多。它处理分页的方式也很 hack：检测每个 run 的 XML 里是否含 `lastRenderedPageBreak` 来累加页码。表格则用 `__extract_table_content` 读出二维文本，再交给 `__compose_table_content` 做和 PDF 表格一样的"表头+单元格拼句子"处理——可见"表格语义化"是 DeepDoc 跨格式统一的诉求，连 `blockType` 这个判断单元格内容类型（日期 `Dt`、数字 `Nu`、英文 `En` 等）的函数都在 DOCX 和 TSR 里各实现了一份几乎相同的版本。

`RAGFlowPptParser` 按 shape 的 `(top // 10, left)` 排序来近似阅读顺序，用 `shape_type == 19` 判表格、`== 6` 判组合形状并递归展开，还会识别项目符号（`a:buChar`/`a:buAutoNum`/`a:buBlip`）给文本加缩进前缀。`RAGFlowExcelParser` 最有意思的是它的格式自适应：`_load_excel_to_workbook` 先读文件头四字节，`PK\x03\x04`（zip）或 `\xd0\xcf\x11\xe0`（旧 OLE）才当 Excel，否则当 CSV 用 pandas 读；它还能输出 HTML（`html` 方法，按 `chunk_rows=256` 分块、每块带表头和 `<caption>` 工作表名）或 Markdown（`df.to_markdown`）两种形态。

HTML 和 Markdown 则把"解析"和"分块"揉在了一起。`RAGFlowHtmlParser.parser_txt` 先用 BeautifulSoup 把 `<style>`/`<script>`/注释/内联样式全清掉，递归遍历 DOM，把 `BLOCK_TAGS`（h1–h6、p、div、li、table 等）当作块边界，标题标签 `TITLE_TAGS` 还会被还原成 Markdown 的 `#` 前缀；表格被单独抽出、按 `chunk_token_num` 用 `rag_tokenizer` 计数切分。`RAGFlowMarkdownParser.extract_tables_and_remainder` 则用一组正则把有边框/无边框的 Markdown 表格、以及 HTML 内嵌表格都抠出来单独处理，剩下的正文另行分块。这两个 parser 已经隐隐越界到了下一章"分块模板"的领地——它们提醒我们，parser 和 chunker 的边界在 RAGFlow 里并不是一刀切的。

## 本章小结

- DeepDoc 由 `deepdoc/vision`（OCR / 版面识别 / 表格结构识别）和 `deepdoc/parser`（按格式分派的解析器）两半构成，定位是把各种"脏文档"准确地变成结构化文本。
- parser 层为每种格式提供独立的 `*Parser` 类（见 `deepdoc/parser/__init__.py`），它们靠 `__call__` 的鸭子类型约定统一，而非公共基类；PDF 一种格式就有 `RAGFlowPdfParser`、`PlainParser`、`VisionParser` 三条路径可选。
- PDF 主线 `RAGFlowPdfParser.__call__` 是一条六道工序的流水线：渲染取字 → 版面识别 → 表格结构识别 → 文本合并/阅读顺序 → 目录过滤 → 表格图片抽取，输出"带位置标签的文本 + 表格/图片列表"。
- DeepDoc 对乱码有主动防御：`_is_garbled_text`（PUA/CID 字符）和 `_is_garbled_by_font_encoding`（子集字体把 CJK 映射成 ASCII）两种策略检测到坏字符层后清空它，自动回退到 OCR，是"优雅降级"。
- OCR 框定位 + PDF 字符层取文的"缝合"设计（`__ocr` 中 `find_overlapped`）让 DeepDoc 既用上模型的准确定位，又尽量复用已有的高质量文字，能不 OCR 就不 OCR。
- 表格自动旋转（`_evaluate_table_orientation`，`TABLE_AUTO_ROTATE` 默认开）用四个角度的 OCR 置信度选方向，并用"得分高出 0.2 且 0° 低于 0.8"的保守闸门防止把摆正的表格误转。
- 段落纵向拼接由 XGBoost 模型 `updown_concat_xgb.model`（仓库 `InfiniFlow/text_concat_xgb_v1.0`）依据三十多维特征判定，模型被刻意固定在 CPU 上以省出 GPU。
- 表格被"语义化"成自然语言：`__desc_table` 把单元格和表头用"的/："（英文 `for`/`in`）拼成句子并标注来源 caption，目的是让嵌入与大模型读懂行列关系。
- 视觉模型全部基于本地 ONNX（默认 YOLOv10 版面 + PaddleOCR 系两阶段 OCR），可经 `LAYOUT_RECOGNIZER_TYPE` 切换昇腾 NPU；位置标签格式 `@@页号\tx0\tx1\ttop\tbottom##` 贯穿全系统，支撑前端溯源高亮。
- Office/HTML/Markdown 走"直接读结构"的轻路径，与 PDF 的"渲染+视觉还原"重路径形成对照，但"表格拼句子"这一诉求跨格式统一存在。

## 动手实验

1. **实验一：追踪 PDF 主流水线的六道工序** — 打开 `/tmp/agent_explore/deepdoc/parser/pdf_parser.py`，定位 `RAGFlowPdfParser.__call__`（约 1673 行），按顺序找到它调用的 `__images__`、`_layouts_rec`、`_table_transformer_job`、`_text_merge`、`_concat_downward`、`_filter_forpages`、`_extract_table_figure` 七个方法，给每个方法写一句话说明它对 `self.boxes` 做了什么改动，验证返回值确实是"文本 + 表格"二元组。

2. **实验二：复现乱码检测的判定** — 阅读 `_is_garbled_char`、`_is_garbled_text`、`_is_garbled_by_font_encoding` 三个方法（约 200–316 行），在 Python 里手动构造几个字符串：一个含 PUA 字符（`chr(0xE001)`）、一个含 `(cid:5)`、一个全是 ASCII 标点。逐个调用 `RAGFlowPdfParser._is_garbled_text` 确认它们都返回 `True`，再把 `threshold` 调高观察判定如何变化，理解"清空 page_chars 回退 OCR"的触发条件。

3. **实验三：观察表格如何被翻译成句子** — 在 `/tmp/agent_explore/deepdoc/vision/table_structure_recognizer.py` 找到 `construct_table`（约 157 行）和 `__desc_table`（约 400 行）。对照 `labels` 列表理解六类表格结构，然后顺着 `__desc_table` 里 `de = "的"`、`from_ = "来自"` 的拼接逻辑，在纸上把一个两列"地区/销售额"的小表手工推演成它会输出的句子；再把 `is_english` 设为 `True` 看连接词如何变成 `for`/`in`。

4. **实验四：体验视觉模型测试入口** — 按 `deepdoc/README.md` 的说明，查看 `deepdoc/vision/t_recognizer.py` 与 `t_ocr.py` 两个测试脚本的 `argparse` 选项。重点读 `t_recognizer.py` 支持的 `--mode {layout,tsr}` 与 `--threshold` 参数，对照 `LayoutRecognizer4YOLOv10.postprocess` 里硬编码的 `thr = 0.08`，思考命令行 `--threshold` 与模型内部阈值分别作用在流水线的哪一环。

> **下一章预告**：DeepDoc 把文档"看懂"并抽成结构化文本之后，这些文本还要被切成适合检索与喂给大模型的"块（chunk）"。第 3 章《分块模板系统》将解构 `rag/app/` 目录下那一组按文档类型定制的分块模板——naive、paper、book、laws、manual、qa、table、resume 等各自如何针对不同文档结构选择切分策略，以及它们如何复用本章 DeepDoc 产出的解析结果。
