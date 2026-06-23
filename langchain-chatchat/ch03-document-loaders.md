# 第 3 章：文档加载器

任何 RAG 系统的第一道工序，都是把五花八门的文件变成统一的文本。PDF 里夹着扫描页、Word 文档里嵌着截图、PPT 里全是图形化的图表——这些内容如果只做"纯文本提取"，会大量丢失藏在图片里的文字。Langchain-Chatchat 的文档加载器层正是围绕这个痛点设计的：它在 LangChain 现成 loader 的基础上，自研了一批带 OCR 能力的中文友好型 loader，确保图片里的文字也能被识别、补回正文。

本章从扩展名到 loader 的注册映射讲起，逐个拆解 `RapidOCRPDFLoader`、`RapidOCRLoader`、`RapidOCRDocLoader`、`RapidOCRPPTLoader` 以及 `FilteredCSVLoader` 的实现，看清这些自研 loader 如何在纯文本之外补回图片中的文字、OCR 引擎如何优雅回退，以及最终它们如何产出 LangChain 的 `Document` 对象。

## 一张表决定一切：扩展名到 loader 的注册映射

整个加载器层的"路由表"是 `knowledge_base/utils.py` 里的 `LOADER_DICT`。它是一个 `loader 类名 -> 扩展名列表` 的字典，把每一种文件后缀绑定到一个 loader：

```python
LOADER_DICT = {
    "UnstructuredHTMLLoader": [".html", ".htm"],
    "TextLoader": [".md"],
    "RapidOCRPDFLoader": [".pdf"],
    "RapidOCRDocLoader": [".docx"],
    "RapidOCRPPTLoader": [".ppt", ".pptx"],
    "RapidOCRLoader": [".png", ".jpg", ".jpeg", ".bmp"],
    ...
}
```

这里有几个值得注意的设计。其一，`.pdf`、`.docx`、`.ppt/.pptx` 以及常见图片格式被显式指向了 `RapidOCR*` 系列自研 loader，而不是 LangChain 自带的 `UnstructuredPDFLoader` 之类——这正是 Chatchat 把 OCR 能力植入加载链路的入口。其二，`FilteredCSVLoader` 被以注释形式预留：`# "FilteredCSVLoader": [".csv"], 如果使用自定义分割csv`，默认仍走 `CSVLoader`，但留好了切换口子。其三，紧随其后的一行 `SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]`，把字典里所有扩展名摊平成"系统支持的文件类型"白名单，后续 `KnowledgeFile` 会用它做校验。

从扩展名反查 loader 的逻辑在 `get_LoaderClass`：它遍历 `LOADER_DICT`，找到包含该扩展名的第一个 loader 类名并返回。注意字典是有序的，所以当同一个扩展名出现在多个 loader 下（比如 `.md` 同时挂在 `TextLoader` 和 `UnstructuredMarkdownLoader`，`.docx` 同时挂在 `RapidOCRDocLoader` 和 `UnstructuredWordDocumentLoader`）时，**靠前者胜出**——这也是为什么 `.md` 实际走 `TextLoader`、`.docx` 实际走 `RapidOCRDocLoader`。

## 从类名到实例：`get_loader` 的动态导入与回退

拿到 loader 类名后，真正实例化交给 `get_loader`。它做了一件关键的事：根据 loader 名字决定**从哪个模块导入**。如果是 `RapidOCRPDFLoader`、`RapidOCRLoader`、`FilteredCSVLoader`、`RapidOCRDocLoader`、`RapidOCRPPTLoader` 这几个自研类，就用 `importlib.import_module("chatchat.server.file_rag.document_loaders")`；否则从 `langchain_community.document_loaders` 导入。

```python
DocumentLoader = getattr(document_loaders_module, loader_name)
```

这套"动态导入 + `getattr`"的写法让加载器层完全由字符串驱动——配置里写个类名就能换 loader，无需改代码。一旦导入失败（比如缺依赖、类名拼错），`except` 分支会打日志，并兜底回退到 `UnstructuredFileLoader`，保证流程不中断。

`get_loader` 还针对不同 loader 注入默认参数：`UnstructuredFileLoader` 默认 `autodetect_encoding=True`；`CSVLoader` 在未指定编码时用 `chardet.detect` 嗅探文件编码，避免中文 CSV 报编码错误；`JSONLoader`/`JSONLinesLoader` 默认 `jq_schema="."` 且 `text_content=False`。这些细节都是为"少踩坑"服务的。

自研 loader 的对外出口在 `document_loaders/__init__.py`，仅四行 `import`，把四个 OCR loader 暴露给上面的动态导入：

```python
from .mydocloader import RapidOCRDocLoader
from .myimgloader import RapidOCRLoader
from .mypdfloader import RapidOCRPDFLoader
from .mypptloader import RapidOCRPPTLoader
```

## OCR 引擎的优雅回退：`get_ocr`

四个自研 loader 里有三个共享同一个 OCR 入口 `document_loaders/ocr.py` 的 `get_ocr` 函数。它的核心是一个 `try/except ImportError`：优先尝试 `rapidocr_paddle`（PaddlePaddle 后端，可用 GPU 加速），失败则回退到纯 CPU 的 `rapidocr_onnxruntime`：

```python
def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    try:
        from rapidocr_paddle import RapidOCR
        ocr = RapidOCR(
            det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda
        )
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr
```

这个设计很务实：装了 paddle 版就吃 GPU、跑得快；没装也不报错，自动落到 ONNXRuntime 版用 CPU。`use_cuda` 默认 `True`，只对检测（det）、方向分类（cls）、识别（rec）三个子模型同时开启 CUDA。两个后端都属于 RapidOCR 项目，输出的调用接口一致——传入图像、返回 `(result, _)`，`result` 中每条是 `[box, text, score]` 结构，因此调用方只取 `line[1]`（文字）即可。值得一提的是 `myimgloader`、`mypdfloader` 用 `get_ocr` 统一回退，而 `mydocloader`、`mypptloader` 里直接 `from rapidocr_onnxruntime import RapidOCR` 硬编码了 ONNX 版，没有走 paddle 回退——这是代码里实际存在的不一致。

关于 RapidOCR 与底层 PaddleOCR、ONNXRuntime 的关系可参阅其官方说明[[RapidOCR 文档]](https://rapidai.github.io/RapidOCRDocs/)。

## `RapidOCRPDFLoader`：文字提取 + 页内图片 OCR

PDF 是最复杂的格式，`mypdfloader.py` 的 `RapidOCRPDFLoader` 继承自 LangChain 的 `UnstructuredFileLoader`，只重写 `_get_elements`。它用 `fitz`（即 PyMuPDF）逐页处理[[PyMuPDF 文档]](https://pymupdf.readthedocs.io/)，核心循环是"先取整页文字，再对页内大图做 OCR"：

```python
text = page.get_text("")
resp += text + "\n"

img_list = page.get_image_info(xrefs=True)
for img in img_list:
    if xref := img.get("xref"):
        bbox = img["bbox"]
        # 过滤掉占页面比例过小的图片
        if (bbox[2]-bbox[0]) / page.rect.width < Settings.kb_settings.PDF_OCR_THRESHOLD[0] \
           or (bbox[3]-bbox[1]) / page.rect.height < Settings.kb_settings.PDF_OCR_THRESHOLD[1]:
            continue
        ...
```

这里的 `PDF_OCR_THRESHOLD` 是性能与召回的平衡阀，默认值在 `settings.py` 中为 `(0.6, 0.6)`，含义是：只有当图片宽度占页面宽度、且高度占页面高度都不低于 60% 时才送去 OCR。小图标、装饰性图片直接 `continue` 跳过，避免在无意义的小图上浪费算力。

对通过阈值的图片，loader 用 `fitz.Pixmap(doc, xref)` 取出像素数据。还有一处细节体现了对真实文档的考量：如果 `page.rotation` 不为 0（页面被旋转过），就先用内部函数 `rotate_img` 把图片反向旋转回正——`rotate_img` 基于 OpenCV 的 `getRotationMatrix2D` 和 `warpAffine`，并重新计算旋转后的画布边界以免裁切。摆正后再调用 `ocr(img_array)`，把识别出的每行文字 `"\n".join` 后追加进 `resp`。整页处理过程用 `tqdm` 显示进度条，按页更新描述。

最后，提取出的纯文本字符串通过 `from unstructured.partition.text import partition_text` 切成结构化元素返回[[unstructured 文档]](https://docs.unstructured.io/)。这一步是所有 `RapidOCR*` loader 的共同收尾：它们都把"自己抽取的文本"丢给 `partition_text`，再由父类 `UnstructuredFileLoader.load()` 包装成 `Document` 列表。这意味着这些 loader 的产物在结构上与其它 Unstructured loader 完全一致，下游切分器无需特判。

## `RapidOCRLoader`：纯图片的 OCR

图片文件的 loader 最简洁。`myimgloader.py` 的 `RapidOCRLoader` 直接把文件路径喂给 `get_ocr` 返回的 ocr 对象——RapidOCR 既能接收 numpy 数组也能直接接收图片路径：

```python
def img2text(filepath):
    resp = ""
    ocr = get_ocr()
    result, _ = ocr(filepath)
    if result:
        ocr_result = [line[1] for line in result]
        resp += "\n".join(ocr_result)
    return resp
```

识别结果同样取 `line[1]`、用换行拼接，再交给 `partition_text`。它服务于 `.png`、`.jpg`、`.jpeg`、`.bmp` 四种格式——对一张图片来说，OCR 出来的文字就是它的全部"内容"。

## `RapidOCRDocLoader`：Word 正文 + 表格 + 嵌入图 OCR

`mydocloader.py` 处理 `.docx`，难点在于 Word 文档的内容是分块结构化的：段落、表格、图片混杂。它用 `python-docx` 打开文档，并自定义 `iter_block_items` 生成器，按文档 body 的子元素顺序依次产出 `Paragraph`（`CT_P`）或 `Table`（`CT_Tbl`），从而**保持正文与表格的原始顺序**。

对每个块分类处理：

- **段落**：先追加 `block.text`；再用 XPath `.//pic:pic` 找出段落内嵌的图片，通过 `.//a:blip/@r:embed` 拿到图片关系 id，用 `doc.part.related_parts[img_id]` 取回图片二进制，确认是 `ImagePart` 后用 PIL 打开、转 numpy 数组送 OCR，识别文字追加进正文。
- **表格**：双层循环遍历 `row.cells` 与 `cell.paragraphs`，把单元格文字逐个追加。

```python
images = block._element.xpath(".//pic:pic")
for image in images:
    for img_id in image.xpath(".//a:blip/@r:embed"):
        part = doc.part.related_parts[img_id]
        if isinstance(part, ImagePart):
            image = Image.open(BytesIO(part._blob))
            result, _ = ocr(np.array(image))
            ...
```

这套实现的价值在于：很多中文 Word 文档会把流程图、公式、截图当图片插入正文，普通文本提取会丢掉这些信息，而 `RapidOCRDocLoader` 能把它们补回来。进度条 `tqdm` 的总量设为段落数加表格数。

## `RapidOCRPPTLoader`：递归遍历形状与组合

`mypptloader.py` 处理 `.ppt`/`.pptx`，用 `python-pptx` 打开演示文稿。PPT 的内容单元是"形状"（shape），一张幻灯片由若干形状构成，因此核心是一个递归函数 `extract_text(shape)`，按形状类型分别处理：

- `has_text_frame`：追加文本框文字；
- `has_table`：遍历表格单元格的段落文字；
- `shape_type == 13`（图片）：用 PIL 打开 `shape.image.blob` 送 OCR；
- `shape_type == 6`（组合）：递归对每个 `child_shape` 调用 `extract_text`，处理嵌套组合。

```python
sorted_shapes = sorted(slide.shapes, key=lambda x: (x.top, x.left))  # 从上到下、从左到右
for shape in sorted_shapes:
    extract_text(shape)
```

遍历前先按 `(top, left)` 对形状排序，模拟"从上到下、从左到右"的人类阅读顺序，让产出的文本顺序更自然。这对 PPT 尤其重要——幻灯片上的形状在文件里的存储顺序往往是杂乱的。

## 收尾与一致性：都回到 LangChain `Document`

四个 OCR loader 殊途同归：抽取出纯文本字符串 → `partition_text(text=text, **self.unstructured_kwargs)` → 由 `UnstructuredFileLoader` 父类包装成 `Document` 列表。它们都不直接 `new Document`，而是借父类机制保证 metadata（如 `source`）的一致性。

## `FilteredCSVLoader`：按列裁剪的结构化加载

CSV 是结构化数据，全表灌进向量库往往噪声太大。`FilteredCSVLoader.py` 继承自 LangChain 的 `CSVLoader`，新增 `columns_to_read` 参数，**只读取指定列**。它重写 `load`/`__read_file`，用 `csv.DictReader` 逐行解析，把选中列拼成 `"列名:值"` 的多行文本作为 `page_content`：

```python
for col in self.columns_to_read:
    if col in row:
        content.append(f"{col}:{str(row[col])}")
    else:
        raise ValueError(f"Column '{self.columns_to_read[0]}' not found in CSV file.")
content = "\n".join(content)
doc = Document(page_content=content, metadata=metadata)
```

每行 CSV 生成一个 `Document`，metadata 记录 `source`（可由 `source_column` 指定，否则用文件路径）和行号 `row`，并可把 `metadata_columns` 里的列写入 metadata 而不进正文。编码处理上，它在 `UnicodeDecodeError` 时按 `autodetect_encoding` 用 `detect_file_encodings` 逐个尝试探测到的编码——对编码混乱的中文 CSV 很友好。注意它是 `LOADER_DICT` 里被注释掉的可选项，需手动启用。

## 加载器如何被调用：`KnowledgeFile`

把这些串起来的是 `utils.py` 里的 `KnowledgeFile`。构造时它取文件扩展名 `self.ext`，先用 `SUPPORTED_EXTS` 校验（不支持则抛 `暂未支持的文件格式`），再 `get_LoaderClass(self.ext)` 算出 `document_loader_name`。`file2docs` 调 `get_loader` 拿到 loader 并 `load()`；如果是 `TextLoader` 还会强制 `encoding = "utf8"`。加载得到的 `docs` 随后交给 `docs2texts` 做切分（CSV 会跳过切分），切分逻辑就是下一章的主题。批量场景下，`files2docs_in_thread` 用线程池并发地对多个文件执行 `file2text`，并以 `status, (kb_name, file_name, docs | error)` 的形式产出结果，单文件失败不影响整体。

## 本章小结

- `LOADER_DICT` 是扩展名到 loader 的注册中心，`get_LoaderClass` 按字典顺序返回首个匹配的 loader 类名，`SUPPORTED_EXTS` 由它摊平而来作为支持白名单。
- `get_loader` 用 `importlib` + `getattr` 字符串驱动地实例化 loader，自研 OCR loader 从 `chatchat.server.file_rag.document_loaders` 导入，其余从 `langchain_community.document_loaders` 导入，导入失败兜底回退 `UnstructuredFileLoader`。
- `get_ocr` 优先用 `rapidocr_paddle`（可走 GPU），缺包时回退 `rapidocr_onnxruntime`（CPU），实现 OCR 引擎的优雅降级；但 doc/ppt loader 实际硬编码了 onnxruntime 版。
- `RapidOCRPDFLoader` 用 PyMuPDF 提取每页文字，并对超过 `PDF_OCR_THRESHOLD`（默认 `(0.6, 0.6)`）的页内大图做 OCR，必要时按 `page.rotation` 用 OpenCV 反向旋转图片。
- `RapidOCRLoader` 直接把图片路径喂给 RapidOCR，把识别文字作为内容。
- `RapidOCRDocLoader` 用 `iter_block_items` 顺序遍历 Word 段落与表格，并通过 XPath 提取段内嵌入图做 OCR，补回截图/流程图里的文字。
- `RapidOCRPPTLoader` 递归遍历形状，区分文本框、表格、图片（`shape_type==13`）、组合（`shape_type==6`），并按 `(top, left)` 排序模拟阅读顺序。
- 四个 OCR loader 都以 `partition_text` 收尾、经 `UnstructuredFileLoader` 父类产出统一的 `Document`，下游无需特判。
- `FilteredCSVLoader` 按 `columns_to_read` 裁剪列、逐行生成 `Document`，是默认被注释的可选增强。
- `KnowledgeFile` 把校验、选 loader、加载、切分串成一条链，`files2docs_in_thread` 提供线程池批量加载且容错。

## 动手实验

1. **实验一：验证扩展名路由** — 在 Python 交互环境导入 `get_LoaderClass`，分别传入 `".pdf"`、`".docx"`、`".md"`、`".csv"`、`".jpg"`，观察返回的 loader 类名，确认 `.docx` 走 `RapidOCRDocLoader` 而非 `UnstructuredWordDocumentLoader`，并解释字典顺序为何决定结果。
2. **实验二：调阈值看 OCR 行为** — 准备一个内含大图与小图标的 PDF，先用默认 `PDF_OCR_THRESHOLD=(0.6,0.6)` 跑 `RapidOCRPDFLoader.load()` 看输出，再把阈值改成 `(0.1,0.1)`，对比被 OCR 的图片数量与耗时变化，体会阈值的性能/召回权衡。
3. **实验三：观察 OCR 引擎回退** — 在未安装 `rapidocr_paddle` 的环境调用 `get_ocr()`，确认它落到 `rapidocr_onnxruntime`；再对一张含中文文字的截图调用返回的 ocr 对象，打印 `result`，看清每条 `[box, text, score]` 结构以及为何代码只取 `line[1]`。
4. **实验四：启用 FilteredCSVLoader** — 取消 `LOADER_DICT` 中 `FilteredCSVLoader` 的注释（或直接实例化），传入 `columns_to_read=["问题","答案"]` 加载一个多列 CSV，打印生成的 `Document`，确认 `page_content` 只含指定列、`metadata` 含 `source` 与 `row`。

> **下一章预告**：文件被加载成 `Document` 之后，紧接着要切成大小合适的片段才能向量化。第 4 章《中文文本切分》将拆解 Chatchat 的 `make_text_splitter` 工厂、中文标点感知的递归切分器，以及中文标题增强 `zh_title_enhance` 等机制，看它如何针对中文语料把"长文本切得既不断句又不超长"。
