# 第 3 章 Processor 注册表与多格式解析

在第 2 章我们看到 `QuivrFile` 如何把一份原始文件包装成带 `file_extension`、`metadata` 与异步流接口的统一对象。但是知道"这是一个 PDF"还远远不够——系统真正要回答的问题是：**该用哪个解析器去拆它？** TXT 可以直接读字符，PDF 需要 OCR/版面分析，DOCX 需要 office 解析库，而这些解析能力往往依赖一堆"可选"的第三方包，用户未必都装了。Quivr 的解法是一套**懒加载注册表**：把"扩展名 → 解析器"的映射先以字符串形式登记下来，等到真正要处理某种文件时才动态 import，import 失败就按优先级优雅地退到下一个候选。

本章钻进 `quivr_core/processor/` 目录，拆解三件事：`registry.py` 的注册表与优先级堆机制、`processor_base.py` 的模板方法如何把横切关注点下沉到基类，以及 `SimpleTxtProcessor` / `TikaProcessor` 两个真实解析器的实现。读完你会理解 Quivr 是如何在"支持十几种格式"和"不强制用户装一堆依赖"之间取得平衡的。

## 注册表：只读视图与字符串映射

`registry.py` 的开头定义了三个全局对象。真正存放"已加载并可用"解析器的是私有字典 `_registry: dict[str, Type[ProcessorBase]]`；对外暴露的则是它的只读包装：

```python
_registry: dict[str, Type[ProcessorBase]] = {}
# external, read only. Contains the actual processors that we are imported and ready to use
registry = types.MappingProxyType(_registry)
```

`types.MappingProxyType` 是 Python 标准库提供的只读字典视图——外部代码可以读 `registry["pdf"]`，但无法直接往里塞东西，必须走 `register_processor()` 这个受控入口。这是一个很克制的封装：内部可变、外部只读，避免使用方误改全局状态。

注意一个关键细节：`_registry` 一开始是**空的**。它不是在模块导入时就把所有解析器加载好，而是随着处理请求逐步填充。真正的"扩展名 → 解析器位置"清单存在另一个结构里。

## ProcEntry 与优先级堆

每一条候选解析器用 `ProcEntry` 表示，它是一个**可排序的 dataclass**：

```python
@dataclass(order=True)
class ProcEntry:
    priority: int
    cls_mod: str = field(compare=False)
    err: str | None = field(compare=False)
```

`@dataclass(order=True)` 让 `ProcEntry` 自动获得比较运算符，而 `cls_mod` 和 `err` 都标了 `field(compare=False)`——这意味着比较时**只看 `priority`**。这正是为了把它塞进堆（heap）：`registry.py` 顶部 `from heapq import heappop, heappush`，每个扩展名对应的候选列表就是一个最小堆，`heappop` 总是弹出 `priority` 数值最小的那一条。

这里有个反直觉但重要的约定：**数值越小，优先级越高**。常量 `_LOWEST_PRIORITY = 100` 是"最低优先级"，而内置的高质量解析器会被赋予比 100 更小的数。`cls_mod` 字段存的是解析器类的**模块路径字符串**（如 `"quivr_core.processor.implementations.tika_processor.TikaProcessor"`），而**不是类本身**——这是懒加载的根基：登记时不 import，只记字符串。`err` 则是 import 失败时要打印的提示。

`base_processors` 给出了最基础的两条映射，TXT 与 PDF 各自指向一个内置解析器，优先级都是 `_LOWEST_PRIORITY`：

```python
base_processors: ProcMapping = {
    FileExtension.txt: [ProcEntry(cls_mod=".../SimpleTxtProcessor", err=None, priority=_LOWEST_PRIORITY)],
    FileExtension.pdf: [ProcEntry(cls_mod=".../TikaProcessor", err=None, priority=_LOWEST_PRIORITY)],
}
```

类型别名 `ProcMapping: TypeAlias = dict[FileExtension | str, list[ProcEntry]]` 表明：键可以是 `FileExtension` 枚举或裸字符串，值是一组候选（一个堆）。

## defaults_to_proc_entries：错误即文档

`base_processors` 只覆盖 TXT 和 PDF，其余十几种格式由 `defaults_to_proc_entries()` 批量补齐。它遍历一张 `(扩展名列表, 解析器名)` 的表，把 CSV、DOCX/DOC、XLS/XLSX、PPTX、Markdown/MD/MDX、EPUB、BibTeX、ODT、HTML、Python、Notebook 等格式逐一注册到 `quivr_core.processor.implementations.default.{processor_name}`：

```python
for ext in supported_extensions:
    ext_str = ext.value if isinstance(ext, FileExtension) else ext
    _append_proc_mapping(
        mapping=base_processors,
        file_exts=[ext],
        cls_mod=f"quivr_core.processor.implementations.default.{processor_name}",
        errtxt=f"can't import {processor_name}. Please install quivr-core[{ext_str}] to access {processor_name}",
        priority=None,
    )
```

这里 `errtxt` 的写法值得品味——它不是一句干巴巴的 "import error"，而是直接告诉用户**怎么修**：`Please install quivr-core[{ext}] to access {processor}`。当某个可选依赖缺失时，日志里给出的就是可执行的修复指令。这是"错误即文档"的设计：异常信息本身就是操作手册。

函数末尾还把 **Megaparse** 作为一个"全能解析器"`_append_proc_mapping` 进十几种格式（txt/pdf/docx/pptx/xlsx/csv/epub/html/markdown 等全都有）。由于 Megaparse 是后 append 的，且 `_append_proc_mapping` 在已存在条目时会取 `prev_proc.priority - 1`，它会获得**比已有解析器更高的优先级**（数值更小），从而在装了 megaparse 时被优先选用。注释里作者也坦承这套顺序是手工维护的：`order of this list is important as resolution of get_processor_class depends on it`。

模块加载时执行 `known_processors = defaults_to_proc_entries(base_processors)`，于是 `known_processors` 成为全量的"扩展名 → 候选堆"清单。`_append_proc_mapping` 的内部逻辑也呼应了堆语义：若扩展名已存在，就 `heappop` 出当前最优的 `prev_proc`，据它算出新 entry 的优先级，再把两者一起 `heappush` 回去；若不存在则新建单元素堆。

## get_processor_class：懒加载与优雅降级

`get_processor_class()` 是整个机制的运行时入口，它的逻辑值得逐行看：

```python
if file_extension not in registry:
    if file_extension not in known_processors:
        raise ValueError(f"Extension not known: {file_extension}")
    entries = known_processors[file_extension]
    while entries:
        proc_entry = heappop(entries)
        try:
            register_processor(file_extension, _import_class(proc_entry.cls_mod))
            break
        except ImportError:
            logger.warn(f"{proc_entry.err}. Falling to the next available processor for {file_extension}")
    if len(entries) == 0 and file_extension not in registry:
        raise ImportError(f"can't find any processor for {file_extension}")
cls = registry[file_extension]
return cls
```

它先看 `registry`（已加载缓存）里有没有——有就直接返回，意味着同一扩展名的 import 只发生一次。没有则去 `known_processors` 取候选堆，用 `heappop` **按优先级从高到低**逐个尝试 `_import_class`。某个解析器因缺依赖抛 `ImportError` 时，并不会让整个流程崩溃，而是 `logger.warn` 出那条"错误即文档"的提示，然后**自动 fall 到下一个候选**。只有当堆被耗尽、依然没人能处理时，才抛 `ImportError("can't find any processor")`。

这就是"可选依赖优雅降级"的完整闭环：用户装了 megaparse 就用 megaparse，没装就退回 Tika，PDF 永远有人兜底；而那些需要重型依赖的格式（如 DOCX 需要 office 解析），缺依赖时报错也只在真正处理该格式时才发生，不会拖累其他格式。

## register_processor 与 _import_class

`register_processor()` 是双形态的：当传入的 `proc_cls` 是**字符串**时，它把这条映射 append 进 `known_processors`（供后续懒加载）；当传入的是**类**时，则做强校验后直接写进 `_registry`：

```python
assert issubclass(proc_cls, ProcessorBase), f"{proc_cls} should be a subclass of quivr_core.processor.ProcessorBase"
if file_ext in registry and override is False:
    ...
else:
    _registry[file_ext] = proc_cls
```

`append`、`override` 两个开关控制"已存在时是追加候选还是覆盖",给了用户注册自定义解析器的弹性。

底层的 `_import_class()` 才是真正执行动态导入的地方。它支持 `module:Class` 和 `module.Class` 两种写法（用 `rsplit(":", 1)` 或 `rsplit(".", 1)` 切分），然后 `importlib.import_module(mod_name)` 加载模块，逐级 `getattr` 取到类对象，最后断言它确实是 `type` 且是 `ProcessorBase` 的子类：

```python
mod = importlib.import_module(mod_name)
for cls in name.split("."):
    mod = getattr(mod, cls)
if not issubclass(mod, ProcessorBase):
    raise TypeError(f"{full_mod_path} is not a subclass of ProcessorBase ")
```

注意：`importlib.import_module` 在目标模块的依赖缺失时会抛 `ImportError`，而这正是 `get_processor_class` 那个 `except ImportError` 捕获并降级的来源——动态导入和优雅降级在此精确咬合。`available_processors()` 则简单返回 `list(known_processors)`，供外部探查支持哪些格式。

## ProcessorBase：模板方法与横切关注点下沉

所有解析器都继承自 `processor_base.py` 的 `ProcessorBase(ABC, Generic[R])`。它用**模板方法模式**把"通用流程"与"格式特定逻辑"分离：`process_file` 是固定的模板，`process_file_inner` 是留给子类实现的抽象钩子。

```python
async def process_file(self, file: QuivrFile) -> ProcessedDocument[R]:
    self.check_supported(file)
    docs = await self.process_file_inner(file)
    try:
        qvr_version = version("quivr-core")
    except PackageNotFoundError:
        qvr_version = "dev"
    for idx, doc in enumerate(docs.chunks, start=1):
        ...
        doc.page_content = doc.page_content.replace(" ", "")
        doc.page_content = doc.page_content.encode("utf-8", "replace").decode("utf-8")
        doc.metadata = {
            "chunk_index": idx,
            "quivr_core_version": qvr_version,
            "language": detect_language(...).value,
            **file.metadata,
            **doc.metadata,
            **self.processor_metadata,
        }
    return docs
```

模板里集中了所有"每个解析器都要做、但又跟具体格式无关"的横切关注点：

- **入口校验**：`check_supported(file)` 比对 `file.file_extension` 是否在子类声明的 `supported_extensions` 中，不支持就抛 `ValueError`。
- **chunk 元数据注入**：从 1 开始给每个 chunk 编 `chunk_index`，写入 `quivr_core_version`（版本读不到时降级为 `"dev"`），并调用 `detect_language` 探测语言写入 `language`。
- **内容清洗**：先把 `original_file_name` 拼到正文前缀，再剔除空字节 ` `，最后做一轮 `encode("utf-8", "replace").decode("utf-8")` 兜底非法字符。
- **metadata 合并顺序**：`{**file.metadata, **doc.metadata, **self.processor_metadata}`——后者覆盖前者，意味着解析器自带的 metadata 优先级最高。

把这些逻辑下沉到基类，子类就只需关心"如何把这种格式变成 chunks"。每个子类必须实现抽象属性 `processor_metadata` 和异步方法 `process_file_inner`。返回值统一封装为泛型 `ProcessedDocument[R]`，它带三个字段：`chunks: List[Document]`、`processor_cls: str`、`processor_response: R`——`R` 是协变类型变量 `TypeVar("R", covariant=True)`，让不同解析器能携带各自的原始响应（字符串、None 或 megaparse 的 Document）。

## SimpleTxtProcessor：手写递归切分

最简单的解析器 `SimpleTxtProcessor` 只支持 `FileExtension.txt`。它用 `aiofiles.open` 异步读全文，包成一个 `Document`，再交给模块级函数 `recursive_character_splitter` 切分：

```python
def recursive_character_splitter(doc, chunk_size, chunk_overlap) -> list[Document]:
    assert chunk_overlap < chunk_size, "chunk_overlap is greater than chunk_size"
    if len(doc.page_content) <= chunk_size:
        return [doc]
    chunk = Document(page_content=doc.page_content[:chunk_size], metadata=doc.metadata)
    remaining = Document(page_content=doc.page_content[chunk_size - chunk_overlap:], metadata=doc.metadata)
    return [chunk] + recursive_character_splitter(remaining, chunk_size, chunk_overlap)
```

这是一个**纯字符级**的手写递归切分：先断言 `chunk_overlap < chunk_size`（否则窗口永远前进不了，会无限递归），文本不超过 `chunk_size` 时直接返回单 chunk；否则切下前 `chunk_size` 个字符为一块，剩余部分从 `chunk_size - chunk_overlap` 处开始（即保留 `chunk_overlap` 个字符的重叠），递归处理。它不理会语义边界、不看 token，纯粹按字符切——胜在零依赖、可预测。`processor_metadata` 里还会把 `splitter_config.model_dump()` 一并记下，便于复现切分参数。

## TikaProcessor：通过 Tika server 解析 PDF

PDF 的默认兜底是 `TikaProcessor`，它把解析这件重活外包给一个独立的 **Apache Tika** 服务 [[Apache Tika]](https://tika.apache.org/)。构造函数里 `tika_url` 默认取环境变量 `TIKA_SERVER_URL`，缺省值是 `http://localhost:9998/tika`，类 docstring 还贴心地给出了启动命令 `docker run -d -p 9998:9998 apache/tika`。

解析走 HTTP：`_send_parse_tika` 把文件流以 `PUT` 请求发给 Tika，请求头 `Accept: text/plain` 要求纯文本输出，并带 `max_retries=3` 的重试循环：

```python
async def _send_parse_tika(self, f: AsyncIterable[bytes]) -> str:
    retry = 0
    headers = {"Accept": "text/plain"}
    while retry < self.max_retries:
        try:
            resp = await self._client.put(self.tika_url, headers=headers, content=f)
            resp.raise_for_status()
            return resp.content.decode("utf-8")
        except Exception as e:
            retry += 1
            logger.debug(f"tika url error :{e}. retrying for the {retry} time...")
    raise RuntimeError("can't send parse request to tika server")
```

拿到纯文本后，切分用的是 LangChain 的 `RecursiveCharacterTextSplitter.from_tiktoken_encoder` [[RecursiveCharacterTextSplitter]](https://python.langchain.com/docs/how_to/recursive_text_splitter/)——这是**按 token 而非按字符**切，`chunk_size`/`chunk_overlap` 都以 token 计；处理器还预置了 `tiktoken.get_encoding("cl100k_base")` 编码器，给每个 chunk 的 metadata 写上 `chunk_size`（该 chunk 的 token 数）。与 `SimpleTxtProcessor` 的字符切分形成鲜明对比：一个零依赖纯字符，一个借助 tiktoken 对齐大模型上下文窗口。

## Megaparse：委托给外部库的全能解析器

`MegaparseProcessor` 声明的 `supported_extensions` 几乎涵盖所有文本类格式（txt/pdf/docx/pptx/xlsx/csv/epub/html/markdown…），这也是它在注册表里被赋予高优先级的原因。它的 `process_file_inner` 把活儿完全委托给 megaparse 库 [[Megaparse]](https://github.com/quivrhq/megaparse)：通过 `MegaParseNATSClient` 连上 megaparse 服务，`await client.parse_file(file=file.path)` 拿回解析结果，再用同样的 `RecursiveCharacterTextSplitter.from_tiktoken_encoder` 按 token 切。它把 megaparse 的原始 `MPDocument` 一并放进 `processor_response`，体现了 `ProcessedDocument[R]` 泛型携带原始响应的价值。

## 本章小结

- `registry.py` 用私有 `_registry` 存"已加载"解析器，对外只暴露 `types.MappingProxyType` 包装的只读 `registry`，内部可变外部只读。
- `ProcEntry` 是 `@dataclass(order=True)` 且只按 `priority` 比较，正是为了塞进最小堆；约定**数值越小优先级越高**，`_LOWEST_PRIORITY = 100`。
- 候选解析器以**模块路径字符串** `cls_mod` 登记而非类本身，这是懒加载的根基——登记不 import，处理时才 import。
- `defaults_to_proc_entries` 批量注册十几种格式，`errtxt` 写成 `Please install quivr-core[{ext}] to access {processor}`，把修复指令直接放进错误信息（错误即文档）；Megaparse 以更高优先级 append 进多种格式。
- `get_processor_class` 用 `heappop` 按优先级逐个 `_import_class`，缺依赖抛 `ImportError` 时 `logger.warn` 并降级到下一候选，全失败才报错——可选依赖的优雅降级。
- `_import_class` 用 `importlib.import_module` 动态导入，支持 `module:Class`/`module.Class` 两种写法，并断言结果是 `ProcessorBase` 子类。
- `ProcessorBase.process_file` 是模板方法：先 `check_supported`，再调子类 `process_file_inner`，最后统一注入 `chunk_index`/`quivr_core_version`/`language` 等元数据并清洗 UTF-8——横切关注点下沉基类。
- 返回值统一为泛型 `ProcessedDocument[R]`（chunks + processor_cls + processor_response），协变 `R` 让各解析器携带各自原始响应。
- `SimpleTxtProcessor` 用手写 `recursive_character_splitter` 做纯字符级切分（断言 `chunk_overlap < chunk_size`），零依赖。
- `TikaProcessor` 通过 HTTP PUT 把文件发给 Tika server（默认 `http://localhost:9998/tika`，`max_retries=3`），并用 `RecursiveCharacterTextSplitter.from_tiktoken_encoder` 按 token 切。

## 动手实验

1. **实验一：观察懒加载与降级日志** — 在 Python 中打开 `quivr_core` 的 logger（`logging.getLogger("quivr_core").setLevel(logging.DEBUG)`），调用 `from quivr_core.processor.registry import get_processor_class; get_processor_class("docx")`。若未安装相应可选依赖，观察控制台打印的 `Please install quivr-core[...]` 警告以及 fall 到下一候选的行为；装上依赖后再跑一次，确认不再降级。

2. **实验二：打印优先级堆** — 导入 `from quivr_core.processor.registry import known_processors`，挑 `FileExtension.pdf` 与 `FileExtension.txt`，把每个扩展名对应的候选列表打印出来（`[(e.priority, e.cls_mod) for e in known_processors[ext]]`），验证 Megaparse 的 `priority` 是否比 Tika/SimpleTxt 更小（更优先）。

3. **实验三：复现纯字符递归切分** — 单独 import `recursive_character_splitter`，传入一个超过 `chunk_size` 的 `Document`，用不同 `chunk_size`/`chunk_overlap` 调用并打印每块长度与相邻块的重叠字符，验证重叠规则；再故意传 `chunk_overlap >= chunk_size`，确认断言 `chunk_overlap is greater than chunk_size` 触发。

4. **实验四：注册自定义 Processor** — 写一个继承 `ProcessorBase` 的最小子类（实现 `processor_metadata` 和 `process_file_inner`，`supported_extensions=["log"]`），用 `register_processor("log", MyProcessor)` 注册，再调用 `get_processor_class("log")` 确认拿回的就是你的类；随后跑通 `await proc.process_file(file)`，检查返回 chunk 的 metadata 里自动出现了 `chunk_index`、`quivr_core_version` 和 `language`。

> **下一章预告**：本章里 `SplitterConfig` 反复出现却一直被当成黑盒——`chunk_size`、`chunk_overlap` 这两个旋钮究竟如何影响检索质量？第 4 章《切分策略与 SplitterConfig》将钻进切分配置的定义与默认值，对比字符级与 token 级切分的取舍，讲清这两个参数是如何在召回率与上下文完整性之间权衡的。
