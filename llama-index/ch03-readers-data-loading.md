# 第 3 章 数据加载 Readers:从外部数据到统一的 Document

任何 RAG 系统都从一个朴素的问题开始:外部世界的数据五花八门——本地磁盘上的 PDF、Markdown、CSV,远端对象存储里的文件,Notion 页面、Slack 频道、一段内存里的字符串——它们必须先被读进系统,变成一种统一的、可被后续切分、嵌入、检索的内部表示。在 LlamaIndex 中,这个内部表示就是 `Document`,而把异构数据源转换成 `Document` 的那一层,就是 Readers 子系统。它是整条 RAG 链路的真正起点:没有 Reader,后面的 NodeParser、Embedding、Index、Retriever 都无米下锅。

本章聚焦 `llama-index-core` 中的 Readers 抽象与最常用的 `SimpleDirectoryReader`。我们不重复第 1 章已经讲过的 `Document`/`Node` 数据模型,而是从"Reader 如何构造 Document、如何填充 metadata、如何抽象文件系统"这几个角度切入,沿着 `llama_index/core/readers/` 目录下的真实源码逐层展开。

## 一、Reader 的统一契约:load_data 返回 List[Document]

所有 Reader 的根在 `llama-index-core/llama_index/core/readers/base.py` 里的 `BaseReader` 类。它是一个 `ABC`,但有意思的是它并没有把 `load_data` 声明为抽象方法,而是给出了一套以 `lazy_load_data` 为核心的默认实现链:

```python
class BaseReader(ABC):
    def lazy_load_data(self, *args, **load_kwargs) -> Iterable[Document]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide lazy_load_data method currently"
        )

    async def alazy_load_data(self, *args, **load_kwargs) -> Iterable[Document]:
        return await asyncio.to_thread(self.lazy_load_data, *args, **load_kwargs)

    def load_data(self, *args, **load_kwargs) -> List[Document]:
        return list(self.lazy_load_data(*args, **load_kwargs))

    async def aload_data(self, *args, **load_kwargs) -> List[Document]:
        return await asyncio.to_thread(self.load_data, *args, **load_kwargs)
```

这段不到二十行的代码,定义了整个子系统的契约。其设计意图很清晰:Reader 的本质职责只有一个——产出 `Document`。框架把"如何懒加载"(`lazy_load_data`)定为最小可重写单元,然后用四个默认方法围绕它派生出全部能力:`load_data` 只是把懒加载迭代器 `list()` 物化;`aload_data` 用 `asyncio.to_thread` 把同步的 `load_data` 丢进线程池得到一个"伪异步"版本;`alazy_load_data` 同理。换句话说,一个最简单的自定义 Reader 只要重写 `lazy_load_data` 一个方法,就自动获得了同步 `load_data`、异步 `aload_data` 与异步懒加载三种调用方式。这是典型的模板方法模式:用一个抽象点支撑四个入口。

值得注意的是 `load_data` 的返回类型注解恒为 `List[Document]`——这是贯穿全书需要反复确认的统一接口。无论数据来自文件、API 还是字符串,Reader 都必须把它压平成一个 `Document` 列表交出去。`base.py` 还提供了 `load_langchain_documents`,它内部调用 `load_data` 然后对每个文档执行 `d.to_langchain_format()`,体现了 LlamaIndex 对 LangChain 生态的兼容意图,但核心数据通路始终是 `List[Document]`。

`base.py` 中还有几个值得点名的角色。`BasePydanticReader` 同时继承 `BaseReader` 与 `BaseComponent`,通过 `model_config = ConfigDict(arbitrary_types_allowed=True)` 让 Reader 成为可被 Pydantic 序列化的组件,并新增了一个 `is_remote: bool` 字段(`default=False`),用来标记数据是来自远端 API 还是本地文件——这对需要把整个摄入配置序列化保存的场景很关键。`ResourcesReaderMixin` 则是一个更精细的混入:它把"数据源"抽象为"资源(resource)"的概念,定义了 `list_resources`、`get_resource_info`、`load_resource` 等抽象方法,以及 `load_resources`、`list_resources_with_info` 等组合方法。注释里给出的例子很形象——对文件系统 Reader 来说资源就是文件,对 Slack Reader 是频道 ID,对 Notion Reader 是页面。这让上层可以"先列举资源、再按需加载单个资源",而不必一次性 `load_data` 全量拉取。`base.py` 末尾的 `ReaderConfig` 把一个 `BasePydanticReader` 连同它的 `reader_args`/`reader_kwargs` 打包成可序列化对象,其 `read()` 方法本质上就是 `self.reader.load_data(*args, **kwargs)`,用于把"用哪个 Reader、传什么参数"作为配置持久化。

最朴素的 Reader 实现可以在 `string_iterable.py` 里看到。`StringIterableReader` 继承 `BasePydanticReader`,它的 `load_data(self, texts: List[str])` 只是遍历字符串列表,对每个字符串 `results.append(Document(text=text))`,然后返回。没有 metadata、没有文件系统,这恰好把 Reader 的最小内核暴露出来:输入异构数据,输出 `Document` 列表。它甚至连 `lazy_load_data` 都不走,直接重写了 `load_data`——这是被允许的,因为 `load_data` 本就不是抽象方法。

## 二、SimpleDirectoryReader:递归扫描与按扩展名分发

真正承担"读一整个目录"重任的是 `llama-index-core/llama_index/core/readers/file/base.py` 中的 `SimpleDirectoryReader`。它的类声明就透露了它的多重身份:

```python
class SimpleDirectoryReader(BaseReader, ResourcesReaderMixin, FileSystemReaderMixin):
```

它既是 `BaseReader`,又混入了 `ResourcesReaderMixin`(因此支持把每个文件当作资源单独加载),还混入了同文件中定义的 `FileSystemReaderMixin`(提供 `read_file_content` 抽象,用于直接读取字节内容)。

### 构造期:确定要读哪些文件

`__init__` 的参数表几乎就是这个 Reader 的全部行为开关,逐一对照源码默认值:`input_dir` 与 `input_files` 二选一,构造器开头就有 `if not input_dir and not input_files: raise ValueError("Must provide either `input_dir` or `input_files`.")`;`exclude_hidden=True`(默认排除点文件)、`exclude_empty=False`、`recursive=False`(默认不递归)、`encoding="utf-8"`、`errors="ignore"`、`filename_as_id=False`、`raise_on_error=False`。还有几个 `Optional` 参数:`required_exts`、`file_extractor`、`num_files_limit`、`file_metadata`、`fs`,默认都是 `None`。

构造期最关键的动作是确定文件清单。如果传了 `input_files`,它逐个校验 `self.fs.isfile(path)` 并收集;如果传了 `input_dir`,先校验 `self.fs.isdir(input_dir)`,再调用 `self._add_files(self.input_dir)` 扫描目录。`_add_files` 是递归扫描的核心:

```python
depth = 1000 if self.recursive else 1
for root, _, files in self.fs.walk(str(input_dir), topdown=True, maxdepth=depth):
    for file in files:
        c += 1
        if limit and c > limit:
            break
        file_refs.append(_Path(root, file))
```

这里有几个设计细节:递归与否被翻译成 `maxdepth`——非递归时 `depth=1`(只看当前层),递归时用一个魔数 `1000` 作为实际上的"无限深度";`num_files_limit` 通过计数器 `c` 在遍历时硬截断;遍历用的是 `self.fs.walk` 而非 `os.walk`,这为后面的 fsspec 抽象埋下伏笔。收集到 `file_refs` 后,代码再逐个判定是否跳过:`skip_because_hidden`(由 `exclude_hidden` 与 `is_hidden` 决定,`is_hidden` 检查路径任一部分是否以 `.` 开头)、`skip_because_empty`(`exclude_empty` + `is_empty_file`,后者用 `fs.info(...).get("size", 0) == 0` 判断)、`skip_because_bad_ext`(若 `required_exts` 非空且 `ref.suffix not in self.required_exts`)、`skip_because_excluded`(命中 `exclude` glob 模式)。`exclude` 参数本身是一组 glob,递归时会拼成 `input_dir/**/pattern`,非递归时是 `input_dir/pattern`,再用 `self.fs.glob` 展开成被拒绝的文件与目录集合。最终 `all_files` 排序后返回;如果一个文件都没有,直接 `raise ValueError(f"No files found in {input_dir}.")`。

### 加载期:按扩展名分发 file_extractor

确定文件后,真正把文件读成 `Document` 的是静态方法 `SimpleDirectoryReader.load_file`(它被刻意写成 `@staticmethod`,源码注释解释 "necessarily as a static method for parallel processing",因为多进程并行时函数必须可 pickle)。它的分发逻辑是整个文件读取系统的心脏:

```python
default_file_reader_cls = SimpleDirectoryReader.supported_suffix_fn()
default_file_reader_suffix = list(default_file_reader_cls.keys())
...
file_suffix = input_file.suffix.lower()
if file_suffix in default_file_reader_suffix or file_suffix in file_extractor:
    if file_suffix not in file_extractor:
        reader_cls = default_file_reader_cls[file_suffix]
        file_extractor[file_suffix] = reader_cls()
    reader = file_extractor[file_suffix]
    ...
    docs = reader.load_data(input_file, **kwargs)
    ...
else:
    fs = fs or get_default_fs()
    with fs.open(input_file, errors=errors, encoding=encoding) as f:
        data = cast(bytes, f.read()).decode(encoding, errors=errors)
    doc = Document(text=data, metadata=metadata or {})
```

逻辑可以这样读:先把文件后缀小写化,然后判断它是否落在"已知能用专用 Reader 处理"的集合里——这个集合是 `default_file_reader_suffix` 与用户传入的 `file_extractor` 键的并集。如果命中,就懒实例化对应的子 Reader(若 `file_extractor` 里还没有这个后缀,就用默认类 `reader_cls()` 新建一个并缓存进 `file_extractor`),再调用它的 `reader.load_data(input_file, extra_info=metadata)`。如果没命中任何专用 Reader,就走兜底分支:用 `fs.open` 打开、按 `encoding` 解码成纯文本,直接 `Document(text=data, metadata=metadata or {})`。这意味着任何"未知"扩展名的文件都会被当作纯文本读入,这是 `SimpleDirectoryReader` 能"吃下整个目录"的关键容错设计。

那张默认的"扩展名→Reader 类"映射表来自 `_try_loading_included_file_formats`(被赋给类属性 `supported_suffix_fn`)。它的实现很有讲究——这些专用 Reader 并不在 `llama-index-core` 里,而是 `try: from llama_index.readers.file import (DocxReader, EpubReader, HWPReader, ImageReader, IPYNBReader, MboxReader, PandasCSVReader, PandasExcelReader, PDFReader, PptxReader, VideoAudioReader)`。如果这个独立包没装,就 `except ImportError` 打印一条警告 "`llama-index-readers-file` package not found...",并返回空字典 `{}`。装上之后,它返回的映射覆盖了常见格式:`.pdf` → `PDFReader`,`.docx` → `DocxReader`,`.pptx`/`.ppt`/`.pptm` → `PptxReader`,`.csv` → `PandasCSVReader`,`.xls`/`.xlsx` → `PandasExcelReader`,`.epub` → `EpubReader`,`.mbox` → `MboxReader`,`.ipynb` → `IPYNBReader`,`.hwp` → `HWPReader`,图片类 `.gif`/`.jpg`/`.jpeg`/`.png`/`.webp` → `ImageReader`,音视频 `.mp3`/`.mp4` → `VideoAudioReader`。这种"核心包定义分发框架、专用解析能力放进可选扩展包"的拆分,正是 LlamaIndex 把核心保持轻量的工程取舍:你只装核心包就能读纯文本目录,要读 PDF 才需要额外安装。

错误处理上也体现了实用主义:子 Reader 的 `load_data` 被包在 `try/except` 里,`ImportError` 一定向上抛(`raise ImportError(str(e))`),好让用户知道缺依赖;其它异常则看 `raise_on_error`——为 `True` 时抛 `Exception("Error loading file")`,为 `False`(默认)时只打印 "Failed to load file ... Skipping..." 然后 `return []` 跳过这个文件。默认不让单个坏文件中断整个目录的加载,这对批量摄入很友好。

最后,如果 `filename_as_id=True`,通过子 Reader 读出的多段文档会被赋 ID `f"{input_file!s}_part_{i}"`,兜底纯文本分支则赋 `doc.id_ = str(input_file)`——这让文档 ID 可追溯到源文件。

## 三、默认 metadata 注入:file_path、file_name 与时间戳

Reader 不只产出文本,还要为每个 `Document` 附上来源信息,这部分由 `default_file_metadata_func` 负责。在 `load_file` 开头有 `if file_metadata is not None: metadata = file_metadata(str(input_file))`,而 `SimpleDirectoryReader.__init__` 中 `self.file_metadata = file_metadata or _DefaultFileMetadataFunc(self.fs)`——也就是说用户不传时,默认就用 `_DefaultFileMetadataFunc`,它是对 `default_file_metadata_func` 的一层可 pickle 封装(`__call__` 里转发 `default_file_metadata_func(file_path, self.fs)`,把 `fs` 存为实例属性以便多进程序列化)。

`default_file_metadata_func` 的产物是这样一个字典:

```python
default_meta = {
    "file_path": file_path,
    "file_name": file_name,
    "file_type": mimetypes.guess_type(file_path)[0],
    "file_size": stat_result.get("size"),
    "creation_date": creation_date,
    "last_modified_date": last_modified_date,
    "last_accessed_date": last_accessed_date,
}
return {k: v for k, v in default_meta.items() if v is not None}
```

可以逐项读出它注入了什么:`file_path` 是原始路径;`file_name` 取自 `os.path.basename(stat_result["name"])`(失败时回退到对原路径取 basename);`file_type` 用标准库 `mimetypes.guess_type` 推断 MIME 类型(如 `text/plain`);`file_size` 来自 `fs.stat` 的 `size`;三个时间戳分别是创建、修改、访问时间,均由 `_format_file_timestamp` 格式化。这个格式化函数把 epoch 浮点数(或已是 `datetime` 的对象)统一转成 UTC,默认输出 `%Y-%m-%d`(`include_time=False`),需要时输出 `%Y-%m-%dT%H:%M:%SZ`。值得一提的是末尾那行字典推导:只保留非 `None` 的键——所以如果某个文件系统拿不到创建时间,`creation_date` 就压根不会出现在 metadata 里,避免污染下游。这些 metadata 最终在 `load_file` 里以 `extra_info=metadata` 传给子 Reader,或在兜底分支直接作为 `Document(text=data, metadata=metadata)` 的 metadata 写入。

光注入还不够,LlamaIndex 还要决定这些 metadata 是否参与嵌入和喂给 LLM。`SimpleDirectoryReader._exclude_metadata` 在每次 `load_data`/`aload_data`/`iter_data` 返回前都会调用,它给每个文档的 `excluded_embed_metadata_keys` 与 `excluded_llm_metadata_keys` 追加了一组键:`file_name`、`file_type`、`file_size`、`creation_date`、`last_modified_date`、`last_accessed_date`。源码注释解释了意图——只保留 `file_path` 进入嵌入和 LLM 上下文(因为路径里常含有关于这块内容的"极重要语境"),而日期等保留在 metadata 里只是为了方便像 `TimeWeightedPostprocessor` 这样的后处理器使用,不应污染嵌入向量和提示词。这两个排除列表是第 1 章讲过的 `Node` 字段(在 `schema.py` 中定义),Reader 在这里只是按约定填充它们,体现了 Reader 与数据模型之间的清晰分工。

## 四、懒加载、迭代与异步:三种消费节奏

`SimpleDirectoryReader` 给出了三种把目录读成文档的节奏,适配不同内存与并发需求。

同步全量是 `load_data(self, show_progress=False, num_workers=None, fs=None)`。它用 `functools.partial` 把静态方法 `load_file` 连同所有实例配置(`file_metadata`、`file_extractor`、`encoding` 等)预绑成 `load_file_with_args`,然后分两条路:当 `num_workers and num_workers > 1` 时,走 `multiprocessing.get_context("spawn").Pool(num_workers)` 多进程并行(并贴心地 `warnings.warn` 提醒 `num_workers` 不要超过 `multiprocessing.cpu_count()`,超过则自动下调),用 `pool.imap` 把文件分发给各进程——这正是 `load_file` 必须是静态方法的原因;否则就单进程顺序遍历 `self.input_files`,每个文件调用一次 `load_file_with_args`。无论哪条路,最后都 `return self._exclude_metadata(documents)`。

迭代式是 `iter_data(self, show_progress=False)`,一个生成器:它逐个文件 `load_file`、`_exclude_metadata`,然后 `if len(documents) > 0: yield documents`。它按文件粒度 `yield`,适合目录极大、不想把所有 `Document` 一次性堆在内存里的场景。注意它产出的是"每个文件对应的文档列表",而不是单个文档。

异步是 `aload_data(self, show_progress=False, num_workers=None, fs=None)`。它为每个文件构造一个 `SimpleDirectoryReader.aload_file(...)` 协程,放进 `coroutines` 列表;若指定了 `num_workers`,用 `run_jobs(coroutines, workers=num_workers)` 控制并发度;否则用 `asyncio.gather`(带进度时走 `get_asyncio_module(show_progress=...)`)。`aload_file` 与 `load_file` 几乎对称,区别只在它 `await reader.aload_data(...)`——这意味着如果某个子 Reader 实现了真正的异步 I/O,这里就能拿到真异步收益;否则借助 `BaseReader` 默认的 `aload_data`(`asyncio.to_thread`)退化为线程池并发。最后同样 `return self._exclude_metadata(documents)`。这三种节奏共用同一套 `load_file`/`aload_file` 内核,只是在"如何调度文件"这一层做文章,是很干净的复用。

## 五、fsspec:本地与远程文件系统的统一抽象

`SimpleDirectoryReader` 之所以能既读本地磁盘、又读 S3 等远端存储,关键在于它把所有文件系统操作都委托给了 fsspec。源码顶部 `import fsspec` 与 `from fsspec.implementations.local import LocalFileSystem`,构造器里 `self.fs = fs or get_default_fs()`,而 `get_default_fs()` 直接返回 `LocalFileSystem()`。整个类内部访问文件一律走 `self.fs.isfile`、`self.fs.isdir`、`self.fs.walk`、`self.fs.glob`、`self.fs.open`、`self.fs.stat`、`self.fs.info`,从不直接调用 `os` 或内置 `open`。这意味着只要传入一个实现了 fsspec 接口的文件系统对象(例如 `s3fs.S3FileSystem`),同一份 Reader 代码就能把远端 bucket 当作本地目录来扫描和读取 [[fsspec — Filesystem Spec]](https://filesystem-spec.readthedocs.io/)。

抽象层并非毫无成本,源码里有两处专门为非本地文件系统打的补丁,很能说明问题。其一是路径类型的切换:`_Path = Path if is_default_fs(self.fs) else PurePosixPath`,在多处出现。`is_default_fs` 判断 `isinstance(fs, LocalFileSystem) and not fs.auto_mkdir`;一旦不是本地默认文件系统,就改用 `PurePosixPath` 而非平台相关的 `Path`,因为远端对象存储的路径分隔符语义是 POSIX 风格的,不能让 Windows 风格的反斜杠掺进来。其二是目录判定:专门写了 `_is_directory` 方法,先尝试标准的 `fs.isdir`,失败后对非默认文件系统再检查 "0 字节且以 `/` 结尾且 type 不是 file" 的对象——注释明说这是因为 "For S3 filesystems, directories are often represented as 0-byte objects ending with '/'"。S3 没有真正的目录概念,LlamaIndex 通过这个启发式让目录扫描在对象存储上也能正确工作。

此外,`load_file` 在把文件交给子 Reader 时有一句 `if fs and not is_default_fs(fs): kwargs["fs"] = fs`——只有非默认文件系统才会把 `fs` 透传给子 Reader,本地情况下让子 Reader 用自己的默认行为,既避免冗余又保持兼容。`FileSystemReaderMixin.read_file_content` 同样以 `fs.open(input_file, errors=..., encoding=...)` 读取字节,完全建立在 fsspec 之上。

## 六、自定义 Reader 怎么写

理解了契约之后,写一个自定义 Reader 就是填空题。最省事的做法是继承 `BaseReader` 并只重写 `lazy_load_data`,返回一个产出 `Document` 的可迭代对象——这样 `load_data`、`aload_data`、`alazy_load_data` 全都免费获得。如果你的数据天然就是一次性拿到的(比如调一次 API 返回全量),也可以像 `StringIterableReader` 那样直接重写 `load_data`,因为它本就不是抽象方法。构造 `Document` 时最小只需 `Document(text=...)`;要附来源信息就传 `metadata=...`(等价的旧写法是 `extra_info=...`,`Document.__init__` 会把 `extra_info` 迁移到 `metadata`、把 `doc_id` 迁移到 `id_`、把 `text` 包进 `text_resource`,这些向后兼容逻辑都在 `schema.py` 的 `Document.__init__` 里)。

如果希望 Reader 可被序列化保存(例如纳入一个可持久化的摄入配置),就继承 `BasePydanticReader` 而非裸 `BaseReader`,并按需设置 `is_remote`。如果数据源天然有"先列举、再按需取"的结构,混入 `ResourcesReaderMixin` 并实现它的三个抽象方法 `list_resources`、`get_resource_info`、`load_resource`,就能复用框架提供的 `load_resources`、`list_resources_with_info` 等组合能力——`SimpleDirectoryReader` 自己就是这么做的,它的 `list_resources` 返回 `[str(x) for x in self.input_files]`,`load_resource` 则转调静态的 `load_file`。

还有一种更轻的扩展点不必写新类:既然 `SimpleDirectoryReader` 的分发表是"`default_file_reader_cls` 键与 `file_extractor` 键的并集",你完全可以在构造时传入 `file_extractor={".myext": MyReader()}`,为某个扩展名挂上自定义解析器,甚至覆盖默认行为(比如用自己的 PDF 解析替换内置的 `PDFReader`)。同理,传入自定义的 `file_metadata` 函数就能改写注入的 metadata 字段。这种"组合优于继承"的扩展方式,往往比派生一个新 Reader 更经济。

## 本章小结

1. Readers 子系统是整条 RAG 链路的起点,其唯一统一职责是把异构数据源转换成 `List[Document]`;`Document` 的统一性让下游 NodeParser、Embedding、Index 得以与数据源解耦。
2. `BaseReader`(`readers/base.py`)用模板方法模式以 `lazy_load_data` 为唯一最小重写点,默认派生出 `load_data`(`list(...)` 物化)、`aload_data`/`alazy_load_data`(`asyncio.to_thread` 伪异步)。
3. `BasePydanticReader` 让 Reader 可被 Pydantic 序列化并带 `is_remote` 标记;`ResourcesReaderMixin` 提供"列举资源 + 按需加载"模型;`ReaderConfig` 把 Reader 与其参数打包成可持久化配置。
4. `SimpleDirectoryReader`(`readers/file/base.py`)同时是 `BaseReader`、`ResourcesReaderMixin`、`FileSystemReaderMixin`;构造期通过 `_add_files` + `fs.walk` 扫描目录,用 `recursive` 翻译成 `maxdepth`(非递归为 `1`、递归为 `1000`),并按 `exclude_hidden`、`exclude_empty`、`required_exts`、`exclude`、`num_files_limit` 过滤。
5. 加载期的核心是静态方法 `load_file`:按 `input_file.suffix.lower()` 在"默认映射 + `file_extractor`"并集里查找专用 Reader,命中则懒实例化并调用其 `load_data(..., extra_info=metadata)`,未命中则用 fsspec 读为纯文本并 `Document(text=data, metadata=...)` 兜底。
6. 默认扩展名映射由 `_try_loading_included_file_formats` 从可选包 `llama-index-readers-file` 导入(缺失则告警并返回 `{}`),覆盖 `.pdf`/`.docx`/`.pptx`/`.csv`/`.xlsx`/`.epub`/`.ipynb`/图片/音视频等;核心包保持轻量,专用解析能力按需安装。
7. `default_file_metadata_func` 为每个文档注入 `file_path`、`file_name`、`file_type`(`mimetypes.guess_type`)、`file_size`、以及创建/修改/访问时间戳(`_format_file_timestamp` 转 UTC),并剔除值为 `None` 的键。
8. `_exclude_metadata` 把除 `file_path` 外的文件元数据加入 `excluded_embed_metadata_keys` 与 `excluded_llm_metadata_keys`,避免文件名、日期等污染嵌入向量与 LLM 提示,同时保留给后处理器使用。
9. 三种消费节奏共享同一内核:`load_data` 支持 `num_workers` 多进程(`spawn` + `imap`,故 `load_file` 必须是静态方法)、`iter_data` 按文件 `yield`、`aload_data` 用 `asyncio.gather`/`run_jobs` 并发。
10. fsspec 抽象让同一份代码读本地与远端;`is_default_fs` 判定后用 `PurePosixPath` 替代 `Path`,`_is_directory` 用"0 字节且以 `/` 结尾"启发式处理 S3 目录占位符,非默认 fs 才把 `fs` 透传给子 Reader。
11. 自定义 Reader 最省事是只重写 `lazy_load_data`;需序列化继承 `BasePydanticReader`;需资源模型混入 `ResourcesReaderMixin`;而通过传 `file_extractor`/`file_metadata` 可在不派生新类的情况下扩展 `SimpleDirectoryReader`。

## 动手实验

1. **实验一:观察统一契约与默认 metadata** — 用 `SimpleDirectoryReader(input_dir="<某含 .txt/.md 的目录>").load_data()` 读取一个目录,打印每个文档的 `doc.metadata`,确认 `file_path`/`file_name`/`file_type`/`file_size` 与时间戳字段的存在,并打印 `doc.excluded_embed_metadata_keys` 验证 `_exclude_metadata` 已把日期等键排除、只留 `file_path`。
2. **实验二:扩展名分发与兜底分支** — 在目录里放一个无扩展名或自造扩展名(如 `.foobar`)的纯文本文件,观察它走 `load_file` 的兜底分支被当作纯文本读入;再在构造时传 `file_extractor={".foobar": StringIterableReader()-like 自定义 Reader}`(或一个最小 `BaseReader` 子类),验证分发表是默认映射与 `file_extractor` 的并集。
3. **实验三:对比三种消费节奏** — 对同一目录分别调用 `load_data()`、`list(iter_data())`、`asyncio.run(reader.aload_data())`,比较返回结构(`iter_data` 按文件分组 `yield`),再给 `load_data(num_workers=4, show_progress=True)` 加并行与进度条,观察 `multiprocessing` 警告在 `num_workers` 超过 CPU 数时触发。
4. **实验四:自定义 Reader 与 fsspec** — 继承 `BaseReader` 写一个只重写 `lazy_load_data` 的 Reader(例如从一个 API 列表逐条 `yield Document(text=..., metadata=...)`),验证你免费得到了 `load_data`/`aload_data`;再用 `fsspec.filesystem("memory")` 构造一个内存文件系统、写入几个文件,把它作为 `fs=` 传给 `SimpleDirectoryReader`,确认 `is_default_fs` 为假时路径切换为 `PurePosixPath` 且仍能正确读取。

> **下一章预告**:Reader 交出的是"整篇文档"级别的 `Document`,而向量检索需要的是粒度更细、长度更可控的片段。第 4 章将进入节点解析与切分 NodeParser,解构 LlamaIndex 如何把 `Document` 拆成带父子关系与重叠窗口的 `Node`——从 `SentenceSplitter` 的分句与 token 预算,到 `NodeParser` 如何在切分时保留并传播 metadata、维护 `relationships`,看清"文档"到"节点"这一步的工程细节。
