# 第 2 章 文件抽象与 Storage 存储层

任何 RAG 系统的第一步，都是把外部世界里五花八门的文件（PDF、txt、docx、甚至 mp3）请进自己的领地。这道门要解决三件事：文件如何被**统一表示**、如何被**指纹化以便去重**、以及如何被**存放**。Quivr 把前两件事交给 `files/file.py` 里的 `QuivrFile`，把第三件事交给 `storage/` 目录下的一套以 `StorageBase` 为抽象基类的存储实现。

本章我们钻进这两层代码，看 Quivr 如何用 `__slots__` 把文件对象做得轻量、用 `hashlib.sha1` 给文件按指纹去重、用 `__init_subclass__` 把「子类必须声明 name」这条约束硬编码进类型系统，以及它如何在零配置时给你一个默认的内存存储。这一章是数据进入 Quivr 的第一道门，也为第 3 章的解析层铺好地基。

## QuivrFile：用 __slots__ 定义的文件实体

Quivr 内部一切关于文件的操作都围绕 `QuivrFile` 展开。它没有继承 pydantic，而是一个朴素的 Python 类，但在类体的第一行就用 `__slots__` 显式列出了所有合法字段：

```python
class QuivrFile:
    __slots__ = [
        "id",
        "brain_id",
        "path",
        "original_filename",
        "file_size",
        "file_extension",
        "file_sha1",
        "additional_metadata",
    ]
```

`__slots__` 的作用是告诉解释器：这个类的实例只会有这八个属性，因此不必为每个实例分配一个 `__dict__`。这带来两个实际收益。第一是**省内存**——在一个 brain 可能加载成百上千个文件的场景里，省掉每实例的字典开销不是小数目。第二是**防止意外属性**：写错字段名（比如把 `file_sha1` 敲成 `file_sha`）时会立刻抛 `AttributeError`，而不是悄悄给实例加一个无人使用的属性。对一个作为「数据契约」反复传递的实体类来说，这种约束远比灵活性重要。

构造函数把八个字段依次赋值，其中 `brain_id`、`file_size`、`metadata` 带默认值，而 `additional_metadata` 的处理用了一个常见的小技巧——`self.additional_metadata = metadata if metadata else {}`，避免了把可变默认参数 `{}` 直接写进函数签名而导致的共享陷阱。

`QuivrFile` 真正对外暴露的核心是 `metadata` property。它并不直接返回内部字段，而是拼出一份带统一前缀键名的字典：

```python
@property
def metadata(self) -> dict[str, Any]:
    return {
        "qfile_id": self.id,
        "qfile_path": self.path,
        "original_file_name": self.original_filename,
        "file_sha1": self.file_sha1,
        "file_size": self.file_size,
        **self.additional_metadata,
    }
```

这套 `qfile_id` / `qfile_path` / `original_file_name` / `file_sha1` / `file_size` 的键名，正是后续解析阶段灌进 LangChain `Document` 的 metadata 字段（参见 [[LangChain Document]](https://python.langchain.com/docs/concepts/documents/)）。把它们集中在一个 property 里生成，意味着「文件级元数据长什么样」只有这一个出处，下游不需要也不应该自己拼。注意 `**self.additional_metadata` 放在最后，这让调用方传入的自定义元数据能够覆盖默认键——是有意为之的开放式设计。

## 异步读取：open() 与 asynccontextmanager

文件内容的读取统一走 `open()` 方法，它被 `@asynccontextmanager` 装饰，内部用 `aiofiles` 做异步 IO：

```python
@asynccontextmanager
async def open(self) -> AsyncGenerator[AsyncIterable[bytes], None]:
    # TODO(@aminediro) : match on path type
    f = await aiofiles.open(self.path, mode="rb")
    try:
        yield f
    finally:
        await f.close()
```

用 `async with qfile.open() as f:` 即可拿到一个异步文件句柄，`try/finally` 保证无论后续处理是否抛异常都会 `await f.close()`。选择 `aiofiles`（见 [[aiofiles]](https://github.com/Tinche/aiofiles)）而不是内建 `open`，是因为 Quivr 的整条摄取链路都是 `async` 的——文件读取不应该阻塞事件循环。源码里那行 `TODO: match on path type` 也透露了设计者的未来意图：目前 `path` 假定是本地路径，将来可能根据路径类型（如远程 URI）切换不同读取方式。

## 序列化：QuivrFileSerialized 与 serialize/deserialize

`QuivrFile` 自身不是 pydantic 模型，但它有一个配对的 pydantic 影子类 `QuivrFileSerialized`，字段与 `__slots__` 一一对应（`id` / `brain_id` / `path` / `original_filename` / `file_size` / `file_extension` / `file_sha1` / `additional_metadata`）。`serialize()` 把运行时对象转成这个模型，`deserialize()` 这个 `classmethod` 再把它还原：

```python
def serialize(self) -> QuivrFileSerialized:
    return QuivrFileSerialized(
        id=self.id,
        brain_id=self.brain_id,
        path=self.path.absolute(),
        ...
    )
```

值得注意 `path=self.path.absolute()`——序列化时强制转绝对路径，这样把 brain 存盘后再换工作目录加载也不会丢失文件位置。这种「运行时用裸类、持久化用 pydantic」的分工，让热路径上的对象保持轻量，又能在落盘时复用 pydantic 的校验与 JSON 能力。

## FileExtension 枚举与扩展名识别

支持哪些格式，由 `FileExtension(str, Enum)` 一次性枚举出来。除了常见的文本与文档类（`.txt` `.pdf` `.csv` `.doc` `.docx` `.pptx` `.xls` `.xlsx` `.md` `.mdx` `.markdown` `.bib` `.epub` `.html` `.odt`），还包含代码类（`.py` `.ipynb`）以及一组音视频格式（`.m4a` `.mp3` `.webm` `.mp4` `.mpga` `.wav` `.mpeg`）——后者预示着 Quivr 在解析层支持音视频转写。继承 `str` 让枚举成员可以直接当字符串用，比如拼接路径时 `f"{file.id}{file.file_extension}"`。

识别扩展名的逻辑在 `get_file_extension`，它有意先信任 MIME 类型、再回退到后缀：

```python
def get_file_extension(file_path: Path) -> FileExtension | str:
    try:
        mime_type, _ = mimetypes.guess_type(file_path.name)
        if mime_type:
            mime_ext = mimetypes.guess_extension(mime_type)
            if mime_ext:
                return FileExtension(mime_ext)
        return FileExtension(file_path.suffix)
    except ValueError:
        warnings.warn(
            f"File {file_path.name} extension isn't recognized. ...",
            stacklevel=2,
        )
        return file_path.suffix
```

它先用 `mimetypes.guess_type` 猜 MIME 类型，再用 `mimetypes.guess_extension` 把 MIME 反推成标准后缀（见 [[mimetypes]](https://docs.python.org/3/library/mimetypes.html)）——这能把诸如 `.htm` 归一成 `.html` 之类的差异抹平。只有当 MIME 路线走不通时才退回 `file_path.suffix`。如果连后缀都不在 `FileExtension` 枚举里，`FileExtension(...)` 会抛 `ValueError`，被捕获后用 `warnings.warn` 提示「请为这个后缀注册一个 parser」，并返回原始后缀字符串而非崩溃。注意返回类型是 `FileExtension | str`：识别成功是枚举，失败则降级为裸字符串，但流程不中断——这是一种**容错优先**的设计，把「未知格式」交给下游 processor 注册表去决定能否处理。

## load_qfile：指纹化与 id 的来历

把磁盘上的一个路径变成 `QuivrFile`，统一入口是异步函数 `load_qfile`。它做三件关键的事。

第一，校验存在性并取大小：路径不存在直接抛 `FileExistsError`，再用 `os.stat(path).st_size` 取字节数。

第二，**计算 SHA-1 指纹**——这是整个去重机制的基础：

```python
async with aiofiles.open(path, mode="rb") as f:
    file_sha1 = hashlib.sha1(await f.read()).hexdigest()
```

它一次性异步读入全部字节，喂给 `hashlib.sha1` 得到十六进制摘要。这个 `file_sha1` 后面会被存储层用来判断「同一份内容是否已经上传过」，内容相同的文件即使文件名不同也会得到相同指纹。

第三，**决定 id 的来源**，这里藏着一个巧妙的约定：

```python
try:
    # NOTE: when loading from existing storage, file name will be uuid
    id = UUID(path.name)
except ValueError:
    id = uuid4()
```

如果文件名本身就是一个合法 UUID，就直接用它作 id；否则生成新的 `uuid4()`。源码注释点破了原因——文件被存进 storage 后，落盘的文件名就是它的 UUID（见下一节的 `dst_path` 拼法），所以从既有存储重新加载时，文件名天然就是 id，无需另存映射表。这是一种用文件名携带身份信息的、轻量的「无数据库」持久化思路。

## StorageBase：把约束固化进类型系统

所有存储实现都继承自 `storage/storage_base.py` 里的抽象基类 `StorageBase(ABC)`。它最值得学习的一笔，是用 `__init_subclass__` 在**定义子类的那一刻**就检查 `name` 属性是否存在：

```python
class StorageBase(ABC):
    name: str

    def __init_subclass__(cls, **kwargs):
        for required in ("name",):
            if not getattr(cls, required):
                raise TypeError(
                    f"Can't instantiate abstract class {cls.__name__} without {required} attribute defined"
                )
        return super().__init_subclass__(**kwargs)
```

`__init_subclass__` 是在子类被**创建**（而不是实例化）时触发的钩子。这意味着：只要有人写了一个忘记声明 `name` 的存储子类，Python 在导入该模块时就会立刻抛 `TypeError`，而不是等到运行时调用 `info()` 才发现问题。这就是「把约束固化进类型系统」的范例——它把一条文档级的约定（每种存储都得有个名字）变成了无法绕过的硬性规则。`info()` 方法也依赖这个 `name`，它返回一个 `StorageInfo(storage_type=self.name, n_files=self.nb_files())`。

四个抽象方法 `nb_files` / `get_files` / `upload_file` / `remove_file` 定义了存储的最小契约：数文件数、取全部文件、上传单个文件、按 id 删除。其中 `get_files` / `upload_file` / `remove_file` 都是 `async`，与 `QuivrFile.open()` 的异步风格一致。每个抽象方法体里还写了 `raise Exception("Unimplemented ...")`，是抽象方法之外的额外保险。

## LocalStorage：落盘、去重与诚实的未实现

`LocalStorage` 是写到磁盘的具体实现，`name = "local_storage"`。它的目录默认值体现了「环境变量优先、否则给个合理缺省」的惯例：

```python
if dir_path is None:
    self.dir_path = Path(
        os.getenv("QUIVR_LOCAL_STORAGE", "~/.cache/quivr/files")
    )
...
os.makedirs(self.dir_path, exist_ok=True)
```

即设置了环境变量 `QUIVR_LOCAL_STORAGE` 就用它，否则落到 `~/.cache/quivr/files`，并用 `exist_ok=True` 幂等地建好目录。

上传逻辑 `upload_file` 把去重和存放方式都讲清楚了：

```python
dst_path = os.path.join(
    self.dir_path, str(file.brain_id), f"{file.id}{file.file_extension}"
)
if file.file_sha1 in self.hashes and not exists_ok:
    raise FileExistsError(f"file {file.original_filename} already uploaded")
if self.copy_flag:
    shutil.copy2(file.path, dst_path)
else:
    os.symlink(file.path, dst_path)
file.path = Path(dst_path)
self.files.append(file)
self.hashes.add(file.file_sha1)
```

几个要点：目标路径按 `dir_path/brain_id/<uuid><ext>` 组织——文件名正是 `file.id`，与上一节 `load_qfile` 的「文件名即 UUID」约定首尾呼应。去重靠 `self.hashes: Set[str]` 这个 SHA-1 集合，命中且 `exists_ok=False` 时抛 `FileExistsError`，避免重复内容入库。`copy_flag` 决定是 `shutil.copy2`（连同元数据一起拷贝）还是 `os.symlink`（建软链，省空间但依赖源文件留在原处）。上传成功后会把 `file.path` 改写成存储后的新路径，并把文件登记进 `self.files` 列表与 `hashes` 集合。

值得点名表扬的是 `remove_file`——它**诚实地** `raise NotImplementedError`，连同 `_load_files` 里的 `TODO(@aminediro): load existing files` 一起，明确标注了哪些能力还没做，而不是用一个空实现假装支持。这种坦白对读源码的人是友好的。`load` 这个 classmethod 则负责从 `LocalStorageConfig` 反序列化：用 `config.storage_path` 重建目录，再把 `config.files.values()` 里的每个 `QuivrFileSerialized` 经 `QuivrFile.deserialize` 还原成对象列表。

## TransparentStorage：零配置的内存默认

与 `LocalStorage` 对照，`TransparentStorage` 是最简实现，`name = "transparent_storage"`。它根本不碰磁盘，只用一个 `self.id_files = {}` 字典在内存里以 `file.id` 为键存放 `QuivrFile`：

```python
class TransparentStorage(StorageBase):
    name: str = "transparent_storage"

    def __init__(self):
        self.id_files = {}

    async def upload_file(self, file: QuivrFile, exists_ok: bool = False) -> None:
        self.id_files[file.id] = file
```

`upload_file` 一行搞定、不去重、不报 `FileExistsError`，`nb_files` 即 `len(self.id_files)`，`get_files` 返回 `list(self.id_files.values())`，`remove_file` 同样 `NotImplementedError`。它存在的意义是当 Brain 工厂方法不显式指定存储时的**零配置默认**——用户调用 `Brain.from_files(...)` 想快速建一个 brain 时，文件直接进内存即可跑通，不必先操心磁盘目录。它的 `load` 从 `TransparentStorageConfig` 反序列化，把 `config.files.items()` 重建回 `id_files` 字典。

## 配置与可发现性：两个 Config 与 StorageInfo

两种存储各自对应一个 pydantic 配置类，定义在 `brain/serialization.py`。`LocalStorageConfig` 带 `storage_type: Literal["local_storage"]`、`storage_path: Path` 和 `files: dict[UUID, QuivrFileSerialized]`；`TransparentStorageConfig` 则少了 `storage_path`，因为它本就不落盘。这两个 `Literal` 字段不是摆设——在 `BrainSerialized` 里，`storage_config` 被声明为 `Union[TransparentStorageConfig, LocalStorageConfig]` 并用 `Field(..., discriminator="storage_type")` 做**判别式联合**，pydantic 据此在反序列化时自动选对类，对应到各存储类的 `load` classmethod。

最后，`brain/info.py` 的 `StorageInfo` 是个 `@dataclass`，只有 `storage_type` 和 `n_files` 两个字段，并提供 `add_to_tree` 把自己渲染进 `rich.tree.Tree`。它由 `StorageBase.info()` 产出，最终汇入 `BrainInfo.to_tree()`，让用户 `print(brain.info())` 时能在终端看到「Storage Type / Number of Files」这样的可视化摘要。这让存储层不仅能存，还能被观察。

## 本章小结

- `QuivrFile` 用 `__slots__` 锁定八个字段，既省内存又防止意外属性，是 Quivr 内部文件操作的统一数据契约。
- `metadata` property 集中拼出 `qfile_id` / `qfile_path` / `original_file_name` / `file_sha1` / `file_size`，并允许 `additional_metadata` 覆盖，是下游灌入 LangChain `Document` 元数据的唯一出处。
- 文件读取统一走被 `@asynccontextmanager` 装饰的 `open()`，底层用 `aiofiles` 异步读、`try/finally` 保证关闭。
- `serialize/deserialize` 配对 pydantic 的 `QuivrFileSerialized`，落盘时把 `path` 转绝对路径——运行时用裸类、持久化用 pydantic 的分工。
- `FileExtension` 枚举一次列全支持格式（含音视频），`get_file_extension` 先信 `mimetypes` 再回退后缀，无法识别时 `warnings.warn` 而不崩溃。
- `load_qfile` 用 `hashlib.sha1` 算指纹做去重基础，并尝试 `UUID(path.name)` 复用文件名作 id，否则 `uuid4()`。
- `StorageBase` 用 `__init_subclass__` 强制子类声明 `name`，把约定变成导入即检查的硬规则，体现「约束固化进类型系统」。
- `LocalStorage` 默认目录取 `QUIVR_LOCAL_STORAGE` 或 `~/.cache/quivr/files`，按 `brain_id/<uuid><ext>` 落盘，用 `hashes` 集合去重，`copy_flag` 切换 copy 与 symlink。
- `LocalStorage.remove_file` 诚实地 `raise NotImplementedError`，连同 `_load_files` 的 TODO 一起明确标注未完成能力。
- `TransparentStorage` 用内存 `id_files` 字典实现零配置默认存储，是 Brain 工厂在不指定存储时的兜底。
- 两个 Config 的 `storage_type` 是 pydantic 判别式联合的判别键，驱动各存储的 `load` 反序列化；`StorageInfo` 让存储状态可被 `rich` 树形可视化。

## 动手实验

1. **实验一：验证 __slots__ 的约束** — 打开 `files/file.py`，在 REPL 里构造一个 `QuivrFile`，尝试给它赋一个不在 `__slots__` 列表里的属性（如 `qf.foo = 1`），观察抛出的 `AttributeError`；再注释掉 `__slots__` 一行对比行为差异，体会它「防止意外属性」的作用。
2. **实验二：复现 SHA-1 去重** — 用 `load_qfile` 加载同一份文件的两个不同文件名副本，打印各自的 `file_sha1`，确认指纹一致；再依次 `await storage.upload_file(...)` 到同一个 `LocalStorage`（`exists_ok=False`），验证第二次会抛 `FileExistsError`，并到 `~/.cache/quivr/files` 下查看落盘文件名是否为 UUID。
3. **实验三：触发 __init_subclass__ 检查** — 仿照 `storage/local_storage.py`，自己写一个继承 `StorageBase` 但故意不声明 `name` 的子类，import 它，观察在子类定义阶段就抛出的 `TypeError` 及其提示文案，理解为何错误发生得这么早。
4. **实验四：对比两种 Storage 与 info()** — 分别实例化 `LocalStorage` 和 `TransparentStorage`，各上传一个文件后调用 `.info()`，对比 `LocalStorage.info()` 多出的 `directory_path` 键；再把 `StorageInfo` 喂给 `rich.tree.Tree` 渲染，看 `add_to_tree` 输出的「Storage Type / Number of Files」摘要。

> **下一章预告**：文件被表示、指纹化、存放之后，真正的难题是把 PDF、docx、音视频等异构格式统一解析成可索引的文本。第 3 章「Processor 注册表与多格式解析」将钻进 Quivr 的 processor 注册机制，看它如何按 `FileExtension` 分派解析器、把一个 `QuivrFile` 转成一串 LangChain `Document`。
