# 第 3 章 输入加载 Input

整条 GraphRAG 索引流水线的第一步,是把磁盘上五花八门的原始文件——`.txt`、`.csv`、`.json`、`.jsonl`、`.parquet`,乃至 PDF/Word——读成一种统一的、下游所有 workflow 都能消费的"文档记录"。这件事看似琐碎,却是后续切分、抽取、社区检测、社区报告生成全部环节的地基:如果第一步产出的 schema 不一致、文档 id 不稳定,后面的增量更新和缓存全都会失效。

GraphRAG 把这件事独立成了一个轻量子包 `graphrag-input`(`pyproject.toml` 里 `name = "graphrag-input"`,`version = "3.1.0"`,`description = "Input document loading utilities for GraphRAG"`)。它只依赖 `graphrag-common`(工厂)、`graphrag-storage`(存储抽象)、`pydantic`、`markitdown`、`pyarrow`,不反向依赖主包。本章逐文件解构它如何把异构文件读成统一的 `TextDocument`,以及主包 `load_input_documents` / `load_update_documents` 两个 workflow 怎么把它接入流水线。

## 统一契约:`TextDocument` 数据类

要谈"统一输入",先看输出长什么样。`text_document.py` 定义了一个 `@dataclass TextDocument`,它就是整个子包对外承诺的契约:

- `id: str` —— 文档的唯一标识;
- `text: str` —— 正文内容;
- `title: str` —— 标题;
- `creation_date: str` —— ISO-8601 格式的创建时间;
- `raw_data: dict[str, Any] | None = None` —— 来自源文件的原始一行/一条数据,默认为 `None`。

注意字段顺序:`id`、`text`、`title`、`creation_date` 是必填位置参数,`raw_data` 才有默认值。`raw_data` 的存在是个关键设计——对于结构化文件(csv/json),它保留整行原始字典,这样用户在配置里指定的任意"额外列"都能在后续被取出来,而不必硬塞进固定的四个字段里。

`TextDocument` 还提供了两个取值方法,体现了"标准字段 + 自由字段"的两段式访问:

- `get(field, default_value=None)`:如果 `field` 属于 `["id", "title", "text", "creation_date"]` 就直接 `getattr`;否则去 `raw_data` 字典里按点号路径(委托给 `get_property`)取,取不到返回 `default_value`。源码注释明确写了这是"two step approach for flexibility"——纯文本只有标准四字段也能用,结构化输入则能透出原始任意列。
- `collect(fields)`:对一组字段批量调用 `get`,把非 `None` 的值收进一个 dict 返回。

这就是下游 workflow 看到的统一形状。无论文件是 txt 还是 parquet,经过 reader 之后都被规整成 `TextDocument` 的列表。

## 配置入口:`InputConfig`

`input_config.py` 用 pydantic 定义了 `InputConfig`,这是用户在 `settings.yaml` 的 `input` 段里能配的全部字段:

- `type: str`,默认 `InputType.Text`,即文件类型;
- `encoding: str | None`,默认 `None`,文件编码;
- `file_pattern: str | None`,默认 `None`,用于匹配文件的正则;
- `id_column / title_column / text_column: str | None`,默认全 `None`,告诉结构化 reader 哪一列当 id、title、text。

两个细节值得记住。第一,`model_config = ConfigDict(extra="allow")`,注释说明这是"Allow extra fields to support custom reader implementations"——它允许用户在配置里塞进官方没定义的字段,这些字段会随 `model_dump()` 一起流进自定义 reader 的构造函数。第二,这里的默认 `encoding`、`file_pattern`、`id_column` 等都是 `None`,真正的兜底默认值不在配置层,而在各 reader 的 `__init__` 里(下文会看到 `encoding` 默认 `utf-8`、各 reader 自带 `file_pattern`)。这是一种"配置层只表达用户意图,默认值由实现层补齐"的分层策略。

## 类型枚举:`InputType`

`input_type.py` 是一个 `StrEnum`,枚举了所有受支持的文件类型:`Csv = "csv"`、`Text = "text"`、`Json = "json"`、`JsonLines = "jsonl"`、`MarkItDown = "markitdown"`、`Parquet = "parquet"`。它重写了 `__repr__` 返回带引号的字符串。因为是 `StrEnum`,这些成员既能当枚举又能当字符串直接比较,工厂里 `match input_strategy` 时正是利用了这一点。

## 抽象基类:`InputReader`

`input_reader.py` 定义了所有 reader 的抽象基类 `InputReader(metaclass=ABCMeta)`。它的构造接收 `storage`、`file_pattern`、`encoding="utf-8"` 以及 `**kwargs`,把它们存进 `self._storage`、`self._file_pattern`、`self._encoding`。这里 `encoding` 的默认值 `utf-8` 就是前面说的"实现层兜底"。

`InputReader` 把"遍历文件"和"读单个文件"拆成两层,这是整个子包最核心的设计:

- `read_file(self, path)` 是 `@abstractmethod`,每个具体 reader 必须实现,负责把一个文件读成 `list[TextDocument]`;
- `_iterate_files()` 是一个 async 生成器,负责调度。它先 `self._storage.find(re.compile(self._file_pattern))` 拿到所有匹配文件;若一个都没匹配上,打一条 warning 并直接返回(不报错);然后逐个文件调用 `await self.read_file(file)`,把产出的 doc 一条条 `yield` 出来。关键在那段 `try/except Exception`:**单个文件解析失败只 warning 并跳过,不会让整批加载崩溃**(`"Warning! Error loading file %s. Skipping..."`)。最后打两条统计日志,报告找到多少文件、实际加载多少行。
- `__aiter__` 返回 `_iterate_files()`,因此可以写 `async for doc in reader`;
- `read_files()` 是便捷方法,`return [doc async for doc in self]`,一次性把所有文档收成一个 list。

这种"基类管遍历 + 容错 + 统计,子类只管解析一个文件"的拆分,让六种 reader 的实现都极薄。

注意 `find` 来自 `graphrag-storage` 的 `Storage` 抽象基类:它接收一个 `re.Pattern`,返回文件 key 的 `Iterator`;`get(key, as_bytes, encoding)` 取内容;`get_creation_date(key)` 取创建日期。reader 完全不关心文件到底在本地磁盘、Blob 还是内存里,这层解耦由 storage 抽象提供。

## 工厂分发:`input_reader_factory`

`input_reader_factory.py` 负责"按类型挑 reader"。`InputReaderFactory` 继承自 `graphrag-common` 的泛型 `Factory[InputReader]`,而 `Factory` 是个用 `__new__` 实现的单例,内部维护 `_service_initializers`(类型名 → `_ServiceDescriptor`)和 `_initialized_services`(singleton 缓存)。`register` 注册一个初始化器及其作用域(`"transient"` 或 `"singleton"`),`create(strategy, init_args)` 按名取初始化器并调用——`create` 里还有一个巧妙处理:`init_args = {k: v for k, v in ... if v is not None}`,**把值为 `None` 的参数全部剔除**,从而让 reader 构造函数里的默认值生效。这正好解释了为什么 `InputConfig` 把那么多字段默认成 `None`:`None` 在工厂层被丢掉,默认值在 reader 层补齐。

对外暴露的两个函数:

- `register_input_reader(input_reader_type, input_reader_initializer, scope="transient")`:供用户注册自定义 reader;
- `create_input_reader(config: InputConfig, storage: Storage) -> InputReader`:主入口。它先 `config_model = config.model_dump()`、`input_strategy = config.type`,然后做一个**懒注册**:如果 `input_strategy` 还没在工厂里,就 `match` 类型,按需 `from graphrag_input.csv import CSVFileReader` 这样延迟导入并注册对应 reader,六种类型各一个 `case`,`case _` 兜底抛 `ValueError` 并列出已注册类型。最后把 `storage` 塞进 `config_model["storage"]`,`return input_reader_factory.create(input_strategy, init_args=config_model)`。

懒注册 + 延迟导入的好处:`pyarrow`、`markitdown` 这种重依赖只有在用户真正选了 `parquet` / `markitdown` 时才被 import,而不是子包一加载就全拉进来。`config_model` 里的所有字段(包括 `extra="allow"` 透出的自定义字段)都作为关键字参数流进 reader 构造函数,这就是自定义 reader 能拿到额外配置的通道。

## 结构化 reader 基类:`StructuredFileReader`

csv、json、jsonl、parquet 这四种"一文件多行"的格式共享一个中间基类 `StructuredFileReader(InputReader)`(`structured_file_reader.py`)。它在 `InputReader` 之上多接收三个列名参数:`id_column=None`、`title_column=None`、`text_column="text"`(注意 `text_column` 默认就是字符串 `"text"`,不是 `None`),存为 `self._id_column / _title_column / _text_column`。

核心是 `process_data_columns(rows, path)`,它把"一堆字典行"统一映射成 `TextDocument` 列表,这段逻辑值得逐条拆:

- **text 是必填**:`text = get_property(row, self._text_column)`。`get_property` 支持点号路径,所以 `text_column` 可以写成 `"content.body"` 这种嵌套路径;取不到会抛 `KeyError`,而这个异常会被基类 `_iterate_files` 的 `try/except` 接住、跳过该文件。
- **id 是可选**:配了 `id_column` 就从行里取,否则 `gen_sha512_hash({"text": text}, ["text"])`——**用正文内容算哈希当 id**。
- **title 是可选**:配了 `title_column` 就取;否则用 `f"{path}{num}"`,其中 `num` 在该文件多于一行时是 `" ({index})"`,单行时为空。也就是说没配标题列时,标题形如 `data.csv (0)`、`data.csv (1)`。
- 每条都 `await self._storage.get_creation_date(path)` 取创建日期,并把整行原始字典塞进 `raw_data=row`。

这套逻辑让所有结构化格式只需各自负责"把文件解析成 `list[dict]`",列映射与文档构造全交给基类。

## 四种结构化 reader 的差异

四个具体类的 `read_file` 都极短,差异只在"如何把文件变成 `list[dict]`":

- **`CSVFileReader`(`csv.py`)**:默认 `file_pattern=".*\\.csv$"`。它 `await self._storage.get(path, encoding=self._encoding)` 拿到字符串,用标准库 `csv.DictReader(io.StringIO(file))` 解析成行字典。文件头部有一段 `csv.field_size_limit(sys.maxsize)`(`OverflowError` 时退回 `100 * 1024 * 1024`),是为了支持单元格里的超长文本字段。
- **`JSONFileReader`(`json.py`)**:默认 `file_pattern=".*\\.json$"`。`json.loads` 后判断:`rows = as_json if isinstance(as_json, list) else [as_json]`——既支持"整文件是一个对象",也支持"整文件是对象数组"。
- **`JSONLinesFileReader`(`jsonl.py`)**:默认 `file_pattern=".*\\.jsonl$"`。`rows = [json.loads(line) for line in text.splitlines()]`,每行一个独立 JSON 对象。
- **`ParquetFileReader`(`parquet.py`)**:默认 `file_pattern=".*\\.parquet$"`。它以 `as_bytes=True` 取字节流,用 `pyarrow.parquet.read_table(io.BytesIO(file_bytes))` 读表后 `table.to_pylist()` 转成行字典列表。

四者最终都 `return await self.process_data_columns(rows, path)`,复用同一套列映射。

## 两种非结构化 reader

剩下两种格式是"一文件一文档",它们直接继承 `InputReader` 而非 `StructuredFileReader`:

- **`TextFileReader`(`text.py`)**:默认 `file_pattern=".*\\.txt$"`。读出全文后构造单个 `TextDocument`:`id` 用 `gen_sha512_hash({"text": text}, ["text"])`,`title` 用 `Path(path).name`(纯文件名),`text` 是全文,`creation_date` 来自 storage,`raw_data=None`。
- **`MarkItDownFileReader`(`markitdown.py`)**:它没有重写 `__init__`,所以走基类的默认行为。`read_file` 以 `as_bytes=True` 取字节,用微软的 `MarkItDown().convert_stream(BytesIO(bytes), stream_info=StreamInfo(extension=Path(path).suffix))` 把 PDF/Word 等任意格式转成 Markdown 文本。`title` 优先用转换结果的 `result.title`,没有才退回文件名;`id` 同样是正文哈希,`raw_data=None`。这是 GraphRAG 支持"非纯文本文档"的统一入口。

## 文档 id:`hashing.py` 与稳定性

`hashing.py` 只有一个函数 `gen_sha512_hash(item, hashcode)`:把 `hashcode` 列出的键对应的值 `str()` 后拼接,再 `sha512(..., usedforsecurity=False).hexdigest()`。前面所有"没指定 id_column"的场景,id 都是 `gen_sha512_hash({"text": text}, ["text"])`,也就是**正文内容的 SHA-512**。

这个选择对整个流水线意义重大:**只要正文不变,文档 id 就不变**。这是内容寻址的思路,让缓存命中、去重、增量更新都有了稳定锚点。如果改用随机 UUID 或行号,同一篇文档每次重跑都会变 id,缓存和增量全失效。当然代价是:正文哪怕改一个字符,id 就变成"另一篇文档"。

## 取值工具:`get_property.py`

`get_property(data, path)` 是个小工具:把 `path` 按 `.` 拆开,逐层下钻字典,任意一层不是 dict 或键不存在就抛 `KeyError("Property '<path>' not found")`。它被 `StructuredFileReader`(取 text/id/title 列)和 `TextDocument.get`(取自由字段)共用,统一了"点号路径访问嵌套结构"的语义,使得 `text_column`、`title_column` 都可以指向嵌套字段。

## 接入流水线:两个 workflow

子包本身不知道流水线的存在,真正把它接进去的是主包 `graphrag/index/workflows/` 下的两个 workflow。

**`load_input_documents.py`** 是全量加载。`run_workflow` 调用 `create_input_reader(config.input, context.input_storage)` 拿到 reader,然后 `context.output_table_provider.open("documents")` 打开一张名为 `documents` 的输出表,交给 `load_input_documents(input_reader, documents_table)`。后者 `async for doc in input_reader` 流式遍历,每条 `asdict(doc)` 转成字典、补一个自增的 `row["human_readable_id"] = idx`、确保有 `raw_data` 键,然后 `await documents_table.write(row)` 落表;同时收集前 `sample_size=5` 条作为 sample。`total_count == 0` 时记 error 并抛 `ValueError("Error reading documents, please see logs.")`。`run_workflow` 还把总数写进 `context.stats.num_documents`,返回前 5 行的 `WorkflowFunctionOutput`。这里值得注意:reader 产出的 `TextDocument` 流式逐条写表,而非一次性全读进内存——配合 `_iterate_files` 的生成器设计,大语料也不会撑爆内存。

**`load_update_documents.py`** 是增量加载。它要求 `context.previous_table_provider` 非空,同样用 `create_input_reader` 建 reader,但 `load_update_documents` 把所有 doc 一次性收成 DataFrame,补 `human_readable_id`,然后调用 `get_delta_docs(input_documents, previous_table_provider)`。`get_delta_docs`(在 `index/update/incremental_index.py`)读取上一轮的 `documents` 表,**按 `title` 列做差集**:输入里 title 不在旧文档中的就是 `new_inputs`,旧文档里 title 不在本次输入中的就是 `deleted_inputs`,封装成 `InputDelta`。update workflow 只取 `new_inputs` 写回 `documents` 表;若没有新文档则 `WorkflowFunctionOutput(result=None, stop=True)` 提前终止整条流水线。

这里有个微妙点值得工程师留意:全量加载靠"正文哈希 id"保证幂等,而**增量 diff 用的是 `title`** 而不是 id。也就是说,增量判断"是不是新文档"是按标题去重的;对结构化输入,如果没配 `title_column`,标题是 `f"{path} ({index})"` 形式,这一点会直接影响增量识别的粒度。

## 本章小结

- `graphrag-input` 是个独立轻量子包(`3.1.0`),职责单一:把异构文件读成统一的 `TextDocument`,只依赖存储与工厂抽象,不反向依赖主包。
- 统一契约是 `TextDocument` 数据类:`id / text / title / creation_date` 四个标准字段加可选的 `raw_data`,后者保留结构化源数据的整行,使任意原始列可被透出。
- `InputReader` 抽象基类把"遍历文件 + 容错 + 统计"放在 `_iterate_files`,把"解析单文件"留给子类 `read_file`;单文件失败只 warning 跳过,不影响整批。
- 工厂 `create_input_reader` 用懒注册 + 延迟导入按 `InputType` 分发,既避免重依赖被无谓加载,又通过 `Factory.create` 剔除 `None` 参数让 reader 默认值生效。
- 四种结构化格式(csv/json/jsonl/parquet)共享 `StructuredFileReader.process_data_columns` 的列映射:text 必填、id/title 可选,差异只在"如何把文件解析成行字典"。
- 两种非结构化格式(text/markitdown)直接产出单文档,markitdown 借助微软 MarkItDown 把 PDF/Word 等转 Markdown,是非纯文本的统一入口。
- 文档 id 默认是正文的 SHA-512(`gen_sha512_hash`),内容寻址保证"内容不变则 id 不变",这是缓存、去重、增量的稳定基础。
- `get_property` 支持点号路径访问嵌套字典,让 `text_column / title_column` 可指向嵌套字段,并被 `TextDocument.get` 复用于自由字段读取。
- `load_input_documents` workflow 流式逐条写 `documents` 表并统计;`load_update_documents` 收成 DataFrame 后用 `get_delta_docs` **按 `title`** 做新增/删除差集,只索引新文档。
- 配置层 `InputConfig` 只表达用户意图(默认多为 `None` 且 `extra="allow"`),真正的兜底默认值由各 reader 的 `__init__` 补齐——一种清晰的配置/实现分层。

## 动手实验

1. **实验一:追踪一条 CSV 行的旅程** — 准备一个含 `text` 列的 `data.csv`,对照 `csv.py` 的 `read_file` 与 `StructuredFileReader.process_data_columns`,手动推演:不配 `id_column / title_column` 时,某一行的 `id`、`title`、`raw_data` 分别会是什么?验证 `id == gen_sha512_hash({"text": <该行text>}, ["text"])`,`title == "data.csv (<行号>)"`。

2. **实验二:验证容错与统计** — 构造一个目录,放两个 `.json` 文件,其中一个内容是坏 JSON(无法 `json.loads`)。阅读 `input_reader.py` 的 `_iterate_files`,解释为什么坏文件只会触发一条 `"...Skipping..."` warning 而好文件仍被加载;再对照末尾两条统计日志,推断 `file_count` 与 `doc_count` 各是多少。

3. **实验三:观察懒注册与依赖隔离** — 阅读 `input_reader_factory.py` 的 `create_input_reader` 与 `Factory.create`。说明当 `config.type` 为 `"parquet"` 时,`pyarrow` 是在哪一行、什么时机才被 import 的;再解释 `Factory.create` 里 `{k: v for k, v in ... if v is not None}` 这一行如何与 `InputConfig` 中默认 `None` 的字段配合,使 `CSVFileReader` 的内置 `file_pattern` 默认值最终生效。

4. **实验四:推演增量更新的 title 依赖** — 对照 `load_update_documents.py` 与 `incremental_index.py` 的 `get_delta_docs`。假设两轮输入是同一个 CSV、内容完全没变但未配 `title_column`,推演 `get_delta_docs` 按 `title` 差集会判定有几条 `new_inputs`;再思考:如果第二轮只改了某一行的正文(id 变了)但 title 仍是 `f"{path} (index)"` 形式,增量逻辑能否识别这次内容变更?给出你的结论与理由。

> **下一章预告**:拿到统一的 `documents` 表后,下一步是把长文档切成可被 LLM 处理的片段。第 4 章《文本切分 Chunking》将解构 GraphRAG 的分块策略——token/sentence 两种切法、chunk 大小与重叠、以及 chunk 如何回挂到源文档 id 上。
