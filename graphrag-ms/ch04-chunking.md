# 第 4 章 文本切分 Chunking

上一章我们把原始文件读成了一张 `documents` 表,每行是一篇完整文档。但 GraphRAG 的图谱抽取并不直接喂整篇文档给 LLM——上下文窗口装不下,长文档里实体与关系的密度也不均匀。GraphRAG 的解法是先把文档切成大小可控的「text unit」(文本单元),再以 text unit 为最小粒度做实体/关系抽取。这一步看似平平无奇,却是整条索引流水线的「分母」:后续每一个实体、每一条关系、每一段协变量,最终都要能回溯到产生它的那一个 text unit。

GraphRAG 3.1.0 把切分逻辑单独抽成了一个 PyPI 子包 `graphrag-chunking`,只依赖 `graphrag-common` 与 `pydantic`(见 `pyproject.toml` 的 `dependencies`),不依赖主索引包。这种「切分自成一包」的设计本身就传递了一个工程信号:切分是一个稳定、可复用、可单测的纯函数式组件。本章我们逐文件解构这个包,再回到主包的 `create_base_text_units` workflow,看文档如何流式地变成 `text_units` 表。

## 为什么切分要单独成包,以及与下游的关系

`graphrag-chunking` 这个包的全部代码只有十一个 `.py` 文件,核心抽象极小。它的 `pyproject.toml` 声明 `version = "3.1.0"`、`description = "Chunking utilities for GraphRAG"`,依赖里只有 `graphrag-common==3.1.0` 和 `pydantic~=2.10`。注意它**不**依赖 tiktoken、也不依赖 nltk 作为强制依赖(nltk 只在 sentence chunker 里 import)——这意味着 token 切分所需的编解码函数是从外部**注入**的,而不是包内硬编码。

这种注入式设计的意义在第 4.5 节会看到:主包的 `create_base_text_units` 用自己的 `Tokenizer`(可能是 tiktoken,也可能是基于 LiteLLM 的 tokenizer)生成 `encode`/`decode` 两个回调,再传给 chunker。切分包因此与具体的 tokenizer 实现解耦,既能被索引流水线用,也能被任何只想「按 token 切文本」的场景复用。

下游关系上,切分产物 `text_units` 表是图谱抽取(第 5 章)的输入。每个 text unit 携带一个 `document_id` 与一个内容哈希 `id`,后续抽取出的实体会记录「来源 text unit id」,从而把图谱节点钉回到原文片段——这是 GraphRAG 能做溯源问答的物理基础。

## 抽象基类 `Chunker` 与结果对象 `TextChunk`

切分包的契约由 `chunker.py` 里的抽象基类 `Chunker` 定义。它是个 `ABC`,只声明两个抽象方法:`__init__(self, **kwargs)` 和 `chunk(self, text, transform=None) -> list[TextChunk]`。`chunk` 接收一段文本和一个可选的 `transform: Callable[[str], str]`,返回一组 `TextChunk`。把 `transform` 作为参数而非构造期固定值,意味着同一个 chunker 实例可以对不同文档套用不同的「文本变换」(比如给某些文档前置元数据)。

结果对象 `TextChunk` 在 `text_chunk.py` 里是个 `@dataclass`,字段刻意区分了「原始」与「最终」两份文本:`original` 是变换前的原始片段,`text` 是变换后(若有 transform)的最终内容,`index` 是该 chunk 在源文档中的 0 基序号,`start_char`/`end_char` 是该 chunk 在源文档中的起止字符位置,`token_count` 是可选的最终文本 token 数。同时保留 `original` 和字符偏移,是为了让 chunk 既能被改写(前置元数据)又能精确定位回原文——这对溯源很关键。

## `ChunkerType` 与 chunker factory 的分发

切分支持哪些策略,由 `chunk_strategy_type.py` 的 `ChunkerType` 枚举(一个 `StrEnum`)给出,目前只有两个成员:`Tokens = "tokens"` 与 `Sentence = "sentence"`。配置侧 `chunking_config.py` 的 `ChunkingConfig` 是个 Pydantic `BaseModel`,`type` 默认 `ChunkerType.Tokens`,`size` 默认 `1200`,`overlap` 默认 `100`,还有 `encoding_model` 与 `prepend_metadata` 两个可选项。它声明了 `model_config = ConfigDict(extra="allow")`,允许携带额外字段,这样自定义 chunker 可以读到非标准配置。

分发逻辑在 `chunker_factory.py`。`ChunkerFactory` 继承自 `graphrag_common.factory.factory.Factory[Chunker]`,而这个 `Factory` 基类本身是一个**单例**(`__new__` 里用 `_instance` 缓存),内部维护 `strategy -> 初始化器` 的注册表,支持 `"singleton"`/`"transient"` 两种服务作用域。模块级实例化了一个全局 `chunker_factory = ChunkerFactory()`。

真正的入口是 `create_chunker(config, encode=None, decode=None)`。它做三件事:第一,把 `config.model_dump()` 摊平成 dict,并把外部传入的 `encode`/`decode` 函数塞进去;第二,做**惰性注册**——只有当 `config.type` 不在 factory 里时,才按 `match chunker_strategy` 现场 import 并注册对应实现(`ChunkerType.Tokens` → `TokenChunker`,`ChunkerType.Sentence` → `SentenceChunker`),命中不了就抛 `ValueError` 并列出已注册类型;第三,调用 `chunker_factory.create(chunker_strategy, init_args=config_model)` 实例化。惰性 import 的好处是:用 token 切分时根本不会 import nltk,反之亦然,避免无谓的重依赖加载。

值得一提的是 `Factory.create` 在实例化前会过滤掉 `init_args` 里值为 `None` 的项(见 `factory.py` 第 98 行),所以 `ChunkingConfig` 里那些默认 `None` 的字段不会污染构造函数,chunker 的默认值得以生效。

## token chunker:用注入的编解码按 token 数切

`token_chunker.py` 的 `TokenChunker` 是默认策略。构造函数接收 `size`、`overlap`、`encode`、`decode` 四个参数——注意它**不自带** tokenizer,编解码全靠外部注入。`chunk` 方法把活儿交给模块级函数 `split_text_on_tokens`,再用 `create_chunk_results` 把字符串列表包成 `TextChunk`。

`split_text_on_tokens` 是整个包里算法最实在的一段。它先 `encode(text)` 把整段文本编成 token 列表 `input_tokens`,然后用一个滑动窗口推进:`start_idx` 从 0 开始,`cur_idx = min(start_idx + chunk_size, len)`,取出 `chunk_tokens` 切片,`decode` 回字符串追加到结果;若 `cur_idx` 已到末尾就 break,否则 `start_idx += chunk_size - chunk_overlap` 前移。这里的关键是**步长是 `chunk_size - chunk_overlap`**:相邻 chunk 之间会重叠 `overlap` 个 token。重叠的目的,是避免一句话/一个实体恰好被切在两个 chunk 边界上而被双方都「截断」,从而保护跨边界的语义。tiktoken 的「编码成整数 token、再解码回文本」正是这种按模型 token 计数切分的标准做法[[tiktoken]](https://github.com/openai/tiktoken)。

按 token 而非按字符切,是因为下游的 LLM 上下文预算以 token 计;`size=1200`、`overlap=100` 这组默认值意味着每个 text unit 约 1200 token、相邻重叠 100 token,在「单次抽取信息量」与「重复成本」之间取了个折中。

## sentence chunker:用 nltk 按句切,与 `bootstrap_nltk`

`sentence_chunker.py` 的 `SentenceChunker` 提供另一种粒度:不按 token 数,而是按自然句。它的构造函数接收一个可选的 `encode`(仅用于事后计 token 数),并在 `__init__` 里调用 `bootstrap()`。`chunk` 方法用 `nltk.sent_tokenize(text.strip())` 把文本切成句子列表,再交给 `create_chunk_results`。NLTK 的 `sent_tokenize` 是其官方推荐的句子切分入口[[NLTK]](https://www.nltk.org/api/nltk.tokenize.html)。

这里有个真实世界的坑被代码显式处理了:nltk 的句子分词器会**裁掉句子之间的空白**,导致按 `create_chunk_results` 假设「chunk 未被裁剪」算出的 `start_char`/`end_char` 会偏移。所以 `SentenceChunker.chunk` 在拿到结果后做了一轮校正(第 36–47 行):对每个 chunk 用 `text.find(txt, start)` 在原文里找回真实起点,算出 `delta` 并把当前 chunk 以及下一个 chunk 的 `start_char`/`end_char` 向后顺移。这段补偿逻辑就是为了让句子级 chunk 的字符偏移仍然能精确对回原文。

`bootstrap_nltk.py` 解释了为什么要「下载 nltk 数据」。NLTK 的分词/词性标注/命名实体模型不是随库附带的,而是要按需从 nltk 数据服务器下载语料与模型包。`bootstrap()` 用一个模块级 `initialized_nltk` 标志做幂等保护(只下载一次),依次 `nltk.download(...)` 了 `punkt`、`punkt_tab`、`averaged_perceptron_tagger`、`averaged_perceptron_tagger_eng`、`maxent_ne_chunker`、`maxent_ne_chunker_tab`、`words`、`wordnet`,最后 `wn.ensure_loaded()` 预载 WordNet。文件顶部还用 `warnings.filterwarnings` 静默了两条来自 numba 的告警。`punkt`/`punkt_tab` 正是 `sent_tokenize` 句子切分所依赖的预训练模型,缺了它句子切分会直接报错——这就是这次下载存在的根本原因。

## `create_chunk_results` 与字符偏移的统一计算

两种 chunker 切出的都是 `list[str]`,把字符串列表升格成 `list[TextChunk]` 的统一逻辑在 `create_chunk_results.py`。它从 `start_char = 0` 开始线性扫描:对第 `index` 个 chunk,`end_char = start_char + len(chunk) - 1`(0 基、闭区间),构造 `TextChunk(original=chunk, text=transform(chunk) if transform else chunk, ...)`,若提供了 `encode` 还会算 `token_count = len(encode(result.text))`,然后 `start_char = end_char + 1` 推进到下一个。

注意它的偏移计算建立在一个明确假设上(docstring 写得很清楚):**chunk 没有相对源文本被裁剪**。这正解释了上一节 sentence chunker 为何要额外做校正——token chunker 的解码结果是连续覆盖原文 token 的,这个假设成立;而 nltk 句子切分会丢空白,假设被打破,故须事后修正。把偏移计算和 transform 应用收敛到这一个函数,是为了让两种策略的结果结构完全一致,下游不必关心它是怎么切出来的。

`transformers.py` 提供了内置变换 `add_metadata(metadata, delimiter=": ", line_delimiter="\n", append=False)`。它返回一个闭包 `transformer(text)`,把元数据 dict 渲染成 `key: value` 的多行字符串,默认**前置**(prepend)到 chunk 文本前,`append=True` 时则追加在后。这就是配置项 `prepend_metadata` 的实现载体。

## `create_base_text_units` workflow:文档 → text unit 表

回到主包,`graphrag/index/workflows/create_base_text_units.py` 是把上面这些零件组装进索引流水线的地方。`run_workflow` 先用 `get_tokenizer(encoding_model=config.chunking.encoding_model)` 拿到 tokenizer,再 `create_chunker(config.chunking, tokenizer.encode, tokenizer.decode)`——这正是「编解码函数从外部注入切分包」的落点。`get_tokenizer`(在 `graphrag/tokenizer/get_tokenizer.py`)的默认路径是构造一个 `TokenizerType.Tiktoken` 的 tokenizer,编码名取自 `config.chunking.encoding_model` 或回退到 `ENCODING_MODEL`。

随后它用 `context.output_table_provider.open(...)` 同时打开输入表 `documents` 与输出表 `text_units`,把两张表连同 tokenizer、chunker、`prepend_metadata` 一起交给 `create_base_text_units`。这个核心协程刻意采用**流式读写**:`async for doc in documents_table` 一行一行读文档,切完立刻 `await text_units_table.write(row)` 写出,不把全量数据载入内存——docstring 明确写了「avoiding loading all data into memory」,这是面向大语料的工程取舍。它还顺带收集前 5 行做 `sample_rows` 返回,供流水线观测。

每个 chunk 写出的行结构很简单:`{"id": "", "document_id": doc["id"], "text": chunk_text, "n_tokens": len(tokenizer.encode(chunk_text))}`。`document_id` 直接取自源文档行的 `id`,这就是把 text unit 钉回原文档的关联键。`n_tokens` 用工作流自己的 tokenizer 再算一遍(不复用 `TextChunk.token_count`)。`id` 先置空,随后 `row["id"] = gen_sha512_hash(row, ["text"])`——**以 chunk 文本内容做 SHA-512 哈希**作为 text unit 的稳定 id。用内容哈希而非自增序号,好处是相同文本切出的 unit 有确定性 id,缓存与增量更新时可天然去重、可复现。

`chunk_document` 负责切单篇文档。若配置了 `prepend_metadata`,它会用 `graphrag_input.TextDocument` 包装该行(带上 `title`、`creation_date`、`raw_data` 等),调用 `document.collect(prepend_metadata)` 收集指定元数据字段,再用 `add_metadata(metadata=metadata, line_delimiter=".\n")` 造出 transformer 传给 `chunker.chunk(doc["text"], transform=transformer)`。最终它只取每个 `TextChunk` 的 `.text` 返回字符串列表——也就是说,workflow 这一层只关心最终文本,偏移与 `original` 留在切分包内部供需要的场景使用。把文档标题、日期等元数据前置进每个 chunk,是为了让被切碎的片段不丢失「它来自哪篇、什么时候」的上下文,从而提升后续抽取与检索的准确性。

## 与遗留 `TokenTextSplitter` 的关系

主包里还留着一份 `graphrag/index/text_splitting/text_splitting.py`,内有 `TokenTextSplitter` 与 `split_single_text_on_tokens`。把它和 `graphrag-chunking` 的 `split_text_on_tokens` 并排看,会发现滑窗算法几乎逐行一致(同样的 `start_idx += tokens_per_chunk - chunk_overlap` 步长)。区别是:`TokenTextSplitter` 自带 `get_tokenizer()` 默认 tokenizer、默认 `chunk_size=8191`(贴着 OpenAI embedding 限制)、并处理 `pd.isna`/空串等输入边界,更像服务于嵌入/向量化场景的旧式工具类;而新的 `graphrag-chunking` 把同一算法抽成了无状态、依赖注入、可注册扩展的独立包。理解这层并存,有助于在阅读流水线时不被两套相似 API 绕晕——索引主路径走的是 `create_chunker`,这份遗留 splitter 服务于另一些消费点。

## 本章小结

- GraphRAG 3.1.0 把切分独立成 `graphrag-chunking` 子包,只依赖 `graphrag-common` 与 `pydantic`,tokenizer 与 nltk 都不是强制依赖,切分逻辑与具体 tokenizer 解耦。
- 契约由抽象基类 `Chunker.chunk(text, transform)` 定义,结果是 `TextChunk` dataclass,刻意区分 `original`/`text` 并保留 `start_char`/`end_char`/`index`/`token_count`,兼顾可改写与可溯源。
- `ChunkerType` 枚举目前只有 `Tokens` 与 `Sentence`;`create_chunker` 通过单例 `chunker_factory` 做**惰性注册**,按需 import 对应实现,避免无谓加载重依赖。
- `TokenChunker` 用外部注入的 `encode`/`decode`,在 `split_text_on_tokens` 里以 `chunk_size - overlap` 为步长滑窗,相邻 chunk 重叠 `overlap` 个 token 以保护跨边界语义;默认 `size=1200`、`overlap=100`。
- `SentenceChunker` 用 `nltk.sent_tokenize` 按句切,并因 nltk 裁剪空白而显式校正字符偏移;`bootstrap_nltk.bootstrap()` 幂等下载 `punkt`/`punkt_tab` 等语料,缺失会导致句子切分失败。
- `create_chunk_results` 在「chunk 未被裁剪」的假设下统一计算 0 基闭区间字符偏移并应用 transform,把两种策略的输出结构对齐。
- `transformers.add_metadata` 是 `prepend_metadata` 的实现:把源文档元数据渲染成多行文本前置进 chunk,保留「来自哪篇/何时」的上下文。
- `create_base_text_units` workflow 用 `get_tokenizer` 造出 `encode`/`decode` 注入切分包,流式读 `documents`、写 `text_units`,避免全量入内存。
- 每个 text unit 行带 `document_id`(回溯原文档)与用 `gen_sha512_hash` 对 `text` 算出的内容哈希 `id`,保证确定性、可去重、可增量复现。
- 主包遗留的 `TokenTextSplitter` 与新包共享同一滑窗算法,但默认参数(`8191`)与定位不同,索引主路径走 `create_chunker`。

## 动手实验

1. **实验一:复现滑窗重叠** — 直接照搬 `token_chunker.py` 的 `split_text_on_tokens`,用 `tiktoken.get_encoding("cl100k_base")` 的 `encode`/`decode` 作回调,对一段长文本设 `chunk_size=20`、`chunk_overlap=5` 运行,打印每个 chunk 的首尾几个词,验证相邻 chunk 确实重叠了约 5 个 token,并把 `overlap` 改成 0 对比边界处的语义截断。

2. **实验二:句子切分与偏移校正** — 安装并 `bootstrap()` nltk 后,用 `SentenceChunker().chunk(text)` 切一段含多句、句间有多空格/换行的文本;打印每个 `TextChunk` 的 `text` 与 `start_char`/`end_char`,再用 `text[start:end+1]` 切原文核对是否对得上,以此理解第 36–47 行偏移校正的必要性(试着注释掉校正逻辑看会差多少)。

3. **实验三:factory 惰性注册** — 调用 `create_chunker(ChunkingConfig(type="tokens"), encode, decode)`,在调用前后分别 `print(chunker_factory.keys())`,观察 `tokens` 是被惰性注册进去的;再传一个不存在的 `type="paragraph"`,确认抛出列出已注册类型的 `ValueError`;最后用 `register_chunker` 注册一个自定义 chunker 并成功创建它。

4. **实验四:prepend_metadata 端到端** — 构造一条假 `documents` 行(含 `id`/`title`/`text`/`creation_date`),分别在 `prepend_metadata=None` 与 `prepend_metadata=["title","creation_date"]` 两种配置下调用 `chunk_document(doc, chunker, prepend_metadata)`,对比每个 chunk 文本开头是否多了元数据行;再对返回文本套 `gen_sha512_hash`,验证相同文本得到相同 `id`、改一个字符 id 即变。

> **下一章预告**:文档已被切成带 `document_id` 与内容哈希的 text unit。第 5 章《图谱抽取 Extract Graph》将进入 GraphRAG 的核心——如何用 LLM 在每个 text unit 上抽取实体与关系、做多轮 gleaning 补抽、并把抽出的图谱节点钉回它来源的 text unit。
