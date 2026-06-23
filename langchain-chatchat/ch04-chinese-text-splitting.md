# 第 4 章：中文文本切分

上一章我们看到 `KnowledgeFile` 如何借助各类 loader 把磁盘上的文件读成一串 `Document` 对象。但 loader 交出来的往往是整篇文档：一个长达上万字的 `page_content`。要把它喂给嵌入模型、写进向量库、再被检索召回，必须先切成长度可控、语义相对完整的片段（chunk）。切分质量直接决定了召回的精度——切得太碎会丢失上下文，切得太大又会让一个 chunk 里混进太多无关内容、稀释向量语义。切分，是 RAG 链路里一个看似不起眼、实则决定上限的环节。

更棘手的是，主流的切分器是为英文设计的。英文以空格分词、以句号加空格断句，而中文是连续书写的方块字，没有词间空格，句末用的是全角的「。！？；」。如果直接拿 LangChain 默认的 `RecursiveCharacterTextSplitter` 处理中文，它会在不该断的地方硬切，把一个完整的句子从中间劈开。Langchain-Chatchat 因此在 `chatchat/server/file_rag/text_splitter/` 下自带了一组中文专用切分器，并通过配置名做成可插拔的工厂。本章就来逐一解构它们。

## 一个目录，四种切分器

切分器模块的入口非常克制，`text_splitter/__init__.py` 只导出四样东西：

```python
from .ali_text_splitter import AliTextSplitter
from .chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
from .chinese_text_splitter import ChineseTextSplitter
from .zh_title_enhance import zh_title_enhance
```

前三者是真正的切分器，对应三种思路：`ChineseRecursiveTextSplitter` 用中文标点感知的递归切分，是项目的默认选择；`ChineseTextSplitter` 用一连串正则做句号/分号断句并兜底超长句；`AliTextSplitter` 调用达摩院的文档分段模型做语义切分。第四个 `zh_title_enhance` 不是切分器，而是切分之后的一道增强工序，负责识别标题并把它注入后续片段。三种切分器都继承自 LangChain 的基类——前两者各自继承 `CharacterTextSplitter`，`ChineseRecursiveTextSplitter` 继承 `RecursiveCharacterTextSplitter`——因此天然兼容 `split_documents`、`chunk_size`、`chunk_overlap` 这套接口。

## 默认主力：ChineseRecursiveTextSplitter

`ChineseRecursiveTextSplitter` 继承自 LangChain 的 `RecursiveCharacterTextSplitter`。后者的核心思想是「按优先级递归回退」：先用最强的分隔符切，切出来的块若仍超过 `chunk_size`，就换下一级更细的分隔符继续切，直到块足够小为止 [[RecursiveCharacterTextSplitter](https://python.langchain.com/docs/concepts/text_splitters/)]。这个递归框架本身是通用的，中文版要做的只是把「分隔符列表」换成贴合中文标点的版本，再修正断句的方向。

构造函数里那份分隔符列表，是整个类的灵魂：

```python
self._separators = separators or [
    "\n\n",
    "\n",
    "。|！|？",
    "\.\s|\!\s|\?\s",
    "；|;\s",
    "，|,\s",
]
```

从上到下优先级递减。最先尝试的是段落级的 `\n\n` 与行级的 `\n`，这两级保留了文档作者天然划分的结构。接着是 `。|！|？`——全角的句号、叹号、问号，这是中文句末断句的主力。再往下是 `\.\s|\!\s|\?\s`，即英文句号/叹号/问号后跟一个空白字符，专门处理中英混排里夹杂的英文句子。然后是分号 `；|;\s`，最后才退到逗号 `，|,\s`。可以看到，每一级分隔符其实是一个正则，用 `|` 同时匹配全角和半角两种写法。这与构造函数默认的 `is_separator_regex=True` 相呼应——分隔符按正则解释，而非字面字符串。

注意它对应的配置项 `keep_separator=True`：切分时要把标点保留在结果里。如果断句时把句号丢掉，重组出来的 chunk 会变成没有标点的连续文字，既不利于阅读也损伤语义。

### 从句末断句：`_split_text_with_regex_from_end`

LangChain 原版的分隔逻辑会把分隔符当作下一句的开头来保留，对中文标点是反直觉的——中文的句号属于上一句的句末，而不是下一句的句首。该文件用一个模块级函数 `_split_text_with_regex_from_end` 重写了这个行为：

```python
def _split_text_with_regex_from_end(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    if separator:
        if keep_separator:
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]
```

关键在 `re.split(f"({separator})", text)`：用带捕获组的正则切分，结果里会交替出现「正文片段、分隔符、正文片段、分隔符……」。函数名中的 `from_end` 体现在 `zip(_splits[0::2], _splits[1::2])` 这一步——它把偶数位的正文和紧随其后的奇数位分隔符两两拼接，于是分隔符被粘到了它前面那段正文的末尾，而不是后面那段的开头。这样「今天天气很好。明天会下雨。」就会被切成 `["今天天气很好。", "明天会下雨。"]`，句号留在各自句尾，符合中文的阅读习惯。末尾的 `if len(_splits) % 2 == 1` 处理的是文本不以分隔符结尾、留下一段没有配对的尾巴的情况。最后 `[s for s in splits if s != ""]` 滤掉空串。

### 递归回退与最终清洗

`_split_text` 方法是对父类同名方法的覆写。它先在 `separators` 里从前往后找第一个能在当前文本中命中的分隔符（用 `re.search` 探测），把它作为本级分隔符，剩下的更细分隔符存进 `new_separators` 留待递归：

```python
for i, _s in enumerate(separators):
    _separator = _s if self._is_separator_regex else re.escape(_s)
    if _s == "":
        separator = _s
        break
    if re.search(_separator, text):
        separator = _s
        new_separators = separators[i + 1 :]
        break
```

随后用 `_split_text_with_regex_from_end` 切出 `splits`，逐个判断长度：小于 `chunk_size` 的累积进 `_good_splits`，攒够了就用父类的 `_merge_splits` 合并成贴近目标长度的块；遇到仍然超长的片段，则带着 `new_separators` 递归 `_split_text` 继续往下切。这正是「递归回退」的体现——长句会被一层层更细的分隔符拆开，直到满足长度约束。

方法结尾还有一道统一清洗，值得单独留意：

```python
return [
    re.sub(r"\n{2,}", "\n", chunk.strip())
    for chunk in final_chunks
    if chunk.strip() != ""
]
```

它把每个 chunk 内部连续两个以上的换行压成一个，`strip()` 去掉首尾空白，并丢弃清洗后为空的片段。这能消除 PDF、Word 抽取时常见的多余空行，让最终 chunk 干净紧凑。

文件末尾的 `if __name__ == "__main__"` 还附了一段自测：用一篇关于对外贸易的长文、`chunk_size=50`、`chunk_overlap=0` 跑一遍切分并逐块打印，方便开发者直观验证切分效果。

## 纯正则路线：ChineseTextSplitter

`ChineseTextSplitter` 继承 `CharacterTextSplitter`，走的是另一条路——不依赖递归，而是用一连串 `re.sub` 在断句符后插入换行 `\n`，再按 `\n` 切开。它的构造函数多了两个参数：`pdf` 控制是否先做 PDF 专属的空白归一化，`sentence_size: int = 250` 是单句的长度上限。

核心的 `split_text` 先在单字符断句符后断行：

```python
text = re.sub(r"([;；.!?。！？\?])([^"’])", r"\1\n\2", text)  # 单字符断句符
```

捕获组 `\1` 是断句符、`\2` 是它后面的非引号字符，替换成 `\1\n\2`，等于在两者之间插入换行。紧接着处理英文省略号 `\.{6}`、中文省略号 `\…{2}`，以及一条专门针对引号的规则——当 `。！？；` 后面跟着收尾引号时，把换行放到引号之后，从而保证「他说：『今天很好。』」这类带引号的句子不会在引号前被截断。

切成句子列表 `ls` 之后，还有一层「超长兜底」：对任何长度超过 `sentence_size` 的句子，依次尝试用逗号 `，`、再用空白、再用单个空格继续细分，层层嵌套地把过长片段压到上限以内。它的注释也很坦诚——方法签名旁直接写着 `##此处需要进一步优化逻辑`，说明这是一套以正则堆叠为主、规则较硬的实现。相比 `ChineseRecursiveTextSplitter` 由长度驱动的递归合并，`ChineseTextSplitter` 更像「先切到句、句太长再硬拆」，输出粒度通常更细。

## 语义切分路线：AliTextSplitter

如果说前两者都是基于标点和长度的「规则派」，`AliTextSplitter` 则是「模型派」。它的 `split_text` 调用 modelscope 上达摩院开源的文档分段模型 `damo/nlp_bert_document-segmentation_chinese-base`，按语义边界而非标点来切分：

```python
from modelscope.pipelines import pipeline

p = pipeline(
    task="document-segmentation",
    model="damo/nlp_bert_document-segmentation_chinese-base",
    device="cpu",
)
result = p(documents=text)
sent_list = [i for i in result["text"].split("\n\t") if i]
return sent_list
```

源码注释明确指出，该模型对应论文见 arXiv:2107.09278，且需要额外安装 `modelscope[nlp]` 才能使用 [[nlp_bert_document-segmentation_chinese-base](https://modelscope.cn/models/iic/nlp_bert_document-segmentation_chinese-base)]。模型推理后会在语义段落之间插入 `\n\t` 标记，代码据此 `split("\n\t")` 还原出段落列表。注意 `device="cpu"` 是写死的——注释解释这是为了照顾低配置 GPU、避免显存压力，需要的话可自行改成显卡 id。它对 `modelscope` 的导入用了 `try/except ImportError`，未安装时给出清晰的安装提示而非直接崩溃。语义切分的好处是切点落在意义转折处，缺点是要加载大模型、推理慢、依赖重——属于「想要更好语义边界、又能接受成本」时的可选项。

## 切分之后：中文标题增强

`zh_title_enhance.py` 不参与切分，而是在切分完成后增强 chunk 的可检索性。它的逻辑基于一个观察：文档里的标题（如「二、中国对外贸易发展环境分析」）本身信息密度极高，但切分时往往被单独切成一个很短的 chunk，与它统领的正文段落脱节。增强的做法是识别出标题，并把标题文字注入它后面那些正文 chunk。

判断「一行是不是标题」由 `is_possible_title` 这个启发式函数完成。它是一串否决式规则，任何一条不满足就判定不是标题：

- 空文本直接否决；
- 用正则 `r"[^\w\s]\Z"`（`ENDS_IN_PUNCT_PATTERN`）检测——以标点结尾的不是标题；
- 长度超过 `title_max_word_length`（默认 20）的不是标题；
- 非字母字符占比过高的不是标题，由辅助函数 `under_non_alpha_ratio` 按 `non_alpha_threshold=0.5` 判定，这能挡掉 `-----BREAK-----` 这类分隔线；
- 以 `,`、`.`、`，`、`。` 结尾的不是标题（防止把「致我亲爱的朋友们，」这类招呼语误判为标题）；
- 纯数字的不是标题；
- 开头 5 个字符内必须含数字，否则不是标题——这是为了匹配「一、」「1.」「第二章」这类带序号的中文标题样式。

通过全部检查后返回 `True`。真正的注入发生在 `zh_title_enhance` 主函数：

```python
def zh_title_enhance(docs: Document) -> Document:
    title = None
    if len(docs) > 0:
        for doc in docs:
            if is_possible_title(doc.page_content):
                doc.metadata["category"] = "cn_Title"
                title = doc.page_content
            elif title:
                doc.page_content = f"下文与({title})有关。{doc.page_content}"
        return docs
```

它顺序遍历切好的 `docs`：一旦某个 chunk 被判为标题，就在它的 `metadata` 里打上 `category="cn_Title"` 标记，并把它记成「当前标题」`title`；之后遇到的非标题 chunk，会被改写成 `下文与({title})有关。{原内容}`。这样正文片段的开头就携带了所属章节的语境，嵌入后的向量也会吸收标题信息，从而在检索时更容易被相关问题召回。需要强调的是，这是一道**默认关闭**的工序——配置项 `ZH_TITLE_ENHANCE` 默认值为 `False`，只有显式开启才会在 `docs2texts` 末尾调用。

## 可插拔的工厂：make_text_splitter

切分器虽多，使用方却不需要直接 `import` 任何一个具体类。`knowledge_base/utils.py` 里的 `make_text_splitter(splitter_name, chunk_size, chunk_overlap)` 充当工厂，按配置里的名字字符串动态实例化对应切分器。它最精巧的一笔是「先自定义、后内置」的双层查找：

```python
try:  # 优先使用用户自定义的text_splitter
    text_splitter_module = importlib.import_module("chatchat.server.file_rag.text_splitter")
    TextSplitter = getattr(text_splitter_module, splitter_name)
except:  # 否则使用langchain的text_splitter
    text_splitter_module = importlib.import_module("langchain.text_splitter")
    TextSplitter = getattr(text_splitter_module, splitter_name)
```

它先在项目自带的 `file_rag.text_splitter` 模块里用 `getattr` 按名字取类，取不到再回退到 `langchain.text_splitter`。于是 `ChineseRecursiveTextSplitter` 这类项目自带类，与 `SpacyTextSplitter`、`RecursiveCharacterTextSplitter` 这类 LangChain 原生类，能用同一个名字字符串统一调度。`MarkdownHeaderTextSplitter` 因为接口特殊（按标题层级切、不吃 `chunk_size`），被单独 `if` 出来用 `headers_to_split_on` 配置构造。

实例化的方式由 `text_splitter_dict` 里每个切分器的 `source` 字段决定：`tiktoken` 走 `from_tiktoken_encoder`、`huggingface` 走 `from_huggingface_tokenizer`（其中 `gpt2` 还会单独加载 `GPT2TokenizerFast`），其余则直接构造。构造时还做了一次 `try/except` 兜底——先试着传 `pipeline="zh_core_web_sm"`（为 spaCy 系切分器准备），失败了再退化成只传 `chunk_size`、`chunk_overlap`。整个函数最外层还包了一层 `except`，任何意外都会兜底回退到 LangChain 的 `RecursiveCharacterTextSplitter`，保证「切分器初始化失败也不至于让入库流程崩溃」。

工厂的调用方是 `KnowledgeFile.docs2texts`。它从 `Settings.kb_settings` 读出三个关键默认值：`TEXT_SPLITTER_NAME`（默认 `"ChineseRecursiveTextSplitter"`）、`CHUNK_SIZE`（默认 `750`）、`OVERLAP_SIZE`（默认 `150`），传给工厂造出切分器，再用 `split_documents` 把 `docs` 切成片段；若开启了 `ZH_TITLE_ENHANCE`，最后追加一遍 `func_zh_title_enhance`。至此，从「整篇文档」到「带标题语境的片段列表」的完整链路就闭合了。

## 本章小结

- 中文没有词间空格、句末用全角标点，直接套用英文切分器会从句中硬切，因此 Langchain-Chatchat 自带了一组中文专用切分器。
- `text_splitter/__init__.py` 导出三种切分器加一个标题增强函数；三种切分器分别继承 LangChain 的 `RecursiveCharacterTextSplitter` 或 `CharacterTextSplitter`，复用其标准接口。
- 默认主力 `ChineseRecursiveTextSplitter` 的分隔符列表按 `\n\n` → `\n` → 中文句末标点 → 英文句末加空格 → 分号 → 逗号 的优先级递减，每级都是兼容全/半角的正则。
- `_split_text_with_regex_from_end` 用带捕获组的 `re.split` 把分隔符粘到前一句句尾，符合「句号属于上句」的中文习惯；`keep_separator=True` 保证标点不丢失。
- `_split_text` 用 `re.search` 选定本级分隔符、长句递归回退到更细分隔符，并在结尾用 `re.sub(r"\n{2,}", "\n", ...)` 与 `strip()` 统一清洗。
- `ChineseTextSplitter` 走纯正则路线：在断句符后插 `\n` 再切，并对超过 `sentence_size`（默认 250）的句子逐层用逗号/空白兜底拆分，粒度更细。
- `AliTextSplitter` 调用达摩院 `nlp_bert_document-segmentation_chinese-base` 模型做语义切分，按 `\n\t` 还原段落，`device="cpu"` 写死，依赖 `modelscope`。
- `zh_title_enhance` 通过 `is_possible_title` 的一串否决式启发式（长度、结尾标点、非字母占比、开头含数字等）识别标题，给标题打 `category="cn_Title"`，并把标题以 `下文与(...)有关。` 的形式注入后续正文 chunk。
- 标题增强由 `ZH_TITLE_ENHANCE` 配置控制，默认 `False`，关闭时不影响切分主流程。
- `make_text_splitter` 是可插拔工厂：按名字字符串先查项目自带模块、再回退 LangChain，按 `source` 字段选择 tiktoken/huggingface/直接构造，并以 `RecursiveCharacterTextSplitter` 作为最终兜底。
- 默认配置为 `TEXT_SPLITTER_NAME="ChineseRecursiveTextSplitter"`、`CHUNK_SIZE=750`、`OVERLAP_SIZE=150`，由 `docs2texts` 串起整条切分链路。

## 动手实验

1. **实验一：观察递归切分的句末断句** — 直接运行 `chinese_recursive_text_splitter.py` 的 `__main__` 块（或自行用 `ChineseRecursiveTextSplitter(chunk_size=50, chunk_overlap=0)` 切一段中文），逐块打印结果，确认句号、叹号、问号都留在 chunk 末尾、没有从句中断开。
2. **实验二：对比三种切分器的粒度** — 取同一篇中文长文，分别用 `ChineseRecursiveTextSplitter`、`ChineseTextSplitter(sentence_size=250)` 切分，打印每种方案产出的 chunk 数量与平均长度，体会「长度驱动的递归合并」与「先切到句再硬拆」在粒度上的差异。
3. **实验三：验证标题增强** — 构造若干 `Document`，其中一个内容是「一、概述」、其余是正文，调用 `zh_title_enhance(docs)`，检查标题 chunk 的 `metadata["category"]` 是否变为 `cn_Title`、后续正文 `page_content` 是否被加上了 `下文与(一、概述)有关。` 前缀；再单独喂几个反例（以逗号结尾、纯数字、开头无数字）给 `is_possible_title`，确认它们都被判为非标题。
4. **实验四：切换切分器并触发兜底** — 把 `Settings.kb_settings.TEXT_SPLITTER_NAME` 改成一个不存在的名字，调用 `make_text_splitter("不存在的名字", 750, 150)`，观察它最终回退到 LangChain 的 `RecursiveCharacterTextSplitter`；再改成 `"SpacyTextSplitter"`，对照 `text_splitter_dict` 里的 `source` 字段理解它走的是哪条加载分支。

> **下一章预告**：切分产出的片段最终要落到一个具体的知识库里——但 Chatchat 支持 FAISS、Milvus、PGVector、ES 等多种后端。第 5 章将解构知识库服务抽象层，看 `KBService` 如何用统一接口屏蔽底层向量库差异、`KBServiceFactory` 如何按类型实例化，以及一份知识库文件从添加到向量化入库的完整生命周期。
