# 第 7 章 上下文提取 ContextExtractor

在第 6 章我们看到 MinerU 把一份文档拆成一长串 `content_list`，每个元素是一张图、一个表、一个公式或一段文本。问题随之而来：当流水线把第 12 项「一张柱状图」单独交给 VLM 去生成描述时，模型看到的只有这张图本身——它不知道这张图出现在哪一节、上文在讨论什么、图下方那行小字 caption 写了什么。孤立地描述一张图、一个表、一个公式，几乎必然丢失语义：同一张「逐月增长的柱状图」，在「营收报告」语境下和在「服务器负载」语境下，应当生成完全不同的描述。

`ContextExtractor` 就是为解决这个问题而生的。它在调用多模态模型**之前**，把当前项**周边**的文本——前后若干页（或前后若干 chunk）、所属标题层级、邻近图表的 caption——抽取出来，作为「上下文」一并喂给模型。这是 RAG-Anything 把多模态描述质量从「看图说话」提升到「结合语境说话」的核心机制。本章逐行拆解 `raganything/modalprocessors.py` 前半部分（40–365 行）的实现。

## 7.1 配置载体：`ContextConfig`

提取行为由一个 dataclass `ContextConfig`（第 40 行）集中描述，它的六个字段构成了上下文提取的全部旋钮：

- `context_window: int = 1`：窗口大小，表示当前项前后各取多少页 / 多少 chunk。
- `context_mode: str = "page"`：提取模式，注释里写明支持 `"page"`、`"chunk"`、`"token"`。
- `max_context_tokens: int = 2000`：上下文 token 预算上限。
- `include_headers: bool = True`：是否把标题 / 层级标记纳入上下文。
- `include_captions: bool = True`：是否把图 / 表的 caption 纳入上下文。
- `filter_content_types: List[str] = None`：允许进入上下文的内容类型。

注意最后一个字段的默认值是 `None` 而非列表——这是 Python 可变默认参数的经典规避写法。`__post_init__`（第 50 行）在实例化后补上真正的默认：

```python
def __post_init__(self):
    if self.filter_content_types is None:
        self.filter_content_types = ["text"]
```

所以**默认只有纯文本会进入上下文**。这是一个有意为之的保守选择：在为某张图生成描述时，把同一页另外几张图的图像数据也塞进上下文没有意义，反而会污染语义、浪费 token。

这些默认值并非凭空写死。在 `config.py`（78–110 行）中，每个字段都通过 `get_env_value` 暴露成环境变量：`CONTEXT_WINDOW`（默认 `1`）、`CONTEXT_MODE`（默认 `"page"`）、`MAX_CONTEXT_TOKENS`（默认 `2000`）、`INCLUDE_HEADERS` / `INCLUDE_CAPTIONS`（默认 `True`）、`CONTEXT_FILTER_CONTENT_TYPES`（默认 `"text"`，按逗号切分成列表）。`raganything.py` 的 `_create_context_config`（第 181 行）把这些配置原样搬进 `ContextConfig`，其中 `filter_content_types` 取自 `context_filter_content_types`。

## 7.2 用 LightRAG 的 tokenizer 构建提取器

`ContextExtractor.__init__`（第 58 行）只持有两样东西：`self.config`（提取配置）与 `self.tokenizer`（分词器）。tokenizer 是可选的，但在真实运行中一定会被注入。`raganything.py` 的 `_create_context_extractor`（第 192 行）规定：必须先初始化 `LightRAG`，然后

```python
return ContextExtractor(config=context_config, tokenizer=self.lightrag.tokenizer)
```

也就是说，上下文提取器**复用了 LightRAG 的同一个 tokenizer**（参见第 2 章对组件装配的讲解）。这一点很关键：上下文最终要拼进 prompt 喂给下游 LLM/VLM，用与主系统一致的分词器来计算 token 数，才能保证「2000 token 预算」与模型实际消耗的口径一致，不会因为换了一套估算方式而高估或低估。

## 7.3 主入口 `extract_context` 与格式分发

`extract_context`（第 68 行）是唯一的对外入口，接收三个参数：`content_source`（内容源，可以是 list / dict / str）、`current_item_info`（当前项信息，含 `page_idx`、`index` 等定位字段）、`content_format`（格式提示，默认 `"auto"`）。

它首先做一个短路判断：`if not content_source and not self.config.context_window: return ""`——既没有内容源、窗口又为 0，直接返回空串。随后按 `content_format` 分发：

- `"minerU"` + list → `_extract_from_content_list`
- `"text_chunks"` + list → `_extract_from_text_chunks`
- `"text"` + str → `_extract_from_text_source`
- 否则进入**自动探测**：按 `content_source` 的实际类型（list / dict / str）选择对应方法，类型不支持就 `logger.warning` 并返回空串。

整个方法体包在 `try/except` 里，任何异常都被 `logger.error` 记录并返回空串。这是一种「上下文是锦上添花，提取失败也不能让主流程崩」的防御式设计——上下文提取再重要，也不应阻断对该项本身的多模态处理。`config.py` 里 `content_format` 默认为 `"minerU"`，与 RAG-Anything 主链路用 MinerU 解析文档相吻合。

## 7.4 两种模式：`page` 与 `chunk`

当内容源是 MinerU 风格的 list 时，`_extract_from_content_list`（第 120 行）根据 `context_mode` 二选一：`"page"` 走 `_extract_page_context`，`"chunk"` 走 `_extract_chunk_context`，任何其他取值都**回退到 page 模式**（`else` 分支）。

**按页提取**（`_extract_page_context`，第 139 行）。它从 `current_item_info` 取出 `page_idx` 作为 `current_page`，以 `context_window` 为半径算出页区间：

```python
start_page = max(0, current_page - window_size)
end_page = current_page + window_size + 1
```

然后遍历整个 `content_list`，对每一项检查两个条件——页号落在 `[start_page, end_page)` 区间内，**且** `item_type` 命中 `filter_content_types`。满足条件就调 `_extract_text_from_item` 取文本。这里有个体贴的细节：若该项不在当前页（`item_page != current_page`），会给文本加上 `[Page {item_page}]` 前缀，让模型知道这段上下文来自相邻哪一页；同页文本则不加标记。最后所有片段以换行拼接，交给 `_truncate_context` 截断。

**按 chunk 提取**（`_extract_chunk_context`，第 179 行）。它改用 `current_item_info` 里的 `index`（项在列表中的序号），区间按序号计算：`start_idx = max(0, current_index - window_size)`、`end_idx = min(len(content_list), current_index + window_size + 1)`。遍历区间内的项时，**显式跳过当前项**（`if i != current_index`），其余项同样经过 `filter_content_types` 过滤后取文本。

两种模式适配不同的文档结构：`page` 模式利用 MinerU 给出的 `page_idx`，适合幻灯片、扫描件这类「页」语义清晰、版面分明的文档；`chunk` 模式不依赖页号，只看元素在序列中的相邻关系，适合长 PDF、网页这类页边界模糊、但内容连续的文档。默认 `page` 模式，因为 RAG-Anything 的主输入正是 MinerU 解析出的带 `page_idx` 的 `content_list`。

## 7.5 单项取文本：`_extract_text_from_item`

`_extract_text_from_item`（第 212 行）决定「从一个 content 元素里能抠出什么文本」，按 `type` 分三类：

- `"text"` 类：取 `item["text"]`。若 `include_headers` 为真且该项有 `text_level > 0`（MinerU 用 `text_level` 标记标题层级），则按层级在前面补 Markdown 井号：`f"{'#' * text_level} {text}"`。这样一级标题变 `#`、二级变 `##`，把文档的**结构层级**也传递给了模型。
- `"image"` 类：仅当 `include_captions` 为真才处理，取 `image_caption`（兼容旧字段 `img_caption`），有 caption 时返回 `f"[Image: {...}]"`。
- `"table"` 类：同样在 `include_captions` 下，取 `table_caption`，返回 `f"[Table: {...}]"`。

关键在于：**默认 `filter_content_types` 只放行 `"text"`**，所以正常情况下 `image`/`table` 分支根本不会被触发——除非用户显式把 `"image"`/`"table"` 加进过滤列表。`include_captions` 的真正价值在这种放宽配置下才体现：它让邻近图表「只贡献一行标题」而非整张图，是一种「轻量纳入」的折中。无 caption 或类型不匹配时，方法返回空串，上游再用 `text_content.strip()` 把空白片段过滤掉。

## 7.6 三种内容源适配

同一个提取器要服务不止一种数据格式，所以除了 MinerU 的 list，还有另外三条适配路径：

- `_extract_from_dict_source`（第 244 行）：面向字典源。优先取 `dict_source["content"]`，其次 `dict_source["text"]`，两者都没有时**兜底**遍历所有值、把字符串类型的值拼起来。结果统一过 `_truncate_context`。
- `_extract_from_text_source`（第 271 行）：面向纯文本源。最简单——直接把整段文本交给 `_truncate_context`，靠截断控制长度。
- `_extract_from_text_chunks`（第 285 行）：面向 `List[str]` 形式的文本块。逻辑与 `_extract_chunk_context` 同构：用 `index` 算窗口区间、跳过当前块（`if i != current_index`）、`strip()` 后非空才纳入，最后拼接截断。区别在于这里的元素是裸字符串，不需要 `type` 过滤，也不取 caption。

这三条路径加上 MinerU 的 list，意味着无论上游送来的是结构化 content_list、纯文本，还是简单的 text_chunks，`ContextExtractor` 都能用统一的接口产出上下文——这正是类注释里所说的 "supporting multiple content source formats"。

## 7.7 token 预算与 `_truncate_context`

所有路径的最后一步都是 `_truncate_context`（第 314 行）。它的职责是确保上下文不会撑爆 prompt——若不加约束，一份长文档的相邻几页文本动辄上万 token，会把宝贵的 prompt 空间挤占殆尽，甚至超出模型上下文窗口。

逻辑分两支。**有 tokenizer** 时（真实运行路径）：先 `encode` 数 token，未超 `max_context_tokens` 就原样返回；超了则截到前 `max_context_tokens` 个 token，再 `decode` 回文本。截断后还做了一步「优雅收尾」：找最后一个句号 `.` 和最后一个换行 `\n` 的位置，若句号位置超过文本长度的 80%（`> len * 0.8`）就截到句号处，否则若换行满足同样条件就截到换行处，再否则就在末尾补 `...`。这避免了把句子拦腰砍断。**无 tokenizer** 时退化为按字符数截断，用 `max_context_tokens` 当字符上限，收尾逻辑相同。

这里体现了一个清醒的工程判断：上下文是「预算受限的资源」。`max_context_tokens=2000` 是一个默认预算，用与下游模型一致的 tokenizer 精确计量，宁可在句子边界处优雅收尾、丢掉尾部，也不让上下文无限膨胀。结合 7.1 的 `filter_content_types=["text"]` 默认，可以看出整个 `ContextExtractor` 的设计哲学：**给模型够用、干净、有边界的语境，而不是把周边内容一股脑倒进去。**

## 本章小结

- 多模态项（图 / 表 / 公式）孤立地交给模型描述会丢失语义，`ContextExtractor` 通过抽取**周边文本**作为上下文来提升描述质量，是 RAG-Anything 的核心增强机制。
- `ContextConfig`（第 40 行）集中六个旋钮：`context_window`（默认 1）、`context_mode`（默认 `"page"`）、`max_context_tokens`（默认 2000）、`include_headers`、`include_captions`（默认 `True`）、`filter_content_types`（`__post_init__` 补默认 `["text"]`）。
- 这些默认值在 `config.py` 中通过 `get_env_value` 全部暴露为环境变量，由 `_create_context_config` 搬进配置对象。
- 提取器复用 `LightRAG` 的同一个 `tokenizer`（`_create_context_extractor`），保证 token 计量口径与下游模型一致。
- `extract_context`（第 68 行）是唯一入口，按 `content_format` 分发或自动探测类型，全程包在 `try/except` 中，失败只记日志返回空串，绝不阻断主流程。
- `page` 模式（按 `page_idx` 取前后页，给跨页文本加 `[Page N]` 标记）与 `chunk` 模式（按 `index` 取前后窗口、跳过当前项）适配不同文档结构，默认 `page`。
- `_extract_text_from_item` 对 `text` 项按 `text_level` 补 Markdown 标题井号；`image`/`table` 项仅在放宽 `filter_content_types` 后才以 `[Image: ...]`/`[Table: ...]` 形式贡献 caption。
- 同一提取器通过 `_extract_from_dict_source`、`_extract_from_text_source`、`_extract_from_text_chunks` 适配字典、纯文本、文本块三种额外来源。
- `_truncate_context` 用 tokenizer 按 `max_context_tokens` 截断，并在句号 / 换行的 80% 阈值处优雅收尾，无 tokenizer 时退化为字符截断，体现「token 预算」设计。

## 动手实验

1. **实验一：观察两种 context_mode 的差异** — 构造一个含 `page_idx` 与 `type` 字段的小 `content_list`（跨 3 页），分别用 `ContextConfig(context_mode="page")` 与 `ContextConfig(context_mode="chunk")` 实例化 `ContextExtractor`，对同一个 `current_item_info` 调 `extract_context`，对比 page 模式输出里的 `[Page N]` 前缀与 chunk 模式按序号取窗的结果。
2. **实验二：放宽 filter_content_types 看 caption 纳入** — 先用默认配置（仅 `["text"]`）跑一次，确认 image/table 不出现在上下文；再设 `filter_content_types=["text","image","table"]` 且 `include_captions=True`，给 content_list 里的图表加上 `image_caption`/`table_caption`，观察输出中出现的 `[Image: ...]`/`[Table: ...]` 片段。
3. **实验三：验证 token 预算截断** — 拼一段远超 2000 token 的长文本作为 `text` 源，注入一个真实 tokenizer，把 `max_context_tokens` 依次设为 50、500、2000，调 `_extract_from_text_source`，用 `tokenizer.encode` 数返回结果的 token 数，并观察末尾是落在句号、换行还是补了 `...`。
4. **实验四：跑通三种来源适配** — 分别用 list（MinerU 风格）、dict（带 `content` 键 / 不带键两种）、`List[str]`（text_chunks）三种 `content_source` 各调一次 `extract_context`，确认它们分别路由到 `_extract_from_content_list`、`_extract_from_dict_source`、`_extract_from_text_chunks`，并对照 dict 兜底分支（无 `content`/`text` 键时拼接所有字符串值）的行为。

> **下一章预告**：拿到了干净的上下文，下一步就是真正调用模型生成描述。第 8 章将拆解构建在 `ContextExtractor` 之上的多模态处理器四件套——`ImageModalProcessor`、`TableModalProcessor`、`EquationModalProcessor` 与 `GenericModalProcessor`，看它们如何把上下文与各自的 prompt 模板组合，分别处理图像、表格、公式与其他模态。
