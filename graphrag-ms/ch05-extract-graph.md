# 第 5 章 图谱抽取 Extract Graph

第 4 章我们看到 GraphRAG 的索引流水线如何把原始文档切成一行行 text unit,每条带着自己的 `id` 与 `text` 列落进 `text_units` 表。切分本身并不产生任何"知识结构",它只是把长文档拆成 LLM 可以稳定处理的片段。真正让 GraphRAG 区别于普通向量 RAG 的一步,从本章开始:把这些纯文本片段读进语言模型,让它**抽出实体(节点)与关系(边)**,从而把一堆离散的文本块编织成一张可遍历、可聚类、可溯源的知识图谱。

本章解构的就是这条抽取链路:从 workflow 入口 `extract_graph` 怎样把 `text_units` 表喂给抽取算子,到 `GraphExtractor` 如何用一段精心设计的 prompt 让 LLM 吐出带分隔符的实体/关系记录、再用 gleaning 多轮补抽提升召回,最后如何把跨片段的同名实体合并、把每个节点和边钉回它的来源 text unit。所有结论都来自 `microsoft/graphrag 3.1.0` 主包 `packages/graphrag/graphrag/` 下的真实源码。

## workflow 入口:从 text_units 到 entities/relationships

抽取链路的最外层是 `graphrag/index/workflows/extract_graph.py` 中的 `run_workflow`。它先用 `DataReader(context.output_table_provider)` 拿到 reader,再 `await reader.text_units()` 把上一阶段产出的 text unit 表整体读进内存的 `pd.DataFrame`。注意这里它一次性把整张表加载出来,而不是流式逐行——这与同目录下基于 NLP 的 `extract_graph_nlp.py` 形成对照:后者用 `context.output_table_provider.open(...)` 打开表并把结果 `await table.write(row)` 流式写出。本章的主线是 LLM 抽取,走的是前者的内存 DataFrame 路径。

`run_workflow` 接着做了一件值得留意的事:它在同一个 workflow 里同时准备了**两个模型**。一个是抽取模型,通过 `config.get_completion_model_config(config.extract_graph.completion_model_id)` 取配置、`create_completion(...)` 实例化,缓存挂在 `context.cache.child(config.extract_graph.model_instance_name)` 下(默认实例名 `"extract_graph"`);另一个是摘要模型,取自 `config.summarize_descriptions`。也就是说,GraphRAG 把"抽取"和"描述摘要"放进了同一个 workflow 顺序执行:先抽出原始实体/关系,再立刻对它们的描述做摘要。两个模型用各自独立的 cache child,缓存分区互不污染。

真正干活的是模块内的协程 `extract_graph`(与 workflow 同名,但是下层函数)。`run_workflow` 把它的返回拆成四个 DataFrame:`entities, relationships, raw_entities, raw_relationships`。前两个是摘要后的最终结果,会无条件 `write_dataframe("entities", ...)` 和 `write_dataframe("relationships", ...)`;后两个是摘要**之前**的原始快照,只有当 `config.snapshots.raw_graph` 为真时才额外落盘成 `raw_entities` / `raw_relationships`。保留原始快照是个朴素但有用的工程取舍:摘要会改写 description,一旦怀疑摘要环节出问题,可以拿原始抽取结果对照排查,而不必重跑昂贵的 LLM 抽取。

## 抽取算子:逐 text unit 抽图、再合并

下沉一层到 `graphrag/index/operations/extract_graph/extract_graph.py` 的 `extract_graph` 算子。它的核心是把"对单条 text unit 抽图"这件事并行化。函数内定义了 `run_strategy(row)`,从每行取出 `row[text_column]`(`"text"`)和 `row[id_column]`(`"id"`),调用 `_run_extract_graph(...)` 完成单片抽取;然后交给 `derive_from_rows(text_units, run_strategy, callbacks, num_threads=..., async_type=...)` 在多个 text unit 之间并发执行。并发度来自 `config.concurrent_requests`,异步模式来自 `config.async_mode`——这两个参数一路从 `run_workflow` 透传下来,本质上是用并发把"对成百上千个片段各做一次 LLM 调用"的总时延压下来。

注释里写得很直白:`# this returns a graph for each text unit, to be merged later`。每条 text unit 抽出的是一张**局部子图**,彼此独立,合并是后续的事。`derive_from_rows` 返回的 `results` 是一组 `(entities_df, relationships_df)` 元组,代码用 `if result:` 过滤掉空结果,再分别 `pd.concat` 进 `entity_dfs` 和 `relationship_dfs`,交给 `_merge_entities` 和 `_merge_relationships` 做跨片段聚合。这种"map 出局部子图、reduce 成全局图"的结构,是把 LLM 抽取并行化的自然做法:单片抽取无状态、可独立缓存、可并发,合并则是纯 pandas 的确定性操作。

## prompt 设计:分隔符约定而非 JSON

单片抽取的灵魂在 prompt。默认模板是 `graphrag/prompts/index/extract_graph.py` 里的 `GRAPH_EXTRACTION_PROMPT`。它的结构是经典的"目标—步骤—示例—真实数据"四段式:`-Goal-` 告诉模型"给定文本和一组实体类型,识别所有该类型实体以及实体之间的关系";`-Steps-` 把任务拆成识别实体、识别关系、汇总输出、收尾四步;接着是三个 few-shot `-Examples-`;最后才是 `-Real Data-` 部分,用 `{entity_types}` 和 `{input_text}` 两个占位符注入真实的实体类型清单和待抽文本。

最值得玩味的是它**不要求 LLM 输出 JSON**,而是规定了一套分隔符约定。实体的格式是:

```
("entity"<|><entity_name><|><entity_type><|><entity_description>)
```

关系的格式是:

```
("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)
```

记录与记录之间用 `##` 分隔,整段输出结束时打一个 `<|COMPLETE|>`。这三个常量在 `graph_extractor.py` 顶部被显式定义:`TUPLE_DELIMITER = "<|>"`(字段分隔符)、`RECORD_DELIMITER = "##"`(记录分隔符)、`COMPLETION_DELIMITER = "<|COMPLETE|>"`(收尾标记)。

为什么不用 JSON?这是个有意识的取舍。一段长文本里可能抽出几十条实体和关系,如果要求严格 JSON,模型一旦在中途某条记录漏了引号或括号,整个 JSON 就解析失败、整片结果作废。而分隔符约定是**逐记录**的:`##` 把输出切成相互独立的片段,即便某一条记录格式有瑕疵,解析器也只丢弃那一条,其余记录照常入库。这种"宽进严解析"的格式对 LLM 自由生成的容错性远高于结构化 JSON,代价是需要自己写解析逻辑——下文会看到这套解析有多防御性。`relationship_strength` 用一个数值打分表示关系强度,后续会被当作边的权重 `weight`。

抽取的实体类型不是写死的,而是来自配置 `config.extract_graph.entity_types`,默认四类(见 `defaults.py` 的 `ExtractGraphDefaults`):`["organization", "person", "geo", "event"]`。在 `GraphExtractor._process_document` 里,这个列表用 `",".join(entity_types)` 拼成逗号串塞进 prompt 的 `{entity_types}`。把类型表外置成配置,意味着换个领域(比如医疗、法律)只要改这一项,就能让模型聚焦该领域关心的实体类别。

## gleaning:循环追问"还有没有漏掉的"

LLM 在一次性抽取长文本时几乎一定会漏。GraphRAG 应对漏抽的机制叫 **gleaning**(拾遗补抽),实现在 `GraphExtractor._process_document`。首轮抽取走标准流程:用 `CompletionMessagesBuilder().add_user_message(...)` 把格式化好的抽取 prompt 作为 user 消息发出,拿到 `response.content` 作为首轮结果,并 `add_assistant_message(results)` 把模型回复追加进对话历史。

随后,只要 `self._max_gleanings > 0`,就进入一个最多 `max_gleanings` 轮的循环。每一轮先追加一条 `CONTINUE_PROMPT` 的 user 消息——其内容是"上一次抽取漏掉了很多实体和关系,记住只输出此前定义过的类型,用相同格式补充在下面"(见 `prompts/index/extract_graph.py` 的 `CONTINUE_PROMPT`)。模型基于完整对话历史补抽,新结果直接 `results += response_text` 拼接到总结果上,同时也加进对话历史。

循环有两个退出条件,代码注释列得很清楚:`(a) we hit the configured max, (b) the model says there are no more entities`。其一是轮数到顶:`if i >= self._max_gleanings - 1: break`,最后一轮补抽完就不再问。其二是模型主动喊停:在非最后一轮,代码会再追加一条 `LOOP_PROMPT`,要求模型只回答单个字母 `Y` 或 `N`——"还有没有遗漏的实体/关系"。只有当 `response.content == "Y"` 时才继续下一轮,否则 `break`。

这个设计的动机直指**召回**。一次抽取漏掉的实体,通过"我知道你漏了,接着补"的追问能再捞回一批;而用 `Y/N` 探针让模型自评是否抽尽,可以在还没到轮数上限时提前止损,避免无谓的 LLM 调用。`max_gleanings` 默认是 `1`(见 `ExtractGraphDefaults.max_gleanings`),即默认在首轮之外再补一轮——在召回收益和 token 成本之间取一个保守平衡。把它调大能进一步提召回,但每加一轮都要多付一次完整对话历史的推理成本。

## 解析:把分隔符文本拆成结构化记录

模型吐出的(可能跨多轮拼接的)长字符串,要在 `GraphExtractor._process_result` 里被解析成两张 DataFrame。这是整段抽取里最讲究防御性的部分。

解析第一步是 `result.split(record_delimiter)` 用 `##` 把整段输出切成一条条 record,并对每条 `strip()`。接着对每条 record 用 `re.sub(r"^\(|\)$", "", raw_record.strip())` 去掉首尾的圆括号——因为 prompt 要求每条记录被一对 `(...)` 包裹。如果清洗后是空串,或者整条就是 `COMPLETION_DELIMITER`(即 `<|COMPLETE|>`),直接 `continue` 跳过。这一步把收尾标记和空记录滤掉,不让它们污染结果。

第二步用 `record.split(tuple_delimiter)` 即 `<|>` 把一条记录拆成字段数组 `record_attributes`,取 `record_attributes[0]` 作为 `record_type`。这里有两道严格的类型与长度校验:只有当 `record_type == '"entity"'` 且字段数 `>= 4` 时才当作实体处理;只有当 `record_type == '"relationship"'` 且字段数 `>= 5` 时才当作关系处理。注意判断的是带引号的字符串 `'"entity"'`,因为 prompt 里写的就是 `"entity"`(连引号一起输出)。任何字段数不足、类型对不上的脏记录,都会被这两道 `if` 静默跳过——这正是分隔符格式相比 JSON 的容错优势:坏记录被逐条丢弃,好记录照常入库。

字段值都过一遍 `clean_str`(来自 `index/utils/string.py`):它做 `html.unescape`、`strip`,并用正则 `re.sub(r"[\x00-\x1f\x7f-\x9f]", "", ...)` 抹掉控制字符。实体的 `entity_name`、`entity_type` 还会被 `.upper()` 统一成大写;关系的 `source`、`target` 同样大写。**全大写归一化是后续合并能跨片段对齐同名实体的关键**——它把 "Apple"、"APPLE"、"apple" 视为同一个节点。关系强度的解析更体现防御性:`weight = float(record_attributes[-1])` 取最后一个字段转浮点,一旦 `ValueError`(模型把分数写歪了),就 `weight = 1.0` 兜底,绝不让一条记录因为分数解析失败而丢掉整条边。

实体记录最终落成 `{"title", "type", "description", "source_id"}` 四个字段,关系记录落成 `{"source", "target", "description", "source_id", "weight"}`。这里 `source_id` 就是当前 text unit 的 `id`——**溯源链在解析这一步就已经钉上了**:每个实体、每条边都记住自己是从哪条 text unit 抽出来的。若整片文本一条有效记录都没解析出来,则返回 `_empty_entities_df()` / `_empty_relationships_df()`,即带正确列名的空表,保证下游 `pd.concat` 不会因为列缺失而炸。

异常的最外层兜底在 `GraphExtractor.__call__`:整个 `_process_document` 用 `try/except` 包住,一旦 LLM 调用或处理抛异常,就调用 `self._on_error(e, traceback, {...})` 记日志(`_run_extract_graph` 注入的回调会 `logger.error("Entity Extraction Error", ...)`),然后返回空实体/关系表。换句话说,**单条 text unit 抽取失败不会让整个 workflow 崩溃**,只是这一片贡献零节点零边,这对处理大规模、质量参差的语料至关重要。

## 合并:跨 text unit 聚合同名实体

回到算子层的 `_merge_entities` 和 `_merge_relationships`,它们把所有局部子图 reduce 成一张全局图。实体合并的逻辑是:`pd.concat` 所有片段的实体表后,按 `["title", "type"]` 分组聚合——也就是说,**实体的同一性由"名字 + 类型"共同决定**,同名但不同类型会被视为不同节点。聚合时三件事同时发生:`description=("description", list)` 把同一实体在各片段里的描述**收集成一个列表**(注意此时并不合并文字,只是攒着,留给下一章的 summarize 去融合);`text_unit_ids=("source_id", list)` 把所有来源 text unit id 收成列表,这就是该实体完整的溯源链;`frequency=("source_id", "count")` 统计它在多少个片段里出现过,这个频次后续会用于重要性排序。

关系合并类似,按 `["source", "target"]` 分组:description 同样收集成列表、`text_unit_ids` 收集来源、而权重用 `weight=("weight", "sum")` **累加**。把同一对实体之间在多个片段反复出现的关系强度相加,等于"出现得越频繁、各处给分越高的关系,边越粗"——这是一种朴素但有效的边权重信号。

合并完还有一道清洗,在 `operations/extract_graph/utils.py` 的 `filter_orphan_relationships`:它把所有实体的 `title` 收进集合,用 `relationships["source"].isin(entity_titles) & relationships["target"].isin(entity_titles)` 过滤,**丢弃那些 source 或 target 没有对应实体行的"悬空边"**。函数 docstring 解释了动机:LLM 可能在关系里幻觉出一个并未作为实体抽出来的名字,这种边指向不存在的节点,会让下游图处理遇到断裂的边。被丢弃的边数会以 `logger.warning` 记下来。把这道过滤放在合并之后,既保证了"边的两端必定是图里真实存在的节点",也顺手兜住了大写归一化后仍对不齐的边角情况。

至此,LLM 抽取这条主线就走完了:`run_workflow` 拿到的 `extracted_entities` / `extracted_relationships` 是带 description 列表、text_unit_ids 溯源、frequency/weight 信号的全局图。`extract_graph`(下层函数)还做了两道硬校验——抽完若 `len(extracted_entities) == 0` 或 `len(extracted_relationships) == 0`,直接 `raise ValueError`,因为没有节点或没有边的"图"对后续社区检测、社区报告毫无意义,早失败远好过让空图悄悄流进下游。校验通过后,它先 `.copy()` 出 `raw_entities` / `raw_relationships` 留快照,再把抽取结果交给 `get_summarized_entities_relationships`,用摘要模型把 description 列表融合成单条描述——那正是下一章的主题。

## 本章小结

- `extract_graph` workflow(`index/workflows/extract_graph.py`)用 `DataReader.text_units()` 一次性读入 text unit 表,产出 `entities` / `relationships` 两张表;开启 `snapshots.raw_graph` 时还会落盘摘要前的 `raw_entities` / `raw_relationships` 快照。
- 该 workflow 同时实例化抽取模型与摘要模型,各挂独立 cache child(抽取默认实例名 `"extract_graph"`),把抽取与描述摘要串在同一个 workflow 内顺序执行。
- 抽取算子按行并发:每条 text unit 抽出一张局部子图(map),`derive_from_rows` 用 `concurrent_requests` 控制并发,再由 `_merge_entities` / `_merge_relationships` 合并成全局图(reduce)。
- 默认 prompt `GRAPH_EXTRACTION_PROMPT` 用分隔符约定而非 JSON:`TUPLE_DELIMITER`(`<|>`)分字段、`RECORD_DELIMITER`(`##`)分记录、`COMPLETION_DELIMITER`(`<|COMPLETE|>`)收尾;实体类型来自 `entity_types`,默认 `organization/person/geo/event`。
- gleaning 多轮补抽(`_process_document`)用 `CONTINUE_PROMPT` 追问补抽、用 `LOOP_PROMPT` 让模型回 `Y`/`N` 自评是否抽尽,退出条件是触顶 `max_gleanings`(默认 `1`)或模型回 `N`,核心目的是提升实体/关系召回。
- 解析(`_process_result`)逐 `##` 切记录、去括号、按 `<|>` 拆字段,严格校验记录类型与字段数,脏记录逐条静默丢弃;`weight` 转浮点失败兜底为 `1.0`,`clean_str` 清洗并对实体名/关系端点 `.upper()` 归一化。
- 全大写归一化让同名实体可跨片段对齐;合并按 `["title","type"]` 聚合实体、按 `["source","target"]` 聚合关系,description 攒成列表、`text_unit_ids` 记溯源、`frequency` 计数、`weight` 累加。
- 溯源链在解析时就钉上(`source_id` = 当前 text unit 的 `id`),保证每个节点和边都能回指来源文本片段。
- 容错分层:单片抽取异常由 `__call__` 的 `try/except` 兜成空表不中断全局;`filter_orphan_relationships` 丢弃指向不存在实体的悬空边;但全局抽取若产出零实体或零关系,则 `raise ValueError` 早失败。

## 动手实验

1. **实验一:观察分隔符输出与解析** — 阅读 `graphrag/prompts/index/extract_graph.py` 的 `GRAPH_EXTRACTION_PROMPT`,挑一段你自己的短文本,手动按 prompt 格式写几条 `("entity"<|>...)` 和 `("relationship"<|>...)` 记录用 `##` 连接,再对照 `_process_result` 的逻辑逐字段走一遍,确认哪些会被解析成实体/关系、哪些会被两道 `if` 校验静默丢弃。
2. **实验二:复现 gleaning 退出条件** — 通读 `GraphExtractor._process_document` 的 gleaning 循环,把 `max_gleanings` 分别假设为 `0`、`1`、`3`,在纸上推演每种情况下会发出几次抽取请求、几次 `CONTINUE_PROMPT`、几次 `LOOP_PROMPT`,并标出两个 `break` 各自在什么条件触发。
3. **实验三:验证合并与溯源** — 构造两条 text unit,让它们都抽出同名实体(注意大小写不同,如 `Apple` 与 `APPLE`),在脑中或用一小段 pandas 脚本模拟 `_merge_entities` 的 `groupby(["title","type"])`,确认归一化后它们合并为一行,且 `text_unit_ids` 同时包含两条来源 id、`frequency` 为 2。
4. **实验四:对比 LLM 抽取与 NLP 抽取** — 并排阅读 `index/workflows/extract_graph.py` 与 `index/workflows/extract_graph_nlp.py`,列出二者在"实体类型来源(配置实体类型 vs `NOUN PHRASE`)、是否调用 LLM、description 是否为空、表写出方式(内存 DataFrame vs 流式 `table.write`)"四个维度上的差异,总结各自适合的场景。

> **下一章预告**:本章抽出的实体和关系都带着一串待融合的 description 列表与悬而未决的图结构。第 6 章《描述摘要与图谱终化》将解构 `summarize_descriptions` 如何用 LLM 把同一实体/关系的多条描述融合成一段连贯说明,以及 `finalize_graph`、`prune_graph` 如何为节点赋予稳定 id、计算度数与布局、并裁掉低价值的节点与边,把"原始抽取图"打磨成可供社区检测使用的终态图谱。
