# 第 8 章 多模态处理器四件套：把图片、表格、公式插进知识图谱

上一章我们拆解了 `ContextExtractor`——它负责为每一个多模态项抽取周边的页面/分块上下文。但上下文只是「配料」，真正把一张图、一张表、一个公式变成可被检索召回的知识，靠的是 `modalprocessors.py` 里四个并列的处理器：`ImageModalProcessor`、`TableModalProcessor`、`EquationModalProcessor` 和兜底的 `GenericModalProcessor`。它们共同回答一个工程问题：**异构的非文本内容，如何统一地转成知识图谱中的实体节点，从而和正文文本一样参与 LightRAG 的双层检索？**

本章聚焦这四个处理器本身。我们会看到它们如何共享一套「上下文注入 → 调模型生成描述 → 健壮解析 JSON → upsert 进图谱」的流水线（模板方法模式），又如何只在 prompt 与响应解析两处分叉。其中最值得细品的，是基类里一整套对抗 LLM 不可靠输出的**防御式 JSON 解析栈**——这是把「玩具 demo」和「能在生产里跑」区分开来的关键工程智慧。

## 8.1 模板方法模式：基类定流程，子类填空

四个处理器全部继承自 `BaseModalProcessor`（`modalprocessors.py` 第 366 行）。基类的 `__init__`（第 369 行）接收三样东西：`lightrag` 实例、`modal_caption_func`(生成描述的模型函数)、可选的 `context_extractor`。它在构造时把 LightRAG 的存储句柄全部「借」过来——`text_chunks`、`chunks_vdb`、`entities_vdb`、`relationships_vdb`、`chunk_entity_relation_graph` 分别赋给 `self.text_chunks_db`、`self.chunks_vdb`、`self.entities_vdb`、`self.relationships_vdb`、`self.knowledge_graph_inst`，并复用 LightRAG 的 `embedding_func`、`llm_model_func`、`tokenizer`、`llm_response_cache`。这意味着多模态实体写进的是**同一套存储**，与文本实体平起平坐。

基类还把整个 LightRAG 对象 `asdict()` 成 `self.global_config`（第 395 行）——后面调 LightRAG 的 `extract_entities` 时要把这份全局配置原样传回去。

模板方法的「骨架」是 `process_multimodal_content` 与它内部调用的 `_create_entity_and_chunk`（第 471 行），二者由基类统一定义流程；而每个子类只需实现两个「空格」：

- `generate_description_only`——基类把它声明为 `raise NotImplementedError`（第 448 行），强制子类实现，差异在于用什么 prompt、调 vision 还是普通 LLM。
- `_parse_*_response`——解析模型返回，各子类几乎一模一样，只是实体兜底类型不同（`image`/`table`/`equation`/任意 `content_type`）。

也就是说，四个子类的代码高度雷同：`generate_description_only` 负责「拼 prompt → 调模型 → 解析」，`process_multimodal_content` 负责「拿到描述 → 拼成 chunk 文本 → 交给基类 `_create_entity_and_chunk` 落库」。这正是模板方法模式的教科书形态：把不变的流程上提到基类，把可变的步骤下沉到子类 [[Template Method Pattern]](https://refactoring.guru/design-patterns/template-method)。

## 8.2 核心落库：`_create_entity_and_chunk` 如何把多模态项变成图谱实体

要理解「多模态进图谱」的价值，必须读懂基类第 471 行的 `_create_entity_and_chunk`。它接收一段 `modal_chunk`(已拼好的描述文本)和 `entity_info`(实体名/类型/摘要)，做了如下五步：

1. **建分块**：用 `compute_mdhash_id(str(modal_chunk), prefix="chunk-")` 算出 `chunk_id`，统计 token 数，组装 `chunk_data`（含 `content`、`chunk_order_index`、`full_doc_id`、`file_path`），先 `upsert` 进 `text_chunks_db`，再 `upsert` 进 `chunks_vdb`(向量库)。这一步让多模态项的文本描述本身**可被向量检索召回**。
2. **建实体节点**：组装 `node_data`(`entity_id`、`entity_type`、`description=summary`、`source_id=chunk_id`、`file_path`、`created_at`)，调 `self.knowledge_graph_inst.upsert_node(entity_info["entity_name"], node_data)`。**这是「图片/表格/公式成为图谱节点」的关键一行**。
3. **实体进向量库**：以 `ent-` 前缀的 mdhash 为 key，把 `entity_name + summary` 拼成 `content` 写入 `entities_vdb`，让实体也能被向量召回。
4. **抽取关联**：调用 `_process_chunk_for_extraction`(第 729 行)，对这段描述文本跑一遍 LightRAG 的 `extract_entities`——从描述里再抽出更细的子实体和关系。
5. **返回三元组**：`(summary, entity_info, chunk_results)`，供上层(批处理/单文档)继续合并。

第 4 步的 `_process_chunk_for_extraction` 还有一个巧妙设计：它把从描述里抽出的每一个子实体，都与「这个多模态项的主实体」建立一条 `belongs_to` 关系（第 766–801 行）。关系描述写成 `Entity {entity_name} belongs to {modal_entity_name}`，关键词 `belongs_to,part_of,contained_in`，权重高达 `10.0`，同时 `upsert_edge` 进图谱、`upsert` 进 `relationships_vdb`。这意味着一张图里提到的每个概念，都会通过高权重边「挂」在这张图的节点上——检索到任一子概念，都能顺藤摸到原图。

最后，当 `batch_mode=False`(非批处理)时，基类调 `merge_nodes_and_edges`(第 807 行)把本次抽取的节点/边合并进全局图谱，并 `await self.lightrag._insert_done()` 确保所有存储落盘。批处理模式则把合并推迟到统一阶段，这是性能优化（详见后续批处理章节）。

## 8.3 防御式 JSON 解析：对抗 LLM 不可靠输出的降级链

LLM 被要求「以 JSON 返回实体信息」时，输出几乎从不可靠：可能被 ` ```json ` 代码块包裹、夹带推理模型的 `<think>` 段、用了中文弯引号、尾随逗号、反斜杠转义错误。如果直接 `json.loads`，生产环境里会有相当比例的多模态项解析失败而丢失。`BaseModalProcessor` 用一套**多级降级链**来「尽力抢救」，入口是第 577 行的 `_robust_json_parse`，它依次尝试四种策略，任一成功即返回：

**策略 1——直接解析候选**：调 `_extract_all_json_candidates`(第 603 行)抽出所有可能的 JSON 片段，逐个交给 `_try_parse_json`(第 648 行，本质是被 try/except 包裹的 `json.loads`，失败返回 `None`)。`_extract_all_json_candidates` 自身就很硬核：先用正则剥掉 `<think>…</think>` 与 `<thinking>…</thinking>`(防推理模型污染)；再用三种方法收集候选——(a) 正则抓 ` ``` ` 代码块里的 `{…}`；(b) **手写括号配平扫描**，逐字符计数 `{` 与 `}`，每当计数归零就截出一个完整对象（能正确处理嵌套）；(c) 一个最宽松的 `\{.*\}` 兜底正则。

**策略 2——基础清理后再解析**：对每个候选跑 `_basic_json_cleanup`(第 658 行)：把中文弯引号 `"` `"` 替换回 `"`、弯撇号替换回 `'`，再用正则 `,(\s*[}\]])` 去掉尾随逗号，然后重试解析。这能救回「内容对、只是标点被模型美化过」的输出 [[JSON 规范 RFC 8259]](https://www.rfc-editor.org/rfc/rfc8259)。

**策略 3——渐进式引号/转义修复**：对每个候选跑 `_progressive_quote_fix`(第 672 行)。它处理最棘手的转义问题：用 `(?<!\\)\\(?=")` 给引号前未转义的反斜杠补上转义；再用一个内部函数 `fix_string_content` 把字符串值里像 `\alpha` 这样的 LaTeX 反斜杠(`\(?=[a-zA-Z])`)双写成 `\\alpha`。这一步专为公式场景设计——`EquationModalProcessor` 返回的描述里经常含 LaTeX 命令，原始反斜杠会让 `json.loads` 崩溃。

**策略 4——正则字段兜底**：前三策略全败时，调 `_extract_fields_with_regex`(第 687 行)。它**彻底放弃把响应当 JSON**，转而用四条正则直接从原始文本里抠出 `detailed_description`、`entity_name`、`entity_type`、`summary` 四个字段，缺失时给默认值(`unknown_entity`/`unknown`/描述前 100 字)。它会 `logger.warning("Using regex fallback for JSON parsing")` 留痕。**这一层保证了「哪怕模型完全没按格式来，也总能产出一个结构化结果」**，而不是抛异常丢内容。

这套降级链体现的工程哲学是：把「解析失败」从「致命错误」降级为「逐层尽力」。每一层都比上一层更宽容、更不依赖输出合法性，直到最后用正则硬抠——宁可拿到残缺数据，也不丢弃整个多模态项。

此外基类第 553 行还有独立的 `_strip_thinking_tags` 静态方法：当 `_parse_*_response` 整体失败、不得不把原始响应当作 fallback 描述存进图谱时，先调它剥掉 `<think>`/`<thinking>` 块，避免把模型的内部思维链(而非真正的内容描述)写进知识图谱污染检索。文件里还保留了 `_extract_json_from_response`(第 720 行)和 `_fix_json_escapes`(第 725 行)两个 legacy 方法，注释明说已被新策略取代，仅做向后兼容——这是真实演进的痕迹。

## 8.4 `ImageModalProcessor`：base64 编码喂给视觉模型

图片处理器(第 832 行)是唯一需要把原始二进制喂给模型的。它的 `_encode_image_to_base64`(第 850 行)以 `"rb"` 打开文件，`base64.b64encode(...).decode("utf-8")` 编码成字符串，失败则记日志返回空串 [[Base64 RFC 4648]](https://www.rfc-editor.org/rfc/rfc4648)。

`generate_description_only`(第 860 行)的流程：先把 `modal_content` 解析成 dict（字符串则尝试 `json.loads`），取出 `img_path`、`image_caption`/`img_caption`、`image_footnote`/`img_footnote`、`_section_path`；校验图片路径存在(`Path(image_path).exists()`，不存在抛 `FileNotFoundError`)；用 `_get_context_for_item` 取上下文，**有上下文就用 `vision_prompt_with_context`、没有就用 `vision_prompt`**(用 `PROMPTS.get(...)` 优雅降级)；编码 base64；最关键的一步是调 `self.modal_caption_func(vision_prompt, image_data=image_base64, system_prompt=PROMPTS["IMAGE_ANALYSIS_SYSTEM"])`——注意它通过 `image_data` 命名参数把图片传给视觉模型(`vision_model_func`)；最后用 `_parse_response` 解析。

`_parse_response`(第 1036 行)调 `_robust_json_parse` 拿到 `detailed_description` 和 `entity_info`，校验三个必填字段，然后做一件统一的事：把实体名改成 `entity_name + " (entity_type)"` 形式(第 1054 行)——给实体名带上类型后缀以降低重名碰撞；若调用方预先指定了 `entity_name` 则覆盖。任何环节抛错，就走 `_strip_thinking_tags` + fallback 实体兜底。

`process_multimodal_content`(第 969 行)拿到 `enhanced_caption` 后，用 `PROMPTS["image_chunk"]` 把 `section_path`、`neighbor_text`、`image_path`、captions、footnotes、增强描述拼成一段完整的 `modal_chunk` 文本，再交给基类 `_create_entity_and_chunk` 落库。

## 8.5 `TableModalProcessor` 与 `EquationModalProcessor`：纯文本模态

表格与公式不需要视觉模型——它们的内容本身就是文本(markdown 表格 / LaTeX)。

`TableModalProcessor`(第 1076 行)的 `generate_description_only`(第 1079 行)取出 `img_path`、用 `normalize_caption_list` 规整的 `table_caption`/`table_footnote`、用 `format_table_body(get_table_body(...))` 规整的 `table_body`；按有无上下文选 `table_prompt_with_context` 或 `table_prompt`；**调 `modal_caption_func` 时不传 `image_data`，只传 prompt 和 `TABLE_ANALYSIS_SYSTEM`**——走的是普通 `llm_model_func`。`_parse_table_response`(第 1231 行)与图片版逻辑完全一致，只是 fallback 实体类型为 `table`。

`EquationModalProcessor`(第 1271 行)更简洁：`get_equation_text_and_format(content_data)` 一次拿到公式文本与格式(如 latex)，按上下文选 `equation_prompt(_with_context)`，调 `modal_caption_func` 配 `EQUATION_ANALYSIS_SYSTEM`。`_parse_equation_response`(第 1414 行)的 docstring 特意写明「with robust JSON handling」——前文 8.3 的 `_progressive_quote_fix` 双写 LaTeX 反斜杠，正是为它服务。

两者的 `process_multimodal_content` 分别用 `PROMPTS["table_chunk"]`、`PROMPTS["equation_chunk"]` 拼 chunk，落库路径与图片版共用基类，再次印证模板方法的复用。

## 8.6 `GenericModalProcessor`：保证「没有内容被丢弃」

最后是兜底的 `GenericModalProcessor`(第 1454 行)。它存在的意义是：解析器(MinerU/Docling)可能产出框架尚未专门支持的模态类型(音频转写、代码块、图表对象……)，而 RAG-Anything 的原则是**不让任何内容被静默丢弃**。

它的 `generate_description_only`(第 1457 行)不假设任何结构，直接把 `str(modal_content)` 塞进 `generic_prompt(_with_context)`，并把 `content_type` 当作模板变量传入——连 system prompt 都是 `PROMPTS["GENERIC_ANALYSIS_SYSTEM"].format(content_type=content_type)`，动态告诉模型「你正在分析一个 X 类型的内容」。`_parse_generic_response`(第 1577 行)多接一个 `content_type` 参数，fallback 时用它作实体类型与命名前缀。其余落库流程与三个专用处理器共用基类——同一套上下文注入、健壮解析、图谱 upsert。

正因如此，无论解析器吐出什么古怪类型，RAG-Anything 都有一个处理器接得住，最终都会变成知识图谱里的一个实体节点。这是「多模态进图谱」承诺的兜底保险。

## 本章小结

- 四个处理器 `ImageModalProcessor`/`TableModalProcessor`/`EquationModalProcessor`/`GenericModalProcessor` 全部继承 `BaseModalProcessor`，是模板方法模式的标准实现：基类定流程(`process_multimodal_content`、`_create_entity_and_chunk`)，子类只填 `generate_description_only` 与 `_parse_*_response` 两处空格。
- 基类构造时复用 LightRAG 的全部存储句柄与配置，使多模态实体与文本实体写进同一套存储、共享同一图谱。
- `_create_entity_and_chunk` 是「多模态进图谱」的核心：建分块 → upsert 节点 → 实体进向量库 → 抽子实体 → 合并落盘，让图片/表格/公式都成为可被双层检索召回的实体节点。
- `_process_chunk_for_extraction` 为每个子实体与主实体建立权重 `10.0` 的 `belongs_to` 高权重边，把一个多模态项内部的概念全部「挂」回主节点。
- `_robust_json_parse` 用四级降级链对抗 LLM 不可靠输出：直接解析 → 基础清理(弯引号/尾逗号) → 渐进式引号/转义修复(双写 LaTeX 反斜杠) → 正则字段兜底，宁可拿残缺数据也不丢内容。
- `_extract_all_json_candidates` 用「括号配平扫描 + 代码块正则 + 宽松正则」三法并用收集 JSON 候选，并预先剥除 `<think>` 推理标签。
- `_strip_thinking_tags` 在解析整体失败、原始响应被当 fallback 存图时剥掉推理链，避免污染知识图谱。
- `ImageModalProcessor` 经 `_encode_image_to_base64` 把图片以 `image_data` 参数喂给视觉模型；表格/公式处理器不传 `image_data`，走普通 LLM。
- 实体名统一改写为 `name (type)` 形式以降低重名碰撞；调用方可用 `entity_name` 覆盖。
- `GenericModalProcessor` 作为兜底，动态把 `content_type` 注入 prompt 与 system prompt，保证任何未识别模态都不被丢弃。

## 动手实验

1. **实验一：追踪一张图的入图全过程** — 在 `ImageModalProcessor._encode_image_to_base64`、`_parse_response`、基类 `_create_entity_and_chunk` 的 `upsert_node` 处分别加 `print`/断点，跑一张含图片的 PDF，观察从 base64 编码 → 视觉模型描述 → 实体节点写入图谱的完整数据流，并核对 `entity_name` 最终形如 `name (image)`。
2. **实验二：压测健壮 JSON 解析** — 手写若干「畸形响应」字符串(被 ` ```json ` 包裹、夹带 `<think>`、含中文弯引号、尾随逗号、含 `\alpha` 未转义的 LaTeX)，直接调用 `BaseModalProcessor._robust_json_parse`，逐个观察它命中策略 1/2/3/4 中的哪一级，验证策略 4 的正则兜底确实不会抛异常。
3. **实验三：验证 `belongs_to` 高权重边** — 处理一张包含多个具名概念的表格后，查询底层图谱中以该表格主实体为端点的边，确认每个子实体都通过 `keywords=belongs_to,part_of,contained_in`、`weight=10.0` 的边连回主节点；尝试检索某个子概念，看是否能召回原表格。
4. **实验四：触发并观察兜底处理器** — 构造一个 `content_type` 为框架未专门支持的值(如 `"audio"`)的 `modal_content`，确认调度落到 `GenericModalProcessor`，并检查它把 `content_type` 同时注入了 `generic_prompt` 与 `GENERIC_ANALYSIS_SYSTEM`，最终仍生成了一个该类型的图谱实体。

> **下一章预告**：处理器把多模态项变成了图谱实体，但这些 `modal_chunk` 文本还需要被妥善地「织」回原文档的文本流中——下一章我们进入内容工具与文本插入，看 RAG-Anything 如何把多模态描述与正文 chunk 编排成连贯的文档表示。
