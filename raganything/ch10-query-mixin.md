# 第 10 章 查询系统 QueryMixin

前面九章我们把文档「灌」进了系统：解析、切分、上下文提取、多模态处理、文本插入，最终沉淀成 LightRAG 的双层知识图谱。本章走到链路的另一端——把知识「取」出来。`RAGAnything` 由 `QueryMixin`、`ProcessorMixin`、`BatchMixin` 三个 mixin 组合而成，`QueryMixin` 正是所有检索查询的入口。它的代码不长（`query.py` 共 868 行），但承载了一个关键设计取舍：**文本检索不重造轮子，直接委托 LightRAG；多模态查询则在 LightRAG 之上做增量增强**。读完本章，你会明白 RAG-Anything 相比纯文本 RAG 多出来的那一层价值，到底加在了哪里。

`QueryMixin` 暴露两类共四个公开入口：纯文本的 `aquery` / `query`，以及带多模态输入的 `aquery_with_multimodal` / `query_with_multimodal`。其中 `query` 与 `query_with_multimodal` 是同步包装，内部用 `always_get_an_event_loop()` 拿到事件循环再 `run_until_complete` 调用异步版本，逻辑全在异步方法里。

## 纯文本查询：站在 LightRAG 的肩上

先看最朴素的路径。`aquery(self, query, mode="mix", system_prompt=None, **kwargs)` 的默认 `mode` 是 `"mix"`，docstring 列出的合法取值为 `"local"`、`"global"`、`"hybrid"`、`"naive"`、`"mix"`、`"bypass"`——这套模式枚举正是第六部讲过的 LightRAG 双层知识图谱检索模式，RAG-Anything 原样透传。

核心委托只有两行：

```python
query_param = QueryParam(mode=mode, **kwargs)
result = await self.lightrag.aquery(
    query, param=query_param, system_prompt=system_prompt
)
```

这里有两点值得品味。第一，`QueryParam` 直接 `import` 自 `lightrag`（`from lightrag import QueryParam`），RAG-Anything 没有定义自己的查询参数类。第二，`**kwargs` 被原封不动塞进 `QueryParam`——这意味着 `top_k`、`response_type`、`only_need_context` 等 LightRAG 支持的所有参数，调用方都能透传进去，无需 RAG-Anything 逐个声明。这是「薄封装」的典型写法：上游能力变了，下游自动跟着变，不必同步维护一份参数白名单。

为什么不重造检索？因为 RAG-Anything 的差异化不在「文本怎么检索」，而在「多模态内容怎么进图、怎么被看见」。文本检索是 LightRAG 已经解决好的成熟问题，复用它既省工又能蹭到上游迭代。代价是 RAG-Anything 与 LightRAG 强耦合——`aquery` 开头就检查 `if self.lightrag is None` 并抛出明确错误，提示用户「先处理文档或提供预初始化的 LightRAG 实例」。

`aquery` 还埋了一套回调钩子：通过 `getattr(self, "callback_manager", None)` 取回调管理器，在查询前 `dispatch("on_query_start")`、出错时 `dispatch("on_query_error")`、完成时 `dispatch("on_query_complete")` 并附带 `duration_seconds` 和 `result_length`。这套可观测性机制是后续章节的主题，这里只需知道查询入口已为它预留了埋点。

## VLM 增强：让回答「看见」图

`aquery` 真正的分叉点在最前面。它会先 `kwargs.pop("vlm_enhanced", None)`，若调用方没显式指定，则按可用性自动判定：

```python
if vlm_enhanced is None:
    vlm_enhanced = (
        hasattr(self, "vision_model_func")
        and self.vision_model_func is not None
    )
```

也就是说，只要初始化时提供了 `vision_model_func`，纯文本 `aquery` 默认就会走 VLM 增强路径，转交给 `aquery_vlm_enhanced`。如果用户要求增强但没有视觉模型，则 `logger.warning` 后回退到普通查询。这个「能视觉就视觉」的默认值，是 RAG-Anything 相比纯文本 RAG 的核心增量。

`aquery_vlm_enhanced(self, query, mode="mix", system_prompt=None, extra_safe_dirs=None, **kwargs)` 的流程在源码注释里清晰分为四步，我们逐一拆解。

**第一步：只取检索 prompt，不生成答案。** 它构造 `QueryParam(mode=mode, only_need_prompt=True, **kwargs)`，调 `self.lightrag.aquery` 拿到 `raw_prompt`。`only_need_prompt=True` 让 LightRAG 完成双层检索、拼好上下文，但**不**调用 LLM 生成最终回答——RAG-Anything 要拦截这个中间产物，自己接管生成环节。这是「站在肩上」之外又「插进流程中间」的巧妙一手。

**第二步：提取并处理图片路径。** `_process_image_paths_for_vlm(raw_prompt, extra_safe_dirs=...)` 用正则在 prompt 里扫描图片路径：

```python
image_path_pattern = (
    r"Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))"
)
```

之所以能匹配到 `Image Path:`，是因为前面章节插入多模态实体时，图片实体的描述里就带了这个标记。对每个命中的路径，`replace_image_path` 做三件事：用 `validate_image_file` 校验文件合法性；做**安全检查**——只允许来自 `Path.cwd()`、`self.config.working_dir`、`self.config.parser_output_dir` 或调用方显式传入的 `extra_safe_dirs` 的图片，用 `Path.is_relative_to` 判定，越界的路径直接拦截并告警。这一步是为了防御「间接 prompt 注入」：检索回来的上下文可能被污染，诱导系统读取任意系统文件，安全目录白名单把这个攻击面关上了。

校验通过后，`encode_image_to_base64(image_path)` 把图片编码为 base64，追加进实例变量 `self._current_images_base64`，并把原文里的路径替换成 `Image Path: {image_path}\n[VLM_IMAGE_{n}]` 这样的占位标记。注意进入这一步前会先 `delattr(self, "_current_images_base64")` 清掉上次查询的缓存，避免图片串台。方法返回 `(enhanced_prompt, images_processed)`——若 `images_found` 为 0，说明检索结果里没图，直接回退普通 LightRAG 查询。

**第三步：构造带图的 VLM 消息。** `_build_vlm_messages_with_images(enhanced_prompt, query, system_prompt)` 负责把文本和图片交织成 vision model 能吃的消息格式。它的精巧之处在于**用占位标记保持图文位置对应**：

```python
text_parts = enhanced_prompt.split("[VLM_IMAGE_")
for i, text_part in enumerate(text_parts):
    if i == 0:
        if text_part.strip():
            content_parts.append({"type": "text", "text": text_part})
    else:
        marker_match = re.match(r"(\d+)\](.*)", text_part, re.DOTALL)
        if marker_match:
            image_num = int(marker_match.group(1)) - 1
            ...
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{images_base64[image_num]}"},
            })
```

按 `[VLM_IMAGE_` 切分文本，逐段还原：标记前的文字作为 `text` 块，标记处插入对应序号的 base64 图片块（`image_num` 减 1 转 0 基索引），标记后的剩余文字再作为下一个 `text` 块。这样图片就被插回它在原文里出现的精确位置，而不是简单堆在末尾——VLM 因此能理解「这张图对应这段描述」。最后追加用户问题文本块，并拼装 `system` + `user` 两条消息。系统提示在固定的 `base_system_prompt`（"You are a helpful assistant that can analyze both text and image content..."）基础上拼接调用方传入的 `system_prompt`。若 `images_base64` 为空则退化为纯文本单消息。

**第四步：调用视觉模型。** `_call_vlm_with_multimodal_content(messages)` 取出 `messages[1]` 的 `content`：若是字符串（纯文本模式），调 `self.vision_model_func(content, system_prompt=...)`；若是列表（多模态模式），则调 `self.vision_model_func("", messages=messages)`——传空 prompt，因为完整对话已在 `messages` 里。至此，一次「检索 → 取回图片路径 → base64 编码 → 构造 VLM messages → 调 vision_model_func」的增强查询闭环完成。回答得以「看见」检索到的图，这正是纯文本 RAG 给不了的。

## 多模态查询：查询里也能带图

上一节说的是「问题是文本、检索结果带图」。还有另一种场景：**用户的问题本身就附带图片或表格**——比如「分析这张图里的内容」。这交给 `aquery_with_multimodal(self, query, multimodal_content=None, mode="mix", system_prompt=None, **kwargs)`。

`multimodal_content` 是一个 dict 列表，每项含 `type` 字段（`"image"`、`"table"`、`"equation"` 等）及类型相关字段（如 `img_path`、`table_data`、`latex`）。流程是：

1. `_ensure_lightrag_initialized()` 确保 LightRAG 就绪，失败则抛 `RuntimeError`。
2. 若 `multimodal_content` 为空，直接回退 `aquery` 纯文本路径。
3. 算缓存键，命中则直接返回（见下一节）。
4. `_process_multimodal_query_content(query, multimodal_content)` 把输入的多模态内容**先理解成文字描述**，拼成「增强查询文本」。
5. 用增强后的文本调 `aquery` 走正常检索。
6. 把结果写回缓存并持久化。

第四步是关键。`_process_multimodal_query_content` 遍历每项内容，用 `get_processor_for_type(self.modal_processors, content_type)` 取到对应的模态处理器（即第八章的处理器四件套），再按类型分发到 `_describe_image_for_query` / `_describe_table_for_query` / `_describe_equation_for_query` / `_describe_generic_for_query`。以图片为例，若 `img_path` 存在，就用 `processor._encode_image_to_base64` 编码后调 `processor.modal_caption_func`，配合 `PROMPTS["QUERY_IMAGE_DESCRIPTION"]` 和 `PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"]` 生成描述；图片不存在则退而拼 caption、footnote 等元信息。表格、公式则分别用 `QUERY_TABLE_ANALYSIS`、`QUERY_EQUATION_ANALYSIS` 模板。所有描述拼进 `enhanced_parts`，最后追加 `PROMPTS["QUERY_ENHANCEMENT_SUFFIX"]`（"Please provide a comprehensive answer based on the user query and the provided multimodal content information."）。

这里的设计哲学是**「把多模态输入翻译成文本，再交给文本检索」**。用户带来的图被理解成一段描述，描述与原始问题合并成更丰富的查询，从而能在知识图谱里检索到相关实体。它和上一节的 VLM 增强互补：一个让「检索结果」可视，一个让「查询输入」可读。

## 多模态查询缓存：别让 VLM 白跑

多模态查询很贵——理解一张输入图要调一次 caption 模型，跑一次检索又要调 LLM。如果同一个带图查询被问两遍，重复开销不可接受。`_generate_multimodal_cache_key` 就是为此而生，它生成一个稳定的缓存键，让相同查询直接命中缓存。

稳定性是它的核心诉求。难点在于 `multimodal_content` 里有文件路径和大块数据，直接序列化既不稳定也不便携。方法做了两类归一化：

```python
if key in ["img_path", "image_path", "file_path"] and isinstance(value, str):
    normalized_item[key] = Path(value).name        # 只取文件名，跨机器可移植
elif key in ["table_data", "table_body"] and isinstance(value, str) and len(value) > 200:
    normalized_item[f"{key}_hash"] = hashlib.md5(value.encode()).hexdigest()  # 大内容存哈希
```

对图片路径只取 `Path(value).name`（basename），让缓存在不同机器、不同绝对路径下仍可命中；对超过 200 字符的表格数据，用 `md5` 存内容哈希而非原文，既缩短键又规避大字段。此外只挑选 `stream`、`response_type`、`top_k`、`max_tokens`、`temperature`、`system_prompt` 等会影响结果的 kwargs 纳入缓存数据，无关参数不参与，避免缓存碎片化。最终 `json.dumps(cache_data, sort_keys=True, ensure_ascii=False)` 后取 `md5`，返回 `f"multimodal_query:{cache_hash}"`，带前缀便于在缓存里区分类型。

缓存的读写复用了 LightRAG 的 `llm_response_cache`：`aquery_with_multimodal` 先 `get_by_id(cache_key)`，命中且有 `return` 字段就直接返回；未命中则查询后 `upsert({cache_key: cache_entry})`，最后 `index_done_callback()` 落盘。整个过程外裹 `if ... enable_llm_cache` 开关判断和 `try/except`——缓存失败只 `debug` 记录、绝不影响主流程。复用 LightRAG 的缓存基础设施，又一次体现了「不重造轮子」。

## 本章小结

- `QueryMixin`（`query.py`，868 行）是 `RAGAnything` 的检索入口，提供 `aquery`/`query`（纯文本）与 `aquery_with_multimodal`/`query_with_multimodal`（带多模态输入）四个公开方法；同步版本是异步版的 `run_until_complete` 包装。
- 默认查询模式是 `mode="mix"`，合法取值 `local`/`global`/`hybrid`/`naive`/`mix`/`bypass` 直接复用 LightRAG 的双层知识图谱检索枚举。
- 纯文本查询「站在 LightRAG 肩上」：`QueryParam` 直接 `import` 自 `lightrag`，`**kwargs`（含 `top_k`）原样透传，RAG-Anything 不重造检索，代价是与 LightRAG 强耦合。
- VLM 增强查询是核心增量：只要有 `vision_model_func`，纯文本 `aquery` 默认转 `aquery_vlm_enhanced`，让回答能「看见」检索到的图。
- 增强四步法：`only_need_prompt=True` 拦截 LightRAG 检索 prompt → `_process_image_paths_for_vlm` 正则提取并 base64 编码图片 → `_build_vlm_messages_with_images` 用 `[VLM_IMAGE_n]` 占位标记保持图文位置对应 → `_call_vlm_with_multimodal_content` 调视觉模型。
- 安全防护：图片路径必须落在 `cwd`/`working_dir`/`parser_output_dir`/`extra_safe_dirs` 内，用 `is_relative_to` 校验，防御间接 prompt 注入读取任意文件。
- 带图输入的查询走「翻译成文本再检索」路线：`_process_multimodal_query_content` 借助模态处理器把图/表/公式描述成文字，与原问题合并成增强查询。
- `_generate_multimodal_cache_key` 通过 basename 归一化路径、对大表格存 md5 哈希、只纳入相关 kwargs，生成稳定可移植的缓存键，复用 LightRAG 的 `llm_response_cache` 避免重复 VLM 调用。
- 查询入口预埋了 `callback_manager` 的 `on_query_start`/`on_query_error`/`on_query_complete` 回调，为可观测性留出钩子。

## 动手实验

1. **实验一：验证默认模式与透传** — 阅读 `aquery` 的签名与 `QueryParam(mode=mode, **kwargs)` 一行，确认默认 `mode="mix"`。然后用 `query("...", mode="local", top_k=3)` 调用并打印日志中的 `Query mode`，验证 `top_k` 是否被透传进 `QueryParam`（可在 LightRAG 侧打断点观察）。
2. **实验二：触发并观察 VLM 增强分叉** — 分别在初始化时提供和不提供 `vision_model_func`，对同一文本查询调用 `aquery`，观察日志：前者应进入 `Executing VLM enhanced query`，后者走普通查询。再显式传 `vlm_enhanced=False` 强制关闭增强，确认分支判定逻辑。
3. **实验三：测试安全目录拦截** — 构造一个含 `Image Path: /etc/passwd.png` 的伪 prompt，直接调用 `_process_image_paths_for_vlm`，确认它命中「Blocking image path outside safe directories」告警且 `images_processed` 为 0；再把图片放进 `parser_output_dir` 重试，确认能被正常 base64 编码。
4. **实验四:验证缓存键稳定性** — 用同一张图但写成两个不同绝对路径（如 `/a/x.jpg` 与 `/b/x.jpg`），分别调 `_generate_multimodal_cache_key`，确认因 basename 归一化二者键相同；再把一段超过 200 字符的 `table_data` 改动一个字符，确认 md5 哈希变化导致键变化。

> **下一章预告**：单条查询讲完了，但真实场景往往要一次性灌入成百上千份文档。第 11 章我们转向 `BatchMixin` 与 `BatchParser`，看 RAG-Anything 如何用并发、进度跟踪与错误隔离把批处理做稳、做快。
