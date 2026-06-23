# 第 10 章:响应合成 ResponseSynthesizers

第 9 章里,Retriever 完成了它的使命:针对一个查询,从索引中取回一批 `NodeWithScore`。但用户要的从来不是一串带分数的文本片段,而是一个连贯、可读、直接回答问题的答案。把这些零散的 node 内容塞进 prompt、交给 LLM、再产出一段成文的回答——这正是 ResponseSynthesizer 的职责。

听上去只是"拼个 prompt 再调一次 LLM",但魔鬼藏在约束里:context window 是有限的。检索器可能返回十几个、几十个 node,它们的总 token 数轻易就能超出模型一次能吃下的上限。于是"如何在窗口限制下,把所有相关内容都纳入考量,又生成一致的答案"就分裂出了一整套策略。LlamaIndex 把这套策略沉淀为 `response_synthesizers` 模块下的九种 `ResponseMode`,每一种都是 token 预算、LLM 调用次数与答案质量之间的一个权衡点。本章逐一拆解。

## BaseSynthesizer:统一的合成抽象

所有合成器都继承自 `base.py` 的 `BaseSynthesizer`(它同时混入了 `PromptMixin` 和 `DispatcherSpanMixin`)。对外的统一入口是 `synthesize`(以及异步孪生 `asynthesize`),对内则把真正的差异收敛到一个抽象方法 `get_response`/`aget_response` 上。这是经典的模板方法模式:`synthesize` 负责通用流程,子类只需实现"给定 query 和一批文本块,怎么生成回答"。

`synthesize` 的骨架很清晰(`base.py`):

```python
if len(nodes) == 0:
    # 返回 self._empty_response(默认 "Empty Response"),streaming 时返回空流
    ...
if isinstance(query, str):
    query = QueryBundle(query_str=query)
with self._callback_manager.event(CBEventType.SYNTHESIZE, ...) as event:
    response_str = self.get_response(
        query_str=query.query_str,
        text_chunks=[n.node.get_content(metadata_mode=MetadataMode.LLM) for n in nodes],
        **response_kwargs,
    )
    source_nodes = list(nodes) + list(additional_source_nodes)
    response = self._prepare_response_output(response_str, source_nodes)
```

几个值得注意的设计:第一,空 node 短路——没有检索结果时,直接返回 `self._empty_response`,绝不浪费一次 LLM 调用。第二,node 内容是用 `get_content(metadata_mode=MetadataMode.LLM)` 抽取的,意味着喂给 LLM 的不只是正文,还包含按 LLM 模式裁剪过的 metadata。第三,`_prepare_response_output` 统一封装返回类型:普通字符串包成 `Response`,`Generator` 包成 `StreamingResponse`,`AsyncGenerator` 包成 `AsyncStreamingResponse`,而 `StructuredLLM` 或带 `output_cls` 的场景包成 `PydanticResponse`——streaming 与结构化输出的分支都在这里收口,子类无需重复处理。第四,可观测性贯穿始终:`synthesize` 用 `@dispatcher.span` 装饰,过程中发出 `SynthesizeStartEvent`/`SynthesizeEndEvent`,并通过 `callback_manager` 埋下 `CBEventType.SYNTHESIZE` 事件——这与第 14 章要讲的 instrumentation 体系一脉相承。

`BaseSynthesizer.__init__` 还在构造期解析两个关键依赖:`prompt_helper`(普通 LLM)与 `chat_prompt_helper`(多模态 chat LLM),都来自参数、`Settings` 或 `PromptHelper.from_llm_metadata(llm.metadata)` 三级回退。`PromptHelper` 正是后面所有 repack/truncate 逻辑的底座——它知道模型的 context window,才能决定一次能塞多少 token。

## ResponseMode 全景与 factory 映射

`type.py` 把所有策略枚举为 `ResponseMode(str, Enum)`,共九个值:`REFINE`、`COMPACT`、`SIMPLE_SUMMARIZE`、`TREE_SUMMARIZE`、`GENERATION`、`NO_TEXT`、`CONTEXT_ONLY`、`ACCUMULATE`、`COMPACT_ACCUMULATE`。每个枚举值的 docstring 就是一句精炼的策略说明。

`factory.py` 的 `get_response_synthesizer` 负责把枚举映射到具体类。它是大多数用户接触合成器的入口,签名里有一行关键默认值:

```python
response_mode: ResponseMode = ResponseMode.COMPACT
```

也就是说,**不指定时默认用 `CompactAndRefine`**。函数体里一长串 `if/elif` 完成映射:`REFINE → Refine`、`COMPACT → CompactAndRefine`、`TREE_SUMMARIZE → TreeSummarize`、`SIMPLE_SUMMARIZE → SimpleSummarize`、`GENERATION → Generation`、`ACCUMULATE → Accumulate`、`COMPACT_ACCUMULATE → CompactAndAccumulate`、`NO_TEXT → NoText`、`CONTEXT_ONLY → ContextOnly`,未知模式抛 `ValueError`。

factory 还统一兜底了 prompt 模板:`text_qa_template` 默认 `DEFAULT_TEXT_QA_PROMPT_SEL`,`refine_template` 默认 `DEFAULT_REFINE_PROMPT_SEL`,`summary_template` 默认 `DEFAULT_TREE_SUMMARIZE_PROMPT_SEL`,`simple_template` 默认 `DEFAULT_SIMPLE_INPUT_PROMPT`,以及对应的 chat 版本 `CHAT_CONTENT_QA_PROMPT` 等。这些 `_SEL` 后缀的是 prompt selector——能按 LLM 是否为 chat 模型自动选不同模板。

## Refine:逐块迭代,本章重头戏

`refine.py` 的 `Refine` 是整个模块的核心,理解了它,其余模式几乎都是它的变体或简化。它的思想直白:既然一次塞不下所有 node,那就一块一块来。第一块用 `text_qa_template` 生成"初答",之后每一块都用 `refine_template`,把"已有答案 + 新一块 context"交给 LLM,让它在需要时修正/补全答案。如 `type.py` 所述,N 个 node 就 refine N-1 次。

为什么必须分块?**就是 context window 限制**——单次 prompt 容纳不下全部内容,只能用迭代把信息分批"喂"进答案。来看两份默认模板(`prompts/default_prompts.py`)。`DEFAULT_TEXT_QA_PROMPT_TMPL` 是标准 RAG 问答:

```
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer:
```

而 `DEFAULT_REFINE_PROMPT_TMPL` 多了一个"已有答案"的注入点:

```
The original query is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
{context_msg}
------------
Given the new context, refine the original answer to better answer the query.
If the context isn't useful, return the original answer.
Refined Answer:
```

迭代主体是 `_run_refine_loop`(及异步 `_arun_refine_loop`)。它先用 `get_biggest_prompt([qa_template, refine_template])` 取两个模板里更"占地方"的那个作为容量基准,再用 `prompt_helper.repack(...)` 把每个 chunk 拆/合到能放进窗口的大小,带 `padding=self._response_padding_size`(默认常量 `DEFAULT_RESPONSE_PADDING_SIZE = 500`,给答案留出余量),装入一个 `deque`。然后循环出队:

```python
if response is None:
    prompt_template = qa_template.partial_format(query_str=query_str)
    prompt_kwargs = make_qa_prompt_kwargs(chunk)
else:
    prompt_template = refine_template.partial_format(
        query_str=query_str, existing_answer=response
    )
    repacked = prompt_helper.repack(prompt_template, [chunk], llm=self._llm)
    if len(repacked) > 1:
        chunks_deque.extendleft(repacked)
        continue
    chunk = repacked[0]
    prompt_kwargs = make_refine_prompt_kwargs(chunk)
program = self._program_factory(prompt_template)
if resp := self._update_response(program, prompt_kwargs, response_kwargs):
    response = resp
```

注意 refine 分支里有一次"二次 repack":因为 `existing_answer` 会随迭代不断变长,挤占可用空间,所以每轮都要拿"已经填进答案的 refine 模板"重新计算 chunk 能否放下;若一块都放不下(`len(repacked) > 1`),就把它拆开塞回队首继续。这是个容易忽略却很关键的细节——它保证了答案膨胀时迭代依然不会溢出窗口。

真正调 LLM 的是 `program`,由 `_program_factory` 产出。默认是 `DefaultRefineProgram`,它直接 `self._llm.predict(...)` 并恒定返回 `query_satisfied=True`——也就是不做过滤。但 `Refine` 支持一个 `structured_answer_filtering` 开关:开启后改用结构化程序,LLM 被要求返回 `StructuredRefineResponse`,其中:

```python
class StructuredRefineResponse(BaseModel):
    query_satisfied: bool = Field(...)
    answer: str = Field(...)
```

`query_satisfied` 字段刻意放在 `answer` 前面——源码注释解释:LLM 几乎总是按字段声明顺序流式输出 json,把这个布尔放在最前,流式场景下前十来个 token 就能判断本块是否相关,不相关就提前放弃、跳到下一块,省下等待整段答案的开销。`_update_response`/`_aupdate_response` 据此决定是否用新答案覆盖 `response`,且对 `ValidationError/ValueError/TypeError` 做了容错降级。

此外 `Refine` 也是多模态合成的承载者:它另有 `get_response_from_messages` 系列方法,改用 `chat_content_qa_template`/`chat_content_refine_template`(默认 `CHAT_CONTENT_QA_PROMPT`/`CHAT_CONTENT_REFINE_PROMPT`),把 chunk 包成 `ChatMessage` 走 chat 路径。

## CompactAndRefine:为什么它是默认模式

`compact_and_refine.py` 的 `CompactAndRefine` 直接继承 `Refine`,只做一件事:在进入 refine 循环之前,先把多个小 chunk **尽量塞满**成更少、更大的 chunk。核心是 `_make_compact_text_chunks`:

```python
text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
refine_template = self._refine_template.partial_format(query_str=query_str)
max_prompt = get_biggest_prompt([text_qa_template, refine_template])
return self._prompt_helper.repack(
    max_prompt, text_chunks, llm=self._llm, padding=self._response_padding_size
)
```

`get_response` 先调它压缩,再 `super().get_response(...)` 复用父类的 refine 逻辑。效果是:原本十几个 node 要 refine 十几次,压缩后可能只剩两三个"满载" chunk,LLM 调用次数随之骤减。正如 `type.py` 中 `COMPACT` 的 docstring 所言——"This mode is faster than refine since we make fewer calls to the LLM"。

它之所以被选为 factory 默认,正是因为它在"覆盖全部检索内容"(不丢信息、仍走 refine 迭代)与"控制成本/延迟"(尽量少调 LLM)之间取得了普适的平衡。纯 `REFINE` 一块一次太慢,`SIMPLE_SUMMARIZE` 又有溢出风险,`COMPACT` 则是绝大多数 RAG 问答的稳妥默认。

## TreeSummarize:自底向上递归摘要

`tree_summarize.py` 的 `TreeSummarize` 面向另一类需求——"总结全部内容"而非"回答某个具体小问题"。它的递归算法在类 docstring 里写得很明确:1) repack 文本块使每块填满 context window;2) 若只剩一块,直接给出最终回答;3) 否则,对每块各做一次摘要,再对这批摘要递归调用自己。

`get_response` 的实现忠实于此:用 `summary_template`(默认 `DEFAULT_TREE_SUMMARIZE_PROMPT_SEL`)`partial_format` 后 repack,`len(text_chunks) == 1` 时收敛产出结果,否则对每块 `self._llm.predict(summary_template, context_str=text_chunk)` 得到一组 summaries,再 `return self.get_response(query_str=query_str, text_chunks=summaries, ...)`。一层层向上合并,直到塌缩成根节点的最终答案——这就是"自底向上建树"。

它和 refine 的关键差异:refine 是**线性、串行**地"已有答案 + 新块"逐步修正;tree summarize 是**树形、可并行**地摘要再合并。同层各块互不依赖,所以 `TreeSummarize` 提供了 `use_async` 开关——`get_response` 里可用 `run_async_tasks`、`aget_response` 用 `asyncio.gather` 并发跑同层摘要,对"长文档全局总结"这类查询能显著提速。它也支持 `output_cls` 结构化输出与 streaming(仅在收敛到单块时对最终回答 `stream`)。

## 其余模式:token 预算到答案质量的权衡谱系

把九种模式排在一条谱系上,看 LLM 调用次数与处理强度的取舍:

- `SIMPLE_SUMMARIZE`(`simple_summarize.py`,`SimpleSummarize`):把所有 chunk 用 `"\n".join` 拼成一段,经 `prompt_helper.truncate` 截断后**只调一次 LLM**。最省钱,但截断意味着超窗内容会被直接丢弃——`type.py` 明说 merged 文本超窗会失败/丢失。
- `GENERATION`(`generation.py`,`Generation`):完全忽略 context,`get_response` 里 `del text_chunks`,只用 `DEFAULT_SIMPLE_INPUT_PROMPT`(模板就是裸 `{query_str}`)让 LLM 自由生成。它甚至重写了 `synthesize`/`asynthesize`,**绕过**了基类的空 node 短路——哪怕没有任何检索结果也照样调 LLM。本质上是"非 RAG"的对照基线。
- `ACCUMULATE`(`accumulate.py`,`Accumulate`):对**每个** chunk 各自独立回答一次,再用分隔符 `"\n---------------------\n"` 把结果拼成 `Response 1: ... / Response 2: ...`。它不合并、不 refine,因此调用次数与 chunk 数成正比;明确不支持 streaming(streaming 时抛 `ValueError`),但支持 `use_async` 并发。适合"对每篇来源分别抽取/打分"这类需要逐源结果的场景。
- `COMPACT_ACCUMULATE`(`compact_and_accumulate.py`,`CompactAndAccumulate`):继承 `Accumulate`,先 `repack` 压缩 chunk 再 accumulate,理由同 compact——少调几次 LLM。
- `CONTEXT_ONLY`(`context_only.py`,`ContextOnly`):**完全不调 LLM**,`get_response` 直接 `"\n\n".join(text_chunks)` 返回拼接后的原文。用于调试、人审或把 context 交给下游处理。
- `NO_TEXT`(`no_text.py`,`NoText`):更极端,`get_response` 返回空串 `""`,连构造时都不需要 `llm`(看 factory:`NoText(callback_manager=..., streaming=..., multimodal=...)` 没传 llm)。它的价值在于 `synthesize` 仍会把 `source_nodes` 挂进 `Response`——也就是"只要检索结果、不要生成答案"。

从 `NO_TEXT`/`CONTEXT_ONLY`(零 LLM 调用)、`SIMPLE_SUMMARIZE`(一次调用、可能丢内容)、`COMPACT`/`TREE_SUMMARIZE`(少量调用、覆盖全部)、到 `REFINE`/`ACCUMULATE`(每块一次、最细但最贵),正好铺成一条从"便宜但粗糙"到"昂贵但充分"的连续谱系。没有银弹,选哪个取决于你的 token 预算、延迟容忍度与对答案质量的要求。

## prompt 模板的可定制性

合成器的"性格"几乎全写在 prompt 模板里,而这些模板都是可替换的。`get_response_synthesizer` 暴露了 `text_qa_template`、`refine_template`、`summary_template`、`simple_template` 及其 chat 版本等参数;运行期还能通过 `PromptMixin` 体系热更新——`Refine._update_prompts` 就支持替换 `text_qa_template`/`refine_template`/`chat_content_qa_template`/`chat_content_refine_template`,`TreeSummarize._update_prompts` 支持替换 `summary_template`/`chat_summary_template`。这意味着你可以在不改一行框架代码的前提下,把默认的英文问答模板换成中文模板、加入引用要求或调整"无关 context 时返回原答案"的措辞,从而精确控制合成行为。

## 本章小结

- ResponseSynthesizer 承接 Retriever 输出的 `NodeWithScore`,职责是在 context window 限制下把 node 内容合成为连贯答案。
- `BaseSynthesizer.synthesize` 是统一模板方法,空 node 短路返回 `"Empty Response"`,用 `get_content(MetadataMode.LLM)` 抽内容,经 `_prepare_response_output` 统一封装 `Response`/`StreamingResponse`/`PydanticResponse`,并埋 `CBEventType.SYNTHESIZE` 与 dispatcher 事件。
- 子类只需实现抽象的 `get_response`/`aget_response`,差异收敛于此。
- `ResponseMode` 定义九种策略,`get_response_synthesizer` 把枚举映射到类,默认 `response_mode=ResponseMode.COMPACT`(即 `CompactAndRefine`)。
- `Refine` 是核心:首块用 `text_qa_template` 生成初答,后续块用 `refine_template` 注入 `existing_answer` 迭代修正;分块的根本原因是 context window 装不下全部内容。
- `Refine` 用 `deque` + `repack` 管理分块,`DEFAULT_RESPONSE_PADDING_SIZE = 500` 留答案余量,refine 时二次 repack 防止膨胀的答案溢出;`structured_answer_filtering` 借 `StructuredRefineResponse.query_satisfied`(刻意置于首字段)实现流式早停过滤。
- `CompactAndRefine` 先 repack 塞满 chunk 再 refine,以更少 LLM 调用覆盖全部内容,故被选为默认模式。
- `TreeSummarize` 自底向上递归摘要建树,同层可并行(`use_async`),适合"总结全部"类查询。
- `SimpleSummarize`(一次调用、截断丢内容)、`Generation`(忽略 context、绕过空 node 短路)、`Accumulate`/`CompactAndAccumulate`(逐块独立回答后拼接、不支持 streaming)、`ContextOnly`(零 LLM、拼原文)、`NoText`(返回空串、只保留 source_nodes)构成完整谱系。
- 九种模式本质是 token 预算、LLM 调用次数与答案质量之间的权衡;prompt 模板通过 factory 参数与 `_update_prompts` 完全可定制。

## 动手实验

1. **实验一:验证默认模式与 factory 映射** — 阅读 `/tmp/llama_explore/llama-index-core/llama_index/core/response_synthesizers/factory.py`,确认 `get_response_synthesizer` 的 `response_mode` 默认值为 `ResponseMode.COMPACT`,并逐条核对 9 个 `if/elif` 分支返回的类;再对照 `type.py` 检查九个枚举值是否一一对应。
2. **实验二:画出 Refine 迭代时序** — 在 `refine.py` 的 `_run_refine_loop` 里跟踪 `response is None` 与 `else` 两个分支,标注首块走 `text_qa_template`、后续块走 `refine_template.partial_format(existing_answer=response)`;特别留意 refine 分支里 `repack` 二次打包与 `len(repacked) > 1` 时 `extendleft` 回填队首的逻辑,说明它如何防止答案膨胀导致溢出。
3. **实验三:对比 compact 与 tree 的容量处理** — 并排阅读 `compact_and_refine.py` 的 `_make_compact_text_chunks` 与 `tree_summarize.py` 的 `get_response`,都基于 `prompt_helper.repack`,但 `CompactAndRefine` 压缩后走线性 refine,`TreeSummarize` 则在 `len(text_chunks) > 1` 时对每块摘要后递归 `self.get_response`。用三五个 node 在纸上推演各自的 LLM 调用次数。
4. **实验四:辨析"轻量/无 LLM"模式的边界** — 读 `generation.py`、`accumulate.py`、`context_only.py`、`no_text.py`,确认:`Generation.synthesize` 重写后绕过基类空 node 短路且 `del text_chunks`;`Accumulate.get_response` 在 `self._streaming` 时抛 `ValueError`;`ContextOnly` 返回 `"\n\n".join(text_chunks)`;`NoText` 返回 `""`。总结四者各自"调几次 LLM、丢不丢内容、能否 streaming"。

> **下一章预告**:第 11 章将进入查询引擎与对话引擎(QueryEngine/ChatEngine)——把本章的 ResponseSynthesizer 与第 9 章的 Retriever 组装成端到端的问答管线,并在此之上加入对话记忆,支撑带上下文的多轮交互。
