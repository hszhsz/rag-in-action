# 第 9 章：RAG 对话链路

前面八章我们已经把知识库的"前半场"拆解完毕：文档怎么加载、怎么切分、怎么向量化、怎么落进 FAISS 或 Milvus、检索器又如何把最相关的若干段文本捞出来。但所有这些铺垫，最终都要汇聚到一个用户真正关心的动作上——**提问，然后得到一段流式生成的中文回答，并附带可追溯的出处**。本章就站在 `chatchat/server/chat/` 这个目录里，把"一次知识库问答"从入参到 SSE 输出的完整数据流走一遍。

Langchain-Chatchat 把对话拆成了三条相互独立又结构高度同构的链路：`kb_chat.py` 负责知识库问答（本章主角），`file_chat.py` 负责临时上传文件的即时问答，`chat.py` 负责带工具调用的 Agent 对话。它们共用同一套 `History`、`get_prompt_template`、`wrap_done` 和 `AsyncIteratorCallbackHandler` 基础设施。读懂 `kb_chat`，另外两条链路就只是参数和 Chain 结构上的微调。

## 一次知识库问答的入参契约

`kb_chat` 是一个 `async def`，所有参数都用 FastAPI 的 `Body(...)` 声明，这意味着它直接就是一个 POST 接口的请求体定义。值得注意的几个默认值都来自全局 `Settings`，体现了"配置即契约"的设计：`top_k` 默认 `Settings.kb_settings.VECTOR_SEARCH_TOP_K`，`score_threshold` 默认 `Settings.kb_settings.SCORE_THRESHOLD` 并被约束在 `ge=0, le=2` 之间，`model` 默认调用 `get_default_llm()` 取当前可用的首选 LLM，`temperature` 默认 `Settings.model_settings.TEMPERATURE`。

最关键的是 `mode` 参数，它是一个 `Literal["local_kb", "temp_kb", "search_engine"]`，决定了"知识从哪来"。这个三选一的设计让同一个接口同时服务本地知识库、临时知识库和联网搜索三种召回源，而后续的"拼 context → 填 prompt → LLM 生成"流程完全复用。另外两个开关型参数也很有意思：`stream`（默认 `True`）控制是否走 SSE 流式，`return_direct`（默认 `False`）控制是否**跳过 LLM、直接把检索结果返回**——这是一个把 Chatchat 当作纯检索服务用的逃生通道。

接口体的最外层先做了一次"前置校验"：当 `mode == "local_kb"` 时，立刻用 `KBServiceFactory.get_service_by_name(kb_name)` 取知识库服务，若为 `None` 直接返回 `BaseResponse(code=404, msg=f"未找到知识库 {kb_name}")`。这步在进入异步生成器之前就拦掉了无效知识库，避免把错误塞进 SSE 流里。

## 三种召回源的统一抽象

真正的业务逻辑被包在内部协程 `knowledge_base_chat_iterator()` 里，它是一个 `AsyncIterable[str]` 异步生成器。函数开头用 `nonlocal history, prompt_name, max_tokens` 声明了对外层变量的可写引用——这三个值在后续逻辑里会被就地改写。

第一件事是把历史消息标准化：`history = [History.from_data(h) for h in history]`。`History.from_data` 是一个分类工厂方法，它能同时吃下 `list`、`tuple` 和 `dict` 三种形态——`["user", "你好"]` 会被解析成 `role/content`，`{"role": ..., "content": ...}` 也照单全收。这种宽容的入参解析让前端无论传数组还是对象都能工作。

接下来按 `mode` 分三路召回，但三路的产出都被对齐成两个变量：`docs`（结构化文档列表，元素是 dict，含 `page_content` 与 `metadata`）和 `source_documents`（给前端展示的出处字符串列表）。这种"两路对齐"的归一化，是后续流程能与召回源彻底解耦的前提——无论知识来自本地库、临时库还是搜索引擎，下游只认 `docs` 和 `source_documents` 这两个变量。

`local_kb` 路是主路径。它先调 `kb.check_embed_model()` 确认 embedding 模型可用，不可用直接 `raise ValueError(msg)` 让外层异常处理接管；再 `await run_in_threadpool(search_docs, query=..., knowledge_base_name=kb_name, top_k=..., score_threshold=..., file_name="", metadata={})`。这里用 `run_in_threadpool` 把同步的向量检索丢进线程池，避免阻塞事件循环——这是把同步 RAG 组件嵌入异步接口的标准做法。顺着 `search_docs` 往下看，它最终落到 `KBService.search_docs`，内部仍先 `check_embed_model()` 把关，再 `do_search(query, top_k, score_threshold)` 交给具体向量库实现（FAISS / Milvus / PGVector 等），把"检索"这件事彻底多态化。召回结果交给 `format_reference(kb_name, docs, api_address(is_public=True))` 格式化成出处。

`temp_kb` 路流程几乎一样，只是改用全局的 `check_embed_model()` 和 `search_temp_docs`，后者从 `memo_faiss_pool` 里 `acquire(knowledge_id)` 取临时 FAISS 库做 `similarity_search_with_score`，再把每条 `x[0].dict()` 化。它服务的是"上传文件后立刻问答"的临时知识，生命周期跟随内存池而非落盘库。

`search_engine` 路则完全不碰向量库：调用 `search_engine(query, top_k, kb_name)`（其中 `kb_name` 此时被复用为搜索引擎名称），把返回结果 `x.dict()` 化，并就地用列表推导拼出 `出处 [i] [filename](source)` 形式的字符串列表。它把联网搜索的结果硬塞进同一套 RAG 流程，让"实时联网问答"和"知识库问答"共享同一段生成逻辑。

`format_reference` 的内部值得一看：它逐条遍历 `docs`，从 `metadata.source` 取文件名，用 `urlencode` 把 `knowledge_base_name` 和 `file_name` 编进查询串，拼出一个指向 `{api_base_url}/knowledge_base/download_doc?...` 的下载 URL，最终格式是 `出处 [N] [filename](url) \n\n{page_content}\n\n`。也就是说，**出处不仅是文件名，而是一个可点击下载原文件的链接**，这让 RAG 回答具备了真正的可追溯性。`api_address(is_public=True)` 则保证在反向代理/云服务器场景下生成的是公网可访问地址——它会优先读取 `API_SERVER` 里配置的 `public_host`/`public_port`，缺省才回落到本机地址。

## 一图看清数据流时序

把上面拆散的环节串成一条时间线，`kb_chat` 一次流式知识库问答的骨架可浓缩成下面这段（保留关键调用、略去 langfuse 与异常分支）：

```python
async def knowledge_base_chat_iterator():
    history = [History.from_data(h) for h in history]          # 1. 归一化历史
    docs = await run_in_threadpool(search_docs, ...)            # 2. 召回
    source_documents = format_reference(kb_name, docs, ...)     # 3. 拼出处
    if return_direct:                                          # 3.5 纯检索逃生
        yield OpenAIChatOutput(..., docs=source_documents); return
    callback = AsyncIteratorCallbackHandler()                  # 4. 流式回调
    llm = get_ChatOpenAI(model_name=model, callbacks=[callback])
    context = "\n\n".join(d["page_content"] for d in docs)      # 5. 拼 context
    if len(docs) == 0: prompt_name = "empty"                   # 6. 无召回退化
    prompt_template = get_prompt_template("rag", prompt_name)   # 7. 取模板
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages(
        [h.to_msg_template() for h in history] + [input_msg])   # 8. 组装 prompt
    chain = chat_prompt | llm                                  # 9. LCEL 串联
    task = asyncio.create_task(wrap_done(                      # 10. 后台生成
        chain.ainvoke({"context": context, "question": query}), callback.done))
    yield OpenAIChatOutput(..., docs=source_documents)         # 11. 先发出处
    async for token in callback.aiter():                       # 12. 逐 token 推
        yield OpenAIChatOutput(content=token, ...)
    await task                                                 # 13. 回收任务
```

这十三步几乎一字不差地对应源码行序，是本章后续每个小节的索引。下面逐段细讲其中的设计要点。

## return_direct：纯检索逃生通道

在召回完成后、调用 LLM 之前，代码先检查 `return_direct`。若为真，立刻 `yield` 一个 `OpenAIChatOutput`，其 `object="chat.completion"`、`content=""`、`finish_reason="stop"`、`docs=source_documents`，然后 `return` 提前结束。这意味着开了 `return_direct` 的请求根本不会创建 callback、不会调用 LLM，纯粹把检索结果当成 docs 字段吐回去。对于"我只要检索、不要生成"的下游应用（比如自己接一个生成模型，或只做相似问题推荐），这是一条零成本的捷径。

注意它复用的是同一个 `OpenAIChatOutput` schema——Chatchat 刻意让自己的对话接口对齐 OpenAI Chat Completion 的响应结构，`id`、`object`、`role`、`choices` 一应俱全，只是额外塞了一个 `docs` 扩展字段（schema 的 `Config.extra = "allow"` 允许这种扩展）。这让前端或第三方客户端可以用现成的 OpenAI SDK 思路来消费它。

## context 拼接与 prompt 模板填充

当需要走 LLM 时，先准备回调与模型。`callback = AsyncIteratorCallbackHandler()` 是 langchain 提供的异步迭代回调，它是整条流式链路的心脏——LLM 每生成一个 token，就通过它推到一个内部队列里。`callbacks = [callback]`，随后可选地追加 Langfuse 的 `CallbackHandler`（仅当环境变量 `LANGFUSE_SECRET_KEY`/`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_HOST` 三者齐备时），用于可观测性追踪。这种"环境变量齐备才挂载"的写法，让 Langfuse 成为一个零侵入的可选插件：不配就完全不引入依赖。

`max_tokens` 做了一次兜底：若为 `None` 或 `0`，回落到 `Settings.model_settings.MAX_TOKENS`。然后 `get_ChatOpenAI(model_name=model, temperature=..., max_tokens=..., callbacks=callbacks)` 构造出一个 `ChatOpenAI` 实例。

`get_ChatOpenAI` 内部默认 `streaming=True`，并会遍历参数字典剔除值为 `None` 的项以避开 OpenAI 客户端的校验报错；它从 `get_model_info(model_name)` 取该模型的 `api_base_url`/`api_key`/`api_proxy`，把"模型名"翻译成"真实 API 端点"，构造失败时不抛异常而是 log 后返回 `None`。它还有一个 `local_wrap` 开关，开启时会把 base url 指向 Chatchat 自己的 `{api_address()}/v1` 本地包装层、api_key 填 `EMPTY`——`kb_chat` 里没开它，直接打到真实模型平台。源码里还能看到一段被注释掉的 reranker 逻辑（`USE_RERANKER`），说明重排在这条链路里目前是预留位，实际重排发生在检索层（见第 8 章）。

context 的拼接极其朴素：`context = "\n\n".join([doc["page_content"] for doc in docs])`——把所有召回片段的正文用双换行连起来，没有再做截断或重排。

prompt 的选择有一个关键分支：`if len(docs) == 0: prompt_name = "empty"`。当一条文档都没召回到时，强制切到 `empty` 模板（注意这里改写的是 `nonlocal` 声明过的外层 `prompt_name`）。随后 `prompt_template = get_prompt_template("rag", prompt_name)` 从配置里取出模板字符串。`get_prompt_template` 的实现是 `Settings.prompt_settings.model_dump().get(type, {}).get(name)`，也就是说所有 prompt 都集中在 `PromptSettings` 里、可经 `prompt_settings.yaml` 外部覆盖，第一个参数 `type`（这里是 `"rag"`）对应模板大类，第二个参数 `name` 对应该类下的具体模板名。`rag` 这一类下默认有两条模板：

- `default`：`【指令】根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说"根据已知信息无法回答该问题"……【已知信息】{{context}}【问题】{{question}}`
- `empty`：`请你回答我的问题:\n{{question}}`

可以看到模板用的是 Jinja2 的 `{{context}}` / `{{question}}` 占位（`PromptSettings` 文档串明确写了"除 Agent 模板使用 f-string 外，其它均使用 jinja2 格式"）。`default` 模板还内置了"无法回答就直说、不许编造、用中文"的硬约束，这是抑制幻觉的第一道闸门；而 `empty` 模板则在无召回时退化成纯 LLM 问答，并在出处里追加一段红色提示"未找到相关文档,该回答为大模型自身能力解答！"，老老实实告诉用户这次没用上知识库。

## 把 History 与 prompt 组装成 ChatPromptTemplate

模板字符串拿到后，要变成 langchain 能消费的消息序列。代码先把当前这条 RAG 提问包成一个 user 消息模板：`input_msg = History(role="user", content=prompt_template).to_msg_template(False)`，再和历史消息拼成完整的 `ChatPromptTemplate`：

```python
chat_prompt = ChatPromptTemplate.from_messages(
    [i.to_msg_template() for i in history] + [input_msg])
```

这里 `History.to_msg_template` 的 `is_raw` 参数是理解多轮安全性的关键。历史消息调用时 `is_raw` 默认为 `True`，会把内容用 `{% raw %}...{% endraw %}` 包起来——因为历史消息是用户/AI 说过的纯文本，**绝不能被当作 Jinja2 模板二次解析**，否则用户消息里偶然出现的 `{{` 就会触发模板注入或渲染异常。而当前这条 RAG prompt 传 `is_raw=False`，因为它**确实需要**把 `{{context}}`/`{{question}}` 当占位符渲染。同一个方法用一个布尔开关精准区分了"待渲染的模板"和"原样保留的历史文本"，这是个很克制的设计。`to_msg_template` 内部还做了角色映射（`assistant`↔`ai`、`user`↔`human`），统一成 langchain 的角色体系。

最终用 LCEL 管道符把 prompt 和 LLM 串起来：`chain = chat_prompt | llm`。在 `file_chat.py` 里则仍用旧式的 `LLMChain(prompt=chat_prompt, llm=model)`——两条链路一新一旧，恰好是 langchain 链式 API 演进的活化石。

## 异步流式：wrap_done + AsyncIteratorCallbackHandler

流式生成是整条链路最精巧的部分。它的核心是"生产-消费分离"：一个后台任务驱动 LLM 生成（生产 token），主协程通过回调的异步迭代器消费 token 并 `yield` 给 SSE。

二者之间靠一个 `asyncio.Event` 通信，互不阻塞。生产端是这样启动的：

```python
task = asyncio.create_task(wrap_done(
    chain.ainvoke({"context": context, "question": query}),
    callback.done),
)
```

`chain.ainvoke(...)` 是 LLM 的异步调用，传入的 dict 正好填上模板里的 `context` 和 `question`。它被 `wrap_done` 包了一层：`wrap_done` 的作用是 `await fn`，无论成功还是抛异常（异常会被 log 下来而不上抛），最后在 `finally` 里执行 `event.set()`。这里的 `event` 就是 `callback.done`——一个 `asyncio.Event`。也就是说，**LLM 一旦生成完毕（或出错），就置位 `done` 事件，通知消费端"没有更多 token 了"**。整个调用又被 `asyncio.create_task` 丢到后台，主协程不会在此阻塞。

消费端则是 `async for token in callback.aiter()`：`AsyncIteratorCallbackHandler` 内部维护一个队列，LLM 每吐一个 token 就入队，`aiter()` 不断出队；当 `done` 事件置位且队列排空，迭代自然结束。这套"`create_task` 生产 + `aiter` 消费 + `done` 事件收尾"的三件套，是 Chatchat 在 FastAPI 异步环境下做 token 级流式的标准范式，`kb_chat`、`file_chat` 完全一致。

为什么非要这么绕？因为 langchain 的 `chain.ainvoke` 本身是"一次性 await 出最终结果"的协程，并不天然产出一个可逐 token 迭代的流。Chatchat 的做法是把"驱动生成"和"读取 token"解耦到两个并发执行体里：生成在后台任务中跑、token 经回调队列旁路流出、`done` 事件充当结束信号。这样既保留了 LCEL Chain 的简洁组合（`chat_prompt | llm`），又拿到了 SSE 所需的增量输出能力。`wrap_done` 里 `except Exception` 只 log 不上抛、`finally` 必定 `event.set()` 的写法也很关键——它保证即便 LLM 调用炸了，消费端的 `aiter()` 也不会永远挂起等待。

## SSE 输出：先发出处，再逐 token 推送

消费循环把 token 包装成 SSE 事件吐出去。在无召回时，先给 `source_documents` 追加那条红色提示。然后分流式和非流式两种返回：

流式分支（`stream=True`）有一个刻意的顺序——**先单独 yield 一个只带 docs、content 为空的 chunk**。这样前端在收到第一个 SSE 包时就能立刻渲染出"参考来源"区块，不必等整段回答生成完：

```python
ret = OpenAIChatOutput(id=..., object="chat.completion.chunk",
                       content="", role="assistant",
                       model=model, docs=source_documents)
yield ret.model_dump_json()
```

这样前端在收到第一个 SSE 包时就能立刻渲染出"参考来源"区块，不必等整段回答生成完。之后才进入 `async for token in callback.aiter()`，把每个 token 包成 `object="chat.completion.chunk"`、`content=token` 的包逐个 yield。

每个包都 `model_dump_json()` 序列化，而 `OpenAIBaseOutput.model_dump` 会把它组织成标准的 `choices[0].delta.content` 结构——和 OpenAI 流式响应格式完全一致。

非流式分支则在循环里把所有 token 累加进 `answer`，最后一次性 yield 一个 `object="chat.completion"` 的完整响应。两个分支结束后都 `await task`，确保后台生成任务被正确回收，不留悬挂协程。

值得专门点出的是 `OpenAIChatOutput` 的序列化逻辑。它继承自 `OpenAIBaseOutput`，后者重写了 `model_dump`：当 `object == "chat.completion.chunk"` 时输出 `choices[0].delta.content`（流式增量），当 `object == "chat.completion"` 时输出 `choices[0].message.content`（完整消息）。`docs` 这种额外字段则通过 `model_extra` 透出（schema 的 `Config.extra = "allow"` 允许）。`model_dump_json` 还特意带了 `ensure_ascii=False`，保证中文不被转义成 `\uXXXX`。一句话：Chatchat 的对话响应在结构上"伪装"成 OpenAI，但又通过扩展字段携带了 RAG 特有的出处信息。

整个生成器外层包了两层异常处理：`asyncio.exceptions.CancelledError` 对应"用户中途断开/取消"，只 log 一句 warning 后静默 `return`；其它 `Exception` 则 yield 一个 `{"data": json.dumps({"error": str(e)})}` 把错误透传给前端。最外层根据 `stream` 决定返回 `EventSourceResponse(...)`（来自 `sse_starlette`，维持 SSE 长连接）还是 `await knowledge_base_chat_iterator().__anext__()`（取生成器的第一个、也是唯一一个产物，对应非流式只 yield 一次的事实）。

## 与 file_chat、chat 的对照

`file_chat` 几乎是 `kb_chat` 的"临时库特化版"。它跳过了知识库工厂，直接对 `memo_faiss_pool` 做 `similarity_search_with_score_by_vector`——先用 `get_Embeddings().aembed_query(query)` 把问题向量化成向量，再拿向量去临时库里搜，取 `top_k` 并按 `score_threshold` 过滤。

它的 context 用单换行 `"\n".join` 拼接（注意与 `kb_chat` 的双换行有别），prompt 在有无召回间硬切 `default`/`empty`，Chain 用旧式 `LLMChain(prompt=chat_prompt, llm=model)` 而非 LCEL 管道，输出格式也是自定义的 `{"answer": ...}` / `{"docs": ...}` 而非 OpenAI 结构。它还在文件前半段保留了上传入口 `upload_temp_docs`，用多线程把文件落到临时目录、`file2text` 切分后向量化进临时 FAISS 库，并返回一个临时库 ID。

`chat.py` 则是另一个量级——它面向带工具调用的 Agent 对话。它通过 `create_models_from_config` 按 `LLM_MODEL_CONFIG` 区分 `action_model`（构造 `ChatPlatformAI`）和普通模型，用 `PlatformToolsRunnable.create_agent_executor` 建出 agent executor。

历史从数据库 `filter_message` 取并反序成正序，跨轮的 `intermediate_steps` 用 langchain 的 `loads`/`dumps` 序列化后持久化进 message 的 metadata，下一轮再读回——这让 Agent 拥有了跨请求的"记忆"。它的流式输出会区分 `PlatformToolsAction`、`PlatformToolsFinish`、`PlatformToolsActionToolStart/End`、`PlatformToolsLLMStatus` 等多种事件类型，把工具调用的每一步（决定调哪个工具、工具开始、工具返回、LLM 续写）都流式化地推给前端。这套 Agent 机制是下一章的主题，这里只需记住：**纯 RAG 走 `kb_chat`/`file_chat` 的轻量 Chain，带工具的智能体走 `chat` 的 agent executor，二者在同一个 `server/chat/` 目录下并存。**

## 三条链路的设计同构与差异一览

把三条链路并排看，能更清楚地体会 Chatchat 的取舍。它们**同构**于：都用 `History` 归一化历史、都经 `get_prompt_template` 取模板、都靠 `AsyncIteratorCallbackHandler` + `wrap_done` 做流式（`chat` 是 Agent 版的回调，但思想一致）、都返回 `EventSourceResponse`。

差异则集中在三处。其一是**召回来源**：`kb_chat` 三选一（本地/临时/搜索），`file_chat` 固定为临时 FAISS 库，`chat` 不做检索而是靠工具调用按需取数据。其二是**Chain 形态**：`kb_chat` 用最新的 LCEL 管道 `chat_prompt | llm`，`file_chat` 用旧式 `LLMChain`，`chat` 用 `PlatformToolsRunnable.create_agent_executor` 建出的 agent executor，三者恰好是 langchain 链式 API 从老到新的三个断面。其三是**输出契约**：`kb_chat`/`chat` 对齐 OpenAI 的 `OpenAIChatOutput`，`file_chat` 用自定义的 `{"answer"}`/`{"docs"}`。

理解了这套"同构骨架 + 局部差异"的组织方式，就能预判：如果要新增一条对话链路（比如多模态问答），最省力的做法是复刻 `kb_chat` 的异步生成器骨架，只替换召回段和 Chain 段，流式与输出层几乎可以原样复用。

## 本章小结

1. `kb_chat` 是一个 FastAPI `async def` 接口，用 `mode`（`local_kb`/`temp_kb`/`search_engine`）统一三种召回源，召回后的"拼 context → 填 prompt → LLM 流式生成"流程完全复用。
2. 所有默认参数都引自全局 `Settings`（`VECTOR_SEARCH_TOP_K`、`SCORE_THRESHOLD`、`TEMPERATURE`、`get_default_llm()`），体现配置即契约。
3. 同步的向量检索通过 `run_in_threadpool` 嵌入异步接口，避免阻塞事件循环。
4. `format_reference` 把每条召回拼成带 `/knowledge_base/download_doc` 下载链接的出处字符串，让 RAG 回答可追溯到原文件。
5. `return_direct=True` 是纯检索逃生通道：跳过 LLM，直接把 `source_documents` 作为 docs 返回。
6. context 用 `"\n\n".join(page_content)` 朴素拼接；无召回时 `prompt_name` 强制切到 `empty` 模板并退化为纯 LLM 问答。
7. prompt 集中在 `PromptSettings.rag` 下的 `default`/`empty`，用 Jinja2 的 `{{context}}`/`{{question}}` 占位，`default` 内置"无法回答就直说、不许编造、用中文"的反幻觉约束。
8. `History.to_msg_template` 用 `is_raw` 开关区分待渲染模板（`False`）与原样保留的历史文本（`True`，用 `{% raw %}` 包裹防模板注入）。
9. 流式核心是 `asyncio.create_task(wrap_done(chain.ainvoke(...), callback.done))` 生产 + `async for token in callback.aiter()` 消费 + `done` 事件收尾的三件套。
10. SSE 流式刻意先 yield 只带 docs 的空 content chunk，让前端先渲染出处；输出对齐 OpenAI Chat Completion 的 `choices[0].delta` 结构。
11. `file_chat` 是临时库特化版（`LLMChain` + 自定义输出），`chat` 是带工具调用的 Agent 链路（agent executor + 多事件类型流式），三者共用 `History`/`get_prompt_template`/`wrap_done` 基础设施。

## 动手实验

1. **实验一：追踪一次完整问答的数据流** — 在 `kb_chat.py` 的 `knowledge_base_chat_iterator` 里，分别在 `docs` 召回后、`context` 拼接后、`chat_prompt` 组装后打印中间值，发起一次 `mode=local_kb` 的请求，观察召回片段如何被填进 `{{context}}`、最终 prompt 长什么样。

2. **实验二：验证 empty 模板的退化路径** — 用一个明显不在知识库里的冷门问题发起请求，确认 `len(docs) == 0` 触发 `prompt_name = "empty"`，并在 SSE 首包里看到那条红色"未找到相关文档"提示；再把 `prompt_settings.yaml` 里 `rag.empty` 改写一句，重启验证模板确实是外部可覆盖的。

3. **实验三：对比 return_direct 与正常生成** — 同一问题分别用 `return_direct=True` 和 `False` 各发一次，对比响应：前者只有 `docs`、`content` 为空、`object="chat.completion"` 且不消耗 LLM；后者走完整流式。据此理解纯检索逃生通道的用途。

4. **实验四：拆解流式三件套** — 在 `wrap_done` 的 `finally` 和 `callback.aiter()` 的消费循环里各加一行带时间戳的日志，发起 `stream=True` 请求，观察"后台任务驱动生成"与"主协程消费 token"如何交错，以及 `done` 事件置位后消费循环为何能自然结束。

> **下一章预告**：本章末尾我们瞥见了 `chat.py` 里 `create_models_from_config` 按 `action_model` 区分模型、`PlatformToolsRunnable` 建出 agent executor 的轮廓。第 10 章《Agent 注册表与多模型适配》将深入 `agents_registry`，拆解 Chatchat 如何在一套接口下注册不同 Agent 类型、用 `get_ChatOpenAI` 与 `ChatPlatformAI` 适配多家模型平台，以及工具调用的中间状态如何跨轮持久化。
