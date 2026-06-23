# 第 11 章 模型接入层

一个 RAG 引擎要落地，绕不开一个朴素却棘手的问题：用户手里的大模型五花八门——有人用 OpenAI，有人用通义千问、DeepSeek、智谱，有人在内网跑 Ollama 或 Xinference，还有人接的是 AWS Bedrock 或 Azure。它们的鉴权方式、请求字段、流式协议、推理思考字段、工具调用格式各不相同。如果让检索、对话、Agent 这些上层逻辑直接面对每一家厂商的差异，代码会迅速腐烂成一堆 `if provider == ...` 的分支。RAGFlow 的解法是在 `rag/llm/` 下建立一层**模型接入抽象**：把模型按能力分成 chat、rerank、cv（视觉）、tts（语音合成）、sequence2txt（语音识别）、ocr、embedding 等若干**类型**，每一类型一个工厂注册表，每一家厂商一个 provider 子类，再用统一的 Base 接口把差异收敛在子类内部。

本章聚焦 chat 及 rerank、cv、tts、sequence2txt、ocr 这些模型类型的接入机制（embedding 已在第 5 章详述，此处不再展开）。我们会从工厂注册模式入手，逐层剖开统一 chat 接口、LiteLLM 多厂商收敛、tool calling 装饰器、重试与 token 统计、密钥安全处理这几个最能体现"统一抽象"工程设计的点。所有论断都来自 `rag/llm/` 下的真实源码。

## 一、按能力分类型，每类型一张工厂表

打开 `rag/llm/__init__.py`，你看到的第一个设计决策就是**按模型能力切分命名空间**。文件顶部声明了八个全局字典：

```python
ChatModel = globals().get("ChatModel", {})
CvModel = globals().get("CvModel", {})
EmbeddingModel = globals().get("EmbeddingModel", {})
RerankModel = globals().get("RerankModel", {})
Seq2txtModel = globals().get("Seq2txtModel", {})
TTSModel = globals().get("TTSModel", {})
OcrModel = globals().get("OcrModel", {})
ModelMeta = globals().get("ModelMeta", {})
```

每个字典就是一类模型的**工厂注册表**：键是厂商名（如 `"OpenAI"`、`"DeepSeek"`），值是对应的 provider 类。上层只需要 `ChatModel["DeepSeek"](key, model_name, base_url)` 就能拿到一个能对话的实例，而完全不必知道 DeepSeek 内部是走 LiteLLM 还是裸 OpenAI SDK。这种"同一个厂商在不同能力维度各有一个实现"的切法很关键：同样是 `"OpenAI"`，在 `ChatModel` 里指向 chat 实现，在 `CvModel` 里指向多模态实现（`cv_model.py` 的 `GptV4`），在 `Seq2txtModel` 里指向 whisper 实现（`sequence2txt_model.py` 的 `GPTSeq2txt`）。能力与厂商是两个正交的维度，用"类型字典 + 厂商键"恰好把它们组织成二维表。

`MODULE_MAPPING` 把模块名和这些字典对应起来，注册过程则是一段反射代码，自动扫描每个模块里的类并按 `_FACTORY_NAME` 落表：

```python
MODULE_MAPPING = {
    "chat_model": ChatModel,
    "cv_model": CvModel,
    "embedding_model": EmbeddingModel,
    "rerank_model": RerankModel,
    "sequence2txt_model": Seq2txtModel,
    "tts_model": TTSModel,
    "ocr_model": OcrModel,
    "model_meta": ModelMeta,
}

for module_name, mapping_dict in MODULE_MAPPING.items():
    full_module_name = f"{package_name}.{module_name}"
    module = importlib.import_module(full_module_name)
    ...
    if base_class is not None:
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, base_class) and obj is not base_class and hasattr(obj, "_FACTORY_NAME"):
                if isinstance(obj._FACTORY_NAME, list):
                    for factory_name in obj._FACTORY_NAME:
                        mapping_dict[factory_name] = obj
                else:
                    mapping_dict[obj._FACTORY_NAME] = obj
```

这里有几个值得玩味的设计意图。其一，**注册是自动的**：新增一个厂商，只需在对应模块写一个继承 `Base` 的类并声明 `_FACTORY_NAME`，`__init__.py` 在导入时通过 `inspect.getmembers` 把它扫进字典，无需手动维护一张巨大的映射表。其二，`_FACTORY_NAME` **支持列表**——`chat_model.py` 里的 `OpenAI_APIChat` 就写成 `_FACTORY_NAME = ["VLLM", "OpenAI-API-Compatible"]`，一个实现同时认领两个厂商名，因为它们走的都是 OpenAI 兼容协议。其三，代码对 `LiteLLMBase` 做了**特殊处理**：扫描时若遇到名为 `"LiteLLMBase"` 的类，会断言它必须带 `_FACTORY_NAME`（`assert hasattr(obj, "_FACTORY_NAME")`），并把它声明的一长串厂商名全部指向自己。这正是下文要讲的"一个类吃下几十家厂商"的收敛策略。文件开头那句注释也透露了维护纪律——`AFTER UPDATING THIS FILE, PLEASE ENSURE THAT docs/references/supported_models.mdx IS ALSO UPDATED`，提醒注册表与文档要同步。

配置侧，`conf/llm_factories.json` 里登记了 64 家 factory，每家用 `tags` 字段标注它支持哪些能力。统计这些标签可以看到生态全貌：`LLM`（chat）53 家、`TEXT EMBEDDING` 41 家、`SPEECH2TEXT`（语音识别）18 家、`IMAGE2TEXT`（视觉）17 家、`TEXT RE-RANK` 16 家、`TTS` 12 家、`MODERATION` 11 家、`OCR` 3 家。这张配置表是给前端展示和能力过滤用的，真正的运行期实现仍在 `rag/llm/` 各模块的注册字典里。

## 二、chat 的统一接口：异步优先的 Base 抽象

chat 是接入面最复杂的类型，`chat_model.py` 超过两千行，但骨架清晰。所有非 LiteLLM 的本地与兼容厂商都继承 `chat_model.py` 里的 `Base(ABC)`，它在构造时同时建了同步与异步两个 OpenAI 客户端：

```python
class Base(ABC):
    def __init__(self, key, model_name, base_url, **kwargs):
        timeout = int(os.environ.get("LLM_TIMEOUT_SECONDS", 600))
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
        self.async_client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout)
        self.model_name = model_name
        self.max_retries = kwargs.get("max_retries", int(os.environ.get("LLM_MAX_RETRIES", 5)))
        self.base_delay = kwargs.get("retry_interval", float(os.environ.get("LLM_BASE_DELAY", 2.0)))
        self.max_rounds = kwargs.get("max_rounds", 5)
        self.is_tools = False
        self.tools = []
        self.toolcall_sessions = {}
```

接口设计上有一个明显倾向：**异步优先**。Base 暴露的主力方法是 `async_chat`、`async_chat_streamly`、`async_chat_with_tools`、`async_chat_streamly_with_tools`，全部是协程或异步生成器；同步的 `chat` / `chat_streamly` 只在少数本地实现（如 `LocalLLM`、`BaiChuanChat`）里出现。这与第 4 章讲的任务执行器、第 9 章的 Agent 画布都跑在异步事件循环上是一脉相承的——模型调用是天然的 I/O 等待，用协程才能在等待网络时把线程让出去。

统一接口约定了一套**消息装配规则**。`async_chat` 接收 `system`、`history`、`gen_conf` 三个参数，会把 system 提示按需插到历史最前：

```python
async def async_chat(self, system, history, gen_conf=None, **kwargs):
    gen_conf = dict(gen_conf or {})
    if system and history and history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": system})
    gen_conf = self._clean_conf(gen_conf)
    for attempt in range(self.max_retries + 1):
        try:
            return await self._async_chat(history, gen_conf, **kwargs)
        except Exception as e:
            e = await self._exceptions_async(e, attempt)
            if e:
                return e, 0
```

返回值统一是 `(answer_text, token_count)` 元组——把"文本"和"消耗的 token"绑在一起返回，是 RAGFlow 全链路计费与配额统计的基础。下文还会看到 rerank、cv、seq2txt 这些类型也都坚守"第二个返回值是 token 数"的约定。

流式接口 `async_chat_streamly` 则是异步生成器，逐块 yield 文本，最后再 yield 一个累计 token 数收尾。底层 `_async_chat_streamly` 在拼装每个 delta 时还顺手处理了**推理模型的思考链**：

```python
_reasoning = getattr(resp.choices[0].delta, "reasoning_content", None) or getattr(resp.choices[0].delta, "reasoning", None)
if kwargs.get("with_reasoning", True) and _reasoning:
    ans = ""
    if not reasoning_start:
        reasoning_start = True
        ans = "<think>"
    ans += _reasoning + "</think>"
```

不同厂商把思考内容放在 `reasoning_content` 或 `reasoning` 字段，这里用 `getattr(..., None) or getattr(..., None)` 兼容两种命名，并统一包进 `<think>...</think>` 标签往外吐——上层只认这一种标记，又一次把厂商差异关在了 Base 内部。流式还监测 `finish_reason == "length"`，命中就追加 `LENGTH_NOTIFICATION_CN` / `LENGTH_NOTIFICATION_EN`，按内容中英文给用户一句"回答被上下文窗口截断"的提示。

那么各厂商子类怎么处理差异？大多数子类几乎不写逻辑，只在构造函数里调整 base_url。`XinferenceChat`、`HuggingFaceChat`、`ModelScopeChat` 都只是把传入的本地 url 拼上 `/v1` 再交给父类；后两者还会对 `model_name.split("___")[0]` 做切分，因为 RAGFlow 在前端用 `___` 拼接了模型名与部署标识。`BaiChuanChat` 则展示了更深的定制——它重写了 `_clean_conf`、`_chat`、`chat_streamly`，在请求里硬塞了百川的联网搜索 `extra_body`，并把采样参数收窄到 `temperature` / `top_p`。这种"能复用就空壳继承、需要差异就局部重写"的分层，正是抽象基类模式的价值所在。

## 三、用 LiteLLM 一个类收敛几十家厂商

如果每家厂商都像 `XinferenceChat` 那样写一个子类，几十家也还能忍；但 chat 厂商实在太多。RAGFlow 的杀手锏是引入 [LiteLLM](https://docs.litellm.ai/)，用一个 `LiteLLMBase` 类吃下绝大多数主流云厂商。它的 `_FACTORY_NAME` 是一张长长的列表：

```python
class LiteLLMBase(ABC):
    _FACTORY_NAME = [
        "Tongyi-Qianwen", "Bedrock", "Moonshot", "xAI", "DeepInfra",
        "Groq", "Cohere", "Gemini", "DeepSeek", "NVIDIA", "TogetherAI",
        "Anthropic", "Ollama", "LongCat", "CometAPI", "SILICONFLOW",
        "OpenRouter", "StepFun", "PPIO", "PerfXCloud", "Upstage",
        "NovitaAI", "01.AI", "GiteeAI", "302.AI", "Jiekou.AI", "ZHIPU-AI",
        "MiniMax", "DeerAPI", "GPUStack", "OpenAI", "Azure-OpenAI",
        "Tencent Hunyuan",
    ]
```

`__init__.py` 的注册循环会把这 33 个名字全部映射到 `LiteLLMBase`。换言之，用户在界面上选"DeepSeek"还是"Anthropic"，最后落到的是同一个类，差异靠两张映射表区分。第一张是 `LITELLM_PROVIDER_PREFIX`，给每家厂商一个 LiteLLM 路由前缀；第二张是 `FACTORY_DEFAULT_BASE_URL`，给每家厂商一个默认端点。构造时把前缀拼到模型名上：

```python
def __init__(self, key, model_name, base_url=None, **kwargs):
    self.provider = kwargs.get("provider", "")
    self.prefix = LITELLM_PROVIDER_PREFIX.get(self.provider, "")
    self.model_name = f"{self.prefix}{model_name}"
    self.api_key = key
    self.base_url = (base_url or FACTORY_DEFAULT_BASE_URL.get(self.provider, "")).rstrip("/")
```

看 `LITELLM_PROVIDER_PREFIX` 的内容很有意思：`DeepSeek` 是 `"deepseek/"`、`Gemini` 是 `"gemini/"`、`Ollama` 是 `"ollama_chat/"`、`Bedrock` 是 `"bedrock/"`、`Anthropic` 是空串（注释写着 `# don't need a prefix`）。而**一大批厂商共享 `"openai/"` 前缀**——`SILICONFLOW`、`OpenRouter`、`StepFun`、`ZHIPU-AI`、`MiniMax`、`GPUStack`、`Tencent Hunyuan` 等等。原因是这些厂商都提供 OpenAI 兼容端点，LiteLLM 用 `openai/` 路由加上各自的 `api_base` 即可对接，无需为每家写专门的适配器。这就是"收敛"的本质：把"OpenAI 协议兼容"当成最大公约数，再用 base_url 区分到具体厂商。

真正发请求的是 `litellm.acompletion`，请求参数由 `_construct_completion_args` 统一拼装。这个方法是整个 LiteLLM 路径的核心，它在通用骨架之上为有特殊鉴权需求的厂商开了专门分支：

```python
def _construct_completion_args(self, history, stream: bool, tools: bool, **kwargs):
    completion_args = {
        "model": self.model_name,
        "messages": history,
        "api_key": self.api_key,
        "num_retries": self.max_retries,
        **kwargs,
    }
    ...
    if self.provider in FACTORY_DEFAULT_BASE_URL:
        completion_args.update({"api_base": self.base_url})
    elif self.provider == SupportedLiteLLMProvider.Bedrock:
        import boto3
        ...
```

AWS Bedrock 的处理最重——它要解析 key 里的 `auth_mode`，支持 `access_key_secret`（直接给 AK/SK）、`iam_role`（用 STS `assume_role` 换临时凭证）以及默认凭证链三种模式，把 `api_key`/`api_base` 删掉换成一组 AWS 参数。Azure 则要额外塞 `api_version`；OpenRouter 支持 `provider_order` 路由优先级；MiniMax 要把 `GroupId` 拼到 URL 的 query 上；Ollama 在反向代理后要补 `Authorization: Bearer` 头（注释引用了 issue #11350）。所有这些诡异的厂商私货都被关进这一个方法里，对上层完全透明。调用 `acompletion` 时还统一带了 `drop_params=True`，让 LiteLLM 自动丢掉目标模型不认识的参数，避免"Unknown parameter"类报错。

LiteLLM 路径还有一个细节体现了对推理模型的照顾：`_need_reasoning_content_back` 只对 DeepSeek 返回 True，在带工具的对话里会把 `reasoning_content` 回写进 assistant 消息（`_append_history` / `_append_history_batch` 的 `reasoning_content` 参数），因为 DeepSeek 的工具调用协议要求把思考链带回上下文。

## 四、生成参数的白名单与模型族策略

把几十家厂商收敛到统一接口，最容易翻车的地方是**参数**：上层传下来的 `gen_conf` 可能混进 RAGFlow 内部的元数据（如 `model_type`），也可能含某些厂商不认的字段。`chat_model.py` 用一个**白名单常量**兜底：

```python
ALLOWED_GEN_CONF_KEYS = frozenset(
    {
        "temperature", "max_completion_tokens", "top_p", "stream",
        "stream_options", "stop", "n", "presence_penalty", "frequency_penalty",
        "functions", "function_call", "logit_bias", "user", "response_format",
        "seed", "tools", "tool_choice", "logprobs", "top_logprobs", "extra_headers",
    }
)
```

注释写得很直白：`gen_conf` 来自对话助手的 `llm_setting`，里面可能带 `model_type` 这种内部字段，"Anything outside this set is dropped so providers don't reject the request"，并明确引用了 issue #15427。`_clean_conf` 就靠 `{k: v for k, v in gen_conf.items() if k in ALLOWED_GEN_CONF_KEYS}` 把请求洗干净。LiteLLM 路径用的是更宽的 `LITELLM_ALLOWED_GEN_CONF_KEYS`，在白名单基础上额外放行 `thinking`、`reasoning_effort`、`extra_body` 这些推理控制参数，因为 LiteLLM 能理解它们。

比白名单更精细的是 `_apply_model_family_policies` 这个函数，它按"模型族"打补丁。逻辑全是真实业务踩坑的沉淀：`qwen3` 系列在非流式请求里要塞 `extra_body={"enable_thinking": False}` 关掉思考；OpenAI/Azure 的 `gpt-5` 要删掉 `temperature`、`top_p`、`logprobs` 等不被支持的采样参数；Anthropic 的 `claude-opus-4-7` / `claude-opus-4-8` 要删 `temperature`、`top_p`、`top_k`；腾讯混元要删 `presence_penalty`、`frequency_penalty`；`kimi-k2.5` / `kimi-k2.6` 要把 `reasoning` 翻译成 `thinking={"type": "enabled"/"disabled"}` 并固定一组采样参数。这些策略以 `backend`（`"base"` 或 `"litellm"`）和 `provider` 为条件分发，`_clean_conf` 在两条路径上各自调用它。把这些"某个模型有某个怪癖"的知识集中到一个纯函数里，而不是散落在各处的 if，是这层抽象保持可维护的关键。

## 五、tool calling：装饰器、会话适配与多轮循环

RAGFlow 的 Agent 要让大模型调用工具，模型接入层为此提供了三块拼图：把普通 Python 函数变成工具的装饰器、把工具列表适配成会话协议的适配器、以及驱动"模型出工具调用→执行→回灌结果"的多轮循环。

第一块在 `tool_decorator.py`。`@tool` 装饰器的全部工作就是给函数挂两个属性：

```python
def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    fn.openai_schema = _build_openai_schema(fn)
    fn._is_tool = True
    return fn
```

`_build_openai_schema` 通过 `inspect.signature` 读签名、`get_type_hints` 读类型注解、再用 `_parse_param_docs` 从 docstring 里抠出函数描述和 `:param name:` 行，自动拼出 OpenAI function-calling 所需的 JSON Schema。类型映射由 `_PY_TO_JSON` 完成（`str→"string"`、`int→"integer"`、`list→"array"` 等），`_py_type_to_json` 还能拆 `Optional[T]` / `T | None` 并递归处理 `list[T]`。无默认值的参数被收进 `required`。这意味着开发者写一个带类型注解和文档字符串的普通函数，加一行 `@tool`，就能直接喂给 LLM，无需手写那一大坨 schema。

第二块是 `FunctionToolSession`，它把一组 `@tool` 函数适配成 chat 模型期望的 `ToolCallSession` 协议——注释特意说明它用鸭子类型而非显式继承，"to avoid pulling the MCP client SDK into this module's import graph"。它维护一个 `tools_map`（函数名→可调用对象），暴露 `schemas` 属性和 `tool_call` / `tool_call_async` 方法。执行时它会判断目标是协程还是同步函数：

```python
if asyncio.iscoroutinefunction(fn):
    coro = fn(**arguments)
else:
    coro = thread_pool_exec(fn, **arguments)
return await asyncio.wait_for(coro, timeout=request_timeout)
```

同步函数被推进线程池（`thread_pool_exec`）以免阻塞事件循环，且整体套了 `asyncio.wait_for` 超时。注释还诚实地标注了局限：同步工具超时后 Python 无法中断底层线程，函数会在后台继续跑。这一点提醒第 12 章 MCP 里要讲的远程工具，与本地函数工具走的是同一套 `ToolCallSession` 抽象——本章的 `FunctionToolSession` 只是这套协议的本地实现之一。

`bind_tools` 把两块拼图接上，它同时支持两种调用风格：旧式 `bind_tools(toolcall_session, tools_schemas)`（Agent/对话层用预构建 schema），新式 `bind_tools(tools=[fn1, fn2])`（传 `@tool` 函数，自动建会话）。判断逻辑是 `all(is_tool(t) for t in tools)`——若全是装饰过的函数，就自动包一个 `FunctionToolSession`。

第三块是多轮工具循环，`async_chat_with_tools` 是其代表。它最外层套 `max_retries` 重试，内层 `for _ in range(self.max_rounds + 1)` 控制对话轮数（默认 `max_rounds=5`）。每一轮带着 `tools=self.tools, tool_choice="auto"` 请求模型；若返回没有 `tool_calls`，说明模型给出了最终答案，直接返回；若有工具调用，则**并发执行**所有工具再回灌：

```python
results = await asyncio.gather(*[_exec_tool(tc) for tc in response.choices[0].message.tool_calls])
history = self._append_history_batch(history, results)
```

`_exec_tool` 用 `json_repair.loads` 解析模型给的参数（容忍模型吐出的不规范 JSON），调用 `tool_call_async`，并把异常作为结果的一部分返回而不抛出——这样一个工具失败不会中断整轮。`_append_history_batch` 严格按 OpenAI 协议组织历史：一条带全部 `tool_calls` 的 assistant 消息，后跟每个调用一条 `role: "tool"` 的结果消息。若轮数耗尽仍未收敛，循环外会追加一句 `"Exceed max rounds"` 提示再做最后一次普通对话兜底。流式版本 `async_chat_streamly_with_tools` 逻辑同构，额外要处理流式分片里 `tool_calls` 参数的增量拼接（按 `index` 把 `function.arguments` 一段段累加）。值得一提的是，LiteLLM 路径的 `LiteLLMBase` 也实现了一份结构几乎相同的工具循环，只是把底层调用换成 `litellm.acompletion` 并多了 DeepSeek 思考链回写——两条路径在工具循环这件事上保持了行为一致。

## 六、横切关注点：重试、错误分类、token 统计

把多厂商接进来，错误处理必须统一，否则上层无法判断该重试还是该放弃。`chat_model.py` 定义了 `LLMErrorCode` 枚举（`ERROR_RATE_LIMIT`、`ERROR_AUTHENTICATION`、`ERROR_QUOTA`、`ERROR_SERVER`、`ERROR_TIMEOUT` 等），并用 `_classify_error` 做**基于关键词的归类**：

```python
keywords_mapping = [
    (["quota", "capacity", "credit", "billing", "balance", "欠费"], LLMErrorCode.ERROR_QUOTA),
    (["rate limit", "429", "tpm limit", "too many requests", "requests per minute"], LLMErrorCode.ERROR_RATE_LIMIT),
    (["auth", "key", "apikey", "401", "forbidden", "permission"], LLMErrorCode.ERROR_AUTHENTICATION),
    ...
]
```

各家厂商的报错文案千差万别，但只要里面出现 `429` 或 `rate limit` 就归到限流。注意关键词表里连中文 `"欠费"` 都考虑到了，可见是面向真实国产厂商打磨过的。`_should_retry` 只把 `ERROR_RATE_LIMIT` 和 `ERROR_SERVER` 视为可重试——限流和服务端 5xx 是瞬时故障，重试有意义；而鉴权、参数错误、配额耗尽这类是确定性失败，重试只会浪费配额，直接拼成 `**ERROR**: ...` 文本返回。重试退避用的是 `_get_delay`，`self.base_delay * random.uniform(10, 150)`——带较大随机抖动的退避，避免多个并发请求在限流后同时重试形成"惊群"。Base 和 LiteLLMBase 各有一份 `_exceptions` / `_exceptions_async` 实现，行为一致：到达 `max_retries` 时把错误码改判为 `ERROR_MAX_RETRIES` 再放弃。重试参数全部可由环境变量调（`LLM_MAX_RETRIES` 默认 5、`LLM_BASE_DELAY` 默认 2.0、`LLM_TIMEOUT_SECONDS` 默认 600）。

token 统计是另一个贯穿所有类型的横切点。chat 用 `common.token_utils` 的 `total_token_count_from_response` 优先从响应的 usage 里取，取不到再用 `num_tokens_from_string` 本地估算 delta 文本——流式场景下厂商不一定每个分片都回 usage，这种"先官方后估算"的双保险保证了计费不会因缺字段而归零。所有 chat 方法的返回值都带着 token 数，这个数字最终汇入 RAGFlow 的用量统计。

## 七、其它模型类型：同构的抽象，各自的协议

chat 之外的几类模型，复用了同一套"Base 抽象 + provider 子类 + 工厂注册 + (文本, token) 返回"的范式，只是各自适配不同的下游协议。

**rerank（重排）** 在 `rerank_model.py`，与第 6 章的混合检索直接呼应。它的 `Base` 设计极具代表性：唯一公开入口 `similarity(query, texts)` 返回 `(rank, token_count)`，内部先短路空输入，再调子类的 `_compute_rank`，最后**强制把分数归一化到 [0, 1]**：

```python
def similarity(self, query: str, texts: List) -> Tuple[np.ndarray, int]:
    if not query or not texts:
        return np.zeros(len(texts) if texts else 0, dtype=float), 0
    rank, token_count = self._compute_rank(query, texts)
    rank = np.asarray(rank, dtype=float)
    ...
    return self._normalize_rank(rank), token_count
```

`_normalize_rank` 的注释解释了为什么这一步不能省：下游的混合打分会把 rerank 分数和关键词相似度在固定的 [0, 1] 区间上加权融合，像 NVIDIA reranker 那种输出无界 logit（甚至负数）的厂商若不归一会污染最终排序。它的策略很讲究——已经落在 [0,1] 的（Cohere、Jina、Voyage 等天然校准的分数）原样返回以保留 `similarity_threshold` 语义，只对越界输出做 min-max 映射，而对没有区分度的批次（单候选或方差极小）改用 `np.clip` 钳位，免得"一个孤立的高分被静默清零"。子类按厂商各写 `_compute_rank`：`JinaRerank` 走 `/v1/rerank` HTTP 接口，`GPUStackRerank` 拼 `URL(base_url) / "v1" / "rerank"`，都从响应的 `results` 里按 `index` 回填 `relevance_score`。这里的工厂表同样用 `_FACTORY_NAME` 注册，`CoHereRerank` 还用列表认领了 `["Cohere", "VLLM"]` 两家。

**cv（多模态视觉）** 在 `cv_model.py`，核心是把图片喂给视觉大模型生成描述。`Base` 提供了一组图片归一化工具：`_blob_to_data_url` 把 bytes / BytesIO / 已是 url 的输入统一转成 `data:image/...;base64,...` 形式，`image2base64` 还会嗅探 JPEG 魔数（`0xFF 0xD8`）来定 MIME，避免厂商因 MIME 不符报错。`_image_prompt` 把文本和图片拼成 OpenAI 多模态消息格式（`{"type": "image_url", "image_url": {"url": ...}}`）。子类 `GptV4`（`_FACTORY_NAME = "OpenAI"`）实现 `describe` 和 `describe_with_prompt`，前者用内置的中英文提示词让模型描述图中时间地点人物，后者接受自定义 prompt。`AzureGptV4` 继承 `GptV4` 只换 Azure 客户端，`xAICV` 同理——又是空壳继承复用的范例。

**sequence2txt（语音识别）** 在 `sequence2txt_model.py`，`Base.transcription` 直接用 OpenAI SDK 的 `client.audio.transcriptions.create`，返回 `(text, num_tokens_from_string(text))`。`GPTSeq2txt` 默认模型 `whisper-1`，`StepFunSeq2txt`、`FuturMixSeq2txt`、`GiteeSeq2txt` 等只改 base_url 继承它；`QWenSeq2txt` 则因为通义走 dashscope SDK 而完全重写 `transcription`。**tts（语音合成）** 在 `tts_model.py`，`HTTPBasedTTS` 封装通用 HTTP 流式请求，子类按厂商定 payload；它甚至用 pydantic 定义了 `ServeTTSRequest` 来约束 Fish Audio 的请求结构。

**ocr** 在 `ocr_model.py`，是个有趣的反例——它接的不是云端大模型，而是把 `deepdoc` 的解析器（`MinerUParser`、`PaddleOCRParser`、`OpenDataLoaderParser`）包装成模型 provider。`MinerUOcrModel` 同时继承 `Base` 和 `MinerUParser`，从 key（可能是 JSON）里解析出 apiserver 地址等配置，对外暴露统一的 `parse_pdf`。这说明"模型接入层"的抽象足够宽泛，连本地 OCR 工具也能套进同一个工厂体系，让上层用一致的方式获取能力。

## 八、密钥安全处理

接入这么多厂商，密钥格式千奇百怪：有的是纯字符串，有的是 JSON（Azure 的 key 里塞了 `api_version`、Bedrock 塞了 `auth_mode` 和 AK/SK、MiniMax 塞了 `group_id`、OpenRouter 塞了 `provider_order`）。模型接入层用两种手法处理。

其一是**就地解析**。`LiteLLMBase.__init__` 针对不同 provider 解析 key：OpenRouter 用 `json.loads(key).get("api_key")` 取出真正的 key 并单独留下 `provider_order`，Azure 取出 `api_key` 和 `api_version`，MiniMax 取出 `api_key` 和 `group_id`，并都对 `JSONDecodeError` 做了降级（解析失败就当纯字符串用）。`key_utils.py` 的 `_normalize_replicate_key` 是这种逻辑的独立工具版，能从 dict 或 JSON 字符串里提取 `api_key`，否则原样返回。

其二是**日志脱敏**。OCR 那几个 provider 在打印解析出的配置前，会遍历键名做敏感词过滤：

```python
for k, v in config.items():
    if any(sensitive_word in k.lower() for sensitive_word in ("key", "password", "token", "secret")):
        redacted_config[k] = "[REDACTED]"
    else:
        redacted_config[k] = v
logging.info(f"Parsed MinerU config (sensitive fields redacted): {redacted_config}")
```

凡键名含 `key`/`password`/`token`/`secret` 的一律替换成 `[REDACTED]` 再落日志。在一个会把请求历史 `logging.info` 出来的系统里（chat 的 `[HISTORY]` 日志），这种主动脱敏是必要的合规防线。

## 本章小结

1. RAGFlow 的模型接入层按**能力类型**分库（chat / rerank / cv / tts / sequence2txt / ocr / embedding），每类型在 `rag/llm/__init__.py` 里维护一张工厂注册表字典，能力与厂商构成正交的二维表。
2. 注册是**反射自动化**的：`__init__.py` 用 `inspect.getmembers` 扫描每个模块里继承 `Base` 的类，按其 `_FACTORY_NAME`（支持单值或列表）落表，新增厂商无需手改映射表。
3. chat 的 `Base` 抽象**异步优先**，主力方法是 `async_chat` / `async_chat_streamly` 及其工具版本，统一返回 `(文本, token_count)`，并把推理模型的思考链统一包成 `<think>...</think>`。
4. `LiteLLMBase` 用一个类的 `_FACTORY_NAME` 列表收敛 33 家厂商，靠 `LITELLM_PROVIDER_PREFIX`（大量厂商共享 `openai/` 前缀）和 `FACTORY_DEFAULT_BASE_URL` 两张表区分，`_construct_completion_args` 集中处理 Bedrock/Azure/OpenRouter/MiniMax/Ollama 等的特殊鉴权。
5. 生成参数用 `ALLOWED_GEN_CONF_KEYS` 白名单过滤掉内部元数据与不支持字段，`_apply_model_family_policies` 按模型族（qwen3、gpt-5、claude-opus、混元、kimi 等）打补丁，把厂商怪癖集中到一个纯函数里。
6. tool calling 由三块拼图组成：`@tool` 装饰器自动从签名与 docstring 生成 OpenAI schema、`FunctionToolSession` 鸭子类型适配会话协议、`async_chat_with_tools` 的多轮循环并发执行工具并按 OpenAI 协议回灌历史。
7. 重试与错误处理统一：`_classify_error` 按关键词（含中文"欠费"）归类，只对限流和 5xx 重试，退避用 `base_delay * random.uniform(10, 150)` 带抖动，参数由 `LLM_MAX_RETRIES` 等环境变量控制。
8. token 统计贯穿所有类型，优先取响应 usage，缺失则用 `num_tokens_from_string` 本地估算，为全链路计费托底。
9. rerank 的 `Base.similarity` 强制把各厂商分数归一化到 [0, 1]，区分"已校准分数原样保留"与"无界 logit 做 min-max"两种策略，服务于第 6 章的混合打分。
10. cv/seq2txt/tts/ocr 复用同一范式：cv 统一图片到 data-url 并嗅探 MIME，seq2txt 走 whisper 风格转写，ocr 则把 deepdoc 本地解析器也包装成 provider。
11. 密钥安全有两手：`LiteLLMBase` 与 `key_utils.py` 就地解析 JSON 形态的复合 key 并对解析失败降级，OCR provider 在落日志前对含 `key/password/token/secret` 的字段做 `[REDACTED]` 脱敏。

## 动手实验

1. **实验一：画出工厂注册表** — 在 `rag/llm/` 目录下写一段脚本 `import rag.llm`，然后打印 `rag.llm.ChatModel`、`rag.llm.RerankModel`、`rag.llm.CvModel` 三个字典的键集合，对照 `conf/llm_factories.json` 里对应 `tags` 的厂商数（LLM 53、TEXT RE-RANK 16、IMAGE2TEXT 17），观察哪些厂商在某一能力维度缺失，理解"能力×厂商"二维表的稀疏性。

2. **实验二：验证 LiteLLM 前缀收敛** — 读 `rag/llm/__init__.py` 的 `LITELLM_PROVIDER_PREFIX`，统计有多少厂商共享 `"openai/"` 前缀、多少用专属前缀。然后挑一家共享 `openai/` 的厂商（如 SILICONFLOW）和一家专属前缀的厂商（如 DeepSeek），分别走查 `LiteLLMBase.__init__` 里 `self.model_name = f"{self.prefix}{model_name}"` 的拼接结果，说明为什么 OpenAI 兼容端点能被当成最大公约数复用。

3. **实验三：给一个 Python 函数加 @tool** — 用 `from rag.llm.tool_decorator import tool` 装饰一个带类型注解和 `:param:` docstring 的函数（例如 `def add(a: int, b: int) -> int`），打印它的 `fn.openai_schema`，确认 `_build_openai_schema` 正确推断出了 `properties`、`required` 与 `description`；再把它放进 `FunctionToolSession([fn])`，调用 `session.schemas` 和 `tool_call("add", {"a":1,"b":2})`，观察同步函数如何被 `thread_pool_exec` 推进线程池。

4. **实验四：复现错误分类与重试判定** — 实例化任意 chat provider，构造几个假异常（消息分别含 `"429 too many requests"`、`"401 invalid api key"`、`"insufficient quota / 欠费"`、`"503 service unavailable"`），调用其 `_classify_error` 看归到哪个 `LLMErrorCode`，再调 `_should_retry` 确认只有限流和 5xx 返回 True；改写 `LLM_BASE_DELAY` 环境变量后打印 `_get_delay()` 几次，体会 `random.uniform(10, 150)` 带来的退避抖动范围。

> **下一章预告**：本章讲的 `FunctionToolSession` 只是 `ToolCallSession` 协议的本地函数实现，而 RAGFlow 还能把工具能力跨进程、跨网络地暴露出去。第 12 章《MCP 接入》将解构 `mcp` 目录下 client 与 server 的实现，看 RAGFlow 如何用 Model Context Protocol 既作为客户端消费外部 MCP 工具、又把自身的检索与知识库能力包装成 MCP server 暴露给其它 Agent，并与本章的工具循环抽象无缝衔接。
