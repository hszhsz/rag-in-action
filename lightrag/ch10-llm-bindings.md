# 第 10 章 LLM 接入与多供应商绑定

第 1 章我们就立下一条贯穿全书的契约：LightRAG 不"内置"任何一家模型，它只规定函数签名——`llm_model_func` 是一个 `async def f(prompt, system_prompt, history_messages, **kwargs) -> str`，`embedding_func` 是一个包了维度元数据的 `EmbeddingFunc`。把符合签名的函数塞进 `LightRAG(...)`，框架就能跑。这一章把视线从"契约"拉到"实现"，看 LightRAG 是怎么用一个 `llm/` 目录把整个模型生态的碎片一口口吃下去的。

整个接入层可以拆成四个相互独立又彼此咬合的部分：

- **生态适配层**：每个供应商一个驱动文件，提供函数式的 `xxx_complete_if_cache` / `xxx_embed`；
- **配置层**：`binding_options.py` 用 dataclass 把每家的可调参数声明成"配置即数据"；
- **精排层**：`rerank.py` 提供检索后重排的可选环节；
- **角色调度层**：`llm_roles.py` 让抽取、关键词、查询、视觉四种任务各自绑不同模型、各走一条并发队列。

读完它们，你会理解 LightRAG 为什么能在"换一个模型只改一行 func"和"给每个阶段配不同模型"两个极端之间自由滑动。

## 函数式契约：一个目录吃掉一个生态

`lightrag/llm/__init__.py` 是空文件——这是个刻意的信号：接入层没有"基类继承体系"，没有 `LLMProvider` 抽象类要你去 subclass。每个供应商就是一个平铺的模块文件，里面是几个自由函数。看一眼目录就知道战场有多大：`openai.py`、`azure_openai.py`、`ollama.py`、`anthropic.py`、`bedrock.py`、`gemini.py`、`zhipu.py`、`hf.py`、`jina.py`、`nvidia_openai.py`、`voyageai.py`、`lollms.py`、`lmdeploy.py`、`llama_index_impl.py`，外加一个 `deprecated/` 目录放历史包袱。

每个文件都遵守同一份"函数式契约"。以补全为例，对比四家的核心签名就能看出收敛得有多齐：

- `openai.py` 的 `openai_complete_if_cache(model, prompt, system_prompt=None, history_messages=None, enable_cot=False, ...)`；
- `anthropic.py` 的 `anthropic_complete_if_cache(model, prompt, system_prompt=None, history_messages=None, enable_cot=False, base_url=None, api_key=None, image_inputs=None, **kwargs)`；
- `ollama.py` 的 `_ollama_model_if_cache(model, prompt, system_prompt=None, history_messages=[], enable_cot=False, image_inputs=None, **kwargs)`；
- `gemini.py` 的 `gemini_complete_if_cache(...)`。

四家来自完全不同的 SDK（`openai`、`anthropic`、`ollama`、`google-genai`），却暴露了几乎逐字相同的入参。这就是契约的价值：上层管线（第 9 章的 Pipeline、第 6 章的检索）调用时永远是同一套关键字参数，根本不需要知道背后是谁。

`**kwargs` 是收纳箱，供应商特有的参数从这里漏进去，再由各驱动各自消化。但"消化"的方式各家不同，这恰恰暴露了契约统一之下的真实差异：

- `openai.py` 把 `response_format={"type": "json_object"}` 这类 dict 原样转发给 chat completions；
- `ollama.py` 的 `_normalize_ollama_response_format` 把 OpenAI 风格的 `response_format` 翻译成 Ollama 原生的 `format` 字段（`{"type": "json_object"}` → `format="json"`）；
- `anthropic.py` 干脆 `kwargs.pop("response_format", None)`——Messages API 没有 JSON 模式，传了也只能丢弃。

也就是说契约统一的是"调用方怎么传"，而不是"模型支持什么"。能力差异被各驱动吸收为"翻译"或"静默忽略"，上层因此可以无脑传递。同理 `enable_cot` 在 OpenAI 是一套思维链状态机，在 Ollama 和 Anthropic 则只打一行 `logger.debug` 说明不支持并忽略——契约保证"传了不报错"，而非"传了都生效"。

在 `xxx_complete_if_cache` 之上，每个驱动还会包几层"便利函数"。`openai.py` 里 `gpt_4o_complete` / `gpt_4o_mini_complete` / `nvidia_openai_complete` 都只是把模型名写死、转手调 `openai_complete_if_cache`；`openai_complete` 则从 `kwargs["hashing_kv"].global_config["llm_model_name"]` 取模型名——这正是 LightRAG 注入缓存句柄的约定通道。

`anthropic.py` 里 `claude_3_opus_complete` / `claude_3_sonnet_complete` / `claude_3_haiku_complete` 同理，`ollama.py` 有 `ollama_model_complete`，`gemini.py` 有 `gemini_model_complete`。这些便利函数就是给用户"开箱即用"的：`llm_model_func=gpt_4o_mini_complete` 一行搞定，连模型名都不用写。`azure_openai.py` 更是个只有几十行的薄壳，几乎全靠 `from .openai import azure_openai_*` 转发——足见这套函数式契约的复用密度。

依赖也是惰性安装的。`openai.py` 顶部 `if not pm.is_installed("openai"): pm.install("openai")`——用 `pipmaster` 在导入时按需装包，所以你装 LightRAG 不必把十几家 SDK 全拖下来，只在真正 import 某个驱动时才装它的依赖。

这是"吃掉生态碎片"却不让依赖爆炸的工程手段：内置十余家供应商，但每个用户的实际依赖只是自己用到的那一两家。各驱动文件顶部还都 `load_dotenv(dotenv_path=".env", override=False)`，刻意用当前目录的 `.env` 且不覆盖已有的 OS 环境变量——这让同一台机器上的多个 LightRAG 实例可以各带一份 `.env` 配置而互不干扰。

## `openai_complete_if_cache`：补全核心的解剖

`openai.py` 的 `openai_complete_if_cache`（行 241）是全书最值得逐行读的函数之一，因为它把"一次 LLM 调用要操心的所有事"都摊开了：客户端构造、缓存边界、重试策略、流式、思维链、结构化输出、token 计量、资源清理——八件事压在一个函数里。

下面分段拆解它的每一层职责。

**客户端构造与 OpenAI 兼容生态。** 真正建客户端的是 `create_openai_async_client`（行 137）。它一个函数同时覆盖标准 OpenAI 和 Azure：`use_azure=True` 时走 `AsyncAzureOpenAI`，并把 `model` 替换成 `azure_deployment`（`api_model = azure_deployment if use_azure and azure_deployment else model`）——因为 Azure 用部署名而非模型名寻址。

`base_url` 缺省读 `OPENAI_API_BASE`，这意味着任何"OpenAI 兼容"端点（vLLM、OpenRouter、DashScope、本地代理）都能复用这套驱动，只要改 `base_url`。代码里那行 `default_headers` 里塞 `LightRAG/{__api_version__}` 的 User-Agent、以及对 `DASHSCOPE_WORKSPACE_ID` 的特判，都是适配真实部署踩出来的坑。参数合并的优先级也写得很清楚：`merged_configs` 以 `client_configs` 打底，再叠加 `default_headers`/`api_key`，最后用显式传入的 `base_url`/`timeout` 覆盖——"显式参数 > client_configs > 默认值"。这种分层覆盖让用户既能用环境变量配全局默认，又能在单次调用里临时改写，互不打架。

值得一提的还有可观测性：文件顶部会探测 `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`，两者都配齐时把 `AsyncOpenAI` 换成 `langfuse.openai` 的版本，做到"配了环境变量就自动接入 trace"，否则无感降级到标准客户端，连 import 失败都被 try/except 兜住。

**缓存的位置。** 函数名里的 `if_cache` 暗示了缓存逻辑，但请注意：`openai_complete_if_cache` 自己并不查缓存，它在入口处 `kwargs.pop("hashing_kv", None)` 直接把缓存句柄丢掉。

真正的"命中则不调模型"发生在更上层——LightRAG 在调用 LLM 前先用 prompt 哈希查 `llm_response_cache`（即 `hashing_kv`），命中就根本不会进到这个函数。所以这一层的职责很纯粹：只管"真的要打 API 时怎么打得稳"。这种分层让缓存策略和供应商驱动彻底解耦：换供应商不影响缓存，换缓存后端（第 7 章的 KV 存储）也不影响驱动。`token_tracker` 参数同理，它把每次调用的 prompt/completion token 数累加进一个外部追踪器，无论流式（从最后一个含 `usage` 的 chunk 取）还是非流式（从 `response.usage` 取）都会回报。

**重试：区分"真错"和"假错"。** 函数被 `@retry` 装饰（基于 `tenacity`），`stop_after_attempt(3)` + `wait_exponential`。重试白名单里除了常规的 `RateLimitError`、`APIConnectionError`、`APITimeoutError`，还有两个 LightRAG 自定义异常很关键：`InvalidResponseError`（响应为空/结构非法时主动抛，触发重试）和 `TransientBadRequestError`。

后者是点睛之笔——HTTP 400 通常是"客户端错误，重试无意义"，但 OpenAI 或前置代理偶尔会因请求体在传输中被截断而返回 "could not parse the JSON body"，这种 400 重试就能成功。代码在 `except BadRequestError` 分支里用 `if "could not parse" in str(e).lower()` 这条启发式把这类瞬时 400 重新包成 `TransientBadRequestError` 抛出，让它进重试；其余真正的 400（参数错、内容策略）则 `raise` 快速失败。注释里诚实地承认这是字符串匹配的"脏"启发式，可能误判，但宁可多重试 3 次也要救回常见的瞬时场景。

`InternalServerError`（≥500）也在白名单，覆盖 OpenAI 的 500 与代理的 upstream 错误。

还有一个常被忽视的资源纪律：每个 `except` 分支都先 `await openai_async_client.close()` 再 `raise`，避免在校验密集或配置错误的场景下泄漏 httpx 连接——非流式分支的 `finally`、流式 `inner()` 的 `finally` 也都做同样的兜底关闭。

**流式与 CoT。** 通过 `hasattr(response, "__aiter__")` 判断是否流式，是则返回一个 `inner()` 异步生成器。流式分支里最复杂的是 Chain-of-Thought 处理。

当 `enable_cot=True` 且模型（DeepSeek 风格）只吐 `reasoning_content` 而 `content` 为空时，驱动会自动用 `<think>...</think>` 标签把推理内容包进输出流；一旦正式 `content` 出现就闭合标签停止 CoT。文档里把这套规则列成 6 条状态机：CoT 仅在"正文为空、reasoning 有内容"时启动；正文一旦出现就停止；两者同时出现则忽略 reasoning；流结束若标签未闭合则在 `finally` 块兜底闭合，保证标签永远配对。

非流式分支则把 `<think>{reasoning_content}</think>` 前置到正文。注意当 `response_format` 被设置（结构化输出）时，代码强制 `enable_cot=False`——结构化 JSON 和思维链标签互斥。流式分支里还顺手处理了两个真实兼容性：用 `safe_unicode_decode` 修复带 `\u` 转义的 Unicode，以及对 Azure"首个 chunk 无 choices（内容过滤结果）"的特判跳过。

**结构化输出与去重的废弃路径。** `_validate_openai_response_format` 只接受 dict 形态的 `response_format`（如 `{"type": "json_object"}`），明确拒绝 Pydantic/typed helper，并在文档里说明结构化结果以原始文本从 `message.content` 返回、不在本层做 schema 校验。

`keyword_extraction` / `entity_extraction` 两个旧布尔参数被降级成兼容垫片：仅当用户没显式传 `response_format` 时，才把它们映射成 `{"type": "json_object"}` 并发 `DeprecationWarning`。这是典型的"留一个发布周期的兼容期再删"工程纪律——三家驱动（OpenAI/Ollama/Anthropic）都实现了这套垫片，只是各自处置不同：OpenAI/Ollama 映射成 JSON 模式，Anthropic 因没有 JSON 模式只能警告后忽略。非流式分支还兼容了 `message.parsed`（beta parse 接口的结构化结果），有则直接 `model_dump_json()`。

## embedding：批处理、维度注入与异步前缀

embedding 侧的契约比补全更"重元数据"。看 `openai_embed`（行 896）的装饰器栈：

```python
@wrap_embedding_func_with_attrs(
    embedding_dim=1536, max_token_size=8192,
    model_name="text-embedding-3-small", supports_asymmetric=True,
)
@retry(...)
async def openai_embed(texts, model="text-embedding-3-small", ...): ...
```

`wrap_embedding_func_with_attrs`（`utils.py`）把裸函数包成 `EmbeddingFunc` 实例，把维度等元数据挂到函数对象上。维度为什么必须挂出来？因为向量库要靠 `embedding_dim` 建表、靠 `model_name` 做 workspace 数据隔离（第 7 章）。

`EmbeddingFunc` 还做了几件聪明事：当底层函数签名支持时，自动把 `embedding_dim`、`max_token_size`、`context` 注入进去（用 `inspect.signature` 探测）——所以函数文档反复强调"不要手动传 `embedding_dim`，它由 wrapper 注入"。`supports_asymmetric=True` 表示这个函数能区分 query/document，于是检索时传 `context="query"`、索引时传 `context="document"`，配合 `query_prefix`/`document_prefix` 实现非对称检索（有些模型要求查询和文档加不同前缀）。

`openai_embed` 内部做两件实事。其一，**客户端侧截断**：若 `max_token_size>0`，用 `tiktoken` 编码（带 `_TIKTOKEN_ENCODING_CACHE` 缓存、未知模型回退 `cl100k_base`）逐条裁到上限，避免超长文本直接被 API 拒，并把截断条数 log 出来。

其二，**批量+维度+编码格式**：一次性把整个 `texts` 列表喂给 `embeddings.create`，靠模块级开关 `EMBEDDING_USE_BASE64`（默认 true）决定 `encoding_format` 走 base64 还是 float——base64 在线路上更省，但 Yandex 等不支持，可关。仅当 `embedding_dim` 非空才加 `dimensions` 参数（OpenAI v3 支持的维度裁剪）。返回时统一解成 `np.float32` 数组：base64 用 `np.frombuffer(base64.b64decode(...))`，float 列表用 `np.array(...)`。

不同供应商的默认维度各异，这是工程上必须知道的事实：OpenAI `text-embedding-3-small` 是 1536，Ollama `bge-m3:latest` 是 1024（`ollama_embed`），Gemini `gemini-embedding-001` 是 1536（`gemini_embed`，`max_token_size=2048`）。换 embedding 模型往往意味着维度变化，需要重建向量库——`EmbeddingFunc` 把 `embedding_dim` 当成数据隔离的一部分，正是为了让这种不兼容在建表期就暴露而非运行期才崩。

`azure_openai_embed` 演示了一个容易踩的复用陷阱：它调的是 `openai_embed.func(...)` 而非 `openai_embed(...)`。因为 `openai_embed` 已经是 `EmbeddingFunc` 实例，直接调会触发二次包装（双重注入 `embedding_dim`、参数冲突），`.func` 取到的才是未包装的原函数。

文档里专门画了调用链图来讲清这一点，并强调 `azure_openai_embed` 自身不加 `@retry`——因为底层 `openai_embed.func` 已带重试，再叠一层就成了双重重试。同样的"用 `.func` 避免双重包装"原则，也适用于用户用 `functools.partial(ollama_embed.func, ...)` 预绑模型名的常见写法。

## `binding_options.py`：配置即数据

接入层最体现"框架品味"的是 `binding_options.py`。它的目标是：给每个供应商声明一组可调参数，让这组参数**同时**能从 CLI 参数、环境变量、`.env` 文件、代码实例四种渠道生成，且零重复代码。

朴素做法是为每家每个参数手写一遍 argparse 注册、再手写一遍环境变量读取、再手写一遍文档——三处分散、极易漂移。LightRAG 的做法是把参数当成有类型的数据声明一次，其余形态全部自动派生。

实现手法是 dataclass + 类自省。基类 `BindingOptions`（行 70）只有两个类变量约定：`_binding_name`（供应商前缀）和 `_help`（每个字段的帮助文案）。具体供应商各自继承并声明字段：

- `OpenAILLMOptions`（行 598）把 `temperature`、`top_p`、`frequency_penalty`、`max_completion_tokens`、`reasoning_effort`、`extra_body` 等逐个写成带默认值的 dataclass 字段；
- `OllamaLLMOptions` / `OllamaEmbeddingOptions` 共享一个 `_OllamaOptionsMixin`，里面塞了 `num_ctx`、`num_predict`、`mirostat`、`num_gpu`、`use_mmap` 等三十多个 Ollama 原生参数——LLM 与 embedding 复用同一组字段，只靠 `_binding_name`（`ollama_llm` vs `ollama_embedding`）区分前缀；
- `GeminiLLMOptions` 有 `thinking_config`、`safety_settings`；`GeminiEmbeddingOptions` 则只有一个 `task_type`，对应 Gemini 的非对称 embedding 任务类型；
- `BedrockLLMOptions` 注释里明确警告：字段必须落在 `bedrock.py` 的 Converse API 白名单内，否则会被驱动静默丢弃。

围绕这些字段，基类提供了一组 classmethod 把"声明"翻译成各种形态：

- `args_env_name_type_value()` 用 `dataclasses.fields(cls)` 遍历字段，按规则生成三元命名：CLI 参数名 `{binding}-{field}`（如 `--openai-llm-temperature`）、环境变量名 `{BINDING}_{FIELD}`（如 `OPENAI_LLM_TEMPERATURE`）、类型、默认值、帮助。一处声明，三种名字自动派生。
- `add_args(parser)` 把这些字段批量注册成 argparse 参数，并对 `List[str]`、`dict`、`bool` 三种类型做特殊解析（JSON 数组/对象解析、自定义布尔解析器），默认值用 `get_env_value` 从环境变量取——于是 CLI 和环境变量天然打通，命令行没传就回落到环境变量。
- `generate_dot_env_sample()` 遍历 `cls.__subclasses__()`，把所有子类的所有字段拼成一份带注释的 `.env` 模板。运行 `python -m lightrag.llm.binding_options` 就能一键导出配置样例，新增供应商无需手写文档。
- `options_dict(args)` 反向操作：从解析后的 `Namespace` 里按前缀过滤出属于本供应商的参数、剥掉前缀，得到干净的 kwargs 字典，直接喂给驱动函数。
- `asdict()` 把实例转字典，方便把一组运行时配置整体传下去。

这五个 classmethod 各管一段链路：`args_env_name_type_value` 管命名派生，`add_args` 管 CLI 注册，`generate_dot_env_sample` 管文档导出，`options_dict` 管运行时提取，`asdict` 管实例序列化——它们共享同一份字段声明，互不重复。

这就是"配置即数据"：参数不是散落在 if-else 里的字符串，而是有类型、有默认、有帮助的 dataclass 字段；新增一家供应商只需写一个 dataclass 子类，CLI、环境变量、`.env` 模板、参数提取全部自动具备。

`add_args` 里对 `List[str]`/`dict`/`bool` 三种类型的特殊处理也值得一看：列表和字典走 JSON 解析器（传非法 JSON 会被 argparse 拒绝），布尔走自定义 `bool_parser`（识别 `true/1/yes/t/on`），避免了 Python `bool("false")` 恒为真的经典坑。`_resolve_optional_type` 还会把 `int | None` 这类 Optional 注解还原成具体类型，让 argparse 拿到正确的 `type=`。这种设计的回报在第 11 章会更明显——API 服务的启动参数大量来自这套机制。

## `llm_roles.py`：让每个阶段绑不同模型

到这里所有驱动还都是"一个 LightRAG 配一个 LLM"。`llm_roles.py` 把粒度细化到"一个 LightRAG 给不同任务配不同 LLM"。

它定义了四个角色（`ROLES`，行 52）：`extract`（实体关系抽取）、`keyword`（关键词抽取）、`query`（最终查询应答）、`vlm`（视觉）。设计哲学写在 `RoleSpec` 的注释里：新增角色是"一行编辑"——往 `ROLES` 元组追加一个 `RoleSpec` 即可，其余组件（API 层的环境变量循环、队列观测、热更新流程）全都遍历这个注册表，没有任何地方硬编码角色名。`RoleSpec` 是 `frozen=True` 的不可变描述符，含 `name`（字典键）、`env_prefix`（环境变量前缀）、`queue_name`（日志显示名）三字段；模块还预生成了 `ROLE_NAMES` 与 `ROLES_BY_NAME` 两个查找表供 O(1) 校验。

为什么要分角色？因为各阶段的成本/质量取舍不同：抽取阶段调用量巨大但任务相对机械，适合用便宜快的模型；最终 `query` 直面用户，值得用最强的模型。`RoleLLMConfig` 允许在 `LightRAG()` 初始化时对每个角色单独覆盖 `func`/`kwargs`/`max_async`/`timeout`，留空的字段回落到基础 LLM 设置；`max_async` 还能从 `{ROLE}_MAX_ASYNC_LLM` 环境变量种入。

`binding_options.py` 的 `options_dict_for_role` 与之配套，支持 `EXTRACT_OPENAI_LLM_TEMPERATURE` 这种"角色前缀 + 供应商 + 字段"的环境变量覆盖，并区分同供应商（继承基础参数再叠加覆盖）与跨供应商（从空字典起步）两种语义——这正是 `RoleSpec` 里 `env_prefix` 字段（如 `EXTRACT`）的用武之地。

运行时由 `_RoleLLMMixin`（混入 `LightRAG`）驱动。它本身不持有状态，而是依赖主类在 `__post_init__` 里初始化的 `_role_llm_states`、`_role_llm_builders` 等属性——典型的 mixin 分工：逻辑在 mixin、数据在主类。

核心是 `_wrap_llm_role_func`：每个角色的裸函数都被 `priority_limit_async_func_call` 单独包一层带优先级的并发队列，`concurrency_group=f"llm:{role_name}"`，并用 `functools.partial` 预绑 `hashing_kv=self.llm_response_cache` 和该角色的 `model_kwargs`。

注意一个反直觉的决定：基础 `llm_model_func` **不**被队列包装——因为所有真正触达 LLM 的代码路径都走角色 wrapper，并发控制在角色层统一收口即可。`_get_effective_role_llm_kwargs` / `_get_effective_role_llm_timeout` / `_get_effective_role_llm_max_async` 三个方法实现"角色有则用角色、无则回落基础"的逐字段继承，跨供应商角色还会刻意返回空 kwargs 而非基础 kwargs（避免把 OpenAI 的参数喂给 Anthropic）。

这套设计还支持热更新。`update_llm_role_config` / `aupdate_llm_role_config` 能在运行时改某个角色的模型、binding、api_key 乃至整套 `provider_options`；改 binding/model 这类需要重建客户端的更新，要先 `register_role_llm_builder` 注册一个构建器。

更新走"快照-尝试-失败回滚"模式：`_apply_llm_role_config_update` 里先 `deepcopy` 出 `snapshot`，异常时逐字段还原，保证一次失败的热更新不会把角色置于半损坏状态。被替换掉的旧 wrapper 交给 `_schedule_retired_llm_queue_cleanup` 优雅排空——`aupdate_*` 会等旧队列 `queue.join()` 把已入队的请求跑完再关，上限约束在 `llm_timeout*2+15` 秒（默认约 6 分 15 秒），绝不无限阻塞。

对外暴露的 `get_llm_role_config` 做了安全处理：`_scrubbed_llm_metadata` 把 `api_key`、`secret`、`password` 等带 `_SECRET_MARKERS` 的字段整条剥掉（不是打码，因为 `***` 对外部消费者毫无信息量），保证 `/health`、WebUI、审计输出永不泄密。真正需要原始密钥的组件（角色构建器、provider 客户端）直接从私有的 `_role_llm_states[role].metadata` 读。配套的 `get_llm_queue_status` / `get_embedding_queue_status` / `get_rerank_queue_status` 还能汇报每条队列的积压情况，是第 11 章 `/health` 端点的数据源。

## `rerank.py`：可选的检索后精排

检索召回的候选未必按相关度排好，`rerank.py`（577 行）提供一个可选的精排环节：把 query 和候选文档丢给专门的 rerank 模型重新打分排序。

它同样是函数式契约，核心是 `generic_rerank_api`，对外暴露 `cohere_rerank`、`jina_rerank`、`ali_rerank` 三个便利函数，分别对应 Cohere、Jina、阿里 DashScope 三家。值得注意的是 rerank 用的是裸 `aiohttp` 直连 REST 端点，而不是各家的官方 SDK——因为 rerank 接口足够简单（一个 POST），自己发请求反而省掉了一堆 SDK 依赖。

三家的请求/响应格式不同，`generic_rerank_api` 用 `request_format` / `response_format` 两个开关统一：阿里走嵌套的 `input/parameters` 结构、响应在 `output.results`；Jina/Cohere 走扁平 `query/documents/top_n`、响应在 `results`。最后都归一成 `[{"index": int, "relevance_score": float}, ...]`。这又是一次"吃掉格式碎片"。三个便利函数的差异也都收在参数里：Cohere 不支持 `return_documents`（传 None），Jina 传 `return_documents=False`，三家的 api_key 各自从 `COHERE_API_KEY`/`JINA_API_KEY`/`DASHSCOPE_API_KEY` 回落，并都兜底到统一的 `RERANK_BINDING_API_KEY`。

`cohere_rerank` 的默认 `max_tokens_per_doc=4096`（v3.5 模型上限较高），而 `jina_rerank`/`ali_rerank` 不默认开分块——这些差异都写在便利函数的默认参数里，对调用方透明。

工程细节有两处值得学。其一，**长文档分块重排**：rerank 模型常有 512 token 的输入上限，`enable_chunking=True` 时先用 `chunk_documents_for_rerank` 把超长文档切成带重叠的块（`overlap_tokens` 会被 clamp 到小于 `max_tokens` 以防死循环，tokenizer 初始化失败还会回退到"1 token≈4 字符"的近似切分），重排后再用 `aggregate_chunk_scores` 按 `max`/`mean`/`first` 把块分数聚合回原文档级，并在文档级而非块级施加 `top_n`——保证"top_n 限制的是文档数而非块数"的语义正确。

其二，**错误清洗**：当 rerank 服务返回 502/503/504 且响应体是 HTML 报错页时，把它替换成人类可读的提示（如"Bad Gateway (502) - Rerank service temporarily unavailable"），而不是把整页 HTML 抛给上层。整个函数被 `@retry` 包住，只对 `aiohttp.ClientError`/`ClientResponseError` 系重试。

rerank 函数怎么进入 LightRAG？通过 `rerank_model_func` 字段（`lightrag.py` 行 572）。`__post_init__` 里若它非空，就用 `priority_limit_async_func_call` 包成带队列的 `concurrency_group="rerank"`（行 1052），和 LLM、embedding 一样纳入统一的并发治理。

检索流程在拿到向量召回结果后调它精排，没配则跳过——这正是"可选环节"的含义：rerank 不是 RAG 必需品，而是质量与延迟的权衡按钮。值得对照的是，rerank 队列、各角色 LLM 队列、embedding 队列三者共用同一套 `priority_limit_async_func_call` 基础设施，只是 `concurrency_group` 与 `queue_name` 不同。这种统一让"限流、超时、积压观测"成为整个接入层的横切能力，而不是每家驱动各写一遍。

## 多模态：统一的 `image_inputs` 契约

视觉接入没有单独的"VLM 驱动"，而是把多模态能力缝进既有补全函数。约定是一个统一的 `image_inputs` 关键字参数，由 `_vision_utils.py` 的 `normalize_image_inputs` 归一化。

它接受三种形态：裸 base64 字符串（用魔数 `_detect_mime` 嗅探 MIME，默认 png）、`data:<mime>;base64,<payload>` 的 data URL、带 `base64` 必填键的 dict。归一化产物是 `NormalizedImage` 数据类，连图片像素宽高都从文件头解析出来（纯手写 PNG/JPEG/GIF/WebP 头解析，不依赖 Pillow）。

各驱动再把归一化结果翻成自己的格式：`openai.py` 拼成 `{"type": "image_url", "image_url": {"url": "data:...;base64,..."}}` 的内容块；`ollama.py` 则把 `img.base64_str` 放进 message 的 `images` 列表。同一份 `NormalizedImage`，到了不同驱动手里翻成不同结构，这又是"统一入口、各自翻译"的契约思路在多模态上的复刻。

`_vision_utils.py` 还区分了两种元数据，这是个容易忽略但很关键的设计：`image_cache_metadata` 喂给缓存键，故意不含 `source_id`/`source_file`，让同一张图在不同文件名下仍命中同一缓存；`image_audit_metadata` 是人类可读审计，含来源指针但绝不含原始 base64。两者都带上从文件头解析出的 `width`/`height`，让诊断能一行看清"发了什么"而无需重新解码。这种"统一入口 + 各驱动各自翻译 + 缓存与审计分离"的设计，让 `vlm` 角色能复用整套补全与缓存基础设施，而不必新建一条多模态专属链路。

## 本章小结

- LightRAG 的 LLM 接入是"函数式契约 + 平铺驱动"：`llm/__init__.py` 是空的，没有继承体系，每个供应商一个模块文件、几个自由函数，符合 `xxx_complete_if_cache` / `xxx_embed` 签名即可被 `llm_model_func` / `embedding_func` 接纳。
- 内置覆盖 openai/azure_openai/ollama/anthropic/bedrock/gemini/zhipu/hf/jina/nvidia_openai/voyageai/lollms/lmdeploy/llama_index 等十余家；`create_openai_async_client` 靠 `base_url` 让所有"OpenAI 兼容"端点复用同一驱动。
- `openai_complete_if_cache` 是补全核心：缓存查在更上层（本函数 `pop("hashing_kv")` 直接丢弃句柄），它专注于稳：`tenacity` 三次重试，并用自定义 `InvalidResponseError` / `TransientBadRequestError` 把"假 400"救回重试、真 400 快速失败。
- 流式与 CoT 是一套状态机：`enable_cot` 下把 DeepSeek 风格 `reasoning_content` 用 `<think>...</think>` 包入输出，`finally` 兜底闭合；结构化 `response_format` 与 CoT 互斥。
- embedding 侧靠 `wrap_embedding_func_with_attrs` 挂维度元数据，自动注入 `embedding_dim`/`context`，支持非对称前缀、客户端 tiktoken 截断、批量与 base64 编码格式开关；换模型常伴随维度变化需重建库。
- `binding_options.py` 用 dataclass 把每家参数声明成"配置即数据"，一处声明自动派生 CLI 参数、环境变量、`.env` 模板与 kwargs 提取，新增供应商只写一个子类。
- `llm_roles.py` 把 `extract`/`keyword`/`query`/`vlm` 四角色各绑一套模型与独立优先级队列，支持环境变量角色覆盖、运行时热更新（快照回滚 + 旧队列优雅排空）、密钥剥离的观测快照。
- `rerank.py` 是可选精排，`generic_rerank_api` 用 request/response 格式开关统一 Cohere/Jina/阿里，含长文档分块-聚合重排和 HTML 错误清洗，经 `rerank_model_func` 纳入统一并发队列。
- 多模态用统一 `image_inputs` 契约，`_vision_utils.py` 归一化为 `NormalizedImage` 并区分缓存键元数据与审计元数据，各驱动各自翻成自家内容块格式。

## 动手实验

1. **实验一：追踪一次补全的重试白名单** — 打开 `openai.py`，对照 `openai_complete_if_cache` 的 `@retry` 装饰器（行 226）与下方 `except BadRequestError` 分支（行 460），列出哪些异常会触发重试、哪些会快速失败；手动构造一个含 "could not parse" 的 400 文案，verify 它会被包成 `TransientBadRequestError`，再换一段普通 400 文案确认它直接 `raise`。
2. **实验二：用 dataclass 生成 .env 模板** — 在 `lightrag/llm/` 目录运行 `python -m lightrag.llm.binding_options`，观察输出的 `.env` 样例；再读 `args_env_name_type_value()`，对 `OpenAILLMOptions` 的 `temperature` 字段，手算出它派生的 CLI 参数名与环境变量名，和输出比对验证命名规则。
3. **实验三：给抽取与查询配不同模型** — 阅读 `llm_roles.py` 的 `ROLES`、`RoleLLMConfig` 与 `_wrap_llm_role_func`，写出一段 `LightRAG(..., role_llm_configs={"extract": RoleLLMConfig(func=gpt_4o_mini_complete), "query": RoleLLMConfig(func=gpt_4o_complete)})` 的初始化代码，并说明为什么基础 `llm_model_func` 不被队列包装而角色函数被包装。
4. **实验四：长文档重排的分块-聚合** — 在 `rerank.py` 中跟读 `chunk_documents_for_rerank` → `generic_rerank_api`（`enable_chunking=True` 分支）→ `aggregate_chunk_scores` 的数据流；构造一篇超过 480 token 的文档加几篇短文档，手推 `doc_indices` 映射，解释为什么 `top_n` 要在聚合后的文档级而非块级施加。

> **下一章预告**：模型已经能稳定接入，但 LightRAG 终究要以服务形态对外。第 11 章进入 `api/`，解构 FastAPI 服务的启动、路由分层、文档与查询端点，以及本章 `binding_options` 与 `llm_roles` 如何在 API 启动参数里落地成一个可运维的 RAG 服务器。
