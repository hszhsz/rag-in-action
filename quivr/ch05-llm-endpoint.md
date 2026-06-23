# 第 5 章 LLMEndpoint 与多供应商模型配置

在前几章中我们看到，Quivr 把检索增强生成（RAG）拆成了一条可配置的流水线。流水线里最核心、也最容易被各种供应商差异污染的环节，就是「调用大模型」这一步。OpenAI、Anthropic、Mistral、Gemini、Groq 各家的 SDK 参数命名不同、上下文窗口不同、密钥环境变量不同，甚至连「能不能传 `temperature`」都各有脾气。Quivr 的做法是用一个统一的抽象——`LLMEndpoint`——把这些差异全部收纳进来，对上层只暴露一个干净的接口。

本章会钻进 `quivr_core/llm/llm_endpoint.py` 与 `quivr_core/rag/entities/config.py` 两个文件，逐层拆解 Quivr 是如何用「配置即数据」「约定优于配置」「约束只能收紧」「实例与 tokenizer 双层缓存」这几条原则，把七家供应商的模型统一到同一套调用模型之下的。读完你应该能回答：一个 `gpt-4o-2024-08-06` 这样的带日期后缀的模型名，是怎么被识别成 OpenAI 供应商、自动拿到正确上下文窗口、配上对应分词器，并最终实例化成一个 LangChain ChatModel 的。

## LLMEndpointConfig：一份带自检的配置对象

一切从 `config.py` 里的 `LLMEndpointConfig` 开始。它继承自 `QuivrBaseConfig`，本质是一个 Pydantic 模型，字段携带了一组「开箱即用」的默认值：

```python
class LLMEndpointConfig(QuivrBaseConfig):
    supplier: DefaultModelSuppliers = DefaultModelSuppliers.OPENAI
    model: str = "gpt-4o"
    tokenizer_hub: str | None = None
    llm_base_url: str | None = None
    env_variable_name: str | None = None
    llm_api_key: str | None = None
    max_context_tokens: int = 20000
    max_output_tokens: int = 4096
    temperature: float = 0.3
    streaming: bool = True
    prompt: BasePromptTemplate | None = None

    _FALLBACK_TOKENIZER = "cl100k_base"
```

也就是说，不传任何参数时，Quivr 默认用 OpenAI 的 `gpt-4o`，上下文上限 `20000`、输出上限 `4096`、温度 `0.3`、开启流式。`_FALLBACK_TOKENIZER` 是一个类级常量，值为 `"cl100k_base"`——这是 tiktoken 的一种通用编码，当一切「精确」分词器都加载失败时退到这里兜底 [[tiktoken encodings]](https://github.com/openai/tiktoken)。

这个配置类的关键不在字段本身，而在它的 `__init__`：

```python
def __init__(self, **data):
    super().__init__(**data)
    self.set_llm_model_config()
    self.set_api_key()
```

构造时它做了两件「副作用」：先 `set_llm_model_config()` 根据 supplier/model 自动校正上下文与输出上限、并填好分词器；再 `set_api_key()` 按约定从环境变量里把密钥读进来。换句话说，`LLMEndpointConfig` 不是一个被动的数据袋，而是一个会在出生时自我体检、自我补全的对象。下面分别看这两步。

另外注意它显式实现了 `__hash__`，把 supplier、model、各种 token 上限、温度、流式、甚至 `repr(self.prompt)` 全都纳入哈希。这个哈希值后面会成为 `LLMEndpoint.from_config` 实例缓存的钥匙，所以这里它需要是稳定且能反映「配置是否等价」的。

## DefaultModelSuppliers：七家供应商的统一枚举

多供应商支持的第一块基石是枚举 `DefaultModelSuppliers`：

```python
class DefaultModelSuppliers(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    META = "meta"
    MISTRAL = "mistral"
    GROQ = "groq"
    GEMINI = "gemini"
```

它继承 `str`，所以每个成员既是枚举又能直接当字符串用——这在后面拼接环境变量名时很方便（可以直接对成员做正则处理）。七家供应商覆盖了主流闭源与开源托管服务。注意 `AZURE` 与 `OPENAI` 是分开的两条线：Azure OpenAI 的端点形态、部署名、API 版本都需要从 URL 里解析，与原生 OpenAI 走的是不同的实例化分支。

## LLMModelConfig：模型元数据表与 startswith 匹配

供应商有了，每个供应商旗下又有许多具体模型，每个模型的上下文窗口、输出上限、分词器都不一样。Quivr 把这些「模型元数据」集中写在 `LLMModelConfig._model_defaults` 这张大字典里，结构是 `supplier -> {model_name -> LLMConfig}`。其中 `LLMConfig` 只有三个字段：

```python
class LLMConfig(QuivrBaseConfig):
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    tokenizer_hub: str | None = None
```

举几个表中的真实条目：OpenAI 的 `gpt-4o` 配 `max_context_tokens=128000, max_output_tokens=16384, tokenizer_hub="Quivr/gpt-4o"`；`gpt-4.1` 配的上下文高达 `1047576`；Anthropic 的 `claude-3-5-sonnet` 配 `max_context_tokens=200000, max_output_tokens=8192, tokenizer_hub="Quivr/claude-tokenizer"` [[Anthropic models overview]](https://docs.anthropic.com/en/docs/about-claude/models)。这些数字都来自各家官方文档，被「固化」成了仓库内的常量表，从而把「外部世界的事实」变成了「代码内可静态查询的数据」。

真正巧妙的是查询逻辑 `get_llm_model_config` 与 `get_supplier_by_model_name`，它们都用 `startswith` 而不是精确相等来匹配：

```python
@classmethod
def get_llm_model_config(cls, supplier, model_name):
    supplier_defaults = cls._model_defaults.get(supplier)
    if not supplier_defaults:
        return None
    for key, config in supplier_defaults.items():
        if model_name.startswith(key):
            return config
    return None
```

为什么是 `startswith`？因为各家供应商频繁给模型名追加日期或版本后缀，例如 `gpt-4o-2024-08-06`、`claude-3-5-sonnet-20241022`。如果用精确相等，每出一个快照就得改表；用前缀匹配，只要表里有 `gpt-4o` 这个基名，所有 `gpt-4o-*` 的快照都能命中同一份元数据。`get_supplier_by_model_name` 同理，它遍历所有供应商的所有基名，一旦 `model.startswith(base_model_name)` 命中就返回对应供应商——这让 `set_llm_model(model)` 这种「只给模型名、自动推断供应商」的便捷用法成为可能。需要留意：前缀匹配依赖字典遍历顺序，因此更具体的基名（如 `gpt-4o-mini`）应排在更短的基名（如 `gpt-4o`）之前，源码中的表正是这么排列的。

## set_api_key：约定优于配置的密钥读取

`set_api_key` 体现了「约定优于配置」与「密钥永不进代码」两条原则：

```python
def set_api_key(self, force_reset=False):
    if not self.supplier:
        return
    if force_reset or not self.env_variable_name:
        self.env_variable_name = (
            f"{normalize_to_env_variable_name(self.supplier)}_API_KEY"
        )
    if not self.llm_api_key or force_reset:
        self.llm_api_key = os.getenv(self.env_variable_name)
    if not self.llm_api_key:
        logger.warning(f"The API key for supplier '{self.supplier}' is not set. ")
        logger.warning(f"Please set the environment variable: '{self.env_variable_name}'. ")
```

它不会要求用户在代码里硬写密钥，而是按约定推导出环境变量名 `{SUPPLIER}_API_KEY`，再用 `os.getenv` 读取。供应商 `openai` 对应 `OPENAI_API_KEY`，`anthropic` 对应 `ANTHROPIC_API_KEY`，以此类推。即便没读到，也只是 `logger.warning` 提示「请设置某某环境变量」，而不是抛错——配置阶段保持宽容，真正调用时才会因缺密钥失败。

供应商名到环境变量名的转换交给了模块级函数 `normalize_to_env_variable_name`：

```python
def normalize_to_env_variable_name(name: str) -> str:
    env_variable_name = re.sub(r"[^A-Za-z0-9_]", "_", name).upper()
    if env_variable_name[0].isdigit():
        raise ValueError(
            f"Invalid environment variable name '{env_variable_name}': Cannot start with a digit."
        )
    return env_variable_name
```

它用正则把所有非「字母、数字、下划线」的字符替换成下划线，再整体大写，并且禁止以数字开头——这保证了任何供应商名都能被安全地转成一个合法的 shell 环境变量名。这同一个函数也被 `RerankerConfig` 复用来推导重排器的密钥变量名，是个小而通用的工具。

## set_llm_model_config：约束只能收紧，错误即文档

`set_llm_model_config` 是配置自检里最有「设计哲学」的一段。它先用前面的 `get_llm_model_config` 拿到模型元数据，然后把用户填的 `max_context_tokens` / `max_output_tokens` 往下夹紧到模型真实上限之内：

```python
_max_context_tokens = (
    llm_model_config.max_context_tokens - llm_model_config.max_output_tokens
    if llm_model_config.max_output_tokens
    else llm_model_config.max_context_tokens
)
if self.max_context_tokens > _max_context_tokens:
    logger.warning(f"Lowering max_context_tokens from {self.max_context_tokens} to {_max_context_tokens}")
    self.max_context_tokens = _max_context_tokens
if self.max_context_tokens < MIN_CONTEXT_TOKENS:
    logger.error(...)
    raise ValueError(f"max_context_tokens is too low: {self.max_context_tokens}. ")
```

这里有两个值得注意的细节。第一，可用上下文是「模型上下文窗口减去输出预留」，因为生成的 token 也要占窗口，所以 Quivr 主动从总窗口里扣掉 `max_output_tokens` 再作为真正可填充上下文的上限。第二，这套逻辑**只会把用户的值往下调，永不上调**——这就是「约束只能收紧」：用户可以请求比模型上限更小的窗口，但永远不会因为配置而超出模型物理能力。

而当收紧后的值跌破下限时，配置直接抛 `ValueError`。下限是两个模块级常量：

```python
MIN_CONTEXT_TOKENS = 4096
MIN_OUTPUT_TOKENS = 4096
```

输出上限走同样的逻辑：超过模型上限则下调，低于 `MIN_OUTPUT_TOKENS` 则抛错。这是一种「错误即文档」的设计——与其让一个上下文只有几百 token 的畸形配置悄悄跑下去、最后在生成阶段莫名其妙地截断或报错，不如在构造配置的那一刻就用清晰的 `ValueError` 告诉调用者「你这个窗口太小了」。最后这段还顺手把 `self.tokenizer_hub` 设成元数据里的 `tokenizer_hub`，为下一步加载分词器埋好伏笔。

## LLMTokenizer：带 LRU 淘汰的分词器缓存

分词器（tokenizer）是 RAG 里高频使用的部件——每次估算上下文长度、决定能塞多少 chunk，都要 `count_tokens`。而从 HuggingFace 加载一个分词器既慢又占内存，因此 `llm_endpoint.py` 里的 `LLMTokenizer` 实现了一套类级别的缓存：

```python
class LLMTokenizer:
    _cache: dict[int, tuple["LLMTokenizer", int, float]] = {}
    _max_cache_size_mb: int = 50
    _max_cache_count: int = 5
    _current_cache_size: int = 0
    _default_size: int = 5 * 1024 * 1024
```

缓存的值是三元组 `(tokenizer, size_bytes, last_access_time)`，并由两个上限共同约束：总体积不超过 `_max_cache_size_mb=50` MB，数量不超过 `_max_cache_count=5`。淘汰策略是 LRU——`load` 方法在插入新实例前，会循环检查是否超限，超限就用 `min(... key=last_access_time)` 找出最久未访问的项弹出：

```python
while (cls._current_cache_size + instance._size_bytes > cls._max_cache_size_mb * 1024 * 1024
       or len(cls._cache) >= cls._max_cache_count):
    oldest_key = min(cls._cache.keys(), key=lambda k: cls._cache[k][2])
    _, removed_size, _ = cls._cache.pop(oldest_key)
    cls._current_cache_size -= removed_size
```

命中缓存时它还会刷新 `last_access_time`，让 LRU 顺序保持准确。这样设计的理由很直接：分词器加载昂贵、但跨多个 `LLMEndpoint` 实例高度可复用，缓存住既省时又省内存，而上限与淘汰保证它不会无限膨胀。

实例化分词器时也有分流逻辑。若 `tokenizer_hub` 存在，就先把 `TOKENIZERS_PARALLELISM` 环境变量设为 `false`（避免 fork 后的并行死锁警告），再分两路加载：模型名里含 `text-embedding-ada-002` 的用 `GPT2TokenizerFast`，否则用通用的 `AutoTokenizer` 从 HuggingFace Hub 拉取 [[Hugging Face Tokenizers]](https://huggingface.co/docs/transformers/main_classes/tokenizer)：

```python
if "text-embedding-ada-002" in self.tokenizer_hub:
    from transformers import GPT2TokenizerFast
    self.tokenizer = GPT2TokenizerFast.from_pretrained(self.tokenizer_hub)
else:
    from transformers import AutoTokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_hub)
```

加载若抛 `OSError`（连不上 HuggingFace 或本地无缓存），它会 `logger.warning` 后回退到 `tiktoken.get_encoding(self.fallback_tokenizer)`，也就是前面那个 `cl100k_base`。如果压根没有 `tokenizer_hub`，则直接用 tiktoken。换言之，分词器加载是「精确优先、永不致命」：拿不到精确分词器就退到一个够用的通用编码，绝不让分词这件事把整个流程拖崩。

类方法 `preload_tokenizers` 提供了预热能力。给定一组模型名，它遍历 `LLMModelConfig._model_defaults`，用 `startswith` 找到匹配模型的 `tokenizer_hub`，去重后逐个 `cls.load` 进缓存；不给模型名则预热表里全部分词器。这适合在服务启动阶段把热点分词器提前装好，避免首个请求承担加载延迟。

## LLMEndpoint.from_config：实例缓存与按供应商分派

最后来到核心类 `LLMEndpoint`。它的工厂方法 `from_config` 做两件事：实例级缓存 + 按供应商分派创建 LangChain ChatModel。

缓存这块沿用了 `LLMEndpointConfig.__hash__`：

```python
@classmethod
def from_config(cls, config=LLMEndpointConfig()):
    hashed_config = hash(config)
    if hashed_config in cls._cache:
        return cls._cache[hashed_config]
    ...
    instance = cls(llm=_llm, llm_config=config)
    cls._cache[hashed_config] = instance
    return instance
```

相同配置只会构建一次底层 ChatModel，之后命中 `cls._cache` 直接复用。这与 `LLMTokenizer` 的缓存构成了「双层缓存」：上层缓存整个 endpoint 实例，下层缓存分词器。

分派部分则是一长串 `if config.supplier == ...` 分支，每个供应商对应一个 LangChain ChatModel 类，并用 `pydantic.SecretStr` 包裹密钥以避免日志泄露 [[LangChain chat models]](https://python.langchain.com/docs/integrations/chat/)：

- `AZURE` → `AzureChatOpenAI`，需要从 `llm_base_url` 解析出 `deployment`（`parsed_url.path.split("/")[3]`）和 `api-version`（从 query string 取），再拼出 `azure_endpoint`；
- `ANTHROPIC` → `ChatAnthropic`，用 `max_tokens_to_sample` 表示输出上限，并 `assert config.llm_api_key`；
- `OPENAI` → `ChatOpenAI`，用 `max_completion_tokens`；
- `MISTRAL` → `ChatMistralAI`；`GEMINI` → `ChatGoogleGenerativeAI`；`GROQ` → `ChatGroq`；
- 兜底 `else` 分支也走 `ChatOpenAI`，让任何未显式列出的「OpenAI 兼容」端点仍可工作。

其中最值得一提的是 OpenAI 分支对 `o` 系列推理模型的特判：

```python
_llm = ChatOpenAI(
    model=config.model,
    api_key=SecretStr(config.llm_api_key) if config.llm_api_key else None,
    base_url=config.llm_base_url,
    max_completion_tokens=config.max_output_tokens,
    temperature=config.temperature if not config.model.startswith("o") else None,
)
```

当模型名以 `"o"` 开头（即 `o1`、`o3-mini`、`o4-mini` 这类推理模型）时，`temperature` 传 `None`——因为这些模型不接受自定义温度，硬传会被 API 拒绝。这又是一处用 `startswith` 抹平供应商差异的细节。整段被包在 `try ... except ImportError` 里：缺少某家 SDK 时抛出友好的提示，引导用户安装 `quivr-core['base']`。

构造完底层 `_llm` 后，`LLMEndpoint.__init__` 还会算一次 `_supports_func_calling = model_supports_function_calling(self._config.model)`。这个函数在 `rag/utils.py` 里非常朴素：

```python
def model_supports_function_calling(model_name: str):
    models_not_supporting_function_calls = ["llama2", "test", "ollama3"]
    return model_name not in models_not_supporting_function_calls
```

采用「黑名单」策略——默认认为模型支持函数调用，只把已知不支持的几个名字排除。结果通过 `supports_func_calling()` 暴露，并进入 `info()` 返回的 `LLMInfo`（定义在 `brain/info.py`，含 `model`、`llm_base_url`、`temperature`、`max_tokens`、`supports_function_calling` 五个字段，可被渲染进 `rich` 的 Brain 信息树）。最后 `clone_llm()` 用 `self._llm.__class__(**self._llm.__dict__)` 复制出一个同配置的底层模型，供需要独立实例的场景使用。

## 本章小结

- `LLMEndpointConfig` 是一份会自检的 Pydantic 配置：构造时自动调用 `set_llm_model_config()` 与 `set_api_key()`，默认 `supplier=OPENAI`、`model="gpt-4o"`、`max_context_tokens=20000`、`max_output_tokens=4096`、`temperature=0.3`、`streaming=True`，并以 `_FALLBACK_TOKENIZER="cl100k_base"` 兜底。
- `DefaultModelSuppliers` 枚举统一了七家供应商（OPENAI/AZURE/ANTHROPIC/META/MISTRAL/GROQ/GEMINI），继承 `str` 便于直接参与字符串处理。
- `LLMModelConfig._model_defaults` 把各模型的上下文窗口、输出上限、`tokenizer_hub` 固化成数据表，`get_llm_model_config` 与 `get_supplier_by_model_name` 都用 `startswith` 匹配，从而兼容带日期/版本后缀的模型快照名。
- `set_api_key` 遵循「约定优于配置」：用 `normalize_to_env_variable_name` 把供应商名规整成 `{SUPPLIER}_API_KEY` 环境变量，再由 `os.getenv` 读取，密钥永不进代码，缺失只警告不报错。
- `set_llm_model_config` 体现「约束只能收紧」：把用户的 token 上限往下夹到模型真实上限（且上下文要扣掉输出预留），低于 `MIN_CONTEXT_TOKENS=4096` / `MIN_OUTPUT_TOKENS=4096` 时直接抛 `ValueError`，做到「错误即文档」。
- `LLMTokenizer` 用类级 LRU 缓存（上限 `_max_cache_size_mb=50` MB、`_max_cache_count=5`，按 `last_access_time` 淘汰）复用昂贵的分词器，并提供 `preload_tokenizers` 预热。
- 分词器加载「精确优先、永不致命」：有 `tokenizer_hub` 时从 HuggingFace 加载（ada-002 用 `GPT2TokenizerFast`，其余用 `AutoTokenizer`），失败回退 tiktoken；无 hub 则直接用 tiktoken。
- `LLMEndpoint.from_config` 以 `hash(config)` 做实例缓存，与分词器缓存构成双层缓存；按 supplier 分派创建对应 LangChain ChatModel，用 `SecretStr` 包裹密钥。
- OpenAI 的 `o` 系列推理模型不传 `temperature`（传 `None`）；Azure 需从 `llm_base_url` 解析 deployment 与 api-version；`else` 兜底走 `ChatOpenAI` 以兼容 OpenAI 兼容端点。
- `model_supports_function_calling` 采用黑名单策略，结果经 `supports_func_calling()`、`info()`（返回 `LLMInfo`）对外暴露；`clone_llm()` 可复制同配置的底层模型。

## 动手实验

1. **实验一：观察环境变量名推导** — 在 Python 中分别用 `supplier=DefaultModelSuppliers.ANTHROPIC`、`GEMINI`、`GROQ` 构造 `LLMEndpointConfig`，打印每个实例的 `env_variable_name`，确认它们分别是 `ANTHROPIC_API_KEY`、`GEMINI_API_KEY`、`GROQ_API_KEY`；再单独调用 `normalize_to_env_variable_name("azure-openai")` 看非字母数字字符被替换成下划线、整体大写的效果。
2. **实验二：触发上下文下限 ValueError** — 构造 `LLMEndpointConfig(model="gpt-4o", max_context_tokens=2000)`，观察构造时因 `set_llm_model_config` 把值收紧后跌破 `MIN_CONTEXT_TOKENS=4096` 而抛出的 `ValueError`；再把值改成 `8000` 重试，对比日志里 `Lowering max_context_tokens` 的下调提示，体会「约束只能收紧」。
3. **实验三：验证 from_config 实例缓存命中** — 用完全相同的参数调用两次 `LLMEndpoint.from_config(config)`，用 `is` 比较两次返回对象是否为同一引用（应为 `True`）；再修改 `temperature` 后第三次调用，确认因 `hash(config)` 改变而得到新实例，从而理解缓存键的构成。
4. **实验四：观察 startswith 模型匹配** — 调用 `LLMModelConfig.get_supplier_by_model_name("gpt-4o-2024-08-06")` 与 `get_llm_model_config(DefaultModelSuppliers.OPENAI, "gpt-4o-mini-2024-07-18")`，确认带日期后缀的模型仍能命中基名 `gpt-4o` / `gpt-4o-mini` 的元数据；再传 `"o3-mini"` 走一遍 `from_config`，验证 OpenAI 分支因模型名以 `o` 开头而把 `temperature` 置为 `None`。

> **下一章预告**：模型只是 RAG 的「嘴」，真正决定它「读到什么」的是检索。下一章我们将转向嵌入与向量库，拆解 Quivr 如何把文档切块、调用 Embedding 模型生成向量、并在向量库中完成相似度检索，看「上下文窗口」之外那条把知识喂进模型的管道是怎样铺成的。
