# 第 2 章 全局配置 Settings 与依赖解析

任何一个 RAG 框架,只要稍微用得深一点,都会遇到同一个尴尬:LLM、嵌入模型、切分器、回调管理器这些"基础设施级"对象,几乎每个组件都要用到。索引构建要用嵌入模型,检索后处理要用 LLM,节点解析要用切分器,几乎没有哪个环节能完全独善其身。如果让用户在每次构造 index、retriever、query engine 时都把这一长串依赖手动传进去,代码会迅速退化成一堆样板参数;而如果框架在内部到处硬编码 `OpenAI()`,又会让"换一个模型"变成一场跨越几十个文件的搜索替换。

LlamaIndex 给出的答案是一个看似朴素、实则设计得相当克制的全局配置对象:`Settings`。它定义在 `llama-index-core/llama_index/core/settings.py`,本质是一个 `dataclass` 加上一组带懒加载语义的 property,再以模块级单例的形式暴露给整个框架。本章我们就通读这份不到 300 行的源码,理解"全局默认"如何与"局部覆盖"统一在同一套依赖注入约定之下。

## 为什么需要全局 Settings:一个对象消解到处传参

`settings.py` 的核心是一个名为 `_Settings` 的 dataclass,文件末尾用一行 `Settings = _Settings()` 把它实例化成模块级单例。注意命名约定:类名带下划线前缀 `_Settings` 表示它是实现细节,真正供外部使用的是那个已经实例化好的 `Settings` 对象。这意味着整个 Python 进程里只有一份配置状态,任何 `from llama_index.core import Settings` 拿到的都是同一个引用。

dataclass 里声明的字段全部以下划线开头并默认为 `None`:

```python
@dataclass
class _Settings:
    """Settings for the Llama Index, lazily initialized."""

    # lazy initialization
    _llm: Optional[LLM] = None
    _embed_model: Optional[BaseEmbedding] = None
    _callback_manager: Optional[CallbackManager] = None
    _tokenizer: Optional[Callable[[str], List[Any]]] = None
    _node_parser: Optional[NodeParser] = None
    _prompt_helper: Optional[PromptHelper] = None
    _chat_prompt_helper: Optional[ChatPromptHelper] = None
    _transformations: Optional[List[TransformComponent]] = None
```

这里有两个值得注意的设计决定。第一,所有真实状态都藏在 `_` 前缀的私有字段里,对外暴露的是同名去掉下划线的 property(`llm`、`embed_model`、`node_parser` 等),读写都要经过 getter/setter。第二,初始值一律是 `None`,而不是直接 `_llm: LLM = OpenAI()` 之类。第二点正是后文要讲的懒加载机制的物理基础——只要用户没碰过,这些字段就一直是空的。

用户的典型用法因此变得极简:在程序启动处写一句 `Settings.llm = OpenAI(model="gpt-4o")`、`Settings.embed_model = ...`,之后构造各种 index、engine 时什么都不用传,组件自己会去 `Settings` 里取默认值。这就把"到处传参"压缩成了"一处配置、处处生效"。

## 懒加载机制:用到才 resolve

`_llm` 默认是 `None`,那么第一次有人访问 `Settings.llm` 时会发生什么?看 getter:

```python
@property
def llm(self) -> LLM:
    """Get the LLM."""
    if self._llm is None:
        self._llm = resolve_llm("default")

    if self._callback_manager is not None:
        self._llm.callback_manager = self._callback_manager

    return self._llm
```

逻辑非常清楚:`_llm` 为空时,才调用 `resolve_llm("default")` 去构造一个默认 LLM 并缓存到 `_llm`;之后每次访问直接复用缓存。这就是懒加载(lazy initialization)的精髓——构造默认 LLM 这种"重"操作(它要 import OpenAI 包、读环境变量、甚至校验 API key)被推迟到真正第一次需要时才执行。如果用户在访问前已经通过 setter 设过 `_llm`,那么 `resolve_llm("default")` 这条分支永远不会触发,默认值的构造代价就被完全省掉了。

`embed_model` 的 getter 是同一套模板,空则 `resolve_embed_model("default")`;`node_parser` 的 getter 也类似,只是默认值不是靠 resolve 函数而是直接 `SentenceSplitter()`:

```python
@property
def node_parser(self) -> NodeParser:
    """Get the node parser."""
    if self._node_parser is None:
        self._node_parser = SentenceSplitter()

    if self._callback_manager is not None:
        self._node_parser.callback_manager = self._callback_manager

    return self._node_parser
```

`callback_manager` 的懒加载更简单:空则 `CallbackManager()`(一个空回调管理器),不需要 resolve。`transformations` 的默认值则巧妙地复用了 node parser——它的 getter 在 `_transformations` 为空时返回 `[self.node_parser]`,把切分器本身当作转换管线的唯一一步,这与第 4、5 章要讲的 IngestionPipeline 把 node parser 当作一个 `TransformComponent` 的设计是一脉相承的。

懒加载还有一个不太显眼但很务实的好处:它让"只用一部分功能"的用户不必为没用到的依赖付费。一个只做纯向量检索、根本不调 LLM 的脚本,即使没装 `llama-index-llms-openai`,只要它从不访问 `Settings.llm`,就永远不会触发那条要求安装 OpenAI 包的导入,也就不会报错。配置对象的"惰性"在这里直接转化为依赖的"按需"。

## resolve_* 归一化:字符串、对象、None 收敛成实例

setter 端的设计同样克制。`llm` 的 setter 不是简单地 `self._llm = llm`,而是经过一道 resolve:

```python
@llm.setter
def llm(self, llm: LLMType) -> None:
    """Set the LLM."""
    self._llm = resolve_llm(llm)
```

这里的关键类型是 `LLMType`,定义在 `llama-index-core/llama_index/core/llms/utils.py`:`LLMType = Union[str, LLM, "BaseLanguageModel"]`。也就是说用户既可以传一个字符串(`"default"`、`"local"`、`"local:/path/to/model"`),也可以传一个现成的 LlamaIndex `LLM` 实例,甚至传一个 LangChain 的 `BaseLanguageModel`。`resolve_llm` 的职责就是把这三类异构输入统统归一化成一个标准的 `LLM` 实例。

`resolve_llm` 的分支结构值得逐段看清楚。首先是 `"default"` 这个魔法字符串:

```python
if llm == "default":
    # if testing return mock llm
    if os.getenv("IS_TESTING"):
        from llama_index.core.llms.mock import MockLLM
        llm = MockLLM()
        llm.callback_manager = callback_manager or Settings.callback_manager
        return llm

    # return default OpenAI model. If it fails, return LlamaCPP
    try:
        from llama_index.llms.openai import OpenAI  # pants: no-infer-dep
        from llama_index.llms.openai.utils import (
            validate_openai_api_key,
        )  # pants: no-infer-dep

        llm = OpenAI()
        validate_openai_api_key(llm.api_key)  # type: ignore
    except ImportError:
        raise ImportError(
            "`llama-index-llms-openai` package not found, "
            "please run `pip install llama-index-llms-openai`"
        )
    except ValueError as e:
        raise ValueError(
            "\n******\n"
            "Could not load OpenAI model. "
            "If you intended to use OpenAI, please check your OPENAI_API_KEY.\n"
            ...
        )
```

这段逻辑揭示了 LlamaIndex 的若干隐含约定。第一,测试环境优先:只要设了环境变量 `IS_TESTING`,`"default"` 就解析为 `MockLLM`,这让整个测试套件不必真的去调用 OpenAI。第二,生产默认就是 OpenAI:`"default"` 在非测试场景下等价于 `OpenAI()`,而且紧接着调用 `validate_openai_api_key` 校验密钥。第三,失败时给出可操作的报错:没装 `llama-index-llms-openai` 包时抛 `ImportError` 并附带精确的 `pip install` 命令;密钥无效时抛 `ValueError` 并提示检查 `OPENAI_API_KEY`。注意源码注释里写着 "If it fails, return LlamaCPP",但当前实现里 `except` 分支其实是直接 raise——这条注释与代码已经轻微脱节,是阅读源码时需要以代码为准的一个小例子。

接着是字符串前缀的处理。如果传入的是普通字符串(非 `"default"`),`resolve_llm` 会按冒号切分:

```python
if isinstance(llm, str):
    splits = llm.split(":", 1)
    is_local = splits[0]
    model_path = splits[1] if len(splits) > 1 else None
    if is_local != "local":
        raise ValueError(
            "llm must start with str 'local' or of type LLM or BaseLanguageModel"
        )
    ...
    llm = LlamaCPP(
        model_path=model_path,
        ...
        model_kwargs={"n_gpu_layers": 1},
    )
```

只有以 `local` 开头的字符串是合法的,`local:` 后面的部分会被当作本地模型路径喂给 `LlamaCPP`。任何其他前缀都会得到一句明确的报错:必须以 `'local'` 开头,或者直接传 `LLM`/`BaseLanguageModel` 实例。再往后,如果传入的是 LangChain 的 `BaseLanguageModel`,会被包进 `LangChainLLM` 适配器;如果传入 `None`,则打印 "LLM is explicitly disabled. Using MockLLM." 并退回到 `MockLLM`——也就是说显式传 `None` 是"我不要 LLM"的语义,与字段默认的 `None`(意味着"还没决定,用默认")在语义上被区分开了。

`resolve_embed_model` 定义在 `llama-index-core/llama_index/core/embeddings/utils.py`,套路完全平行,但前缀更丰富。`EmbedType = Union[BaseEmbedding, "LCEmbeddings", str]`。`"default"` 在测试态解析为 `MockEmbedding(embed_dim=8)`,生产态解析为 `OpenAIEmbedding()` 并校验密钥;不同的是它的 `ValueError` 报错文案里额外提示了 "Consider using embed_model='local'." 并给出文档链接,引导没有 OpenAI 密钥的用户改用本地嵌入。字符串前缀方面,以 `clip` 开头的解析为多模态的 `ClipEmbedding`(冒号后是模型名,缺省 `ViT-B/32`);以 `local` 开头的解析为 `HuggingFaceEmbedding`,并把模型缓存到 `get_cache_dir()` 下的 `models` 子目录。传入 LangChain 的 `Embeddings` 会被包进 `LangchainEmbedding`;传入 `None` 则打印 "Embeddings have been explicitly disabled." 退回 `MockEmbedding(embed_dim=1)`。

把这两个 resolve 函数放在一起看,可以提炼出 LlamaIndex 配置归一化的统一哲学:**对外接受尽可能宽的输入类型(字符串简写、第三方对象、现成实例、None),对内统一收敛成一个标准接口实例,并在收敛失败时给出带安装命令或配置提示的人性化报错。** setter 调用 resolve 而非裸赋值,正是为了把这套归一化逻辑前移到"写入配置"那一刻,而不是留到每个消费点各自处理。

## 全局默认与局部覆盖如何共存

有了全局 `Settings`,框架还得回答一个问题:如果用户在某个具体组件上想用一个不同的 LLM,怎么办?LlamaIndex 的答案不是再造一套局部配置容器,而是一个极其简单、却贯穿了整个 core 的代码约定——`llm or Settings.llm`。

随手 grep 一下就能看到这个模式遍地开花。在 chat engine 里(`chat_engine/simple.py`、`condense_question.py`、`context.py` 等):`llm = llm or Settings.llm`;在各种 selector 里(`selectors/llm_selectors.py`):`llm = llm or Settings.llm`,`selectors/embedding_selectors.py`:`embed_model = embed_model or Settings.embed_model`;在 postprocessor、program、evaluation、extractors 等模块里同样如此,例如 `extractors/metadata_extractors.py` 甚至写成 `llm=llm or llm_predictor or Settings.llm`,把已废弃的 `llm_predictor` 参数也纳入了这条短路链。

这个 `or` 的语义恰到好处:Python 里 `a or b` 在 `a` 为假值(`None`)时取 `b`。于是每个组件的构造函数都把 LLM 设计成 `Optional`,默认 `None`,然后在内部 `self._llm = llm or Settings.llm`。用户传了就用用户的,没传就回退到全局 `Settings`。**局部参数永远优先于全局默认,而全局默认又永远兜底**,二者通过一行短路表达式自然共存,没有任何额外的优先级配置或合并规则需要记忆。

索引基类 `indices/base.py` 把这套约定演绎得更完整。它的 `as_query_engine` 接受一个 `Optional[LLMType]`,处理方式是:

```python
llm = (
    resolve_llm(llm, callback_manager=self._callback_manager)
    if llm
    else Settings.llm
)
```

这里多了一层细节:当用户传了 `llm`(可能是字符串简写),走的是 `resolve_llm(llm, ...)` 把它就地归一化成实例,并顺手把索引自己的 `_callback_manager` 注入进去;只有当用户什么都没传(`llm` 为假)时,才退回 `Settings.llm`。换句话说,局部覆盖路径也复用了同一个 `resolve_llm`,保证了无论从全局还是局部进入,LLM 最终都经过相同的归一化。`as_chat_engine` 用的是完全一样的写法。这条约定的好处是:消费方代码不需要知道用户到底传的是字符串还是实例,`resolve_llm`/`Settings.llm` 已经替它把类型摆平了。

## callback_manager 的自动注入

回调管理器(`CallbackManager`)是观测与追踪的中枢(详见第 14 章 instrumentation),它需要被"织入"到 LLM、嵌入模型、node parser 等每一个会产生事件的组件里。LlamaIndex 在两个层面做了自动注入,避免用户手动给每个对象挂回调。

第一层在 `Settings` 的 getter 里。回头看 `llm`、`embed_model`、`node_parser` 三个 getter,它们在返回对象之前都有同一段:

```python
if self._callback_manager is not None:
    self._llm.callback_manager = self._callback_manager
```

也就是说,只要用户设过 `Settings.callback_manager`,那么每次通过 `Settings` 取 LLM/嵌入/切分器时,都会顺手把全局回调管理器盖到那个对象上。这是一种"读时注入":配置对象在交付依赖的同一时刻完成回调的绑定,用户改了 `Settings.callback_manager` 后无需再去逐个组件重新设置。

第二层在 resolve 函数的末尾。`resolve_llm` 返回前有一行:

```python
llm.callback_manager = (
    callback_manager or llm.callback_manager or Settings.callback_manager
)
```

这条三级短路的优先级很讲究:显式传入的 `callback_manager` 最优先,其次保留对象自带的 `llm.callback_manager`,最后兜底到 `Settings.callback_manager`。`resolve_embed_model` 的结尾则更直接:`embed_model.callback_manager = callback_manager or Settings.callback_manager`。无论哪条路径,新解析出来的模型都不会"裸奔"——它一定带着一个回调管理器,要么是调用方指定的,要么是全局的。

消费端也有兜底。基类 retriever(`base/base_retriever.py`)的 `_check_callback_manager` 在对象还没有 `callback_manager` 属性时,直接 `self.callback_manager = Settings.callback_manager`;router retriever、各 chat engine、citation query engine 等在构造子组件时也普遍显式传 `callback_manager=Settings.callback_manager`。可以说回调管理器的注入在 LlamaIndex 里是"多重保险"的:配置 getter、resolve 函数、消费方构造逻辑三处都会去对齐 `Settings.callback_manager`,确保观测链路不会因为某个组件漏挂回调而断裂。

值得一提的是 tokenizer 和 prompt_helper 的处理与上面几个略有不同。`tokenizer` 的 getter 实际上读的是 `llama_index.core.global_tokenizer`,为空时回退到 `get_tokenizer()`——后者定义在 `utils.py`,默认按 `model_name="gpt-3.5-turbo"` 用 `tiktoken` 构造一个编码器。setter 则做了一个体贴的适配:如果传进来的是 HuggingFace 的 `PreTrainedTokenizerBase`,会自动 `partial(tokenizer.encode, add_special_tokens=False)` 包装成可调用对象再交给 `set_global_tokenizer`。`prompt_helper` 的 getter 则会在 `_llm` 已存在时用 `PromptHelper.from_llm_metadata(self._llm.metadata)` 按 LLM 的元数据(上下文窗口等)推导出合适的 prompt helper,否则退回 `PromptHelper()` 默认值。这些细节都印证了同一个原则:Settings 不只是被动存储,它在交付每一种依赖时都尽量"做对的默认事"。

## 从 ServiceContext 到 Settings 的演进

如果你读过老版本的 LlamaIndex 教程,会记得一个叫 `ServiceContext` 的容器——当年正是它扮演今天 `Settings` 的角色,把 LLM、嵌入模型、node parser 等打包在一起,通过 `service_context=` 参数到处传递。这段历史在源码里留下了清晰的痕迹。

`llama-index-core/llama_index/core/service_context.py` 至今还在,但已经被彻底掏空成一块"墓碑"。它的 `__init__` 和 `from_defaults` 都不再做任何事,而是直接抛异常:

```python
class ServiceContext:
    """
    Service Context container.

    NOTE: Deprecated, use llama_index.settings.Settings instead or pass in
    modules to local functions/methods/interfaces.
    """

    def __init__(self, **kwargs: Any) -> None:
        raise ValueError(
            "ServiceContext is deprecated. Use llama_index.settings.Settings instead, "
            "or pass in modules to local functions/methods/interfaces.\n"
            "See the docs for updated usage/migration: \n"
            "https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/"
        )
```

`from_defaults` 和模块级的 `set_global_service_context` 也是同样的处理:一律 raise 同一段迁移指引。这种"保留符号、移除实现、抛错引导"的做法是一种相当负责任的弃用方式——老代码 import `ServiceContext` 不会立刻 `ImportError`,而是在真正实例化时才报错,并把用户精确地指向迁移文档。

从设计角度看,这次演进的核心变化有两点。其一,从"显式容器到处传"变成"全局单例隐式取":过去要把 `service_context` 作为参数在调用链上层层透传,现在组件直接 `Settings.llm` 兜底,调用链干净了很多。其二,弃用说明里那句 "or pass in modules to local functions/methods/interfaces" 点明了配套的另一半哲学——LlamaIndex 鼓励对需要定制的场景直接把模块(`llm=`、`embed_model=`)作为局部参数传给具体接口,而不是再造一个大而全的上下文对象。全局 `Settings` 兜底加局部参数覆盖,这一进一退之间,正好就是本章前面分析的 `llm or Settings.llm` 模式。可以说,`Settings` 不只是 `ServiceContext` 的替代品,它代表了一种更轻、更显式、更符合直觉的依赖注入取向。

## 本章小结

1. `Settings` 是 LlamaIndex 的全局配置单例,定义在 `settings.py`,以 `Settings = _Settings()` 一行实例化;类名带下划线前缀表明 `_Settings` 是实现细节,对外只暴露已实例化的 `Settings`。
2. `_Settings` 是一个 dataclass,所有真实状态藏在 `_llm`、`_embed_model`、`_callback_manager`、`_node_parser` 等以下划线开头、默认为 `None` 的私有字段中,对外通过同名 property 读写。
3. 懒加载是核心机制:getter 在私有字段为 `None` 时才构造默认值并缓存,例如 `Settings.llm` 首次访问才 `resolve_llm("default")`,`node_parser` 首次访问才 `SentenceSplitter()`;用户提前设过则跳过默认构造,从而不为未用到的依赖付费。
4. setter 不做裸赋值,而是调用 `resolve_llm`/`resolve_embed_model` 把异构输入归一化:`LLMType = Union[str, LLM, "BaseLanguageModel"]`,`EmbedType = Union[BaseEmbedding, "LCEmbeddings", str]`。
5. `"default"` 是魔法字符串:测试态(环境变量 `IS_TESTING`)解析为 `MockLLM`/`MockEmbedding`,生产态解析为 `OpenAI()`/`OpenAIEmbedding()` 并调 `validate_openai_api_key` 校验;失败时抛带 `pip install` 命令或 `OPENAI_API_KEY` 提示的报错。
6. 字符串前缀约定:LLM 仅支持 `local`/`local:<path>`(走 `LlamaCPP`);嵌入支持 `local:`(走 `HuggingFaceEmbedding`)与 `clip`(走 `ClipEmbedding`,缺省 `ViT-B/32`);显式传 `None` 表示"禁用",退回 Mock 实现。
7. 全局默认与局部覆盖通过 `llm or Settings.llm` 这一短路约定共存,遍布 chat engine、selector、postprocessor、program 等模块;局部参数优先,全局 `Settings` 兜底。
8. `indices/base.py` 的 `as_query_engine`/`as_chat_engine` 在用户传值时走 `resolve_llm(llm, callback_manager=...)` 就地归一化,未传时退回 `Settings.llm`,保证两条路径殊途同归。
9. callback_manager 自动注入分两层:Settings 的 getter 在返回 LLM/嵌入/切分器前盖上 `_callback_manager`;resolve 函数末尾以 `callback_manager or 对象自带 or Settings.callback_manager` 三级短路兜底,确保模型不会无回调"裸奔"。
10. tokenizer 默认按 `gpt-3.5-turbo` 用 `tiktoken` 构造,setter 会自动把 HuggingFace 的 `PreTrainedTokenizerBase` 包装成可调用对象;prompt_helper 在已有 LLM 时按其 metadata 推导。
11. `ServiceContext` 已彻底弃用,`service_context.py` 中的类与 `set_global_service_context` 一律 raise 迁移指引;新范式是"全局 `Settings` 兜底 + 局部参数覆盖",更轻、更显式。

## 动手实验

1. **实验一:观察懒加载触发时机** — 在不设置任何 `Settings.llm` 的前提下,于 `settings.py` 的 `llm` getter 内 `resolve_llm("default")` 一行打断点或加 print,然后写一个只做向量检索、从不访问 LLM 的脚本,确认默认 LLM 从未被构造;再加一行访问 `Settings.llm`,确认此时才触发 resolve,直观体会"用到才 resolve"。
2. **实验二:复现归一化的多种输入** — 分别执行 `resolve_llm("default")`、`resolve_llm("local:/不存在的路径")`、`resolve_llm(None)`,观察三者各自的返回类型与报错/打印("LLM is explicitly disabled")文案;对照 `resolve_embed_model("clip")` 与 `resolve_embed_model("local:BAAI/bge-small-en")`,理解字符串前缀如何映射到不同实现包。
3. **实验三:验证局部覆盖优先于全局默认** — 先 `Settings.llm = MockLLM()`,再构造一个 chat engine 但显式传入另一个 `llm=`,在 `chat_engine/simple.py` 的 `llm = llm or Settings.llm` 处加 print,确认拿到的是局部传入的那个;随后不传 `llm`,确认回退到 `Settings.llm`。
4. **实验四:追踪 callback_manager 的自动注入** — 设置一个带自定义 handler 的 `Settings.callback_manager`,然后访问 `Settings.llm` 与 `Settings.embed_model`,打印它们的 `callback_manager` 是否就是你设的那个;再用 `resolve_llm(some_llm, callback_manager=另一个)` 验证 resolve 末尾三级短路的优先级顺序。

> **下一章预告**:配置就绪后,数据要先进得来。第 3 章我们将解构 LlamaIndex 的数据加载层 Readers——`SimpleDirectoryReader` 如何按扩展名分派解析器、`BaseReader` 的统一接口如何把五花八门的数据源收敛成 `Document` 列表,以及 reader 与本章的 `Settings`、下一步的 node parser 如何衔接成完整的摄入起点。
