# 第 11 章 工具系统 Tools Factory

在前一章我们看到，Langchain-Chatchat 的 Agent 把一组工具交给大模型，
由模型在推理过程中自行决定调用哪一个、传什么参数。
这一章我们把镜头对准这些工具的源头：`chatchat/server/agent/tools_factory` 目录。
它是一座"工具工厂"，把本地知识库检索、互联网搜索、Text2SQL、Shell 执行、文生图、
天气、ArXiv、维基百科等十余种能力，统一收编进一张全局注册表。
无论这些能力底层来自 LangChain 社区封装、第三方 HTTP API，还是 Chatchat 自己写的链路，
它们对外都呈现为同一种形态——一个带有 `name`、`description`、`args_schema` 的 LangChain `BaseTool`。

工具系统真正有意思的地方，在于它如何用一个不到一百五十行的 `tools_registry.py` 解决三个工程问题：
怎样让新增工具"写完就自动登记"无需手工维护列表；
怎样绕过 LangChain 当时对 pydantic `Field` 声明参数的限制；
以及怎样让一个既要给 LLM 看（字符串）又要给前端用（结构化数据）的工具输出同时满足两边的需要。
本章以 `tools_registry.py` 为骨架，挑 `search_local_knowledgebase`、`text2sql`、
`search_internet`、`shell` 四个代表性工具讲透其内部机制，其余工具做横向概述。

## 一张全局注册表与 `regist_tool` 装饰器

整个工具系统的中枢是 `tools_registry.py` 顶部那个朴素的模块级字典：

```python
_TOOLS_REGISTRY = {}
```

每一个工具文件不需要把自己手工加进某个列表，只要用 `@regist_tool` 装饰函数即可。
`regist_tool` 是对 LangChain 原生 `tool` 装饰器的一层包装，它的核心动作在内部函数 `_parse_tool(t: BaseTool)` 里：
拿到生成好的 `BaseTool` 实例后，它做的第一件事就是 `_TOOLS_REGISTRY[t.name] = t`，
以工具名为键登记进全局表。这是"写完即注册"的根本所在。
紧接着它做两件"补全"工作：补描述与补标题。

补全描述时，如果调用方没有显式传入 `description`，
它会回退到被装饰函数的 `__doc__` 文档字符串（协程函数则取 `coroutine.__doc__`），
然后用 `" ".join(re.split(r"\n+\s*", description))` 把多行文档压成单行——
因为这段描述最终要塞进 prompt 给 LLM，单行更干净。
补全标题时，若未传 `title`，就把工具名按下划线拆开、每段首字母大写拼成一个人类可读的名字，
例如 `search_local_knowledgebase` 会得到 `SearchLocalKnowledgebase`。

`regist_tool` 的签名兼容了 LangChain `tool` 装饰器的两种用法。
当 `len(args) == 0`（即写成 `@regist_tool(title="天气查询")` 这种带括号形式）时，
返回一个 `wrapper` 闭包，由它接收被装饰函数、调用 `tool(...)` 生成实例再 `_parse_tool`；
当直接写成 `@regist_tool`（无括号，被装饰函数作为第一个位置参数）时，则立即构造并登记。
`wolfram.py` 用的就是后一种无括号写法，其余工具大多用前一种。

这样一来，新增工具的流程被压缩到极致：
写一个函数、加一行装饰器、在 `tools_factory/__init__.py` 里 import 一下触发模块加载，工具就活了。
该 `__init__.py` 正是 `from .arxiv import arxiv`、`from .text2sql import text2sql` 这样一串 import 语句，
把各工具逐一引入，import 的副作用就是执行装饰器、完成注册。
也就是说，注册表的填充时机不是某个集中的初始化函数，而是分散在每个工具模块被导入的那一刻。

## 对 LangChain BaseTool 的两处 monkeypatch

`tools_registry.py` 在模块加载时直接对 `BaseTool` 这个类做了运行时打补丁，
注释里明确写着这是针对 LangChain issue #15855 的 workaround。

第一处补丁是一行：

```python
BaseTool.Config.extra = Extra.allow
```

LangChain 的 `BaseTool` 默认不允许携带 schema 之外的额外字段，
而 Chatchat 想给每个工具挂一个 `title` 属性（前面 `_parse_tool` 里 `t.title = title` 就依赖它）。
把 pydantic 的 `extra` 配置改成 `Extra.allow`，工具实例就能自由附加 `title` 这样的扩展字段，
前端 `/tools` 接口才能把 `t.title` 取出来展示给用户。

第二处补丁更关键，它替换了 `BaseTool._parse_input` 和 `BaseTool._to_args_and_kwargs` 两个方法：

```python
BaseTool._parse_input = _new_parse_input
BaseTool._to_args_and_kwargs = _new_to_args_and_kwargs
```

目的是支持用 pydantic `Field` 直接在函数签名里声明参数，例如：

```python
def search_local_knowledgebase(
    database: str = Field(description="Database for Knowledge Search", choices=[...]),
    query: str = Field(description="Query for Knowledge Search"),
):
```

新的 `_new_parse_input` 在输入是字符串时取 `args_schema` 的第一个字段做校验并原样返回；
在输入是字典时用 `input_args.parse_obj(tool_input)` 解析成模型再 `.dict()`。
新的 `_new_to_args_and_kwargs` 处理把 `tool_input` 拆成位置参数与关键字参数的逻辑，
并对名为 `args` 的字段做了 `*args` 展开的特殊处理。

有了这两处替换，Chatchat 的工具作者就能用熟悉的 `Field(description=...)` 写法声明每个参数的语义，
这些语义会进入工具的 `args_schema`，再被序列化进 prompt，帮助 LLM 准确填参。
值得一提的是 `search_local_knowledgebase` 还在 `Field` 里用了 `choices=[...]`
把可选的知识库名列出来，等于直接告诉模型"database 只能从这几个里挑"。

## `BaseToolOutput`：一份输出，两种消费者

几乎每个工具的返回值都不是裸字符串，而是被 `BaseToolOutput` 包了一层。
它定义在 `langchain_chatchat/agent_toolkits/all_tools/tool.py`，类文档一句话点破了设计动机：
"LLM 要求 Tool 的输出为 str，但 Tool 用在别处时希望它正常返回结构化数据。"

`BaseToolOutput` 继承自 LangChain 的 `Serializable`，
持有 `data`（原始数据，任意类型）、`format`（格式化方式）、`data_alias`、`extras` 几个字段。
它的灵魂是 `__str__` 方法：

```python
def __str__(self) -> str:
    if self.format == "json":
        return json.dumps(self.data, ensure_ascii=False, indent=2)
    elif hasattr(self, "_format_callable") and callable(self._format_callable):
        return self._format_callable(self)
    else:
        return str(self.data)
```

当 Agent 需要把工具结果拼回 prompt 时，Python 会对其调用 `str()`，此时分三种情况——
`format == "json"` 就用 `json.dumps(..., ensure_ascii=False, indent=2)` 输出格式化 JSON；
存在可调用的格式化器时调用之；否则退化为 `str(self.data)`。
这样工具内部可以始终返回包含 `docs`、`message_type` 等键的字典结构供前端或别处使用，
而面向 LLM 时又能"现场"转成一段合适的文本。

与之配套的是 `tools_registry.py` 里的 `format_context(self: BaseToolOutput)` 函数，
它是专门给"检索类"工具用的格式化器。
它从 `self.data["docs"]` 里逐条取出文档，用 `DocumentWithVSId.parse_obj(doc)` 反序列化后收集 `page_content`，
再用空行拼接成一整段上下文；若一条都没有，则返回"没有找到相关文档,请更换关键词重试"这句友好提示。
`search_local_knowledgebase`、`search_internet`、`url_reader` 都把 `format=format_context`
传给 `BaseToolOutput`，于是它们对 LLM 呈现的就是干净的纯文本知识，
而原始的 doc 结构仍保留在 `data` 里供引用展示。

## 代表工具一：本地知识库检索与 `return_direct`

`search_local_knowledgebase.py` 是 RAG 应用里最核心的工具，
也是理解"工具如何接回前几章检索链路"的关键。
它的描述 `template_knowledge` 在模块加载时就动态拼好：
把 `Settings.kb_settings.KB_INFO` 里所有知识库的名称与说明列出来，
告诉模型"只能从这些库里挑一个，并且只有问本地数据时才用这个工具"。

函数体很短——调用 `get_tool_config("search_local_knowledgebase")` 取出 `top_k` 与 `score_threshold`，
再委托给 `search_docs`（也就是前面章节讲过的知识库检索服务）完成实际检索，
最后把 `{"knowledge_base": database, "docs": docs}` 包进 `BaseToolOutput(ret, format=format_context)` 返回。

更进一步，由于知识库工具的 `choices` 依赖运行时实际存在的知识库，
Chatchat 在 `utils.py` 的 `update_search_local_knowledgebase_tool()` 里做了动态刷新：
每次 `get_tool()` 被调用时，它会从数据库 `list_kbs_from_db()` 读出当前所有知识库，
重新拼出 description，并把 `search_local_knowledgebase_tool.args["database"]["choices"]`
改成最新的库名列表。
这意味着用户新建一个知识库后，无需重启，工具的可选项就自动跟上了。

知识库工具值得单独点名的另一处设计是 `return_direct`。
`regist_tool`（以及它包装的 LangChain `tool`）支持 `return_direct=True`，
表示工具一旦执行完，其结果直接作为最终回答返回，不再回灌给 LLM 做二次加工。
文生图工具 `text2images` 就用了 `return_direct=True`——
因为生成的图片路径没必要再让 LLM 复述一遍。
知识库检索是否直接返回则取决于具体配置与编排策略，
但 `return_direct` 这一开关的存在，
让"检索即答案"与"检索后再让模型总结"两种产品形态都能被同一套工具机制支撑。

## 代表工具二：Text2SQL 的只读防护

`text2sql.py` 展示了一个"重"工具如何在工具函数这薄薄一层里组织起完整的 LangChain 链路。
工具本体 `text2sql` 只接收一个自然语言 `query`，把繁重逻辑全交给 `query_database(query, config)`。
后者从配置里取出 `model_name`、`sqlalchemy_connect_str`、`read_only`、`top_k`、
`table_names`、`table_comments` 等一整套参数，用 `SQLDatabase.from_uri` 连库，
用 `get_ChatOpenAI` 拿到一个独立指定的模型。
注意 `text2sql` 在 `ToolSettings` 里有自己的 `model_name`（默认 `qwen-plus`），与前端选择的对话模型解耦。

它的两层安全设计很有教学价值。当 `read_only` 为真时，
它先跑一条 `READ_ONLY_PROMPT_TEMPLATE` 让 LLM 预判"这条需求在只读模式下能否执行"，
若模型回答 "SQL cannot be executed normally" 就直接返回"当前数据库为只读状态，无法满足您的需求！"；
与此同时，它还通过 SQLAlchemy 的事件机制注册了一个底层拦截器：

```python
event.listen(db._engine, "before_cursor_execute", intercept_sql)
```

`intercept_sql` 检查语句是否以 `insert/update/delete/create/drop/alter/truncate/rename`
这些写操作关键字开头，命中就抛 `OperationalError`。
一层靠模型判断给友好提示，一层靠引擎拦截兜底——
即便模型判断失误，写操作也会在执行前被拒。

链路选择上它也有讲究：
若配置里指定了 `table_names`，就走 `SQLDatabaseChain` 直接把指定表结构喂给模型；
若没指定，则走 `SQLDatabaseSequentialChain`，先让模型预测要用哪些表，再把相关表交给后续链——
这是为了避免把全量表结构一次塞给模型导致 token 超长。
注释里还坦白了一个实践痛点：Sequential 模式靠表名预测，容易误判，
所以提供了 `table_comments` 让用户给表加说明，
由于 LangChain 固定了入参，这些说明只能通过拼接进 `query` 传递。

`text2promql.py` 的结构与之同源，区别是它把自然语言翻成 PromQL 再打 Prometheus 的 HTTP API，
同样用一个 LCEL 链 `prompt | llm | output_parser` 完成转换，
最后由 `execute_promql_request` 拼出 `/api/v1/{query_type}` 路径发起请求。

## 代表工具三与四：互联网搜索与 Shell

`search_internet.py` 体现了"一个工具，多个后端"的可插拔思路。
它在 `SEARCH_ENGINES` 字典里登记了 `bing`、`duckduckgo`、`metaphor`、`searx` 四个搜索后端，
每个对应一个函数：

```python
SEARCH_ENGINES = {
    "bing": bing_search,
    "duckduckgo": duckduckgo_search,
    "metaphor": metaphor_search,
    "searx": searx_search,
}
```

`search_engine()` 根据配置里的 `search_engine_name` 选用对应实现，
`top_k` 缺省时回退到配置或 `Settings.kb_settings.SEARCH_ENGINE_TOP_K`，
最后用 `search_result2docs` 把各家返回的 `snippet/link/title` 统一成 LangChain `Document`，
再包进 `BaseToolOutput(..., format=format_context)`。
其中 `metaphor` 后端还额外支持把长结果用 `RecursiveCharacterTextSplitter` 切块、
用 `NormalizedLevenshtein` 算相似度重排取 top_k，相当于在搜索工具内部又做了一轮迷你检索。
工具入口 `search_internet` 本身只声明一个 `query` 参数，把多后端复杂度完全藏在了内部。

`shell.py` 则是另一个极端——它是全书最短的工具，函数体只有三行：

```python
@regist_tool(title="系统命令")
def shell(query: str = Field(description="The command to execute")):
    """Use Shell to execute system shell commands"""
    tool = ShellTool()
    return BaseToolOutput(tool.run(tool_input=query))
```

它实例化 LangChain 社区的 `ShellTool`，调用 `tool.run(tool_input=query)`，再用 `BaseToolOutput` 包一下。
本质上是把一个外部 `BaseTool` "嫁接"进 Chatchat 的注册表并统一输出格式。
`arxiv`、`wikipedia_search`、`wolfram` 走的也是同一条路子：
分别套用 `ArxivQueryRun` [[LangChain Arxiv 工具]](https://python.langchain.com/docs/integrations/tools/arxiv/)、
`WikipediaQueryRun`（指定 `lang="zh"`）[[LangChain Wikipedia 工具]](https://python.langchain.com/docs/integrations/tools/wikipedia/)、
`WolframAlphaAPIWrapper` [[LangChain Wolfram Alpha 工具]](https://python.langchain.com/docs/integrations/tools/wolfram_alpha/)，
几行代码完成"借壳上市"。

需要提醒的是，Shell 工具直接执行系统命令、Text2SQL 可触达数据库，二者都带有显著安全风险，
因此它们在 `ToolSettings` 里默认 `use: False`，需用户明确开启并自行评估隔离与权限边界。

## 其余工具与按配置启用

除上面四个，工厂里还有一批以 HTTP API 为主的轻量工具：

- `amap_weather` 调高德地图的行政区划与天气接口 [[高德开放平台 Web 服务 API]](https://lbs.amap.com/api/webservice/summary)，先用 `get_adcode` 拿城市编码再查天气；
- `weather_check` 调心知天气 `api.seniverse.com`，抽取出 `temperature` 与 `description`；
- `url_reader` 借 jina-ai 的 `r.jina.ai` 把网页正文转成 LLM 友好的文本，并复用 `format_context`；
- `calculate` 用 `numexpr` 对数学表达式求值，出错时返回 `wrong: ...`；
- `text2images` 调 OpenAI 兼容的文生图接口，把结果落盘到 `MEDIA_PATH` 并以 `format="json"` 输出。

它们的共同模式是：用 `get_tool_config(name)` 取自己的配置块（含 `api_key` 等），
发请求，把结果包进 `BaseToolOutput`。

所有工具的开关都集中在 `settings.py` 的 `ToolSettings` 类里，
它从 `tool_settings.yaml`/`tool_settings.json` 加载（`extra="allow"`），
为每个工具提供一个字典配置项，统一带 `use` 字段，绝大多数默认 `False`。
`get_tool_config(name)` 通过 `Settings.tool_settings.model_dump().get(name, {})`
把对应配置返还给工具函数。

注册与启用是两件事：`@regist_tool` 让工具进入 `_TOOLS_REGISTRY`（注册），
而是否真正交给 Agent 使用，则在对话链路里按需筛选。
`chat.py` 中拿到全表后用工具名做交集：

```python
all_tools = get_tool().values()
tools = [tool for tool in all_tools if tool.name in tool_config]
tools = [t.copy(update={"callbacks": callbacks}) for t in tools]
```

仅保留本次请求声明要用的工具，并 `copy(update={"callbacks": callbacks})` 注入回调。
前端的工具列表则由 `tool_routes.py` 的 `/tools` 接口提供，
它把每个工具的 `name/title/description/args/config` 打包返回，
`/tools/call` 还支持直接 `ainvoke` 单个工具做调试。

## 本章小结

- 工具系统的中枢是模块级字典 `_TOOLS_REGISTRY`，`@regist_tool` 装饰器在内部 `_parse_tool` 中执行 `_TOOLS_REGISTRY[t.name] = t` 实现"写完即注册"。
- `regist_tool` 是对 LangChain `tool` 装饰器的包装，会自动用函数 `__doc__` 补全描述并压成单行、用工具名首字母大写补全 `title`，并兼容带括号与无括号两种装饰写法。
- 模块加载时对 `BaseTool` 做了两处 monkeypatch：`Config.extra = Extra.allow` 以支持挂载 `title` 等额外字段；替换 `_parse_input`/`_to_args_and_kwargs` 以支持用 pydantic `Field` 声明参数（针对 LangChain #15855）。
- `args_schema` 里的 `Field(description=..., choices=...)` 会被序列化进 prompt，帮助 LLM 理解每个参数语义甚至枚举取值，如知识库工具用 `choices` 限定可选库名。
- `BaseToolOutput` 让一份工具输出同时服务两类消费者：`data` 保留结构化数据供前端使用，`__str__` 按 `format`（`"json"`、可调用格式化器或裸 `str`）转成文本供 LLM 消费。
- `format_context` 是检索类工具的专用格式化器，从 `data["docs"]` 抽取 `page_content` 拼成上下文，无结果时返回友好提示；被知识库、互联网搜索、URL 阅读三类工具复用。
- `search_local_knowledgebase` 委托 `search_docs` 完成检索，并由 `update_search_local_knowledgebase_tool()` 在运行时动态刷新 description 与 `database` 的 `choices`，新建知识库无需重启即可生效。
- `return_direct=True` 让工具结果直接作为最终答案不再过 LLM，`text2images` 即采用此模式。
- `text2sql` 采用"模型预判 + SQLAlchemy 引擎拦截器 `intercept_sql`"双层只读防护，并按是否指定 `table_names` 在 `SQLDatabaseChain` 与 `SQLDatabaseSequentialChain` 之间切换以控制 token 规模。
- `search_internet` 用 `SEARCH_ENGINES` 字典支持 bing/duckduckgo/metaphor/searx 多后端可插拔；`shell`、`arxiv`、`wikipedia_search`、`wolfram` 则直接套用 LangChain 社区工具"借壳"进注册表。
- 工具的注册与启用解耦：`ToolSettings` 用每个工具的 `use` 字段（多数默认 `False`）控制可用性，对话链路按 `tool.name in tool_config` 二次筛选，高风险的 Shell、Text2SQL 默认关闭。

## 动手实验

1. **实验一：写一个自定义工具并验证自动注册** — 在 `tools_factory` 下新建 `dice.py`，用 `@regist_tool(title="掷骰子")` 装饰一个 `roll(sides: int = Field(description="骰子面数"))` 函数返回 `BaseToolOutput(random.randint(1, sides))`，在 `__init__.py` 中 import 它；启动后调用 `get_tool("dice")`，确认它已进入 `_TOOLS_REGISTRY` 且 `title`、`description`、`args` 均被正确填充。
2. **实验二：观察 BaseToolOutput 的双面性** — 在 Python REPL 中构造 `out = BaseToolOutput({"docs": [...]}, format=format_context)`，分别打印 `out.data`（结构化字典）与 `str(out)`（经 `format_context` 转出的纯文本上下文）；再换成 `format="json"` 与不传 `format`，对比三种 `__str__` 输出差异，理解"一份输出两种消费者"的设计。
3. **实验三：给 Text2SQL 套上只读防护** — 在 `tool_settings` 中配置一个本地 SQLite 的 `sqlalchemy_connect_str` 并设 `read_only: True`、`use: True`，先用一句自然语言查询验证可正常返回结果，再尝试"把某条记录删掉"这类写意图，观察是先被 `READ_ONLY_PROMPT` 拦下给出友好提示，还是被 `intercept_sql` 拦截器抛出 `OperationalError`。
4. **实验四：切换互联网搜索后端** — 把 `search_internet` 的 `use` 改为 `True`，将 `search_engine_name` 从 `duckduckgo` 切到 `searx`（填入可用的 `host`），同一个查询对比两种后端经 `search_result2docs` 与 `format_context` 后给到 LLM 的文本差异，并阅读 `metaphor` 分支体会"工具内部再做一轮切块重排"的写法。

> **下一章预告**：注册表里的工具都是 Chatchat 进程内置的"本地能力"。但真实业务往往还需要接入外部独立服务暴露的工具——文件系统、浏览器、第三方 SaaS 等。第 12 章我们将进入 **MCP（Model Context Protocol）集成**，看 Langchain-Chatchat 如何把远端 MCP Server 暴露的工具动态拉进同一套 Agent 调用体系，与本章的 Tools Factory 协同工作。
