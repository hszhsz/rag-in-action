# 第 1 章 项目全景与多进程启动

每一个能"离线跑起来"的 RAG 应用，背后都藏着一段不太起眼却极其关键的代码：进程是怎么被拉起来的。

Langchain-Chatchat 选择了一条与众不同的路线。它不是一个单一的 Web 服务，而是由一个父进程亲手 `spawn` 出两个相互独立的子进程：一个跑 FastAPI，对外提供 OpenAI 风格的 HTTP 接口；一个跑 Streamlit，提供可点击的网页对话界面。本章要解构的，正是这套"父进程编排、子进程各司其职"的启动骨架，以及它背后那些关于跨平台兼容、就绪同步、启动前建表的工程取舍。

理解这一章为什么重要，要先理解 Chatchat 的定位。它是一个面向"本地知识库 + LLM"的可离线部署应用，使用者往往是在自己的服务器甚至个人电脑上一键拉起整套系统。

这意味着启动逻辑必须足够健壮：既要能在 Windows 上正确创建子进程，又要保证 WEBUI 不会在 API 还没准备好时就抢先连上去。我们将以 `startup.py`、`cli.py`、`init_database.py` 这三个文件为唯一事实来源，逐层拆开这台机器的"点火装置"。

## monorepo 与 libs 双子包结构

Chatchat 仓库的根目录采用 monorepo 布局，真正的代码集中在 `libs/` 下，并被切成两个子包：`chatchat-server` 与 `python-sdk`。

从 `libs/chatchat-server/pyproject.toml` 可以读到，这个子包同时打包了两个 Python 包——`packages = [{include = "chatchat"}, {include = "langchain_chatchat"}]`。

前者 `chatchat` 是应用本体，包含 `startup.py`、`cli.py`、`webui.py` 以及整个 `server/` 目录；后者 `langchain_chatchat` 是抽离出来、可被外部复用的 SDK 层。浏览 `server/` 目录能看到 `agent`、`api_server`、`chat`、`db`、`file_rag`、`knowledge_base`、`reranker` 等子目录，整本书后续章节几乎都将围绕它们展开。

`python-sdk` 则是另一个独立子包，目录里能看到 `agents`、`agent_toolkits`、`chat_models`、`embeddings`、`callbacks` 等模块，定位是把对接 LangChain 的能力沉淀成库。这种"应用与库分家"的切分，让核心模型接入能力可以脱离 Web 框架被单独引用。

值得注意的是，仓库根目录的 `pyproject.toml` 与子包的 `pyproject.toml` 并不一致。根目录那份的 `name` 写的是 `Chatchat`、`version` 是 `0.3.0`、且 `package-mode = false`，更像是给 monorepo 顶层做 lint 与工具配置用的占位元信息。

而真正定义可安装包、版本号与依赖清单的，是 `libs/chatchat-server/pyproject.toml`，其 `name = "langchain-chatchat"`、`version = "0.3.1.3"`。这个版本号与 `chatchat/__init__.py` 里的 `__version__ = "0.3.1.3"` 严格对齐。

`__version__` 在多处被引用：`dump_server_info` 打印"项目版本"，`webui.py` 在侧边栏显示"当前版本"，`server_app.py` 把它塞进 `FastAPI(title="Langchain-Chatchat API Server", version=__version__)`。

把版本号收敛到单一常量，是个朴素但有效的工程习惯——改一处，全局同步，不会出现接口文档和侧边栏版本对不上的尴尬。

子包的 `pyproject.toml` 还藏着两条关键信息。

其一，`[tool.poetry.scripts]` 声明了 `chatchat = 'chatchat.cli:main'`，这正是用户敲下 `chatchat` 命令时实际执行的入口，也解释了为什么 `cli.py` 是整个程序的真正起点。

其二，Python 版本被锁在 `>=3.10,<3.12,!=3.9.7`，且 `langchain = "0.1.17"`、`fastapi = "~0.109.2"`、`streamlit = "1.34.0"`、`uvicorn = ">=0.27.0.post1"` 等核心依赖都被钉死或加了上界。对一个强调"可离线复现"的项目而言，这种近乎苛刻的版本固定，是保证别人能在另一台机器上跑出同样行为的前提。

## click 三层命令树：从 `chatchat` 到 `start`

入口 `cli.py` 用 Click 搭了一棵命令树。顶层 `@click.group(help="chatchat 命令行工具")` 定义了空的 `main()` 作为命令组，本身不做任何事，只负责聚合子命令。

Click 的 group/command 机制让一个可执行入口可以承载多个子命令，这正契合 Chatchat "一个 `chatchat` 命令、多种用途"的设计。

子命令通过两种方式挂载。`main.add_command(startup_main, "start")` 与 `main.add_command(kb_main, "kb")` 把两个独立模块的命令引进来——这里的 `startup_main` 就是 `chatchat.startup` 里的 `main`，`kb_main` 是 `chatchat.init_database` 里的 `main`。

换句话说，`chatchat start` 实际跳转到 `startup.py`，`chatchat kb` 跳转到 `init_database.py`，而 `chatchat init` 则是用 `@main.command("init")` 直接定义在 `cli.py` 自身里的子命令。这种"命令分散在各模块、再聚合到一棵树"的组织方式，让启动逻辑与知识库迁移逻辑彼此解耦。

`init` 子命令承担"第一次部署"的职责。它先 `Settings.set_auto_reload(False)` 关闭配置热重载，避免在初始化过程中配置被反复读盘；接着 `Settings.basic_settings.make_dirs()` 建好所有数据目录；再把内置的 `data/knowledge_base/samples` 样例知识库用 `shutil.copytree` 复制到 `KB_ROOT_PATH` 下的 `samples`（且有一层判断，源路径与目标路径相同时跳过复制）。

随后 `init` 调用 `create_tables()` 建好数据库表，按命令行参数覆盖 `DEFAULT_LLM_MODEL`、`DEFAULT_EMBEDDING_MODEL` 与 `MODEL_PLATFORMS[0].api_base_url`，最后用 `Settings.createl_all_templates()` 生成默认 YAML 配置文件，再 `set_auto_reload(True)` 把热重载打开。

`init` 的两个默认值很能说明产品取向：`--llm-model` 的帮助文案默认是 `glm4-chat`，`--embed-model` 的帮助文案写的是 `bge-large-zh-v1.5`，都是面向中文场景的模型选择；而 `--xinference-endpoint` 默认指向 `http://127.0.0.1:9997/v1`，暗示 Xinference 是其推荐的本地推理后端。要注意的是，这里的帮助文案与代码实际默认值已脱节——用户不传 `--embed-model` 时实际生效的是 `Settings.model_settings.DEFAULT_EMBEDDING_MODEL`，其真实默认在 `settings.py` 里是 `bge-m3`（见第 2 章），`bge-large-zh-v1.5` 只是一处未同步更新的注释文案。

整条 `init` 流程跑完后，若带了 `-r/--recreate-kb` 就会顺手 `folder2db(...)` 重建向量库，否则日志会引导用户"执行 `chatchat kb -r` 初始化知识库，然后 `chatchat start -a` 启动服务"——这恰好串起了 `init`、`kb`、`start` 三个命令的使用次序。

`start` 命令本身只暴露三个开关：`-a/--all`、`--api`、`-w/--webui`，分别对应"全开"、"只开 API"、"只开 WEBUI"。

`main(all, api, webui)` 把它们塞进一个临时的 `class args: ...` 占位类里，再交给 `start_main_server(args)`。这种"用一个空类当参数包"的写法虽不优雅，却避免了在异步主流程的函数签名里铺一长串关键字参数。而 `start_main_server` 开头还会用 `if args.all:` 把 `args.api` 与 `args.webui` 一并置真，于是 `-a` 等价于同时打开两者。

## spawn 启动方式与 Manager Event 编排

`start` 命令的真正引擎是 `startup.py` 里的 `async def start_main_server(args)`。它做的第一件大事，是把多进程启动方式显式设为 spawn：`mp.set_start_method("spawn")`。

这是整章最值得品味的一行。Python 在 Linux 上默认用 `fork` 创建子进程，`fork` 会连同父进程的内存、已导入模块、甚至已打开的句柄一并复制。对于一个加载了大模型相关库、可能持有 CUDA 上下文或网络连接的进程，`fork` 极易引发死锁和状态污染。

而 `spawn` 会启动一个全新的解释器、重新 import 目标函数所需的模块，行为干净且跨平台一致——Windows 本就只支持 spawn。

代价是子进程无法继承父进程的内存，所有要传递的东西都得能被序列化，这也正是后面 `Manager().Event()` 登场的原因 [[multiprocessing — Process-based parallelism]](https://docs.python.org/3/library/multiprocessing.html)。把启动方式钉死为 spawn，本质上是用一点点序列化成本换取部署环境的可预测性，对一个要在各种陌生机器上"一键运行"的项目而言，这笔交易非常划算。

紧接着 `manager = mp.Manager()` 创建了一个进程间共享对象的管理器。代码为 API 与 WEBUI 各申请了一个事件：`api_started = manager.Event()` 与 `webui_started = manager.Event()`。

此外还有一个细节：`start_main_server` 里把 `run_mode` 初始化为 `None`，并原样透传给两个子进程的 `kwargs`。这个变量是为 lite 精简模式预留的开关，当前默认流程下保持关闭。

为什么用 `Manager().Event()` 而不是普通的 `mp.Event()`？

因为在 spawn 模式下，普通事件难以可靠地跨进程共享，而 `Manager().Event()` 由管理器进程托管、通过代理对象访问，正好能跨越 spawn 出来的子进程边界。这两个事件就是父子进程之间的"握手信号"：子进程就绪后 `set()`，父进程在 `wait()` 处放行。

子进程通过标准库的 `Process` 创建，两者的参数有意做了区分。API 进程 `target=run_api_server`、`name="API Server"`、`daemon=False`；WEBUI 进程 `target=run_webui`、`name="WEBUI Server"`、`daemon=True`。

这里的 daemon 差异是有意为之。把 WEBUI 设为守护进程，意味着父进程退出时它会被自动回收；而 API 进程作为核心服务则不设守护，交由后续 `finally` 里的清理逻辑显式处理。

两个进程对象被存进 `processes` 字典，键分别是 `"api"` 和 `"webui"`，便于后面统一管理与回收；代码还顺手定义了一个 `process_count()` 闭包返回字典长度。值得留意的是，这两个 `Process` 的构造都只是"声明"，真正的 `start()` 要等到后面的串行拉起阶段才发生。

## 串行拉起：先 API 后 WEBUI 的就绪等待

进程的实际启动是严格串行的。`start_main_server` 里先用海象表达式 `if p := processes.get("api"):` 取出 API 进程并 `p.start()` 拉起它，给它的 `name` 追加上 pid，然后调用 `api_started.wait()`——父进程在这里阻塞，直到 API 子进程把事件置位。

WEBUI 进程同理：先 `p.start()` 再 `webui_started.wait()`。这种"启动一个、等它就绪、再启动下一个"的顺序，保证了 WEBUI 被拉起时 API 已经能够响应，避免了网页端在后端尚未监听端口时就发起请求而报错。

值得对照的是：尽管两个进程是依次启动的，`start_main_server` 在判断 `if p := processes.get("api")` 与 `processes.get("webui")` 时彼此独立，因此 `chatchat start --webui` 单独拉 WEBUI 也能工作，只是这种情况下 WEBUI 连接的 API 需由用户另行提供。

事件是如何被置位的？答案藏在两个 target 函数里，且两者机制不同，很能说明问题。

`run_api_server` 走的是 FastAPI 的生命周期钩子。它通过 `_set_app_event(app, started_event)` 给 app 装上一个自定义的 `lifespan` 异步上下文管理器，在 `lifespan` 进入时执行 `started_event.set()`，随后 `yield`。

`_set_app_event` 的实现是把这个 `lifespan` 赋给 `app.router.lifespan_context`，从而替换掉 FastAPI 默认的空生命周期。这是一种相当巧妙的"侵入式注入"：不改动 `create_app` 的内部逻辑，就在外层把就绪信号挂了进去。

也就是说，只有当 uvicorn 真正完成 app 启动、进入 lifespan，事件才被置位——这是一个相当精确的"服务已就绪"信号 [[FastAPI Lifespan Events]](https://fastapi.tiangolo.com/advanced/events/)。

在置位之前，`run_api_server` 还做了几件准备工作：打印 `MODEL_PLATFORMS`、调用 `set_httpx_config()` 统一 httpx 行为、用 `create_app(run_mode=run_mode)` 构建应用、配置好日志。它的尾部 `uvicorn.run(app, host=host, port=port)` 会一直阻塞运行，host 与 port 取自 `Settings.basic_settings.API_SERVER`。

顺带一提，被构建出来的 app 在 `server_app.py` 的 `create_app` 里完成了路由装配：`MakeFastAPIOffline(app)` 把 swagger 静态资源本地化（呼应"离线"主旨），随后 `include_router` 依次挂上 `chat_router`、`kb_router`、`tool_router`、`openai_router`、`server_router`、`mcp_router`，并 `mount` 了 `/media` 与 `/img` 两个静态目录。

`run_webui` 则朴素得多。它把一大坨 Streamlit 的 `flag_options` 准备好——其中 `theme_primaryColor` 设为 `#165dff`、`theme_base` 设为 `"light"`、`server_fileWatcherType` 关成 `"none"`，其余几十个选项几乎全是 `None`，目的是显式覆盖 Streamlit 默认值、关闭文件监听以减少开销。

把文件监听设为 `"none"` 尤其关键：生产部署时没必要让 Streamlit 监视源码变更并自动 rerun，关掉它能省下一个常驻的文件系统监听线程。

接着它导入 `streamlit.web.bootstrap`（旧版本回退到 `streamlit.bootstrap`），调用 `bootstrap.load_config_options(flag_options=flag_options)` 与 `bootstrap.run(script_dir, False, args, flag_options)`，直接以编程方式拉起 Streamlit，跑的脚本正是同目录下的 `webui.py`（`script_dir` 由 `os.path.dirname(os.path.abspath(__file__))` 拼出）。

注意 `started_event.set()` 写在 `bootstrap.run(...)` 之后，而 `bootstrap.run` 通常是阻塞的。这意味着对 WEBUI 而言，事件的"就绪"语义并不像 API 那样精确——这是源码里一处真实可见的不对称设计，读者不妨在实验中亲自验证它的行为。

好在这并不致命：由于 WEBUI 是最后一个被拉起的进程，`webui_started.wait()` 即便提前返回，也只是让父进程稍早进入守护循环，不会破坏整体启动顺序。

至于被拉起的 `webui.py`，它用 `is_lite = "lite" in sys.argv` 判断是否进入精简模式（`run_webui` 在 `run_mode == "lite"` 时会向 `args` 追加 `["--", "lite"]`），随后用 `streamlit_antd_components` 的 `sac.menu` 渲染出"多功能对话 / RAG 对话 / 知识库管理 / MCP 管理"四个页面入口。代码里 `# TODO: remove lite mode` 的注释说明该模式正在被淘汰。

## 守护循环与信号清理

两个子进程都起来后，父进程调用 `dump_server_info(after_start=True, args=args)` 打印出 `Chatchat Api Server` 与 `Chatchat WEBUI Server` 的访问地址，地址由 `server/utils.py` 的 `api_address()`、`webui_address()` 计算得到。

`dump_server_info` 其实在启动前后各调用了一次：第一次（`after_start=False`）打印操作系统、Python 版本、`__version__`、langchain 版本、数据目录、分词器 `TEXT_SPLITTER_NAME` 与默认 Embedding 等环境信息；第二次（`after_start=True`）才追加打印服务端访问地址。这种"环境自报"对排查部署问题极有价值——用户贴一段启动日志，维护者就能一眼看清版本与配置。

`api_address` 有个值得记的细节：当配置的 host 是 `0.0.0.0` 时会被替换成 `127.0.0.1`，因为 `0.0.0.0` 是监听地址而非可访问地址，直接拿去拼 URL 是连不上的。

它还支持 `is_public` 模式，从 `API_SERVER` 里读 `public_host`、`public_port`，以便在云服务器或反向代理后生成正确的公网地址（如知识库文档下载链接）。`webui_address` 则简单地用 `WEBUI_SERVER` 的 host 与 port 拼出地址，不做替换。这些地址会在 `dump_server_info` 里被打印成醒目的访问入口。

随后父进程进入守护循环：`while processes:` 中对每个进程 `p.join(2)`，若 `not p.is_alive()` 就把它从字典里弹出。

这个"每 2 秒轮询一次存活状态"的循环让父进程持续盯着子进程，任一进程退出都能被感知。`join(2)` 带超时而非无限阻塞，是为了让循环能周期性地检查所有进程，而不会卡死在某一个上。

整段逻辑包在 `try/except/finally` 里。捕获到异常（包括被信号处理器转化来的 `KeyboardInterrupt`）时记录日志，最后在 `finally` 中对剩余进程逐个 `p.kill()` 强杀。

源码注释坦承 "Queues and other inter-process communication primitives can break when process is killed, but we don't care here"——在退出阶段，干脆利落地杀干净比优雅收尾更重要。

信号处理也提前布置好了。`start_main_server` 开头用闭包工厂 `handler(signalname)` 为 `SIGINT` 与 `SIGTERM` 各注册了一个处理器，它们的作用是把系统信号统一转译成 `KeyboardInterrupt` 抛出，从而汇入上面的异常清理路径。

源码注释还解释了用闭包的原因：Python 3.9 才有 `signal.strsignal`，为兼容更早版本，作者用闭包把信号名字提前捕获进 `f`。这样无论用户是按 Ctrl-C 还是 `kill` 进程，都会走同一条优雅停机的通道。

## 启动前建表：create_tables 的位置学问

把视线拉回 `startup.py` 的 click 入口 `main()`，会发现它在调用 `start_main_server` 之前，先做了一件容易被忽略的事：`from chatchat.server.knowledge_base.migrate import create_tables` 后立即 `create_tables()`。

`create_tables` 的实现极简——`Base.metadata.create_all(bind=engine)`，即用 SQLAlchemy 把所有已声明的 ORM 模型对应的表"建好（若不存在）"。

这里的 `Base, engine` 来自 `server/db/base.py`，而 `migrate.py` 顶部刻意 import 了 `ConversationModel`、`MessageModel` 等模型模块，正是为了让它们在 `create_all` 之前被注册进 `Base.metadata`。如果不预先 import，SQLAlchemy 就无从知晓这些表的存在，`create_all` 会漏建。

把建表放在拉起任何子进程之前，保证了 API 与 WEBUI 一启动就能直接读写数据库，不必各自处理"表还没建"的竞态。考虑到 spawn 出来的子进程是全新解释器，若让它们各自建表，反而可能并发触碰同一个数据库文件，父进程统一建表是更稳妥的选择。

`main()` 里还有两处兼容性铺垫。一是 `cwd = os.getcwd(); sys.path.append(cwd)`，把当前工作目录加进模块搜索路径，便于加载用户在运行目录下的自定义内容。

二是 `mp.freeze_support()`，这是在 Windows 上把程序打包成 exe 后正确支持多进程的必要调用，没有它，spawn 出来的子进程在冻结环境里会重复执行入口逻辑。

最后对事件循环的获取也做了版本分支：`sys.version_info < (3, 10)` 时走 `asyncio.get_event_loop()`，否则尝试 `get_running_loop()`、失败再 `new_event_loop()` 并 `set_event_loop`，最终 `loop.run_until_complete(start_main_server(args))` 把异步主流程跑起来。这段分支是为了兼容 3.10 前后 asyncio 在"获取事件循环"语义上的变化。

`init_database.py` 的 `main` 则展示了另一种进程模型。它把实际工作 `worker(kwds)` 放进一个 `daemon=True` 的子进程跑，父进程用 `while p.is_alive(): time.sleep(0.1)` 轮询，捕获 `KeyboardInterrupt` 时 `p.terminate()` 后 `sys.exit()`。

为什么连建表这种"一次性任务"也要丢到子进程？一个合理的解释是：知识库迁移往往涉及加载嵌入模型、读写大量文件，把它隔离到独立进程，能让主进程始终保有响应 Ctrl-C 的能力，中断时干净地 `terminate` 掉子进程即可。

`worker` 内部根据传入开关分派到不同的知识库迁移操作：`create_tables`、`reset_tables`、`folder2db`（按 `recreate_vs / update_in_db / increment` 三种 mode）、`import_from_db`、`prune_db_docs`、`prune_folder_files`，并打印总计用时。这个文件的 `if __name__ == "__main__":` 块同样调用了 `mp.set_start_method("spawn")`，与启动逻辑保持一致的多进程风格。

## 本章小结

- Chatchat 是 monorepo 布局，代码集中在 `libs/`，分为应用本体 `chatchat-server`（含 `chatchat` 与 `langchain_chatchat` 两个包）和可复用 `python-sdk` 子包。
- 真正生效的项目元信息在 `libs/chatchat-server/pyproject.toml`：`version = "0.3.1.3"`，与 `chatchat/__init__.py` 的 `__version__` 一致；根目录 `pyproject.toml` 是 `package-mode = false` 的工具配置占位。
- 命令入口由 `[tool.poetry.scripts]` 的 `chatchat = 'chatchat.cli:main'` 声明；Click 命令组把 `start`、`kb` 从独立模块挂载进来，`init` 定义在 `cli.py` 自身。
- `init` 负责首次部署：建目录、复制 samples、`create_tables`、覆盖默认模型（LLM 默认 `glm4-chat`，embedding 实际默认 `bge-m3`，CLI 帮助文案中的 `bge-large-zh-v1.5` 已与代码脱节）、生成 YAML 模板。
- `start_main_server` 用 `mp.set_start_method("spawn")` 选择跨平台、状态干净的子进程创建方式，规避了 `fork` 复制内存带来的死锁与污染风险。
- 父子进程通过 `Manager().Event()` 创建的 `api_started`、`webui_started` 完成就绪握手，这种事件能可靠跨越 spawn 进程边界。
- API 进程 `daemon=False`、WEBUI 进程 `daemon=True`；二者被串行拉起：先 `start` API 再 `api_started.wait()`，然后才启动 WEBUI 并 `webui_started.wait()`。
- API 的就绪信号经由 FastAPI `lifespan` 钩子里的 `started_event.set()` 精确触发；WEBUI 的 `started_event.set()` 写在阻塞的 `bootstrap.run` 之后，语义并不对称。
- 父进程进入 `while processes` 守护循环以 `p.join(2)` 轮询存活，`finally` 中 `p.kill()` 强制清理；`SIGINT/SIGTERM` 被统一转译为 `KeyboardInterrupt`。
- `startup.py` 的 `main()` 在拉起任何子进程前先 `create_tables()`（`Base.metadata.create_all(bind=engine)`），消除子进程的建表竞态。
- 跨平台与跨版本细节：`mp.freeze_support()` 服务 Windows 打包，事件循环按 `sys.version_info` 分支获取。

## 动手实验

1. **实验一：画出启动时序** — 通读 `start_main_server`，按 `p.start()` 与 `api_started.wait()`/`webui_started.wait()` 的先后顺序，在纸上画出"父进程、API 进程、WEBUI 进程"三条泳道及事件置位的时刻；重点标注 API 事件由 `_set_app_event` 的 `lifespan` 触发、而 WEBUI 事件在 `bootstrap.run` 之后触发这一不对称点。

2. **实验二：验证 spawn 的影响** — 在 `run_api_server` 函数体最前面临时加一行 `print(f"child import, ppid={os.getpid()}")`，再用 `chatchat start --api` 启动；观察该打印是否出现，体会 spawn 模式下子进程会重新执行模块导入、而不是继承父进程内存的行为（验证后请删除该行）。

3. **实验三：追踪建表时机** — 从 `startup.py` 的 `main()` 里的 `create_tables` 调用出发，跳到 `server/knowledge_base/migrate.py` 确认其实现是 `Base.metadata.create_all(bind=engine)`，再回溯 `Base, engine` 来自 `server/db/base.py`；解释为什么把建表放在 `start_main_server` 之前而非子进程内部。

4. **实验四：对比两种进程模型** — 把 `startup.py` 中"父进程亲手 `Process(...)` 起两个子进程"的写法，与 `init_database.py` 中"`main` 起一个 `daemon=True` 的 `worker` 子进程并轮询"的写法并列阅读，列出两者在 daemon 设置、清理方式（`p.kill()` vs `p.terminate()`）、循环间隔（`join(2)` vs `sleep(0.1)`）上的差异，并思考各自适配的场景。

> **下一章预告**：本章多次出现 `Settings.basic_settings`、`Settings.model_settings`、`set_auto_reload`、`createl_all_templates` 这些配置入口——它们正是整个系统的"控制台"。第 2 章将深入配置系统 Settings，拆解 Chatchat 如何用 pydantic-settings 把 YAML 文件、默认值与运行期热重载编织成一套可离线、可覆盖的配置体系。
