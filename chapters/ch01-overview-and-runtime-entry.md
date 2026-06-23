# 第 1 章 项目全景与运行时入口

要读懂一个 RAG 系统，最好的起点不是它的算法，而是它的**进程**——当你 `docker compose up` 之后，到底有哪些程序被拉起、它们各自负责什么、数据在它们之间如何流动。RAGFlow 是一个相当庞大的 monorepo，截至本书解构的版本，它在 `pyproject.toml` 中声明的版本号是 `0.26.1`，并要求 `requires-python = ">=3.13,<3.14"`。在钻进深度文档理解、向量检索、GraphRAG 这些"招牌算法"之前，本章先把整个系统的骨架摊开：它的目录如何分层、它启动时跑起来的是哪几个进程、这些进程依赖哪些外部基础设施。读完本章，你应该能在脑子里画出一张"RAGFlow 运行时拓扑图"，并知道每一块对应源码里的哪个文件。

## 一个 monorepo，两类运行时角色

把 RAGFlow 仓库 clone 下来，根目录下的目录数量足以让人一时不知从何看起。但如果按"运行时角色"而非"技术栈"来归类，结构立刻清晰。核心的几个 Python 包各司其职：

- `api/`——HTTP 服务层。入口是 `api/ragflow_server.py`，Web 应用对象定义在 `api/apps/__init__.py`。
- `rag/`——RAG 的算法主体。它下面又分出 `rag/app`（按文档类型分块的模板）、`rag/nlp`（查询、分词、检索）、`rag/llm`（模型接入）、`rag/svr`（后台任务执行器）、`rag/graphrag`、`rag/raptor.py` 等。
- `deepdoc/`——深度文档理解模块，下分 `deepdoc/parser`（各类格式解析器）和 `deepdoc/vision`（视觉/OCR/版面分析）。
- `agent/`——Agent 编排，包含 `agent/canvas.py`、`agent/component`、`agent/tools`、`agent/sandbox`、`agent/plugin`。
- `common/`——跨模块复用的基础设施，例如 `common/settings.py`、`common/log_utils.py`、`common/config_utils.py`，以及一个很值得注意的 `common/ssrf_guard.py`。
- `mcp/`——Model Context Protocol 的 client/server 实现。
- `web/`——前端（独立的前端工程，本书聚焦后端实现，前端只在必要时提及）。

光看目录还不够。真正决定"系统怎么跑"的是启动脚本。RAGFlow 的容器入口是 `docker/entrypoint.sh`，它揭示了一个关键事实：**RAGFlow 在运行时被拆成两类截然不同的进程**——一个是面向用户的 HTTP 服务，另一个是面向文档的后台任务执行器。这个"双进程模型"是理解整个系统的第一把钥匙。

## 双进程模型：API 服务与任务执行器

在 `docker/entrypoint.sh` 里可以清楚看到这两类进程是如何被分别拉起的。HTTP 服务这一侧，脚本在 `ENABLE_WEBSERVER` 为 1 时先启动 `nginx`，随后在一个 `while true` 循环里反复执行 `"$PY" api/ragflow_server.py`：

```bash
if [[ "${ENABLE_WEBSERVER}" -eq 1 ]]; then
    echo "Starting nginx..."
    /usr/sbin/nginx
    while true; do
        echo "Attempt to start RAGFlow server..."
        "$PY" api/ragflow_server.py ${INIT_SUPERUSER_ARGS}
        echo "RAGFlow python server started."
        sleep 1;
    done &
fi
```

这个 `while true ... sleep 1` 的包裹不是装饰，而是一种朴素但有效的"进程级看门狗"：一旦 Python 服务异常退出，循环会在 1 秒后把它重新拉起。任务执行器那一侧也是同样的处理思路，定义在 `task_exe` 函数里：

```bash
function task_exe() {
    local consumer_id="$1"
    local host_id="$2"
    JEMALLOC_PATH="$(pkg-config --variable=libdir jemalloc)/libjemalloc.so"
    while true; do
        LD_PRELOAD="$JEMALLOC_PATH" \
        "$PY" rag/svr/task_executor.py -i "${host_id}_${consumer_id}" -t "common" &
        wait;
        sleep 1;
    done
}
```

这里有两个细节值得停下来看。其一，任务执行器通过 `LD_PRELOAD` 预加载了 `jemalloc`——文档解析与向量化是典型的内存密集、长时间运行的负载，用 jemalloc 替换默认的 glibc malloc 是为了缓解长跑进程的内存碎片问题。其二，每个 task executor 启动时都被赋予一个形如 `${host_id}_${consumer_id}` 的标识符（通过 `-i` 参数传入），这意味着 RAGFlow 的后台是**可以横向扩展的多消费者模型**——同一主机上可以跑多个 consumer，多台主机又各有自己的 host_id，它们共同从任务队列里消费文档处理任务。

除了开发者直接使用的脚本 `docker/launch_backend_service.sh`，它的 usage 也印证了这一点：`Usage: $0 [ragflow|task_executor|admin|data_sync]...`，并注明"Without arguments, starts ragflow and task_executor"。换句话说，**默认部署形态就是 API 服务 + 任务执行器两者并存**，但二者可以被独立启动、独立扩缩容。这正是把"在线请求"与"离线重活"解耦的经典做法：用户上传文档后立即得到响应，真正耗时的解析、分块、embedding 在后台异步进行。

## API 服务的启动引导：ragflow_server.py

把镜头推近到 `api/ragflow_server.py`，它是 HTTP 服务的真正入口。这个文件的开头就透露了不少工程考量。第一行业务代码之前，它先设置了一个环境变量：

```python
# LiteLLM fetches a model cost map from GitHub during import unless this is set.
# The API server should not block startup on external network access.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
```

这条注释本身就是一份"为什么这样写"的文档：RAGFlow 通过 LiteLLM 接入各家大模型，而 LiteLLM 在 import 时会默认去 GitHub 拉一份模型价格表。对一个需要快速、可靠启动的服务进程来说，让启动过程依赖外部网络是不可接受的，于是用 `LITELLM_LOCAL_MODEL_COST_MAP=True` 强制走本地副本。`rag/svr/task_executor.py` 的开头做了一模一样的处理，注释甚至直接写明"no internet, save about 10s"——可见这是一条被反复应用的工程纪律：**进程启动不应阻塞在外部网络访问上**。

`__main__` 块里的启动序列也很有条理。它先 `init_root_logger("ragflow_server")` 初始化日志，打印那张 ASCII 艺术字的 RAGFlow logo 和版本号，然后依次：

1. 解析命令行参数（`--version`、`--debug`、`--init-superuser`），其中 `--version` 会直接打印版本号并 `sys.exit(0)`，这是给运维探活/版本核对用的轻量路径。
2. 通过 `RuntimeConfig.init_env()` 和 `RuntimeConfig.init_config(...)` 装配运行时配置。
3. 调用 `GlobalPluginManager.load_plugins()` 加载插件。
4. 注册 `SIGINT`/`SIGTERM` 信号处理器。
5. 延迟 1 秒启动 `update_progress` 后台线程，并启动聊天通道服务线程。
6. 最后 `app.run(...)` 把 Quart 应用跑起来。

信号处理器 `signal_handler` 的实现体现了"优雅关闭"的意识——它不是简单地 `sys.exit`，而是先 `shutdown_all_mcp_sessions()` 关掉所有 MCP 会话，再 `stop_event.set()` 通知后台线程停止，等待 1 秒后才退出。这说明 RAGFlow 在进程内同时管理着 HTTP 服务、后台进度线程、MCP 会话等多个生命周期，关闭时需要协调它们一起收尾。

值得一提的是那个 `update_progress` 线程。它的实现用到了 `RedisDistributedLock("update_progress", ...)`：

```python
def update_progress():
    lock_value = str(uuid.uuid4())
    redis_lock = RedisDistributedLock("update_progress", lock_value=lock_value, timeout=60)
    while not stop_event.is_set():
        try:
            if redis_lock.acquire():
                DocumentService.update_progress()
                redis_lock.release()
        ...
            stop_event.wait(6)
```

为什么更新文档处理进度要加一把 Redis 分布式锁？因为如前所述，RAGFlow 可以部署多个 API 服务实例，但"刷新所有文档的处理进度"这件事在全局只需要有一个实例去做。用一把带 60 秒超时、值为随机 UUID 的分布式锁，保证同一时刻只有一个实例在跑 `DocumentService.update_progress()`，每轮间隔 6 秒（`stop_event.wait(6)`）。这是一个很小但很典型的分布式协调细节。

## Web 应用对象：Quart 而非 Flask

`app` 这个 Quart 应用对象定义在 `api/apps/__init__.py`。这里有个容易被忽略却很重要的事实：RAGFlow 用的是 **Quart**（`from quart import Blueprint, Quart, request, ...`），而不是更广为人知的 Flask。Quart 与 Flask API 基本兼容，但底层是 `asyncio` 异步模型。对一个要长时间等待大模型流式响应的 RAG 服务而言，异步框架几乎是必然选择。

紧接着的几行配置把这层意图坐实了：

```python
app.config["RESPONSE_TIMEOUT"] = int(os.environ.get("QUART_RESPONSE_TIMEOUT", 600))
app.config["BODY_TIMEOUT"] = int(os.environ.get("QUART_BODY_TIMEOUT", 600))
```

注释解释得很直白：Quart 默认的超时是 60 秒，对很多 LLM 后端（尤其是在 CPU 上跑的本地 Ollama）来说太短了，于是把响应和请求体超时都默认放宽到 600 秒。这是一个被真实生产场景倒逼出来的默认值——**默认值的选择往往藏着对运行环境的假设**，这里的假设是"模型可能很慢，框架不能先于模型超时"。

模块加载时还立即执行了 `settings.init_settings()`，并配置了基于 Redis 的会话存储（`app.config["SESSION_TYPE"] = "redis"`），以及全局的错误处理器 `app.errorhandler(Exception)(server_error_response)`——任何未捕获的异常都会被统一转成结构化的错误响应，而不是把栈直接抛给客户端。

## 运行时拓扑：它依赖的那些基础设施

RAGFlow 不是一个自给自足的单体，它把"重活"外包给了一组成熟的基础设施。这些依赖在 `docker/docker-compose-base.yml` 里一览无余，它定义了 RAGFlow 运行所需的全部底座服务（镜像名摘自该文件）：

- **检索/向量存储**（三选一的文档存储后端）：`elasticsearch`、`opensearchproject/opensearch:2.19.1`、或 `infiniflow/infinity:v0.7.0`。RAGFlow 支持多种 doc store，这也是为什么 `common/` 下会有 `metadata_es_filter.py` 和 `metadata_infinity_filter.py` 两套元数据过滤实现——不同后端的查询语法不同，需要各自适配。
- **关系数据库**：`mysql:8.0.39`（也支持 OceanBase 系列），存放用户、知识库、文档元信息等结构化数据。
- **对象存储**：MinIO（`pgsty/minio:...`），存放上传的原始文档与切分产物。
- **缓存/队列/锁**：`valkey/valkey:8`（Redis 协议兼容），承担前面提到的会话存储、分布式锁、任务队列等职责。
- **消息**：`nats:2.14.2`。
- **嵌入服务**：TEI（Text Embeddings Inference，CPU/GPU 两个镜像变体）。
- **沙箱执行器**：`infiniflow/sandbox-executor-manager`，对应 README 中提到的代码执行特性，且 README 明确指出它需要 [gVisor](https://gvisor.dev/docs/user_guide/install/) 才能启用。

把这些拼起来，一张完整的运行时拓扑就浮现了：用户请求经 nginx 进入 `ragflow_server.py`（Quart）；文档上传后落到 MinIO，元信息写进 MySQL，任务投递到 Redis/NATS；后台的 `task_executor.py` 消费任务，调用 `deepdoc` 解析、`rag` 分块与向量化，把向量与文本写进 ES/OpenSearch/Infinity；检索时再从这些 doc store 召回、重排、交给 `rag/llm` 接入的大模型生成答案。后续每一章，本质上都是在放大这张拓扑图里的某一个环节。

## 本章小结

- RAGFlow 是一个 monorepo，按运行时角色可分为 `api`（HTTP 服务）、`rag`（算法主体）、`deepdoc`（文档理解）、`agent`（编排）、`common`（基础设施）、`mcp`、`web` 等核心包；解构版本为 `pyproject.toml` 声明的 `0.26.1`，要求 Python `>=3.13,<3.14`。
- 系统在运行时是**双进程模型**：`api/ragflow_server.py` 提供在线 HTTP 服务，`rag/svr/task_executor.py` 在后台异步处理文档；二者可独立启停与扩缩容，由 `docker/entrypoint.sh` 与 `docker/launch_backend_service.sh` 编排。
- 两类进程都用 `while true ... sleep 1` 做进程级看门狗自动重启；task executor 用 `LD_PRELOAD` 预加载 `jemalloc` 应对内存密集长跑负载，并以 `${host_id}_${consumer_id}` 标识支持多消费者扩展。
- 两个入口都在 import 阶段设置 `LITELLM_LOCAL_MODEL_COST_MAP=True`，避免启动阻塞在外部网络——"启动不依赖外网"是一条被反复应用的工程纪律。
- HTTP 层用的是 **Quart**（异步）而非 Flask，并把请求/响应超时默认放宽到 600 秒以适配可能很慢的 LLM 后端。
- `update_progress` 用 `RedisDistributedLock` 保证多实例下全局只有一个实例刷新文档进度，是典型的分布式协调细节。
- RAGFlow 把重活外包给一组基础设施：ES/OpenSearch/Infinity（doc store 三选一）、MySQL/OceanBase、MinIO、Valkey(Redis)、NATS、TEI 嵌入服务、以及需要 gVisor 的沙箱执行器。

## 动手实验

1. **实验一：画出你自己的运行时拓扑图** — 打开 `docker/docker-compose-base.yml`，逐个数出其中定义的 service 及其镜像；再打开 `docker/entrypoint.sh`，找出 `ragflow_server.py` 与 `task_executor.py` 分别在哪个分支被启动。把"进程"和"基础设施"两类节点画在一张图上，用箭头标出文档上传 → 解析 → 入库 → 检索的数据流向。

2. **实验二：验证"双进程模型"** — 在 `docker/launch_backend_service.sh` 中定位 usage 字符串，确认 `ragflow|task_executor|admin|data_sync` 四种可独立启动的角色。再在 `task_exe` 函数里找到 `-i "${host_id}_${consumer_id}"` 这一行，思考：如果你想在一台机器上跑 4 个文档处理消费者，应该如何设置 `WS`（worker 数）这个环境变量？（提示：看脚本里 `WS` 的默认值处理逻辑。）

3. **实验三：追踪"启动不依赖外网"这条纪律** — 在仓库里用 `grep -rn "LITELLM_LOCAL_MODEL_COST_MAP"` 搜索，列出所有设置该变量的文件。阅读每处旁边的注释，总结 RAGFlow 为什么如此在意"import 阶段不访问外部网络"。再想想：如果去掉这一行，对一个没有外网的私有化部署会发生什么？

4. **实验四：读懂优雅关闭** — 阅读 `api/ragflow_server.py` 中的 `signal_handler` 函数，列出它在收到 `SIGTERM` 时依次做了哪几件事（提示：MCP 会话、stop_event、等待、退出）。再找到 `update_progress` 函数，解释 `RedisDistributedLock` 的 `timeout=60` 和 `stop_event.wait(6)` 这两个数字各自的含义。

> **下一章预告**：RAGFlow 的招牌是"深度文档理解"——它能从 PDF、扫描件、PPT、Excel 这些复杂格式里"看懂"版面并抽取结构化知识。下一章我们钻进 `deepdoc/`，从 `deepdoc/parser` 的各类解析器和 `deepdoc/vision` 的 OCR 与版面分析入手，解构 RAGFlow 是如何把一份杂乱的原始文档变成可被分块和检索的干净文本的。
