# 第 10 章 沙箱与代码执行

到上一章为止，我们看到的所有数据流——解析、分块、向量化、检索、图谱、画布编排——处理的都是“被动的数据”。但 RAGFlow 的 agent 画布里有一个特殊的组件 `CodeExec`，它把 LLM 生成的、或用户写下的一段 Python / JavaScript 源码当作执行体跑起来，再把返回值接回工作流。这是一种本质不同的能力：系统不再只是读写数据，而是要在自己的机器上**运行不可信的代码**。一旦这段代码能直接命中宿主机，攻击者就能读 `/etc/passwd`、发起反向 shell、删库、挖矿、把整台机器的内存吃光——RAG 服务瞬间变成入侵跳板。

正因为如此，RAGFlow 没有把代码执行做成一个轻量的 `eval()`，而是单独建了一个子系统 `agent/sandbox/`，用一整套纵深防御把“执行不可信代码”这件事关进笼子里。本章就沿着一段代码从 agent 画布出发、穿过客户端、provider、执行器管理器、最终落进一个 gVisor 隔离容器里的全过程，逐层解构它的安全设计意图：哪些是边界、为什么默认拒绝、攻击在哪一层被挡下。

## 从画布组件到沙箱客户端：一次调用的起点

代码执行能力在 agent 画布中以 `CodeExec` 组件的形式暴露，实现位于 `agent/tools/code_exec.py`。组件参数 `CodeExecParam` 在 `meta` 里给 LLM 写明了契约——“This tool has a sandbox that can execute code written in 'Python'/'Javascript'”，并强制要求代码必须包含一个 `main` 函数，Python 返回 dict、JavaScript 必须 `module.exports = { main }`。组件被触发时，`_invoke` 收集脚本与参数，转交给 `_execute_code`。

`_execute_code` 的第一选择不是直接发 HTTP，而是走新的 provider 体系。它从 `agent.sandbox.client` 导入 `execute_code`、`get_provider_info`、`reload_provider`，先 `reload_provider()` 重新加载配置，再把代码派发出去：

```python
from agent.sandbox.client import execute_code as sandbox_execute_code
...
reload_provider()
provider_info = get_provider_info()
provider_type = provider_info.get("provider_type") or "unknown"
...
result = sandbox_execute_code(code=code, language=language, timeout=timeout_seconds, arguments=arguments)
```

这里有一处很关键的 fail-closed 设计：如果 provider 体系抛出 `SandboxProviderConfigError`（即“管理员明确选了某个 provider，但它不可用”），`code_exec.py` 不会偷偷降级，而是直接把错误写进 `_ERROR` 输出并返回。只有在 `ImportError`（provider 模块根本不存在）这种情况下，它才退回到旧的“直接 HTTP 请求 `http://{settings.SANDBOX_HOST}:9385/run`”的兼容路径。换句话说，“配置错了”和“没装这套东西”被刻意区分开：前者宁可报错也不冒险用一个未经确认的后端跑代码。

`agent/sandbox/client.py` 是组件与具体执行后端之间的解耦层。它维护一个全局 `_provider_manager`，并通过 `_resolve_provider_type()` 从系统设置 `sandbox.provider_type` 里读出当前激活的 provider 类型，缺省值是 `"self_managed"`。随后 `_load_provider_config` 读取 `sandbox.{provider_type}` 对应的 JSON 配置，把 provider 实例化并 `initialize`。`client.py` 里登记的 provider 类一共有五种：

```python
provider_classes = {
    "self_managed": SelfManagedProvider,
    "aliyun_codeinterpreter": AliyunCodeInterpreterProvider,
    "e2b": E2BProvider,
    "local": LocalProvider,
    "ssh": SSHProvider,
}
```

注意 `_load_provider_from_settings` 里对初始化失败的处理也是分级的：当 `local` 或 `ssh` 这两个 provider 初始化失败时，会抛 `SandboxProviderConfigError` 终止；而其余 provider 失败则记录日志后返回。`local` 和 `ssh` 之所以被特殊对待，是因为它们的安全假设最脆弱（后面会看到 `LocalProvider` 自己都在注释里承认“it is not a sandbox boundary”），配置不对必须立刻让使用者知道。

真正决定“代码在哪儿、怎么跑”的对外入口是 `client.py` 的 `execute_code`：

```python
def execute_code(code, language="python", timeout=30, arguments=None):
    provider_manager = get_provider_manager()
    if not provider_manager.is_configured():
        raise RuntimeError("No sandbox provider configured. ...")
    provider = provider_manager.get_provider()
    ...
    instance = provider.create_instance(template=language)
    try:
        result = provider.execute_code(instance_id=instance.instance_id, code=code, ...)
        return result
    finally:
        provider.destroy_instance(instance.instance_id)
```

这段代码体现了两个朴素但重要的工程纪律。其一，没有配置 provider 就直接 `raise`，绝不静默吞掉——这又是 fail-closed。其二，无论执行成功还是抛异常，`finally` 块都会调用 `destroy_instance` 销毁实例。对沙箱来说，“用完即焚”不是性能优化，而是安全本身：每次执行都在一个生命周期受控的实例里完成，不让上一段代码留下的状态污染下一段。

## provider 抽象：把“在哪执行”做成可插拔接口

所有执行后端都实现 `agent/sandbox/providers/base.py` 里的抽象基类 `SandboxProvider`。它定义了一组最小契约：`initialize`、`create_instance`、`execute_code`、`destroy_instance`、`health_check`、`get_supported_languages`，以及用于配置校验的 `get_config_schema` 与 `validate_config`。执行结果统一封装为 `ExecutionResult` dataclass，包含 `stdout`、`stderr`、`exit_code`、`execution_time` 和 `metadata`。这种抽象的价值在于：上层 `client.py` 与 `code_exec.py` 完全不关心代码究竟是跑在本机进程、远程 SSH 主机、自管 Docker 池，还是阿里云 / E2B 的托管沙箱里——安全强度由具体 provider 决定，而调用方代码一行不用改。

最值得对比的是 `LocalProvider`（`local.py`）与 `SelfManagedProvider`（`self_managed.py`）。前者是“直接在本机起子进程”，类文档字符串写得很直白：

```python
class LocalProvider(SandboxProvider):
    """
    Execute code as a local child process.
    This provider is intentionally gated by SANDBOX_LOCAL_ENABLED because it is
    not a sandbox boundary. Use a low-privilege runtime account.
    """
```

它“不是一个沙箱边界”，所以默认被开关 `SANDBOX_LOCAL_ENABLED` 关着。但即便在这种弱隔离模式下，`LocalProvider` 仍然尽力施加了多重资源限制：通过 `subprocess.Popen` 的 `preexec_fn=self._limit_child_process` 在子进程里用 `resource.setrlimit` 设置 CPU 时间（`RLIMIT_CPU`）、地址空间内存（`RLIMIT_AS`）、单文件大小（`RLIMIT_FSIZE`）、以及打开文件描述符上限 64（`RLIMIT_NOFILE`）；用 `start_new_session=True` 把子进程放进独立进程组，超时后用 `os.killpg(process.pid, signal.SIGKILL)` 整组杀死，避免漏网的孙进程；还用 `_build_child_env` 把环境变量收窄到只剩 `HOME`、`PATH`、`TMPDIR` 等几项，并强制 `MPLBACKEND=Agg`。`_set_resource_limit` 还做了 `min(value, hard)` 的处理，绝不尝试抬高硬上限。这些细节说明：哪怕承认自己不是强隔离，作者也按“最小权限”的原则去收缩攻击面。

`SelfManagedProvider` 则是 RAGFlow 推荐的默认形态。它本身只是个 HTTP 客户端，把代码 base64 编码后 POST 给 `http://sandbox-executor-manager:9385/run`，真正的隔离逻辑全在那个独立的 `executor_manager` 服务里：

```python
code_b64 = base64.b64encode(code.encode("utf-8")).decode("utf-8")
payload = {"code_b64": code_b64, "language": normalized_lang, "arguments": arguments or {}}
url = f"{self.endpoint}/run"
response = requests.post(url, json=payload, timeout=exec_timeout, ...)
```

把执行器拆成一个独立进程/容器，本身就是安全设计：RAGFlow 主服务与“跑不可信代码的地方”被进程边界和网络边界隔开，即使沙箱内部被攻破，攻击者拿到的也只是 executor manager 那个受限容器，而不是 RAGFlow 主应用。下面几节我们就钻进 `executor_manager` 内部，看它如何用 gVisor、容器池和静态分析层层设防。

## 第一道闸：执行前的 AST / 正则静态审查

代码进入 `executor_manager` 后，第一个接触它的不是容器，而是 `api/handlers.py` 里的 `run_code_handler`。在真正分配容器之前，它先做静态安全审查：

```python
@limiter.limit("5/second")
async def run_code_handler(req: CodeExecutionRequest, request: Request):
    async with _CONTAINER_EXECUTION_SEMAPHORES[req.language]:
        code = base64.b64decode(req.code_b64).decode("utf-8")
        ...
        is_safe, issues = analyze_code_security(code, language=req.language)
        if not is_safe:
            issue_details = "\n".join([f"Line {lineno}: {issue}" for issue, lineno in issues])
            return CodeExecutionResult(status=ResultStatus.PROGRAM_RUNNER_ERROR, ..., exit_code=-999, detail="Code is unsafe")
```

审查逻辑在 `services/security.py`。Python 走的是 AST 路线：`SecurePythonAnalyzer` 继承 `ast.NodeVisitor`，对解析出的语法树做遍历。它维护两张黑名单——`DANGEROUS_IMPORTS` 涵盖 `os`、`subprocess`、`sys`、`shutil`、`socket`、`ctypes`、`pickle`、`threading`、`multiprocessing`、`asyncio`、`http.client`、`ftplib`、`telnetlib`、`builtins`；`DANGEROUS_CALLS` 涵盖 `eval`、`exec`、`open`、`__import__`、`compile`、`input`、`system`、`popen`、`remove`、`chmod`、`getattr`、`setattr`、`globals`、`locals`，以及 `subprocess.Popen`、`pickle.loads` 等属性式调用名。

它不只盯着 `import os` 这种最直白的写法，而是把 AST 的多种节点都覆盖到：`visit_Import` / `visit_ImportFrom` 拦截危险导入；`visit_Call` 同时处理 `ast.Name`（直接调用 `eval(...)`）和 `ast.Attribute`（属性式调用 `builtins.exec(...)`，并额外打一条 warning 日志方便事后审计）；`visit_Attribute` 拦截 `os.system` 这类属性访问；`visit_FunctionDef` 拦截用户自定义一个名叫 `eval` 的函数来混淆；`visit_Lambda`、`visit_ListComp`、`visit_DictComp`、`visit_SetComp`、`visit_Yield` 则覆盖把危险调用藏进 lambda 和各种推导式里的绕过手法；甚至 `visit_BinOp` 会把两个常量字符串相加（`"os." + "system"`）标记为“可能的不安全拼接”，专门针对 `eval("os." + "system")` 这种拼装攻击。

JavaScript 没有现成的轻量 AST 库可用，所以 `SecureJavaScriptAnalyzer` 退而用正则匹配 `DANGEROUS_PATTERNS`：`require('child_process')`、`require('fs')`、`require('worker_threads')`、`eval(`、`Function(`、`process.binding(`。值得注意的是它的正则用 `['"`+\`]` 同时覆盖了单引号、双引号和反引号，因为 `require(\`child_process\`)` 这种模板字符串写法会绕过只匹配引号的简单正则——`tests/test_security.py` 里专门有 `test_javascript_child_process_template_literal_is_rejected` 和 `test_javascript_fs_template_literal_is_rejected` 两个用例来钉死这个边界。

这一层的设计意图需要看清楚：AST/正则审查**不是**沙箱的主防线，而是“便宜的早期拒绝”。`README.md` 里把它定位为“an extra layer of protection”——在代码还没碰到容器、没占用任何执行资源之前，就把明显恶意的请求驳回，既省资源也缩短攻击窗口。真正兜底的隔离仍然是后面的 gVisor。`analyze_code_security` 对“无法解析的 Python”（`ast.parse` 抛异常）和“不支持的语言”都返回 `False`，这同样是 fail-closed——看不懂的代码一律当成不安全。

## 核心隔离：gVisor runsc 容器池

通过静态审查后，请求落到 `services/execution.py` 的 `execute_code`，它先从容器池里 `allocate_container_blocking` 借一个空闲容器。容器池由 `core/container.py` 管理，启动时由 `core/config.py` 的 `_lifespan` 调 `init_containers(size)` 预热——`size` 来自环境变量 `SANDBOX_EXECUTOR_MANAGER_POOL_SIZE`（`docker-compose.yml` 默认 5）。池子按语言分成 Python 与 Node.js 两个 `Queue`，并各自配一个 `asyncio.Semaphore(size)` 来限制并发。

整章最该精读的，是每个沙箱容器是怎么被创建出来的。`create_container` 拼出的 `docker run` 参数集中体现了这套系统对“不可信代码运行环境”的全部假设：

```python
create_args = [
    "docker", "run", "-d",
    "--runtime=runsc",
    "--name", name,
    "--read-only",
    "--tmpfs", "/workspace:rw,exec,size=100M,uid=65534,gid=65534",
    "--tmpfs", "/tmp:rw,exec,size=50M",
    "--user", "nobody",
    "--workdir", "/workspace",
]
```

逐项拆开看它在防什么。`--runtime=runsc` 是整套隔离的基石：`runsc` 是 gVisor 的容器运行时，它在用户态实现了一个独立的内核（Sentry），拦截容器内进程发出的系统调用并在用户态重新实现，而不是直接转发给宿主机内核 [[gVisor Security Model]](https://gvisor.dev/docs/architecture_guide/security/)。这等于在不可信代码和宿主机内核之间多插了一层“软内核”，把宿主机内核暴露给攻击者的攻击面大幅收窄，对容器逃逸和提权这类经典攻击提供了远超普通 `runc` 容器的防护 [[What is gVisor]](https://gvisor.dev/docs/)。`README.md` 把它概括为“a user-space kernel, to isolate code execution from the host system”，FAQ 里也专门给出 `docker run --rm --runtime=runsc hello-world` 来验证 gVisor 是否装好。

`--read-only` 把容器根文件系统设为只读，不可信代码无法篡改镜像里的任何文件；唯一可写的是两块 `tmpfs`：100M 的 `/workspace`（带 `uid=65534,gid=65534`，即 `nobody`）和 50M 的 `/tmp`。tmpfs 是内存文件系统，容器销毁即蒸发，天然做到“执行痕迹不落盘”。`--user nobody` 让代码以非特权用户运行，配合 `docker-compose.yml` 里 executor manager 自身的 `no-new-privileges:true`，进一步堵死提权路径。`--memory` 默认 `256m`（由 `SANDBOX_MAX_MEMORY` 控制，并经 `is_valid_memory_limit` 用正则 `[1-9]\d*(b|k|m|g)` 校验合法性，非法则回落到 256m），把内存耗尽攻击的天花板钉死。

还有一个细节藏着设计意图：可选的 seccomp。当 `SANDBOX_ENABLE_SECCOMP` 打开时，`create_container` 会追加 `--security-opt seccomp=/app/seccomp-profile-default.json`。这个 profile 的 `defaultAction` 是 `SCMP_ACT_ERRNO`（默认拒绝一切系统调用），只放行白名单里列出的 `read`、`write`、`mmap`、`openat`、`clock_gettime` 等少数 syscall。这是教科书式的白名单（allowlist）安全模型——默认拒绝，按需放行。但它默认是关的，`README.md` 解释原因是“Enabling it may cause compatibility issues with some dependencies”：作者在“零信任级 syscall 控制”和“开箱即用的开发者体验”之间做了取舍，把最严格的那档留给愿意承担兼容成本的高安全场景，而日常默认依赖 gVisor 提供的隔离。

容器并非用一次就扔，而是回收复用：`release_container` 把容器还回队列前会先 `container_is_running` 检查存活，崩溃的容器则 `recreate_container` 重建后再入队。Node.js 容器创建后还要把 `/app/node_modules` 拷进可写的 `/workspace`，因为根文件系统是只读的。这种“预热池 + 健康检查 + 自愈重建”的组合，兼顾了启动延迟与隔离纯净度。

## 代码注入与执行：runner 包裹与超时双保险

借到容器后，`execute_code` 并不会直接把用户代码丢进去跑，而是用 `_build_execution_bundle` 在宿主机临时目录 `/tmp/sandbox_{task_id}`（权限 `0o700`）里组装一个三件套：用户代码 `main.py`/`main.js`、一个 `runner.py`/`runner.js` 包裹器、以及参数文件 `args.json`。runner 负责从 `args.json` 读参数、调用用户的 `main(**args)`、再把返回值用一个固定前缀打印出来。

这三个文件不是用 `docker cp` 而是用 `tar` 流式注入容器：宿主机 `tar czf -` 打包后，通过 `docker exec -i ... tar xzf -` 解到容器内 `/workspace/{task_id}`。这么做有现实原因——`_collect_artifacts` 的注释指出 “docker cp doesn't work with gVisor tmpfs”，gVisor 的 tmpfs 不支持 `docker cp`，所以读写产物都改走 `docker exec` 内部命令。

真正运行用户代码的命令由 `_build_container_run_args` 拼出，超时控制用了双保险：

```python
run_args = ["docker", "exec", "--workdir", f"/workspace/{task_id}", container,
            "timeout", str(TIMEOUT), language]
if language == SupportLanguage.PYTHON:
    run_args.extend(["-I", "-B"])
run_args.append(runner_name)
```

外层 `timeout str(TIMEOUT)` 用 GNU `timeout` 命令在容器内强制限时（`TIMEOUT` 由 `SANDBOX_TIMEOUT` 解析，默认 `10s`）；与此同时，`async_run_command(*run_args, timeout=TIMEOUT + 5)` 又在宿主机侧设了一个比容器内多 5 秒的兜底超时。两道超时一里一外，即使容器内的 `timeout` 因某种原因失效，宿主机侧的 `asyncio.TimeoutError` 也会触发，并执行 `docker exec container pkill -9 language` 把进程强杀。Python 解释器特意加了 `-I`（isolated mode，忽略环境变量与用户 site-packages）和 `-B`（不写 `.pyc`），进一步收紧运行时行为。

执行结束后，`execute_code` 按退出码精确分类资源耗尽类型：`returncode == 124` 是 GNU `timeout` 的超时信号，映射为 `ResourceLimitType.TIME`；`returncode == 137`（128+9，即被 OOM killer 用 SIGKILL 干掉）映射为 `ResourceLimitType.MEMORY`。其他非零退出则交给 `analyze_error_result`，它从 stderr 文本里识别 `"Permission denied"`（文件越权 → `UnauthorizedAccessType.FILE_ACCESS`）、`"Operation not permitted"`（被拦截的 syscall → `DISALLOWED_SYSCALL`）、`"MemoryError"`（内存 → `MEMORY`）。这套分类让上层能区分“代码本身写错了”和“代码试图越界被沙箱挡了”，对运维排查和安全审计都很有用。整个流程的 `finally` 块同样践行用完即焚：并发删除容器内的 `/workspace/{task_id}` 和宿主机的临时目录，再 `release_container` 把容器还池。

## 结果回传：带外标记与产物白名单

不可信代码的输出也是攻击面之一——代码可以往 stdout 打印任意内容，包括伪造的“结果”。RAGFlow 用一个带外标记协议来安全地把 `main()` 的真实返回值从噪声里捞出来。`result_protocol.py` 与 `execution.py` 都约定了同一个前缀常量 `RESULT_MARKER_PREFIX = "__RAGFLOW_RESULT__:"`。runner 把 `main()` 的返回值包成 `{"present": True, "value": ..., "type": "json"}`，JSON 序列化后再 base64 编码，最后以该前缀打印一行。回传侧的 `_extract_result_envelope` 逐行扫描 stdout，只对以该前缀开头的行做 base64 解码并用 `ExecutionStructuredResult.model_validate_json` 严格校验，其余行原样保留为普通 stdout：

```python
for line in str(stdout).splitlines():
    if line.startswith(RESULT_MARKER_PREFIX):
        payload_b64 = line[len(RESULT_MARKER_PREFIX):].strip()
        ...
        payload = base64.b64decode(payload_b64).decode("utf-8")
        envelope = ExecutionStructuredResult.model_validate_json(payload)
```

base64 + 固定前缀 + Pydantic 校验三件套，既能区分“结构化结果”与“普通日志”，又能防止用户代码随便打印一行就冒充结果——解析失败会被 `try/except` 兜住，降级成普通 stdout 而不会让整个回传崩掉。

代码产生的文件产物（图表、PDF、CSV 等）走 `_collect_artifacts`，这里的防御同样密集。它只在容器内 `/workspace/{task_id}/artifacts` 目录下用 `find -maxdepth 1 -type f` 列文件，然后对文件名做反越权清洗——拒绝包含 `/`、`\`、`..` 或以 `.` 开头的名字；对扩展名做白名单 `ALLOWED_ARTIFACT_EXTENSIONS`（仅 `.png`/`.jpg`/`.jpeg`/`.svg`/`.pdf`/`.csv`/`.json`/`.html`，各自映射固定 MIME）；用 `MAX_ARTIFACT_COUNT = 10` 限制数量、`MAX_ARTIFACT_SIZE = 10MB` 限制单文件大小；读取内容也是用 `docker exec container base64 file_path`（同样因为 gVisor tmpfs 不能用 `docker cp`）。`LocalProvider._collect_artifacts` 还额外拒绝 symlink（`if path.is_symlink(): raise`），防止用软链接指向 `/etc/passwd` 之类敏感文件偷渡出来。这些产物最终由 `code_exec.py` 的 `_upload_artifacts` 存进对象存储桶 `SANDBOX_ARTIFACT_BUCKET`，并由 `_ensure_bucket_lifecycle` 设置 `SANDBOX_ARTIFACT_EXPIRE_DAYS` 天的自动过期，避免不可信代码产物在系统里无限堆积。

## 把防线串起来：安全测试里看见的攻击图谱

一套安全系统到底防住了什么，最诚实的答案往往写在它的测试里。`tests/sandbox_security_tests_full.py` 的 `get_test_cases` 是一份攻击清单，每条都标了 `should_fail`，把前面所有防线串成了一张可验证的图谱。

死循环 / CPU 耗尽（用例 1-6，多个 `while True: pass` 并发提交）依赖容器内外双重 `timeout` 强制终止，退出码 124 被识别为 `RESOURCE_LIMIT_EXCEEDED`。内存耗尽（用例 17，试图分配 300MB 超过 256m 上限）被 `--memory` 限制触发 OOM，退出码 137 被识别为内存超限。危险导入（用例 11 `import os`、用例 12 `from subprocess import Popen`）和危险调用（用例 13 `eval(...)`、用例 14 `shutil.rmtree`）在 AST 审查阶段就被拒，根本进不了容器。最能体现“防绕过”用心的是用例 15 和 16：前者把命令字符串拆成 `"os." + "system"` 再 `eval`，被 `visit_BinOp` 的常量拼接检测拿下；后者把 `eval` 藏进一个自定义函数 `eval_function` 里，被 `visit_FunctionDef` / `visit_Call` 配合识破。

与之对照的是“正常用例应当通过”——用例 7-10 覆盖了无依赖 Python、带 pandas 的 Python、用 `https`/`axios` 发网络请求的 Node.js。这里透露出一个重要的边界选择：**沙箱默认并不禁止出站网络**。Node.js 用例直接访问 `https://example.com/` 且 `should_fail=False`，`README.md` 的 JavaScript 示例也演示了 `axios.get('https://github.com/...')`。这与很多“断网沙箱”不同，是 RAGFlow 为了支持“agent 写代码去抓取/调用外部 API”这类真实场景做的有意取舍——它选择用 gVisor 的内核级隔离、非特权用户、只读文件系统、资源上限和静态审查来兜底，而非简单地一刀切断网。`models/enums.py` 里虽然定义了 `UnauthorizedAccessType.NETWORK_ACCESS` 这个枚举值，但在默认配置下网络访问是被允许的，需要靠部署侧的网络策略或 seccomp 才能进一步收紧。理解这个边界，对评估 RAGFlow 沙箱的实际安全等级至关重要。

至此，一段不可信代码从 agent 画布出发，经过 client 的 fail-closed 派发、provider 抽象的后端选择、AST/正则静态审查的早期拒绝、gVisor runsc 容器的内核级隔离、双重超时与资源上限的运行时约束、带外标记与产物白名单的安全回传——每一层都默认怀疑输入、默认拒绝越界，共同构成了 RAGFlow “安全执行不可信代码”的纵深防御。

## 本章小结

- RAGFlow 把代码执行单独建为 `agent/sandbox/` 子系统，因为运行 LLM 生成的不可信代码若不隔离，会直接把 RAG 服务变成入侵跳板。
- 调用链是 `CodeExec` 组件（`agent/tools/code_exec.py`）→ `client.py` 的 `execute_code` → 具体 `SandboxProvider` → `executor_manager`；各层用统一的 `ExecutionResult` 解耦，安全强度由 provider 决定。
- 全程贯彻 fail-closed：没配 provider 直接 `raise`，`local`/`ssh` 配置失败抛 `SandboxProviderConfigError`，看不懂的代码一律判为不安全，`finally` 里 `destroy_instance` 用完即焚。
- 执行前有一道便宜的静态审查（`services/security.py`）：Python 用 `ast.NodeVisitor` 黑名单覆盖导入、调用、属性、推导式、lambda、常量拼接等多种绕过；JavaScript 用正则并兼容反引号模板字符串。
- 核心隔离是 gVisor `--runtime=runsc`，在用户态实现独立内核拦截 syscall，配合 `--read-only`、`tmpfs`、`--user nobody`、`--memory 256m`、`no-new-privileges` 构成最小权限运行环境。
- seccomp 白名单（`defaultAction: SCMP_ACT_ERRNO`）作为可选的“零信任级”加固默认关闭，是“安全 vs 开发者体验”的有意取舍。
- 容器以预热池形式复用，借还都做健康检查，崩溃则重建；代码与产物因 gVisor tmpfs 不支持 `docker cp` 而改用 `tar` 流和 `docker exec base64` 进出。
- 超时是容器内 GNU `timeout` 与宿主机 `asyncio` 双保险，退出码 124/137 被精确映射为时间/内存超限，stderr 文本被分类为文件越权 / 非法 syscall。
- 结果回传用带外标记 `__RAGFLOW_RESULT__:` + base64 + Pydantic 校验，防止用户代码用普通 stdout 伪造结构化结果。
- 产物收集做了文件名反越权清洗、扩展名白名单、数量/大小上限、symlink 拒绝，并设对象存储桶生命周期自动过期。
- 安全测试 `sandbox_security_tests_full.py` 验证了死循环、内存耗尽、危险导入/调用、拼接绕过等攻击均被拦；同时确认默认**允许出站网络**，这是支持 agent 联网编码的有意边界选择。

## 动手实验

1. **实验一：画出一次执行的防线时序** — 从 `agent/tools/code_exec.py` 的 `_execute_code` 入手，沿 `client.py::execute_code` → `self_managed.py::execute_code` → `executor_manager/api/handlers.py::run_code_handler` → `services/execution.py::execute_code` 跟读，画一张时序图，在每一跳标注它施加的安全控制（fail-closed / AST 审查 / 容器隔离 / 超时 / 产物白名单），体会“纵深防御”是如何分层落地的。

2. **实验二：给 AST 审查找绕过** — 阅读 `services/security.py` 的 `SecurePythonAnalyzer`，列出它覆盖的所有 AST 节点类型；然后构造几段“逻辑上能执行危险操作但黑名单未覆盖”的 Python（例如通过 `__class__.__bases__` 链或 f-string 间接构造），对照 `DANGEROUS_CALLS`/`DANGEROUS_IMPORTS` 判断哪些会漏网。再回想：即便绕过了这道静态审查，后面的 gVisor + 非特权用户 + 只读文件系统还能挡住什么？

3. **实验三：核对 docker run 的每一个隔离参数** — 把 `core/container.py::create_container` 拼出的 `create_args` 完整列出，逐项对照 gVisor 官方安全文档 [[gVisor Security Model]](https://gvisor.dev/docs/architecture_guide/security/) 解释 `--runtime=runsc`、`--read-only`、`--tmpfs`、`--user nobody`、`--memory`、`no-new-privileges` 各自防御的攻击类型；再打开 `seccomp-profile-default.json`，统计白名单里放行了多少个 syscall，思考为什么它默认是关闭的。

4. **实验四：复盘攻击测试用例** — 通读 `tests/sandbox_security_tests_full.py` 的 `get_test_cases` 与 `tests/test_security.py`，把每个 `should_fail=True` 的用例映射到“它在哪一层被拦下”（AST 审查 / 超时 / 内存上限 / 容器隔离）；特别关注用例 9、10 这两个 `should_fail=False` 的联网用例，结合 `models/enums.py` 里的 `NETWORK_ACCESS` 枚举，写一段分析：RAGFlow 沙箱在默认配置下对网络的真实态度是什么，要彻底断网应该改哪一层。

> **下一章预告**：代码能安全地跑起来了，但 RAG 系统真正的“大脑”是各种大模型——chat、rerank、embedding、CV。第 11 章《模型接入层》将解构 `rag/llm` 目录下的 provider 抽象：它如何用统一接口屏蔽 OpenAI、通义、Ollama 等数十家厂商的差异，如何通过 LiteLLM 接入海量模型，以及 chat / rerank / cv 等不同能力的基类是怎么设计的。
