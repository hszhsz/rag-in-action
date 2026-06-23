# 第 6 章：向量缓存与嵌入接入

知识库问答系统在运行期反复做两件昂贵的事：把一段查询文本变成向量（调用 embedding 模型），以及把整个 FAISS 索引从磁盘读进内存再做相似度检索。前者意味着每次提问都要走一次网络或本地推理，后者意味着一个动辄几百兆的 `index.faiss` 文件如果每个请求都重新 `load_local`，磁盘 IO 与内存抖动会迅速拖垮服务。Langchain-Chatchat 用一个内存缓存层把这两类资源管起来：向量库实例被装进一个带 LRU 淘汰的缓存池常驻内存，embedding 客户端按多后端按需构造。

更棘手的是并发。FastAPI 后端会把多个请求分发到不同线程，而 FAISS 的 `add`、`delete`、`save_local` 都不是线程安全的——同一个知识库被一个线程写、另一个线程读，结果就是脏数据甚至崩溃。本章围绕 `kb_cache` 目录里两个文件展开：`base.py` 给出通用的线程安全缓存原语 `ThreadSafeObject` 与 `CachePool`，`faiss_cache.py` 在其上派生出真正管理 FAISS 索引的 `KBFaissPool`。随后我们再看 embedding 一侧：`get_Embeddings` 如何按平台分发到 OpenAI、Ollama、智谱、LocalAI 四类后端，以及 `LocalAIEmbeddings` 与 `ZhipuAIEmbeddings` 各自的批处理与重试策略。

## 一把可重入锁加一个事件：ThreadSafeObject

缓存的基本单元是 `ThreadSafeObject`（`kb_cache/base.py`）。它不直接持有 FAISS，而是把"被缓存的对象"包一层：`self._obj` 存真正的资源，`self._key` 存键，`self._pool` 回指所属缓存池。线程安全靠两个 `threading` 原语：

```python
self._lock = threading.RLock()
self._loaded = threading.Event()
```

`_lock` 选的是 `RLock`（可重入锁）而不是普通 `Lock`，这点很关键。因为 `ThreadSafeFaiss.save` 内部会 `with self.acquire()` 拿一次锁，而调用方往往已经在 `acquire()` 上下文里持锁——同一线程重复获取同一把普通锁会死锁，`RLock` 允许同线程重入，所以这种"持锁再调持锁方法"的写法才成立。

对外暴露的加锁入口是 `acquire`，它是一个 `@contextmanager`：

```python
@contextmanager
def acquire(self, owner: str = "", msg: str = "") -> Generator[None, None, FAISS]:
    owner = owner or f"thread {threading.get_native_id()}"
    try:
        self._lock.acquire()
        if self._pool is not None:
            self._pool._cache.move_to_end(self.key)
        ...
        yield self._obj
    finally:
        ...
        self._lock.release()
```

它在 `try` 里加锁、`yield` 出 `self._obj`，在 `finally` 里释放，保证即使 `with` 体内抛异常锁也一定被还回去。`owner` 默认取 `threading.get_native_id()`，配合 debug 日志能看到"哪个线程在操作哪个 key"，这是排查并发问题的抓手。注意 `acquire` 进来后立刻 `self._pool._cache.move_to_end(self.key)`——每次有人来操作，就把这个 key 顶到 `OrderedDict` 末尾，这是 LRU 的"最近使用"信号，下一节细说。

`_loaded` 这个 `Event` 解决的是另一类竞态：加载耗时。一个大索引从磁盘读进来要几秒，期间如果第二个线程也来 `get` 同一个库，不能让它拿到一个还没填好 `obj` 的半成品。于是有三个方法配合：`start_loading()` 清事件、`finish_loading()` 置事件、`wait_for_loading()` 阻塞等待。谁先创建谁负责加载并最终 `finish_loading`，后来者在 `CachePool.get` 里会 `wait_for_loading()` 一直等到加载完成才拿到对象。

## OrderedDict 撑起的 LRU 缓存池

`CachePool` 是缓存容器，底层就是一个 `OrderedDict` 加一把池级别的 `RLock`（命名为 `self.atomic`）：

```python
def __init__(self, cache_num: int = -1):
    self._cache_num = cache_num
    self._cache = OrderedDict()
    self.atomic = threading.RLock()
```

LRU 淘汰逻辑藏在 `_check_count` 里，每次 `set` 后被调用：

```python
def _check_count(self):
    if isinstance(self._cache_num, int) and self._cache_num > 0:
        while len(self._cache) > self._cache_num:
            self._cache.popitem(last=False)
```

`cache_num` 是容量上限，`-1` 或非正数表示不限。超量时 `popitem(last=False)` 弹出 `OrderedDict` 的头部——而头部正是"最久未被 `move_to_end` 顶过"的那个 key。"末尾=最近使用、头部=最久未用、淘汰从头弹"，这套约定把一个普通字典变成了标准 LRU。

`get` 方法把"等待加载"封进了取数路径：

```python
def get(self, key: str) -> ThreadSafeObject:
    if cache := self._cache.get(key):
        cache.wait_for_loading()
        return cache
```

拿到缓存项后先 `wait_for_loading()` 再返回，所以任何拿到 `ThreadSafeObject` 的调用方都能确信里面的 `obj` 已经就绪。`pop` 支持两种语义：传 key 就删指定项，不传就 `popitem(last=False)` 弹最旧的；`acquire` 则是"取出缓存项并直接进入它的 `acquire` 上下文"的便捷封装，取不到时抛 `RuntimeError(f"请求的资源 {key} 不存在")`。

值得一提的是，这一版 `base.py` 里并没有独立的 `EmbeddingsPool`——embedding 不走常驻缓存池，而是每次需要时由 `get_Embeddings` 现造（见后文）。缓存池的全部精力都用在 FAISS 索引上，因为索引才是又大又重、值得常驻的资源。

## ThreadSafeFaiss：给 FAISS 套上安全壳

`faiss_cache.py` 里的 `ThreadSafeFaiss` 继承 `ThreadSafeObject`，把通用壳特化成 FAISS 专用。它重写 `__repr__` 时多打了一个 `docs_count()`，直接数 `self._obj.docstore._dict` 的长度，方便日志里看库里有多少条文档。真正体现"安全"价值的是 `save` 与 `clear`：

```python
def save(self, path: str, create_path: bool = True):
    with self.acquire():
        if not os.path.isdir(path) and create_path:
            os.makedirs(path)
        ret = self._obj.save_local(path)
        logger.info(f"已将向量库 {self.key} 保存到磁盘")
    return ret
```

`save_local` 与 `clear` 里的 `delete` 都是会修改索引内部状态的危险操作，全部包在 `with self.acquire()` 内——同一时刻只有一个线程能写盘或清库，其他读检索的线程被挡在锁外。`clear` 删完还加了一句 `assert len(self._obj.docstore._dict) == 0` 自检，确保删除真的把 docstore 清空了。

文件顶部还有一处对 langchain 的猴子补丁值得注意：它把 `InMemoryDocstore.search` 替换成自定义的 `_new_ds_search`，在返回 `Document` 时往 `doc.metadata["id"]` 里塞回文档 id [[InMemoryDocstore]](https://python.langchain.com/api_reference/community/docstore/langchain_community.docstore.in_memory.InMemoryDocstore.html)。这样检索结果里就能拿到原始 id，便于后续删除或定位——这是对上游 FAISS 封装的一处补强。

## 加载向量库：双重检查加锁的并发协议

`KBFaissPool.load_vector_store` 是整章并发设计的核心，它把池级锁 `atomic` 与项级锁 `acquire` 编织成一套精巧的协议：

```python
def load_vector_store(self, kb_name, vector_name=None, create=True, embed_model=...):
    self.atomic.acquire()
    locked = True
    vector_name = vector_name or embed_model.replace(":", "_")
    cache = self.get((kb_name, vector_name))
    try:
        if cache is None:
            item = ThreadSafeFaiss((kb_name, vector_name), pool=self)
            self.set((kb_name, vector_name), item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                locked = False
                ...
                item.obj = vector_store
                item.finish_loading()
        else:
            self.atomic.release()
            locked = False
    except Exception as e:
        if locked:
            self.atomic.release()
        ...
```

逐步拆解这套协议的意图：

第一，缓存键用元组 `(kb_name, vector_name)` 而非拼接字符串，源码注释明说"用元组比拼接字符串好一些"——避免知识库名里含特殊字符时与分隔符冲突。`vector_name` 默认由 `embed_model.replace(":", "_")` 生成，意味着同一个知识库用不同 embedding 模型嵌入会落到不同缓存项，互不污染。

第二，先用池锁 `atomic` 保护"检查—占位"这段临界区：进来即 `atomic.acquire()`，查 `get`，若不存在就立即 `set` 一个空壳占位。这一步必须串行，否则两个线程都查到 None、都去创建，就会重复加载。

第三，占位成功后切换到项锁：`with item.acquire()` 拿到这个项自己的锁，然后**立刻 `self.atomic.release()`** 放掉池锁。这是关键的锁粒度降级——耗时几秒的磁盘加载（`FAISS.load_local`）只在项锁保护下进行，池锁被尽快释放，别的线程可以并发去加载别的知识库，不会被一个大库的加载阻塞整个池。加载完成后填 `item.obj` 并 `item.finish_loading()` 放行所有等待者。

第四，`locked` 标志加异常分支处理了一个微妙问题：注释写道"we don't know exception raised before or after atomic.release"。因为池锁可能在 `with` 块内已被释放，也可能还没释放就抛了异常，用 `locked` 布尔精确追踪当前是否仍持池锁，避免在 `except` 里重复 release 一把已经放掉的锁。

真正读盘的分支也很清楚：若 `vs_path` 下存在 `index.faiss` 文件就 `FAISS.load_local(..., normalize_L2=True, allow_dangerous_deserialization=True)`；否则在 `create=True` 时调 `new_vector_store` 造一个空库并存盘，再否则抛 `RuntimeError`。`new_vector_store` 的造法是个小技巧：先用一条 `Document(page_content="init")` 通过 `FAISS.from_documents` 建库（FAISS 不能从零文档建立），再把这条占位删掉，得到一个结构完整的空库。

池末尾实例化了两个全局单例：`kb_faiss_pool = KBFaissPool(cache_num=Settings.kb_settings.CACHED_VS_NUM)` 与 `memo_faiss_pool = MemoFaissPool(cache_num=Settings.kb_settings.CACHED_MEMO_VS_NUM)`。两者默认容量分别是 `1` 与 `10`（`settings.py`）：持久知识库通常体量大、只缓存一个最近用的；而 `MemoFaissPool` 服务于"文件对话"这种临时向量库，纯内存、不落盘（`new_temp_vector_store` 与 `KBFaissPool` 的区别就是建好不存磁盘），数量多但每个小，故缓存 10 个。

## get_Embeddings：一函数四后端

embedding 一侧的统一入口是 `server/utils.py` 的 `get_Embeddings`。它先用 `get_model_info` 查出模型配置，再按 `platform_type` 分发：

- `openai` → langchain 的 `OpenAIEmbeddings` [[OpenAIEmbeddings]](https://python.langchain.com/docs/integrations/text_embedding/openai/)
- `ollama` → `OllamaEmbeddings`，并把 `api_base_url` 里的 `/v1` 去掉（Ollama 原生接口不带该后缀）
- `zhipuai` → 项目自带的 `ZhipuAIEmbeddings`
- 其余 → `LocalAIEmbeddings`（OpenAI 兼容协议的通用兜底）

`local_wrap=True` 时会把 base_url 指向 `f"{api_address()}/v1"`、api_key 写死 `"EMPTY"`，用于走项目自己拉起的本地推理服务。默认模型由 `get_default_embedding()` 决定：它从 `get_config_models(model_type="embed")` 拿可用列表，若配置的 `DEFAULT_EMBEDDING_MODEL` 在列表里就用它，否则打 warning 退而用列表第一个——这种"配置缺失时降级而非崩溃"的策略贯穿整个项目。配套的 `check_embed_model` 用一句 `embeddings.embed_query("this is a test")` 做连通性探针，返回 `(bool, msg)` 供启动检查使用。

## LocalAIEmbeddings：多线程批处理 + tenacity 重试

`localai_embeddings.py` 是对 OpenAI 兼容 embedding 接口的封装。它的批处理思路与上游 langchain 不同——不是把文本切成 chunk 顺序请求，而是**一文本一线程并发请求**：

```python
def embed_documents(self, texts, chunk_size=0):
    def task(seq, text):
        return (seq, self._embedding_func(text, engine=self.deployment))
    params = [{"seq": i, "text": text} for i, text in enumerate(texts)]
    result = list(run_in_thread_pool(func=task, params=params))
    result = sorted(result, key=lambda x: x[0])
    return [x[1] for x in result]
```

每条文本封成一个 `task` 丢进 `run_in_thread_pool`（`utils.py` 里基于 `ThreadPoolExecutor` 的封装）。由于线程池 `as_completed` 返回顺序不确定，task 里给每条文本带了序号 `seq`，结果回来后再 `sorted(key=lambda x: x[0])` 恢复原顺序——保证嵌入向量和输入文本一一对应。对于嵌入这种 IO 密集型调用，多线程并发能显著压低整批耗时。

重试由 `tenacity` 实现 [[tenacity]](https://tenacity.readthedocs.io/)。`_create_retry_decorator` 配置了指数退避：`wait_exponential(multiplier=1, min=4, max=10)`、`stop_after_attempt(embeddings.max_retries)`（默认 `max_retries=3`），且只对 `openai.Timeout / APIError / APIConnectionError / RateLimitError / InternalServerError` 这几类瞬时错误重试，业务错误不会被无谓重试。`reraise=True` 保证重试耗尽后把原始异常抛出而非吞掉。还有个细节 `_check_response`：若返回的某个 embedding 长度为 1，判定为"空 embedding"并主动抛 `APIError` 触发重试，规避上游接口偶发返回残缺向量的坑。同步走 `embed_with_retry`、异步走 `async_embed_with_retry`，两套对称实现。

## ZhipuAIEmbeddings：顺序分批与密钥保护

`langchain_chatchat/embeddings/zhipuai.py` 是智谱 embedding 的实现，默认 `model="embedding-2"`，底层用官方 `zhipuai.ZhipuAI(**client_params).embeddings` 客户端 [[ZhipuAI Embedding]](https://docs.bigmodel.cn/cn/guide/models/text-embedding/embedding-2)。它的批处理走另一条路线——`_get_len_safe_embeddings` 按 `chunk_size`（默认 1000）顺序切片请求：

```python
for i in _iter:
    response = self.client.create(input=texts[i : i + _chunk_size], **self._invocation_params)
    if not isinstance(response, dict):
        response = response.dict()
    batched_embeddings.extend(r["embedding"] for r in response["data"])
```

相比 `LocalAIEmbeddings` 的"逐条多线程"，智谱这里是"成批顺序"，因为智谱接口本身支持一次传入多条文本，批量请求更省连接开销；可选 `show_progress_bar` 时用 `tqdm` 显示进度。密钥处理上它用了 `SecretStr` 类型与 `convert_to_secret_str`，把 `zhipuai_api_key` 包成密文，避免日志或 repr 意外泄露明文 key；`embed_query` 则直接复用 `embed_documents([text])[0]`。两个类都继承 `BaseModel, Embeddings`，用 `root_validator` 在实例化时校验环境变量与构造客户端，符合 langchain Embeddings 接口约定 [[langchain Embeddings]](https://python.langchain.com/api_reference/core/embeddings/langchain_core.embeddings.embeddings.Embeddings.html)。

## 本章小结

- `ThreadSafeObject` 用 `threading.RLock`（可重入）保护被缓存对象，`acquire` 以 `@contextmanager` 形式保证异常时锁必释放；选 `RLock` 是为支持 `save` 等"持锁方法内再持锁"的写法。
- `threading.Event`（`_loaded`）解决加载竞态：`start/finish/wait_for_loading` 三方法让后来线程在 `get` 时阻塞等待，直到对象就绪才返回。
- `CachePool` 以 `OrderedDict` 实现 LRU：`acquire` 时 `move_to_end` 标记最近使用，`_check_count` 在超过 `cache_num` 时 `popitem(last=False)` 淘汰最旧项。
- `ThreadSafeFaiss` 把 `save`、`clear` 等危险写操作全部包进 `acquire`，并在文件级猴子补丁 `InMemoryDocstore.search` 以回填文档 id。
- `KBFaissPool.load_vector_store` 用"池锁保护检查占位、随即降级到项锁做耗时加载"的双层锁协议，`locked` 标志精确处理异常路径的锁释放。
- 缓存键用元组 `(kb_name, vector_name)`，`vector_name` 由 embedding 模型名派生，使同库不同模型互不污染。
- `kb_faiss_pool`（默认缓存 1 个，落盘）与 `memo_faiss_pool`（默认缓存 10 个，纯内存临时库）是两个全局单例，分别服务持久知识库与文件对话。
- 本版没有独立 embedding 缓存池，embedding 客户端由 `get_Embeddings` 按需现造，按 `platform_type` 分发到 OpenAI / Ollama / 智谱 / LocalAI 四类后端。
- `LocalAIEmbeddings.embed_documents` 用 `run_in_thread_pool` 逐条并发并以 `seq` 排序还原顺序；`tenacity` 指数退避只重试瞬时错误，`_check_response` 拦截空向量。
- `ZhipuAIEmbeddings` 按 `chunk_size` 顺序分批请求，用 `SecretStr` 保护 api_key，复用 `embed_documents` 实现 `embed_query`。
- `get_default_embedding` 与 `check_embed_model` 体现"配置缺失即降级、启动前先探针"的容错风格。

## 动手实验

1. **实验一：观察 LRU 淘汰** — 把 `Settings.kb_settings.CACHED_VS_NUM` 临时改为 `1`，依次 `load_vector_store` 加载两个不同知识库，打印 `kb_faiss_pool.keys()`，确认第一个库被 `popitem(last=False)` 淘汰；再交替访问两库观察 `move_to_end` 如何改变淘汰顺序。
2. **实验二：复现可重入锁的必要性** — 把 `ThreadSafeObject._lock` 临时换成 `threading.Lock()`，调用 `ThreadSafeFaiss.save`（其内部在已持锁上下文里再 `acquire`），观察是否死锁；换回 `RLock` 后恢复正常，理解为何必须可重入。
3. **实验三：验证加载竞态保护** — 起多个线程同时 `load_vector_store` 同一个尚未缓存的库，在加载分支里 `time.sleep(3)` 放大窗口，确认只有一个线程真正读盘、其余线程经 `wait_for_loading` 等待后拿到同一实例（用 `id()` 比对）。
4. **实验四：对比两种批处理策略** — 给 `LocalAIEmbeddings` 与 `ZhipuAIEmbeddings` 各传入 50 条文本调 `embed_documents`，分别计时；再给 `LocalAIEmbeddings` 故意配一个会超时的 base_url，观察 `tenacity` 指数退避日志（`before_sleep_log`）与最终 `reraise` 行为。

> **下一章预告**：向量库与 embedding 都备齐后，知识库还需要一套管理文档全生命周期的服务——增删改查、源文件与向量的同步、不同存储后端之间的迁移。第 7 章《知识库文档管理与迁移》将解构 `KnowledgeFile`、文档入库与删除链路，以及 FAISS 到 Milvus/PG 等后端之间的数据迁移工具。
