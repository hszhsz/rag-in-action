# 第 3 章 文本切分 Chunker

文档进入 LightRAG 后的第一道实质性加工，就是切分。一篇论文、一份合同、一张满是表格的财报，最终都要被切成一段段大小可控、语义尽量完整的 chunk，才能去做向量化、实体抽取、入图入库。

切分质量直接决定了下游召回的天花板：切得太碎，实体被拦腰斩断、上下文丢失；切得太大，向量被稀释、token 预算被浪费。LightRAG 没有把切分写死成一个函数，而是在 `lightrag/chunker/` 下提供了四种可插拔的策略，并用一套统一的 chunk 契约把它们串起来——无论哪种策略，吐出来的都是同构的字典列表，下游完全不必关心它是怎么切出来的。

本章我们从 `chunker/__init__.py` 的注册与契约谈起，逐一拆开 token_size、recursive_character、paragraph_semantic、semantic_vector 四种实现，再回到 `chunk_schema.py` 看 chunk 这个数据结构到底承载了什么。读完你会明白：所谓"多策略可插拔"，本质是用一份稳定的输出 schema，给不同的切分算法留出统一的接缝。

为了让脉络清晰，本章的顺序是先讲两套调用契约（解释"为什么有两个入口"），再按"从不看内容到完全看语义"的复杂度递增顺序拆解四个策略，然后用一小节横向对比"什么时候用哪个"，最后落到把它们粘合起来的那份 chunk schema。你会反复看到同一个工程母题：每个策略都在自己的内部逻辑之外，额外承担了"把切出来的内容精确溯源回原文"的责任，这正是 LightRAG 在切分层做得比许多框架更细的地方。

## 两套调用契约：为什么 token_size 有两个入口

打开 `chunker/__init__.py`，模块 docstring 开宗明义地讲了一件容易被忽略的事：LightRAG 里"故意"并存着两套契约。

第一套是**遗留契约**（legacy contract），由 `chunking_by_token_size` 承担，它保留了历史上的 6 个位置参数签名 `(tokenizer, content, split_by_character, split_by_character_only, chunk_overlap_token_size, chunk_token_size)`。

这个签名之所以不能动，是因为它是 `LightRAG.chunking_func` 的默认值，外部用户可能传入自己实现的 `chunking_func` 来替换默认行为；一旦改签名，所有第三方实现就全挂了。它只在 `process_options` 没有显式指定切分选择器时才被调用——典型场景是直接调用 `LightRAG.ainsert` 灌入裸文本。

换言之，遗留契约服务的是"我直接给一段文本、按默认方式切"这类最简单的入口；它不需要知道文档来自哪个文件、用什么策略，只要保持那个老签名不变，历史代码与第三方扩展就能继续工作。这是一种典型的"为兼容性付出表面冗余"的取舍——多一个入口，换来不破坏任何现存调用方。

第二套是**文件切分契约**（file-chunker contract）。当一篇文档的 `process_options` 明确选择了某种切分策略时，`_PipelineMixin.process_single_document` 里的分发器会读 `doc_process_opts.chunking` 字段，按照标准签名 `(tokenizer, content, chunk_token_size, *, <策略专属 kwargs>)` 路由到对应 chunker。

当前四个出厂 chunker 各自对应一个单字母策略码：`"F"` 是 `chunking_by_fixed_token`（固定 token 窗口，与遗留算法同源）、`"R"` 是 `chunking_by_recursive_character`（递归字符）、`"V"` 是 `chunking_by_semantic_vector`（向量语义）、`"P"` 是 `chunking_by_paragraph_semantic`（段落语义）。`__init__.py` 的 `__all__` 把这五个函数（含遗留的 `chunking_by_token_size`）统一导出，这就是整个切分子系统对外的全部表面。

单字母策略码的设计也透露出 LightRAG 的偏好：切分策略是一个可以序列化进配置、跟着文档状态一起持久化的小标量，而不是一个需要传函数对象的复杂选项。这让"某文档用什么策略切的"成为一条可记录、可审计、可复现的元数据。

理解这两套契约的关键在于：标准签名的前三个参数 `(tokenizer, content, chunk_token_size)` 是所有文件 chunker 的**公共前缀**，策略差异全部塞进 keyword-only 的尾部参数。正因为前缀对齐，分发器才能用同一套代码调用任意策略，这就是可插拔的物理基础。

`chunking_by_fixed_token` 本身没有任何新逻辑，它只是把遗留算法重新包装成标准签名的"门面"，函数体直接转调 `chunking_by_token_size`。

可见两套契约虽然签名不同，却共享同一份切分内核——这是 LightRAG 在"不破坏向后兼容"与"统一分发"之间的折中：旧入口原样保留给第三方 `chunking_func`，新入口让管线分发器有一致的调用面。这种"一份算法、两副签名"的做法在成熟代码库里很常见，关键是让二者真正调用同一段实现，而不是各写一份慢慢漂移——`chunking_by_fixed_token` 的函数体里没有任何独立逻辑，正是这一原则的体现。

## token_size：固定窗口的基础切分

`token_size.py` 是 LightRAG 的默认策略，也是理解其余三者的基准。它的思路最朴素：把全文用 `tokenizer.encode(content)` 编码成 token 序列，然后开一个固定大小的滑动窗口逐段取出。之所以选 token 而非字符作为单位，是因为下游嵌入模型与 LLM 的上下文预算都是按 token 计的，按 token 切才能让"每个 chunk 不超模型上限"这件事变得可控。

核心两个旋钮写在 `chunking_by_token_size` 的默认值里——`chunk_token_size=1200` 是每个 chunk 的 token 上限，`chunk_overlap_token_size=100` 是相邻 chunk 之间的重叠量。窗口推进的步长是二者之差，体现在那行 `range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)`：每次前进 `1200-100=1100` 个 token，于是每段尾部 100 个 token 会和下一段开头重叠。重叠的意义是给跨段的句子、实体一个被完整覆盖的机会，避免边界处的语义被硬生生切断。

每个窗口取出后用 `tokenizer.decode(tokens[start:end])` 还原成文本。

值得注意的是 `tokens` 字段的计法：代码写的是 `min(chunk_token_size, len(tokens) - start)`，也就是说它记录的是这个窗口"应该"覆盖的 token 数（满窗为 `chunk_token_size`，末段为剩余量），而非 decode 回来再重新 encode 的实际数。

`_make_chunk` 把每段封装成 `{"tokens", "content", "chunk_order_index"}` 三件套，`content` 在落库前会 `.strip()` 去掉首尾空白，`chunk_order_index` 是顺序号，下游靠它还原 chunk 在原文中的先后。这三个字段就是后文要反复提到的"最小 chunk 契约"——无论哪种策略，至少要吐出这三样。

当调用方传入 `split_by_character` 时，算法会先用该分隔符把全文 `content.split()` 成若干段。在切分前它会先记录每一段的原始字符区间 `raw_spans`，按段长加上分隔符长度推进游标，这样即便后续要再细切，每段也都能溯源回原文位置。

若同时 `split_by_character_only=True`，则每段原样作为一个 chunk，但有硬约束：一旦某段超过 `chunk_token_size`，会直接 `raise ChunkTokenLimitExceededError`——因为用户明确表态"只按这个字符切，不许再细分"，超限就只能报错而非偷偷切开。若 `split_by_character_only=False`，则对超限的段落退回到固定窗口继续细切，且窗口锚点是**段内相对**的（每段重置为 `(0,0)`），算完再整体平移回全局偏移。无论哪条路径，结果都汇入同一个 `results` 列表，编号连续。

`token_size.py` 里还有一段相当克制的工程细节值得一提：`_token_window_source_span` 负责把一个 decode 出来的 token 窗口精确映射回原文的字符区间（`_source_span` 的 `{"start","end"}`）。难点在于字节级 BPE 的 decode 在多字节 UTF-8 边界上不可拼接，逐窗口重新 decode 整个前缀会退化成 O(N²)。

它的解法是只 decode 上一窗口已验证锚点到当前起点之间的增量（约一步的量），预测出起点字符偏移，再拿 `content[start:end] != window` 验证，不符就在 ±32 字符范围内用 `find` 兜底校正，并把验证后的位置作为新锚点，从而把累计误差压住，整体保持 O(N) 而定位仍是字节精确。

这个 span 不是花架子：下游若要把一个召回的 chunk 高亮回原文、或在 citation 里给出精确位置，就需要它。`_source_span` 只在 `_emit_source_span=True` 时附带到 chunk 上——默认关闭，是因为大多数纯文本场景并不需要精确回溯，把这份开销做成可选项是合理的取舍。

## recursive_character：分隔符回退链

`"R"` 策略 `chunking_by_recursive_character` 包装了 LangChain 的 `RecursiveCharacterTextSplitter`，思路是"先按强语义边界切，切不动再降级"。

它的分隔符级联在 `separators` 参数里，默认沿用 LangChain 的 `["\n\n", "\n", " ", ""]`——优先在段落空行处切，其次单换行，再次空格，最后退到逐字符（空字符串）。

这条级联背后的直觉是：段落边界（双换行）通常是最强的语义分界，越往后越弱；逐字符切是最后的"什么都切不开时也得切"的保底手段。`_split_text_with_spans` 实现了这条回退链的核心控制流：它从级联里挑出第一个能在当前文本中命中的分隔符，把文本切成片段；对每个片段，若其 token 长度小于 `chunk_size` 就收进 `good_splits` 待合并，否则若还有更弱的分隔符就**递归**用剩余级联去切这个超长片段，没有更弱分隔符可用时才原样保留。这就是"递归"二字的来历：强分隔符切不开的硬块，会被逐级降级处理。

这里有一个 LightRAG 自己的关键改造：token 计量。`length_function` 被定义成 `len(tokenizer.encode(text))`，强行把 `chunk_size`/`chunk_overlap` 的单位从字符改成 token——否则 `chunk_token_size` 这个名字就名不副实了。默认值同样是 `chunk_token_size=1200`、`chunk_overlap_token_size=100`，与 token_size 对齐。

`_merge_splits_with_spans` 复刻了 LangChain 的 `_merge_splits`：把小片段贪心地拼到接近 `chunk_size` 再切开，并在切开时按 `chunk_overlap` 回退保留尾部重叠。这一步是 R 与 F 在效果上的关键差异：F 是无脑等距滑窗，R 则是"尽量在自然边界拼到接近上限"，于是 R 切出来的 chunk 更倾向于在段落/句子边界收口，而非在某个词中间生切。

为什么 LightRAG 不直接用 LangChain 原版、非要把 split/merge 流程整套重写一遍？`chunking_by_recursive_character` 的注释给了答案：它**故意不用** LangChain 的 `add_start_index`，因为当 `length_function` 是基于 token 时，那个 start_index 是用字符与 token 混算出来的，对重复文本块的文本搜索回溯也有歧义。

值得细看的是 span 是怎么一路保住的。`_split_text_with_regex_spans` 在按正则切分时，不只返回片段文本，还返回 `(text, start, end)` 三元组；它甚至完整复刻了 LangChain 的 `keep_separator` 三态行为——`"end"` 把分隔符留在前一片尾部、真值留在后一片首部、假值丢弃分隔符——每一种都同步维护字符偏移。

`_join_span_pieces` 在拼回片段时按字符逐位记录 `char_offsets`，再做 strip，于是即使首尾空白被裁掉，剩余内容的起止字符位也能精确回算。`_merge_splits_with_spans` 的 overlap 回退是一个 `while` 循环：当累计 token 超过 `chunk_overlap` 或加上下一片会超 `chunk_size` 时，就从 `current_doc` 头部弹出片段并扣减 token，从而把尾部一小段保留进下一个 chunk。这一整套"镜像上游 + 全程带偏移"的写法，代价是要维护两百行与 LangChain 行为对齐的代码，换来的是每个 chunk 都能字节精确地回指原文。

于是 LightRAG 镜像了 LangChain 的控制流，但让每个切分单元一路携带自己的源字符偏移 `_SpanPiece = (text, start, end)`，最终给每个 chunk 都算出精确的 `_source_span`。

还有两处兜底：模块缺失 `langchain_text_splitters` 时抛 `ImportError`；splitter 只产出空白片段时打 warning 并退回单个 stripped chunk，保证非空输入永远至少产出一个 chunk。另外注释明确说明 R **不在内部强制 token 上限**——切不开的超大段会被放出去，由 `enforce_chunk_token_limit_before_embedding` 在向量化前做最终硬切。这是一个反复出现的分层约定：切分器负责"尽量切好"，最终的"绝不超限"由向量化前那道闸统一兜底，避免每个切分器都各自实现一遍硬切逻辑。

## semantic_vector：用句向量找断点

`"V"` 策略 `chunking_by_semantic_vector` 包装 LangChain 的 `SemanticChunker`（来自 `langchain-experimental`），它不靠字符或 token，而是靠**句子的语义相似度**找切分点。

流程是：先按句子切分正则把文本切成句子，把相邻 `buffer_size` 个句子组成窗口算 embedding，再计算相邻窗口的余弦距离，距离超过某个阈值处就认为是语义断点。

这背后的假设是：语义连贯的句子，其向量在空间上彼此靠近；当话题切换时，相邻句向量之间会出现一个明显的距离"跳变"，那个跳变点就是天然的切分边界。

阈值类型由 `breakpoint_threshold_type` 决定，默认 `percentile`，还支持 `standard_deviation`、`interquartile`、`gradient`。句子切分正则默认是 `DEFAULT_SENTENCE_SPLIT_REGEX`，其值为 `(?<=[.?!])\s+|(?<=[。？！])`——在 LangChain 原版纯英文模式上**扩展了中文句末标点** `。？！`，这样中英混排和纯中文输入才能正确断句。这是一处很小但很关键的本地化：上游纯英文正则在中文上几乎切不出句子，整篇会被当成一句、语义切分就退化成不切。

V 是四个 chunker 里唯一的 `async` 函数，因为 LightRAG 的 `EmbeddingFunc` 是异步的。但 `SemanticChunker` 是同步接口，于是这里有一段精巧的桥接：`_AsyncEmbeddingFuncAdapter` 把异步的 `EmbeddingFunc` 适配成 LangChain 同步的 `Embeddings`，它在构造时捕获当前事件循环引用，真正的 `embed_documents`/`embed_query` 调用通过 `asyncio.run_coroutine_threadsafe` 弹回主循环执行。

而整个 `SemanticChunker` 的同步切分则用 `asyncio.to_thread` 丢到工作线程里跑，避免阻塞事件循环。注意 adapter 必须在运行中的事件循环里构造才能抓到 loop 引用，`embed_documents`/`embed_query` 还会把返回向量逐元素 `float()` 规整成纯 Python list，喂给 LangChain。这套"同步库跑在线程、回调弹回主循环"的模式，是把一个本质同步的第三方库嵌进异步框架的标准做法，值得在自己的项目里借鉴。

`_semantic_groups_with_spans` 是另一处典型的"重写上游以保留源 span"。它直接复刻了 `SemanticChunker.split_text` 的函数体，目的是让每个语义组携带精确源 span `text[start:end]`，而不是上游那种 `" ".join(sentences)` 的重排文本（重排会丢失原文的精确位置）。代价是它依赖了一批**私有成员**（`sentence_split_regex`、`breakpoint_threshold_type`、`_calculate_sentence_distances` 等），所以注释里写明只对 `langchain-experimental` 0.3.2–0.4.x 这个 `pyproject.toml` 锁定区间逐字节验证过，并有测试用漂移守卫比对自家镜像与上游真实输出。

span 的恢复靠 `_sentence_spans`：它拿着切出的句子去原文里 `text.find(sentence, cursor)` 逐句定位，找不到就退而在全文找、再退到游标位，保证每句都有一个起止区间；`_group` 再把若干句的 span 首尾相连、`_trim_span` 去掉两端空白，得到该语义组的源区间。

函数对边界情形也很细致：只有一句时直接成组；`gradient` 阈值类型且恰好两句时分成两组（gradient 至少要三个点才能算梯度）；其余情形才进入"距离超阈处断开"的主循环，并用 `min_chunk_size` 把过小的组跳过。这些边界分支看似琐碎，却是镜像上游 `split_text` 时必须逐字对齐的地方——少一个分支，自家镜像与上游就会在某类输入上产出不同结果，drift 守卫测试就会报警。

V 同样**不强制 token 上限**——`SemanticChunker` 没有内置 size cap，`chunk_token_size` 在这里只是 advisory。所以函数在产出每个语义组后会检查其 token 数，凡超过 `target_max` 的组就调 `chunking_by_recursive_character` 用 R 再切一遍，并且传 `chunk_overlap_token_size=0` 以保持 V"语义边界天然不重叠"的特性。

注释里特别点明：这一步是有意为之的，否则超大组就只能依赖向量化前那个用 `embedding_token_limit`（而非 `chunk_token_size`）的硬兜底来切，用户配置的尺寸就形同虚设了。这里把"尊重用户配置"这件事提前到了切分阶段，而不是拖到嵌入阶段。

还有一个重要降级路径：当 `embedding_func` 为 `None`（没有嵌入函数可用），V 的唯一差异化能力就没了，于是它打 warning 后回退到 R——同样用 0 重叠，理由是 R 是"只看结构"的最近邻替代。这两处对 R 的引用都用了懒导入，专门绕开 `recursive_character` 与 `semantic_vector` 之间的循环依赖。

## paragraph_semantic：标题感知的结构化切分

`"P"` 策略 `chunking_by_paragraph_semantic` 是四者中最复杂的（`paragraph_semantic.py` 长达两千余行），也是 LightRAG 切分思想的集大成者。它的输入不是裸文本，而是解析阶段产出的 `.blocks.jsonl` sidecar——由任意支持 sidecar 的解析器（native / mineru / docling）生成，里面是按标题驱动切好的结构化块（HeadingBlocks），表格被整块保留。

这个分工本身就是一个值得品味的设计决策。换句话说，**解析器只做标题层级的结构切分，所有块大小的调整都搬到这里来做**，且以 `chunk_token_size` 为参数，让 chunk 大小目标跟随用户的 RAG 配置走。

这样做的好处是：同一份解析产物（`.blocks.jsonl`）可以被不同的 `chunk_token_size` 配置反复消费，而不必每次重新解析文档——解析昂贵、切分廉价，把可变的部分留在廉价侧。注意 P 的 `chunk_token_size` 默认是 `2000`，比其余策略的 1200 大，因为结构化块本身更倾向于保留完整语义单元。

模块 docstring 把整条流水线拆成五个阶段。**HeadingBlocks** 在解析期完成、持久化为 jsonl 每行一块。

**TableRowSplit** 处理超大表：当内嵌的 `<table format="json">`（或 html）超过 `table_max` 时，优先按结构化行边界（JSON 列表项、HTML 的 `<tr>`）切，让每个碎片仍是合法 `<table>`；只有无行边界可切、或单行就超限时才退回 `chunking_by_recursive_character`。它还会把解析期抽到 `.tables.json` 的表头按 slice 的 `<table>` id 重新注入到非首片，且**先把表头 token 从每片预算里扣掉再切**，保证每片含表头后仍 ≤ `target_max`。

注释还点出一个连带约束：因为表头从一开始就嵌进了 slice，被切开的表的各 slice 会被"冻结"以免参与 LevelMerge——否则把同一张表的两个 slice 重新合并会让表头出现两次。当一个 heading 块里两张超大表之间夹着短文本时，这段桥接文本可能被复制进两个表边界 chunk，让每张表都保留临近上下文，较长的中段残余则单独成块。

**AnchorSplit** 用短段落（≤ 100 字符，常量 `_MAX_ANCHOR_CANDIDATE_LENGTH`）作为锚点拆分超长块，向 `target_ideal` 重新平衡。其直觉是：一段很短的独立段落（比如一个小标题式的引导句）往往是话题转折的天然分割点，比在长段落中间硬切更自然。

**HeadingGlue** 把"只有标题没有正文"的块向前粘到它第一个更深的子块上，免得标题与正文失散。**LevelMerge** 自底向上、按层级把过小块合并进同级或更浅级邻居：先吸收同级（Phase A）、再吸收更浅级（Phase B），最后对尾部残块做一遍 tail-absorption。这五个阶段串起来，本质上是在"不破坏文档结构"的前提下，把过大块切小、把过小块并大，最终让每个 chunk 既不超 `target_max`、又尽量接近 `target_ideal`。

这里全是相对阈值而非魔数。函数开头把一组比例常量乘到 `target_max` 上：`target_ideal = target_max * _IDEAL_RATIO`（0.75）、`table_max`（0.625）、`table_ideal`（0.375）、`table_min_last`（0.32）、`small_tail_threshold`（0.125）。

注释说明这些比值是从 docx 解析器审计模式的 6000/8000、5000/8000 等绝对值推导来的，让那套打磨过的权衡曲线原样平移，但绝对值随用户配置缩放。这种"比例化"设计的好处是：用户只需调一个 `chunk_token_size`，整条表格/正文/理想/上限的尺度链就协调地一起变。反过来说，如果这些值是写死的绝对数，用户把 `chunk_token_size` 调小一半后，表格阈值就会相对偏大、整套平衡被打破——比例化把"一组互相关联的尺寸"压缩成了"一个可调旋钮"。

P 在两个场景需要 overlap，所以保留了 `chunk_overlap_token_size`（默认 100），并经 `_bounded_overlap` 夹到 `[0, target_max-1]` 防止越界：一是长正文退回 R 切分时，二是两个相邻超大表之间桥接文本的两侧复制预算。结构化的表格行切分本身仍是行边界、不重叠——overlap 只在"不得不退回字符切分"或"桥接文本复制"这两种破坏结构的场景里出现。当一个原始 jsonl content 行被切成多片时，每片标题会被打上行内 `[part n]` 后缀（`_apply_part_suffix`），未切的行保留原标题；`_strip_generated_heading_suffixes` 会先剥掉历史遗留的 `[表格片段N]`/`[part N]` 后缀再重新编号，避免后缀叠加。

最关键的兜底是：当 `blocks_path` 为空、读不到、或无内容行时，P 直接退回 `chunking_by_recursive_character`——这是 `FileProcessingConfiguration-zh.md` 明文规定的契约（P 降级到 R），保证非结构化文档或边缘解析也绝不静默丢内容。注意这条降级硬要求装了 `langchain-text-splitters`，宁可抛 `ImportError` 也不再悄悄退到 F，因为契约写的是退 R。`drop_references` 旋钮可在切分前丢掉尾部参考文献段：一个块只有同时满足"位于最后 `references_tail_n`（默认 2）个块内"且"标题命中 `DEFAULT_P_REFERENCES_HEADINGS`（`References`/`Bibliography`/`参考文献`）前缀"才会被删，tail 窗口是防止误删文档中部某个 References 子节的安全闸。

## 策略如何被选中：per-document 快照

四个策略并非全局二选一，而是**逐文档**生效的。

`__init__.py` 的 docstring 指明：是否走遗留契约取决于 `chunking_explicit` 这个标记——`process_options` 没指定切分选择器时（典型是 `ainsert` 灌裸文本）才落到遗留的 `chunking_by_token_size`；一旦某文档的 `process_options` 显式选了策略，文件分发器就读 `doc_process_opts.chunking` 的单字母码，路由到 `F/R/V/P` 之一。

这里有一个对"可复现"很重要的设计：部分旋钮会被快照进 `chunk_options`、并记录到 `doc_status.metadata['chunk_opts']`，部分则不会。

以 P 策略为例，`drop_references` 是唯一被快照的参考文献旋钮（它可能来自 per-file hint），而 `references_tail_n`/`references_headings` 这两个检测窗口默认从环境变量 `CHUNK_P_REFERENCES_TAIL_N`/`CHUNK_P_REFERENCES_HEADINGS` **运行时实时读取**、不快照。差别在于：被快照的旋钮在重跑时行为稳定，不快照的旋钮则允许你改环境变量后，重跑直接生效而无需重新入队。这种"哪些参数冻结、哪些参数活取"的取舍，正是切分层在可复现性与可运维性之间的精细权衡。

## 四种策略的取舍：什么时候用哪个

把四者并排看，它们其实覆盖了从"完全不看内容"到"完全看语义"的一条谱带。`F`（token_size）只数 token，最快、最可预测、零外部依赖，是默认与兜底；它对结构无感知，纯文本流式灌入时最合适。

`R`（recursive_character）多走一步，按 `\n\n`/`\n`/空格/逐字符的级联尽量在自然边界切，对 Markdown、代码、带换行结构的纯文本效果好，且只需 `langchain-text-splitters`，是其余两个高级策略的公共降级目标。可以把 R 理解为"结构感知的 F"——它仍然只看字符表面、不懂语义，但比 F 更尊重排版边界。

`V`（semantic_vector）和 `P`（paragraph_semantic）则是两种"语义优先"路线，但语义来源不同。V 的语义来自**嵌入模型**——它用句向量距离找话题转折，适合长篇连续散文（论文正文、长文章），但代价是要算 embedding、要 `langchain-experimental`、且是 async；没有 `embedding_func` 它就退化成 R。

P 的语义来自**文档结构**——它信任解析器给出的标题层级与表格边界，适合规范化的 docx/PDF（合同、报告、带大量表格的文档），它能保留 heading 面包屑、把表格按行切而非按字符切，这是 V 做不到的；但它强依赖 `.blocks.jsonl` sidecar，拿不到就退回 R。一句话区分：V 问"这两句话在讲不在同一件事吗"，P 问"这两段在不在同一个章节、同一张表里"。

可以这样记忆这条降级链：P 失败退 R，V 失败退 R，R 失败退单 chunk，而 F 永远可用。四个策略不是互斥的竞品，而是一个有明确兜底顺序的策略簇——这正是"统一契约 + 可插拔实现"带来的鲁棒性。任何一层失效，系统都能优雅地退到更简单但更可靠的下一层，绝不会因为一个可选依赖缺失就丢掉整篇文档。

## 统一的 chunk 契约：chunk_schema.py

走到这里，四种策略的内部机制已经各异其趣：固定窗口、分隔符级联、句向量断点、结构化五阶段流水线。但它们能被同一条下游管线（向量化、实体抽取、入库）无差别消费，靠的不是巧合，而是一份共同的 chunk schema。这一节我们就把这份 schema 拆开，看它到底约束了什么、又为各策略留了哪些可选的扩展位。

最基础的三个字段——`tokens`、`content`、`chunk_order_index`——是所有 chunker 的最小公约数；带定位能力的 chunker 还会附 `_source_span`。可以把前三者理解为"必填项"、`_source_span` 理解为"可选的精确溯源"，再往上才是 P 独有的结构化富化字段。

P 策略在此之上额外富化两个结构化字段：嵌套的 `heading`（`{"level", "heading", "parent_headings"}`）和可选的 `sidecar`（`{"type", "id", "refs"}`，仅当源行带 `blockid` 时存在）。`heading` 让 KG 抽取能感知文档层级，`sidecar` 把 chunk 溯源回它来自哪些原始块。

注意 `full_doc_id` 这类把 chunk 绑回所属文档的字段不在 chunker 里产生，而是由更上游的管线在写库时补齐——chunker 只负责"切"，不负责"归属"。这条职责边界很重要：它意味着你可以把任意一个 chunker 单独拿出来在 REPL 里测试，输入文本、得到 chunk 列表，完全不必构造文档上下文。

`chunk_schema.py` 的职责正是给这些字段定义统一的规范化规则，让 chunker 和管线消费同一套逻辑、不会漂移。`normalize_chunk_heading` 把历史上扁平的 `heading`/`parent_headings`/`level` 三元组，和新的嵌套形式，统一塌缩成规范的 `{"level", "heading", "parent_headings"}` 字典；空输入归为 `None`，调用方就可以干脆省略该字段。这种"同时接受新旧两种形状、塌缩成一种"的写法，是 schema 演进时保持向后兼容的常见手法——旧数据不必迁移，读取时归一化即可。

`format_parent_headings` 和 `format_heading_context` 把标题链拼成 `h1 → h2 → h3` 的面包屑（分隔符常量 `HEADING_BREADCRUMB_SEP = " → "`），前者只拼父链、后者把当前节标题也接上，喂给实体抽取 LLM 让它知道这段文字在文档结构中的位置。

两者都先经 `_clean_heading_text` 清洗：把面包屑分隔符 `→` 换成空格（防止它在被重新按 ` → ` 拆回层级时伪造出一个额外层级）、剔除所有 Unicode 控制（Cc）/格式（Cf）字符（零宽符、NUL、方向控制码这类只给 prompt 添 token 噪声的东西）、最后把各种空白折叠成单个空格。清洗顺序被刻意固定为"清洗 → 去空 → 截断"，并对每个标题层级按 `DEFAULT_HEADING_LEVEL_MAX_CHARS`（80）硬截断，防止一个超长标题撑爆 prompt。这套清洗对中文是安全的：普通 CJK、拉丁、数字、标点都原样保留，相邻 CJK 字符之间也不会被插入空格。

`normalize_chunk_sidecar` 则校验 sidecar 负载并保证 `refs` 永远物化成至少含主 id 的列表——单源 chunk 也会落库成 `refs=[{type,id}]`，下游消费者就不必为字段是否存在写特例分支。

它还会校验 `type` 必须落在 `_SIDECAR_TYPES`（`block`/`drawing`/`table`/`equation`）这个白名单内、`id` 非空，否则整条 sidecar 归为 `None`。"宁可返回 None 也不返回半成品"是这一组 normalize 函数共同的姿态：它们把"字段缺失/非法"统一收敛成一个明确的空值信号，让调用方只需判断有没有，而不必逐字段防御性校验。

`strip_internal_multimodal_markup_for_extraction` 在构建抽取 prompt 时清掉 `<cite>`/`<drawing>`/`<equation>`/`<table>` 里那些解析器塞进去、不该让 LLM 看到的标识属性。

它的尺度很克制：`<cite>` 整体退化为可见文本，`<drawing>` 只留 `caption` 丢掉 `id`/`path`/`src`/`format`（连 caption 都没有的整个删掉），`<equation>`/`<table>` 则剥掉 `id`/`refid` 但**保留 `format` 与 caption 和正文**，让抽取器仍能识别"这是一张结构化表/一个公式"并据此 grounding。原生解析器发出的 `tb-<doc>-NNNN` 这类内部表 id 若不剥掉，会泄进 prompt 变成一个噪声实体。

最关键的是原始 `chunk["content"]` 绝不被改动——清洗串只用于 prompt。这条"原文不可变、清洗只服务于抽取"的纪律，保证了 chunk 落库的内容与最终用于召回展示的内容一致，抽取时看到的"干净版"只是临时视图。

把这些拼起来看：chunk schema 既是 chunker 的"输出合同"，也是抽取与查询阶段的"输入合同"。

正因为合同稳定，LightRAG 才能在不动下游一行代码的前提下，自由地增删切分策略——这正是第 1 章里"可插拔"设计哲学在切分层的具体落地。回头看本章拆过的四个实现，它们内部从最简单的等距滑窗到两千行的结构化流水线，差异不可谓不大；但它们对外都只承诺同一件事："给我文本，还你一列 `{tokens, content, chunk_order_index, ...}`"。这份承诺，就是多策略可插拔得以成立的全部基础。

## 本章小结

- LightRAG 在 `chunker/__init__.py` 里并存两套契约：遗留 6 参签名 `chunking_by_token_size`（默认 `chunking_func`，灌裸文本时用）与标准签名的文件 chunker（前缀 `(tokenizer, content, chunk_token_size)` + keyword-only 旋钮），后者按策略码 `F/R/V/P` 分发。
- `token_size.py` 是基础策略：默认 `chunk_token_size=1200`、`chunk_overlap_token_size=100`，滑动窗口步长为二者之差（1100），重叠保护跨段语义；`chunking_by_fixed_token`（`"F"`）只是它的标准签名门面，转调同一内核。
- `recursive_character.py`（`"R"`）包装 LangChain，按 `["\n\n","\n"," ",""]` 分隔符级联从强到弱递归切分，用 `length_function = len(tokenizer.encode(...))` 把单位改成 token；故意不用 `add_start_index`，自带 `_source_span` 精确溯源。
- `semantic_vector.py`（`"V"`）用句向量余弦距离找断点，默认 `percentile` 阈值，句切正则 `DEFAULT_SENTENCE_SPLIT_REGEX` 扩展了中文 `。？！`；它是唯一 async 策略，靠 `_AsyncEmbeddingFuncAdapter` 桥接异步嵌入；无 `embedding_func` 时降级到 R。
- `paragraph_semantic.py`（`"P"`）消费 `.blocks.jsonl` sidecar，跑 HeadingBlocks→TableRowSplit→AnchorSplit→HeadingGlue→LevelMerge 五阶段流水线，阈值全部以比例（`_IDEAL_RATIO=0.75` 等）乘 `target_max` 得到，默认 `chunk_token_size=2000`；sidecar 缺失时降级到 R。
- R、V、P 都**不在内部强制 token 上限**，超限块或交给 `enforce_chunk_token_limit_before_embedding` 硬切，或主动调 R 再切（V/P）。
- 多个 chunker 之间用懒导入互调以绕开循环依赖（V→R、P→R）。
- chunk 的最小公约数字段是 `tokens`/`content`/`chunk_order_index`，P 额外富化嵌套 `heading`（`level`/`heading`/`parent_headings`）与可选 `sidecar`；`full_doc_id` 等归属字段由上游管线补齐。
- `chunk_schema.py` 给 heading/sidecar 定义统一规范化与清洗规则（面包屑 `→`、剔除 Cc/Cf、层级截断 80 字符），让 chunker 与抽取/查询阶段共享同一份合同，这正是切分层"可插拔"的物理基础。

## 动手实验

1. **实验一：观察重叠与窗口步长** — 在 Python 里构造一段约 3000 token 的长文本，分别用默认参数和 `chunk_overlap_token_size=0` 调用 `chunking_by_token_size`，打印每个 chunk 的 `tokens` 与 `chunk_order_index`，验证步长从 1100 变为 1200、相邻 chunk 是否还共享尾部内容。
2. **实验二：跟踪递归回退链** — 准备一段含双换行、单换行、长无分隔符硬块混排的文本，用 `chunk_token_size=50` 调 `chunking_by_recursive_character`，对照 `_split_text_with_spans` 的逻辑观察哪些块在 `\n\n` 处切开、哪些被递归降级到更弱分隔符甚至逐字符。
3. **实验三：中文断句对比** — 取 `DEFAULT_SENTENCE_SPLIT_REGEX`，分别用 LangChain 原版英文正则和它对同一段中英混排文本做 `re.split`，对比中文 `。？！` 处是否被正确断开，理解 V 策略为何要扩展句切正则。
4. **实验四：标题面包屑规范化** — 构造一个 `heading` 含全角空格、零宽字符、内嵌 `→` 且超过 80 字符的 chunk dict，调用 `format_heading_context` 打印结果，验证 `_clean_heading_text` 的清洗与 `DEFAULT_HEADING_LEVEL_MAX_CHARS` 截断如何把它变成一行干净面包屑。

> **下一章预告**：chunk 切得再好，前提也是文档已经被正确读进内存、解析成纯文本与结构化 sidecar。第 4 章我们走进文档解析 Parser，看 LightRAG 如何在 native / mineru / docling 之间分发，把 PDF、docx、表格、公式还原成切分器能消费的 `.blocks.jsonl`。
