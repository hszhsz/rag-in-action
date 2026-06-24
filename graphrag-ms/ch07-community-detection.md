# 第 7 章 社区检测 Leiden

第 6 章结束时,我们手里已经有了一张「干净」的知识图谱:实体经过去重与描述摘要,关系拥有了合并后的权重,整张图被终化(finalize)成两张规整的表 `entities` 与 `relationships`。但这张图是「平的」——它只告诉你哪两个实体之间有边,却没有任何关于「主题聚簇」的结构。一部小说里的人物可能分属几条故事线,一份技术文档里的概念可能分属几个子系统。GraphRAG 想要回答「整篇语料讲了什么」这类全局问题,就必须先把这张平图组织成**多粒度的层次社区**:粗粒度社区给出宏观主题,细粒度社区给出局部细节。

本章解构 `create_communities` 工作流。它的核心是把 `relationships` 边表喂给 graspologic 的层次化 Leiden 算法,得到「每个节点在每个 level 上属于哪个社区」的划分,再把这些划分整理成带父子关系的社区树,落地为 `communities` 表。这张表是第 8 章社区报告生成、以及后续 Global Search「全局检索」map-reduce 的直接输入——没有社区,就没有 GraphRAG 区别于普通向量 RAG 的那套全局问答能力。

## 工作流入口:从边表到社区表

`run_workflow` 定义在 `create_communities.py`。它的输入并不是节点表,而是关系表:`reader.relationships()` 读出 `relationships`,然后从配置 `config.cluster_graph` 取出三个参数 `max_cluster_size`、`use_lcc`、`seed`,打开 `entities` 与 `communities` 两张表后调用 `create_communities`。

为什么聚类只看边表而不看节点表?因为社区检测是纯粹的**图拓扑**问题:节点的归属完全由它与谁相连、连接有多强决定,实体本身的描述文本在这一步无关紧要。节点表的作用要等到后面把社区里的实体 title 映射回 `entity_id` 时才出现——代码里用 `title_to_entity_id` 字典完成这件事,逐行遍历 `entities_table` 建立 `title -> id` 映射。

`ClusterGraphConfig` 定义在 `config/models/cluster_graph_config.py`,只有三个字段,默认值来自 `config/defaults.py` 的 `ClusterGraphDefaults`:`max_cluster_size = 10`、`use_lcc = True`、`seed = 0xDEADBEEF`。注意 `seed` 默认是那个经典的十六进制魔数 `0xDEADBEEF`,这不是随手写的——它是「可复现」这条设计原则的物理体现,后文会专门讲。

## 为什么是 Leiden 而非 Louvain

社区检测最广为人知的算法是 Louvain,但 GraphRAG 选择了 Leiden。`hierarchical_leiden.py` 里 `hierarchical_leiden` 函数直接调用 `graspologic_native as gn` 的 `gn.hierarchical_leiden`。graspologic 是微软自家的图统计库,其 native 实现底层是 Rust 写的 Leiden。

Leiden 相对 Louvain 的关键改进在于它修复了 Louvain 一个著名缺陷:Louvain 可能产生**内部不连通**的社区(一个社区里的节点实际上分成了几块互不相连的部分)。Leiden 通过额外的「细化(refinement)」阶段保证每个社区内部都是连通的,同时通常能收敛到更高的模块度(modularity)。对 GraphRAG 来说,社区将被整体送进 LLM 生成一份报告,如果一个社区内部根本不连通、主题割裂,生成的报告就会语义混乱——所以社区的连通性质量直接决定下游报告质量,这是选 Leiden 的实质动机 [[graspologic 仓库]](https://github.com/graspologic-org/graspologic)。

`hierarchical_leiden` 调用时固定了一组参数:`resolution=1.0`(分辨率,控制社区粒度)、`randomness=0.001`、`use_modularity=True`(用模块度作优化目标)、`iterations=1`、`starting_communities=None`。其中真正暴露给用户配置的只有 `max_cluster_size` 和 `random_seed`(对应 `seed`)。

## 层次化体现在哪:level 与递归细分

普通 Leiden 给你一层划分:每个节点属于一个社区。**层次化** Leiden 给你多层。理解层次化的关键是 `max_cluster_size`(默认 `10`)。

算法先做一遍 Leiden,得到 level 0 的若干社区。然后检查每个社区的大小:如果某个社区的节点数超过 `max_cluster_size`,这个社区「太大、太笼统」,算法就**只在这个大社区内部**再跑一遍 Leiden,把它打散成 level 1 的若干子社区;子社区如果还超标,继续在它内部细分出 level 2……如此递归,直到所有叶子社区都不超过 `max_cluster_size`。

这就解释了为什么同一个节点会在不同 level 出现:在 level 0 它属于一个大社区(宏观主题),在 level 1 它属于那个大社区拆出来的某个子社区(中观),在 level 2 又属于更细的孙社区(微观)。`max_cluster_size` 本质上是一个「粒度闸门」:它越小,层次越深、社区越细;它越大,层次越浅、社区越粗。

`graspologic_native` 把这套层次结果返回成一个 `HierarchicalCluster` 列表,每个元素带四个关键字段:`node`(节点名)、`cluster`(它在这一层所属社区 id)、`level`(层级)、`parent_cluster`(父社区 id)。`hierarchical_leiden.py` 里还配套提供了 `first_level_hierarchical_clustering`(取 `level == 0` 的划分)和 `final_level_hierarchical_clustering`(取 `is_final_cluster` 为真的叶子划分)两个便捷函数,供需要单层结果的场景使用。

测试 `test_cluster_graph.py` 里有一个很直观的真实数据回归:对《圣诞颂歌》语料的 `relationships.parquet` 用 `max_cluster_size=10, use_lcc=True, seed=0xDEADBEEF` 聚类,得到 122 个社区,各 level 分布恰好是 `{0: 23, 1: 65, 2: 32, 3: 2}`——四层结构,中间层社区最多,印证了「大社区逐层细分」的形态。

## cluster_graph:把原始划分整理成社区元组

`cluster_graph` 操作(`operations/cluster_graph.py`)是 workflow 与 graspologic 之间的适配层。它做三件事。

第一,**预处理边表**(在内部函数 `_compute_leiden_communities` 里)。先 `edges.copy()` 避免污染输入;然后把每条边的 `source`/`target` 用 `min`/`max` 规范化方向(无向图里 `A-B` 与 `B-A` 是同一条边),`drop_duplicates(..., keep="last")` 去重,刻意复刻 NetworkX「保留最后一行属性」的行为;若 `use_lcc=True` 则调用 `stable_lcc` 过滤;最后把边表转成排序后的 `(source, target, weight)` 三元组列表——`weight` 列缺失时默认全 `1.0`。这一连串的「先 min/max 规范方向、再排序」操作不是为了正确性,而是为了**确定性**:同样的图无论行序如何,喂给算法的边列表都完全一致。

第二,**调用算法并拆分结果**。`community_mapping` 是 `HierarchicalCluster` 列表,代码遍历它构造两个字典:`results[level][node] = cluster` 记录每层每个节点的社区归属;`hierarchy[cluster] = parent_cluster`(若 `parent_cluster is None` 则记为 `-1`)记录每个社区的父社区。`-1` 是 level 0 社区的父标记——它们没有父社区,是树根。

第三,**汇成元组列表**。`cluster_graph` 按 level 把节点按社区 id 聚成 `dict[level][community_id] -> [node...]`,再展开成 `Communities = list[tuple[int, int, int, list[str]]]`,每个元组是 `(level, cluster_id, parent, nodes)`。测试 `test_output_tuple_structure` 把这个契约钉死:四元组,前三个 int,最后是字符串节点列表;`test_level_zero_has_parent_minus_one` 钉死 level 0 的 parent 恒为 `-1`。

## use_lcc:只在最大连通分量上聚类

`use_lcc`(默认 `True`)控制是否先把图裁剪到**最大连通分量**(Largest Connected Component)再聚类。实现在 `stable_lcc.py` 的 `stable_lcc`,它依赖 `connected_components.py` 里用并查集(union-find,带路径压缩)实现的 `largest_connected_component`。

为什么要裁掉小分量?抽取出来的图常有「孤岛」:几个只在某个边缘段落出现、与主线毫无关联的实体自成一个小连通块。这些孤岛对「全局主题」毫无贡献,却会污染社区集合。`use_lcc=True` 直接丢弃它们,只对主干图聚类。测试 `test_lcc_filters_to_largest_component` 验证了这一点:两个互不相连的三角形,`use_lcc=True` 时只剩最大的那个(3 个节点)。

`stable_lcc` 的「stable」同样是确定性诉求。它做了四件确定化的事:节点名 `_normalize_name`(HTML 反转义、大写、去空白)、过滤到 LCC、方向规范化(小的节点恒为 source)、按字典序排序。文件头的 docstring 明说目标是「同样的输入边列表,无论原始行序如何,总产生同样的输出」。这也是为什么前面《圣诞颂歌》样例里节点名是大写的 `SCROOGE`、`JACOB MARLEY`——大写发生在 `_normalize_name` 这一步。

## 确定性:seed 如何让聚类可复现

Leiden 是随机算法(`randomness=0.001`),不固定种子的话每次运行社区划分都会有微小差异,导致整条索引流水线不可复现——这对生产环境、对调试、对增量更新都是灾难。

GraphRAG 把确定性做成了贯穿全链路的硬约束。`seed` 从配置一路透传:`run_workflow` 取 `config.cluster_graph.seed` → `create_communities` → `cluster_graph` → `_compute_leiden_communities` → `hierarchical_leiden(..., random_seed=seed)` → `gn.hierarchical_leiden(seed=random_seed)`。默认值 `0xDEADBEEF` 让开箱即用也是确定的。配合前面边表的 min/max 规范化、排序,以及 `stable_lcc` 的全套稳定化,整个聚类对「相同输入产生相同输出」给出了强保证。测试 `test_same_seed_same_result` 直接断言 `c1 == c2`,`test_does_not_mutate_input` 还顺带验证 `cluster_graph` 不修改入参 DataFrame。

## 社区表的最终结构:双向树与归属集合

回到 `create_communities`,拿到 `clusters` 元组列表后,它要把这些拓扑结果「填上肉」,组装成下游可用的 `communities` 表。

先 `pd.DataFrame(clusters, columns=["level", "community", "parent", "title"]).explode("title")`——把每个社区的节点列表炸开成多行,每行一个节点 title。然后做三组聚合:

- **entity_ids**:把每个社区里的节点 title 经 `title_to_entity_id` 映射成 `entity_id`,按 `community` 分组聚成列表。
- **relationship_ids 与 text_unit_ids**:这一步只统计**社区内部边**(intra-community edges)。代码逐 level 处理,把 `relationships` 两次 merge 进当层社区表(一次按 source、一次按 target),只保留 `community_x == community_y`(源和目标在同一社区)的边,再 `explode("text_unit_ids")` 后按 `(community, parent)` 聚合出 `relationship_ids` 和 `text_unit_ids`。注释明说「逐层处理以保持中间 DataFrame 较小」,这是对大图的内存优化。两个列表最后都 `sorted(set(...))` 去重——又是确定性。

组装阶段再补上若干字段:`id` 用 `uuid4()` 生成;`human_readable_id` 等于 `community` 整数;`title` 拼成 `"Community " + 社区号`(所以社区标题形如 `Community 5`);`parent` 转成 int。

特别值得注意的是**children 的构造**。社区树本来只有「子指父」(每行有 `parent`),代码额外按 `parent` 分组 `agg(children=("community", "unique"))`,再 merge 回去,给每个社区补上 `children` 列表——注释写道「collect the children so we have a tree going both ways」。于是 `communities` 表同时持有 `parent` 和 `children`,是一棵**双向可遍历的树**。这非常实用:Global Search 需要按 level 自上而下展开社区,而报告生成又常需从子社区聚合到父社区,双向指针让两个方向都是 O(1) 跳转。没有边的 `parent` 会被填成空列表(`np.ndarray` 判断兜底)。

最后补上 `period`(当前 UTC 日期,用于增量更新追踪)和 `size`(`entity_ids` 的长度,即社区含多少实体),按 `COMMUNITIES_FINAL_COLUMNS` 选列输出。这套列在 `data_model/schemas.py` 定义,与 `data_model/community.py` 的 `Community` dataclass 对应:`id`、`human_readable_id`、`community`、`level`、`parent`、`children`、`title`、`entity_ids`、`relationship_ids`、`text_unit_ids`、`period`、`size`。写表前 `_sanitize_row` 把 numpy 类型转回原生 Python 类型,保证序列化干净,并返回最多 5 行样本供日志查看。

## 设计动机:层次社区是全局检索的地基

把这一章串起来看,`create_communities` 产出的不是一个扁平的社区列表,而是一棵**多粒度社区树**,每个节点(社区)都自带它的实体、内部关系、相关文本单元、父、子、大小。这个结构正是 GraphRAG 全局问答的地基:

第 8 章会对**每个社区**生成一份摘要报告(`create_community_reports`),粗粒度社区报告概述宏观主题,细粒度社区报告补充局部细节。第 12 章的 Global Search 则按某个 level 取出所有社区报告,做 map-reduce:每份报告并行回答一遍用户问题(map),再把部分答案汇总成最终答案(reduce)。这种「先把语料压缩成分层主题、再在主题层面做检索」的范式,是 GraphRAG 相对普通向量 RAG 能回答「整篇讲了什么」「有哪些主要冲突」这类全局问题的根本原因——而这一切,都从本章这张 `communities` 表开始。

## 本章小结

- `create_communities` 工作流以 `relationships` 边表为输入,经 Leiden 层次聚类,产出带父子层级的 `communities` 表,是第 8 章报告生成和 Global Search 的直接地基。
- 社区检测只看图拓扑(边),节点描述文本无关;实体归属在事后通过 `title_to_entity_id` 字典映射回 `entity_id`。
- GraphRAG 选 graspologic 的 `hierarchical_leiden`(`graspologic_native`,Rust 实现)而非 Louvain,因为 Leiden 保证社区内部连通、模块度更高,直接决定下游报告质量。
- 层次化由 `max_cluster_size`(默认 `10`)驱动:超过阈值的大社区会在其内部递归再跑 Leiden 细分出下一 level,同一节点因此在不同 level 属于不同粒度社区。
- `cluster_graph` 把算法返回的 `HierarchicalCluster` 整理成 `(level, cluster_id, parent, nodes)` 四元组;level 0 社区的 `parent` 恒为 `-1`(树根)。
- `use_lcc`(默认 `True`)只在最大连通分量上聚类,丢弃孤岛;`stable_lcc` 用并查集 + 名称规范化 + 排序保证结果稳定。
- 确定性是贯穿全链路的硬约束:`seed`(默认 `0xDEADBEEF`)层层透传到 `gn.hierarchical_leiden`,配合边表 min/max 规范化与排序、列表 `sorted(set(...))` 去重,保证相同输入相同输出。
- `communities` 表只统计社区内部边(`community_x == community_y`)来聚合 `relationship_ids` 与 `text_unit_ids`,并逐 level 处理以控制内存。
- 表同时持有 `parent` 与 `children`,构成一棵双向可遍历的社区树,便于上下两个方向遍历;社区 `title` 形如 `Community N`,`size` 为社区内实体数。

## 动手实验

1. **实验一:观察层次结构** — 在 `tests/unit/indexing/test_cluster_graph.py` 找到 `test_level_distribution`,本地运行它,确认《圣诞颂歌》语料聚出 `{0: 23, 1: 65, 2: 32, 3: 2}` 的四层分布;然后把 `max_cluster_size` 从 `10` 改成 `30` 再跑一次,观察层数与各层社区数如何变浅、变少,体会粒度闸门的作用。
2. **实验二:验证确定性与种子** — 写一段脚本,对同一份 `relationships` DataFrame 用相同 `seed` 调两次 `cluster_graph`,断言结果完全相等;再换一个不同的 `seed`,对比 level 0 的社区划分是否发生变化,理解随机种子对可复现性的意义。
3. **实验三:打开/关闭 LCC** — 构造一个含两个互不相连子图的边表(可参考 `test_lcc_filters_to_largest_component`),分别用 `use_lcc=True` 和 `use_lcc=False` 调 `cluster_graph`,对比输出社区数,确认 `True` 时孤岛被丢弃;再去 `stable_lcc.py` 跟读 `_normalize_name` 如何把节点名大写规范化。
4. **实验四:读懂双向树** — 对一份真实 `communities` 表(或在 `create_communities` 末尾打印 `final_communities`),挑一个 level 0 社区,顺着 `children` 列向下找到它的子社区,再从子社区的 `parent` 列跳回父社区,确认父子指针一致;同时检查该社区的 `size` 是否等于 `entity_ids` 列表长度。

> **下一章预告**:有了这棵多粒度社区树,下一步就是给每个社区「写一份介绍」。第 8 章《社区报告生成》将解构 `create_community_reports` 与 `summarize_communities`,看 GraphRAG 如何把一个社区的实体、关系、文本单元打包成上下文,调 LLM 生成结构化的社区摘要报告,并处理「报告太长放不下上下文窗口」时的子社区递归汇总。
