# 当前 Agent 设计与代码组织

模型指令使用英文，共享规则、价格约定和监督指令在 `agents/prompts/instructions.py`，仿真动作指令与对应的 SFT 消息同放在 `agents/prompts/customer.py` 和 `agents/prompts/seller.py`，反思使用 `agents/memory/prompts.py`。反思和 Tick 专用指令独立于 SFT 契约。Query 输出中文，原始商品、品牌及搜索文本保持原样。

当前入口：[运行与样本说明](../agent_simulation/README.md)。完整样本可用文档中的 `datasets.labels preview` 命令按需生成。

流程图：[Customer](assets/agent_architecture/customer_architecture.svg)、[Seller](assets/agent_architecture/seller_architecture.svg)、[Memory与Retrieval](assets/agent_architecture/memory_retrieval.svg)。这些图保留生成时的架构，具体字段以本页和当前代码为准。11月3日真实Qwen已执行2家店、2名Customer、3个Tick，闭环检查通过；11月2日的5次中文调用仅是离线档案。

## 上下文按业务组织

Seller 的任务是为指定商品报价。`decision` 给出目标与截止点，业务证据分为三块：

| 业务块 | 当前模型字段 | 用途 |
|---|---|---|
| 本店当前状态与近期表现 | `current_state` | 商品属性、最近观测报价、仿真有效报价、最近报价段及同段曝光／点击／购买；店铺共同窗口报表 |
| 本店历史依据 | `history.price_changes`、`history.similar_evidence`，仿真有记录时附 `history.long_term_memory` | 较长价格变化轨迹、与本次相关的具体事实片段，以及有来源的经验 |
| 外部价格参照 | `external_reference` | 近期同类市场统计和最多三家异店竞品；市场总体和精选案例分别解释 |

报价和对应反馈放在同一商品的观察序列中。同价且同状态的相邻观察可压成一段，区间只表示首尾观察，保留观察次数。检索结果按独立片段展示并各有引用。仿真反馈结算后进入事实历史，观测索引、结算步和逻辑时间各自保留。长期经验是派生判断，SFT 没有这类监督，不能把它当成另一份经营记录。

v15 的 `price_changes` 明确累计 `observation_range`（首末观测索引，均含端点），所有记录严格早于来源的 `end_exclusive`。累计变化次数和不同价格数覆盖完整历史前缀，`runs` 最多保留最近四段；只有价格、范围、观测数和反馈计数全部与近期块一致时才省略，并以 `recent_evidence_ref` 指向近期证据。部分重叠保留，不猜测缺失期间；不同检索片段即使价格相同也分别展示。观测索引不是天数，连续观测段不表示两端之间一直维持该价。

Customer 的两个决策共享画像和历史，当前任务信息按阶段出现：

| 业务块 | 当前模型字段 | Query / Response |
|---|---|---|
| 用户背景 | `profile` | 两阶段共用 |
| 最近消费行为 | `short_term_memory.search_history` | 最近 10 次请求，搜索词与点击／购买及当时价格一起看 |
| 相关历史与经验 | `retrieved_context.history`、`long_term_memory` | Query 用过去行为检索；Response 可用本次 Query；SFT 长期经验为空 |
| 当前选购任务 | `query`、`displayed_items` | 仅 Response 出现，包含当前批次商品和价格；答案单独作为标签 |

Customer 字段名仍偏技术术语，但不需要为统一外观再加一层摘要。两类 Agent 可以共享编码、排序、权限／时间规则，业务投影分别实现：Seller 需要价格段和反馈统计，Customer 需要请求顺序与商品曝光。

## 本轮文件职责核查

[逐文件清单](../../audit/context_cleanup_20261107/inventory.json)记录当前源码、配置、测试和文档的职责、定义及导入；[核查结论](../../audit/context_cleanup_20261107/result.json)区分已修复项与建议。模型、原始数据、历史分片和审计快照不作为重复源码删除。

| 层 | 职责与边界 | 整理判断 |
|---|---|---|
| `agent_simulation/data` | 读取事实、时间窗口、纯统计、来源核验 | 数据代码位于项目包内，只依赖数据层 |
| `agents/context` | 按决策选择并组装授权事实 | 组织角色事实与业务投影；已删除未被模型使用的店铺保留窗口汇总 |
| `agents/context/*_evidence.py` | 业务证据投影、去重与审计映射 | 已迁入 `context`，两角色独立；消息格式仍在 `prompts` |
| `agents/prompts/{seller,customer}.py` | 模型消息与紧凑表示 | 两角色保留独立；报告复用其投影／解码，不自行重建业务视图 |
| `agents/retrieval` | 选取相关事实／经验／竞品，共用 BGE、余弦和 RRF | 共享底层已存在；保留三类证据各自的候选和权限规则 |
| `datasets` | 标签、按时间回放、完整请求展开、容量与分片 | 与运行时共用业务规则；直接读取和增量回放保留不同访问方式 |
| `environment`、`simulation` | 前者掌管状态／权限／结算，后者编排时钟与角色 | 不合并；Agent 不直接写数据库 |
| `inference`、`training`、`evaluation` | 执行模型、训练、评价，各有生命周期 | 不合并；共享 tokenizer／样本格式即可 |
| `scripts`、`docs` | 诊断、展示、当前说明与历史档案 | 修复 Token 归因和报告对旧结构的假设；当前说明不链接不存在的模块 |

已完成小模块整理：Customer Response 和 Seller 定价的分批上下文均由 `environment/context.py` 准备，Customer 的仿真和 SFT 共用 `environment/batching.py` 的纯切分函数；JSON 持久化统一调用 `logs.write_report`，由该函数建目录并原子替换。删除两个无消费方的 `shop_summary.py`，保留真实共同窗口 `observed_shop_period`、仿真经营报表、原始覆盖统计和历史检索；不再为目标样本维护一套未展示的店铺汇总。

## 数据、Agent 与环境

```text
KuaiSearch原始记录
  → 标准化事实、经营观测、冻结划分
  → Customer／Seller共享输入构造
  → Agent提交动作
  → 环境校验、价格生效和反馈结算
  → 新经历进入历史、检索及长期经验
```

数据层负责事实、历史筛选和纯统计，不导入Agent代码。两类Agent分别组织模型输入和决策。Environment负责权限、价格和结算，Agent不直接改数据库。SFT只添加真实监督，不维护第二套清洗逻辑。

## 代码结构

`agents/` 收拢 Customer、Seller 以及它们共用的输入、协议、Memory 和检索。`datasets/samples/` 构造角色样本，`datasets/labels.py` 按需展开输入，`training/` 执行模型训练。环境与推荐流水线分别位于 `environment/` 和 `platform/`。

```text
agent_simulation/
├── agents/
│   ├── customer.py
│   ├── seller.py
│   ├── memory/
│   ├── retrieval/
│   ├── base.py
│   ├── context/               shopping.py、pricing.py、market.py
│   ├── actions.py             共用动作格式与校验
│   ├── decisions.py
│   ├── schemas.py
│   └── prompts/
├── logs.py                   模型调用、角色执行记录与 JSON 报告写入
├── data/
│   ├── catalog.py  事实读取入口
│   ├── common.py / history_records.py
│   ├── build/           经营表、官方划分与目录构建
│   └── market/               商品、市场、分布统计与缓存
├── datasets/
│   ├── samples/                  Customer、Seller 样本构造
│   ├── labels.py                 标签导出、样本读取和预览
│   ├── sampling.py               分层抽样与供应配对
│   ├── tokens.py                 Token 文件构建、读写和恢复
│   └── tokenization.py            编码、答案掩码和组批
├── simulation/
│   ├── run.py
│   ├── episode.py             Tick 调度与逻辑时间
│   ├── replay.py              读取已提交轨迹并导出或展示回放
│   └── templates/             HTML 回放模板
├── environment/
│   ├── market.py / store.py    环境入口与状态事务
│   ├── context.py / batching.py  上下文装配与纯分批
│   ├── memory.py              证据与长期记忆持久化
│   └── operations/            actions.py、queries.py、reporting.py、api.py
├── platform/
├── inference/
├── training/
├── evaluation/
└── tests/
```

从一个 Seller 报价回合追代码，可以沿这条路径读。

1. [context.py](../agent_simulation/environment/context.py) 读取本轮市场快照，确定店主、决策时点、活动价格和已结算反馈。
2. [pricing.py](../agent_simulation/agents/context/pricing.py) 组织自有商品和经营记录，[market.py](../agent_simulation/agents/context/market.py) 补充该卖家可见的市场统计。训练取样直接调用这些事实函数。
3. [prompts/seller.py](../agent_simulation/agents/prompts/seller.py) 把事实写成消息。[actions.py](../agent_simulation/agents/actions.py) 为训练、评估和仿真提供同一份 JSON 报价约束。
4. [seller.py](../agent_simulation/agents/seller.py) 发起决策，[operations/actions.py](../agent_simulation/environment/operations/actions.py) 校验本批草案，环境在规定的轮次边界统一生效。

只读事实查询由 `agent_simulation.data.catalog.DataRepository` 提供。`environment/context.py` 组织模型输入，Customer／Seller 选择动作。`environment/market.py` 管理市场状态和结算，`simulation/episode.py` 管理逻辑时间和多主体调度。

## 从训练样本到仿真与评估

| 想核对的事情 | 入口 |
|---|---|
| 谁被抽中、一次请求展开成几条训练样本、冻结清单是否仍匹配 | [datasets/labels.py](../agent_simulation/datasets/labels.py)。正式训练按 [datasets/labels.py](../agent_simulation/datasets/labels.py) 遍历标签池；逐条对照和统一读取入口见 [datasets/labels.py](../agent_simulation/datasets/labels.py)。 |
| 哪些 token 参与监督、不同批次怎样加权 | [datasets/tokenization.py](../agent_simulation/datasets/tokenization.py) 生成掩码和权重，[training/data_loader.py](../agent_simulation/training/data_loader.py) 组装张量，[training/trainer.py](../agent_simulation/training/trainer.py) 计算损失并处理 epoch、验证和保存。 |
| 一次模型请求怎样执行、何时超时 | [inference/local/qwen.py](../agent_simulation/inference/local/qwen.py) 执行本地模型，[inference/local/service.py](../agent_simulation/inference/local/service.py) 管理队列和请求时限。 |
| 一轮仿真何时启动模型服务、卖家报价何时生效 | [simulation/run.py](../agent_simulation/simulation/run.py) 按运行配置启动并关闭服务，[simulation/episode.py](../agent_simulation/simulation/episode.py) 编排一轮内的角色行动和结算。 |
| 怎样比较预测与历史行为 | [单步评估](../agent_simulation/evaluation/micro/behavior_prediction.py) 读取留出样本并执行预测，将商品批次合回原请求再计分。[轨迹对照](../agent_simulation/evaluation/macro/trajectories.py) 比较已完成的仿真轨迹。 |

## Customer

一名Agent对应一个真实用户。输入包括真实Profile、最近10次搜索／点击／购买、BGE检索的较早经历和已有长期经验。当前商品列表只在Response阶段加入，和历史分开。

最近 10 次请求和较早检索共享同一份经过归属、截止点校验的历史快照。提示词按搜索请求保留查询、点击、购买和时间，价格取当时的展示快照。原始记录只有未知价时保留 `null`，仿真有明确的未知价原因时一并保留。较早检索片段保留各自的时钟，片段时间可直接从事件读取。SFT 的原始会话 ID 可在样本 `input` 中查到。仿真记录实际模型请求，并按近期和检索分组保存对应的原始会话 ID。

SFT同一请求提供两步监督：此前历史→真实Query；此前历史＋真实Query＋曝光商品→真实点击／购买集合。目标Query不提前进入Query输入，响应标签不进入Response输入。当前长期经验为空，不伪造工具、理由、退出或经验标签。

仿真每Tick可搜索、等待或退出。搜索之后再响应当前列表，允许空集合、多点击、多购买；当前数据不强制购买属于点击集合。等待、空响应和退出分别表示。最多 64 件商品时一次响应。更长的列表按固定数量分批，批次各自判断点击和购买，合并后按原请求结算一次。分批决策只比较本批展示商品。每次模型调用仍检查实际 Token 上限。

## Seller

一店一个Agent，管理多个商品。默认接收全店概览、目标商品近期事实、从目标较早历史和同店其他商品近期／较早历史中检索的片段、同类市场摘要、最多 3 件异店竞品例子及已有长期经验。成本、库存等未知字段不编造。公开竞品的发现和读取仅供后端使用，模型不主动调用查询。

近期经营窗口按商品保留最近若干个观察点，结算反馈用独立的仿真时钟。基础上下文保留购买记录数，统一店铺周期的经营总量另列。近期明细只随报价目标展开，同店其他商品仍可作为检索参考。 同一轮仿真或同一时点的 SFT 目标共用只读店铺索引，每个目标视图独立补充市场和检索证据。报价目标必须在概览的明细范围内。

SFT每条监督一个可靠商品价格观测点，输入包含其他商品上下文；无同一时点标签的商品不补造价格。不变价样本保留，目标观测价不称为真实后台编辑动作或最优价格。

仿真可保持或调整商品价格。大店分批作草案，全部批次成功后统一生效。下一次判断由全局Tick调度；Seller当前不输出自主复查时间，也没有退出动作。

市场默认摘要突出价格和计数。稀疏计数的三个分位数全为零时，用非零商品数和可比商品数替代精细排名；比率附分子分母，完整分布仍可查询。价格观测点、区间统计与累计统计各自标明时间。价格原值按沙箱分单位约定解释，来源单位尚未确认。

## 事实与长期经验

短期 Memory 由 `agents/context/shopping.py` 和 `agents/context/pricing.py` 从完整历史中选取结构化记录。长期经验由 `agents/memory/` 管理。并列的 `agents/retrieval/` 同时服务较早事实查询和长期经验检索。

以卖家再次报价为例，三类读取各有入口。

- **找过去发生过的事。** [history.py](../agent_simulation/agents/retrieval/history.py) 从目标商品较早记录以及同店其他商品的近期和较早记录中挑出经营片段。顾客同样从自己的较早请求中检索。训练和仿真共用这些事实规则。
- **找此前形成的判断。** [long_term.py](../agent_simulation/agents/retrieval/long_term.py) 筛出当时属于该主体、仍有效的总结，按本次决策排序，并核对来源。仿真自动准备上下文和显式的 `query_long_term_memory` 都经过 [environment/memory.py](../agent_simulation/environment/memory.py) 再调用它。
- **查指定区间的原始记录。** 程序调用 `query_business`、`query_customer_history` 查询后端时（当前模型不开放这两个工具），[environment/operations/queries.py](../agent_simulation/environment/operations/queries.py) 按主体和时间范围读取记录。工具调用先由环境校验权限和参数。

写入发生在反思阶段。[reflection.py](../agent_simulation/agents/memory/reflection.py) 按时间处理下一批 10 条经历，可从已处理事件里再挑最多 5 条作对照，然后由模型显式选择保存、修订或撤回。模型按 Customer／Seller 角色生成简短经验总结，事实通过局部 E 引用支撑总结，M 引用指定需替换或撤回的旧记忆；不使用纯事实摘录替代长期经验。证据快照的 SQLite 读写、总结及处理进度的事务提交归 [environment/memory.py](../agent_simulation/environment/memory.py)。

| 职责 | 当前代码 |
|---|---|
| 纯历史记录与共享窗口配置 | `data/history_records.py`、`inference/model_policy.json` |
| Seller近期事实上下文 | `agents/context/pricing.py` |
| 全店概览与历史展开 | `agents/context/pricing.py` |
| 同类市场与竞品公开信息 | `agents/context/market.py` |
| 把角色事实写成模型消息 | `agents/prompts/customer.py`、`agents/prompts/seller.py` |
| 更早事实片段检索 | `agents/retrieval/history.py` |
| 中文BGE编码与语义排序 | `agents/retrieval/bge_encoder.py`、`agents/retrieval/similarity.py` |
| 已有长期经验检索 | `agents/retrieval/long_term.py` 在主体可见的经验中检索与当前决策相关的证据 |
| 反思触发条件 | `agents/memory/reflection.py` 中的 `ReflectionTrigger` |
| 长期经验总结 | `agents/memory/reflection.py` 组织证据并调用模型，`agents/memory/validation.py` 校验输出与引用 |
| 长期经验的证据与版本 | `agents/memory/evidence.py`、`agents/memory/versions.py` |
| 运行时经验持久化 | `environment/memory.py` |

表中Agent路径相对`agent_simulation/`。

近期事实不经LLM改写。Customer保留跨主题历史；Seller每商品保留最近10点。更早事实默认由中文BGE结合商品关系和经营数值选择，最多5段；近期记录不被重复检索替代。完整事实仍可查。

v16 只按完整观测身份去除重复引用：商品、来源时钟、索引及内容必须相同；不同时间的同价、同反馈片段仍分别参与相关性排序，最多选五段。近期和历史之间的展示去重仍由业务投影处理。相同特征和文档向量可以共享计算，不能因此把不同日期当成同一事实。当前没有额外模式配额或相似度阈值。

较早事实和长期经验使用以下检索规则：

- 较早事实：先限定本用户或本店、可用时间及权限；Customer 排除近期窗口，Seller 只排除当前目标已展示的近期记录，再融合语义、实体关系和可用经营数值的排序名次。Customer每段是一条请求；Seller同一商品、同一时钟下每段至多3个原始点。返回的是原始记录，数值不由LLM重写。
- 长期经验：先限定本Agent有效、当时可用的经验版本，先以BGE余弦相似度排序，同分时新近程度优先。Seller经验若引用了同商品的经营明细，还比较每个观测点的平均曝光和购买记录数，与语义名次融合；最多返回5条。返回角色相关的经验总结或工作假设，保留事实来源和版本；推断不视为原始事实。
- 市场参照：按同完整三级类目、异店、截止点及统计区间计算分布，是结构化统计查询，不使用BGE。默认竞品例子先按属性、关键词和价格缩小候选，再用 BGE 排序；后台竞品查找接口按字面条件过滤、ID 分页。商品筛选与经营片段检索是不同候选池。

#### Seller 市场信息的可见性

普通 Seller 的信息分为以下四类。同一披露口径用于 SFT、仿真首轮报价、只读查询和记忆证据。

| 信息 | 可见事实 | 边界 |
|---|---|---|
| 本店经营 | 本店历史报价、曝光、点击、购买记录及本店比率 | 明确截止点和分母，历史、已结算仿真记录分别统计。 |
| 竞品公开报价 | 同三级类目其他店铺的公开属性、价格、价格分布和覆盖范围 | 标注商品和店铺数、观测索引；近期只有一件时直接呈现其报价。 |
| 竞品公开购买 | 窗口内购买事件计数及分布；可查具体公开商品的购买数 | 按本项目设定公开。每条计数代表一条日志购买标记，真实售出件数、成交金额均未知。零表示该日志窗口内没有购买记录。 |
| 竞品流量 | 未披露 | 曝光、点击及其分位数、排名、CTR、购买率均不返回；不能用零代替未知。平台分析函数可计算这些量，Seller 端只取明确允许的字段。 |

默认报价只读取同三级类目、异店、最近 10,000 个观测索引内的报价中位数和覆盖范围。竞品例子单独排序，较早报价保留原观测时间。仿真另给当时有效报价和指定窗口内购买记录的中位数，同一店铺、类目和时间窗复用摘要。需要价格带、历史购买记录和完整分布时使用后台查询。

可见性策略发生变化后，新仿真使用 v4 存储；v3 中原有的市场证据和自由文本总结不能直接用于新报价。旧日志保留用于离线复盘。

长期总结的来源若同时出现在召回事实片段中，Seller 模型输入通过 `evidence_ref` 标明关联，原始片段号、行号和 `source_ids` 保留在审计信息中。

Customer的Query阶段用近期历史中的搜索词检索较早经历，不读待预测的真实Query；Response阶段可以用已知的本次Query。相关性参与排序，但不会删除近期跨主题历史。Seller用关注商品的标题和当前已知经营特征检索较早同店片段，允许参考同店其他商品。BGE只编码这些检索视图，长文本分块编码后合并向量，完整原始记录继续保留。

当前默认workflow自动执行上述读取和检索。查询后端中的`query_business`、`query_customer_history`按ID和时间查询更多事实，`query_long_term_memory`查询独立经验库；接口分别返回事实和生成经验，当前模型工具权限见[工具说明](../agent_simulation/README.md#tools)。

反思以每个主体的已读游标为界。满 10 条时，本批 10 条按时间顺序全部进入总结输入；检索只从游标之前选最多 5 条旧事实作对照，游标以后的下一批尚不可见。已有长期总结与事实分栏，允许新增、修订、撤回或不更新。合法的不更新只推进游标，失败保留待处理批次。来源、生成时点与版本供后续决策追溯。

长期总结是可修订经验。模型草稿在写入前统一核验字段、证据和版本；坏草稿保留原始输出与原因，不消耗事实批次，同一 Tick 不重复尝试，下一 Tick 可再次尝试该批次。此限制在当前环境实例内跨 Agent 共享；非 Tick 模式不抑制下一次显式调用。配置、容量、生成器内部或设备错误向上抛出。主体隔离、时间边界、编码器身份与事务内状态一致性分别在对应边界检查。

全局 revision 覆盖市场与长期记忆状态；调度先完成记忆更新，再冻结同一阶段的决策输入。Agent 日志中的 `model_calls[].call_id` 关联模型原始输出日志；`memory_updates[].seconds` 记录每批成功或草稿被拒绝的更新总耗时，包括上下文准备、检索、生成与持久化，不是纯 LLM 推理时间。

## 全局 Tick

1. 所有Seller根据此前已结算状态决定价格或等待。
2. 价格统一生效，各Seller不能提前看到其他商家本Tick草案。
3. Customer决定搜索、等待或退出；平台返回列表，Customer生成响应。
4. 所有响应决定后统一结算，更新用户历史、商品经营记录及后续经验来源。
5. 推进Tick；达到上限或全部Customer退出时结束。

完整无活动区间可压缩展示起止范围，历史数据缺口仍为未知。历史索引、仿真事件索引、逻辑Tick是不同的时钟，不能直接换算成自然日期。

程序自动准备上下文，模型只提交最终动作。纯查询、Agent决策、环境状态变更仍分别实现。`environment/context.py`装配运行输入，`agents/prompts/`组织本轮角色消息，`agents/decisions.py`准备请求并解析输出，Customer/Seller 控制器调用环境执行动作，`environment/operations/reporting.py`提供结算后的经营统计。

## 实际完成范围

- Lite全量真实标签已整理；完整Prompt按需展开，未声称全量动态容量逐条通过。
- Full 与 Lite 共用原始数据、时间划分和 SFT 取样实现。真实数据已在 Lite 抽查。Full 的 32 分区布局另用四份小型 JSONL 测通了原始数据到训练取样的整条链，覆盖 64 位商品 ID、空分区、完整标签池及答案掩码。Full 原始文件仍需就位后单独验证。[构造与取样入口](../agent_simulation/README.md#sft-data)
- `simulation/run.py --sellers` 支持多个 Seller 共享同一市场。
- 当前平台是随机或BM25接通后端，正式Recall／Ranking与Bandit仍待训练接入。
- SFT输入、工具、结算与历史检索可执行；模型行为质量、自由运行真实性和政策效果需要继续验证。

## 参考架构的使用边界

[E-CommerceBench](https://github.com/QwenLM/E-CommerceBench)用于参考Agent、工具与统一环境的职责分工；不会引入其营业日时钟或固定需求模型。Mirroring Users相关讨论用于参考用户情境、商品信息和行为决策的组织，不改变当前真实点击／购买监督。文献细节沿用原研究材料，本次只做代码组织调整，未重新核验论文结论。

代码、依赖及命令见[Agent说明](../agent_simulation/README.md)、[数据层实现](../agent_simulation/data)、[样本准备](../agent_simulation/README.md#sft-data)和[模型训练](../agent_simulation/README.md#training)。核验结果集中在[审计目录](../../audit)。
## 与 AgentSociety、MALLES 的对应关系

当前 Episode 给每个活跃主体每 Tick 一次决策机会，分别负责参与者选择、阶段执行和统一结算；Customer、Seller 决定具体动作。Customer、Seller 和记忆总结共用一个本机 Qwen 推理进程。Seller 准备完记忆后共享一份带版本号的市场快照，各店只保存本店原价和报价草案。提交时在事务内检查版本，旧报价不能覆盖新状态。每轮报价按 Seller ID 排序，在一个 SQLite 事务中生效，然后才调度 Customer。按角色设置唤醒周期仍属扩展设计。

阶段内使用有并发上限的 worker：商品或响应批次完成一个就接下一个，返回结果仍按输入顺序排列；任一任务失败会取消尚未完成的任务并报错，不重试。Seller 的首个批次仍先决定是否整店等待，其余批次才并发。Customer 在记忆准备结束后共享 Query 快照，在全部搜索完成后另取 Response 快照；所有响应完成后统一结算，确认每个结算回执后再处理反馈记忆。并发请求上限和市场结算屏障保持独立。

训练和留出评估共用 `datasets/labels.py` 选择业务记录。Customer 的选择单位是一条请求，展开时保留 Query 及全部 Response 批次；评估将各批次合回同一请求后才比较购买行为。观测价格对照以同一商品的不同价位计算中位数，在中位数两侧分别按真实曝光次数计算购买率，再对有对照的商品等权平均。

| 需求 | 可以借鉴的设计 | 本项目的边界 |
|---|---|---|
| 不同角色和行动频率 | 共享时钟，由调度器按各主体的下一次唤醒时间选择参与者 | 沿用独立的 Customer、Seller 输入和动作协议，分别记录获得机会、主动等待和未获调度 |
| 多主体共享推理 | 一个 Qwen 服务按请求角色选择适配器，集中限制在途请求并记录排队和执行时间 | 请求携带 Tick、主体和阶段标识，迟到或重复响应不能重复结算 |
| 共享环境和 Ray 分组 | 按组分发只读快照，并行计算同一阶段的提案 | 市场统一提交价格，随后收集 Customer 响应并集中结算。分组不能使后一个主体提前看到同轮反馈 |
| 实验中途改政策 | 在预先声明的 Tick 边界应用带版本的干预，保存生效时点与参数 | 例如从第 10 Tick 起改变低价排序权重，用相同初态和种子与原规则对照，保留两条轨迹 |

单机推理服务使用进程间队列和管道，由 Episode 命令启动并回收。可按角色加载 Customer 和 Seller 的行为 LoRA，记忆总结使用底座。Customer 的 Query 和 Response 分阶段并发，各阶段共用同一请求上限。跨机器的 Ray 分组与按 Tick 切换政策属于后续扩展。[AgentSociety 官方说明](https://agentsociety.readthedocs.io/en/latest/)

留出评估入口 `python -m agent_simulation.evaluation.micro.behavior_prediction` 支持在既有 Customer 命令上追加 `--price-response`。同一请求固定原始展示顺序、其他商品价格和历史，按种子选择一个展示商品，比较原价及上下 20% 条件。每个改价条件只对目标所在批次推理一次，复用其他批次的原价结果。评估文件 `price_response.json` 保存每个条件下完整请求的购买集合，并在同一批有效配对上报告目标商品、其他商品及整单购买比例。这些比例来自单次贪心解码的模型选择。原价预测和真实标签另用于比较商品及一级品类的购买分布，并报告同商品高低价组的曝光数和购买率差。真实日志只记录实际展示价下的购买，组间差异是观察性关联。

MALLES 的价格与促销条件、群体稳定性和校准实验可以借鉴。建议先报告原始 Agent 的配对响应和分布，再将 Mean-field 反馈作为独立实验条件。若做额外校准，只用训练／验证期拟合，在独立测试期并列报告校准前后结果，并按品类、顾客和曝光频次检查是否掩盖局部偏差。[MALLES 原文](https://arxiv.org/html/2603.17694v1)

### 原始历史背景与行为事实

Customer 的 `behavioral_summary` 保存决策截止点之前最近一次 Ranking 快照：
30 天曝光、点击、订单、GMV，以及原始近期点击/购买 ID。`item_details` 从静态
商品目录补充标题、品牌、类目等属性，不补造当时价格或行为时间。原始数组保留重复，
但不解释为购买件数，也不与搜索历史合并计数。没有更早快照时为 null。
近期搜索仍为最近 10 次请求，较早历史仍走现有检索，长期记忆仍是可修正的经验。

Seller 的自有商品 `business_summary` 保存最近已知观测的原始 30 天经营统计，
与近期观测计数、已结算反馈分开。它进入最终 Seller Prompt，但不进入竞品公开信息，
也不包含消费者个人统计。SFT 和仿真复用同一组装规则。

两类背景均采用严格 `time_index < cutoff`：不使用当前监督行，也不使用同时间其他
请求。原始近期列表缺少逐事件时间，当前不能证明它在 Query 生成前可见，因此统一
使用较早已知快照。搜索入口也不从待预测请求注入 Query。仿真固定初始化 cutoff，
原始背景不随 Tick 更新；新行为只进入现有 History Store，不伪装成新的滚动 30 天值。

构建完整 Catalog 时会生成 `facts/customer_background.parquet`。已有 Catalog 可运行：

```bash
python -m agent_simulation.data.build.catalog --config data/kuaisearch/lite/config.json \
  --output data/kuaisearch/lite/source/catalog --customer-background-only
```

这只补充每请求一条的事实快照，不改标签、旧 Token 缓存或模型权重。商品属性在读取时
关联，不在快照里重复存储。新 Prompt 必须生成到新的 Token 缓存目录；构建时绑定背景
文件摘要，防止续建时混入旧上下文分片。训练读取旧缓存仍使用其原有 Token，不重扫数据。
旧模型生成评估若使用新上下文，应作为新的输入条件记录，不能混入原评估输出目录。


Customer 背景商品描述可用 `--compact-customer-background` 构建对照缓存：保留
`item_id`、`item_title`、`brand_name`、`category_level3_name`；近期点击/购买 ID、
30 天统计、搜索历史、检索和当前展示商品均保持不变。默认仍保留完整商品描述。
该选项只改变模型可见投影，不改原始事实或监督答案。缓存记录这一选项，必须写入
独立目录；不会覆盖完整信息版。只构建 Customer 时使用 `--role customer`。
训练和运行时消息函数共用这项投影选项，运行时默认行为仍不变。

### Customer 分阶段 LoRA 对照

旧上下文实验直接读取原 `runs/lite_2pct_goal/tokens`，用训练入口 `--stage query`
或 `--stage response` 选择样本，不重新渲染 Prompt，不重分批，也不调整 Response 的
`1/batch_count` 权重。两个独立输出目录都从冻结的 Qwen Base 初始化自己的 LoRA。
训练阶段的隔离不改变 History、Retrieval、Platform 或 Environment。

`--loss-breakdown` 在评估的同一次前向计算中记录 Query/Response 各自的答案 Loss，
以及 Response 的结构、点击 ID、购买 ID 三组 Token Loss。ID 分组依据真实答案中
数字的字符跨度；结构组包含字段名、标点、空列表和结束符，因此并非全部是无业务
含义的固定格式。分组 Token Loss 按请求权重加权后除以该组 Token 权重；另记录
各组对答案平均 Loss 的贡献，三组贡献之和等于 Response 答案 Loss。没有 ID 的
样本不虚构该组 Token。诊断不改变反向传播的原有答案平均 CE。

`--evaluate-train` 在初始评估和每轮 Test 评估后，用同一时刻的权重、关闭 dropout，
完整计算 Train Loss；不更新参数，也不消费或修改训练流的 shuffle 状态。
指标与 epoch/step 一起记录到 Trainer 的 `log_history`，保存于各 checkpoint 的
`trainer_state.json`，初始值保存于 `initial_evaluation.json`。Test 仍用于选择
checkpoint，并非独立于选择过程的最终测试集。`--keep-checkpoints 0` 保留所有完整
checkpoint（含 optimizer/scheduler/RNG），不只保存 Adapter 权重。

本次旧上下文对照实际训练 5 轮，使用 `--lr-schedule-epochs 10`，匹配旧联合训练
“按 10 轮衰减、停在第 5 轮”的每轮学习率进度。默认未指定此选项时，学习率仍按
正常训练步数衰减。Response-only 先运行，随后运行 Query-only；Seller 不在队列中。
