# Customer 数据构造流程

当前主流程用于离线构造训练数据：作者式候选／干扰项 → 独立 Query 与强弱 Response EKB／直接动作轨迹 → 同一行为预测器的选项概率与熵 → 场景／偏好筛选 → SFT、DPO 导出。候选场景在模型调用前冻结。Query、原始多点击／购买记录、空点击／空购买、fixed／autonomous 两种信息获取模式都保留；Response 按真实正例拆成单正例短列表。

## 作者代码与适配

直接来源：[UserMirrorer](https://github.com/Joinn99/UserMirrorer)，commit `b6e628cdc6e9a439265d5375cdc44bf5d8a29e80`。`agent_simulation/vendor/usermirrorer/` 保存 MIT 许可证、原文件 SHA256 和实际摘取的函数清单。EKB 模板完整复制；熵、概率、场景／偏好筛选、候选评分／合并／插入的函数体原样保留。

候选构造实际调用作者的随机评分、内容相似、协同过滤谱嵌入、Top-K、随机合并去重、历史交互过滤和随机目标插入，不再用 BGE 按当前 Query 截取前 10 件替代这条流程。内容编码器是可替换部件，本项目沿用已提供的中文 BGE；协同过滤沿用作者的度归一化及随机 SVD。

数据域适配位于 `datasets/customer_candidate_sampling.py`、`datasets/customer_scenes.py`：

- 候选池只取当前请求的真实曝光商品，直接在池内调用作者随机、内容相似、协同过滤三路评分／合并。不按 Query 相似度阈值或商品类目硬筛，不使用全站未曝光商品补齐。共享 catalog 仅提供商品事实和协同特征拟合空间。
- 类目仍可参与商品文本编码和数据域统计，Query 在无正例且无可用历史锚点时仍可用于内容评分；两者均不再决定商品是否有资格进入曝光池。scope 记录曝光数、排除其他正例后的采样池大小，以及 query_category_filter=false。
- 协同特征仍只使用冻结截止点前的训练交互。计算用户表示时保留池外历史，再将得分限制到当前候选池，避免曝光池裁剪把用户历史清空；数据 time_index 按原索引比较，不当作 Unix 时间。
- 对点击与购买并集中的每个不同商品分别构造一张列表，每张只插入该商品。其他真实正例从全部干扰候选中排除，不会标成负例。
- 子场景只保留目标商品对应的点击／购买状态：仅点击、仅购买、两者均有均可；同一商品点击且购买只生成一条。原始完整标签、原请求 ID、子场景 ID 和排除的正例 ID 保存在元数据中，不送入模型。
- 每张列表默认总长 2–12；单正例对应 1–11 个干扰项。不再因多正例突破最大长度。无法满足最小长度时报告 insufficient_distractors，不生成全正例列表。
- 不点击且不购买时不插入正例，其候选仅取自该次真实曝光的负例，避免给未曝光商品伪造不点击标签。
- 默认候选总数 2–12、至少 3 个过去正交互，与作者训练配置对应。当前曝光商品进入池；干扰项沿用过去交互过滤，协同特征及历史价格严格使用截止点之前的数据。
- 所有候选统一取严格过去的最近有效价格；无记录时为未知。不会只给真实正例保留当前价格。
- 小矩阵的 SVD 维度作边界适配。构造随机种子局部隔离，场景保存 seed、池大小、正例位置、来源和协议。

这属于作者式、结果条件化的离线选择题构造。真实目标参与题目构造是该协议的一部分；生成模型只看到候选事实，不能看到正例位置和当前动作标签。它与真实市场曝光下的行为评估是两个明确的协议。旧 `full`／`retrieved` 仅作为显式可选协议，默认使用准备好的 `sampled`（采样候选） 场景。

## 完整 Lite Train 的生成前选样

主实验候选池使用完整 Lite 官方 Train，不受旧 quality-3k 或 5% 时间窗口限制。
标签、原始事实和历史仍完整保留；只选择本轮需要构造场景的请求。
在项目目录下：

```bash
./agent_simulation/python.sh -m agent_simulation.datasets.labels build \
  --catalog data/kuaisearch/lite/source/catalog \
  --output result/lite-full-training-NEW/sft

dataset=result/lite-full-training-NEW/sft
./agent_simulation/python.sh -m agent_simulation.datasets.customer_selection \
  --dataset "$dataset" --output result/customer-selection-NEW --per-scene 100
```

`--per-scene` 是可调的**每个行为/历史/曝光分层的原请求上限**，不是全局 3,000 条限制，
也不是每个请求生成多少条 CoT。100 只是预算示例，不是验证过的最佳规模。
选择器检查完整 Train 的 ID 集合，拒绝旧时间窗口；Test 不进入排序和历史质量计算。

分层包括点击/购买四种组合、单/多正例、冷启动/短历史/较丰富历史，以及
1–12、13–32、33–128、129+ 件曝光。层内优先曝光商品标题完整、覆盖不同用户的请求，
再按标题覆盖率和固定种子打破平局。检查所有曝光，不能仅检查真实正例来构造有利上下文。
历史仅计严格早于请求的 Train 事件；同一时刻、未来和 Test 记录不参与。
冷启动、无交互和长列表分层保留，信息不足不能自动等同于错误行为。
缺失标题仍可能以 partial 进入预算；全部曝光均无标题等无法支持该场景的记录列入排除原因。
这是输入完整性代理和覆盖控制，不是自然语言、语义或行为真实性认证。

输出 `candidates.parquet` 保留整个 Train 的统计、分层、排名、选择状态和排除原因；
`selected.parquet` 保存本轮选中的 ID；manifest 保存数据版本、哈希、总量和每层覆盖。
筛选改变训练分布，尤其不能把分层后的购买率当成自然购买率；原请求权重不自动校正此偏移。
完整标签池未删改，可以独立用于直接动作训练。

场景构造与生成入口直接读取同一清单，避免重新随机抽样：

```bash
./agent_simulation/python.sh -m agent_simulation.datasets.customer_scenes \
  --dataset "$dataset" --selection result/customer-selection-NEW \
  --output result/customer-scenes-NEW

# 或在完整曝光协议下生成；尚未通过实际 token 预算检查的长请求仍可能超长。
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_curation \
  --dataset "$dataset" --selection result/customer-selection-NEW \
  --candidate-policies full --output result/customer-teacher-NEW \
  --split train --modes fixed --samples 1 --sft-only
```

`--selection` 消费整个已选清单，不再加 `--count` 截取，否则会破坏各层配额。
准备短列表时，历史不足等原有可构造性检查仍可能跳过 Response 子场景；Query 保持独立。
生成前输入预算检查以及生成后的行为/证据筛选仍然需要执行。
这些命令说明运行方式，不表示已启动模型生成或训练。

## 紧凑事实上下文

Customer 轨迹初始上下文共用 `encode_context`，覆盖场景长度检查、direct/EKB、
真实动作 SFT 和 curation 对照评估。相同来源的公共元数据集中到 `fact_sources`；
`facts` 的每行按 `fact_layout` 给出来源、字段、原值及可选的其他元数据。
原引用 ID 不变，事实值、历史归属、时间、未知价格及“非当前报价”语义全部保留，
没有通过删除商品、裁剪标签或改写历史来缩短输入。JSON 去掉非必要空白。
证据检查能展开新格式，也能读取旧轨迹；旧结果文件不重写。
这属于可逆的上下文表示优化，尚未声称等价模型效果或特定生成加速比。

2026-11-26 在完整 Lite Train 上执行了 CPU 选样审计：539,012 个 Customer 请求、
74 个非空分层；每层最多 100 个请求的示例选中 6,313 条，覆盖 5,995 位用户，
包含 2,097 条冷启动和 1,348 条 129+ 曝光请求。此批标签通过了全部基础完整性检查，
且曝光标题覆盖率均为 100%，所以该示例主要起分层与用户覆盖作用，不能宣传为已识别并剔除语义噪声。
完整标签位于 result/lite-full-training-20261126/sft；
候选审计与示例清单位于其同级 customer-scenes-100，均为不提交 Git 的本地数据产物。

使用 Qwen3.5-9B tokenizer 对三条实际请求验证了紧凑格式可完整还原原事实：
6/13/40 件曝光的上下文 token 分别由 5,234/9,884/19,240 降至 3,562/6,707/12,392，
减少约 32%–36%。这里只统计事实上下文，未包含系统提示，三个例子不代表全量平均。
没有运行教师生成或正式训练，也未验证模型行为效果。

## 1. 仅准备场景

```bash
dataset=result/lite-full-training-NEW/sft
./agent_simulation/python.sh -m agent_simulation.datasets.customer_scenes \
  --dataset "$dataset" --output result/customer-scenes \
  --split train --count 100 --min-items 2 --max-items 12
```

本阶段不调用生成模型。只有启用 content 路径才加载编码器；单独 random 或 collaborative 不需要 BGE。内容编码默认使用 GPU（`--bge-device cuda`）；可指定 `cuda:1` 或通过 `CUDA_VISIBLE_DEVICES=1` 选择物理 GPU 1。CUDA 不可用时明确报错，不静默退回 CPU。默认三种评分均启用；`--methods` 可显式选择。可通过 `--max-input-tokens` 和 `--tokenizer` 按实际 Query／Response 初始提示、schema、工具定义作长度过滤。输出 `scenes.jsonl`、`skipped.jsonl`、`manifest.json`，保留每种跳过原因及原始标签。场景文件使用 customer-single-positive-scenes-v2：每个源请求保存一份 Query 与若干 Response 子场景。旧 v1 多正例场景须重新准备，不能静默按新协议读取。Query 输入始终不含当前真实 Query 或候选列表。
Query 每个原请求只生成一组（每模型／每种信息模式各按 samples 采样），不随正例数重复。Response 候选不足、历史不足或超长仅跳过对应子场景；Query 独立检查输入长度并保留。count 和 request-id 均按原请求计数，manifest 分别记录原请求、Query 场景和 Response 子场景数量。

## 首轮建议：保留真实动作的 SFT-only

先使用完整曝光和固定信息，避免把短列表、工具选择与依据监督混在同一轮改动中：

```bash
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_curation \
  --dataset "$dataset" --output result/customer-sft-only-NEW \
  --split train --candidate-policies full --count 100 \
  --modes fixed --samples 2 --sft-only
```

`--sft-only` 只连接 strong 服务，不加载弱模型 tokenizer、不调用公共选项预测器、不构造 DPO；训练生成时不重复生成 Query，直接保存真实 Query。这个示例只描述运行方式，不表示已进行教师生成或正式训练。

新的生成入口总会保存 `observed-actions.jsonl`：每个入选 Train 原请求一条真实 Query 和一条完整曝光真实 Response，与教师是否成功无关。真实动作前缀重新由固定事实和直接动作协议构造，不拼接失败解释或工具轨迹。目标商品必须属于原曝光，非法标签明确失败。Test/evaluation-only 不导出训练样本。

`sft.jsonl` 合并真实动作与合格依据增强轨迹。增强轨迹仅保留两个动作集合都完全匹配标签、通过现有证据检查的 EKB；不会把部分匹配的教师动作改写成真实动作。直接动作和 Query 教师轨迹由真实动作替代，原生成记录仍留在 records.jsonl。未人工审阅的依据仍标记 not_reviewed，不称为语义已验证。funnel 单列 observed_action_rows，原有生成轨迹漏斗不把真实动作伪装成教师通过样本。

离线 customer_filtering 默认读取 records.jsonl 同目录的 observed-actions.jsonl，也可用 `--observed-actions` 明确指定。侧文件缺失时仍支持旧的纯轨迹导出，并在 sft_protocol 中区别标记；新实验应保留该文件及 manifest。离线 `--sft-only` 可跳过偏好配对。

## 2. 生成 EKB 和信息获取轨迹

```bash
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_curation \
  --dataset "$dataset" --scenes result/customer-scenes \
  --output result/customer-teacher --split train \
  --modes fixed --samples 10
```

默认仅使用 sampled（采样候选）场景和固定信息模式，不自动展开完整比较矩阵。`--count 0` 使用所有已准备场景；`--request-id` 可固定请求。`--modes fixed` 使用固定事实，`--modes autonomous` 自主选择只读工具，也可显式同时生成两种模式。强模型默认 Qwen3-32B，弱解释模型与默认公共行为预测器均改为 Qwen3.5-9B；弱模型服务和 tokenizer 通过 `--weak-url/model/tokenizer` 指定。公共行为预测器可使用独立的 `--predictor-url/model/tokenizer`，不必与弱解释生成器绑定。

Customer 生成入口中的 BGE 检索及 Query 相似度评分也默认使用 `--bge-device cuda`。设备参数会进入运行 manifest；这不代表当前已启动 GPU 任务。

EKB 直接保留作者详细解释的要求，不额外限制 100 字、单句或字段文本长度。作者示例编码器为 `gte-Qwen2-1.5B-instruct`，本项目当前配置的内容编码器是中文 BGE。

EKB 保留作者 Stimulus → Knowledge → Evaluation、因素和 Evaluation Style，追加项目自己的 evidence_refs。作者 vendor 原文件与函数不改动；证据标记、动作限制、联合概率、严格偏好条件与独立评估均为项目适配。

Query 正式入口始终直接输出查询 JSON，不生成 EKB，不含当前真实 Query、当前曝光和当前动作标签。每个请求的 Query 只生成一次（每模型／信息模式／采样序号），不随 Response 的 EKB 消融分支重复。

Response 的 sampled 单正例短列表最多选择一个不同商品，点击与购买字段仍分别保留：空动作、仅点击、仅购买、同一商品点击且购买。不同商品分别出现在点击和购买中也不合法。full/retrieved 协议保留多选，不能与 sampled 合并计算效果。

### EKB 证据标记

代码从实际可见历史和商品构造事实表，为每条事实生成稳定编号（H… 历史、P… 商品、Q… 当前查询）。每条事实由代码保存来源、字段、值、时间、用户与商品标识；点击和购买使用不同字段。提示中用事实表替换重复的历史／商品正文，不再重复发送一份 evidence_sources。

模型保留原始 Stimulus、Knowledge、Evaluation，只增加 evidence_refs：三个阶段各列引用编号；无依据时使用空列表，在原文中说明未知或暂定推断。模型不再生成字段路径、value_json、Observed claim 或重复抄写事实。

代码检查编号确实出现在实际 user/tool 消息中、历史与历史价格早于截止点、用户及商品标识一致、行为字段类型合法，以及原始最终 JSON 与保存的 EKB／动作一致。当前 Query 和静态商品事实可作为当前已可见信息，不声称它们属于过去交互。

machine_checked 只表示来源检查通过，不能证明自然语言推断成立。保留少量人工语义审阅：semantic-review.jsonl 按模型与原请求抽取最多 20 条 EKB，重点看点击误称购买、价格偏好推断、缺失事实和解释／动作冲突。审阅结论写入对应轨迹 semantic_review.verdict（pass/fail），离线重筛时 fail 禁止进入 SFT 或作为 chosen；未审阅始终标记 not_reviewed，不能报告成语义已验证。此过程不自动调用第二个 LLM。

旧路径／值式证据输出不会自动转成新编号，旧记录保留；重新生成才使用新协议。

## 3. 两类实验边界

A：独立行为预测，是当前 customer_curation 的协议。

- Query：过去历史 → 独立生成 Query → 与真实 Query 比较。
- Response：固定真实 Query、同一候选列表、相同可见历史 → direct 或 EKB → 预测点击／购买。
- direct 直接输出动作 JSON，不要求任何解释；EKB 输出原始解释与动作。
- 默认 fixed、EKB；显式选择 direct ekb 才展开消融。仅比较解释影响时使用 fixed，避免工具查询差异成为混杂因素。

以下仅是以后运行的示例，本轮没有执行：

~~~bash
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_curation \
  --dataset "$dataset" --scenes result/customer-scenes-test \
  --output result/customer-ekb-ablation --split test \
  --modes fixed --response-styles direct ekb --samples 1 --evaluation-only
~~~

evaluation-only 保存所有原始尝试、概率和报告，不执行训练数据筛选导出。报告按阶段、模型、信息模式、候选协议、EKB/direct、数据域、候选数与行为类型分组；提供点击／购买 F1、Recall、完整动作匹配、无点击正确率、Log Loss、证据检查数和有效数量。真实标签为空的头 Recall 为 null，单列无点击正确率；不以空集 Recall=1 稀释指标。所有失败尝试保留，不能只报告筛选后的高分样本。

B：端到端仿真是另一个协议：历史 → 预测 Query → 平台搜索返回列表 → Response → 结算。现有仿真运行器负责这条路径，不能拿新搜索列表的动作直接和旧曝光标签比较，也不能与 A 混报。本轮没有新增或运行 B 实验。

## 4. 概率、筛选与训练导出

~~~bash
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_filtering \
  --input result/customer-teacher/records.jsonl \
  --output result/customer-training-data
~~~

离线筛选不调用模型、不修改原始 EKB 或动作。正常生成入口也会导出一次；evaluation-only 禁用这一步。

### 概率协议

- sampled 单正例列表有 1+3N 个互斥合法动作：无动作，以及每件商品的仅点击／仅购买／两者都有。公共预测器对这些动作评分，保留完整概率、熵、argmax、采样动作；Log Loss 为真实完整动作的负 log2 概率，单位 bits/decision。
- full/retrieved 多选保留逐商品四状态边缘概率；Log Loss 为均值 bits/item，不是完整集合的联合概率。
- 不能把两种单位混合排序、配对或汇总。候选数量不同的场景也分别分层。
- Query 是自由文本，保留 BGE 相似度，不伪造选项熵。
- 这些概率来自同一个公共行为预测器，不能称为强弱生成器自身的置信度，也不保证已校准。

单正例概率最大需要 37 个选项（12 件商品），服务需允许对应数量的 logprobs；项目 serve.sh 设置 --max-logprobs 64。它不会自动启动服务。

### SFT

独立收集 strong 原始轨迹，不依赖偏好对或概率评分成功。以下阈值描述原轨迹筛选；新混合 SFT 的依据增强部分另外要求完整动作精确匹配，真实动作池独立保留。Query 相似度默认 ≥0.7；Response 点击与购买 F1 分别默认 ≥0.8，真实空头必须预测为空。EKB 还须通过证据检查；direct 分支没有解释证据门槛。单正例合法动作下，上述 Response 门槛实际要求匹配真实动作。完整曝光多选仍按两个集合判断。

### Response DPO

chosen 不按模型大小指定，强弱任一方都可成为 chosen。相同场景、输入、工具权限、候选、任务协议和概率单位才可配对。

1. chosen 原始点击与购买必须同时完整匹配真实标签，并通过证据检查。
2. rejected 行为不正确时，要求 chosen 平均 F1 优势至少 action-min-gap（默认 0.05），两头均不劣。
3. 两边行为都正确时，只有 rejected 存在可确定的证据引用错误，才可构造证据质量偏好；缺少旧标记仅为未验证，不能自动判定解释更差。
4. 默认 logloss 协议还要求 chosen 的 Log Loss 低至少 logloss-min-gap（默认 0.05，单位取决于任务协议）。仅更低 Log Loss 不足以配对。
5. 两边行为都错、chosen 行为正确但证据不通过、分数方向冲突、概率不足均不配对。
6. 原始解释、动作和各分支实际工具结果逐字保留，不把公共预测器采样动作拼接回 EKB。

sampled_match 是显式可选作者采样对照，也执行 chosen 原始动作完全正确、证据检查通过的门槛；它仍不使用 Log Loss 差距。主流程默认 logloss。Query DPO 独立按真实 Query 相似度分差选择，不套用 Response 的 EKB 质量判定。

作者 calculate_metrics/data_filtering 继续用于强弱解释条件下的不确定性统计和分层场景选择，新的严格配对条件位于项目适配层。数据域、候选数量、当前行为类型分层；统计同时保留原请求多点击类型。缺少有效偏好对是正常结果，不为凑数放宽正确性。

### 防泄漏、导出与统计

拆分前确定官方原请求 split。场景读取检查父请求、全部子列表与 manifest 的划分一致；导出再次检查同一数据版本下的 source_request_id 不得跨划分（不因数据域不同而放行）。训练读取只接受 train，不重新随机划分子列表。source_request_id、scene_id、原始标签及任务协议传到训练元数据。test 不进入训练导出。

- sft.jsonl / preferences.jsonl：接现有轨迹 SFT、DPO 训练器。
- filter-decisions.jsonl：每条行为与证据检查、保留状态、淘汰原因、概率失败原因。
- pair-decisions.jsonl：场景为何配对／未配对。
- filter-funnel.json：按阶段、模型、数据域、候选数量、当前／原始行为类型、候选协议及 EKB/direct 统计保留率；原因可能重叠。
- uncertainty-metrics.jsonl：作者强弱分歧指标，采样匹配率不等于生成动作正确率。

数据没有可用的真实曝光顺序，不使用位置特征、不把列表排列解释成真实展示排名。短列表效果、完整曝光多选效果、端到端仿真结果分别报告。旧结果保留，不修改为新协议结果。

默认模型仍是 Qwen3-32B、Qwen3.5-9B；BGE small zh v1.5 使用 CUDA。生成默认 temperature=0.7、seed=42、samples=10、max-new-tokens=2048、context-limit=16384，fixed 每次最多一次生成调用；自主模式显式启用，默认最多 6 次模型调用、8 次工具调用。2048 是输出 token 预算，不是 EKB 字数限制。

## 单正例拆分的解释边界

例如原请求点击 A、B，购买 B，生成两条 Response：A + 干扰项（排除 B），标签为点击 A、无购买；B + 干扰项（排除 A），标签为点击 B、购买 B。Query 只保留一组。无点击且无购买的请求生成一张真实曝光负例列表，两个动作集合均为空。

拆分能复用更多单交互训练场景，但这些子场景共享同一请求，不是独立用户。划分继承原请求 split；SFT 和 DPO 记录保留 source_request_id、scene_id、source_response_target，用于追溯和请求级分组。不能对拆分行重新随机划分训练／测试，也不能把子列表预测的并集直接当作原完整曝光条件下的多点击评估结果。训练器按原请求/阶段分配总权重，再分给子场景和轨迹，避免多正例与高通过率请求仅因行数多而放大。完整曝光真实动作也保留在同一原请求下。

候选仍是从真实曝光抽出的结果条件化短列表；干扰项只表示该次请求未点击且未购买的商品，不代表用户永久不喜欢。正例与无点击请求均只使用各自的真实曝光池。没有真实曝光顺序字段，不推断位置。内容编码保持 GPU 默认，原始数据及旧实验结果不改写。


## 曝光池与作者协议的区别

当前默认直接在曝光池内做三路采样，不按 Query 或类目硬筛；相关阈值 CLI 参数已移除。旧已生成场景不会原地修改，必须重新准备才能采用当前规则；manifest 和每个子场景 scope 的 query_category_filter=false 区分本次规则。仅实际用到的曝光商品编码，避免编码整个共享 catalog。

作者 MIND 的 preprocessing/DataProcessor_MIND.py 保留原始 impression_list，但只保留恰好一个点击、列表不超过 10、历史至少 5 的请求。scene_sampling.py 有条件复用 MIND 的真实曝光列表；该分支还存在 sampled_dataframe / sample_dataframe 变量不一致，不能说其所有默认路径均正确复用了曝光。其通用路径仍调用三路采样。我们不复制 MIND 的单点击请求剔除策略：多正例拆分、不点击和 Query 均保留。

作者 Amazon 预处理把评分／评论记录映射为单一 item_id 交互，模板称为购买历史，但没有独立的点击、购买双标签或转化漏斗。我们继续保留 clicked_item_ids / purchased_item_ids 两套真实标签，且不由点击推断购买。曝光未点击只是该请求下未点击，并不等于永久不喜欢；从完整曝光抽出的短列表仍是结果条件化训练构造，不等价于用户真的只看到了这几件商品。

## 多采样、请求级指标与性能

正式默认每模型每场景 samples=10，与作者示例 n=10 对应；sample-concurrency 默认 2。小规模格式／耗时试跑显式指定 samples=1 或 2，不能把这种试跑当成已完成多采样筛选。SFT 独立保留合格轨迹；DPO 还需同场景的合格 chosen 与有明确差距的 rejected。全部错误时不配对；只有一个正确分支时 SFT 可以保留，DPO 仍可能没有合格对。错误且证据可靠的分支可成为 rejected，正确但证据不可靠的分支不得成为 chosen。

request-metrics.json 按模型、协议、信息模式、解释模式、原请求和采样序号聚合，合并子场景动作并去重，记录 Click/Purchase Precision、Recall、F1 和各自 Exact Match。检查预期子场景数，缺失或失败标记 incomplete；不能丢弃失败请求后宣称无偏效果。每个采样序号独立报告，不从十条中挑最正确的一条计算预测准确率。

sampled 聚合标为 split_reconstructed；它仍受结果条件化候选影响。真正的原始曝光评估使用 candidate-policies full，与构造短列表重建结果分开。正式样本队列必须覆盖购买正例；本轮旧样例全部没有购买，不能证明购买预测能力，也没有擅自生成新的购买样例。

stream-timing 默认开启，模型调用日志保留 input_tokens、output_tokens、seconds、ttft_seconds、decode_tokens_per_second。TTFT 是客户端发出 HTTP 请求到首个非空 SSE 内容／工具增量的时间，包含排队与网络；decode rate 按服务端输出 token 数与首尾增量时间近似计算，不能当作精确逐 token 的内核速度。关闭流式时 TTFT 和 decode rate 为 null，不用总耗时冒充 TTFT。没有增加额外模型调用来做计时。

## 提示词与上下文审计

固定模式精简提示词移除工具规划包装、重复的 EKB 决策类型说明和重复输出说明，保留作者 default_system 全文、项目事实规则、证据解释与 JSON 适配。不新增字数上限。自主模式仍保留工具规划说明。

上下文另外提供 history_events（按时间排序，每个事件链接 Query、点击、购买事实）与 candidate_context（按商品关联属性）。历史价格标记 historical_observation 和 not_current_quote；未知当前报价显式保留，不以事实缺失暗示价格可推断。事实表仍保存来源和时间；事件索引只保存引用，不重复商品正文。

新增保守的价格陈述检查：仅对一句话中明确出现单个候选商品 ID、价格词和数字的情况，检查该商品价格是否未知、数值是否吻合，以及历史价格有没有时间限定。多商品、代词、隐含比较等无法确定时仍交给语义审阅。这不是完整自然语言事实验证，不能据此声称每句话都被证明。

prompt-only 对照读取上一轮实际 user 消息，保持原样上下文、模型、seed=42/43、temperature=0.7、max_new_tokens=2048、并发 2 与流式方式，仅替换系统提示的重复包装。本次上下文索引与价格标记不混入这组效率对照；事实检查新增失败原因也不代表生成行为本身变差。

## Independent EKB effect evaluation

For label-blind A/B/C comparisons and the matching direct multi-sample control, see
[Customer EKB controlled evaluation](customer-ekb-ablation.md). This evaluation-only
entry point uses a simplified EKB prompt and never exports SFT/DPO data.
