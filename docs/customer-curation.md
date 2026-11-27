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
再按标题覆盖率、是否有严格过去的已记录点击/购买证据、固定种子排序。过去行为仅作有无判断，不按次数奖励重度用户。检查所有曝光，不能仅检查真实正例来构造有利上下文。
历史仅计严格早于请求的 Train 事件；同一时刻、未来和 Test 记录不参与。
冷启动、无交互和长列表分层保留，信息不足不能自动等同于错误行为。
缺失标题仍可能以 partial 进入预算；全部曝光均无标题等无法支持该场景的记录列入排除原因。
这是输入完整性代理和覆盖控制，不是自然语言、语义或行为真实性认证。
过去行为证据的优先级只在同一分层内部生效，不会删除冷启动或当前无动作场景，
也不把过去没有点击/购买判为坏数据。该统计复用已有 Train 历史扫描，不新增模型调用。
后续生成仍须检查解释是否真的受到这些历史事实支持，不能把可用证据当成已证明的偏好。

输出 `candidates.parquet` 保留整个 Train 的统计、分层、排名、选择状态和排除原因；
`selected.parquet` 保存本轮选中的 ID；manifest 保存数据版本、哈希、总量和每层覆盖。
筛选改变训练分布，尤其不能把分层后的购买率当成自然购买率；原请求权重不自动校正此偏移。
完整标签池未删改，可以独立用于直接动作训练。

训练选样与 CoT 调用使用独立预算。原始动作监督无需逐条配上生成解释；
例如教师阶段可先限制为每层最多 2 个请求，而不是沿用上面的每层 100 个预算。
74 个非空层对应最多 148 个教师请求；这只是预算示例，不是质量或最佳规模结论。
场景构造与生成读取该教师清单，避免重新随机抽样：

```bash
./agent_simulation/python.sh -m agent_simulation.datasets.customer_selection \
  --dataset "$dataset" --output result/customer-teacher-selection-NEW --per-scene 2

./agent_simulation/python.sh -m agent_simulation.datasets.customer_scenes \
  --dataset "$dataset" --selection result/customer-teacher-selection-NEW \
  --output result/customer-scenes-NEW

# 或在完整曝光协议下生成；尚未通过实际 token 预算检查的长请求仍可能超长。
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_curation \
  --dataset "$dataset" --selection result/customer-teacher-selection-NEW \
  --candidate-policies full --output result/customer-teacher-NEW \
  --split train --modes fixed --samples 1 --sft-only
```

上述 148 是 full/fixed/samples=1 下 Response 教师调用的请求上限；
作者式 sampled 协议可能按真实正例展开多个子场景，应按场景 manifest 再计算调用预算。
教师筛选失败不删除对应真实动作监督，也不默认继续重采样直到命中。

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
随后仅增加同层的“有过往行为证据优先”排序，重新从完整 Train 选取同样的 6,313 条：
有过往点击/购买的请求由 3,280 增至 4,126；74 层、2,097 条冷启动、1,348 条长曝光配额不变。
用户覆盖由 5,995 变为 5,947，846 个请求被替换。新清单位于 customer-scenes-evidence-100，
旧清单保留。这说明选中输入的可用行为证据增加，尚不证明 CoT 或预测准确率提高。
小样本 SFT 的冻结队列不因这次排序调整而更换，以免混淆实验结果。

完整标签位于 result/lite-full-training-20261126/sft；
候选审计与示例清单位于其同级 customer-scenes-100，均为不提交 Git 的本地数据产物。

使用 Qwen3.5-9B tokenizer 对三条实际请求验证了紧凑格式可完整还原原事实：
6/13/40 件曝光的上下文 token 分别由 5,234/9,884/19,240 降至 3,562/6,707/12,392，
减少约 32%–36%。这里只统计事实上下文，未包含系统提示，三个例子不代表全量平均。
上述 CPU 选样与编码审计不包含教师生成或训练效果验证。后续预算受限验证见 trajectory-training.md。

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
  --split train --scenes result/customer-scenes --candidate-policies sampled \
  --modes fixed --samples 2 --sft-only
```

`--sft-only` 只连接 strong 服务，不加载弱模型 tokenizer、不调用公共选项预测器、不构造 DPO；训练生成时不重复生成 Query，直接保存真实 Query。这个示例只描述运行方式，不表示已进行教师生成或正式训练。

新的生成入口总会保存 `observed-actions.jsonl`，与教师是否成功无关。默认 sampled 协议使用已冻结的短列表：每个启用 Query 的原请求只保留一条真实 Query，每个子场景保留一条对应候选列表的真实 Response；原请求多正例分别拆分，空动作保留空标签。显式 full/retrieved 运行仍保留原请求的完整曝光监督。真实动作前缀由固定事实和直接动作协议构造，不拼接失败解释或工具轨迹。目标商品必须属于当前场景的可见候选，非法标签明确失败。Test/evaluation-only 不导出训练样本。

`sft.jsonl` 合并真实动作与合格依据增强轨迹。增强轨迹仅保留两个动作集合都完全匹配标签、通过现有证据检查的 EKB；不会把部分匹配的教师动作改写成真实动作。直接动作和 Query 教师轨迹由真实动作替代，原生成记录仍留在 records.jsonl。未人工审阅的依据仍标记 not_reviewed，不称为语义已验证。funnel 单列 observed_action_rows，原有生成轨迹漏斗不把真实动作伪装成教师通过样本。

离线 customer_filtering 默认读取 records.jsonl 同目录的 observed-actions.jsonl，也可用 `--observed-actions` 明确指定。sampled 合并按原请求、阶段和子场景匹配，并要求候选顺序及目标动作一致；旧的“sampled 解释＋full 真实动作”侧文件需要重新生成，不能直接混合。侧文件缺失时仍支持旧的纯轨迹导出，并在 sft_protocol 中区别标记；新实验应保留该文件及 manifest。离线 `--sft-only` 可跳过偏好配对。

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

## 轻量 SFT 与跨场景并发（2026-11-27）

正式入口现支持显式 `--sft-only --cot-preset lightweight`。未指定预设时仍沿用 original 研究配置；不会把 DPO 的多采样实验静默改成单采样。

| 设置 | original | lightweight |
| --- | --- | --- |
| 每模型每场景 samples | 10 | 1 |
| temperature | 0.7 | 0 |
| max-new-tokens | 2048 | 768 |
| scene-concurrency | 1 | 4 |
| EKB 提示 | 原始 EKB | 复用已审计的 concise EKB |
| 模式与模型 | 原有 fixed/autonomous、强弱模型协议 | fixed、仅 strong，须显式 sft-only |

上述数值可被显式参数覆盖；重复采样仍要求非零 temperature。轻量预设保留相同 EKB schema、事实上下文、证据检查和原始生成动作，不在失败后改写答案。Train Query 仍直接用原始监督，不调用 teacher。短解释不等于已通过语义审阅。

`--scene-concurrency` 控制同时处理的场景分支数量，`--sample-concurrency` 控制每分支内样本并发。最多同时执行前者 × 后者个采样工作流；轻量默认每分支只有一个样本，因此最多 4 个。固定模式每工作流一次生成调用。调度按输入顺序落盘、保留独立样本 seed，惰性拉取有限任务，不为全量场景一次建立全部 asyncio tasks。错误会取消并回收待处理任务；已提交的 HTTP 请求不能保证服务端立即停止。新逻辑没有启动或扩容模型服务。

仅供后续明确启动时使用，本次未执行生成：

~~~bash
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_curation \
  --dataset "$dataset" --scenes "$scenes" --output "$new_output" \
  --sft-only --cot-preset lightweight --scene-concurrency 4 \
  --teacher-budget "$teacher_budget"
~~~

`--teacher-budget` 是可选的 **Response 子场景总预算**，仅用于 lightweight / Train / sampled / fixed / EKB 的训练构造，不用于 Test 或 evaluation-only。省略它仍处理输入中的全部场景；显式设为 0 只导出原始监督，不建立 teacher 客户端。若覆盖 samples，生成次数上限相应为预算 × samples。

预算分配复用既有已准备场景，按类目域与观察到的行为类型分组，在各组内依次优先：Query 与标题完整、没有缺少对应 ID 历史支持的可见属性冲突、有相同商品 ID 历史、有规范化后相同 Query 历史。其余同分使用 seed 和 scene ID 稳定打散，再轮流给各组分配预算。标签只用于行为覆盖分组，不进入质量优先级或模型输入；预算小于分组数时不能保证覆盖每一组。它是保守的输入证据启发式，不是质量分类器或 PRM，不保证 teacher 一定正确。

- `teacher-scene-evidence.jsonl`：逐场景输入证据与属性冲突标记。
- `teacher-budget.json`：考虑数量、选择数量、预算、被选 scene IDs。
- `observed-actions.jsonl` 与 SFT 导出：输入范围内所有有效原始 Query / Response 都保留，未获 teacher 预算不等于删除行为数据。
- 部分子场景未生成时，已有请求级指标会标记不完整；预算筛选后的生成诊断不能当作全体请求的无偏准确率。

预算扫描不调用 LLM，但仍需遍历已准备场景。现有 prepared scene 加载、原始监督与导出集合仍驻留内存；本次没有重写全量数据存储，也没有用小型离线测试宣称全量内存或吞吐已验证。全集 Train 可以作为场景来源，teacher 预算与原始数据覆盖范围分别控制。

### 不同 ID、相同可见属性的来源核查

核对原始 `data/kuaisearch/lite/source/raw/items_lite.jsonl`：第 4,335,912 行的 34262 与第 5,593,803 行的 89935，除 item_id 外所有字段完全相同，包括酸梅条标题、品牌、同一店铺（40751）、各级类目 ID/名称。catalog 保留了这两条独立记录，并非提示压缩或标题拼接造成。同一原请求缓存（44867）中，两者 price_raw 也都是 490；这是价格观测，不能当成已知当前报价或直接推断货币单位。

源商品表不提供能解释区别的图片、规格等字段。因此不能断定是不同 SKU、重复上架或其他平台内部区分，也不能将两个 ID 自动合并。新预算逻辑对标题、品牌、三级类目、价格完全一致的候选组做标记；仅同标题不算属性完全相同。有对应 ID 历史只表示额外证据可用，不证明用户会复购或一定点击。真实动作仍按原 ID 保留。

## 同一 Response 内候选去重（same-shop-content-v1）

新构造的 sampled 场景默认在三路采样前，对该请求的真实曝光池去重。比较完整的标题、店铺 ID/名称、品牌 ID/名称、各级类目 ID/名称，以及最终提供给模型的历史价格值和已知/未知状态。静态比较字段缺失时不合并；不同店铺、不同价格或不同价格状态都保留。

价格在每个原请求上统一查询一次，仅使用严格早于 cutoff 的有效观测；各子场景复用同一结果。相同历史价格但观测时间不同，不据此认定商品内容不同；保留的商品仍携带它自己的价格时间，不移接另一个 ID 的证据。原请求的当前价格不参与该去重判断。

- 本子场景的真实正例优先保留。其他真实正例仍各自建立子场景，原始 source_response_target、父请求及 Query 不改写。
- 其余相同内容按 seed、原请求 ID 和商品 ID 的稳定散列选择代表，再在剩余曝光池内执行原来的随机／内容／协同采样，避免重复项占满候选后才删除。
- 同组任一 ID 有该用户严格在先的点击／购买证据，整组免于内容合并。检查输入历史、该用户的历史交互池和在先背景快照中的近期交互 ID；其他用户或未来交互不提供保护。既有采样器对已交互干扰项的过滤规则保持不变，“免于合并”不代表强制入选。
- 去重造成候选不足时保留可用短列表，标记 short_pool；即使只剩真实正例或一个无动作候选，也不会为凑数量加回重复项。其他既有构造失败规则继续适用。
- 每个子场景的 candidate.scope.deduplication 记录策略版本、removed_to_retained 和 history_protected_groups；这些离线审计字段不送入模型。目录 manifest 的 candidate_deduplication 记录版本，读取时检查子场景一致性。
- 老场景仍可读取，不自动重写；没有冻结价格的旧 builder 调用不启用内容合并。完整曝光评估与原始商品表不去重，新场景构造与旧构造的指标应分别报告。

离线核对原有 44867:positive:34262 案例：两条历史价格均为 490，时间分别为 467064 与 424670，该用户历史中没有两个 ID 的交互证据；新规则保留真实正例 34262，移除相同内容的干扰项 89935。没有生成新 teacher 数据或运行训练。

## 历史 token 预算、等价提示压缩与 Seller 证据（2026-11-27）

轻量入口新增 `--history-token-budget`，默认 2048；original 预设默认不启用，显式设为 0 则不展示历史事件。预算按 strong tokenizer 计算“带历史的上下文 JSON”相对“同一基础上下文、不带历史事件”的 token 增量，包含事实表与来源索引的开销；不是整段输入的长度上限。总输入与输出预算仍由原有 context-limit 检查。

在默认最多 10 条事件的范围内，保留近期行为，并为较早的同商品记录和规范化后相同 Query 的记录提供候选位置；同 Query 的真实无动作记录也参与。预算不足时按优先级选择完整事件，超长事件整条跳过，不截断标题、事实、动作或证据引用。预算模式对同商品和同 Query 两路分别召回，避免两个同商品记录占完名额后完全丢掉同 Query 无动作证据。极小预算不能保证各类证据都保留；上下文报告实际 selected_history_events。Query 阶段仍不读取当前 Query 或 Response 标签。

同一配置用于 teacher、direct 对照及 observed-action 输入，事实保持对齐。teacher-budget 的证据优先级也以预算后的实际历史为依据，避免给最终不会展示的历史证据加分。预算改变可见历史，不改变原始行为标签；来源时间、所属用户和输入不可变检查继续生效。

动作 schema 仅在系统提示中用等价的 enum 列出原来的 1+3N 个联合动作，省掉每个选项的 anyOf/const 包装。实际解码和校验继续使用原有 schema，跨商品“点击 A、购买 B”等非法动作仍不允许。本地 LMFE 不需要增加新解析能力；正式 vLLM/XGrammar 的实际解码 schema 已做 CPU 离线转换检查。

历史 clicked_items / purchased_items 中的多商品属性复用项目现有 $table 格式：共享字段只写一次，逐商品差异按列保存。只有包括格式说明后的总 JSON 字符数仍减少至少 12 时才启用；history_item_tables 提供格式说明。展开后与原事实严格相等，包含商品 ID、顺序、重复交互、null、每次历史价格、来源和时间；事实 ID 不变，旧原始／紧凑记录仍可读取。冲突列、错行宽、缺失格式标记等会被拒绝。JSON 字符节省不是每个 tokenizer 上都保证等比例的 token 节省。

Seller 的共享行为特征补充最近有效报价的时间点和最近两次有效报价变化的前后观测，稳定报价尾段较长时仍能看到较早的变化；跳过未知报价的区间继续标为不可推断。正式冻结 Seller 上下文同时保留已加载商品的市场窗口、类目和 recent 统计，省略重复分析文本与竞争品检索正文，不额外查询市场。市场窗口必须不晚于决策 cutoff。该快照对应当前决策窗口，不冒充历史变价当时的市场快照；报价差异与市场统计不能证明变价因果效果。

### 离线 token 核对

使用本地 Qwen3-32B tokenizer，从既有 customer-sampled-cohorts 场景中按无动作、仅点击、仅购买、点击且购买各取 4 个不同原请求的已有子场景，共 16 条。基线为 31066af；没有重采候选、调用模型或训练。

| 测量块 | 原值总 token | 优化后总 token | 减少 |
| --- | ---: | ---: | ---: |
| 上下文 JSON，仅无损属性表压缩 | 59,620 | 56,958 | 4.5% |
| 上下文 JSON，再启用 2048 历史预算 | 59,620 | 52,882 | 11.3% |
| 动作 schema 的提示表达 | 6,069 | 5,224 | 13.9% |

16 条的历史 token 增量均不超过 2048，保留历史事件由合计 125 条变为 114 条；少数本来已满足预算的场景会因预算说明增加几个 token。这是分层小样本的输入长度核对，不是全量吞吐或模型效果评估。

可复核 scene IDs：90476:no_action、3568:no_action、172349:no_action、275831:no_action；
249491:positive:202712、16209:positive:561989、396277:positive:211776、139495:positive:1390614；
125619:positive:227325、345637:positive:121021、44867:positive:34262、333208:positive:1457724；
144433:positive:1360759、197762:positive:3810606、519335:positive:8149、518977:positive:6469219。

## 新场景构造的真实小样本配对检查（2026-11-27）

在代码 3757030 上，从已有完整官方 Train 的 fit 分区缓存请求中，按无动作、单点击无购买、多点击无购买、单点击有购买、多点击有购买、无点击有购买各取 4 个请求；每个请求最多 3 个不同正例，共 24 个真实请求。包含原先重复案例 44867。此选择用于覆盖检查，不代表全集分布；不使用 dev/audit 调参。

使用正式 prepare_scenes 和 random/content/collaborative 三条路线，seed=20261126、min_items=2、max_items=12、min_history=3。对照仅关闭内容去重，仍冻结相同的全局严格历史价格；两侧复用相同 CF 快照和 BGE 编码缓存。BGE 在 CPU 上运行，只加载 Qwen3-32B tokenizer，不调用 LLM、不生成 CoT、不训练。24 条请求的用户、cutoff、点击与购买标签均与当前 catalog 核对一致，源缓存哈希也与原 manifest 相符；输入未被修改。

| 指标 | 去重前 | 去重后 |
| --- | ---: | ---: |
| 成功 Response 子场景 | 32 | 32 |
| 可采样池总候选数（按成功子场景累计） | 1,253 | 1,247 |
| 最终短列表总候选数 | 198 | 202 |
| 原有规则跳过的 Response 子场景 | 2 | 2 |
| 无动作请求遗漏 | 0 | 0 |

3 条子场景发生合并，共移除 6 个池内候选；1 组因真实用户历史而受保护。候选数是按子场景累计的，不是全局去重商品数。32 条成功场景的点击／购买投影、Query 输入、曝光来源、历史时间与用户归属、历史价格时间、上下文编码往返均通过检查，去重没有新增行为丢失。

| 案例 | 可采样池 | 最终列表 | 说明 |
| --- | --- | --- | --- |
| 68573:no_action | 138 → 137 | 10 → 10 | 移除 2730402、保留 2730371；2698909/2730361 因此前点击 2698909 而不合并 |
| 352338:positive:1684561 | 16 → 15 | 5 → 9 | 移除 1056423，保留真实正例 1684561 |
| 44867:positive:34262 | 159 → 155 | 4 → 4 | 移除 34265/89935 等重复项，保留真实点击与购买 34262 |

44867 的旧短列表为 [40925,17443,34262,89935]，新列表为 [42846,17443,34262,555638]。去重在采样前发生，其他曝光可以补足短列表。352338 的长度增加另有明确原因：merge_ranking_list 的 shuffle 与随后 target_size 共用 Python 随机流；候选池变小时 shuffle 消耗变化，因此同一个 seed 的长度也会变化。此次保留现状并记录差异，没有把最终数量变化直接归因为去重压缩。历史保护只免于合并，不免除原有历史商品过滤与后续采样。

**发现原有监督遗漏：** 请求 392266 的两件曝光商品 138544、430121 均被购买。拆为单正例后排除另一正例，便没有干扰项；新旧代码都因 insufficient_distractors 跳过两条 Response。实际 observed_action_rows 仅剩 Query 行。源数据没有被删，但当前 sampled SFT 路径会漏掉这两条购买监督，不能宣称全部真实行为已覆盖。建议下一步将非空单商品场景保留为原始行为监督，标记 short_pool，并降低其 CoT 优先级；不从未曝光商品补干扰项。

历史检查在相同的新候选下，对照旧的事件数策略与 2048-token 历史预算，避免混入候选变化：

| 指标（32 条子场景累计） | 事件数策略 | token 预算 |
| --- | ---: | ---: |
| 历史事件 | 270 | 207 |
| 同商品事件 | 6 | 6 |
| 精确同 Query 事件 | 30 | 29 |
| 同 Query 无动作事件 | 10 | 10 |
| 上下文 JSON tokens | 140,812 | 120,196 |

32 条均保留最近事件，历史贡献增量为 1,074–2,035 tokens，均不超过 2048；轻量 EKB 初始消息含工作流预算提示及 chat template 为 3,597–6,281 tokens。一个场景的同 Query 事件由 4 条减至 3 条，其同商品证据仍保留。没有截断单个历史事件。这里的相关性统计只计同 ID／精确同 Query，不能当作语义相关性评估。

还剩 3 组“提示可见属性相同且无同 ID 历史支持”的候选：两组实际属于不同店铺，去重正确保留，但当前商品事实没有展示店铺；另一组同店铺但三级类目为 UNKNOWN，被完整字段规则保守保留。建议仅在候选冲突时补充店铺与有效上级类目事实，避免给所有商品扩充提示，也不要把没有三级类目直接等同于低质量。

后续优先顺序：先补齐无干扰项时的原始行为监督；再为冲突候选补最小可见差异、将目标列表长度随机流与排名合并随机流分离。历史方面可在相同预算内增加轻量的类目／标题相关性：本样本 19/32 场景没有同 ID 或精确同 Query 历史，并不意味着它们没有语义相关历史。场景预过滤的长度计算也应与正式轻量入口共享预算配置，防止按旧长上下文提前误筛。

本次只验证和记录，未修改采样策略或训练入口。服务器项目内的复核脚本、请求、配对场景、逐场景检查与汇总保存在 tmp/response-dedup-real-audit-20261127/；最终一次 CPU 配对构造与上下文复核耗时 18.74 秒（复用已有请求缓存，不代表全集构造耗时）。所选请求 SHA256：caff91ca82555ed80dbeab2dd766122fcc8c16512228ef3519b4c1fd8d2e7127。

## 修复短列表监督与轻量上下文复核（2026-11-27）

在 80d51e2 的真实小样本审计之后，仅改现有采样、证据和历史选择路径：

- 候选长度范围改为目标长度：只要最终列表非空，就保留真实场景并用 short_pool 标记不足。没有干扰项的正例保留为单商品场景；单商品无动作也可保留。原有历史商品过滤后完全空的无动作列表仍跳过，不虚构候选。单候选场景在启用 teacher-budget 时，于同域／同行为分层内降低优先级；原始 observed-action 监督不受 teacher 预算影响。min_history 等其他资格约束不变。
- 仅当原可见字段相同、且已知店铺字段能提供差异时，补充 seller_id／seller_name 的 P 类事实及来源引用。相同店铺或不冲突的候选不增加冗余字段，未知店铺不补造。teacher 证据优先级按实际可见店铺信息判断冲突。
- target_size 使用由运行 seed 和 scene_id 派生的独立 Random 实例，并记录 target_size_policy=scene-seeded-v1；去重、池大小和采样路线不再消耗其随机流。实际列表仍可因曝光不足短于目标值。新规则会改变部分旧场景的目标长度，旧文件不重写，旧场景仍可读取。
- 仅在启用历史 token 预算的 Response 路径加入轻量相关性：规范化文本的字符二元组重合（至少两个共有二元组、Jaccard ≥ 0.3，忽略纯数字二元组），以及已知二／三级类目匹配；不运行额外模型。较早弱相关事件最多额外召回两条，近期、同 ID、精确同 Query 优先级仍更高；最多 10 条完整事件及 2048-token 默认历史贡献预算不变。Query 不读取当前 Response 信息。该规则是检索提示，不是已验证的偏好证据或语义判断器。

再次使用完全相同的 24 个真实 fit 请求、seed=20261126 和三条正式采样路线（CPU BGE、全局严格历史价格／CF）。结果：

| 检查 | 上轮 | 本轮 |
| --- | ---: | ---: |
| 成功 Response 子场景 | 32 | 34 |
| 跳过的 Response 子场景 | 2 | 0 |
| 本轮开关去重的目标长度不一致 | 未独立固定 | 0/34 |

请求 392266 恢复两条单商品 Response：[138544] 与 [430121]，均保留仅购买、未点击标签。通过 observed_action_rows → combine_observed_sft 的离线检查，产生 24 条 Query 和 34 条 Response 原始监督，共 58 条，没有调用 teacher。所有源标签按子场景聚合后完整恢复，没有添加未曝光商品或改变 Query 输入。

本轮内部配对仅开关去重，两侧均使用修复后的长度与短列表规则：最终候选总数均为 194；3 条子场景池内共移除 6 个重复项，1 组因历史差异受保护。44867 的目标长度两侧均为 11，34262 的真实点击／购买保留，34265 和 89935 被去重。此处不能把 194 与旧轮 202 的差值单独解释为去重收益，随机长度策略也已改变。

店铺效果另用上轮固定候选回放，排除重采样混淆：68573 的 2698921/2698941，以及 71516 的 2779664/2779678，均新增各自真实店铺事实，不再被标为可见属性完全相同。278746 的 1575/25659 同店铺且三级类目未知，仍保守保留冲突标记，没有宣称可以区分。

历史比较使用相同的新候选、相同 2048-token 预算，对照 80d51e2 选择规则与新规则：

| 指标（34 条子场景累计） | 原规则 | 新规则 |
| --- | ---: | ---: |
| 历史事件 | 225 | 222 |
| 同商品事件 | 6 | 6 |
| 精确同 Query 事件 | 33 | 33 |
| 同 Query 无动作事件 | 14 | 14 |
| 非同 ID／精确同 Query、但匹配文本或类目的事件 | 25 | 30 |
| 上下文 JSON tokens | 124,645 | 124,482 |

34 条均保留最近事件，历史贡献最多 2,039 tokens。例：当前“钓鱼中药酒”补入“中药酒鲤鱼”“支架钓鱼支架”，替换较无关的洁面与烧水壶事件；“花老虎鱼”补入观赏鱼历史。仍按完整事件保留，类目／文本匹配可能较宽，不能由此声称模型准确率提升。历史时间、用户归属、价格时间、编码还原和输入不变检查均通过。

128 项相关回归测试通过，覆盖单候选行为监督与导出、无动作、空候选保护、三条采样路线、长度随机流隔离、店铺事实、Query 隔离、预算和正式生成入口兼容性。此次没有启动 LLM 生成或训练，也未修改旧数据。脚本与新审计产物位于项目内 tmp/run_scene_refinement_audit.py 和 tmp/response-scene-refinement-real-20261127/，原审计保留；样本输入 SHA256 与上轮一致。

## 正例位置、长度预过滤与共享上下文整理（2026-11-27）

在 9d45e94 上修复两个已复现的问题，并做局部整理，没有改 vendor 或重写旧场景：

- 正例插入在项目适配层使用独立、按场景派生的随机流，取值包含末尾插入位置，空干扰项列表仍可插入唯一正例。记录 target_position_policy=scene-seeded-uniform-v1；目标长度策略不变。
- 场景构造入口新增 --cot-preset、--modes、--history-token-budget，长度预过滤与正式生成入口共用 resolve_context_options。轻量预过滤默认 fixed、简化 EKB 提示、2048-token 历史预算；original 保留原来的默认行为。使用轻量 EKB 生成时，应在预过滤端也明确指定同一 preset、tokenizer 和历史预算，例如 --cot-preset lightweight --max-input-tokens 6656。这里 6656 是复核阈值，不是修改项目全局上下文上限。
- 将 CurationTools、历史选择／预算及固定请求构造移至 evaluation/customer_context.py，旧 customer_curation 导入路径继续兼容。统一 Query 规范化；历史相关性返回具名结果，历史召回集中到 select_history。生成 manifest 补充共享模块的源码哈希。
- 父适配器提供原始基础上下文，子适配器在最终边界统一编码。预算试选复用已构造的商品／Query 事实，避免反复进行商品冲突检测；仍以完整序列化结果精确计数，不用近似 token 相加替代。
- 采样器初始化时建立按用户、时间排序的交互索引，每个子场景用二分查询严格早于 cutoff 的该用户历史，避免重复遍历全局交互；全局 CF 数据与快照规则保持不变。去掉商品回填中的恒真成员判断，以及单正例模式下恒为假的 over_max_to_preserve_positives 输出字段。

复用前两轮相同的 24 个真实 fit 请求，三条采样路线均启用，CPU BGE，无 LLM 调用。逐场景与 9d45e94 对比，34 条子场景的候选集合、目标长度、去重审计、历史交互数量及行为标签一致，仅正例位置按新规则变化。34 条 Response 与 24 条 Query 仍可形成 58 条原始行为监督。

| 检查 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 非单候选正例场景 | 28 | 28 |
| 正例位于末尾 | 0 | 7 |
| 固定模式预过滤跳过的 Response（阈值 6656） | 18 | 2 |
| 相对实际轻量 EKB 提示的误筛 | 16 | 0 |

三条两候选正例场景的原正例位置均为 0，新位置分别为 0、1、1。另以固定长度 2/3/5、64 个确定种子做测试，所有合法插入位置均可到达；小样本位置数量不是总体均匀性的统计证明。

两条仍跳过的场景确实超出复核阈值：396277:positive:211776 为 6909 tokens，44867:positive:34262 为 6786 tokens。新预过滤每条保留场景的计数与实际轻量 EKB 初始消息（含工作流提示和 chat template）完全一致，故在此阈值下保留 32 条 Response，而非放宽阈值掩盖超限。

共享上下文在相同候选下，分别比较 Query／Response、原 fixed／预算 fixed／autonomous，共 204 组，新旧内容完全相同；编码往返检查也通过。141 项相关测试通过，新增覆盖插入末位、严格历史边界及不扫描全局交互、长度精确边界、无效配置、CLI 配置透传和共享模块版本记录。

项目内复核脚本为 tmp/run_scene_review_fixes.py，结果为 tmp/scene-review-fixes-real-20261127/；保留此前审计。未启动 CoT 生成或训练。旧场景的位置与旧过滤结果不会自动改变，后续构造需使用新入口及与生成端一致的配置。

## 并发补位、跨阶段上下文复用与证据检查（2026-11-27）

在 7214a57 上做局部性能修复，不迁移 FrozenSampleTools、不引入 SceneJob，也不改采样或质量门槛。

- bounded_map 按已完成任务补位，慢的首任务不再挡住其他任务。仍按需消费输入、限制未交付任务数量，并在失败或关闭时取消和回收剩余任务。不同场景按完成顺序写入 records.jsonl，比较结果应按场景／轨迹 ID 对齐；同一场景的样本交付后按原 variant／tid 顺序整理，保留去重与并列选择规则。
- 正式入口为原始行为监督、teacher budget 选择、教师初始提示共享一个运行内 ContextCache，最多 128 个条目，按最近使用淘汰。键包含完整输入快照指纹、阶段、信息模式、历史条数／token 预算及 tokenizer 实例身份；只复用 fixed 输入上下文，不缓存标签或生成结果。缓存读写隔离副本，保留冻结检查；不同输入必须重新构造并执行原时间、用户归属、事实校验。大规模 budget 扫描中，早期入选场景可能已被淘汰，此时安全重算，不保证全部三阶段都命中。
- 同一轮同步导出复用已计算的 Response 证据检查结果，SFT 筛选和偏好筛选不再重复检查同一轨迹。独立调用或再次导出会重新检查，绝不信任文件里已有的 evidence_quality。生成边界与离线导出边界仍分别校验；split、schema、行为质量、上下文、时间／归属及原始输出一致性检查保留。

离线对照复用冻结的 24 个真实 fit 请求、34 个子场景，Qwen3-32B 本地 tokenizer、2 个 CPU 线程。同一进程预热一轮后交替运行新旧实现三轮，每轮使用新缓存，计时包括原始监督构造、budget 扫描与选择、教师初始消息准备。未调用模型，未启动 CoT 生成或训练。

| 指标 | 7214a57 | 本次 |
| --- | ---: | ---: |
| 三阶段准备耗时中位数 | 5.703 秒 | 2.674 秒 |
| 历史 token 预算构造次数 | 126 | 58 |
| 原始行为监督 | 58 条 | 58 条 |
| 入选子场景 | 34 条 | 34 条 |

该样本的准备耗时减少约 53%，不是端到端生成或训练加速结论。逐项比较 58 条监督（含标签）、budget 审计及选中 ID、34 组 schema／提示词、完整上下文与事实 ID，以及原始监督导出文件，全部一致，输入未被修改。

并发另用 8 个真实场景 ID 配合可控延迟做调度模拟：并发 2，首任务 120ms、其他任务 10ms。第三个任务启动时间约从 120ms 提前到 10ms，总耗时中位数约从 151ms 降至 121ms，结果按 ID 对齐一致。该模拟仅证明补位行为，不是 LLM 吞吐实测。

149 项相关测试通过，包括慢首任务补位、背压与取消、正式入口三阶段缓存复用、缓存淘汰／配置隔离／防修改、时间和归属保护、证据检查调用次数及修改后重新校验。审计脚本、基线副本、计时与结果在项目内 tmp/concurrency-context-audit-20261127/；真实场景输入 SHA256 为 35134a80be17b8df85e2a3dc81cfd78655053a5e5c57ec3fc6b8e2a0df2d14de。旧数据和既有运行产物未重写。

## 明确的 CoT SFT 链路与配对对照（2026-11-27）

历史最后的 customer-context-sft-v1 是 596 条原始行为、0 条解释的 SFT，不能用于证明 CoT 收益。新版通过明确的生成模式、离线审查导出和实际 token 检查，将 CoT 训练目标与纯行为 SFT 区分开。以下命令是后续使用说明，本次未执行真实模型生成或训练。

正式生成入口使用 --cot-sft --sft-only --cot-preset lightweight。必须是 Train、fixed、EKB；teacher budget 不得为 0。输入场景应是冻结的 fit 分区，按父请求与 dev/audit 隔离；不能因为都属于官方 Train 就混入评估样本。例：

~~~bash
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_curation \
  --dataset <dataset-release> --scenes <frozen-fit-scenes> \
  --cot-preset lightweight --sft-only --cot-sft --teacher-budget <scene-budget> \
  --output <new-generation-run>
~~~

该模式保存 records.jsonl、observed-actions.jsonl、cot-filter-decisions.jsonl、cot-report.json 和 cot-review.jsonl，不自动写 sft.jsonl。状态明确为 awaiting_semantic_review 或 no_eligible_rationales。CoT 审查队列包括全部机械合格候选，不受旧 semantic-review.jsonl 的 20 条预览限制。轻量提示补充中文解释、保留原始商品/查询用词和 ID、不同 ID 不可混同；效果仍须后续实测，不能由代码变更宣称通过率提高。

机械筛选保留既有 split、用户、标签和 schema 校验，并要求单次固定输入原始输出、与原始监督相同的候选/标签、除系统输出指令外相同的输入消息、真实行为完全匹配、非空证据引用和重新计算的证据检查。不会修改预测动作来凑标签，也不信任保存的 evidence_quality。语义审查在队列副本中填写 verdict=pass/fail、reason 和 reviewer，保留 trajectory_id、messages_sha256；没有明确通过、哈希不符或失败的记录不进入 CoT 数据。

离线导出不调用模型，不再依赖整个 cohort 文件哈希相同：未改变的原始消息和同一场景可在新观察集合中重新审核复用，新增其他场景不会自动废弃它。真实上下文/候选变化则拒绝，不能将旧答案移植到新提示。

~~~bash
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_cot \
  --run <generation-run> --reviews <reviewed-cot-review.jsonl> \
  --output <new-comparison-release> --min-rationale-scenes 32
~~~

--min-rationale-scenes 必须显式指定正整数；32 只是示例数量门槛，并非统计充分性保证。还需根据 report.json 的行为分层、解释覆盖率决定是否扩量，不能用全部无动作解释代表正例。数量不足时保存失败报告，不产生任何训练 JSONL。成功后得到：

- action_only.jsonl：全部原始动作，每个场景一条。已选教师样本的动作 JSON 写法与教师原文中的 action 子串一致，避免空格、键顺序等序列化差异混入对照；标签不变，保留来源 ID。
- cot.jsonl：完全相同的场景顺序、候选、输入信息和标签；每个通过审查的 Response 最多用一条原始教师解释/动作替换对应行，Query 与未选 Response 保持原样，不叠加额外样本。
- manifest.json：v2 协议，两臂文件哈希、源 records/observed/reviews/生成 manifest 哈希、comparison_id、解释门槛、行为分层和对照协议。教师原文及审查哈希保持不变；action 在 rationale 之前的输出按 action_precedes_rationale 排除，不计作先推理后决策监督。

每条训练记录携带 sft_objective、comparison_id 和解释门槛。加载训练数据会按实际编码后的 rationale loss mask 检查非零解释及不同场景数，--limit 截掉解释也会失败；混合两臂、混合对照身份、无 CoT 对照中混入解释均被拒绝。CoT 训练同时拒绝 rationale_weight=0。旧的无标记文件可显式使用训练或编码入口的 --require-rationales。训练配置记录实际 rationale_scenes/rationale_tokens，而非只记录声明的权重。

可先只加载 tokenizer 做预检查，不加载模型：

~~~bash
./agent_simulation/python.sh -m agent_simulation.datasets.trajectory \
  --mode sft --input <comparison-release>/cot.jsonl \
  --tokenizer models/Qwen3.5-9B --max-length 8192 --require-rationales
~~~

action_only.jsonl 使用同一编码入口、同一长度阈值，去掉 --require-rationales。若一臂超长，应回到同一场景集合共同处理，不能分别删样本或截断。后续训练固定相同底座/初始化、种子、epoch、batch、梯度累积、学习率、动作及父请求权重；只改变解释监督与相应输出指令。两臂样本数和优化步数匹配，CoT 的 token/耗时会更多，需单独报告。评估使用相同且未参与训练或筛选的 dev/audit 场景，分开两种口径：

- direct（默认）：两组都直接输出动作，用于隔离训练监督的影响；保留 Query 与 Response 的既有指标。
- generated_cot：仅比较 Response，action-only 模型直接输出动作，CoT 模型自行生成短解释后输出动作。两组共享用户上下文，解释不是输入中的标准答案；该口径同时改变推理时输出协议，不能将收益全部归因于训练。

~~~bash
./agent_simulation/python.sh experiments/cot_sft_pilot.py compare \
  --cohorts <frozen-evaluation-cohorts> \
  --baseline <action-only-predictions> --sft <cot-predictions> \
  --response-protocol generated_cot \
  --comparison-release <v2-comparison-release> \
  --action-only-training <completed-action-only-run> \
  --cot-training <completed-cot-run> --output <comparison-report.json>
~~~

生成预测时分别使用 style=direct 与 style=concise；direct 对照则两组都使用 direct。新生成记录保存解码实现哈希，防止混用修复前后的 JSON 约束。generated_cot 允许不同输出上限并明确报告；报告包括所有尝试的有效性与请求耗时，含有耗时记录的失败请求。请求耗时之和不是并发任务的墙钟耗时。解释质量统计是机械证据检查，不能代替语义审查。

声明为配对 CoT-SFT 实验时，上述三个数据/训练路径须一起提供。评估核对实际训练输入、完整样本数、已完成状态、相同初始化/训练选项/步数、非零解释 token，并拒绝训练父请求出现在评估中。不提供这些路径时标记 training_control=not_verified。此检查验证本地训练产物，服务启动时仍须核对实际加载的 adapter 身份，不能仅凭服务模型名证明已加载对应权重。

本轮离线检查使用已有 32 条真实教师输出及对应 63 条原始监督，保留历史审查通过的 2 条，两臂均为 63 条、输入/标签/顺序/父请求权重一致。实际 Qwen3.5-9B tokenizer 编码得到 action_only 为 0、cot 为 732 个 rationale token；最低数量设为 3 时导出停止。这里 2 条仅用于验证链路，不作为正式 CoT 实验的充分数据。审计保存在 tmp/cot-chain-real-audit-20261127/，原始输入未修改。本次 174 项相关测试通过（含离线 mock、编码和 CPU 微型夹具）；没有启动服务器上的真实教师服务、生成或训练任务。


v2 接入复核：真实历史数据导出后仍为每臂 63 条、2 条已审查解释，实际解释 token 为 0 对 732；输入、标签、权重和教师原文一致。历史 Response/Query 报告完整重算一致。新增协议及相关回归共 55 项测试通过；本轮只做离线复核，没有启动模型生成或训练。产物：tmp/cot-chain-real-audit-20261127/v2-paired/ 和 tmp/response-metrics-protocol-replay-20261127.json。

### Response diagnosis and low-cost CoT follow-up (2026-11-27)

The generation report now includes `response_metrics` pooled within each
mode/policy/model/style/domain, plus the same metrics within existing strata.
The pilot comparison uses the same helper. Click and purchase each report item
TP/FP/FN, micro precision/recall/F1, scene macro F1, positive-scene macro F1,
false-action rate **among empty-target scenes**, and predicted-action rate among
all attempts. Every arm includes an always-no-action baseline on identical
targets. Undefined ratios are null. Invalid outputs remain in denominators,
receive no empty-match credit, and contribute missing positive items as FN;
they are reported separately rather than counted as false actions.

Historical parent-weighted mean F1 remains explicitly labeled as legacy:
empty/empty earns one, child scores are averaged within each parent, then parents
receive equal weight. Its bootstrap samples parents. Adding baseline reporting
does not alter any existing metric or interval (separate RNG). The pilot gate
now also requires the legacy improvement interval to exclude zero, beating the
no-action legacy score, and improving pooled action micro-F1. This conservative
pilot screen is not a significance test for micro-F1 or a claim of generalization.
Parent reconstruction reports complete/incomplete coverage; metrics over complete
parents are separately labeled and do not establish full-exposure performance.

Offline replay of the frozen Train-internal temporal audit (115 child scenes,
78 parents; this is not official Test):

| Metric | Zero-shot | Action-only SFT |
| --- | ---: | ---: |
| Click micro recall | 3/83 = 0.0361 | 4/83 = 0.0482 |
| Purchase micro recall | 0/17 = 0 | 2/17 = 0.1176 |
| Click micro-F1 | 0.0674 | 0.0825 |
| Purchase micro-F1 | 0 | 0.1600 |
| Click false-action rate on empty targets | 2/32 = 0.0625 | 2/32 = 0.0625 |
| Purchase false-action rate on empty targets | 0/98 = 0 | 5/98 = 0.0510 |
| Any-action prediction rate | 6/115 | 14/115 |

Legacy F1 is 0.616239 -> 0.618162; always-no-action is 0.620513.
Its paired improvement interval crosses zero. The new conservative pilot gate
therefore fails. There are 59 complete and 19 incomplete parents (115 of 150
expected children). The old child metric cannot demonstrate complete multi-click
or multi-purchase recovery. Do not tune models/thresholds on this frozen audit.

The actual 596-row training input hash matches the saved config. It contains
247 Query and 349 Response rows, with zero rationale tokens. Of Response rows,
82 are empty, 228 click-only, 35 click+purchase, and 4 purchase-only. Although
empty rows are 23.5% of examples, their share after the existing parent/scene
weighting is 40%; positive responses retain 60%. Purchase-positive weight is
14.51%. This is not simply an overwhelmingly empty training set.

Using the actual tokenizer and supervised masks, under normalized action loss
the coefficient on a head's first ID-or-close token is row_weight / action_tokens.
The summed coefficient is empty/positive = 36.5474/15.4803 for click and
49.4213/2.6065 for purchase. Short empty arrays concentrate coefficients on this
decision more than longer IDs. These are **coefficients, not measured losses or
gradients**; other action tokens remain supervised. The saved logs lack classwise
loss/gradient measurements. Keep loss/learning-rate settings fixed until a
bounded model diagnostic can test this hypothesis. The optional existing
Response decision auxiliary loss is a candidate for a separate future ablation,
not a change made in this replay.

Replaying current mechanical checks and checking original review hashes gives:
64 old attempts -> 13 exact actions -> 11 mechanically eligible -> 9 reviewed
passes; 32 sampled attempts -> 11 exact -> 8 mechanically eligible -> 2 passes.
These are historical outputs, not a measurement of the newer Chinese short-CoT
prompt. Their errors include mistranslation, confusing catalog prices with user
exposure, conflating equal-title IDs, unbounded claims from a finite history,
overgeneralizing candidate relevance, and an unsupported Habitual style.
The short prompt now also explicitly limits price provenance and absence claims.
Fact-reference existence remains weaker than semantic entailment.

Follow-up protocol (initial offline plan; executed probes are reported below):

1. Freeze a small stratified **dev** subset before viewing new outputs. Compare
   the saved base and SFT on identical original prompts and original action schema;
   no optimizer steps. Record native JSON action continuation log probabilities,
   relevant empty-vs-ID decision scores, predicted selections and actual targets.
   If scores are normalized over a finite menu, label them as conditional menu
   probabilities, not calibrated real-world behavior probabilities. The existing
   OptionPredictor changes the prompt to option labels and cannot alone diagnose
   the original JSON decoder.
2. On the same dev cases, ablate historical events together with all historical
   facts/derived summaries that expose them, preserving current query/candidates.
   Recheck differences to ensure the history did not leak through another field.
   Compare matched base/SFT accuracy as well as probability changes. Do not use
   audit cases to choose context, decoding or thresholds.
3. Use a small fixed **fit** batch for the current short EKB teacher, one attempt
   per scene, 768 output tokens and bounded concurrency. Record all attempts,
   exact-action yield, mechanical yield, semantic yield, tokens and latency.
   Review hard factual errors separately from weak interpretations. Prefer this
   measurement before adding more sampling, PRM, strong/weak entropy passes or DPO.
   Repair, if later tested, gets at most one targeted retry and separate provenance;
   answer-conditioned repair must be a separate experimental arm.
4. CoT-SFT remains a later authorized experiment. Both arms receive the same
   context at inference; the explanation is generated output supervision, never
   a gold rationale appended to inference input. Keep rows, labels, initialization,
   steps and evaluation fixed, and require nonzero reviewed rationale supervision.

Local replay artifacts: `tmp/response-metrics-replay-20261127.json` and
`tmp/response-diagnosis-20261127/{training-supervision,training-decision-masks,cot-audit}.json`.
The offline replay above made no Customer model calls. Later GPU headroom became
available, allowing the bounded inference probes below. No training was started.


### Frozen native-probability probe

`experiments.customer_response_probe` prepares a small diagnostic without model
calls. It selects dev Response scenes with distinct parents, stratified by the
projected click/purchase labels, before any new predictions. The default cap of
five candidates limits the cost of scoring every valid single-item action.
This selected subset is not a representative accuracy estimate.

```bash
./agent_simulation/python.sh -m experiments.customer_response_probe prepare \
  --cohorts result/cot-sft-pilot-20261126/customer-context-cohorts \
  --output tmp/response-probe-plan.json --count 8 --max-candidates 5
```

The paired conditions preserve the original system, runtime instructions,
current query, candidate order, candidate product facts and static profile.
The history ablation removes historical event facts, history lists, available
history counts and the complete behavior-features block (including the
background activity snapshot). It states that history is unavailable, rather
than claiming the user has no history. Unknown context/profile fields require
review before ablation so new derived memories cannot silently leak.

Once a GPU can hold the existing model, this separate command performs inference
only, sequentially scoring Base and the saved SFT adapter on the exact same plan:

```bash
./agent_simulation/python.sh -m experiments.customer_response_probe score \
  --plan tmp/response-probe-plan.json --model models/Qwen3.5-9B \
  --sft-run result/cot-sft-pilot-20261126/customer-context-sft-v1 \
  --output tmp/response-probe-scores --device cuda:0
```

Scoring reuses the actual assistant token masks and model loader. It compares
native complete JSON continuation likelihoods, including end tokens, without
changing the task into option-label prediction. One canonical JSON serialization
is used per action. Normalization is only across those exact strings: it is not
the probability of every possible serialization of an action, not calibrated
click/purchase probability, and not free generation. Mean token log probability
is saved descriptively; it does not replace the native summed likelihood for
ranking. Purchase first-token scores condition on that option's preceding click
choice, and an ID's first token does not encode its entire selection probability.

Outputs include all option scores, matched action metrics with no-action
baselines, and gold action NLL/first-token log probabilities by behavior class.
Base weights are checked against the SFT config; frozen request/plan hashes and
adapter hashes are recorded. Partial files remain marked running until all
conditions finish. There is no optimizer, backward pass, or parameter update.
Real parameter-gradient magnitudes remain unmeasured. Decoding variants are
reported below; they must not be conflated with full-sequence menu ranking.

The prepared real plan at `tmp/response-diagnosis-20261127/dev-probe-plan.json`
contains 8 parents: 3 no-action, 1 purchase-only, 2 click-only, 2 click+purchase.
Official-tokenizer verification covers 190 continuations per model (both context
conditions), with a maximum of 4,428 tokens. All decision masks and ablations
passed. The subsequent real 9B runs used this same frozen plan; results follow.


### Real decoder and history probes (2026-11-27)

The frozen 8-case dev plan was executed with the existing 9B base and unchanged
Customer SFT adapter. Native menu scoring completed 380 continuations across
the two models in about 112 seconds of model scoring. Separate short-generation
runs checked HF greedy, vLLM greedy without a response-format constraint, and
the original vLLM JSON-schema constraint. No parameters were updated.

A concrete decoder artifact was found. The single-item contract is an
`anyOf` of whole-object `const` actions. In the installed xgrammar version,
these compile to literal compact JSON strings even with `any_whitespace=True`.
For example, the compact form below was accepted while the spaced form was
rejected, despite identical parsed objects:

```json
{"clicked_item_ids":[2453715],"purchased_item_ids":[]}
{"clicked_item_ids": [2453715], "purchased_item_ids": []}
```

On these inputs the base model naturally emitted the spaced style, whereas the
SFT adapter emitted compact JSON. Within the same vLLM backend and identical
messages, removing only the response-format constraint changed 4/8 base actions
and 0/8 SFT actions. The unconstrained full-context outputs matched HF greedy
for all 16 model/case pairs. This isolates a format-enforcement effect for this
probe; it is not evidence that every omitted action has this cause.

The minimal correction is in `inference.vllm._generation_schema`: only
single-item Response object constants are expanded into equivalent typed object
and array schemas. Prompt text, original validation schema, candidates, allowed
actions and model weights stay unchanged. Other constants remain untouched.
This preserves empty actions, purchase without click, and the single-ID
constraint without requiring compact serialization. It adds no model calls or
extra review stage.

Verification included semantic action-set equivalence, plus the actual vLLM
environment accepting 285 valid compact/spaced/indented strings and rejecting
24 invalid/out-of-scope/multi-ID actions. The old grammar rejected all 95 spaced
legal actions. In 16 real full-context replays, the corrected strict decoder
matched unconstrained outputs exactly while keeping action validation.

Both history conditions were then evaluated under the corrected strict decoder:

| Model / context | Scenes predicting an action | Click TP / FP | Purchase TP / FP |
| --- | ---: | ---: | ---: |
| Base / full | 4/8 | 2 / 2 | 0 / 0 |
| Base / history withheld | 7/8 | 1 / 6 | 0 / 0 |
| SFT / full | 2/8 | 2 / 0 | 1 / 0 |
| SFT / history withheld | 6/8 | 1 / 5 | 0 / 2 |

The cohort contains four true click items and three true purchase items.
Removing historical events **and** derived behavioral statistics increased
actions but worsened item selection in this probe. Retain relevant history;
these results do not identify whether individual events or aggregate features
drive the effect, and do not establish population-level improvement.

Gold action-token NLL fell after SFT for all eight full-context cases, yet that
did not imply that the true action became the preferred output. The normalized
canonical-string menu and greedy decoding sometimes rank actions differently.
Do not interpret the menu's all-empty SFT argmax as the deployed decoder's
behavior. Per-class losses are available in `native-probe/report.json`;
actual parameter-gradient diagnostics remain outstanding.

All results above concern existing action-only SFT, not CoT-SFT. The frozen audit
was not rerun or used for model/threshold selection. Its old numbers remain
historical results under the old decoder and cannot establish reliable CoT
benefits. Future controlled CoT evaluation must freeze the corrected decoder
and account for output serialization/instructions as well as shared user context.

Artifacts under `tmp/response-diagnosis-20261127/`: `native-probe/`,
`native-greedy/`, `constrained-probe/`, `unconstrained-probe/`,
`decoder-grammar-check.json`, `repaired-decoder-probe/`,
`repaired-history-probe/`, and the final `repaired-comparison.json`.
All newly started model processes/services were allowed to finish and only our
own services were stopped. New 32B teacher CoT generation remains pending
sufficient GPU headroom; it was not replaced with a different teacher.
