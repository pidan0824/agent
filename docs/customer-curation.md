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

### 后续低成本蒸馏与 DPO：待验证方案

[Mirroring Users 第 3.2 节](https://arxiv.org/html/2508.18142v1#S3.S2)
对强弱模型分别生成多条解释，再用同一行为预测器计算条件分布；论文实验每模型每场景采样 10 次。当前适配层保留该计算方式，但轻量 --sft-only 路径明确跳过弱模型、选项预测器和 DPO。单次贪心 CoT 的质量试跑不能算作完整不确定性蒸馏。

历史 comparison 产物离线重算：27 条有概率的 Response 分支，7 个有配对的场景，其中 4 个只有 2 对、3 个只有 1 对。新增采样数之外，全部原指标数值不变；这些旧小样本不支持稳定的场景排名结论。result 中现有训练配置/结果只有已完成 SFT，未找到已完成 DPO。此次只重算指标，没有模型调用。

后续按阶段控制预算，不把所有步骤叠到每个场景：

1. 先完成已冻结 11 个 fit 场景的单次、768-token 短 CoT 质量试跑。记录全部失败，分别检查动作匹配、证据引用和语义；没有合格解释时保留真实动作，不认定原始日志一定有噪声。
2. 若短 CoT 通过率可用，再从同一 fit 集冻结至多 4 个场景做不确定性诊断，兼顾行为类型和一部分随机对照。强弱模型各采样 3 次、相同非零温度和完整种子列表，并使用同一固定版本的预测器。此阶段最多 24 次解释生成和 24 次选项打分；全 11 场景各 10 次则为 220+220 次。该预算不含前面的 11 次贪心质量试跑，N=3 只是低成本诊断，不是论文 N=10 复现。
3. 不把 temperature=0 的旧输出混入非零温度样本来凑采样数，也不只对动作正确的解释计算不确定性；否则估计的分布已被正确性筛选改变。报告全部尝试、有效配对数及重复解释。选项打分虽只输出一个 token，仍需完整输入预填充，不能当作零成本。需要更多采样时保留既有结果，新增等量双边样本，而不是重跑整个集合。
4. DPO 等 CoT-SFT 对照后再推进。优先复用相同提示下已有的真实错误分支，不额外调用大模型编造 rejected。chosen 必须动作精确匹配且通过语义审查；仅有低熵、较低 NLL 或较长解释都不够。没有可信 chosen/rejected 时保留合格 SFT 样本，不强制造对。答案条件修正另设实验；给过真实动作的生成提示不能伪装成普通未见答案的轨迹。

分阶段生成／离线概率评分现已接入，见下方恢复与分阶段运行说明；原在线路径继续保留。lightweight 现可用于 sft-only 或 defer-probabilities：分阶段强弱生成复用同一短提示与历史预算，重复采样显式设置非零温度。新离线评分入口默认要求 Response 解释通过原文哈希绑定的语义审查，才进入解释 SFT 或成为 DPO chosen；机械检查不能代替语义审查。此次不改变原公式、筛选阈值及 vendor 代码。

离线审计：tmp/uncertainty-cost-audit-20261127/report.json 与 historical-metrics.json；驱动 tmp/audit_uncertainty_cost.py。该驱动比较修改前后公式输出，新增字段单测同时覆盖单对、多对和未配对分支。

### 防泄漏、导出与统计

拆分前确定官方原请求 split。场景读取检查父请求、全部子列表与 manifest 的划分一致；导出再次检查同一数据版本下的 source_request_id 不得跨划分（不因数据域不同而放行）。训练读取只接受 train，不重新随机划分子列表。source_request_id、scene_id、原始标签及任务协议传到训练元数据。test 不进入训练导出。

- sft.jsonl / preferences.jsonl：接现有轨迹 SFT、DPO 训练器。
- filter-decisions.jsonl：每条行为与证据检查、保留状态、淘汰原因、概率失败原因。
- pair-decisions.jsonl：场景为何配对／未配对。
- filter-funnel.json：按阶段、模型、数据域、候选数量、当前／原始行为类型、候选协议及 EKB/direct 统计保留率；原因可能重叠。
- uncertainty-metrics.jsonl：作者强弱分歧指标，采样匹配率不等于生成动作正确率。 现在额外记录 matched_sample_count，即同场景双方均有有效结果的配对采样数；未配对分支不计入。只有 1 对时，混合分布熵减平均条件熵恒为零，不能据此断言没有数据歧义。

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

## 随机种子、断点恢复与分阶段概率评分（2026-11-27）

后处理概率采样改为按运行 seed、场景 cid、模型分支和样本编号派生独立 RNG。并发完成顺序不再改变 sampled_action/preference；模型本身的生成 seed 沿用原协议。与旧共享 RNG 的随机采样结果可能不同，原始概率、熵和 Log Loss 公式不变。

新生成运行使用 v3 manifest。原命令加 --resume 复用已完成样本；允许调整两层并发数，数据、场景、种子、模型配置及相关代码身份必须一致。旧 v2 产物不会静默升级为可续跑的新运行。原始观察监督在模型调用前保存，generated.jsonl 每完成一条生成即追加并刷盘，probabilities.jsonl 独立保存概率结果。传输失败会停止当前运行，修复服务后显式续跑；失败请求及未完成请求可重新执行，已持久化的有效或无效输出都不重复生成。只丢弃没有结束换行的中断尾行，完整行损坏则报错；运行锁防止同目录被两个写入者同时使用。

completed.json 只在整个阶段完成后写入；中断运行的部分 records.jsonl 不视为完整产物。完全缓存命中时，恢复不请求模型服务。尚未落盘的在途 HTTP 请求不保证服务端取消，也不保证从网络层实现 exactly-once。

完整强弱路径可显式延迟概率评分。以下为后续运行模板，本轮未运行真实模型服务；两个生成阶段使用同一冻结场景和相同设置，仅变更所选分支、服务地址及输出目录：

~~~bash
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_curation \
  --dataset <release> --scenes <frozen-fit-scenes> --output <strong-run> \
  --cot-preset lightweight --defer-probabilities --generation-variants strong \
  --samples 3 --temperature 0.7 --max-new-tokens 768 --bge-device cpu

./agent_simulation/python.sh -m agent_simulation.evaluation.customer_curation \
  --dataset <release> --scenes <frozen-fit-scenes> --output <weak-run> \
  --cot-preset lightweight --defer-probabilities --generation-variants weak \
  --samples 3 --temperature 0.7 --max-new-tokens 768 --bge-device cpu

./agent_simulation/python.sh -m agent_simulation.evaluation.customer_scoring \
  --run <strong-run> --run <weak-run> --output <scored-run> \
  --predictor-url <fixed-service-url> --predictor-model <fixed-model> \
  --predictor-tokenizer models/Qwen3.5-9B --concurrency 4
~~~

实际运行时分别配置对应的 --strong-url/--weak-url；只检查被选中的生成器服务。生成阶段不创建概率预测器，也不导出训练文件；保留原始监督、生成轨迹与 label-free probability_context。评分入口只加载预测器 tokenizer、连接一个预测器服务，校验上下文哈希、原始提示和候选顺序；观察标签和生成动作不进入预测器。不同分支文件须同配置、同场景且无重复任务才可合并；不完整或变更的生成结果拒绝评分。评分运行也支持 --resume，复用已完成概率，无须重跑教师。评分后先输出 response-review.jsonl 审查队列，并保留原始动作监督；没有经外部确认的语义 pass 时，Response 解释不进入解释 SFT 或 DPO chosen。审查后用 customer_filtering 离线导出，无须重新生成或重新打分；原在线研究路径的默认机械筛选规则保留。服务权重仍须按既有流程核对；服务模型名不等于远端权重验证。

语义审查只在队列副本中填写 verdict、reason、reviewer，保留 trajectory_id、messages_sha256：

~~~bash
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_filtering \
  --input <scored-run>/records.jsonl --observed-actions <scored-run>/observed-actions.jsonl \
  --reviews <reviewed-response-review.jsonl> --require-reviewed-rationales \
  --output <new-reviewed-export>
~~~

解释审查优先检查：当前需求／历史偏好是否对应具体候选属性，是否说明相对其他候选的选择理由，以及点击／购买解释是否与动作一致。允许合理、明确为推测的模拟动机，不以心理状态不可观测本身判失败。仅有类目兴趣或复述历史记为 weak_decision_support；不能区分候选时允许表达不确定性，不将其自动认定为原始行为日志有噪声。同时保留商品 ID、原始属性、历史行为和价格的必要核对；动作是否匹配真实标签仍由独立门槛判断。

--reviews 会自动启用严格语义门槛；原文哈希不符、未知或重复审查 ID 会报错。严格模式要求 Response 解释的 pass 同时具有正确原文哈希、原因和审查人；保存记录中的旧 semantic_review 不会自动算作外部审查。该门槛同时作用于解释 SFT 和 DPO chosen，direct 动作监督不需要解释审查。未通过解释不会导致原始行为监督丢失。

复核使用 3 个真实 Train 父请求、5 个 Response 子场景和 8 条原始监督，CPU 离线回放历史输出并注入生成/评分中断，两阶段分别复用已完成的 2 条任务；16 个最终分支的原始轨迹、概率及采样结果与在线路径一致，全部父请求行为保留，历史窗口未变。此次概率使用测试分布、没有真实模型调用，不能用于判断新 CoT 质量或蒸馏效果。记录：tmp/customer-recovery-real-20261127/verification.json；驱动：tmp/verify_customer_recovery_real.py。

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
| 模式与模型 | 原有 fixed/autonomous、强弱模型协议 | fixed；sft-only 仅 strong，或 defer-probabilities 分阶段强弱生成 |

上述数值可被显式参数覆盖；重复采样仍要求非零 temperature。轻量预设保留相同 EKB schema、事实上下文、证据检查和原始生成动作，不在失败后改写答案。Train Query 仍直接用原始监督，不调用 teacher。短解释不等于已通过语义审阅。

`--scene-concurrency` 控制同时处理的场景分支数量，`--sample-concurrency` 控制每分支内样本并发。最多同时执行前者 × 后者个采样工作流；轻量默认每分支只有一个样本，因此最多 4 个。固定模式每工作流一次生成调用。调度按完成顺序接收结果，全部完成后按稳定身份整理最终记录；保留独立样本 seed，惰性拉取有限任务，不为全量场景一次建立全部 asyncio tasks。错误会取消并回收待处理任务；已提交的 HTTP 请求不能保证服务端立即停止。新逻辑没有启动或扩容模型服务。

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

### Small fit-only teacher input preparation (2026-11-27)

The next teacher diagnostic is frozen under `tmp/new-cot-fit-probe-20261127/`.
It uses 8 distinct users/parents from the complete-Train selection (539,012
parents; 6,313 selected), not the old 3,000-request subset. Selection is limited
to the existing fit time interval [1, 655707], excludes parent IDs in existing
dev/audit case files, and takes two parents from each observed behavior group.
These are diagnostic strata, not an unbiased accuracy sample. This small batch
uses established histories; it does not test cold-start performance.

The current scene builder produced 11 Response children plus 8 Query rows in
30.96 seconds on CPU, with no skipped scenes. Parent actions are recovered
exactly by the union of child targets. Response children comprise 2 no-action,
4 click-only, 3 purchase-only and 2 click-and-purchase cases. Two children have
only one candidate: their real purchase supervision is retained, but they
cannot measure discrimination between products.

Offline audit uses the production history budget (2048), scene budget selector,
prompt builder and both actual tokenizers. Each child retains 4–9 strictly prior
history events; 5 have an exact-query history match, 1 has a candidate-item
interaction match, and none has an unsupported identical-visible-product group.
Changing current target labels leaves all teacher requests unchanged. The
non-system messages match the paired observed-action inputs. History timestamps
and feature cutoffs are checked, as is recovery of every parent's actions.

| Input token count | Total over 11 calls | Range |
| --- | ---: | ---: |
| Short prompt, Qwen3-32B tokenizer | 46,577 | 3,274–5,595 |
| Same context, original explanation instructions | 56,059 | 4,136–6,457 |
| Short prompt, Qwen3.5-9B tokenizer | 46,319 | 3,263–5,568 |

Short instructions reduce input tokens by 16.9% on this batch at the same
history budget. Every input fits 8192 tokens with 768 reserved for output, using
either tokenizer. Eleven one-sample calls cap output at 8,448 tokens; neither
this cap nor input-token reduction proves wall-time savings or teacher quality.

`selection.json`, `input-audit.json`, `teacher-requests.jsonl` and
`generation-command.json` record the selection, input/source hashes and exact
planned production command. Reproducible local drivers are
`tmp/prepare_new_cot_fit_probe.py` and `tmp/audit_new_cot_fit_probe.py`.
The teacher remains Qwen3-32B BF16. This preparation did not start teacher
generation, update model parameters, or touch other GPU processes. Actual
generation quality, latency and matched training remain unverified.

### 候选属性与用户偏好的映射：本轮质量诊断

本轮优先修正解释的判别能力，不要求模拟动机都是可观测事实。允许将合理动机明确写成推测；核心是将当前需求／历史偏好映射到具体候选属性，并解释相比最有竞争力的另一候选为何更合适。泛泛的类目兴趣、复述历史、仅列商品属性，都不能单独完成这一步。普通与 lightweight EKB 共用该要求，保留原 schema；轻量版只将 knowledge/evaluation 放宽为各 1–2 句，输出预算仍为 768，不增加 PRM、逐商品调用或第二个审查模型。

不能区分近似候选时可以表达不确定性，但不能把不确定性直接等同于不点击／不购买；缺少同商品历史购买不意味着本次不会购买。购买和点击分别解释。只有一个候选时不虚构对照商品，判断该候选与需求的关系及行为类型。审查关注解释是否提供选择依据，不因合理的假设性心情而单独拒绝；商品属性、实际历史与价格仍不能改写。

论文对比（不同协议的数值不能直接横比）：

- [Mirroring Users](https://arxiv.org/html/2508.18142v1)：固定 5 候选单选、5 次随机输出取平均 Accuracy；除 MIND 外采用混合策略构造曝光。原 EKB Knowledge 已要求候选属性与偏好关联，示例包含具体候选比较。Table 2 的随机数据对照，有解释／无解释在 MIND 为 31.4%／25.7%，合成曝光为 53.9%／48.3%；这是该训练流程的总体收益，不能证明每条解释都有区分力或本项目已获益。
- [Agent4Rec](https://arxiv.org/html/2310.10108v3)：MovieLens、Amazon-Book、Steam，1000 个 agent 各 20 个候选的偏好二分类，报告 Accuracy/Precision/Recall/F1；另评估评分分布、社会特征及交互反馈。MovieLens 正负比从 1:1 到 1:9 时，F1 从 0.6982 降至 0.3232。未交互项作为负例，不等于已曝光但拒绝的商品。
- [SimUSER](https://arxiv.org/html/2504.12722v1)：同类 20 候选偏好分类，并报告评分 RMSE/MAE、交互指标及 55 次真实 A/B 测试中模拟／实际业务指标的 Spearman 相关性。MovieLens F1 在 1:1 为 0.7771、1:9 为 0.4972。相关性、评分误差与逐次点击／购买准确率是不同任务。
- [RecAgent](https://arxiv.org/html/2306.02552v3)：20 名 MovieLens 用户的后续 a 件真实商品与 b 件负例混合，要求选择 a 件，以命中重合比例评估；另有人类对行为序列的真假偏好判断、聊天／广播 1–5 分可信度与信息茧房熵。预先给定选择数量简化了行为数量判断，主观可信度也不等于精确预测命中率。

本项目仍需保留真实不行动样本和点击／购买区分，分别报告正例行为表现、空动作误报与原请求聚合；不能为追求论文式高分删除困难评估样本。候选质量筛选用于训练数据预算分配，不把“老师没选对”当作原始日志有错。当前尚未完成同数据／同上下文的 CoT 训练收益验证。

本轮真实复核：原短提示的 11 个 fit 子场景（8 个原请求）全部输出合法且通过机械引用检查，但动作精确匹配仅 2/11，均为空动作；9 个有行为场景无一完全匹配，点击 TP/FP/FN=0/6/6，购买 TP/FP/FN=0/0/5。正式 CoT 暂存报告明确列出 9 条 action_mismatch、2 条 semantic_review_pending，不导出零解释训练。问题已在生成阶段出现，不能归因于筛选删掉了大量好解释。产物 tmp/short-cot-live-20261127/quality-audit.json。

修订提示后，复用鞋和手机壳两个原场景及正式 AgenticClient，模型、BF16、温度 0、种子、768-token 上限及非 system 消息不变。两条开始比较具体候选，但动作均仍错误且均漏购买；鞋例将一次点击推为低价偏好，手机壳例将缺少品牌偏好推为无品牌优势，仍不足以支持选择。鞋例原基线使用同步 CPU offload，本次为异步 prefetch；这只是定性诊断，不是严格性能 A/B。不宣称质量或准确率已改善，不将这两条纳入 CoT 训练。产物 tmp/choice-mapping-probe-20261127/report.json、逐请求 calls/result.json。

冻结的 11 条真实输入仅 system 提示改变，每条增加 177 tokens（总输入 46,577 → 48,524）；候选、历史和 action schema 完全一致，均满足 8192 上下文并预留 768 输出。80 项相关单测通过，日志 tmp/choice-mapping-regression.log。所有本轮自建推理服务均在小规模试跑后退出，无训练、概率打分或其他进程操作。后续输入审计已发现背景商品级历史在共享上下文投影时丢失（见下一节）；应优先修复信息通路与候选关联，不以挑选可区分场景代替修复，也不因比较句式出现就扩大生成。

### 原论文提示词与实际上下文对照：信息丢失和候选关联（2026-11-27）

本次只进行源码／真实输入审计及 6 次单场景推理诊断，不修改正式生成、筛选或训练实现，不启动训练。上一轮把问题主要归为提示词和近似商品难区分不够准确：原始背景快照包含有用的商品级历史，实际模型上下文却丢失了这部分信息。不能以删掉这些场景或回退动作监督代替修复。

**确定的输入缺失。** Catalog.customer_summary 读取同用户、严格早于当前 cutoff 的 train 快照，包含 recent_clicked_item_ids、recent_purchased_item_ids、item_details。FrozenSampleTools._base_context 仅经 customer_behavior_features 传递该背景；后者的 background_snapshot 只保留时间和 statistics_30d，未传递上述三个字段。丢失发生在上下文投影阶段，并非 history token budget 截断。场景去重反而已经读取这些历史 ID 来保护有历史差异的候选，形成“构造场景知道差异、教师看不到差异”的不一致。

冻结的 11 个子场景（8 个原请求）全部有这类遗漏，背景商品详情为 24–39 项；只有 2 个子场景的快照与当前候选存在同 ID 交集。不能把同 ID 恢复的收益外推到其余场景：它们还需要历史商品属性与新候选的关联。手机壳场景背景含 14 条购买 ID 及有品牌／卡通款式商品，而当前教师只看到更稀疏的搜索事件；“没有品牌偏好，所以无品牌更好”并无充分依据。

鞋例 125619:positive:227325 的背景时间为 169024，当前 cutoff 为 169025，已与实时 catalog 逐字段核对。候选 227325 在保留的历史点击列表出现 6 次、购买列表出现 1 次；227326 分别为 3 次和 0 次。两者同属 seller 60738，标题分别为“082-13真皮超轻女士休闲特价板鞋”和“55-9牛皮超轻女士特价休闲板鞋”。这是快照列表出现次数，不是终身交互总数，不能与搜索事件相加；历史购买也不意味着本次必然复购。原始输入有差异，教师之前没有收到。

**原提示词与实际输入的差别。** 未修改的 vendor/usermirrorer/template.py（上游 b6e628cdc6e9a439265d5375cdc44bf5d8a29e80）在 Knowledge 明确要求关联候选属性和用户偏好。当前共享提示也已表达该要求，但模型实际收到的是压缩事实表：candidate_context 不直接包含标题、品牌等属性，需要跨 facts/fact_sources 按 ID 关联；背景商品列表更是在此前已丢失。压缩表可无损解码，不能把编码本身说成丢数据，但它增加了关联负担。当前协议还增加引用 ID、JSON schema 和多种约束，与原论文的输入输出并不等价。只复制原文提示不能恢复缺失信息。

**单场景受控诊断。** 使用 Qwen3-32B BF16、温度 0、相同种子和候选顺序／动作 schema，不向输入加入当前动作标签。各组均保留失败输出，不筛掉场景。前 4 种有效对照输出上限均为 768：

| 输入处理 | 输入 tokens | 预测点击 | 预测购买 |
| --- | ---: | --- | --- |
| 当前正式上下文和提示 | 5,772 | 227350 | 空 |
| 只恢复原始历史点击／购买 ID 列表 | 6,236 | 227350 | 空 |
| 只将已有属性与候选放在一起，信息无损且不补历史 | 7,366 | 227350 | 空 |
| 恢复历史，并为所有候选展示标题及其在历史列表中的出现次数 | 6,603 | 227325 | 空 |
| 原论文风格提示，沿用当前上下文和项目输出协议，输出上限 1536 | 6,084 | 227350 | 空 |

原论文风格在 768 上限的首次调用被截断，先输出了较长的引用 ID 列表；表中为增加到 1536 后完整输出的补充诊断。它仍保留项目动作空间、事实规则和引用 schema，不能视为原论文复现或同成本比较。完整输出还存在将 raw price 写成“元”、错误概括无历史点击等问题。

候选对齐组按全部 8 个候选统一计算次数，没有手工指定优先商品；解释明确比较 227325 的 6 次与 227326 的 3 次历史点击。真实目标是点击且购买 227325，因此只修正了点击，完整动作仍不匹配。它还没有在 evidence_refs 中引用新增的背景事实，解释中的类目关联仍较弱；不能直接判为合格 CoT 或作为训练收益证据。单例只能支持修复输入通路的方向，不能证明属性偏好泛化、购买预测或总体质量已解决。

**正式修复应遵循的顺序。**

1. 在共享上下文入口保留严格先于 cutoff、同用户的背景商品级历史及相关详情，保留来源／窗口，不把快照和事件重复累计。生成、动作对照和概率打分使用同一上下文。
2. 使用确定性的候选视图，将候选属性、同 ID 历史及相关历史商品属性放在一起；无同 ID 交互仍应保留品牌、款式、类目、店铺等可关联的信息。依据输入和已有预算选取历史，不依据正确答案挑证据，不删除困难场景。
3. 将这部分证据接入已有 owner/time 校验和引用映射；减少反复输出引用／格式的负担，保留必要事实检查。不能用高频历史规则代替用户偏好推断，也不能把“缺少记录”解释为反向偏好。
4. 固定原来的 11 个场景复核输入完整性、实际属性比较、点击／购买及引用正确性；不只验 JSON 或只看点击命中。所有失败样本保留并报告，CoT 未合格不能宣布 CoT-SFT 已完成。

审计产物：tmp/cot-mapping-rootcause-20261127/coverage-audit.json、input-audit.json、requests.jsonl、各组 calls/result；后续对照为 tmp/cot-mapping-followup-20261127/。两轮合计 6 次推理，正式源代码和训练数据未更改。两轮自建推理服务及其作业均已退出，其他进程未操作。

### 背景历史正式接入与小规模复核（2026-11-27）

共享 FrozenSampleTools / CurationTools 现已传递背景快照。customer_background.py 生成带用户、快照时间和商品 ID 的 B 类证据，完整保留近期点击／购买 ID 序列（包括重复项），并关联历史商品属性。完整序列与候选本身的出现次数始终可见；商品详情和搜索事件共用已有 2048-token 历史预算，逐条完整选取并记录 available/shown，不删场景或动作目标。背景和搜索事件分别呈现，不累计成一个伪造的总次数；缺失购买列表不解释为零购买。

候选旁展示其属性、快照交互次数及相关历史引用。关联依据为同 ID、已知品牌、店铺、类目和标题词，不按点击次数给商品打分，也不读取当前动作标签。“无品牌／UNKNOWN／其他”不会自动产生正向品牌匹配。候选商品属性以 P 引用 ID 为键存储在候选行，避免重复整张事实表；旧平铺记录及此前 attribute-facts-v1 编码仍可读取。快照引用校验覆盖用户、严格先于 cutoff 的时间、来源和商品身份，但不把可引用误当作文本语义正确。

生成、无 CoT 请求和延迟打分复用同一个上下文。概率打分也使用共享的无损编码；实际 Qwen tokenizer 会拆分部分两位数字选项，因此保留可用的单 token 数字，并以单 token 字母补足其余选项，动作集合和 1+3N 概率协议不变。新上下文实现与编码器纳入生成／打分恢复身份，旧任务不能在新协议下混入缓存概率。

所有原有 11 个场景均保留。最后一次离线预检：
- 全部原始背景 ID 列表与 catalog 一致，用户／时间检查通过；当前动作标签变更不影响输入。
- 历史详情显示 2–10 项，原快照包含 24–39 项；未显示的详情由预算选择，不意味着没有相关行为。
- Qwen3-32B 输入 4,861–7,160 tokens；Qwen3.5-9B 为 4,865–7,163。全部满足 8192 总长度并预留 768 输出。
- 实际 OptionPredictor 请求在带上第二轮解释后为 3,998–6,226 tokens；逐条解码回原上下文一致。这是离线请求检查，没有发送概率推理或训练请求。

已完成两轮各 11 次 Qwen3-32B BF16 推理，温度 0、同种子和原候选顺序：
| 版本 | 完整动作匹配 | 点击 TP/FP/FN | 购买 TP/FP/FN |
| --- | ---: | --- | --- |
| 此前缺失背景历史的短 CoT | 2/11（均为空动作） | 0/6/6 | 0/0/5 |
| 恢复背景并关联候选 | 3/11 | 2/7/4 | 1/0/4 |
| 再调整引用输出顺序／提示 | 2/11 | 2/8/4 | 1/0/4 |

这不是可靠收益结论：样本很小，迭代同时调整上下文和提示。鞋例两轮均预测对点击和购买；打底衫开始比较历史的显瘦／设计属性与当前候选，但仍把真实购买预测成点击。第一轮鞋例有跨商品价格混淆，第二轮把 227326 的历史点击也概括成购买。手机壳仍有价格单位和品牌偏好推断问题。仅换引用顺序没有解决候选属性漏引，两轮 knowledge/evaluation 都没有 P 类引用。机械检查通过不等于解释合格。11 条逐条问题保存在 manual-quality-notes.json，没有据此删原始场景，也没有用动作监督宣布 CoT-SFT 完成。

因此最后又将候选属性编码从“属性名 → [引用 ID, 值]”改为“P 引用 ID → [属性名, 值]”，使引用身份直接可见，不增加字段值副本。该格式已完成无损往返、旧格式兼容和真实输入预检；此前第三轮两次启动都在显存检查阶段退出，未停止其他任务，也未以更小教师替代。后续资源空闲后，最终格式已完成原 11 场景推理，结果见下节；不能声称质量已解决。

目前评估测试 207 项、相关数据构造测试 46 项通过；追加输入校验及恢复身份变更后还做了针对性回归。两轮各自总耗时约 187 秒、191 秒（含服务启动等，不是可横比的纯推理吞吐），自建服务／生成子进程均已退出。没有训练、权重更新或大规模生成。

产物：
- tmp/background-context-20261127/input-audit.json、requests.jsonl：最终格式输入及一致性审计；
- generation/、generation-v2/：两轮完整输出，包含所有失败场景；
- quality-audit.json、manual-quality-notes.json：指标和逐场景质量问题；
- tmp/background-context-service-20261127/、tmp/background-context-v2-service-20261127/：服务与作业完成记录。


### 最终编码复核与 Seller 输入通路审计（2026-11-27）

补完同一批 11 个 Customer 场景：10 条格式有效，1 条因 evidence_refs 重复 ID 无效；
完整动作匹配 0/11，点击 TP/FP/FN=1/9/5，购买=0/0/5（无购买预测）。
共有 6 个 knowledge/evaluation 阶段开始引用 P 属性，机械证据检查通过 9/11，
但引用出现不等于解释正确。鞋例再次错选 227350，仍把 227326 的点击概括为购买，
并用未知价格推导相对便宜；手机壳例前后对候选标题是否包含 Z5x 的描述自相矛盾。
另有 raw price 被写成元。所有失败输出保留。最终格式并未证明优于此前两轮，
不据这 11 条诊断数据选择训练版本，也不将其记为合格 CoT-SFT 数据。

已核对实际请求中的候选 P 属性及引用 schema：P 值确实传入，schema 允许字符串引用，
没有只允许历史引用的枚举限制。概率打分输入再次无损还原为同一上下文。
问题同时包括此前的字段丢失和修复后模型仍会错误使用证据；不能继续只靠编码、
追加提示或引用合法性，宣称偏好与属性对应已解决。

Seller 存在同类入口分叉：普通 seller_sample/runtime 保留 business_summary 和具体竞品；
旧 FrozenSellerSampleTools 丢失这些字段，且未合并 product_details 中的品牌和类目；
旧 pilot 的 seller_case 又独立构造了一份简化上下文。旧 Seller 训练文件 136 行
（128 动作 + 8 解释）均没有 business_summary / competitor_examples。
Customer 旧 596 行动作训练上下文同样遗漏了背景商品列表，所以不是仅 CoT 路径的问题。

现增加 observed_seller_context，共享给 FrozenSellerSampleTools 和 pilot seller_case。
按商品放在一起展示属性、带时点的近 30 天经营快照、近 10 条报价观测、
市场统计和已有检索排序前 3 个竞品（不足则全部），保留 available/shown、
报价时间及店铺，不调用额外模型。快照与报价事件不能累加；同类不等于同规格。
同一商品可引用自身证据 ID，引用检查验证实际嵌套块；这仍不是语义判真。
新 pilot 准备协议记录上下文源代码身份，防止混用旧的部分准备结果。

冻结 6 个真实 fit Seller 样本（不变价、上涨、下跌各 2），全部保留：
经营快照恢复 6/6；原始具体竞品数 27/64/0/64/64/11，显示 3/3/0/3/3/3。
generic/pilot、direct/CoT 的事实输入一致，改变当前目标不改变输入，时间和店铺约束通过。
例如商品 931598 的保留报价事件没有点击购买，但源快照有 1337 曝光、46 点击、2 单；
其报价为 2080，类目中位数 990，而相似竞品报价也是 2080。
旧简化上下文缺少区分这些证据的依据。

6 次 Seller 推理全部格式有效、引用机械检查通过，但都维持上次报价，
仅 2 个不变价样本完整命中；没有稳定使用新增的竞品属性。
商品 911288 解释用到了恢复的 3 单快照；商品 109119 仍泛称没有近期购买，
未交代含 32 单的快照窗口；商品 660170 甚至把 3380 描述成高于 p75=4785。
因此本次只证实输入通路修复，不能宣称 Seller CoT 质量改善。
没有同批旧 Seller 输入的推理对照，也不能从六条结果估计整体收益。

评估目录 212 项测试通过，包括字段保留、同事实输入、目标不进入上下文、
越界时点/错误店铺拒绝和嵌套引用篡改测试。本次 17 场景诊断总耗时约 424 秒，
包含服务启动等；自建服务和作业均已退出。未训练、未覆盖旧训练文件，
未操作其他进程。新的输入应重新准备冻结样本；旧模型效果不能归因于本次修复。

产物：tmp/dual-context-audit-20261127/ 下的 seller-samples.jsonl、
seller-verified.json、seller-verified.jsonl、seller-predictions.jsonl、
seller-quality-audit.json 及逐调用日志；Customer 全部结果在
tmp/background-context-20261127/generation-v3/ 和 quality-audit.json。

### 为什么最新一轮降到 0/11：同样本实际输入输出审计（2026-11-27）

本节直接核对保存的请求与原始模型输出，不以测试通过代替 CoT 质量检查。
before 指 generation-v2（2/11），after 指 generation-v3（0/11）；更早缺失背景的版本也是 2/11，
但两个命中均为空动作，与本次 before 的两个正例命中不同，不能混为同一个基线。

11/11 对的事实解码后完全相同，候选及顺序、动作 schema、模型/种子/温度一致。
逐条反向变换候选表示后，用户输入也一致；系统提示唯一文字差异是说明属性编码的一句话。
before 为 属性名→[P 引用 ID, 值, 元信息]，after 为 P 引用 ID→[属性名, 值, 元信息]。
字段遍历顺序未变；不是把属性重新排序。每条输入仅增加 3 tokens；11 条均 stop 正常结束，
输出 768 上限未造成截断。服务启动参数一致，但历史并发调度不等于严格可复现的同批次推理。
因此可以排除本轮再次丢字段、候选改变、动作空间改变和截断；尚不能把因果单独归到
属性排列或那一句提示，不能从两次运行推断普遍退步。

原始动作只改变 4 条：鞋 227325、外套 202712、卸妆水 138079、手机壳 1457724。
完整命中的两个正例（227325、202712）都丢失；手机壳则从错选商品变成点击对商品但漏购买。
新版本仍有 3 个正例选中目标商品（2941589、1457724、138544），但都未正确预测购买。
无效的 430121 虽因重复引用未通过格式检查，其原始动作也是空、目标为购买；
即使忽略引用校验，完整动作仍为 0/11。没有删除场景，没有通过修补输出制造命中。

| 场景 | before → after（点击；购买） | 偏好—属性对应审计 |
|---|---|---|
| 138079:no_action | [2549788]；[] → [275870]；[] | 两版对价格敏感性给出相反假设，缺少足够依据；raw price 单位也误写成元。 |
| 90476:no_action | [2861208]；[] → [2861208]；[] | 候选均与查询相关，但并无个人偏好证据区分其一；两版都把相关性过度转成点击。 |
| 187074:positive:1045264 | [656694]；[] → [656694]；[] | 有真实历史对应，但将熟悉度当作再次选择倾向，未有效比较套装数量/起发条件。 |
| 249491:positive:202712 | [202712]；[] → [1089105]；[] | 这是第二个完整命中丢失；两版 related_history 均只关联同二级类目的运动袜。 |
| 249491:positive:353512 | [1089105]；[] → [1089105]；[] | 历史冬季/加绒/保暖外套未充分用于候选区别，模型主要依赖类目标签。 |
| 254183:positive:2941589 | [2941589]；[] → [2941589]；[] | 两版都选中实际购买商品，却仅预测点击；不能称完全没有属性对应。 |
| 392266:positive:138544 | [138544]；[] → [138544]；[] | 品牌属性有对应，但购买头仍未预测；不能由历史价差直接推出此次不购买。 |
| 392266:positive:430121 | []；[] → []；[] | 未证明过去购买某品牌就排斥其他品牌；即使忽略引用格式错误，原始动作也不匹配目标。 |
| 333208:positive:1457724 | [5565385]；[] → [1457724]；[] | 新版本点击选品改善，解释仍自相矛盾，且遗漏真实购买。 |
| 125619:positive:227325 | [227325]；[227325] → [227350]；[] | 事实关联断在价格归属和点击/购买区分；这是完整命中丢失之一。 |
| 125619:positive:3407032 | [3407034]；[] → [3407034]；[] | 历史休闲板鞋不足以支持具体款式和无品牌偏好；原始标题/类目本身也有不一致风险。 |

**两个掉分案例的证据。** 鞋 227325 的快照点击/购买为 6/1，227326 为 3/0；两者价格均未知。
227350 是棉麻绣花半拖，历史价格 2990，与前两双真皮/牛皮休闲板鞋在材料和款式上不同。
before 选对但已误称 227326 也购买过；after 仍有这一误读，并用无法成立的低价比较改选 227350。
模型知道共同的店铺和休闲鞋兴趣，却没有稳定将皮质/板鞋历史对应到候选差异。
不能用历史次数最多作为新选择规则，也不能把旧命中当作解释合格。

外套 202712 标题含 跑步/连帽/骑行/速干/防风；1089105 标题为撞色短款拉链外套，三级类目是跑步服装。
历史中有真实点击的加绒/保暖/冬季跑步外套，以及购买的运动袜。after 把购买宝图曼运动袜
泛化成偏好科里芙运动外套，并把未知对手价与 4990 比较；历史品牌与候选品牌并不相同。
更具体的组织问题是：两候选的 related_history 都只指向运动袜，因为二级类目同为运动服。
曾点击的加绒外套虽在事实表中，标题二字片段 Jaccard 分数不足 .15，未进入候选旁关联。
例如历史 353509 对 202712 的分数约 .118；其他瑜伽/户外外套因类目不同也未被关联。
这会突出粗类目而弱化用户当前需求相关的历史，但该组织缺陷两版都有，不能独自解释本轮掉分。

**结论与下一步边界。** 新编码提升部分 P 引用出现率，没有证明提升属性理解。
当前有三类不同问题：表示变化伴随输出不稳定；共享历史关联过粗；解释中混淆事实和偏好推断，
以及点击/购买预测本身不足。未知价格比较和把点击说成购买是可核查错误；
未命中真实行为并不自动证明所有偏好推断错误，也不能保证仅凭这些输入恢复随机个体行为。
下一步应先用字段名直观可读的表示、同一提示作受控比较，再改候选旁历史关联；
共同类目/店铺证据与能区分候选的材料、版型、功能证据应分开呈现，避免重复粗关联。
不应仅因旧版在这 11 条命中更高就选择旧提示，不应继续堆约束或用筛选替代生成质量。

已准备两失败场景共 12 次顺序诊断请求：原请求重放、统一格式说明下的两种表示、
只抄录候选属性/历史价格/快照点击购买次数的读数探针。目标标签不进入请求；
预检全部满足 8192 上限。启动前显存检查失败，未加载模型、未发送这些请求。
因此暂不能区分表示本身、一句格式提示和推理调度各自的因果贡献，也不能判定
模型是读不准字段还是能读取却在决策时忽略。正式生成代码本轮未变更，没有训练或干预其他进程。

审计产物：tmp/cot-regression-audit-20261127/paired-summary.json、paired-records.jsonl、
paired-details.md、manual-findings.json、requests.jsonl、diagnosis-status.json。
原始调用日志仍保留在各 generation 目录；诊断脚本为 tmp/audit_cot_regression.py、
tmp/run_cot_regression_calls.py。


### 12 次受控诊断完成：排列影响与重现性边界（2026-11-28）

在修改历史关联之前，使用昨天准备的同两场景、同 12 份请求完成诊断。
请求文件 SHA256 与预备版本一致；日志逐条 request 与对应准备请求一致。
Qwen3-32B BF16、temperature=0、seed=20261127、8192 客户端长度限制；
每次仅一个请求在服务中运行，普通生成上限 768，事实抄录探针上限 2048。
GPU 0 启动前空闲 83132 MiB；GPU 1 已满，不操作其他进程。

| 场景 / 条件 | 属性名作键 | P 引用 ID 作键 |
|---|---|---|
| 鞋：原请求重放 | 点击 227350；不购买 | 点击并购买 227325 |
| 鞋：统一格式说明 | 点击 227325；不购买 | 点击并购买 227325 |
| 外套：原请求重放 | 点击 202712；不购买 | 点击 202712；不购买 |
| 外套：统一格式说明 | 点击 202712；不购买 | 点击 202712；不购买 |
| 鞋：只抄录候选事实 | 8 商品 × 7 字段全部正确 | 8 商品 × 7 字段全部正确 |
| 外套：只抄录候选事实 | 2 商品 × 7 字段全部正确 | 2 商品 × 7 字段全部正确 |

“统一格式说明”使用完全相同的系统提示、输出 schema、候选顺序及解码后的事实，
仅输入表示不同；这两对均在同一个服务进程内顺序执行。鞋例的购买判断发生变化，
外套动作相同。因此有本例对表示敏感的证据，不能推出 P ID 作键普遍更差；
本次鞋例恰好是 P ID 作键的完整动作命中。样本只有两个，未重复每个条件，
不足以估计效应大小或宣称总体质量提升。

固定排列、仅替换格式说明：同一个服务内的三对动作均不变，但解释内容会变。
例如外套旧排列的原提示主要比较类目，统一说明后引用了历史加绒保暖外套，
再比较候选标题中的跑步用途；新排列原提示仍捏造“与历史运动袜价格接近”，
统一说明后转向候选的防风功能。动作一致不能当作解释质量一致。
鞋旧排列的动作也随提示变化，但这一对跨了服务重启，不能据此单独归因给提示。

需要特别纠正此前的因果判断：4 份原请求重放中有 3 份动作与昨天不同，
包括鞋两种排列和外套新排列。日志保留 schema 的字典顺序，生成客户端和
schema 转换代码 hash 与原运行一致；不是日志重新排序或这些代码变更造成。
模型文件修改时间早于原实验，但本次未重新计算全部权重文件 hash。
原运行跨场景并发，本次顺序调用；GPU/批次等运行条件的作用尚未逐一隔离。
因此昨天的 2/11→0/11 不能全归因于排列或一句提示，不能把旧版两次命中当成稳定基线。

4 次事实抄录共 140 字段全部正确，包含鞋 227325/227326 的未知价格、
6/1 与 3/0 的点击/购买次数。这说明两种表示在这个读数任务中均可正确读取；
不代表决策时一定使用正确，也不代表推断出的偏好忠实。
例如原请求鞋例仍把 227326 的点击概括为购买；历史关联和决策推断仍需修复，
不能根据格式有效或动作碰巧命中就接受为高质量 CoT。

执行途中，第一个请求完成后，诊断包装脚本因重复日志文件名退出。
修正为每次调用独立日志后保留首条结果，接着执行剩余 11 条，未重复调用。
两次自建服务均正常退出。总计恰好 12 次模型调用，全部格式有效、stop 结束；
模型调用日志累计约 505 秒，两段服务总耗时约 644 秒（包含两次启动）。
没有训练，没有停止其他进程，没有在诊断中修改历史关联。

完整结果：tmp/cot-regression-audit-20261127/outputs.jsonl、
calls.jsonl、calls-02.jsonl 至 calls-12.jsonl、controlled-summary-20261128.json。
服务记录：tmp/cot-regression-service-20261128/ 和
tmp/cot-regression-resume-service-20261128/。上述结果均基于修复历史关联之前的输入。


### 单独修复缺失或只有粗类目的历史关联（2026-11-28）

上述 12 次诊断全部完成、服务退出后，才修改 customer_background.candidate_links。
保留现有 P ID 作键编码、提示、输出 schema、历史预算和数据目标。
当某候选已能关联同 ID、已知品牌、店铺、细分类目或标题时，沿用原规则；
仅在没有具体匹配、或只匹配到二级类目时，补充查询相关的历史标题关联。
候选和历史标题至少共享两个非纯数字二字片段，且有共享片段出现在当前查询中，
才允许补充标题关联。展示 matched_query_terms 供核对；这是可见词片段相同，
不是偏好结论、同款证明或必须点击的规则。仍最多显示三条关联，
不会用当前正确动作或历史行为次数选择优先候选。

先固定旧快照单独重建关联，11 条均确认只有 related_history 变化，
其余事实完全一致；再通过正式入口重建原 11 条场景。
9 条完整输入未变，仅 249491 的 202712、353512 两个子场景变化：
候选 202712 从只关联运动袜 353520，变为优先关联真实点击的 353509，
其标题包含加绒、外套、冬季跑步、保暖；运动袜与防风冲锋衣作为后续相关历史保留。
候选 1089105 也能关联真实点击的拉链/加绒外套，不再只有运动袜。

正式入口仍服从原 2048-token 历史预算。由于新增关联和匹配词占用空间，
上述两条分别少显示历史商品 124764（针织开衫）、573867（皮衣外套）的详情；
完整快照 ID 序列、候选逐项点击/购买次数、目标和候选顺序均保持。
因此固定快照的关联验证与正式预算后的输入变化分别记录，不能混作严格相同事实的生成对照。
其余 9 条输入逐项相同；手机壳、鞋等已有具体关联未受影响。

全部 11 条正式输入仍满足教师/学生 8192 总长度（含预留 768 输出），
direct/CoT/概率上下文一致，改变目标不影响输入，owner/time 校验通过。
4 项新增回归覆盖长标题跨类目关联、无查询时不放宽、单片段误匹配拒绝、
已有同 ID 具体关联保持；评估测试总计 216 项通过。
没有在关联修复后追加模型调用或训练，尚未证明该修复带来 CoT 质量/准确率改善。

产物：tmp/history-links-20261128/before/、after/ 的 requests.jsonl 与 input-audit.json；
comparison.json 保留固定快照与正式重建的两种对照，以及预算省略详情的逐条记录。


### 偏好对选择协议与小样本不确定性标记（2026-11-28）

新增 --preference-selection-protocol：默认 uncertainty-ranked-v2，
legacy-v1 保留原先缺失指标时使用 0 排序的兼容路径。新协议仅允许具有有效、
有限 filter_column 指标的候选偏好对参与不确定性排序；无指标场景在
pair-decisions 中记录 missing_uncertainty_metric，不影响独立的 SFT 选择。filter_column=random 保留随机选择路径，
不要求不确定性指标，报告明确其不是不确定性排序；SFT-only 记录未执行偏好选择。
vendor 的行为抽样、指标计算及分层过滤函数未修改。延迟打分读取旧生成
manifest 时缺省使用 legacy-v1；新生成 manifest 的 args 记录实际协议。

max_pairs 保持兼容默认 10000。filter-funnel 和离线导出 manifest 的
preference_selection 记录候选对数、可排序对数、缺失指标对数、实际入选对数、
实际是否截断。由于 vendor 按分层分配，依据实际返回数判断是否截断，
不把 max_pairs 小于候选数直接当作截断证据。未截断时明确报告
“不确定性排序未起作用”，含义是没有缩小候选集合，仍可能改变导出行顺序。
这不表示引用/动作质量门没有作用，也不是不确定性蒸馏效果的证据。

uncertainty-metrics 增加 low_sample=(matched_sample_count<3)，报告统计数量；
不自动删除这些场景，不改 vendor 指标值，也不将 N=3 当作统计可靠性的保证。

离线重放 result/customer-mirroring-pilot-20261121/comparison/records.jsonl：
48 条记录中有 32 条 Response，7 个强弱匹配场景（4 个 N=2、3 个 N=1）
全部标记 low_sample。原代码与 legacy-v1 的 SFT 和偏好对逐项相同；
新协议的 SFT、两个 Query 偏好对也相同。两版均没有通过质量门的 Response
偏好对，所以该既有导出集合没有变化，指标除新增 low_sample 外逐项不变。
若其他旧导出中包含 selection_metric_available=false 的 Response 对，
新协议会移除它们；有截断时，这也可能改变剩余有指标场景的名额分配。
对有完整指标且不截断的输入，原始 chosen/rejected 内容保持不变。
历史 result 产物未覆盖。

新增测试覆盖有效与缺失指标混合、正负指标截断/不截断、旧路径兼容、
N=1/2/3 标记及导出报告。最终 evaluation 回归 220 项通过、0 项失败。
这里验证的是选择和报告行为，没有调用模型、
训练或声称偏好数据质量/训练效果提升。
重放及对照结果见 tmp/preference-filter-audit-20261128/report.json、
before/、legacy-v1/、uncertainty-ranked-v2/ 和 replay.py。
