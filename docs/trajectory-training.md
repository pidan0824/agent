# 单模型自主规划、轨迹 SFT 与 DPO

Customer 与 Seller 现在都可由同一个 Qwen3-32B 选择查询、读取证据、决定停止并输出动作。Customer 使用 EKB 作为可否定的解释候选，Seller 使用竞争、需求、稳定报价、参考价和促销假设。最终输出为 `{"rationale": {...}, "action": {...}}`；环境只提交验证后的 action，rationale 是生成的简短依据，不是真实心理标签。默认 SFT 仿真路径保持不变。

运行时分别用 `--customer-agentic`、`--seller-agentic`，可同时启用。Seller 沿用现有目标分批与统一报价提交；每个目标批次由同一模型做查询与决策，预算按批次计算。工具只读冻结市场，本角色权限、时间截止和调用预算由代码执行。不同目标批次不共享隐式对话。Customer Query 与同次 Response 续接对话，但当前只执行一次真实搜索，不包含多轮试搜。

Customer 作者式候选／干扰项构造、结构化 EKB 和强弱解释筛选见 [Customer 数据构造](customer-curation.md)。下方 trajectories 入口保留原始曝光轨迹收集。Customer Query 输出直接动作；Response 理论开启时使用 EKB 加证据标记，关闭时直接输出动作。此入口没有公共预测器概率，故 Customer Response 不导出 DPO 偏好对，须经 customer_curation / customer_filtering；不能绕过新证据与 Log Loss 条件。Seller 与 Query 的结果分差规则保留。

## 生成与筛选

```bash
dataset=result/lite-5pct-quality-3k-20261112/release/sft
./agent_simulation/python.sh -m agent_simulation.evaluation.trajectories \
  --dataset "$dataset" --output result/customer-trajectories \
  --role customer --split train --count 100 \
  --samples-per-decision 4 --temperature 0.7 \
  --model Qwen3-32B --tokenizer models/Qwen3-32B

./agent_simulation/python.sh -m agent_simulation.evaluation.trajectories \
  --dataset "$dataset" --output result/seller-trajectories \
  --role seller --split train --count 100 \
  --samples-per-decision 4 --temperature 0.7
```

默认使用本机 `http://127.0.0.1:8001/v1`。`--record-id` 可重复指定同一组请求进行消融；Customer 使用 session ID，Seller 使用 `seller:商家:商品:观察索引`。`--no-theory` 移除理论解释，保留相同事实、工具、预算和最终动作协议。理论开启时核心候选解释直接进入输入，不再依赖模型是否主动读取 reference。

整个决策组生成完后才读取当前标签评分。Query 用 BGE 语义相似度；Response 报告点击和购买集合 F1 的平均值（两个空集对应 F1=1），但接受判定分别检查两个集合：有交互的头各自达到 F1 门槛，无交互的头必须预测空集；Seller 同时检查相对真实报价误差、是否变价和涨跌方向。默认 Query 阈值 0.7、点击和购买各 0.8、Seller 误差上限 5%，均为可配置的开发起点，应只在预留验证集上调整。Seller 的 `--seller-change-tolerance` 默认为 0，严格区分观测涨跌与持平；该参数是相对前价的容差，应按价格精度在开发集确定。缺少过去有效报价时不接受轨迹。Seller 依据中的每条 evidence 必须引用输入或实际工具返回的 `[S…]` 编号，引用不可用或人工语义审阅失败时不得进入 SFT/chosen；编号验证仅证明来源可用，不证明自然语言推断正确。旧轨迹没有编号时不自动补写。

Customer 的公共预测器 Log Loss 配对、原始动作保护和分层筛选统计见 [Customer 数据构造流程](customer-curation.md)。

输出保留所有有效、失败和未通过轨迹。`sft.jsonl` 包含通过筛选且去重的轨迹；`preferences.jsonl` 每个决策最多生成一个“最好通过轨迹 vs 最差有效轨迹”的偏好对，默认分差至少 0.05，平局跳过。两分支必须同人物、同初始状态、同任务、同初始 prompt/tools，分别保留自己的真实工具返回。没有通过轨迹或没有偏好对是合法结果，不能自动认定原数据有噪声。

`--split test` 只保存全样本预测与评分，训练导出为空。当前 Response 仍是给定真实 Query/曝光的条件预测，不能冒充由预测 Query 触发的新曝光下的端到端准确率。报告保存各人各决策的通过率、查询序列与简短假设，供检查同人稳定性和跨人差异；没有把这些统计自动解释为个性化已经成立。

## 训练

```bash
# 先检查真实 student 模板能否编码，并核对 assistant token 数量。
./agent_simulation/python.sh -m agent_simulation.datasets.trajectory \
  --input result/customer-trajectories/sft.jsonl --mode sft \
  --tokenizer models/Qwen3.5-9B

CUDA_VISIBLE_DEVICES=1 ./agent_simulation/python.sh -m agent_simulation.training.trajectory \
  --mode sft --input result/customer-trajectories/sft.jsonl \
  --model models/Qwen3.5-9B --output result/customer-trajectory-sft

CUDA_VISIBLE_DEVICES=1 ./agent_simulation/python.sh -m agent_simulation.training.trajectory \
  --mode dpo --input result/customer-trajectories/preferences.jsonl \
  --model models/Qwen3.5-9B --init-adapter result/customer-trajectory-sft/adapter \
  --output result/customer-trajectory-dpo
```

训练后用已有 vLLM adapter 导出入口准备服务副本，在支持原生工具调用的本机服务中注册 student adapter，再将 `--agentic-model` 指向该服务名称。默认 Transformers 最终 JSON 后端不执行工具循环。

Seller 使用自己的轨迹和独立 adapter，不混合角色。SFT 可用 `--init-adapter` 从现有行为 SFT 起步；DPO 必须显式指定 SFT adapter。默认 1 epoch、LoRA rank 64、学习率 1e-5、DPO beta 0.1，都是需要在预留验证集上选择的起始配置。训练入口仅接受明确 train 来源，不读 Test 选检查点；不自动调参或声称训练损失就是行为改进。

SFT 对 action、rationale、工具决策、格式 token 分别求平均 NLL，再加权相加。动作权重为 1；`--rationale-weight`、`--tool-weight` 默认 1，`--format-weight` 默认 0.25，均为开发起点，不是验证过的最优值。直接动作样本没有 rationale 损失。分段依据最终 JSON 的顶层值区间及真实 tokenizer offset；若模板改写内容或 token 对不上则明确失败，不猜测边界。日志分别报告各项 NLL 与 token 数。

读取训练集后，每个数据版本、角色、人物、原请求、阶段分配相同总权重，再均分给子场景及保留轨迹；权重在整个读取集合内归一化到均值 1，不能在单条 minibatch 内归一化。Query 与 Response 各有独立请求预算。没有 source_request_id 的旧数据按 initial_state_id 分组；没有 scene_id 的按初始状态分组。DPO 同样使用请求权重，但概率仍为完整分支 token 对数概率之和，不使用 SFT 的分段系数。

仅 assistant 实际生成的简短依据、工具名称/参数、停止与最终动作参与监督和 DPO 概率。system、user、工具返回和模板插入的 assistant 头全部 mask。每个分支用自己的工具观察计算条件概率，按 assistant 决策求和；不能把 chosen 的工具结果复制给 rejected，也不能把工具返回当作输出训练。

DPO 在任何更新前以初始 SFT 的 eval/no_grad 状态计算并保存固定 reference log-prob，避免再常驻一份 9B。policy/reference 使用相同分支与相同 mask。关闭 dropout，记录数据、模型、tokenizer、adapter 身份。输出 checkpoint、optimizer/scheduler/RNG、日志、固定 reference 和最终 adapter。当前入口要求全新输出目录，不支持原地恢复 optimizer；需要继续训练时从已保存 adapter 新开运行，不能称为精确续训。

## 传统 Seller 对照

```bash
./agent_simulation/python.sh -m agent_simulation.evaluation.micro.seller_baselines \
  --dataset "$dataset" --output result/seller-traditional \
  --train-count 2000 --count 500
```

实现保持前价、滚动中位数、EWMA、保持历史竞品相对价、保持历史竞品分位、两阶段 Logistic+Ridge、GBDT。监督模型只用 train 拟合，固定超参数不在 test 调整；缺失竞品时两种竞品规则退回前价，并单独报告覆盖量。`--rules-only` 显式跳过监督模型；后者需要项目的 `baselines` 可选依赖 scikit-learn。

每种方法保存独立 JSONL 和统一 Seller 指标，包括持平误调价、变价召回、方向 macro-F1、相对幅度误差和按商家平均 MAE。`--sample-id` 可重复指定与 LLM 相同的测试请求。各方法共享严格过去的自身报价与近期同类目其他店铺公开报价；这是共同可访问事实，不意味着自主 agent 实际读取了全部事实。价格单位未确认，目标是复现报价，没有利润最优或因果弹性解释。

## 本轮范围

目录审计修复已部署，完整回归 699 项中 698 通过、1 项旧快照资产缺失跳过。新增链路只做定向逻辑检查、小模型训练闭环、真实 Qwen tokenizer 兼容和少量真实数据联通；没有再次运行全套，也没有启动完整 9B 训练或把小样本当效果结论。具体实跑结果保存于 `result/agentic-training-20261119/`。

研究路线与 [Mirroring Users](https://aclanthology.org/2026.acl-long.273/) 的理论辅助生成、结果筛选、SFT+DPO 一致；工具轨迹、Customer 多标签与 Seller 连续报价是本项目扩展。下一轮实验应预先固定训练内验证与最终评估队列，对比默认行为 SFT、轨迹 SFT、SFT+DPO、理论消融和传统 Seller 方法。已经用于旧 checkpoint 选择的 Test 不再声称是未使用的盲测。


## 完整 Train 来源的小样本验证

`experiments/cot_sft_pilot.py` 是预算受限的 Customer/Seller 验证入口。
候选来源是完整 Lite Train 标签与完整候选统计，不读取旧 quality-3k。
完整 Customer Train 有 539,012 个原请求，Seller Train 有 11,874,348 个报价样本。
每个场景选多少请求与每个请求生成几条解释是两个独立预算；
先筛选、构造上下文，再调用教师，不给全集逐条生成长 CoT。

该验证把 Train 按时间划分为 fit/dev/audit，分别固定 128/64/128 个原请求或报价。
Customer 按四种点击/购买组合保留样本，Seller 保留不变/涨价/降价。
当前小实验受 8,192 token 服务窗口限制，只覆盖曝光数不超过 32 的 Customer 请求，
并按实际 token 数检查输入；它不能代表完整版中的长曝光场景。
正式选样仍使用 datasets.customer_selection 的全部曝光分层，不能套用小实验的容量限制。
样本少是控制本轮验证成本，不是把今后的训练候选池缩成固定小集合。

上下文在调用前冻结：Customer 使用严格过去的描述统计、最近十次历史和完整原曝光；
Query 隐藏当前 Query 和曝光，Response 才使用已记录 Query。
Seller 使用严格过去的自身报价统计、最近报价和历史市场位置。
事实复用紧凑编码；不因真实点击/购买结果删候选、不把缺失价格补成当前报价。
“最近历史 + 与当前查询相关的较早历史”可作为下一项独立对照，
但当前实验尚未实现或验证这种检索策略，不能宣称已提升效果。

简洁教师提示保留原动作/EKB schema、事实引用、未知信息和行为预测要求，
将解释限定为短句，单次调用；Query 直接采用真实动作监督。
生成保存所有成功和失败尝试，显式同身份续跑跳过已完成 ID，并使用新的调用日志。
机械检查通过仅表示动作吻合和引用可追溯，不等于文字推断正确。
可在教师目录保存 semantic-reviews.json，以 case_id 对应 verdict、reason 和
messages_sha256；导出核对原文哈希，语义失败的解释不能进入训练。

SFT 同时保留所有 fit 的真实动作，只有合格教师原文作为额外解释轨迹，
避免“教师只猜对无动作”把购买正例从训练集中删除。
导出复核上下文、schema、原始回答及证据，不重写动作凑出正确 CoT。
Customer/Seller 分开训练 LoRA，复用现有 assistant-only、请求/阶段平衡训练器。
dev 用于选择配置；audit 只在配置冻结后评估，官方 Test 不参与本轮调参。

比较要求相同输入、输出预算、解码参数和 base 配置，运行时还须核对实际 base 权重。
Customer 报告点击/购买 F1、分层结果和 Query 相似度；Seller 报告相对报价误差、
涨/跌/不变 macro-F1、误调价率以及沿用上次价格基线。失败输出计入分母。
小样本均值上升不自动等于稳定提升；配对区间和保留评估结果仍须报告。
训练 loss 下降、代码测试通过或教师匹配率不能代替 SFT 对 zero-shot 的行为比较。


### 开发集定位后的监督修正（保留集尚未开启）

第一轮 Customer 的平均 Response F1 上升，但真实点击样本 F1 从 0.343 降至 0.044；
购买正例 F1 在基线和 SFT 中都为零。不能把这种更多输出空动作的变化当作达标。
8 条 Train 诊断样本也未学好正例，说明问题不只是开发集泛化。

审计发现：直接 Response 答案约 45.1% 的 token 属于固定 JSON 格式，
但旧轨迹编码把整个直接 JSON 都记为 action，format-weight 并没有降低这些固定键的权重。
现在直接及解释包装内的 Response 只把两个动作数组的值记为 action，
固定键和外层格式记为 format；空数组仍是需要学习的真实决策，不被掩掉。

轨迹 SFT 增加显式 click-weight / purchase-weight（默认均为 1，不改变旧权重配置）。
第二轮采用原训练器已有的 2/4 方案，按原请求取较大的正例系数，
再分配给子场景和解释变体，保留无动作请求；Query 和 Seller 不套用这些系数。
样本、上下文、底座、种子、学习率与轮数保持一致，从同一底座重新训练。
这是一组针对已定位监督问题的修正，不声称单独隔离了每项改动的因果贡献。

开发集诊断后，Customer 的通过检查还要求点击正例 F1 不降低、购买正例 F1 提高，
以及存在 Query 对照时其相似度不降低；平均分提高本身不足以通过。
这些检查在开启保留集之前确定，原始基线和第一轮结果保留。
