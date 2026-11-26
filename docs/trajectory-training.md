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
