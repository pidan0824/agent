# Agent 实现与单模型自主查询

当前保留原 SFT 路径，并提供可独立开启的 Customer/Seller agentic 路径，默认服务模型为本机 Qwen3-32B。同一模型选择证据、查询和停止，再形成搜索、点击/购买或报价。Customer EKB、Seller 定价候选理论、轨迹筛选、SFT→DPO 与传统 Seller 基线均已提供独立入口，见[实现与命令](trajectory-training.md)。

## 当前代码如何运行

`simulation/run.py` 装配市场、模型和两个角色；`simulation/episode.py` 推进 Tick。Seller 在冻结状态上形成报价草案，环境统一生效后，Customer 决定 Query，环境执行一次搜索，Customer 对展示列表形成 Response，最后统一结算和更新记忆。

原路径不是“通用 planner 加 SFT actor”：它由 `agents/context/` 和检索模块按固定流程准备上下文，`agents/decisions.py` 请求模型输出一次最终 JSON，拒绝模型工具调用。历史摘要、竞品筛选和记忆反思可以调用其他模型，但不是自主规划器。原有 `prompts/skills/*.md` 是直接插入 Prompt 的决策说明。

| 层 | 入口 | 职责 |
|---|---|---|
| Customer 生命周期 | `agents/customer.py` | Query、展示、Response、结算；开关选择原路径或 agentic 路径 |
| 自主查询循环 | `agents/agentic.py` | 工具白名单、预算、结果大小、冻结版本检查、完整轨迹 |
| 外部模型 | `inference/agentic.py` | OpenAI 兼容 Chat Completions，保留原生 tool calls，默认 Qwen3-32B |
| 描述性特征 | `agents/context/customer_features.py` | 从严格更早的本人事实计算品牌重复、点击数量、查询重复和类目范围 |
| Customer skill | `agents/prompts/skills/customer-shopping/SKILL.md` | 核心决策说明，证据规则与按需读取的理论参考 |
| 冻结数据评估 | `evaluation/micro/customer_agentic.py` | 同请求条件预测、多轨迹、理论消融、可选基线比较 |

## Customer 自主查询边界

Query 与 Response 使用同一 agentic 模型；运行时 Response 续接本次 Query 的可见对话与查询结果。每阶段分别冻结快照并计算 SHA256，阶段内市场 revision 改变时直接失败。

工具只允许读取本人历史、本人长期记忆、当前展示商品，以及加载 skill reference、记录简短且未验证的行为假设。工具直接绑定查询后端和快照副本，不经过会登记 `memory_evidence` 的通用 `runtime.execute`。模型不能改价、结算、写记忆或访问别人的历史。

`search_products` 会创建 pending request，是市场动作，因此本版不把它当只读查询。模型在 Query 阶段输出一次最终搜索词后，由环境执行搜索；Response 最终决定点击与购买，再由环境结算。多次试搜、回退搜索及纯沙箱搜索尚未实现。记忆反思仍在阶段外发生；比较多条轨迹应从同一份已准备好的快照开始。

每阶段默认最多 6 次模型调用、8 次工具调用，每次模型响应最多调用 3 个工具。单工具返回最多 12,000 字符，工具返回总量最多 40,000 字符，模型响应总量最多 32,000 字符；每次生成默认最多 2,048 tokens，输入沿用项目 32,768 tokens 总上下文限制。返回过大时明确标注截断，原始结果仍写入轨迹。最后一次调用只允许最终 JSON；耗尽预算、非法最终动作或输入超长均显式失败，不自动提交空动作。

agentic Response 对完整展示集形成一次决策，不使用旧 ranker 的 Top 3 封顶，也不静默拆分或截短候选。超过输入容量的请求计为失败。沿用既有报告口径：F1 针对有效完整输出，同时报告所有请求的失败率与端到端精确集合匹配率；不能只展示有效输出的分数。

## 个性化与理论的含义

特征使用严格早于当前历史 cutoff 的数据；仿真反馈使用独立的 settled-step 时钟。30 天背景统计单独呈现，不与事件求和。静态 profile 的源表没有记录日期，沿用原项目背景信息，不能声称 profile 也通过了时点验证。特征是观察摘要，不是人格标签。

已有历史不包含完整历史曝光候选，因此不能据此计算“购买商品在当时曝光列表的价格分位”。点击数量是交互量的代理，不等于真实比较顺序或深度。品牌重复也不能直接认定品牌忠诚。数据不足时保留 unknown。

EKB、习惯性购买和广泛比较作为可被否定的候选解释，在理论开启时直接进入提示词；核心 skill 不规定固定的查证顺序。`--no-agentic-theory`（仿真）或 `--no-theory`（评估）关闭理论内容及其 reference，但保留完全相同的事实解释规则。工具返回、参考文本和商品文本均不能改变工具权限及输出协议。

## 评估与后续蒸馏

`evaluation/micro/customer_agentic.py` 只评估并保留全部轨迹。新增 `evaluation/trajectories.py` 在生成后评分，导出 train 的 SFT 与偏好数据；独立训练入口执行轨迹 SFT 和 DPO。Query 预测时隐藏当前真实 Query 和曝光；Response 条件预测时提供真实 Query 与原曝光，真实点击、购买标签只在推理结束后附加到输出文件。

自由仿真产生的新搜索结果与原日志曝光不同，不能直接用原日志点击集合计算同口径 F1。条件预测和端到端仿真应分别报告。`--baseline` 接受原 `behavior_prediction` 格式 JSONL，要求同样本 ID、同标签、每样本恰好一条；旧分批 Response 预测应先按请求合并，不能用局部批次冒充整请求基线。

此前 499 条 validation 来自官方 train 池的预留请求，已经参与多轮诊断。复用其 ID 时选择 `--split train`；现有 release 没有独立 validation 分区。原训练流程可能使用 test loss 选择 checkpoint，因此也不能把已有 test 描述为从未使用的独立检验集。

后续训练闭环应先冻结请求级划分和时间 cutoff，再从 train 采样轨迹，用 validation 决定筛选阈值，在预先保留的最终评估集上只做一次比较。保留失败与未入选轨迹，报告各人的接受率；不能从多次 test 采样中挑最好的一条冒充单次预测。个性化诊断应区分同人稳定性、不同人差异与随机采样变化，并比较带理论和不带理论两组。

Seller 已复用同一循环，开放只读经营与公开市场查询。竞争导向、需求导向、Hi-Lo、稳定报价、尾数偏好与参考价约束可作为策略候选。报价观察序列不是实际调价时间；价格相关性也不能证明商家看过竞品。缺少成本数据时不计算成本加成策略。分析子任务仅负责按主模型要求查数或压缩，不代替它决定查询顺序。

## 运行入口

Qwen3-32B 的模型目录为 `models/Qwen3-32B`。本机 vLLM 0.20.2 的服务命令示例：

```bash
source ../env.sh
CUDA_VISIBLE_DEVICES=1 ../vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3-32B --served-model-name Qwen3-32B \
  --host 127.0.0.1 --port 8001 --max-model-len 32768 \
  --max-num-seqs 4 --gpu-memory-utilization 0.86 \
  --enable-auto-tool-choice --tool-call-parser hermes --no-enable-log-requests
```

Qwen3 使用 Hermes 工具调用格式，见 [Qwen 官方说明](https://qwen.readthedocs.io/en/latest/framework/function_call.html)。服务仅监听本机；需要鉴权时通过 `AGENTIC_API_KEY`（或现有 `VLLM_API_KEY`）传入，代码和日志不保存凭据。

冻结 Lite 数据小样本评估：

```bash
python -m agent_simulation.evaluation.micro.customer_agentic \
  --dataset result/lite-5pct-quality-3k-20261112/release/sft \
  --split test --count 10 --stage both \
  --output result/customer-agentic-test-small \
  --url http://127.0.0.1:8001/v1 --model Qwen3-32B \
  --tokenizer models/Qwen3-32B
```

添加 `--no-theory` 使用另一全新输出目录做消融；固定 `--request-id` 或 `--candidates` 保证同请求。多样本探索用 `--samples-per-decision` 配合非零 `--temperature`；全部样本按 sample index 报告，不按答案挑选。

已有仿真命令追加 `--customer-agentic --agentic-url http://127.0.0.1:8001/v1` 即启用新路径。原 `--model`、角色适配器及反思模型仍控制已有 Seller/Memory 组件；Customer 自主查询使用单独的 `--agentic-model` 和 `--agentic-tokenizer`。未加开关时原 SFT 行为不变。

输出保留 `manifest.json`、`predictions.jsonl`、`model-calls.jsonl` 和 `report.json`；运行时完整轨迹还写入原有 `agent_runs`。记录模型可见消息、工具参数、工具结果、简短依据、动作 JSON、预算和失败原因。显式依据是可审查的生成假设，不声称是原始参与者的思考过程。

## 2026 年 11 月 19 日核验

在旧 SFT 第 1 轮已有预测的 Lite test 请求中，按 `SHA256(request_id:42)` 预选 10 条，不按真实结果筛选。分别评估 Query 和固定真实 Query、曝光的 Response，两组各 20 次决策均为有效输出。完整结果在 `result/customer-agentic-20261119/comparison.json`。

| 模式 | Query 语义相似度 | 点击命中与预测数 | 点击 micro F1 | 购买命中 | 自主工具调用 |
|---|---:|---|---:|---:|---:|
| 旧 9B SFT 第 1 轮 | 0.516 | 命中 1/12，预测 47 | 0.034 | 0/1 | 原固定上下文流程 |
| Qwen3-32B 带理论 | 0.332 | 命中 2/12，预测 20 | 0.125 | 0/1，预测为空 | 0 |
| Qwen3-32B 无理论 | 0.332 | 命中 2/12，预测 20 | 0.125 | 0/1，预测为空 | 0 |

两组汇总指标相同，但其中一条 Response 的预测集合不同。模型均未读取理论参考、查历史或记录行为假设。明确要求调用工具的合成协议测试能返回有效 `query_customer_history`，说明服务具备工具调用能力；本次行为评估尚未出现自主信息获取，不能证明个性化查询；它也没有检验 EKB 显式解释→轨迹 SFT→DPO 路线。单次 teacher 准确率不是是否蒸馏的硬门槛，应检查多轨迹质量、覆盖率和独立评估收益。这是 10 条请求的实现试跑，不是完整效果结论。

首次实现回归为 684 项：680 通过、1 跳过、3 个旧问题失败。随后的完整目录审计已经修复这三处问题（缓存测试磁盘隔离、上游排序代码来源、data 层依赖），并补充了数据、状态与评估边界修复。当前验证结果与完整修改清单见[目录审计](directory-review.md)和 `result/directory-review-20261119/validation.json`，不再把这三个问题列为待办。

最新逐件监督训练和两份续训 checkpoint、其基线数据及依赖均保留。清理归档与源码部署备份已于 2026-11-21 按用户要求删除；本机实验日志仍保留在 result，不进入 Git。

## 已实现的训练闭环

Customer 用 EKB、Seller 用定价理论作为可否定的解释候选，Qwen3-32B 在隐藏当前目标的冻结状态上生成多条可见依据、工具与最终动作轨迹，再由外部评分器使用真实观测筛选。先进行轨迹 SFT，再用同人物、同任务、同初始状态的 chosen/rejected 进行 DPO。teacher/student 是训练阶段；部署时每个角色由同一模型选择信息并决策。

生成、数据契约、assistant-only 训练、冻结 SFT reference、Seller 传统模型和运行限制统一记录于[轨迹训练](trajectory-training.md)。`enable_thinking=False` 关闭模型的隐藏 thinking 模式，但最终输出必须包含简短可见依据。新训练按每轮实际前缀编码，适配 Qwen3.5 对历史模板的重排；工具返回只参与条件上下文，不能计入输出 loss。

传统 Seller 对照已实现保持前价、滚动中位数/EWMA、保持历史竞品相对价/分位、两阶段 Logistic+Ridge 和 GBDT。保持相同截止时间和信息权限，并报告变价召回、持平误调价和分组误差。尚未开展完整效果对照或完整 9B 训练，不能据此宣布个性化或 DPO 收益已经成立。
