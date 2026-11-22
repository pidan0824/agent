# Customer 数据构造流程

当前主流程用于离线构造训练数据：作者式候选／干扰项 → 强弱结构化 EKB 轨迹 → 同一行为预测器的选项概率与熵 → 场景／偏好筛选 → SFT、DPO 导出。候选场景在模型调用前冻结。Query、原始多点击／购买记录、空点击／空购买、fixed／autonomous 两种信息获取模式都保留；Response 按真实正例拆成单正例短列表。

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

## 1. 仅准备场景

```bash
dataset=result/lite-5pct-quality-3k-20261112/release/sft
./agent_simulation/python.sh -m agent_simulation.datasets.customer_scenes \
  --dataset "$dataset" --output result/customer-scenes \
  --split train --count 100 --min-items 2 --max-items 12
```

本阶段不调用生成模型。只有启用 content 路径才加载编码器；单独 random 或 collaborative 不需要 BGE。内容编码默认使用 GPU（`--bge-device cuda`）；可指定 `cuda:1` 或通过 `CUDA_VISIBLE_DEVICES=1` 选择物理 GPU 1。CUDA 不可用时明确报错，不静默退回 CPU。默认三种评分均启用；`--methods` 可显式选择。可通过 `--max-input-tokens` 和 `--tokenizer` 按实际 Query／Response 初始提示、schema、工具定义作长度过滤。输出 `scenes.jsonl`、`skipped.jsonl`、`manifest.json`，保留每种跳过原因及原始标签。场景文件使用 customer-single-positive-scenes-v2：每个源请求保存一份 Query 与若干 Response 子场景。旧 v1 多正例场景须重新准备，不能静默按新协议读取。Query 输入始终不含当前真实 Query 或候选列表。
Query 每个原请求只生成一组（每模型／每种信息模式各按 samples 采样），不随正例数重复。Response 候选不足、历史不足或超长仅跳过对应子场景；Query 独立检查输入长度并保留。count 和 request-id 均按原请求计数，manifest 分别记录原请求、Query 场景和 Response 子场景数量。

## 2. 生成 EKB 和信息获取轨迹

```bash
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_curation \
  --dataset "$dataset" --scenes result/customer-scenes \
  --output result/customer-teacher --split train \
  --modes fixed --samples 2
```

默认仅使用 sampled（采样候选）场景和固定信息模式，不自动展开完整比较矩阵。`--count 0` 使用所有已准备场景；`--request-id` 可固定请求。`--modes fixed` 使用固定事实，`--modes autonomous` 自主选择只读工具，也可显式同时生成两种模式。强模型默认 Qwen3-32B，弱解释模型与默认公共行为预测器均改为 Qwen3.5-9B；弱模型服务和 tokenizer 通过 `--weak-url/model/tokenizer` 指定。公共行为预测器可使用独立的 `--predictor-url/model/tokenizer`，不必与弱解释生成器绑定。

Customer 生成入口中的 BGE 检索及 Query 相似度评分也默认使用 `--bge-device cuda`。设备参数会进入运行 manifest；这不代表当前已启动 GPU 任务。

EKB 直接保留作者详细解释的要求，不额外限制 100 字、单句或字段文本长度。作者示例编码器为 `gte-Qwen2-1.5B-instruct`，本项目当前配置的内容编码器是中文 BGE。

EKB 保留作者 Stimulus → Knowledge → Evaluation、因素和 Evaluation Style，按项目 JSON 输出。不能虚构心理状态；多选／空集覆盖原模板强制单选约束。原始模型动作、解释、工具返回、四状态选项概率、argmax 和采样行为分别保存。此处“单正例”指列表和监督标签；动作仍使用点击／购买两个集合，并保留空集合能力，公共预测器仍使用四状态概率，不在本次切换为作者的单一字母分类器。

## 3. 独立筛选和导出

```bash
./agent_simulation/python.sh -m agent_simulation.evaluation.customer_filtering \
  --input result/customer-teacher/records.jsonl \
  --output result/customer-training-data
```

可重用已保存轨迹重新筛选，本阶段不调用任何模型。生成结束时也会按相同函数导出一次。

生成与离线导出使用同一套筛选函数；离线重筛只读取保存的动作、标签、分数与概率，不重新调用模型。默认 Response 配对协议为 `--pair-protocol logloss`，可显式选择 `sampled_match`。作者 vendor 函数保持原样，新的动作约束、Log Loss 配对和统计属于项目适配。

### SFT：分别检查点击、购买与空集合

SFT 独立收集 strong 的原始完整轨迹，不要求预测器成功、弱模型正确或存在 rejected；不改写 EKB、最终动作、工具调用及返回内容。

- Query 保留原始生成文本，按真实 Query 相似度筛选，默认 `--query-min 0.7`。
- 有点击时单独检查点击 F1，有购买时单独检查购买 F1。默认两个门槛均为 0.8；`--click-min`、`--purchase-min` 可分别设置，`--response-min` 是两者的共同默认值。
- 真实点击为空时必须预测空点击；真实购买为空时必须预测空购买。即使另一项得分很高，也不能抵消虚构点击／购买。
- 多点击、多购买、购买但没有点击记录均按两个原始集合处理；不强制单选，也不强制购买必须出现在点击集合中。
- 离线未覆盖门槛时采用轨迹保存的门槛；旧记录无门槛则使用 Query 0.7、点击／购买各 0.8。旧 Response 只有 accepted 标记而缺少真实标签时，不静默沿用旧平均分判定，会报告缺少标签。
- 行为分层按当前场景标签采用无／单／多点击 × 有／无购买，分别报告保留率；单正例协议下源请求的多点击类别另保存在 source_response_target 中，不能把拆分后的单点击计数解释为源用户行为分布；不以分层为由伪造、补齐样本，也不默认限制各组配额。

### Response DPO：解释评分与原始动作方向一致

公共预测器只评分。四状态 A/B/C/D 是逐商品的均无／仅点击／仅购买／两者均有；保留选项概率、熵、argmax 和采样行为作为诊断。逐商品 Log Loss 是真实状态概率的负对数均值，沿用作者 `get_logloss` 的 **log2，即 bits/item**，不是完整点击集合的联合概率，也不宣称已经校准。

默认 `logloss` 协议在相同初始状态、prompt、工具权限、候选与目标内比较：

1. chosen 的原始动作通过上述分层质量门槛。
2. chosen 原始动作的平均 F1 比 rejected 高至少 `--action-min-gap 0.05`，且点击、购买两项都不劣于 rejected。
3. chosen 的真实标签平均 Log Loss 比 rejected 低至少 `--logloss-min-gap 0.05` bits/item。
4. 任一信号方向冲突、差距不足或没有足够概率记录时，不构造该偏好对；合格 SFT 仍独立保留。
5. chosen 可以来自 strong 或 weak；按质量选择，模型来源保留在数据中。每个场景最多一对，优先 Log Loss 差距较大的合格对。

因此，当前完整 Agent 协议不把“动作完全相同、只有解释分数不同”的轨迹变成偏好对。这类解释单独优化不在本次实现范围内。

`sampled_match` 保留作者 `behavior_sampling` 的选项采样匹配选择，但增加原始动作保护：chosen 的采样动作和原始动作均完整匹配真实集合，rejected 两者均不完整匹配，并通过原始动作分差约束。此协议不使用 Log Loss 差距门槛。**两种协议导出的 chosen/rejected 都逐条保留原始生成轨迹，绝不把预测器采样动作拼接回去。**

强弱模型相同 sample ID 的概率用于作者不确定性统计。场景排序继续采用 `--filter-column`（默认强弱熵差），并用作者 `data_filtering` 按数据域、候选数量、行为分层选择；`--max-pairs` 限制 Response 对数。缺少成对强弱样本但有合格概率分支的 Log Loss 场景仍可配对，其不确定性排序值为 0 并标记不可用，不伪造强弱差异。`--sampling-column entropy|logloss` 只控制 `sampled_match` 的作者分支抽样。

Query DPO 保留同场景真实 Query 匹配分差规则（同样由 `--action-min-gap` 控制），不捏造自由文本的选项熵。DPO 两分支保留各自实际工具返回；现有训练器只对 assistant 生成内容计算损失，工具返回不参与生成概率。

### 导出与统计

- `sft.jsonl`、`preferences.jsonl`：接现有[轨迹 SFT→DPO 训练入口](trajectory-training.md)。
- `uncertainty-metrics.jsonl`：作者强弱概率／熵统计；teacher_hit/student_hit 仍指预测器采样完整匹配率。
- `filter-decisions.jsonl`：每条轨迹的实际门槛、质量分、是否进入 SFT、剔除原因、概率可用性与错误。
- `pair-decisions.jsonl`：每个场景是否配对，或因缺概率、无合格 chosen、动作分差／方向冲突、Log Loss 分差／方向冲突、配额等原因未配对。
- `filter-funnel.json`：原始数、有效数、质量通过数、Query／Response SFT 与偏好对数、各分层保留率、剔除原因计数。行级原因允许重叠，配对原因按场景计数，二者不相加。`test` 来源不进入训练导出。

数据没有可用的真实曝光位置，数组顺序不代表原始展示排名。以上匹配与筛选均按商品 ID 集合，不使用位置特征或位置偏差假设。候选是事先冻结的，同一场景各模型收到相同排列；短列表中的排列是构造顺序，不声称恢复了真实曝光顺序。

之前的 2026-11-21 两请求效果试跑属于旧 BGE 截短协议，结果仍保留在原目录；不作为本次作者式候选流程的效果结论。本次改动只做固定小数据的流程检查，不启动性能比较或训练。

固定信息模式不开放查询工具，直接提供历史与候选后生成 EKB 和动作；自主查询模式是本项目的可选扩展，须显式设置 `--modes autonomous`。搜索 Query 的生成与查询工具调用是两个独立概念，Query 阶段始终保留。

## 单正例拆分的解释边界

例如原请求点击 A、B，购买 B，生成两条 Response：A + 干扰项（排除 B），标签为点击 A、无购买；B + 干扰项（排除 A），标签为点击 B、购买 B。Query 只保留一组。无点击且无购买的请求生成一张真实曝光负例列表，两个动作集合均为空。

拆分能复用更多单交互训练场景，但这些子场景共享同一请求，不是独立用户。划分继承原请求 split；SFT 和 DPO 记录保留 source_request_id、scene_id、source_response_target，用于追溯和请求级分组。不能对拆分行重新随机划分训练／测试，也不能把子列表预测的并集直接当作原完整曝光条件下的多点击评估结果。当前不额外按原请求做损失重加权，因此多正例请求贡献的 Response 样本较多。

候选仍是从真实曝光抽出的结果条件化短列表；干扰项只表示该次请求未点击且未购买的商品，不代表用户永久不喜欢。正例与无点击请求均只使用各自的真实曝光池。没有真实曝光顺序字段，不推断位置。内容编码保持 GPU 默认，原始数据及旧实验结果不改写。


## 曝光池与作者协议的区别

当前默认直接在曝光池内做三路采样，不按 Query 或类目硬筛；相关阈值 CLI 参数已移除。旧已生成场景不会原地修改，必须重新准备才能采用当前规则；manifest 和每个子场景 scope 的 query_category_filter=false 区分本次规则。仅实际用到的曝光商品编码，避免编码整个共享 catalog。

作者 MIND 的 preprocessing/DataProcessor_MIND.py 保留原始 impression_list，但只保留恰好一个点击、列表不超过 10、历史至少 5 的请求。scene_sampling.py 有条件复用 MIND 的真实曝光列表；该分支还存在 sampled_dataframe / sample_dataframe 变量不一致，不能说其所有默认路径均正确复用了曝光。其通用路径仍调用三路采样。我们不复制 MIND 的单点击请求剔除策略：多正例拆分、不点击和 Query 均保留。

作者 Amazon 预处理把评分／评论记录映射为单一 item_id 交互，模板称为购买历史，但没有独立的点击、购买双标签或转化漏斗。我们继续保留 clicked_item_ids / purchased_item_ids 两套真实标签，且不由点击推断购买。曝光未点击只是该请求下未点击，并不等于永久不喜欢；从完整曝光抽出的短列表仍是结果条件化训练构造，不等价于用户真的只看到了这几件商品。
