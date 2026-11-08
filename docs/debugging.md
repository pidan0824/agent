**用真实数据阅读与调试 E-commerce Policy Sandbox**

先完整跟踪一轮 Seller → Customer，再按问题深入模块。一次运行可以验证调用链和状态变化；跨主体泄漏、并发顺序、异常回滚则要配合局部测试。模型做出“合理”的动作，不能单独证明仿真行为真实。

**1．打开正确的工作副本**

用 VS Code 打开项目根目录的 [ecommerce-agent-simulation.code-workspace](../ecommerce-agent-simulation.code-workspace)，或直接打开当前 `ecommerce-agent-simulation` 文件夹。

本机当前工作目录：

```text
/Users/pidan/Documents/Playground/ecommerce_policy_optimization_20261105/ecommerce-agent-simulation
```

已经配置好 [.vscode/launch.json](../.vscode/launch.json)，直接使用项目的 `.venv-memory/bin/python` 和 `.runtime` 里的 DuckDB，不需要重新装环境。VS Code 需要 Python、Python Debugger 扩展，工作区已列为推荐扩展。先不要自动更新项目依赖。

**2．按这棵树阅读**

```text
ecommerce-agent-simulation/
├── ecommerce-agent-simulation.code-workspace   打开当前工作副本
├── .vscode/launch.json                F5 调试入口
├── agent_simulation/
│   ├── simulation/                    第一站：谁行动、阶段顺序
│   │   ├── run.py                     装配真实数据、模型、两个 Agent
│   │   ├── episode.py                 Tick 调度、价格生效和反馈结算
│   │   ├── replay.py                  回放与时间线
│   │   └── templates/                 HTML 回放模板
│   ├── agents/                        第二站：两个 Agent 如何决策
│   │   ├── seller.py            目标商品报价、整店等待
│   │   ├── customer.py          Query → Response → 合并选择
│   │   ├── base.py                    完整模型请求和原始回复
│   │   ├── decisions.py               单次最终动作校验
│   │   ├── concurrency.py             有上限的并发任务接续
│   │   ├── context/                   近期事实、店铺摘要、市场参照、目标投影
│   │   ├── prompts/                   最终发给模型的文本与表格编码
│   │   ├── actions.py                 共用动作格式与校验
│   │   ├── retrieval/                 历史事实、长期经验、竞品的筛选
│   │   └── memory/                    反思、修订、撤回与证据验证
│   ├── environment/                   第三站：动作如何改变真实运行状态
│   │   ├── context.py                 将快照、历史、检索、记忆组装成输入
│   │   ├── batching.py                SFT 与仿真共用的 Customer 固定切分
│   │   ├── market.py                  身份绑定、工具执行、阶段事务
│   │   ├── store.py                   SQLite 状态与 revision
│   │   ├── operations/                动作、查询、报表与工具协议
│   │   │   ├── actions.py             报价暂存及搜索、购买反馈结算
│   │   │   ├── queries.py / reporting.py  查询口径与经营统计
│   │   │   └── api.py                 程序接口定义与调用分派
│   │   └── memory.py                  记忆证据存储与提交
│   ├── platform/                      找商品：随机抽取、标题匹配、商品搜索与排序模型
│   ├── inference/                     第四站：模型调用和 model_policy.json 默认值
│   │   ├── client.py                  模型接口与外部模型适配
│   │   ├── local/service.py           共享模型子进程与请求队列
│   │   ├── local/qwen.py              本地 Qwen 生成
│   │   └── vllm.py                    可选远端推理接口
│   ├── datasets/                      真实标签 → SFT 样本
│   │   ├── samples/                  Customer、Seller 样本构造
│   │   ├── labels.py                 标签导出、样本读取和预览
│   │   ├── sampling.py               分层抽样与供应配对
│   │   ├── tokens.py                 Token 文件构建、读写和恢复
│   │   └── tokenization.py            编码、答案掩码和组批
│   ├── training/                      LoRA、loss、checkpoint 与导出
│   ├── evaluation/                    单步预测、价格响应、轨迹比较
│   └── tests/                         各模块的最小可复现约束
├── agent_simulation/data/    历史数据来源与统计口径
│   ├── catalog.py     运行时真实历史读取
│   ├── build/                 经营表、官方划分和目录构建
│   └── market/                     商品、经营统计、公开市场和缓存
├── data/kuaisearch/lite/source/catalog/     当前调试使用的只读真实数据
├── models/                           已下载的 Qwen 与 BGE
└── runs/debug/<本次名字>/             独立调试结果，不覆盖 SFT 分片
```

第一次沿 `run.py → episode.py → 两个 agent.py → environment` 走。只有看到具体问题，再进入检索、提示词、数据层；不必先阅读整个 Transformers 或所有测试。

**3．F5 跑一次真实双 Agent**

在“运行和调试”中选择 **01 真实数据 + Qwen：Seller → Customer 单步**，设置下文断点，再按 F5。输入新目录名，例如 `manual-001`；下一次换成 `manual-002`。程序拒绝覆盖已有输出。

默认案例固定如下：

| 配置 | 值与原因 |
|---|---|
| Seller | `319675`（云上酒坊），截止前三件注册商品，包含该顾客历史关注商品 |
| 商品 | `2118809` 水井坊；`2944136` 小糊涂仙；`5715815` 白云边15年 |
| Customer | `83860`，截止前有 2 次真实请求，历史中关注过白云边15年 |
| 历史截止 | `time_index < 850769`；边界本身不读入历史 |
| 仿真长度 | 1 Tick，逻辑步长 1；不是现实时间的 1 秒 |
| 模型 | 本地 Qwen2.5-3B-Instruct 底座；没有加载正式 SFT adapter |
| 历史检索 | 本地 BGE；小案例可能没有足够较早片段或长期记忆 |
| 搜索 | `random`，seed `42`，三件候选、展示三件；用于检查链路，不代表官方推荐策略 |
| 并发 | `1`，先看清顺序；之后可改成 `2` 对照 |
| 推理 | 本机 Apple GPU / MPS / float16，最多输出 1024 tokens；初次加载仍需要等待 |
| 请求期限 | 调试专用 86400 秒，包含排队和断点暂停时间；正式运行仍用原来的默认值 |

本机正常进程已确认 MPS 可用；受限执行环境可能只能看到 CPU。当前配置明确指定 MPS，不可用时直接报错。若换到没有 Apple GPU 的机器，需要明确改为 `--device cpu --dtype float32`（较慢）或当地可用的 CUDA 设备。

`price_raw` 保留数据原始数值，本例初始值为 `29900`、`15800`、`49900`，不要在调试时自行当作元或除以 100。该案例是链路检查用的小市场，不代表总体市场。

Seller 可以选择整店等待；Customer 可以选择等待或退出。因此一次真实运行不保证出现报价、搜索或购买。先核对实际输出，不修改回复来“补齐流程”。第二个调试配置提供确定的分支覆盖，但它使用测试模型和合成数据，与真实推理明确区分。

不使用 VS Code 时，可在项目根目录执行同一案例：

```bash
./agent_simulation/python.sh -m agent_simulation.simulation.run \
  --catalog data/kuaisearch/lite/source/catalog \
  --model models/Qwen2.5-3B-Instruct \
  --output runs/debug/manual-001 \
  --seller 319675 --customers 83860 --cutoff 850769 \
  --mode episode --ticks 1 --tick-size 1 \
  --top-k 3 --candidate-size 3 --search-backend random --seed 42 \
  --history-retrieval bge --max-concurrent-requests 1 \
  --seller-price-batch-size 1 --customer-response-batch-size 64 \
  --max-new-tokens 1024 --request-timeout 86400 --device mps --dtype float16
```

**4．第一遍只打这些断点**

在表中的函数内找到所述语句，点击行号左侧打断点。断点应放在赋值后的下一条可执行语句，才能看见刚生成的变量。F10 看本层，F11 进入自己的函数，Shift+F11 返回，F5 到下一断点。

| 顺序 | 文件与函数 | 停在哪里、看什么 |
|---|---|---|
| 1 | [run.py](../agent_simulation/simulation/run.py) `run` | `initial = runtime.snapshot()` 后；看 `args`、`initial['products']`、`initial['customers']`、历史截止；确认当前文件来自 `work` |
| 2 | [episode.py](../agent_simulation/simulation/episode.py) `_seller_proposals` | `baseline = self.runtime.freeze()` 后；看 `baseline.revision`、`baseline.state['active_prices']`，这是本轮共享的原价 |
| 3 | [seller.py](../agent_simulation/agents/seller.py) `_quote` | `prompt, tools = seller_prompt(batch)` 后；看 `batch`、目标商品、近期事实、竞品例子、`retrieved_context`、`long_term_memory`、`tools` |
| 4 | [base.py](../agent_simulation/agents/base.py) `_complete` | `request, count = ...` 后与 `response = await ...` 后分别停；看完整 `request['messages']`、`request['tools']`、token 数、原始 `response` |
| 5 | [operations/actions.py](../agent_simulation/environment/operations/actions.py) `stage` | 报价暂存后看 `self.decisions`、`self.original_prices`；此时市场尚未应用草案 |
| 6 | [episode.py](../agent_simulation/simulation/episode.py) `step` | `applied = ...apply_seller_quotes(...)` 后；看 `quotes`、`applied`，对照正式生效价格 |
| 7 | [customer.py](../agent_simulation/agents/customer.py) `choose_query` / `open_display` | Query 返回后看 `choice`；搜索后看 `run['displayed_list']`，尤其是 `request_id` 和展示价格 |
| 8 | [customer.py](../agent_simulation/agents/customer.py) `choose_response` | `base = ...inputs.customer(...)` 后；看 `base`、商品列表及 Query，随后看 `responses` 与合并的 `run['result']` |
| 9 | [episode.py](../agent_simulation/simulation/episode.py) `step` | `self.customer.settle_staged(...)` 前后对照；响应前没有本轮新购买反馈，结算后才更新历史与经营反馈 |
| 10 | [base.py](../agent_simulation/agents/base.py) `_update_long_term_memory` | 看触发条件；未达到门槛时没有生成调用是正常行为。需要追反思时进入 `environment/memory.py → agents/memory/reflection.py` |

所有模型调用都经过第 4 个断点，因此可以先只用 **1、4、6、9** 四个位置看全程，然后补其余断点。Seller 当前只提交最终报价；`decisions.run_workflow` 中的 `name`、`args` 可以解释动作如何验证和执行。

在 `_complete` 的局部作用域中，可添加条件断点 `run['role'] == 'seller'`，或 `stage == 'response'`。看长期记忆生成时要跟踪 `SummaryGenerator.__call__`，它不走行为决策的 `_complete`。

Debug Console 中可以展开已有变量；例如 Seller 断点看 `batch['requested_price_item_ids']`，Customer 响应断点看 `base['displayed_items']`。需要比较环境时，手动读取一次 `self.runtime.snapshot()`。不要把 `step()`、`run()`、`_execute()` 或 `inputs.customer()` 放进 Watch 自动求值：前几项推进状态，最后一项默认记录记忆证据。

**5．确实进入模型推理时怎么跟**

```text
父进程：BaseAgent._complete
  → inference/client.py：complete_once
  → inference/local/service.py：LocalInferenceService.acomplete
  → 请求队列
子进程：inference/local/service.py：_serve
  → inference/local/qwen.py：QwenLocal.complete_batch
  → self.model.generate(...)
  → 原始文本与结构化回复
父进程：_complete 收到 response → decisions.run_workflow
```

配置已经开启 `subProcess: true`，可以在 [qwen.py](../agent_simulation/inference/local/qwen.py) 的 `complete_batch` 内、`self.model.generate(...)` 前后打断点。在 VS Code 的“调用堆栈”中选择对应子进程。第一遍直接 F10 跨过 `generate`，先看 token 输入和输出；只有研究底层模型时，再把 `justMyCode` 改为 `false`。[VS Code 官方调试配置说明](https://code.visualstudio.com/docs/python/debugging#_subprocess)

不要为了调试再启动一份独立模型。当前两个行为 Agent 和记忆生成共用同一模型服务。模型加载阶段没有 Agent 输出；prefill 阶段也可能暂时没有输出日志，不能仅凭日志未新增就认定进程停了。

**6．运行后看哪些文件**

| 文件 | 核查用途 |
|---|---|
| `report.json` | `status`、`real_model_inference`、`model_id`、设备、配置、耗时；同时明确 `training=false`、`behavior_validated=false` |
| `initial_state.json` | 初始化商品、真实历史、原价、截止边界 |
| `final_state.json` | 正式生效价格、购买反馈、Customer 历史、Tick 状态；不能把报价草案当作已生效价格 |
| `model_calls.jsonl` | 每次实际模型输入/原始输出、token 数、耗时；与 Agent 动作对应查看 |
| `runtime.sqlite` | `agent_runs` 保存 Agent 完整轨迹；`calls` 保存工具调用；`ticks` 保存统一结算结果；`state` 是当前市场 |

回放不会重新推理，也不会训练：

```bash
./agent_simulation/python.sh -m agent_simulation.simulation.replay \
  runs/debug/manual-001/runtime.sqlite --html runs/debug/manual-001/timeline.html
```

可在 Debug Console 直接观察 `run['model_calls']`、`run['tools']`、`run['result']`。如需离线检查数据库，使用只读连接，例如 `sqlite3.connect('file:runs/debug/manual-001/runtime.sqlite?mode=ro', uri=True)`。

**7．看到问题后，沿哪条线深入**

| 观察到的问题 | 下一步阅读 | 要核查的事实 |
|---|---|---|
| 输入混入未来信息或其他店私有数据 | `environment/context.py → agents/context/ → data/catalog.py` | 历史截止、主体归属、公开/私有字段，以及实际传入模型的值 |
| 历史或竞品没有选对 | `agents/retrieval/history.py / ranking.py / competitors.py` | 候选池、目标 query、相关性与时间因素、最终保留证据；空结果可能只是小案例没有候选 |
| 长期记忆没有生成或内容不实 | `environment/memory.py → agents/memory/reflection.py / validation.py` | 是否到触发门槛、实际证据、生成/修订/撤回、失败审计 |
| 工具被调用得不对 | `agents/actions.py → agents/decisions.py → environment/operations/api.py` | 动作 JSON、程序接口参数与 actor 绑定；环境查询不向当前决策模型开放 |
| 别人提前看到了本轮反馈 | `simulation/episode.py → environment/market.py / actions.py` | 共享快照、草案、事务提交、输入时点 |
| 并发变快但结果变了 | `agents/concurrency.py → tests/simulation/test_episode.py` | 请求集合、主体顺序、最终状态；浮点推理差异与调度造成的状态泄漏要分开 |
| 仿真输入与 SFT 输入不一致 | `datasets/samples/ → agents/prompts/ → datasets/tokenization.py` | 目标事实、动作格式、监督边界；仿真控制动作和工具查询不等于都有 SFT 标签 |
| 样本所属 split 或 2% 抽样不对 | `datasets/labels.py / sampling.py → data/build/splits.py` | 官方 train/test 标记、固定 seed、分层配额；不在调试过程中重建标签 |
| loss 正常但轨迹不真实 | `evaluation/micro/behavior_prediction.py` 和 `evaluation/macro/trajectories.py` | 分别评价单步预测与连续轨迹 |

**8．用局部测试验证具体约束**

选择第二个 F5 配置 **02 局部回归：阶段、快照与并发（测试模型）**。这三个测试覆盖多轮结算、慢任务下的并发接续、旧报价拒绝；无需加载 Qwen 权重，也不触碰生产样本。

```bash
./agent_simulation/python.sh -m unittest -v \
  agent_simulation.tests.simulation.test_episode.EpisodeTests.test_phase_barriers_and_next_tick_context \
  agent_simulation.tests.agents.test_customer_parallel.CustomerParallelTests.test_next_batch_starts_while_first_batch_is_still_running \
  agent_simulation.tests.environment.test_state_revision.StateRevisionTests.test_staged_review_rejects_change_then_restore_before_state_read
```

第一轮的完成标准：能指出两类 Agent 的实际输入和输出、价格何时生效、反馈何时进入下一轮，并能从运行文件重建这个过程。随后再扩大店铺、顾客、Tick 和并发数，一次只改变一个维度。

**9．已经观察到的真实失败案例**

Seller `344929` 搭配 Customer `83860` 的运行保存在 [real_agents_20261106_mps](../../work/runs/debug/real_agents_20261106_mps/report.json)。这次是真实 Qwen 推理，共 8 次调用：Seller 对两件商品分别查市场两次、提交维持原价；Customer 生成 Query 后，在 Response 中选择了 `5715815`，而本次只展示 `[3590059, 5715820]`。

错误在 `agents/actions.py:validate_response` 被拒绝，运行标为 `failed`，没有写入 Customer 反馈，待结算展示列表被保留。Seller 阶段已经提交，不会因为随后 Customer 失败而回滚整轮。这个案例可以用来检查历史商品与当前列表的区分、校验位置和失败审计。

该失败版本只在应用层检查展示商品归属。现在 `decisions.action_output` 将**当前 Response 批次**的 ID 写入两个数组的 `enum`；空列表使用固定空响应，最终 `validate_response` 仍保留。运行时、预览和离线行为评价共用该约束。没有替换非法 ID、静默删除选择或失败重试。

[同提示词真实复验](../../work/runs/debug/agent_boundaries_20261106_v2/response_replay.json)只改变生成 schema：原输出为历史商品 `5715815`，新输出为当前展示的 `5715820`。这验证选择边界，不证明点击偏好预测正确。

**10．已完成的真实运行，可先看结果再 F5**

默认案例的实跑结果位于 [real_agents_20261106_shopping/report.json](../../work/runs/debug/real_agents_20261106_shopping/report.json)，[初始状态](../../work/runs/debug/real_agents_20261106_shopping/initial_state.json)、[最终状态](../../work/runs/debug/real_agents_20261106_shopping/final_state.json)和[模型调用](../../work/runs/debug/real_agents_20261106_shopping/model_calls.jsonl)均已保存。本机 MPS 实跑约 155 秒，12 次模型调用，`status=complete`、`real_model_inference=true`。

- Seller 的长期记忆生成返回 `no_update`，已处理 10 条证据，没有新增经验。
- Seller 对三件商品分别进行了两次市场查询，并提交维持原价的报价。三次模型报价在环境里统一提交；`price_actions=0` 表示没有改变价格，不是 Seller 未执行。重复查询是本次底座模型的真实表现。
- Customer 搜索“白云边15年陈酿42度兼香型白酒500ml礼盒装纯粮固态法酿造礼品”，点击了展示商品 `5715815`，没有购买。
- 结算产生 3 条商品展示反馈，只有一件被点击，购买为零；`pending={}`，真实初始化历史未被覆盖。

可以直接打开[已导出的时间线](../../work/runs/debug/real_agents_20261106_shopping/timeline.html)，无需再次加载模型。

另外，[real_agents_20261106_complete/report.json](../../work/runs/debug/real_agents_20261106_complete/report.json) 是 Seller `344929`、Customer `73109` 的正常等待案例：Seller 维持原价，Customer 返回 `no_action`，没有调用搜索或 Response。第 9 节则是另一个明确失败的案例。案例是为了观察不同分支选择的，不能把这些结果当作总体成功率。

以上是修复前保存的案例，原始文件保留不变。F5 配置使用更长的调试请求期限，其余业务参数与默认案例实跑一致。不同设备和运行环境下，模型输出仍应以新日志为准。

**11．选择边界、工具、记忆和检索的历史核查**

本节保留关闭主动查询之前的实测记录。当前 Seller 已关闭主动查询，查询预算和重复缓存已删除；第 12 节说明最新策略。

以下使用本地 **Qwen2.5-3B-Instruct 底座 + BGE**，没有加载 SFT adapter。功能跑通与行为质量评价分开记录。

| 核查 | 实际证据与限制 |
|---|---|
| 重复市场查询 | 首次结果完整进入下一次 `market_queries`。模型仍可能重复。`agents/queries.py` 在一个目标决策内按工具名和参数复用只读结果；重复仍耗查询预算，不跨主体或快照共享，也不缓存报价等写动作。`read_queries` 记录所有尝试，`tools` 记录实际执行。缓存不减少重复那次模型生成。 |
| 非法查询参数 | 三轮案例的第 3 轮产生 `start=before=840769`，被后台拒绝；没有部分应用 Seller 报价。工具描述已明确 `0 <= start < before <= history_cutoff_exclusive`，保留严格校验。普通 JSON schema 无法保证两个自由整数之间的大小关系，描述改进不等于彻底消除此类错误。 |
| Customer 较早历史 | Customer `20762` 有 65 条截止前请求。初始输入近期 10 条、较早候选 55 条、选入 5 条；后续候选变为 56、57 条。重算排序与实际输入一致，检查了主体、时间边界和分数。 |
| 时间衰减 | 历史检索采用相关性融合后乘新鲜度因子；本例旧记录因子仅约 `0.900000–0.900025`，几乎无法区分新旧。参数未校准；本次未改检索策略或重建 SFT。 |
| 连续反馈 | 前两轮完成，第 2 轮购买 `329710`。第三轮 Seller 的真实输入包含该购买，Customer 的下一轮输入包含先前结算；这是反馈传递证据，尚不能证明模型的决策机制真实。 |
| 长期记忆 | 普通真实运行仍为 `no_update`。独立功能探针使用真实历史和真实模型，在明确分支指令下完成 Add → 检索进入 Query → Revise / Retract；修订、撤回针对隔离副本中主动注入的错误记忆，原运行不变。旧版本不再被检索。这个结果证明分支机制，不证明模型会自主、正确地选择操作。 |

主要记录：[多轮原始运行](../../work/runs/debug/agent_boundaries_20261106_v2/report.json)、[候选与排序审计](../../work/runs/debug/agent_boundaries_20261106_v2/retrieval_audit.json)、[记忆探针说明](../../work/runs/debug/agent_memory_probe_20261106/report.json)、[记忆进入真实模型请求](../../work/runs/debug/agent_memory_probe_20261106/memory_in_decision.json)。探针引用 ID 合法，但 Add 和 Retract 引用了过多记录；逐条证据是否支持文本仍需单独评价。

修改工具说明后，仅从第 2 轮完成快照分叉，[重跑第 3 轮](../../work/runs/debug/agent_boundaries_tick2_20261106/report.json)成功：两家 Seller 查询窗口合法、维持原价，Customer 搜索后没有新增点击或购买，`pending={}`，Tick 达到 3。这不是同一版本从头连续三轮的成功率测试；前两轮未重跑，原始失败运行也未覆盖。

复现脚本位于项目相邻的 `audit/agent_boundaries_20261106/`。正式 2% SFT 样本和既有训练结果未重跑。

**12．历史策略：直接决策与精简引用**

本节为改成事实摘要之前的实测，当前长期记忆见第 13 节。

Seller 只开放 `submit_prices` 最终报价；Customer 仍输出最终 JSON。两者均没有主动查询，市场摘要、最多 3 件竞品例子、历史检索和长期记忆由程序自动提供。查询后端仍可用于上下文构造和人工诊断。

删除了查询预算、重复结果缓存、查询后再次生成的递归路径，以及针对这些路径的过时测试。保留最终动作、主体和时间边界、版本及引用合法性检查；没有增加每条样本的审查调用。

记忆提示词要求聚焦单条观察或修正，用最多 3 条充分证据支持一句短结论；购买结论必须引用该商品实际购买的记录，修订保留旧结论时也要保留支持证据。生成 schema 与最终校验限制最多 3 条引用；没有截断或自动修复模型引用。合法 ID 仍不等于文本被证据支持，需要对真实输出单独评价。

后续真实复验：[直接报价单轮](../../work/runs/debug/passive_actions_20261106/report.json)中两家 Seller 各生成一次，原同快照案例合计为 6 次 Seller 调用，现在为 2 次。

引用优化同时发现并修复了解码缓存漏放 `.",` 这类跨字符串结束边界 token 的问题：保留缓存快速路径，额外对这些 token 做完整语法检查，不扩大合法输出范围。强制先输出证据的实验未保留，最终仍使用原有内容／引用协议。

[最终引用探针](../../work/runs/debug/citation_decoder_fixed_20261106/report.json)的三条输出均正常结束，引用数分别从 10、3、10 变为 3、3、3。但语义核查没有通过：Add 结论涉及两家店，三个引用只支持其中一家；预期修订和撤回的两条探针均生成 Add，没有退休旧版本。因此，这一轮证明了生成边界、条数约束和直接报价链路；未证明自动反思的引用完整性或操作选择正确。探针是明确指令下的功能检查，不是自然反思成功率。

当前保留必要的边界、提交和记忆版本检查；删除查询缓存／预算路径及其过时测试，并删除依赖提示词中是否出现特定普通单词的脆弱断言。最终全量回归 509 项通过。SFT 数据、输入契约及原始项目保持不变。

**13．历史实验：选择证据，保存事实摘要**

该收缩方案已按用户最新要求撤回；当前角色化总结见第 14 节。

按用户选择，当前先采用可靠事实摘要，后续再补自由反思。每个触发批次仍只调用一次模型。模型只能输出 action、evidence_refs 和必要的 memory_refs，不允许输出 content。E 引用按展示顺序编号，新增购买事实优先展示，全部输入记录保留；购买证据另外按商品和店铺列出索引。

- add/revise：选择 1–3 条事实，由程序从所选原始记录生成正文，保留商品、店铺、价格、窗口和时钟；不从别的记录拼接结论。修订必须指定 M 引用，旧版本退出检索。
- retract：只指定旧记忆。撤回是管理动作，不附伪装成反证的正文引用；当前提供的证据批次记为 retirement_context_source_ids，与 retirement_source_ids 区分。
- no_update：不写入记忆，推进已读游标。

[protocol.py](../agent_simulation/agents/memory/protocol.py)负责局部引用、操作 schema 和事实正文；原有存储格式及版本退休机制复用。新事实条目标记 content_format=cited_observations_v1，运行时 evidence_kind=selected_factual_observations。自由反思暂不入库，原有历史条目不批量改写。事实正文来自引用记录，但选择哪些事实值得记、何时忘记仍是模型决策。

删除了旧自由总结的生成 schema、空数组解码特例及过时的提示词字面断言。默认事实路径跳过自由文本的货币单位／复述相似度检查，保留引用范围、新事实、目标版本及事务检查；没有增加第二个审查模型或自动修复重试。

真实复验使用同一批历史失败案例，Qwen 底座在一次生成中选择 Add、Revise、Retract。新增和修订正文均由程序从所选记录生成。撤回采用最终管理协议，只返回 action 与 M 引用，不提交反证引用。[实际操作及提交检查](../../work/runs/debug/factual_memory_final_20261106/commit_checks.json)将真实输出送入生产更新逻辑，保存到新 SQLite 后关闭重开，确认新增可检索、修订后旧版本退出、撤回后无活跃条目；这是受控案例功能验证，尚不是自然长期记忆决策质量评估。

本次全量回归 502 项通过；此前过时的自由总结 schema、空数组专用解析器和提示词字面断言已清理。正式 SFT 输入指纹、manifest、contract 和受保护的 282 个文件保持一致，没有重新构造或训练。

**14．当前长期记忆：角色化总结与工作假设**

核对本地 PolicySim 官方仓库 commit 045446b7dd5120437233e75da0e997d68d7c267a：`src/prompt/base_prompt.py:642` 将画像、当前信息及相关记忆交给模型生成简短长期总结；`src/agent/base_agent.py` 在周期触发后检索最多 5 条相关记忆再总结。当前项目采用这一“角色＋体验＋相关记忆→高层总结”的形式，但不照搬 SNS 角色或重试循环。

Customer 总结消费偏好、重复兴趣、搜索比较习惯、购买模式及有价格证据时的价格反应。Seller 总结同商品价格／经营经验、竞品定位及关于平台曝光分配的暂定假设。提示词区分观察、解释和未知信息；不把平台假设写成真实推荐算法，不臆造成本、库存、其他店的私有流量或消费者满意度。

模型 add/revise 的 content 直接作为记忆正文，不再由程序改写成事实副本；证据单独通过 E 引用映射回 canonical source_ids，沿原有历史与证据库追溯。显式 action、M 版本目标、单次生成和必要提交检查保留。新条目标记 `role_experience_summary_v1`，决策输入仍归为 `model_summary_or_hypothesis`。旧事实实验条目不批量改写。

真实 Qwen 底座各尝试一次，已生成 Customer 偏好总结和 Seller 推流假设，记录在 `runs/debug/role_memory_restore_20261106/`；这证明形式恢复，不证明总结准确。Customer 有满意度／店铺归因过度推断；Seller 原始输出出现 `$12400` 和无依据的多次改价描述，前者会被既有单位检查拒绝。没有因此重试或再切回纯事实。提示词已针对这些歧义补充要求，未把未微调模型的总结错误当成可证明的知识。

**SFT 独立运行。** 正式版本沿用官方 train/test、seed=42 候选和真实 Query／点击购买／报价标签，不加入记忆或反思监督。新入口为 `data/kuaisearch/lite/sft`，候选为 `data/kuaisearch/lite/candidates`。本轮只做有界验证，尚未启动全量构造或训练；旧版本的进程、冻结源码及分片状态应在其原目录查询，不能混用。
