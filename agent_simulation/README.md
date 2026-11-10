# Agent 代码与运行

Customer、Seller 与环境共用一个市场状态。行为和长期记忆设计见[架构说明](../docs/architecture.md)。

## 当前数据入口

标签入口为 `data/kuaisearch/lite/sft`，配对后的 2% 候选在 `data/kuaisearch/lite/candidates`。候选中的展示商品有同一 split 内、请求当时或更早的 Seller 报价样本。候选不等于已编码的训练样本，最终容量判断在 Token 构造阶段执行。

Customer 和 Seller 共用近期 10 条观察、较早历史检索与最终输入编码规则。Customer 的流式历史按时间增量更新；Seller 按同店同时间点构造共享快照，再投影到目标商品。训练流不复制完整历史 ID 列表，Seller 样本不附加全店覆盖审计报告。

竞品限定同三级类目、其他店铺及最近 10,000 个观察索引。先按标题、品牌和新近程度粗选，再做 BGE 语义排序，候选上限为64件，由模型选择可比商品，数量不限于三件；近期不足不补旧报价。原始价格作为决策证据，不参与竞品身份选择。

模型配置在进程启动后固定读取；修改配置后重启构造任务。BGE 自动优先选择 CUDA、其次 MPS，否则 CPU。已有文本向量复用，首次模型加载和新文本编码仍是主要额外成本。实际构造速度应按本次数据、设备和缓存状态测量，不沿用旧版本的计时数字。

## 代码阅读顺序

可直接使用[真实数据调试指南](../docs/debugging.md)和[现成的 F5 配置](../.vscode/launch.json)，先观察输入、输出和状态提交，再按下面顺序读代码。

1. [simulation/run.py](simulation/run.py) 装配数据、模型客户端、Agent 和 Episode。
2. [simulation/episode.py](simulation/episode.py) 推进 Tick，协调价格生效与反馈结算。
3. [agents/customer.py](agents/customer.py) 生成 Query、响应商品列表并合并分批选择。
4. [agents/seller.py](agents/seller.py) 组织店铺决策，收集各批商品的定价草案。
5. [environment/context.py](environment/context.py) 从已结算状态装配上下文。

| 模块 | 职责 |
|---|---|
| [agents/](README.md#agent-core) | 两类 Agent、共用输入、Memory 和检索 |
| [datasets/](README.md#sft-data) | 监督数据构造、读取与 Token 化 |
| `simulation/` | 仿真启动、Tick 推进、Agent 调度与记录回放 |
| `environment/` | 共享状态、权限、价格生效及反馈结算 |
| [platform/](README.md#platform) | 按搜索词找商品，决定展示哪些商品及顺序 |
| [inference/](README.md#inference) | 模型调用、请求构造和约束解码 |
| [training/](README.md#training) | 模型参数训练与检查点 |
| [evaluation/](README.md#evaluation) | 行为预测、轨迹与分布评价 |
| [tests/](README.md#tests) | 程序测试 |

[共享规则、价格约定和 SFT 决策指令](agents/prompts/instructions.py)、角色 Prompt（[Customer](agents/prompts/customer.py)、[Seller](agents/prompts/seller.py)）和[记忆反思指令](agents/memory/prompts.py)按用途分开。指令和长期总结要求英文，Query 输出中文，原始商品与搜索文本保留原文。

## Environment 与 Platform 的职责

`agents/` 中，`base.py` 保存两类 Agent 的共用调用与记忆更新；`context/` 选择可见事实；`decisions.py` 统一最终动作调用；`customer.py` 与 `seller.py` 实现各自的决策流程。

`environment/` 根目录保留环境入口 `market.py`、状态事务 `store.py`、记忆持久化 `memory.py`、上下文装配 `context.py` 和纯分批函数 `batching.py`。业务操作集中在 `operations/`：`actions.py` 执行搜索、报价暂存与行为结算；`queries.py` 查询授权历史；`reporting.py` 计算经营报表；`api.py` 定义程序调用的参数并分派操作。环境查询接口仍可由程序直接调用。运行库只接受当前 v5 格式，不再兼容旧 v4。

决策输入沿 `Agent → context → agents/context、retrieval、memory` 构建；动作沿 `Agent/Episode → market → operations/api → actions/queries/reporting` 执行。`market` 统一授权、动作记录和提交，`store` 提供事务及版本机制。长期记忆的生成算法在 `agents/memory`，环境的 `memory.py` 负责证据存取和提交。

校验按边界保留：初始化检查种子历史的归属、截止时间和重复记录，并保存时间顺序；查询检查本次请求范围；动作检查商品可见性、所有权和当前状态；提交检查版本及原子性。Customer 历史查询直接读取已验证的本用户种子，不再包装身份后重复校验。直接查询 API 保留，但不向当前决策模型开放。

`platform/search.py` 根据运行参数决定怎么找商品，可随机抽取、按标题匹配，或调用模型。`platform/pipeline.py` 统一加载模型并串起召回、排序、取 Top-K；`platform/recall.py` 找候选，`platform/ranking/` 给候选打分。搜索结果的相关性评估位于 `evaluation/micro/relevance.py`。

## 数据层的职责

`data/` 负责事实，供 SFT、仿真和评估共同读取。`datasets/` 负责把事实变成训练样本。实际 Parquet、配置和清单保存在项目根目录的 `data/kuaisearch/`。

| 位置 | 职责 |
|---|---|
| `data/build/` | 从原始日志生成经营表、官方划分和 Catalog |
| `data/catalog.py` | 统一读取请求、商品、历史和公开市场信息 |
| `data/market/` | 商品候选、市场报价查询与统计；现有市场库的构建和读取也在 `quotes.py` |
| `data/history_records.py` | 真实与仿真历史的整理、近期记录和增量价格摘要 |
| `data/common.py` | 配置、DuckDB、文件读写 |
| `datasets/samples/` | Customer 和 Seller 的单条及增量样本构造 |
| `datasets/labels.py` | 标签导出、样本读取和预览 |
| `datasets/sampling.py` | 分层抽样与 Customer/Seller 供应配对 |
| `datasets/tokens.py` | Token 文件构建、读写和断点恢复 |
| `datasets/tokenization.py` | 模型编码、答案掩码和组批 |

价格冲突和缺失状态在经营表构建时处理；官方划分和更早有效价格条件在划分构建时检查。Catalog 保留跨来源的 Recall/Ranking 对照，但不再重新聚合经营表和重算 Seller 监督索引。标签导出检查实际输出，不再扫描全量历史验证旧 Catalog，也不重新计算所有源文件哈希。需要查文件完整性时使用现有的 `--verify`，不增加逐样本审计。

`build/` 按数据生成步骤分成三份：`seller_tables.py` 从 Ranking 汇总经营表，`splits.py` 保存官方划分和报价监督点，`catalog.py` 整理统一的事实读取目录。只在准备数据时运行。

`market/` 按读取内容分成三份：`products.py` 处理商品和公开候选，`quotes.py` 处理近期报价、类目快照和价格参照，`statistics.py` 计算经营指标与价格分布。SFT 与仿真通过这些共用函数读取数据，不代表模型在主动调用工具。

构建事实表使用 `python.sh -m agent_simulation.data.build.<模块>`；市场库入口为 `python.sh -m agent_simulation.data.market.quotes --dataset ... --output ...`。

## 模型接口与依赖

项目命令统一使用 `agent_simulation/python.sh`。它选择 `.venv-memory` 的 Python 并配置包路径；可用 `SANDBOX_PYTHON` 指定其他解释器。`agent_simulation/__init__.py` 是 Python 包入口，提供 `PACKAGE_ROOT` 和 `PROJECT_ROOT` 两个稳定路径常量。

[client.py](inference/client.py) 定义模型调用接口及 PydanticAI 适配，[requests.py](inference/requests.py) 处理官方模板和请求容量，[Qwen 后端](inference/local/qwen.py) 提供本地 Transformers 执行。

本地 Qwen 仿真、行为评估和 LoRA 训练默认用 `--device auto`，依次选择 CUDA、MPS、CPU。显式指定的设备不可用时会报错。KuaiSearch 资产清单也默认使用 `device: auto`，DPR 查询编码与候选点积分块使用所选设备，排序模型使用同一设备。

仿真和 `evaluation.micro.behavior_prediction` 均支持 `--inference-backend vllm`。默认仍是 Transformers。vLLM 客户端使用本地匹配 tokenizer 生成原始 chat-template token，经 `/v1/completions` 发送；服务端负责并发调度和前缀缓存。Customer 决策、Seller 报价／等待、反思的 JSON 通过 `structured_outputs` 约束，失败直接记录并抛出，不重试、不切回本地模型。两类后端共享决策输出 schema。

在 GPU 机器上启动支持 `structured_outputs.json` 的 vLLM 服务。若使用训练导出的两个 Agent LoRA，在服务端分别注册它们的名称和路径。

```bash
vllm serve /path/to/Qwen2.5-3B-Instruct --served-model-name qwen \
  --max-model-len 32768 --enable-prefix-caching --generation-config vllm \
  --enable-lora --max-lora-rank 64 --max-loras 2 \
  --lora-modules customer-sft=/path/to/customer/adapter seller-sft=/path/to/seller/adapter
```

在现有仿真命令上增加 `--inference-backend vllm --vllm-url http://127.0.0.1:8000/v1 --served-model qwen --customer-served-model customer-sft --seller-served-model seller-sft`。两个角色用各自的 SFT 模型决策。`--served-model` 指定反思和未单独配置角色时使用的默认模型。所有名称都以服务的 `/v1/models` 为准。`--model` 指向与服务器相同 revision 的本地 tokenizer 目录，行为评估使用其已有 `--tokenizer`。需要认证时设置 `VLLM_API_KEY`，不写入模型日志。

本地测试覆盖 HTTP 协议、prompt token 保持、JSON Schema、角色路由和失败处理。真实 vLLM 服务上的生成和吞吐量需另行实测。服务模型列表核对不等于远端权重哈希核验，运行报告明确记录此区别。API 依据：[vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)。

`simulation/run.py` 的 `run(args, load_model=...)` 支持注入其他模型。加载器返回 `(client, 模型信息)`，client 提供 `complete()`、Tokenizer 和生成预算，Tokenizer 与上下文预算需匹配所用模型。程序自动准备上下文后调用模型，查询后端和模型动作权限见[工具说明](README.md#tools)。

依赖集中在 [pyproject.toml](pyproject.toml)。从项目根目录按需安装开发或训练依赖，PyTorch 后端需与目标设备匹配。

```bash
python -m pip install -e './agent_simulation[dev]'
python -m pip install -e './agent_simulation[training,dev]'
./agent_simulation/python.sh -m agent_simulation.simulation.run --help
```

## 仿真运行与时间线

运行顺序固定为 `run.py` 装配 → `episode.py` 调度 Tick → Environment 准备上下文 → Agent 调用 Inference → Environment 统一结算。Evaluation 从已保存样本或已提交轨迹读取结果；回放只读取运行库。

本地仿真和单步行为评估默认不扫描模型权重哈希，也不生成整套源码指纹。需要核对本地权重版本及完整性时显式添加 `--verify`，行为评估的同一选项也检查数据文件。默认仍由模型加载器检查文件可用性与结构，LoRA 检查角色及加载后的参数键；完整底座权重一致性对照只在 `--verify` 下执行。

| 文件 | 职责 |
|---|---|
| `simulation/run.py` | 装配数据、模型、Agent 和市场，启动 Episode 并保存报告 |
| `simulation/episode.py` | 管理逻辑时钟，按 Tick 协调 Seller 定价、Customer 行动和统一结算 |
| `simulation/replay.py` | 读取已结算记录，导出时间线和动画，或提供本地回放服务 |

每次运行写入独立输出目录。`model_calls.jsonl` 保存模型输出，`runtime.sqlite` 保存状态、动作与 Tick，`report.json` 汇总运行结果。回放时使用该次运行的输出目录。

运行失败时，`report.json` 的 `error` 记录主因。资源关闭或结果汇总也出错时，`finalization_errors` 按发生顺序记录阶段和错误；程序仍会尝试保存最终市场状态。若决策已经完成但收尾失败，运行报告标为 `failed`，已提交的 Tick 可在 `final_state.json` 中核对。

在 `simulation.run` 命令后追加 `--max-concurrent-requests 4 --request-timeout 120`，每个决策阶段最多四个主体并发。Seller 的报价统一生效后，Customer 并发决定搜索、等待或退出，市场按顾客 ID 生成展示列表，再并发收集列表响应并统一结算。同一顾客的 Response 分批也受限并发，结果按原列表顺序合并后只结算一次，与不同主体共享全局请求上限。默认上限为 2，每个推理请求含服务排队的期限默认为 120 秒；并发为 1 时每阶段逐人执行，所有人仍在同一阶段边界读取市场。

Seller 默认使用 `--seller-price-batch-size 1`，每次明确要求一个目标商品报价，与单商品 SFT 对齐。可显式设置 2–16 个目标进行小批联合报价实验；该模式尚不代表已有多目标 SFT 或行为效果验证。整轮共享冻结快照，报价只暂存，收齐后统一生效；后续商品不会看到本轮先前的草案。首个目标开始前可在 Tick 中选择整店等待，开始报价后必须完成本轮。当前生效价格未知时，可提交 null 保持未知，或提交正价；已知价格不能被清空。统一提交会核对快照版本，失败不发布部分报价。旧 capacity 模式及 `--seller-pricing-mode` 参数已删除。离线监督只使用真实单商品正价，包括维持已知价格。

两个角色完成训练并导出 adapter 后，运行 Episode 时可以将下列参数加入已有命令；这些是计划输出路径，目前尚未生成。两份 LoRA 在工作进程内共用同一底座，按请求角色选择。可只指定一个角色，另一角色使用底座。记忆总结也使用底座，训练数据只监督了行为动作。原始调用日志在启用角色路由时记录 `active_adapter`，底座行记为 `__base__`。

```bash
--model models/Qwen2.5-3B-Instruct \
--customer-adapter training_runs/customer_2pct_v14_recent/adapter \
--seller-adapter training_runs/seller_2pct_v14_recent/adapter
```

每次运行只启动一个本机推理子进程，持有一份 Qwen 权重。它在短收集窗口内合并已经就绪的请求，每行保留自己的输出约束。Customer 和记忆总结也使用该服务。Seller 全部完成后才提交价格；报价超时或失败会使 Tick 失败，迟到的结果不能写回市场。

将 `EPISODE_DIR` 设为实际运行目录，即可打开像素市场、Tick 指标和决策时间线。本地服务可按需展开 Context、Prompt 和模型输出。离线导出包含同一套回放界面和已结算事件，`--animation` 是 `--html` 的别名。购买金额沿用原始价格单位，CTR 和购买率都以曝光为分母。

```bash
./agent_simulation/python.sh -m agent_simulation.simulation.replay "$EPISODE_DIR/runtime.sqlite" --html /tmp/episode.html
./agent_simulation/python.sh -m agent_simulation.simulation.replay "$EPISODE_DIR/runtime.sqlite" --serve 8765
```

加 `--play` 可从同一份市场配置开一场新的本地仿真，任选一位已注册顾客或商家进行人工控制，其余角色由 Agent 运行。角色切换在下一 Tick 生效。新的 SQLite 记录存放在源运行同级的 `play-*` 目录，原回放仍可从主菜单打开。若源报告里的 Catalog 路径已迁移，可传入 `--play-catalog <目录>`。 加 `--play-markets agent_simulation/simulation/markets.json` 可先选择卫衣或保温杯市场。顾客名单来自所选主题的历史搜索，商家和可搜索商品按对应的 Catalog 类目限定。

<a id="tools"></a>

## 工具与动作

Customer 的搜索／选购和 Seller 报价共用 `agents/actions.py`，训练答案与环境动作使用同一份格式定义。两类角色的直接查询由 `environment/operations/api.py` 定义，经环境接口校验和执行。

当前项目未接入 MCP 客户端或服务端。工具通过本地 Python 接口执行；Function Calling Schema 描述工具参数，`environment/operations/api.py` 分派操作，`inference/client.py` 对接模型。这些接口目前不通过 MCP 提供服务。

程序读取历史、统计市场和检索 Memory，再调用模型。Seller 的 SFT 和仿真使用相同的报价 JSON。Customer 输出 Query、响应或 Tick 控制动作的 JSON，环境负责执行。

价格参数保持 `price_raw`，Agent 接口明确按分解释，数值不转换；该单位是沙箱约定，原始来源单位尚未确认。

### 模型实际收到的工具

| 模型阶段 | 实际 tools | 输出与执行方式 |
|---|---|---|
| Seller 每个目标报价批次 | `[]` | 输出 `{"prices":[...]}`，覆盖本批全部目标商品 |
| Customer Query | `[]` | 输出 `{"query":"保温杯"}`，程序据此执行 `search_products`；Tick 模式还可输出 `{"action":"no_action"}` 或 `{"action":"exit"}` |
| Customer Response | `[]` | 输出 `clicked_item_ids`、`purchased_item_ids` 两个数组；程序合并所有分批结果后执行一次 `submit_response` |

Seller 的市场统计、竞品发现、详情和本店历史查询仅保留后端接口，供上下文构造及人工诊断使用。模型不再进入查询循环，也没有查询预算或重复结果缓存。一次返回多个调用、调用未开放接口、参数错误或执行失败，都直接报错并留下轨迹，不自动重试。

环境先暂存每批 Seller 报价。维持已知价格时重复当前价格，维持未知价格时填写 `null`。独立决策等该店所有批次成功后统一提交，Episode 等所有 Seller 完成后统一提交。

Customer 当前没有自主 tool calling。`search_products`、`submit_response` 和 `exit_customer` 是程序把已验证的 JSON 决策转成环境动作；查历史、查记忆、查商品详情虽有后端接口，却不在 Customer 模型的 tools 中。两个角色的反思也由程序触发，模型返回经验 JSON，程序验证后新增、修订或撤回记忆。

SFT 监督 Seller 最终报价 JSON 和 Customer 的 Query／Response JSON。市场查询轨迹、反思、等待和退出属于仿真功能。单步 `behavior_prediction` 使用相同监督协议。Seller 从工具调用切换到 JSON 后，Lite 数据契约已更新，旧 Token 缓存需按新契约重建。

程序准备事实和检索结果，并校验模型返回的 JSON。环境的 `submit_prices` 和直接查询接口供程序执行和诊断使用。

### 查询后端使用示范

以下 JSON 演示程序调用查询后端的语法，不表示模型拥有全部工具权限，也不是人为生成的 SFT 查询标签。`query_business` 是本店经营记录查询：保留的 CPU 基线也未向默认 Seller 模型开放该接口；当前 Seller 由程序自动提供近期事实并检索同店相关片段，模型不开放主动查询。ID 和截止点取自本次输入，不能照抄到其他主体。

| 需要补充的信息 | 调用示例 | 无需调用的情况 |
|---|---|---|
| Seller 商品更早历史 | `query_business({"item_ids":[579227],"before_time_index":643335,"limit":10})` | 已给出的近期轨迹足够；实际边界取历史最早点以查更早记录 |
| Seller 完整市场统计或其他窗口 | `query_market_position({"item_id":579227,"start_time_index":643335,"before_time_index":850769})` | 默认摘要已提供本次所需统计与窗口；其他统计或自定义窗口可通过此接口查询 |
| Seller 过去经营经验 | `query_long_term_memory({"query":"痘痘贴降价后曝光与购买变化"})` | 相关经验已提供；返回空结果后不重复查 |
| Customer 更早购物历史 | `query_customer_history({"before_time_index":100,"limit":5})` | 当前历史足够；100 仅为语法示例，实际取合法历史截止点 |
| Customer 当前商品详情 | `query_displayed_products({"item_ids":[2919]})` | 商品完整信息已提供；2919 须属于该 Customer 尚待响应的展示列表 |
| 最终定价 | `{"prices":[{"item_id":579227,"price_raw":880}]}` | 无需改价时可保持原价；大店按批次规则显式覆盖待处理商品 |

Customer 的选择随原展示请求 ID 一起交给环境结算。构造商品选择提示词时先核对这个 ID，读取对应展示时保存的价格。最近搜索时间包含已发起但尚未结算的搜索。单独运行和 Tick 批量结算还会在写入事务内核对这个 ID；待处理请求已更换时整笔事务回滚。结算成功后先保存回执，再反思，后续反思失败仍能查到已生效的反馈。

只读调用不推进时钟、不改价格、不记录点击或购买。返回结构化事实或统计，模型动作由环境校验执行。调用失败保留错误，不制造动作补救。

### 后端完整工具清单

下表列出查询后端和环境动作。标记“查询接口”的条目可由程序调用，模型可调用的 Seller 工具单独标出。

| 使用者 | 工具 | 实际作用 |
|---|---|---|
| Customer 查询接口 | `query_customer_history` | 查自己的更早搜索、点击和购买，历史索引与仿真步骤分开截止 |
| Customer Response 查询接口 | `query_displayed_products` | 按商品 ID 查询当前待响应展示列表中的已有属性与展示价；不产生点击或购买、不返回真实标签 |
| Customer 查询接口 | `query_long_term_memory` | 从自己的已保存长期经验中检索相关消费经验 |
| Customer 阶段结束后由环境执行 | `search_products` | 用最终 Query 搜索，保存当前展示及价格快照 |
| Customer 阶段结束后由环境执行 | `submit_response` | 校验最终点击／购买集合并一次结算 |
| Customer 退出后由环境执行 | `exit_customer` | 仅 Tick 仿真：将最终 JSON `{"action":"exit"}` 转成退出状态并保存时间，后续不再调度该 Customer；不加入模型查询工具或 SFT Schema |
| Seller 查询接口 | `query_shop_products` | 按商品ID分页遍历完整本店目录，每页最多20件，返回已有属性与当前仿真价格 |
| Seller 查询接口 | `query_business` | 查目标商品或同店其他商品的完整属性、带时间参考价、历史、反馈和已执行价格动作，返回覆盖范围与续查游标 |
| Seller 查询后端（不向模型开放） | `discover_public_products` | 按完整三级类目、不同商家、截止前已有观测查公开候选，支持标题／品牌过滤和分页 |
| Seller 查询后端（不向模型开放） | `query_public_products` | 查看候选商品公开属性与带索引参考价；不返回竞品私有经营结果 |
| Seller 查询接口 | `query_long_term_memory` | 查自己的经营经验和平台曝光机制假设 |
| Seller 市场查询后端 | `query_market_position` | 查同类其他商家的公开价格与购买记录数分布。竞品曝光、点击和转化率不披露；本店仍保留自己的经营指标 |
| Seller 查询接口 | `read_business_report` | Tick 模式查看自上次决策以来的经营报告；独立单次调用返回全部已结算反馈。重复读取不重置窗口 |
| Seller 模型动作 | 报价 JSON | Agent 暂存覆盖本批全部目标的报价，整轮成功后通过环境的 `submit_prices` 统一提交 |

Customer 查询后端支持历史和长期记忆，商品详情查询限定于 Response 当前列表。当前 SFT 使用自动组装上下文与最终答案协议，没有查询调用监督。`environment.operations.api.runtime_tools()` 在仿真中加入 `read_business_report`，Tick 模式进一步为 `query_business` 加入可选逻辑时间边界。旧复查调度接口已经移除。

`query_business` 后端支持 `item_ids` 同时包含目标商品和其他本店商品，包括未纳入当前仿真市场的本店历史商品；这些商品只返回历史部分，并列入 `outside_market_item_ids`。其他卖家商品被拒绝。历史范围为 `[start_time_index, before_time_index)`；`limit` 按商品限制明细，`summary_of_returned_series` 只汇总返回明细，显式请求 `metrics` 时的 `statistics` 才覆盖整个查询窗口、不受明细条数限制。每个商品分别返回续查游标；仿真反馈与改价动作也各有自己的游标。空窗口的无分母比率返回缺失，不能解释为零转化。成功的只读查询不再写入完整调用日志；长期记忆需要的证据由记忆服务保存。查询不改变市场价格、反馈或时钟。

Tick 模式的 `query_business` 可指定 `[start_logical_time, before_logical_time)`，边界必须是 Tick 边界，并且不超出已完成区间；不与事件序号过滤混用。返回各商品 `activity_timeline`，包含压缩后的空白区间、价格和区间经营数量；`limit` 限制明细区间数量，`next_before_logical_time` 可继续查询更早区间。历史观测仍单独使用 `before_time_index`，不会把逻辑时钟推进解释为新真实数据。

默认 workflow 和 SFT 提供市场摘要，并在 `market_position.competitor_examples` 展示模型从最多64件候选中选出的可比商品，不限制最终件数或每店一件；没有合格候选时可为空。`public_product_pages` 默认不预填完整查询页，不等于没有竞品例子。后端发现接口按属性条件／字面关键词筛选、按 ID 分页；自动例子另用 BGE 与 RRF 排序。BGE 也用于事实与长期记忆检索。

Seller SFT 和仿真都用本批目标商品 ID 约束报价 JSON，仿真允许保留未知价。历史和报告查询不推进市场状态。搜索保存展示快照，价格提交和反馈结算由环境执行。

程序直接调用后端时，已知且在可见范围内的竞品 ID 可以查询公开属性与参考价，无需先调用发现接口；这两个接口均不在模型 tools 中。`discover_public_products.focal_item_id` 用本店商品定义待搜索类目，不能填竞品 ID。范围不合法的只读查询会被环境拒绝。失败请求也保留在工具轨迹中。

<a id="inference"></a>

## 模型推理

`inference/` 与 `training/` 并列。这里管理模型的执行，`training/` 管理模型参数更新。

| 文件 | 职责 |
|---|---|
| `client.py` | `ModelClient`、输出异常、同步／异步调用及 PydanticAI 适配 |
| `requests.py` | 官方 Tokenizer、Chat Template、请求预算和两种推理后端共用的输入 Token |
| `model_policy.json` | 上下文容量、输出预留、Customer 分批和两类 Agent 的近期历史窗口 |
| `__init__.py` | 本地 Torch 设备选择，供推理、训练和样本构造复用 |
| `vllm.py` | 远程请求、按角色路由模型和原生 JSON Schema |
| `local/qwen.py` | 本地 Qwen 生成、JSON 约束解码、多请求推理和 KV 前缀缓存 |
| `local/checkpoints.py` | 识别基础权重版本，按角色加载 LoRA |
| `local/service.py` | 共享推理进程、请求排队、并发和超时 |

### 接入其他模型

Customer、Seller、共享 Agent 基类和长期总结通过 `ModelClient` 调用模型。客户端提供 `tokenizer`、`max_new_tokens`、`generator_id`，并实现 `complete(request, call_id=...)`，返回标准 Assistant 消息及可选 `tool_calls`。

`simulation/run.py` 的 `run(args, load_model=...)` 接受自定义加载器，返回 `(client, checkpoint_report)`。内置 Qwen 和 vLLM 直接实现模型接口。已有 PydanticAI provider 可通过 `FrameworkClient` 接入，使用匹配的 Tokenizer、生成预算和模型核验。仿真入口根据本次运行配置启动共享服务、绑定角色 LoRA，并在结束时关闭服务。

仿真和行为评估默认使用本地 Qwen2.5，也可配置 vLLM。两个后端复用同一套 JSON 决策协议。

业务允许哪些动作由 `agents/decisions.py` 定义，长期经验的字段及引用规则由 `agents/memory/validation.py` 定义。推理层执行对应的输出约束，环境校验并结算动作。

<a id="agent-core"></a>

## 一次决策实际看什么

Seller 直接从默认上下文提交报价或按 Tick 规则等待，不自主查询。默认市场摘要、竞品例子、事实检索和长期记忆仍自动提供；输入齐全不代表决策质量已经通过验证。

| 决策阶段 | 默认输入 | 检索围绕什么 |
|---|---|---|
| Customer Query：接下来搜什么 | 用户画像＋最近 10 次搜索／点击／购买事实＋最多 5 条较早事实＋最多 5 条适用长期记忆 | 只用此前已知的搜索与行为；不能使用尚未生成的本次 Query |
| Customer Response：当前列表点／买什么 | 相同类型的画像和历史＋本次 Query＋当前展示商品属性与价格＋适用长期记忆 | 较早事实用本次 Query 检索；长期记忆还参考展示类目与价格概况。结果可与 Query 阶段不同 |
| Seller：目标商品报什么价 | 全店经营摘要＋目标商品属性、当前价和近期经营事实＋目标类目的公开市场摘要＋最多 5 个相关经营片段＋最多 5 条适用长期记忆 | 目标商品较早片段，以及同店其他商品的近期和较早片段共同参与；候选完整，输入只展示选中的片段 |

Seller 的近期窗口按商品保留最近 10 个经营观测／结算点，历史观测与后续仿真反馈按各自时钟标记，合计保留 10 点。相关经营片段每段最多 3 点；5 段不是 5 件商品，也没有另一个固定的“同店相似商品”名额。竞品默认展示其他店铺同类目的公开价格、购买统计和模型选出的相关例子；模型不再自主查询具体竞品详情。竞品私有曝光、点击、转化率不进入输入。

同一店本轮的目标上下文来自同一冻结快照，彼此看不到尚未生效的报价。Customer Response 分批共享决策前历史，不把前一批的选择当作后一批的新历史。历史观测只取截止点之前的数据，当前监督答案不进入输入。

边界：仿真经营明细和检索限于该店已注册商品；历史全店统计可以覆盖未注册商品，输入注明范围，不能混用分母。当前 SFT 长期记忆为空，运行时可有生成的记忆；SFT 也未监督市场查询轨迹。这个上下文对应价格、曝光、点击和购买反馈仿真，并不包含日志中不存在的成本或库存。

## Agent、Memory 与检索

```text
agents/
├── customer.py                发起搜索、响应列表与分批
├── seller.py                  逐店决策与商品分批
├── memory/
│   ├── evidence.py            原始经历与证据
│   ├── reflection.py          经验总结与更新
│   ├── versions.py            版本可见性与旧经验撤回
│   ├── validation.py          总结输出协议
│   └── prompts.py             反思指令
├── retrieval/                 决定时检索历史事实（history.py）与长期总结（long_term.py）
├── base.py                    共用 Agent 基类
├── context/                  可见事实：shopping.py、pricing.py、pricing_evidence.py、market.py
├── actions.py                 Customer/Seller 动作格式与业务校验
├── decisions.py               请求准备、输出约束和单次决策
├── prompts/                  将事实写成角色消息，附上指令与价格约定
└── schemas.py                 JSON 和工具参数协议
```

决策路径为：`customer.py`／`seller.py` → `prompts/` → `decisions.py` → 模型调用 → Environment 暂存与结算。`actions.py` 供训练与运行时复用动作格式。

`memory/` 管理长期经验、证据、版本和更新；`retrieval/` 提供检索能力，既查询较早的原始事实，也为长期经验排序。

“较早”指未进入本次近期窗口的已可见记录，并非固定天数。Customer 的 Query 和 Response 都包含画像、近期事实、较早事实检索与适用长期记忆；Response 额外根据当前 Query 和展示商品形成检索需求，结果不要求与 Query 阶段完全一致。当前 SFT 两阶段长期记忆都为空，运行时由已生成记忆提供。决策长期记忆上限为 5 条；“有效”指主体、可见时间、引用和版本合法，相关性目前按分数排序，没有经过校准的最低相关性阈值。

长期记忆只在仿真运行时更新，与当前真实动作 SFT 独立。每批读取 10 条新经历，可检索最多 5 条旧事实与适用旧总结。模型按角色写一个简短经验总结，并显式选择 add/revise/retract/no_update。Customer 侧总结产品／品牌／类目偏好、搜索比较习惯、购买模式和有依据的价格反应；Seller 侧总结定价经营经验及平台曝光分配的工作假设。观察和推断要区分，平台假设不视为已知推荐算法。事实通过 1–3 条 E 引用支撑总结，保存时映射回原始 ID，不把事实副本当作总结正文。修订、撤回必须指定旧 M 记忆，旧版本退出检索；撤回的当轮证据作为审计上下文，不伪装成反证引用。当前 2% SFT 不新增长期记忆或反思标签。每批一次生成，不增加审查模型和重试链；保留引用范围、新证据和动作一致性检查，不用文本正则审查总结措辞。

短期 Memory 不另建一套数据库：`context/shopping.py` 从本用户历史选择近期搜索、点击和购买；`context/pricing.py` 从本店经营历史选择逐商品近期轨迹。完整历史继续由数据仓库和环境状态保存。

Customer 训练取样和仿真都调用 [context/shopping.py](agents/context/shopping.py)。两边默认保留本用户最近 10 次请求，从更早的经历中最多取 5 条。生成搜索词时只依据已有历史，选择商品时再用本次搜索词检索，并读取本次展示的商品。两边都用 [environment/batching.py](environment/batching.py) 的 `response_batches()` 先用模型的聊天模板和 tokenizer 计算完整 Response 输入。历史、曝光列表与输出预留都能放进预算时，一次处理整单。超过预算才按原顺序分批。训练从历史日志读取展示列表和行为标签。仿真由当轮平台返回列表，结算后的行为进入后续历史。

| 文件 | 职责 |
|---|---|
| `agents/prompts/context_format.py` | 对指定列表进行无损 JSON 表格编码，支持当前输入的无损解码 |
| `agents/prompts/instructions.py` | 顾客搜索和选购、卖家报价的指令，以及共享的价格单位约定 |
| `agents/decisions.py` | 准备模型请求、约束输出、解析一次 JSON 决策 |
| `agents/actions.py` | Customer 搜索／选购、Seller 报价的共用动作定义 |
| `agents/schemas.py` | JSON、标量、工具 Schema 与动作参数校验 |
| [logs.py](logs.py) | 逐次模型调用的 JSONL、调用汇总，SQLite 中的 Agent 运行轨迹，以及报告和清单的 JSON 写入 |
| `agents/retrieval/bge_encoder.py` | 本地中文 BGE、完整 token 分块、批量编码与编码器身份；分块向量合并不保证语义无损 |
| `agents/retrieval/retrieval_text.py` | 构造检索 Query，以及商品、历史经历的待检索文本 |
| `agents/retrieval/history.py` | 近期窗口之外的事实片段、分段规则与 RRF 排序 |
| `agents/retrieval/similarity.py` | 直接余弦评分、所有权／时间过滤与同分顺序；不重复构建向量索引 |
| `agents/memory/evidence.py` | 授权证据快照、事实经历、引用及证据选择 |
| `agents/memory/reflection.py` | 总结触发条件、模型请求和经验更新 |
| `agents/memory/validation.py` | 总结格式、引用范围与更新操作校验 |
| `agents/memory/versions.py`、`agents/retrieval/long_term.py` | 前者管理版本可见性与撤回，后者按当前决策上下文检索有证据的经验 |
| `environment/memory.py` | 将经验读写和检索接入环境 |

`datasets/` 复用角色上下文、`agents/prompts/context_format.py`、`agents/schemas.py` 和事实检索，不加载运行中的 Agent、市场环境或长期经验生成服务。模型调用位于 `inference/`，模型训练位于 `training/`。

<a id="sft-data"></a>

## SFT 样本准备

正式目录：`data/kuaisearch/lite/sft/`。这是**全量真实监督池与可展开输入**，模型 Prompt 按需生成，容量在读取时检查。

```text
sft/
├── manifest.json              全量标签计数、文件哈希与完成范围
├── customer/{train,test}/labels.parquet
├── seller/{train,test}/labels.parquet
```

固定 Lite 样本的回归测试重建两条 Customer 请求和一条 Seller 报价，比较 Prompt、答案、Token、答案起点和权重。确认输出变化后更新基准，并用新目录生成 Token 分片。

```bash
./agent_simulation/python.sh -m unittest agent_simulation.tests.datasets.test_sft_regression
```

Customer 每行是一条原始请求，保存真实 Query、曝光集合、点击集合、购买集合及划分。同一请求提供 Query 和 Response 两步监督；超大曝光集合按已有规则生成多个 Response 批次，批次共用原请求划分。当前标签池共 555,553 条 Customer 请求（train 539,012，test 16,541）。Response 行数取决于实际输入长度。直接读取样本训练时，先展开一遍以确定每轮行数；使用 Token 缓存时直接读取已保存的行数。

两步都输入 Profile、近期搜索／点击／购买历史及较早事实检索片段，长期 Memory 当前为空。Query 的检索只依据此前记录；Response 在同一历史上增加本次真实 Query 与曝光商品，并可用该 Query 重新检索。当前 Query 不进入 Query 阶段的输入；当前点击／购买不进入 Response 的输入。当前示例可用 `datasets.labels preview` 按需生成，不在标签目录中维护旧版副本。价格在 Agent 接口统一按分解释，这是沙箱约定，原始数值不转换，来源单位仍未确认。

Seller 有 12,250,786 条商品价格监督记录。每条都要求同店同商品在当期和严格更早的时点各有有效价，其中包括维持原价的观测。

当前划分和标签数量以 [manifest.json](../data/kuaisearch/lite/sft/manifest.json) 为准；时间边界见下文[时间划分与跨主体来源](#时间划分与跨主体来源)。Seller train 11,874,348 条、test 376,438 条；当前没有 validation 目录。

`labels.parquet` 保存监督与来源键，不直接进入模型。`SFTDataset` 从冻结 Catalog 读取商品、Profile 和严格截止前的历史，调用现有 SFT 构造器，返回完整 `messages`、`tools` 和答案。短期历史、公开竞品权限和价格单位保持原有含义；Seller 使用下述目标上下文，Customer 使用固定商品数分批规则，不生成额外查询、理由、经验或复查时间标签。

Catalog 的 DuckDB 视图使用绝对路径读取本地事实表、Seller 经营表和冻结划分。外部经营表与划分文件列在 Catalog 清单的 `business_sources` 和 `split_files`。迁移时需要这些文件，并按新路径重建 Catalog 视图和引用它的 SFT 清单。

当前 SFT 和仿真均默认用本地中文 BGE 检索较早事实；长期 Memory 为空不代表关闭历史检索。近期历史保留，较早片段按商品关系、语义和可比较数值融合排序。构造器显式传 `history_embedder=None` 才使用无语义的对照模式，正式输入使用 BGE。这只运行编码器，不调用 Qwen、不生成解释或经验。

BGE 默认按 CUDA、Apple MPS、CPU 的顺序选择可用设备。可用 `MEMORY_BGE_DEVICE=mps`、`MEMORY_BGE_DEVICE=cuda:0` 或 `MEMORY_BGE_DEVICE=cpu` 指定设备。向量缓存按模型和编码规则区分，可跨设备复用。

构建 Token 缓存时，每个 worker 各加载一份 BGE。GPU 默认 1 个 worker，CPU 默认 24 个，可以通过 `--workers` 覆盖。

具体 Memory 和市场摘要的组织见[架构说明](../docs/architecture.md)。

从项目根目录运行，展开一条完整真实样本：

```bash
./agent_simulation/python.sh -m agent_simulation.datasets.labels preview \
  --dataset data/kuaisearch/lite/sft \
  --tokenizer models/Qwen2.5-3B-Instruct \
  --sample-id query:14 --sample-id response:14 \
  --sample-id seller:108629:579227:850769 \
  --output /absolute/new_examples
```

训练读取接口（这里只展示读取，不执行训练）：

```python
from pathlib import Path
from agent_simulation.datasets.labels import SFTDataset

data = SFTDataset(Path("data/kuaisearch/lite/sft"), Path("models/Qwen2.5-3B-Instruct"))
try:
    sample = data.sample("seller:108629:579227:850769")
    messages, tools = sample["messages"], sample["tools"]
    # Default iteration covers all actor buckets. Workers can specify a disjoint bucket explicitly.
    for sample in data.iter_samples("customer", split="train"):
        pass  # Hand complete messages/tools to a separate trainer.
finally:
    data.close()
```

读取时实际计算官方模板Token数，超限会明确报错，不静默丢商品或历史。全量标签池完成不代表所有动态上下文都已逐条通过；后续训练前仍需处理实际读取中的例外。Response 权重 `batching.request_weight` 保留，由当前 LoRA 训练器消费。

完整样本的预构造统一使用 `datasets.tokens`，写入可恢复的 NPZ Token 分片。旧的全量消息 Parquet 导出路径已删除；需要查看模型输入时使用指定样本预览。

<a id="sample-reuse"></a>

### 已有样本是否需要重新生成

| 产物 | 当前状态 |
|---|---|
| 原始事实、Catalog 和官方 train/test 标签 | 直接复用 |
| seed=42 的 2% 候选 | 使用当前配对清单。Customer train/test 为 10,780/331 个请求，Seller 为 237,487/7,529 条报价 |
| Token 缓存 | 使用当前配对清单构造到 `runs/lite_sft_tokens` |

旧候选目录和旧 Token 分片已清理。历史构造日志和冻结源码保留。

### 数据集代码与开发命令

`datasets/samples/` 只放 Customer 和 Seller 的上下文与监督构造。标签、抽样、Token 文件和编码各有一个入口，训练与评估复用相同读取函数。

| 命令 | 用途 |
|---|---|
| `python -m agent_simulation.datasets.labels build --catalog ... --output ...` | 导出标签池 |
| `python -m agent_simulation.datasets.labels preview --dataset ... --tokenizer ... --sample-id ... --output ...` | 查看指定样本的模型输入和答案 |
| `python -m agent_simulation.datasets.sampling --help` | 查看配对抽样参数 |
| `python -m agent_simulation.datasets.tokens --help` | 查看 Token 构建参数 |

候选清单通过标签清单的 SHA256 确认数据来源。同一个买家或卖家的记录归到同一组，交给一个 Worker 按时间处理。构造中断后在同一个输出目录续跑。显式使用 `--verify` 可检查文件哈希。

Customer Response 的输入预算为模型上下文上限减去输出预留，两项默认值由 `inference/model_policy.json` 配置。运行时请求的输出上限更高时，按更高值预留。`environment/batching.py` 的 `response_batches()` 按列表原顺序连续切分，各批各自持有本次 Query、决策前的历史和商品数据副本。这些批次是计算切片，各自独立判断本批商品。仿真由 `response_concurrency` 控制单用户并发，由 `--max-concurrent-requests` 控制阶段总并发。全部批次成功后按原请求顺序合并，只结算一次。SFT 构建器为每个 Response 批次赋予 `1 / 批数` 的权重。

### Seller 增量构造

顺序遍历通过 `SFTDataset.iter_samples` 按 Seller 和时间读取一次历史，`samples/seller.py` 保留每商品最近 10 点。滑出窗口的记录增量组成较早的三点片段；未满三点也参与检索，已完成片段复用记录与统计特征。相同时间索引的样本全部生成后，才将该时刻记录加入历史。同一时刻的全店基础上下文只构造一次。单条随机读取 `seller_sample` 保留按截止点查询的路径。

### Seller 目标上下文 v2

SFT 和默认仿真共用 `agents/context/pricing.py`。输入为全店摘要、报价目标的近期事实、目标三级类目的公开市场参照，以及本店相关经营片段的检索结果。报价目标与参考商品分开：`requested_price_item_ids` 指定输出，`product_details` 可以包含检索命中的其他自有商品，但不增加报价标签。

整店近期概览及事件索引每个店铺时点只准备一次，选中目标后才查询其市场数据。SFT 不再先计算整店容量分批；最终完整 prompt 和真实答案仍按官方 tokenizer 检查，超限明确失败。全店共同窗口统计保留；默认竞品参照只包含公开属性、价格和报价时间，不暴露竞品流量／转化率，不声称价格因果效应。

历史检索的统一候选池包含目标商品较早片段，以及同店其他商品的近期和较早片段。目标近期事实已直接展示，不重复检索；同商品关系只是排序信号，不保证保留目标自己的较早片段。沿用实体、语义和经营数值的 RRF 排序，最多取五段。余弦计算不再为每次查询新建 LlamaIndex；主体内按确切文本缓存最多 2,048 个历史向量，文本变化即重新编码。仿真一轮仅分段一次，后续目标复用片段；SFT 同一店铺时点共享候选池，各商品的近期分段仅在新观测到来后重建。市场查询缓存绑定连接、卖家、精确截止点、类目或目标集合；最多 32 项，不使用近似时间桶。当前模拟价格／反馈按每轮快照单独组织，不能复用旧轮模拟结果。

同店报价共享冻结快照、全店摘要、历史准备和按类目索引的模拟市场数据。各目标独立构造 prompt，按 `price_concurrency` 有限并发；Episode 中还受全阶段 `--max-concurrent-requests` 约束，现有本地推理服务可将这些请求合并成推理 batch。全部报价通过后统一提交，任一失败则取消同组待完成任务并丢弃草稿。Tick 首个目标需先决定是否全店等待，选择报价后再并发剩余目标；独立决策可直接并发。

Prompt、Tokenizer 或检索设置改变后，用新目录重新生成 Token 分片。只有标签数据变化时才需要重建标签池和候选清单。

先做小规模 direct／rolling、标签、时间边界和 token 检查，再扩大构造规模。输入更短、构造更快不等于已验证模型行为效果，仍需按当前 test 选型协议和仿真评价检查效果。默认逐商品报价会增加模型调用次数；上下文构造提速不代表整轮推理更快，总推理耗时与成本需要单独测量。

### 固定 seed 的分层比例抽样

`agent_simulation/datasets/lite_2pct_sampling.json` 配置 seed=42，以及官方 train／test 各自的 0.02 比例，不再单留 validation。当前数据是 `data/kuaisearch/lite/sft`，候选清单是 `data/kuaisearch/lite/candidates`。原验证段已归回 train；先在各官方 split 内分层抽样，再构造上下文。Customer 的抽样单位是完整请求，之后展开 Query 和全部 Response 批次；Seller 是商品时点报价记录。因此 Customer 的 2% 不是展开后行数或 token 数的 2%。

默认 Customer 按购买／仅点击／无行为 × 冷启动／近期历史／需检索历史分层。Seller 按涨价／降价／维持 × 短历史／已有历史分层，也可在 `strata` 中选择 `category_level1_id`。每组总预算按完整标签池的 2% 四舍五入，各层以原始比例为目标，并按 seed 固定的哈希顺序抽取。

每个 split 分开配对。先筛出整张展示列表都有供给的 Customer 请求，再按分层配额抽取完整请求。每件展示商品预留该 split 最早一条 Seller 报价样本，报价索引须早于或等于请求索引。同一商品的报价可覆盖多个请求。Seller 剩余名额按分层配额补齐。某层因供给不足或预留名额超出原配额时，会调整各层名额并保留总预算。候选清单的 `supply` 记录按实际入选报价核对的请求、展示件次和商品数。

这里是按原占比分配，不是把各动作均衡到相同数量。2% 测试候选仅用于快速检查；完整原始测试标签池保留，用于自然分布的最终单步评价。L2 必须另取连续主体／时间窗口，不能把抽散的 2% 标签连接成轨迹。上下文超限等构造失败仍应计入覆盖率，并报告接受后分布，不能用删除失败样本后的分数代替整体结果。

当前 2% 清单已存在。改动抽样配置时可用第一条命令生成新目录。日常直接用现有清单构造。

```bash
./agent_simulation/python.sh -m agent_simulation.datasets.sampling \
  --dataset data/kuaisearch/lite/sft \
  --config agent_simulation/datasets/lite_2pct_sampling.json \
  --output /absolute/new_candidates

./agent_simulation/python.sh -m agent_simulation.datasets.tokens \
  --dataset data/kuaisearch/lite/sft \
  --tokenizer models/Qwen2.5-3B-Instruct \
  --candidates data/kuaisearch/lite/candidates/manifest.json \
  --output runs/lite_sft_tokens \
  --market-cache /Users/pidan/Documents/Playground/ecommerce_policy_sandbox/runs/lite_full/benchmarks/market-cache-production \
  --seed 42 --workers 1 --buckets 32 --reserve-gib 8
```

`datasets.tokens` 根据清单中的实际 split 构造缓存。当前只构造 train／test；训练使用全部已接受 train 行，评估和 checkpoint 选择使用全部已接受 test 行。test 不参与梯度更新，报告明确标为参与模型选择。构造失败数和接受后分布分别报告。

完整前序历史仍保留，未抽中请求也可作为后续请求的合法历史。超限按整个请求／报价记录明确排除，并分别报告各 split 的排除数及分层分布；不会静默补抽或只留下 Customer 的半个请求。2% 是候选比例，最终接受量可能因超限少于它。

本地已生成配对后的 2% 候选。Customer train 为 10,780、test 为 331，Seller train 为 237,487、test 为 7,529。完整标签池分别是 Customer 539,012／16,541、Seller 11,874,348／376,438。当前配对清单待构造 Token，容量检查会另报排除数。

### 构造热路径与失败记录

`datasets/tokens.py` 在主进程确认候选清单对应当前标签数据，不加载 Tokenizer。加 `--verify` 才读取标签、事实、候选及市场缓存文件做哈希核验。worker 只读挂载市场库。Seller SFT 不再生成训练未使用的证据审计副本；模型输入与答案保持不变。

单 split 和多 split 共用一条按主体、时间排序的构造路径。启动 worker 前，主进程用初始化锁准备共享向量库；样本循环不增加锁重试或异常兜底。构造过程中只捕获容量超限，数据或程序错误直接报告。

分片写入后原子提交 `progress.json`，记录已接受的业务 ID 与分片哈希。恢复时检查已提交分片存在，跳过这些记录的上下文构造和编码。加 `--verify` 可核对续跑的分片哈希。已记录的容量排除随失败日志保留。后续样本所需的完整历史仍会读取。未提交的缓冲记录重新构造，不再逐条重算已提交 token 做比对。旧格式的未完成缓存缺少恢复信息，需要使用新的输出目录；完整缓存继续可用。

生产序列化只编码，不逐条解码并比较整个上下文。无损表示、曝光顺序、标签重构和跨批权重由独立测试覆盖；Seller 的诊断接口仍可显式打开 roundtrip。Customer 按配置的商品数切片（默认 64），只对最终批次编码并检查容量。

每条最终样本编码后检查实际输入／答案长度，并复用 token 写入缓存。任何一批超限，该请求的 Query 和所有 Response 一起排除；失败立即写入对应 bucket 的 `failures.jsonl`（record_id、role、split、reason、detail），并继续下一条。完成后 `selection.json` 汇总接受与排除的记录，后续可单独处理失败清单。

新版 Customer Token 缓存记录按 token 分批的规则，构建时使用新输出目录。既有缓存可以继续用于复现实验。若历史和单件商品已超预算，缓存构建会记录该请求无法容纳。

### 构造和推理复用

Customer 默认近期窗口为 10 次请求，更早记录仍参与最多 5 条事实检索。配置见 `agent_simulation/inference/model_policy.json`，新建仿真和 SFT 共用；既有仿真数据库保留创建时的窗口，不自动迁移。Customer 分层抽样同步使用这一阈值，seed 仍为 42。阈值变化可改变选中的 Customer 请求及其配对的 Seller 报价，各 split 的 2% 总预算按完整标签池计算。清单记录 `feature_policy`。

v2 候选清单按角色和主体 bucket 合并回放清单实际包含的 split（当前为 train／test），一次遍历完整前序历史，再按原 split 分别输出。完成标记、超限排除和恢复检查仍按 split 独立保存；已完成 split 不重新构造。当前构造入口只接受 `sft-split-candidates-v2`；旧版 train-only 构造分支已删除。不再读取未绑定抽样清单的旧缓存。Customer 非选中请求只更新历史，不构造监督输入。

BGE 将多个缺失文档的 token 块按长度组批，每批最多 8 块，按原文块顺序及长度加权还原文档向量，不截断原文。模型进程缓存最多 8,192 个确切文本／查询模式的向量，主体缓存保留最多 2,048 项；缓存只保存向量，权限、时间过滤和检索排名仍每次计算。批量浮点计算可能有微小数值差异，因此编码器版本及输入 release 更新，旧记忆 embedding 不默认为兼容。

Seller 在市场与检索证据齐备后渲染一次；Customer Query 渲染一次，Response 先渲染整单，超预算时再用相同历史测量各切片。容量检查产生的官方模板 token 经 tokenizer 和内容身份核对后直接传入本地 Qwen，避免再次分词。Qwen worker 还缓存确切 token 前缀的 KV：默认最多 4 项、每项最多 2,048 token；用 `--prefix-cache-tokens 0` 关闭。不同 adapter 不共用 KV，批内混用 adapter 时不共享前缀；生成前复制缓存，生成结果不会写回公共前缀。长短不同的请求保留正确 attention mask 和位置，不改变各自的动作 grammar。这里是有界前缀复用，不是完整的持续组批调度器。

Token 缓存绑定对应的标签清单和候选清单。小规模合并／分开回放检查 token、监督边界和权重一致，KV 用真实小型 Qwen 和角色 LoRA 验证 logits／贪心输出；这些检查不代表已测得生产模型或 CPU 服务器的总体吞吐。

### 决策范围与业务边界

仿真 Seller 的 `shop_scope` 明确区分：报价目标、近期明细、自动历史检索和仿真经营报表覆盖纳入当前市场的本店商品；`observed_shop_period` 统计该卖家全部历史商品，可能含市场之外的商品。SFT 则从历史记录中的合格自有商品构造候选。不能将仿真的商品子集称为现实店铺的完整商品池。目标输入默认提供近期市场价格摘要及模型选出的其他店铺竞品例子。完整市场查询仅保留在后端供分析，模型不主动调用查询接口。

逐商品报价使用同一旧快照，但不读取其他商品本轮报价，因此没有联合优化替代品、搭售或店内价格阶梯。Customer 各响应批独立判断并取并集，没有全列表预算、库存或互斥购买约束；并发只改变执行方式，不改变这种决策假设。当前 SFT 监督真实观测价格和行为，不能将历史拟合等同于最优定价或反事实需求校准。成本和利润、库存、钱包余额尚未进入结算约束；购买计数是记录数，金额按一条购买一件估算。官方 Ranking 不直接使用价格，价格先影响 Customer 的选择，之后可通过已结算点击历史间接改变 Ranking 输入。

### Customer 的共享历史准备

同一请求的 Query／Response 监督共享历史候选准备。Query 用近期历史中的搜索词检索更早经历，Response 用本次真实搜索词检索。流式 SFT 的 `CustomerHistory` 按原始数据划分决定可见历史，train 只读更早的 train 请求，test 读所有更早的请求。同一时间索引的请求共用更新前的历史，等这一组处理完才写入。候选抽样只控制哪些请求生成监督样本。仿真 `AgentContextBuilder` 按用户与近期窗口保存同类缓存。缓存复用事实、文本与向量，每次决策按本阶段的搜索词和截止点重新排序。

当前近期窗口按用户选择改为 10：最近 10 次请求按时间保留，每条包含当时的搜索词、点击和购买商品；较早历史最多检索 5 条。Response 没有额外重排、裁剪或总结最近 10 次行为。返回的两个上下文彼此独立，修改一个不会污染另一个。仿真 Query 与 Response 可复用同一历史准备对象，但 Response 读取搜索后形成的展示快照；两阶段的查询、排名和商品可见范围各自确定。

### 答案Token与训练批次

`agent_simulation.datasets.tokenization.encode_sample` 使用官方工具聊天模板，生成完整Token序列；上下文对应的 `labels=-100`，只有最终Assistant答案参与监督。`collate` 右侧补齐长度，Padding同样不参与loss。返回的 `model_inputs` 仅含 `input_ids`、`attention_mask`、`labels`；`sample_weights` 和 `metadata` 在外部。

数据层输出 Python 列表；`training.data_loader.TensorCollator` 将其张量化，`training.trainer.BehaviorTrainer` 在 loss 中消费每条 Response 的 `1/批数` 权重。Query 和 Seller 权重为 1。Seller 与 Customer 分别训练 adapter；Customer 内保留 Query、Response 两阶段。

### 下一步：模型训练

样本与 Token 准备完成后，由 [模型训练模块](README.md#training) 消费这些结果。这里不运行模型权重更新。

<a id="training"></a>

## 模型训练

本目录读取 [datasets](README.md#sft-data) 提供的完整训练样本，将 Token 批次交给 LoRA 训练器，执行 loss、反向传播、验证和 checkpoint 保存。

| 文件 | 职责 |
|---|---|
| `run.py` | 依赖预检、训练命令及恢复入口 |
| `data_loader.py` | 按需构造样本、读取 NPZ 缓存、按 seed 和 epoch 打乱、组装张量批次 |
| `trainer.py` | 答案 loss、样本权重及训练器 |

历史 pilot 清单已清理，不再作为当前命令入口。当前完整 2% 使用 `data/kuaisearch/lite/candidates/manifest.json` 固定候选，并在构造完成后通过 `--token-cache` 训练。小规模试跑可通过 `--train-samples` 和 `--eval-samples` 控制记录数。

全量 SFT 直接使用对应数据版本的 train 标签池。当前 Lite 使用 `--dataset data/kuaisearch/lite/sft`；Full 应从官方 Full train／test 单独建立，不能与 Lite 混用。未使用 token cache 时，以固定 seed 抽取 `--eval-samples` 条完整 test 记录作评估，默认 128；设为 0 使用完整 test。只支持当前官方 train/test 数据。

[rolling.py](datasets/labels.py) 按主体和时间滚动生成样本。Customer 保留最近 10 条请求，从该用户严格更早的其余全部请求中检索最多 5 条。Seller 按商品保留最近 10 个经营观测点，从目标较早片段及同店其他商品全部前序片段中检索最多 5 段。两者均在本时间索引的全部样本构造完成后才更新历史。

### 运行入口

统一入口是 `agent_simulation.training.run`。用 `--model` 选择本地 Qwen2.5、Qwen3 或 Qwen3.5 权重，默认读取该模型自带的 tokenizer。训练 Qwen3.5 使用其文本模型，LoRA 同时覆盖全注意力和线性注意力的输入、输出投影。Qwen3 和 Qwen3.5 的动作模板关闭 thinking，监督从 JSON 答案开始。

更换模型时，用相应的 `--tokenizer models/<模型目录>` 构建新 Token 缓存，冻结标签和请求划分可以复用。CUDA 支持时默认采用 BF16，显式 `--dtype` 可以覆盖。项目默认上下文预算仍为 32,768 token。恢复旧检查点使用其训练时的模型、Tokenizer 和依赖版本。

使用 `--token-cache` 时，当前角色的全部 split／bucket 完成即可开始训练。训练默认读取完成清单和分片路径。恢复训练继续使用模型权重指纹。根目录 `complete.json` 表示两个角色全部完成。

```bash
./agent_simulation/python.sh -m agent_simulation.training.run check \
  --dataset data/kuaisearch/lite/sft --model models/Qwen2.5-3B-Instruct --role seller
./agent_simulation/python.sh -m agent_simulation.training.run check \
  --dataset data/kuaisearch/lite/sft --model models/Qwen2.5-3B-Instruct --role customer
```

`check` 校验版本及少量 train／评估记录的完整输入，不加载生成模型权重；传 `--token-cache` 时直接检查完成的缓存。当前评估 split 为 test，输出明确记录 `test_used=true`。它不是全量动态 Prompt 容量扫描。批量构造遇到超长记录时，记录并跳过，不截断或缩批重试；版本、标签或存储错误仍终止。

当前计划是完整 2% 的 Customer 和 Seller 各训练 10 个 epoch。以下命令仅在新版本对应角色的 Token 缓存完成后执行；目前该缓存尚未构造。以 Seller 为例：

```bash
CUDA_VISIBLE_DEVICES=0 ./agent_simulation/python.sh -m agent_simulation.training.run train \
  --dataset data/kuaisearch/lite/sft --model models/Qwen2.5-3B-Instruct \
  --token-cache runs/lite_sft_tokens \
  --seed 42 --role seller --output runs/seller_2pct \
  --device cuda --dtype bfloat16 --rank 64 --epochs 5
# Customer 使用 --role customer 和独立的 --output。
```

这条命令消费候选缓存中的全部已接受 train／test，test 用于选模。不传 `--token-cache` 时，默认读取整个标签池。

默认 LoRA rank=64、学习率=1e-6，参考 [PolicySim §4.1.4](https://arxiv.org/html/2603.19649v1#S4.SS1.SSS4)。`q/k/v/o_proj`、alpha=2×rank、dropout=0.05、微批大小=1、梯度累积=16 是当前显式配置，不称为官方复现。支持单设备训练和同机 CUDA 多卡 DDP。DDP 每张卡各加载一份基座，处理不同样本并同步 LoRA 梯度，要求已完成的角色 Token 缓存。

两张卡训练 Customer 时用同一条训练命令和已完成的新缓存，通过 `torchrun` 启动每卡一个进程：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 -m agent_simulation.training.run train \
  --dataset data/kuaisearch/lite/sft --model models/Qwen3-32B \
  --token-cache runs/customer_tokens --role customer --output runs/customer_ddp \
  --device cuda --dtype bfloat16 --epochs 5 --batch-size 1 --accumulation 8 \
  --activation-offload-tokens 20000 --loss-breakdown
```

批大小和梯度累积按卡设置。上例每步最多使用 16 条真实样本，末批的填充行不计入 loss 或评估。多卡 checkpoint 保存每个进程的随机状态，续训沿用原卡数和训练参数；改卡数时可用 `--init-adapter` 新开运行。每卡都需要足够显存放一份模型。

小规模试训可用 `--train-samples` 和 `--eval-samples` 控制记录数。设备缓存压力较大时，可用 `--empty-cache-steps 1` 启用 Transformers 的未使用缓存释放机制；样本、答案和 loss 保持原样，该参数记入运行配置。

Customer 只训练 Response 正例时，使用 `--stage response --positive-train-responses`。此选项只保留训练侧至少含一项点击或购买的 Response 行，沿用每行原始请求权重。评估仍读取完整 Test Response，包括两个动作都为空的记录。

loss 为每条答案 Token 的平均交叉熵乘以样本权重，再对当前梯度累积组中的样本数取平均；不会在单条样本里把 `1/批数` 约掉。上下文和 Padding 不参与监督。Qwen 只对预测答案所需的位置计算词表 logits，完整上下文仍参与注意力计算。

数据按需读取，有限缓冲区打乱训练顺序。当前官方两段协议用 test loss 选 checkpoint；test 保持独立的数据流，只评估，不参与反向传播。固定评估记录随 seed 写入配置，恢复时保持相同记录。该 test 结果参与模型选择，不称为独立盲测。配置、数据和权重身份写入 `run_config.json`，不再对源码文件生成续训指纹。

默认预算为 10 个完整训练轮次，短跑可显式指定 `--max-steps`。Seller 每条标签生成一个训练样本。使用 Token 缓存时，从已完成清单读取接受后的准确行数；按需训练 Customer 时，只扫描标签的曝光数组长度，按 `1 + max(1, ceil(曝光数 / 批大小))` 累加 Query／Response 行数，不执行检索、渲染或分词。每轮数量存入 `run_config.json`，恢复时复用；短跑也沿用这一计数规则。

默认训练 5 轮，每轮结束时用 test 评估并保存 checkpoint。保留 test 损失最低和最新的 checkpoint，并将最优 adapter 导出到 `adapter/`。显式使用 `--max-steps` 时可通过 `--save-steps` 调整保存和评估间隔。答案损失用于选模，实际生成的搜索词、点击和购买由行为评估入口统计。

训练路径为：`run.py` 选择数据和训练参数 → `data_loader.py` 读取样本并组成 Tensor → `trainer.py` 计算加权 Loss、评估与保存。

续训时给原训练命令追加 `--resume latest`，也可以指定本次输出目录里的 `checkpoint-N`。每个完整快照保存 LoRA 权重、AdamW 状态、学习率调度器、随机状态和 Trainer 进度，MPS 另存设备随机状态。恢复会按原种子重建该轮数据顺序，跳过已完成的批次，从上次完整保存处继续。续训比对数据、模型、配置和依赖版本；不比较源码哈希，注释或文件组织变化不会单独阻止续训。使用 Token 缓存时还比对本角色实际读取的分片清单，使用 `--verify` 可核对清单中的分片哈希。

需要在已完成的训练之后再加几轮时，指定新的输出目录并使用 `--init-adapter` 加载已有 LoRA 权重。新训练会重新初始化优化器和学习率调度器。

当前行为 SFT 没有长期总结、工具查询轨迹、推理理由或退出标签；不生成这些标签。完整标签池并不等于已完成全量训练。小模型协议测试也不等于 Qwen 3B 的训练效果验证。

<a id="evaluation"></a>

## 行为评估

| 入口 | 层次 | 可用标签 |
|---|---|---|
| `micro/behavior_prediction.py` | 微观行为预测 | 读取 SFT `test`，用真实 Query、曝光列表内的点击／购买集合、Seller 商品时点的观测价格打分 |
| 单步评估加 `--price-response` | 微观价格干预及分布汇总 | 真实展示价下有购买标签，改价后的购买结果来自模型 |
| `macro/trajectories.py` | 宏观轨迹对照 | 在已完成 Tick 覆盖的历史索引窗口内比较搜索活动、点击／购买率和价格曲线 |
| `micro/relevance.py` | 微观搜索相关性 | 官方 `rank.test.jsonl` 提供 Query–商品二元相关性标签 |

单步评估用本地中文 BGE 比较生成 Query 与真实 Query，报告截到 0–1 的平均余弦相似度及逐条分数。端到端均值将无效输出记 0，完全文本匹配率留作参考。相似度是连续分数，品牌、型号和规格是否一致仍可结合逐条 Query 检查。单步评估合并 Customer 分批结果后计算集合一致性和价格误差。轨迹对照使用从 cutoff 到首个未完成 Tick 的左闭右开窗口。Seller 真实报价映射到最近的已完成 Tick，同一 Tick 只取最近一条，距离相同取较早的一边。保留真实报价的原始索引，对照该 Tick 决策后的模拟价，只在有真实报价的 Tick 计算价格误差、方向一致率和探索性 DTW。初始价单列，用于计算仿真首轮的调价幅度。Customer 比较同一窗口内的完整搜索次数、间隔和搜索后的行为分布，仿真侧另报行动机会、等待和退出。

现有 L2 是描述性轨迹比较，尚不构成动态机制验证。Customer 已统计相邻请求的相同 Query 比例，但没有类别转移、连续未购买后的条件购买概率；Seller 已统计相邻价格变化幅度、涨跌持平占比及连续同价观测长度，但没有经营状态条件下的价格转移。尚未实现按时间窗口的群体分布演化、多种长度的误差曲线，以及多 seed 与经验转移／历史重采样基线的统一实验。

扩展时保留上述入口，分别增加个体条件转移、群体分布演化、不同长度偏差三组报告。先固定训练期定义的类目、价格带和长尾分组，再观察测试期变化；无行为／无购买窗口保留样本量，比例无分母时记为缺失。Query 类目需要独立可复现的标注或分类规则。真实 `time_index` 和仿真逻辑时间共用相对索引尺度，搜索间隔按索引差报告。比较不同窗口长度时固定主体队列并报告覆盖率。

Lite 完整 Seller 测试标签池中，`369584/376438`（约 98.18%）相对上一次有效价格观测保持不变。单步评估的 `price_changes` 以同商品上一次有效报价为参照，报告调价方向准确率、相对调价幅度的绝对误差，并按真实涨价、降价、持平分组。沿用上次报价的基线在同一批有效预测上计分。经验转移和历史重采样基线只从训练期拟合。多 seed 实验共用初始状态、平台策略和评估队列，固定 seed 列表，并记录不同运行长度和中途失败。

价格变化与前期购买表现的条件关系是观察性关联。当前随机 Recall／Ranking 与历史曝光策略不同，轨迹差异同时包含平台策略和 Agent 的影响；新政策没有真实反事实轨迹标签。相同真实历史初始化后的自由仿真只能使用之后生成的反馈；逐步注入真实反馈的单步回放不能作为自由仿真的长期真实性证据。

### 时间划分与跨主体来源

当前 Customer 和 Seller 直接使用官方 train/test 时间边界：train 截止 `936724`，test 为 `936726..965265`。Seller 首个合法训练标签为 `29`，Customer 为 `1`。同一时间索引整体保留，允许同主体跨 split，但不会出现 train 时间 300、test 时间 200 的交错。

动态输入按当前目标严格向前读取。Customer 点击／购买记录可以同时成为 Seller 经营反馈和公开市场购买统计；三者共享来源时间，跨主体共享本身不等于泄漏。当前评估是滚动真实历史协议：较早 test 结果可以进入较晚 test 样本上下文，不参与模型参数拟合。若训练 Customer 模型与 Seller 模型，共用这一边界；不能把某一方的测试期标签加入另一方训练。静态用户与商品画像没有快照日期，无法独立证明这些属性在每个历史时点已经可用，严格时点结论须保留这一限制。

按当前实验约定，test 用于选择 embedding、训练配置和 checkpoint，均在报告中注明；不声称该结果来自独立盲测。已有 test embedding 对照直接复用，不因取消 validation 重跑。验证段上的复跑保留为开发诊断，该段现已归回官方 train。

```bash
./agent_simulation/python.sh -m agent_simulation.evaluation.micro.behavior_prediction --help
./agent_simulation/python.sh -m agent_simulation.evaluation.macro.trajectories --help
```

单步评估只使用目标前的真实历史，每个 Query、Response 分批或 Seller 目标去掉监督答案后各调用模型一次，不执行可选市场查询。单条生成、格式或动作参数错误记为 invalid，保留原始输出并继续下一条，不修改答案或重试。Response 任一分批失败时，该请求整体记为失败。逐条预测随运行写盘，汇总在结束时生成。需要扫描数据文件哈希时加 `--verify`。设备内存不足、模型执行故障、文件和数据库等基础设施错误仍抛出，不计入预测失败率。

程序测试集中在 `tests/`，覆盖数据层、输入构造、Agent 决策、Memory、环境结算和训练接口。运行输出写入 `runs/`；测试不参与日常样本构造。

<a id="tests"></a>

## 程序测试

这些是可重复执行的程序检查。使用模型替身或随机小模型的测试验证接口和流程；真实行为是否可信由 `evaluation/` 中的评估任务判断。

| 目录 | 检查什么 |
|---|---|
| `agents/` | Customer／Seller 决策、最终动作协议和提示词组装 |
| `environment/` | Tick、搜索、结算、历史权限与经营统计 |
| `agents/memory/` | 长期总结、事实证据、更新与持久化 |
| `datasets/` | 真实监督、历史边界、分批、无损表示与 Token mask |
| `training/` | 答案 loss、样本权重、LoRA、验证抽样和恢复 |
| `models/` | 模型输出格式、解码、checkpoint 与调用日志 |
| `platform/` | 召回、排序和重排接入；相关性评价位于 `evaluation/` |
| `evaluation/` | 行为指标与工具实验自身的计算逻辑 |
| `structure/` | 模块导入和依赖边界 |

`fixtures.py` 提供共用的测试市场，不是真实训练标签。测试覆盖自动上下文、最终动作、Memory、分批与环境结算。

从项目根目录执行全部测试或某一组：

```bash
./agent_simulation/python.sh -m unittest discover -s agent_simulation/tests -t .
./agent_simulation/python.sh -m unittest discover -s agent_simulation/tests/agents/memory -t .
```

<a id="platform"></a>

## 商品搜索与排序

比如顾客搜「保温杯」，仿真可以从市场随机挑商品，也可以用 BM25 找标题里匹配的商品。商品搜索与排序模型来自 KuaiSearch，它先找一批与搜索词相关的候选，再结合顾客的点击历史和商品信息给候选打分。模型和所需数据由 [pipeline.py](platform/pipeline.py) 加载。

| 阶段 | 位置 | 处理方式 |
|---|---|---|
| 找候选（召回） | [recall.py](platform/recall.py) | DPR 比较搜索词与可见商品的向量，选出最匹配的一批 |
| 给候选打分（排序） | [ranking/](platform/ranking) | 把搜索词、顾客点击历史和候选商品整理成模型输入，一次给多件商品打分 |
| 决定展示列表（重排） | [pipeline.py](platform/pipeline.py) | 默认展示得分最高的前 K 件商品，也可以传入自定义展示规则 |

排序模型给每件商品打分，估计顾客点击或购买的可能性。环境在展示时附上价格，结算后的点击会用于下一次排序。模型找出的候选商品、排序分数和最终展示列表都记在 `platform_trace` 中。模型与输入整理代码的来源和版本记录在 [kuaisearch.json](platform/kuaisearch.json)，其中源码路径相对于 `platform/`。

分数相同时，商品 ID 较小的排在前面。自定义重排可以从完整召回列表里重新选出要展示的商品，最多 K 件，并须填写自己的 `reranker_id` 供实验记录使用。单独比较重排策略时，共用召回、排序和初始市场设置，各轮市场状态由各自的用户和卖家行为推进。

[模型文件清单样例](platform/assets.example.json)列出召回模型、商品向量、排序模型权重、特征和 ID 映射。文件路径相对于清单所在目录。备齐文件后，用 `--search-backend kuaisearch --platform-assets /path/to/assets.json` 启动仿真。当前使用测试资产验证接入，实际运行需要官方训练后的权重。

### 相关性评估

相关性评估核对搜索词和商品文本，计算有标签样本对的 ROC-AUC 和 Average Precision。官方验证集有 417 对，测试集有 4,643 对，均只提供搜索词、商品文本和二元标签。测试集中只有两个搜索词各出现两次，数据缺少搜索请求 ID 和完整候选列表，因此按单个样本对评估。分类阈值可在验证集上确定。 [评估脚本](evaluation/micro/relevance.py) · [相关性参考数据](../data/kuaisearch/relevance) · [官方训练源码](https://github.com/benchen4395/KuaiSearch/tree/88a545d6ada3a9a237e2b231b3321ddb83deb0a2/relevance)

```bash
./agent_simulation/python.sh -m agent_simulation.evaluation.micro.relevance \
  --reference data/kuaisearch/relevance/rank.test.jsonl \
  --predictions /path/to/pair_scores.jsonl --score-model '<checkpoint + revision>' \
  --output runs/relevance_test.json
```

`pair_scores.jsonl` 每行保留从 0 开始的 `source_row`、原 `query`、原 `text` 和 `score`。`--score-model` 可填写权重版本或 SHA-256，报告会保存该标识及输入文件的 SHA-256。

给无标签的平台列表打分时，使用 `request_id`、`item_id` 和 `score`，并省略 `--reference`。未评分商品的 `score` 填 null。每个请求的平均分只计有分数的商品，要和 `request_score_coverage` 一起看。比较平均分时应使用同一模型和同一分数尺度。

### 更新与失败边界

Seller 每批按本批商品检索相关事实和长期经验，事实候选池包含目标较早片段及本店其他商品的近期、较早片段；实际检索结果加入后再检查容量。环境的历史区间查询在 `environment/operations/queries.py`，两类角色的动作定义共用 `agents/actions.py`。Seller 运行时不再额外构造可逆的上下文审计副本；需要调试来源时，可显式调用 Prompt 的 `audit_sink`。

长期总结的生成接口接收完整消息，返回原始 JSON 文本或已解析的草稿对象，必须明确给出 action；不再按空内容或 supersedes 猜测更新操作。模型输出被截断、JSON 无效、引用越界等草稿问题只拒绝本次更新，不推进已读游标；同一 Tick 不重复尝试。合法的 no_update 会推进游标。请求预检（含容量）、回调内部、编码器或设备故障，以及主体、时间和事务冲突均向调用方抛出。运行库的 calls 表保存反思请求及返回的原始输出，Agent 轨迹用 audit_id 关联该行；模型底层调用可另查 model_calls.jsonl。

样本读取先按 split／stage 选择监督目标，再展开对应 Prompt；未选择目标的较早事实仍进入后续历史。Query 单独读取时不展开当前曝光列表，Response 单独读取时不生成 Query 样本。

容量检查与训练组批复用同一次按需分词结果。Token 缓存仅存在于本次读取的内存对象中，不进入 JSON／Parquet；输入变化时重新编码。Seller 已删除旧 capacity 模式，只检查选中目标的最终请求。


### Seller 市场上下文

全店经营汇总覆盖历史中出现的全部自有商品，使用统一的最近 10,000 个时间索引。报价清单仅纳入截止前有过有效报价的商品。默认近期竞品价格摘要按其他店铺同三级类目、截止前 10,000 个索引计算；不混入扩展补样或旧价分位数。默认竞品也仅取这个近期窗口，数量不足不补旧报价。市场查询后端仍支持指定窗口及扩展补样。

Seller 每个批次根据本批商品和市场上下文报价。默认市场摘要和历史检索仍自动提供。后端历史查询遵守冻结截止点，实时仿真价格另列。SFT 仍只监督真实价格答案。

默认 Seller 报价先选择目标，再按决策用途取摘要。同轮目标共享冻结快照和类目市场数据。近期同类价格摘要取每件商品窗口内的最新观测，仅有效价格进入统计；最新观测无效时不回填旧价，仿真当前价格和已结算反馈使用独立时钟。

竞品沿用同三级类目、其他店铺和严格截止时间的公开候选池。候选超过 64 件时，先用已有标题、品牌和新鲜度粗排缩至最多 64 件（粗排每店最多四件）；小候选池直接保留。BGE 和 RRF 排序后交给上下文模型选择。模型读取公开属性、报价和全类目统计，Seller 上下文保留入选商品的原始报价卡片和完整类目统计。未配置上下文模型时直接提供排序后的候选。

市场摘要覆盖同类目其他店铺的全部合格近期报价，提供有效报价商品/店铺数、未知价格覆盖、价格最小值/P25/中位数/P75/最大值、报价索引范围及中位索引。本商品最近观测价在该分布中的分位为0..1，价格相同时使用中位秩；不足两件可比商品时不报告分位。同品牌参照是全类目统计的子集，不代表规格一致；无品牌或未知品牌不归成同品牌竞争组。店铺商品占比仅衡量报价商品集中度，不代表销量份额。最新无效报价保持未知，不用旧有效价格补齐。

上下文模型在一次调用内从目标标题摘出 1 至 3 个能说明商品类型与必要规格的短词组，再选择直接替代商品的 ID 和市场比较角度。入选标题须能匹配这些词组，比较时统一大小写、全半角和空白，并识别运营商常用简称。目标标题明确标注套装时，候选也须明确是套装，允许套盒、件套或套装等不同写法。词组只用于当次筛选，商品标题始终显示原文。原始标题须支持相同购物用途和可比较的销售形态。套装之间可以比较不同尺寸或件数；适配型号、话费运营商和面额等决定用途或计价单位的条件仍须匹配。无合适竞品时可以返回空列表。Seller 用入选商品的原始标题、报价和观测时间作比较。市场概况按模型选中的角度，由已有统计呈现中部报价与上端极值、同品牌与全类目中位数、报价覆盖或店铺供给集中度。报价位置复用已计算的价格带，计算与同品牌中位数的差值，并明确窗口前旧报价。两部分仍进入 Seller 上下文，标记为 derived_analysis。

Customer 的背景商品表可供完全相同商品 ID 和标题复用：item_title 中的 {title_ref:i} 指向 behavioral_summary.item_details[i]。编码时同时比较直接表格与引用形式，计入说明开销，只选择更短的形式；解码恢复原始标题。品牌、类目、价格、事件顺序和原始重复 ID 不变，不按商品 ID 合并不同历史快照。共用来源说明只写一次，长期记忆说明只在有记忆时加入。

同店同时间同类目的统计、品牌分组和排序价格数组复用已有 `decision_cached`，目标商品的价格分位直接在共享数组上查找。同一时点已有市场快照和经营汇总继续复用，不新增缓存层。64件候选复用已有 `$table` 格式无损压缩重复字段及共有值。逐商品模型分析仍逐条执行，没有声称已经实现多目标联合推理或并发加速；实际耗时需另测。使用本版提示词必须在新的 Token 缓存目录构造，旧分片不续写，2%清单不变。

历史事实检索在原 RRF 分数后乘以弱新鲜度系数：最多扣减 10%，历史时钟半衰期为 10,000 个索引，仿真时钟为 10 个结算步，均不解释为自然日。参数尚未校准；年龄不作为硬过滤条件，强相关旧证据仍可入选。长期生成经验不统一按年龄衰减，继续由反思、修订、撤回和可见性规则维护。共享排序函数位于 `agents/retrieval/ranking.py`；竞品筛选独立位于 `competitors.py`，复用已有编码器和缓存。

Customer 搜索与列表响应保留选中的历史事实、原请求价格、日期和顺序，默认不展示内部检索得分。列表响应保留本次全部候选的标题、标价与身份，超出模型输入预算时才按原顺序分批。

本品最近观测价无效时，默认报价行保留原来的 `last_observed_price_raw=null`、状态和索引，并用 `last_valid_reference` 给出严格更早的有效标价及其索引。最近观测本身有效时，一份标价即可表达两者。滚动读取在短期窗口外仍保留最后有效参考点。


### 用模型整理上下文

Customer 和 Seller 的决策指令、历史摘要和竞品选择指令统一用英文。生成的历史摘要、市场概况和报价位置也用英文。商品标题、历史搜索词及其摘录保留原文。

Customer 和 Seller 共用 `agents/context` 中的摘要入口。每部分历史先按紧凑编码测量，不超过 `SIMULATION_HISTORY_TOKENS`（默认 4096 tokens）时保持事实原文；超过时优先一次总结，仅输入预算放不下时才分组，不增加二次汇总调用。摘要模型读取带明确商品 ID 的原始事实表格，不把共享属性索引当商品 ID。Customer 的近期历史与较早检索结果分别判断，不因其中一段较长而替换另一段短记录；同一 Response 的各批次复用已准备的历史。摘要要求合并重复搜索、保留具体事件和变化，避免每组重写人物介绍或重复缺失说明；提示词要求400字符以内，生成格式上限500字符。Seller 的历史变化与检索证据、各商品近期历史分别按长度判断。原始事实留在数据和环境中，摘要只替换模型输入中的长历史。

新增的顾客近期行为背景、30 天统计、Seller 经营汇总、当前报价及市场统计不交给模型重算。历史生成文字标记为 `model_summary`，市场选择与数值比较标记为 `derived_analysis`，都不是新增观测事实。精确价格留在原始报价字段，摘要不换算单位。

服务器 `env.sh` 配置 Qwen3-32B 作为上下文模型，使用 `cuda:1`；只在需要选择竞品或总结长历史时加载。也可以手动配置：

```bash
export SIMULATION_CONTEXT_MODEL=/path/to/Qwen3-32B
export SIMULATION_CONTEXT_DEVICE=cuda:1
export SIMULATION_HISTORY_TOKENS=4096
# 如已有同模型 vLLM 服务，填写 /v1 地址；服务模型名须使用上述模型路径。
# export SIMULATION_CONTEXT_URL=http://127.0.0.1:8000/v1
```

本地上下文模型构建 Token 缓存使用 `--workers 1`，避免各进程重复加载权重；多个构建进程可共用 vLLM 服务。旧缓存保持不变；使用新上下文时另建输出目录。设置 `SIMULATION_CONTEXT_MODEL=''` 可关闭模型整理，用于无摘要对照。该开关不恢复旧的固定三个竞品规则。配置和实现变化均未触发训练或重建现有缓存。

### SFT 默认模型与上下文预算

`python -m agent_simulation.training.run check/train` 未填写 `--model` 时使用项目内 `models/Qwen3.5-9B`；`--model models/Qwen3.5-27B` 或 `--model models/Qwen3-32B` 可选择其他模型。Tokenizer 默认取所选模型目录，不单独指定时不会继承旧 3B tokenizer。Token 缓存应与所选 tokenizer 一致。上下文整理模型仍由 `SIMULATION_CONTEXT_MODEL` 独立配置。

Customer 与 Seller 共用 32,768 tokens 的项目总预算，至少预留 2,048 给输出，因此常规输入预算为 30,720。选择更大模型不会自动扩大这个预算。Customer 按真实 Prompt tokens 在超限时切分曝光列表；Seller SFT 是单商品监督，仿真仍按 `price_batch_size` 分组（默认 1，最多 16），不是按 tokens 自动分组。

4,096 tokens 是长历史调用摘要模型的触发阈值，不是所有历史或背景字段的硬上限。顾客 `behavioral_summary` 包含原始 30 天统计、近期点击/购买 ID 与商品属性；商家 `business_summary` 是原始商品 30 天曝光/点击/订单快照。它们保持原始数值和快照时间，不与自己重建的近期记录累加。日志曝光不证明注意到商品，未点击也不是明确负偏好。

### 新上下文 SFT 准备

Customer 的近期行为背景现在统一使用商品 ID、标题、品牌和三级类目名称，SFT Query、Response 与仿真共用这一表示。原始曝光列表、统计、近期点击/购买 ID 的顺序和重复、真实行为标签不变；完整背景属性仍留在原数据中。已删除仅在缓存构建时开启的紧凑背景开关。

原相对时间2%的 Train/Test 窗口不变。Customer Train/Test 和 Seller Test 保持原清单；Seller Train 在构造上下文之前保留全部观测价变化样本，并按 `SHA256(record_id:seed)` 固定抽取最多四倍数量的持平样本。这里的变化相对于同商品严格更早的最近有效报价，无效价格不会成为比较基准。只筛选监督目标，完整历史和公开市场供给仍从原 Catalog 读取。

先生成独立候选清单（输出目录必须尚不存在；已有清单可直接用于后续构造）：

```bash
source /root/autodl-tmp/zhz/llm_simulation/env.sh
python -m agent_simulation.datasets.sampling \
  --dataset runs/lite-relative-time-2pct/sft \
  --candidates runs/lite-relative-time-2pct/token-records/manifest.json \
  --output runs/lite-relative-time-2pct/token-records-seller-4to1 \
  --seller-keep-ratio 4
```

当前2%清单中，Seller Train 从255,894条缩至25,360条，包括5,072条变化、20,288条持平。Seller Test 6,692条、Customer Train 10,656条及Test 311条不变。该清单不再要求每条Customer曝光都对应一条被选中的Seller训练标签；商品事实与公开历史报价仍完整保留。

新缓存使用独立目录；旧缓存不覆盖、不续写。下列命令只构造Token，不启动训练：

```bash
source /root/autodl-tmp/zhz/llm_simulation/env.sh
python -m agent_simulation.datasets.tokens \
  --dataset runs/lite-relative-time-2pct/sft \
  --candidates runs/lite-relative-time-2pct/token-records-seller-4to1/manifest.json \
  --tokenizer models/Qwen3.5-9B \
  --market-cache runs/lite-relative-time-2pct/token-market \
  --output runs/lite-relative-time-2pct/tokens-Qwen3.5-9B-actions \
  --workers 1
```

此命令准备两个角色。可加 `--role customer` 或 `--role seller` 分开构建。上下文模型的选择与长历史总结会增加构造时间；大量 Seller 样本建议使用现有共享 vLLM 接口，避免多个进程分别加载 32B 权重。不要将小样本预览当作完成的训练缓存。

历史检索保留现有 BGE/RRF：Customer 最近 10 条独立呈现，较早候选不含这些记录；Query 阶段按近期搜索主题检索，Response 按当前 Query 检索。Seller 按本品属性及经营变化检索本店历史，排除已展示的本品近期片段。最多五个片段是控制输入规模的设置，并不代表每条都高度相关；本次没有凭空增加相似度阈值或新的检索 Agent。

服务器 `env.sh` 将现有 BGE 向量缓存的数据库上限设为 8 GiB，复用已有约 7.2 GiB 的向量文件；原默认 2 GiB 无法打开该文件。磁盘预留和淘汰机制继续沿用现有实现。此配置不创建新缓存层，也不删除已有向量。

### 训练行为权重

Customer Response 默认权重为未记录点击/购买1、有点击无购买2、有购买4；同时有点击和购买取4，不叠乘。Query保持1。每个Response Batch原有的 `1 / batch_count` 权重继续保留，额外系数乘在整条答案的平均Token损失上，不是单独给购买ID加权。日志中未记录互动不等于用户明确拒绝商品。

Seller观测价变化权重为2，持平为1。筛选与加权复用同一套严格更早有效价格比较。训练和 Test 损失使用相同行为权重，checkpoint 按加权后的评估损失选择。损失分项也应用这些权重，保留原有样本或请求权重的归一化。生成式行为评估仍按真实请求计算指标。默认参数为 `--click-weight 2 --purchase-weight 4 --price-change-weight 2`，可显式设为1做无加权对照。行为加权读取完成的Token缓存；Query-only不要求行为权重索引。

Customer Response 默认使用 `--response-format-weight 0.25`，仅降低固定 JSON 格式 Token 的权重。商品 ID、整个动作列表（包括空列表、逗号和结束符）及特殊终止 Token 的权重保持 1。跨越格式与动作边界的合并 Token 也保留权重 1。每条回答先计算 `sum(token_weight × CE) / sum(token_weight)`，再应用请求权重和动作权重；训练、Test 和 checkpoint 选择使用同一口径。Query 和 Seller 不应用此项降权。设为 1 可恢复等 Token 权重，已有 Token 缓存可复用；不同权重设置的 Loss 数值不能直接比较，续训须沿用原设置，改变目标应另开运行。`--loss-breakdown` 将 Response 分为固定格式、点击 ID、购买 ID、列表决策及终止 Token 四组。

样本构造时，同一 Customer 请求的 Query/Response 共用近期历史摘要，较早历史仍按各阶段检索结果分别处理。Seller 候选集行为评估复用按时间推进的历史读取，避免逐目标重读完整历史。竞品分析保留全类目原始统计和入选竞品的原始卡片，模型选择比较角度和有依据的直接替代商品，市场概况和报价位置中的数值关系由现成统计计算。


新的样本选择和权重是新的实验设置。已有按无加权评估损失选优的加权训练不能直接沿用旧的最佳 checkpoint 记录续训。旧Token缓存、训练产物和原候选目录均保留。上述筛选不会调用BGE、上下文模型或Tokenizer；下一步构造时才为保留的监督目标准备上下文。
