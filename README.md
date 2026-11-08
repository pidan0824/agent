# E-commerce Agent Simulation

使用 KuaiSearch Lite 构造 Customer 与 Seller Agent，通过共享市场环境运行搜索、定价、点击与购买。Full 和 Lite 共用数据处理脚本，本地使用 Lite 核验。

数据划分直接沿用官方 train/test：Customer 保留原始 `split`，Seller 保留经营记录的 `source_split`。不额外切验证集，也不需要恢复划分。

## 当前代码与数据

本目录是电商 Agent 项目的正式入口。日常使用固定路径，代码和数据快照由 Git 记录。

| 交付项 | 路径 |
|---|---|
| Customer、Seller 和仿真代码 | `agent_simulation/` |
| SFT 标签池和示例 | [sft](data/kuaisearch/lite/sft) |
| 模型默认参数与近期历史窗口 | [model_policy.json](agent_simulation/inference/model_policy.json) |
| KuaiSearch 数据源配置 | [Lite](data/kuaisearch/lite/config.json) · [Full](data/kuaisearch/full/config.json) |
| 2% 抽样配置 | [lite_2pct_sampling.json](agent_simulation/datasets/lite_2pct_sampling.json) |
| seed=42 的 2% 候选，按展示商品配对 Customer 与 Seller | [candidates](data/kuaisearch/lite/candidates) |

Git 包含四个标签 Parquet、四个候选 Parquet，以及各自的清单和校验值。完整训练输入由标签和事实表按需构造。数据清单中的绝对路径绑定当前机器上的事实目录。重新构造样本还需要本机的 KuaiSearch 源事实、模型和 Tokenizer。原始数据、模型权重和虚拟环境使用本地依赖，运行结果和缓存由 `runs/` 管理。迁移到其他机器时需准备相同的依赖，并按新路径重建数据清单。

本机的 `models`、`.venv-memory` 和 `.runtime` 复用旧沙箱的模型、Python 环境与 DuckDB。`data/kuaisearch/lite/source/` 集中放置本机链接。`catalog` 和 `splits` 指向同级 `work` 中官方划分对应的事实目录，`raw` 和 `seller_tables` 指向旧沙箱的原始数据与经营表。

已保存的样本可直接阅读。现有机器可从项目根目录通过 `./agent_simulation/python.sh` 运行构造和测试入口。

## 阅读入口

建议先按[真实数据端到端调试指南](docs/debugging.md)打开 [VS Code 工作区](ecommerce-agent-simulation.code-workspace)，用断点看一轮 Seller → Customer，再沿模块深入。

[架构设计](docs/architecture.md)说明两个角色各自能看见什么、哪些动作有 SFT 监督，以及一个 Tick 如何结算。

| 想了解什么 | 文档 |
|---|---|
| Agent、Memory 与仿真如何连接 | [架构设计](docs/architecture.md) |
| 代码入口与运行方法 | [Agent 使用说明](agent_simulation/README.md) |
| 原始表、经营事实、数据划分和重建 | [市场数据代码](agent_simulation/data) |
| SFT 标签、完整输入、Token 和样本导出 | [SFT 样本准备](agent_simulation/README.md#sft-data) |
| 模型调用、后端与约束解码 | [Inference](agent_simulation/README.md#inference) |
| LoRA、test 选模和恢复训练 | [SFT 训练](agent_simulation/README.md#training) |
| 查询接口、动作与权限 | [工具说明](agent_simulation/README.md#tools) |
| Memory、检索及共用代码 | [共用能力](agent_simulation/README.md#agent-core) |
| 商品怎么找、怎么排序 | [商品搜索与排序](agent_simulation/README.md#platform) |
| 行为预测与轨迹评价 | [评估入口](agent_simulation/README.md#evaluation) |

## 当前可查看的结果

- [从一次 Episode 的运行库生成时间轴和动画](agent_simulation/README.md#仿真运行与时间线)。
- 本轮核验日志和报告位于同级 `../audit/`；旧图文报告反映各自生成时的版本。

Lite 的完整监督标签池已整理，模型输入按需展开。当前联调用随机抽取或 BM25 标题匹配来找商品。商品搜索与排序模型已接入，负责找候选并打分，实际运行还需要训练后的权重。自定义重排接口已接入，Bandit 尚未实现。由程序准备检索上下文，模型输出最终动作。长期记忆的读写链路已接通，基础 3B 的总结事实质量仍需验证。

## 项目目录

| 目录 | 职责 |
|---|---|
| `agent_simulation/agents/` | Customer、Seller、输入装配、Memory 与检索 |
| `agent_simulation/datasets/` | SFT 标签、输入构造、按需读取和 Token 化 |
| `agent_simulation/environment/` | 市场状态、操作权限、价格生效和反馈结算 |
| `agent_simulation/simulation/` | 逻辑时钟、Tick 调度、运行入口和时间线回放 |
| `agent_simulation/platform/` | 找候选商品、打分、决定展示顺序 |
| `agent_simulation/inference/` | 模型客户端、推理后端和输出解码 |
| `agent_simulation/training/` | 模型训练、验证与检查点 |
| `agent_simulation/evaluation/` | 行为预测、轨迹和分布评价 |
| `agent_simulation/data/` | KuaiSearch 事实表、划分、来源校验和只读查询 |
| `data/kuaisearch/` | Lite 发布数据、抽样候选和相关性参考对 |
| `docs/` | 阅读文档和展示素材 |
| `runs/` | 本地仿真、训练及评估输出和可重建缓存 |
| `agent_simulation/platform/`、`models/` | 平台召回、排序基线与本地模型资产 |

测试命令见[测试说明](agent_simulation/README.md#tests)，依赖及源码许可见 [源码来源和许可](docs/third-party-notices.txt)。
