# 服务器环境与目录

服务器工作区为 `/root/autodl-tmp/zhz/llm_simulation`。代码仓库为其下 `ecommerce-agent-simulation/`。运行环境统一放在 `environments/`，不要把依赖安装到性能实验、tmp 或 cache。

| 路径 | 用途 |
|---|---|
| `environments/train` | 训练、仿真、评估与传统定价基线，Python 3.12 / Torch 2.8 / Transformers 5.3 |
| `environments/vllm` | vLLM 0.20.2 服务环境，与训练环境隔离 |
| `environments/python` | 管理式 Python 解释器 |
| `environments/tools` | uv 与性能分析工具 |
| `cache` | 可再生下载、编译与字节码缓存，不再存放必须的运行依赖 |
| `tmp` | 一次性文件与临时进程通信；清理时检查活动 socket |
| `state` | Shell 历史与环境迁移记录 |

```bash
source /root/autodl-tmp/zhz/llm_simulation/env.sh
./agent_simulation/python.sh -m agent_simulation.training.run --help
./agent_simulation/python.sh -m agent_simulation.simulation.run --help
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server --help
```

`env.sh` 激活正式 train 环境并进入仓库；`SANDBOX_PYTHON` 和 `VLLM_PYTHON` 分别指定训练/仿真与服务解释器。无需额外设置指向 `runs-kernel-bench` 或 `cache/kuaisearch-deps` 的 PYTHONPATH。

迁入正式 train 环境的依赖包括 FLA、causal-conv1d、einops、scikit-learn、SciPy 及其支持包。CUDA 扩展仍使用原有与 Torch 2.8 匹配的文件，未混入 vLLM 的独立 Torch 环境。版本清单保存在 `environments/train-packages.json` 和 `vllm-packages.json`，这些机器资产不进 Git。

旧路径 `simulation`、`vllm-env`、`python` 已移除，不保留兼容链接。环境内启动器、激活脚本、Python 编译配置、演示与历史实验启动脚本均直接使用 `environments/` 路径。未修改历史实验的模型、数据和训练参数。`agent_simulation/python.sh` 默认只使用正式 train 环境，也允许显式设置 `SANDBOX_PYTHON`，不再回退旧环境。

原性能测试的脚本、日志和指标保留在 `result/kernel-benchmark-20261110/`；原安装 wheel 和已迁移的包目录不再需要。模型、数据、检查点、检索向量和其他实验结果保留。迁移验证仅覆盖入口、依赖和小模型训练/双角色仿真，不表示重新训练了完整模型。
