# 本机 vLLM 启动与首请求诊断

2026-11-22，本机 RTX PRO 6000 Blackwell / vLLM 0.20.2 的实际请求多次停在 FlashAttention 2 的 varlen_fwd 原生调用。去掉 JSON 约束仍复现，不能归因为结构化 JSON。9B 切换 TRITON_ATTN 后，同一 Query 请求在 22.32 秒内返回 13 token。此证据定位到本机后端路径，不等于已证明所有驱动／硬件上的通用缺陷。

另外，容器可见 208 个逻辑 CPU，但 cpu.max=5000000/100000，即约 50 核配额。采样 5 秒期间每个调度周期均触发限流，vLLM 主线程只得到约 0.13 秒 CPU。模块加载、Triton 首次编译会因此拖慢。没有修改共享容器配额或其他任务。

## 启动

在项目根目录运行 ./agent_simulation/serve.sh Qwen3.5-9B 8011 1；或将模型改为 Qwen3-32B。小样例依次使用同一张 GPU。

启动器固定使用 TRITON_ATTN，限制本服务线程／编译并发，沿用项目 env.sh 的持久缓存路径。仅绑定本机；不会自动停止已有服务。先核对指定 GPU 可用。

## 真正就绪

另一个终端激活项目环境后运行：
./agent_simulation/python.sh -m agent_simulation.inference.readiness --url http://127.0.0.1:8011/v1 --model Qwen3.5-9B

该检查分别完成普通和 JSON-schema 生成，验证输出并报告耗时。/health 返回成功只说明服务可接收请求，不代表所有首次内核路径已预热。业务任务应在该检查通过后启动。300 秒是启动探针上限，不用于掩盖业务请求异常。

保留 cache/triton、cache/cuda、cache/torchinductor，避免每次清理后重新编译。首次编译仍可能较慢；不要仅凭 GPU 0% 判断服务死锁。结合 Python 栈、请求队列、输出 token 计数区分等待、编译和实际执行。出现失败时保留日志，核对 PID／命令后只停止本项目启动的进程组。

最终验证：9B 普通／JSON 探针分别 3.11 / 5.48 秒；32B 分别 2.21 / 1.57 秒。两模型均完成 3 Query + 4 EKB，共 14 条，服务自动清理。该结论是本机后端切换后的恢复验证，不宣称已证明底层驱动缺陷。
