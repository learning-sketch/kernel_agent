# kernel-opt-agent

一个"在目标芯片上实现算子并自动把它优化到最佳性能"的 Agent 骨架。核心是一条闭环:

```
算子规格 (spec)                 ┌──────────────────────────────────────────────────┐
  ├─ 参考实现 (numpy)           │  propose ─→ compile ─→ verify ─→ benchmark ─→ feedback │
  ├─ 函数原型 / ABI             └──────────────────────────────────────────────────┘
  ├─ 基准形状 + 边界形状              ▲                                        │
  └─ FLOPs / 字节数                   └───────── 生成器 (模板自动调参 / LLM 改写) ◀─┘
```

1. **baseline**:先跑一个朴素但正确的实现,作为正确性锚点和加速比分母。
2. **autotune**:对手写的参数化模板(分块大小、线程数、调度策略、编译选项…)做搜索空间采样,每个配置都重新编译、验证、计时。
3. **LLM refine**:把"当前最优内核 + 实测指标 + 硬件信息 + 上一轮失败原因(编译报错 / 错误元素 / 崩溃信号 / 越界写)"喂给 LLM,让它提出一个全新的内核;只有**正确且更快**的候选才会成为新的最优。

三个阶段共用同一个评估器,所以"更快"永远意味着"在所有测试形状上都正确的前提下更快"。

## 快速开始

```bash
pip install -e .            # 仅依赖 numpy;需要 gcc(支持 OpenMP)
kopt list-ops
kopt run --op matmul --autotune-budget 12
kopt run --op softmax --shape 4096 1024 --autotune-budget 12
kopt show --op matmul       # 打印找到的最优内核
```

输出示例(4 核 x86 虚拟机,512³ float32 GEMM):

```
  # trial  latency ms   GFLOP/s     GB/s  speedup  candidate
------------------------------------------------------------
  1     7      1.6921     158.6      1.9   63.56x  autotune[KB=128,MB=8,NB=256,SCHEDULE=dynamic,THREADS=4]
  2    10      1.8062     148.6      1.7   59.54x  autotune[KB=64,MB=8,NB=512,SCHEDULE=dynamic,THREADS=4]
 ...
 11     1    107.5456       2.5      0.0    1.00x  baseline[fa1c6ad2fe9c]
```

结果落在 `results/<op>/`:`best.c`(最优内核)、`trials.jsonl`(每次试验的状态与指标)、`candidates/`(所有候选源码)、`summary.json`。

### 开启 LLM 改写阶段

任何 OpenAI 兼容接口都可以(OpenAI / DeepSeek / Qwen-DashScope / vLLM / Ollama):

```bash
export KOPT_LLM_API_KEY=sk-...
export KOPT_LLM_BASE_URL=https://api.deepseek.com/v1   # 默认 https://api.openai.com/v1
export KOPT_LLM_MODEL=deepseek-chat                    # 默认 gpt-4o
kopt run --op matmul --autotune-budget 12 --llm-rounds 8 --llm-patience 4
```

没有配置密钥时 LLM 阶段会被跳过并给出提示,其余阶段照常运行。

## 评估器如何保证"最佳性能"是可信的

| 风险 | 处理方式 |
| --- | --- |
| 生成的内核段错误 / 死循环 | 内核在独立子进程 (`kopt_agent/runner.py`) 里执行,崩溃变成 `runtime_error`(附信号名和提示),超时变成 `timeout`,Agent 本身不受影响 |
| 只写了一部分输出 | 输出缓冲区预填 NaN,任何未写入的元素都会被判为 `incorrect` |
| 分块循环不处理余数、越界写 | 每个张量两侧各放 4 KiB 金丝雀页,运行后校验;同时用质数/极小形状(如 `7x13x5`、`1x1x1`)做正确性测试 |
| 修改了 `const` 输入 | 运行后逐字节比对输入快照 |
| 精度被"快速数学"破坏 | 容差 (`atol`/`rtol`) 与参考实现 (float64 numpy) 比对;`-ffast-math` 等只是搜索空间里的一个可选项 |
| 计时噪声 | 预热 + 多次重复取中位数;报告 GFLOP/s 与有效带宽,便于对照 roofline |
| 重复编译相同候选 | 按源码 + 编译选项哈希缓存编译产物;搜索空间无放回采样;LLM 回复与历史重复时直接反馈而不评估 |

## 代码结构

```
kopt_agent/
  spec.py          OperatorSpec / TensorSpec / TestCase:算子契约(原型、参考实现、形状、容差、FLOPs)
  candidate.py     Candidate:一份候选源码 + 来源 + 参数 + 额外编译选项
  evaluator.py     编译 → 多形状验证 → 计时,产出 TrialResult(状态、延迟、GFLOP/s、GB/s、最大误差)
  agent.py         OptimizationAgent:baseline → autotune → LLM refine 三阶段调度
  history.py       试验记录、最优追踪、排行榜、结果落盘
  generators/
    template.py    参数化模板 + 搜索空间(带约束的枚举 / 无放回随机采样)
    llm.py         OpenAI 兼容接口的内核改写器(提示词包含硬件、指标、失败反馈、优化手册)
  backends/
    base.py        Backend 协议:compile / run / hardware_summary / language_guidance
    cpu_c.py       gcc + OpenMP 后端,子进程隔离执行
  runner.py        子进程内核执行器(ctypes 加载 .so,对齐 + 金丝雀 + 计时)
  hardware.py      读取 CPU 型号 / SIMD 指令集 / 缓存,供报告与 LLM 使用
ops/
  matmul.py        float32 GEMM:朴素基线 + 分块模板(MB/NB/KB/线程/调度)
  softmax.py       行 softmax:朴素基线 + 模板(线程/调度/在线算法/快速数学)
tests/             评估器失败路径分类、模板采样、LLM 回复解析
```

## 如何扩展

**新增一个算子**:在 `ops/` 下新建模块,提供 `build(shape) -> OperatorBundle`(包含 `OperatorSpec`、基线源码、可选模板),然后在 `ops/__init__.py` 的 `OP_REGISTRY` 注册。关键是把边界形状(非分块倍数、维度为 1、极大值输入等)写进 `edge_shapes` / `input_generator`,评估器会替你把所有投机取巧的候选筛掉。

**移植到 GPU / NPU**:实现一个新的 `Backend` 子类即可,Agent、评估器、生成器不需要改动:

- `compile`:调用 `nvcc` / `hipcc` / `bisheng`(昇腾 Ascend C)/ Triton JIT,产出可加载的产物。
- `run`:在隔离进程里搬运数据、同步、用设备事件计时,把输出拷回主机。
- `hardware_summary` / `language_guidance`:告诉 LLM 目标是 CUDA C++ 还是 Ascend C、SM 数量 / AI Core 数量、共享内存 / UB 大小、允许的头文件与 API。
- 想给 LLM 更强的反馈,可在 `RunResult` 里附带 profiler 指标(如 `ncu` 的达成占用率、访存效率、bank conflict),并在 `llm.py` 的提示词中加入这些字段。

**换搜索策略**:`agent.py` 里的三个 `_phase_*` 是纯函数式的调用顺序;可以把 autotune 换成贝叶斯优化,或把 LLM 阶段改成"保留 top-k 候选做进化搜索"。评估器和历史记录接口保持不变。

## 运行测试

```bash
pip install -e ".[dev]"
pytest -q
```
