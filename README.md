# kernel-opt-agent

一个"在目标芯片上实现算子并自动把它优化到最佳性能"的 Agent。核心是一条闭环:

```
算子规格 (spec)                 ┌──────────────────────────────────────────────────┐
  ├─ 参考实现 (numpy)           │  propose ─→ compile ─→ verify ─→ benchmark ─→ feedback │
  ├─ 函数原型 / ABI             └──────────────────────────────────────────────────┘
  ├─ 基准形状 + 边界形状              ▲                                        │
  └─ FLOPs / 字节数                   └───── 生成器 (模板调参 / 进化搜索 / LLM 改写) ◀─┘
```

1. **baseline**:朴素但正确的实现,作为正确性锚点和加速比分母。
2. **autotune**:对参数化模板(分块、寄存器块、线程数、调度策略、编译选项…)做搜索:
   先用知识库里相似形状的最优配置热启动,再随机采样,最后对 top-k 配置做邻域变异(进化搜索)。
3. **LLM refine**:每轮并发向 LLM 请求 N 个候选(best-of-N),提示词里包含当前最优内核、
   roofline 位置(算强度、离可达峰值的百分比)、gcc 向量化报告、上一轮每个样本的结果
   (编译报错 / 错误元素 / 崩溃信号 / 越界写)。只有**正确且更快**的候选才会成为新的最优。

三个阶段共用同一个评估器,所以"更快"永远意味着"在所有测试形状上都正确的前提下更快"。

## 快速开始

```bash
pip install -e .            # 仅依赖 numpy;需要 gcc(支持 OpenMP)
kopt list-ops               # 列出算子及其模板
kopt run --op matmul --autotune-budget 16
kopt run --op softmax --shape 4096 1024 --autotune-budget 12
kopt show --op matmul       # 打印找到的最优内核
```

输出示例(4 核 x86 虚拟机,AVX-512,512³ float32 GEMM,16 个模板配置、13 秒):

```
roofline: peak FMA throughput ~1013 GFLOP/s (all threads), streaming bandwidth ~133.6 GB/s (cached)
  # trial  latency ms   GFLOP/s     GB/s  speedup  roof%  candidate
-------------------------------------------------------------------
  1    18      0.7753     346.2      4.1  138.82x    34%  evolve[KC=256,MC=32,MR=8,NC=512,NR=32,SCHEDULE=static,THREADS=4,WIDE=0]
  2    17      0.7987     336.1      3.9  134.76x    33%  evolve[KC=256,MC=32,MR=6,NC=1024,NR=32,SCHEDULE=static,THREADS=4,WIDE=0]
  3     5      0.8782     305.7      3.6  122.56x    30%  autotune[KC=256,MC=64,MR=4,NC=512,NR=64,SCHEDULE=static,THREADS=4,WIDE=0]
 ...
```

同一台机器上手工点测 `packed` 模板的 `MR=6,NR=32,MC=128,KC=128,NC=1024,WIDE=1` 达到 552 GFLOP/s(numpy/OpenBLAS 同形状为 672 GFLOP/s);更大的搜索预算或 LLM 阶段会继续逼近。

结果落在 `results/<op>/`:`best.c`(最优内核)、`trials.jsonl`(每次试验的状态与指标、roofline、编译器报告)、
`candidates/`(所有候选源码)、`knowledge_<template>.jsonl`(跨运行知识库)、`summary.json`。

### 常用参数

| 参数 | 作用 |
| --- | --- |
| `--template packed\|blocked` | 选择要调参的模板(`kopt list-ops` 里带 `*` 的是默认) |
| `--autotune-budget N` / `--evolve-fraction 0.5` | 模板搜索总预算,以及其中用于进化变异的比例 |
| `--warm-start 3` | 从知识库取多少个相似形状的最优配置作为种子(0 关闭) |
| `--workers 4` | 并行编译 / 验证的进程数,也是 LLM 并发请求数 |
| `--top-k 3` / `--quick-repeats 5` | 每批只有 top-k 候选获得完整计时;其余用粗筛计时(取最小值抗噪) |
| `--no-roofline` | 跳过峰值探测(FMA 吞吐 + 流式带宽,结果按硬件缓存在 `~/.cache/kopt`) |
| `--llm-rounds 8 --llm-samples 3 --llm-patience 4` | LLM 轮数、每轮 best-of-N 样本数、连续无提升多少轮后停止 |

### 开启 LLM 改写阶段

任何 OpenAI 兼容接口都可以(OpenAI / DeepSeek / Qwen-DashScope / vLLM / Ollama):

```bash
export KOPT_LLM_API_KEY=sk-...
export KOPT_LLM_BASE_URL=https://api.deepseek.com/v1   # 默认 https://api.openai.com/v1
export KOPT_LLM_MODEL=deepseek-chat                    # 默认 gpt-4o
kopt run --op matmul --autotune-budget 16 --llm-rounds 8 --llm-samples 3
```

没有配置密钥时 LLM 阶段会被跳过并给出提示,其余阶段照常运行。

## 评估器如何保证"最佳性能"是可信的

| 风险 | 处理方式 |
| --- | --- |
| 生成的内核段错误 / 死循环 | 内核在独立子进程 (`kopt_agent/runner.py`) 里执行,崩溃变成 `runtime_error`(附信号名和提示),超时变成 `timeout`,Agent 本身不受影响 |
| 只写了一部分输出 | 输出缓冲区预填 NaN,任何未写入的元素都会被判为 `incorrect` |
| 分块循环不处理余数、越界写 | 每个张量两侧各放 4 KiB 金丝雀页,运行后校验;同时用质数/极小形状(如 `7x13x5`、`1x1x1`)做正确性测试,并且小形状先跑 |
| 修改了 `const` 输入 | 运行后逐字节比对输入快照 |
| 精度被"快速数学"破坏 | 容差 (`atol`/`rtol`) 与参考实现 (float64 numpy) 比对;`-ffast-math`、`-mprefer-vector-width=512` 只是搜索空间里的可选项 |
| 计时噪声 | 完整计时:预热 + 多次重复取中位数;粗筛计时取最小值(虚拟机上偶发的抢占会污染中位数);只有完整计时的结果能成为最优 |
| 重复工作 | 编译产物按源码 + 编译选项哈希缓存;搜索空间无放回采样;LLM 回复与历史重复时直接反馈不评估 |

## 提高效率与性能上限的机制

| 机制 | 位置 | 效果 |
| --- | --- | --- |
| 分级评估 | `evaluator.py: evaluate_batch` | 并行编译 + 验证 → 串行粗筛计时 → 仅 top-k 完整计时,慢候选不再消耗完整基准时间 |
| 进化搜索 | `generators/template.py: mutate` + `agent.py` | 在 top 配置的邻域内移动(分块翻倍/减半),比纯随机采样更快收敛 |
| 跨运行知识库 | `knowledge.py` | 按"同硬件 → 形状最近 → GFLOP/s 最高"排序,新形状直接从历史最优配置起步 |
| roofline 反馈 | `roofline.py` | 一次性探测机器 FMA 峰值与流式带宽,给每个试验标注 compute/memory-bound 与达成比例 |
| 编译器反馈 | `backends/cpu_c.py` | 解析 `-fopt-info-vec-missed/optimized`,告诉 LLM 哪一行循环没被向量化以及原因 |
| best-of-N | `generators/llm.py: propose_many` | 每轮并发请求 N 个候选,每个样本被引导到不同优化方向(寄存器分块 / 打包 / 并行划分 / 指令级 SIMD / 访存) |
| BLIS 风格 GEMM 模板 | `ops/matmul_packed.py` | A/B 面板打包 + MR×NR 寄存器微内核 + 三级缓存分块,把 matmul 起点从 ~160 提升到 ~350–550 GFLOP/s |

## 代码结构

```
kopt_agent/
  spec.py          OperatorSpec / TensorSpec / TestCase:算子契约(原型、参考实现、形状、容差、FLOPs)
  candidate.py     Candidate:一份候选源码 + 来源 + 参数 + 额外编译选项
  evaluator.py     编译 → 多形状验证 → 计时(单个 / 分级批量),产出 TrialResult
  agent.py         OptimizationAgent:baseline → 热启动/随机/进化调参 → best-of-N LLM 改写
  history.py       试验记录、最优追踪、排行榜、结果落盘
  knowledge.py     跨运行知识库与热启动
  roofline.py      峰值探测与 roofline 分析
  generators/
    template.py    参数化模板:搜索空间、约束、无放回采样、邻域变异
    llm.py         OpenAI 兼容接口的内核改写器(best-of-N、聚合反馈、roofline + 编译器报告)
  backends/
    base.py        Backend 协议:compile / run / hardware_summary / language_guidance
    cpu_c.py       gcc + OpenMP 后端,子进程隔离执行,向量化报告解析
  runner.py        子进程内核执行器(ctypes 加载 .so,对齐 + 金丝雀 + 计时)
  hardware.py      读取 CPU 型号 / SIMD 指令集 / 缓存,供报告与 LLM 使用
ops/
  matmul.py        float32 GEMM:朴素基线 + 两个模板(packed*、blocked)
  matmul_packed.py BLIS 风格打包 + 寄存器微内核模板
  softmax.py       行 softmax:朴素基线 + 模板(线程/调度/在线算法/快速数学)
tests/             评估器失败路径分类、分级批量、模板采样与变异、知识库、roofline、LLM 回复解析
```

## 如何扩展

**新增一个算子**:在 `ops/` 下新建模块,提供 `build(shape) -> OperatorBundle`(包含 `OperatorSpec`、基线源码、一个或多个模板),然后在 `ops/__init__.py` 的 `OP_REGISTRY` 注册。关键是把边界形状(非分块倍数、维度为 1、极大值输入等)写进 `edge_shapes` / `input_generator`,评估器会替你把所有投机取巧的候选筛掉。

**移植到 GPU / NPU**:实现一个新的 `Backend` 子类即可,Agent、评估器、生成器不需要改动:

- `compile`:调用 `nvcc` / `hipcc` / `bisheng`(昇腾 Ascend C)/ Triton JIT,产出可加载的产物;把编译器给出的寄存器占用、spill、occupancy 等信息放进 `CompileResult.optimization_report`。
- `run`:在隔离进程里搬运数据、同步、用设备事件计时,把输出拷回主机。
- `hardware_summary` / `language_guidance`:告诉 LLM 目标是 CUDA C++ 还是 Ascend C、SM 数量 / AI Core 数量、共享内存 / UB 大小、允许的头文件与 API。
- roofline 探针(`roofline.py`)目前是 CPU C 源码;GPU 后端可以换成自己的峰值内核,或直接填入厂商标称值。

**换搜索策略**:`agent.py` 里的三个 `_phase_*` 是纯函数式的调用顺序;可以把随机 + 进化换成贝叶斯优化,或把 LLM 阶段改成"保留 top-k 候选做进化"。评估器和历史记录接口保持不变。

## 运行测试

```bash
pip install -e ".[dev]"
pytest -q
```
