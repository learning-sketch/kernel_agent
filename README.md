# kernel-opt-agent

一个"在目标芯片上实现算子并自动把它优化到最佳性能"的 Agent。核心是一条闭环:

```
算子规格 (spec)                 ┌──────────────────────────────────────────────────┐
  ├─ 参考实现 (numpy, fp64)     │  propose ─→ compile ─→ verify ─→ benchmark ─→ feedback │
  ├─ 函数原型 / ABI / dtype     └──────────────────────────────────────────────────┘
  ├─ 负载 (shape, 调用次数)          ▲                 ▲                      │
  ├─ 基准形状 + 边界形状              │        roofline 裁决 (到顶即停 / 导向)   │
  └─ FLOPs / 字节数                   └───── 生成器 (模板调参 / 进化搜索 / LLM 改写) ◀─┘
```

1. **baseline**:朴素但正确的实现,作为正确性锚点和加速比分母。
2. **autotune**:对参数化模板(分块、寄存器块、线程数、调度策略、编译选项…)做搜索:
   先用知识库里相似形状的最优配置热启动,再随机采样,最后对 top-k 配置做邻域变异(进化搜索)。
3. **LLM refine**:每轮并发向 LLM 请求 N 个候选(best-of-N),提示词里包含当前最优内核、
   roofline 位置(算强度、离可达峰值的百分比)、gcc 向量化报告、上一轮每个样本的结果
   (编译报错 / 错误元素 / 崩溃信号 / 越界写)。只有**正确且更快**的候选才会成为新的最优。

三个阶段共用同一个评估器,所以"更快"永远意味着"在所有测试形状上都正确的前提下更快"。
每个阶段结束后都会用 roofline 裁决当前最优:已达可达上限的 85%(可调)就宣布"到顶"并停止烧预算;
否则把瓶颈类型(compute / memory / dispatch-overhead)对应的优化手法喂给 LLM 与搜索。

## 快速开始

```bash
pip install -e .            # 仅依赖 numpy;需要 gcc(支持 OpenMP)
kopt list-ops               # 列出算子及其模板
kopt run --op matmul --autotune-budget 16
kopt run --op softmax --shape 4096 1024 --autotune-budget 12
kopt run --op matmul --dtype bf16 --autotune-budget 12          # fp32 / fp16 / bf16 / fp64
kopt run --op matmul --dtype bf16 --output-dtype fp32 --accumulate-dtype fp32   # 混合精度:bf16 输入、fp32 输出与累加
kopt run --op matmul --workload profile.json --autotune-budget 16   # 负载驱动:按调用频次加权
kopt run --op matmul_bias_relu --autotune-budget 12             # 融合算子,末尾报告"融合 vs 分开跑"
kopt workload --trace trace.csv --op matmul --dims M N K --profile profile.json   # 从 profiler trace 生成负载
kopt show --op matmul       # 打印找到的最优内核
kopt export --op matmul     # (重新)生成 results/matmul/bundle/ 可集成包
```

`profile.json` 是某算子在一次端到端 trace 里的 (shape, 调用次数) 签名集合:

```json
{"shapes": [{"shape": [512, 512, 512], "count": 120}, {"shape": [64, 512, 512], "count": 900}]}
```

它可以手写,也可以用 `kopt workload` 从 CSV / JSONL / JSON trace 聚合出来:一行一次调用(或带 `count`/`calls` 列),
形状列写 `512x512x512` / `[512, 512, 512]`,或者按 `--dims M N K` 指定维度列;`--op` 只保留该算子的行,非法行会被跳过并报告。

给了负载后,每个候选在所有形状上计时,排行榜同时给出逐形状与按 count 加权的两栏,**最优按加权总时间选**;
未指定 `--shape` 时基准(展示)形状取 count×FLOPs 最大的那个。

输出示例(4 核 x86 虚拟机,AVX-512,fp32 GEMM,负载 = 3 个形状 / 350 次调用,4 个模板配置、6.5 秒):

```
roofline: peak FMA throughput ~1021 GFLOP/s (all threads), streaming bandwidth ~130.6 GB/s, launch floors: serial call ~0.46 us, parallel region ~1.75 us
workload: 350 calls over 3 shapes - objective is the call-weighted total time: 128x128x128 x40, 64x256x64 x300, 256x64x128 x10
[evolve g1] trial 6 dead end: WIDE 1->0 -> regression +35% vs parent

  # trial  weighted ms primary ms  GFLOP/s  speedup    A/B  roof% grade            candidate
--------------------------------------------------------------------------------------------
  1     4       6.4412     0.0164    128.1   31.65x 31.77x    22% within-N-ULP     autotune[KC=128,MC=128,MR=4,NC=256,NR=32,SCHEDULE=static,THREADS=1,WIDE=1]
  2     3       8.8146     0.0229     91.5   23.13x 23.27x    16% within-N-ULP     autotune[KC=128,MC=128,MR=4,NC=1024,NR=16,SCHEDULE=dynamic,THREADS=1,WIDE=1]
 ...
  6     1     203.8726     0.4647      4.5    1.00x      -     1% within-N-ULP     baseline[6d6f92867f05]

per-shape breakdown of trial 4:
shape                  calls  latency ms  baseline ms  speedup  weighted ms  roofline
128x128x128               40      0.0304       1.4109   46.41x       1.2161  compute-bound, 19% of ceiling
64x256x64                300      0.0164       0.4647   28.37x       4.9131  compute-bound, 22% of ceiling
256x64x128                10      0.0312       0.8036   25.75x       0.3120  compute-bound, 18% of ceiling
weighted total           350                 203.8726   31.65x       6.4412

roofline verdict: compute-bound; headroom 4.74x to the attainable 1.3580 ms. Compute-bound: the FMA units are the limiter. Work on register tiling ...
```

精度敏感算子(softmax)的排行榜多一列 `grade`;等级为 `reduced-precision` 的候选会带上
`[excluded: reduced precision]` 标记且不参与选优,如果它比当前最优还快,排行榜末尾会追加一行
`precision <-> speed: trial N is X.XXx faster than the best but graded reduced-precision (scaled ULP error ..., bitwise match ...%). Re-run with --allow-reduced-precision to accept it.`

小算子会被裁决为 dispatch-overhead-bound 并建议融合而不是抠内层循环:

```
verdict after baseline: overhead-bound; headroom 1.29x to the attainable 0.0005 ms. Dispatch-overhead-bound: the operator is so small that launching the kernel / spawning the parallel region costs more than the work. Do not micro-optimize the inner loop. Fuse this operator with its neighbours, batch several calls into one launch, or skip the parallel region below a size threshold.
```

同一台机器上手工点测 `packed` 模板的 `MR=6,NR=32,MC=128,KC=128,NC=1024,WIDE=1` 在 512³ 上达到 552 GFLOP/s(numpy/OpenBLAS 同形状为 672 GFLOP/s);更大的搜索预算或 LLM 阶段会继续逼近。

结果落在 `results/<op>/`:`best.c`(最优内核)、`parity_test.py`(随最优内核生成的独立对拍/回归测试)、
`trials.jsonl`(每次试验的状态、指标、数值等级、逐形状结果、roofline、结构化 profile)、
`candidates/`(所有候选源码)、`knowledge_<template>.jsonl`(跨运行知识库:最优配置 + 死路)、`summary.json`(含裁决、死路清单、融合收益)。

### 可集成包(价值兑现口)

运行结束时夺冠内核被打成自包含的 `results/<op>/bundle/`,供下游算子库直接接入;`kopt export --op <op>` 可从 `summary.json` + `best.c` 重新生成:

| 文件 | 内容 |
| --- | --- |
| `kernel.c` | 夺冠内核源码原样 |
| `kernel.h` | 原型、launch ABI(指针空间 / stream 参数 / 是否同步)、张量布局约定、`extern int kopt_fast_path_active`(若有快路) |
| `manifest.json` | 机器可读契约:签名、逐张量 dtype + 累加 dtype、launch ABI、选中的模板参数与编译选项、数值等级(scaled ULP / 逐位一致率 / 容差)、逐形状性能与 A/B、roofline 裁决、硬件 |
| `build.sh` | 验证时使用的精确编译命令 |
| `parity_test.py` | 独立对拍测试(参考 vs 内核、全部形状、特殊值、输入不可变、毒值预填);通过 `pip install` 本仓库或 `KOPT_REPO=<checkout>` 找到参考实现 |
| `README.md` | 契约、数值、性能表、构建与复验方法 |

### 常用参数

| 参数 | 作用 |
| --- | --- |
| `--template packed\|blocked` | 选择要调参的模板(`kopt list-ops` 里带 `*` 的是默认) |
| `--autotune-budget N` / `--evolve-fraction 0.5` | 模板搜索总预算,以及其中用于进化变异的比例 |
| `--warm-start 3` | 从知识库取多少个相似形状的最优配置作为种子(0 关闭) |
| `--workers 4` | 并行编译 / 验证的进程数,也是 LLM 并发请求数 |
| `--top-k 3` / `--quick-repeats 5` | 每批只有 top-k 候选获得完整计时;其余用粗筛计时(取最小值抗噪) |
| `--dtype fp32\|fp16\|bf16\|fp64` | 输入元素类型;参考实现始终 fp64,再按**输出** dtype 舍入后比对 |
| `--output-dtype` / `--accumulate-dtype` | 混合精度:输出类型(默认同输入)与累加精度(默认至少 fp32,fp64 I/O 时为 fp64);每个 `TensorSpec` 自带 dtype,融合算子的 bias 跟输出同类型 |
| `--workload profile.json` | 负载驱动:多形状计时,目标函数是按调用次数加权的总时间 |
| `--allow-reduced-precision` | 允许数值等级为 `reduced-precision` 的候选在精度敏感算子上当选(默认不允许) |
| `--ceiling-fraction 0.85` / `--no-stop-at-ceiling` | roofline 裁决"到顶"的阈值;是否到顶即停 |
| `--no-roofline` | 跳过峰值探测(FMA 吞吐、流式带宽、串行调用地板、并行区发射地板,结果按硬件缓存在 `~/.cache/kopt`) |
| `--no-fusion-report` | 融合算子不再额外测"分开跑"的时间 |
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
| 生成的内核段错误 / 死循环 | 内核在独立子进程 (`kopt_agent/runner.py`) 里执行,崩溃变成 `runtime_error`(附信号名和提示),超时变成 `timeout`,Agent 本身不受影响;子进程加载任何内核前先把 `RLIMIT_CORE` 置 0,故意崩掉的候选不会甩出几百 MB 的 core 文件(`.gitignore` 亦忽略 `core`、`core.*`) |
| 混合精度内核"偷偷"用窄类型累加 | 规格里显式写明 `accumulate_dtype`,提示词与模板按 `in_t / out_t / acc_t` 生成;参考按输出 dtype 舍入后打分,窄累加表现为更差的 scaled-ULP 等级(`within-N-ULP` → `reduced-precision`)甚至 `incorrect` |
| 只写了一部分输出 / 依赖输出缓冲原有内容 | 验证跑两遍:先预填 NaN(0xFF),再预填毒值 `0x5C`;未写入的元素、残留毒值、两次结果不一致且第二次超出容差(即"往输出里累加"),全部判 `incorrect` |
| "假成功":原样拷贝输入、全零输出 | 输出与同形状输入逐位相同、或全零而参考不为零,直接给出诊断消息判 `incorrect` |
| 只在基准形状上能跑 | 失败后继续跑完所有形状统计覆盖率;基准形状通过但边界形状失败的候选标为 `shape-specialized`(禁形状白名单),并把这一根因喂回 LLM / 知识库 |
| 快路"其实没走" | 候选若声明 `// fast_path: <谓词>`(模板用 `fast_path=` 钩子),必须导出 `int kopt_fast_path_active`;评估器断言谓词为真的形状上标志确实置 1,谓词为假的形状上回退仍正确 |
| 分块循环不处理余数、越界写 | 每个张量两侧各放 4 KiB 金丝雀页,运行后校验;同时用质数/极小形状(如 `7x13x5`、`1x1x1`、`24x48x40`)做正确性测试,并且小形状先跑 |
| 修改了 `const` 输入 | 运行后逐字节比对输入快照 |
| 低精度 dtype 的误判 | 参考实现 fp64 算完 → 按目标 dtype 舍入 → 再比对;16-bit 判据是"按输出量级缩放的 ULP + 逐位一致率",不用逐元素 ULP(近零处会把 ULP 放大成假报警) |
| "更快但降精度"悄悄夺冠 | 每个候选打数值等级 `bitwise-equal` / `within-N-ULP` / `reduced-precision`(阈值 = max(dtype 策略, 2×基线自身误差));精度敏感算子(如 softmax)默认只在等精度候选里选最快,排行榜并列展示"精度↔速度"取舍 |
| 计时噪声 / 漂移 | 完整计时里最优与基线**同进程交替**测(每次计时前各补一次不计时调用,避免线程唤醒偏差);中位数落在"快模式"之外超过 1/4 样本时改用快模式中位数并标注 noisy;粗筛取最小值;只有完整计时的结果能成为最优 |
| 主机受限被误读为内核慢 | 分别报 host(wall)与 device(kernel)时间;报 cpu/wall 比 → 线程利用率,低于 50% 标记 host-bound;`RunResult.timing_source` 标明计时来源(`host_wall` / `device_timer`),H2D / D2H 时间单独报在 `h2d_ms` / `d2h_ms`,永不计入内核时间 |
| 重复工作 / 重蹈覆辙 | 编译产物按源码 + 编译选项哈希缓存;搜索空间无放回采样;LLM 回复与历史重复时直接反馈不评估;知识库里的死路方向在变异时被剪枝 |

## 提高效率与性能上限的机制

| 机制 | 位置 | 效果 |
| --- | --- | --- |
| 负载驱动的目标函数 | `spec.py: WorkloadProfile` + `evaluator.py` + `history.py` | 优化的是 trace 里按调用次数加权的总时间,而不是单个基准形状;逐形状 + 加权两栏排行 |
| 分级评估 | `evaluator.py: evaluate_batch` | 并行编译 + 验证 → 串行粗筛计时 → 仅 top-k 完整计时,慢候选不再消耗完整基准时间 |
| 进化搜索 + 死路剪枝 | `generators/template.py: mutate` + `knowledge.py: is_pruned` | 在 top 配置的邻域内移动;同一硬件上失败 ≥2 次且从未带来提升的方向不再尝试 |
| 跨运行知识库(正 + 负) | `knowledge.py` | 正知识:同硬件 → 形状最近 → GFLOP/s 最高的配置热启动;负知识:回退方向 + 根因标签(编译错误 / 越界 / 形状特化 / 退步 N% 且丢失向量化 / 线程空转 / 访存变多) |
| roofline 裁决 | `roofline.py: analyze / verdict_for` + `agent.py: _at_ceiling` | 探测 FMA 峰值、流式带宽、串行调用地板与并行区发射地板;分类 compute / memory / dispatch-overhead-bound;给出可达时间;到顶即停;按瓶颈导向手法(overhead→融合/批处理,memory→带宽/向量化,compute→分块/ILP) |
| 融合算子 | `ops/matmul_bias_relu.py` + `agent.py: _phase_fusion_report` | `fused_stages` + 单一融合参考 + 单一签名;packed GEMM 模板带 `EPILOGUE` 钩子,尾操作只在最后一个 K 块应用;结束时报告"融合 vs 各阶段分开跑"的加速比 |
| 后端中立 profile schema | `backends/base.py: ProfileReport` | 向量宽度、已/未向量化循环、寄存器/spill、occupancy、达成带宽%/算力%、stall 原因;后端填多少填多少,以固定版式喂 LLM |
| best-of-N + 裁决导向 | `generators/llm.py: FOCUS_BY_BOUND` | 每轮并发 N 个候选;样本的优化方向由瓶颈类型决定;提示词带 roofline 裁决、"已知无效"清单、结构化 profile、A/B 与线程利用率 |
| 可移植对拍测试 | `history.py: write_parity_test` | 每次刷新最优都生成 `parity_test.py`:参考 vs 内核、逐 dtype 容差、全部形状、NaN/Inf/±0/denormal、输入不可变、毒值预填、按契约分配 64 字节对齐缓冲 |
| 可集成包 | `bundle.py` + `kopt export` | `kernel.c` + `kernel.h` + `manifest.json`(签名 / 逐张量与累加 dtype / launch ABI / 参数 / 数值等级 / 性能)+ `build.sh` + `parity_test.py` + README |
| 混合精度规格 | `spec.py: output_dtype / accumulate_dtype` + `ops/*` | 逐张量 dtype + 显式累加 dtype;`bf16 -> fp32 (acc fp32)` 这类推理常见配置是一份规格,与单精度同一套数值等级 |
| dtype 泛化模板 | `ops/matmul_packed.py` / `ops/softmax.py` | `in_t / out_t / acc_t` 三类型;`_Float16` / `__bf16` 装载转累加类型计算再存回;输出比累加类型窄时 K 维单块以保住累加精度;bf16 在 numpy 侧用 uint16 存储 + RNE 编解码 |
| 后端 / launch ABI 契约 | `backends/protocol.py` + `backends/base.py: LaunchABI` | 验证 + 计时协议只写一次,建立在 `KernelSession` 的设备原语上(放置缓冲 / 发射 / 设备计时器 / 拷回);加速器后端只填原语,Agent / 评估器 / 生成器不变;`backends/accelerator_stub.py` 是文档化的填空模板 |
| 负载摄入 | `workload.py` + `kopt workload` | 从 CSV / JSONL / JSON trace 聚合 (shape, count),过滤算子、跳过并报告非法行 |
| BLIS 风格 GEMM 模板 | `ops/matmul_packed.py` | A/B 面板打包 + MR×NR 寄存器微内核 + 三级缓存分块,把 matmul 起点从 ~160 提升到 ~350–550 GFLOP/s |

## 代码结构

```
kopt_agent/
  dtypes.py        DType(fp32/fp16/bf16/fp64):C 类型、numpy 存储、编解码、每 dtype 数值策略;widest / default_accumulate_dtype
  spec.py          OperatorSpec / TensorSpec / TestCase / WorkloadProfile:算子契约、逐张量 + 累加 dtype、负载、precision_sensitive、快路谓词求值
  candidate.py     Candidate:候选源码 + 来源 + 参数 + 编译选项 + 快路谓词
  evaluator.py     编译 → 多形状验证(毒值、快路断言、覆盖率、数值等级)→ 多形状计时(加权目标、A/B、host/device、计时来源)
  agent.py         OptimizationAgent:baseline → 调参/进化(死路剪枝)→ LLM;每阶段 roofline 裁决;融合收益报告;结束导出 bundle
  history.py       试验记录、精度门控的最优选择、逐形状 + 加权排行榜、parity_test.py 生成
  bundle.py        夺冠内核可集成包:kernel.c / kernel.h / manifest.json / build.sh / parity_test.py / README
  workload.py      从 CSV / JSONL / JSON trace 聚合 WorkloadProfile
  knowledge.py     跨运行知识库:最优配置热启动 + 死路(方向 + 根因)剪枝与"已知无效"清单
  roofline.py      峰值 / 发射地板探测(或后端自报峰值)、roofline 分析、Verdict(瓶颈分类、可达上限、到顶即停、导向)
  generators/
    template.py    参数化模板:搜索空间、约束、无放回采样、邻域变异、fast_path 钩子
    llm.py         OpenAI 兼容接口的内核改写器(best-of-N、裁决导向的样本焦点、死路清单、结构化 profile)
  backends/
    protocol.py    验证 + 计时协议(NaN/毒值双跑、金丝雀、输入快照、快路标志、交替 A/B),建立在 KernelSession 设备原语之上
    base.py        Backend 契约 + LaunchABI + ProfileReport + CompileResult / RunResult;默认 run() = open_session() + 协议
    cpu_c.py       gcc + OpenMP 后端:子进程隔离执行同一协议,向量化报告 → ProfileReport
    accelerator_stub.py 文档化的加速器后端填空模板(设备指针 + stream 的 launcher、设备计时器、H2D/D2H 不计时)
  runner.py        子进程内核执行器:关闭 core dump,HostSession(ctypes + 金丝雀缓冲)驱动协议
  hardware.py      读取 CPU 型号 / SIMD 指令集 / 缓存,供报告与 LLM 使用
ops/
  matmul.py        GEMM:朴素基线 + packed* 模板 + blocked 模板(含 ALIGNED 快路示例);in/out/acc dtype 可分别指定
  matmul_packed.py BLIS 风格打包 + 寄存器微内核模板,in_t/out_t/acc_t 泛化,带 EPILOGUE 融合钩子
  softmax.py       行 softmax(精度敏感):朴素基线 + 模板(线程/调度/在线算法/快速数学),acc_t 决定 expf/exp
  bias_relu.py     逐元素 Y = max(X + bias, 0):小形状下用于演示 dispatch-overhead-bound 裁决
  matmul_bias_relu.py 融合算子 relu(A@B + bias):阶段序列 + 融合参考 + 单一签名 + 分开跑的各阶段 bundle
tests/             失败路径分类、分级批量、数值等级与门控、反假成功、快路契约、形状特化、负载目标、A/B、
                   roofline 裁决、负知识库、profile 渲染、提示词内容、融合参考、假 LLM 端点的端到端回路、
                   后端契约(假设备后端过评估器、core dump 限制、stub)、混合精度、bundle 导出、负载摄入
Makefile / tox.ini 本地 pre-ship 检查:make preship 在 3.10 与默认解释器上各跑一遍 compileall + pytest
```

## 如何扩展

**新增一个算子**:在 `ops/` 下新建模块,提供 `build(shape, dtype="fp32", workload=None, output_dtype=None, accumulate_dtype=None) -> OperatorBundle`(包含 `OperatorSpec`、基线源码、一个或多个模板),然后在 `ops/__init__.py` 的 `OP_REGISTRY` 注册。`make_case(shape, dtype)` 负责构造张量,每个 `TensorSpec` 自带 dtype,所以输入 / 输出 / 辅助张量可以各用各的类型;`OperatorSpec.output_dtype` / `accumulate_dtype` 声明输出与累加精度,基线与模板按 `in_t / out_t / acc_t` 生成;`reference` 直接返回 fp64 结果(评估器按输出 dtype 舍入);`scalar_names` 给出 int 标量名(快路谓词用);关键是把边界形状(非分块倍数、维度为 1、极大值输入等)写进 `edge_shapes` / `input_generator`,评估器会替你把所有投机取巧的候选筛掉。输出是概率 / 会被下游放大的算子请标 `precision_sensitive=True`。

**新增一个融合算子**:参考 `ops/matmul_bias_relu.py`——把各阶段的 `reference` 组合成融合参考,填 `fused_stages`,在 `OperatorBundle.fusion_parts` 里放各阶段在相同形状上的 bundle,Agent 会在结束时报告融合加速比。packed GEMM 模板的 `epilogue` 参数可以直接把逐元素尾操作折进最后一个 K 块。

**移植到 GPU / NPU**:复制 `kopt_agent/backends/accelerator_stub.py`,填空即可,Agent、评估器、生成器不需要改动。契约如下:

- 候选 = 设备 kernel + 一个具有算子 `c_signature` 的 host launcher;launcher 收到的是**设备指针**,`LaunchABI.stream_argument=True` 时在 int 标量后多一个 `void* stream`,launcher 负责选 grid 并入队。快路契约与 CPU 相同(`// fast_path:` 注释 + 导出 `int kopt_fast_path_active`)。
- `compile`:调用 `nvcc` / `hipcc` / `bisheng`(昇腾 Ascend C)/ Triton JIT,产出可加载的产物;把寄存器占用、spill、occupancy、stall 原因填进结构化的 `CompileResult.profile: ProfileReport`(填多少算多少,提示词版式统一)。
- `open_session` 返回一个 `KernelSession`,只实现原语:`upload`(H2D,后端决定缓冲放哪)、`allocate_output` / `fill_bytes`(设备 memset)、`download`(D2H,只在验证阶段被调用)、`launch`(同步)、`timed_launch`(设备事件圈住 kernel,返回 ms)、`fast_path_flag`。NaN / 毒值双跑、输入快照比对、快路断言、warmup、同进程交替 A/B 全部由 `backends/protocol.py` 里同一份协议驱动;H2D / D2H 由协议在原语外计时,报在 `h2d_ms` / `d2h_ms`,永不进入 `timings_ms`(`timing_source="device_timer"`)。
- `measure_peaks`:返回设备的 FMA 峰值 / 带宽 / 发射地板(标称值或自测),roofline 裁决就用它;`portable_compile_command` 给 bundle 与对拍测试用。
- `hardware_summary` / `language_guidance`:告诉 LLM 目标是 CUDA C++ 还是 Ascend C、SM 数量 / AI Core 数量、共享内存 / UB 大小、允许的头文件与 API;`launch_abi.describe()` 已经把指针空间 / stream / 同步语义写好。
- 在 `backends/__init__.py` 的 `BACKENDS` 注册后即可 `kopt run --backend <name>`。`tests/test_backend_contract.py` 里的 `FakeDeviceBackend` 是一个用 ctypes 模拟设备内存与设备计时器的最小实现,可以作为参照。

**换搜索策略**:`agent.py` 里的三个 `_phase_*` 是纯函数式的调用顺序;可以把随机 + 进化换成贝叶斯优化,或把 LLM 阶段改成"保留 top-k 候选做进化"。评估器和历史记录接口保持不变。

## 运行测试与 pre-ship 检查

```bash
pip install -e ".[dev]"
pytest -q                   # 默认解释器
make preship                # 3.10(最低声明版本)+ 默认解释器各跑一遍 compileall + pytest;推送前必须绿
make test-310               # 只跑 3.10;有 uv 时自动准备 3.10 + numpy + pytest(uv python install 3.10 一次即可)
tox -e py310                # 等价的 tox 入口(tox.ini 覆盖 py310 / py311 / py312)
```

仓库没有 CI,`make preship` 就是本地关卡:`requires-python = ">=3.10"`,任何只在高版本合法的写法(3.12 的 f-string 嵌套引号、3.11 的 `Template.get_identifiers()` 等)都会在这里被抓住。
需要 gcc + OpenMP;fp16 / bf16 相关用例在编译器不支持 `_Float16` / `__bf16`(x86 上分别需要 gcc ≥ 12 / ≥ 13)时自动跳过,`kopt run --dtype fp16|bf16` 在这种机器上会给出明确报错而不是编译失败。
