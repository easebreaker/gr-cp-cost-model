# 08 · NPU 实验设计：测什么、怎么测、先测哪个

面向 **真实 NPU（Ascend 910B / CANN）环境**的一次性测量方案。目标不是"多跑几个点"，而是**用最少的机时产出两类数据**：

1. `grcost` 的五参数（`eff_inf, c_star, gamma, alpha, bw_comm`）+ 三个 pin 值；
2. `timepred` 框架的 `Sample` 行（同一批测量，第二种 schema），让两套模型同时可用。

完整清单是可执行代码：`python -m grcost manifest --out data/npu_manifest.csv`。本文解释它**为什么这样排**。

---

## 0. 一个前提判断：90% 的标定不需要 CP 已经实现

这是整个设计的支点。成本模型是三项的乘积，而三项可以**分别测**：

```
t(CP) = t_fixed + max_r[ recomp·(flops_r/(P·eff(c)) + bytes_r/(BW·bw_util)) + (1-η)(α + V/bw_comm) ]
                     ↑ 工作量            ↑ 效率曲线                ↑ 通信
```

| 项 | 怎么单独测 | 需要 CP 吗 |
|---|---|---|
| `eff(c)` | **CP=1 时 `c ≡ T`**，扫 `T` 就是扫 `c` | 不需要 |
| 负载不均衡 | 只跑某个 rank 的 query 行，`K/V` 保留全长，不加集合通信 | 不需要 |
| `α, bw_comm` | 任意集合通信原语的 时间–字节 曲线 | 不需要 |
| 掩码系数 | 单卡上切换 static/causal/candidate 比例 | 不需要 |
| `t_fixed` | 空 kernel 斜率 + 逐段相减 | 不需要 |
| 显存 `T_max(CP)` | 二分 OOM 边界，每卡一次 | 不需要 |

只有**最后的组合验证**需要真 CP。所以计划的第一版可以在 NPU 上**本周就跑完**，CP 实现到位后再补 12 组做验证。

`scripts/npu/e2e.py compose` 就是这个思路的可执行版本：它读前两步的 CSV，**直接反解出 `eff(c)`**（不是联合拟合），再组合出整条 CP 曲线。在合成数据上的回代是精确的：

```
extracted points (no joint fitting):
  c=512    eff=0.084   ...   c=16384  eff=0.386
fitted: eff_inf=0.420  c_star=1800  gamma=1.10   rmse(log)=0.00%
collective fit: alpha=25.0us  bw=120.0 GB/s
composed CP prediction (no CP run used as input):
 CP  chunk     ms  speedup    eff  imbal   comm_ms
  1   4096  519.51     1.00  0.299   1.00      0.03
  2   2048  366.38     1.42  0.225   1.01     25.19
  4   1024  297.49     1.75  0.147   1.04     37.77
  8    512  273.06     1.90  0.084   1.10     44.07
```

真值 `eff_inf=0.42, c_star=1800, gamma=1.1, α=25μs, bw=120GB/s` **全部被精确还原**。这不是因为拟合好，而是因为 `eff` 是**代数解出来的**（`compose.extract_eff`），只有一个一维三参数形状需要拟合。

---

## 1. 测量协议（先固定，否则数据不可比）

沿用已有方案 P3 的口径，不做修改：

```
warmup 5 次丢弃
计时 15 次，每次前后各 record 一个 Event
计时前后 torch.npu.synchronize()
报告 中位数 + IQR（不报均值）
不 flush L2（线上也不 flush）
```

Ascend 上的具体写法（CANN 8.0+ 已验证的 API）：

```python
import torch, torch_npu
torch.npu.synchronize()
start = torch_npu.npu.Event(enable_timing=True)   # 必须 enable_timing=True
end   = torch_npu.npu.Event(enable_timing=True)
start.record(); fn(); torch.npu.synchronize(); end.record(); torch.npu.synchronize()
ms = start.elapsed_time(end)                       # 单位 ms
```

> 不加末尾的 `synchronize()` 时，你测的是 **CPU 下发时间**，可能比设备时间短 10–100 倍。这是这类测量最常见的错误。

**每个数都带 CV 一起报。** 下游所有判据都是"差异 vs CV"的比较，CV 不给出来，结论就不可审计。`harness.noise_floor()` 专门产出这个门禁用数。

内核级时间（用于拆 `b`）走 profiler，导出后读 `kernel_details.csv` 的 `Task Duration`：

```python
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=2, repeat=1, skip_first=1),
    experimental_config=torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,   # Level1 才有 communication.json
        export_type=torch_npu.profiler.ExportType.Text),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof"),
) as prof:
    for step in range(5):
        step_fn(); prof.step()
```

产出 `ASCEND_PROFILER_OUTPUT/`：`kernel_details.csv`（算子耗时/形状）、`op_statistic.csv`（按类型汇总）、`api_statistic.csv`（框架侧）、`communication.json` + `communication_matrix.json`（Level1，集合通信）。
以及 `mstx` 打点用来标定"前向/反向/优化器"边界，避免靠算子名猜。

---

## 2. L0：常数（约 2 小时，全部单卡可做）

每一项都对应一个**从拟合里删掉的参数**。

| ID | 测什么 | 怎么构造 | 得到 | 判据 |
|---|---|---|---|---|
| M1.1/M1.2 | 同输入重复性 | 同一配置连测 10 次 ×3 组，在最短路/最长序列各一组 | **CV（噪声底）** | CV > 5% → 放弃细粒度判据（G0） |
| M2.1 | 下发成本 | 空 kernel 循环 `n = 1/8/64/256` | `c_dispatch` (μs/launch) | 斜率是否 ∝ n；`t_fixed` 由此定 |
| M2.2 | 单次启动 vs 重复 | 同一 kernel 打 1/2/4/8 次 | 每启动成本 | 与 M2.1 一致 |
| M3.1 | HBM 带宽 | 64/256 MB 纯 copy | `bw_util = 实测/标称` | 这个数**必须 pin**，不能拟合 |
| M3.2 | 可达算力 | 4096³ matmul，工作 dtype | `eff_inf` 上限 | 与标称 328 TFLOPS(bf16) 的比值 |
| M4.1 | 同步开销 | `synchronize()` ×1 / ×100 | `t_sync` | 过大会污染所有计时 |
| M5.* | 集合通信 时间–字节 | 见 §3 | `α`, `bw_comm`, 重叠是否可信 | 见 G1 |
| M6.1 | 显存边界 | 二分最大 T（单卡 / 每档 CP） | `a` bytes/token, `M_static` | 见 G2 |

### 为什么 M5 是"逐字节曲线"而不是"带宽数字"

公开实测（910B，8 卡机内 Full Mesh）显示同一个集合通信的"算法带宽"强烈依赖参与卡数与消息大小：

| 卡数 | AllGather | AllReduce |
|---|---|---|
| 2 | 38 GB/s | 18 GB/s |
| 4 | 64 GB/s | 30 GB/s |
| 8 | 149 GB/s | 72 GB/s |

HCCS 单链路约 56 GB/s、聚合约 392 GB/s。**标称 392 与实测 149 差 2.6 倍**，而 `c* = F/BW` 直接吃这个数。所以：

- 不要用标称带宽算 `c*`，用**实测的、与你实际 CP 度对应的**那一条；
- 拟合 `α` 与 `bw` 用**同一种集合通信原语**（CP 用的是 ring/all-to-all，不是 all-reduce）；
- 2 卡是最坏情况（38 GB/s），意味着 `c*` 在 CP=2 时比 CP=8 大 4 倍。

参考命令（CANN 自带）：

```bash
cd $ASCEND_HOME/tools/hccl_test
mpirun -np 8 ./bin/all_gather_test -p 8 -b 8K -e 1G -f 2 -d fp16 -w 20 -n 100
mpirun -np 8 ./bin/alltoall_test  -p 8 -b 8K -e 1G -f 2 -d fp16 -w 20 -n 100
npu-smi info -t topo        # 先确认链路类型 HCCS / SYS / PHB
```

`-b/-e` 起始/结束字节、`-f` 倍增因子、`-w/-n` 预热/运行次数。**先把 topo 打出来**：不是所有 8 卡机器都是 Full Mesh，混了 `SYS`/`PHB` 的那几对会让 2 卡结果异常低，而这个异常会直接变成错误的 `c*`。

---

## 3. L1：曲线与不均衡（约 1.5 小时，全部不需要 CP）

### S1 — `eff(c)`：靠 CP=1 扫 T

```
T ∈ {512, 1024, 2048, 4096, 8192, 16384}   (跨度 32×)
B, d, heads, attn_dim, value_dim 固定
```

为什么这样可以：`c = T/CP`，CP=1 时 `c = T`。所以单卡扫 T 就是**直接描出 `eff(c)`**。

三个必须遵守的细节：

1. **跨度要跨过 `c*`**（910B 上 `c*` 约 800–2200 token，取决于是哪条链路）。只在 `c*` 以上取点，`c_star` 不可辨识，`gamma` 也会跑飞（G2）。
2. **两种构造都测**：纯 attention kernel（`S1.1`）和整层（`S1.2`）。前者给 kernel 的效率，后者把投影部分也带进来；两者的差就是"非 L² 项"的实证。
3. **`attn_dim ≠ value_dim` 的配置也要测**（官方 HSTU 是 `attention_dim=16, linear_dim=32, H=4, d=64`）。L² 项的系数是 `H·(a+v)`，**不是 `2d`**——按 `2d` 记会低估 1.5 倍。这正是已有方案 E2 要判的事，这里的价值是：它同时决定 FLOPs 表里要不要 `value_dim` 这一列。

`scripts/npu/sweep.py --only eff` 输出每个 T 的中位数，`e2e.py compose` 会代数反解 `eff` 并拟合形状。

### S2 — 不均衡：只跑一个 rank 的 query 行

构造（这就是 CP rank 在通信前的本地计算）：

```python
# 全序列 x_full: (B, T, d)；本 rank 拥有 query 位置 idx
xq   = x_full[:, idx]        # (B, |idx|, d)
rows = mask_full[idx]        # (|idx|, T)   —— key 侧保留全长
# 计时：xq 对 全长 K/V 做注意力
```

由此得到每个 rank 的**纯计算时间**，其 `max/mean` 就是负载不均衡。它应当与闭式 `2 − 1/CP`（纯因果、连续分块）在因果掩码下吻合；在 MTGR 布局下应显著更低。

三个方案各跑一遍（`block` / `balanced` / `striped`），**这才是"重平衡值不值得"的直接证据**，而且不需要任何分布式实现。

判据（G3）：`block/balanced` 的收益 **不能超过** `2 − 1/CP`。超过就说明闭式错了，回去看掩码。

### S3 — 掩码拓扑：矩形 vs 三角

固定 `T`，扫 `static_share ∈ {0, 0.5, 0.85, 0.95}`。

预测：`pairs ≈ S² + [P(P+1) − S(S+1)]/2 + K(P+1)`，`static_share=0` 退化为三角。所以时间应当随 share **上升**（矩形更贵），而不均衡应当随 share **下降**。两条都要看——只验证一条会被"时间随 share 涨是因为别的原因"反驳。

### S4 — padding 浪费

同一 `B`、同一 kernel，比较 `T = max(length)` 与 `T = mean(length)`：

```
waste = (B·Tmax − B·Tmean) / (B·Tmax)
```

这个数**只报，不除**。它归到 `t_fixed`（jagged gather/scatter 的实现成本），或者如果 kernel 支持 packed，就变成一种真实收益。把它偷偷从分母里除掉，是"造出 CP 加速比"的经典做法。

> 不要构造 `(B·T)²` 的块对角掩码来做 packed 对照：`B=32, T=6000` 时是 3.7×10¹⁰ 个元素。而且那样会改变 kernel 启动次数，把 padding 收益和调度开销混在一起。

---

## 4. L2：端到端（约 1.5 小时，需要 CP）

只有在 §0–§3 完成后才有意义，因为它是**验证**而不是标定。

| ID | 配置 | 用途 |
|---|---|---|
| E1.1 | `T ∈ {2048, 4096}` × `CP ∈ {1,2,4,8}`，`B=32` | 12 组标定表；`cp=1` 必须先 < 5%（G4） |
| E1.2 | `T=4096`, `B ∈ {16, 64}`, `CP ∈ {2,4}` | `α` 项：batch 应当缩放 step、不缩放延迟 |
| E2.1 | `T=4096`, `CP=4`, `balanced` | 实测重平衡收益 vs `2−1/CP` 上界 |
| E3.1 | `hstu-base` / `longer`，不重新标定 | 迁移性（G6） |
| E4.1 | 同配置换 kernel（Triton vs 融合） | `eff` 是不是 kernel 的属性（G5） |

同一批测量写两份文件，格式不同：

- `data/npu_calibration.csv` → `grcost.fit.load_csv` 直接读；
- `data/timepred_samples.csv` → 列 `batch, time_s, peak_mem_bytes, expert_frac, plan, len_<block>, t_<block>`，`plan` 写成 `attention=cp:4`。

`harness.write_calibration` / `harness.write_timepred_sample` 就是这两个契约。**一次 NPU 运行，两套模型同时可用**，不需要中间转换。

---

## 5. 门禁：先写下来，再看数据

写在 `manifest.gates()` 里，任何人在任何时间都可以 diff：

| 门 | 检查 | 不通过怎么办 |
|---|---|---|
| G0 | M1.1 同点 CV | > 5%：放弃台阶类细粒度判据，只报粗曲线，**不拟合 gamma** |
| G1 | `c* = F/BW_实测` 与拟合出的 `c_star` 同量级（10× 内） | 不同量级 → 曲线在吸收别的东西，停下 profile |
| G2 | S1 的曲线在扫描区间内**有拐点** | 没拐点 → 扩大扫描区间，不是加参数 |
| G3 | `block/balanced` 收益 ≤ `2−1/CP` | 超过 → 闭式或掩码假设错 |
| G4 | E1.1 的 `cp=1` MAPE < 5% | 不通过 → 工作量计数器错，**停止标定** |
| G5 | E4.1 换 kernel 后 `c_star` 变化 < 3× | 超过 → 撤掉"标定一次"的说法 |
| G6 | E3.1 跨模型误差 < 20% | 超过 → 改为按模型标定，缩小claim |

这张表的用处：**在看到数据之前**决定什么算失败。否则每个残差都会有一个事后解释——这是性能建模最常见的失败方式。

---

## 6. 顺序与成本

```
G0 噪声(M1)  →  M2/M3/M4 常数  →  M5 通信  →  M6 显存
                                      ↓
                              S1 eff 曲线  →  S2 不均衡  →  S3 掩码  →  S4 padding
                                      ↓
                              e2e.py compose  ⇒  预测整条 CP 曲线（无需 CP）
                                      ↓
                              E1/E2/E3/E4 验证（需要 CP）
```

| 层 | 作业数 | 机时 | 需要 CP |
|---|---:|---:|---|
| L0 常数 | 11 | ~2.1 h | 0 |
| L1 曲线 | 5 | ~1.5 h | 0 |
| L2 验证 | 5 | ~1.4 h | 全部 |
| 合计 | 21 | ~5.0 h | 1.4 h（28%） |

**3.6 小时就能拿到完整的 `eff(c)` + 通信常数 + 显存模型 + 不均衡实证**，足以回答组会上"CP 到底值不值"的问题。

---

## 7. 如果 NPU 上缺东西

这是最需要提前想清楚的一节，因为升腾栈的 CP 支持不会比 GPU 好。

| 缺什么 | 退路 |
|---|---|
| **没有 CP 实现** | 不影响标定。§0 的三项分开测，`e2e.py compose` 组合。E1/E2 降级为"后续验证" |
| **kernel 不支持 jagged/packed** | S4 量出 padding 浪费，计入 `t_fixed`，并**显式报告** |
| **拿不到 kernel 级时间** | 差分计时：同配置跑两次，一次去掉某段，差值即该段。每段两次作业 |
| **只有单卡** | 通信常数用 `hccl_test` 在能借到的 2 卡上量（α 与 BW 是节点属性，不是模型属性）。单卡仍能完成 `eff` 与不均衡 |
| **CANN 版本差异导致 kernel 变** | `kernel` 列记下 CANN 版本，`eff` 按 kernel 分组标定 |
| **`torch_npu.Event` 不可用** | 用 `time.perf_counter()` + `torch.npu.synchronize()` 包裹，精度够（step 是 ms 级） |

---

## 8. 落地检查表

跑之前逐条打勾：

- [ ] `npu-smi info -t topo` 已保存（确认 HCCS 拓扑，找出 `SYS`/`PHB` 对）
- [ ] CANN / torch_npu 版本已记入 `environment()`
- [ ] 计时协议固定：warmup 5 / 中位数 15 / 前后 synchronize / 不 flush
- [ ] `t_fixed` 用两条独立路径测（层数倍比 + 空 kernel）
- [ ] `bw_util`、`overlap`、`recompute` **pin 住**，不进拟合
- [ ] `eff` 扫描区间跨过实测 `c*` 两侧
- [ ] `attn_dim ≠ value_dim` 至少一组（E2 的判据）
- [ ] 每个数都带 CV
- [ ] 同点重复 ≥ 2 次（噪声底比多一个配置值钱）
- [ ] 输出同时写 `npu_calibration.csv` 与 `timepred_samples.csv`
- [ ] 门禁表在看数据前已定稿

---

## 9. 与已有 31 条方案的分工

已有的 `复杂度测定-原理方案.md`（P1–P3、E1–E5）解决的是**单层/单卡上的机理辨识**：`b` 的三分、两个 GEMM 分开、`ceil` 台阶、并行结构、regime 拐点 `L*`。

本文补的是**CP 维度**：

| 已有 | 本文 | 关系 |
|---|---|---|
| E5 定 `L* = 2P/BW`（regime 拐点） | `c* = F/BW`（ring 通信拐点） | **同一个物理量的两种口径**，差一个因子 2（算术强度取 `L/2`） |
| P2 独立测 `b` | M2 空 kernel + M4 同步 | 同一件事，`t_fixed` 的两种测法 |
| E1c 通信轮次阶梯 | M5 时间–字节曲线 | 前者定 `s(N)` 计数，后者定 `α` 与 `BW` |
| E2 两个 GEMM 的宽度 | `value_dim` 进 `pair_width()` | 本文已经把结论落进代码：`HstuModel.pair_width()` |
| E4 并行结构 / 核数 `C` | S2 查询块计时 | 前者是单卡内并行，后者是卡间负载；同一种 max/mean 逻辑 |

两条线合起来才闭合：**E5 给出单卡何时从访存受限转向计算受限，本文给出多卡时 CP 何时开始亏**。两者都形如 `长度 ≳ 机器比值`，这是同一个结构约束的两个投影。

---

## 10. 代码入口

```bash
# 清单（无依赖，笔记本上就能看）
PYTHONPATH=src python -m grcost manifest --out data/npu_manifest.csv

# L0 / L1 / L2 的 dry-run（不 import torch，不需要 NPU）
python scripts/npu/micro.py --dry-run
python scripts/npu/sweep.py --dry-run
python scripts/npu/e2e.py compose --dry-run   # 若 CSV 缺失会明确报缺

# 真跑（NPU 上）
python scripts/npu/micro.py --only noise,dispatch,hbm,peak,sync
torchrun --nproc_per_node=8 scripts/npu/micro.py --only collective
python scripts/npu/micro.py --only memory --B 32
python scripts/npu/sweep.py --only eff,shard,static,pad

# 组合出整条 CP 曲线（不需要 CP）
python scripts/npu/e2e.py compose --sweep data/npu_sweep.csv \
    --micro data/npu_micro.csv --hardware ascend-910b --T 4096

# 有 CP 之后的端到端
torchrun --nproc_per_node=8 scripts/npu/e2e.py measure --cp 8 \
    --step-module my_hstu_step.py
```
