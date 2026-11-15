# NPU 实验脚本

面向**真实 Ascend NPU** 的测量驱动。没有 NPU 也能看：所有脚本支持 `--dry-run`，且 `grcost` 本身零依赖。

```
harness.py     计时协议 + 三种 CSV 写出器（grcost / timepred / 通用）
hstu_layer.py  算子形状与官方实现一致的 HSTU 层（用于测时延）+ 三种掩码族
micro.py       L0：常数（下发、带宽、算力、同步、集合通信、显存边界）
sweep.py       L1：曲线（eff、不均衡、掩码拓扑、padding）
e2e.py         L2：端到端；以及 compose 模式（不需要 CP 就能预测整条 CP 曲线）
```

设计依据见 `docs/08-npu-experiment-design.md`。

## 为什么可以用"替身层"测时延

成本模型关心的是**算子形状**，不是权重。`hstu_layer.py` 的算子顺序与官方
`SequentialTransductionUnitJagged` 一致（已按 `执行时间预测/HSTU核对.md` 的形状核对）：

```
uvqk mm -> silu -> qk^T einsum -> silu(LxL) -> mask -> p.v einsum -> o mm -> ln -> gate
```

所以用对的 `(d, H, attention_dim, linear_dim)` 计时，得到的就是官方 kernel 的成本。
它唯一测不到的是 jagged gather/scatter —— 那由 `sweep.py --only pad` 单独量。

## 计时协议（固定）

```
warmup 5 -> 计时 15 次 -> 中位数 + IQR（不报均值）
每次前后 torch.npu.synchronize()
不 flush L2
```

`torch_npu.npu.Event(enable_timing=True)` + `elapsed_time()`，单位 ms。
不加末尾 `synchronize()` 会测成 CPU 下发时间（短 10-100 倍），这是最常见的错误。

## 用法

```bash
# 看计划（任何机器）
python scripts/npu/micro.py --dry-run
python scripts/npu/sweep.py --dry-run
PYTHONPATH=src python -m grcost manifest --out data/npu_manifest.csv

# L0：常数
python scripts/npu/micro.py --only noise,dispatch,hbm,peak,sync
torchrun --nproc_per_node=8 scripts/npu/micro.py --only collective
python scripts/npu/micro.py --only memory --B 32

# L1：曲线（全部不需要 CP）
python scripts/npu/sweep.py --only eff,shard,static,pad

# 组合出整条 CP 曲线（不需要 CP）
python scripts/npu/e2e.py compose --sweep data/npu_sweep.csv \
    --micro data/npu_micro.csv --hardware ascend-910b --T 4096 --t-fixed 6.0

# L2：端到端（有 CP 之后）
torchrun --nproc_per_node=8 scripts/npu/e2e.py measure --cp 8 \
    --step-module my_hstu_step.py
```

## `--step-module` 的契约

`e2e.py measure` 不内置模型。它导入你给的文件，调用

```python
build_step(cfg) -> callable
# cfg = {"T": int, "B": int, "cp": int, "device": "npu", "dtype": "bf16"}
```

返回的函数必须是**一次完整的 step**（前向 + 反向 + 优化器，或你要测的那部分），
无参数、可重复调用。

## 输出

| 文件 | 谁写 | 谁读 |
|---|---|---|
| `data/npu_micro.csv` | `micro.py` | `e2e.py compose` |
| `data/npu_sweep.csv` | `sweep.py` | `e2e.py compose` |
| `data/npu_calibration.csv` | `e2e.py` | `grcost.fit.load_csv` |
| `data/timepred_samples.csv` | `e2e.py` | `timepred.io.load_samples` |
| `out/params_composed.json` | `e2e.py compose` | `grcost` CLI（`--params`） |

同一个测量写两份 schema，两套模型同时可用，见 `docs/09-bridge-to-timepred.md`。

## 已知限制

- `collective` 作业在单进程下会**明确跳过**（不是静默给 0）；`alpha` 与 `BW` 是节点属性，
  单卡测不出来。
- `memory` 作业单卡只给一个点；`fit_memory` 需要**不同的 per-rank chunk 长度**，
  否则会返回 `identifiable=False`。这是故意的。
- `sweep.py --only pad` 用 `T=mean` 与 `T=max` 对比来量化 padding 浪费，
  不构造 `(B*T)^2` 的块对角掩码（`B=32, T=6000` 时那会是 3.7e10 个元素）。
- Ascend 的标称 TFLOPS 与实测差距很大，`hardware.py` 里的值只是量级参考，
  绝对时间必须用你实测的 `bw_util` / `eff_inf`。
