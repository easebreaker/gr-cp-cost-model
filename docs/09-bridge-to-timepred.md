# 09 · 与 `timepred` 的接口：κ 是被推导出来的，不是拟合的

仓库里已有的 `执行时间预测/timepred` 用这个式子描述一个被切分的块：

```
t_hat = t_cut / g(N) + τ + v·Vol ,        g(N) = N / (1 + κ(N−1))
```

其中 `κ` 是"离理想并行有多远"的拟合系数，`τ`、`v` 是未掩盖的通信。它的判据是 `speedup_upper_bound(s) = 1/s`，也就是 Amdahl。

本文的模型给出同一件事的**结构分解**：

```
Δ(N) = 被切分部分在 N 下的时间 / 在 1 下的时间
g(N) = 1/Δ(N)
κ(N) = (N·g − 1)/(N − 1)
r    = 可切分部分占比（attn_only 模式下 = 注意力占比）
c    = (1−overlap)(α + Vol/BW) / t_0
```

`src/grcost/bridge.py` 实现了它，并且**回代是精确的**（`tests/test_bridge.py`，误差 0.00e+00）。所以两个框架不是两套模型，而是**同一个模型的两种分辨率**：`timepred` 拟合 `κ`，`grcost` 从 `eff(c)`、不均衡、通信量算出 `κ`。

---

## 1. 一个必须说清的定义问题

`λ` 必须定义在**时间**上，不能定义在 pair 数上。

原因：dense 投影在 rank 之间是均分的，只有 attention 项带掩码不均衡。用 pair 的 `max/mean` 会**高估**时间不均衡，回代误差 4%–13%，且随 N 增大。第一版就是这样错的，被 `test_bridge` 抓住。

所以 `CPDecomposition` 同时给出三个量，定位用途不同：

| 字段 | 含义 | 用途 |
|---|---|---|
| `g`, `kappa` | 由**时间**比定义 | 回代 `timepred`，精确 |
| `eff_ratio` | `eff(c₁)/eff(c_N)` | 效率损失有多大 |
| `imbalance_pairs` / `imbalance_flops` | pair / FLOP 的 max÷mean | 诊断；两者之差就是"dense 压平不均衡"的实证 |

---

## 2. 由此得到的一条可证伪预测：κ 不是常数

`eff(c)` 随 chunk 变小而下降，所以 `κ(N)` 随 N **上升**。在 910B（`bw_comm=150 GB/s`）上：

```
T=4096   c(cp1)=4096   c(cp8)=512
  N=2   g=1.535  κ= 0.303
  N=4   g=2.034  κ= 0.322
  N=8   g=2.319  κ= 0.350
  κ: N=2 0.303 | N=4 0.322 | N=8 0.350   (drift 1.16x)
  constant-κ fitted on N∈{2,4}, used at N=8: -6.8% error
```

三条序列（`T = 2048/4096/8192`）都得到 drift ≈ 1.15–1.16×，常数-κ 外推**低估损失 5.6%–7.3%**——即**过于乐观**。

三条推论：

1. **用 `N=2,4` 拟合的 κ 预测 `N=8` 会低报损失。** 这解释了为什么"小规模验证通过、上大规模不达预期"是常态，而不是偶发。
2. **`κ` 的物理含义是可分解的**：`κ` 上升来自 `eff(c)` 的滚降，不是来自并行度本身。把 `κ` 当成一个常数，等于假设 `eff(c)` 在扫描区间内是平的——而这个假设正好在 `c` 接近 `c*` 时失效。
3. **只需一次 `eff` 标定就能算出 drift**，不需要任何 NPU 时间。这是"先算再测"的典型例子：`kappa_drift()` 在跑实验之前就告诉你，你的 κ 会在哪个 N 上失准。

如果实测的 `κ` 真的不随 N 移动，那就说明 `eff(c)` 在扫描区间内是平的——这本身是一个结论（`c*` 远小于最小 chunk），而且会推翻"效率曲线是主要损失来源"的说法。**两种结果都有信息量**，这是判据设计合格的标准。

---

## 3. 两个 shard 模式，别混用

| 模式 | 假设 | 谁是这样 |
|---|---|---|
| `full` | CP 切整个块，投影也被切 | 真实 CP（序列维切分后，投影自然按 token 切） |
| `attn_only` | 只切 attention，投影在每卡复制 | 某些只切 KV 的实现；或 kernel 不支持分片投影时的退路 |

两者的 `r` 不同（1.0 vs 注意力占比），预测也不同（`attn_only` **更慢**或在某些点更快，取决于 `dense/total`）。`test_attn_only_is_not_cheaper_than_full` 只是一个方向的检查；真实差别要实测。

**先说清你的实现属于哪一种**，再谈 κ。混用会让两个框架永远对不上，而且很难查。

---

## 4. 数据怎么流

一次 NPU 测量，写两份 CSV：

```
scripts/npu/harness.write_calibration()      -> data/npu_calibration.csv
                                                 (model, layout, T, B, cp, step_ms, t_fixed_ms, ...)
                                                 ==> grcost.fit.load_csv

scripts/npu/harness.write_timepred_sample()  -> data/timepred_samples.csv
                                                 (batch, time_s, peak_mem_bytes, plan=attention=cp:4, ...)
                                                 ==> timepred.io.load_samples
```

于是：

```
                 ┌── grcost: 解析工作量 + 拟合 eff(c) ──► 预测任意 (T, CP, model)
一次 NPU step ───┤
                 └── timepred: 拟合 (κ, τ, v) ──────────► 预测同分布下的配置
```

两条路可以互相校验：`bridge.decompose()` 从 `grcost` 参数算出 `κ`，`timepred` 从数据拟合出 `κ`。**两者不一致的那一组配置，就是两个模型都还没抓住的物理。** 这是比任何单一 MAPE 都更有诊断力的检查。

---

## 5. 具体怎么对账

```python
from grcost.bridge import decompose
from grcost.cost import Params
from grcost.presets import resolve_model, mtgr_layout
from grcost.hardware import get

p = Params.from_json(open("out/params_composed.json").read())
m, lay, hw = resolve_model("mtgr-large"), mtgr_layout(4096), get("ascend-910b")

for n in (2, 4, 8):
    d = decompose(m, lay, 4096, 32, n, hw, p, t_fixed_ms=6.0)
    print(d.row())
# N=2  g=1.535  κ= 0.303  eff×1.329  imb=1.015  comm=  11.09ms  t=  354.89ms
# ...
```

把这里的 `κ` 与 `timepred` 拟合出的 `κ` 并排：

| 来源 | κ(CP=2) | κ(CP=4) | κ(CP=8) |
|---|---|---|---|
| `grcost` 解析 | 0.303 | 0.322 | 0.350 |
| `timepred` 拟合 | ? | ? | ? |
| 差异 | ? | ? | ? |

若解析值随 N 上升而拟合值接近常数，说明拟合用的是**平均值**，那么它在 N 更大时会继续低估——可以直接用 `kappa_drift()` 报出下一点的误差量级。

---

## 6. 这条桥的价值

- 已有框架**不用改**：它继续拟合，只是现在有了先验和外部校验。
- 新框架**不用重跑**：`eff(c)` 一次标定即可支撑任意 N 的 κ。
- 报告时可以说：**"我们没有拟合 κ，我们算出了它，并预测了它在 N=8 会移动 16%。"** 这比"我们拟合得更好"具体得多，也更容易被证伪。

一句话：`κ` 是 `eff(c)` 与掩码拓扑的一个函数，而那个函数只有一条一维曲线需要测量。
