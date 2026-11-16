# 11 · 集合通信：为什么「带宽随消息大小变」，以及怎么建模

> 结论先说：**「按消息大小分段」是在描述症状，不是建模。** 现象的本质是
> 一次 all-gather 由若干**串行**步组成，延迟按步付、字节按总量付；
> 两者之比随消息变大而移动，于是「测出来的带宽」自然是一条曲线。
> 这条曲线只有一个形状、两个参数，而且它的拐点**随卡数移动**——所以分段
> 拟合在 n=8 上得到的断点，到 n=2 或 n=16 上全错位。

---

## 1. 机制

all-gather 不是「一次传完」。它把消息切成若干步，每步都有一次串行启动：

```
t(V) = S(N, V) · alpha  +  V / BW_inf
```

- `alpha` 是**单步**延迟（μs 级）
- `S` 是**串行步数**：递归倍增 `ceil(log2 N)`；ring 退化为 `N−1`
- `BW_inf` 是渐近带宽

只有第一项按步计费，第二项按字节计费。

## 2. 「表观带宽」是曲线

实测时你会用 `V / t` 反过来算带宽：

```
BW_app(V) = V / t = BW_inf / (1 + V*/V),      V* = S · alpha · BW_inf
```

这是**饱和曲线**，和 `eff(c) = eff∞/(1 + (c*/c)^γ)` 完全同形。

```
 alpha = 15 us/step, BW_inf = 149 GB/s, n = 8 (S = 3)

    消息      时间    BW_app   占峰值
   0.07M    45.4us     1.4G       1%     <- 延迟主导
   1.05M    52.0us    20.2G      14%
   4.19M    73.1us    57.3G      38%
  16.78M   157.6us   106.5G      71%     <- 拐点附近
  67.11M   495.4us   135.5G      91%
 268.44M  1846.6us   145.4G      98%
1073.74M  7251.3us   148.1G      99%     <- 带宽主导
```

**所以那些「小消息带宽低」的点不是异常，而是同一个规律的低 V 段。**

## 3. 为什么不能分段：拐点随卡数移动

`V* = S·alpha·BW_inf`，而 `S` 取决于 N：

| 卡数 | 递归倍增 S | V* | ring S | V* |
|---|---|---|---|---|
| 2 | 1 | 2.23 MB | 1 | 2.23 MB |
| 4 | 2 | 4.47 MB | 3 | 6.71 MB |
| 8 | 3 | 6.71 MB | 7 | 15.64 MB |
| 16 | 4 | 8.94 MB | 15 | **33.52 MB** |

**n=8 拟合的断点搬到 n=16 上，位置差 5 倍。** 这就是分段不可外推的定量理由。
（顺带：这也解释了为什么 8 卡的「算法带宽」看起来比 2 卡高一个量级——不是链路变好，
而是 `S` 变了、拐点移了。）

## 4. 一个 α 一个 BW 就够了

把整条曲线（64 KB – 1 GB）丢给两参数拟合，`alpha = 15.0us`、`bw = 149 GB/s` 被精确还原，
留出检验（只用大消息拟合、预测小消息）误差 **0%**。

`diagnose_size_dependence()` 会给出这张表，并明确告诉你落在哪个 regime：

```
  n=8  steps=3  alpha=15.0us  bw=149.0 GB/s
  break-even message size V* = 6.70 MB
  fit residual: 0.0%        held-out: 0% error
       bytes   measured      model  BW_app GB/s     regime
       0.10M      45.7us      45.7us          2.2    latency
      16.78M     157.6us     157.6us        106.5  bandwidth
    1000.00M    6756.4us    6756.4us        148.0  bandwidth
```

**留出检验是判据，不是 in-sample 残差。** 因为自由截距能吸收很多结构：
下面这条「两段不同带宽」的曲线 in-sample 只有 6.3% 残差，留出却是 **92%**。

```
  10.00M     226.7us     459.2us         44.1    latency
 100.00M    1726.7us    1047.0us         57.9  bandwidth   <- 斜率变了
  3000.00M   20060.0us   19986.3us       149.6  bandwidth
  >> the single-step-count model does not explain this curve.
     Do NOT fit piecewise breakpoints; check whether the implementation
     changes algorithm mid-range (v_switch), or whether the number of
     collectives changes with size.
```

如果留出确实不过，正确的补救是给模型加**一个整数** `v_switch`（算法切换点），
而不是给曲线加断点。

## 5. 真正让延迟变贵的是「集合通信次数」，不是字节

这是 CP 在 jagged 输入上最容易犯的错。同一步 6.04 GB 的字节总量，两种实现：

| 实现 | 每次集合通信 | 次数 | 时间 |
|---|---|---|---|
| 每层一次 | 402.65 MB | 15 | **41.2 ms** |
| 每层每桶一次 | 12.58 MB | 480 | **62.1 ms** |

字节总量完全相同，差 21 ms —— 全是延迟（480 次 × 3 步 × 15 μs = 21.6 ms）。
**所以模型必须分开记两个数**：

- `rounds`：一次 step 里有几次集合通信（scheme × 层数 × 桶数）
- `S(N,V)`：每次内部有几跳

把两者相乘当一次，或者把跳数当成 `rounds`，都会把延迟项放大 `cp−1` 倍。
（这正是早先版本把通信判成主因的那个错误的来源。）

## 6. all-gather 与 ring：字节一样，步数与显存不同

```
每 rank 体积（两者相同）  = (N−1)/N · V
每 rank 峰值缓冲  all-gather = V       ring = V/N
```

**所以「换成 all-gather」是延迟与显存的改动，不是带宽的改动。**
把它们建模成不同的通信量，是一个常见错误。

代价也很实际：all-gather 的峰值缓冲是 `V`，`V = 6 GB` 量级时直接决定能不能跑。
这就是 CP 的第一性约束仍然是显存、而不是时间的原因。

## 7. 910C 上 overlap 必须 pin 0

HCCL 在这一代不做「集合通信与计算流水线重叠」，所以延迟项**完全暴露**。
代码里用 `Hardware.pipeline_overlap=False` 表达这个事实，`Params.overlap` 默认 0，
`e2e.py compose` 遇到这类卡会强制 pin 并打印原因。

这改的是结论方向：早先按 `overlap=0.5` 算出的加速比偏乐观，pin 0 之后通信
全部计入关键路径。

## 8. 这条分析改了什么

| 原做法 | 现做法 | 理由 |
|---|---|---|
| per-size 分段拟合 `alpha` | 一个 `alpha` + 步数 `S` | 断点随 N 移动，分段不可外推 |
| per-size 分段拟合 `bw_comm` | 一个渐近 `bw` + 表观带宽公式 | 表观带宽是 latency 摊薄的产物 |
| ring 与 all-gather 用不同体积 | 同体积，不同步数与峰值缓冲 | `(N−1)/N·V` 两者相同 |
| `rounds = (cp−1)×层数` | `rounds = 层数×桶数`，跳数在 `S` 里 | 否则延迟被放大 `cp−1` 倍 |
| `overlap` 拟合 | Ascend 上 pin 0 | HCCL 无流水线重叠 |
| 无 `t_emb` | `t_emb_ms` + `dedup`，按 cp 切分 | 见 §9 |

## 9. 嵌入查表：一个必须单独建模的项

查表**永久带宽受限**（算术强度 1 FLOP/byte），而且和 attention 竞争同一块 HBM。
两件事必须单独记：

- **去重比 `dedup`**：`B·T` 个 token 不等于 `B·T` 次行读取。按 token 建模会**高估 1/dedup 倍**
  （生产流量上这个比值常是 2–10）。
- **按 cp 切分（需求 7）**：每卡只查自己那段，查表时间 ∝ 1/cp。但**不是免费的**——
  激活要做一次稠密 all-gather。算一下就很清楚：

```
  cp=8: 查表省下  0.764 ms
        re-gather 1344.00 MB 花 9.065 ms  -> DOES NOT pay
        （改成 gather id 只搬 4.00 MB，而不是 1344.00 MB）
```

因为 `row_bytes = d·dtype`，re-gather 的体积恰好是表读取的 `1/dedup` 倍，
而它走的是集合通信带宽。**所以朴素实现（切查表 + 收激活）永远不划算。**

生产上的正解通常不是切查表，而是**gather id**（4 字节）然后在每张卡上各自查表 ——
因为 attention 本来就需要全序列的 K/V。`EmbedModel(credit_regather=True)` 表达
「这次 re-gather 与 CP 本来就要付的传输融合了」的情形。

**需求 7 的正确表述**：`t_emb` 按 `1/cp` 缩小的前提是 re-gather 被融合或本就要付；
否则该项不应该缩。

## 10. 要测什么（清单已改）

1. **尺寸扫描跨过 `V*` 两侧**：节点内链路约 64 KB – 256 MB。
   区间没跨过 `V*`，`alpha` 与 `bw` 不可分，拟合值会互相抵消。
2. **至少两个卡数**（2 与 8）：拐点随 N 移动，这一步是在**检验**步数模型，
   而不是假设它。
3. **单独记录「每 step 集合通信次数」**：jagged 实现会把这个数悄悄乘大，
   而字节总量看不出来。
4. **跑 `diagnose_size_dependence()`**：留出误差大就说明实现中途换了算法，
   加 `v_switch`（一个整数），不要加断点。
5. **`t_emb` 与 `dedup` 分开测**，并记录 re-gather 是否融合。

对应作业：`M5.all_gather` / `M5.all_to_all`（尺寸扫描 × 两个卡数）、`M2.4`（查表与去重比）。
