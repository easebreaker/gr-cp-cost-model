# NPU 测量填报单

在 `值` 列填数字即可。填完运行：

```
PYTHONPATH=src python scripts/read_fill.py data/measurement_fill.md
```

它会检查前置依赖、跑门禁、生成 `out/params.json` 并打印归因表。

**五个绝不能进拟合的量**：`t_fixed` · `overlap` · `bw_util` · `recompute` · `layers`
前四个要么能直接测、要么在公式里只以乘积出现；`layers` 是口径。
任何一个搞错，拟合都会给出内部自洽但物理错误的结果。

## 只填「值」这一列

`指标` / `变体` / `怎么测` / `依赖` 都是**预设的，不要改动**。
- `变体` 列是这一行对应的配置（如 `cp=2`、`collective=all_gather`），**不是填写位**。
- 所有数据都填在 `值` 列；`CV%` 列填该次测量的变异系数（可留空）。
- 环境类（机型、版本、拓扑）填文字也可以，会被当作"已填"。

`依赖` 列非空的行，请等前置填完再填 —— 否则算出来的数没有意义。

**单位按表里写的填**（us、GB/s、token 等），读入时会自动换算成 SI：

| 符号 | 表里填 | 换算 |
|:--|:--|:--|
| `alpha` | us | ×1e-6 |
| `bw_comm` `BW_hbm` | GB/s | ×1e9 |
| `F_real` `F_nom` | TFLOPS | ×1e12 |
| `BW_nom` | TB/s | ×1e12 |
| `a_act` | KiB/token | ×1024 |
| 其余 | 表里的单位 | 原样 |

---

## 目录


**P0 · 预检（不占机时）**
- `env` 环境与配置
- `topo` 链路拓扑（决定 c* 的上界）
- `spec` 标称规格（只用于给上界，不用于拟合）

**L0 · 常数（把参数从拟合里删掉）**
- `M1.1` 同点噪声 · 短序列
- `M1.2` 同点噪声 · 长序列
- `M2.1` 下发成本（空 kernel 斜率）
- `M2.2` 单次启动成本（独立口径）
- `M2.3` 不可切部分 t_fixed（独立口径）
- `M3.1` HBM 带宽与效率
- `M3.3` M3.3
- `M3.2` 可达算力（eff_inf 的物理上限）
- `M2.4` M2.4
- `M4.1` 同步开销
- `M5` 集合通信 时间–字节 曲线
- `M6.1` 显存墙与激活字节

**L1 · 曲线与不均衡（不需要 CP）**
- `decl` 口径声明（层数）
- `S1.1` 效率曲线 eff(c) —— 唯一要拟合的东西
- `S1.3` S1.3
- `S1.2` 整栈交叉验算
- `S2.1` 负载不均衡（只跑一个 rank 的 query）
- `S3.1` 掩码拓扑：矩形 vs 三角
- `S4.1` padding 浪费

**C0 · 组合（不需要 CP）**
- `C1` 效率曲线三参数
- `C2` 两个拐点

**L2 · 验证与判决（需要 CP）**
- `E1.1` 标定表：长度 × CP
- `E1.2` batch 扫描（判 alpha 是否真实）
- `E2.1` balancer 实测（判闭式上界）
- `E3.1` 跨模型迁移（判“标定一次”）
- `E4.1` 换 kernel（判 eff 是 kernel 属性）

**G · 门禁（看数据前定稿）**
- `G0` 噪声底
- `G1` c_star 与 F/BW 是否同量级
- `G2` 曲线是否有拐点
- `G3` 重平衡收益是否超过 2−1/cp
- `G4` cp=1 是否准
- `G5` 换 kernel 后 c_star 是否移动
- `G6` 跨模型误差

---

# P0 · 预检（不占机时）

## env · 环境与配置

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 机型与卡数 |  | npu-smi info |  | Ascend910B 8卡机（自述 Ascend9392，8卡16核） |  |
| 单卡显存 *(`M_dev`)* _GB_ |  | npu-smi info |  | 64 |  |
| CANN 版本 |  | cat $ASCEND_HOME_PATH/version.cfg |  | 8.1.T18 |  |
| torch / torch_npu 版本 |  | python -c 'import torch, torch_npu' |  | 2.5.1 |  |
| 是否开启 activation checkpoint *(`recompute`)* |  | 读训练脚本的配置 |  |  |  |

## topo · 链路拓扑（决定 c* 的上界）

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 每对卡的链路类型 |  | npu-smi info -t topo |  |  |  |

## spec · 标称规格（只用于给上界，不用于拟合）

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 标称算力 *(`F_nom`)* _TFLOPS_ |  | 厂商规格，注明 dtype |  |  |  |
| 标称 HBM 带宽 *(`BW_nom`)* _TB/s_ |  | 厂商规格 |  |  |  |
| 标称链路带宽 *(`BW_link_nom`)* _GB/s_ |  | HCCS 聚合 |  |  |  |

# L0 · 常数（把参数从拟合里删掉）

## M1.1 · 同点噪声 · 短序列

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 同点 CV（短序列） *(`CV_short`)* _%_ |  | 同一配置连测 10 次 ×3 组 |  |  |  |

## M1.2 · 同点噪声 · 长序列

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 同点 CV（长序列） *(`CV_long`)* _%_ |  | 同上，最长序列 |  |  |  |

## M2.1 · 下发成本（空 kernel 斜率）

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 下发成本 *(`c_dispatch`)* _us/launch_ |  | 空 kernel 循环 1/8/64/256 次，取斜率 | CV_short |  |  |

## M2.2 · 单次启动成本（独立口径）

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 单次启动成本（独立口径） _us_ |  | 同一 kernel 打 1/2/4/8 次 |  |  |  |

## M2.3 · 不可切部分 t_fixed（独立口径）

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 不可切部分 *(`t_fixed`)* _ms_ |  | dense 主干换 no-op，只留 embedding+all2all+optimizer |  |  |  |

## M3.1 · HBM 带宽与效率

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 实测 HBM 带宽 *(`BW_hbm`)* _GB/s_ |  | 64/256 MB 纯 copy |  |  |  |
| HBM 效率 *(`bw_util`)* |  | BW_hbm / BW_nom | BW_hbm |  |  |

## M3.3 · M3.3

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 分数矩阵的访存遍数 *(`hops`)* _pass_ |  | 把 L² 分数矩阵单独写出再读回，按 HBM 带宽反算遍数 |  |  |  |

## M3.2 · 可达算力（eff_inf 的物理上限）

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 实测峰值算力 *(`F_real`)* _TFLOPS_ |  | 大 matmul，工作 dtype |  |  |  |

## M2.4 · M2.4

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 单卡查表时间 *(`t_emb`)* _ms_ |  | 只跑 embedding 查表，cp=1 |  |  |  |
| id 去重比 *(`dedup`)* |  | 不同 id 数 / 总 token 数（真实流量） |  |  |  |

## M4.1 · 同步开销

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 同步开销 *(`t_sync`)* _us_ |  | synchronize() ×1 / ×100 |  |  |  |

## M5 · 集合通信 时间–字节 曲线

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 每次集合通信延迟 *(`alpha`)* _us_ | collective=all_gather | 时间-字节曲线截距 ÷ 步数（S = ceil(log2 n)）；整条曲线两参数拟合 |  |  |  |
| 每次集合通信延迟 *(`alpha`)* _us_ | collective=all_to_all | 时间-字节曲线截距 ÷ 步数（S = ceil(log2 n)）；整条曲线两参数拟合 |  |  |  |
| 渐近集合通信带宽 *(`bw_comm`)* _GB/s_ | collective=all_gather | 时间-字节曲线斜率倒数；渐近值，不是小消息的表观带宽 |  |  |  |
| 渐近集合通信带宽 *(`bw_comm`)* _GB/s_ | collective=all_to_all | 时间-字节曲线斜率倒数；渐近值，不是小消息的表观带宽 |  |  |  |
| 扫描跨过拐点的证据 *(`v_star`)* _MB_ |  | V* = S·alpha·BW；检查扫描区间是否跨过它两侧 | alpha,bw_comm |  |  |
| 重叠比例 *(`overlap`)* |  | 910B/910C 的 HCCL 不做流水线重叠 -> 直接 pin 0 | bw_comm |  |  |

## M6.1 · 显存墙与激活字节

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 最大可跑序列长度 *(`T_max`)* _token_ | cp=1,2,4,8 | 二分 OOM 边界 |  |  |  |
| 每 token 激活字节 *(`a_act`)* _KiB/token_ |  | 由两档不同 chunk 的 T_max 反解 | T_max |  |  |
| 静态显存占用 *(`M_static`)* _GB_ |  | 同上 | T_max |  |  |

# L1 · 曲线与不均衡（不需要 CP）

## decl · 口径声明（层数）

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 标定时的 batch *(`b_cal`)* _sample_ |  | sweep.py --B 的值 |  |  |  |
| 计时模块的层数 *(`layers`)* _layer_ |  | sweep.py --layers 的值 |  |  |  |

## S1.1 · 效率曲线 eff(c) —— 唯一要拟合的东西

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 单层/整栈扫描：序列长度 *(`T`)* _token_ | T=512 | sweep.py --lengths |  |  |  |
| 单层/整栈扫描：序列长度 *(`T`)* _token_ | T=1024 | sweep.py --lengths |  |  |  |
| 单层/整栈扫描：序列长度 *(`T`)* _token_ | T=2048 | sweep.py --lengths |  |  |  |
| 单层/整栈扫描：序列长度 *(`T`)* _token_ | T=4096 | sweep.py --lengths |  |  |  |
| 单层/整栈扫描：序列长度 *(`T`)* _token_ | T=8192 | sweep.py --lengths |  |  |  |
| 单层/整栈扫描：序列长度 *(`T`)* _token_ | T=16384 | sweep.py --lengths |  |  |  |
| 对应中位数时间 *(`t_med`)* _ms_ |  | 固定协议：warmup5 / 中位数15 / 前后 synchronize / 不 flush |  |  |  |
| 对应 CV *(`cv`)* _%_ |  | 同上 |  |  |  |
| 提取出的效率 *(`eff`)* |  | eff = flops / (P·((t−t_fixed)/recomp − bytes/(BW·bw_util))) | t_med,t_fixed,bw_util,layers |  |  |

## S1.3 · S1.3

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| tile 对齐代价 *(`align_meas`)* |  | 同 kernel 在 cp=3/5/6 与 cp=4/8 的对比 | t_med |  |  |

## S1.2 · 整栈交叉验算

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 整栈扫描时间（交叉验算） *(`t_med_full`)* _ms_ |  | sweep.py --layers {model_layers} |  |  |  |

## S2.1 · 负载不均衡（只跑一个 rank 的 query）

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 每 rank 计算不均衡 *(`imbal_time`)* | scheme=block | 只跑某 rank 的 query 行，key 保留全长 |  |  |  |
| 每 rank 计算不均衡 *(`imbal_time`)* | scheme=balanced | 只跑某 rank 的 query 行，key 保留全长 |  |  |  |
| 每 rank 计算不均衡 *(`imbal_time`)* | scheme=striped | 只跑某 rank 的 query 行，key 保留全长 |  |  |  |
| 闭式预测的不均衡 *(`imbal_pred`)* |  | 2 − 1/cp（纯因果、连续分块） | imbal_time |  |  |

## S3.1 · 掩码拓扑：矩形 vs 三角

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| static 占比对时间的影响 *(`t_vs_static`)* _ms_ | share=0 | 扫 static_share |  |  |  |
| static 占比对时间的影响 *(`t_vs_static`)* _ms_ | share=0.5 | 扫 static_share |  |  |  |
| static 占比对时间的影响 *(`t_vs_static`)* _ms_ | share=0.85 | 扫 static_share |  |  |  |
| static 占比对时间的影响 *(`t_vs_static`)* _ms_ | share=0.95 | 扫 static_share |  |  |  |
| static 占比对不均衡的影响 *(`imbal_vs_static`)* |  | 同上的 max/mean | t_vs_static |  |  |

## S4.1 · padding 浪费

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| padding 浪费比例 *(`pad_waste`)* _%_ |  | T=max 与 T=mean 两次计时对比 |  |  |  |

# C0 · 组合（不需要 CP）

## C1 · 效率曲线三参数

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 效率渐近值 *(`eff_inf`)* |  | fit_curve 三参数之一 | eff |  |  |
| 半效率 chunk 长度 *(`c_star`)* _token_ |  | fit_curve | eff |  |  |
| 滚降陡度 *(`gamma`)* |  | fit_curve | eff |  |  |
| 拟合残差 *(`rmse_log`)* _%_ |  | fit_curve 的 rmse | eff_inf,c_star,gamma |  |  |

## C2 · 两个拐点

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| breakeven chunk *(`c_comm`)* _token_ |  | F_real / bw_comm（不是标称） | F_real,bw_comm |  |  |
| roofline 拐点 *(`c_ridge`)* _token_ |  | F_real / BW_hbm | F_real,BW_hbm |  |  |

# L2 · 验证与判决（需要 CP）

## E1.1 · 标定表：长度 × CP

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 端到端 step 时间 *(`step_ms`)* _ms_ | cp=1 | 标定表：未见过的长度与 CP 度<br>_T=4096 与 T=2048，B=32_ |  |  |  |
| 端到端 step 时间 *(`step_ms`)* _ms_ | cp=2 | 标定表：未见过的长度与 CP 度<br>_T=4096 与 T=2048，B=32_ |  |  |  |
| 端到端 step 时间 *(`step_ms`)* _ms_ | cp=4 | 标定表：未见过的长度与 CP 度<br>_T=4096 与 T=2048，B=32_ |  |  |  |
| 端到端 step 时间 *(`step_ms`)* _ms_ | cp=8 | 标定表：未见过的长度与 CP 度<br>_T=4096 与 T=2048，B=32_ |  |  |  |
| 对应 CV *(`step_cv`)* _%_ |  | 同一配置的重复测量<br>_T=4096 与 T=2048，B=32_ |  |  |  |
| 模型预测值 *(`step_pred`)* _ms_ |  | grcost predict，用 C0 的参数<br>_T=4096 与 T=2048，B=32_ | eff_inf,c_star,gamma |  |  |
| 误差 *(`err`)* _%_ |  | (pred−meas)/meas<br>_T=4096 与 T=2048，B=32_ |  |  |  |

## E1.2 · batch 扫描（判 alpha 是否真实）

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 端到端 step 时间 *(`step_ms`)* _ms_ | B=16 | batch 是否改善 CP 效率（判 alpha 是否真实）<br>_cp=4，T=4096_ |  |  |  |
| 端到端 step 时间 *(`step_ms`)* _ms_ | B=64 | batch 是否改善 CP 效率（判 alpha 是否真实）<br>_cp=4，T=4096_ |  |  |  |
| 对应 CV *(`step_cv`)* _%_ |  | 同一配置的重复测量<br>_cp=4，T=4096_ |  |  |  |
| 模型预测值 *(`step_pred`)* _ms_ |  | grcost predict，用 C0 的参数<br>_cp=4，T=4096_ | eff_inf,c_star,gamma |  |  |
| 误差 *(`err`)* _%_ |  | (pred−meas)/meas<br>_cp=4，T=4096_ |  |  |  |

## E2.1 · balancer 实测（判闭式上界）

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 端到端 step 时间 *(`step_ms`)* _ms_ | scheme=block | balancer 实测收益 vs 2−1/cp 上界<br>_T=4096, cp=4, B=32_ |  |  |  |
| 端到端 step 时间 *(`step_ms`)* _ms_ | scheme=balanced | balancer 实测收益 vs 2−1/cp 上界<br>_T=4096, cp=4, B=32_ |  |  |  |
| 对应 CV *(`step_cv`)* _%_ |  | 同一配置的重复测量<br>_T=4096, cp=4, B=32_ |  |  |  |
| 模型预测值 *(`step_pred`)* _ms_ |  | grcost predict，用 C0 的参数<br>_T=4096, cp=4, B=32_ | eff_inf,c_star,gamma |  |  |
| 误差 *(`err`)* _%_ |  | (pred−meas)/meas<br>_T=4096, cp=4, B=32_ |  |  |  |
| 重平衡实测收益 *(`gain_rebal`)* _%_ |  | block 与 balanced 的时间比减 1 | step_ms |  |  |

## E3.1 · 跨模型迁移（判“标定一次”）

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 端到端 step 时间 *(`step_ms`)* _ms_ | model=hstu-base | 跨模型不重标定（判'标定一次'）<br>_T=4096, cp=1 与 cp=4, B=32_ |  |  |  |
| 端到端 step 时间 *(`step_ms`)* _ms_ | model=longer | 跨模型不重标定（判'标定一次'）<br>_T=4096, cp=1 与 cp=4, B=32_ |  |  |  |
| 对应 CV *(`step_cv`)* _%_ |  | 同一配置的重复测量<br>_T=4096, cp=1 与 cp=4, B=32_ |  |  |  |
| 模型预测值 *(`step_pred`)* _ms_ |  | grcost predict，用 C0 的参数<br>_T=4096, cp=1 与 cp=4, B=32_ | eff_inf,c_star,gamma |  |  |
| 误差 *(`err`)* _%_ |  | (pred−meas)/meas<br>_T=4096, cp=1 与 cp=4, B=32_ |  |  |  |

## E4.1 · 换 kernel（判 eff 是 kernel 属性）

| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |
|:--|:--|:--|:--|:--|:--|
| 端到端 step 时间 *(`step_ms`)* _ms_ | kernel=triton | 换 kernel 后 c_star 是否移动（判'标定一次'）<br>_T=4096, cp=1 与 cp=4, B=32_ |  |  |  |
| 端到端 step 时间 *(`step_ms`)* _ms_ | kernel=fused | 换 kernel 后 c_star 是否移动（判'标定一次'）<br>_T=4096, cp=1 与 cp=4, B=32_ |  |  |  |
| 对应 CV *(`step_cv`)* _%_ |  | 同一配置的重复测量<br>_T=4096, cp=1 与 cp=4, B=32_ |  |  |  |
| 模型预测值 *(`step_pred`)* _ms_ |  | grcost predict，用 C0 的参数<br>_T=4096, cp=1 与 cp=4, B=32_ | eff_inf,c_star,gamma |  |  |
| 误差 *(`err`)* _%_ |  | (pred−meas)/meas<br>_T=4096, cp=1 与 cp=4, B=32_ |  |  |  |
| 第二实现的 c_star *(`c_star_alt`)* _token_ |  | 只重拟合 eff 三参数，其余不动 | c_star |  |  |

# G · 门禁（看数据前定稿）

## G0 · 噪声底

| 检查什么 | 不通过怎么办 | 结论 |
|:--|:--|:--|
| 同点 CV < 5% | 放弃细粒度判据，不拟合 gamma |  |

## G1 · c_star 与 F/BW 是否同量级

| 检查什么 | 不通过怎么办 | 结论 |
|:--|:--|:--|
| c_star 与 c_comm 同量级（10× 内） | 曲线在吸收别的东西，停 |  |

## G2 · 曲线是否有拐点

| 检查什么 | 不通过怎么办 | 结论 |
|:--|:--|:--|
| eff 曲线在扫描区间内有拐点 | 扩区间，不是加参数 |  |

## G3 · 重平衡收益是否超过 2−1/cp

| 检查什么 | 不通过怎么办 | 结论 |
|:--|:--|:--|
| 重平衡收益 ≤ 2−1/cp | 闭式或掩码假设错 |  |

## G4 · cp=1 是否准

| 检查什么 | 不通过怎么办 | 结论 |
|:--|:--|:--|
| cp=1 的 MAPE < 5% | 工作量计数器错，停止标定 |  |

## G5 · 换 kernel 后 c_star 是否移动

| 检查什么 | 不通过怎么办 | 结论 |
|:--|:--|:--|
| 换 kernel 后 c_star 移动 < 3× | 撤'标定一次' |  |

## G6 · 跨模型误差

| 检查什么 | 不通过怎么办 | 结论 |
|:--|:--|:--|
| 跨模型误差 < 20% | 改为按模型标定，缩小 claim |  |

