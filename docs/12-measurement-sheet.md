# NPU 测量数据填报表

> 生成自 `grcost.sheet.registry()`，与 `manifest` 的作业 ID 同步。
> `依赖` 列为空表示可独立填；非空的格子**必须先有前置值**，
> 否则得到的模型内部自洽但无法被证伪。

## 怎么用

1. 先填 P0 与 G 两段的判据（判据要在看数据前定稿）。
2. 按 L0 → L1 → C0 → L2 顺序填，每段结束检查门禁。
3. `provenance` 列的五个值含义：
   `measured` 你测的 · `derived` 由测量算出 · `fitted` 优化器选的 ·
   `pinned` 必须给定且不许拟合 · `declared` 配置事实。
4. `变体` 非空的行，CSV 里已展开成多行，每行填一个配置；`固定配置` 列说明该行其余参数。

## 五个绝对不要让它进拟合的量

`t_fixed` · `overlap` · `bw_util` · `recompute` · `layers`

前四个要么可由微基准直接测、要么在公式里只以乘积出现；`layers` 是口径。
任何一个搞错，拟合都会给出内部自洽但物理错误的结果。

## P0 · 预检（不占机时）

| 作业 | 指标 | 符号 | 单位 | 来源 | 怎么得到 | 期望/判据 | 依赖 | 变体 | 固定配置 | 值 | CV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| env | 机型与卡数 | `` |  | declared | npu-smi info | e.g. Atlas 800T A2 x8 |  |  |  |  |  |
| env | 单卡显存 | `M_dev` | GB | declared | npu-smi info |  |  |  |  |  |  |
| env | CANN 版本 | `` |  | declared | cat $ASCEND_HOME_PATH/version.cfg |  |  |  |  |  |  |
| env | torch / torch_npu 版本 | `` |  | declared | python -c 'import torch, torch_npu' |  |  |  |  |  |  |
| topo | 每对卡的链路类型 | `` |  | declared | npu-smi info -t topo<br>混有 SYS/PHB 时 2 卡带宽会异常低，直接改变 c* | 记录是否有非 HCCS 的对 |  |  |  |  |  |
| spec | 标称算力 | `F_nom` | TFLOPS | declared | 厂商规格，注明 dtype | bf16/fp16 各一 |  |  |  |  |  |
| spec | 标称 HBM 带宽 | `BW_nom` | TB/s | declared | 厂商规格 |  |  |  |  |  |  |
| spec | 标称链路带宽 | `BW_link_nom` | GB/s | declared | HCCS 聚合<br>仅用于给 c* 一个上界，不能用于拟合 |  |  |  |  |  |  |
| env | 是否开启 activation checkpoint | `recompute` |  | declared | 读训练脚本的配置<br>pin 住；直接由配置决定，不要拟合 | 1.0 或 4/3 |  |  |  |  |  |

## L0 · 常数（把参数从拟合里删掉）

| 作业 | 指标 | 符号 | 单位 | 来源 | 怎么得到 | 期望/判据 | 依赖 | 变体 | 固定配置 | 值 | CV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1.1 | 同点 CV（短序列） | `CV_short` | % | measured | 同一配置连测 10 次 ×3 组<br>这是后面所有判据的分辨率 | < 5% |  |  |  |  |  |
| M1.2 | 同点 CV（长序列） | `CV_long` | % | measured | 同上，最长序列 | < 5% |  |  |  |  |  |
| M2.1 | 下发成本 | `c_dispatch` | us/launch | measured | 空 kernel 循环 1/8/64/256 次，取斜率 |  | CV_short |  |  |  |  |
| M2.2 | 单次启动成本（独立口径） | `` | us | measured | 同一 kernel 打 1/2/4/8 次 | 与 c_dispatch 一致 |  |  |  |  |  |
| M2.3 | 不可切部分 | `t_fixed` | ms | measured | dense 主干换 no-op，只留 embedding+all2all+optimizer<br>第二条独立口径；两条路同量级才可信 |  |  |  |  |  |  |
| M3.1 | 实测 HBM 带宽 | `BW_hbm` | GB/s | measured | 64/256 MB 纯 copy |  |  |  |  |  |  |
| M3.1 | HBM 效率 | `bw_util` |  | derived | BW_hbm / BW_nom<br>pin 住，不要拟合 | 0.5-0.8 | BW_hbm |  |  |  |  |
| M3.3 | 分数矩阵的访存遍数 | `hops` | pass | measured | 把 L² 分数矩阵单独写出再读回，按 HBM 带宽反算遍数<br>DAV_2201 无 L0C→UB，至少要写一次读一次；A5 才有直通 | 2–4 |  |  |  |  |  |
| M3.2 | 实测峰值算力 | `F_real` | TFLOPS | measured | 大 matmul，工作 dtype<br>eff_inf 的物理上限 |  |  |  |  |  |  |
| M2.4 | 单卡查表时间 | `t_emb` | ms | measured | 只跑 embedding 查表，cp=1<br>按 cp 切分时乘 1/cp；但要注意随之而来的 re-gather |  |  |  |  |  |  |
| M2.4 | id 去重比 | `dedup` |  | measured | 不同 id 数 / 总 token 数（真实流量）<br>按 token 数建模会高估 1/dedup 倍 | 0.1-0.5 |  |  |  |  |  |
| M4.1 | 同步开销 | `t_sync` | us | measured | synchronize() ×1 / ×100 | 过大会污染所有计时 |  |  |  |  |  |
| M5 | 每次集合通信延迟<br>*每* `collective=all_gather\|collective=all_to_all` | `alpha` | us | fitted | 时间-字节曲线截距 ÷ 步数（S = ceil(log2 n)）；整条曲线两参数拟合<br>不是分段拟合：V* = S·alpha·BW 随卡数移动，分段无法外推 |  |  | collective=all_gather\|collective=all_to_all |  |  |  |
| M5 | 渐近集合通信带宽<br>*每* `collective=all_gather\|collective=all_to_all` | `bw_comm` | GB/s | fitted | 时间-字节曲线斜率倒数；渐近值，不是小消息的表观带宽<br>表观带宽随消息变小而低，是 latency 摊薄，不是链路的性质 | 大消息处应接近实测峰值 |  | collective=all_gather\|collective=all_to_all |  |  |  |
| M5 | 扫描跨过拐点的证据 | `v_star` | MB | derived | V* = S·alpha·BW；检查扫描区间是否跨过它两侧<br>若 V* 落在区间外，alpha 与 bw 不可分，拟合值不可信 |  | alpha,bw_comm |  |  |  |  |
| M5 | 重叠比例 | `overlap` |  | pinned | 910B/910C 的 HCCL 不做流水线重叠 -> 直接 pin 0<br>pin！放开拟合会退化成 eta->1, bw->0 | 0（Ascend）；GPU 上另测 | bw_comm |  |  |  |  |
| M6.1 | 最大可跑序列长度<br>*每* `cp=1,2,4,8` | `T_max` | token | measured | 二分 OOM 边界<br>各档 chunk 长度必须不同，否则 a 不可辨识 |  |  | cp=1,2,4,8 |  |  |  |
| M6.1 | 每 token 激活字节 | `a_act` | KiB/token | fitted | 由两档不同 chunk 的 T_max 反解<br>chunk 相同时 fit_memory 返回 identifiable=False |  | T_max |  |  |  |  |
| M6.1 | 静态显存占用 | `M_static` | GB | fitted | 同上 |  | T_max |  |  |  |  |

## L1 · 曲线与不均衡（不需要 CP）

| 作业 | 指标 | 符号 | 单位 | 来源 | 怎么得到 | 期望/判据 | 依赖 | 变体 | 固定配置 | 值 | CV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| decl | 标定时的 batch | `b_cal` | sample | declared | sweep.py --B 的值<br>eff 是 B*T/CP 的函数；预测时的 batch 与它相差不得超过 2 倍（G7） | = 32 |  |  |  |  |  |
| decl | 计时模块的层数 | `layers` | layer | declared | sweep.py --layers 的值<br>必须与预测目标一致，否则 eff 错 n_layers 倍 | = 15 或 1 |  |  |  |  |  |
| S1.1 | 单层/整栈扫描：序列长度<br>*每* `T=512\|T=1024\|T=2048\|T=4096\|T=8192\|T=16384` | `T` | token | declared | sweep.py --lengths |  |  | T=512\|T=1024\|T=2048\|T=4096\|T=8192\|T=16384 |  |  |  |
| S1.1 | 对应中位数时间 | `t_med` | ms | measured | 固定协议：warmup5 / 中位数15 / 前后 synchronize / 不 flush |  |  |  |  |  |  |
| S1.1 | 对应 CV | `cv` | % | measured | 同上 | < 5% |  |  |  |  |  |
| S1.1 | 提取出的效率 | `eff` |  | derived | eff = flops / (P·((t−t_fixed)/recomp − bytes/(BW·bw_util)))<br>代数解出，不是联合拟合 |  | t_med,t_fixed,bw_util,layers |  |  |  |  |
| S1.3 | tile 对齐代价 | `align_meas` |  | measured | 同 kernel 在 cp=3/5/6 与 cp=4/8 的对比<br>若实测与 1/u(c) 不符，说明 eff 已含该效应，勿重复计算（G8） | = 1/u(c) | t_med |  |  |  |  |
| S1.2 | 整栈扫描时间（交叉验算） | `t_med_full` | ms | measured | sweep.py --layers {model_layers}<br>与 S1.1 × 层数 应当吻合 |  |  |  |  |  |  |
| S2.1 | 每 rank 计算不均衡<br>*每* `scheme=block\|scheme=balanced\|scheme=striped` | `imbal_time` |  | measured | 只跑某 rank 的 query 行，key 保留全长<br>每个 scheme 再按 cp=2,4,8 各一行 |  |  | scheme=block\|scheme=balanced\|scheme=striped |  |  |  |
| S2.1 | 闭式预测的不均衡 | `imbal_pred` |  | derived | 2 − 1/cp（纯因果、连续分块）<br>实测不得超过此上界，超过说明掩码假设错 |  | imbal_time |  |  |  |  |
| S3.1 | static 占比对时间的影响<br>*每* `share=0\|share=0.5\|share=0.85\|share=0.95` | `t_vs_static` | ms | measured | 扫 static_share<br>预测：时间随 share 上升 |  |  | share=0\|share=0.5\|share=0.85\|share=0.95 |  |  |  |
| S3.1 | static 占比对不均衡的影响 | `imbal_vs_static` |  | derived | 同上的 max/mean<br>预测：不均衡随 share 下降。两条必须同时成立 |  | t_vs_static |  |  |  |  |
| S4.1 | padding 浪费比例 | `pad_waste` | % | measured | T=max 与 T=mean 两次计时对比<br>只报不除；归入 t_fixed |  |  |  |  |  |  |

## C0 · 组合（不需要 CP）

| 作业 | 指标 | 符号 | 单位 | 来源 | 怎么得到 | 期望/判据 | 依赖 | 变体 | 固定配置 | 值 | CV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 效率渐近值 | `eff_inf` |  | fitted | fit_curve 三参数之一<br>超过 F_real/F_nom 说明在补偿别的东西 | 0.2-0.5 | eff |  |  |  |  |
| C1 | 半效率 chunk 长度 | `c_star` | token | fitted | fit_curve<br>这是 G1 的判据 | 与 F/BW 同量级（10× 内） | eff |  |  |  |  |
| C1 | 滚降陡度 | `gamma` |  | fitted | fit_curve | 0.5-2 | eff |  |  |  |  |
| C1 | 拟合残差 | `rmse_log` | % | derived | fit_curve 的 rmse | < 3% | eff_inf,c_star,gamma |  |  |  |  |
| C2 | breakeven chunk | `c_comm` | token | derived | F_real / bw_comm（不是标称）<br>跨节点时另算：F/BW_link_out |  | F_real,bw_comm |  |  |  |  |
| C2 | roofline 拐点 | `c_ridge` | token | derived | F_real / BW_hbm<br>与 c_comm 是两个不同门槛 |  | F_real,BW_hbm |  |  |  |  |

## L2 · 验证与判决（需要 CP）

| 作业 | 指标 | 符号 | 单位 | 来源 | 怎么得到 | 期望/判据 | 依赖 | 变体 | 固定配置 | 值 | CV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1.1 | 端到端 step 时间<br>*每* `cp=1\|cp=2\|cp=4\|cp=8` | `step_ms` | ms | measured | 标定表：未见过的长度与 CP 度 | cp=1 必须先 < 5%，否则停止标定 |  | cp=1\|cp=2\|cp=4\|cp=8 | T=4096 与 T=2048，B=32 |  |  |
| E1.1 | 对应 CV | `step_cv` | % | measured | 同一配置的重复测量 | < 5% |  |  | T=4096 与 T=2048，B=32 |  |  |
| E1.1 | 模型预测值 | `step_pred` | ms | derived | grcost predict，用 C0 的参数 |  | eff_inf,c_star,gamma |  | T=4096 与 T=2048，B=32 |  |  |
| E1.1 | 误差 | `err` | % | derived | (pred−meas)/meas |  |  |  | T=4096 与 T=2048，B=32 |  |  |
| E1.2 | 端到端 step 时间<br>*每* `B=16\|B=64` | `step_ms` | ms | measured | batch 是否改善 CP 效率（判 alpha 是否真实） | 若拟合出的 alpha~0，模型会预测无 batch 效应，与文献矛盾 |  | B=16\|B=64 | cp=4，T=4096 |  |  |
| E1.2 | 对应 CV | `step_cv` | % | measured | 同一配置的重复测量 | < 5% |  |  | cp=4，T=4096 |  |  |
| E1.2 | 模型预测值 | `step_pred` | ms | derived | grcost predict，用 C0 的参数 |  | eff_inf,c_star,gamma |  | cp=4，T=4096 |  |  |
| E1.2 | 误差 | `err` | % | derived | (pred−meas)/meas |  |  |  | cp=4，T=4096 |  |  |
| E2.1 | 端到端 step 时间<br>*每* `scheme=block\|scheme=balanced` | `step_ms` | ms | measured | balancer 实测收益 vs 2−1/cp 上界 | 收益不得超过 2−1/cp；MTGR 布局下应远小于文献的 37% |  | scheme=block\|scheme=balanced | T=4096, cp=4, B=32 |  |  |
| E2.1 | 对应 CV | `step_cv` | % | measured | 同一配置的重复测量 | < 5% |  |  | T=4096, cp=4, B=32 |  |  |
| E2.1 | 模型预测值 | `step_pred` | ms | derived | grcost predict，用 C0 的参数 |  | eff_inf,c_star,gamma |  | T=4096, cp=4, B=32 |  |  |
| E2.1 | 误差 | `err` | % | derived | (pred−meas)/meas |  |  |  | T=4096, cp=4, B=32 |  |  |
| E3.1 | 端到端 step 时间<br>*每* `model=hstu-base\|model=longer` | `step_ms` | ms | measured | 跨模型不重标定（判'标定一次'） | 误差 < 20%，否则改为按模型标定 |  | model=hstu-base\|model=longer | T=4096, cp=1 与 cp=4, B=32 |  |  |
| E3.1 | 对应 CV | `step_cv` | % | measured | 同一配置的重复测量 | < 5% |  |  | T=4096, cp=1 与 cp=4, B=32 |  |  |
| E3.1 | 模型预测值 | `step_pred` | ms | derived | grcost predict，用 C0 的参数 |  | eff_inf,c_star,gamma |  | T=4096, cp=1 与 cp=4, B=32 |  |  |
| E3.1 | 误差 | `err` | % | derived | (pred−meas)/meas |  |  |  | T=4096, cp=1 与 cp=4, B=32 |  |  |
| E4.1 | 端到端 step 时间<br>*每* `kernel=triton\|kernel=fused` | `step_ms` | ms | measured | 换 kernel 后 c_star 是否移动（判'标定一次'） | c_star 移动 < 3×，否则撤'标定一次' |  | kernel=triton\|kernel=fused | T=4096, cp=1 与 cp=4, B=32 |  |  |
| E4.1 | 对应 CV | `step_cv` | % | measured | 同一配置的重复测量 | < 5% |  |  | T=4096, cp=1 与 cp=4, B=32 |  |  |
| E4.1 | 模型预测值 | `step_pred` | ms | derived | grcost predict，用 C0 的参数 |  | eff_inf,c_star,gamma |  | T=4096, cp=1 与 cp=4, B=32 |  |  |
| E4.1 | 误差 | `err` | % | derived | (pred−meas)/meas |  |  |  | T=4096, cp=1 与 cp=4, B=32 |  |  |
| E4.1 | 第二实现的 c_star | `c_star_alt` | token | fitted | 只重拟合 eff 三参数，其余不动<br>超过则撤'标定一次' | < 3× c_star | c_star |  |  |  |  |
| E2.1 | 重平衡实测收益 | `gain_rebal` | % | derived | block 与 balanced 的时间比减 1<br>MTGR 布局下应远小于文献的 37% | ≤ 2−1/cp | step_ms |  |  |  |  |

## G · 门禁（看数据前定稿）

| 门 | 判据 | 不过怎么办 | 结论 |
|---|---|---|---|
| G0 | 同点 CV < 5% | 放弃细粒度判据，不拟合 gamma |  |
| G1 | c_star 与 c_comm 同量级（10× 内） | 曲线在吸收别的东西，停 |  |
| G2 | eff 曲线在扫描区间内有拐点 | 扩区间，不是加参数 |  |
| G3 | 重平衡收益 ≤ 2−1/cp | 闭式或掩码假设错 |  |
| G4 | cp=1 的 MAPE < 5% | 工作量计数器错，停止标定 |  |
| G5 | 换 kernel 后 c_star 移动 < 3× | 撤'标定一次' |  |
| G6 | 跨模型误差 < 20% | 改为按模型标定，缩小 claim |  |

