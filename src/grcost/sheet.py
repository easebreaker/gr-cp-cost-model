"""The blank measurement sheet: what to write down, per stage and per metric.

Single source of truth for the data-collection form.  Generated rather than
hand-written so that the job IDs here can never drift away from
``manifest.build_plan()`` and the parameters can never drift away from
``cost.Params``.

Every row declares its **provenance**, because that is the distinction that
matters when someone later asks "where did this number come from?":

    measured   you timed it
    derived    computed from measurements (e.g. bw_util = measured / nominal)
    fitted     the optimiser chose it
    pinned     a constant you must supply and must NOT fit
    declared   a configuration fact (model depth, dtype, topology)

Rows with ``requires`` set are hard prerequisites: filling a later row while an
earlier prerequisite is still blank produces a model that cannot be falsified.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

STAGES = {
    "P0": "预检（不占机时）",
    "L0": "常数（把参数从拟合里删掉）",
    "L1": "曲线与不均衡（不需要 CP）",
    "C0": "组合（不需要 CP）",
    "L2": "验证与判决（需要 CP）",
    "G": "门禁（看数据前定稿）",
}

PROVENANCE = ("declared", "measured", "derived", "fitted", "pinned")

#: Factors that convert the *form's* unit into the unit ``Params`` expects.
#: Every entry here is a factor of 1000 or more, i.e. exactly the kind of
#: slip that produces a model which fits and predicts nonsense: 26 written as
#: ``alpha`` in microseconds becomes 26 *seconds* per collective if nobody
#: converts, which is a million times too big and still "fits" because the
#: optimiser compensates elsewhere.
CANONICAL = {
    "alpha": 1e-6,      # us        -> s
    "bw_comm": 1e9,     # GB/s      -> bytes/s
    "F_real": 1e12,     # TFLOPS    -> FLOPS
    "F_nom": 1e12,
    "BW_hbm": 1e9,      # GB/s      -> bytes/s
    "BW_nom": 1e12,     # TB/s      -> bytes/s
    "BW_link_nom": 1e9,
    "M_dev": 1.0,       # GB, stays human-readable
    "a_act": 1024.0,    # KiB/token -> bytes/token
}

#: Symbols whose form unit is already the model's unit.
IDENTITY = ("eff_inf", "c_star", "gamma", "overlap", "bw_util", "recompute",
            "t_fixed", "t_med", "cv", "err", "rmse_log", "c_comm", "c_ridge",
            "step_ms", "step_pred", "M_static", "CV_short", "CV_long",
            "c_dispatch", "T_max", "layers", "pad_waste", "gain_rebal")


def to_model_units(symbol: str, value: float) -> float:
    """Convert a number typed into the form into the unit the model uses."""
    return value * CANONICAL.get(symbol, 1.0)


@dataclass(frozen=True)
class Row:
    stage: str
    section: str          # job id or sub-stage
    metric: str           # what to write down
    symbol: str = ""
    unit: str = ""
    source: str = "measured"
    how: str = ""         # how to obtain it
    target: str = ""      # expected value / gate
    requires: str = ""    # prerequisite metric symbols
    repeat: str = ""      # "a|b|c" -> one CSV line per value
    config: str = ""      # fixed part of the configuration for every line
    note: str = ""


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------
def registry(T_ref: int = 4096, B: int = 32, L: Sequence[int] = (512, 1024, 2048, 4096, 8192, 16384),
             model_layers: int = 15, d: int = 768, heads: int = 3) -> List[Row]:
    rows: List[Row] = []
    a = rows.append

    # ---------------- P0 -------------------------------------------------
    a(Row("P0", "env", "机型与卡数", "", "", "declared",
          "npu-smi info", "e.g. Atlas 800T A2 x8"))
    a(Row("P0", "env", "单卡显存", "M_dev", "GB", "declared", "npu-smi info", ""))
    a(Row("P0", "env", "CANN 版本", "", "", "declared",
          "cat $ASCEND_HOME_PATH/version.cfg"))
    a(Row("P0", "env", "torch / torch_npu 版本", "", "", "declared",
          "python -c 'import torch, torch_npu'", ""))
    a(Row("P0", "topo", "每对卡的链路类型", "", "", "declared",
          "npu-smi info -t topo", "记录是否有非 HCCS 的对",
          note="混有 SYS/PHB 时 2 卡带宽会异常低，直接改变 c*"))
    a(Row("P0", "spec", "标称算力", "F_nom", "TFLOPS", "declared",
          "厂商规格，注明 dtype", "bf16/fp16 各一"))
    a(Row("P0", "spec", "标称 HBM 带宽", "BW_nom", "TB/s", "declared", "厂商规格"))
    a(Row("P0", "spec", "标称链路带宽", "BW_link_nom", "GB/s", "declared",
          "HCCS 聚合", note="仅用于给 c* 一个上界，不能用于拟合"))

    # ---------------- L0 -------------------------------------------------
    a(Row("L0", "M1.1", "同点 CV（短序列）", "CV_short", "%", "measured",
          "同一配置连测 10 次 ×3 组", "< 5%", note="这是后面所有判据的分辨率"))
    a(Row("L0", "M1.2", "同点 CV（长序列）", "CV_long", "%", "measured",
          "同上，最长序列", "< 5%"))
    a(Row("L0", "M2.1", "下发成本", "c_dispatch", "us/launch", "measured",
          "空 kernel 循环 1/8/64/256 次，取斜率", "", requires="CV_short"))
    a(Row("L0", "M2.2", "单次启动成本（独立口径）", "", "us", "measured",
          "同一 kernel 打 1/2/4/8 次", "与 c_dispatch 一致"))
    a(Row("L0", "M2.3", "不可切部分", "t_fixed", "ms", "measured",
          "dense 主干换 no-op，只留 embedding+all2all+optimizer",
          "", note="第二条独立口径；两条路同量级才可信"))
    a(Row("L0", "M3.1", "实测 HBM 带宽", "BW_hbm", "GB/s", "measured",
          "64/256 MB 纯 copy", ""))
    a(Row("L0", "M3.1", "HBM 效率", "bw_util", "", "derived",
          "BW_hbm / BW_nom", "0.5-0.8", requires="BW_hbm",
          note="pin 住，不要拟合"))
    a(Row("L0", "M3.3", "分数矩阵的访存遍数", "hops", "pass", "measured",
          "把 L² 分数矩阵单独写出再读回，按 HBM 带宽反算遍数", "2–4",
          note="DAV_2201 无 L0C→UB，至少要写一次读一次；A5 才有直通"))
    a(Row("L0", "M3.2", "实测峰值算力", "F_real", "TFLOPS", "measured",
          "大 matmul，工作 dtype", "", note="eff_inf 的物理上限"))
    a(Row("L0", "M2.4", "单卡查表时间", "t_emb", "ms", "measured",
          "只跑 embedding 查表，cp=1", "",
          note="按 cp 切分时乘 1/cp；但要注意随之而来的 re-gather"))
    a(Row("L0", "M2.4", "id 去重比", "dedup", "", "measured",
          "不同 id 数 / 总 token 数（真实流量）", "0.1-0.5",
          note="按 token 数建模会高估 1/dedup 倍"))
    a(Row("P0", "env", "是否开启 activation checkpoint", "recompute", "", "declared",
          "读训练脚本的配置", "1.0 或 4/3",
          note="pin 住；直接由配置决定，不要拟合"))
    a(Row("L0", "M4.1", "同步开销", "t_sync", "us", "measured",
          "synchronize() ×1 / ×100", "过大会污染所有计时"))
    a(Row("L0", "M5", "每次集合通信延迟", "alpha", "us", "fitted",
          "时间-字节曲线截距 ÷ 步数（S = ceil(log2 n)）；整条曲线两参数拟合",
          "", repeat="collective=all_gather|collective=all_to_all",
          note="不是分段拟合：V* = S·alpha·BW 随卡数移动，分段无法外推"))
    a(Row("L0", "M5", "渐近集合通信带宽", "bw_comm", "GB/s", "fitted",
          "时间-字节曲线斜率倒数；渐近值，不是小消息的表观带宽",
          "大消息处应接近实测峰值",
          repeat="collective=all_gather|collective=all_to_all",
          note="表观带宽随消息变小而低，是 latency 摊薄，不是链路的性质"))
    a(Row("L0", "M5", "扫描跨过拐点的证据", "v_star", "MB", "derived",
          "V* = S·alpha·BW；检查扫描区间是否跨过它两侧", "",
          requires="alpha,bw_comm",
          note="若 V* 落在区间外，alpha 与 bw 不可分，拟合值不可信"))
    a(Row("L0", "M5", "重叠比例", "overlap", "", "pinned",
          "910B/910C 的 HCCL 不做流水线重叠 -> 直接 pin 0",
          "0（Ascend）；GPU 上另测", requires="bw_comm",
          note="pin！放开拟合会退化成 eta->1, bw->0"))
    a(Row("L0", "M6.1", "最大可跑序列长度", "T_max", "token", "measured",
          "二分 OOM 边界", "", repeat="cp=1,2,4,8",
          note="各档 chunk 长度必须不同，否则 a 不可辨识"))
    a(Row("L0", "M6.1", "每 token 激活字节", "a_act", "KiB/token", "fitted",
          "由两档不同 chunk 的 T_max 反解", "",
          requires="T_max", note="chunk 相同时 fit_memory 返回 identifiable=False"))
    a(Row("L0", "M6.1", "静态显存占用", "M_static", "GB", "fitted",
          "同上", "", requires="T_max"))

    # ---------------- L1 -------------------------------------------------
    a(Row("L1", "decl", "标定时的 batch", "b_cal", "sample", "declared",
          "sweep.py --B 的值", f"= {B}",
          note="eff 是 B*T/CP 的函数；预测时的 batch 与它相差不得超过 2 倍（G7）"))
    a(Row("L1", "decl", "计时模块的层数", "layers", "layer", "declared",
          "sweep.py --layers 的值", f"= {model_layers} 或 1",
          note="必须与预测目标一致，否则 eff 错 n_layers 倍"))
    a(Row("L1", "S1.1", "单层/整栈扫描：序列长度", "T", "token", "declared",
          "sweep.py --lengths", "",
          repeat="|".join(f"T={x}" for x in L)))
    a(Row("L1", "S1.1", "对应中位数时间", "t_med", "ms", "measured",
          "固定协议：warmup5 / 中位数15 / 前后 synchronize / 不 flush", ""))
    a(Row("L1", "S1.1", "对应 CV", "cv", "%", "measured", "同上", "< 5%"))
    a(Row("L1", "S1.1", "提取出的效率", "eff", "", "derived",
          "eff = flops / (P·((t−t_fixed)/recomp − bytes/(BW·bw_util)))", "",
          requires="t_med,t_fixed,bw_util,layers",
          note="代数解出，不是联合拟合"))
    a(Row("L1", "S1.3", "tile 对齐代价", "align_meas", "", "measured",
          "同 kernel 在 cp=3/5/6 与 cp=4/8 的对比", "= 1/u(c)",
          requires="t_med",
          note="若实测与 1/u(c) 不符，说明 eff 已含该效应，勿重复计算（G8）"))
    a(Row("L1", "S1.2", "整栈扫描时间（交叉验算）", "t_med_full", "ms", "measured",
          "sweep.py --layers {model_layers}", "",
          note="与 S1.1 × 层数 应当吻合"))
    a(Row("L1", "S2.1", "每 rank 计算不均衡", "imbal_time", "", "measured",
          "只跑某 rank 的 query 行，key 保留全长", "",
          repeat="scheme=block|scheme=balanced|scheme=striped",
          note="每个 scheme 再按 cp=2,4,8 各一行"))
    a(Row("L1", "S2.1", "闭式预测的不均衡", "imbal_pred", "", "derived",
          "2 − 1/cp（纯因果、连续分块）", "", requires="imbal_time",
          note="实测不得超过此上界，超过说明掩码假设错"))
    a(Row("L1", "S3.1", "static 占比对时间的影响", "t_vs_static", "ms", "measured",
          "扫 static_share", "",
          repeat="share=0|share=0.5|share=0.85|share=0.95",
          note="预测：时间随 share 上升"))
    a(Row("L1", "S3.1", "static 占比对不均衡的影响", "imbal_vs_static", "", "derived",
          "同上的 max/mean", "", requires="t_vs_static",
          note="预测：不均衡随 share 下降。两条必须同时成立"))
    a(Row("L1", "S4.1", "padding 浪费比例", "pad_waste", "%", "measured",
          "T=max 与 T=mean 两次计时对比", "",
          note="只报不除；归入 t_fixed"))

    # ---------------- C0 -------------------------------------------------
    a(Row("C0", "C1", "效率渐近值", "eff_inf", "", "fitted",
          "fit_curve 三参数之一", "0.2-0.5",
          requires="eff", note="超过 F_real/F_nom 说明在补偿别的东西"))
    a(Row("C0", "C1", "半效率 chunk 长度", "c_star", "token", "fitted",
          "fit_curve", "与 F/BW 同量级（10× 内）",
          requires="eff", note="这是 G1 的判据"))
    a(Row("C0", "C1", "滚降陡度", "gamma", "", "fitted", "fit_curve", "0.5-2",
          requires="eff"))
    a(Row("C0", "C1", "拟合残差", "rmse_log", "%", "derived",
          "fit_curve 的 rmse", "< 3%", requires="eff_inf,c_star,gamma"))
    a(Row("C0", "C2", "breakeven chunk", "c_comm", "token", "derived",
          "F_real / bw_comm（不是标称）", "",
          requires="F_real,bw_comm", note="跨节点时另算：F/BW_link_out"))
    a(Row("C0", "C2", "roofline 拐点", "c_ridge", "token", "derived",
          "F_real / BW_hbm", "", requires="F_real,BW_hbm",
          note="与 c_comm 是两个不同门槛"))

    # ---------------- L2 -------------------------------------------------
    # Each L2 job varies a *different* knob, so the grid and the note must be
    # per-job.  A single generic "cp=1..8 at T, B" is wrong for three of them.
    l2_jobs = [
        ("E1.1", "标定表：未见过的长度与 CP 度",
         "cp=1|cp=2|cp=4|cp=8", f"T={T_ref} 与 T={T_ref // 2}，B={B}",
         "cp=1 必须先 < 5%，否则停止标定"),
        ("E1.2", "batch 是否改善 CP 效率（判 alpha 是否真实）",
         "B=16|B=64", f"cp=4，T={T_ref}",
         "若拟合出的 alpha~0，模型会预测无 batch 效应，与文献矛盾"),
        ("E2.1", "balancer 实测收益 vs 2−1/cp 上界",
         "scheme=block|scheme=balanced", f"T={T_ref}, cp=4, B={B}",
         "收益不得超过 2−1/cp；MTGR 布局下应远小于文献的 37%"),
        ("E3.1", "跨模型不重标定（判'标定一次'）",
         "model=hstu-base|model=longer", f"T={T_ref}, cp=1 与 cp=4, B={B}",
         "误差 < 20%，否则改为按模型标定"),
        ("E4.1", "换 kernel 后 c_star 是否移动（判'标定一次'）",
         "kernel=triton|kernel=fused", f"T={T_ref}, cp=1 与 cp=4, B={B}",
         "c_star 移动 < 3×，否则撤'标定一次'"),
    ]
    for name, why, grid, cfg, gate in l2_jobs:
        a(Row("L2", name, "端到端 step 时间", "step_ms", "ms", "measured",
              why, gate, repeat=grid, config=cfg))
        a(Row("L2", name, "对应 CV", "step_cv", "%", "measured",
              "同一配置的重复测量", "< 5%", config=cfg))
        a(Row("L2", name, "模型预测值", "step_pred", "ms", "derived",
              "grcost predict，用 C0 的参数", "", requires="eff_inf,c_star,gamma",
              config=cfg))
        a(Row("L2", name, "误差", "err", "%", "derived", "(pred−meas)/meas", "",
              config=cfg))

    a(Row("L2", "E4.1", "第二实现的 c_star", "c_star_alt", "token", "fitted",
          "只重拟合 eff 三参数，其余不动", "< 3× c_star",
          requires="c_star", note="超过则撤'标定一次'"))
    a(Row("L2", "E2.1", "重平衡实测收益", "gain_rebal", "%", "derived",
          "block 与 balanced 的时间比减 1", "≤ 2−1/cp",
          requires="step_ms", note="MTGR 布局下应远小于文献的 37%"))

    # ---------------- G --------------------------------------------------
    for gid, check, action in (
        ("G0", "同点 CV < 5%", "放弃细粒度判据，不拟合 gamma"),
        ("G1", "c_star 与 c_comm 同量级（10× 内）", "曲线在吸收别的东西，停"),
        ("G2", "eff 曲线在扫描区间内有拐点", "扩区间，不是加参数"),
        ("G3", "重平衡收益 ≤ 2−1/cp", "闭式或掩码假设错"),
        ("G4", "cp=1 的 MAPE < 5%", "工作量计数器错，停止标定"),
        ("G5", "换 kernel 后 c_star 移动 < 3×", "撤'标定一次'"),
        ("G6", "跨模型误差 < 20%", "改为按模型标定，缩小 claim"),
    ):
        a(Row("G", gid, check, "", "", "measured", "由前面各阶段的数据判定",
              action, note="判定结果填 pass / fail / 不可测"))
    return rows


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------
# 15 columns, in this exact order.  `variant` is what makes each expanded line
# different; `fixed_config` is what stays the same across them.  Splitting the
# two is what stops "which line am I filling?" ambiguity in a spreadsheet.
SHEET_HEADER = [
    "stage", "section", "metric", "symbol", "unit", "provenance",
    "how_to_measure", "target_or_gate", "requires", "variant", "fixed_config",
    "value", "cv_pct", "layers", "note",
]


def expand(rows: Sequence[Row]) -> List[List[str]]:
    """One CSV line per (row × repeat value), with blanks for the fills."""
    out = []
    for r in rows:
        reps = ([x.strip() for x in r.repeat.split("|") if x.strip()]
                if "|" in r.repeat else ([""] if not r.repeat else [r.repeat]))
        for rep in reps:
            out.append([
                r.stage, r.section, r.metric, r.symbol, r.unit, r.source,
                r.how, r.target, r.requires,
                rep, r.config,
                "", "", "", r.note,
            ])
    return out


def write_csv(path: str, rows: Sequence[Row]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(SHEET_HEADER)
        w.writerows(expand(rows))
    return path


def _md(text: str) -> str:
    """Escape what would break a Markdown table cell."""
    return text.replace("|", "\\|")


def write_markdown(path: str, rows: Sequence[Row]) -> str:
    lines = [
        "# NPU 测量数据填报表",
        "",
        "> 生成自 `grcost.sheet.registry()`，与 `manifest` 的作业 ID 同步。",
        "> `依赖` 列为空表示可独立填；非空的格子**必须先有前置值**，",
        "> 否则得到的模型内部自洽但无法被证伪。",
        "",
        "## 怎么用",
        "",
        "1. 先填 P0 与 G 两段的判据（判据要在看数据前定稿）。",
        "2. 按 L0 → L1 → C0 → L2 顺序填，每段结束检查门禁。",
        "3. `provenance` 列的五个值含义：",
        "   `measured` 你测的 · `derived` 由测量算出 · `fitted` 优化器选的 ·",
        "   `pinned` 必须给定且不许拟合 · `declared` 配置事实。",
        "4. `变体` 非空的行，CSV 里已展开成多行，每行填一个配置；`固定配置` 列说明该行其余参数。",
        "",
        "## 五个绝对不要让它进拟合的量",
        "",
        "`t_fixed` · `overlap` · `bw_util` · `recompute` · `layers`",
        "",
        "前四个要么可由微基准直接测、要么在公式里只以乘积出现；`layers` 是口径。",
        "任何一个搞错，拟合都会给出内部自洽但物理错误的结果。",
        "",
    ]
    for stage, title in STAGES.items():
        sub = [r for r in rows if r.stage == stage]
        if not sub:
            continue
        lines.append(f"## {stage} · {title}")
        lines.append("")
        if stage == "G":
            lines.append("| 门 | 判据 | 不过怎么办 | 结论 |")
            lines.append("|---|---|---|---|")
            for r in sub:
                lines.append(f"| {r.section} | {_md(r.metric)} | {_md(r.target)} |  |")
            lines.append("")
            continue
        lines.append("| 作业 | 指标 | 符号 | 单位 | 来源 | 怎么得到 | 期望/判据 "
                     "| 依赖 | 变体 | 固定配置 | 值 | CV |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sub:
            rep = f"<br>*每* `{_md(r.repeat)}`" if r.repeat else ""
            rep_raw = r.repeat
            how = _md(r.how) + (f"<br>{_md(r.note)}" if r.note else "")
            lines.append(
                f"| {_md(r.section)} | {_md(r.metric)}{rep} | `{r.symbol}` | "
                f"{r.unit} | {r.source} | {how} | {_md(r.target)} | "
                f"{_md(r.requires)} | {_md(rep_raw)} | {_md(r.config)} |  |  |")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_markdown_file(path: str, rows: Sequence[Row]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(write_markdown(path, rows))
    return path


# --------------------------------------------------------------------------
# Fill mode: the compact sheet you actually type into
# --------------------------------------------------------------------------
FILL_HEADER = [
    "阶段", "作业", "指标", "变体", "固定配置", "怎么测", "值", "CV%", "备注",
]

_SHORT = {
    "P0": "P0 预检",
    "L0": "L0 常数",
    "L1": "L1 曲线",
    "C0": "C0 组合",
    "L2": "L2 验证",
    "G": "G  门禁",
}


def fill_rows(rows: Sequence[Row]) -> List[List[str]]:
    """One line per thing to fill, with only the columns you need to type in.

    The 15-column machine sheet stays the authority (it carries provenance and
    prerequisites, which the analysis reads).  This is the human-facing view:
    stage, what, which variant, and a blank for the number.
    """
    out = []
    for r in rows:
        reps = ([x.strip() for x in r.repeat.split("|") if x.strip()]
                if "|" in r.repeat else ([""] if not r.repeat else [r.repeat]))
        for rep in reps:
            hint = r.how
            if r.stage == "G":
                hint = r.target
            out.append([
                _SHORT.get(r.stage, r.stage), r.section, r.metric,
                rep, r.config, hint, "", "", r.note,
            ])
    return out


def write_fill_csv(path: str, rows: Sequence[Row]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(FILL_HEADER)
        w.writerows(fill_rows(rows))
    return path


#: A one-line purpose for each block, so every table says why it exists.
SECTION_TITLES = {
    ("P0", "env"): "环境与配置",
    ("P0", "topo"): "链路拓扑（决定 c* 的上界）",
    ("P0", "spec"): "标称规格（只用于给上界，不用于拟合）",
    ("L0", "M1.1"): "同点噪声 · 短序列",
    ("L0", "M1.2"): "同点噪声 · 长序列",
    ("L0", "M2.1"): "下发成本（空 kernel 斜率）",
    ("L0", "M2.2"): "单次启动成本（独立口径）",
    ("L0", "M2.3"): "不可切部分 t_fixed（独立口径）",
    ("L0", "M3.1"): "HBM 带宽与效率",
    ("L0", "M3.2"): "可达算力（eff_inf 的物理上限）",
    ("L0", "M4.1"): "同步开销",
    ("L0", "M5"): "集合通信 时间–字节 曲线",
    ("L0", "M6.1"): "显存墙与激活字节",
    ("L1", "decl"): "口径声明（层数）",
    ("L1", "S1.1"): "效率曲线 eff(c) —— 唯一要拟合的东西",
    ("L1", "S1.2"): "整栈交叉验算",
    ("L1", "S2.1"): "负载不均衡（只跑一个 rank 的 query）",
    ("L1", "S3.1"): "掩码拓扑：矩形 vs 三角",
    ("L1", "S4.1"): "padding 浪费",
    ("C0", "C1"): "效率曲线三参数",
    ("C0", "C2"): "两个拐点",
    ("L2", "E1.1"): "标定表：长度 × CP",
    ("L2", "E1.2"): "batch 扫描（判 alpha 是否真实）",
    ("L2", "E2.1"): "balancer 实测（判闭式上界）",
    ("L2", "E3.1"): "跨模型迁移（判“标定一次”）",
    ("L2", "E4.1"): "换 kernel（判 eff 是 kernel 属性）",
    ("G", "G0"): "噪声底",
    ("G", "G1"): "c_star 与 F/BW 是否同量级",
    ("G", "G2"): "曲线是否有拐点",
    ("G", "G3"): "重平衡收益是否超过 2−1/cp",
    ("G", "G4"): "cp=1 是否准",
    ("G", "G5"): "换 kernel 后 c_star 是否移动",
    ("G", "G6"): "跨模型误差",
}


def blocks(rows: Sequence[Row]) -> List[Tuple[str, str]]:
    """(stage, section) in registry order, so the form reads top to bottom."""
    seen: List[Tuple[str, str]] = []
    for r in rows:
        key = (r.stage, r.section)
        if key not in seen:
            seen.append(key)
    return seen


def section_title(stage: str, section: str) -> str:
    return SECTION_TITLES.get((stage, section), section)


FILL_INTRO = """# NPU 测量填报单

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
"""


def write_fill_markdown(path: str, rows: Sequence[Row]) -> str:
    """One table per job block, grouped under its stage.

    A single 90-row table is impossible to navigate while filling.  Splitting by
    block means each table is short enough to hold in view, and the heading says
    what the block is for -- so "where do I write this number" has an answer that
    does not require reading the whole form.
    """
    lines = [FILL_INTRO]

    # --- contents ------------------------------------------------------
    lines.append("## 目录")
    lines.append("")
    cur = None
    for stage, section in blocks(rows):
        if stage != cur:
            lines.append("")
            lines.append(f"**{stage} · {STAGES[stage]}**")
            cur = stage
        lines.append(f"- `{section}` {section_title(stage, section)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- one table per block -------------------------------------------
    for stage, title in STAGES.items():
        stage_blocks = [b for b in blocks(rows) if b[0] == stage]
        if not stage_blocks:
            continue
        lines.append(f"# {stage} · {title}")
        lines.append("")
        for _, section in stage_blocks:
            sub = [r for r in rows if r.stage == stage and r.section == section]
            lines.append(f"## {section} · {section_title(stage, section)}")
            lines.append("")

            if stage == "G":
                lines.append("| 检查什么 | 不通过怎么办 | 结论 |")
                lines.append("|:--|:--|:--|")
                for r in sub:
                    lines.append(f"| {_md(r.metric)} | {_md(r.target)} |  |")
                lines.append("")
                continue

            lines.append("| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |")
            lines.append("|:--|:--|:--|:--|:--|:--|")
            for r in sub:
                reps = ([x.strip() for x in r.repeat.split("|") if x.strip()]
                        if "|" in r.repeat else ([""] if not r.repeat else [r.repeat]))
                for rep in reps:
                    hint = _md(r.how)
                    if r.config:
                        hint += f"<br>_{_md(r.config)}_"
                    name = _md(r.metric)
                    if r.symbol:
                        name += f" *(`{r.symbol}`)*"
                    if r.unit:
                        name += f" _{_md(r.unit)}_"
                    lines.append(
                        f"| {name} | {_md(rep)} | {hint} | {_md(r.requires)} "
                        f"|  |  |")
            lines.append("")
    return "\n".join(lines) + "\n"


def write_fill_markdown_file(path: str, rows: Sequence[Row]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(write_fill_markdown(path, rows))
    return path


def counts(rows: Sequence[Row]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[r.source] = out.get(r.source, 0) + 1
    return out
