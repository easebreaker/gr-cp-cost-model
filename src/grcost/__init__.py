"""grcost -- analytic step-time and memory prediction for CP-sharded GR models.

Quick start
-----------
>>> from grcost import Params, H100_SXM, resolve_model, resolve_layout, predict
>>> m = resolve_model("mtgr-large")
>>> lay = resolve_layout("mtgr", 4096)
>>> predict(m, lay, T=4096, B=32, cp=2, hw=H100_SXM, p=Params(), t_fixed_ms=3.0)
"""

from .bridge import CPDecomposition, decompose, kappa_drift
from .compose import EffPoint, extract_eff, fit_curve
from .cost import Ablation, Params, amdahl_ceiling, efficiency, predict, scaling_factor, speedup
from .fit import FitResult, Record, fit, load_csv, predict_record
from .hardware import A100_80, ASCEND_910B, H100_SXM, PRESETS, Hardware, get
from .frontier import (Point, best, breakeven_alpha, gpu_hours_for_tokens,
                       longer_only, longer_with_cp, pareto, prior_memory_fit,
                       throughput_tokens_per_hour, tokens_for_quality)
from .manifest import Job, build_plan, cost_report, gates
from .quality import QualityLaw, QualityPoint, fit_quality
from .memory import MemoryFit, check_table, estimate_act_bytes_per_token, fit_memory
from .metrics import Report, by_group, kendall_tau, mape, report
from .presets import MODELS, list_models, mtgr_layout, resolve_layout, resolve_model
from .spec import HstuModel, Layout, LongerModel, Phase, assign, crossover_report, work_per_rank
from .work import RankWork, flops_per_example, imbalance, rank_work, total_flops

__version__ = "0.1.0"

__all__ = [
    "Params", "predict", "speedup", "efficiency",
    "roofline_flops_per_sec", "Hardware", "H100_SXM", "A100_80", "ASCEND_910B",
    "get", "PRESETS", "Layout", "HstuModel", "LongerModel", "Phase", "assign",
    "work_per_rank", "crossover_report", "rank_work", "RankWork", "imbalance",
    "total_flops", "flops_per_example", "MemoryFit", "fit_memory",
    "check_table", "estimate_act_bytes_per_token", "Record", "fit", "load_csv",
    "predict_record", "FitResult", "report", "by_group", "kendall_tau", "mape",
    "Report", "resolve_model", "resolve_layout", "mtgr_layout", "MODELS",
    "list_models", "Ablation", "amdahl_ceiling", "scaling_factor",
    "CPDecomposition", "decompose", "kappa_drift",
    "EffPoint", "extract_eff", "fit_curve",
    "Job", "build_plan", "cost_report", "gates",
    "QualityLaw", "QualityPoint", "fit_quality",
    "Point", "best", "pareto", "breakeven_alpha", "throughput_tokens_per_hour",
    "gpu_hours_for_tokens", "tokens_for_quality", "longer_only",
    "longer_with_cp", "prior_memory_fit",
]
