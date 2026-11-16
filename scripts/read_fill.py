"""Read a filled-in form back and do something with it.

Parses either the Markdown form (``data/measurement_fill.md``) or the CSV
(``data/measurement_fill.csv``), then:

* reports which rows are still blank, in dependency order;
* refuses to compute anything whose ``requires`` list is incomplete;
* assembles a ``Params`` from whatever *is* available;
* runs the gates that can be decided;
* writes ``out/params.json``.

Why the dependency check is not optional
----------------------------------------
``eff`` cannot be solved without ``t_fixed``, ``bw_util`` and ``layers``: the
same measurement yields a different efficiency for every choice of those three.
Producing a number anyway is how a model becomes unfalsifiable. So a missing
prerequisite is reported as *not computable*, never as a default.

Usage
-----
    PYTHONPATH=src python scripts/read_fill.py data/measurement_fill.md
    PYTHONPATH=src python scripts/read_fill.py data/measurement_fill.csv --out out/params.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grcost.sheet import Row, registry, to_model_units  # noqa: E402

NUM = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _num(text: str) -> Optional[float]:
    t = text.strip().strip("`*_ ").replace(",", "")
    if not t or t in ("-", "—", "?"):
        return None
    if not NUM.match(t):
        return None
    return float(t)


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------
@dataclass
class Filled:
    """One answered (or unanswered) line."""

    symbol: str
    section: str
    metric: str
    variant: str
    value: Optional[float]
    cv: Optional[float]
    raw: str
    text: str = ""       # the cell as typed; non-empty means "answered"

    @property
    def key(self) -> str:
        return self.symbol or f"{self.section}/{self.metric}"

    @property
    def answered(self) -> bool:
        """Answered is not the same as numeric.

        ``机型``, ``CANN 版本`` and ``链路类型`` are legitimately text. Treating
        them as blank would make the reader nag about rows the user has in fact
        filled in, which is the fastest way to teach someone to ignore it.
        """
        return bool(self.text.strip())


#: The fill form carries one table per block, so the parser has to know which
#: table it is inside: the block heading gives the job id, and only 6-column
#: tables are data (the gate tables have 3).
_HEADING = re.compile(r"^##\s+`?([A-Za-z0-9._]+)`?\s*·")
_SYMBOL = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
_UNIT = re.compile(r"_`?([^`_]*?)`?_\s*$")


def parse_markdown(path: str) -> List[Filled]:
    out: List[Filled] = []
    section = ""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            m = _HEADING.match(stripped)
            if m:
                section = m.group(1)
                continue
            if not stripped.startswith("|") or stripped.startswith("|:--"):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) != 6:          # header, or a 3-column gate table
                continue
            name, variant = cells[0], cells[1]
            if not name or "指标" in name:
                continue
            value, cv = _num(cells[4]), _num(cells[5])
            sym = _SYMBOL.search(name)
            symbol = sym.group(1) if sym else ""
            metric = re.sub(r"\s*\*\(`[^`]+`\)\*", "", name)
            metric = _UNIT.sub("", metric).strip()
            out.append(Filled(symbol, section, metric, variant, value, cv,
                              stripped, text=cells[4]))
    return out


def parse_csv(path: str) -> List[Filled]:
    out: List[Filled] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            sym = (row.get("symbol") or "").strip()
            raw_val = (row.get("value") or "").strip()
            out.append(Filled(
                symbol=sym,
                section=(row.get("section") or "").strip(),
                metric=(row.get("metric") or "").strip(),
                variant=(row.get("variant") or "").strip(),
                value=_num(raw_val),
                cv=_num(row.get("cv_pct") or ""),
                raw="",
                text=raw_val,
            ))
    return out


def load(path: str) -> List[Filled]:
    return parse_csv(path) if path.lower().endswith(".csv") else parse_markdown(path)


# --------------------------------------------------------------------------
# Completeness, in dependency order
# --------------------------------------------------------------------------
def completeness(filled: Sequence[Filled]) -> Tuple[Dict[str, int], List[str]]:
    # Convert on the way in, once.  The form is in human units (us, GB/s);
    # Params is in SI (s, bytes/s).  Doing it anywhere else invites the same
    # number being divided by 1e6 in one place and not another.
    have = {f.symbol: to_model_units(f.symbol, f.value)
            for f in filled if f.symbol and f.value is not None}
    answered = {f.symbol for f in filled if f.symbol and f.answered}
    missing = []
    for r in registry():
        if not r.symbol:
            continue
        if r.symbol in answered:
            continue
        unmet = [d for d in (x.strip() for x in r.requires.split(",")) if d
                 and d not in have]
        why = f" (needs {', '.join(unmet)})" if unmet else ""
        missing.append(f"{r.stage} {r.section}: {r.metric} `{r.symbol}`{why}")
    return have, missing


def build_params(have: Dict[str, float]):
    from grcost.cost import Params

    p = Params()
    kw = {}
    for name in ("eff_inf", "c_star", "gamma", "alpha", "bw_comm", "overlap",
                 "bw_util", "recompute"):
        if name in have:
            kw[name] = have[name]
    return p.copy(**kw) if kw else p


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
@dataclass
class GateResult:
    gate: str
    verdict: str          # pass | fail | undecidable
    detail: str


def run_gates(have: Dict[str, float]) -> List[GateResult]:
    res: List[GateResult] = []

    def need(*keys):
        return all(k in have for k in keys)

    # G0: noise floor
    cvs = [v for k, v in have.items() if k.startswith("CV")]
    if cvs:
        worst = max(cvs)
        res.append(GateResult("G0", "pass" if worst < 5 else "fail",
                              f"同点 CV 最大 {worst:.2f}% (阈值 5%)"))
    else:
        res.append(GateResult("G0", "undecidable", "未填 CV_short / CV_long"))

    # G1: c_star vs measured break-even.  ``have`` is already in model units
    # (FLOPS, bytes/s, tokens), so no further scaling belongs here.
    if need("c_star", "F_real", "bw_comm"):
        c_comm = have["F_real"] / have["bw_comm"]
        ratio = have["c_star"] / c_comm
        ok = 0.1 <= ratio <= 10
        res.append(GateResult("G1", "pass" if ok else "fail",
                              f"c_star/bw_comm = {ratio:.2f} "
                              f"(c_star={have['c_star']:.0f}, c_comm={c_comm:.0f})"))
    else:
        res.append(GateResult("G1", "undecidable", "需要 c_star, F_real, bw_comm"))

    # G4: cp=1 accuracy
    if need("step_ms", "step_pred"):
        err = abs(have["step_pred"] / have["step_ms"] - 1) * 100
        res.append(GateResult("G4", "pass" if err < 5 else "fail",
                              f"cp=1 误差 {err:.1f}% (阈值 5%)"))
    else:
        res.append(GateResult("G4", "undecidable", "需要 step_ms 与 step_pred"))

    # G5: kernel transfer
    if need("c_star", "c_star_alt"):
        r = max(have["c_star"], have["c_star_alt"]) / max(
            min(have["c_star"], have["c_star_alt"]), 1e-9)
        res.append(GateResult("G5", "pass" if r < 3 else "fail",
                              f"换 kernel 后 c_star 移动 {r:.2f}x (阈值 3x)"))
    else:
        res.append(GateResult("G5", "undecidable", "需要 c_star 与 c_star_alt"))

    # G6: cross-model
    if "err" in have:
        res.append(GateResult("G6", "pass" if abs(have["err"]) < 20 else "fail",
                              f"跨模型误差 {have['err']:+.1f}% (阈值 20%)"))
    else:
        res.append(GateResult("G6", "undecidable", "需要 err"))

    # G3: rebalancing bound
    if need("gain_rebal", "step_cv"):
        res.append(GateResult("G3", "undecidable",
                              "需要 cp 度才能算上界 2−1/cp；请手填结论"))
    else:
        res.append(GateResult("G3", "undecidable", "需要 gain_rebal"))

    for g in ("G2",):
        res.append(GateResult(g, "undecidable", "看图判定：曲线在区间内是否单调且有拐点"))
    return res


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("form", help="data/measurement_fill.md or .csv")
    ap.add_argument("--out", default="out/params.json")
    ap.add_argument("--all", action="store_true", help="list every blank row")
    args = ap.parse_args(argv)

    if not os.path.exists(args.form):
        print(f"no such file: {args.form}")
        print("generate it first: PYTHONPATH=src python scripts/make_sheet.py")
        return 2

    filled = load(args.form)
    n_ans = sum(1 for f in filled if f.answered)
    n_num = sum(1 for f in filled if f.value is not None)
    print(f"parsed {len(filled)} rows from {args.form}: "
          f"{n_ans} answered ({n_num} numeric), {len(filled) - n_ans} blank")

    have, missing = completeness(filled)
    if missing:
        show = missing if args.all else missing[:12]
        print(f"\nstill blank ({len(missing)}):" if args.all
              else f"\nstill blank: {len(missing)}  (--all to list)")
        for m in show:
            print(f"  {m}")
        if not args.all and len(missing) > len(show):
            print(f"  ... and {len(missing) - len(show)} more")

    params = build_params(have)
    print("\nparameters assembled from what is present "
          "(converted to model units):")
    for name in ("eff_inf", "c_star", "gamma", "alpha", "bw_comm",
                 "overlap", "bw_util", "recompute"):
        src = "filled" if name in have else "default"
        print(f"  {name:<10} {getattr(params, name):>12.4g}   [{src}]")

    print("\ngates:")
    for g in run_gates(have):
        mark = {"pass": "PASS", "fail": "FAIL", "undecidable": " ?  "}[g.verdict]
        print(f"  {g.gate}  {mark}  {g.detail}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(params.to_json())
    print(f"\nwritten to {args.out}")
    if missing:
        print("NOTE: some parameters are still at their defaults. Predictions "
              "made now are\n      provisional; the gates above say which "
              "conclusions they cannot support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
