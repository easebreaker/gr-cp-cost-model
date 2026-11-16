"""Generate the blank measurement sheet (CSV + printable Markdown).

Usage:
    PYTHONPATH=src python scripts/make_sheet.py
    PYTHONPATH=src python scripts/make_sheet.py --model-layers 15 --T 4096
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grcost.sheet import (  # noqa: E402
    STAGES, counts, expand, registry, write_csv, write_fill_csv,
    write_fill_markdown_file, write_markdown_file,
)


def existing_answers(path: str) -> int:
    """How many rows already carry an answer, for any supported layout.

    Deliberately format-agnostic: it looks for a non-empty cell in the last two
    columns of any 5..8 column table, which covers both the old 7-column form and
    the current 6-column one.  A strict parser here would fail open, and fail
    open means losing someone's data.
    """
    if not os.path.exists(path):
        return 0
    n = 0
    try:
        if path.endswith(".csv"):
            with open(path, newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    if (row.get("value") or "").strip():
                        n += 1
            return n
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                st = line.strip()
                if not st.startswith("|") or st.startswith("|:--"):
                    continue
                cells = [c.strip() for c in st.strip("|").split("|")]
                if "CV%" in cells or "结论" in cells:
                    continue          # header row, not an answer
                if 5 <= len(cells) <= 8 and any(cells[-2:]):
                    n += 1
    except OSError:
        return 0
    return n


def _answers(path: str, csv_mode: bool):
    """(key -> (value, cv)) from an existing form, whatever its layout."""
    out = {}
    if csv_mode:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                key = (r.get("section", ""), r.get("metric", ""), r.get("variant", ""))
                v = (r.get("value") or "").strip()
                if v:
                    out[key] = (v, (r.get("cv_pct") or "").strip())
        return out
    sec = ""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            st = line.strip()
            m = re.match(r"^##\s+`?([A-Za-z0-9._]+)`?\s*·", st)
            if m:
                sec = m.group(1)
                continue
            if not st.startswith("|") or st.startswith("|:--") or "CV%" in st:
                continue
            cells = [c.strip() for c in st.strip("|").split("|")]
            if len(cells) != 6:
                continue
            name, variant, val, cv = cells[0], cells[1], cells[4], cells[5]
            sym = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", name)
            metric = re.sub(r"\s*\*\(`[^`]+`\)\*", "", name)
            metric = re.sub(r"_`?[^`_]*?`?_\s*$", "", metric).strip()
            if val:
                out[(sec, metric, variant)] = (val, cv)
                if sym:
                    out[(sec, sym, variant)] = (val, cv)
    return out


def _merge_answers(path: str, rows, csv_mode: bool) -> None:
    """Rewrite the form so it contains the new rows *and* the old answers."""
    import tempfile
    old = _answers(path, csv_mode)
    tmp = path + ".new"
    if csv_mode:
        write_fill_csv(tmp, rows)
        with open(tmp, newline="", encoding="utf-8-sig") as fh:
            mine = list(csv.DictReader(fh))
            cols = list(mine[0].keys())
        for r in mine:
            hit = old.get((r["section"], r["metric"], r["variant"]))
            if hit:
                r["value"], r["cv_pct"] = hit
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(mine)
        os.remove(tmp)
        return

    write_fill_markdown_file(tmp, rows)
    with open(tmp, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    os.remove(tmp)
    sec, out = "", []
    for line in lines:
        st = line.strip()
        m = re.match(r"^##\s+`?([A-Za-z0-9._]+)`?\s*·", st)
        if m:
            sec = m.group(1)
        if st.startswith("|") and not st.startswith("|:--") and "CV%" not in st:
            cells = [c for c in st.strip("|").split("|")]
            if len(cells) == 6 and cells[4].strip() == "" and cells[5].strip() == "":
                name, variant = cells[0].strip(), cells[1].strip()
                hit = old.get((sec, variant and name, variant))
                if not hit:
                    sym = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", name)
                    metric = re.sub(r"\s*\*\(`[^`]+`\)\*", "", name)
                    metric = re.sub(r"_`?[^`_]*?`?_\s*$", "", metric).strip()
                    for key in ((sec, metric, variant), (sec, sym.group(1), variant)
                                if sym else (sec, metric, variant)):
                        hit = old.get(key)
                        if hit:
                            break
                if hit:
                    cells[4], cells[5] = f" {hit[0]} ", f" {hit[1]} "
                    line = "|" + "|".join(cells) + "|"
        out.append(line)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-csv", default="data/measurement_sheet.csv")
    ap.add_argument("--out-md", default="docs/12-measurement-sheet.md")
    ap.add_argument("--out-fill", default="data/measurement_fill.csv",
                    help="compact spreadsheet to type into")
    ap.add_argument("--out-fill-md", default="data/measurement_fill.md",
                    help="compact markdown form to type into in the editor")
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--model-layers", dest="model_layers", type=int, default=15)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--force", action="store_true",
                    help="overwrite even if the form already has answers")
    ap.add_argument("--merge", action="store_true",
                    help="keep existing answers, add the new rows around them")
    args = ap.parse_args(argv)

    rows = registry(T_ref=args.T, B=args.B, model_layers=args.model_layers,
                    d=args.d, heads=args.heads)

    # Never clobber someone's measurements.  Regenerating is cheap; re-deriving
    # a half-filled sheet on a cluster you no longer have access to is not.
    targets = [args.out_fill, args.out_fill_md]
    filled = {t: existing_answers(t) for t in targets}
    clobbered = {t: n for t, n in filled.items() if n}
    if clobbered and args.merge:
        for t in clobbered:
            _merge_answers(t, rows, csv_mode=t.endswith(".csv"))
        write_csv(args.out_csv, rows)
        write_markdown_file(args.out_md, rows)
        print(f"  merged into existing forms: "
              + ", ".join(f"{t} ({n})" for t, n in clobbered.items()))
        print(f"  new metric count: {len(rows)}")
        return 0
    if clobbered and not args.force:
        print("REFUSING to overwrite files that already contain answers:")
        for t, n in clobbered.items():
            print(f"  {t}: {n} answered row(s)")
        print("\n  move them aside, or pass --force to overwrite anyway.")
        return 3

    print(f"  {len(rows)} distinct metrics across {len(STAGES)} stages")
    for stage, title in STAGES.items():
        n = sum(1 for r in rows if r.stage == stage)
        print(f"    {stage}  {title:<34} {n:>2} 行")
    print()
    print("  provenance breakdown:")
    for k, v in sorted(counts(rows).items()):
        print(f"    {k:<10} {v:>2}")
    print()
    print("  the five things that must never enter a fit:")
    print("    t_fixed · overlap · bw_util · recompute · layers")

    write_csv(args.out_csv, rows)
    write_markdown_file(args.out_md, rows)
    write_fill_csv(args.out_fill, rows)
    write_fill_markdown_file(args.out_fill_md, rows)
    n_lines = len(expand(rows))
    print()
    print(f"  CSV  : {args.out_csv}  ({n_lines} 填数行, 机器读)")
    print(f"  MD   : {args.out_md}  (可打印)")
    print(f"  FILL : {args.out_fill}  (表格软件)")
    print(f"  FORM : {args.out_fill_md}  <- 在编辑器里直接填这一份")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
