"""The fill form must round-trip: blank -> filled -> Params, with units converted.

The interesting failure mode is not a crash.  It is a form that parses, produces
a full ``Params``, and is wrong by a factor of 1e6 because microseconds were
written into a field that expects seconds.  These tests pin the units and the
dependency gate, because those are what make the form safe to hand to someone.
"""

import json
import math
import os
import tempfile
import unittest

from grcost.sheet import CANONICAL, registry, to_model_units

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import make_sheet  # noqa: E402
import read_fill  # noqa: E402


class TestUnits(unittest.TestCase):
    def test_microseconds_become_seconds(self):
        self.assertAlmostEqual(to_model_units("alpha", 26.0), 26e-6)

    def test_gigabytes_per_second_become_bytes_per_second(self):
        self.assertAlmostEqual(to_model_units("bw_comm", 148.0), 148e9)

    def test_dimensionless_symbols_pass_through(self):
        for sym in ("eff_inf", "c_star", "gamma", "overlap", "bw_util",
                    "recompute", "frac"):
            self.assertAlmostEqual(to_model_units(sym, 0.42), 0.42, msg=sym)

    def test_every_scaled_symbol_appears_in_the_registry(self):
        """A conversion for a symbol nobody fills is dead code."""
        syms = {r.symbol for r in registry() if r.symbol}
        for sym in CANONICAL:
            self.assertIn(sym, syms, f"{sym} has a conversion but no row")

    def test_no_accidental_identity_in_canonical(self):
        """A factor that is not a round scaling value is a typo.

        ``alpha`` is 1e-6 (us -> s, shrinking), ``bw_comm`` is 1e9
        (GB/s -> bytes/s, growing), ``a_act`` is 1024 (KiB -> bytes).  Those are
        the three shapes that occur; anything else is probably a slip like 1e8
        for GB/s or 60 for minutes.
        """
        allowed = {1e-6, 1e-3, 1024.0, 1e9, 1e12}
        for sym, f in CANONICAL.items():
            if f == 1.0:
                continue
            self.assertTrue(
                any(abs(f / a - 1.0) < 1e-9 for a in allowed),
                f"{sym}: factor {f} is not one of the expected scalings")


class TestParsing(unittest.TestCase):
    def _form(self, fills):
        """Build a minimal form in the real shape: block heading + 6 columns."""
        lines = ["## L0 · 测试块", "",
                 "| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |",
                 "|:--|:--|:--|:--|:--|:--|"]
        for sym, val in fills.items():
            lines.append(f"| thing *(`{sym}`)* _us_ |  | how |  | {val} |  |")
        return "\n".join(lines) + "\n"

    def test_parses_markdown_values(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.md")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(self._form({"alpha": 26, "bw_comm": 148}))
            filled = read_fill.load(p)
        have = {f.symbol: to_model_units(f.symbol, f.value)
                for f in filled if f.value is not None}
        self.assertAlmostEqual(have["alpha"], 26e-6)
        self.assertAlmostEqual(have["bw_comm"], 148e9)

    def test_blank_cells_are_none_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.md")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(self._form({"alpha": "  "}))
            filled = read_fill.load(p)
        self.assertIsNone(filled[0].value)

    def test_scientific_notation_and_thousands_separators(self):
        self.assertEqual(read_fill._num("1.5e-5"), 1.5e-5)
        self.assertEqual(read_fill._num("1,900"), 1900.0)
        self.assertIsNone(read_fill._num("n/a"))
        self.assertIsNone(read_fill._num("—"))


class TestDependencyGate(unittest.TestCase):
    def test_missing_prerequisite_is_reported_not_defaulted(self):
        """``eff`` must be undecidable without t_fixed / bw_util / layers."""
        filled = [read_fill.Filled("t_med", "S1.1", "time", "", 12.0, None, "")]
        have, missing = read_fill.completeness(filled)
        eff_line = [m for m in missing if "`eff`" in m]
        self.assertTrue(eff_line)
        self.assertIn("needs", eff_line[0])

    def test_satisfied_prerequisite_drops_the_note(self):
        filled = [
            read_fill.Filled("t_med", "S1.1", "time", "", 12.0, None, ""),
            read_fill.Filled("t_fixed", "M2.3", "fixed", "", 6.0, None, ""),
            read_fill.Filled("bw_util", "M3.1", "util", "", 0.55, None, ""),
            read_fill.Filled("layers", "decl", "layers", "", 15.0, None, ""),
        ]
        _, missing = read_fill.completeness(filled)
        eff_line = [m for m in missing if "`eff`" in m]
        self.assertTrue(eff_line)
        self.assertNotIn("needs", eff_line[0])

    def test_all_blank_is_all_missing(self):
        _, missing = read_fill.completeness([])
        self.assertGreater(len(missing), 40)


class TestGates(unittest.TestCase):
    def _have(self, **kw):
        base = {
            "eff_inf": 0.42, "c_star": 1900.0, "gamma": 1.05,
            "alpha": 26e-6, "bw_comm": 148e9, "overlap": 0.45,
            "bw_util": 0.57, "recompute": 1.333,
            "CV_short": 2.1, "CV_long": 2.4,
            "F_real": 298e12, "BW_hbm": 910e9,
            "step_ms": 412.0, "step_pred": 428.0, "err": 3.9,
        }
        base.update(kw)
        return base

    def test_all_pass_on_a_consistent_set(self):
        res = {g.gate: g for g in read_fill.run_gates(self._have())}
        self.assertEqual(res["G0"].verdict, "pass")
        self.assertEqual(res["G1"].verdict, "pass")
        self.assertEqual(res["G4"].verdict, "pass")

    def test_g0_fails_on_a_noisy_machine(self):
        res = {g.gate: g for g in read_fill.run_gates(self._have(CV_short=7.5))}
        self.assertEqual(res["G0"].verdict, "fail")

    def test_g1_fails_when_the_curve_disagrees_with_f_over_bw(self):
        """c_star two orders of magnitude off means the curve ate something else."""
        res = {g.gate: g for g in read_fill.run_gates(self._have(c_star=190000.0))}
        self.assertEqual(res["G1"].verdict, "fail")

    def test_g4_fails_when_cp1_is_wrong(self):
        res = {g.gate: g for g in read_fill.run_gates(self._have(step_pred=500.0))}
        self.assertEqual(res["G4"].verdict, "fail")

    def test_undecidable_is_not_fail(self):
        """A gate with no data must say so; it must not pass silently."""
        res = {g.gate: g for g in read_fill.run_gates({})}
        for g in res.values():
            self.assertIn(g.verdict, ("undecidable", "fail"))
        self.assertEqual(res["G0"].verdict, "undecidable")

    def test_g1_uses_model_units_without_double_conversion(self):
        """F already in FLOPS and bw in bytes/s must not be scaled again."""
        have = self._have()
        g = {x.gate: x for x in read_fill.run_gates(have)}["G1"]
        self.assertEqual(g.verdict, "pass", g.detail)


class TestFormStructure(unittest.TestCase):
    """One table per job block, so a row is never more than a glance away."""

    def _doc(self, rows=None):
        from grcost.sheet import write_fill_markdown
        return write_fill_markdown("x", rows or registry())

    def test_one_heading_per_block(self):
        from grcost.sheet import blocks
        text = self._doc()
        for stage, section in blocks(registry()):
            self.assertIn(f"## {section} ·", text, f"no table for {stage}/{section}")

    def test_every_table_has_a_header_and_separator(self):
        text = self._doc()
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("## ") and "·" in line:
                # the next non-blank lines must be header + separator
                nxt = [x for x in lines[i + 1:i + 4] if x.strip()]
                self.assertTrue(nxt and nxt[0].startswith("|"), line)
                self.assertTrue(nxt[1].startswith("|:--"), line)

    def test_tables_are_short(self):
        """A block table must fit on screen; the whole point of splitting."""
        text = self._doc()
        block, n = None, 0
        for line in text.splitlines() + ["## end · x"]:
            if line.startswith("## ") and "·" in line:
                if block:
                    self.assertLessEqual(n, 12, f"{block} has {n} rows")
                block, n = line, 0
            elif line.startswith("|") and not line.startswith("|:--"):
                n += 1
            elif not line.strip() and block and not line.startswith("|"):
                pass

    def test_contents_lists_every_block(self):
        from grcost.sheet import blocks
        text = self._doc()
        toc = text.split("## 目录")[1].split("# P0")[0]
        for stage, section in blocks(registry()):
            self.assertIn(f"`{section}`", toc, f"{stage}/{section} missing from ToC")

    def test_unit_is_visible_in_the_row(self):
        text = self._doc()
        self.assertIn("_GB/s_", text)
        self.assertIn("_us_", text)


class TestTextAnswers(unittest.TestCase):
    """Environment rows are text; treating them as blank trains the user to ignore the reader."""

    def test_text_answer_counts_as_answered(self):
        f = read_fill.Filled("", "env", "CANN 版本", "", None, None, "", text="8.1.T18")
        self.assertTrue(f.answered)
        self.assertIsNone(f.value)

    def test_text_answer_is_not_computable(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.md")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("## L0 · x\n\n| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |\n"
                         "|:--|:--|:--|:--|:--|:--|\n"
                         "| thing *(`alpha`)* _us_ |  | how |  | 26 |  |\n"
                         "| other *(`c_star`)* |  | how |  | text |  |\n")
            filled = read_fill.load(p)
        have, _ = read_fill.completeness(filled)
        self.assertIn("alpha", have)
        self.assertNotIn("c_star", have)

    def test_answered_suppresses_the_blank_warning(self):
        filled = [read_fill.Filled("c_star", "C1", "x", "", None, None, "",
                                   text="见附录")]
        _, missing = read_fill.completeness(filled)
        self.assertFalse([m for m in missing if "`c_star`" in m])


class TestClobberGuard(unittest.TestCase):
    """Regenerating must never silently destroy measurements."""

    def _answers(self, path, text):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_counts_answers_in_the_6_column_form(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._answers(os.path.join(d, "f.md"),
                              "| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |\n"
                              "|:--|:--|:--|:--|:--|:--|\n"
                              "| a |  | h |  | 12 |  |\n"
                              "| b |  | h |  |  |  |\n")
            self.assertEqual(make_sheet.existing_answers(p), 1)

    def test_counts_answers_in_the_old_7_column_form(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._answers(os.path.join(d, "f.md"),
                              "| 作业 | 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |\n"
                              "|:--|:--|:--|:--|:--|:--|:--|\n"
                              "| env | a |  | h |  | 12 |  |\n")
            self.assertEqual(make_sheet.existing_answers(p), 1)

    def test_headers_are_not_counted(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._answers(os.path.join(d, "f.md"),
                              "| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |\n"
                              "|:--|:--|:--|:--|:--|:--|\n")
            self.assertEqual(make_sheet.existing_answers(p), 0)

    def test_blank_form_is_safe_to_regenerate(self):
        from grcost.sheet import write_fill_markdown_file
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.md")
            write_fill_markdown_file(p, registry())
            self.assertEqual(make_sheet.existing_answers(p), 0)


class TestRoundTrip(unittest.TestCase):
    """Fill the generated form and read it back."""

    def _fill(self, path, vals):
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        out = []
        for line in lines:
            st = line.strip()
            if st.startswith("|") and not st.startswith("|:--"):
                cells = st.strip("|").split("|")
                if len(cells) == 6 and cells[4].strip() == "" and cells[5].strip() == "":
                    for sym, v in vals.items():
                        if f"`{sym}`" in cells[0]:
                            cells[4], cells[5] = f" {v} ", " 2.0 "
                            line = "|" + "|".join(cells) + "|"
                            break
            out.append(line)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")

    def test_generated_form_round_trips(self):
        from grcost.sheet import write_fill_markdown_file
        from grcost.sheet import to_model_units
        vals = {"CV_short": "2.1", "CV_long": "2.4", "t_fixed": "5.8",
                "BW_hbm": "910", "bw_util": "0.569", "F_real": "298",
                "alpha": "26", "bw_comm": "148", "overlap": "0.45",
                "eff_inf": "0.42", "c_star": "1900", "gamma": "1.05"}
        with tempfile.TemporaryDirectory() as d:
            form = os.path.join(d, "f.md")
            write_fill_markdown_file(form, registry())
            self._fill(form, vals)
            filled = read_fill.load(form)
        got = {f.symbol for f in filled if f.value is not None}
        self.assertEqual(got, set(vals))
        by = {f.symbol: f.value for f in filled}
        self.assertAlmostEqual(to_model_units("alpha", by["alpha"]), 26e-6)
        self.assertAlmostEqual(to_model_units("bw_comm", by["bw_comm"]), 148e9)

    def test_block_heading_is_picked_up_as_section(self):
        from grcost.sheet import write_fill_markdown_file
        with tempfile.TemporaryDirectory() as d:
            form = os.path.join(d, "f.md")
            write_fill_markdown_file(form, registry())
            self._fill(form, {"c_star": "1900"})
            filled = read_fill.load(form)
        row = [f for f in filled if f.symbol == "c_star"][0]
        self.assertEqual(row.section, "C1")


class TestEndToEnd(unittest.TestCase):
    def test_blank_form_produces_defaults_and_warns(self):
        with tempfile.TemporaryDirectory() as d:
            form = os.path.join(d, "f.md")
            out = os.path.join(d, "p.json")
            header = ("## L0 · x\n\n| 指标 | 变体 | 怎么测 | 依赖 | 值 | CV% |\n"
                      "|:--|:--|:--|:--|:--|:--|\n")
            with open(form, "w", encoding="utf-8") as fh:
                fh.write(header)
            rc = read_fill.main([form, "--out", out])
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as fh:
                p = json.load(fh)
        self.assertIn("c_star", p)

    def test_missing_file_exits_nonzero(self):
        self.assertEqual(read_fill.main(["nope.md"]), 2)


if __name__ == "__main__":
    unittest.main()
