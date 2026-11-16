"""The sheet is a contract, so it gets tested like one.

Three properties matter:

1. every row you must fill has a way to be obtained, and the ones that cannot be
   fitted are marked ``pinned``;
2. every symbol the cost model needs appears somewhere;
3. the CSV is loadable and re-generatable, and the Markdown survives a table
   parser (pipes escaped).
"""

import csv
import os
import tempfile
import unittest
from dataclasses import fields

from grcost.cost import Params
from grcost.sheet import (
    PROVENANCE, STAGES, Row, counts, expand, registry, write_csv,
    write_markdown_file,
)

ROWS = registry()
BY_SYMBOL = {r.symbol: r for r in ROWS if r.symbol}


class TestCoverage(unittest.TestCase):
    def test_every_free_parameter_appears(self):
        for name in Params.FREE:
            self.assertIn(name, BY_SYMBOL, f"{name} has no row in the sheet")

    def test_constants_that_must_not_be_fitted(self):
        """The five that cannot be identified from step time.

        `t_fixed` and `bw_util` are measured, `overlap` is pinned, `recompute`
        and `layers` are declared by the config. None may be `fitted`.
        """
        allowed = ("measured", "derived", "pinned", "declared")
        for name in ("t_fixed", "overlap", "bw_util", "recompute", "layers"):
            self.assertIn(name, BY_SYMBOL, name)
            self.assertIn(BY_SYMBOL[name].source, allowed,
                          f"{name} must not be fittable")
            self.assertNotEqual(BY_SYMBOL[name].source, "fitted", name)

    def test_layers_is_declared_not_measured(self):
        self.assertEqual(BY_SYMBOL["layers"].source, "declared")

    def test_all_six_stages_present(self):
        got = {r.stage for r in ROWS}
        self.assertEqual(got, set(STAGES))

    def test_provenance_values_are_from_the_enum(self):
        for r in ROWS:
            self.assertIn(r.source, PROVENANCE, f"{r.section}/{r.metric}")

    def test_every_row_explains_how_to_get_it(self):
        for r in ROWS:
            if r.stage == "G":
                continue
            self.assertTrue(r.how.strip(), f"{r.section}/{r.metric} has no method")

    def test_gates_have_both_a_check_and_a_consequence(self):
        gates = [r for r in ROWS if r.stage == "G"]
        self.assertEqual(len(gates), 7)
        for g in gates:
            self.assertTrue(g.metric.strip())
            self.assertTrue(g.target.strip(), f"{g.section} has no fallback")

    def test_prerequisite_symbols_exist(self):
        for r in ROWS:
            for dep in [x.strip() for x in r.requires.split(",") if x.strip()]:
                self.assertIn(dep, BY_SYMBOL,
                              f"{r.section}/{r.metric} requires unknown {dep}")


class TestEmission(unittest.TestCase):
    def test_csv_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.csv")
            write_csv(path, ROWS)
            with open(path, newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), len(expand(ROWS)))
        self.assertIn("value", rows[0])
        self.assertEqual(rows[0]["value"], "")   # blank by construction

    def test_each_row_has_a_value_slot(self):
        from grcost.sheet import SHEET_HEADER
        for line in expand(ROWS):
            self.assertEqual(len(line), len(SHEET_HEADER))
        # and the value column must actually be blank in the template
        vi = SHEET_HEADER.index("value")
        for line in expand(ROWS):
            self.assertEqual(line[vi], "")

    def test_repeat_expands_to_multiple_lines(self):
        rep = [r for r in ROWS if r.repeat and "|" in r.repeat]
        self.assertTrue(rep)
        n = sum(1 for line in expand(ROWS) if line[9])
        self.assertGreater(n, 0)

    def test_variant_values_are_self_describing(self):
        """A bare '2' in the variant column is unusable in a spreadsheet."""
        for line in expand(ROWS):
            v = line[9]
            if not v:
                continue
            self.assertRegex(v, r"[=:]",
                             f"variant {v!r} does not say what it varies")

    def test_l2_jobs_vary_different_knobs(self):
        by_job = {}
        for line in expand(ROWS):
            if line[0] == "L2" and line[9]:
                by_job.setdefault(line[1], set()).add(line[9].split("=")[0])
        self.assertEqual(by_job["E1.1"], {"cp"})
        self.assertEqual(by_job["E1.2"], {"B"})
        self.assertEqual(by_job["E2.1"], {"scheme"})
        self.assertEqual(by_job["E3.1"], {"model"})
        self.assertEqual(by_job["E4.1"], {"kernel"})

    def test_descriptive_repeat_is_not_falsely_expanded(self):
        """`T×cp 网格` is prose; it must stay on one line."""
        row = Row("L2", "E1.1", "x", repeat="T×cp 网格")
        self.assertEqual(len(expand([row])), 1)

    def test_markdown_tables_are_well_formed(self):
        """Every table row must have the same cell count as its header."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.md")
            write_markdown_file(path, ROWS)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        self.assertIn("\\|", text)      # at least one escaped pipe exists
        block, cells = [], None
        for line in text.splitlines() + [""]:
            if line.startswith("|"):
                n = line.replace("\\|", "").count("|")
                if cells is None:
                    cells = n
                block.append((n, line))
            else:
                if block:
                    for n, ln in block:
                        self.assertEqual(n, cells, f"ragged row: {ln[:70]}")
                block, cells = [], None

    def test_markdown_has_a_section_per_stage(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.md")
            write_markdown_file(path, ROWS)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        for stage, title in STAGES.items():
            self.assertIn(f"## {stage} · {title}", text)


class TestCounts(unittest.TestCase):
    def test_measured_dominates(self):
        c = counts(ROWS)
        self.assertGreater(c["measured"], c.get("fitted", 0))
        self.assertGreater(c["measured"], c.get("derived", 0))

    def test_only_one_pinned(self):
        self.assertEqual(counts(ROWS).get("pinned", 0), 1)


if __name__ == "__main__":
    unittest.main()
