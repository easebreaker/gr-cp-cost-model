"""The measurement plan is a document that happens to be code.

It must be complete (every model parameter has a job that pins it), ordered
(gates before the things they gate), and cheap enough that nobody argues about
running it.
"""

import os
import tempfile
import unittest

from grcost.manifest import (
    Job, build_plan, cost_report, gates, unresolved, write_manifest,
)


class TestPlanCoverage(unittest.TestCase):
    def setUp(self):
        self.jobs = build_plan("mtgr-large", 4096, 32, 8)
        self.by_purpose = {}
        for j in self.jobs:
            self.by_purpose.setdefault(j.purpose, []).append(j)

    def test_every_model_parameter_is_pinned(self):
        """Each fitted or pinned parameter must trace back to at least one job."""
        needed = {
            "eff_curve": "eff_inf / c_star / gamma",
            "collective": "alpha / bw_comm / overlap",
            "hbm_bw": "bw_util",
            "dispatch": "t_fixed / c_dispatch",
            "memory": "a / M_static",
            "imbalance": "the max-over-ranks composition",
            "mask_topology": "the pair-count coefficient",
        }
        for purpose, why in needed.items():
            self.assertIn(purpose, self.by_purpose, f"no job pins {why}")

    def test_ids_are_unique(self):
        ids = [j.id for j in self.jobs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_gates_reference_real_jobs(self):
        ids = {j.id for j in self.jobs}
        prefixes = {"M1.1", "M3.1", "S1.1", "S2.1", "E1.1", "E4.1", "E3.1"}
        self.assertTrue(prefixes <= ids)

    def test_cp_free_jobs_dominate(self):
        """Most of the calibration must not require a CP implementation."""
        free = sum(j.cost for j in self.jobs if not j.needs_cp)
        total = sum(j.cost for j in self.jobs)
        self.assertGreater(free / total, 0.6)

    def test_eff_curve_grid_spans_16x(self):
        job = self.by_purpose["eff_curve"][0]
        lengths = job.knobs["T"]
        self.assertGreaterEqual(max(lengths) / min(lengths), 16)
        self.assertIn(1, [job.knobs["cp"]])

    def test_plan_fits_in_a_workday(self):
        total = sum(j.cost for j in self.jobs) / 60.0
        self.assertLess(total, 8.0)

    def test_cost_is_repeats_times_minutes(self):
        j = Job("x", "L0", "p", {}, "m", "d", repeats=3, est_minutes=2.0)
        self.assertAlmostEqual(j.cost, 6.0)


class TestOutput(unittest.TestCase):
    def test_manifest_writes_union_of_knobs(self):
        jobs = build_plan("mtgr-large", 4096, 32, 4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.csv")
            write_manifest(jobs, path)
            with open(path, encoding="utf-8") as fh:
                header = fh.readline().strip().split(",")
                n = sum(1 for _ in fh)
        self.assertEqual(n, len(jobs))
        self.assertIn("knob_op", header)
        self.assertIn("est_minutes", header)

    def test_reports_are_text(self):
        jobs = build_plan()
        for text in (cost_report(jobs), gates(jobs), unresolved(jobs)):
            self.assertIsInstance(text, str)
            self.assertIn("\n", text)

    def test_cp_max_is_respected(self):
        jobs = build_plan("mtgr-large", 4096, 32, cp_max=4)
        for j in jobs:
            npus = j.knobs.get("npus")
            if isinstance(npus, list) and npus and all(isinstance(x, int) for x in npus):
                self.assertLessEqual(max(npus), 4)
            cps = j.knobs.get("cp")
            if isinstance(cps, list) and cps and all(isinstance(x, int) for x in cps):
                self.assertLessEqual(max(cps), 4)


if __name__ == "__main__":
    unittest.main()
