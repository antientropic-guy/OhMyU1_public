"""Small integration checks; synthetic cases do not tune benchmark parameters."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import run_assignment as experiment
import numpy as np


class AssignmentTests(unittest.TestCase):
    def test_feasibility(self):
        good = np.eye(3, dtype=np.int32).reshape(1, -1)
        self.assertTrue(experiment.feasible(good, 3)[0])
        self.assertFalse(experiment.feasible(np.ones((1, 9), dtype=np.int32), 3)[0])
        self.assertFalse(experiment.feasible(2 * good, 3)[0])

    def test_shared_real_data(self):
        base = experiment.ROOT / "data/protes_comparison/local_calibration/inputs"
        if not base.exists():
            self.skipTest("Optional local calibration data absent")
        for name in ("linear", "quadratic"):
            meta, data, value, costs = experiment.load_input(base / f"{name}_12_1")
            self.assertEqual(data.shape, (400, 144))
            self.assertTrue(experiment.feasible(data, 12).all())

    def test_upstream_end_to_end_and_resume(self):
        with tempfile.TemporaryDirectory(dir=experiment.HERE) as temporary:
            folder = Path(temporary)
            rng = np.random.default_rng(42)
            initial = np.array([np.eye(3, dtype=np.uint8)[rng.permutation(3)].ravel() for _ in range(400)])
            # Deliberately nonsymmetric Q tests x'Qx and Julia column-major export.
            q = np.arange(81, dtype=np.float64).reshape(9, 9) / 13
            costs = np.einsum("bi,ij,bj->b", initial, q, initial)
            raw = initial.tobytes()
            (folder / "initial.u8").write_bytes(raw)
            q.ravel(order="F").astype("<f8").tofile(folder / "objective.f64")
            np.savetxt(folder / "initial_costs.csv", costs, delimiter=",")
            fingerprint = hashlib.sha256(b"(9, 400):" + raw).hexdigest()
            (folder / "metadata.tsv").write_text(
                "objective\tn\tinstance\tseed\tinitial_sha256\tsource_sha256\n"
                f"quadratic\t3\t1\t42\t{fingerprint}\tsynthetic\n")
            (folder / "eels_timing.tsv").write_text("seconds\tc_min\n2.0\t0.0\n")
            experiment.warmup(9)
            path = folder / "result.json"
            result = experiment.run_one(folder, path)
            self.assertTrue(experiment.feasible(np.array(result["incumbent"])[None, :], 3)[0])
            self.assertLessEqual(result["c_min"], float(costs.min()) + 1e-9)
            self.assertGreater(result["diagnostics"]["generated"], 0)
            self.assertGreater(result["diagnostics"]["feasible_generated"], 0)
            self.assertTrue(result["initial_training_complete"])
            self.assertEqual(experiment.PROCESS.cpu_affinity(), [experiment.CPU])
            self.assertEqual(result, experiment.run_one(folder, path))
            self.assertLess(result["deadline_overshoot_seconds"], 1.0)
            original = (folder / "initial.u8").read_bytes()
            (folder / "initial.u8").write_bytes(bytes([1-original[0]]) + original[1:])
            with self.assertRaises(AssertionError):
                experiment.load_input(folder)


if __name__ == "__main__":
    unittest.main()
