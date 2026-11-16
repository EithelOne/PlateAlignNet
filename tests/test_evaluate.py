"""统一评估模块的单元测试"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.data.plate_dataset import PlateDataset, build_synthetic_dataset
from src.eval.evaluate import evaluate_dataset, format_table, load_checkpoint
from src.models.plate_recognizer import build_recognizer
from src.models.stn import STN_INPUT_SIZE
from src.train.train_end2end import save_checkpoint


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temp.name)
        cls.manifest = build_synthetic_dataset(cls.root, count=4, seed=11)
        cls.dataset = PlateDataset.from_manifest(
            cls.manifest, image_root=cls.root, size=STN_INPUT_SIZE
        )
        torch.manual_seed(0)
        cls.model = build_recognizer("off")
        cls.model.eval()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_metrics_have_expected_keys(self) -> None:
        metrics = evaluate_dataset(self.model, self.dataset, torch.device("cpu"))
        for key in (
            "samples",
            "accuracy",
            "cer",
            "latency_ms",
            "fallback",
            "by_length",
            "num_failures",
            "failures",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["samples"], 4)

    def test_metric_ranges(self) -> None:
        metrics = evaluate_dataset(self.model, self.dataset, torch.device("cpu"))
        self.assertGreaterEqual(float(metrics["accuracy"]), 0.0)
        self.assertLessEqual(float(metrics["accuracy"]), 1.0)
        self.assertGreaterEqual(float(metrics["cer"]), 0.0)
        self.assertGreaterEqual(float(metrics["latency_ms"]), 0.0)

    def test_by_length_matches_targets(self) -> None:
        metrics = evaluate_dataset(self.model, self.dataset, torch.device("cpu"))
        expected = {str(len(self.dataset[i].text)) for i in range(len(self.dataset))}
        self.assertEqual(set(metrics["by_length"]), expected)

    def test_batch_size_one_works(self) -> None:
        metrics = evaluate_dataset(self.model, self.dataset, torch.device("cpu"), batch_size=1)
        self.assertEqual(metrics["samples"], 4)

    def test_failures_limited_and_sorted(self) -> None:
        metrics = evaluate_dataset(
            self.model, self.dataset, torch.device("cpu"), max_failures=2
        )
        failures = metrics["failures"]
        self.assertLessEqual(len(failures), 2)
        distances = [int(item["edit_distance"]) for item in failures]
        self.assertEqual(distances, sorted(distances, reverse=True))

    def test_format_table_contains_row(self) -> None:
        text = format_table(
            [{"mode": "affine", "parameters": 164364, "samples": 246,
              "accuracy": 0.5, "cer": 0.1, "latency_ms": 0.8, "num_failures": 123}]
        )
        self.assertIn("| 模式 |", text)
        self.assertIn("affine", text)
        self.assertIn("50.00%", text)

    def test_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ckpt.pt"
            save_checkpoint(self.model, path, "off", {"note": "unit-test"})
            loaded, payload = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(payload["mode"], "off")
            self.assertEqual(loaded.describe()["mode"], "off")
            self.assertEqual(loaded.describe()["total_parameters"], 107558)
            self.assertIn("charset", payload)


if __name__ == "__main__":
    unittest.main()





