"""STN+LPRNet组合模型的单元测试"""
from __future__ import annotations

import unittest

import torch

from src.inference.schemas import RectificationMode
from src.models.lprnet_spec import trace_layers
from src.models.plate_recognizer import (
    PlateRecognizer,
    build_recognizer,
    character_error_rate,
    edit_distance,
)
from src.models.stn import STN_INPUT_SIZE, STN_OUTPUT_SIZE
from src.utils.plate_charset import NUM_CLASSES

def _images(batch: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(batch, 3, STN_INPUT_SIZE[1], STN_INPUT_SIZE[0]) * 2 - 1

class ShapeTests(unittest.TestCase):
    def test_all_modes_produce_same_shapes(self) -> None:
        for mode in RectificationMode:
            model = build_recognizer(mode)
            model.eval()
            with torch.no_grad():
                details = model(_images(3), return_details=True)
            self.assertEqual(
                tuple(details.logits.shape),
                (trace_layers().time_steps, 3, NUM_CLASSES),
                msg=mode,
            )
            self.assertEqual(
                tuple(details.rectified.shape),
                (3, 3, STN_OUTPUT_SIZE[1], STN_OUTPUT_SIZE[0]),
                msg=mode,
            )

    def test_default_forward_returns_logits_only(self) -> None:
        model = build_recognizer(RectificationMode.AFFINE)
        model.eval()
        with torch.no_grad():
            output = model(_images(2))
        self.assertEqual(tuple(output.shape), (23, 2, 70))

    def test_input_size_mismatch_raises(self) -> None:
        model = build_recognizer(RectificationMode.AFFINE)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3, 24, 94))

class ParameterTests(unittest.TestCase):
    def test_parameter_counts_per_mode(self) -> None:
        expected = {
            RectificationMode.OFF: (0, 107558, 107558),
            RectificationMode.AFFINE: (56806, 107558, 164364),
            RectificationMode.PERSPECTIVE: (56936, 107558, 164494),
        }
        for mode, (stn, lprnet, total) in expected.items():
            info = build_recognizer(mode).describe()
            self.assertEqual(info["stn_parameters"], stn, msg=mode)
            self.assertEqual(info["lprnet_parameters"], lprnet, msg=mode)
            self.assertEqual(info["total_parameters"], total, msg=mode)

    def test_describe_fields(self) -> None:
        info = build_recognizer(RectificationMode.AFFINE).describe()
        self.assertEqual(info["mode"], "affine")
        self.assertEqual(info["input_size"], [188, 48])
        self.assertEqual(info["output_size"], [94, 24])
        self.assertTrue(info["stn_trainable"])

    def test_mode_accepts_string(self) -> None:
        model = PlateRecognizer("perspective")
        self.assertEqual(model.mode, RectificationMode.PERSPECTIVE)

class FreezeTests(unittest.TestCase):
    def test_freeze_stn_marks_parameters(self) -> None:
        model = build_recognizer(RectificationMode.AFFINE, freeze_stn=True)
        assert model.stn.localization is not None
        self.assertFalse(any(p.requires_grad for p in model.stn.localization.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.lprnet.parameters()))
        self.assertFalse(model.describe()["stn_trainable"])

    def test_set_stn_trainable_roundtrip(self) -> None:
        model = build_recognizer(RectificationMode.AFFINE)
        model.set_stn_trainable(False)
        assert model.stn.localization is not None
        self.assertFalse(any(p.requires_grad for p in model.stn.localization.parameters()))
        model.set_stn_trainable(True)
        self.assertTrue(any(p.requires_grad for p in model.stn.localization.parameters()))

    def test_frozen_stn_still_forwards(self) -> None:
        model = build_recognizer(RectificationMode.AFFINE, freeze_stn=True)
        model.eval()
        with torch.no_grad():
            output = model(_images(2))
        self.assertEqual(tuple(output.shape), (23, 2, 70))

class GradientTests(unittest.TestCase):
    def test_gradient_reaches_stn_and_lprnet(self) -> None:
        model = build_recognizer(RectificationMode.AFFINE)
        images = _images(2)
        logits = model(images)
        loss = logits.sum()
        loss.backward()
        assert model.stn.localization is not None
        stn_grad = model.stn.localization.fc2.weight.grad
        lprnet_grad = model.lprnet.head.weight.grad
        self.assertIsNotNone(stn_grad)
        self.assertIsNotNone(lprnet_grad)
        self.assertGreater(float(stn_grad.abs().sum()), 0.0)
        self.assertGreater(float(lprnet_grad.abs().sum()), 0.0)

    def test_frozen_stn_gets_no_gradient(self) -> None:
        model = build_recognizer(RectificationMode.AFFINE, freeze_stn=True)
        model(_images(2)).sum().backward()
        assert model.stn.localization is not None
        self.assertIsNone(model.stn.localization.fc2.weight.grad)

class RecognizeTests(unittest.TestCase):
    def test_recognize_returns_texts(self) -> None:
        model = build_recognizer(RectificationMode.AFFINE)
        texts = model.recognize(_images(3))
        self.assertEqual(len(texts), 3)
        self.assertTrue(all(isinstance(text, str) for text in texts))

    def test_recognize_works_with_batch_one(self) -> None:
        """推理走 eval 模式，所以 batch=1 也能跑（训练时 BatchNorm 不允许 batch=1）。"""
        model = build_recognizer(RectificationMode.PERSPECTIVE)
        self.assertEqual(len(model.recognize(_images(1))), 1)

class MetricTests(unittest.TestCase):
    def test_edit_distance_identical(self) -> None:
        self.assertEqual(edit_distance("皖A355U6", "皖A355U6"), 0)

    def test_edit_distance_single_substitution(self) -> None:
        self.assertEqual(edit_distance("皖A355U6", "皖A357U6"), 1)

    def test_edit_distance_insert_and_delete(self) -> None:
        self.assertEqual(edit_distance("皖A355U", "皖A355U6"), 1)
        self.assertEqual(edit_distance("皖A355U67", "皖A355U6"), 1)

    def test_cer(self) -> None:
        self.assertAlmostEqual(character_error_rate(["皖A355U6"], ["皖A355U6"]), 0.0)
        self.assertAlmostEqual(
            character_error_rate(["皖AL0000"], ["皖A10000"]), 1 / 7, places=6
        )
        self.assertAlmostEqual(character_error_rate([], []), 0.0)


if __name__ == "__main__":
    unittest.main()        









