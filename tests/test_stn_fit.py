"""STN几何自拟合实验的单元测试"""
from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from src.inference.schemas import RectificationMode
from src.models.stn import STN_INPUT_SIZE
from src.train.fit_stn_demo import (
    affine_matrix,
    fit_stn,
    invert_affine,
    make_target_plate,
    run_case,
    theta_to_angle_scale,
    warp,
)

OUT_HW = (STN_INPUT_SIZE[1], STN_INPUT_SIZE[0])

class MatrixToolTests(unittest.TestCase):
    def test_zero_rotation_is_identity(self) -> None:
        theta = affine_matrix(0.0)
        self.assertTrue(
            torch.allclose(theta, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        )

    def test_ninety_degree_rotation(self) -> None:
        theta = affine_matrix(90.0)
        expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(theta, expected, atol=1e-5))

    def test_angle_scale_roundtrip(self) -> None:
        theta = affine_matrix(12.0, scale=1.1)
        angle, scale = theta_to_angle_scale(theta)
        self.assertAlmostEqual(angle, 12.0, places=2)
        self.assertAlmostEqual(scale, 1.1, places=3)

    def test_invert_rotation(self) -> None:
        inverse = invert_affine(affine_matrix(12.0))
        self.assertTrue(torch.allclose(inverse, affine_matrix(-12.0), atol=1e-5))

    def test_invert_with_scale(self) -> None:
        inverse = invert_affine(affine_matrix(0.0, scale=0.5))
        angle, scale = theta_to_angle_scale(inverse)
        self.assertAlmostEqual(angle, 0.0, places=4)
        self.assertAlmostEqual(scale, 2.0, places=4)

class WarpTests(unittest.TestCase):
    def test_identity_warp_preserves_image(self) -> None:
        torch.manual_seed(0)
        image = torch.rand(1, 3, OUT_HW[0], OUT_HW[1]) * 2 - 1
        warped = warp(image, affine_matrix(0.0), OUT_HW)
        self.assertLess(float((warped - image).abs().max()), 1e-4)

    def test_target_plate_shape_and_range(self) -> None:
        plate = make_target_plate()
        self.assertEqual(tuple(plate.shape), (1, 3, OUT_HW[0], OUT_HW[1]))
        self.assertGreaterEqual(float(plate.min()), -1.0)
        self.assertLessEqual(float(plate.max()), 1.0)

    def test_rotation_actually_changes_pixels(self) -> None:
        plate = make_target_plate()
        tilted = warp(plate, affine_matrix(12.0), OUT_HW)
        self.assertGreater(float((tilted - plate).abs().mean()), 0.01)


class FitAffineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """整个类只训练一次，避免每条测试都重新拟合。"""
        cls.target = make_target_plate()
        cls.applied = affine_matrix(12.0)
        cls.tilted = warp(cls.target, cls.applied, OUT_HW)
        cls.stn, cls.history = fit_stn(cls.target, cls.tilted, steps=250)
        with torch.no_grad():
            cls.rectified = cls.stn(cls.tilted)
        assert cls.stn.localization is not None
        cls.learned = cls.stn.localization(cls.tilted).view(1, 2, 3)[0]

    def test_loss_decreases(self) -> None:
        self.assertLess(self.history[-1], self.history[0])
        with torch.no_grad():
            after = float(F.mse_loss(self.rectified, self.target))
        self.assertLess(after, float(F.mse_loss(self.tilted, self.target)))

    def test_learned_angle_is_opposite_to_applied(self) -> None:
        """核心判据：把图转了 +12°，STN 必须学出约 −12°。"""
        learned_angle, _scale = theta_to_angle_scale(self.learned)
        expected_angle, _ = theta_to_angle_scale(invert_affine(self.applied))
        self.assertAlmostEqual(expected_angle, -12.0, places=2)
        self.assertLess(abs(learned_angle - expected_angle), 1.5)
        self.assertLess(learned_angle, 0.0)

    def test_learned_scale_is_close_to_one(self) -> None:
        _angle, scale = theta_to_angle_scale(self.learned)
        self.assertAlmostEqual(scale, 1.0, places=1)

class FitScaleTests(unittest.TestCase):
    def test_recovers_inverse_scale(self) -> None:
        target = make_target_plate()
        applied = affine_matrix(-8.0, scale=0.9)
        tilted = warp(target, applied, OUT_HW)
        stn, _history = fit_stn(target, tilted, steps=250)
        assert stn.localization is not None
        with torch.no_grad():
            learned = stn.localization(tilted).view(1, 2, 3)[0]
        _angle, learned_scale = theta_to_angle_scale(learned)
        _angle2, expected_scale = theta_to_angle_scale(invert_affine(applied))
        self.assertLess(abs(learned_scale - expected_scale) / expected_scale, 0.08)

class PerspectiveFitTests(unittest.TestCase):
    def test_perspective_recovers_rotation(self) -> None:
        target = make_target_plate()
        applied = affine_matrix(6.0)
        tilted = warp(target, applied, OUT_HW)
        stn, _history = fit_stn(
            target, tilted, mode=RectificationMode.PERSPECTIVE, steps=250
        )
        assert stn.localization is not None
        with torch.no_grad():
            raw = stn.localization(tilted)
        self.assertEqual(tuple(raw.shape), (1, 8))

        matrix = torch.zeros(3, 3)
        matrix[0] = raw[0, 0:3]
        matrix[1] = raw[0, 3:6]
        matrix[2, 0:2] = raw[0, 6:8]
        matrix[2, 2] = 1.0
        matrix = matrix / float(matrix[2, 2])
        angle, _scale = theta_to_angle_scale(matrix[:2])
        self.assertLess(abs(angle - (-6.0)), 1.5)

class RunCaseTests(unittest.TestCase):
    def test_run_case_reports_pass(self) -> None:
        result = run_case("rot+10", affine_matrix(10.0), make_target_plate(), steps=250)
        self.assertTrue(bool(result["passed"]), msg=str(result))
        self.assertEqual(result["case"], "rot+10")
        self.assertLess(float(result["mse_after"]), float(result["mse_before"]))
        self.assertLess(float(result["angle_error"]), 1.5)

    def test_grad_clip_can_be_disabled(self) -> None:
        target = make_target_plate()
        tilted = warp(target, affine_matrix(5.0), OUT_HW)
        stn, history = fit_stn(target, tilted, steps=20, grad_clip=0.0)
        self.assertTrue(math.isfinite(history[-1]))
        self.assertIsNotNone(stn.localization)


if __name__ == "__main__":
    unittest.main()






