"""STN校正模块的测试"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from PIL import Image

from src.inference.schemas import RectificationMode
from src.models.stn import (
    ALIGN_CORNERS,
    IDENTITY_PARAMS,
    PADDING_MODE,
    STN_INPUT_SIZE,
    STN_OUTPUT_SIZE,
    LocalizationNet,
    STN,
    count_parameters,
    identity_grid,
    pil_to_tensor,
    rectify_pil,
    tensor_to_pil,
)

def _random_batch(batch: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(batch, 3, STN_INPUT_SIZE[1], STN_INPUT_SIZE[0]) * 2 - 1


class SizeTests(unittest.TestCase):
    def test_default_sizes(self) -> None:
        self.assertEqual(STN_INPUT_SIZE, (188, 48))
        self.assertEqual(STN_OUTPUT_SIZE, (94, 24))

    def test_output_shape_affine(self) -> None:
        stn = STN(RectificationMode.AFFINE)
        with torch.no_grad():
            output = stn(_random_batch(2))
        self.assertEqual(tuple(output.shape), (2, 3, STN_OUTPUT_SIZE[1], STN_OUTPUT_SIZE[0]))

    def test_output_shape_perspective(self) -> None:
        stn = STN(RectificationMode.PERSPECTIVE)
        with torch.no_grad():
            output = stn(_random_batch(3))
        self.assertEqual(tuple(output.shape), (3, 3, 24, 94))

    def test_input_size_mismatch_raises(self) -> None:
        stn = STN(RectificationMode.AFFINE)
        with self.assertRaises(ValueError):
            stn(torch.zeros(1, 3, 24, 94))

    def test_three_dim_input_raises(self) -> None:
        stn = STN(RectificationMode.AFFINE)
        with self.assertRaises(ValueError):
            stn(torch.zeros(3, 48, 188))


class IdentityInitTests(unittest.TestCase):
    def test_affine_bias_is_identity(self) -> None:
        stn = STN(RectificationMode.AFFINE)
        self.assertIsNotNone(stn.localization)
        self.assertEqual(
            stn.localization.fc2.bias.tolist(), list(IDENTITY_PARAMS[RectificationMode.AFFINE])
        )
        self.assertTrue(bool((stn.localization.fc2.weight == 0).all()))

    def test_perspective_bias_is_identity(self) -> None:
        stn = STN(RectificationMode.PERSPECTIVE)
        self.assertEqual(
            stn.localization.fc2.bias.tolist(),
            list(IDENTITY_PARAMS[RectificationMode.PERSPECTIVE]),
        )

    def test_affine_identity_preserves_same_size_input(self) -> None:
        """同尺寸输入输出时，恒等初始化的 STN 应该近似"原图复制"。"""
        stn = STN(RectificationMode.AFFINE, output_size=STN_INPUT_SIZE)
        sample = _random_batch(2)
        with torch.no_grad():
            output = stn(sample)
        self.assertLess(float((output - sample).abs().max()), 1e-4)

    def test_perspective_identity_preserves_same_size_input(self) -> None:
        stn = STN(RectificationMode.PERSPECTIVE, output_size=STN_INPUT_SIZE)
        sample = _random_batch(2)
        with torch.no_grad():
            output = stn(sample)
        self.assertLess(float((output - sample).abs().max()), 1e-4)

    def test_identity_grid_is_identity_mapping(self) -> None:
        grid = identity_grid(1, 24, 94)
        self.assertEqual(tuple(grid.shape), (1, 24, 94, 2))
        self.assertLessEqual(float(grid.abs().max()), 1.0 + 1e-6)


class OffModeTests(unittest.TestCase):
    def test_off_has_no_localization_net(self) -> None:
        stn = STN(RectificationMode.OFF)
        self.assertIsNone(stn.localization)

    def test_off_equals_bilinear_baseline_exactly(self) -> None:
        """关掉 STN 必须和"什么都不做的基线"逐元素一致，否则消融对比不干净。"""
        stn = STN(RectificationMode.OFF)
        sample = _random_batch(2)
        with torch.no_grad():
            output = stn(sample)
            baseline = F.interpolate(
                sample, size=stn.output_hw, mode="bilinear", align_corners=ALIGN_CORNERS
            )
        self.assertTrue(torch.equal(output, baseline))

    def test_identity_stn_close_to_off_baseline(self) -> None:
        """恒等初始化的仿射 STN ≈ 关闭 STN（差异只来自采样实现）。"""
        off_stn = STN(RectificationMode.OFF)
        affine_stn = STN(RectificationMode.AFFINE)
        sample = _random_batch(2)
        with torch.no_grad():
            off_output = off_stn(sample)
            affine_output = affine_stn(sample)
        self.assertLess(float((off_output - affine_output).abs().max()), 1e-4)


class PerspectiveSafetyTests(unittest.TestCase):
    def test_extreme_translation_triggers_fallback(self) -> None:
        """把位置参数推到离谱的值，越界点会超过一半 → 必须整体回退成恒等变换。

        默认 STN 使用有界残差，极端 bias 会被夹到安全范围、不再触发回退；
        这里改用大 bound 的无界 STN 来单独验证回退安全网本身仍可用。
        """
        stn = STN(RectificationMode.PERSPECTIVE, residual_bounds=(2.0,) * 8)
        sample = _random_batch(2)
        with torch.no_grad():
            stn.localization.fc2.bias.copy_(
                torch.tensor((1.0, 0.0, 5.0, 0.0, 1.0, 5.0, 0.0, 0.0))
            )
            warped, stats = stn(sample, return_stats=True)
            reference = F.grid_sample(
                sample,
                identity_grid(2, *stn.output_hw),
                align_corners=ALIGN_CORNERS,
                padding_mode=PADDING_MODE,
            )
        self.assertGreater(stats.fallback_fraction, 0.0)
        self.assertLess(float((warped - reference).abs().max()), 1e-4)

    def test_moderate_translation_reports_out_of_range(self) -> None:
        stn = STN(RectificationMode.PERSPECTIVE)
        sample = _random_batch(2)
        with torch.no_grad():
            stn.localization.fc2.bias.copy_(
                torch.tensor((1.0, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0))
            )
            _warped, stats = stn(sample, return_stats=True)
        self.assertGreater(stats.out_of_range_fraction, 0.0)
        self.assertEqual(stats.mode, "perspective")

    def test_no_nan_in_any_mode(self) -> None:
        sample = _random_batch(2)
        for mode in RectificationMode:
            stn = STN(mode)
            with torch.no_grad():
                output = stn(sample)
            self.assertFalse(bool(torch.isnan(output).any()), msg=mode)


class LearnabilityTests(unittest.TestCase):
    def test_gradient_reaches_localization_net(self) -> None:
        stn = STN(RectificationMode.AFFINE, output_size=STN_INPUT_SIZE)
        sample = _random_batch(2)
        target = torch.roll(sample, shifts=4, dims=-1)
        warped = stn(sample)
        loss = F.mse_loss(warped, target)
        loss.backward()
        gradient = stn.localization.fc2.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_a_few_steps_move_parameters_away_from_identity(self) -> None:
        """能学的前提：参数确实会被优化器改动。"""
        stn = STN(RectificationMode.AFFINE, output_size=STN_INPUT_SIZE)
        sample = _random_batch(2)
        target = torch.roll(sample, shifts=4, dims=-1)
        optimizer = torch.optim.Adam(stn.parameters(), lr=1e-2)
        for _ in range(5):
            optimizer.zero_grad()
            loss = F.mse_loss(stn(sample), target)
            loss.backward()
            optimizer.step()
        changed = not torch.allclose(
            stn.localization.fc2.bias.detach(),
            torch.tensor(IDENTITY_PARAMS[RectificationMode.AFFINE]),
        )
        self.assertTrue(changed)

    def test_param_count_affine_vs_perspective(self) -> None:
        self.assertEqual(LocalizationNet(6).num_params, 6)
        self.assertEqual(LocalizationNet(8).num_params, 8)
        self.assertGreater(count_parameters(STN(RectificationMode.AFFINE)), 1000)


class PilBridgeTests(unittest.TestCase):
    def test_pil_to_tensor_shape_and_range(self) -> None:
        image = Image.new("RGB", (300, 90), "#0b5ba6")
        tensor = pil_to_tensor(image)
        self.assertEqual(tuple(tensor.shape), (1, 3, STN_INPUT_SIZE[1], STN_INPUT_SIZE[0]))
        self.assertGreaterEqual(float(tensor.min()), -1.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_tensor_to_pil_roundtrip(self) -> None:
        image = Image.new("RGB", (188, 48), "#0b5ba6")
        back = tensor_to_pil(pil_to_tensor(image))
        self.assertEqual(back.size, (188, 48))
        self.assertEqual(back.mode, "RGB")

    def test_rectify_pil_returns_image(self) -> None:
        image = Image.new("RGB", (300, 90), "#0b5ba6")
        stn = STN(RectificationMode.AFFINE)
        result = rectify_pil(stn, image)
        self.assertEqual(result.size, STN_OUTPUT_SIZE)

    def test_rectify_pil_with_stats(self) -> None:
        image = Image.new("RGB", (300, 90), "#0b5ba6")
        stn = STN(RectificationMode.PERSPECTIVE)
        result, stats = rectify_pil(stn, image, return_stats=True)
        self.assertEqual(result.size, STN_OUTPUT_SIZE)
        self.assertEqual(stats.mode, "perspective")

if __name__ == "__main__":
    unittest.main()