"""LPRNet（PyTorch 版）的单元测试"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.data.plate_dataset import PLATE_SIZE
from src.models.lprnet import (
    LPRNet,
    build_model,
    compute_ctc_loss,
    ctc_loss_from_log_probs,
    decode_batch,
    log_probs_from_logits,
    spec_consistency,
    to_ctc_targets,
)
from src.models.lprnet_spec import trace_layers
from src.utils.plate_charset import BLANK_INDEX, NUM_CLASSES, encode_text, greedy_decode

TIME_STEPS = trace_layers().time_steps

def _perfect_log_probs(text: str, time_steps: int = TIME_STEPS) -> torch.Tensor:
    """构造一份"完美预测"的 log 概率：标签按顺序铺开，重复字符之间插 blank。"""
    indices = encode_text(text)
    path: list[int] = []
    previous: int | None = None
    for index in indices:
        if index == previous:
            path.append(BLANK_INDEX)
        path.append(index)
        previous = index
    assert len(path) <= time_steps, f"{text} 需要 {len(path)} 帧，超过 {time_steps}"
    path += [BLANK_INDEX] * (time_steps - len(path))

    chosen = 0.999
    other = (1.0 - chosen) / (NUM_CLASSES - 1)
    probs = torch.full((time_steps, 1, NUM_CLASSES), other, dtype=torch.float32)
    for frame, index in enumerate(path):
        probs[frame, 0, index] = chosen
    return torch.log(probs)

def _targets(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    labels = np.array([encode_text(text)], dtype=np.int64)
    lengths = np.array([len(text)], dtype=np.int64)
    return to_ctc_targets(labels, lengths)


class ModelTests(unittest.TestCase):
    def test_forward_shape(self) -> None:
        model = build_model()
        model.eval()
        images = torch.zeros(2, 3, PLATE_SIZE[1], PLATE_SIZE[0])
        with torch.no_grad():
            output = model(images)
        self.assertEqual(tuple(output.shape), (TIME_STEPS, 2, NUM_CLASSES))

    def test_parameter_count_matches_spec(self) -> None:
        parameters = sum(parameter.numel() for parameter in LPRNet().parameters())
        self.assertEqual(parameters, trace_layers().total_parameters)
        self.assertEqual(parameters, 107558)

    def test_spec_consistency_helper(self) -> None:
        info = spec_consistency()
        self.assertTrue(info["consistent"], msg=str(info))
        self.assertEqual(info["actual_time_steps"], TIME_STEPS)

class LogProbTests(unittest.TestCase):
    def test_log_probs_are_normalised(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(TIME_STEPS, 2, NUM_CLASSES)
        log_probs = log_probs_from_logits(logits)
        total = log_probs.exp().sum(dim=2)
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-5))

    def test_log_probs_are_non_positive(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(TIME_STEPS, 2, NUM_CLASSES)
        self.assertLessEqual(float(log_probs_from_logits(logits).max()), 0.0)


class CTCLossTests(unittest.TestCase):
    def test_perfect_alignment_has_near_zero_loss(self) -> None:
        log_probs = _perfect_log_probs("京A12345")
        targets, lengths = _targets("京A12345")
        loss = ctc_loss_from_log_probs(log_probs, targets, lengths)
        self.assertLess(float(loss), 0.1)

    def test_perfect_alignment_with_repeated_characters(self) -> None:
        """相邻重复字符必须靠 blank 分隔，这条验证的就是 CTC 的核心机制。"""
        text = "京A112233"
        log_probs = _perfect_log_probs(text)
        targets, lengths = _targets(text)
        loss = ctc_loss_from_log_probs(log_probs, targets, lengths)
        self.assertLess(float(loss), 0.1)

    def test_uniform_logits_give_large_loss(self) -> None:
        logits = torch.zeros(TIME_STEPS, 1, NUM_CLASSES)
        targets, lengths = _targets("京A12345")
        loss = compute_ctc_loss(logits, targets, lengths)
        self.assertGreater(float(loss), 8.0)
        self.assertLess(float(loss), 20.0)


class TargetConversionTests(unittest.TestCase):
    def test_padding_minus_one_becomes_zero(self) -> None:
        labels = np.array([[1, 2, -1], [3, -1, -1]], dtype=np.int64)
        lengths = np.array([2, 1], dtype=np.int64)
        targets, target_lengths = to_ctc_targets(labels, lengths)
        self.assertEqual(targets.dtype, torch.long)
        self.assertTrue(bool((targets >= 0).all()))
        self.assertEqual(targets[0].tolist(), [1, 2, 0])
        self.assertEqual(target_lengths.tolist(), [2, 1])

class DecodeTests(unittest.TestCase):
    def test_decode_recovers_text_from_perfect_log_probs(self) -> None:
        log_probs = _perfect_log_probs("京A12345")
        self.assertEqual(decode_batch(log_probs), ["京A12345"])

    def test_decode_recovers_text_with_repeats(self) -> None:
        log_probs = _perfect_log_probs("京A112233")
        self.assertEqual(decode_batch(log_probs), ["京A112233"])

    def test_decode_matches_charset_greedy_decode(self) -> None:
        torch.manual_seed(3)
        logits = torch.randn(TIME_STEPS, 2, NUM_CLASSES)
        decoded = decode_batch(logits)
        probs = log_probs_from_logits(logits).exp().numpy()
        for column in range(2):
            expected, _path, _confidence = greedy_decode(probs[:, column, :])
            self.assertEqual(decoded[column], expected)


if __name__ == "__main__":
    unittest.main()












