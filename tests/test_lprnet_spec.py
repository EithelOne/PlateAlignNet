"""LPRNET结构规格的单元测试"""

from __future__ import annotations

import unittest

from src.models.lprnet_spec import (
    INPUT_SHAPE,
    LPRNET_HEAD,
    LPRNET_LAYERS,
    LayerSpec,
    check_time_steps,
    conv_output,
    conv_parameters,
    describe,
    trace_layers,
)
from src.utils.plate_charset import NUM_CLASSES

class ConvOutputTests(unittest.TestCase):
    def test_same_padding_keeps_size(self) -> None:
        self.assertEqual(conv_output(94, kernel=3, stride=1, padding=1), 94)
        self.assertEqual(conv_output(24, kernel=3, stride=1, padding=1), 24)

    def test_half_downsampling(self) -> None:
        self.assertEqual(conv_output(24, kernel=2, stride=2), 12)
        self.assertEqual(conv_output(47, kernel=2, stride=2), 23)
        self.assertEqual(conv_output(94, kernel=2, stride=2), 47)

    def test_one_by_one_keeps_size(self) -> None:
        self.assertEqual(conv_output(23, kernel=1, stride=1), 23)

    def test_invalid_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            conv_output(0, kernel=3)
        with self.assertRaises(ValueError):
            conv_output(10, kernel=3, stride=0)

    def test_too_large_kernel_raises(self) -> None:
        with self.assertRaises(ValueError):
            conv_output(3, kernel=5, stride=2)

class ParameterTests(unittest.TestCase):
    def test_conv_parameters_without_batchnorm(self) -> None:
        self.assertEqual(conv_parameters(3, 32, 3), 3 * 32 * 9 + 32)

    def test_conv_parameters_with_batchnorm(self) -> None:
        self.assertEqual(conv_parameters(3, 32, 3, batchnorm=True), 3 * 32 * 9 + 32 + 64)

    def test_one_by_one_head_parameters(self) -> None:
        self.assertEqual(conv_parameters(64, NUM_CLASSES, 1), 64 * NUM_CLASSES + NUM_CLASSES)

class TraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = trace_layers()

    def test_input_shape_is_94_by_24(self) -> None:
        self.assertEqual(INPUT_SHAPE, (3, 24, 94))

    def test_feature_shape_before_head(self) -> None:
        self.assertEqual(self.trace.feature_shape, (NUM_CLASSES, 6, 23))

    def test_time_steps(self) -> None:
        self.assertEqual(self.trace.time_steps, 23)

    def test_num_classes(self) -> None:
        self.assertEqual(self.trace.num_classes, NUM_CLASSES)
        self.assertEqual(self.trace.num_classes, 70)

    def test_total_parameters_are_locked(self) -> None:
        """参数量锁定：以后谁改了结构，这条测试会立刻失败。"""
        self.assertEqual(self.trace.total_parameters, 107558)

    def test_per_layer_parameters(self) -> None:
        by_name = {report.name: report for report in self.trace.reports}
        self.assertEqual(by_name["conv1"].parameters, 960)
        self.assertEqual(by_name["pool1"].parameters, 0)
        self.assertEqual(by_name["pool2"].parameters, 0)
        self.assertEqual(by_name["head"].parameters, 4550)

    def test_label_shape_for_ctc(self) -> None:
        self.assertEqual(self.trace.label_shape(4), (23, 4, 70))

    def test_channel_mismatch_raises(self) -> None:
        broken = (LayerSpec("convX", "conv", in_channels=5, out_channels=8, kernel=3, padding=1),)
        with self.assertRaises(ValueError):
            trace_layers(broken)

    def test_unknown_kind_raises(self) -> None:
        broken = (LayerSpec("weird", "attention"),)
        with self.assertRaises(ValueError):
            trace_layers(broken)

    def test_head_channel_mismatch_raises(self) -> None:
        bad_head = LayerSpec("head", "conv", in_channels=32, out_channels=NUM_CLASSES, kernel=1)
        with self.assertRaises(ValueError):
            trace_layers(head=bad_head)

class CapacityTests(unittest.TestCase):
    def test_normal_plates_fit(self) -> None:
        self.assertEqual(check_time_steps(["京A12345", "粤AD12345", "京A112233"], 23), [])

    def test_too_narrow_input_flags_long_labels(self) -> None:
        trace = trace_layers(input_shape=(3, 24, 24))
        self.assertEqual(trace.time_steps, 6)
        self.assertEqual(check_time_steps(["京A112233"], trace.time_steps), ["京A112233"])

    def test_width_48_gives_12_steps(self) -> None:
        trace = trace_layers(input_shape=(3, 24, 48))
        self.assertEqual(trace.time_steps, 12)
        self.assertEqual(check_time_steps(["京A12345"], trace.time_steps), [])

class DescribeTests(unittest.TestCase):
    def test_description_mentions_time_steps_and_parameters(self) -> None:
        text = describe(trace_layers())
        self.assertIn("T=23", text)
        self.assertIn("107,558", text)
        self.assertIn("conv5", text)


if __name__ == "__main__":
    unittest.main()




