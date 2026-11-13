from __future__ import annotations

import unittest

import numpy as np

"""字符表与 CTC 编解码的单元测试"""
from src.utils.plate_charset import (
    BLANK_INDEX,
    CHARSET,
    CHAR_TO_INDEX,
    NUM_CLASSES,
    ctc_collapse,
    decode_indices,
    encode_text,
    greedy_decode,
    labels_to_batch,
    min_time_steps,
    validate_label,
)
A = CHAR_TO_INDEX["A"]
B = CHAR_TO_INDEX["B"]

class CharsetTests(unittest.TestCase):
    def test_charset_is_frozen(self) -> None:
        """字符表一旦冻结，长度、唯一性、blank 位置都不许再变。"""
        self.assertEqual(len(CHARSET), 69)
        self.assertEqual(len(set(CHARSET)), 69)
        self.assertEqual(BLANK_INDEX, 69)
        self.assertEqual(NUM_CLASSES, 70)
        self.assertEqual(CHARSET[0], "皖")
        self.assertEqual(CHARSET[12], "京")
        self.assertEqual(CHARSET[32], "学")
        self.assertEqual(CHARSET[33], "O")
        self.assertEqual(CHARSET[35], "A")

    def test_encode_decode_roundtrip(self) -> None:
        for text in ("京A12345", "粤AD12345", "沪B6A521"):
            self.assertEqual(decode_indices(encode_text(text)), text)

    def test_unknown_char_raises(self) -> None:
        with self.assertRaises(KeyError):
            encode_text("京A1234!")

class CTCBridgeTests(unittest.TestCase):
    def test_collapse_merges_repeats_before_removing_blank(self) -> None:
        self.assertEqual(ctc_collapse([A, BLANK_INDEX, A]), "AA")
        self.assertEqual(ctc_collapse([A, A, BLANK_INDEX]), "A")
        self.assertEqual(ctc_collapse([A, BLANK_INDEX, BLANK_INDEX, A]), "AA")
        self.assertEqual(ctc_collapse([A, B, B]), "AB")
        self.assertEqual(ctc_collapse([BLANK_INDEX, BLANK_INDEX]), "")
        self.assertEqual(ctc_collapse([]), "")

    def test_greedy_decode_recovers_text(self) -> None:
        target = "京A12345"
        frames = []
        for index in encode_text(target):
            row = np.zeros(NUM_CLASSES, dtype=np.float32)
            row[index] = 0.9
            row[BLANK_INDEX] = 0.1
            frames.append(row)
        text, path, confidence = greedy_decode(np.stack(frames))
        self.assertEqual(text, target)
        self.assertEqual(len(path), len(target))
        self.assertAlmostEqual(confidence, 0.9, places=5)

    def test_greedy_decode_rejects_bad_shape(self) -> None:
        with self.assertRaises(ValueError):
            greedy_decode(np.zeros((3, NUM_CLASSES, 2), dtype=np.float32))

    def test_min_time_steps_counts_repeats(self) -> None:
        self.assertEqual(min_time_steps("京A12345"), 7)
        self.assertEqual(min_time_steps("京A11223"), 9)

    def test_labels_to_batch_pads_and_reports_lengths(self) -> None:
        matrix, lengths = labels_to_batch(["京A12345", "粤AD12345"])
        self.assertEqual(matrix.shape, (2, 8))
        self.assertEqual(lengths.tolist(), [7, 8])
        self.assertEqual(int(matrix[0, 7]), -1)

    def test_labels_to_batch_handles_empty_input(self) -> None:
        matrix, lengths = labels_to_batch([])
        self.assertEqual(matrix.shape, (0, 0))
        self.assertEqual(lengths.shape, (0,))

class ValidateLabelTests(unittest.TestCase):
    def test_legal_plates(self) -> None:
        self.assertEqual(validate_label("京A12345"), [])
        self.assertEqual(validate_label("粤AD12345"), [])

    def test_illegal_length(self) -> None:
        self.assertTrue(validate_label("京A1234"))

    def test_first_char_must_be_province(self) -> None:
        self.assertTrue(validate_label("XA12345"))


if __name__ == "__main__":
    unittest.main()

