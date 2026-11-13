"""CCPD 文件名解析与数据清单的单元测试。"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from src.data.ccpd_parser import (
    ParseError,
    Point,
    assign_split,
    build_manifest,
    build_sample,
    check_sample,
    collect_samples,
    make_synthetic_filename,
    parse_ccpd_filename,
    subset_from_path,
    to_tl_tr_br_bl,
    validate_quad,
)

# 一条格式完整的样例：京A12345，字符索引 12_35_60_61_62_63_64
SAMPLE_NAME = (
    "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-12_35_60_61_62_63_64-37-15.jpg"
)
# 新能源 8 位：粤AD12345
GREEN_NAME = (
    "050-0_0-154&383_386&473-386&473_177&454_154&383_363&402-19_35_38_60_61_62_63_64-100-5.jpg"
)


class ParseFilenameTests(unittest.TestCase):
    def test_basic_fields(self) -> None:
        fields = parse_ccpd_filename(SAMPLE_NAME)
        self.assertEqual(fields["area_ratio"], 25)
        self.assertEqual(fields["tilt"], (95, 113))
        self.assertEqual(fields["bbox"].x1, 154)
        self.assertEqual(fields["bbox"].y1, 383)
        self.assertEqual(fields["bbox"].x2, 386)
        self.assertEqual(fields["bbox"].y2, 473)
        self.assertEqual(fields["brightness"], 37)
        self.assertEqual(fields["blurriness"], 15)

    def test_plate_text_is_decoded(self) -> None:
        self.assertEqual(parse_ccpd_filename(SAMPLE_NAME)["plate_text"], "京A12345")
        self.assertEqual(parse_ccpd_filename(GREEN_NAME)["plate_text"], "粤AD12345")

    def test_corners_are_normalised(self) -> None:
        corners = parse_ccpd_filename(SAMPLE_NAME)["corners"]
        self.assertEqual(corners, (Point(154, 383), Point(363, 402),
                                   Point(386, 473), Point(177, 454)))

    def test_wrong_field_count_raises(self) -> None:
        with self.assertRaises(ParseError):
            parse_ccpd_filename("025-95_113-154&383_386&473-12_35_60_61_62_63_64-37-15.jpg")

    def test_bad_suffix_raises(self) -> None:
        with self.assertRaises(ParseError):
            parse_ccpd_filename(SAMPLE_NAME.replace(".jpg", ".bmp"))

    def test_index_out_of_range_raises(self) -> None:
        broken = SAMPLE_NAME.replace("12_35_60_61_62_63_64", "12_35_99_60_61_62_63")
        with self.assertRaises(ParseError):
            parse_ccpd_filename(broken)

    def test_inverted_bbox_raises(self) -> None:
        broken = SAMPLE_NAME.replace("154&383_386&473", "386&473_154&383")
        with self.assertRaises(ParseError):
            parse_ccpd_filename(broken)

    def test_plate_length_must_be_7_or_8(self) -> None:
        broken = SAMPLE_NAME.replace("12_35_60_61_62_63_64", "12_35_59_60_61_62")
        with self.assertRaises(ParseError):
            parse_ccpd_filename(broken)


class GeometryTests(unittest.TestCase):
    def test_reorder_maps_raw_order_to_standard(self) -> None:
        raw = (Point(386, 473), Point(177, 454), Point(154, 383), Point(363, 402))
        self.assertEqual(to_tl_tr_br_bl(raw), (Point(154, 383), Point(363, 402),
                                               Point(386, 473), Point(177, 454)))

    def test_validate_quad_accepts_normal_quad(self) -> None:
        good = (Point(154, 383), Point(363, 402), Point(386, 473), Point(177, 454))
        self.assertEqual(validate_quad(good), [])

    def test_validate_quad_detects_wrong_order(self) -> None:
        """如果原始角点顺序假设错了，这里必须报出来，而不是悄悄算出错误的单应矩阵。"""
        wrong = (Point(154, 383), Point(177, 454), Point(386, 473), Point(363, 402))
        self.assertTrue(validate_quad(wrong))

    def test_validate_quad_detects_degenerate(self) -> None:
        flat = (Point(0, 0), Point(10, 0), Point(20, 0), Point(5, 0))
        self.assertTrue(validate_quad(flat))


class SampleCheckTests(unittest.TestCase):
    def test_good_sample_passes(self) -> None:
        sample = build_sample(SAMPLE_NAME, Path("ccpd_base") / SAMPLE_NAME)
        self.assertEqual(check_sample(sample), [])
        self.assertEqual(sample.plate_text, "京A12345")
        self.assertEqual(sample.subset, "base")
        self.assertIn(sample.split, {"train", "val", "test"})

    def test_corner_bbox_mismatch_is_reported(self) -> None:
        """四角点整体偏移很多时，外接矩形对不上边界框，必须报问题。"""
        shifted = SAMPLE_NAME.replace(
            "386&473_177&454_154&383_363&402", "500&600_300&600_300&500_500&500"
        )
        sample = build_sample(shifted, Path("ccpd_base") / shifted)
        self.assertTrue(check_sample(sample))

    def test_subset_from_path(self) -> None:
        self.assertEqual(subset_from_path(Path("data/CCPD2019/ccpd_tilt/1.jpg")), "tilt")
        self.assertEqual(subset_from_path(Path("data/misc/1.jpg")), "unknown")


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic(self) -> None:
        path = Path("data/CCPD2019/ccpd_base/025-95_113-x.jpg")
        self.assertEqual(assign_split(path), assign_split(path))

    def test_split_covers_all_three(self) -> None:
        splits = {assign_split(Path(f"data/img_{i}.jpg")) for i in range(300)}
        self.assertEqual(splits, {"train", "val", "test"})

    def test_split_ratio_is_roughly_8_1_1(self) -> None:
        paths = [Path(f"data/img_{i}.jpg") for i in range(4000)]
        train_ratio = sum(assign_split(p) == "train" for p in paths) / len(paths)
        self.assertAlmostEqual(train_ratio, 0.8, delta=0.03)


class ManifestTests(unittest.TestCase):
    def test_collect_and_write_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "ccpd_base").mkdir()
            (root / "ccpd_tilt").mkdir()
            (root / "ccpd_base" / SAMPLE_NAME).write_bytes(b"")
            (root / "ccpd_tilt" / GREEN_NAME).write_bytes(b"")
            (root / "ccpd_tilt" / "broken-name.jpg").write_bytes(b"")

            samples, issues = collect_samples(root)
            self.assertEqual(len(samples), 2)
            self.assertEqual(len(issues), 1)
            self.assertEqual({s.subset for s in samples}, {"base", "tilt"})

            out = root / "manifest"
            summary = build_manifest(samples, issues, out, root_label="temp")
            self.assertEqual(summary["total_samples"], 2)
            self.assertEqual(summary["total_issues"], 1)
            self.assertEqual(summary["by_plate_length"], {"7": 1, "8": 1})
            self.assertEqual(sum(summary["by_split"].values()), 2)

            lines = (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["plate_text"], "京A12345")
            self.assertEqual(len(first["corners"]), 4)
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "issues.jsonl").exists())


class SyntheticTests(unittest.TestCase):
    def test_synthetic_filenames_are_parsable(self) -> None:
        """没有数据集时，用 200 条合成文件名验证整条解析 + 检查链路。"""
        rng = random.Random(7)
        for index in range(200):
            name = make_synthetic_filename(rng)
            sample = build_sample(name, Path("synthetic") / f"case{index}.jpg", subset="synthetic")
            self.assertEqual(check_sample(sample), [], msg=f"{name} -> {sample.to_dict()}")
            self.assertIn(len(sample.plate_text), (7, 8))


if __name__ == "__main__":
    unittest.main()