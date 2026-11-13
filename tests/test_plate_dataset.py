"""识别数据集的单元测试"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image



from src.data.plate_dataset import (
    PLATE_SIZE,
    PlateDataset,
    augment_plate,
    build_synthetic_dataset,
    crop_with_margin,
    labels_fit_time_steps,
    prepare_plate_image,
)
from src.inference.schemas import BoundingBox
from src.utils.plate_charset import encode_text, min_time_steps

class CropTests(unittest.TestCase):
    def test_crop_expands_by_margin(self) -> None:
        image = Image.new("RGB", (100, 100), "white")
        cropped = crop_with_margin(image, BoundingBox(40, 40, 60, 60), margin=(0.5, 0.5))
        self.assertEqual(cropped.size, (40, 40))

    def test_crop_clamps_to_image(self) -> None:
        image = Image.new("RGB", (100, 100), "white")
        cropped = crop_with_margin(image, BoundingBox(0, 0, 20, 20), margin=(1.0, 1.0))
        self.assertEqual(cropped.size, (40, 40))
        self.assertLessEqual(cropped.size[0], image.size[0])

    def test_crop_rejects_empty_box(self) -> None:
        image = Image.new("RGB", (100, 100), "white")
        with self.assertRaises(ValueError):
            crop_with_margin(image, BoundingBox(10, 10, 10, 20))

class PrepareImageTests(unittest.TestCase):
    def test_shape_dtype_and_range(self) -> None:
        plate = Image.new("RGB", (200, 70), "#0b5ba6")
        tensor = prepare_plate_image(plate)
        self.assertEqual(tensor.shape, (3, PLATE_SIZE[1], PLATE_SIZE[0]))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertGreaterEqual(float(tensor.min()), -1.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_deterministic_without_rng(self) -> None:
        plate = Image.new("RGB", (120, 40), "#0b5ba6")
        first = prepare_plate_image(plate)
        second = prepare_plate_image(plate)
        self.assertTrue(np.array_equal(first, second))

    def test_white_pixel_normalises_to_one(self) -> None:
        plate = Image.new("RGB", (40, 40), "white")
        tensor = prepare_plate_image(plate)
        self.assertAlmostEqual(float(tensor.max()), 1.0, places=5)

    def test_augment_keeps_size_and_changes_pixels(self) -> None:
        plate = Image.new("RGB", (120, 40), "#0b5ba6")
        rng = random.Random(1)
        augmented = augment_plate(plate, rng)
        self.assertEqual(augmented.size, plate.size)
        tensor_plain = prepare_plate_image(plate)
        tensor_aug = prepare_plate_image(augmented)
        self.assertFalse(np.array_equal(tensor_plain, tensor_aug))

class TimeStepTests(unittest.TestCase):
    def test_short_labels_fit(self) -> None:
        self.assertEqual(labels_fit_time_steps(["京A12345", "粤AD12345"], 24), [])

    def test_repeated_characters_need_more_steps(self) -> None:
        self.assertEqual(min_time_steps("京A112233"), 11)
        self.assertEqual(labels_fit_time_steps(["京A112233"], 11), [])
        self.assertEqual(labels_fit_time_steps(["京A112233"], 10), ["京A112233"])

class SyntheticDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temp.name)
        cls.manifest = build_synthetic_dataset(cls.root, count=5, seed=3)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_files_are_written(self) -> None:
        self.assertTrue(self.manifest.exists())
        images = list((self.root / "images").glob("*.jpg"))
        self.assertEqual(len(images), 5)
        lines = self.manifest.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 5)
        row = json.loads(lines[0])
        self.assertEqual(len(row["bbox"]), 4)
        self.assertIn(len(row["plate_text"]), (7, 8))

    def test_dataset_length_and_item(self) -> None:
        dataset = PlateDataset.from_manifest(self.manifest, image_root=self.root)
        self.assertEqual(len(dataset), 5)
        record = dataset[0]
        self.assertEqual(record.image.shape, (3, PLATE_SIZE[1], PLATE_SIZE[0]))
        self.assertEqual(record.image.dtype, np.float32)
        self.assertEqual(record.label_length, len(record.text))
        self.assertTrue(record.image_path.exists())
        self.assertEqual(record.subset, "synthetic")

    def test_indices_match_charset(self) -> None:
        dataset = PlateDataset.from_manifest(self.manifest, image_root=self.root)
        for index in range(len(dataset)):
            record = dataset[index]
            self.assertEqual(record.label_indices, tuple(encode_text(record.text)))

    def test_all_labels_fit_24_steps(self) -> None:
        dataset = PlateDataset.from_manifest(self.manifest, image_root=self.root)
        texts = [dataset[index].text for index in range(len(dataset))]
        self.assertEqual(labels_fit_time_steps(texts, 24), [])

    def test_iter_batches_shapes(self) -> None:
        dataset = PlateDataset.from_manifest(self.manifest, image_root=self.root)
        batches = list(dataset.iter_batches(batch_size=2))
        self.assertEqual(len(batches), 3)  # 5 条 = 2 + 2 + 1
        self.assertEqual(batches[0].images.shape, (2, 3, PLATE_SIZE[1], PLATE_SIZE[0]))
        self.assertEqual(batches[0].labels.shape[0], 2)
        self.assertEqual(batches[0].lengths.shape, (2,))
        self.assertEqual(batches[-1].batch_size, 1)

    def test_iter_batches_covers_all_samples(self) -> None:
        dataset = PlateDataset.from_manifest(self.manifest, image_root=self.root)
        expected = sorted(dataset[index].text for index in range(len(dataset)))
        collected = sorted(
            text for batch in dataset.iter_batches(batch_size=2) for text in batch.texts
        )
        self.assertEqual(collected, expected)

    def test_labels_are_padded_beyond_length(self) -> None:
        dataset = PlateDataset.from_manifest(self.manifest, image_root=self.root)
        for batch in dataset.iter_batches(batch_size=5):
            for row, length in enumerate(batch.lengths):
                self.assertTrue(np.all(batch.labels[row, length:] == -1))

    def test_shuffle_keeps_all_samples(self) -> None:
        dataset = PlateDataset.from_manifest(self.manifest, image_root=self.root)
        expected = sorted(dataset[index].text for index in range(len(dataset)))
        shuffled = sorted(
            text
            for batch in dataset.iter_batches(batch_size=2, shuffle=True)
            for text in batch.texts
        )
        self.assertEqual(shuffled, expected)

    def test_filter_by_subset_and_split(self) -> None:
        everything = PlateDataset.from_manifest(self.manifest, image_root=self.root)
        synthetic = PlateDataset.from_manifest(
            self.manifest, image_root=self.root, subset="synthetic"
        )
        self.assertEqual(len(synthetic), len(everything))

        train = PlateDataset.from_manifest(self.manifest, image_root=self.root, split="train")
        train_texts = {train[index].text for index in range(len(train))}
        all_train = {
            everything[index].text
            for index in range(len(everything))
            if everything[index].split == "train"
        }
        self.assertEqual(train_texts, all_train)

    def test_augmented_dataset_still_produces_tensors(self) -> None:
        dataset = PlateDataset.from_manifest(
            self.manifest, image_root=self.root, augment=True, seed=5
        )
        first = dataset[0].image
        second = dataset[0].image
        self.assertEqual(first.shape, second.shape)
        self.assertFalse(np.array_equal(first, second))

class ErrorTests(unittest.TestCase):
    def test_missing_image_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "samples.jsonl"
            row = {
                "image_path": "images/missing.jpg",
                "subset": "synthetic",
                "split": "train",
                "bbox": [1, 1, 30, 12],
                "label_indices": list(encode_text("京A12345")),
                "plate_text": "京A12345",
            }
            manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            dataset = PlateDataset.from_manifest(manifest, image_root=temp)
            with self.assertRaises(FileNotFoundError):
                _ = dataset[0]

    def test_empty_manifest_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "samples.jsonl"
            manifest.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                PlateDataset.from_manifest(manifest, image_root=temp)

    def test_bad_batch_size_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = build_synthetic_dataset(temp, count=2, seed=1)
            dataset = PlateDataset.from_manifest(manifest, image_root=temp)
            with self.assertRaises(ValueError):
                list(dataset.iter_batches(batch_size=0))


if __name__ == "__main__":
    unittest.main()



