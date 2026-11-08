from __future__ import annotations

import unittest

from PIL import Image

from src.inference.demo_components import create_sample_image
from src.inference.factory import create_pipeline
from src.inference.schemas import BoundingBox, InferenceOptions, InferenceStatus, RectificationMode


class BoundingBoxTests(unittest.TestCase):
    def test_clamp_keeps_box_inside_image(self) -> None:
        box = BoundingBox(-3, -5, 120, 90).clamp(100, 80)
        self.assertEqual(box, BoundingBox(0, 0, 100, 80))


class PipelineTests(unittest.TestCase):
    def test_demo_sample_completes_pipeline(self) -> None:
        response = create_pipeline().run(create_sample_image())
        self.assertEqual(response.status, InferenceStatus.OK)
        self.assertTrue(response.plates)
        self.assertEqual(response.backend, "demo-heuristic")
        self.assertIsNotNone(response.annotated_image)

    def test_all_rectification_modes_are_supported(self) -> None:
        pipeline = create_pipeline()
        image = Image.new("RGB", (400, 240), "white")
        for mode in RectificationMode:
            response = pipeline.run(image, InferenceOptions(rectification=mode))
            self.assertNotEqual(response.status, InferenceStatus.ERROR, mode)


if __name__ == "__main__":
    unittest.main()
