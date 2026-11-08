from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .schemas import BoundingBox, Detection, Recognition, RectificationMode


class HeuristicPlateDetector:
    """Colour-based detector used only to keep the demo runnable without weights."""

    def __init__(self, allow_center_fallback: bool = True) -> None:
        self.allow_center_fallback = allow_center_fallback

    def detect(self, image: Image.Image, confidence_threshold: float) -> Sequence[Detection]:
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        blue_mask = (blue > 85) & (blue > red + 50) & (blue > green + 25)
        green_mask = (green > 80) & (green > red + 35) & (green > blue + 20)

        candidates = [
            self._from_mask(blue_mask, image.size, "blue"),
            self._from_mask(green_mask, image.size, "green"),
        ]
        detections = [item for item in candidates if item and item.confidence >= confidence_threshold]
        if detections:
            return sorted(detections, key=lambda item: item.confidence, reverse=True)
        if not self.allow_center_fallback:
            return []

        width, height = image.size
        box_width = max(32, int(width * 0.46))
        box_height = max(16, int(height * 0.16))
        x1 = (width - box_width) // 2
        y1 = int(height * 0.62)
        bbox = BoundingBox(x1, y1, x1 + box_width, min(height, y1 + box_height))
        return [Detection(bbox=bbox, confidence=0.25, is_fallback=True)]

    @staticmethod
    def _from_mask(mask: np.ndarray, image_size: tuple[int, int], plate_type: str) -> Detection | None:
        ys, xs = np.where(mask)
        if xs.size < max(30, int(mask.size * 0.0015)):
            return None
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        width, height = x2 - x1, y2 - y1
        if width < 24 or height < 8 or width / max(height, 1) < 1.8:
            return None

        image_width, image_height = image_size
        margin_x, margin_y = int(width * 0.05), int(height * 0.12)
        bbox = BoundingBox(x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y)
        bbox = bbox.clamp(image_width, image_height)
        density = float(mask[y1:y2, x1:x2].mean())
        confidence = min(0.92, 0.48 + density * 0.5)
        return Detection(bbox=bbox, confidence=confidence, plate_type=plate_type)


class DemoRectifier:
    """Normalises ROI size while reserving the STN replacement point."""

    def rectify(
        self,
        plate_image: Image.Image,
        mode: RectificationMode,
        output_size: tuple[int, int],
    ) -> Image.Image:
        if mode == RectificationMode.OFF:
            return plate_image.copy()
        method = Image.Resampling.BICUBIC if mode == RectificationMode.PERSPECTIVE else Image.Resampling.BILINEAR
        return ImageOps.fit(plate_image.convert("RGB"), output_size, method=method)


class DemoRecognizer:
    """Deterministic placeholder recognizer; never presented as a trained result."""

    _BLUE_PLATES = ("京A12345", "沪B6A521", "粤C8M726")
    _GREEN_PLATES = ("粤AD12345", "京AF02368", "沪BD51888")

    def recognize(self, plate_image: Image.Image) -> Recognition:
        sample = plate_image.convert("RGB").resize((16, 8))
        digest = hashlib.sha1(sample.tobytes()).digest()
        pixels = np.asarray(sample, dtype=np.float32)
        means = pixels.mean(axis=(0, 1))
        is_green = means[1] > means[2] * 1.05
        values = self._GREEN_PLATES if is_green else self._BLUE_PLATES
        return Recognition(text=values[digest[0] % len(values)], confidence=0.58)


def create_sample_image() -> Image.Image:
    image = Image.new("RGB", (960, 540), "#dce7ef")
    draw = ImageDraw.Draw(image)
    draw.rectangle((75, 210, 885, 470), fill="#26333d")
    draw.rounded_rectangle((210, 105, 750, 330), radius=65, fill="#697985")
    draw.rectangle((320, 340, 640, 430), fill="#1258a6", outline="white", width=5)
    draw.text((405, 372), "A 12345", fill="white", anchor="mm")
    draw.ellipse((145, 410, 275, 540), fill="#15191d")
    draw.ellipse((685, 410, 815, 540), fill="#15191d")
    return image
