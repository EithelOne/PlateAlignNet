from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .schemas import PlateResult


def draw_results(image: Image.Image, plates: list[PlateResult]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(canvas.size) / 180))
    font = _load_font(max(13, round(min(canvas.size) / 28)))
    for index, plate in enumerate(plates, start=1):
        box = plate.detection.bbox
        colour = "#26d980" if not plate.detection.is_fallback else "#ffbf3f"
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=colour, width=line_width)
        label = f"#{index} {plate.recognition.text}  {plate.detection.confidence:.0%}"
        label_box = draw.textbbox((box.x1, box.y1), label, font=font)
        label_height = label_box[3] - label_box[1] + 8
        top = max(0, box.y1 - label_height)
        draw.rectangle((box.x1, top, box.x1 + label_box[2] - label_box[0] + 10, box.y1), fill=colour)
        draw.text((box.x1 + 5, top + 3), label, fill="#07120c", font=font)
    return canvas


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()
