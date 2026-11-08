from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class InferenceStatus(str, Enum):
    OK = "ok"
    NO_PLATE = "no_plate"
    ERROR = "error"


class RectificationMode(str, Enum):
    OFF = "off"
    AFFINE = "affine"
    PERSPECTIVE = "perspective"


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def clamp(self, image_width: int, image_height: int) -> "BoundingBox":
        x1 = min(max(self.x1, 0), image_width)
        y1 = min(max(self.y1, 0), image_height)
        x2 = min(max(self.x2, x1), image_width)
        y2 = min(max(self.y2, y1), image_height)
        return BoundingBox(x1, y1, x2, y2)


@dataclass(frozen=True)
class Detection:
    bbox: BoundingBox
    confidence: float
    plate_type: str = "unknown"
    is_fallback: bool = False


@dataclass(frozen=True)
class Recognition:
    text: str
    confidence: float


@dataclass(frozen=True)
class InferenceOptions:
    rectification: RectificationMode = RectificationMode.AFFINE
    confidence_threshold: float = 0.25
    plate_size: tuple[int, int] = (188, 48)
    max_detections: int = 3


@dataclass
class PlateResult:
    detection: Detection
    recognition: Recognition
    timing_ms: dict[str, float]
    raw_plate: Any = field(repr=False)
    rectified_plate: Any = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": asdict(self.detection.bbox),
            "detection_confidence": round(self.detection.confidence, 4),
            "plate_type": self.detection.plate_type,
            "is_fallback_detection": self.detection.is_fallback,
            "plate_text": self.recognition.text,
            "recognition_confidence": round(self.recognition.confidence, 4),
            "timing_ms": {k: round(v, 3) for k, v in self.timing_ms.items()},
        }


@dataclass
class InferenceResponse:
    status: InferenceStatus
    backend: str
    message: str
    timing_ms: dict[str, float]
    plates: list[PlateResult] = field(default_factory=list)
    annotated_image: Any | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "backend": self.backend,
            "message": self.message,
            "timing_ms": {k: round(v, 3) for k, v in self.timing_ms.items()},
            "plates": [plate.to_dict() for plate in self.plates],
        }
