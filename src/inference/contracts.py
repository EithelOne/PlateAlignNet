from __future__ import annotations

from typing import Any, Protocol, Sequence

from .schemas import Detection, Recognition, RectificationMode


class Detector(Protocol):
    def detect(self, image: Any, confidence_threshold: float) -> Sequence[Detection]: ...


class Rectifier(Protocol):
    def rectify(
        self,
        plate_image: Any,
        mode: RectificationMode,
        output_size: tuple[int, int],
    ) -> Any: ...


class Recognizer(Protocol):
    def recognize(self, plate_image: Any) -> Recognition: ...
