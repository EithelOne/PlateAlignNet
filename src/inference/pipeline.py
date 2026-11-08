from __future__ import annotations

from time import perf_counter

from PIL import Image

from .contracts import Detector, Recognizer, Rectifier
from .schemas import (
    InferenceOptions,
    InferenceResponse,
    InferenceStatus,
    PlateResult,
)
from .visualization import draw_results


class PlatePipeline:
    def __init__(
        self,
        detector: Detector,
        rectifier: Rectifier,
        recognizer: Recognizer,
        backend_name: str,
    ) -> None:
        self.detector = detector
        self.rectifier = rectifier
        self.recognizer = recognizer
        self.backend_name = backend_name

    def run(self, image: Image.Image, options: InferenceOptions | None = None) -> InferenceResponse:
        options = options or InferenceOptions()
        total_started = perf_counter()
        try:
            source = image.convert("RGB")
            stage_started = perf_counter()
            detections = list(self.detector.detect(source, options.confidence_threshold))
            detect_ms = self._elapsed_ms(stage_started)
            detections = detections[: options.max_detections]
            if not detections:
                return InferenceResponse(
                    status=InferenceStatus.NO_PLATE,
                    backend=self.backend_name,
                    message="未检测到车牌。",
                    timing_ms={"detect": detect_ms, "total": self._elapsed_ms(total_started)},
                    annotated_image=source,
                )

            plates: list[PlateResult] = []
            for detection in detections:
                bbox = detection.bbox.clamp(*source.size)
                raw_plate = source.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))

                stage_started = perf_counter()
                rectified = self.rectifier.rectify(raw_plate, options.rectification, options.plate_size)
                rectify_ms = self._elapsed_ms(stage_started)

                stage_started = perf_counter()
                recognition = self.recognizer.recognize(rectified)
                recognize_ms = self._elapsed_ms(stage_started)
                plates.append(
                    PlateResult(
                        detection=detection,
                        recognition=recognition,
                        raw_plate=raw_plate,
                        rectified_plate=rectified,
                        timing_ms={
                            "rectify": rectify_ms,
                            "recognize": recognize_ms,
                        },
                    )
                )

            total_ms = self._elapsed_ms(total_started)
            annotated = draw_results(source, plates)
            return InferenceResponse(
                status=InferenceStatus.OK,
                backend=self.backend_name,
                message=f"完成 {len(plates)} 个车牌候选的演示推理。",
                timing_ms={"detect": detect_ms, "total": total_ms},
                plates=plates,
                annotated_image=annotated,
            )
        except Exception as exc:
            return InferenceResponse(
                status=InferenceStatus.ERROR,
                backend=self.backend_name,
                message=f"推理失败：{exc}",
                timing_ms={"total": self._elapsed_ms(total_started)},
            )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return (perf_counter() - started) * 1000
