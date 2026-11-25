"""YOLOv8 车牌检测器：实现 Detector 协议，接入统一推理流水线。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from src.inference.schemas import BoundingBox, Detection


class YoloDetector:
    """用训练好的 YOLOv8 检测车牌，返回边界框 + 置信度。"""

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = str(weights_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.model = YOLO(self.weights_path)

    def detect(self, image: Image.Image, confidence_threshold: float) -> Sequence[Detection]:
        """检测车牌，返回按置信度降序的 Detection 列表。"""
        results = self.model.predict(
            source=image,
            conf=max(confidence_threshold, self.conf),
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                detections.append(
                    Detection(
                        bbox=BoundingBox(int(round(x1)), int(round(y1)),
                                        int(round(x2)), int(round(y2))),
                        confidence=conf,
                        plate_type="plate",
                        is_fallback=False,
                    )
                )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
