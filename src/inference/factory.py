from __future__ import annotations

from .demo_components import DemoRecognizer, DemoRectifier, HeuristicPlateDetector
from .pipeline import PlatePipeline


def create_pipeline(backend: str = "demo") -> PlatePipeline:
    if backend != "demo":
        raise ValueError(
            f"未知后端 {backend!r}。请注册真实 YOLOv8、STN 与 LPRNet 适配器后再启用。"
        )
    return PlatePipeline(
        detector=HeuristicPlateDetector(),
        rectifier=DemoRectifier(),
        recognizer=DemoRecognizer(),
        backend_name="demo-heuristic",
    )
