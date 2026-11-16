from __future__ import annotations

from pathlib import Path

from .demo_components import DemoRecognizer, DemoRectifier, HeuristicPlateDetector
from .pipeline import PlatePipeline


def create_pipeline(
    backend: str = "demo",
    checkpoint: Path | str | None = None,
    manifest: Path | str | None = None,
    manual_box: tuple[int, int, int, int] | None = None,
) -> PlatePipeline:
    """按名称创建推理流水线"""
    if backend == "demo":
        return PlatePipeline(
            detector=HeuristicPlateDetector(),
            rectifier=DemoRectifier(),
            recognizer=DemoRecognizer(),
            backend_name="demo-heuristic",
        )

    if backend == "real":
        if checkpoint is None:
            raise ValueError("real 后端需要 checkpoint（端到端权重文件路径）")
        # 延迟导入：real_backend 会拉起 torch 与 models，而 models 又依赖 data 层，
        # 若在模块顶层导入会形成 data -> inference -> factory -> real_backend -> data 的循环。
        from .real_backend import create_real_pipeline

        pipeline, _info = create_real_pipeline(
            checkpoint_path=checkpoint,
            manifest_path=manifest,
            manual_box=manual_box,
        )
        return pipeline

    raise ValueError(
        f"未知后端 {backend!r}。可用后端：'demo'（启发式占位）、'real'（真实 STN + LPRNet）。"
    )
