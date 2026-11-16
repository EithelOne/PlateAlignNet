"""真实模型推理后端（S7b）：把训练好的 STN + LPRNet 接进现有流水线"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from src.inference.pipeline import PlatePipeline
from src.inference.schemas import (
    BoundingBox,
    Detection,
    InferenceOptions,
    InferenceResponse,
    Recognition,
    RectificationMode,
)
from src.models.lprnet import decode_batch
from src.models.plate_recognizer import PlateRecognizer, build_recognizer
from src.models.stn import STN_INPUT_SIZE, STN_OUTPUT_SIZE, pil_to_tensor, tensor_to_pil


@dataclass(frozen=True)
class ModelInfo:
    """一个权重文件的自描述信息，界面顶部会显示它"""

    checkpoint: str
    mode: str
    stn_parameters: int
    lprnet_parameters: int
    total_parameters: int

    def as_dict(self) -> dict[str, object]:
        return {
            "权重": Path(self.checkpoint).name,
            "几何模式": self.mode,
            "STN 参数": self.stn_parameters,
            "LPRNet 参数": self.lprnet_parameters,
            "总参数": self.total_parameters,
        }


def load_recognizer(
    checkpoint_path: Path | str, device: torch.device | None = None
) -> tuple[PlateRecognizer, ModelInfo]:
    """加载端到端权重"""
    device = device or torch.device("cpu")
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    mode = str(payload.get("mode", "affine"))
    model = build_recognizer(mode, device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    description = model.describe()
    info = ModelInfo(
        checkpoint=str(checkpoint_path),
        mode=str(description["mode"]),
        stn_parameters=int(description["stn_parameters"]),
        lprnet_parameters=int(description["lprnet_parameters"]),
        total_parameters=int(description["total_parameters"]),
    )
    return model, info


class ManifestDetector:
    """用数据清单里的真值框充当检测结果"""

    def __init__(self, manifest_path: Path | str | None = None) -> None:
        self.boxes: dict[str, tuple[list[int], str]] = {}
        self.hint: str | None = None
        self.manifest_path = Path(manifest_path) if manifest_path else None
        if self.manifest_path and self.manifest_path.exists():
            with self.manifest_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    name = Path(str(row["image_path"])).name
                    self.boxes[name] = (list(row["bbox"]), str(row.get("plate_text", "")))

    def set_hint(self, image_name: str | None) -> None:
        """告诉检测器"这次要处理的图片叫什么"（界面在上传后调用）。"""
        self.hint = image_name

    def clear_hint(self) -> None:
        self.hint = None

    def detect(self, image: Image.Image, confidence_threshold: float) -> Sequence[Detection]:
        for key in (self.hint, Path(str(self.hint)).name if self.hint else None):
            if key and key in self.boxes:
                bbox_values, _text = self.boxes[key]
                box = BoundingBox(*bbox_values).clamp(*image.size)
                return [Detection(bbox=box, confidence=1.0, plate_type="ground-truth")]
        return []


class ManualBoxDetector:
    """手工指定一个矩形框（界面滑块给值），用于任意图片"""

    def __init__(self) -> None:
        self.box: tuple[int, int, int, int] | None = None

    def set_box(self, box: tuple[int, int, int, int] | None) -> None:
        self.box = box

    def detect(self, image: Image.Image, confidence_threshold: float) -> Sequence[Detection]:
        if self.box is None:
            return []
        x1, y1, x2, y2 = self.box
        box = BoundingBox(int(x1), int(y1), int(x2), int(y2)).clamp(*image.size)
        if box.width < 4 or box.height < 4:
            return []
        return [Detection(bbox=box, confidence=1.0, plate_type="manual", is_fallback=True)]


class STNRectifier:
    """实现 Rectifier 协议：裁剪图 → 188×48 → 模型内的 STN → 94×24 矫正图。"""

    def __init__(self, model: PlateRecognizer, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.last_stats = None

    def rectify(
        self,
        plate_image: Image.Image,
        mode: RectificationMode,
        output_size: tuple[int, int],
    ) -> Image.Image:
        """执行几何矫正"""
        tensor = pil_to_tensor(plate_image, size=STN_INPUT_SIZE).to(self.device)
        with torch.no_grad():
            warped, stats = self.model.stn(tensor, return_stats=True)
        self.last_stats = stats
        return tensor_to_pil(warped)


class TorchRecognizer:
    """实现 Recognizer 协议：94×24 矫正图 → LPRNet → 车牌文本 + 置信度。"""

    def __init__(self, model: PlateRecognizer, device: torch.device) -> None:
        self.model = model
        self.device = device

    def recognize(self, plate_image: Image.Image) -> Recognition:
        """识别车牌文本"""
        tensor = pil_to_tensor(plate_image, size=STN_OUTPUT_SIZE).to(self.device)
        with torch.no_grad():
            logits = self.model.lprnet(tensor)
        text = decode_batch(logits)[0]
        probabilities = logits.softmax(dim=2).max(dim=2).values
        confidence = float(probabilities.mean())
        return Recognition(text=text, confidence=confidence)


class RealPlatePipeline(PlatePipeline):
    """在原有流水线基础上，支持【告诉检测器这次处理的图片文件名】"""

    def run(  # type: ignore[override]
        self,
        image: Image.Image,
        options: InferenceOptions | None = None,
        image_name: str | None = None,
    ) -> InferenceResponse:
        detector = self.detector
        if hasattr(detector, "set_hint"):
            detector.set_hint(image_name)  # type: ignore[attr-defined]
        try:
            return super().run(image, options)
        finally:
            if hasattr(detector, "clear_hint"):
                detector.clear_hint()  # type: ignore[attr-defined]


def create_real_pipeline(
    checkpoint_path: Path | str,
    manifest_path: Path | str | None = None,
    manual_box: tuple[int, int, int, int] | None = None,
    device: torch.device | None = None,
) -> tuple[RealPlatePipeline, ModelInfo]:
    """构建真实后端流水线"""
    device = device or torch.device("cpu")
    model, info = load_recognizer(checkpoint_path, device)

    recognizer = TorchRecognizer(model, device)
    rectifier = STNRectifier(model, device)

    if manual_box is not None:
        detector: object = ManualBoxDetector()
        detector.set_box(manual_box)  # type: ignore[attr-defined]
        backend_name = "real-model/manual-box"
    elif manifest_path is not None:
        detector = ManifestDetector(manifest_path)
        backend_name = "real-model/ground-truth-box"
    else:
        raise ValueError("必须提供 manifest_path（真值框）或 manual_box（手工框）之一")

    pipeline = RealPlatePipeline(
        detector=detector,  # type: ignore[arg-type]
        rectifier=rectifier,  # type: ignore[arg-type]
        recognizer=recognizer,  # type: ignore[arg-type]
        backend_name=backend_name,
    )
    return pipeline, info


def main() -> int:
    """命令行冒烟测试：用真实权重 + 数据集里的一张图跑一遍完整推理"""
    import time

    from src.data.plate_dataset import PlateDataset

    root = Path(__file__).resolve().parents[2]
    checkpoint = root / "outputs" / "end2end_affine.pt"
    manifest = root / "data" / "ccpd_manifest_base" / "samples.jsonl"
    if not checkpoint.exists() or not manifest.exists():
        print(f"缺少 {checkpoint} 或 {manifest}，无法自检")
        return 1

    pipeline, info = create_real_pipeline(checkpoint, manifest)
    print(json.dumps(info.as_dict(), ensure_ascii=False, indent=2))

    dataset = PlateDataset.from_manifest(manifest, split="test")
    for index in range(3):
        record = dataset[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
        started = time.perf_counter()
        response = pipeline.run(image, image_name=record.image_path.name)
        elapsed = (time.perf_counter() - started) * 1000
        text = response.plates[0].recognition.text if response.plates else "-"
        mark = "OK " if text == record.text else "ERR"
        print(f"{mark} 真值={record.text:<10} 预测={text:<10} 状态={response.status.value} 耗时={elapsed:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
