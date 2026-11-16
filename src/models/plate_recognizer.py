"""STN+LPRNet组合模型：把【几何矫正】和【字符识别】串成一条可训练链路"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from src.inference.schemas import RectificationMode
from src.models.lprnet import LPRNet, decode_batch
from src.models.stn import STN, STN_INPUT_SIZE, STN_OUTPUT_SIZE, RectifyStats, count_parameters

@dataclass(frozen=True)
class RecognizerOutput:
    """一次前向的完整产物（训练用 logits，演示用 rectified）。"""

    logits: torch.Tensor
    rectified: torch.Tensor
    stats: RectifyStats

class PlateRecognizer(nn.Module):
    """车牌识别完整模型：几何矫正 + 序列识别。"""

    def __init__(
        self,
        mode: RectificationMode | str = RectificationMode.AFFINE,
        input_size: tuple[int, int] = STN_INPUT_SIZE,
        stn_hidden_dim: int = 64,
        freeze_stn: bool = False,
    ) -> None:
        super().__init__()
        self.mode = RectificationMode(mode)
        self.input_size = input_size
        self.stn = STN(
            self.mode,
            input_size=input_size,
            output_size=STN_OUTPUT_SIZE,
            hidden_dim=stn_hidden_dim,
        )
        self.lprnet = LPRNet()
        self.freeze_stn = freeze_stn
        if freeze_stn:
            self.set_stn_trainable(False)

    def set_stn_trainable(self, trainable: bool) -> None:
        """控制STN是否参与训练"""
        self.freeze_stn = not trainable
        if self.stn.localization is not None:
            for parameter in self.stn.localization.parameters():
                parameter.requires_grad_(trainable)


    def forward(
        self, images: torch.Tensor, return_details: bool = False
    ) -> torch.Tensor | RecognizerOutput:
        """前向：矫正 → 识别"""
        rectified, stats = self.stn(images, return_stats=True)
        logits = self.lprnet(rectified)
        if return_details:
            return RecognizerOutput(logits=logits, rectified=rectified, stats=stats)
        return logits

    @torch.no_grad()
    def recognize(self, images: torch.Tensor) -> list[str]:
        """推理：直接出车牌文本列表（演示/评估用）。"""
        self.eval()
        return decode_batch(self(images))

    def describe(self) -> dict[str, object]:
        """参数量明细，报告 §3.12 结果表要填的字段就来自这里。"""
        stn_parameters = count_parameters(self.stn)
        lprnet_parameters = count_parameters(self.lprnet)
        return {
            "mode": self.mode.value,
            "input_size": list(self.input_size),
            "output_size": list(STN_OUTPUT_SIZE),
            "stn_parameters": stn_parameters,
            "lprnet_parameters": lprnet_parameters,
            "total_parameters": stn_parameters + lprnet_parameters,
            "stn_trainable": not self.freeze_stn,
        }

def build_recognizer(
    mode: RectificationMode | str = RectificationMode.AFFINE,
    device: torch.device | None = None,
    freeze_stn: bool = False,
) -> PlateRecognizer:
    """创建组合模型，可选搬到指定设备。"""
    model = PlateRecognizer(mode=mode, freeze_stn=freeze_stn)
    if device is not None:
        model = model.to(device)
    return model

def edit_distance(prediction: str, target: str) -> int:
    """莱文斯坦编辑距离"""
    if prediction == target:
        return 0
    previous = list(range(len(target) + 1))
    for row, pred_char in enumerate(prediction, start=1):
        current = [row]
        for column, target_char in enumerate(target, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (pred_char != target_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """一批样本的字符错误率 = 总编辑距离 / 总真值字符数。"""
    if not targets:
        return 0.0
    total_errors = sum(edit_distance(p, t) for p, t in zip(predictions, targets))
    total_characters = sum(len(t) for t in targets)
    return total_errors / max(total_characters, 1)



def main() -> int:
    """命令行自检：三种模式的参数量与输出形状"""
    torch.manual_seed(0)
    images = torch.rand(2, 3, STN_INPUT_SIZE[1], STN_INPUT_SIZE[0]) * 2 - 1
    print(f"{'模式':<14}{'STN参数':>10}{'LPRNet参数':>12}{'合计':>10}  输出形状")
    print("-" * 70)
    for mode in (RectificationMode.OFF, RectificationMode.AFFINE, RectificationMode.PERSPECTIVE):
        model = build_recognizer(mode)
        model.eval()
        with torch.no_grad():
            details = model(images, return_details=True)
        info = model.describe()
        print(
            f"{info['mode']:<14}{info['stn_parameters']:>10,}{info['lprnet_parameters']:>12,}"
            f"{info['total_parameters']:>10,}  {tuple(details.logits.shape)}"
            f" / rectified {tuple(details.rectified.shape)}"
        )
    print()
    print("编辑距离自检：", edit_distance("皖A355U6", "皖A355U6"), edit_distance("皖A355U6", "皖A357U6"))
    print("CER 自检：", round(character_error_rate(["皖A355U6", "皖AL0000"], ["皖A355U6", "皖A10000"]), 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
