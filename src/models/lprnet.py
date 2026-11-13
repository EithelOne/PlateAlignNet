"""LPRNet网络实现（PyTorch 版，逐层照 lprnet_spec 的层表写）"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.lprnet_spec import (
    INPUT_SHAPE,
    LPRNET_HEAD,
    LPRNET_LAYERS,
    LayerSpec,
    trace_layers,
)
from src.utils.plate_charset import BLANK_INDEX, ctc_collapse

def build_backbone(layers: tuple[LayerSpec, ...] = LPRNET_LAYERS) -> nn.Sequential:
    """按规格列表拼出猪肝网络
       把【数据形式的网络结构】变成真正的PyTorch模块，保证实现和规格表一致 
    """
    modules: list[nn.Module] = []
    for spec in layers:
        if spec.kind == "conv":
            modules.append(
                nn.Conv2d(
                    spec.in_channels,
                    spec.out_channels,
                    kernel_size=spec.kernel,
                    stride=spec.stride,
                    padding=spec.padding,
                )
            )
            if spec.batchnorm:
                modules.append(nn.BatchNorm2d(spec.out_channels))
            modules.append(nn.ReLU(inplace=True))
        elif spec.kind == "pool":
            modules.append(
                nn.MaxPool2d(kernel_size=spec.kernel, stride=spec.stride, padding=spec.padding)
            )
        else:
            raise ValueError(f"未知层类型 {spec.kind!r}（只支持 conv / pool）")
    return nn.Sequential(*modules)

class LPRNet(nn.Module):
    """轻量全卷积车牌识别网络：输出沿宽度方向的字符概率序列。"""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = build_backbone()
        self.head = nn.Conv2d(
            LPRNET_HEAD.in_channels,
            LPRNET_HEAD.out_channels,
            kernel_size=LPRNET_HEAD.kernel,
            stride=LPRNET_HEAD.stride,
            padding=LPRNET_HEAD.padding,
        )
        self.num_classes = LPRNET_HEAD.out_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """输入 (N, 3, 24, 94)，输出 logits (T, N, C)。"""
        features = self.backbone(images)
        logits = self.head(features)
        squeezed = logits.mean(dim=2)
        return squeezed.permute(2, 0, 1)


def build_model(device: torch.device | None = None) -> LPRNet:
    """创建模型，可选直接搬到指定设备。"""
    model = LPRNet()
    if device is not None:
        model = model.to(device)
    return model


def log_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """logits (T, N, C) -> log 概率 (T, N, C)，在类别维归一化    """
    return logits.log_softmax(dim=2)

def to_ctc_targets(
    labels: np.ndarray, lengths: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """把数据集的标签矩阵转成CTCLoss要求的张量
       CTCLoss 要求目标值都在 [0, C-1] 内， 所以补位的 -1 必须先换成合法值
          （这里换成 0）。超出 target_lengths 的位置不会被读取，换什么都行
    """
    padded = np.where(labels < 0, 0, labels).astype(np.int64)
    return torch.from_numpy(padded), torch.from_numpy(lengths.astype(np.int64))

def ctc_loss_from_log_probs(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_index: int = BLANK_INDEX,
) -> torch.Tensor:
    """直接用log概率算CTC损失"""
    time_steps, batch_size, _ = log_probs.shape
    input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=log_probs.device)
    loss_fn = nn.CTCLoss(blank=blank_index, reduction="mean", zero_infinity=True)
    return loss_fn(
        log_probs,
        targets.to(log_probs.device),
        input_lengths,
        target_lengths.to(log_probs.device),
    )

def compute_ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_index: int = BLANK_INDEX,
) -> torch.Tensor:
    """"训练用的损失：logits->log_softmax=>CTCLoss"""
    return ctc_loss_from_log_probs(
        log_probs_from_logits(logits), targets, target_lengths, blank_index
    )

def decode_batch(logits: torch.Tensor, blank_index: int = BLANK_INDEX) -> list[str]:
    """"把logits解码成文本（贪心解码）"""
    path = logits.detach().argmax(dim=2).cpu().numpy()
    return [
        ctc_collapse([int(value) for value in path[:, column]], blank_index)
        for column in range(path.shape[1])
    ]


def spec_consistency(device: torch.device | None = None) -> dict[str, object]:
    """运行时校验：实际网络的输出形状/参数量必须与规格表一致
       这是【规格表 <-> 实现】的锚点。改了网络却忘了同步规格表时，这里会报 False 
    """
    trace = trace_layers()
    model = build_model(device)
    model.eval()
    dummy = torch.zeros(2, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], device=device or "cpu")
    with torch.no_grad():
        output = model(dummy)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "expected_time_steps": trace.time_steps,
        "actual_time_steps": int(output.shape[0]),
        "expected_classes": trace.num_classes,
        "actual_classes": int(output.shape[2]),
        "expected_parameters": trace.total_parameters,
        "actual_parameters": parameters,
        "consistent": (
            int(output.shape[0]) == trace.time_steps
            and int(output.shape[2]) == trace.num_classes
            and parameters == trace.total_parameters
        ),
    }

def main() -> int:
    """命令行自检：打印规格一致性检查结果。"""
    info = spec_consistency()
    for key, value in info.items():
        print(f"{key}: {value}")
    return 0 if info["consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())











