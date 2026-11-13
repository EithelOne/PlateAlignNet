"""""LPRNet 结构规格：形状与参数量计算"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from src.utils.plate_charset import NUM_CLASSES, min_time_steps


# 输入形状 (通道, 高, 宽)，与 plate_dataset.PLATE_SIZE = (94, 24) 一致
INPUT_SHAPE: tuple[int, int, int] = (3, 24, 94)


@dataclass(frozen=True)
class LayerSpec:
    """一层网络的规格描述（卷积或池化）"""
    name: str
    kind: str
    in_channels: int = 0
    out_channels: int = 0
    kernel: int = 1
    stride: int = 1
    padding: int = 0
    batchnorm: bool = False

# 主干：5 个卷积 + 2 个池化，总下采样 4 倍（宽度 94 -> 47 -> 23）
LPRNET_LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec("conv1", "conv", in_channels=3, out_channels=32, kernel=3, padding=1, batchnorm=True),
    LayerSpec("pool1", "pool", kernel=2, stride=2),
    LayerSpec("conv2", "conv", in_channels=32, out_channels=32, kernel=3, padding=1, batchnorm=True),
    LayerSpec("conv3", "conv", in_channels=32, out_channels=64, kernel=3, padding=1, batchnorm=True),
    LayerSpec("pool2", "pool", kernel=2, stride=2),
    LayerSpec("conv4", "conv", in_channels=64, out_channels=64, kernel=3, padding=1, batchnorm=True),
    LayerSpec("conv5", "conv", in_channels=64, out_channels=64, kernel=3, padding=1, batchnorm=True),
)

# 输出头：1x1 卷积把通道数变成类别数，然后对高度方向取平均，得到 (C, W)
LPRNET_HEAD = LayerSpec(
    "head", "conv", in_channels=64, out_channels=NUM_CLASSES, kernel=1, padding=0, batchnorm=False
)

@dataclass(frozen=True)
class LayerReport:
    """一层网络算完之后的结果。"""

    name: str
    kind: str
    shape: tuple[int, int, int]
    parameters: int


@dataclass(frozen=True)
class SpecTrace:
    """整条网络的形状与参数量推算结果。"""

    reports: tuple[LayerReport, ...]
    total_parameters: int
    feature_shape: tuple[int, int, int]
    time_steps: int
    num_classes: int

    def label_shape(self, batch_size: int) -> tuple[int, int, int]:
        """CTC 需要的 log_probs 形状 (T, N, C)。"""
        return (self.time_steps, batch_size, self.num_classes)


def conv_output(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """卷积/池化后的输出边长。
        所有形状推算的原子操作；`(size + 2p - k) // s + 1` 这一个公式反复用。
    """
    if size <= 0:
        raise ValueError(f"输入边长必须为正，收到 {size}")
    if kernel <= 0 or stride <= 0:
        raise ValueError(f"核大小和步长必须为正，收到 kernel={kernel}, stride={stride}")
    result = (size + 2 * padding - kernel) // stride + 1
    if result <= 0:
        raise ValueError(
            f"输出边长为 {result}：size={size}, kernel={kernel}, stride={stride}, padding={padding}"
        )
    return result

def conv_parameters(
    in_channels: int, out_channels: int, kernel: int, batchnorm: bool = False
) -> int:
    """一个卷积层的参数量（含偏置，可选含 BatchNorm 的 2×通道数）。"""
    weights = in_channels * out_channels * kernel * kernel
    bias = out_channels
    norm = 2 * out_channels if batchnorm else 0
    return weights + bias + norm

def trace_layers(
    layers: Sequence[LayerSpec] = LPRNET_LAYERS,
    head: LayerSpec = LPRNET_HEAD,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
) -> SpecTrace:
    """逐层推算形状与参数量
       把【网络结构】变成【可检查的数字】 
    """
    channels, height, width = input_shape
    reports: list[LayerReport] = []
    total = 0

    for layer in layers:
        if layer.kind == "conv":
            if layer.in_channels != channels:
                raise ValueError(f"{layer.name} 输入通道 {layer.in_channels} 与上一层 {channels} 不符")
            height = conv_output(height, layer.kernel, layer.stride, layer.padding)
            width = conv_output(width, layer.kernel, layer.stride, layer.padding)
            channels = layer.out_channels
            parameters = conv_parameters(
                layer.in_channels, layer.out_channels, layer.kernel, layer.batchnorm
            )
        elif layer.kind == "pool":
            height = conv_output(height, layer.kernel, layer.stride, layer.padding)
            width = conv_output(width, layer.kernel, layer.stride, layer.padding)
            parameters = 0
        else:
            raise ValueError(f"未知层类型 {layer.kind!r}（只支持 conv / pool）")
        reports.append(LayerReport(layer.name, layer.kind, (channels, height, width), parameters))
        total += parameters

    if head.in_channels != channels:
        raise ValueError(f"{head.name} 输入通道 {head.in_channels} 与主干输出 {channels} 不符")
    height = conv_output(height, head.kernel, head.stride, head.padding)
    width = conv_output(width, head.kernel, head.stride, head.padding)
    channels = head.out_channels
    head_parameters = conv_parameters(head.in_channels, head.out_channels, head.kernel, head.batchnorm)
    reports.append(LayerReport(head.name, head.kind, (channels, height, width), head_parameters))
    total += head_parameters

    return SpecTrace(
        reports=tuple(reports),
        total_parameters=total,
        feature_shape=(channels, height, width),
        time_steps=width,
        num_classes=channels,
    )

def check_time_steps(texts: Sequence[str], time_steps: int) -> list[str]:
    """检查哪些标签装不进time_steps这个时间步"""
    return [text for text in texts if min_time_steps(text) > time_steps]

def describe(trace: SpecTrace, input_shape: tuple[int, int, int] = INPUT_SHAPE) -> str:
    """把推算结果整理成一张可读的层表。"""
    lines = [f"输入: C={input_shape[0]} H={input_shape[1]} W={input_shape[2]}"]
    for report in trace.reports:
        channels, height, width = report.shape
        lines.append(
            f"{report.name:<8} {report.kind:<5} -> C={channels:<4} H={height:<3} W={width:<3}"
            f" 参数={report.parameters}"
        )
    lines.append("-" * 60)
    lines.append(
        f"池化/卷积后特征图: C={trace.feature_shape[0]} H={trace.feature_shape[1]} W={trace.feature_shape[2]}"
    )
    lines.append(f"高度方向取平均后: (C={trace.num_classes}, T={trace.time_steps})")
    lines.append(f"CTC 需要的 log_probs 形状: (T={trace.time_steps}, N, C={trace.num_classes})")
    lines.append(f"总参数量: {trace.total_parameters:,}")
    return "\n".join(lines)

def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：打印层表、时间步数与容量检查"""
    parser = argparse.ArgumentParser(description="LPRNet 结构规格与容量自检")
    parser.add_argument(
        "--worst-case",
        action="store_true",
        help="额外打印 8 位车牌（含最多重复字符）的时间步需求",
    )
    args = parser.parse_args(argv)

    trace = trace_layers()
    print(describe(trace))

    samples = ["京A12345", "粤AD12345", "京A112233", "粤A88888"]
    too_long = check_time_steps(samples, trace.time_steps)
    print("-" * 60)
    print(f"检查样本: {samples}")
    print(f"装不下（需要更多时间步）的样本: {too_long if too_long else '无'}")
    if args.worst_case:
        worst = max(min_time_steps(text) for text in samples)
        print(f"最坏情况需要 {worst} 个时间步，当前 T={trace.time_steps}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())



