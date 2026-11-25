"""空间变换网络STN：车牌几何校正"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.data.plate_dataset import NORM_MEAN, NORM_STD, prepare_plate_image
from src.inference.schemas import RectificationMode

# STN 输入/输出尺寸，统一用 (宽, 高)
STN_INPUT_SIZE: tuple[int, int] = (188, 48)
STN_OUTPUT_SIZE: tuple[int, int] = (94, 24)

ALIGN_CORNERS = False
PADDING_MODE = "zeros"
# 透视分母的绝对值下限：低于它就认为该点"不可采样"
MIN_DENOM = 0.1
# 一批样本里越界/非法点占比超过这个阈值，就整体回退成恒等变换
INVALID_LIMIT = 0.5

IDENTITY_PARAMS: dict[RectificationMode, tuple[float, ...]] = {
    RectificationMode.AFFINE: (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    RectificationMode.PERSPECTIVE: (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
}
NUM_PARAMS: dict[RectificationMode, int] = {
    RectificationMode.OFF: 0,
    RectificationMode.AFFINE: 6,
    RectificationMode.PERSPECTIVE: 8,
}

# 有界残差参数化：定位网络输出 raw，实际使用的 theta = identity + tanh(raw - identity) * bound。
# 这样 STN 在训练起点严格等于恒等，且无论如何都不可能把比例/平移推到塌缩区。
# bound 每个分量的含义（归一化坐标 [-1,1]）：
#   前 4 个 = 2x2 线性部分；非对角项（旋转/剪切）放宽到 ±0.80，允许约 ±53° 旋转，
#           以覆盖 CCPD rotate(10°~45°)/tilt(10°~60°) 的大角度倾斜；
#   中间 2 个 = 平移，±0.30；
#   透视 h31/h32 = 分母项，±0.25 保证分母不塌缩。
RESIDUAL_BOUNDS: dict[RectificationMode, tuple[float, ...]] = {
    RectificationMode.AFFINE: (0.35, 0.80, 0.30, 0.80, 0.35, 0.30),
    RectificationMode.PERSPECTIVE: (0.35, 0.80, 0.30, 0.80, 0.35, 0.30, 0.25, 0.25),
}

@dataclass(frozen=True)
class RectifyStats:
    """一次矫正的几何统计，用来监控"STN 是否在乱来"。"""

    mode: str
    invalid_fraction: float = 0.0
    out_of_range_fraction: float = 0.0
    fallback_fraction: float = 0.0


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """卷积 + ReLU（不接 BatchNorm，这样 batch=1 也能前向）。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
    )


def _conv_channels(depth: int) -> list[int]:
    """按卷积层数给出各层输出通道：16 / 32 / 64 / 64 / ..."""
    if depth < 1:
        raise ValueError(f"卷积层数应 >= 1，收到 {depth}")
    base = [16, 32, 64]
    if depth <= len(base):
        return base[:depth]
    return base + [64] * (depth - len(base))


class LocalizationNet(nn.Module):
    """定位网络：可配置层数的卷积（16/32/64/...）+ 池化 + 两层全连接。

    num_conv_blocks 控制卷积块数（E2 深度消融用）；hidden_dim 控制全连接宽度（E2 宽度消融用）。
    默认 3 层卷积 + 64 维隐藏层，与既有 E1 权重完全一致。
    """
    def __init__(
        self,
        num_params: int = 6,
        hidden_dim: int = 64,
        pool_size: tuple[int, int] = (2, 4),
        identity: tuple[float, ...] | None = None,
        residual_bounds: tuple[float, ...] | None = None,
        num_conv_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.num_conv_blocks = int(num_conv_blocks)
        channels = _conv_channels(self.num_conv_blocks)

        modules: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            modules.append(_conv_block(in_channels, out_channels))
            modules.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        self.features = nn.Sequential(*modules)
        final_channels = channels[-1]

        # 3 层卷积时特征图固定为 64×6×23：沿用「右侧补 1 列 + 固定平均池化」保持与既有 E1
        # 权重逐位一致（也兼容 MPS）。其它层数用自适应池化，因为特征图尺寸随层数变化。
        if self.num_conv_blocks == 3:
            self.pool: nn.Module = nn.Sequential(
                nn.ReflectionPad2d((0, 1, 0, 0)),  # 宽度 23 -> 24
                nn.AvgPool2d(kernel_size=(3, 6), stride=(3, 6)),
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
        flattened = final_channels * pool_size[0] * pool_size[1]
        self.fc1 = nn.Linear(flattened, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, num_params)
        self.num_params = num_params
        self.identity: tuple[float, ...] | None = None
        self.residual_bounds: tuple[float, ...] | None = None
        if residual_bounds is not None:
            if len(residual_bounds) != num_params:
                raise ValueError(
                    f"residual_bounds 应有 {num_params} 个，收到 {len(residual_bounds)} 个"
                )
            self.residual_bounds = tuple(float(b) for b in residual_bounds)
        self.reset_identity(identity)

    def reset_identity(self, identity: tuple[float, ...] | None = None) -> None:
        """恒等初始化：输出层权重清零、偏置设为恒等参数"""
        if identity is not None:
            if len(identity) != self.num_params:
                raise ValueError(f"恒等参数应有 {self.num_params} 个，收到 {len(identity)} 个")
            self.identity = tuple(float(v) for v in identity)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()
            if self.identity is not None:
                self.fc2.bias.copy_(
                    torch.tensor(self.identity, dtype=self.fc2.bias.dtype)
                )


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = self.pool(features).flatten(1)
        hidden = self.act(self.fc1(pooled))
        raw = self.fc2(hidden)
        if self.residual_bounds is not None and self.identity is not None:
            identity = torch.tensor(self.identity, dtype=raw.dtype, device=raw.device)
            bounds = torch.tensor(self.residual_bounds, dtype=raw.dtype, device=raw.device)
            raw = identity + torch.tanh(raw - identity) * bounds
        return raw


def identity_grid(
    batch_size: int,
    out_h: int,
    out_w: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """生成恒等变换的采样网格 (N, out_h, out_w, 2)。"""
    theta = (
        torch.tensor(IDENTITY_PARAMS[RectificationMode.AFFINE], dtype=dtype, device=device)
        .view(1, 2, 3)
        .repeat(batch_size, 1, 1)
    )
    return F.affine_grid(theta, size=(batch_size, 1, out_h, out_w), align_corners=ALIGN_CORNERS)

class STN(nn.Module):
    """可插拔的车牌几何矫正模块，支持关闭 / 仿射 / 透视三种模式。"""

    def __init__(
        self,
        mode: RectificationMode | str = RectificationMode.AFFINE,
        input_size: tuple[int, int] = STN_INPUT_SIZE,
        output_size: tuple[int, int] = STN_OUTPUT_SIZE,
        hidden_dim: int = 64,
        pool_size: tuple[int, int] = (2, 4),
        residual_bounds: tuple[float, ...] | None = None,
        num_conv_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.mode = RectificationMode(mode)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.localization: LocalizationNet | None = None
        if self.mode is not RectificationMode.OFF:
            count = NUM_PARAMS[self.mode]
            bounds = RESIDUAL_BOUNDS[self.mode] if residual_bounds is None else residual_bounds
            self.localization = LocalizationNet(
                num_params=count,
                hidden_dim=hidden_dim,
                pool_size=pool_size,
                identity=IDENTITY_PARAMS[self.mode],
                residual_bounds=bounds,
                num_conv_blocks=num_conv_blocks,
            )

    @property
    def output_hw(self) -> tuple[int, int]:
        """(高, 宽) 形式的输出尺寸。"""
        return (self.output_size[1], self.output_size[0])

    def forward(
        self, images: torch.Tensor, return_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, RectifyStats]:
        """执行矫正。

        输入：images (N, 3, 输入高, 输入宽)；return_stats 是否同时返回几何统计。
        输出：矫正后的图 (N, 3, 输出高, 输出宽)，可选附带 RectifyStats。
        """
        if images.dim() != 4:
            raise ValueError(f"images 需要四维 (N, C, H, W)，收到 {tuple(images.shape)}")
        height, width = int(images.shape[-2]), int(images.shape[-1])
        if (width, height) != self.input_size:
            raise ValueError(
                f"输入尺寸应为 (宽, 高)={self.input_size}，收到 {(width, height)}"
            )
        out_h, out_w = self.output_hw

        if self.mode is RectificationMode.OFF:
            warped = F.interpolate(
                images, size=(out_h, out_w), mode="bilinear", align_corners=ALIGN_CORNERS
            )
            stats = RectifyStats(mode=self.mode.value)
        elif self.mode is RectificationMode.AFFINE:
            warped, stats = self._warp_affine(images, out_h, out_w)
        else:
            warped, stats = self._warp_perspective(images, out_h, out_w)

        return (warped, stats) if return_stats else warped

    def _warp_affine(
        self, images: torch.Tensor, out_h: int, out_w: int
    ) -> tuple[torch.Tensor, RectifyStats]:
        """六参数仿射：theta (N, 2, 3) 直接交给 affine_grid。"""
        batch_size = images.shape[0]
        assert self.localization is not None
        theta = self.localization(images).view(batch_size, 2, 3)
        grid = F.affine_grid(
            theta,
            size=(batch_size, images.shape[1], out_h, out_w),
            align_corners=ALIGN_CORNERS,
        )
        out_of_range = float((grid.abs() > 1).any(dim=-1).float().mean().item())
        warped = F.grid_sample(
            images, grid, align_corners=ALIGN_CORNERS, padding_mode=PADDING_MODE
        )
        return warped, RectifyStats(self.mode.value, 0.0, out_of_range, 0.0)

    def _warp_perspective(
        self, images: torch.Tensor, out_h: int, out_w: int
    ) -> tuple[torch.Tensor, RectifyStats]:
        """八参数透视：手写网格，因为 affine_grid 不支持单应变换。

        逆向采样公式：
            d  = h31*x_t + h32*y_t + 1
            x_s = (h11*x_t + h12*y_t + h13) / d
            y_s = (h21*x_t + h22*y_t + h23) / d
        """
        batch_size = images.shape[0]
        assert self.localization is not None
        raw = self.localization(images)
        matrix = torch.zeros(batch_size, 3, 3, device=raw.device, dtype=raw.dtype)
        matrix[:, 0, :3] = raw[:, 0:3]
        matrix[:, 1, :3] = raw[:, 3:6]
        matrix[:, 2, 0:2] = raw[:, 6:8]
        matrix[:, 2, 2] = 1.0

        base = identity_grid(batch_size, out_h, out_w, device=raw.device, dtype=raw.dtype)
        ones = torch.ones_like(base[..., 0])
        coords = torch.stack([base[..., 0], base[..., 1], ones], dim=1)
        mapped = torch.einsum("nij,njhw->nihw", matrix, coords)

        denom = mapped[:, 2:3]
        invalid = denom.abs() < MIN_DENOM
        safe_denom = torch.where(invalid, torch.ones_like(denom), denom)
        grid = torch.cat([mapped[:, 0:1] / safe_denom, mapped[:, 1:2] / safe_denom], dim=1)
        grid = grid.permute(0, 2, 3, 1)

        invalid_per_sample = invalid.flatten(1).float().mean(dim=1)
        outside_per_sample = (grid.abs() > 1).any(dim=-1).flatten(1).float().mean(dim=1)
        fallback = (invalid_per_sample > INVALID_LIMIT) | (outside_per_sample > INVALID_LIMIT)
        if bool(fallback.any()):
            grid = torch.where(fallback.view(-1, 1, 1, 1), base, grid)

        warped = F.grid_sample(
            images, grid, align_corners=ALIGN_CORNERS, padding_mode=PADDING_MODE
        )
        stats = RectifyStats(
            mode=self.mode.value,
            invalid_fraction=float(invalid_per_sample.mean().item()),
            out_of_range_fraction=float(outside_per_sample.mean().item()),
            fallback_fraction=float(fallback.float().mean().item()),
        )
        return warped, stats

def count_parameters(model: nn.Module) -> int:
    """统计一个模块的参数量。"""
    return sum(parameter.numel() for parameter in model.parameters())


def pil_to_tensor(
    image: Image.Image, size: tuple[int, int] = STN_INPUT_SIZE
) -> torch.Tensor:
    """PTL图->STN输入张量（1,3,高,宽）,取值[-1,1]"""
    array = prepare_plate_image(image, size=size)
    return torch.from_numpy(array).unsqueeze(0)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """STN 输出张量 -> PIL 图（把 [-1,1] 反归一化回 0~255）。"""
    array = tensor.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = np.transpose(array, (1, 2, 0))
    array = array * NORM_STD + NORM_MEAN
    return Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))

def rectify_pil(
    stn: STN, image: Image.Image, return_stats: bool = False
) -> Image.Image | tuple[Image.Image, RectifyStats]:
    """给一张 PIL 车牌图做矫正，返回 PIL 图（演示/界面直接用这个）。"""
    tensor = pil_to_tensor(image, size=stn.input_size)
    result = stn(tensor, return_stats=return_stats)
    if return_stats:
        warped, stats = result  # type: ignore[misc]
        return tensor_to_pil(warped), stats
    return tensor_to_pil(result)  # type: ignore[arg-type]

def main() -> int:
    """命令行自检：恒等初始化、关闭 STN 的基线一致性、透视回退。"""
    torch.manual_seed(0)
    device = torch.device("cpu")

    print("=== 1) 参数量 ===")
    for mode in (RectificationMode.AFFINE, RectificationMode.PERSPECTIVE):
        model = STN(mode)
        print(f"{mode.value:<12} 定位网络参数量 = {count_parameters(model):,}")
    print("off          定位网络参数量 = 0（无 STN 基线）")

    print("\n=== 2) 恒等初始化：同尺寸下输出应约等于输入 ===")
    same_size_stn = STN(RectificationMode.AFFINE, output_size=STN_INPUT_SIZE)
    sample = torch.rand(2, 3, STN_INPUT_SIZE[1], STN_INPUT_SIZE[0]) * 2 - 1
    with torch.no_grad():
        identity_out = same_size_stn(sample)
    print(f"最大绝对误差 = {float((identity_out - sample).abs().max()):.3e}（应 < 1e-4）")
    print(f"初始参数 = {same_size_stn.localization.fc2.bias.tolist()}")

    print("\n=== 3) 关闭 STN 与双线性缩放基线完全一致 ===")
    off_stn = STN(RectificationMode.OFF)
    with torch.no_grad():
        off_out = off_stn(sample)
        baseline = F.interpolate(
            sample, size=off_stn.output_hw, mode="bilinear", align_corners=ALIGN_CORNERS
        )
    print(f"最大绝对误差 = {float((off_out - baseline).abs().max()):.3e}（应为 0）")

    print("\n=== 4) 透视：有界残差 vs 无界（触发回退） ===")
    perspective = STN(RectificationMode.PERSPECTIVE)
    with torch.no_grad():
        normal_out, normal_stats = perspective(sample, return_stats=True)
        print(f"恒等初始化 stats = {normal_stats}")
        extreme = (1.0, 0.0, 5.0, 0.0, 1.0, 5.0, 0.0, 0.0)
        perspective.localization.fc2.bias.copy_(torch.tensor(extreme))
        bounded_out, bounded_stats = perspective(sample, return_stats=True)
        print(f"极端平移（有界残差） stats = {bounded_stats}")
        print(f"有界残差下是否出现 nan：{bool(torch.isnan(bounded_out).any())}（应为 False）")

    # 无界 STN 才允许触发回退，用来验证安全网本身仍可用。
    unbounded = STN(RectificationMode.PERSPECTIVE, residual_bounds=(2.0,) * 8)
    with torch.no_grad():
        unbounded.localization.fc2.bias.copy_(torch.tensor(extreme))
        extreme_out, extreme_stats = unbounded(sample, return_stats=True)
        print(f"极端平移（无界） stats = {extreme_stats}")
        identity = identity_grid(2, *unbounded.output_hw)
        reference = F.grid_sample(
            sample, identity, align_corners=ALIGN_CORNERS, padding_mode=PADDING_MODE
        )
        print(f"回退后与恒等变换的最大误差 = {float((extreme_out - reference).abs().max()):.3e}（应 ≈ 0）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


