"""STN 几何自监督拟合实验：验证"方向没搞反、而且真的能学回来"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.plate_dataset import render_plate_image
from src.inference.schemas import BoundingBox, RectificationMode
from src.models.stn import (
    ALIGN_CORNERS,
    PADDING_MODE,
    STN_INPUT_SIZE,
    STN,
    pil_to_tensor,
    tensor_to_pil,
)

def affine_matrix(
    angle_deg: float, scale: float = 1.0, translate: tuple[float, float] = (0.0, 0.0)
) -> torch.Tensor:
    """构造 2×3 仿射矩阵（旋转 + 等比缩放 + 平移）"""
    radians = math.radians(angle_deg)
    cos = math.cos(radians)
    sin = math.sin(radians)
    return torch.tensor(
        [
            [cos * scale, -sin * scale, translate[0]],
            [sin * scale, cos * scale, translate[1]],
        ],
        dtype=torch.float32,
    )

def theta_to_angle_scale(theta: torch.Tensor) -> tuple[float, float]:
    """从 2×3 仿射矩阵反解出旋转角（度）和面积缩放"""
    theta = theta.detach()
    a, b = float(theta[0, 0]), float(theta[0, 1])
    d, e = float(theta[1, 0]), float(theta[1, 1])
    angle = math.degrees(math.atan2(d, a))
    determinant = abs(a * e - b * d)
    return angle, math.sqrt(determinant)


def invert_affine(theta: torch.Tensor) -> torch.Tensor:
    """求 2×3 仿射矩阵的逆（补充成 3×3 后求逆再取前两行）。"""
    matrix = torch.eye(3, dtype=theta.dtype)
    matrix[:2] = theta
    return torch.linalg.inv(matrix)[:2]


def warp(
    images: torch.Tensor, theta: torch.Tensor, out_hw: tuple[int, int]
) -> torch.Tensor:
    """用给定 theta 对一批图做采样（造歪图用）。"""
    batch = images.shape[0]
    theta_batch = theta.unsqueeze(0).repeat(batch, 1, 1) if theta.dim() == 2 else theta
    grid = F.affine_grid(
        theta_batch,
        size=(batch, images.shape[1], out_hw[0], out_hw[1]),
        align_corners=ALIGN_CORNERS,
    )
    return F.grid_sample(images, grid, align_corners=ALIGN_CORNERS, padding_mode=PADDING_MODE)

def make_target_plate(
    text: str = "京A12345", size: tuple[int, int] = STN_INPUT_SIZE
) -> torch.Tensor:
    """画一张端正的车牌并转成 STN 输入张量 (1,3,H,W)"""
    width, height = size
    bbox = BoundingBox(2, 1, width - 2, height - 1)
    image: Image.Image = render_plate_image(bbox, text, canvas_size=(width, height))
    return pil_to_tensor(image, size=size)


def fit_stn(
    target: torch.Tensor,
    tilted: torch.Tensor,
    mode: RectificationMode = RectificationMode.AFFINE,
    steps: int = 300,
    learning_rate: float = 2e-3,
    seed: int = 17,
    grad_clip: float = 1.0,
) -> tuple[STN, list[float]]:
    """训练 STN 把 tilted 还原成 target"""
    torch.manual_seed(seed)
    stn = STN(mode, input_size=STN_INPUT_SIZE, output_size=STN_INPUT_SIZE)
    optimizer = torch.optim.Adam(stn.parameters(), lr=learning_rate)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.mse_loss(stn(tilted), target)
        loss.backward()
        # 梯度裁剪：STN 一旦把位置参数推过头，采样坐标会飞出画面，
        # 损失变成一大片黑边，梯度随之爆炸。这是训练 STN 必须加的保险。
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(stn.parameters(), grad_clip)
        optimizer.step()
        history.append(float(loss.item()))
    return stn, history

def _learned_affine(stn: STN, tilted: torch.Tensor) -> torch.Tensor:
    """取出网络学到的 2×3 仿射矩阵"""
    assert stn.localization is not None
    with torch.no_grad():
        raw = stn.localization(tilted)
    if raw.shape[1] == 6:
        return raw.view(1, 2, 3)[0].clone()

    matrix = torch.zeros(3, 3, dtype=raw.dtype)
    matrix[0] = raw[0, 0:3]
    matrix[1] = raw[0, 3:6]
    matrix[2, 0:2] = raw[0, 6:8]
    matrix[2, 2] = 1.0
    denominator = float(matrix[2, 2])
    if abs(denominator) > 1e-8:
        matrix = matrix / denominator
    return matrix[:2].clone()


def run_case(
    name: str,
    applied_theta: torch.Tensor,
    target: torch.Tensor,
    mode: RectificationMode = RectificationMode.AFFINE,
    steps: int = 300,
    learning_rate: float = 2e-3,
    save_dir: Path | None = None,
) -> dict[str, object]:
    """跑一个案例：造歪图 → 训练 → 比较学到的矩阵与理论逆矩阵"""
    tilted = warp(target, applied_theta, (STN_INPUT_SIZE[1], STN_INPUT_SIZE[0]))
    stn, history = fit_stn(target, tilted, mode=mode, steps=steps, learning_rate=learning_rate)
    with torch.no_grad():
        rectified = stn(tilted)

    mse_before = float(F.mse_loss(tilted, target))
    mse_after = float(F.mse_loss(rectified, target))
    learned = _learned_affine(stn, tilted)
    expected = invert_affine(applied_theta)

    applied_angle, applied_scale = theta_to_angle_scale(applied_theta)
    learned_angle, learned_scale = theta_to_angle_scale(learned)
    expected_angle, expected_scale = theta_to_angle_scale(expected)
    angle_error = abs(learned_angle - expected_angle)
    scale_error = abs(learned_scale - expected_scale) / max(expected_scale, 1e-6)

    # 判定标准（注意：MSE 只能"明显下降"，不能要求降到 0）：
    #   转歪/缩放时有一部分像素被移出画面，那部分信息已经丢了，
    #   任何变换都拿不回来，所以残差有下限。真正要看的是"几何参数学对没有"。
    passed = (
        angle_error < max(1.5, abs(expected_angle) * 0.1)
        and scale_error < 0.08
        and mse_after < mse_before
    )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        tensor_to_pil(target).save(save_dir / f"{name}_1_target.png")
        tensor_to_pil(tilted).save(save_dir / f"{name}_2_tilted.png")
        tensor_to_pil(rectified).save(save_dir / f"{name}_3_rectified.png")

    return {
        "case": name,
        "applied_angle": round(applied_angle, 2),
        "learned_angle": round(learned_angle, 2),
        "expected_angle": round(expected_angle, 2),
        "angle_error": round(angle_error, 2),
        "applied_scale": round(applied_scale, 3),
        "learned_scale": round(learned_scale, 3),
        "expected_scale": round(expected_scale, 3),
        "scale_error": round(scale_error, 3),
        "mse_before": round(mse_before, 5),
        "mse_after": round(mse_after, 5),
        "first_loss": round(history[0], 5),
        "last_loss": round(history[-1], 5),
        "passed": passed,
    }

def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：跑旋转/缩放/平移/透视四个案例。"""
    parser = argparse.ArgumentParser(description="STN 几何自拟合实验（验证方向与可学性）")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--text", type=str, default="京A12345")
    parser.add_argument("--save", action="store_true", help="把目标/歪图/矫正图存到 outputs/stn_fit")
    args = parser.parse_args(argv)

    target = make_target_plate(args.text)
    save_dir = Path("outputs") / "stn_fit" if args.save else None

    results = [
        run_case(
            "rot+12",
            affine_matrix(12.0),
            target,
            steps=args.steps,
            learning_rate=args.lr,
            save_dir=save_dir,
        ),
        run_case(
            "rot-8_scale0.9",
            affine_matrix(-8.0, scale=0.9),
            target,
            steps=args.steps,
            learning_rate=args.lr,
            save_dir=save_dir,
        ),
        run_case(
            "shift+0.15",
            affine_matrix(0.0, translate=(0.15, -0.05)),
            target,
            steps=args.steps,
            learning_rate=args.lr,
            save_dir=save_dir,
        ),
        run_case(
            "perspective+rot6",
            affine_matrix(6.0),
            target,
            mode=RectificationMode.PERSPECTIVE,
            steps=args.steps,
            learning_rate=args.lr,
            save_dir=save_dir,
        ),
    ]

    header = (
        f"{'案例':<18}{'施加角度':>9}{'学到角度':>9}{'理论角度':>9}{'角度误差':>9}"
        f"{'学到缩放':>9}{'理论缩放':>9}{'MSE前':>9}{'MSE后':>9}{'结论':>6}"
    )
    print(header)
    print("-" * 24 + "-" * (len(header) - 24))
    for row in results:
        verdict = "OK" if row["passed"] else "FAIL"
        print(
            f"{row['case']:<18}{row['applied_angle']:>9.2f}{row['learned_angle']:>9.2f}"
            f"{row['expected_angle']:>9.2f}{row['angle_error']:>9.2f}"
            f"{row['learned_scale']:>9.3f}{row['expected_scale']:>9.3f}"
            f"{row['mse_before']:>9.4f}{row['mse_after']:>9.4f}{verdict:>6}"
        )

    print()
    print("解读：")
    print("  · 「施加角度」是我们故意转歪的角度，「理论角度」= 它的逆向（符号相反）")
    print("  · 「学到角度」越接近「理论角度」，说明 STN 学对了方向")
    print("  · 若「学到角度」≈ 施加角度的两倍甚至同号，说明逆向采样方向搞反了")
    print("  · MSE 只会明显下降、不会到 0：转出画面的那部分像素信息已经丢了，任何变换都拿不回来")
    if save_dir is not None:
        print(f"\n对照图已保存到 {save_dir}（每例 3 张：目标 / 歪图 / 矫正后）")

    print()
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(bool(row["passed"]) for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())












