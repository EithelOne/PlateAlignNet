"""LPRNet 训练入口：先做过拟合检查
    TODO:数据到位后再做正式训练
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.data.plate_dataset import NORM_MEAN, NORM_STD, PLATE_SIZE, PlateDataset, PlateRecord
from src.models.lprnet import build_model, compute_ctc_loss, decode_batch, to_ctc_targets
from src.models.lprnet_spec import trace_layers
from src.utils.plate_charset import BLANK_INDEX, CHARSET

def get_device() -> torch.device:
    """有 GPU 用 GPU，没有就用 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def records_to_tensors(
    records: Sequence[PlateRecord], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """一批样本 -> (images, targets, target_lengths) 三个张量"""
    images = np.stack([record.image for record in records])
    max_length = max(len(record.label_indices) for record in records)
    labels = np.full((len(records), max_length), -1, dtype=np.int64)
    for row, record in enumerate(records):
        labels[row, : len(record.label_indices)] = record.label_indices
    lengths = np.array([len(record.label_indices) for record in records], dtype=np.int64)

    targets, target_lengths = to_ctc_targets(labels, lengths)
    return (
        torch.from_numpy(images).to(device),
        targets.to(device),
        target_lengths.to(device),
    )

def evaluate(model: torch.nn.Module, images: torch.Tensor, texts: Sequence[str]) -> tuple[float, list[str]]:
    """在当前这批图上算整牌准确率"""
    model.eval()
    with torch.no_grad():
        logits = model(images)
        decoded = decode_batch(logits)
    correct = sum(1 for predicted, truth in zip(decoded, texts) if predicted == truth)
    return correct / len(texts), decoded

def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """保存权重，并把【解码所必需的信息】一起存进去"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "charset": list(CHARSET),
            "blank_index": BLANK_INDEX,
            "plate_size": list(PLATE_SIZE),
            "normalize": {"mean": NORM_MEAN, "std": NORM_STD},
            "time_steps": trace_layers().time_steps,
        },
        path,
    )

def train_overfit(
    manifest: Path,
    image_root: Path | None,
    count: int,
    epochs: int,
    learning_rate: float,
    print_every: int,
    device: torch.device,
) -> dict[str, object]:
    """在小样本上过拟合，验证整条训练链路是否正确"""

    dataset = PlateDataset.from_manifest(manifest, image_root=image_root, augment=False)
    take = min(count, len(dataset))
    if take < 2:
        raise SystemExit(f"样本太少（{take} 条），至少需要 2 条才能做过拟合检查")
    if take < count:
        print(f"提示：清单里只有 {len(dataset)} 条样本，实际使用 {take} 条")

    records = [dataset[index] for index in range(take)]
    texts = [record.text for record in records]
    images, targets, target_lengths = records_to_tensors(records, device)

    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(
        f"设备={device}  样本数={take}  images={tuple(images.shape)}  "
        f"T={trace_layers().time_steps}  C={trace_layers().num_classes}"
    )
    print(
        f"参数量={sum(p.numel() for p in model.parameters()):,}  "
        f"轮数={epochs}  学习率={learning_rate}"
    )

    started = time.perf_counter()
    history: list[dict[str, float]] = []
    passed = False
    final_loss = float("inf")
    final_accuracy = 0.0
    decoded: list[str] = []

    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(images)
        loss = compute_ctc_loss(logits, targets, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            final_accuracy, decoded = evaluate(model, images, texts)
            history.append(
                {"epoch": epoch, "loss": round(final_loss, 4), "accuracy": round(final_accuracy, 4)}
            )
            print(f"epoch {epoch:>4}  loss={final_loss:9.4f}  accuracy={final_accuracy:6.2%}")
            if final_accuracy >= 1.0 and final_loss < 0.05:
                print(f"epoch {epoch:>4}  已完全记住训练样本，提前结束")
                passed = True
                break

    final_accuracy, decoded = evaluate(model, images, texts)
    elapsed = time.perf_counter() - started

    checkpoint = Path("outputs") / "overfit_lprnet.pt"
    save_checkpoint(model, checkpoint)

    print("-" * 70)
    for truth, predicted in zip(texts, decoded):
        mark = "OK " if truth == predicted else "ERR"
        print(f"{mark} 真值={truth:<10} 预测={predicted}")
    print("-" * 70)
    print(f"最终损失={final_loss:.4f}  整牌准确率={final_accuracy:.2%}  用时={elapsed:.1f}s")
    print(f"权重已保存：{checkpoint}")
    if passed:
        print("结论：过拟合检查通过 ✅ 可以继续训练/接 STN 了")
    else:
        print("结论：未通过 ❌ 不要继续往下训。最常见原因是 CTC 输入形状或标签补位值不对")

    return {
        "passed": passed,
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_accuracy, 4),
        "epochs_run": history[-1]["epoch"] if history else 0,
        "elapsed_seconds": round(elapsed, 2),
        "checkpoint": str(checkpoint),
        "history": history,
        "samples": take,
    }

def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="LPRNet 识别器训练 / 过拟合检查")
    parser.add_argument("--manifest", type=Path, required=True, help="samples.jsonl 路径")
    parser.add_argument("--image-root", type=Path, default=None, help="清单里相对路径的根目录")
    parser.add_argument("--overfit", type=int, default=0, help="过拟合检查用的样本数（>0 才启用）")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = get_device()

    if args.overfit <= 0:
        parser.error(
            "当前只实现了过拟合检查：请加 --overfit 20。"
            "正式训练（--split train 全量 + 验证集）等真实数据到位后再启用。"
        )

    result = train_overfit(
        manifest=args.manifest,
        image_root=args.image_root,
        count=args.overfit,
        epochs=args.epochs,
        learning_rate=args.lr,
        print_every=args.print_every,
        device=device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
