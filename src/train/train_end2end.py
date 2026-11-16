"""端到端训练：在真实CCPD数据上联合优化STN+LPRNet"""


from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.data.plate_dataset import NORM_MEAN, NORM_STD, PlateDataset
from src.inference.schemas import RectificationMode
from src.models.lprnet import compute_ctc_loss, decode_batch, to_ctc_targets
from src.models.plate_recognizer import build_recognizer, character_error_rate
from src.models.stn import STN_INPUT_SIZE, STN_OUTPUT_SIZE
from src.train.train_recognizer import get_device
from src.utils.plate_charset import BLANK_INDEX, CHARSET

def batch_to_tensors(
    batch: object, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 PlateDataset.iter_batches 产出的 Batch 变成三个 torch 张量。

    输入：Batch（含 images / labels / lengths）。
    输出：(images, targets, target_lengths)，都在指定设备上。
    """
    images = torch.from_numpy(batch.images).to(device)  # type: ignore[attr-defined]
    targets, lengths = to_ctc_targets(batch.labels, batch.lengths)  # type: ignore[attr-defined]
    return images, targets.to(device), lengths.to(device)

def train_one_epoch(
    model: torch.nn.Module,
    dataset: PlateDataset,
    device: torch.device,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = 5.0,
) -> dict[str, float]:
    """跑一个epoch的训练"""
    model.train()
    total_loss = 0.0
    total_fallback = 0.0
    steps = 0
    for batch in dataset.iter_batches(batch_size=batch_size, shuffle=True):
        images, targets, lengths = batch_to_tensors(batch, device)
        output = model(images, return_details=True)
        loss = compute_ctc_loss(output.logits, targets, lengths)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        total_fallback += float(output.stats.fallback_fraction)
        steps += 1
    return {
        "loss": total_loss / max(steps, 1),
        "steps": float(steps),
        "fallback": total_fallback / max(steps, 1),
    }

@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module, dataset: PlateDataset, device: torch.device, batch_size: int = 32
) -> dict[str, object]:
    """在某个数据集上评估：整牌准确率+字符错误率+回退比例"""
    model.eval()
    predictions: list[str] = []
    truths: list[str] = []
    fallbacks: list[float] = []
    for start in range(0, len(dataset), batch_size):
        indices = range(start, min(start + batch_size, len(dataset)))
        records = [dataset[index] for index in indices]
        images = torch.from_numpy(np.stack([record.image for record in records])).to(device)
        output = model(images, return_details=True)
        predictions.extend(decode_batch(output.logits))
        fallbacks.append(float(output.stats.fallback_fraction))
        truths.extend(record.text for record in records)

    correct = sum(1 for prediction, truth in zip(predictions, truths) if prediction == truth)
    examples = [
        {"truth": truth, "prediction": prediction}
        for truth, prediction in list(zip(truths, predictions))[:8]
    ]
    return {
        "accuracy": correct / max(len(truths), 1),
        "cer": character_error_rate(predictions, truths),
        "fallback": sum(fallbacks) / max(len(fallbacks), 1),
        "samples": len(truths),
        "examples": examples,
    }

def save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    mode: str,
    extra: dict[str, object] | None = None,
) -> None:
    """保存权重 + 解码必需的全部信息（报告 §2.1 的要求）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "state_dict": model.state_dict(),
        "mode": mode,
        "charset": list(CHARSET),
        "blank_index": BLANK_INDEX,
        "stn_input_size": list(STN_INPUT_SIZE),
        "lprnet_input_size": list(STN_OUTPUT_SIZE),
        "normalize": {"mean": NORM_MEAN, "std": NORM_STD},
    }
    if extra:
        payload["metrics"] = extra
    torch.save(payload, path)

def _print_progress(epoch: int, train_info: dict[str, float], val_info: dict[str, object]) -> None:
    print(
        f"epoch {epoch:>3}  loss={train_info['loss']:8.4f}  "
        f"val_acc={val_info['accuracy']:6.2%}  val_cer={val_info['cer']:6.4f}  "
        f"STN回退={float(val_info['fallback']):5.1%}"
    )

def run_overfit(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    """过拟合检查：用少量真实样本反复训，直到整牌全对"""
    dataset = PlateDataset.from_manifest(
        args.manifest, image_root=args.image_root, augment=False, size=STN_INPUT_SIZE
    )
    take = min(args.overfit, len(dataset))
    samples = PlateDataset(
        [dataset.samples[index] for index in range(take)],
        image_root=dataset.image_root,
        size=STN_INPUT_SIZE,
        augment=False,
    )
    model = build_recognizer(args.mode, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    info = model.describe()
    print(
        f"模式={info['mode']}  样本数={len(samples)}  批大小={len(samples)}  "
        f"STN参数={info['stn_parameters']:,}  LPRNet参数={info['lprnet_parameters']:,}  "
        f"合计={info['total_parameters']:,}"
    )

    started = time.perf_counter()
    passed = False
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_info = train_one_epoch(
            model, samples, device, len(samples), optimizer, args.grad_clip
        )
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            val_info = evaluate_dataset(model, samples, device, len(samples))
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(train_info["loss"], 4),
                    "accuracy": round(float(val_info["accuracy"]), 4),
                }
            )
            _print_progress(epoch, train_info, val_info)
            if float(val_info["accuracy"]) >= 1.0 and train_info["loss"] < 0.05:
                print(f"epoch {epoch:>3}  已完全记住训练样本，提前结束")
                passed = True
                break

    final = evaluate_dataset(model, samples, device, len(samples))
    elapsed = time.perf_counter() - started
    checkpoint = args.output or Path("outputs") / f"end2end_{args.mode}_overfit.pt"
    save_checkpoint(model, checkpoint, args.mode, {"overfit": len(samples), "final": final})

    print("-" * 78)
    for example in final["examples"]:  # type: ignore[union-attr]
        mark = "OK " if example["truth"] == example["prediction"] else "ERR"
        print(f"{mark} 真值={example['truth']:<10} 预测={example['prediction']}")
    print("-" * 78)
    print(
        f"最终 整牌准确率={float(final['accuracy']):.2%}  CER={float(final['cer']):.4f}  "
        f"用时={elapsed:.1f}s  权重={checkpoint}"
    )
    print("结论：" + ("过拟合检查通过 ✅ 可以做 E1 消融了" if passed else "未通过 ❌ 先别做消融"))
    return {
        "mode": args.mode,
        "passed": passed,
        "samples": len(samples),
        "accuracy": round(float(final["accuracy"]), 4),
        "cer": round(float(final["cer"]), 4),
        "elapsed_seconds": round(elapsed, 2),
        "checkpoint": str(checkpoint),
        "history": history,
        "parameters": info,
    }
def run_training(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    """正式训练：训练集训练、验证集选模型、测试集出最终数字。"""
    common = dict(image_root=args.image_root, size=STN_INPUT_SIZE)
    train_set = PlateDataset.from_manifest(
        args.manifest, split="train", augment=not args.no_augment, **common
    )
    if args.limit > 0 and args.limit < len(train_set):
        train_set = PlateDataset(
            train_set.samples[: args.limit],
            image_root=train_set.image_root,
            size=STN_INPUT_SIZE,
            augment=not args.no_augment,
        )
    val_set = PlateDataset.from_manifest(args.manifest, split="val", augment=False, **common)
    test_set = PlateDataset.from_manifest(args.manifest, split="test", augment=False, **common)

    model = build_recognizer(args.mode, device)
    if args.init is not None and Path(args.init).exists():
        loaded = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(loaded["state_dict"])
        print(f"已从 {args.init} 加载权重继续训练（warm start）")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    info = model.describe()
    print(
        f"模式={info['mode']}  训练={len(train_set)} 验证={len(val_set)} 测试={len(test_set)}  "
        f"参数量={info['total_parameters']:,}  批大小={args.batch_size}  学习率={args.lr}"
    )

    started = time.perf_counter()
    history: list[dict[str, object]] = []
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        train_info = train_one_epoch(
            model, train_set, device, args.batch_size, optimizer, args.grad_clip
        )
        val_info = evaluate_dataset(model, val_set, device, args.batch_size)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "loss": round(train_info["loss"], 4),
                "val_accuracy": round(float(val_info["accuracy"]), 4),
                "val_cer": round(float(val_info["cer"]), 4),
            }
        )
        _print_progress(epoch, train_info, val_info)
        if float(val_info["accuracy"]) > best_accuracy:
            best_accuracy = float(val_info["accuracy"])
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_info = evaluate_dataset(model, test_set, device, args.batch_size)
    elapsed = time.perf_counter() - started

    checkpoint = args.output or Path("outputs") / f"end2end_{args.mode}.pt"
    save_checkpoint(
        model,
        checkpoint,
        args.mode,
        {
            "best_val_accuracy": round(best_accuracy, 4),
            "test_accuracy": round(float(test_info["accuracy"]), 4),
            "test_cer": round(float(test_info["cer"]), 4),
            "train_samples": len(train_set),
        },
    )

    print("-" * 78)
    print(
        f"最佳验证准确率={best_accuracy:.2%}  →  测试集：整牌准确率={float(test_info['accuracy']):.2%}  "
        f"CER={float(test_info['cer']):.4f}  STN回退={float(test_info['fallback']):.1%}"
    )
    for example in test_info["examples"]:  # type: ignore[union-attr]
        mark = "OK " if example["truth"] == example["prediction"] else "ERR"
        print(f"{mark} 真值={example['truth']:<10} 预测={example['prediction']}")
    print(f"总用时={elapsed:.1f}s  权重={checkpoint}")
    return {
        "mode": args.mode,
        "train_samples": len(train_set),
        "best_val_accuracy": round(best_accuracy, 4),
        "test_accuracy": round(float(test_info["accuracy"]), 4),
        "test_cer": round(float(test_info["cer"]), 4),
        "test_fallback": round(float(test_info["fallback"]), 4),
        "elapsed_seconds": round(elapsed, 2),
        "checkpoint": str(checkpoint),
        "history": history,
        "parameters": info,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="STN + LPRNet 端到端训练")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="affine",
        choices=[RectificationMode.OFF.value, RectificationMode.AFFINE.value,
                 RectificationMode.PERSPECTIVE.value],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 条训练样本（快速试跑）")
    parser.add_argument("--overfit", type=int, default=0, help=">0 时改为过拟合检查")
    parser.add_argument("--no-augment", action="store_true", help="训练时关闭数据增强（诊断用）")
    parser.add_argument("--init", type=Path, default=None, help="从已有权重继续训练（warm start）")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    result = run_overfit(args, device) if args.overfit > 0 else run_training(args, device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
