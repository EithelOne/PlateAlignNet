"""统一评估：加载权重，在指定划分上出【报告级】指标"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.data.plate_dataset import PlateDataset
from src.models.lprnet import decode_batch
from src.models.plate_recognizer import (
    PlateRecognizer,
    build_recognizer,
    character_error_rate,
    edit_distance,
)
from src.models.stn import STN_INPUT_SIZE
from src.train.train_recognizer import get_device

def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[PlateRecognizer, dict[str, object]]:
    """加载权重并重建模型"""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mode = str(payload.get("mode", "affine"))
    model = build_recognizer(mode, device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload

@torch.no_grad()
def evaluate_dataset(
    model: PlateRecognizer,
    dataset: PlateDataset,
    device: torch.device,
    batch_size: int = 32,
    max_failures: int = 20,
) -> dict[str, object]:
    """在数据集上跑一遍，产出全部指标"""
    model.eval()
    predictions: list[str] = []
    truths: list[str] = []
    fallbacks: list[float] = []
    elapsed_total = 0.0

    for start in range(0, len(dataset), batch_size):
        indices = range(start, min(start + batch_size, len(dataset)))
        records = [dataset[index] for index in indices]
        images = torch.from_numpy(np.stack([record.image for record in records])).to(device)

        started = time.perf_counter()
        output = model(images, return_details=True)
        elapsed_total += time.perf_counter() - started

        predictions.extend(decode_batch(output.logits))
        fallbacks.append(float(output.stats.fallback_fraction))
        truths.extend(record.text for record in records)

    correct = sum(1 for p, t in zip(predictions, truths) if p == t)

    by_length: dict[str, dict[str, float]] = {}
    for prediction, truth in zip(predictions, truths):
        key = str(len(truth))
        bucket = by_length.setdefault(key, {"samples": 0.0, "correct": 0.0})
        bucket["samples"] += 1
        bucket["correct"] += 1 if prediction == truth else 0
    for bucket in by_length.values():
        bucket["accuracy"] = round(bucket["correct"] / max(bucket["samples"], 1), 4)

    failures: list[dict[str, object]] = []
    for prediction, truth in zip(predictions, truths):
        if prediction == truth:
            continue
        failures.append(
            {
                "truth": truth,
                "prediction": prediction,
                "edit_distance": edit_distance(prediction, truth),
            }
        )
    failures.sort(key=lambda item: -int(item["edit_distance"]))  # type: ignore[arg-type]

    return {
        "samples": len(truths),
        "accuracy": round(correct / max(len(truths), 1), 4),
        "cer": round(character_error_rate(predictions, truths), 4),
        "latency_ms": round(elapsed_total * 1000.0 / max(len(truths), 1), 3),
        "fallback": round(sum(fallbacks) / max(len(fallbacks), 1), 4),
        "by_length": dict(sorted(by_length.items())),
        "num_failures": len(failures),
        "failures": failures[:max_failures],
        "predictions": predictions,
        "texts": truths,
    }

def format_table(rows: Sequence[dict[str, object]]) -> str:
    """把一行或多行结果排成Markdown表"""
    header = (
        "| 模式 | 参数量 | 样本数 | 整牌准确率 | CER | 单张耗时(ms) | STN回退 | 失败数 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row.get('mode', '-')} | {row.get('parameters', '-')} | {row.get('samples', '-')} "
            f"| {float(row.get('accuracy', 0.0)):.2%} | {float(row.get('cer', 0.0)):.4f} "
            f"| {float(row.get('latency_ms', 0.0)):.2f} "
            f"| {float(row.get('fallback', 0.0)):.2%} | {row.get('num_failures', '-')} |"
        )
    return "\n".join(lines)



def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：评估一个权重文件在指定划分上的表现。"""
    parser = argparse.ArgumentParser(description="统一评估：整牌准确率 / CER / 耗时")
    parser.add_argument("--checkpoint", type=Path, required=True, help="权重文件路径")
    parser.add_argument("--manifest", type=Path, required=True, help="samples.jsonl 路径")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test", "all"]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None, help="把结果同时写进这个 JSON")
    args = parser.parse_args(argv)

    device = get_device()
    model, payload = load_checkpoint(args.checkpoint, device)
    info = model.describe()

    split = None if args.split == "all" else args.split
    dataset = PlateDataset.from_manifest(
        args.manifest, image_root=args.image_root, split=split, size=STN_INPUT_SIZE
    )
    print(
        f"权重={args.checkpoint}  模式={info['mode']}  参数量={info['total_parameters']:,}  "
        f"划分={args.split}  样本数={len(dataset)}"
    )

    metrics = evaluate_dataset(
        model, dataset, device, batch_size=args.batch_size, max_failures=args.max_failures
    )
    row = {
        "mode": info["mode"],
        "parameters": info["total_parameters"],
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        **{key: metrics[key] for key in ("samples", "accuracy", "cer", "latency_ms", "fallback")},
        "num_failures": metrics["num_failures"],
        "by_length": metrics["by_length"],
    }
    print()
    print(format_table([row]))
    print()
    print(f"STN 回退比例 = {float(metrics['fallback']):.2%}（透视模式下越低越好）")
    print(f"失败样本数 = {metrics['num_failures']}（最多列 {args.max_failures} 条）")
    for failure in metrics["failures"][:10]:  # type: ignore[union-attr]
        print(
            f"  真值={failure['truth']:<10} 预测={failure['prediction']:<10} "
            f"编辑距离={failure['edit_distance']}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({**row, "failures": metrics["failures"]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n结果已写入 {args.output}")

    print()
    print(json.dumps({**row, "checkpoint_metadata_keys": sorted(payload.keys())},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








