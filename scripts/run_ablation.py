#!/usr/bin/env python
"""E1消融主表：把三种模式的评估结果汇总成CSV+Markdown表格"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.plate_dataset import PlateDataset  # noqa: E402
from src.eval.evaluate import evaluate_dataset, format_table, load_checkpoint  # noqa: E402
from src.models.stn import STN_INPUT_SIZE  # noqa: E402
from src.train.train_recognizer import get_device  # noqa: E402

MODES = ("off", "affine", "perspective")
CANDIDATE_NAMES = ("end2end_{mode}.pt", "end2end_{mode}_overfit.pt")

def find_checkpoint(directory: Path, mode: str) -> Path | None:
    """按优先级找一个模式的权重文件；都没有就返回 None。"""
    for template in CANDIDATE_NAMES:
        candidate = directory / template.format(mode=mode)
        if candidate.exists():
            return candidate
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E1 消融主表：off / affine / perspective")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args(argv)

    device = get_device()
    split = None if args.split == "all" else args.split
    dataset = PlateDataset.from_manifest(
        args.manifest, image_root=args.image_root, split=split, size=STN_INPUT_SIZE
    )
    print(f"评估划分={args.split}  样本数={len(dataset)}  设备={device}\n")

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for mode in MODES:
        checkpoint = find_checkpoint(args.checkpoint_dir, mode)
        if checkpoint is None:
            missing.append(mode)
            print(f"[跳过] {mode}：在 {args.checkpoint_dir} 下找不到 {mode} 的权重")
            continue
        model, _payload = load_checkpoint(checkpoint, device)
        info = model.describe()
        metrics = evaluate_dataset(model, dataset, device, batch_size=args.batch_size)
        rows.append(
            {
                "mode": info["mode"],
                "parameters": info["total_parameters"],
                "samples": metrics["samples"],
                "accuracy": metrics["accuracy"],
                "cer": metrics["cer"],
                "latency_ms": metrics["latency_ms"],
                "fallback": metrics["fallback"],
                "num_failures": metrics["num_failures"],
                "checkpoint": str(checkpoint),
            }
        )
        print(
            f"[完成] {mode:<12} 准确率={float(metrics['accuracy']):6.2%}  "
            f"CER={float(metrics['cer']):.4f}  耗时={float(metrics['latency_ms']):6.2f}ms"
        )

    if not rows:
        print("\n没有可汇总的结果。请先训练至少一个模式的权重。")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "ablation_e1.json"
    csv_path = args.out_dir / "ablation_e1.csv"
    md_path = args.out_dir / "ablation_e1.md"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(
        "# E1 变换类型消融（%s 划分，%d 个样本）\n\n%s\n"
        % (args.split, len(dataset), format_table(rows)),
        encoding="utf-8",
    )

    print()
    print(format_table(rows))
    print(f"\n已写出：\n  {json_path}\n  {csv_path}\n  {md_path}")
    if missing:
        print(f"\n注意：这些模式还没有权重，未纳入统计 -> {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())