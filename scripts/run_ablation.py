#!/usr/bin/env python
"""E1消融主表：把三种模式的评估结果汇总成CSV+Markdown表格"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验（配对样本的比例差异是否显著）"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)

def find_checkpoint(directory: Path, mode: str) -> tuple[Path | None, bool]:
    """找一个模式的权重文件"""
    full = directory / f"end2end_{mode}.pt"
    if full.exists():
        return full, False
    overfit = directory / f"end2end_{mode}_overfit.pt"
    if overfit.exists():
        return overfit, True
    return None, False



def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E1 消融主表：off / affine / perspective")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--allow-overfit",
        action="store_true",
        help="允许用「过拟合检查权重」参与统计（仅用于调试，结果不能写进报告）",
    )
    args = parser.parse_args(argv)

    device = get_device()
    split = None if args.split == "all" else args.split
    dataset = PlateDataset.from_manifest(
        args.manifest, image_root=args.image_root, split=split, size=STN_INPUT_SIZE
    )
    print(f"评估划分={args.split}  样本数={len(dataset)}  设备={device}")
    print(f"权重目录={args.checkpoint_dir}\n")

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    blocked: list[str] = []
    predictions_by_mode: dict[str, list[str]] = {}
    truths: list[str] = []

    for mode in MODES:
        checkpoint, is_overfit = find_checkpoint(args.checkpoint_dir, mode)
        if checkpoint is None:
            missing.append(mode)
            print(f"[跳过] {mode:<12} 找不到 end2end_{mode}.pt（全量训练还没跑）")
            continue
        if is_overfit and not args.allow_overfit:
            blocked.append(mode)
            print(
                f"[拒绝] {mode:<12} 只有过拟合权重 end2end_{mode}_overfit.pt。\n"
                f"       它只记住了 20 多张图，评估结果会是 1% 左右的假数字，不能进消融表。\n"
                f"       请先跑全量训练：\n"
                f"         python -m src.train.train_end2end --manifest <清单> --mode {mode} "
                f"--epochs 60 --lr 5e-4 --init outputs/end2end_{mode}_overfit.pt"
            )
            continue

        model, _payload = load_checkpoint(checkpoint, device)
        info = model.describe()
        metrics = evaluate_dataset(model, dataset, device, batch_size=args.batch_size)
        predictions_by_mode[info["mode"]] = list(metrics["predictions"])  # type: ignore[arg-type]
        truths = list(metrics["texts"])  # type: ignore[arg-type]
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
                "checkpoint": checkpoint.name,
                "overfit_weights": is_overfit,
            }
        )
        print(
            f"[完成] {info['mode']:<12} 准确率={float(metrics['accuracy']):6.2%}  "
            f"CER={float(metrics['cer']):.4f}  耗时={float(metrics['latency_ms']):5.2f}ms  "
            f"权重={checkpoint.name}"
            + ("   ⚠️ 用的是过拟合权重！" if is_overfit else "")
        )

    if not rows:
        print("\n没有任何可汇总的结果。请先训练至少一个模式的全量权重。")
        return 1

    # ---- 配对比较：同一批测试样本上逐样本对比（报告 §3.12 要求）----
    paired: list[dict[str, object]] = []
    baseline = "off" if "off" in predictions_by_mode else next(iter(predictions_by_mode))
    for mode, predictions in predictions_by_mode.items():
        if mode == baseline:
            continue
        base = predictions_by_mode[baseline]
        baseline_only = sum(
            1 for pb, pm, t in zip(base, predictions, truths) if pb == t and pm != t
        )
        mode_only = sum(
            1 for pb, pm, t in zip(base, predictions, truths) if pb != t and pm == t
        )
        p_value = mcnemar_exact(baseline_only, mode_only)
        paired.append(
            {
                "baseline": baseline,
                "mode": mode,
                "baseline_only_correct": baseline_only,
                "mode_only_correct": mode_only,
                "p_value": round(p_value, 4),
                "significant_at_0.05": p_value < 0.05,
            }
        )
        print(
            f"[配对] {baseline} vs {mode}: 只有{baseline}对={baseline_only}  "
            f"只有{mode}对={mode_only}  McNemar 精确 p={p_value:.4f}  "
            + ("差异显著" if p_value < 0.05 else "差异不显著（可能只是噪声）")
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "ablation_e1.json"
    csv_path = args.out_dir / "ablation_e1.csv"
    md_path = args.out_dir / "ablation_e1.md"

    json_path.write_text(
        json.dumps(
            {"rows": rows, "paired_comparisons": paired, "missing": missing, "blocked": blocked},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    notes: list[str] = []
    if missing:
        notes.append("以下模式还没有全量训练权重，未纳入统计：" + "、".join(missing))
    if blocked:
        notes.append("以下模式只有过拟合权重，已拒绝统计：" + "、".join(blocked))
    table = format_table(rows)
    if paired:
        table += (
            "\n\n配对比较（McNemar 精确检验，基线="
            + baseline
            + "）：\n\n| 对比 | 只有基线对 | 只有该模式对 | p 值 | 结论 |\n"
            "| --- | ---: | ---: | ---: | --- |\n"
        )
        for item in paired:
            verdict = "显著" if bool(item["significant_at_0.05"]) else "不显著"
            table += (
                f"| {item['baseline']} vs {item['mode']} | {item['baseline_only_correct']} "
                f"| {item['mode_only_correct']} | {item['p_value']} | {verdict} |\n"
            )
    md_path.write_text(
        "# E1 变换类型消融（%s 划分，%d 个样本）\n\n%s\n" % (args.split, len(dataset), table)
        + ("\n" + "\n".join(f"> {note}\n" for note in notes) if notes else ""),
        encoding="utf-8",
    )

    print()
    print(table)
    print(f"\n已写出：\n  {json_path}\n  {csv_path}\n  {md_path}")
    for note in notes:
        print(f"注意：{note}")
    if blocked:
        print("\n结论：消融主表不完整，请先补齐被拒绝的模式再出报告数字。")
        return 1
    if missing:
        print("\n结论：消融主表不完整（缺模式），补齐后再出报告数字。")
        return 1
    print("\n结论：三个模式齐全，可以写进报告 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
