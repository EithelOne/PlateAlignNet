"""车牌识别数据集：把数据清单变成 (图像张量, 标签索引, 标签长度)"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from src.data.ccpd_parser import build_sample, make_synthetic_filename, parse_ccpd_filename
from src.inference.schemas import BoundingBox
from src.utils.plate_charset import encode_text, labels_to_batch, min_time_steps

# LPRNet 输入尺寸 (宽, 高)
PLATE_SIZE: tuple[int, int] = (94, 24)

# 归一化：(x / 255 - 0.5) / 0.5 -> [-1, 1]，训练与推理必须完全一致
NORM_MEAN = 0.5
NORM_STD = 0.5

# 裁剪外扩比例
DEFAULT_MARGIN: tuple[float, float] = (0.05, 0.05)

# 中文字体候选（含 Windows 路径，避免注释里的中文变成方块）
FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)

def _as_set(value: str | Sequence[str] | None) -> set[str] | None:
    """把字符串或序列统一成集合，方便写过滤条件。"""
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)

def crop_with_margin(
    image: Image.Image,
    bbox: BoundingBox,
    margin: tuple[float, float] = DEFAULT_MARGIN,
) -> Image.Image:
    """按边界框裁剪车牌，并在四周留出一点边距
       1. 检测框常常贴得太紧，稍留 5% 边距能避免把首尾字符切掉 2. 统一在这里做 clamp，越界框不会让 crop 崩掉
    """
    margin_x = round(bbox.width * margin[0])
    margin_y = round(bbox.height * margin[1])
    expanded = BoundingBox(
        bbox.x1 - margin_x, bbox.y1 - margin_y, bbox.x2 + margin_x, bbox.y2 + margin_y
    ).clamp(*image.size)
    if expanded.width < 2 or expanded.height < 2:
        raise ValueError(f"裁剪区域过小：{expanded}")
    return image.crop((expanded.x1, expanded.y1, expanded.x2, expanded.y2))

def augment_plate(image: Image.Image, rng: random.Random) -> Image.Image:
    """对车牌小图做轻度数据增强（注意：这个只在训练时用）
        目的：模拟真实场景的亮度波动、轻微倾斜和模糊，让模型不要死记某一种成像条件
        要点：绝对不做水平翻转——因为车牌有字符顺序，翻转后语义就变了
    """
    result = image.convert("RGB")
    result = ImageEnhance.Brightness(result).enhance(rng.uniform(0.7, 1.3))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.8, 1.2))
    angle = rng.uniform(-15.0, 15.0)
    if abs(angle) > 1.0:
        result = result.rotate(
            angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(0, 0, 0)
        )
    if rng.random() < 0.2:
        result = result.filter(ImageFilter.GaussianBlur(rng.choice([0.5, 1.0])))
    return result

def prepare_plate_image(
    image: Image.Image,
    size: tuple[int, int] = PLATE_SIZE,
    rng: random.Random | None = None,
) -> np.ndarray:
    """车牌小图 -> 模型输入张量（numpy）
       把不同尺寸的车牌统一成固定大小并归一化
    """
    working = augment_plate(image, rng) if rng is not None else image.convert("RGB")
    resized = working.resize(size, Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - NORM_MEAN) / NORM_STD
    return np.transpose(array, (2, 0, 1))


def labels_fit_time_steps(texts: Sequence[str], time_steps: int) -> list[str]:
    """找出【标签太长、模型输出时间步装不下】的样本
       相邻重复字符必须多占一个时间步，所以 8 位车牌理论最坏要 8+7=15 步

    """
    return [text for text in texts if min_time_steps(text) > time_steps]

@dataclass(frozen=True)
class PlateRecord:
    """一条已解码的识别样本。"""

    image: np.ndarray = field(repr=False)
    label_indices: tuple[int, ...]
    text: str
    image_path: Path
    split: str
    subset: str

    @property
    def label_length(self) -> int:
        return len(self.label_indices)

@dataclass(frozen=True)
class Batch:
    """一个 batch 的三件套。"""

    images: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)
    lengths: np.ndarray = field(repr=False)
    texts: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.images.shape[0])

class PlateDataset:
    """识别数据集：按需从磁盘读图、裁剪、缩放、编码标签。"""

    def __init__(
        self,
        samples: Sequence[dict[str, Any]],
        image_root: Path | str | None = None,
        size: tuple[int, int] = PLATE_SIZE,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.samples = [dict(item) for item in samples]
        if not self.samples:
            raise ValueError("样本为空：请检查清单文件或过滤条件")
        self.image_root = Path(image_root) if image_root is not None else None
        self.size = (int(size[0]), int(size[1]))
        self.augment = augment
        self.rng = random.Random(seed)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path | str,
        image_root: Path | str | None = None,
        split: str | Sequence[str] | None = None,
        subset: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> "PlateDataset":
        """从samples.jsonl读样本，可按split/subset过滤
           训练脚本只要写 split="train"，验证写 split="val"，就不用再关心清单细节 
        """
        wanted_splits = _as_set(split)
        wanted_subsets = _as_set(subset)
        rows: list[dict[str, Any]] = []
        with Path(manifest_path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if wanted_splits is not None and row.get("split") not in wanted_splits:
                    continue
                if wanted_subsets is not None and row.get("subset") not in wanted_subsets:
                    continue
                rows.append(row)
        return cls(rows, image_root=image_root, **kwargs)

    def __len__(self) -> int:
        return len(self.samples)

    def image_path(self, index: int) -> Path:
        """算出第 index 条样本的真实图片路径。"""
        raw = Path(self.samples[index]["image_path"])
        if raw.is_absolute() or self.image_root is None:
            return raw
        return self.image_root / raw

    def __getitem__(self, index: int) -> PlateRecord:
        """取一条样本：读图 -> 裁剪 -> 缩放归一化 -> 编码标签。"""
        sample = self.samples[index]
        path = self.image_path(index)
        if not path.exists():
            raise FileNotFoundError(f"找不到图片：{path}")

        with Image.open(path) as handle:
            image = handle.convert("RGB")

        bbox = BoundingBox(*sample["bbox"])
        cropped = crop_with_margin(image, bbox)
        tensor = prepare_plate_image(cropped, self.size, self.rng if self.augment else None)

        text = str(sample["plate_text"])
        indices = tuple(int(item) for item in sample["label_indices"])
        if len(indices) != len(text):
            raise ValueError(f"标签索引与文本长度不一致：{indices} vs {text!r}")
        if indices != tuple(encode_text(text)):
            raise ValueError(f"标签索引与字符表不匹配（字符表顺序可能改过）：{text!r}")

        return PlateRecord(
            image=tensor,
            label_indices=indices,
            text=text,
            image_path=path,
            split=str(sample.get("split", "")),
            subset=str(sample.get("subset", "")),
        )


    def iter_batches(self, batch_size: int = 8, shuffle: bool = False) -> Iterator[Batch]:
        """按批产出三件套，是训练循环的唯一入口。"""
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        order = list(range(len(self)))
        if shuffle:
            self.rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            records = [self[index] for index in chunk]
            images = np.stack([record.image for record in records])
            labels, lengths = labels_to_batch([record.text for record in records])
            yield Batch(
                images=images,
                labels=labels,
                lengths=lengths,
                texts=tuple(record.text for record in records),
            )


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """按字号加载一个含中文的字体，找不到就退回默认字体。"""
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()
    
def render_plate_image(
    bbox: BoundingBox,
    text: str,
    canvas_size: tuple[int, int] = (720, 1160),
    background: str = "#2e3a44",
    plate_colour: str = "#0b5ba6",
) -> Image.Image:
    """在指定位置画一块带文字的车牌，生成【假图】
    TODO：后续弄一个数据集后删掉
    """
    image = Image.new("RGB", canvas_size, background)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (bbox.x1, bbox.y1, bbox.x2, bbox.y2),
        fill=plate_colour,
        outline="white",
        width=max(2, bbox.height // 20),
    )
    font = _load_font(max(10, int(bbox.height * 0.6)))
    centre = ((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2)
    draw.text(centre, text, fill="white", anchor="mm", font=font)
    return image

def build_synthetic_dataset(out_dir: Path | str, count: int = 24, seed: int = 0) -> Path:
    """生成一整套【假数据集】：N张假车牌图+一份清单
        TODO:后续整理好一个数据集后删掉
    """
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for _ in range(count):
        file_name = make_synthetic_filename(rng)
        fields = parse_ccpd_filename(file_name)
        canvas = render_plate_image(fields["bbox"], fields["plate_text"])
        canvas.save(images_dir / file_name)
        samples.append(build_sample(file_name, Path("images") / file_name, subset="synthetic"))

    manifest = out_dir / "samples.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    return manifest


def describe_batch(batch: Batch, time_steps: int = 24) -> dict[str, Any]:
    """把 batch 的形状与内容整理成一段可读摘要。"""
    too_long = labels_fit_time_steps(batch.texts, time_steps)
    return {
        "images": list(batch.images.shape),
        "labels": list(batch.labels.shape),
        "lengths": batch.lengths.tolist(),
        "texts": list(batch.texts),
        "image_range": [round(float(batch.images.min()), 3), round(float(batch.images.max()), 3)],
        "fits_in_time_steps": not too_long,
        "too_long": too_long,
    }

def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：造假数据+跑一个batch验证整条链路
        TODO：后续弄出数据集后考虑修改
    """
    parser = argparse.ArgumentParser(description="识别数据集自检（无真实数据时使用）")
    parser.add_argument("--out", type=Path, default=Path("data/_synthetic_plates"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    manifest = build_synthetic_dataset(args.out, count=args.count, seed=args.seed)
    print(f"清单已生成：{manifest}")

    dataset = PlateDataset.from_manifest(manifest, image_root=args.out)
    print(f"数据集样本数：{len(dataset)}")

    for index, batch in enumerate(dataset.iter_batches(batch_size=args.batch_size)):
        print(f"第 {index + 1} 批：" + json.dumps(describe_batch(batch), ensure_ascii=False))
        if index >= 1:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



