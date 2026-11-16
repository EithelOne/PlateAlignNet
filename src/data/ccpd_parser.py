"""CCPD 文件名解析与数据清单生成（DEMO.md 验收清单第 2 步）。

CCPD 把边界框、四角点、车牌文本等信息全部编码在**文件名**里，格式为 7 段，
用 "-" 分隔（坐标写成 "x&y"，同一个点内部用 "&"，不同点之间用 "_"）：

    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-12_35_60_61_62_63_64-37-15.jpg
     │   │       │              │                                    │                  │  └ 6 模糊度
     │   │       │              │                                    │                  └── 5 亮度
     │   │       │              │                                    └── 4 车牌字符索引（7 位或 8 位）
     │   │       │              └── 3 四角点（原始顺序：右下、左下、左上、右上）
     │   │       └── 2 边界框（左上点_右下点）
     │   └── 1 倾斜角（水平_垂直，单位按官方说明，这里只原样保存不解释）
     └── 0 车牌面积占全图比例（百分数）

"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.inference.schemas import BoundingBox
from src.utils.plate_charset import PROVINCE_CHARS, encode_text, validate_label


FIELD_COUNT = 7
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# CCPD 原始四角点顺序（官方说明）：右下、左下、左上、右上
CCPD_CORNER_ORDER = ("bottom_right", "bottom_left", "top_left", "top_right")
# 项目内部统一使用的顺序
STANDARD_CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")
# 原始下标 -> 标准顺序要取的原始位置：左上=2，右上=3，右下=0，左下=1
ORDER_MAP = (2, 3, 0, 1)

# 目录名 -> 子集短标签
SUBSET_ALIASES = {
    "ccpd_base": "base",
    "ccpd_blur": "blur",
    "ccpd_challenge": "challenge",
    "ccpd_db": "db",
    "ccpd_fn": "fn",
    "ccpd_rotate": "rotate",
    "ccpd_tilt": "tilt",
    "ccpd_weather": "weather",
    "ccpd_np": "np",
    "ccpd_green": "green",
    "ccpd2019_base": "base",
}
CCPD_LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
CCPD_AD_CHARS = CCPD_LETTERS + [str(d) for d in range(10)]
CCPD_PROVINCE_COUNT = 31

class ParseError(Exception):
    """某个文件名无法解析成合法样本时抛出。"""


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}

@dataclass(frozen=True)
class PlateSample:
    image_path: Path
    subset: str
    split: str
    area_ratio: int
    tilt: tuple[int, int]
    bbox: BoundingBox
    corners: tuple[Point, Point, Point, Point]
    label_indices: tuple[int, ...]
    plate_text: str
    brightness: int
    blurriness: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path.as_posix(),
            "subset": self.subset,
            "split": self.split,
            "area_ratio": self.area_ratio,
            "tilt": list(self.tilt),
            "bbox": [self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2],
            "corners": [point.to_dict() for point in self.corners],
            "label_indices": list(self.label_indices),
            "plate_text": self.plate_text,
            "brightness": self.brightness,
            "blurriness": self.blurriness,
        }

@dataclass(frozen=True)
class ParseIssue:
    image_path: Path
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"image_path": self.image_path.as_posix(), "reason": self.reason}


def _int_field(text: str, name: str) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise ParseError(f"字段 {name} 不是整数：{text!r}") from exc


def _int_list(text: str, sep: str, expected: int, name: str) -> tuple[int, ...]:
    items = text.split(sep)
    if len(items) != expected:
        raise ParseError(f"字段 {name} 应含 {expected} 项，实际 {len(items)} 项：{text!r}")
    try:
        return tuple(int(item) for item in items)
    except ValueError as exc:
        raise ParseError(f"字段 {name} 含非整数项：{text!r}") from exc


    
def split_corner_field(text: str) -> tuple[Point, ...]:
    """把坐标字段拆成点序列。

    输入："154&383_386&473" 这样的字符串（点之间 "_"，x/y 之间 "&"）。
    输出：tuple[Point, ...]。
    作用：边界框字段（2 个点）和四角点字段（4 个点）共用这一套坐标语法。
    """
    points: list[Point] = []
    for chunk in text.split("_"):
        pair = chunk.split("&")
        if len(pair) != 2:
            raise ParseError(f"坐标项应为 'x&y'，实际 {chunk!r}")
        try:
            points.append(Point(int(pair[0]), int(pair[1])))
        except ValueError as exc:
            raise ParseError(f"坐标项含非整数：{chunk!r}") from exc
    return tuple(points)

def to_tl_tr_br_bl(corners_raw: Sequence[Point]) -> tuple[Point, Point, Point, Point]:
    """把CCPD原始角点顺序改成项目统一顺序
    原始顺序 (右下, 左下, 左上, 右上)，长度 4
    输出：(左上, 右上, 右下, 左下)
    """
    if len(corners_raw) != 4:
        raise ParseError(f"四角点应为 4 个，实际 {len(corners_raw)} 个")
    return tuple(corners_raw[i] for i in ORDER_MAP)  # type: ignore[return-value]


def cross(o: Point, a: Point, b: Point) -> int:
    """叉积（z 分量），用来判断三点的转折方向。"""
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def quad_area(corners: Sequence[Point]) -> float:
    """鞋带公式求多边形面积（像素²）。"""
    total = 0
    for index in range(len(corners)):
        current = corners[index]
        following = corners[(index + 1) % len(corners)]
        total += current.x * following.y - following.x * current.y
    return abs(total) / 2


def validate_quad(corners: Sequence[Point]) -> list[str]:
    """检查统一顺序后的四角点是否构成一个正常的车牌四边形
    输入：按 (左上, 右上, 右下, 左下) 排好的 4 个点
    输出：问题列表，空列表表示通过
    作用：1. 抓"顺序假设错了"——如果数据不是右下起顺时针，左上点就不会在左下点
          上方，这里会立刻报出来；2. 抓退化的四边形（共线、面积过小），避免后面
          求单应矩阵时出现病态结果。
    """

    problems: list[str] = []
    if len(corners) != 4:
        return [f"四角点应为 4 个，实际 {len(corners)} 个"]
    top_left, top_right, bottom_right, bottom_left = corners

    if top_left.y >= bottom_left.y:
        problems.append(f"左上点不在左下点上方：y {top_left.y} -> {bottom_left.y}")
    if top_left.x >= top_right.x:
        problems.append(f"左上点不在右上点左侧：x {top_left.x} -> {top_right.x}")

    signs = []
    for index in range(4):
        signs.append(
            cross(corners[index], corners[(index + 1) % 4], corners[(index + 2) % 4])
        )
    if any(sign == 0 for sign in signs):
        problems.append("存在共线点，四边形退化")
    elif all(sign < 0 for sign in signs):
        problems.append("四角点走向为逆时针（图像坐标下应为顺时针），角点顺序可能反了")
    elif not all(sign > 0 for sign in signs):
        problems.append("四边形非凸（转折方向不一致）")

    area = quad_area(corners)
    if area < 100:
        problems.append(f"四边形面积过小：{area:.1f}")
    return problems

def decode_ccpd_indices(indices: Sequence[int]) -> str:
    """把文件名里的数字索引按 CCPD 自己的编码表还原成车牌文本。
         CCPD 的索引不是我们那张,69字符扁平表的下标，直接拿 CHARSET 去查会解出乱码
    """
    if not indices:
        raise ParseError("车牌字符索引为空")
    if indices[0] >= CCPD_PROVINCE_COUNT:
        raise ParseError(f"省份索引 {indices[0]} 超出 0..{CCPD_PROVINCE_COUNT - 1}")
    text = PROVINCE_CHARS[indices[0]]
    for position, value in enumerate(indices[1:], start=1):
        if value >= len(CCPD_AD_CHARS):
            raise ParseError(f"第 {position} 位索引 {value} 超出 0..{len(CCPD_AD_CHARS) - 1}")
        text += CCPD_AD_CHARS[value]
    return text



def parse_ccpd_filename(file_name: str) -> dict[str, Any]:
    """把一条CCPD文件名解析成字段字典
        解析失败时抛出 ParseError
    """
    path = Path(file_name)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ParseError(f"不支持的图片后缀：{path.suffix!r}")

    parts = path.stem.split("-")
    if len(parts) != FIELD_COUNT:
        raise ParseError(f"字段数应为 {FIELD_COUNT}，实际 {len(parts)}：{path.stem}")

    area_text, tilt_text, bbox_text, corner_text, plate_text, bright_text, blur_text = parts

    area_ratio = _int_field(area_text, "area")
    tilt = _int_list(tilt_text, "_", 2, "tilt")

    bbox_points = split_corner_field(bbox_text)
    if len(bbox_points) != 2:
        raise ParseError(f"边界框字段应含 2 个点，实际 {len(bbox_points)} 个")
    top_left_corner, bottom_right_corner = bbox_points
    x1, y1 = top_left_corner.x, top_left_corner.y
    x2, y2 = bottom_right_corner.x, bottom_right_corner.y
    if x1 >= x2 or y1 >= y2:
        raise ParseError(f"边界框坐标顺序异常：({x1},{y1})-({x2},{y2})")
    bbox = BoundingBox(x1, y1, x2, y2)

    corners = to_tl_tr_br_bl(split_corner_field(corner_text))
    try:
        indices = tuple(int(item) for item in plate_text.split("_"))
    except ValueError as exc:
        raise ParseError(f"车牌字符索引含非整数项：{plate_text!r}") from exc
    if len(indices) not in (7, 8):
        raise ParseError(f"车牌字符应为 7 位或 8 位，实际 {len(indices)} 位")
    plate_text = decode_ccpd_indices(indices)
    label_indices = tuple(encode_text(plate_text))

    return {
        "area_ratio": area_ratio,
        "tilt": tilt,
        "bbox": bbox,
        "corners": corners,
        "ccpd_indices": indices,
        "label_indices": label_indices,
        "plate_text": plate_text,
        "brightness": _int_field(bright_text, "brightness"),
        "blurriness": _int_field(blur_text, "blurriness"),
    }

def subset_from_path(image_path: Path) -> str:
    """从所在目录名推断挑战子集标签。
        CCPD 的挑战子集靠目录名区分（不是文件名），六个子集的细粒度结果就靠这个字段分组统计
    """
    for parent in image_path.parents:
        alias = SUBSET_ALIASES.get(parent.name.lower())
        if alias:
            return alias
    return "unknown"

def assign_split(image_path: Path, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), salt: str = "v1") -> str:
    """用稳定哈希把样本分到 train / val / test
    TODO:后续改成按身份分组
    """

    digest = hashlib.md5(f"{salt}|{image_path.as_posix()}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") / 2**32
    train_end = ratios[0]
    val_end = ratios[0] + ratios[1]
    if value < train_end:
        return "train"
    if value < val_end:
        return "val"
    return "test"

def build_sample(
    file_name: str,
    image_path: Path,
    subset: str | None = None,
    split: str | None = None,
) -> PlateSample:
    """文件名+路径 -> 一条完整的样本记录"""
    fields = parse_ccpd_filename(file_name)
    return PlateSample(
        image_path=image_path,
        subset=subset or subset_from_path(image_path),
        split=split or assign_split(image_path),
        area_ratio=fields["area_ratio"],
        tilt=fields["tilt"],
        bbox=fields["bbox"],
        corners=fields["corners"],
        label_indices=fields["label_indices"],
        plate_text=fields["plate_text"],
        brightness=fields["brightness"],
        blurriness=fields["blurriness"],
    )

def parse_sample(image_path: Path, split: str | None = None) -> PlateSample:
    """从真实文件路径解析一条样本。输入 Path，输出 PlateSample。"""
    return build_sample(image_path.name, image_path, split=split)


def check_sample(sample: PlateSample) -> list[str]:
    """对一条已解析样本做语义与稽核监察，返回问题列表
        注：解析成功≠数据正确
            这里检查：1. 号牌规则（长度/省份/第二位字母）2. 四角点四边形是否正常3. 四角点的外接矩形是否与文件名里的边界框一致

    """
    problems = list(validate_label(sample.plate_text))
    problems.extend(validate_quad(sample.corners))

    xs = [point.x for point in sample.corners]
    ys = [point.y for point in sample.corners]
    
    corner_x1, corner_x2 = min(xs), max(xs)
    corner_y1, corner_y2 = min(ys), max(ys)
    overlap_x = max(0, min(corner_x2, sample.bbox.x2) - max(corner_x1, sample.bbox.x1))
    overlap_y = max(0, min(corner_y2, sample.bbox.y2) - max(corner_y1, sample.bbox.y1))
    overlap = overlap_x * overlap_y
    area_bbox = sample.bbox.width * sample.bbox.height
    area_corner = (corner_x2 - corner_x1) * (corner_y2 - corner_y1)
    union = area_bbox + area_corner - overlap
    iou = overlap / union if union > 0 else 0.0
    if iou < 0.5:
        problems.append(
            f"四角点与边界框严重不重叠（IoU={iou:.2f}）："
            f"corners x[{corner_x1},{corner_x2}] y[{corner_y1},{corner_y2}] vs "
            f"bbox x[{sample.bbox.x1},{sample.bbox.x2}] y[{sample.bbox.y1},{sample.bbox.y2}]"
        )
    return problems


def collect_samples(
    root: Path, splits: Sequence[str] | None = None
) -> tuple[list[PlateSample], list[ParseIssue]]:
    """遍历目录，收集所有能解析的样本与所有异常
        一次扫描就把"好样本"和"坏样本"分开记账；异常不丢弃、只记录"""
    samples: list[PlateSample] = []
    issues: list[ParseIssue] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            sample = parse_sample(path)
            problems = check_sample(sample)
            if problems:
                issues.append(ParseIssue(path, "；".join(problems)))
                continue
            if splits is not None and sample.split not in splits:
                continue
            samples.append(sample)
        except ParseError as exc:
            issues.append(ParseIssue(path, str(exc)))
    return samples, issues

def make_synthetic_filename(rng: random.Random | None = None) -> str:
    """生成一条格式合法但内容随机的CCPD风格文件名
        只用来验证解析、统计、划分、写文件是否正确"""
    generator = rng or random.Random(0)
    width, height = 720, 1160
    plate_w = generator.randint(60, 180)
    plate_h = max(18, round(plate_w / 3.4))
    x1 = generator.randint(20, width - plate_w - 20)
    y1 = generator.randint(20, height - plate_h - 20)
    x2 = x1 + plate_w
    y2 = y1 + plate_h

    province = generator.randrange(CCPD_PROVINCE_COUNT)
    second = generator.randrange(len(CCPD_LETTERS))
    rest = [generator.randrange(len(CCPD_AD_CHARS)) for _ in range(5)]
    indices = [province, second] + rest
    if generator.random() < 0.25:  # 25% 概率造一条 8 位新能源车牌
        indices.append(generator.randrange(len(CCPD_AD_CHARS)))

    bbox_field = f"{x1}&{y1}_{x2}&{y2}"
    corner_field = f"{x2}&{y2}_{x1}&{y2}_{x1}&{y1}_{x2}&{y1}"
    plate_field = "_".join(str(item) for item in indices)
    return (
        f"{generator.randint(1, 90):03d}"
        f"-{generator.randint(0, 30)}_{generator.randint(0, 30)}"
        f"-{bbox_field}"
        f"-{corner_field}"
        f"-{plate_field}"
        f"-{generator.randint(10, 200)}"
        f"-{generator.randint(0, 50)}"
        ".jpg"
    )

def build_manifest(
    samples: Sequence[PlateSample],
    issues: Sequence[ParseIssue],
    out_dir: Path,
    root_label: str = "",
) -> dict[str, Any]:
    """把样本与异常写成 JSON 清单，并返回统计摘要
        训练和评估以后只读这些清单，不再重复解析文件名，避免"不同脚本各解析一遍"导致实现分叉
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_path = out_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    issues_path = out_dir / "issues.jsonl"
    with issues_path.open("w", encoding="utf-8") as handle:
        for issue in issues:
            handle.write(json.dumps(issue.to_dict(), ensure_ascii=False) + "\n")

    by_subset: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_length: dict[str, int] = {}
    for sample in samples:
        by_subset[sample.subset] = by_subset.get(sample.subset, 0) + 1
        by_split[sample.split] = by_split.get(sample.split, 0) + 1
        key = str(len(sample.plate_text))
        by_length[key] = by_length.get(key, 0) + 1

    summary = {
        "root": root_label,
        "total_samples": len(samples),
        "total_issues": len(issues),
        "by_subset": dict(sorted(by_subset.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_plate_length": dict(sorted(by_length.items())),
        "sample_file": samples_path.as_posix(),
        "issue_file": issues_path.as_posix(),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：解析数据集目录，或者生成合成清单"""
    parser = argparse.ArgumentParser(description="CCPD 文件名解析与数据清单生成")
    parser.add_argument("--root", type=Path, help="CCPD 数据集根目录")
    parser.add_argument("--out", type=Path, required=True, help="清单输出目录")
    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="不读数据集，生成 N 条合成样本（无数据时验证流程用）",
    )
    args = parser.parse_args(argv)

    if args.synthetic > 0:
        rng = random.Random(42)
        samples: list[PlateSample] = []
        issues: list[ParseIssue] = []
        for index in range(args.synthetic):
            file_name = make_synthetic_filename(rng)
            path = Path("synthetic") / f"case{index:03d}" / file_name
            try:
                sample = build_sample(file_name, path, subset="synthetic")
                problems = check_sample(sample)
                if problems:
                    issues.append(ParseIssue(path, "；".join(problems)))
                else:
                    samples.append(sample)
            except ParseError as exc:
                issues.append(ParseIssue(path, str(exc)))
        summary = build_manifest(samples, issues, args.out, root_label="synthetic")
    else:
        if args.root is None:
            parser.error("需要 --root 指向数据集，或使用 --synthetic 生成合成样本")
        samples, issues = collect_samples(args.root)
        summary = build_manifest(samples, issues, args.out, root_label=args.root.as_posix())

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


