from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# 1) 省份简称（含 警、学）——索引 0..32
PROVINCE_CHARS = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "警", "学",
]

# 2) 数据集字符表里存在、但真实车牌不使用的两个易混字母——索引 33..34
SYMBOL_CHARS = ["O", "I"]

# 3) 可用字母（去掉 I、O，避免与 1、0 混淆）——索引 35..58
LETTER_CHARS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")

# 4) 数字——索引 59..68
DIGIT_CHARS = [str(d) for d in range(10)]

CHARSET: list[str] = PROVINCE_CHARS + SYMBOL_CHARS + LETTER_CHARS + DIGIT_CHARS
CHAR_TO_INDEX: dict[str, int] = {ch: i for i, ch in enumerate(CHARSET)}
BLANK_INDEX: int = len(CHARSET)
NUM_CLASSES: int = len(CHARSET) + 1

def encode_text(text: str) -> list[int]:
    """将车牌文本编码为索引列表"""
    return [CHAR_TO_INDEX[ch] for ch in text]

def decode_indices(indices: Sequence[int]) -> str:
    """将索引列表解码为车牌文本"""
    return "".join(CHARSET[i] for i in indices)

def ctc_collapse(path: Sequence[int], blank_index: int = BLANK_INDEX) -> str:
    """CTC 解码：去掉重复字符和空白索引"""
    collapsed: list[int] = []
    previous = None
    for index in path:
        if index != previous:
            collapsed.append(index)
        previous = index
    return "".join(CHARSET[i] for i in collapsed if i != blank_index)

def greedy_decode(probs: np.ndarray, blank_index: int = BLANK_INDEX) -> tuple[str, list[int], float]:
    """"贪心解码：每个时间步取概率最大的类别，再做CTC折叠"""
    if probs.ndim != 2:
        raise ValueError(f"probs 需要二维 (T, C)，收到 {probs.shape}")
    path = [int(i) for i in probs.argmax(axis=1)]
    text = ctc_collapse(path, blank_index)
    confidence = float(probs[np.arange(len(path)), path].mean()) if path else 0.0
    return text, path, confidence

def min_time_steps(text: str) -> int:
    """该文本在CTC下最少需要多少个时间步"""
    repeats = sum(1 for a, b in zip(text, text[1:]) if a == b)
    return len(text) + repeats

def labels_to_batch(texts: Iterable[str], pad_value: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """一批文本 -> (填充后的索引矩阵 (N, Lmax), 真实长度 (N,))"""
    encoded = [encode_text(t) for t in texts]
    if not encoded:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    max_len = max(len(item) for item in encoded)
    matrix = np.full((len(encoded), max_len), pad_value, dtype=np.int64)
    lengths = np.zeros((len(encoded),), dtype=np.int64)
    for row, item in enumerate(encoded):
        matrix[row, : len(item)] = item
        lengths[row] = len(item)
    return matrix, lengths

def validate_label(text: str) -> list[str]:
    """检查一条车牌文本是否符合号牌规则，返回问题列表（空列表=没问题）"""

    problems: list[str] = []
    if len(text) not in (7, 8):
        problems.append(f"长度应为 7 或 8，实际 {len(text)}")
    if text and text[0] not in PROVINCE_CHARS:
        problems.append(f"首字符 {text[0]!r} 不是省份简称")
    if len(text) > 1 and text[1] not in LETTER_CHARS:
        problems.append(f"第二位 {text[1]!r} 应为字母")
    for position, char in enumerate(text):
        if char not in CHAR_TO_INDEX:
            problems.append(f"第 {position} 位 {char!r} 不在字符表内")
    return problems


