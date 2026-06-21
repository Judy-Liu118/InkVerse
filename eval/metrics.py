"""
eval.metrics -- 评估统计工具
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """返回 mean / std / median / min / max / n。"""
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0,
                "min": 0.0, "max": 0.0}
    return {
        "n":      len(values),
        "mean":   statistics.fmean(values),
        "std":    statistics.pstdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min":    min(values),
        "max":    max(values),
    }


def paired_delta(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """配对差值统计：b - a 的每条差，再返回 summary + positive_rate。

    用于「同一条 input 跑 A 和 B」这种配对实验。
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"n": 0, "mean_delta": 0.0, "std_delta": 0.0,
                "positive_rate": 0.0, "win_rate_b": 0.0}
    deltas = [y - x for x, y in pairs]
    pos = sum(1 for d in deltas if d > 0)
    return {
        "n":             len(deltas),
        "mean_delta":    statistics.fmean(deltas),
        "std_delta":     statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
        "positive_rate": pos / len(deltas),     # b > a 的比例
        "win_rate_b":    pos / len(deltas),     # alias
        "median_delta":  statistics.median(deltas),
    }


def pass_rate(values: Sequence[float], threshold: float) -> float:
    """通过率：values 中 >= threshold 的比例。"""
    values = [v for v in values if v is not None]
    if not values:
        return 0.0
    return sum(1 for v in values if v >= threshold) / len(values)


def _paired(a: Sequence[float], b: Sequence[float]):
    """对齐两列，剔除任一为 None 的行；返回 (xs, ys)。"""
    xs, ys = [], []
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def _ranks(values: Sequence[float]) -> List[float]:
    """计算 fractional ranks（同值取平均秩），用于 Spearman 相关系数。"""
    indexed = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0   # 秩从 1 起
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Pearson 线性相关系数 ∈ [-1, 1]。样本不足或方差为 0 返回 None。"""
    xs, ys = _paired(a, b)
    if len(xs) < 2:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx == 0 or dy == 0:
        return None
    return num / (dx ** 0.5 * dy ** 0.5)


def spearman_corr(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Spearman 秩相关系数 ∈ [-1, 1]：等价于秩转换后的 Pearson。
    更鲁棒、不要求线性，适合"判图文契合度"这种序数关系评估。
    """
    xs, ys = _paired(a, b)
    if len(xs) < 2:
        return None
    return pearson_corr(_ranks(xs), _ranks(ys))


__all__ = [
    "summarize", "paired_delta", "pass_rate",
    "pearson_corr", "spearman_corr",
]
