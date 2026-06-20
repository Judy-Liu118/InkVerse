"""
eval.metrics -- 评估统计工具
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Sequence


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


__all__ = ["summarize", "paired_delta", "pass_rate"]
