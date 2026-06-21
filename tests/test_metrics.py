"""
test_metrics -- eval.metrics 中相关性 / 配对差值 / 通过率的数值正确性
"""
import math

import pytest

from eval.metrics import (
    pearson_corr, spearman_corr, paired_delta, summarize, pass_rate,
)


# ── pearson_corr ──────────────────────────────────────────────────────────
def test_pearson_perfect_positive():
    assert pearson_corr([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)


def test_pearson_perfect_negative():
    assert pearson_corr([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_no_correlation_handled():
    """方差为 0 的列应返回 None，不抛 ZeroDivisionError。"""
    assert pearson_corr([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert pearson_corr([1, 2, 3, 4], [5, 5, 5, 5]) is None


def test_pearson_skips_none_pairs():
    """任一为 None 的配对应被剔除。"""
    r = pearson_corr([1, 2, None, 4, 5], [2, 4, 6, None, 10])
    # 只剩 (1,2),(2,4),(5,10) 三对 → 完全线性
    assert r == pytest.approx(1.0)


def test_pearson_too_few_returns_none():
    assert pearson_corr([1], [2]) is None
    assert pearson_corr([], []) is None


# ── spearman_corr ─────────────────────────────────────────────────────────
def test_spearman_monotonic_nonlinear():
    """y = x^3 不是线性，但单调递增 → Spearman = 1。"""
    xs = [1, 2, 3, 4, 5]
    ys = [x ** 3 for x in xs]
    assert spearman_corr(xs, ys) == pytest.approx(1.0)
    # Pearson 此时 < 1（因为非线性）
    assert pearson_corr(xs, ys) < 1.0


def test_spearman_handles_ties():
    """相同值给平均秩，不应崩。"""
    r = spearman_corr([1, 2, 2, 3], [10, 20, 20, 30])
    assert r == pytest.approx(1.0)


def test_spearman_constant_returns_none():
    assert spearman_corr([1, 1, 1], [1, 2, 3]) is None


# ── paired_delta ──────────────────────────────────────────────────────────
def test_paired_delta_basic():
    d = paired_delta([1, 2, 3], [2, 4, 5])
    assert d["n"] == 3
    assert d["mean_delta"] == pytest.approx(5/3)
    assert d["positive_rate"] == 1.0


def test_paired_delta_empty():
    d = paired_delta([], [])
    assert d["n"] == 0


# ── summarize ─────────────────────────────────────────────────────────────
def test_summarize_basic():
    s = summarize([1, 2, 3, 4, 5])
    assert s["n"] == 5
    assert s["mean"] == 3
    assert s["min"] == 1 and s["max"] == 5


def test_summarize_filters_none():
    s = summarize([1, None, 3])
    assert s["n"] == 2
    assert s["mean"] == 2


# ── pass_rate ─────────────────────────────────────────────────────────────
def test_pass_rate_basic():
    assert pass_rate([0.1, 0.5, 0.8, 0.9], 0.5) == 0.75
