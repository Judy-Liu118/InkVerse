"""
test_poem_refiner -- 擂台 pairwise A/B 位随机化（2026-07-06 B3a）

不触发真实 LLM：stub scorer 验证位置分配、胜负映射、弃权透传。
背景：固定"擂主A/挑战者B"布局会把 judge 位置偏置计入攻擂率
（sweep 报告 §6.3 #1），随机化把位置效应中和成对称噪声。
"""
from unittest.mock import MagicMock

from core.agent.poem_refiner import pairwise_judge_randomized


class _FixedRng:
    """rng.random() 返回固定值：<0.5 → 挑战者在 B 位；≥0.5 → A 位。"""

    def __init__(self, value: float):
        self._value = value

    def random(self) -> float:
        return self._value


def _scorer(reply):
    scorer = MagicMock()
    scorer.compare_poems.return_value = reply
    return scorer


def test_challenger_at_b_wins_when_judge_says_b():
    scorer = _scorer("B")
    won, pos = pairwise_judge_randomized(
        scorer, "擂主诗", "挑战诗", "春景", None, rng=_FixedRng(0.2))
    assert (won, pos) == (True, "B")
    # B 位布局：champion 传在 A 位（第一个参数）
    args = scorer.compare_poems.call_args[0]
    assert args[0] == "擂主诗" and args[1] == "挑战诗"


def test_challenger_at_b_loses_when_judge_says_a():
    won, pos = pairwise_judge_randomized(
        _scorer("A"), "擂主诗", "挑战诗", "春景", None, rng=_FixedRng(0.2))
    assert (won, pos) == (False, "B")


def test_challenger_at_a_wins_when_judge_says_a():
    scorer = _scorer("A")
    won, pos = pairwise_judge_randomized(
        scorer, "擂主诗", "挑战诗", "春景", None, rng=_FixedRng(0.9))
    assert (won, pos) == (True, "A")
    # A 位布局：challenger 传在第一个参数
    args = scorer.compare_poems.call_args[0]
    assert args[0] == "挑战诗" and args[1] == "擂主诗"


def test_challenger_at_a_loses_when_judge_says_b():
    won, pos = pairwise_judge_randomized(
        _scorer("B"), "擂主诗", "挑战诗", "春景", None, rng=_FixedRng(0.9))
    assert (won, pos) == (False, "A")


def test_abstain_passes_through_as_none():
    """compare_poems 弃权（None）→ chal_won=None，位置照常返回。"""
    won, pos = pairwise_judge_randomized(
        _scorer(None), "擂主诗", "挑战诗", "春景", None, rng=_FixedRng(0.2))
    assert won is None and pos == "B"


def test_position_assignment_is_balanced_with_seeded_rng():
    """seeded rng 下位置分配应接近 50/50（中和位置偏置的前提）。"""
    import random
    rng = random.Random(42)
    positions = [
        pairwise_judge_randomized(_scorer("A"), "甲", "乙", "题", None, rng=rng)[1]
        for _ in range(200)
    ]
    b_ratio = positions.count("B") / len(positions)
    assert 0.4 < b_ratio < 0.6
