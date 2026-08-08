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


# ── 诗评 prompt 里的 CLIP 分：无图阶段不得出现哨兵值 ──────────────────────
# 擂台进化跑在生图之前（autonomous_full_run 第 2 步 vs 第 4 步），此时
# clip_score_final=0.0，_raw_clip 返回 -1.0。曾把该哨兵直接拼进评委 prompt。
from core.agent.poem_refiner import _PoemRefineMixin
from core.agent.state import AgentState


class _CritiqueHarness(_PoemRefineMixin):
    """只为调 _auto_poem_critique：捕获发给评委的 messages，不发真实请求。"""

    def __init__(self):
        self.captured = None
        self.generation_adapter = None

        def _generate(messages, **kwargs):
            self.captured = messages
            return "点评正文"

        adapter = MagicMock()
        adapter.generate.side_effect = _generate
        self.score_adapter = adapter

    @staticmethod
    def _raw_clip(state):
        from core.agent.agent import PoetryAgent
        return PoetryAgent._raw_clip(state)


def _critique_user_msg(clip_score_final: float) -> str:
    h = _CritiqueHarness()
    state = AgentState()
    state.poem = "微飔穿竹径\n清响落云间\n松影摇溪浅\n苔痕浸水闲"
    state.clip_score_final = clip_score_final
    h._auto_poem_critique(state)
    return h.captured[1]["content"]


def test_critique_omits_clip_line_before_any_image():
    """擂台阶段尚未生图 → prompt 不提图文一致性，更不能出现 -1.000。"""
    msg = _critique_user_msg(0.0)
    assert "图文一致性" not in msg
    assert "-1.000" not in msg
    assert "微飔穿竹径" in msg


def test_critique_includes_clip_line_once_scored():
    """真有分数时照常提供（norm 0.65 → raw 0.30）。"""
    msg = _critique_user_msg(0.65)
    assert "图文一致性得分 0.300" in msg
