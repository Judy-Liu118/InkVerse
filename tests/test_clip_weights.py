"""
test_clip_weights -- CLIP 双锚点权重自适应逻辑
"""
import pytest
from core.agent.agent import PoetryAgent


@pytest.fixture
def cfg():
    """读取 config 里和权重相关的常量，避免硬编码。"""
    from config import (
        CLIP_POEM_WEIGHT, CLIP_PROMPT_WEIGHT,
        CLIP_SPARSE_POEM_WEIGHT, CLIP_SPARSE_PROMPT_WEIGHT,
        CLIP_SPARSE_WORD_THRESHOLD,
    )
    return {
        "normal": (CLIP_POEM_WEIGHT, CLIP_PROMPT_WEIGHT),
        "sparse": (CLIP_SPARSE_POEM_WEIGHT, CLIP_SPARSE_PROMPT_WEIGHT),
        "threshold": CLIP_SPARSE_WORD_THRESHOLD,
    }


def test_no_keywords_falls_back_to_prompt_only(cfg):
    """无诗锚点时，全权重给提示词锚点。"""
    wa, wb = PoetryAgent._clip_anchor_weights("")
    assert (wa, wb) == (0.0, 1.0)


def test_rich_keywords_use_normal_weights(cfg):
    """关键词丰富（>=阈值词）走标准权重 0.6 / 0.4。"""
    keywords = "spring, willow, river, swallow, breeze, dawn, mountain"
    wa, wb = PoetryAgent._clip_anchor_weights(keywords)
    assert (wa, wb) == cfg["normal"]


def test_sparse_keywords_switch_to_prompt_dominant(cfg):
    """关键词稀疏（哲理/抽象诗常见）应自动降诗锚权重。"""
    keywords = "mood, thought"  # 2 词
    wa, wb = PoetryAgent._clip_anchor_weights(keywords)
    assert (wa, wb) == cfg["sparse"]
    assert wa < cfg["normal"][0], "稀疏分支必须降低诗锚权重"
    assert wb > cfg["normal"][1], "稀疏分支必须升高提示词锚点权重"


def test_weights_always_sum_to_one(cfg):
    """两套权重之和都应该是 1.0（保证最终 CLIP 分仍在 [-1, 1] 区间）。"""
    assert sum(cfg["normal"]) == pytest.approx(1.0, abs=1e-6)
    assert sum(cfg["sparse"]) == pytest.approx(1.0, abs=1e-6)


def test_short_tokens_are_ignored():
    """单字符 token（如标点遗留）不应计入词数。"""
    # 7 个真实词 + 一堆单字符，应仍判定为 rich
    wa, wb = PoetryAgent._clip_anchor_weights("spring, , a, b, willow, river, swallow, breeze, dawn, mountain")
    from config import CLIP_POEM_WEIGHT
    assert wa == CLIP_POEM_WEIGHT
