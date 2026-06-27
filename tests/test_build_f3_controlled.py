"""
test_build_f3_controlled -- 验证 controlled pair 筛逻辑（纯函数，不调 API）

F3 实验里"controlled pair"定义复杂（pingze 阈值 + 意境维度均值 + margin），
单测把这些边界全列出来防回归。
"""
import pytest

from eval.build_f3_controlled import find_controlled_pairs, _mean_intent_dims


def _cand(poem: str, pingze: float, imagery: float = 0.7,
          cohesion: float = 0.7, aesthetics: float = 0.7):
    return {
        "poem": poem,
        "scores": {
            "pingze": pingze, "rhyme": 0.8,
            "intent": 0.7,
            "imagery": imagery, "cohesion": cohesion, "aesthetics": aesthetics,
        },
    }


# ── _mean_intent_dims ─────────────────────────────────────────────────────
def test_mean_intent_only_uses_imagery_cohesion_aesthetics():
    """intent / pingze / rhyme 不进意境均值（按设计排除）。"""
    s = {"intent": 0.0, "pingze": 1.0, "rhyme": 1.0,
         "imagery": 0.6, "cohesion": 0.6, "aesthetics": 0.6}
    assert _mean_intent_dims(s) == pytest.approx(0.6)


def test_mean_intent_skips_none():
    """None 维度被跳过；剩余维度算均值。"""
    s = {"imagery": 0.8, "cohesion": None, "aesthetics": 0.6}
    assert _mean_intent_dims(s) == pytest.approx(0.7)


def test_mean_intent_all_none_returns_zero():
    """全 None → 0.0（后续筛逻辑过滤掉这种 case）。"""
    s = {"imagery": None, "cohesion": None, "aesthetics": None}
    assert _mean_intent_dims(s) == 0.0


# ── find_controlled_pairs ────────────────────────────────────────────────
def test_filters_out_base_with_passing_pingze():
    """base.pingze >= threshold → 不算"严重出律"，不进 controlled。"""
    base = [_cand("b", pingze=0.6, imagery=0.9)]  # 高于默认 0.5
    lora = [_cand("l", pingze=0.95, imagery=0.6)]
    assert find_controlled_pairs(base, lora) == []


def test_filters_out_when_lora_intent_higher():
    """base.意境均值 < lora.意境均值 → base 不"占意境优势"，不进 controlled。"""
    base = [_cand("b", pingze=0.2, imagery=0.5, cohesion=0.5, aesthetics=0.5)]
    lora = [_cand("l", pingze=0.95, imagery=0.8, cohesion=0.8, aesthetics=0.8)]
    assert find_controlled_pairs(base, lora) == []


def test_keeps_canonical_controlled_pair():
    """典型 F3 case：base 严重出律 (pingze=0.2) + 意境领先 → 进入池。"""
    base = [_cand("b", pingze=0.2, imagery=0.85, cohesion=0.8, aesthetics=0.85)]
    lora = [_cand("l", pingze=0.95, imagery=0.7, cohesion=0.7, aesthetics=0.7)]
    pairs = find_controlled_pairs(base, lora)
    assert len(pairs) == 1
    pr = pairs[0]
    assert pr["base_poem"] == "b"
    assert pr["lora_poem"] == "l"
    assert pr["base_intent_mean"] > pr["lora_intent_mean"]
    assert pr["pingze_diff"] < 0    # base 平仄差 → diff 是负数


def test_margin_blocks_borderline_pair():
    """base.意境 = lora.意境 时 margin=0 通过、margin>0 拦截。"""
    base = [_cand("b", pingze=0.2, imagery=0.7, cohesion=0.7, aesthetics=0.7)]
    lora = [_cand("l", pingze=0.95, imagery=0.7, cohesion=0.7, aesthetics=0.7)]
    assert len(find_controlled_pairs(base, lora, intent_margin=0.0)) == 1
    assert find_controlled_pairs(base, lora, intent_margin=0.05) == []


def test_cartesian_product_n_squared():
    """N base × N lora 候选笛卡尔积；所有 base 都极端出律且意境领先 →
    应得 N × N 对。"""
    base = [_cand(f"b{i}", pingze=0.2, imagery=0.9) for i in range(3)]
    lora = [_cand(f"l{i}", pingze=0.95, imagery=0.6) for i in range(3)]
    assert len(find_controlled_pairs(base, lora)) == 9


def test_none_pingze_skipped():
    """base.pingze=None（评分失败兜底）→ 不抛 TypeError，跳过该 base。"""
    base = [_cand("b", pingze=None, imagery=0.9)]
    base[0]["scores"]["pingze"] = None
    lora = [_cand("l", pingze=0.95)]
    assert find_controlled_pairs(base, lora) == []


def test_custom_pingze_threshold():
    """threshold=0.3：pingze=0.4 的 base 不再算"严重出律"。"""
    base = [_cand("b", pingze=0.4, imagery=0.9)]
    lora = [_cand("l", pingze=0.95, imagery=0.6)]
    assert find_controlled_pairs(base, lora, pingze_threshold=0.3) == []
    assert len(find_controlled_pairs(base, lora, pingze_threshold=0.5)) == 1
