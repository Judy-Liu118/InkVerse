"""
test_scorer -- PoemScorer 的本地规则维度（不依赖 LLM）
覆盖：平仄、押韵、堆砌词硬门控、合掌检测、多评委 intent 合成。
"""
import pytest
from unittest.mock import MagicMock
from core.poem.scorer import PoemScorer, _SYNONYM_CLASH_GROUPS


@pytest.fixture(scope="module")
def scorer():
    return PoemScorer()


# ── 平仄 ────────────────────────────────────────────────────────────────────
def test_pingze_returns_unit_interval(scorer):
    """合规的近体诗平仄分应落在 [0, 1]。"""
    poem = "春眠不觉晓\n处处闻啼鸟\n夜来风雨声\n花落知多少"
    score = scorer._score_pingze(poem, num_lines=4, chars_per_line=5)
    assert 0.0 <= score <= 1.0


def test_pingze_garbage_text_low(scorer):
    """随机汉字应给一个较低的平仄分（不会触发 KeyError 等异常）。"""
    poem = "啊啊啊啊啊\n啊啊啊啊啊\n啊啊啊啊啊\n啊啊啊啊啊"
    score = scorer._score_pingze(poem, num_lines=4, chars_per_line=5)
    assert isinstance(score, float)
    assert score <= 1.0


# ── 押韵 ────────────────────────────────────────────────────────────────────
def test_rhyme_returns_unit_interval(scorer):
    poem = "春眠不觉晓\n处处闻啼鸟\n夜来风雨声\n花落知多少"
    score = scorer._score_rhyme(poem, num_lines=4)
    assert 0.0 <= score <= 1.0


def test_rhyme_non_rhyming_low(scorer):
    """全诗韵脚不押的诗应得 0 或接近 0。"""
    poem = "山高水又长\n树绿草青青\n云白风轻轻\n月明星不归"
    score = scorer._score_rhyme(poem, num_lines=4)
    assert score < 0.7


# ── 同义合掌库 ─────────────────────────────────────────────────────────────
def test_synonym_clash_groups_non_empty():
    """合掌库不能为空 —— 是改诗质量护栏的核心数据。"""
    assert len(_SYNONYM_CLASH_GROUPS) >= 5
    for group in _SYNONYM_CLASH_GROUPS:
        assert isinstance(group, set)
        assert len(group) >= 2, "合掌组内至少两个同义词才有意义"


def test_synonym_clash_groups_disjoint():
    """合掌组之间不应有重叠词，避免分类二义。"""
    seen = set()
    for group in _SYNONYM_CLASH_GROUPS:
        overlap = seen & group
        assert not overlap, f"重叠词: {overlap}"
        seen |= group


# ── BAD_PATTERNS 黑名单 ────────────────────────────────────────────────────
def test_bad_patterns_loaded():
    """堆砌词黑名单是硬门控的核心，必须有内容。"""
    from config import BAD_PATTERNS
    assert isinstance(BAD_PATTERNS, set)
    assert len(BAD_PATTERNS) >= 5
    # 抽检几个标志性堆砌词
    assert "璀璨" in BAD_PATTERNS


# ── 必须意象切分（回归 bug：贪婪 {1,4} 把"柳树和燕子"切成["柳树和燕","子"]）──
def test_required_keywords_split_by_connector(scorer):
    """要求 "柳树和燕子" 应被切成 [柳树, 燕子] 两个 item，而不是 [柳树和燕, 子]。

    诗里出现 "柳" 和 "燕"（都是古诗惯用语），应满足两项要求 → coeff=1.0。
    """
    user_req = "写一首春天的五言绝句，要有柳树和燕子"
    poem = "春风吹柳舞\n燕归人未回\n柳条经雨后\n故作弄风来"
    coeff = scorer._check_required_keywords(poem, user_req)
    assert coeff == 1.0, f"期待 coeff=1.0（柳/燕都命中），实际={coeff}"


def test_required_keywords_synonym_yan_zi_matches_yan(scorer):
    """'燕子' 通过同义词表能命中 '燕'（古诗里 燕 比 燕子 更常见）。"""
    from core.poem.theme import get_imagery_synonyms
    syns = set(get_imagery_synonyms("燕子"))
    assert "燕" in syns, f"燕子 同义词表应包含 '燕'，实际={syns}"


def test_required_keywords_liu_shu_matches_liu(scorer):
    """'柳树' 通过同义词表命中 '柳'/'杨柳' 等古诗惯用语。"""
    from core.poem.theme import get_imagery_synonyms
    syns = set(get_imagery_synonyms("柳树"))
    assert "柳" in syns and "杨柳" in syns


def test_required_keywords_missing_imagery_penalized(scorer):
    """诗里没出现任何'柳/燕'同义词时，应触发 ×0.75 惩罚。"""
    user_req = "写一首春天的五言绝句，要有柳树和燕子"
    poem = "桃花满径开\n春水绕村流\n远山含黛色\n孤舟泛碧波"
    coeff = scorer._check_required_keywords(poem, user_req)
    # 两个 item 都缺 → 0.75 × 0.75 = 0.5625
    assert coeff < 0.6, f"期待两项都缺，coeff 应 < 0.6，实际={coeff}"


# ── B 方案：LLM 4 维评分 + 多评委中位数 ─────────────────────────────────
def _fake_judge_4dim(intent=3, imagery=3, cohesion=2, aesthetics=2, total=None):
    """构造 mock judge：可单独控制 4 维分。
    intent/imagery 满分 3，cohesion/aesthetics 满分 2，total 默认 = 4 维加和。
    """
    if total is None:
        total = intent + imagery + cohesion + aesthetics
    adapter = MagicMock()
    adapter.backend = "qwen"
    adapter.generate.return_value = (
        f"主题匹配度:{intent},意象完整性:{imagery},"
        f"意境连贯度:{cohesion},语言优美度:{aesthetics},总分:{total}"
    )
    return adapter


def test_parse_reply_returns_4dim_dict(scorer):
    """parser 应返回 4 维归一化 dict + total。"""
    reply = "主题匹配度:3,意象完整性:1.5,意境连贯度:2,语言优美度:1.5,总分:8"
    parsed = scorer._parse_llm_score_reply(reply)
    assert parsed is not None
    assert abs(parsed["intent"]     - 1.0) < 1e-6   # 3/3
    assert abs(parsed["imagery"]    - 0.5) < 1e-6   # 1.5/3
    assert abs(parsed["cohesion"]   - 1.0) < 1e-6   # 2/2
    assert abs(parsed["aesthetics"] - 0.75) < 1e-6  # 1.5/2
    assert abs(parsed["total"]      - 0.8) < 1e-6   # 8/10


def test_parse_reply_missing_component_falls_back_to_total(scorer):
    """缺失维度时用 total 兜底，不让单维度查询崩。"""
    reply = "总分:7"
    parsed = scorer._parse_llm_score_reply(reply)
    assert parsed["total"] == 0.7
    assert parsed["intent"] == 0.7  # 兜底


def test_parse_reply_empty_returns_none(scorer):
    assert scorer._parse_llm_score_reply("") is None
    assert scorer._parse_llm_score_reply(None) is None


def test_multi_judge_2_judges_takes_mean(scorer):
    """2 评委时 intent/imagery/cohesion/aesthetics 都取均值。"""
    poem = "春风吹柳舞\n燕子归巢忙\n桃花笑水落\n人间四月天"
    # judge_a 给 intent=3 imagery=3 cohesion=2 aesthetics=2 → 全 1.0
    # judge_b 给 intent=2 imagery=2 cohesion=1 aesthetics=1 → 各 0.667/0.667/0.5/0.5
    judge_a = _fake_judge_4dim(intent=3, imagery=3, cohesion=2, aesthetics=2)
    judge_b = _fake_judge_4dim(intent=2, imagery=2, cohesion=1, aesthetics=1)
    result = scorer.score_single_multi_judge(
        poem, "春诗", [("a", judge_a), ("b", judge_b)],
    )
    # 2 评委取均值
    assert result["aggregation_method"] == "mean"
    assert abs(result["intent"]     - (1.0 + 2.0/3) / 2) < 1e-3
    assert abs(result["imagery"]    - (1.0 + 2.0/3) / 2) < 1e-3
    assert abs(result["cohesion"]   - (1.0 + 0.5) / 2)   < 1e-3
    assert abs(result["aesthetics"] - (1.0 + 0.5) / 2)   < 1e-3


def test_multi_judge_3_judges_takes_median(scorer):
    """3 评委时取中位数（对异常打分更健壮）。"""
    poem = "春风吹柳舞\n燕子归巢忙\n桃花笑水落\n人间四月天"
    judge_high   = _fake_judge_4dim(intent=3, imagery=3, cohesion=2, aesthetics=2)  # 全 1.0
    judge_mid    = _fake_judge_4dim(intent=2, imagery=2, cohesion=1, aesthetics=1)  # 2/3, 2/3, 0.5, 0.5
    judge_outlier = _fake_judge_4dim(intent=0, imagery=0, cohesion=0, aesthetics=0)  # 全 0
    result = scorer.score_single_multi_judge(
        poem, "春诗",
        [("hi", judge_high), ("mid", judge_mid), ("low", judge_outlier)],
    )
    assert result["aggregation_method"] == "median"
    # 中位数应是中间那个（judge_mid 的分），不被极端值影响
    assert abs(result["intent"]     - 2.0/3) < 1e-3
    assert abs(result["imagery"]    - 2.0/3) < 1e-3
    assert abs(result["cohesion"]   - 0.5) < 1e-3
    assert abs(result["aesthetics"] - 0.5) < 1e-3


def test_multi_judge_records_disagreement_per_dim(scorer):
    """judge_disagreement 应记录每维度的 max - min spread。"""
    poem = "春风吹柳舞\n燕子归巢忙\n桃花笑水落\n人间四月天"
    judge_a = _fake_judge_4dim(intent=3, imagery=1, cohesion=2, aesthetics=2)  # i=1.0 im=0.33 co=1.0 a=1.0
    judge_b = _fake_judge_4dim(intent=2, imagery=3, cohesion=2, aesthetics=1)  # i=0.67 im=1.0 co=1.0 a=0.5
    result = scorer.score_single_multi_judge(poem, "春诗",
        [("a", judge_a), ("b", judge_b)])
    d = result["judge_disagreement"]
    assert d["intent"]     > 0.3   # 1.0 - 0.67 ≈ 0.33
    assert d["imagery"]    > 0.6   # 1.0 - 0.33 ≈ 0.67
    assert d["cohesion"]   < 0.01  # 一致
    assert abs(d["aesthetics"] - 0.5) < 1e-3  # 1.0 - 0.5


def test_multi_judge_full_scores_by_judge_visible(scorer):
    """scores_by_judge 应保留每个 judge 的 4 维明细，便于报告。"""
    poem = "春风吹柳舞\n燕子归巢忙\n桃花笑水落\n人间四月天"
    judge = _fake_judge_4dim(intent=3, imagery=2, cohesion=1, aesthetics=2)
    result = scorer.score_single_multi_judge(poem, "春诗", [("solo", judge)])
    s = result["scores_by_judge"]["solo"]
    assert s["intent"]     == 1.0
    assert abs(s["imagery"]    - 2.0/3) < 1e-3
    assert s["cohesion"]   == 0.5
    assert s["aesthetics"] == 1.0


def test_multi_judge_empty_raises(scorer):
    with pytest.raises(ValueError):
        scorer.score_single_multi_judge("春风吹柳舞\n燕子归巢忙\n桃花笑水落\n人间四月天",
                                         "any", [])


def test_multi_judge_failing_judge_falls_back_to_0_5(scorer):
    """一个 judge 抛异常时降级为 0.5，不阻断流程。"""
    poem = "春风吹柳舞\n燕子归巢忙\n桃花笑水落\n人间四月天"
    judge_ok   = _fake_judge_4dim(intent=3, imagery=3, cohesion=2, aesthetics=2)
    judge_fail = MagicMock()
    judge_fail.backend = "qwen"
    judge_fail.generate.side_effect = RuntimeError("api down")
    result = scorer.score_single_multi_judge(
        poem, "春诗", [("ok", judge_ok), ("fail", judge_fail)],
    )
    # ok 给 1.0；fail 兜底到 0.5；均值 0.75
    assert abs(result["intent"] - 0.75) < 1e-3
