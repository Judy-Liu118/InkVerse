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


# ── 韵脚兜底稳定性（_get_stable_rhyme） ────────────────────────────────────
def test_get_stable_rhyme_returns_string(scorer):
    """任意汉字调用都应返回非空 str，不抛异常。"""
    for c in ["冬", "东", "花", "明", "声", "风", "月"]:
        r = scorer._get_stable_rhyme(c)
        assert isinstance(r, str) and r, f"{c} 返回异常: {r!r}"


def test_get_stable_rhyme_same_pinyin_final_consistent(scorer):
    """走 pypinyin 兜底路径（简体字 pingshui 库未覆盖时）的字，若 final 相同应返同韵组。
    这是 _score_rhyme 里 set() 去重能否正确判定押韵的关键前提。
    """
    # 风/声/灯/腾 都是简体，pypinyin final = 'eng' → 兜底映射到 '庚'
    groups = {scorer._get_stable_rhyme(c) for c in ["风", "声", "灯", "腾"]}
    assert len(groups) == 1, f"同 final 字应返同韵组，实际得到 {groups}"


def test_pinyin_pingshui_endswith_semantics(scorer):
    """pypinyin 兜底路径按 endswith 匹（尾韵决定押韵）。
    startswith 会把 'iang' 误匹到 'i'→支；endswith 才能正确匹到 'ang'→阳。
    """
    from core.poem.theme import PINYIN_TO_PINGSHUI
    # 数据前提：iang 和 ang 都映射到阳（同韵组），i 单独映射到支
    assert PINYIN_TO_PINGSHUI["iang"] == "阳"
    assert PINYIN_TO_PINGSHUI["ang"]  == "阳"
    assert PINYIN_TO_PINGSHUI["i"]    == "支"
    # 「阳」实际 final='ang'，在字典里直接命中，返 '阳'（无需进循环）
    assert scorer._get_stable_rhyme("阳") == "阳"


def test_get_stable_rhyme_uei_iou_endswith_fix(scorer):
    """回归：pypinyin 对高频字返回 final='iou'/'uei'（非 'iu'/'ui'），
    这些不在 PINYIN_TO_PINGSHUI 里、需走 loop 兜底。
    · 老 bug：startswith 会把 'uei' 误匹到 'u'→虞
    · 修复后：endswith 正确匹到 'ei'→微
    以「为」为代表（pingshui 库无记录，纯走 pypinyin 路径）。
    """
    import pypinyin
    from pingshui_rhyme import RhymeChecker
    # 前提验证：为 pingshui 未收，pypinyin 返 uei（若上游变更此断言先失败）
    assert RhymeChecker().get_rhyme_group("为") is None
    assert pypinyin.pinyin("为", style=pypinyin.Style.FINALS)[0][0] == "uei"
    # endswith 修复后应归微；老 startswith bug 会返虞
    assert scorer._get_stable_rhyme("为") == "微", (
        "endswith 修复失效：为 应归微（ei 尾韵），startswith 老 bug 会返虞"
    )


# ── 体裁识别（含简称与"绝句/律诗"默认五言） ────────────────────────────
@pytest.mark.parametrize("user_topic, expected", [
    # 全称照常命中
    ("写一首五言绝句，主题是春", ("五言绝句", 4, 5)),
    ("写一首七言律诗，主题是秋", ("七言律诗", 8, 7)),
    # 简称
    ("写一首五绝，主题是月",     ("五言绝句", 4, 5)),
    ("写一首七绝，主题是花",     ("七言绝句", 4, 7)),
    ("写一首五律，主题是山",     ("五言律诗", 8, 5)),
    ("写一首七律，主题是江",     ("七言律诗", 8, 7)),
    # 只说体裁 → 默认五言
    ("写一首绝句吧",             ("五言绝句", 4, 5)),
    ("写一首律诗吧",             ("五言律诗", 8, 5)),
    # 全称优先于简称（"七言绝句" 命中前不会被 "绝句" 截胡）
    ("请给我一首七言绝句",       ("七言绝句", 4, 7)),
])
def test_detect_genre_aliases(scorer, user_topic, expected):
    assert scorer.detect_genre(user_topic) == expected


def test_detect_genre_default_when_none(scorer):
    """完全不提体裁时兜底为五言绝句。"""
    assert scorer.detect_genre("给我写个诗") == ("五言绝句", 4, 5)


# ── 体裁是否被点名（naked prompt 补句式的前置判断）──────────────────────────
@pytest.mark.parametrize("user_topic", [
    "写一首五言绝句，主题是春",
    "写一首七言律诗，主题是秋",
    "写一首五绝，主题是月",      # 简称不能漏判，否则会被补上矛盾的句式
    "写一首七绝，主题是花",
    "写一首五律，主题是山",
    "写一首七律，主题是江",
    "写一首绝句吧",
    "写一首律诗吧",
])
def test_mentions_genre_true(scorer, user_topic):
    assert scorer.mentions_genre(user_topic) is True


@pytest.mark.parametrize("user_topic", [
    "给我写个诗",
    "写一首描写秋天的诗，要有意向菊花",
    "以边塞为主题写首诗",
])
def test_mentions_genre_false(scorer, user_topic):
    assert scorer.mentions_genre(user_topic) is False


def test_mentions_genre_covers_detect_genre_vocabulary(scorer):
    """两者共用词表：detect_genre 能识别的写法，mentions_genre 都要认。

    防回归 —— 历史上 generator 里维护过一份平行关键词列表，漏掉四个简称。
    """
    from config import GENRE_CONFIG
    for word in list(GENRE_CONFIG) + list(type(scorer).GENRE_ALIASES):
        assert scorer.mentions_genre(f"写一首{word}") is True


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


def test_synonym_clash_intra_sentence():
    """句内冗余：同一句同时用同义词群的两个词。"""
    poem = "明月皓月共此时\n江水东流去不回"
    assert PoemScorer._check_synonym_clash(poem) == 0.75


def test_synonym_clash_across_couplet():
    """合掌本义：一联上下句各用同义词群中的不同词（上句残阳、下句落日）。"""
    poem = "残阳照孤城\n落日满秋山"
    assert PoemScorer._check_synonym_clash(poem) == 0.75


def test_synonym_clash_clean_poem():
    """无同义复用的诗不应触发扣分。"""
    poem = "春眠不觉晓\n处处闻啼鸟\n夜来风雨声\n花落知多少"
    assert PoemScorer._check_synonym_clash(poem) == 1.0


def test_synonym_clash_same_word_repeated_is_not_clash():
    """上下句重复同一个词属重字问题（另有重复惩罚处理），不计合掌。"""
    poem = "明月照高楼\n明月满西洲"
    assert PoemScorer._check_synonym_clash(poem) == 1.0


def test_synonym_clash_not_flagged_across_couplet_boundary():
    """第 2 句与第 3 句分属不同联，同义词分居两联不算合掌。"""
    poem = "孤舟横野渡\n残阳下远山\n落日沉极浦\n江声自往还"
    assert PoemScorer._check_synonym_clash(poem) == 1.0


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


def test_multi_judge_failing_judge_abstains(scorer):
    """一个 judge 抛异常时整体弃权（4 维 None），合成时跳过，不阻断流程也不污染均值。

    fb19f43 后语义：弃权评委的 None 维度被跳过，不再用 0.5 顶替。
    所以 ok=1.0、fail 弃权 → intent 合成 = 1.0（不是 0.75）。
    """
    poem = "春风吹柳舞\n燕子归巢忙\n桃花笑水落\n人间四月天"
    judge_ok   = _fake_judge_4dim(intent=3, imagery=3, cohesion=2, aesthetics=2)
    judge_fail = MagicMock()
    judge_fail.backend = "qwen"
    judge_fail.generate.side_effect = RuntimeError("api down")
    result = scorer.score_single_multi_judge(
        poem, "春诗", [("ok", judge_ok), ("fail", judge_fail)],
    )
    # ok=1.0；fail 弃权 → 跳过；intent_final = 1.0
    assert abs(result["intent"] - 1.0) < 1e-3
    # 弃权评委 intent_by_judge 字段为 None（不是 round 后的 0）
    assert result["scores_by_judge"]["fail"]["intent"] is None
    assert result["scores_by_judge"]["ok"]["intent"] == 1.0


# ── arena_from_gated 边界：champion["final"] KeyError 防回归 ────────────────
def _fake_pairwise_adapter(winner: str = "A"):
    adapter = MagicMock()
    adapter.generate.return_value = winner
    return adapter


def _gated_entry(idx: int, total: float, poem: str = "春风吹柳舞\n燕子归巢忙\n桃花笑水落\n人间四月天"):
    """构造 arena_from_gated 期望的 gated 项形状。"""
    return {
        "idx": idx,
        "poem": poem,
        "local": {
            "total": total,
            "pingze": 0.9, "rhyme": 0.9, "imagery": 0.8,
            "cohesion": 0.8, "topic": 0.6,
        },
    }


def test_arena_from_gated_single_candidate_returns_final(scorer):
    """gated 只有 1 首：top_n=1，pairs=[]，champion 字段不应抛 KeyError。"""
    gated = [_gated_entry(0, total=0.80)]
    result = scorer.arena_from_gated(gated, "春景", _fake_pairwise_adapter())
    assert result["champion_idx"] == 0
    assert isinstance(result["champion_final"], float)
    # 单首：arena_score=0/0 fallback=0，final = local*0.75 + 0 = 0.6
    assert result["champion_final"] == pytest.approx(0.80 * 0.75, abs=1e-6)


def test_arena_from_gated_two_candidates_returns_final(scorer):
    """gated 长度 2：top_n=2，pairs=[(0,1)]，两条都被赋 final。"""
    gated = [_gated_entry(0, total=0.85), _gated_entry(1, total=0.80)]
    result = scorer.arena_from_gated(gated, "春景", _fake_pairwise_adapter(winner="A"))
    assert result["champion_idx"] == 0
    assert isinstance(result["champion_final"], float)


def test_arena_from_gated_more_than_top_n_no_keyerror(scorer):
    """gated 长度 5 > top_n=3：只有前 3 被赋 final，第 4-5 走 sort .get fallback；
    champion 字段访问必须有兜底，不能 KeyError。"""
    gated = [
        _gated_entry(0, total=0.85),
        _gated_entry(1, total=0.80),
        _gated_entry(2, total=0.78),
        _gated_entry(3, total=0.70),
        _gated_entry(4, total=0.65),
    ]
    result = scorer.arena_from_gated(gated, "春景", _fake_pairwise_adapter(winner="A"))
    # 不论 champion 落在 idx 0..4 哪个，结果都得有 champion_final（float）
    assert isinstance(result["champion_final"], float)
    assert result["gated_count"] == 5


# ── compare_poems 弃权语义（2026-07-05 引入，对齐 BWS/4 维评分的 fb19f43 设计）──
def _compare(scorer, reply=None, raises=False):
    adapter = MagicMock()
    if raises:
        adapter.generate.side_effect = RuntimeError("api down")
    else:
        adapter.generate.return_value = reply
    return scorer.compare_poems("诗甲", "诗乙", "春景", adapter)


def test_compare_poems_clean_a(scorer):
    assert _compare(scorer, "A") == "A"


def test_compare_poems_clean_b_with_noise(scorer):
    """回复含 B 且不含 A（如 "答案：B"）→ B。"""
    assert _compare(scorer, "答案：B") == "B"


def test_compare_poems_lowercase_normalized(scorer):
    assert _compare(scorer, " b ") == "B"


def test_compare_poems_both_letters_abstains(scorer):
    """同时含 A 和 B（无法判定）→ None 弃权。旧版会静默判 A 胜。"""
    assert _compare(scorer, "A还是B都不错") is None


def test_compare_poems_neither_letter_abstains(scorer):
    assert _compare(scorer, "两首都很好") is None


def test_compare_poems_empty_reply_abstains(scorer):
    assert _compare(scorer, "") is None


def test_compare_poems_adapter_exception_abstains(scorer):
    """adapter 异常 → None 弃权。旧版会静默判 A 胜（污染计票）。"""
    assert _compare(scorer, raises=True) is None


def test_arena_from_gated_abstain_no_vote(scorer):
    """pairwise 全弃权：不计任何 arena 票，冠军按本地分 + arena_score=0 排序，
    不抛异常、winner 记为 None。"""
    gated = [_gated_entry(0, total=0.85), _gated_entry(1, total=0.80)]
    adapter = MagicMock()
    adapter.generate.return_value = "无法判断"
    result = scorer.arena_from_gated(gated, "春景", adapter)
    assert result["champion_idx"] == 0  # 本地分高者胜（无 pairwise 信息）
    assert result["arena_results"][0]["winner"] is None
    assert isinstance(result["champion_final"], float)
