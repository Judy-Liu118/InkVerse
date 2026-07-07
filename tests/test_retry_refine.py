"""
test_retry_refine -- CLIP 不达标时的阶梯式提示词精炼（_refine_prompt_for_retry）

第 1 次重试只保留内容承载段；第 2 次起剥掉段标签压成逗号短语，
但必须保留风格锚（style 后缀行 + Art Style 段），避免末次重试画风漂离。
"""
from core.agent.agent import PoetryAgent

EN_PROMPT = """Chinese ink wash painting, sumi-e, monochrome, minimalist, Song Dynasty style
Subject: lone fishing boat, misty river
Environment: cold river at dusk, light snow
Atmosphere: solitary, serene
Color Palette: ink monochrome, washed gray
Art Style: traditional Chinese ink wash painting
Composition: rule of thirds, vast negative space"""

CN_PROMPT = """主体: 孤舟蓑笠，寒江独钓
环境: 暮雪江面，远山如黛
氛围: 空寂清冷
色调: 水墨留白，淡墨微晕
艺术风格: 中国传统水墨画
构图: 三分法，大留白"""


def test_retry1_keeps_content_sections_only():
    out = PoetryAgent._refine_prompt_for_retry(EN_PROMPT, 1)
    assert out.startswith("clear composition, detailed,")
    assert "lone fishing boat" in out
    assert "Color Palette" not in out
    assert "Composition:" not in out


def test_retry2_strips_labels_but_keeps_style_anchor():
    out = PoetryAgent._refine_prompt_for_retry(EN_PROMPT, 2)
    assert "Subject:" not in out
    assert "lone fishing boat" in out
    # 风格锚双保留：style_suffix 前缀行 + Art Style 段值
    assert "sumi-e" in out
    assert "traditional Chinese ink wash painting" in out
    # 样板段（Color Palette / Composition）剥离
    assert "washed gray" not in out
    assert "rule of thirds" not in out


def test_retry1_matches_cn_headers():
    """中文模板段头（主体:/环境:/氛围:）也应命中，不再退化为整段原文。"""
    out = PoetryAgent._refine_prompt_for_retry(CN_PROMPT, 1)
    assert "孤舟蓑笠" in out
    assert "色调" not in out


def test_retry2_cn_keeps_style_drops_boilerplate():
    out = PoetryAgent._refine_prompt_for_retry(CN_PROMPT, 2)
    assert "主体" not in out
    assert "孤舟蓑笠" in out
    assert "中国传统水墨画" in out
    assert "水墨留白" not in out


def test_retry_fallback_on_unstructured_prompt():
    """无任何已知段头时不应崩溃，兜底返回可用文本。"""
    plain = "a lone boat drifting on a misty river at dusk"
    assert PoetryAgent._refine_prompt_for_retry(plain, 2) == plain
