"""
test_prompts -- 集中化 prompt loader 行为
"""
import pytest

from core.prompts import (
    PROMPTS_DIR, render_messages, list_prompts, get_metadata, clear_cache,
)


def test_prompts_directory_exists():
    assert PROMPTS_DIR.is_dir(), f"prompts/ 目录缺失: {PROMPTS_DIR}"


def test_list_prompts_returns_known_set():
    """目前迁移完成的 4 个 prompt 必须能被枚举到。"""
    names = set(list_prompts())
    expected = {
        "agent.keyword_extract",
        "agent.title_generation",
        "agent.refine_poem",
        "agent.prompt_review",
    }
    missing = expected - names
    assert not missing, f"缺少 prompt: {missing}"


def test_metadata_has_version_and_description():
    """每个 prompt 必须声明 version 和 description，便于 git 版本管理。"""
    for name in list_prompts():
        meta = get_metadata(name)
        assert meta["version"] is not None, f"{name} 缺少 version"
        assert meta["description"], f"{name} 缺少 description"


# ── 渲染行为 ────────────────────────────────────────────────────────────────
def test_render_returns_openai_messages_shape():
    msgs = render_messages("agent.keyword_extract", poem="春风拂柳过")
    assert isinstance(msgs, list)
    assert all("role" in m and "content" in m for m in msgs)
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"


def test_simple_variable_substitution():
    msgs = render_messages("agent.keyword_extract", poem="床前明月光")
    user_msg = msgs[-1]["content"]
    assert "床前明月光" in user_msg
    # system 段不含变量，应不会有遗留的 {} 占位
    assert "{poem}" not in msgs[0]["content"]


def test_multi_variable_substitution():
    msgs = render_messages(
        "agent.refine_poem",
        expected_chars=7, expected_lines=4,
        old_poem="原诗内容", feedback="加强意境",
    )
    sys, usr = msgs[0]["content"], msgs[1]["content"]
    # system 段也有插值变量
    assert "7" in sys
    assert "4" in sys
    assert "原诗内容" in usr
    assert "加强意境" in usr


def test_missing_variable_raises_keyerror():
    """缺变量应直接 KeyError —— 不允许静默生成残缺 prompt。"""
    with pytest.raises(KeyError):
        render_messages("agent.refine_poem", expected_chars=5)


def test_unknown_prompt_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        render_messages("agent.nonexistent_xyz", poem="...")


def test_cache_returns_same_dict_object():
    """同一 prompt 多次读取应命中缓存。"""
    from core.prompts import _load_raw
    a = _load_raw("agent.keyword_extract")
    b = _load_raw("agent.keyword_extract")
    assert a is b


def test_clear_cache_forces_reload(tmp_path, monkeypatch):
    """clear_cache 后下一次读取应重新走盘。"""
    from core.prompts import _load_raw
    _load_raw("agent.keyword_extract")
    info_before = _load_raw.cache_info()
    assert info_before.currsize >= 1
    clear_cache()
    info_after = _load_raw.cache_info()
    assert info_after.currsize == 0


def test_prompt_review_user_includes_all_anchors():
    """提示词自检的 user 段必须把所有关键上下文都注入，防止 LLM 凭空发挥。"""
    msgs = render_messages(
        "agent.prompt_review",
        user_input="春天主题",
        title="春晓",
        poem="春眠不觉晓",
        visual_keywords_en="spring, dawn",
        prompt="ink wash painting of dawn",
    )
    user = msgs[1]["content"]
    for needle in ("春天主题", "春晓", "春眠不觉晓", "spring, dawn", "ink wash painting"):
        assert needle in user, f"user 段缺少 {needle!r}"
