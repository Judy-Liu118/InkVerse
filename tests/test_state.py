"""
test_state -- AgentState 数据结构 + 序列化/反序列化往返
"""
from core.agent.state import AgentState, Phase, AgentStep, ModelUsage


def test_defaults_are_sensible():
    s = AgentState()
    assert s.phase == Phase.INIT
    assert s.poem == ""
    assert s.image is None
    assert s.trace == []
    assert s.qualified_candidates == []
    assert s.clip_score_final == 0.0


def test_log_appends_trace_step():
    s = AgentState()
    s.log("测试阶段", "动作 A", "结果 X", model="qwen-plus", score=0.85)
    assert len(s.trace) == 1
    step = s.trace[0]
    assert step.phase == "测试阶段"
    assert step.action == "动作 A"
    assert step.score == 0.85
    assert step.model == "qwen-plus"


def test_trace_md_renders_all_steps():
    s = AgentState()
    s.log("规划", "解析意图", "OK")
    s.log("生成", "生成候选", "已完成", model="qwen-max", score=0.7)
    md = s.trace_md()
    assert "规划" in md
    assert "qwen-max" in md
    assert "0.700" in md


def test_model_usage_as_dict_has_all_modules():
    mu = ModelUsage(poem_gen="qwen-plus", title_gen="qwen-max")
    d = mu.as_dict()
    assert d["诗歌生成"] == "qwen-plus"
    assert d["诗名生成"] == "qwen-max"
    # 未设置的字段返回 dash 占位
    assert d["提示词生成"] == "—"
    # CLIP 字段始终有默认值
    assert "CLIP" in d["CLIP 评估"] or d["CLIP 评估"]


def test_serialize_deserialize_round_trip():
    """app.py 里的 _serialize_state / _deserialize_state 序列化往返。"""
    from app import _serialize_state, _deserialize_state

    s = AgentState(
        user_input="春日山行",
        lang="中文",
        style_suffix="水墨",
    )
    s.poem = "春风拂柳过\n燕子掠檐低"
    s.title = "春日"
    s.prompt = "ink wash painting of spring willows"
    s.creative_brief = "春日意象，柳与燕"
    s.visual_keywords_en = "willow, spring, swallow"
    s.clip_score_poem   = 0.61
    s.clip_score_prompt = 0.58
    s.clip_score_final  = 0.595
    s.model_usage.poem_gen = "qwen-plus"
    s.log("阶段A", "动作B", "结果C", model="qwen-plus", score=0.7)

    js = _serialize_state(s)
    s2 = _deserialize_state(js)

    assert s2.poem == s.poem
    assert s2.title == s.title
    assert s2.prompt == s.prompt
    assert s2.creative_brief == s.creative_brief
    assert s2.visual_keywords_en == s.visual_keywords_en
    assert s2.model_usage.poem_gen == "qwen-plus"
    # trace 应被还原
    assert any(step.action == "动作B" for step in s2.trace)


def test_phase_enum_values_stable():
    """Phase 名称作为序列化键被外部依赖，避免被悄无声息地重命名。"""
    expected = {
        "INIT", "PLAN", "POEM_GEN", "KEYWORD_EXTRACT",
        "TITLE_GEN", "PROMPT_GEN", "PROMPT_REVIEW",
        "IMAGE_GEN", "CLIP_EVAL", "REFLECT", "DONE", "ERROR",
    }
    actual = {p.name for p in Phase}
    assert expected == actual
