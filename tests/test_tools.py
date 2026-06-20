"""
test_tools -- Tool 抽象 + ToolRegistry 的注册与调度
"""
from core.agent import PoetryAgent, ToolRegistry, AgentTool, build_default_registry
from core.agent.state import AgentState
from core.agent.tools import (
    GeneratePoemTool, PlanTool, RefinePoemTool,
)


def test_default_registry_contains_full_pipeline():
    """默认 registry 应注册创作流水线全部 10 个工具。"""
    agent = PoetryAgent()
    reg = agent.tool_registry

    expected = {
        "plan", "generate_poem", "extract_visual_keywords",
        "generate_title", "generate_image_prompt", "review_image_prompt",
        "generate_image", "reflect", "refine_poem", "edit_image",
    }
    assert set(reg.names) == expected
    assert len(reg) == len(expected)


def test_registry_lookup_and_contains():
    agent = PoetryAgent()
    reg = agent.tool_registry

    assert "generate_poem" in reg
    assert "nonexistent" not in reg
    tool = reg.get("generate_poem")
    assert tool is not None
    assert isinstance(tool, GeneratePoemTool)
    assert reg.get("nonexistent") is None


def test_tool_registry_lazy_cached():
    """同一个 PoetryAgent 多次访问 tool_registry 应返回同一个对象。"""
    agent = PoetryAgent()
    r1 = agent.tool_registry
    r2 = agent.tool_registry
    assert r1 is r2


def test_function_schemas_are_openai_compatible():
    """每个 Tool 的 schema 必须满足 OpenAI Function Calling 形状。"""
    agent = PoetryAgent()
    schemas = agent.tool_registry.to_function_schemas()

    assert len(schemas) == len(agent.tool_registry)
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params


def test_refine_poem_tool_declares_feedback_param():
    """refine_poem 必须在 schema 里声明 feedback 字段。"""
    agent = PoetryAgent()
    tool = agent.tool_registry.get("refine_poem")
    assert isinstance(tool, RefinePoemTool)
    schema = tool.to_function_schema()["function"]["parameters"]
    assert "feedback" in schema["properties"]
    assert "feedback" in schema["required"]


def test_registry_execute_unknown_raises():
    """调度未注册的工具应抛 KeyError，便于上层捕获。"""
    agent = PoetryAgent()
    state = AgentState()
    import pytest
    with pytest.raises(KeyError):
        agent.tool_registry.execute("not_a_real_tool", state)


def test_register_duplicate_overrides_silently():
    """同名工具二次注册应覆盖（带 warning），不应抛异常。"""
    agent = PoetryAgent()
    reg = ToolRegistry()
    reg.register(PlanTool(agent))
    reg.register(PlanTool(agent))   # 不应崩
    assert len(reg) == 1


def test_register_tool_without_name_raises():
    class _Anon(AgentTool):
        name = ""
        def execute(self, state, **kwargs):
            return state
    import pytest
    with pytest.raises(ValueError):
        ToolRegistry().register(_Anon())


def test_plan_tool_executes_and_updates_state():
    """PlanTool 是纯本地逻辑（无 LLM），可端到端验证。"""
    agent = PoetryAgent()
    state = AgentState(user_input="写一首春天的七言绝句")
    state = agent.tool_registry.execute("plan", state)
    assert state.creative_brief
    assert state.agent_plan
    assert state.trace, "PlanTool 应在 trace 留下记录"
