"""
test_tools -- AgentTool 基类 + ToolRegistry 的注册与调度（通用行为）

生产使用方是 controller.build_loop_registry（见 test_controller.py）；
这里用 dummy 工具验证基类与注册表本身的契约。
"""
import pytest

from core.agent import ToolRegistry, AgentTool
from core.agent.state import AgentState


class _MarkTool(AgentTool):
    """把调用参数记到 state.error 字段上（借用现成字段，免 mock）。"""
    name = "mark"
    description = "测试用：记录被调用时收到的参数"
    parameters = {
        "type": "object",
        "properties": {
            "feedback": {"type": "string", "description": "任意标记文本"},
        },
        "required": ["feedback"],
    }

    def execute(self, state, feedback: str = "", **kwargs):
        state.error = f"mark:{feedback}"
        return state


class _NoopTool(AgentTool):
    name = "noop"
    description = "测试用：什么都不做"
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self, state, **kwargs):
        return state


def _registry():
    return ToolRegistry().register(_MarkTool()).register(_NoopTool())


def test_registry_lookup_and_contains():
    reg = _registry()
    assert set(reg.names) == {"mark", "noop"}
    assert len(reg) == 2
    assert "mark" in reg
    assert "nonexistent" not in reg
    assert isinstance(reg.get("mark"), _MarkTool)
    assert reg.get("nonexistent") is None


def test_function_schemas_are_openai_compatible():
    """每个 Tool 的 schema 必须满足 OpenAI Function Calling 形状。"""
    reg = _registry()
    schemas = reg.to_function_schemas()

    assert len(schemas) == len(reg)
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params


def test_declared_params_appear_in_schema():
    schema = _MarkTool().to_function_schema()["function"]["parameters"]
    assert "feedback" in schema["properties"]
    assert "feedback" in schema["required"]


def test_registry_execute_dispatches_with_kwargs():
    """按名调度应把 kwargs 透传给工具并返回更新后的 state。"""
    reg = _registry()
    state = AgentState()
    state = reg.execute("mark", state, feedback="hello")
    assert state.error == "mark:hello"


def test_registry_execute_unknown_raises():
    """调度未注册的工具应抛 KeyError，便于上层捕获。"""
    with pytest.raises(KeyError):
        _registry().execute("not_a_real_tool", AgentState())


def test_register_duplicate_overrides_silently():
    """同名工具二次注册应覆盖（带 warning），不应抛异常。"""
    reg = ToolRegistry()
    reg.register(_NoopTool())
    reg.register(_NoopTool())   # 不应崩
    assert len(reg) == 1


def test_register_tool_without_name_raises():
    class _Anon(AgentTool):
        name = ""
        def execute(self, state, **kwargs):
            return state
    with pytest.raises(ValueError):
        ToolRegistry().register(_Anon())
