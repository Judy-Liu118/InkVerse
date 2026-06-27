"""
test_controller -- LLM-driven 改图循环 controller 单元测试

不触发真实 LLM 调用：用 stub adapter / stub agent 验证：
  · _parse 兼容 bare JSON、markdown 围栏、坏 JSON
  · decide 在 LLM 异常 / JSON 解析失败 / 非白名单工具时回退到 fallback
  · dispatch stop → (state, True)
  · dispatch 真实工具 → 走 ToolRegistry.execute，调用 agent 对应方法
  · build_loop_registry 注册了两个循环工具
"""
import json
from types import SimpleNamespace

import pytest

from core.agent.controller import (
    ImageLoopController,
    LoopEditImageTool,
    LoopRefinePoemAndRegenTool,
    build_loop_registry,
    DEFAULT_FALLBACK_DECISION,
)
from core.agent.state import AgentState
from core.agent.tools import ToolRegistry


# ── 测试用 stub ────────────────────────────────────────────────────────────
class _StubAdapter:
    """可注入 raw 返回值或异常，用于驱动 controller.decide。"""
    def __init__(self, raw=None, raise_exc=None):
        self.raw = raw
        self.raise_exc = raise_exc
        self.last_messages = None
        self.calls = 0

    def generate(self, messages, max_tokens=200, temperature=0.3):
        self.calls += 1
        self.last_messages = messages
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.raw


class _StubAgent:
    """记录 autonomous_improve_image / refine_poem_and_regen_image 调用入参。"""
    def __init__(self):
        self.improve_calls = []
        self.refine_calls = []

    def autonomous_improve_image(self, state, image_mode="rewrite_regen", **kwargs):
        self.improve_calls.append({"state": state, "image_mode": image_mode, **kwargs})
        state.log("test", "improve_image", image_mode)
        return state

    def refine_poem_and_regen_image(self, state, feedback="", **kwargs):
        self.refine_calls.append({"state": state, "feedback": feedback, **kwargs})
        state.log("test", "refine_and_regen", feedback)
        return state


def _make_controller(raw=None, raise_exc=None):
    adapter = _StubAdapter(raw=raw, raise_exc=raise_exc)
    agent = _StubAgent()
    registry = build_loop_registry(agent)
    controller = ImageLoopController(adapter=adapter, registry=registry)
    return controller, adapter, agent, registry


def _decide_kwargs(state):
    """controller.decide 所需的固定 kwargs。"""
    return dict(
        state=state, best_score=0.25, target=0.30,
        round_used=0, max_rounds=2,
        stale_count=0, prev_score=None, history=[],
    )


# ── _parse ────────────────────────────────────────────────────────────────
def test_parse_valid_bare_json():
    raw = '{"tool": "edit_image", "feedback": "x"}'
    out = ImageLoopController._parse(raw)
    assert out == {"tool": "edit_image", "feedback": "x"}


def test_parse_markdown_fenced():
    raw = '```json\n{"tool": "stop", "reason": "done"}\n```'
    out = ImageLoopController._parse(raw)
    assert out == {"tool": "stop", "reason": "done"}


def test_parse_invalid_returns_none():
    assert ImageLoopController._parse("not json at all") is None
    assert ImageLoopController._parse("") is None
    assert ImageLoopController._parse("{ broken json") is None


def test_parse_non_dict_returns_none():
    assert ImageLoopController._parse('["a", "b"]') is None


# ── decide ────────────────────────────────────────────────────────────────
def test_decide_returns_valid_llm_decision():
    raw = '{"tool": "edit_image", "feedback": "加强柳枝", "mode": "rewrite_regen"}'
    controller, adapter, _, _ = _make_controller(raw=raw)
    state = AgentState(poem="test", visual_keywords_en="willow")
    decision = controller.decide(**_decide_kwargs(state))
    assert decision["tool"] == "edit_image"
    assert decision.get("_fallback") is None
    assert adapter.calls == 1


def test_decide_llm_failure_returns_fallback():
    controller, _, _, _ = _make_controller(raise_exc=RuntimeError("api down"))
    state = AgentState(poem="x", visual_keywords_en="y")
    decision = controller.decide(**_decide_kwargs(state))
    assert decision["_fallback"] is True
    assert decision["tool"] == DEFAULT_FALLBACK_DECISION["tool"]


def test_decide_invalid_json_returns_fallback():
    controller, _, _, _ = _make_controller(raw="not json blob")
    state = AgentState(poem="x", visual_keywords_en="y")
    decision = controller.decide(**_decide_kwargs(state))
    assert decision["_fallback"] is True


def test_decide_non_whitelisted_tool_returns_fallback():
    raw = '{"tool": "drop_database", "feedback": "x"}'
    controller, _, _, _ = _make_controller(raw=raw)
    state = AgentState(poem="x", visual_keywords_en="y")
    decision = controller.decide(**_decide_kwargs(state))
    assert decision["_fallback"] is True


def test_decide_allows_stop_tool():
    raw = '{"tool": "stop", "reason": "done"}'
    controller, _, _, _ = _make_controller(raw=raw)
    state = AgentState(poem="x", visual_keywords_en="y")
    decision = controller.decide(**_decide_kwargs(state))
    assert decision["tool"] == "stop"
    assert decision.get("_fallback") is None


# ── dispatch ──────────────────────────────────────────────────────────────
def test_dispatch_stop_returns_should_stop_true():
    controller, _, _, _ = _make_controller()
    state = AgentState()
    new_state, should_stop = controller.dispatch({"tool": "stop"}, state)
    assert should_stop is True
    assert new_state is state


def test_dispatch_edit_image_routes_to_agent():
    controller, _, agent, _ = _make_controller()
    state = AgentState()
    decision = {
        "tool": "edit_image",
        "feedback": "加强柳枝低垂",
        "mode": "rewrite_regen",
        "reasoning": "tweak",
    }
    new_state, should_stop = controller.dispatch(decision, state)
    assert should_stop is False
    assert len(agent.improve_calls) == 1
    assert agent.improve_calls[0]["image_mode"] == "rewrite_regen"
    # LoopEditImageTool 应把 feedback 注入 image_edit_history（避免重复方向）
    assert "加强柳枝低垂" in state.image_edit_history


def test_dispatch_refine_routes_to_agent():
    controller, _, agent, _ = _make_controller()
    state = AgentState()
    decision = {
        "tool": "refine_poem_and_regen",
        "feedback": "末句意象偏弱",
        "reasoning": "需改诗",
    }
    new_state, should_stop = controller.dispatch(decision, state)
    assert should_stop is False
    assert len(agent.refine_calls) == 1
    assert agent.refine_calls[0]["feedback"] == "末句意象偏弱"


def test_dispatch_unknown_tool_keeps_state():
    """ToolRegistry.execute 抛 KeyError → controller 应吞掉并原地返回。"""
    controller, _, _, _ = _make_controller()
    state = AgentState()
    new_state, should_stop = controller.dispatch({"tool": "ghost_tool"}, state)
    assert should_stop is False
    assert new_state is state


# ── build_loop_registry ───────────────────────────────────────────────────
def test_build_loop_registry_has_two_tools():
    registry = build_loop_registry(_StubAgent())
    assert set(registry.names) == {"edit_image", "refine_poem_and_regen"}
    assert isinstance(registry.get("edit_image"), LoopEditImageTool)
    assert isinstance(registry.get("refine_poem_and_regen"), LoopRefinePoemAndRegenTool)


def test_loop_tools_expose_function_schemas():
    registry = build_loop_registry(_StubAgent())
    schemas = registry.to_function_schemas()
    assert len(schemas) == 2
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["name"] in {"edit_image", "refine_poem_and_regen"}
        assert "feedback" in fn["parameters"]["properties"]


def test_controller_default_allowed_includes_stop_and_registry():
    controller, _, _, _ = _make_controller()
    assert "stop" in controller.allowed_tools
    assert "edit_image" in controller.allowed_tools
    assert "refine_poem_and_regen" in controller.allowed_tools


def test_controller_custom_allowed_restricts_set():
    adapter = _StubAdapter()
    registry = build_loop_registry(_StubAgent())
    controller = ImageLoopController(
        adapter=adapter, registry=registry,
        allowed_tools={"edit_image", "stop"},
    )
    raw = '{"tool": "refine_poem_and_regen", "feedback": "x"}'
    adapter.raw = raw
    state = AgentState(poem="x", visual_keywords_en="y")
    decision = controller.decide(**_decide_kwargs(state))
    # refine 不在 allowed → 应 fallback
    assert decision["_fallback"] is True
