"""
core.agent.controller -- LLM-driven 改图循环控制器

每轮把 state snapshot + ToolRegistry 注入的工具 JSON Schema 喂给 LLM，
让它输出严格 JSON 决策，再走 ToolRegistry.execute 调度。

设计要点：
  · decide() 返回 decision dict（不副作用 state），失败时返回 fallback
  · dispatch() 走 registry.execute（让 schema 注入与工具调度都是真事实，
    不只是 review 担心的 "未来对接" facade）
  · stop 是 controller 自带语义，不是 registry 真实工具
  · 解析失败 / 工具非白名单 / LLM 异常 → fallback 到 edit_image
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from core.agent.tools import AgentTool, ToolRegistry
from core.logger import get_logger
from core.prompts import render_messages

if TYPE_CHECKING:
    from core.agent.agent import PoetryAgent
    from core.agent.state import AgentState

_log = get_logger(__name__)


DEFAULT_FALLBACK_DECISION: Dict[str, Any] = {
    "tool": "edit_image",
    "feedback": "继续提升画面与诗的对齐",
    "mode": "rewrite_regen",
    "reasoning": "controller fallback（LLM 异常或 JSON 解析失败）",
    "_fallback": True,
}


# ── 循环专用 Tools（包装 autonomous_improve_image / refine_poem_and_regen_image）
class _LoopBoundTool(AgentTool):
    def __init__(self, agent: "PoetryAgent") -> None:
        self.agent = agent


class LoopEditImageTool(_LoopBoundTool):
    """改图（带 mode 选择，封装 autonomous_improve_image）。"""
    name = "edit_image"
    description = (
        "在不改诗的前提下改图。每次调用消耗 1 次改图预算。"
        "mode='rewrite_regen' 改写绘画 prompt 后重生图（覆盖大）；"
        "mode='edit_api' 在原图基础上调用图像编辑 API 保留构图微调（覆盖小）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "feedback": {
                "type": "string",
                "description": "≤60 字、具体可操作、仅引用诗与 visual_keywords 已有元素的改图意见",
            },
            "mode": {
                "type": "string",
                "enum": ["rewrite_regen", "edit_api"],
                "description": "改图模式。rewrite_regen 覆盖大；edit_api 保留构图微调",
            },
        },
        "required": ["feedback"],
    }

    def execute(self, state, feedback: str = "", mode: str = "rewrite_regen", **kwargs):
        # autonomous_improve_image 内部会让 LLM 重新生成 feedback；
        # 把 controller 已经给出的 feedback 注入 image_edit_history，
        # 避免后续 round 提相同方向。
        if feedback and feedback not in state.image_edit_history:
            state.image_edit_history.append(feedback)
        return self.agent.autonomous_improve_image(state, image_mode=mode)


class LoopRefinePoemAndRegenTool(_LoopBoundTool):
    """改诗 + 重新生成关键词/prompt + 重生图（复合 action，代价大）。"""
    name = "refine_poem_and_regen"
    description = (
        "改诗 + 重新提取视觉关键词 + 重写绘画 prompt + 重生图。"
        "代价大（同时消耗 1 次改诗与 1 次改图预算），仅在诗-prompt 错位、"
        "反复改图无果时使用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "feedback": {
                "type": "string",
                "description": "≤60 字、指出诗的不足处、不引入新元素的改诗方向",
            },
        },
        "required": ["feedback"],
    }

    def execute(self, state, feedback: str = "", **kwargs):
        return self.agent.refine_poem_and_regen_image(state, feedback=feedback)


def build_loop_registry(agent: "PoetryAgent") -> ToolRegistry:
    """构造 LLM-driven 改图循环专用 registry。不影响默认 registry。"""
    registry = ToolRegistry()
    registry.register(LoopEditImageTool(agent))
    registry.register(LoopRefinePoemAndRegenTool(agent))
    return registry


# ── Controller
class ImageLoopController:
    """LLM-driven 改图循环控制器（decide → dispatch）。"""

    STOP_TOOL = "stop"  # controller 自带语义，非 registry 真实工具

    def __init__(
        self,
        adapter,
        registry: ToolRegistry,
        allowed_tools: Optional[Set[str]] = None,
    ) -> None:
        self.adapter = adapter
        self.registry = registry
        self.allowed_tools = (
            allowed_tools if allowed_tools is not None
            else (set(registry.names) | {self.STOP_TOOL})
        )

    def _tool_schemas_json(self) -> str:
        """把 registry 中允许的工具 schema 序列化为 JSON 字符串注入 prompt。"""
        schemas = [
            t.to_function_schema() for t in self.registry
            if t.name in self.allowed_tools
        ]
        return json.dumps(schemas, ensure_ascii=False, indent=2)

    def decide(
        self, *,
        state: "AgentState",
        best_score: float,
        target: float,
        round_used: int,
        max_rounds: int,
        stale_count: int,
        prev_score: Optional[float] = None,
        history: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """让 LLM 决策下一步动作。返回 dict 含 'tool'；失败时 fallback。"""
        history = history or []
        delta_str = (
            f"{(best_score - prev_score):+.3f}" if prev_score is not None else "首轮"
        )
        history_block = (
            "\n".join(f"  - Round {i+1}: {h}" for i, h in enumerate(history))
            if history else "  （首轮）"
        )

        try:
            msg = render_messages(
                "agent.image_loop_controller",
                title=getattr(state, "title", "") or "",
                poem=getattr(state, "poem", "") or "",
                prompt_preview=(getattr(state, "prompt", "") or "")[:200],
                visual_keywords_en=getattr(state, "visual_keywords_en", "") or "（未提取）",
                reflection=(
                    getattr(state, "final_reflection", "")
                    or getattr(state, "clip_msg", "")
                    or ""
                )[:300],
                best_score=f"{best_score:.3f}",
                target=f"{target:.3f}",
                delta_str=delta_str,
                stale_count=stale_count,
                round_used=round_used,
                max_rounds=max_rounds,
                round_remaining=max(0, max_rounds - round_used),
                history_block=history_block,
                tools_schema_json=self._tool_schemas_json(),
            )
        except Exception as e:
            _log.warning("[controller] prompt 渲染失败: %s → fallback", e)
            return dict(DEFAULT_FALLBACK_DECISION)

        try:
            raw = self.adapter.generate(msg, max_tokens=200, temperature=0.3)
        except Exception as e:
            _log.warning("[controller] LLM 调用失败: %s → fallback", e)
            return dict(DEFAULT_FALLBACK_DECISION)

        decision = self._parse(raw)
        if decision is None:
            _log.warning("[controller] JSON 解析失败 raw=%r → fallback", (raw or "")[:120])
            return dict(DEFAULT_FALLBACK_DECISION)

        tool = decision.get("tool")
        if tool not in self.allowed_tools:
            _log.warning(
                "[controller] LLM 返回非法工具 %r（允许=%s）→ fallback",
                tool, sorted(self.allowed_tools),
            )
            return dict(DEFAULT_FALLBACK_DECISION)
        return decision

    def dispatch(
        self, decision: Dict[str, Any], state: "AgentState",
    ) -> Tuple["AgentState", bool]:
        """根据 decision 调度工具。返回 (updated_state, should_stop)。"""
        tool = decision.get("tool")
        if tool == self.STOP_TOOL:
            return state, True
        kwargs = {
            k: v for k, v in decision.items()
            if k not in ("tool", "reasoning", "reason", "_fallback") and v is not None
        }
        try:
            new_state = self.registry.execute(tool, state, **kwargs)
        except (KeyError, TypeError) as e:
            _log.warning(
                "[controller] dispatch %s 失败 %s → 留在原 state，本轮空转",
                tool, e,
            )
            return state, False
        return new_state, False

    @staticmethod
    def _parse(raw: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 返回，宽容 markdown 围栏。"""
        if not raw:
            return None
        s = raw.strip()
        if s.startswith("```"):
            lines = s.split("\n")
            s = "\n".join(l for l in lines if not l.strip().startswith("```"))
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j == -1 or j <= i:
            return None
        try:
            obj = json.loads(s[i:j+1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        return None


__all__ = [
    "ImageLoopController",
    "LoopEditImageTool",
    "LoopRefinePoemAndRegenTool",
    "build_loop_registry",
    "DEFAULT_FALLBACK_DECISION",
]
