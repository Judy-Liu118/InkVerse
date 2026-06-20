"""
core.agent.tools -- Agent 工具抽象层

设计目标：
  1. 把 PoetryAgent 内部的 _phase_* 方法以 **Tool** 的形式对外暴露，
     提供统一的可枚举、可 introspection 的接口；
  2. 不重复业务实现 —— Tool 是 PoetryAgent 上对应方法的 **轻量 facade**；
  3. 每个 Tool 自带 OpenAI Function Calling 风格的 JSON Schema 参数描述，
     未来对接 Function Calling / MCP / 远端 Agent 时无需再改业务代码；
  4. ToolRegistry 统一注册/查找，便于上层做 Plan-and-Execute、并行调度等扩展。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from core.logger import get_logger

if TYPE_CHECKING:
    from core.agent.agent import PoetryAgent
    from core.agent.state import AgentState

_log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 基类
# ═══════════════════════════════════════════════════════════════════════════════
class AgentTool(ABC):
    """工具基类。子类需声明 name / description / parameters，并实现 execute()。"""

    name: str = ""
    description: str = ""
    # JSON Schema 风格的参数描述，对齐 OpenAI Function Calling 规范
    parameters: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    def execute(self, state: "AgentState", **kwargs) -> "AgentState":
        """执行工具逻辑，返回更新后的 AgentState。"""

    # ── 对接 Function Calling 的统一 schema ────────────────────────────────
    def to_function_schema(self) -> Dict[str, Any]:
        """返回 OpenAI Function Calling 风格的 schema，便于未来对接 LLM tools API。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"Tool({self.name})"


# ═══════════════════════════════════════════════════════════════════════════════
# ToolRegistry
# ═══════════════════════════════════════════════════════════════════════════════
class ToolRegistry:
    """工具注册表：注册、查找、列表，并可一次性导出全部 function schemas。"""

    def __init__(self) -> None:
        self._tools: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> "ToolRegistry":
        if not tool.name:
            raise ValueError(f"Tool {tool!r} 缺少 name 字段")
        if tool.name in self._tools:
            _log.warning("工具 %s 已注册，将被覆盖", tool.name)
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[AgentTool]:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> List[str]:
        return list(self._tools.keys())

    def list(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def to_function_schemas(self) -> List[Dict[str, Any]]:
        """导出全部工具的 OpenAI Function Calling schema。"""
        return [t.to_function_schema() for t in self._tools.values()]

    def execute(self, name: str, state: "AgentState", **kwargs) -> "AgentState":
        """按名调度。未注册的工具名会抛 KeyError，便于上层捕获。"""
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"未注册的工具: {name!r}（可用：{self.names}）")
        _log.debug("[ToolRegistry] 调度工具 %s", name)
        return tool.execute(state, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# 业务工具：作为 PoetryAgent._phase_* 的轻量 facade
# ═══════════════════════════════════════════════════════════════════════════════
class _AgentBoundTool(AgentTool):
    """绑定到具体 PoetryAgent 实例的工具基类。"""

    def __init__(self, agent: "PoetryAgent") -> None:
        self.agent = agent


class PlanTool(_AgentBoundTool):
    name = "plan"
    description = "解析用户创作要求，生成结构化任务计划与创作 brief"
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self, state, **kwargs):
        return self.agent._phase_plan(state)


class GeneratePoemTool(_AgentBoundTool):
    name = "generate_poem"
    description = "生成候选诗歌，经多维度评分（意图/平仄/押韵/意象/聚合/重复）后选出最优"
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self, state, **kwargs):
        return self.agent._phase_poem(state)


class ExtractKeywordsTool(_AgentBoundTool):
    name = "extract_visual_keywords"
    description = "从诗歌抽取英文视觉关键词，作为 CLIP 双锚点中的「诗-图」锚点"
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self, state, **kwargs):
        return self.agent._phase_keyword_extract(state)


class GenerateTitleTool(_AgentBoundTool):
    name = "generate_title"
    description = "为诗歌生成 2-8 字的古典诗名"
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self, state, **kwargs):
        return self.agent._phase_title(state)


class GeneratePromptTool(_AgentBoundTool):
    name = "generate_image_prompt"
    description = "将诗歌翻译为面向扩散模型的英文绘画提示词，并附加风格后缀"
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self, state, **kwargs):
        return self.agent._phase_prompt(state)


class ReviewPromptTool(_AgentBoundTool):
    name = "review_image_prompt"
    description = "自检提示词是否带入未授权意象（人物/动物/道具），必要时自动改写"
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self, state, **kwargs):
        return self.agent._phase_prompt_review(state)


class GenerateImageTool(_AgentBoundTool):
    name = "generate_image"
    description = "根据提示词生成图像，并在 CLIP 双锚点评分下做最多 N 次重试"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, state, **kwargs):
        return self.agent._phase_image_clip(state)


class ReflectTool(_AgentBoundTool):
    name = "reflect"
    description = "根据 CLIP 双锚点综合分给出验收结论，指导后续是否需要改图/改诗"
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self, state, **kwargs):
        return self.agent._phase_reflect(state)


class RefinePoemTool(_AgentBoundTool):
    name = "refine_poem"
    description = "根据反馈或方向性诗评对当前诗作进行格律安全的改写"
    parameters = {
        "type": "object",
        "properties": {
            "feedback": {
                "type": "string",
                "description": "用户反馈或自动生成的方向性诗评",
            },
        },
        "required": ["feedback"],
    }

    def execute(self, state, feedback: str = "", **kwargs):
        return self.agent.refine_poem(state, feedback=feedback)


class EditImageTool(_AgentBoundTool):
    name = "edit_image"
    description = "基于反馈改写绘画提示词并重新生图（或调用百炼图像编辑 API 保留构图）"
    parameters = {
        "type": "object",
        "properties": {
            "feedback": {"type": "string", "description": "改图意见"},
            "edit_model": {
                "type": "string",
                "description": "百炼图像编辑模型名，留空则走改写重生图路径",
            },
        },
        "required": ["feedback"],
    }

    def execute(self, state, feedback: str = "", edit_model: Optional[str] = None, **kwargs):
        return self.agent.edit_image_by_feedback(
            state, feedback=feedback, edit_model=edit_model,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 工厂方法
# ═══════════════════════════════════════════════════════════════════════════════
def build_default_registry(agent: "PoetryAgent") -> ToolRegistry:
    """构造 InkVerse 默认工具集，按创作流水线的逻辑顺序注册。"""
    registry = ToolRegistry()
    for tool_cls in (
        PlanTool,
        GeneratePoemTool,
        ExtractKeywordsTool,
        GenerateTitleTool,
        GeneratePromptTool,
        ReviewPromptTool,
        GenerateImageTool,
        ReflectTool,
        RefinePoemTool,
        EditImageTool,
    ):
        registry.register(tool_cls(agent))
    return registry


__all__ = [
    "AgentTool",
    "ToolRegistry",
    "build_default_registry",
    "PlanTool",
    "GeneratePoemTool",
    "ExtractKeywordsTool",
    "GenerateTitleTool",
    "GeneratePromptTool",
    "ReviewPromptTool",
    "GenerateImageTool",
    "ReflectTool",
    "RefinePoemTool",
    "EditImageTool",
]
