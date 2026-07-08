"""
core.agent.tools -- Agent 工具抽象基类与注册表

  1. AgentTool：OpenAI Function Calling 风格的工具基类
     （name / description / parameters + execute），
     to_function_schema() 导出标准 tools 描述；
  2. ToolRegistry：注册 / 查找 / 枚举 / 按名调度。

生产使用方是 core.agent.controller.build_loop_registry —— LLM-driven
改图循环把工具 schema 注入 controller prompt，并经 ToolRegistry.execute
真实调度。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.logger import get_logger

if TYPE_CHECKING:
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


__all__ = [
    "AgentTool",
    "ToolRegistry",
]
