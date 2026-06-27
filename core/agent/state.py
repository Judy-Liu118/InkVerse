"""
core.agent.state -- Agent 全局状态与追踪数据结构
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Phase(Enum):
    INIT             = "初始化"
    PLAN             = "任务规划"
    POEM_GEN         = "诗歌生成"
    KEYWORD_EXTRACT  = "视觉关键词提取"
    TITLE_GEN        = "诗名生成"
    PROMPT_GEN       = "提示词生成"
    PROMPT_REVIEW    = "提示词自检"
    IMAGE_GEN        = "图像生成"
    CLIP_EVAL        = "CLIP评分"
    REFLECT          = "结果反思"
    DONE             = "完成"
    ERROR            = "错误"


@dataclass
class ModelUsage:
    """记录每个模块实际使用的模型，用于报告展示。"""
    poem_gen:   str = ""
    poem_scorer: str = ""
    title_gen:  str = ""
    prompt_gen: str = ""
    image_gen:  str = ""
    clip_eval:  str = "clip-vit-base-patch32（CPU 推理）"

    def as_dict(self) -> Dict[str, str]:
        return {
            "诗歌生成":   self.poem_gen   or "—",
            "诗歌评分":   self.poem_scorer or "—",
            "诗名生成":   self.title_gen  or "—",
            "提示词生成": self.prompt_gen or "—",
            "图像生成":   self.image_gen  or "—",
            "CLIP 评估":  self.clip_eval,
        }


@dataclass
class AgentStep:
    """单步 Trace 记录。"""
    phase:     str
    action:    str
    result:    str
    model:     str            = ""
    score:     Optional[float] = None
    timestamp: float          = field(default_factory=time.time)
    is_retry:  bool           = False
    extra:     Dict           = field(default_factory=dict)

    def summary(self) -> str:
        score_str = f" | 分数={self.score:.3f}" if self.score is not None else ""
        retry_str = " [重试]" if self.is_retry else ""
        model_str = f" ({self.model})" if self.model else ""
        return (
            f"[{self.phase}]{retry_str} {self.action}{model_str}"
            f"{score_str}: {self.result[:200]}"
        )


@dataclass
class AgentState:
    """创作全局状态，贯穿整个创作流程。"""

    # ── 输入 ────────────────────────────────────────────────────────────────
    user_input:      str            = ""
    direct_poem:     str            = ""
    lang:            str            = "英文"
    style_suffix:    str            = ""
    image_backend:   str            = "local"
    image_api_key:   Optional[str]  = None
    image_api_model: Optional[str]  = None

    # ── 各阶段输出 ───────────────────────────────────────────────────────────
    poem:               str = ""
    backup_poem:        str = ""    # pairwise 五选二的次优候选
    title:              str = ""
    prompt:             str = ""
    creative_brief:     str = ""
    agent_plan:         str = ""
    prompt_review:      str = ""
    final_reflection:   str = ""
    visual_keywords_en: str = ""
    image:              Any = None
    last_image_path:    str = ""
    image_history:      List[Dict[str, str]] = field(default_factory=list)
    image_edit_history: List[str] = field(default_factory=list)

    # ── 诗歌得分 ─────────────────────────────────────────────────────────────
    best_poem_score:       float = 0.0   # 含 required_coeff 的最终选择分
    best_poem_art_quality: float = 0.0   # 不含 required_coeff 的艺术品质分

    # ── Arena 海选结果 ─────────────────────────────────────────────────────
    champion_topic:       float = 0.5    # 冠军的切题分（arena 实测值，进化沿用）
    champion_local_total: float = 0.0    # 冠军的本地总分（供品质门槛判断）

    # ── 候选诗追踪（供品质筛选和改诗使用）─────────────────────────────────
    qualified_candidates: List[Dict] = field(default_factory=list)  # 合格候选 [{poem, scores}, ...]
    rejected_candidates:  List[Dict] = field(default_factory=list)  # 不合格候选
    refined_candidates:   List[Dict] = field(default_factory=list)  # 改后候选
    poem_selection_mode:  str = ""  # "qualified_only" | "fallback"

    # ── CLIP 双锚点分数 ─────────────────────────────────────────────────────
    clip_score_poem:   float = 0.0
    clip_score_prompt: float = 0.0
    clip_score_final:  float = 0.0
    clip_msg:          str   = ""

    # ── 模型使用记录 ────────────────────────────────────────────────────────
    model_usage: ModelUsage = field(default_factory=ModelUsage)

    # ── 运行追踪 ────────────────────────────────────────────────────────────
    trace:        List[AgentStep]   = field(default_factory=list)
    phase:        Phase             = Phase.INIT
    error:        str               = ""
    retry_counts: Dict[str, int]    = field(default_factory=dict)

    # LLM-driven 改图循环每轮决策的结构化记录（eval 诚实性指标用，不进 trace）。
    # 字段: round, tool, is_fallback, score_before, score_after, stale_override
    llm_loop_decisions: List[Dict] = field(default_factory=list)

    def log(
        self,
        phase:    str,
        action:   str,
        result:   str,
        model:    str            = "",
        score:    Optional[float] = None,
        is_retry: bool           = False,
        extra:    Dict           = None,
    ) -> None:
        step = AgentStep(
            phase=phase, action=action, result=result,
            model=model, score=score, is_retry=is_retry,
            extra=extra or {},
        )
        self.trace.append(step)
        from core.logger import get_logger
        get_logger("agent").info(step.summary())

    def trace_md(self) -> str:
        """返回 Markdown 格式的完整 Trace。"""
        lines = []
        for s in self.trace:
            retry = " *(重试)*" if s.is_retry else ""
            score = f"  `{s.score:.3f}`" if s.score is not None else ""
            model = f" · *{s.model}*" if s.model else ""
            lines.append(f"**[{s.phase}]** {s.action}{retry}{model}{score}  \n{s.result}")
        return "\n\n".join(lines)
