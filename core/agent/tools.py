"""
core.agent.tools -- 工具抽象层

将每个创作步骤封装为独立 Tool，支持 ToolRegistry 注册与调度。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from core.logger import get_logger

_log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 基类
# ═══════════════════════════════════════════════════════════════════════════════
class AgentTool(ABC):
    """工具基类。每个工具负责一个独立的创作步骤。"""

    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, state: "AgentState", **kwargs) -> "AgentState":
        """执行工具逻辑，返回更新后的 AgentState。"""
        ...

    def __repr__(self) -> str:
        return f"Tool({self.name})"


# ═══════════════════════════════════════════════════════════════════════════════
# 工具注册表
# ═══════════════════════════════════════════════════════════════════════════════
class ToolRegistry:
    """工具注册表，管理所有可用工具及其调度。"""

    def __init__(self):
        self._tools: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        if tool.name in self._tools:
            _log.warning("工具 %s 已注册，将被覆盖", tool.name)
        self._tools[tool.name] = tool
        _log.debug("注册工具: %s — %s", tool.name, tool.description)

    def get(self, name: str) -> Optional[AgentTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# 具体工具实现
# ═══════════════════════════════════════════════════════════════════════════════

class GeneratePoemTool(AgentTool):
    name = "generate_poem"
    description = "根据用户要求生成诗歌候选，多维度评分后选出最优"

    def __init__(self, generation_adapter=None, score_adapter=None):
        self.generation_adapter = generation_adapter
        self.score_adapter = score_adapter

    def execute(self, state, **kwargs) -> "AgentState":
        from core.poem.generator import PoemGenerator
        gen = PoemGenerator()
        # LoRA 模型忽略 brief（避免干扰）；API 模型利用规划阶段生成的 brief 提升质量
        use_lora = getattr(self.generation_adapter, "backend", "") == "local_lora"
        brief = "" if use_lora else (getattr(state, "creative_brief", "") or "")
        result = gen.generate(
            state.user_input, self.score_adapter, self.generation_adapter,
            creative_brief=brief,
        )
        if len(result) == 4:
            genre_name, poem, best_score, art_quality = result
        elif len(result) == 3:
            genre_name, poem, best_score = result
            art_quality = best_score
        else:
            genre_name, poem = result
            best_score = art_quality = 0.0

        state.poem = poem
        state.best_poem_score = best_score
        state.best_poem_art_quality = art_quality
        from core.agent.state import Phase
        if "生成失败" in poem:
            state.phase = Phase.ERROR
            state.error = poem
        return state


class ExtractKeywordsTool(AgentTool):
    name = "extract_keywords"
    description = "从诗歌提取英文视觉关键词，作为 CLIP 诗歌锚点"

    def __init__(self, score_adapter=None):
        self.score_adapter = score_adapter

    def execute(self, state, **kwargs) -> "AgentState":
        if not state.poem:
            return state
        msg = [
            {"role": "system",
             "content": (
                 "You are a Chinese classical poetry visual analyst. "
                 "Extract 6-10 key visual elements (objects, scenes, colors, "
                 "lighting, atmospheric mood) from the poem and translate them to English. "
                 "Only include elements explicitly present in the poem; do not add people, animals, kites, boats, buildings, tools, or story props unless named. "
                 "Treat ambiguous emotion/thought as mood, not as a new object. "
                 "Output ONLY a comma-separated list of short English phrases. "
                 "No explanations, no Chinese characters."
             )},
            {"role": "user",
             "content": f"Extract visual keywords from:\n{state.poem}"},
        ]
        try:
            result = self.score_adapter.generate(msg, max_tokens=80, temperature=0.2)
            state.visual_keywords_en = result.strip().replace("\n", ", ")
        except Exception as e:
            _log.warning("关键词提取失败: %s", e)
            state.visual_keywords_en = ""
        return state


class GenerateTitleTool(AgentTool):
    name = "generate_title"
    description = "为诗歌生成2-8字的古典诗名"

    def __init__(self, title_adapter=None):
        self.title_adapter = title_adapter

    def execute(self, state, **kwargs) -> "AgentState":
        import re
        user_req_hint = (
            f"\n注意：题名不得与创作要求相矛盾（创作要求：{state.user_input[:60]}），"
            "尤其不能出现相反的季节、时段、情绪等词汇。" if state.user_input else ""
        )
        msg = [
            {"role": "system",
             "content": (
                 "你是一个古诗命名专家。直接输出诗名，不要输出任何解释、标点、书名号或多余字符。"
                 f"诗名长度2-8字。{user_req_hint}"
             )},
            {"role": "user",
             "content": f"为下面这首诗起一个2-8字的诗名，只输出诗名：\n\n{state.poem}"},
        ]
        for attempt in range(2):
            try:
                raw = self.title_adapter.generate(msg, max_tokens=20, temperature=0.5)
                m = re.search(r"[一-鿿]{2,8}", raw.strip())
                title = m.group() if m else ""
                if title and 2 <= len(title) <= 8:
                    state.title = title
                    return state
            except Exception as e:
                _log.warning("诗名生成尝试 %d 异常: %s", attempt + 1, e)
        pure = "".join(ch for ch in state.poem.split("\n")[0] if "一" <= ch <= "鿿")
        state.title = pure[:4] or "无题"
        return state


class GeneratePromptTool(AgentTool):
    name = "generate_prompt"
    description = "为诗歌生成结构化的绘画提示词"

    def __init__(self, prompt_adapter=None):
        self.prompt_adapter = prompt_adapter

    def execute(self, state, **kwargs) -> "AgentState":
        from core.image.prompt import PromptGenerator
        gen = PromptGenerator()
        prompt_text = gen.generate(state.poem, state.lang, self.prompt_adapter)
        if prompt_text is None:
            from core.agent.state import Phase
            state.phase = Phase.ERROR
            state.error = "提示词生成失败"
            return state
        if state.style_suffix:
            prompt_text = f"{state.style_suffix}\n{prompt_text}"
        state.prompt = prompt_text
        return state


class ReviewPromptTool(AgentTool):
    name = "review_prompt"
    description = "自检提示词质量，必要时自动改写"

    def __init__(self, prompt_adapter=None):
        self.prompt_adapter = prompt_adapter

    def execute(self, state, **kwargs) -> "AgentState":
        if not state.prompt:
            return state
        import re
        adapter = self.prompt_adapter
        msg = [
            {
                "role": "system",
                "content": (
                    "You are a strict prompt quality controller for poetry-to-image generation. "
                    "The prompt may ONLY depict visual elements supported by the title, poem, and visual anchors. "
                    "Remove any unsupported people, animals, tools, kites, boats, buildings, or narrative objects. "
                    "Do not use planning brief as a visual source. "
                    "If it is already faithful, return KEEP followed by one short reason. "
                    "If it contains unsupported elements or needs improvement, return REWRITE: followed only by the improved English prompt."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User request (context only, not a source for extra objects):\n{state.user_input}\n\n"
                    f"Title:\n{state.title}\n\n"
                    f"Poem:\n{state.poem}\n\n"
                    f"Visual anchors:\n{state.visual_keywords_en}\n\n"
                    f"Prompt:\n{state.prompt}"
                ),
            },
        ]
        try:
            review = adapter.generate(msg, max_tokens=520, temperature=0.25).strip()
            state.prompt_review = review[:500]
            if review.lower().startswith("rewrite"):
                rewritten = re.sub(r"^rewrite\s*[:：]\s*", "", review, flags=re.I).strip()
                rewritten = rewritten.strip().strip("`")
                rewritten = re.sub(r"^```[a-zA-Z]*\s*", "", rewritten)
                rewritten = re.sub(r"\s*```$", "", rewritten)
                rewritten = re.sub(r"^(revised image prompt|prompt)\s*[:：]\s*", "", rewritten, flags=re.I)
                kept = []
                for line in rewritten.splitlines():
                    if re.match(r"\s*(forbidden|negative prompt|禁加元素|负面提示词)\s*[:：]", line, flags=re.I):
                        continue
                    kept.append(line)
                rewritten = "\n".join(kept).strip()
                if len(rewritten) >= 30:
                    state.prompt = rewritten
        except Exception as e:
            _log.warning("提示词自检异常: %s", e)
        return state


class GenerateImageTool(AgentTool):
    name = "generate_image"
    description = "根据提示词生成图像（支持本地 Z-Image 和百炼 API 双后端）"

    def execute(self, state, **kwargs) -> "AgentState":
        from core.image.generator import ImageGenerator
        gen = ImageGenerator()
        image = gen.generate(
            prompt=state.prompt, backend=state.image_backend,
            api_key=state.image_api_key, api_model=state.image_api_model,
        )
        state.image = image
        model_name = state.image_api_model or f"local-Z-Image"
        state.model_usage.image_gen = f"{state.image_backend}/{model_name}"
        return state


class EvaluateCLIPTool(AgentTool):
    name = "evaluate_clip"
    description = "用 CLIP 双锚点评分评估图文一致性"

    def execute(self, state, **kwargs) -> "AgentState":
        from config import CLIP_ENABLED, CLIP_POEM_WEIGHT, CLIP_PROMPT_WEIGHT
        if not CLIP_ENABLED or state.image is None:
            return state
        from core.evaluation.clip import CLIPEvaluator
        clip_eval = CLIPEvaluator()
        raw_b = clip_eval.score_raw_cosine(state.image, state.prompt)
        norm_b = (raw_b + 1.0) / 2.0
        if state.visual_keywords_en:
            raw_a = clip_eval.score_raw_cosine(state.image, state.visual_keywords_en)
            norm_a = (raw_a + 1.0) / 2.0
        else:
            raw_a = norm_a = 0.0
        state.clip_score_poem = norm_a
        state.clip_score_prompt = norm_b
        state.clip_score_final = CLIP_POEM_WEIGHT * norm_a + CLIP_PROMPT_WEIGHT * norm_b if state.visual_keywords_en else norm_b
        return state


class ReflectTool(AgentTool):
    name = "reflect"
    description = "生成结果反思，评估生成质量并给出优化建议"

    def execute(self, state, **kwargs) -> "AgentState":
        raw_final = state.clip_score_final * 2 - 1 if state.clip_score_final else 0.0
        if raw_final >= 0.28:
            verdict = "接受当前结果：图像与诗歌意象、提示词执行均较一致。"
        elif raw_final >= 0.22:
            verdict = "接受当前结果但建议微调：主体和氛围基本匹配，可继续优化细节。"
        elif state.image is not None:
            verdict = "保留最优结果：评分偏低，建议改图增强主体、季节、光线或空间关系。"
        else:
            verdict = "没有可用图像结果，需要检查图像后端或提示词。"
        state.final_reflection = (
            f"{verdict}\n"
            f"诗歌锚点分: {state.clip_score_poem:.3f}；"
            f"提示词锚点分: {state.clip_score_prompt:.3f}；"
            f"综合分: {state.clip_score_final:.3f}。"
        )
        return state
