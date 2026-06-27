"""
core.agent.agent -- 诗画创作引擎

双锚点 CLIP（消弭 Poem→Prompt→Image 链的语义损耗）：
  anchor_a = visual_keywords_en  ← 从诗歌直接提取，代表诗的意境
  anchor_b = english_prompt      ← 扩散模型提示词，代表模型的理解
  clip_score_final = 0.6 × clip(image, a) + 0.4 × clip(image, b)

每个 phase 方法独立修改 AgentState，上层可按需逐步调用或一次性调用 run()。
"""
from __future__ import annotations

import re
import time
import traceback
from typing import Any, Dict, List, Optional

from core.agent.state import AgentState, Phase, AgentStep
from core.logger import get_logger

_log = get_logger(__name__)


class PoetryAgent:
    """诗画创作引擎。"""

    def __init__(
        self,
        generation_adapter=None,
        score_adapter=None,
        title_adapter=None,
        prompt_adapter=None,
    ):
        self.generation_adapter = generation_adapter
        self.score_adapter = score_adapter
        self.title_adapter = title_adapter
        self.prompt_adapter = prompt_adapter
        self._poem_gen = None
        self._prompt_gen = None
        self._image_gen = None
        self._clip_eval = None
        self._clip_init_failed = False
        self._tool_registry = None

    # ── Tool 抽象层 ─────────────────────────────────────────────────────────
    @property
    def tool_registry(self):
        """懒加载 ToolRegistry。把每个 _phase_* 方法包装成可枚举的 Tool，
        便于未来对接 Function Calling / MCP 或外部调度。"""
        if self._tool_registry is None:
            from core.agent.tools import build_default_registry
            self._tool_registry = build_default_registry(self)
        return self._tool_registry

    # ── 懒加载属性 ───────────────────────────────────────────────────────────
    @property
    def poem_gen(self):
        if self._poem_gen is None:
            from core.poem.generator import PoemGenerator
            self._poem_gen = PoemGenerator()
        return self._poem_gen

    @property
    def prompt_gen(self):
        if self._prompt_gen is None:
            from core.image.prompt import PromptGenerator
            self._prompt_gen = PromptGenerator()
        return self._prompt_gen

    @property
    def image_gen(self):
        if self._image_gen is None:
            from core.image.generator import ImageGenerator
            self._image_gen = ImageGenerator()
        return self._image_gen

    def _get_clip_eval(self):
        """单例懒加载 CLIPEvaluator。初始化失败后标记为不可用，避免反复重试。

        返回 None 表示 CLIP 不可用（首次初始化失败、未启用、或模型缺失）。
        """
        if self._clip_eval is not None or self._clip_init_failed:
            return self._clip_eval
        try:
            from core.evaluation.clip import CLIPEvaluator
            self._clip_eval = CLIPEvaluator()
            _log.info("[CLIP] 评估器初始化完成（单例）")
        except Exception as e:
            self._clip_init_failed = True
            _log.exception("[CLIP] 评估器初始化失败，本次会话内将跳过 CLIP 评分")
            self._clip_init_error = str(e)
        return self._clip_eval

    # ═════════════════════════════════════════════════════════════════════════
    # 主运行入口
    # ═════════════════════════════════════════════════════════════════════════
    def run(self, state: AgentState) -> AgentState:
        """完整运行创作流水线。"""
        try:
            state = self._phase_plan(state)
            if state.phase == Phase.ERROR:
                return state
            state = self._phase_poem(state)
            if state.phase == Phase.ERROR:
                return state
            state = self._phase_keyword_extract(state)
            state = self._phase_title(state)
            state = self._phase_prompt(state)
            if state.phase == Phase.ERROR:
                return state
            state = self._phase_prompt_review(state)
            if state.phase == Phase.ERROR:
                return state
            state = self._phase_image_clip(state)
            state = self._phase_reflect(state)
            state.phase = Phase.DONE
        except Exception as e:
            state.phase = Phase.ERROR
            state.error = str(e)
            _log.error("Agent 意外错误: %s", e)
            _log.debug(traceback.format_exc())
        return state

    # ═════════════════════════════════════════════════════════════════════════
    # 逐步调用入口（流式 UI 用）
    # ═════════════════════════════════════════════════════════════════════════
    def step_poem(self, state: AgentState) -> AgentState:
        return self._phase_poem(state)

    def step_plan(self, state: AgentState) -> AgentState:
        return self._phase_plan(state)

    def step_keywords(self, state: AgentState) -> AgentState:
        return self._phase_keyword_extract(state)

    def step_title(self, state: AgentState) -> AgentState:
        return self._phase_title(state)

    def step_prompt(self, state: AgentState) -> AgentState:
        return self._phase_prompt(state)

    def step_prompt_review(self, state: AgentState) -> AgentState:
        return self._phase_prompt_review(state)

    def step_image(self, state: AgentState) -> AgentState:
        state = self._phase_image_clip(state)
        if state.phase != Phase.ERROR:
            state = self._phase_reflect(state)
        return state

    # ═════════════════════════════════════════════════════════════════════════
    # Phase 实现
    # ═════════════════════════════════════════════════════════════════════════
    def _phase_plan(self, state: AgentState) -> AgentState:
        """轻量级意图解析：只提取结构化参数，不调用 LLM。"""
        state.phase = Phase.PLAN
        user_goal = state.direct_poem or state.user_input
        if not user_goal.strip():
            state.phase = Phase.ERROR
            state.error = "请输入创作要求或直接提供诗文。"
            state.log("任务规划", "输入校验失败", state.error)
            return state

        state.creative_brief = self._sanitize_brief(user_goal)
        state.agent_plan = (
            "1. 解析体裁与主题 → 2. 生成候选诗 → 3. 多维度评分选优 → "
            "4. 提取视觉关键词 → 5. 生成诗名 → 6. 生成绘画提示词 → "
            "7. 生图 + CLIP 双锚点评分 → 8. 反思与自主优化"
        )
        state.log("任务规划", "轻量意图解析（本地提取关键参数）",
                  f"用户要求：{user_goal[:160]}")
        return state

    def _phase_poem(self, state: AgentState) -> AgentState:
        state.phase = Phase.POEM_GEN
        model_desc = self._adapter_desc(self.generation_adapter)

        if state.direct_poem.strip():
            lines = [l.strip() for l in state.direct_poem.split("\n") if l.strip()][:8]
            state.poem = "\n".join(lines)
            state.model_usage.poem_gen = "用户直接提供"
            state.model_usage.poem_scorer = self._adapter_desc(self.score_adapter)
            state.log("诗歌生成", "使用用户提供的诗歌", state.poem, model="用户输入")
            return state

        state.log("诗歌生成", "开始生成候选（品质筛选模式）",
                  f"用户要求: {state.user_input[:60]}", model=model_desc)
        try:
            from config import (
                POEM_QUALITY_THRESHOLD, POEM_MAX_DISCARD_PER_BATCH,
                POEM_MAX_GENERATION_ROUNDS, POEM_MIN_QUALIFIED,
            )
            result = self.poem_gen.generate_with_quality_control(
                state.user_input, self.score_adapter, self.generation_adapter,
                creative_brief="",
                quality_threshold=POEM_QUALITY_THRESHOLD,
                max_discard_per_batch=POEM_MAX_DISCARD_PER_BATCH,
                max_rounds=POEM_MAX_GENERATION_ROUNDS,
                min_qualified=POEM_MIN_QUALIFIED,
            )
            genre_name       = result["genre_name"]
            poem             = result["best_poem"]
            best_score       = result["best_score"]
            art_quality      = result["best_art_quality"]
            selection_mode   = result["selection_mode"]

            if "生成失败" in poem:
                state.phase = Phase.ERROR
                state.error = poem
                state.log("诗歌生成", "失败", poem)
                return state

            state.poem = poem
            state.best_poem_score = best_score
            state.best_poem_art_quality = art_quality
            state.qualified_candidates = result["qualified"]
            state.rejected_candidates  = result["rejected"]
            state.poem_selection_mode  = selection_mode
            state.model_usage.poem_gen = model_desc
            state.model_usage.poem_scorer = self._adapter_desc(self.score_adapter)

            qual_count = len(result["qualified"])
            rej_count  = len(result["rejected"])
            state.log("诗歌生成",
                      f"体裁: {genre_name}，品质筛选完成"
                      f"（合格 {qual_count} + 废弃 {rej_count}，"
                      f"模式={selection_mode}，"
                      f"最优候选得分={best_score:.3f}）",
                      poem, model=model_desc)
        except Exception as e:
            state.phase = Phase.ERROR
            state.error = str(e)
            _log.exception("诗歌生成 [_phase_poem] 异常")
            state.log("诗歌生成", "异常", str(e))
        return state

    # ── Arena 诗歌生成（全自主模式专用）───────────────────────────────────
    def _phase_poem_arena(self, state: AgentState) -> AgentState:
        """Arena 海选：生成 5 首 → 硬门控 → 本地分 Top3 → arena pairwise → 冠军。"""
        state.phase = Phase.POEM_GEN
        model_desc = self._adapter_desc(self.generation_adapter)

        if state.direct_poem.strip():
            lines = [l.strip() for l in state.direct_poem.split('\n') if l.strip()]
            state.poem = "\n".join(lines)
            state.backup_poem = ""
            state.model_usage.poem_gen = "用户直接提供"
            state.model_usage.poem_scorer = self._adapter_desc(self.score_adapter)
            state.log("诗歌生成", "使用用户提供的诗歌", state.poem, model="用户输入")
            return state

        from config import POEM_QUALITY_THRESHOLD
        _MAX_RETRIES = 2  # 额外再生最多 2 轮，共 3 轮
        state.log("诗歌生成", "开始生成候选（arena 海选模式）",
                  f"用户要求: {state.user_input[:80]}", model=model_desc)

        # 攒合格诗：门控通过 + 本地分 ≥ 0.7
        qualified_pool = []  # [{idx, poem, local}, ...]
        genre_name = ""
        for gen_round in range(1 + _MAX_RETRIES):
            try:
                result = self.poem_gen.generate_and_score(
                    state.user_input, self.score_adapter, self.generation_adapter,
                    creative_brief=state.creative_brief,
                )
                genre_name = result.get("genre_name", "")
                gated = result.get("gated", [])
                # 收集本轮合格的诗
                new_qualified = [g for g in gated
                                 if g["local"]["total"] >= POEM_QUALITY_THRESHOLD]
                qualified_pool.extend(new_qualified)
                state.log("诗歌生成",
                          f"第{gen_round+1}轮",
                          f"门控通过 {len(gated)} 首，合格 {len(new_qualified)} 首，"
                          f"累计合格 {len(qualified_pool)} 首（需 ≥ 3）",
                          model=model_desc)
                if len(qualified_pool) >= 3:
                    break
            except Exception as e:
                state.phase = Phase.ERROR
                state.error = str(e)
                _log.exception("诗歌生成 [arena 重试轮] 异常")
                state.log("诗歌生成", "异常", str(e))
                return state

        if len(qualified_pool) < 3:
            state.log("诗歌生成",
                      f"经 {1 + _MAX_RETRIES} 轮仅攒到 {len(qualified_pool)} 首合格诗，"
                      f"直接选用最优",
                      model=model_desc)
            # 不足 3 首时从所有门控通过的诗中补足
            if len(qualified_pool) < 1:
                state.phase = Phase.ERROR
                state.error = "无合格候选诗"
                return state

        # Arena 选冠军（合格池 < 3 时 arena_from_gated 自动退化为 1-2 首对决）
        arena_result = self.poem_gen.scorer.arena_from_gated(
            qualified_pool, state.user_input, self.score_adapter,
        )

        state.poem = arena_result["champion"]
        state.backup_poem = arena_result["backup"]
        state.champion_topic = arena_result.get("champion_topic", 0.5)
        state.champion_local_total = arena_result.get("champion_local_total", 0.0)
        state.model_usage.poem_gen = model_desc
        state.model_usage.poem_scorer = self._adapter_desc(self.score_adapter)
        state.poem_selection_mode = "arena"
        state.log("诗歌生成",
                  f"体裁: {genre_name}，arena 海选完成",
                  f"候选 {len(qualified_pool)} 首，"
                  f"冠军本地分={state.champion_local_total:.3f}，"
                  f"切题={state.champion_topic:.2f}",
                  model=model_desc)
        return state

    def _phase_keyword_extract(self, state: AgentState) -> AgentState:
        state.phase = Phase.KEYWORD_EXTRACT
        if not state.poem:
            return state
        from core.prompts import render_messages
        msg = render_messages("agent.keyword_extract", poem=state.poem)
        try:
            result = self.score_adapter.generate(msg, max_tokens=80, temperature=0.2)
            state.visual_keywords_en = result.strip().replace("\n", ", ")
            state.log("视觉关键词提取", "从诗歌提取 CLIP 锚点（诗-图直连）",
                      result, model=self._adapter_desc(self.score_adapter))
        except Exception as e:
            state.visual_keywords_en = ""
            state.log("视觉关键词提取", "提取失败（将仅用提示词锚点）", str(e))
        return state

    def _phase_title(self, state: AgentState) -> AgentState:
        state.phase = Phase.TITLE_GEN
        model_desc = self._adapter_desc(self.title_adapter)
        from core.prompts import render_messages
        user_req_hint = (
            f"\n注意：题名不得与创作要求相矛盾（创作要求：{state.user_input[:60]}），"
            "尤其不能出现相反的季节、时段、情绪等词汇。" if state.user_input else ""
        )
        msg = render_messages(
            "agent.title_generation",
            poem=state.poem, user_req_hint=user_req_hint,
        )
        for attempt in range(2):
            try:
                raw = self.title_adapter.generate(msg, max_tokens=20, temperature=0.5)
                m = re.search(r"[一-鿿]{2,8}", raw.strip())
                title = m.group() if m else ""
                if title and 2 <= len(title) <= 8:
                    state.title = title
                    state.model_usage.title_gen = model_desc
                    state.log("诗名生成", f"尝试 {attempt+1}", title, model=model_desc)
                    return state
            except Exception as e:
                state.log("诗名生成", f"尝试 {attempt+1} 异常", str(e))
        pure = "".join(ch for ch in state.poem.split("\n")[0] if "一" <= ch <= "鿿")
        state.title = pure[:4] or "无题"
        state.model_usage.title_gen = f"{model_desc}（回退）"
        state.log("诗名生成", "回退标题", state.title)
        return state

    def _phase_prompt(self, state: AgentState) -> AgentState:
        state.phase = Phase.PROMPT_GEN
        model_desc = self._adapter_desc(self.prompt_adapter)
        prompt_text = self.prompt_gen.generate(state.poem, state.lang, self.prompt_adapter,
                                                user_request=state.user_input)
        if prompt_text is None:
            state.phase = Phase.ERROR
            state.error = "提示词生成失败"
            state.log("提示词生成", "失败", "返回 None", model=model_desc)
            return state
        if state.style_suffix:
            prompt_text = f"{state.style_suffix}\n{prompt_text}"
        state.prompt = prompt_text
        state.model_usage.prompt_gen = model_desc
        state.log("提示词生成", "成功", prompt_text[:120], model=model_desc)
        return state

    def _phase_prompt_review(self, state: AgentState) -> AgentState:
        state.phase = Phase.PROMPT_REVIEW
        if not state.prompt:
            return state
        adapter = self.prompt_adapter or self.score_adapter
        model_desc = self._adapter_desc(adapter)
        from core.prompts import render_messages
        msg = render_messages(
            "agent.prompt_review",
            user_input=state.user_input,
            title=state.title,
            poem=state.poem,
            visual_keywords_en=state.visual_keywords_en,
            prompt=state.prompt,
        )
        try:
            review = adapter.generate(msg, max_tokens=520, temperature=0.25).strip()
            state.prompt_review = review[:500]
            if review.lower().startswith("rewrite"):
                rewritten = re.sub(r"^rewrite\s*[:：]\s*", "", review, flags=re.I).strip()
                rewritten = self._clean_revised_prompt(rewritten)
                if len(rewritten) >= 30:
                    state.prompt = rewritten
                    state.log("提示词自检", "发现缺口并自动改写", rewritten[:180], model=model_desc)
                    self._check_prompt_alignment(state)
                    return state
            state.log("提示词自检", "通过", review[:180], model=model_desc)
        except Exception as e:
            state.prompt_review = f"Prompt 自检失败: {e}"
            state.log("提示词自检", "异常，沿用原 Prompt", str(e), model=model_desc)

        # ── CLIP 诗-提示词语义一致性检查 ──────────────────────────────────
        self._check_prompt_alignment(state)

        return state

    def _check_prompt_alignment(self, state: AgentState) -> None:
        """用 CLIP text encoder 检查诗锚点与提示词的语义一致性。

        两段文本走 CLIP text encoder 做余弦相似度。偏低说明提示词可能
        遗漏或曲解了诗歌的核心视觉意象。
        """
        from config import CLIP_ENABLED, CLIP_PROMPT_ALIGN_THRESHOLD
        if not CLIP_ENABLED or not state.visual_keywords_en or not state.prompt:
            return
        try:
            clip = self._get_clip_eval()
            if clip is None:
                return
            raw = clip.score_text_text(state.visual_keywords_en, state.prompt)
            norm = (raw + 1.0) / 2.0
            if raw < CLIP_PROMPT_ALIGN_THRESHOLD:
                _log.warning(
                    "[提示词匹配] ⚠ 诗-提示词语义一致性偏低 "
                    "raw=%.4f < 阈值 %.2f | 归一化=%.3f",
                    raw, CLIP_PROMPT_ALIGN_THRESHOLD, norm,
                )
                state.log("提示词匹配", "⚠ 语义一致性偏低",
                          f"CLIP 诗-提示词余弦 raw={raw:.4f}（阈值 {CLIP_PROMPT_ALIGN_THRESHOLD}），"
                          f"归一化={norm:.3f}。提示词可能遗漏诗歌核心意象，建议关注。")
            else:
                _log.debug(
                    "[提示词匹配] ✓ 诗-提示词语义一致性正常 raw=%.4f | 归一化=%.3f",
                    raw, norm,
                )
        except Exception as e:
            _log.debug("[提示词匹配] 检查跳过: %s", e)

    def _phase_image_clip(self, state: AgentState) -> AgentState:
        state.phase = Phase.IMAGE_GEN
        if not state.prompt:
            state.phase = Phase.ERROR
            state.error = "提示词为空，无法生图（请检查提示词生成步骤）"
            state.log("图像生成", "跳过", state.error)
            return state

        from config import CLIP_ENABLED, CLIP_THRESHOLD, CLIP_MAX_RETRIES

        backend_desc = state.image_backend
        if state.image_api_model:
            backend_desc += f"/{state.image_api_model}"
        state.model_usage.image_gen = backend_desc

        clip_eval = None
        if CLIP_ENABLED:
            clip_eval = self._get_clip_eval()
            if clip_eval is None and self._clip_init_failed:
                state.log("CLIP初始化", "失败（跳过评分）",
                          getattr(self, "_clip_init_error", "未知错误"))

        best_image = None
        best_score = -1.0
        max_attempts = (CLIP_MAX_RETRIES + 1) if clip_eval else 1
        current_prompt = state.prompt

        for attempt in range(max_attempts):
            is_retry = attempt > 0
            if is_retry:
                current_prompt = self._refine_prompt_for_retry(state.prompt, attempt)
                state.log("提示词精炼", f"第 {attempt} 次重试精炼",
                          current_prompt[:80], is_retry=True)

            state.log("图像生成", f"尝试 {attempt+1}/{max_attempts}",
                      f"backend={state.image_backend}", model=backend_desc, is_retry=is_retry)

            try:
                image = self.image_gen.generate(
                    prompt=current_prompt, backend=state.image_backend,
                    api_key=state.image_api_key, api_model=state.image_api_model,
                )
            except Exception as e:
                _log.exception("图像生成失败 (backend=%s, model=%s)",
                               state.image_backend, state.image_api_model)
                state.log("图像生成", "失败", str(e), is_retry=is_retry)
                state.phase = Phase.ERROR
                state.error = str(e)
                return state

            if clip_eval is None:
                state.image = image
                state.clip_msg = "✓ 生成完成（CLIP 评分未启用）"
                state.log("CLIP评分", "跳过", "CLIP 未启用")
                return state

            # ── CLIP 双锚点评分 ──────────────────────────────────────────────
            state.phase = Phase.CLIP_EVAL
            try:
                raw_b = clip_eval.score_raw_cosine(image, state.prompt)
                norm_b = (raw_b + 1.0) / 2.0

                if state.visual_keywords_en:
                    raw_a = clip_eval.score_raw_cosine(image, state.visual_keywords_en)
                    norm_a = (raw_a + 1.0) / 2.0
                    wa, wb = self._clip_anchor_weights(state.visual_keywords_en)
                    final_raw = wa * raw_a + wb * raw_b
                    final_norm = wa * norm_a + wb * norm_b
                    score_desc = (
                        f"诗-图锚点={raw_a:.3f}(×{wa}) "
                        f"提示词-图锚点={raw_b:.3f}(×{wb}) "
                        f"→ 综合={final_raw:.3f}"
                    )
                else:
                    raw_a = norm_a = 0.0
                    final_raw = raw_b
                    final_norm = norm_b
                    score_desc = f"提示词-图={raw_b:.3f}（无诗歌关键词锚点）"

                state.clip_score_poem = norm_a
                state.clip_score_prompt = norm_b
                state.clip_score_final = final_norm

                state.log("CLIP评分", f"尝试 {attempt+1}", score_desc,
                          score=final_raw, is_retry=is_retry,
                          extra={"raw_a": raw_a, "raw_b": raw_b, "final": final_raw})

                if final_raw > best_score:
                    best_score = final_raw
                    best_image = image

                if final_raw >= CLIP_THRESHOLD:
                    state.image = best_image
                    state.clip_msg = self._clip_status(final_raw, attempt + 1, exhausted=False)
                    state.log("CLIP评分",
                              f"达标（超过基础生成阈值 {CLIP_THRESHOLD}，自主模式目标另计）",
                              state.clip_msg)
                    return state
                elif attempt < max_attempts - 1:
                    state.log("CLIP评分", "未达基础阈值，准备重试",
                              f"{final_raw:.3f} < 基础阈值 {CLIP_THRESHOLD}")

            except Exception as e:
                state.log("CLIP评分", "评分异常（跳过）", str(e))
                state.image = image
                state.clip_msg = f"✓ 生成完成（CLIP 评分异常: {e}）"
                return state

        state.image = best_image
        state.clip_msg = self._clip_status(best_score, max_attempts, exhausted=True)
        state.log("CLIP评分", "重试耗尽，返回最优结果", state.clip_msg)
        return state

    def _phase_clip_only(self, state: AgentState) -> AgentState:
        from config import CLIP_ENABLED
        if not CLIP_ENABLED or state.image is None:
            state.clip_msg = "✓ 编辑完成（CLIP 评分未启用或无图像）"
            return state
        try:
            clip_eval = self._get_clip_eval()
            if clip_eval is None:
                state.clip_msg = "✓ 编辑完成（CLIP 评估器不可用）"
                return state
            raw_b = clip_eval.score_raw_cosine(state.image, state.prompt)
            norm_b = (raw_b + 1.0) / 2.0
            if state.visual_keywords_en:
                raw_a = clip_eval.score_raw_cosine(state.image, state.visual_keywords_en)
                norm_a = (raw_a + 1.0) / 2.0
                wa, wb = self._clip_anchor_weights(state.visual_keywords_en)
                final_raw = wa * raw_a + wb * raw_b
                final_norm = wa * norm_a + wb * norm_b
                desc = f"诗-图={raw_a:.3f}(×{wa}) 词-图={raw_b:.3f}(×{wb}) → 综合={final_raw:.3f}"
            else:
                raw_a = norm_a = 0.0
                final_raw = raw_b
                final_norm = norm_b
                desc = f"词-图={raw_b:.3f}"
            state.clip_score_poem = norm_a
            state.clip_score_prompt = norm_b
            state.clip_score_final = final_norm
            state.clip_msg = self._clip_status(final_raw, 1, exhausted=False)
            state.log("CLIP重评（编辑后）", "完成", desc, score=final_raw)
        except Exception as e:
            state.log("CLIP重评（编辑后）", "异常（跳过）", str(e))
            state.clip_msg = f"✓ 编辑完成（CLIP 评分异常: {e}）"
        return state

    def _phase_reflect(self, state: AgentState) -> AgentState:
        state.phase = Phase.REFLECT
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
        state.log("结果反思", "生成验收结论", state.final_reflection)
        return state

    # ═════════════════════════════════════════════════════════════════════════
    # 用户反馈注入：修改诗歌
    # ═════════════════════════════════════════════════════════════════════════
    def refine_poem(
        self, state: AgentState, feedback: str,
        refine_adapter=None, score_tolerance: float = 0.03,
    ) -> AgentState:
        adapter = refine_adapter or self.generation_adapter
        model_desc = self._adapter_desc(adapter)

        if getattr(adapter, 'backend', '') in ('local', 'local_lora'):
            state.log("诗歌修改", "⚠ 已跳过",
                      "LoRA 模型不具备改诗能力，请在「改诗模型」下拉框中选择 API 模型再试。",
                      model=model_desc)
            return state

        old_poem = state.poem
        orig_score = state.best_poem_score
        orig_lines = [l for l in old_poem.split("\n") if l.strip()]
        expected_lines = len(orig_lines)
        expected_chars = len(orig_lines[0]) if orig_lines else 5

        from core.prompts import render_messages
        msg = render_messages(
            "agent.refine_poem",
            expected_chars=expected_chars,
            expected_lines=expected_lines,
            old_poem=old_poem,
            feedback=feedback,
        )
        try:
            raw = adapter.generate(msg, max_tokens=120, temperature=0.75)
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            clean = ["".join(ch for ch in l if "一" <= ch <= "鿿") for l in lines]
            char_ok = [l for l in clean if len(l) == expected_chars]
            if len(char_ok) < expected_lines:
                state.log("诗歌修改", "⚠ 字数不符（已回滚）",
                          f"改后各行字数: {[len(l) for l in clean if l]}，"
                          f"应为 {expected_chars} 字×{expected_lines} 行。",
                          model=model_desc)
                return state
            candidate = "\n".join(char_ok[:expected_lines])
            new_score_dict = self.poem_gen.score_single_poem(candidate, state.user_input, self.score_adapter)
            if isinstance(new_score_dict, dict):
                new_score = new_score_dict.get("total", 0.0)
                detail = (
                    f"意图={new_score_dict.get('intent',0):.3f}"
                    f"[LLM={new_score_dict.get('intent_llm',0):.3f}] | "
                    f"平仄={new_score_dict.get('pingze',0):.3f} | "
                    f"押韵={new_score_dict.get('rhyme',0):.3f} | "
                    f"意象={new_score_dict.get('imagery',0):.3f} | "
                    f"聚合={new_score_dict.get('cohesion',0):.3f} | "
                    f"重复={new_score_dict.get('penalty',1):.3f} | "
                    f"必须意象={new_score_dict.get('required_coeff',1):.3f} | "
                    f"原始={new_score_dict.get('raw_total',0):.3f}"
                )
                _log.info("改后分数明细: %s", detail)
            else:
                new_score = float(new_score_dict)
                detail = f"总分={new_score:.3f}"

            threshold = max(0.0, orig_score - score_tolerance)
            if orig_score > 0 and new_score < threshold:
                state.log("诗歌修改", f"⚠ 改后得分不足（已回滚）",
                          f"改后得分 {new_score:.3f} < 原始得分 {orig_score:.3f} - "
                          f"容差 {score_tolerance} = {threshold:.3f}\n"
                          f"改后各维度：{detail}",
                          model=model_desc)
                return state

            state.poem = candidate
            state.best_poem_score = new_score
            state.model_usage.poem_gen += f" → 修改({model_desc})"

            # ── 打印改前/改后对比 ──────────────────────────────────────────
            old_one_line = " | ".join(l.strip() for l in old_poem.split('\n') if l.strip())
            new_one_line = " | ".join(l.strip() for l in candidate.split('\n') if l.strip())
            _log.info("=" * 60)
            _log.info("【改诗对比】")
            _log.info("-" * 60)
            _log.info("[改前 总分=%.3f] %s", orig_score, old_one_line)
            _log.info("[改后 总分=%.3f] %s", new_score, new_one_line)
            _log.info("  各维度: %s", detail)
            _log.info("=" * 60)

            if new_score > orig_score:
                verdict = f"✓ 提升 | 得分 {orig_score:.3f}→{new_score:.3f}"
            elif new_score == orig_score:
                verdict = f"⚠ 持平 | 得分 {orig_score:.3f}→{new_score:.3f}"
            else:
                verdict = f"⚠ 略降（容差内）| 得分 {orig_score:.3f}→{new_score:.3f}"
            state.log("诗歌修改", verdict,
                      f"修改方向：{feedback[:80]}\n改后各维度：{detail}",
                      model=model_desc, score=new_score)
        except Exception as e:
            state.log("诗歌修改", "修改异常", str(e))
        return state

    # ── 批量改诗（全自主模式用）────────────────────────────────────────────
    def refine_multiple_poems(
        self, state: AgentState, candidates: List[Dict],
        refine_adapter=None, score_tolerance: float = 0.03,
    ) -> List[Dict]:
        """对多首候选诗逐一修改，返回所有改后结果 [{poem, scores, original_poem, refined}]

        candidates: [{"poem": str, "scores": dict}, ...]，每项自带分数
        每首诗独立走 refine_poem 流程，修改成功/失败都会记录。
        """
        adapter = refine_adapter or self.generation_adapter

        if getattr(adapter, 'backend', '') in ('local', 'local_lora'):
            _log.warning("批量改诗跳过：LoRA 模型不具备改诗能力")
            return []

        results = []
        for i, cand in enumerate(candidates):
            poem_text = cand["poem"]
            orig_total = cand.get("scores", {}).get("total", state.best_poem_score)

            # 临时替换 state 的诗和分数
            saved_poem  = state.poem
            saved_score = state.best_poem_score
            state.poem  = poem_text
            state.best_poem_score = orig_total

            _log.info("批量改诗 [%d/%d] 原分=%.3f，开始修改...",
                     i + 1, len(candidates), orig_total)
            try:
                # 动态生成改诗方向（用 auto critique 管线，不写死）
                critique = self._auto_poem_critique(state)
                _log.info("批量改诗 [%d/%d] 点评: %s", i + 1, len(candidates), critique[:120])
                auto_fb = self._auto_poem_feedback(state, critique=critique)
                _log.info("批量改诗 [%d/%d] 方向: %s", i + 1, len(candidates), auto_fb)

                state = self.refine_poem(state, auto_fb, refine_adapter=adapter,
                                         score_tolerance=score_tolerance)
                refined_poem = state.poem

                if refined_poem != poem_text:
                    new_score_dict = self.poem_gen.score_single_poem(
                        refined_poem, state.user_input, self.score_adapter)
                    new_score = new_score_dict.get("total", 0.0) if isinstance(new_score_dict, dict) else float(new_score_dict)
                    results.append({
                        "poem": refined_poem,
                        "scores": new_score_dict if isinstance(new_score_dict, dict) else {"total": new_score},
                        "original_poem": poem_text,
                        "refined": True,
                    })
                    _log.info("批量改诗 [%d/%d] ✓ 成功 %.3f→%.3f",
                             i + 1, len(candidates), orig_total, new_score)
                else:
                    _log.info("批量改诗 [%d/%d] - 未变化，保留原诗", i + 1, len(candidates))
            except Exception as e:
                _log.exception("批量改诗 [%d/%d] ✗ 异常", i + 1, len(candidates))

            # 恢复 state
            state.poem            = saved_poem
            state.best_poem_score = saved_score

        return results

    # ═════════════════════════════════════════════════════════════════════════
    # 守擂进化：硬门控 + 混合制评分
    # ═════════════════════════════════════════════════════════════════════════

    # 混合制权重：本地客观维度 0.75 + pairwise 主观审美 0.25（见 config ARENA_*_WT）
    _CHALLENGER_PROMPT = (
        "你是一位精通中国古典诗词的创作专家。\n"
        "当前冠军诗作：\n{champion}\n\n"
        "创作方向：\n{feedback}\n\n"
        "请根据上述方向，创作一首全新的诗作为挑战者。\n"
        "格律铁则：每句必须恰好 {chars_per_line} 个汉字，共 {num_lines} 句，"
        "每句换行，不加任何标点或解释。\n"
        "不要重复冠军诗的措辞，尝试用不同的意象、不同的视角表达相近的意境。\n"
        "保持押韵和平仄规范。\n"
        "输出修改后的完整诗句（每行一句，{chars_per_line} 个汉字，纯汉字）："
    )

    def _local_score_champion(self, poem: str, num_lines: int,
                              chars_per_line: int,
                              state: AgentState = None) -> dict:
        """本地评分。进化阶段沿用 arena 的切题分，不虚高。"""
        topic = getattr(state, 'champion_topic', 1.0) if state else 1.0
        return self.poem_gen.scorer.local_score_poem(poem, num_lines,
                                                      chars_per_line,
                                                      topic_score=topic)

    def _pairwise_delta(self, won: bool) -> float:
        """pairwise 审美 delta：赢 +0.17，输 −0.05（见 config PAIRWISE_*_DELTA）。"""
        from config import PAIRWISE_WIN_DELTA, PAIRWISE_LOSE_DELTA
        return PAIRWISE_WIN_DELTA if won else PAIRWISE_LOSE_DELTA

    def _format_score_log(self, local: dict, pairwise_delta: float) -> str:
        """格式化评分明细。"""
        topic_part = f"切题={local.get('topic', 1.0):.2f} " if local.get('topic') is not None else ""
        return (f"本地={local['total']:.3f}("
                f"平仄={local['pingze']:.2f} 押韵={local['rhyme']:.2f} "
                f"意象={local['imagery']:.2f} 连贯={local['cohesion']:.2f} "
                f"{topic_part})"
                f" pairwise={'✓+' if pairwise_delta > 0 else '✗'}"
                f"{pairwise_delta:+.2f}"
                f" 综合={local['total'] + pairwise_delta:.3f}")

    def hard_gate_check(self, poem: str, num_lines: int,
                        chars_per_line: int) -> dict:
        """硬门控：本地规则检查，不消耗 LLM 调用。"""
        from config import BAD_PATTERNS
        reasons = []
        lines = [l.strip() for l in poem.split('\n') if l.strip()]
        if len(lines) < num_lines:
            reasons.append(f"行数不足（需{num_lines}行）")
        char_ok = [l for l in lines[:num_lines] if len(l) == chars_per_line]
        if len(char_ok) < num_lines:
            bad_info = [f"第{i+1}行{len(l)}字" for i, l in enumerate(lines[:num_lines])
                        if len(l) != chars_per_line]
            reasons.append(f"字数不符: {', '.join(bad_info[:3])}")
        if reasons:
            return {"passed": False, "reasons": reasons, "rhyme": 0.0, "pingze": 0.0}

        rhyme_score = self.poem_gen.scorer._score_rhyme(poem, num_lines)
        pingze_score = self.poem_gen.scorer._score_pingze(poem, num_lines, chars_per_line)
        if rhyme_score < 0.6:
            reasons.append(f"押韵不合格（{rhyme_score:.2f} < 0.60）")
        if pingze_score < 0.6:
            reasons.append(f"平仄不合格（{pingze_score:.2f} < 0.60）")
        poem_text = ''.join(lines)
        hits = [w for w in BAD_PATTERNS if w in poem_text]
        if hits:
            reasons.append(f"AI堆砌词汇: {', '.join(hits)}")

        return {"passed": len(reasons) == 0, "reasons": reasons,
                "rhyme": rhyme_score, "pingze": pingze_score}

    def _generate_challenger(self, champion: str, feedback: str,
                             num_lines: int, chars_per_line: int,
                             adapter) -> str | None:
        """生成一首挑战者诗，失败返回 None。"""
        prompt = self._CHALLENGER_PROMPT.format(
            champion=champion, feedback=feedback,
            chars_per_line=chars_per_line, num_lines=num_lines,
        )
        messages = [
            {"role": "system",
             "content": "你是一位精通中国古典诗词的创作专家。只输出诗句，不含任何解释。"},
            {"role": "user", "content": prompt},
        ]
        from config import POEM_TEMPERATURE
        raw = adapter.generate(messages, max_tokens=120,
                               temperature=POEM_TEMPERATURE + 0.05)
        clines = [l.strip() for l in raw.split('\n') if l.strip()]
        challenger_lines = [
            "".join(ch for ch in l if '一' <= ch <= '鿿')
            for l in clines
        ]
        challenger_lines = [l for l in challenger_lines
                            if len(l) == chars_per_line][:num_lines]
        if len(challenger_lines) < num_lines:
            return None
        return '\n'.join(challenger_lines)

    def _try_challenger(self, state: AgentState, champion: str,
                        feedback: str, direction_label: str,
                        num_lines: int, chars_per_line: int,
                        adapter, round_num: int) -> dict | None:
        """试一个挑战者：生成 → 门控 → 本地评分 → pairwise → 返回结果或 None。"""
        challenger = self._generate_challenger(
            champion, feedback, num_lines, chars_per_line, adapter,
        )
        if challenger is None:
            _log.info("  挑战者%s 格式不符，跳过", direction_label)
            return None

        gate = self.hard_gate_check(challenger, num_lines, chars_per_line)
        if not gate["passed"]:
            _log.info("  挑战者%s 硬门控拦截: %s",
                      direction_label, "; ".join(gate["reasons"]))
            state.log("擂台", f"第{round_num}轮{direction_label}·门控拦截",
                      "; ".join(gate["reasons"]))
            return None

        chal_local = self._local_score_champion(challenger, num_lines, chars_per_line, state=state)
        winner = self.poem_gen.scorer.compare_poems(
            champion, challenger, state.user_input, self.score_adapter,
        )
        chal_won = (winner == "B")
        delta = self._pairwise_delta(chal_won)
        combined = chal_local["total"] + delta
        return {"poem": challenger, "local": chal_local,
                "pairwise_won": chal_won, "delta": delta, "combined": combined,
                "direction": direction_label}

    def _evolve_champion(self, state: AgentState, refine_adapter=None,
                         evolution_rounds: int = 3):
        """单轨守擂进化：每轮 2 个不同方向挑战者，硬门控 + 本地评分 + pairwise。

        是一个生成器，每轮 yield state 供 UI 刷新。
        """
        from config import CHALLENGERS_PER_ROUND
        adapter = refine_adapter or self.score_adapter
        champion = state.poem
        lines = [l for l in champion.split('\n') if l.strip()]
        num_lines = len(lines)
        chars_per_line = len(lines[0]) if lines else 5

        champ_local = self._local_score_champion(champion, num_lines, chars_per_line, state=state)
        _log.info("擂台·当前擂主 %s", self._format_score_log(champ_local, 0.0))

        topic_score = getattr(state, 'champion_topic', 1.0)
        topic_hint = ""
        if topic_score < 0.7:
            topic_hint = (f"\n\n【重要】当前诗作与用户要求的主题契合度偏低（切题分仅{topic_score:.1f}），"
                          "请优先考虑增强主题相关性，使诗中意象、场景、情感紧紧围绕用户要求展开。")

        for evo_round in range(evolution_rounds):
            # 1. 生成多个不同方向的改诗建议
            critique = self._auto_poem_critique(state)
            if topic_hint:
                critique = topic_hint + "\n" + critique
            directions = []
            for ci in range(CHALLENGERS_PER_ROUND):
                fb = self._auto_poem_feedback(state, critique=(
                    critique if ci == 0 else
                    critique + "\n请从另一个完全不同的维度给出修改建议，"
                    "不要重复之前的建议方向。"
                ))
                if fb not in [d[0] for d in directions]:
                    directions.append((fb, f"方向{ci+1}"))
            _log.info("擂台第%d轮·%d个方向:", evo_round + 1, len(directions))
            for fb, label in directions:
                _log.info("  %s: %s", label, fb)

            # 2. 尝试每个方向的挑战者
            best_result = None
            for fb, label in directions:
                result = self._try_challenger(
                    state, champion, fb, label,
                    num_lines, chars_per_line, adapter, evo_round + 1,
                )
                if result:
                    _log.info("  %s 综合=%.3f (本地=%.3f pairwise=%s)",
                              label, result["combined"], result["local"]["total"],
                              "胜" if result["pairwise_won"] else "败")
                    if best_result is None or result["combined"] > best_result["combined"]:
                        best_result = result

            if best_result is None:
                state.log("擂台", f"第{evo_round+1}轮·无有效挑战者",
                          "所有挑战者均被门控拦截或格式不符")
                yield state
                continue

            # 3. 综合判定
            champ_combined = champ_local["total"]
            chal = best_result

            _log.info("-" * 60)
            _log.info("【擂台第%d轮·%s】", evo_round + 1, chal["direction"])
            _log.info("  擂主: %s", self._format_score_log(champ_local, 0.0))
            _log.info("  挑战: %s",
                      self._format_score_log(chal["local"], chal["delta"]))

            if chal["combined"] > champ_combined:
                _log.info("  → 攻擂成功 ✓ 综合 %.3f > %.3f",
                          chal["combined"], champ_combined)
                _log.info("  新擂主:")
                for line in chal["poem"].strip().split('\n'):
                    _log.info("    %s", line.strip())
                champion = chal["poem"]
                champ_local = chal["local"]
                state.poem = champion
                state.log("擂台", f"第{evo_round+1}轮·攻擂成功 ✓",
                          f"{chal['direction']} 综合{chal['combined']:.3f} > "
                          f"擂主{champ_combined:.3f}"
                          f"{' | pairwise审美胜出' if chal['pairwise_won'] else ''}")
            else:
                _log.info("  → 守擂成功 擂主 %.3f ≥ 挑战 %.3f",
                          champ_combined, chal["combined"])
                _log.info("  挑战者（被拒）:")
                for line in chal["poem"].strip().split('\n'):
                    _log.info("    %s", line.strip())
                state.log("擂台", f"第{evo_round+1}轮·守擂成功",
                          f"{chal['direction']} 综合{chal['combined']:.3f} ≤ "
                          f"擂主{champ_combined:.3f}"
                          f"{' | pairwise审美胜出但本地拖累' if chal['pairwise_won'] else ''}")
            _log.info("-" * 60)

            yield state

        state.poem = champion

    # ═════════════════════════════════════════════════════════════════════════
    # 用户反馈注入：改图
    # ═════════════════════════════════════════════════════════════════════════
    def refine_and_regen_image(
        self, state: AgentState, new_prompt: str,
        image_backend: str = None, image_api_key: Optional[str] = None,
        image_api_model: Optional[str] = None,
    ) -> AgentState:
        state.prompt = new_prompt
        if image_backend:
            state.image_backend = image_backend
        if image_api_key:
            state.image_api_key = image_api_key
        if image_api_model:
            state.image_api_model = image_api_model
        state.log("提示词更新", "用户手动编辑", new_prompt[:80])
        return self._phase_image_clip(state)

    def refine_poem_and_regen_image(
        self, state: AgentState, feedback: str,
        refine_adapter=None,
    ) -> AgentState:
        """LLM-driven controller 复合动作：改诗 → 重提关键词 → 重写 prompt → 重生图。

        消耗 1 次改诗 + 1 次改图预算，仅在诗-prompt 错位、反复改图无果时调度。
        """
        state.log("自主优化", "复合动作：改诗+重生图", (feedback or "")[:60])
        adapter = refine_adapter or self.score_adapter or self.prompt_adapter
        state = self.refine_poem(state, feedback, refine_adapter=adapter)
        if state.phase == Phase.ERROR:
            return state
        state = self._phase_keyword_extract(state)
        state = self._phase_prompt(state)
        state = self._phase_prompt_review(state)
        if state.phase == Phase.ERROR:
            return state
        return self._phase_image_clip(state)

    def edit_image_by_feedback(
        self, state: AgentState, feedback: str,
        planner_adapter=None, image_backend: str = None,
        image_api_key: Optional[str] = None, image_api_model: Optional[str] = None,
        edit_model: Optional[str] = None, edit_strength: float = 0.75,
        edit_api_key: Optional[str] = None,
    ) -> AgentState:
        feedback = (feedback or "").strip()
        if not feedback:
            state.log("改图规划", "跳过", "用户没有输入改图要求")
            return state
        adapter = planner_adapter or self.prompt_adapter
        model_desc = self._adapter_desc(adapter)
        if image_backend:
            state.image_backend = image_backend
        if image_api_key:
            state.image_api_key = image_api_key
        if image_api_model:
            state.image_api_model = image_api_model

        original_prompt = state.prompt or ""
        user_req_note = f"\nUser request (context only): {state.user_input}" if state.user_input else ""
        msg = [
            {
                "role": "system",
                "content": (
                    "You are an image editor specialized in Chinese poetry paintings. "
                    "Rewrite the diffusion prompt according to the user's edit request. "
                    "CRITICAL RULES:\n"
                    "1. The title, poem and visual anchors are the ONLY whitelist of drawable subjects.\n"
                    "2. Do NOT preserve unsupported subjects from the old prompt.\n"
                    "3. Apply the user's edit only to existing poem-supported elements.\n"
                    "4. Keep the art style intact.\n"
                    "Output ONLY the revised English image prompt. No explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Poem:\n{state.poem}"
                    f"\n\nTitle:\n{state.title}"
                    f"{user_req_note}\n\n"
                    f"Poem visual anchors:\n{state.visual_keywords_en}\n\n"
                    f"Current prompt:\n{original_prompt}\n\n"
                    f"User edit request:\n{feedback}\n\n"
                    "Revised image prompt:"
                ),
            },
        ]
        try:
            revised = adapter.generate(msg, max_tokens=420, temperature=0.45).strip()
            revised = self._clean_revised_prompt(revised)
            if len(revised) < 20:
                revised = self._fallback_edit_prompt(original_prompt, feedback)
                action = "模型输出过短，使用规则补全"
            else:
                action = "LLM 规划并改写 Prompt（保留原有意象）"
            state.prompt = revised
            state.model_usage.prompt_gen += f" → 改图规划({model_desc})"
            diff_log = (
                f"【改图意见】{feedback[:80]}\n\n"
                f"【原 Prompt 摘要】\n{original_prompt[:200]}\n\n"
                f"【新 Prompt】\n{revised[:300]}"
            )
            state.log("改图规划", action, diff_log, model=model_desc)
        except Exception as e:
            state.prompt = self._fallback_edit_prompt(original_prompt, feedback)
            state.log("改图规划", "规划异常，使用规则补全", str(e), model=model_desc)

        _edit_key = edit_api_key or state.image_api_key
        if edit_model and state.image is not None and _edit_key:
            try:
                edited = self.image_gen.edit(
                    image=state.image, instruction=feedback,
                    edit_model=edit_model, strength=edit_strength, api_key=_edit_key,
                )
                state.image = edited
                state.model_usage.image_gen += f" → 编辑({edit_model})"
                state.log("图像编辑", f"百炼指令编辑({edit_model})", f"指令: {feedback[:60]}")
                return self._phase_clip_only(state)
            except Exception as e:
                _log.exception("[图像编辑] 编辑 API 失败")
                state.log("图像编辑", "⚠ 编辑 API 失败，自动降级",
                          f"原因: {e}\n"
                          f"已用改写后的 Prompt（融入「{feedback[:40]}…」）重新生图。"
                          f"如对结果不满意可手动点「改写重生图」。")
        elif edit_model and not _edit_key:
            state.log("图像编辑", "⚠ 跳过编辑 API（无 API Key）",
                      "DASHSCOPE_API_KEY 未配置，已自动降级为「改写重生图」"
                      "（编辑意见已融入 Prompt）。")
        return self._phase_image_clip(state)

    def autonomous_improve_image(
        self, state: AgentState, planner_adapter=None,
        image_backend: str = None, image_api_key: Optional[str] = None,
        image_api_model: Optional[str] = None, image_mode: str = "rewrite_regen",
        edit_model: str = "wanx2.1-imageedit", edit_api_key: Optional[str] = None,
        edit_strength: float = 0.75,
    ) -> AgentState:
        adapter = planner_adapter or self.prompt_adapter or self.score_adapter
        model_desc = self._adapter_desc(adapter)
        raw_final = state.clip_score_final * 2 - 1 if state.clip_score_final else 0.0

        if state.image_edit_history:
            history_lines = "\n".join(
                f"  Round {i + 1}: {h}"
                for i, h in enumerate(state.image_edit_history)
            )
            history_block = (
                f"\n\n⚠ REPEAT PREVENTION: The edits below were already tried. "
                f"You MUST propose an edit on a COMPLETELY DIFFERENT visual aspect — "
                f"different element, different area of the composition, different quality to adjust. "
                f"Do NOT rephrase or use synonyms of a prior edit.\n"
                f"Already attempted:\n{history_lines}"
            )
        else:
            history_block = ""

        msg = [
            {
                "role": "system",
                "content": (
                    "You are a critic specialized in poetry-to-image alignment. "
                    "Your task: propose ONE concrete image edit that improves alignment between the painting and the poem. "
                    "STRICT RULE: you may ONLY suggest elements that appear in the title, poem, or 'Poem visual anchors' below. "
                    "Do NOT introduce any element from your own imagination. "
                    "Focus on composition, lighting, color balance, or making existing poem elements more vivid. "
                    "Output only the edit request in Chinese, under 60 characters."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Poem visual anchors:\n  {state.visual_keywords_en or '（未提取，请根据诗句判断）'}\n\n"
                    f"Title: {state.title}\n"
                    f"Poem:\n{state.poem}\n"
                    f"Current prompt summary: {state.prompt[:200]}\n"
                    f"CLIP raw final: {raw_final:.3f}\n"
                    f"Reflection: {state.final_reflection or state.clip_msg}"
                    f"{history_block}"
                ),
            },
        ]
        try:
            feedback = adapter.generate(msg, max_tokens=120, temperature=0.35).strip()
            feedback = feedback.splitlines()[0][:80]
            if not feedback:
                feedback = self._default_auto_feedback(state)
            state.image_edit_history.append(feedback)
            state.log("自主优化", "提出下一轮改图目标", feedback, model=model_desc)
        except Exception as e:
            feedback = self._default_auto_feedback(state)
            state.image_edit_history.append(feedback)
            state.log("自主优化", "目标生成异常，使用规则目标", f"{feedback}\n{e}", model=model_desc)

        if image_mode == "edit_api":
            from config import DASHSCOPE_API_KEY as _CFG_KEY
            _key = edit_api_key or state.image_api_key or _CFG_KEY
            return self.edit_image_instruction(
                state, instruction=feedback, edit_model=edit_model, edit_api_key=_key,
                edit_strength=edit_strength,
            )
        return self.edit_image_by_feedback(
            state, feedback=feedback, planner_adapter=adapter,
            image_backend=image_backend, image_api_key=image_api_key,
            image_api_model=image_api_model,
        )

    def edit_image_instruction(
        self, state: AgentState, instruction: str,
        edit_model: str, edit_api_key: str = None, edit_strength: float = 0.75,
    ) -> AgentState:
        from config import DASHSCOPE_API_KEY as _CFG_KEY
        _key = edit_api_key or state.image_api_key or _CFG_KEY
        model_desc = edit_model
        if not edit_model:
            state.log("图像编辑", "⚠ 未选择编辑模型", "请在「改图模型」下拉框选择百炼编辑模型。")
            return state
        if state.image is None:
            state.log("图像编辑", "⚠ 无原图", "需要先有生成图像才能进行指令编辑。")
            return state
        if not _key or "XXX" in str(_key):
            state.log("图像编辑", "⚠ API Key 未配置", "请在 config.py 中配置有效的 DASHSCOPE_API_KEY。")
            return state
        try:
            edited = self.image_gen.edit(
                image=state.image, instruction=instruction,
                edit_model=edit_model, strength=edit_strength, api_key=_key,
            )
            state.image = edited
            state.model_usage.image_gen += f" → 指令编辑({model_desc})"
            state.log("图像编辑（指令）", f"✓ {model_desc}",
                      f"指令: {instruction[:80]}（保留原图构图，仅修改指令涉及内容）",
                      model=model_desc)
            return self._phase_clip_only(state)
        except Exception as e:
            state.log("图像编辑（指令）", "失败", str(e), model=model_desc)
            return state

    # ═════════════════════════════════════════════════════════════════════════
    # 工具方法
    # ═════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _adapter_desc(adapter) -> str:
        if adapter is None:
            return "无"
        backend = getattr(adapter, "backend", "unknown")
        model = getattr(adapter, "api_model", "") or ""
        if backend == "local":
            return "Qwen2.5-1.5B（本地基础）"
        elif backend == "local_lora":
            return "Qwen2.5-1.5B+LoRA（本地微调）"
        return f"{model}（{backend}）" if model else backend

    @staticmethod
    def _refine_prompt_for_retry(original: str, retry_idx: int) -> str:
        lines = original.strip().split("\n")
        if retry_idx == 1:
            key_lines = [
                l.strip() for l in lines
                if any(l.strip().lower().startswith(k) for k in ("subject:", "environment:", "atmosphere:"))
            ]
            core = " ".join(key_lines) if key_lines else original
            return f"clear composition, detailed, {core}"
        else:
            parts = [l.split(":", 1)[1].strip() for l in lines if ":" in l]
            return ", ".join(parts[:4]) if parts else original

    @staticmethod
    def _clean_revised_prompt(text: str) -> str:
        text = text.strip().strip("`")
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"^(revised image prompt|prompt)\s*[:：]\s*", "", text, flags=re.I)
        kept = []
        for line in text.splitlines():
            if re.match(r"\s*(forbidden|negative prompt|禁加元素|负面提示词)\s*[:：]", line, flags=re.I):
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    @staticmethod
    def _fallback_edit_prompt(original: str, feedback: str) -> str:
        original = (original or "").strip()
        edit_clause = (
            "Apply this user-requested image edit while preserving the original poem imagery: "
            f"{feedback}. Keep composition coherent, visually explicit, high quality."
        )
        return f"{original}\n{edit_clause}" if original else edit_clause

    @staticmethod
    def _default_auto_feedback(state: AgentState) -> str:
        if state.visual_keywords_en:
            return f"强化诗歌核心意象：{state.visual_keywords_en[:60]}，让主体更清晰、光线更集中"
        return "强化画面主体、季节氛围和空间层次，让构图更清晰"

    @staticmethod
    def _clip_status(raw: float, attempt: int, exhausted: bool = False) -> str:
        from config import CLIP_THRESHOLD
        norm = (raw + 1.0) / 2.0
        level = "优秀 ✓" if raw >= 0.28 else ("良好" if raw >= 0.22 else "较低")
        base = f"图文一致性 {level} | 综合余弦 {raw:.3f} | 归一化 {norm:.3f}"
        if exhausted:
            base += f" | 已重试 {attempt - 1} 次，返回最优结果"
        elif attempt > 1:
            base += f" | 第 {attempt} 次尝试达标"
        return base

    @staticmethod
    def _sanitize_brief(user_goal: str) -> str:
        goal = re.sub(r"\s+", " ", (user_goal or "")).strip()
        if not goal:
            return "用户未提供明确要求，仅执行保守诗画流程。"
        return f"用户明确要求：{goal[:160]}"

    @staticmethod
    def _parse_brief_plan(raw: str) -> tuple:
        brief = ""
        plan = ""
        brief_match = re.search(r"BRIEF\s*[:：]\s*(.*?)(?=\n\s*PLAN\s*[:：]|$)", raw, flags=re.I | re.S)
        plan_match = re.search(r"PLAN\s*[:：]\s*(.*)$", raw, flags=re.I | re.S)
        if brief_match:
            brief = brief_match.group(1).strip()
        if plan_match:
            plan = plan_match.group(1).strip()
        if not brief:
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            brief = lines[0] if lines else "围绕用户要求生成诗画作品。"
        if not plan:
            plan = raw.strip()
        return brief[:220], plan[:500]

    @staticmethod
    def _clip_anchor_weights(keywords_en: str) -> tuple:
        """根据视觉关键词丰富度返回 (诗锚点权重, 提示词锚点权重)。

        关键词稀疏（<阈值词，哲理/抽象诗常见）→ 降诗锚权重避坑。
        """
        from config import (
            CLIP_POEM_WEIGHT, CLIP_PROMPT_WEIGHT,
            CLIP_SPARSE_POEM_WEIGHT, CLIP_SPARSE_PROMPT_WEIGHT,
            CLIP_SPARSE_WORD_THRESHOLD,
        )
        if not keywords_en:
            return 0.0, 1.0
        word_count = len([w for w in keywords_en.replace(",", " ").split() if len(w) > 1])
        if word_count < CLIP_SPARSE_WORD_THRESHOLD:
            _log.info("[CLIP锚点] 视觉关键词稀疏（%d 词），切换至提示词锚点主导（%.1f/%.1f）",
                      word_count, CLIP_SPARSE_POEM_WEIGHT, CLIP_SPARSE_PROMPT_WEIGHT)
            return CLIP_SPARSE_POEM_WEIGHT, CLIP_SPARSE_PROMPT_WEIGHT
        return CLIP_POEM_WEIGHT, CLIP_PROMPT_WEIGHT

    @staticmethod
    def _raw_clip(state: AgentState) -> float:
        return round(state.clip_score_final * 2 - 1, 3) if state.clip_score_final else -1.0

    @staticmethod
    def _copy_state(state: AgentState) -> AgentState:
        import copy
        snap = copy.copy(state)
        snap.trace = list(state.trace)
        snap.model_usage = copy.copy(state.model_usage)
        snap.retry_counts = dict(state.retry_counts)
        snap.image_edit_history = list(state.image_edit_history)
        snap.image_history = list(getattr(state, "image_history", []) or [])
        return snap

    def _auto_poem_critique(self, state: AgentState) -> str:
        adapter = self.score_adapter or self.generation_adapter
        raw = self._raw_clip(state)
        lines = [l.strip() for l in state.poem.split("\n") if l.strip()]
        chars_per = len(lines[0]) if lines else 7
        msg = [
            {
                "role": "system",
                "content": (
                    "你是一位精通古典诗词格律的文学评论家。"
                    "请从意境深度、画面美感、情感力度、语言锤炼、押韵合规、平仄规范六个维度，"
                    "对这首诗写一段简短的点评（150字以内）：\n"
                    "  • 先用1-2句肯定其可取之处\n"
                    "  • 再指出1-2处最值得打磨的不足（如有押韵或平仄硬伤，优先指出）\n"
                    "注意：不要直接给出修改方案，只需分析不足在哪里；"
                    "不要凭空建议新增诗中没有的人物、器物、动物或情节。"
                    "语言简练，直接输出点评，无需标题或前缀。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"诗歌（{chars_per}言，共{len(lines)}句）：\n{state.poem}\n\n"
                    f"当前图文一致性得分 {raw:.3f}（>0.28 为优秀），"
                    f"说明部分意象在画面转化时仍有空间。\n请点评："
                ),
            },
        ]
        try:
            critique = adapter.generate(msg, max_tokens=200, temperature=0.5).strip()
            return critique[:300] if len(critique) > 300 else critique or "诗意尚佳，但部分意象仍可更具体鲜明。"
        except Exception as e:
            _log.warning("自主改诗点评生成失败: %s", e)
            return "整体意境可取，但仍有句子意象不够鲜明，建议深化视觉细节。"

    def _auto_poem_feedback(self, state: AgentState, critique: str = "") -> str:
        adapter = self.score_adapter or self.generation_adapter
        lines = [l.strip() for l in state.poem.split("\n") if l.strip()]
        chars_per = len(lines[0]) if lines else 7
        msg = [
            {
                "role": "system",
                "content": (
                    "你是古典诗词改诗规划专家。"
                    "根据以下诗评，提炼出一条修改方向（60字以内），"
                    "告诉改诗模型应该在哪个维度、哪一句、如何提升。\n"
                    "要求：\n"
                    "  ① 方向要具体（指出哪句/哪联有问题），但不要给出完整替换句\n"
                    f"  ② 改后每行必须仍是 {chars_per} 个汉字，不得增减\n"
                    "  ③ 不要引入原诗和用户要求之外的具体人物、器物、动物或情节\n"
                    "  ④ 不要说'将X改为Y'这种直接替换格式，而是说'某句可以……，使意境……'\n"
                    "  ⑤ 如果原诗押韵合规，强调'必须保留原韵脚，不得改动偶句末字'\n"
                    "  ⑥ 如果修改涉及任何偶句末字，必须注明'新韵脚需与全诗协调'\n"
                    "直接输出修改方向，不要前缀或解释。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"当前诗歌：\n{state.poem}\n\n"
                    f"诗评：\n{critique}\n\n"
                    "修改方向："
                ),
            },
        ]
        try:
            fb = adapter.generate(msg, max_tokens=80, temperature=0.4).strip()
            fb = fb.splitlines()[0].strip()
            return fb[:80] or "深化诗中视觉意象，使画面感更鲜明，意境更有深度"
        except Exception as e:
            _log.warning("自主改诗修改方向生成失败: %s", e)
            return "深化诗中视觉意象，使画面感更鲜明，意境更有深度"
