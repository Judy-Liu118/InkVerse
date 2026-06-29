"""
core.agent.agent -- 诗画创作引擎

双锚点 CLIP（消弭 Poem→Prompt→Image 链的语义损耗）：
  anchor_a = visual_keywords_en  ← 从诗歌直接提取，代表诗的意境
  anchor_b = english_prompt      ← 扩散模型提示词，代表模型的理解
  clip_score_final = 0.6 × clip(image, a) + 0.4 × clip(image, b)

每个 phase 方法独立修改 AgentState，上层可按需逐步调用或一次性调用 run()。
诗歌修改/守擂逻辑见 poem_refiner.py，图像编辑逻辑见 image_editor.py。
"""
from __future__ import annotations

import re
import traceback
from typing import Optional

from core.agent.state import AgentState, Phase
from core.agent.poem_refiner import _PoemRefineMixin
from core.agent.image_editor import _ImageEditMixin
from core.logger import get_logger

_log = get_logger(__name__)


class PoetryAgent(_PoemRefineMixin, _ImageEditMixin):
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
        _MAX_RETRIES = 2
        state.log("诗歌生成", "开始生成候选（arena 海选模式）",
                  f"用户要求: {state.user_input[:80]}", model=model_desc)

        qualified_pool = []
        genre_name = ""
        for gen_round in range(1 + _MAX_RETRIES):
            try:
                result = self.poem_gen.generate_and_score(
                    state.user_input, self.score_adapter, self.generation_adapter,
                    creative_brief=state.creative_brief,
                )
                genre_name = result.get("genre_name", "")
                gated = result.get("gated", [])
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
            if len(qualified_pool) < 1:
                state.phase = Phase.ERROR
                state.error = "无合格候选诗"
                return state

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
