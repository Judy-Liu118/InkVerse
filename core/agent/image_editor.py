"""图像编辑 Mixin（用户反馈注入改图、自主优化图像）。"""
from __future__ import annotations

from typing import Optional

from core.agent.state import AgentState, Phase
from core.logger import get_logger

_log = get_logger(__name__)


class _ImageEditMixin:
    """图像编辑方法集。作为 Mixin 挂载在 PoetryAgent 上使用。"""

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
