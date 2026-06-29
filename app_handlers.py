"""
app_handlers.py -- Gradio 事件处理器与 AgentState 序列化。

所有 on_* 函数绑定到 UI 组件，负责编排 Agent 调用并 yield 给前端刷新。
"""
import json
import os
import traceback

import gradio as gr
from PIL import Image

from core.agent.agent import PoetryAgent
from core.agent.autonomous import AutonomousConfig, autonomous_full_run
from core.agent.state import AgentState, Phase, ModelUsage, AgentStep
from config import DASHSCOPE_API_KEY
from app_utils import (
    _poem_html, _make_adapter, _parse_image_backend, _parse_edit_model,
    _save_state_image, _image_caption, _auto_image_caption, _image_history_gallery,
    IMAGE_EDIT_DEFAULT_MODEL, LAST_IMAGE_PATH,
    get_style_suffix,
)


# ── AgentState 序列化 / 反序列化 ─────────────────────────────────────────────
def _serialize_state(state: AgentState) -> str:
    """将 AgentState 中需要跨按钮传递的字段序列化为 JSON 字符串。"""
    clip_info = {}
    if state.clip_score_final > 0:
        clip_info = {
            "poem":   round(state.clip_score_poem   * 2 - 1, 4),
            "prompt": round(state.clip_score_prompt * 2 - 1, 4),
            "final":  round(state.clip_score_final  * 2 - 1, 4),
        }
    return json.dumps({
        "poem":               state.poem,
        "backup_poem":        state.backup_poem,
        "title":              state.title,
        "prompt":             state.prompt,
        "creative_brief":     state.creative_brief,
        "agent_plan":         state.agent_plan,
        "prompt_review":      state.prompt_review,
        "final_reflection":   state.final_reflection,
        "visual_keywords_en": state.visual_keywords_en,
        "last_image_path":    state.last_image_path,
        "image_history":      getattr(state, "image_history", []) or [],
        "champion_topic":       state.champion_topic,
        "champion_local_total": state.champion_local_total,
        "best_poem_score":    state.best_poem_score,
        "image_backend":      state.image_backend,
        "image_api_model":    state.image_api_model or "",
        "model_usage":        state.model_usage.as_dict(),
        "clip_info":          clip_info,
        "qualified_candidates": getattr(state, "qualified_candidates", []) or [],
        "rejected_candidates":  getattr(state, "rejected_candidates", []) or [],
        "refined_candidates":   getattr(state, "refined_candidates", []) or [],
        "poem_selection_mode":  getattr(state, "poem_selection_mode", ""),
        "trace":              [s.__dict__ for s in state.trace],
    }, ensure_ascii=False)


def _deserialize_state(state_json: str) -> AgentState:
    """从 JSON 字符串还原 AgentState（仅还原可序列化字段）。"""
    state = AgentState()
    try:
        data = json.loads(state_json)
        state.poem               = data.get("poem", "")
        state.backup_poem        = data.get("backup_poem", "")
        state.title              = data.get("title", "")
        state.prompt             = data.get("prompt", "")
        state.creative_brief     = data.get("creative_brief", "")
        state.agent_plan         = data.get("agent_plan", "")
        state.prompt_review      = data.get("prompt_review", "")
        state.final_reflection   = data.get("final_reflection", "")
        state.visual_keywords_en = data.get("visual_keywords_en", "")
        state.last_image_path    = data.get("last_image_path", "")
        state.image_history      = data.get("image_history", []) or []
        state.champion_topic       = float(data.get("champion_topic", 0.5) or 0.5)
        state.champion_local_total = float(data.get("champion_local_total", 0.0) or 0.0)
        state.best_poem_score    = float(data.get("best_poem_score", 0.0) or 0.0)
        state.image_backend      = data.get("image_backend", "local")
        state.image_api_model    = data.get("image_api_model") or None
        state.qualified_candidates = data.get("qualified_candidates", []) or []
        state.rejected_candidates  = data.get("rejected_candidates", []) or []
        state.refined_candidates   = data.get("refined_candidates", []) or []
        state.poem_selection_mode  = data.get("poem_selection_mode", "")
        if state.last_image_path and os.path.exists(state.last_image_path):
            try:
                state.image = Image.open(state.last_image_path).convert("RGB")
            except Exception as img_err:
                print(f"[State] 图片恢复失败: {img_err}")
        elif os.path.exists(LAST_IMAGE_PATH):
            try:
                state.image = Image.open(LAST_IMAGE_PATH).convert("RGB")
                state.last_image_path = LAST_IMAGE_PATH
            except Exception as img_err:
                print(f"[State] last_image 恢复失败: {img_err}")
        mu_dict = data.get("model_usage", {})
        state.model_usage = ModelUsage(
            poem_gen    = mu_dict.get("诗歌生成", ""),
            poem_scorer = mu_dict.get("诗歌评分", ""),
            title_gen   = mu_dict.get("诗名生成", ""),
            prompt_gen  = mu_dict.get("提示词生成", ""),
            image_gen   = mu_dict.get("图像生成", ""),
        )
        for item in data.get("trace", []):
            if isinstance(item, dict):
                state.trace.append(AgentStep(**{k: item[k] for k in item if k in AgentStep.__dataclass_fields__}))
    except Exception as e:
        print(f"[State] 反序列化失败: {e}")
    return state


# ── 主创作流程（流式，Agent 逐步执行）────────────────────────────────────────
def on_create(
    user_req, poem_edit, lang, style,
    poem_model_val, intent_model_val, title_model_val, prompt_model_val,
    image_backend_val,
):
    """
    outputs 顺序：
      title_out | poem_edit | poem_display | prompt_out |
      image_out | clip_score_out | agent_trace_out | agent_state_json
    """
    style_suffix = get_style_suffix(style, lang)

    generation_adapter = _make_adapter(poem_model_val, allow_lora_fallback=True)
    intent_adapter     = _make_adapter(intent_model_val)
    title_adapter      = _make_adapter(title_model_val)
    prompt_adapter     = _make_adapter(prompt_model_val)

    img_backend, img_api_model = _parse_image_backend(image_backend_val)

    agent = PoetryAgent(generation_adapter, intent_adapter, title_adapter, prompt_adapter)
    state = AgentState(
        user_input      = user_req,
        direct_poem     = poem_edit,
        lang            = lang,
        style_suffix    = style_suffix,
        image_backend   = img_backend,
        image_api_key   = DASHSCOPE_API_KEY if img_backend == "bailian" else None,
        image_api_model = img_api_model,
    )

    empty_state_json = "{}"

    yield "", "", _poem_html("", placeholder=True), "", None, "Agent 正在理解任务…", [], "", empty_state_json

    state = agent.step_plan(state)
    if state.phase == Phase.ERROR:
        yield gr.update(value="", visible=False), state.error, _poem_html(state.error, ""), "", None, state.error, [], state.trace_md(), empty_state_json
        return
    yield "", "", _poem_html("", placeholder=True), "", None, "已完成任务规划，正在生成诗文…", [], state.trace_md(), empty_state_json

    state = agent.step_poem(state)
    if state.phase == Phase.ERROR:
        yield gr.update(value="", visible=False), state.error, _poem_html(state.error, ""), "", None, state.error, [], state.trace_md(), empty_state_json
        return
    yield gr.update(value="", visible=False), state.poem, _poem_html(state.poem, ""), "正在为诗起名…", None, "", [], state.trace_md(), empty_state_json

    state = agent.step_keywords(state)
    kw_info = f"🔍 视觉锚点（CLIP 诗-图锚点）：{state.visual_keywords_en}" if state.visual_keywords_en else ""
    if kw_info:
        print(f"[Agent] {kw_info}")
        yield (
            gr.update(value="", visible=False), state.poem,
            _poem_html(state.poem, ""), "正在为诗起名…",
            None, kw_info, [], state.trace_md(), empty_state_json,
        )

    state = agent.step_title(state)
    yield gr.update(value=state.title, visible=False), state.poem, _poem_html(state.poem, state.title), "生成提示词中…", None, "", [], state.trace_md(), empty_state_json

    state = agent.step_prompt(state)
    state = agent.step_prompt_review(state)
    if state.phase == Phase.ERROR:
        yield (
            gr.update(value=state.title, visible=False), state.poem, _poem_html(state.poem, state.title),
            "❌ 提示词生成失败，请重试或修改创作要求。",
            None, "", [], state.trace_md(), empty_state_json,
        )
        return
    yield (
        gr.update(value=state.title, visible=False), state.poem, _poem_html(state.poem, state.title),
        state.prompt, None, "正在生成图像…", [], state.trace_md(), empty_state_json,
    )

    state = agent.step_image(state)

    _save_state_image(state, _image_caption(state, "开始创作", "基于 Prompt 直接生成"))

    state_json = _serialize_state(state)

    yield (
        gr.update(value=state.title, visible=False), state.poem,
        _poem_html(state.poem, state.title),
        state.prompt, state.image,
        state.clip_msg,
        _image_history_gallery(state),
        state.trace_md(),
        state_json,
    )


# ── 改诗（用户反馈注入）────────────────────────────────────────────────────────
def on_refine_poem(
    feedback: str,
    state_json: str,
    poem_model_val: str,
    intent_model_val: str,
    refine_poem_model_val: str,
    prompt_model_val: str,
    title_model_val: str,
    lang: str,
    style: str,
    image_backend_val: str,
):
    """根据用户意见修改诗歌，改完后自动重新生成提示词并重新生图。"""
    state = _deserialize_state(state_json)
    empty_state_json = "{}"
    if not feedback.strip():
        yield (
            gr.update(value=state.title, visible=False),
            state.poem, _poem_html(state.poem, state.title),
            state.prompt, state.image, state.clip_msg,
            _image_history_gallery(state), state.trace_md(), state_json,
        )
        return

    generation_adapter = _make_adapter(poem_model_val, allow_lora_fallback=True)
    intent_adapter     = _make_adapter(intent_model_val)
    refine_adapter     = _make_adapter(refine_poem_model_val)
    prompt_adapter     = _make_adapter(prompt_model_val)
    title_adapter      = _make_adapter(title_model_val)

    style_suffix = get_style_suffix(style, lang)
    img_backend, img_api_model = _parse_image_backend(image_backend_val)

    agent = PoetryAgent(generation_adapter, intent_adapter, title_adapter, prompt_adapter)

    state.lang            = lang
    state.style_suffix    = style_suffix
    state.image_backend   = img_backend
    state.image_api_key   = DASHSCOPE_API_KEY if img_backend == "bailian" else None
    state.image_api_model = img_api_model

    state = agent.refine_poem(state, feedback, refine_adapter=refine_adapter)
    state = agent._phase_keyword_extract(state)
    if agent.title_adapter is not None:
        state = agent._phase_title(state)
    yield (
        gr.update(value=state.title, visible=False),
        state.poem, _poem_html(state.poem, state.title),
        state.prompt, state.image,
        "诗已修改，正在重新生成提示词…",
        _image_history_gallery(state), state.trace_md(), _serialize_state(state),
    )

    state = agent._phase_prompt(state)
    state = agent._phase_prompt_review(state)
    yield (
        gr.update(value=state.title, visible=False),
        state.poem, _poem_html(state.poem, state.title),
        state.prompt, state.image,
        "提示词已更新，正在重新生图…",
        _image_history_gallery(state), state.trace_md(), _serialize_state(state),
    )

    state = agent._phase_image_clip(state)
    _save_state_image(state, _image_caption(state, "改诗后重生图", f"修改意见「{feedback[:40]}」"))

    state = agent._phase_reflect(state)
    state_json = _serialize_state(state)
    yield (
        gr.update(value=state.title, visible=False),
        state.poem, _poem_html(state.poem, state.title),
        state.prompt, state.image, state.clip_msg,
        _image_history_gallery(state), state.trace_md(), state_json,
    )


# ── 仅重新生图 ───────────────────────────────────────────────────────────────
def on_regen_image(prompt: str, image_backend_val: str, state_json: str):
    """仅重新生图，走 CLIP 双锚点评分 loop。"""
    state = _deserialize_state(state_json)
    img_backend, img_api_model = _parse_image_backend(image_backend_val)

    state.prompt = prompt
    state.image_backend   = img_backend
    state.image_api_key   = DASHSCOPE_API_KEY if img_backend == "bailian" else None
    state.image_api_model = img_api_model

    agent = PoetryAgent()
    state = agent.refine_and_regen_image(
        state, new_prompt=prompt,
        image_backend=img_backend,
        image_api_key=state.image_api_key,
        image_api_model=img_api_model,
    )

    _save_state_image(state, _image_caption(state, "仅重新生图", "基于当前 Prompt 直接生成"))

    state_json = _serialize_state(state)
    return state.image, state.clip_msg, _image_history_gallery(state), state.trace_md(), state_json


def on_rewrite_regen(
    feedback: str,
    prompt: str,
    image_backend_val: str,
    state_json: str,
):
    """改写重生图：LLM 将意见融入 Prompt 后用当前生图后端重新生成。"""
    state = _deserialize_state(state_json)
    feedback = (feedback or "").strip()
    if not feedback:
        return state.image, state.clip_msg or "请输入改图要求。", state.prompt or prompt, _image_history_gallery(state), state.trace_md(), state_json

    if prompt and prompt.strip():
        state.prompt = prompt

    img_backend, img_api_model = _parse_image_backend(image_backend_val)
    planner_adapter = _make_adapter("qwen-plus")
    agent = PoetryAgent(prompt_adapter=planner_adapter)

    state = agent.edit_image_by_feedback(
        state,
        feedback=feedback,
        planner_adapter=planner_adapter,
        image_backend=img_backend,
        image_api_key=DASHSCOPE_API_KEY if img_backend == "bailian" else None,
        image_api_model=img_api_model,
        edit_model=None,
        edit_api_key=DASHSCOPE_API_KEY,
    )

    _save_state_image(state, _image_caption(state, "改写重生图", f"修改意见「{feedback[:40]}」"))

    state_json = _serialize_state(state)
    return state.image, state.clip_msg, state.prompt, _image_history_gallery(state), state.trace_md(), state_json


def on_edit_image_api(
    feedback: str,
    state_json: str,
    edit_image_model_val: str,
):
    """图像编辑：直接调用百炼图像编辑 API，保留原图构图，仅按指令修改内容。"""
    state = _deserialize_state(state_json)
    feedback = (feedback or "").strip()
    if not feedback:
        return state.image, "请输入编辑指令。", _image_history_gallery(state), state.trace_md(), state_json

    edit_model, edit_strength = _parse_edit_model(edit_image_model_val)
    agent = PoetryAgent()

    state = agent.edit_image_instruction(
        state,
        instruction=feedback,
        edit_model=edit_model or IMAGE_EDIT_DEFAULT_MODEL,
        edit_api_key=DASHSCOPE_API_KEY,
        edit_strength=edit_strength,
    )

    _save_state_image(state, _image_caption(state, "图像编辑", f"在上一张图上执行「{feedback[:40]}」"))

    state_json = _serialize_state(state)
    return state.image, state.clip_msg, _image_history_gallery(state), state.trace_md(), state_json


def on_sync_display(poem_text: str):
    return _poem_html(poem_text, "")


# ── 全自主创作（autonomous_full_run）─────────────────────────────────────────
def on_autonomous_create(
    user_req: str,
    poem_edit: str,
    lang: str,
    style: str,
    poem_model_val: str,
    intent_model_val: str,
    title_model_val: str,
    prompt_model_val: str,
    image_backend_val: str,
    auto_target_score: float,
    auto_max_img_rounds: int,
    auto_allow_poem_refine: bool,
    auto_max_poem_rounds: int,
    auto_image_mode_val: str = "改写重生图（LLM 改写 Prompt 后重新生图）",
    auto_edit_model_val: str = "",
    auto_llm_driven_loop: bool = False,
):
    """全自主创作模式：Agent 自主完成生成→图像优化→改诗→重生图的完整循环。"""
    style_suffix = get_style_suffix(style, lang)

    generation_adapter = _make_adapter(poem_model_val, allow_lora_fallback=True)
    intent_adapter     = _make_adapter(intent_model_val)
    title_adapter      = _make_adapter(title_model_val)
    prompt_adapter     = _make_adapter(prompt_model_val)

    img_backend, img_api_model = _parse_image_backend(image_backend_val)
    empty_state_json = "{}"

    agent = PoetryAgent(generation_adapter, intent_adapter, title_adapter, prompt_adapter)
    state = AgentState(
        user_input      = user_req,
        direct_poem     = poem_edit,
        lang            = lang,
        style_suffix    = style_suffix,
        image_backend   = img_backend,
        image_api_key   = DASHSCOPE_API_KEY if img_backend == "bailian" else None,
        image_api_model = img_api_model,
    )
    _img_mode = "edit_api" if "图像编辑" in str(auto_image_mode_val) else "rewrite_regen"
    auto_edit_model, _ = _parse_edit_model(auto_edit_model_val)
    config = AutonomousConfig(
        target_clip_score        = float(auto_target_score),
        max_image_improve_rounds = int(auto_max_img_rounds),
        allow_poem_refine        = bool(auto_allow_poem_refine),
        max_poem_refine_rounds   = int(auto_max_poem_rounds),
        image_improve_mode       = _img_mode,
        edit_model               = auto_edit_model or IMAGE_EDIT_DEFAULT_MODEL,
        image_loop_llm_driven    = bool(auto_llm_driven_loop),
    )

    yield (
        gr.update(value="", visible=False), "", _poem_html("", placeholder=True), "",
        None, "🤖 全自主模式启动，每轮图像完成后自动刷新界面…",
        [],
        "Agent 正在规划，第一轮结果将很快出现…",
        empty_state_json,
    )

    try:
        for state in autonomous_full_run(agent, state, config=config):
            if state.image is not None:
                _save_state_image(state, _auto_image_caption(state, config))
            rounds_info = _build_auto_summary(state, config)
            state_json  = _serialize_state(state)
            clip_line   = f"{state.clip_msg}  |  {rounds_info}" if rounds_info else state.clip_msg
            yield (
                gr.update(value=state.title, visible=False),
                state.poem,
                _poem_html(state.poem, state.title),
                state.prompt,
                state.image,
                clip_line,
                _image_history_gallery(state),
                state.trace_md(),
                state_json,
            )
    except Exception as e:
        traceback.print_exc()
        yield (
            gr.update(value=state.title, visible=False),
            state.poem,
            _poem_html(state.poem, state.title),
            state.prompt,
            state.image,
            f"❌ 自主创作出错：{e}",
            _image_history_gallery(state),
            state.trace_md(),
            _serialize_state(state),
        )


def _build_auto_summary(state: AgentState, config: AutonomousConfig) -> str:
    """从 trace 中统计自主模式执行了多少轮次，生成摘要文本。"""
    img_rounds  = sum(
        1 for s in state.trace
        if ("改图循环：第" in s.action and "完成" in s.action)
        or ("图像优化：第" in s.action and "完成" in s.action)
    )
    poem_rounds = sum(
        1 for s in state.trace
        if ("品质改诗第" in s.action and "完成" in s.action)
        or ("自主改诗：第" in s.action and "完成" in s.action)
    )
    parts = [f"自主循环：改图 {img_rounds} 轮"]
    if config.allow_poem_refine:
        parts.append(f"改诗 {poem_rounds} 轮")
    return " | ".join(parts)


# ── 生成报告 ─────────────────────────────────────────────────────────────────
def on_report(user_req: str, title: str, poem: str, prompt: str, img, state_json: str):
    try:
        if img is None:
            return "⚠ 请先生成画作再导出报告"

        from core.report.generator import ReportGenerator

        model_usage_dict = {}
        clip_info        = {}
        try:
            data = json.loads(state_json)
            model_usage_dict = data.get("model_usage", {})
            clip_info        = data.get("clip_info", {})
        except Exception:
            pass

        path = ReportGenerator.generate(
            user_input  = user_req,
            poem_title  = title,
            poem_text   = poem,
            prompt_text = prompt,
            image       = img,
            model_usage = model_usage_dict or None,
            clip_info   = clip_info or None,
        )
        if path and (path.startswith("错误") or path.startswith("生成报告失败")):
            return f"❌ {path}"
        return f"✓ 报告已生成：{path}"
    except Exception as e:
        traceback.print_exc()
        return f"❌ 报告生成异常：{str(e)}"
