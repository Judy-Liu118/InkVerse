"""
app.py -- 纯 UI 层，无业务逻辑
更新：
  · 集成 PoetryAgent（core.agent），流水线改为 Agent 逐步执行
  · gr.State 保存 AgentState，供"改诗"、"改图"和报告按钮复用
  · 标点渲染修复：poem 显示区和 HTML 报告的「。」均用独立 span + 可靠字体
  · 报告生成自动注入 ModelUsage 和 CLIP 双锚点分数
  · 新增「改诗」按钮：用用户的意见 + 可选不同模型修改诗歌
  · 新增「🤖 全自主创作」按钮：Agent 自主完成生成→改图→改诗的完整优化循环
"""
import os
import json
import time
import gradio as gr
from PIL import Image

from core.agent.state import AgentState, Phase
from core.agent.agent import PoetryAgent
from core.agent.autonomous import AutonomousConfig, autonomous_full_run
from core.models.adapter import ModelAdapter
from core.logger import setup_logging
from config import (
    DEEPSEEK_API_KEY, DASHSCOPE_API_KEY,
    POEM_CANDIDATE_COUNT, IMAGE_API_MODEL,
    LOCAL_LLM_AVAILABLE, LOCAL_LORA_AVAILABLE, LOCAL_IMAGE_AVAILABLE,
)

# 初始化日志系统
setup_logging()

_ROOT = os.path.dirname(os.path.abspath(__file__))
LAST_IMAGE_PATH = os.path.join(_ROOT, "outputs", "last_image.png")
IMAGE_HISTORY_DIR = os.path.join(_ROOT, "outputs", "image_history")
os.makedirs(os.path.dirname(LAST_IMAGE_PATH), exist_ok=True)
os.makedirs(IMAGE_HISTORY_DIR, exist_ok=True)


def _load_last_image():
    if os.path.exists(LAST_IMAGE_PATH):
        try:
            return Image.open(LAST_IMAGE_PATH)
        except Exception:
            return None
    return None


def _save_state_image(state: AgentState, caption: str = "") -> None:
    if state.image is None:
        return
    try:
        state.image.save(LAST_IMAGE_PATH)
        state.last_image_path = LAST_IMAGE_PATH
        if caption:
            _append_image_history(state, caption)
    except Exception as e:
        print(f"[图像] 保存失败: {e}")


def _append_image_history(state: AgentState, caption: str) -> None:
    history = list(getattr(state, "image_history", []) or [])
    score = _raw_clip_from_state(state)
    if score is not None:
        caption = f"{caption} | CLIP raw={score:.3f}"
    # 所有条目去重，避免同一轮图像重复出现
    existing_captions = {item.get("caption") for item in history}
    if caption not in existing_captions:
        filename = f"img_{int(time.time() * 1000)}.png"
        path = os.path.join(IMAGE_HISTORY_DIR, filename)
        try:
            state.image.save(path)
            item = {
                "path": path,
                "caption": caption,
                "ts": str(time.time()),
            }
            history.insert(0, item)
            state.image_history = history[:20]
            state.last_image_path = path
        except Exception as e:
            print(f"[图像历史] 保存失败: {e}")


def _image_history_gallery(state: AgentState):
    items = []
    for item in getattr(state, "image_history", []) or []:
        path = item.get("path", "")
        caption = item.get("caption", "")
        if path and os.path.exists(path):
            items.append((path, caption))
    return items


def _raw_clip_from_state(state: AgentState):
    if getattr(state, "clip_score_final", 0) > 0:
        return state.clip_score_final * 2 - 1
    return None


def _image_caption(state: AgentState, action: str, detail: str = "") -> str:
    backend = state.image_backend or "local"
    if state.image_api_model:
        backend = f"{backend}/{state.image_api_model}"
    detail = f"：{detail}" if detail else ""
    return f"{action}{detail}（{backend}）"


def _auto_image_caption(state: AgentState, config: AutonomousConfig) -> str:
    """根据 trace 中实际的动作字符串，为每轮生成的图像生成唯一 caption，
    确保图像历史去重逻辑不会因同 caption 而跳过不同轮的图。"""
    # 改图循环轮次（当前 trace 动作为 "改图循环：第 X 轮完成"）
    img_rounds = sum(1 for s in state.trace if "改图循环：第" in s.action and "完成" in s.action)
    # 改诗后重生图（当前 trace 动作为 "改诗第X轮·重走全流程"）
    poem_regen = sum(1 for s in state.trace if "改诗第" in s.action and "重走全流程" in s.action)
    # 初始生图
    initial_done = sum(1 for s in state.trace if s.action == "生图完成")

    if poem_regen:
        return _image_caption(state, f"改诗第 {poem_regen} 轮后重生图")
    if img_rounds:
        mode = "图像编辑" if config.image_improve_mode == "edit_api" else "改写重生图"
        return _image_caption(state, f"自主{mode}第 {img_rounds} 轮",
                              f"强度 {getattr(config, 'edit_strength', 0.75):.2f}" if mode == "图像编辑" else "")
    if initial_done:
        return _image_caption(state, "自主创作初始图", "基于最终诗生成")
    return _image_caption(state, "自主创作", "")


# 图像风格映射统一从 config 引入，保证 eval 与 UI 同源
from config import STYLE_MAP, STYLE_MAP_CN, get_style_suffix  # noqa: E402

# 标点字体 style（确保「，」「。」使用包含这些字形的字体）
_PUNCT_STYLE = (
    "font-family:'Noto Serif SC','SimSun','宋体','Microsoft YaHei',serif;"
    "font-size:inherit;"
)


# ── 诗文 HTML 渲染 ────────────────────────────────────────────────────────────
def _poem_html(poem_text: str, title: str = "", placeholder: bool = False) -> str:
    if placeholder or not (poem_text and poem_text.strip()):
        return (
            '<div style="font-family:\'ZCOOL XiaoWei\',\'STKaiti\',\'楷体\',serif;'
            'color:#c9b897;text-align:center;padding:60px 20px;'
            'background:#fdf5e6;border:1px solid #d4c29a;border-radius:3px;'
            'font-size:1rem;letter-spacing:0.3em;">'
            "· 诗文将在此呈现 ·</div>"
        )

    import re
    lines = [l.strip() for l in poem_text.strip().split("\n") if l.strip()]
    clean_lines = []
    for line in lines:
        clean = re.sub(r"[，。！？；、,.;!?]", "", line).strip()
        if clean:
            clean_lines.append(clean)

    rows_html = ""
    for i in range(0, len(clean_lines), 2):
        # 标点使用独立 span + 可靠字体，解决 ZCOOL XiaoWei 不含「。」字形的问题
        first  = clean_lines[i] + f'<span style="{_PUNCT_STYLE}">，</span>'
        if i + 1 < len(clean_lines):
            second = (
                clean_lines[i + 1]
                + f'<span style="{_PUNCT_STYLE}">。</span>'
            )
            rows_html += f'<div style="margin:0.45em 0;">{first}{second}</div>'
        else:
            rows_html += f'<div style="margin:0.45em 0;">{first}</div>'

    title_block = ""
    if title and title.strip():
        title_block = (
            f'<div style="font-size:1.3rem;letter-spacing:0.5em;color:#a83030;'
            f'font-weight:500;margin-bottom:18px;padding-bottom:10px;'
            f'border-bottom:1px solid #d4c29a;">{title.strip()}</div>'
        )

    return (
        '<div style="'
        'font-family:\'ZCOOL XiaoWei\',\'Noto Serif SC\',\'STKaiti\',\'楷体\',\'SimSun\',\'Microsoft YaHei\',serif;'
        'font-size:1.5rem;line-height:2.1;text-align:center;'
        'padding:32px 36px 28px;letter-spacing:0.22em;'
        'background:#fdf5e6;color:#28190a;'
        'border:1px solid #c8b48a;border-radius:3px;'
        'box-shadow:inset 0 0 24px rgba(197,153,62,0.05),'
        '3px 5px 14px rgba(40,25,10,0.10);'
        'min-height:160px;">'
        f"{title_block}{rows_html}</div>"
    )


# ── Adapter 工厂 ──────────────────────────────────────────────────────────────
def _make_adapter(model_choice: str, *, allow_lora_fallback: bool = False) -> ModelAdapter:
    """
    构造 ModelAdapter。

    `allow_lora_fallback`：只在调用方是「诗歌生成」时设 True。其它任务
    （评分/起名/提示词等）需要结构化输出或英文，LoRA 给不了正确结果，
    API 失败时静默走 LoRA 反而会让下游静默崩坏，所以默认禁止降级。
    """
    if not model_choice:
        model_choice = "qwen-plus"
    kwargs = {"allow_lora_fallback": allow_lora_fallback}
    if model_choice == "local_base":
        return ModelAdapter(backend="local",      api_key=None, api_model=None, **kwargs)
    elif model_choice == "local_lora":
        return ModelAdapter(backend="local_lora", api_key=None, api_model=None, **kwargs)
    elif model_choice.startswith("deepseek"):
        return ModelAdapter(backend="deepseek",   api_key=DEEPSEEK_API_KEY, api_model=model_choice, **kwargs)
    elif model_choice.startswith("qwen"):
        return ModelAdapter(backend="qwen",       api_key=DASHSCOPE_API_KEY, api_model=model_choice, **kwargs)
    return ModelAdapter(backend="qwen", api_key=DASHSCOPE_API_KEY, api_model="qwen-plus", **kwargs)


def _parse_image_backend(val: str):
    """解析图像后端选择值，返回 (backend, api_model)。"""
    if val and val.startswith("bailian:"):
        return "bailian", val.split(":", 1)[1]
    return "local", None


def _parse_edit_model(val: str):
    """解析改图模型选择值，返回 (edit_model_or_None, edit_strength)。
    空字符串 → 全量重生图（返回 None）；"edit:xxx" → 百炼编辑 API。
    """
    if val and val.startswith("edit:"):
        return val[5:], 0.75   # 默认强度 0.75
    return None, 0.75


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

    # ── 构建 Agent 和初始状态 ────────────────────────────────────────────────
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

    # 0. 重置
    yield "", "", _poem_html("", placeholder=True), "", None, "Agent 正在理解任务…", [], "", empty_state_json

    # 0b. Agent 先规划
    state = agent.step_plan(state)
    if state.phase == Phase.ERROR:
        yield gr.update(value="", visible=False), state.error, _poem_html(state.error, ""), "", None, state.error, [], state.trace_md(), empty_state_json
        return
    yield "", "", _poem_html("", placeholder=True), "", None, "已完成任务规划，正在生成诗文…", [], state.trace_md(), empty_state_json

    # 1. 诗歌生成
    state = agent.step_poem(state)
    if state.phase == Phase.ERROR:
        yield gr.update(value="", visible=False), state.error, _poem_html(state.error, ""), "", None, state.error, [], state.trace_md(), empty_state_json
        return
    yield gr.update(value="", visible=False), state.poem, _poem_html(state.poem, ""), "正在为诗起名…", None, "", [], state.trace_md(), empty_state_json

    # 1b. 视觉关键词提取（提取后立即显示锚点，帮用户理解 CLIP 锚点来源）
    state = agent.step_keywords(state)
    kw_info = f"🔍 视觉锚点（CLIP 诗-图锚点）：{state.visual_keywords_en}" if state.visual_keywords_en else ""
    if kw_info:
        print(f"[Agent] {kw_info}")
        yield (
            gr.update(value="", visible=False), state.poem,
            _poem_html(state.poem, ""), "正在为诗起名…",
            None, kw_info, [], state.trace_md(), empty_state_json,
        )

    # 2. 诗名生成
    state = agent.step_title(state)
    yield gr.update(value=state.title, visible=False), state.poem, _poem_html(state.poem, state.title), "生成提示词中…", None, "", [], state.trace_md(), empty_state_json

    # 3. 提示词生成
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

    # 4. 图像生成 + CLIP 双锚点评分
    state = agent.step_image(state)

    _save_state_image(state, _image_caption(state, "开始创作", "基于 Prompt 直接生成"))

    # 序列化 AgentState 供后续按钮使用
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
    """
    根据用户意见修改诗歌，改完后自动重新生成提示词并重新生图。
    流程：refine_poem → _phase_prompt → _phase_prompt_review → _phase_image_clip → _phase_reflect
    """
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

    # ── 更新 state 的图像参数（改诗后重生图需要）────────────────────────────
    state.lang            = lang
    state.style_suffix    = style_suffix
    state.image_backend   = img_backend
    state.image_api_key   = DASHSCOPE_API_KEY if img_backend == "bailian" else None
    state.image_api_model = img_api_model

    # ── Step 1：改诗 ─────────────────────────────────────────────────────────
    state = agent.refine_poem(state, feedback, refine_adapter=refine_adapter)
    # 提取关键词供后续提示词自检使用
    state = agent._phase_keyword_extract(state)
    # 改诗后重新生成标题（避免旧标题与新诗内容或创作要求矛盾）
    if agent.title_adapter is not None:
        state = agent._phase_title(state)
    yield (
        gr.update(value=state.title, visible=False),
        state.poem, _poem_html(state.poem, state.title),
        state.prompt, state.image,
        "诗已修改，正在重新生成提示词…",
        _image_history_gallery(state), state.trace_md(), _serialize_state(state),
    )

    # ── Step 2：重新生成提示词 ────────────────────────────────────────────────
    state = agent._phase_prompt(state)
    state = agent._phase_prompt_review(state)
    yield (
        gr.update(value=state.title, visible=False),
        state.poem, _poem_html(state.poem, state.title),
        state.prompt, state.image,
        "提示词已更新，正在重新生图…",
        _image_history_gallery(state), state.trace_md(), _serialize_state(state),
    )

    # ── Step 3：重新生图 + CLIP 评分 ─────────────────────────────────────────
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

    # 用编辑后的 prompt 更新状态
    state.prompt = prompt
    state.image_backend   = img_backend
    state.image_api_key   = DASHSCOPE_API_KEY if img_backend == "bailian" else None
    state.image_api_model = img_api_model

    # 复用 PoetryAgent 的图像 + CLIP 步骤（不需要 LLM adapter）
    agent = PoetryAgent()  # 纯图像操作，无需文本生成适配器
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
    """
    改写重生图：LLM 将意见融入 Prompt 后用当前生图后端重新生成。
    不保留原图构图，适合大幅度改动。
    """
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
        edit_model=None,        # 不使用编辑 API，走全量重生图路径
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
    """
    图像编辑：直接调用百炼图像编辑 API，保留原图构图，仅按指令修改内容。
    需要：① 已有原图  ② 选择了百炼编辑模型  ③ 配置了 DASHSCOPE_API_KEY
    注意：edit_image_model_val 格式为 "edit:模型名"，如 "edit:wanx2.1-imageedit"。
    """
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
    """
    全自主创作模式：Agent 自主完成生成→图像优化→改诗→重生图的完整循环，
    无需用户在中间干预任何步骤。

    outputs 顺序（与 on_create 相同，方便复用同一组输出组件）：
      title_out | poem_edit | poem_display | prompt_out |
      image_out | clip_score_out | agent_trace_out | agent_state_json
    """
    style_suffix = get_style_suffix(style, lang)

    generation_adapter = _make_adapter(poem_model_val, allow_lora_fallback=True)
    intent_adapter     = _make_adapter(intent_model_val)
    title_adapter      = _make_adapter(title_model_val)
    prompt_adapter     = _make_adapter(prompt_model_val)

    img_backend, img_api_model = _parse_image_backend(image_backend_val)
    empty_state_json = "{}"

    # ── 构建 Agent、State 和自主配置 ─────────────────────────────────────────
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

    # 启动提示
    yield (
        gr.update(value="", visible=False), "", _poem_html("", placeholder=True), "",
        None, "🤖 全自主模式启动，每轮图像完成后自动刷新界面…",
        [],
        "Agent 正在规划，第一轮结果将很快出现…",
        empty_state_json,
    )

    # ── 消费生成器：每轮图像完成后立即刷新 UI ────────────────────────────────
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
        import traceback
        traceback.print_exc()
        # 错误不写入诗文框，保留当前诗文，只在 CLIP 状态栏显示错误
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

        # 从 state_json 还原 model_usage 和 clip 分数
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
        import traceback
        traceback.print_exc()
        return f"❌ 报告生成异常：{str(e)}"


# ── AgentState 序列化 / 反序列化 ─────────────────────────────────────────────
def _serialize_state(state: AgentState) -> str:
    """将 AgentState 中需要跨按钮传递的字段序列化为 JSON 字符串。"""
    clip_info = {}
    if state.clip_score_final > 0:
        # 还原 raw 分数（norm = (raw+1)/2 → raw = 2×norm - 1）
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
    from core.agent.state import ModelUsage
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
        mu_dict                  = data.get("model_usage", {})
        state.model_usage = ModelUsage(
            poem_gen   = mu_dict.get("诗歌生成", ""),
            poem_scorer = mu_dict.get("诗歌评分", ""),
            title_gen  = mu_dict.get("诗名生成", ""),
            prompt_gen = mu_dict.get("提示词生成", ""),
            image_gen  = mu_dict.get("图像生成", ""),
        )
        from core.agent.state import AgentStep
        for item in data.get("trace", []):
            if isinstance(item, dict):
                state.trace.append(AgentStep(**{k: item[k] for k in item if k in AgentStep.__dataclass_fields__}))
    except Exception as e:
        print(f"[State] 反序列化失败: {e}")
    return state


# ── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=ZCOOL+XiaoWei&family=Noto+Serif+SC:wght@400;500;600&display=swap');

:root {
    --bg:        #f4ecda;
    --bg2:       #ede0c4;
    --ink:       #28190a;
    --ink-mid:   #5a4430;
    --ink-faint: #9e8870;
    --red:       #a83030;
    --red-h:     #c04040;
    --gold:      #b8892e;
    --border:    #cbb98a;
    --panel:     #faf3e2;
    --shadow:    rgba(40,25,10,0.13);
}

html, body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Noto Serif SC', '宋体', serif !important;
    color: var(--ink) !important;
}

#app-header { text-align:center; padding:30px 0 18px; border-bottom:1px solid var(--border); margin-bottom:26px; }
#app-header h1 { font-family:'ZCOOL XiaoWei','楷体',serif !important; font-size:2.9rem !important; letter-spacing:0.5em !important; color:var(--ink) !important; margin:0 !important; font-weight:400 !important; text-shadow:1px 2px 8px var(--shadow); }
#app-header .sub { font-size:0.8rem; letter-spacing:0.38em; color:var(--ink-faint); margin-top:7px; }

.sec { font-size:0.7rem; letter-spacing:0.45em; color:var(--gold); text-align:center; margin:18px 0 7px; display:flex; align-items:center; gap:10px; }
.sec::before, .sec::after { content:''; flex:1; height:1px; background:var(--border); opacity:0.65; }

label > span.svelte-1gfkn6j, .label-wrap > span { display:none !important; }

textarea, input[type="text"] { background:var(--panel) !important; border:1px solid var(--border) !important; border-radius:3px !important; color:var(--ink) !important; font-family:'Noto Serif SC','宋体',serif !important; font-size:0.96rem !important; line-height:1.75 !important; padding:10px 14px !important; resize:vertical !important; transition:border-color 0.2s,box-shadow 0.2s; }
textarea:focus, input[type="text"]:focus { border-color:var(--gold) !important; box-shadow:0 0 0 3px rgba(184,137,46,0.12) !important; outline:none !important; }
textarea::placeholder { color:var(--ink-faint) !important; }

#title-out textarea, #title-out input { font-family:'ZCOOL XiaoWei','楷体',serif !important; font-size:2rem !important; text-align:center !important; color:var(--red) !important; letter-spacing:0.38em !important; background:transparent !important; border:none !important; border-bottom:1px solid var(--border) !important; border-radius:0 !important; padding:6px 0 10px !important; font-weight:500 !important; }

#poem-edit textarea { font-size:1.04rem !important; line-height:1.9 !important; }
#prompt-out textarea { font-family:'Courier New',monospace !important; font-size:0.82rem !important; color:var(--ink-mid) !important; line-height:1.65 !important; border-style:dashed !important; }
#report-status textarea { font-size:0.83rem !important; color:var(--ink-mid) !important; background:transparent !important; border:none !important; border-top:1px dashed var(--border) !important; border-radius:0 !important; padding:5px 0 !important; }

#image-out { border:1px solid var(--border) !important; border-radius:3px !important; background:var(--panel) !important; box-shadow:3px 5px 18px var(--shadow) !important; overflow:hidden; }

button.primary { background:var(--red) !important; color:#fff !important; border:none !important; font-family:'Noto Serif SC',serif !important; font-size:1.02rem !important; letter-spacing:0.22em !important; padding:11px 0 !important; border-radius:2px !important; box-shadow:0 2px 8px rgba(168,48,48,0.28) !important; transition:background 0.18s,transform 0.12s !important; }
button.primary:hover { background:var(--red-h) !important; transform:translateY(-1px) !important; }
button.secondary { background:transparent !important; border:1px solid var(--border) !important; color:var(--ink-mid) !important; font-family:'Noto Serif SC',serif !important; font-size:0.9rem !important; letter-spacing:0.15em !important; border-radius:2px !important; transition:border-color 0.18s,color 0.18s !important; }
button.secondary:hover { border-color:var(--gold) !important; color:var(--ink) !important; }

#seal { width:54px; height:54px; border:2px solid var(--red); border-radius:4px; color:var(--red); font-family:'ZCOOL XiaoWei','STKaiti','楷体',serif; font-size:0.66rem; display:flex; flex-direction:column; align-items:center; justify-content:center; opacity:0.7; transform:rotate(-8deg); margin:18px auto 6px; line-height:1.5; letter-spacing:0.06em; user-select:none; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg2); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
footer { display:none !important; }

/* 全自主创作按钮：深青色，区别于普通红色 primary */
#auto-btn { background:#1a5c52 !important; box-shadow:0 2px 8px rgba(26,92,82,0.32) !important; letter-spacing:0.28em !important; font-size:1.05rem !important; }
#auto-btn:hover { background:#236e62 !important; }
#auto-desc p { font-size:0.78rem !important; color:var(--ink-faint) !important; line-height:1.6 !important; margin:4px 0 10px !important; }
"""

# API 模型列表（始终可用）
_API_MODEL_CHOICES = [
    ("DeepSeek-Chat", "deepseek-chat"),
    ("DeepSeek-V4-Flash", "deepseek-v4-flash"),
    ("DeepSeek-V4-Pro", "deepseek-v4-pro"),
    ("DeepSeek-R1", "deepseek-reasoner"),
    ("通义千问-Turbo", "qwen-turbo"),
    ("通义千问-Plus", "qwen-plus"),
    ("通义千问-Max", "qwen-max"),
    ("通义千问-Qwen3.7-Plus（慢但质量高）", "qwen3.7-plus"),
    ("通义千问-Qwen3.6-Flash（高速）", "qwen3.6-flash"),
    ("Qwen2.5-1.5B API（非量化，对比 LoRA 用）", "qwen2.5-1.5b-instruct"),
]

# 本地选项仅在权重目录存在时注入（路径由 .env 中的 BASE_MODEL_PATH / LORA_PATH 控制）
_LOCAL_BASE_CHOICE = [("本地基础模型 (Qwen2.5-1.5B)", "local_base")] if LOCAL_LLM_AVAILABLE else []
_LOCAL_LORA_CHOICE = [("本地微调模型 (Qwen2.5-1.5B+LoRA)", "local_lora")] if LOCAL_LORA_AVAILABLE else []

MODEL_CHOICES = _LOCAL_BASE_CHOICE + _LOCAL_LORA_CHOICE + _API_MODEL_CHOICES

# 诗歌生成把 LoRA 放第一位；LoRA 不可用时回退到 API 列表
POEM_MODEL_CHOICES = (_LOCAL_LORA_CHOICE + _API_MODEL_CHOICES) if LOCAL_LORA_AVAILABLE else _API_MODEL_CHOICES

# 改诗专用：只允许 API 模型（LoRA 不具备改诗能力）
REFINE_POEM_MODEL_CHOICES = _API_MODEL_CHOICES

# 默认模型：LoRA 可用时优先，否则用 qwen-plus
_DEFAULT_POEM_MODEL = "local_lora" if LOCAL_LORA_AVAILABLE else "qwen-plus"

# 图像编辑模型（百炼图像编辑 API）
# 以下为已确认可用的模型 ID（2025 年百炼平台）：
IMAGE_EDIT_DEFAULT_MODEL = "qwen-image-edit-max"

IMAGE_EDIT_MODEL_CHOICES = [
    ("百炼 · qwen-image-edit-max（通义图像编辑 Max）",      "edit:qwen-image-edit-max"),
    ("百炼 · qwen-image-edit-plus（通义图像编辑 Plus）",    "edit:qwen-image-edit-plus"),
    ("百炼 · qwen-image-edit（通义图像编辑）",              "edit:qwen-image-edit"),
    ("百炼 · qwen-image-edit-max-2026-01-16（指定版本）",   "edit:qwen-image-edit-max-2026-01-16"),
    ("百炼 · wanx2.1-imageedit（万象图像编辑）",            "edit:wanx2.1-imageedit"),
    ("全量重生图（LLM 改写 Prompt，不保留构图）",            ""),
]

_LOCAL_IMAGE_CHOICE = [("本地 Z-Image（离线量化）", "local")] if LOCAL_IMAGE_AVAILABLE else []

# 已基于 eval_clip spot check 剪掉测过效果不水墨的 wanx2.1 系列（工笔/插画倾向）；
# 测过的 best 三档（wan2.7-image-pro / Qwen-Image Max / 2.0 Pro）按风格标注；
# 其余未测档位保留原状，留作用户自行尝试。
IMAGE_BACKEND_CHOICES = _LOCAL_IMAGE_CHOICE + [
    ("百炼 · Z-Image Turbo API（非量化，推荐）",                "bailian:z-image-turbo"),
    ("百炼 · wan2.7-image-pro（极简禅意，留白极致）",           "bailian:wan2.7-image-pro"),
    ("百炼 · Qwen-Image Plus",                                 "bailian:qwen-image-plus"),
    ("百炼 · Qwen-Image Max（清润水墨）",                       "bailian:qwen-image-max"),
    ("百炼 · Qwen-Image 2.0",                                  "bailian:qwen-image-2.0"),
    ("百炼 · Qwen-Image 2.0 Pro（推荐，大写意水墨，跨主题最稳）",  "bailian:qwen-image-2.0-pro"),
    ("百炼 · Qwen-Image 2.0 2026-03-03",                       "bailian:qwen-image-2.0-2026-03-03"),
]

# 默认图像后端：本地 Z-Image 可用时优先（生成快），否则用 Qwen-Image 2.0 Pro
# （n=10 baseline 全维度优于 Max：§5 Spearman 0.308 / 翻车修复 #4 #9 / CLIP 天花板 0.397）
_DEFAULT_IMAGE_BACKEND = "local" if LOCAL_IMAGE_AVAILABLE else "bailian:qwen-image-2.0-pro"

# ── Gradio Blocks ─────────────────────────────────────────────────────────────
with gr.Blocks(
    title="诗画墨语",
    theme=gr.themes.Base(
        primary_hue="orange", neutral_hue="stone",
        font=[gr.themes.GoogleFont("Noto Serif SC"), "serif"],
    ),
    css=CSS,
) as demo:

    # 持久化 AgentState（JSON 序列化，跨按钮共享）
    agent_state = gr.State("{}")

    gr.HTML("""
    <div id="app-header">
        <h1>诗　画　墨　语</h1>
        <div class="sub">以诗入画 &middot; 以意生境 &middot; InkVerse</div>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ── 左侧控制面板 ──────────────────────────────────────────────────────
        with gr.Column(scale=4, min_width=280):
            gr.HTML('<div class="sec">创 作 要 求</div>')
            user_req = gr.Textbox(
                show_label=False,
                placeholder="例：写一首以春天为主题的七言绝句……",
                lines=4,
            )

            gr.HTML('<div class="sec">诗 文 输 入 / 编 辑</div>')
            poem_edit = gr.Textbox(
                show_label=False,
                placeholder="粘贴已有诗作可直接配图；\nAI 生成后诗文也将显示于此，可自由修改。",
                lines=5, elem_id="poem-edit", interactive=True,
            )

            # ── 改诗区域（Agent 新增功能）─────────────────────────────────
            gr.HTML('<div class="sec">改 诗（可选不同模型）</div>')
            with gr.Row():
                refine_feedback = gr.Textbox(
                    show_label=False,
                    placeholder="例：意境太浅，改得更深沉、更有禅意",
                    lines=2, scale=3,
                )
                refine_poem_model = gr.Dropdown(
                    choices=REFINE_POEM_MODEL_CHOICES,
                    value="qwen-plus",
                    label="改诗模型（仅 API）", show_label=True, scale=2,
                )
            refine_poem_btn = gr.Button("✦ 改诗", variant="secondary")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="sec">语言</div>')
                    lang_radio = gr.Radio(
                        choices=["英文", "中文"], value="英文",
                        show_label=False, interactive=True,
                    )
                with gr.Column(scale=2):
                    gr.HTML('<div class="sec">图 像 风 格</div>')
                    style_drop = gr.Dropdown(
                        choices=list(STYLE_MAP.keys()), value="水墨画",
                        show_label=False, interactive=True,
                    )

            # ── 模型配置 ───────────────────────────────────────────────────
            gr.HTML('<div class="sec">模 型 配 置</div>')
            with gr.Row():
                intent_model = gr.Dropdown(
                    choices=MODEL_CHOICES, value="qwen-plus",
                    label="意图评分模型", show_label=True,
                )
                poem_model = gr.Dropdown(
                    choices=POEM_MODEL_CHOICES, value=_DEFAULT_POEM_MODEL,
                    label="诗歌生成模型", show_label=True,
                )
                title_model = gr.Dropdown(
                    choices=MODEL_CHOICES, value="qwen-plus",
                    label="诗名生成模型", show_label=True,
                )
                prompt_model = gr.Dropdown(
                    choices=MODEL_CHOICES, value="qwen-max",
                    label="提示词生成模型", show_label=True,
                )
            gr.Markdown("💡 各步骤可独立选择模型；改诗时可再换模型。")

            # ── 图像后端 ───────────────────────────────────────────────────
            gr.HTML('<div class="sec">图 像 生 成 后 端</div>')
            image_backend = gr.Dropdown(
                choices=IMAGE_BACKEND_CHOICES,
                value=_DEFAULT_IMAGE_BACKEND,
                show_label=False,
                interactive=True,
            )

            gr.HTML('<div class="sec">操　　作</div>')
            with gr.Row():
                submit_btn = gr.Button("✦ 开始创作", variant="primary", scale=3)
                report_btn = gr.Button("生成报告", variant="secondary", scale=2)
            report_out = gr.Textbox(
                show_label=False, interactive=False,
                elem_id="report-status", lines=1,
            )

            # ── 全自主创作区域 ─────────────────────────────────────────────
            gr.HTML('<div class="sec">🤖 全 自 主 创 作</div>')
            gr.Markdown(
                "Agent 自主完成：生成诗 → 品质筛选 → 改诗 → 提取意象 → 生图 → CLIP 评分 → 改图，"
                "无需人工介入，最终返回最优结果。",
                elem_id="auto-desc",
            )
            with gr.Row():
                auto_target_score = gr.Slider(
                    minimum=0.10, maximum=0.40, value=0.30, step=0.01,
                    label="目标 CLIP 分（raw，达到提前停止）",
                    interactive=True, scale=3,
                )
                auto_max_img_rounds = gr.Slider(
                    minimum=0, maximum=4, value=2, step=1,
                    label="最大改图轮次",
                    interactive=True, scale=1,
                )
            with gr.Row():
                auto_allow_poem_refine = gr.Checkbox(
                    value=True, label="允许 Agent 自主改诗",
                    interactive=True, scale=1,
                )
                auto_max_poem_rounds = gr.Slider(
                    minimum=0, maximum=3, value=1, step=1,
                    label="最大改诗轮次",
                    interactive=True, scale=2,
                )
            auto_image_mode = gr.Radio(
                choices=[
                    "改写重生图（LLM 改写 Prompt 后重新生图）",
                    "图像编辑（百炼 API，保留原图构图）",
                ],
                value="改写重生图（LLM 改写 Prompt 后重新生图）",
                label="自主改图模式",
                interactive=True,
            )
            auto_edit_model = gr.Dropdown(
                choices=IMAGE_EDIT_MODEL_CHOICES[:-1],
                value=f"edit:{IMAGE_EDIT_DEFAULT_MODEL}",
                label="自主图像编辑模型",
                interactive=True,
            )
            auto_llm_driven_loop = gr.Checkbox(
                value=False,
                label="LLM 驱动改图循环（实验）",
                info="勾选后改图循环由 LLM 决定调 edit_image / refine_poem_and_regen / stop；默认走写死流程。",
                interactive=True,
            )
            auto_btn = gr.Button("🤖 全自主创作", variant="primary", elem_id="auto-btn")

            gr.HTML('<div id="seal">詩<br>畫<br>工坊</div>')

        # ── 右侧展示面板 ──────────────────────────────────────────────────────
        with gr.Column(scale=8, min_width=500):
            title_out = gr.Textbox(
                show_label=False, interactive=False,
                elem_id="title-out", lines=1, visible=False,  # 初始隐藏；回调中按 visible=False 更新
            )
            gr.HTML('<div class="sec">诗 文 展 示</div>')
            poem_display = gr.HTML(value=_poem_html("", placeholder=True))

            with gr.Row(equal_height=True):
                with gr.Column(scale=5):
                    gr.HTML('<div class="sec">绘画提示词（可编辑）</div>')
                    prompt_out = gr.Textbox(
                        show_label=False, lines=12, interactive=True,
                        elem_id="prompt-out",
                        placeholder="Prompt will appear here.\nYou may edit before regenerating.",
                    )
                    regen_btn = gr.Button("↺ 仅重新生图", variant="secondary")
                    gr.HTML('<div class="sec">Agent 改 图</div>')
                    image_edit_feedback = gr.Textbox(
                        show_label=False,
                        lines=3,
                        placeholder="例：把画面改成雨后傍晚，增加远山和一盏孤灯，人物更小一些。",
                    )
                    with gr.Row():
                        edit_image_model = gr.Dropdown(
                            choices=IMAGE_EDIT_MODEL_CHOICES,
                            value=f"edit:{IMAGE_EDIT_DEFAULT_MODEL}",
                            label="编辑模型（仅「图像编辑」按钮使用）",
                            show_label=True,
                            scale=3,
                        )
                    gr.Markdown(
                        "**图像编辑**：百炼编辑 API，保留原图构图，按指令微调（需配置 DASHSCOPE_API_KEY）。  \n"
                        "**改写重生图**：LLM 将意见融入 Prompt 后重新生图，不保留构图（适合大幅改动）。",
                        elem_id="edit-help",
                    )
                    with gr.Row():
                        edit_image_api_btn  = gr.Button("✦ 图像编辑", variant="secondary", scale=1)
                        rewrite_regen_btn   = gr.Button("✦ 改写重生图", variant="secondary", scale=1)
                with gr.Column(scale=5):
                    gr.HTML('<div class="sec">生 成 画 作</div>')
                    image_out = gr.Image(
                        show_label=False, interactive=False,
                        elem_id="image-out", value=None,
                    )
                    image_history_gallery = gr.Gallery(
                        label="图像历史",
                        show_label=True,
                        elem_id="image-history-gallery",
                        columns=3,
                        rows=2,
                        object_fit="contain",
                    )

            clip_score_out = gr.Textbox(
                show_label=False, interactive=False,
                elem_id="clip-score", lines=1,
                placeholder="CLIP 双锚点图文一致性分数将在生成后显示…",
            )
            gr.HTML('<div class="sec">Agent 思 考 轨 迹</div>')
            agent_trace_out = gr.Markdown(
                value="Agent 的规划、执行、自检和反思会显示在这里。",
            )

    # ── 事件绑定 ──────────────────────────────────────────────────────────────
    # 「开始创作」按钮点击时自动清空诗文输入（如果创作要求非空），避免用户同时保留两者
    def _clear_poem_on_submit(user_req_val):
        return "" if (user_req_val and user_req_val.strip()) else gr.update()

    # 点击开始创作时，若创作要求非空则先清空诗文输入（避免打字时闪烁）
    submit_btn.click(
        fn=lambda v: "" if (v and v.strip()) else gr.update(),
        inputs=[user_req], outputs=[poem_edit],
    )
    # 开始创作（限制并发，防止重复点击冲垮后端）
    submit_btn.click(
        fn=on_create,
        inputs=[
            user_req, poem_edit, lang_radio, style_drop,
            poem_model, intent_model, title_model, prompt_model,
            image_backend,
        ],
        outputs=[
            title_out, poem_edit, poem_display, prompt_out,
            image_out, clip_score_out, image_history_gallery, agent_trace_out, agent_state,
        ],
        concurrency_limit=1,
        show_progress="minimal",
    )

    # 诗文编辑同步
    poem_edit.change(fn=on_sync_display, inputs=[poem_edit], outputs=[poem_display])

    # 改诗
    refine_poem_btn.click(
        fn=on_refine_poem,
        inputs=[
            refine_feedback, agent_state,
            poem_model, intent_model, refine_poem_model,
            prompt_model, title_model, lang_radio, style_drop, image_backend,
        ],
        outputs=[
            title_out, poem_edit, poem_display,
            prompt_out, image_out, clip_score_out,
            image_history_gallery, agent_trace_out, agent_state,
        ],
    )

    # 仅重新生图
    regen_btn.click(
        fn=on_regen_image,
        inputs=[prompt_out, image_backend, agent_state],
        outputs=[image_out, clip_score_out, image_history_gallery, agent_trace_out, agent_state],
    )

    # 图像编辑（百炼 imgedit API，保留构图）
    edit_image_api_btn.click(
        fn=on_edit_image_api,
        inputs=[image_edit_feedback, agent_state, edit_image_model],
        outputs=[image_out, clip_score_out, image_history_gallery, agent_trace_out, agent_state],
    )

    # 改写重生图（LLM 改写 Prompt 后重新生图）
    rewrite_regen_btn.click(
        fn=on_rewrite_regen,
        inputs=[image_edit_feedback, prompt_out, image_backend, agent_state],
        outputs=[image_out, clip_score_out, prompt_out, image_history_gallery, agent_trace_out, agent_state],
    )

    # 生成报告
    report_btn.click(
        fn=on_report,
        inputs=[user_req, title_out, poem_edit, prompt_out, image_out, agent_state],
        outputs=[report_out],
    )

    # 全自主创作
    _auto_shared_outputs = [
        title_out, poem_edit, poem_display, prompt_out,
        image_out, clip_score_out, image_history_gallery, agent_trace_out, agent_state,
    ]
    auto_btn.click(
        fn=lambda v: "" if (v and v.strip()) else gr.update(),
        inputs=[user_req], outputs=[poem_edit],
    )
    # 全自主创作（限制并发=1，防止重复点击冲垮本地显存和 API 并发）
    auto_btn.click(
        fn=on_autonomous_create,
        inputs=[
            user_req, poem_edit, lang_radio, style_drop,
            poem_model, intent_model, title_model, prompt_model,
            image_backend,
            auto_target_score, auto_max_img_rounds,
            auto_allow_poem_refine, auto_max_poem_rounds,
            auto_image_mode, auto_edit_model, auto_llm_driven_loop,
        ],
        outputs=_auto_shared_outputs,
        concurrency_limit=1,
        show_progress="full",
    )


def main():
    # 防止本机代理（Clash/v2ray 等）劫持 Gradio 启动自检请求 (gradio_api/startup-events)
    # 出现 502 时多为代理转发到 127.0.0.1 失败，显式把 localhost 加入 NO_PROXY 即可
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        if "127.0.0.1" not in existing or "localhost" not in existing:
            parts = [p for p in (existing, "localhost", "127.0.0.1") if p]
            os.environ[key] = ",".join(dict.fromkeys(parts))

    print("\n" + "=" * 60)
    print("诗画墨语 · Agent 模式启动")
    print(f"  候选诗数量: {POEM_CANDIDATE_COUNT}")
    print(f"  图像风格: 支持 {len(STYLE_MAP)} 种")
    print(f"  DeepSeek API Key: {'已设置' if DEEPSEEK_API_KEY else '未设置'}")
    print(f"  通义千问 API Key: {'已设置' if DASHSCOPE_API_KEY else '未设置'}")
    print(f"  本地 LLM 基座:    {'可用' if LOCAL_LLM_AVAILABLE else '未启用（API 模式）'}")
    print(f"  本地 LoRA Adapter: {'可用' if LOCAL_LORA_AVAILABLE else '未启用'}")
    print(f"  本地 Z-Image:     {'可用' if LOCAL_IMAGE_AVAILABLE else '未启用（百炼 API 模式）'}")
    print("  Agent 特性:")
    print("    · 双锚点 CLIP（诗-图 × 0.6 + 提示词-图 × 0.4）")
    print("    · 改诗注入 / Agent 改图规划 / 模型追踪 / 报告含模型配置")
    print("    · 🤖 全自主创作：自动改图循环 + 自主改诗 + best-state 快照")
    print("=" * 60 + "\n")
    demo.launch()


if __name__ == "__main__":
    main()
