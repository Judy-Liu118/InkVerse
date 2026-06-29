"""
app_utils.py -- UI 工具函数、路径常量、模型/图像后端选项。

供 app_handlers.py 和 app.py 共同引用，不依赖 Gradio 组件对象。
"""
import os
import time

from PIL import Image

from core.agent.state import AgentState
from core.agent.autonomous import AutonomousConfig
from core.models.adapter import ModelAdapter
from config import (
    DEEPSEEK_API_KEY, DASHSCOPE_API_KEY,
    LOCAL_LLM_AVAILABLE, LOCAL_LORA_AVAILABLE, LOCAL_IMAGE_AVAILABLE,
    STYLE_MAP, get_style_suffix,
)

# ── 路径常量 ──────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
LAST_IMAGE_PATH = os.path.join(_ROOT, "outputs", "last_image.png")
IMAGE_HISTORY_DIR = os.path.join(_ROOT, "outputs", "image_history")
os.makedirs(os.path.dirname(LAST_IMAGE_PATH), exist_ok=True)
os.makedirs(IMAGE_HISTORY_DIR, exist_ok=True)

# ── 模型列表 ──────────────────────────────────────────────────────────────────
_API_MODEL_CHOICES = [
    ("DeepSeek-Chat",                              "deepseek-chat"),
    ("DeepSeek-V4-Flash",                          "deepseek-v4-flash"),
    ("DeepSeek-V4-Pro",                            "deepseek-v4-pro"),
    ("DeepSeek-R1",                                "deepseek-reasoner"),
    ("通义千问-Turbo",                              "qwen-turbo"),
    ("通义千问-Plus",                               "qwen-plus"),
    ("通义千问-Max",                                "qwen-max"),
    ("通义千问-Qwen3.7-Plus（慢但质量高）",          "qwen3.7-plus"),
    ("通义千问-Qwen3.6-Flash（高速）",              "qwen3.6-flash"),
    ("Qwen2.5-1.5B API（非量化，对比 LoRA 用）",   "qwen2.5-1.5b-instruct"),
]

_LOCAL_BASE_CHOICE = [("本地基础模型 (Qwen2.5-1.5B)", "local_base")] if LOCAL_LLM_AVAILABLE else []
_LOCAL_LORA_CHOICE = [("本地微调模型 (Qwen2.5-1.5B+LoRA)", "local_lora")] if LOCAL_LORA_AVAILABLE else []

MODEL_CHOICES = _LOCAL_BASE_CHOICE + _LOCAL_LORA_CHOICE + _API_MODEL_CHOICES
POEM_MODEL_CHOICES = (_LOCAL_LORA_CHOICE + _API_MODEL_CHOICES) if LOCAL_LORA_AVAILABLE else _API_MODEL_CHOICES
REFINE_POEM_MODEL_CHOICES = _API_MODEL_CHOICES

_DEFAULT_POEM_MODEL = "local_lora" if LOCAL_LORA_AVAILABLE else "qwen-plus"

# ── 图像后端选项 ───────────────────────────────────────────────────────────────
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

IMAGE_BACKEND_CHOICES = _LOCAL_IMAGE_CHOICE + [
    ("百炼 · Z-Image Turbo API（非量化，推荐）",                "bailian:z-image-turbo"),
    ("百炼 · wan2.7-image-pro（极简禅意，留白极致）",           "bailian:wan2.7-image-pro"),
    ("百炼 · Qwen-Image Plus",                                 "bailian:qwen-image-plus"),
    ("百炼 · Qwen-Image Max（清润水墨）",                       "bailian:qwen-image-max"),
    ("百炼 · Qwen-Image 2.0",                                  "bailian:qwen-image-2.0"),
    ("百炼 · Qwen-Image 2.0 Pro（推荐，大写意水墨，跨主题最稳）",  "bailian:qwen-image-2.0-pro"),
    ("百炼 · Qwen-Image 2.0 2026-03-03",                       "bailian:qwen-image-2.0-2026-03-03"),
]

_DEFAULT_IMAGE_BACKEND = "local" if LOCAL_IMAGE_AVAILABLE else "bailian:qwen-image-2.0-pro"

# ── 图像持久化工具 ─────────────────────────────────────────────────────────────
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
    """根据 trace 中实际的动作字符串，为每轮生成的图像生成唯一 caption。"""
    img_rounds = sum(1 for s in state.trace if "改图循环：第" in s.action and "完成" in s.action)
    poem_regen = sum(1 for s in state.trace if "改诗第" in s.action and "重走全流程" in s.action)
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


# ── 诗文 HTML 渲染 ────────────────────────────────────────────────────────────
_PUNCT_STYLE = (
    "font-family:'Noto Serif SC','SimSun','宋体','Microsoft YaHei',serif;"
    "font-size:inherit;"
)


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
    """构造 ModelAdapter。

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
    """解析改图模型选择值，返回 (edit_model_or_None, edit_strength)。"""
    if val and val.startswith("edit:"):
        return val[5:], 0.75
    return None, 0.75
