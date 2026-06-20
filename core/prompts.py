"""
core.prompts -- 集中化 prompt 管理 loader

所有 system / user prompt 均以 YAML 形式存放于项目根目录的 `prompts/` 下，
通过 `render_messages(name, **vars)` 渲染成 OpenAI Chat Messages 格式。

设计要点：
  · YAML 文件结构稳定：name / version / description / locale / system / user
  · 变量插值用标准 `str.format()` 风格，不引入 Jinja2 依赖
  · `lru_cache` 进程内单次加载，零 IO 开销
  · 缺变量直接抛 KeyError，避免静默生成残缺 prompt
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from core.logger import get_logger

_log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = _ROOT / "prompts"


@functools.lru_cache(maxsize=128)
def _load_raw(name: str) -> Dict[str, Any]:
    """加载 prompts/{name}.yaml；结果常驻进程缓存。

    `name` 支持点号路径，如 `agent.refine_poem` → `prompts/agent/refine_poem.yaml`。
    """
    rel = Path(*name.split(".")).with_suffix(".yaml")
    path = PROMPTS_DIR / rel
    if not path.is_file():
        raise FileNotFoundError(
            f"Prompt 文件不存在: {path}（name={name!r}）"
        )
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Prompt {name!r} 顶层结构必须是 dict，实际为 {type(data).__name__}")
    return data


def get_metadata(name: str) -> Dict[str, Any]:
    """读取 prompt 的元数据（version / description / locale）—— 不渲染。"""
    data = _load_raw(name)
    return {k: data.get(k) for k in ("name", "version", "description", "locale")}


def render_messages(name: str, **vars: Any) -> List[Dict[str, str]]:
    """渲染成 OpenAI Chat Messages 格式。

    YAML 中的 `system` 字段可选，`user` 字段必选；
    两者都支持 `{var}` 风格插值。
    """
    data = _load_raw(name)
    user_tpl = data.get("user")
    if not user_tpl:
        raise ValueError(f"Prompt {name!r} 缺少必需的 `user` 字段")

    try:
        user_content = user_tpl.format(**vars) if vars else user_tpl
    except KeyError as e:
        raise KeyError(
            f"Prompt {name!r} 渲染时缺少变量: {e}; 已提供: {list(vars)}"
        ) from e

    messages: List[Dict[str, str]] = []
    system_tpl = data.get("system")
    if system_tpl:
        try:
            system_content = system_tpl.format(**vars) if vars else system_tpl
        except KeyError as e:
            raise KeyError(
                f"Prompt {name!r} 的 system 段渲染时缺少变量: {e}"
            ) from e
        messages.append({"role": "system", "content": system_content.strip()})
    messages.append({"role": "user", "content": user_content.strip()})
    return messages


def list_prompts() -> List[str]:
    """枚举 prompts/ 下所有可用的 prompt 名（用点号路径）。"""
    if not PROMPTS_DIR.is_dir():
        return []
    names = []
    for path in sorted(PROMPTS_DIR.rglob("*.yaml")):
        rel = path.relative_to(PROMPTS_DIR).with_suffix("")
        names.append(".".join(rel.parts))
    return names


def clear_cache() -> None:
    """开发期热重载用：清空 prompt 缓存，下次访问会重新读盘。"""
    _load_raw.cache_clear()


__all__ = [
    "PROMPTS_DIR",
    "get_metadata",
    "render_messages",
    "list_prompts",
    "clear_cache",
]
