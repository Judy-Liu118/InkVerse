"""
eval.vlm_judge -- 用多模态大模型（Qwen-VL / GLM-4V）做图文契合度评审

定位（与生产层 CLIP 严格区分）：
  · CLIP（core/evaluation/clip.py）：生产路径用，本地、快、便宜
  · VLM judge（本模块）：**离线评测层**用，做 CLIP 的外部 ground truth
    锚定 — 算 Spearman/Pearson 看 CLIP 各锚点策略与 VLM 的相关性。

不依赖生产 ModelAdapter；直调 OpenAI 兼容接口（DashScope / ZhipuAI 等）。
返回归一化到 [0, 1] 的 score，方便和 CLIP raw 同量纲对照。
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from PIL import Image

from core.logger import get_logger

_log = get_logger(__name__)


# ── VLM 后端 & 默认模型 ───────────────────────────────────────────────────
QWEN_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ZHIPU_BASE          = "https://open.bigmodel.cn/api/paas/v4/"

_QWEN_VL_MODELS = {"qwen-vl-max", "qwen-vl-plus", "qwen-vl-max-latest", "qwen2-vl-72b-instruct"}
_ZHIPU_VL_MODELS = {"glm-4v", "glm-4v-plus", "glm-4v-flash"}

DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 0.2  # judge 应保守，少创造


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass
class VLMVerdict:
    """单次 VLM 评分结果。score ∈ [0, 1]，已从原始 0–10 归一化。"""
    score: float          # [0, 1]，解析失败时为 None
    raw_score: float      # [0, 10]，原始分
    reasoning: str
    model: str
    error: Optional[str] = None  # 非 None 表示解析或调用失败

    def as_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "raw_score": self.raw_score,
            "reasoning": self.reasoning,
            "model": self.model,
            "error": self.error,
        }


# ── prompt 加载（不依赖 core.prompts._load_raw 私有 API）──────────────────
def _load_judge_prompt() -> Dict[str, str]:
    """加载 prompts/eval/vlm_image_poem_judge.yaml；返回 {system, user}。"""
    path = _PROMPTS_DIR / "eval" / "vlm_image_poem_judge.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"VLM judge prompt 缺失: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or not data.get("user"):
        raise ValueError(f"judge prompt 结构错误: {path}")
    return {"system": (data.get("system") or "").strip(),
            "user": data["user"]}


# ── 图像 → base64 data URL ────────────────────────────────────────────────
def _image_to_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    """PIL Image → data URL（OpenAI multimodal 兼容）。

    水墨画通常无透明，JPEG 体积更小，VLM API 限流更友好。
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


# ── 后端路由 ─────────────────────────────────────────────────────────────
def _resolve_backend(model: str) -> str:
    m = model.lower()
    if m in _QWEN_VL_MODELS or m.startswith("qwen-vl") or m.startswith("qwen2-vl"):
        return "qwen"
    if m in _ZHIPU_VL_MODELS or m.startswith("glm-4v"):
        return "zhipu"
    raise ValueError(
        f"未识别的 VLM 模型 {model!r}；支持: qwen-vl-*、glm-4v-*"
    )


def _resolve_api_key(backend: str, api_key: Optional[str]) -> str:
    if api_key:
        return api_key
    env_var = {"qwen": "DASHSCOPE_API_KEY", "zhipu": "ZHIPU_API_KEY"}[backend]
    key = os.getenv(env_var, "")
    if not key:
        raise RuntimeError(f"{backend} VLM 需要环境变量 {env_var}")
    return key


def _resolve_base_url(backend: str) -> str:
    return {"qwen": QWEN_DASHSCOPE_BASE, "zhipu": ZHIPU_BASE}[backend]


# ── JSON 解析（宽容 markdown 围栏 / 0–10 → 0–1）─────────────────────────
_SCORE_RE = re.compile(r'"?score"?\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)')


def _parse_verdict(raw: str, model: str) -> VLMVerdict:
    """从 raw text 提取 {score, reasoning}。解析失败时 score=None + error。"""
    if not raw:
        return VLMVerdict(score=None, raw_score=None, reasoning="",
                          model=model, error="empty response")
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(l for l in lines if not l.strip().startswith("```"))
    i, j = s.find("{"), s.rfind("}")
    obj = None
    if i != -1 and j != -1 and j > i:
        try:
            cand = json.loads(s[i:j+1])
            if isinstance(cand, dict):
                obj = cand
        except json.JSONDecodeError:
            pass

    if obj is None:
        # JSON 解析失败也尝试用正则抠 score，让 oracle 不至于全废
        m = _SCORE_RE.search(s)
        if m is None:
            return VLMVerdict(score=None, raw_score=None,
                              reasoning=s[:80], model=model,
                              error="no JSON & no score field")
        try:
            raw_score = float(m.group(1))
        except ValueError:
            return VLMVerdict(score=None, raw_score=None,
                              reasoning=s[:80], model=model,
                              error="score not float")
        reasoning = s[:80]
    else:
        if "score" not in obj:
            return VLMVerdict(score=None, raw_score=None,
                              reasoning=str(obj)[:80], model=model,
                              error="JSON missing 'score'")
        try:
            raw_score = float(obj["score"])
        except (TypeError, ValueError):
            return VLMVerdict(score=None, raw_score=None,
                              reasoning=str(obj.get("reasoning", ""))[:80],
                              model=model, error="score not numeric")
        reasoning = str(obj.get("reasoning", ""))[:120]

    # 越界裁剪：少数 VLM 会返回 11、-1 之类
    if raw_score < 0:
        raw_score = 0.0
    if raw_score > 10:
        raw_score = 10.0
    return VLMVerdict(score=raw_score / 10.0, raw_score=raw_score,
                      reasoning=reasoning, model=model, error=None)


# ── 核心 judge ────────────────────────────────────────────────────────────
class VLMJudge:
    """图文契合度 oracle judge。

    用法：
        judge = VLMJudge("qwen-vl-max")
        verdict = judge.score(image=pil_img, poem=poem, visual_keywords_en=kw)
        verdict.score  # ∈ [0, 1]
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        client=None,            # 注入 stub client（便于测试）
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._prompt = _load_judge_prompt()

        if client is not None:
            self._client = client
            self.backend = "injected"
            return

        self.backend = _resolve_backend(model)
        self.api_key = _resolve_api_key(self.backend, api_key)
        self.base_url = base_url or _resolve_base_url(self.backend)

        try:
            from openai import OpenAI  # 与 ModelAdapter 同源依赖
        except ImportError as e:
            raise RuntimeError(
                "VLMJudge 需要 openai 包 (pip install openai)"
            ) from e
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _build_messages(
        self, image: Image.Image, poem: str, visual_keywords_en: str,
    ) -> List[Dict[str, Any]]:
        user_text = self._prompt["user"].format(
            poem=poem.strip() or "（空）",
            visual_keywords_en=(visual_keywords_en or "").strip() or "（无）",
        )
        messages: List[Dict[str, Any]] = []
        if self._prompt["system"]:
            messages.append({"role": "system", "content": self._prompt["system"]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": _image_to_data_url(image)}},
                {"type": "text", "text": user_text},
            ],
        })
        return messages

    def score(
        self, *, image: Image.Image, poem: str, visual_keywords_en: str = "",
    ) -> VLMVerdict:
        """对单张图打分。所有失败都 wrap 成 VLMVerdict(error=...)，不抛异常。"""
        if image is None:
            return VLMVerdict(score=None, raw_score=None, reasoning="",
                              model=self.model, error="image is None")
        try:
            messages = self._build_messages(image, poem, visual_keywords_en)
        except Exception as e:
            return VLMVerdict(score=None, raw_score=None, reasoning="",
                              model=self.model,
                              error=f"build_messages: {e}")
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=self.max_tokens, temperature=self.temperature,
            )
        except Exception as e:
            _log.warning("[vlm_judge] %s 调用失败: %s", self.model, e)
            return VLMVerdict(score=None, raw_score=None, reasoning="",
                              model=self.model, error=f"api: {e}")
        try:
            raw = resp.choices[0].message.content or ""
        except (AttributeError, IndexError) as e:
            return VLMVerdict(score=None, raw_score=None, reasoning="",
                              model=self.model, error=f"resp parse: {e}")
        return _parse_verdict(raw, model=self.model)


__all__ = [
    "VLMJudge",
    "VLMVerdict",
    "_parse_verdict",
    "_image_to_data_url",
    "_resolve_backend",
]
