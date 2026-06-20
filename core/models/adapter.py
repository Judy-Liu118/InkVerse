"""
core.models.adapter -- 统一模型适配层

支持本地模型（Qwen/LoRA）、DeepSeek API、阿里百炼（通义千问）API 无缝切换。

重型依赖（torch）延迟到 _generate_local 内 import，纯 API 用户无需安装。
"""
import os
from typing import List, Dict, Optional
from core.logger import get_logger

_log = get_logger(__name__)

# ── 后端标识常量 ────────────────────────────────────────────────────────────────
BACKEND_LOCAL    = "local"
BACKEND_DEEPSEEK = "deepseek"
BACKEND_QWEN     = "qwen"
BACKEND_ZHIPU    = "zhipu"
BACKEND_MOONSHOT = "moonshot"
BACKEND_LOCAL_LORA = "local_lora"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
QWEN_BASE_URL     = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ZHIPU_BASE_URL    = "https://open.bigmodel.cn/api/paas/v4/"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"

DEFAULT_MODEL = {
    BACKEND_DEEPSEEK: "deepseek-chat",
    BACKEND_QWEN:     "qwen-plus",
    BACKEND_ZHIPU:    "glm-4-plus",
    BACKEND_MOONSHOT: "moonshot-v1-32k",
}
ENV_KEY_MAP = {
    BACKEND_DEEPSEEK: "DEEPSEEK_API_KEY",
    BACKEND_QWEN:     "DASHSCOPE_API_KEY",
    BACKEND_ZHIPU:    "ZHIPU_API_KEY",
    BACKEND_MOONSHOT: "MOONSHOT_API_KEY",
}


_SUPPORTED_BACKENDS = (
    BACKEND_LOCAL, BACKEND_LOCAL_LORA,
    BACKEND_DEEPSEEK, BACKEND_QWEN, BACKEND_ZHIPU, BACKEND_MOONSHOT,
)


class ModelAdapter:
    """统一文本生成接口。

    `allow_lora_fallback`：API 调用失败时是否降级到本地 LoRA。
        默认 False —— 评分/起名/提示词等需要结构化输出或英文的任务，
        LoRA 给不了正确结果，静默降级反而会让下游静默崩坏；
        仅诗歌生成 adapter 显式设 True，让 API 瞬时故障时还能兜底产出。
    """

    def __init__(
        self,
        backend: str = BACKEND_LOCAL,
        api_key: Optional[str] = None,
        api_model: Optional[str] = None,
        allow_lora_fallback: bool = False,
    ):
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"未知后端: {backend!r}，请使用 {_SUPPORTED_BACKENDS} 之一"
            )
        self.backend = backend
        self.allow_lora_fallback = allow_lora_fallback
        env_var      = ENV_KEY_MAP.get(backend, "")
        self.api_key = api_key or os.getenv(env_var, "")
        self.api_model = api_model or DEFAULT_MODEL.get(backend, "")

        if backend not in (BACKEND_LOCAL, BACKEND_LOCAL_LORA):
            _log.info("后端=%s | 模型=%s", backend, self.api_model)
            if not self.api_key:
                _log.warning("未检测到 API Key，请在 config.py 设置或设置系统环境变量 %s", env_var)
        elif backend == BACKEND_LOCAL_LORA:
            _log.info("后端=local_lora（本地微调模型）")
        else:
            _log.info("后端=local（本地基础模型）")

    def generate(
        self,
        messages: List[Dict],
        max_tokens: int = 400,
        temperature: float = 0.7,
    ) -> str:
        is_local = self.backend in (BACKEND_LOCAL, BACKEND_LOCAL_LORA)
        try:
            if is_local:
                return self._generate_local(messages, max_tokens, temperature)
            return self._generate_openai_compat(messages, max_tokens, temperature)
        except Exception as e:
            _log.error("生成失败 (%s): %s", self.backend, e)
            if not is_local and self._can_fallback_to_lora():
                _log.warning(
                    "API 调用失败，自动降级到本地 LoRA 模型重试"
                    "（仅适用于诗歌生成任务）"
                )
                try:
                    saved_backend = self.backend
                    self.backend = BACKEND_LOCAL_LORA
                    result = self._generate_local(messages, max_tokens, temperature)
                    _log.info("降级生成成功")
                    return result
                except Exception as e2:
                    _log.error("降级生成也失败: %s", e2)
                    raise e
                finally:
                    self.backend = saved_backend
            raise

    def _can_fallback_to_lora(self) -> bool:
        """是否允许降级 —— 同时需要本任务允许、且本地 LoRA 真的可用。"""
        if not self.allow_lora_fallback:
            return False
        try:
            from config import LOCAL_LORA_AVAILABLE
        except ImportError:
            return False
        return bool(LOCAL_LORA_AVAILABLE)

    def _generate_local(self, messages, max_tokens, temperature) -> str:
        import torch
        from core.models.manager import ModelManager
        mm = ModelManager()
        if self.backend == BACKEND_LOCAL_LORA:
            model = mm.fine_model
            tokenizer = mm.fine_tokenizer
            _log.debug("使用本地微调模型 (LoRA)")
        else:
            model = mm.base_model
            tokenizer = mm.base_tokenizer
            _log.debug("使用本地基础模型")

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        if hasattr(model, "generation_config") and \
                getattr(model.generation_config, "max_length", None) is not None:
            model.generation_config.max_length = None
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0.01,
            )
        return tokenizer.decode(
            out[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        ).strip()

    def _generate_openai_compat(self, messages, max_tokens, temperature) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请先安装 openai SDK：pip install openai")

        base_url_map = {
            BACKEND_DEEPSEEK: DEEPSEEK_BASE_URL,
            BACKEND_QWEN:     QWEN_BASE_URL,
            BACKEND_ZHIPU:    ZHIPU_BASE_URL,
            BACKEND_MOONSHOT: MOONSHOT_BASE_URL,
        }
        base_url = base_url_map.get(self.backend, QWEN_BASE_URL)
        client = OpenAI(api_key=self.api_key, base_url=base_url)

        # 思考型模型（reasoner / v4-pro 等）需要更大 token 预算才能跑完 reasoning
        # 阶段并吐出最终 content；用太小的 max_tokens 会导致 content 为空。
        is_thinking_model = (
            self.backend == BACKEND_DEEPSEEK and
            ("reasoner" in (self.api_model or "") or
             "v4" in (self.api_model or "").lower() or
             "v5" in (self.api_model or "").lower())
        )
        effective_max_tokens = max(max_tokens, 1024) if is_thinking_model else max_tokens

        kwargs: Dict = dict(model=self.api_model, messages=messages,
                             max_tokens=effective_max_tokens)
        if not is_thinking_model:  # reasoner 系列不接受 temperature
            kwargs["temperature"] = temperature
            # 显式 top_p=0.9，与本地路径（generator._call_model）对齐，避免 DashScope
            # 默认 top_p=0.8 在 logits 集中时把采样池截到 1 个 token → mode collapse。
            kwargs["top_p"] = 0.9

        response = client.chat.completions.create(**kwargs, timeout=180)
        if not (response and response.choices and response.choices[0].message):
            _log.warning("API 响应结构异常: %s", self.api_model)
            return ""

        msg = response.choices[0].message
        content = (getattr(msg, "content", None) or "").strip()
        # 思考型模型的最终答案有时只在 reasoning_content；做兜底
        if not content:
            reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
            if reasoning:
                _log.info("API 内容为空，回落 reasoning_content（%d 字）",
                          len(reasoning))
                content = reasoning
            else:
                _log.warning("API 返回空内容: model=%s, finish_reason=%s",
                             self.api_model,
                             getattr(response.choices[0], "finish_reason", "?"))
        _log.debug("API 响应完成: %s", self.api_model)
        return content


def get_adapter_from_config() -> "ModelAdapter":
    """从 config.py 读取配置自动构建 ModelAdapter。"""
    try:
        from config import LLM_BACKEND, LLM_API_KEY, LLM_API_MODEL
    except ImportError:
        LLM_BACKEND   = BACKEND_LOCAL
        LLM_API_KEY   = ""
        LLM_API_MODEL = ""
    backend   = os.getenv("POETRY_BACKEND", LLM_BACKEND)
    api_key   = LLM_API_KEY or ""
    api_model = LLM_API_MODEL or None
    return ModelAdapter(backend=backend, api_key=api_key, api_model=api_model)
