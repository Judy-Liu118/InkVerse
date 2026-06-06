"""
core.models.adapter -- 统一模型适配层

支持本地模型（Qwen/LoRA）、DeepSeek API、阿里百炼（通义千问）API 无缝切换。
"""
import os
import torch
from typing import List, Dict, Optional
from core.logger import get_logger

_log = get_logger(__name__)

# ── 后端标识常量 ────────────────────────────────────────────────────────────────
BACKEND_LOCAL    = "local"
BACKEND_DEEPSEEK = "deepseek"
BACKEND_QWEN     = "qwen"
BACKEND_LOCAL_LORA = "local_lora"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
QWEN_BASE_URL     = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DEFAULT_MODEL = {
    BACKEND_DEEPSEEK: "deepseek-chat",
    BACKEND_QWEN:     "qwen-plus",
}
ENV_KEY_MAP = {
    BACKEND_DEEPSEEK: "DEEPSEEK_API_KEY",
    BACKEND_QWEN:     "DASHSCOPE_API_KEY",
}


class ModelAdapter:
    """统一文本生成接口。"""

    def __init__(
        self,
        backend: str = BACKEND_LOCAL,
        api_key: Optional[str] = None,
        api_model: Optional[str] = None,
    ):
        self.backend = backend
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
        try:
            if self.backend in (BACKEND_LOCAL, BACKEND_LOCAL_LORA):
                return self._generate_local(messages, max_tokens, temperature)
            elif self.backend in (BACKEND_DEEPSEEK, BACKEND_QWEN):
                return self._generate_openai_compat(messages, max_tokens, temperature)
            else:
                raise ValueError(f"未知后端: '{self.backend}'")
        except Exception as e:
            _log.error("生成失败 (%s): %s", self.backend, e)
            if self.backend not in (BACKEND_LOCAL, BACKEND_LOCAL_LORA):
                _log.warning("API 调用失败，自动降级到本地 LoRA 模型重试")
                try:
                    saved_backend = self.backend
                    self.backend = BACKEND_LOCAL_LORA
                    result = self._generate_local(messages, max_tokens, temperature)
                    _log.info("降级生成成功")
                    return result
                except Exception as e2:
                    self.backend = saved_backend
                    _log.error("降级生成也失败: %s", e2)
                    raise e
                finally:
                    self.backend = saved_backend
            raise

    def _generate_local(self, messages, max_tokens, temperature) -> str:
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

        base_url = DEEPSEEK_BASE_URL if self.backend == BACKEND_DEEPSEEK else QWEN_BASE_URL
        client = OpenAI(api_key=self.api_key, base_url=base_url)
        kwargs: Dict = dict(model=self.api_model, messages=messages, max_tokens=max_tokens)
        if self.api_model != "deepseek-reasoner":
            kwargs["temperature"] = temperature

        response = client.chat.completions.create(**kwargs, timeout=120)
        if response and response.choices and response.choices[0].message:
            _log.debug("API 响应完成: %s", self.api_model)
        return response.choices[0].message.content.strip()


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
