"""
test_adapter -- ModelAdapter 后端选择 + 环境变量回退
不接通真实 API，避免外部依赖；仅校验初始化路径。
"""
import os
import pytest

from core.models.adapter import (
    ModelAdapter,
    BACKEND_LOCAL, BACKEND_LOCAL_LORA, BACKEND_DEEPSEEK, BACKEND_QWEN,
    DEFAULT_MODEL, ENV_KEY_MAP, get_adapter_from_config,
)


def test_default_model_resolved_when_omitted():
    """构造时不传 api_model，应从 DEFAULT_MODEL 查到默认值。"""
    a = ModelAdapter(backend=BACKEND_DEEPSEEK, api_key="dummy")
    assert a.api_model == DEFAULT_MODEL[BACKEND_DEEPSEEK]


def test_explicit_model_overrides_default():
    a = ModelAdapter(backend=BACKEND_QWEN, api_key="dummy", api_model="qwen-max")
    assert a.api_model == "qwen-max"


def test_env_key_picked_up_when_arg_omitted(monkeypatch):
    """未显式传 api_key 时应从环境变量读。"""
    monkeypatch.setenv(ENV_KEY_MAP[BACKEND_QWEN], "qwen-env-key")
    a = ModelAdapter(backend=BACKEND_QWEN)
    assert a.api_key == "qwen-env-key"


def test_explicit_key_beats_env(monkeypatch):
    monkeypatch.setenv(ENV_KEY_MAP[BACKEND_QWEN], "env-key")
    a = ModelAdapter(backend=BACKEND_QWEN, api_key="arg-key")
    assert a.api_key == "arg-key"


def test_local_backends_do_not_require_api_key():
    """本地后端不应因为没 key 报错；只是 log warning。"""
    a_local = ModelAdapter(backend=BACKEND_LOCAL)
    a_lora  = ModelAdapter(backend=BACKEND_LOCAL_LORA)
    assert a_local.backend == BACKEND_LOCAL
    assert a_lora.backend  == BACKEND_LOCAL_LORA


def test_unknown_backend_raises_at_construct_time():
    """未知 backend（拼写错误等）应在构造期就被 fail-fast，而不是延迟到 generate
    时再静默走 LoRA 降级 —— 静默降级会让评分/起名/提示词任务输出垃圾。"""
    with pytest.raises(ValueError, match="未知后端"):
        ModelAdapter(backend="not-a-real-backend")


def test_local_backends_have_no_default_model():
    """本地后端不走 DEFAULT_MODEL（模型由 BASE_MODEL_PATH 指定，不需要别名）。"""
    a_local = ModelAdapter(backend=BACKEND_LOCAL)
    a_lora  = ModelAdapter(backend=BACKEND_LOCAL_LORA)
    assert a_local.api_model == ""
    assert a_lora.api_model == ""


# ── LoRA 降级闸门 ────────────────────────────────────────────────────────────
def test_fallback_disabled_by_default():
    """默认不允许 LoRA 降级——评分/起名/提示词等任务必须诚实抛错。"""
    a = ModelAdapter(backend=BACKEND_QWEN, api_key="dummy")
    assert a.allow_lora_fallback is False
    assert a._can_fallback_to_lora() is False


def test_fallback_requires_both_flag_and_lora_availability(monkeypatch):
    """打开 flag 还不够，本地 LoRA 真的可用才允许降级。"""
    a = ModelAdapter(backend=BACKEND_QWEN, api_key="dummy", allow_lora_fallback=True)
    # 模拟 LoRA 不可用
    import config
    monkeypatch.setattr(config, "LOCAL_LORA_AVAILABLE", False, raising=False)
    assert a._can_fallback_to_lora() is False
    # 模拟 LoRA 可用
    monkeypatch.setattr(config, "LOCAL_LORA_AVAILABLE", True, raising=False)
    assert a._can_fallback_to_lora() is True


def test_strict_adapter_propagates_api_error(monkeypatch):
    """没开 fallback 的 adapter 遇 API 错误应原样向上抛，绝不静默走 LoRA。"""
    a = ModelAdapter(backend=BACKEND_QWEN, api_key="dummy", allow_lora_fallback=False)

    def _boom(*args, **kwargs):
        raise RuntimeError("API 模拟故障")

    monkeypatch.setattr(a, "_generate_openai_compat", _boom)
    # 即便 LoRA 看起来可用，也不能降级
    import config
    monkeypatch.setattr(config, "LOCAL_LORA_AVAILABLE", True, raising=False)
    with pytest.raises(RuntimeError, match="API 模拟故障"):
        a.generate([{"role": "user", "content": "ping"}], max_tokens=4)


def test_fallback_adapter_invokes_local_path_on_api_failure(monkeypatch):
    """开了 fallback 的 adapter，API 失败时应进入本地路径重试。"""
    a = ModelAdapter(backend=BACKEND_QWEN, api_key="dummy", allow_lora_fallback=True)
    import config
    monkeypatch.setattr(config, "LOCAL_LORA_AVAILABLE", True, raising=False)

    monkeypatch.setattr(a, "_generate_openai_compat",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("API 模拟故障")))

    invoked_with = {}

    def _fake_local(messages, max_tokens, temperature):
        invoked_with["backend"] = a.backend
        return "fallback-output"

    monkeypatch.setattr(a, "_generate_local", _fake_local)

    out = a.generate([{"role": "user", "content": "ping"}], max_tokens=4)
    assert out == "fallback-output"
    assert invoked_with["backend"] == BACKEND_LOCAL_LORA
    # 降级后应该把 backend 还原回 qwen，避免污染后续调用
    assert a.backend == BACKEND_QWEN


def test_get_adapter_from_config_respects_env(monkeypatch):
    """POETRY_BACKEND 环境变量应覆盖 config.LLM_BACKEND。"""
    monkeypatch.setenv("POETRY_BACKEND", BACKEND_QWEN)
    a = get_adapter_from_config()
    assert a.backend == BACKEND_QWEN
