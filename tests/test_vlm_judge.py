"""
test_vlm_judge -- VLM ground-truth judge 的解析、归一化、错误兜底

不触发真实 multimodal API：注入 stub client 验证全链路。
"""
from types import SimpleNamespace

import pytest
from PIL import Image

from eval.vlm_judge import (
    VLMJudge, VLMVerdict, _parse_verdict, _image_to_data_url, _resolve_backend,
)


# ── _parse_verdict ────────────────────────────────────────────────────────
def test_parse_bare_json():
    v = _parse_verdict('{"score": 7.5, "reasoning": "意象到位"}', model="m")
    assert v.error is None
    assert v.raw_score == 7.5
    assert v.score == 0.75
    assert v.reasoning == "意象到位"


def test_parse_markdown_fenced():
    raw = '```json\n{"score": 8, "reasoning": "ok"}\n```'
    v = _parse_verdict(raw, model="m")
    assert v.error is None
    assert v.score == 0.8


def test_parse_score_clipped_to_0_10():
    v_hi = _parse_verdict('{"score": 12, "reasoning": "x"}', model="m")
    assert v_hi.raw_score == 10.0 and v_hi.score == 1.0
    v_lo = _parse_verdict('{"score": -3, "reasoning": "x"}', model="m")
    assert v_lo.raw_score == 0.0 and v_lo.score == 0.0


def test_parse_score_via_regex_when_json_broken():
    """JSON 解析失败但能从文字里抓到 score → 仍可用"""
    raw = '我觉得 score: 6.5 / 10，原因是...'
    v = _parse_verdict(raw, model="m")
    assert v.error is None
    assert v.raw_score == 6.5


def test_parse_no_score_field_returns_error():
    v = _parse_verdict('{"reasoning": "just text"}', model="m")
    assert v.score is None
    assert v.error is not None


def test_parse_empty_returns_error():
    v = _parse_verdict("", model="m")
    assert v.score is None
    assert v.error == "empty response"


def test_parse_garbage_returns_error():
    v = _parse_verdict("this is not json at all", model="m")
    assert v.score is None
    assert v.error is not None


# ── _image_to_data_url ────────────────────────────────────────────────────
def test_image_to_data_url_returns_jpeg_b64():
    img = Image.new("RGB", (16, 16), color=(128, 0, 64))
    url = _image_to_data_url(img)
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > 30


def test_image_to_data_url_converts_rgba():
    """RGBA → RGB 自动转换，不抛 cannot write mode RGBA as JPEG。"""
    img = Image.new("RGBA", (8, 8), (1, 2, 3, 255))
    url = _image_to_data_url(img)
    assert url.startswith("data:image/jpeg;base64,")


# ── _resolve_backend ──────────────────────────────────────────────────────
def test_resolve_backend_qwen_vl():
    assert _resolve_backend("qwen-vl-max") == "qwen"
    assert _resolve_backend("qwen-vl-plus") == "qwen"
    assert _resolve_backend("qwen2-vl-72b-instruct") == "qwen"


def test_resolve_backend_glm():
    assert _resolve_backend("glm-4v-plus") == "zhipu"
    assert _resolve_backend("glm-4v") == "zhipu"


def test_resolve_backend_unknown_raises():
    with pytest.raises(ValueError):
        _resolve_backend("gpt-4-vision-preview")


# ── VLMJudge 端到端（注入 stub client）────────────────────────────────────
class _StubClient:
    """伪装 openai.OpenAI 客户端：可指定要返回的 raw 文本，记录调用入参。"""
    def __init__(self, raw_text=None, raise_exc=None):
        self.raw_text = raw_text
        self.raise_exc = raise_exc
        self.last_messages = None
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, max_tokens, temperature):
        self.calls += 1
        self.last_messages = messages
        if self.raise_exc:
            raise self.raise_exc
        msg = SimpleNamespace(content=self.raw_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _make_judge(raw_text=None, raise_exc=None):
    return VLMJudge(
        model="qwen-vl-max", api_key="stub",
        client=_StubClient(raw_text=raw_text, raise_exc=raise_exc),
    )


def test_judge_score_success_path():
    judge = _make_judge(raw_text='{"score": 7, "reasoning": "ok"}')
    img = Image.new("RGB", (8, 8))
    v = judge.score(image=img, poem="春风又绿江南岸", visual_keywords_en="spring breeze")
    assert v.error is None
    assert v.score == 0.7

    # multimodal content 形状校验：必须既含 image_url 又含 text
    msgs = judge._client.last_messages
    user_msg = msgs[-1]
    assert user_msg["role"] == "user"
    types = [p["type"] for p in user_msg["content"]]
    assert "image_url" in types
    assert "text" in types
    # text 必须填入 poem
    text_part = next(p for p in user_msg["content"] if p["type"] == "text")
    assert "春风又绿江南岸" in text_part["text"]


def test_judge_api_error_wrapped_in_verdict():
    judge = _make_judge(raise_exc=RuntimeError("rate limit"))
    img = Image.new("RGB", (8, 8))
    v = judge.score(image=img, poem="x", visual_keywords_en="y")
    assert v.score is None
    assert v.error is not None
    assert "rate limit" in v.error


def test_judge_none_image_returns_error_not_raise():
    judge = _make_judge(raw_text='{"score": 5}')
    v = judge.score(image=None, poem="x", visual_keywords_en="y")
    assert v.score is None
    assert v.error == "image is None"
    # 关键：未真正调 API
    assert judge._client.calls == 0


def test_judge_unparseable_response_returns_error():
    judge = _make_judge(raw_text="garbage with no score")
    img = Image.new("RGB", (8, 8))
    v = judge.score(image=img, poem="x", visual_keywords_en="y")
    assert v.score is None
    assert v.error is not None


def test_verdict_as_dict_roundtrip():
    v = VLMVerdict(score=0.6, raw_score=6.0, reasoning="r", model="m", error=None)
    d = v.as_dict()
    assert d == {"score": 0.6, "raw_score": 6.0, "reasoning": "r",
                 "model": "m", "error": None}
