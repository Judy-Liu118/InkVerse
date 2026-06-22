"""
core.image.api -- 阿里百炼文生图 + 图像编辑 API 客户端
"""
import io
import base64
import re
import time
import requests
from PIL import Image
from typing import Optional
from core.logger import get_logger
from config import (
    API_TIMEOUT_SUBMIT, API_TIMEOUT_SYNC, API_TIMEOUT_POLL,
    API_TIMEOUT_DOWNLOAD, API_MAX_RETRIES,
    API_POLL_INTERVAL, API_POLL_MAX_WAIT,
)

_log = get_logger(__name__)

_SYNTHESIS_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
_MULTIMODAL_GENERATION_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
_TASK_QUERY_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"

_POLL_INTERVAL = API_POLL_INTERVAL
_POLL_MAX_WAIT = API_POLL_MAX_WAIT


_RETRIABLE_EXC = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _request_with_retry(method: str, url: str, *, op: str, **kwargs):
    """对连接/超时类错误做指数退避（1s, 2s, 4s …）。

    HTTP 4xx/5xx 不重试 —— 由调用方走 raise_for_status 让上层判断。
    400 状态码也不在此处特殊处理，调用方需要 fallback 的会自行处理 resp。
    """
    last_exc = None
    for attempt in range(API_MAX_RETRIES):
        try:
            return requests.request(method, url, **kwargs)
        except _RETRIABLE_EXC as e:
            last_exc = e
            if attempt == API_MAX_RETRIES - 1:
                break
            wait = 2 ** attempt
            _log.warning("[API %s] %s 第 %d/%d 次失败 (%s)，%ds 后重试",
                         op, method.upper(), attempt + 1, API_MAX_RETRIES,
                         type(e).__name__, wait)
            time.sleep(wait)
    _log.error("[API %s] %s 重试 %d 次后仍失败: %s",
               op, method.upper(), API_MAX_RETRIES, last_exc)
    raise last_exc


def _post(url, *, op, **kwargs):
    return _request_with_retry("POST", url, op=op, **kwargs)


def _get(url, *, op, **kwargs):
    return _request_with_retry("GET", url, op=op, **kwargs)


class BailianImageAPI:
    """阿里百炼文生图 API 封装。"""

    def __init__(self, api_key: str, model: str = "wanx2.1-t2i-turbo"):
        self.api_key = api_key
        self.model   = model
        if not api_key:
            raise ValueError("[图像API] DASHSCOPE_API_KEY 未配置")

    def generate(
        self, prompt: str, negative_prompt: str = "",
        width: int = 512, height: int = 512,
    ) -> Image.Image:
        if self._is_modern_multimodal_model(self.model):
            return self._generate_via_multimodal(prompt, width, height)
        # 旧 wanx2.1 等：v1 异步 text2image 接口
        size_str = self._resolve_size(width, height)
        task_id  = self._submit_task(prompt, negative_prompt, size_str)
        img_url  = self._poll_until_done(task_id)
        return self._download_image(img_url)

    def _generate_via_multimodal(self, prompt: str, width: int, height: int) -> Image.Image:
        """新版同步多模态生图接口；z-image / qwen-image / wan2.x 共用。"""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        size = self._resolve_modern_size(width, height)
        payload = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
            "parameters": {"size": size, "watermark": False},
        }
        _log.info("调用同步多模态生图接口 (%s, %s)", self.model, size)
        resp = _post(_MULTIMODAL_GENERATION_URL, op=self.model,
                     json=payload, headers=headers, timeout=API_TIMEOUT_SYNC)
        resp.raise_for_status()
        data = resp.json()
        img_url = self._extract_image_url(data)
        if not img_url:
            raise RuntimeError(f"{self.model} 未返回图片 URL: {data}")
        return self._download_image(img_url)

    def _submit_task(self, prompt: str, negative_prompt: str, size: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                   "X-DashScope-Async": "enable"}
        payload = {"model": self.model, "input": {"prompt": prompt}, "parameters": {"size": size, "n": 1}}
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        _log.debug("提交生图任务 (%s, %s)", self.model, size)
        resp = _post(_SYNTHESIS_URL, op="submit-t2i",
                     json=payload, headers=headers, timeout=API_TIMEOUT_SUBMIT)
        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"未获取到 task_id，返回: {data}")
        _log.debug("任务已提交: %s", task_id)
        return task_id

    def _poll_until_done(self, task_id: str) -> str:
        url = _TASK_QUERY_URL.format(task_id=task_id)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        waited = 0
        while waited < _POLL_MAX_WAIT:
            time.sleep(_POLL_INTERVAL)
            waited += _POLL_INTERVAL
            try:
                resp = _get(url, op="poll-t2i", headers=headers, timeout=API_TIMEOUT_POLL)
                data = resp.json()
            except _RETRIABLE_EXC as e:
                _log.warning("[轮询] task=%s 单次失败 (%s)，继续等待", task_id, type(e).__name__)
                continue
            status = data.get("output", {}).get("task_status", "UNKNOWN")
            if status == "SUCCEEDED":
                results = data["output"].get("results", [])
                if not results or "url" not in results[0]:
                    raise RuntimeError(f"任务成功但未返回图像 URL: {data}")
                img_url = results[0]["url"]
                _log.info("生图完成（耗时 ~%ss）", waited)
                return img_url
            elif status == "FAILED":
                code = data["output"].get("code", "")
                message = data["output"].get("message", "未知错误")
                raise RuntimeError(f"任务失败 [{code}]: {message}")
            elif status in ("PENDING", "RUNNING"):
                _log.info("仍在生成中… (%ss)", waited)
        raise TimeoutError(f"任务超时（>{_POLL_MAX_WAIT}s），task_id={task_id}")

    @staticmethod
    def _download_image(url: str) -> Image.Image:
        _log.debug("正在下载图像…")
        if url.startswith("data:image"):
            _, encoded = url.split(",", 1)
            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            _log.debug("图像解码完成，尺寸=%s", image.size)
            return image
        resp = _get(url, op="download", timeout=API_TIMEOUT_DOWNLOAD)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        _log.debug("图像下载完成，尺寸=%s", image.size)
        return image

    @staticmethod
    def _extract_image_url(data: dict) -> str:
        output = data.get("output", {}) if isinstance(data, dict) else {}
        for path in (
            ("url",), ("image_url",), ("results", 0, "url"),
            ("choices", 0, "message", "content", 0, "image"),
            ("choices", 0, "message", "content", 0, "url"),
        ):
            cur = output
            ok = True
            for key in path:
                try:
                    cur = cur[key]
                except (KeyError, IndexError, TypeError):
                    ok = False
                    break
            if ok and isinstance(cur, str) and cur.startswith(("http://", "https://", "data:image")):
                return cur
        m = re.search(r'https?://[^"\']+\.(?:png|jpg|jpeg|webp)(?:\?[^"\']*)?', str(data))
        return m.group(0) if m else ""

    @staticmethod
    def _is_modern_multimodal_model(model: str) -> bool:
        """识别走新同步多模态接口的模型：z-image / qwen-image / wan2.x。"""
        name = (model or "").lower()
        if name.startswith("z-image"):
            return True
        if name.startswith("qwen-image") and "edit" not in name:
            return True
        # wan2.2 / 2.5 / 2.6 / 2.7 等新代万相；不匹配老的 wanx2.x
        if re.match(r"^wan2\.\d", name):
            return True
        return False

    @staticmethod
    def _resolve_size(w: int, h: int) -> str:
        """旧 wanx2.1 异步接口用：64 对齐 + [512, 1024] 钳制。"""
        def snap(v: int) -> int:
            v = max(512, min(1024, v))
            return (v // 64) * 64
        return f"{snap(w)}*{snap(h)}"

    @staticmethod
    def _resolve_modern_size(w: int, h: int) -> str:
        """新同步接口用：归到 1024*1024 / 1024*768 / 768*1024 三档。"""
        if abs(w - h) < 96:
            return "1024*1024"
        return "1024*768" if w > h else "768*1024"


# ═══════════════════════════════════════════════════════════════════════════════
# 百炼图像编辑 API
# ═══════════════════════════════════════════════════════════════════════════════
_EDIT_URL        = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis"
_EDIT_TASK_URL   = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
_EDIT_POLL_INTERVAL = 3
_EDIT_POLL_MAX_WAIT = 120


class BailianImageEditAPI:
    """阿里百炼图像编辑 API 封装。"""

    def __init__(self, api_key: str, model: str = "wanx2.1-imageedit"):
        self.api_key = api_key
        self.model   = model
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY 未配置")

    def edit(self, image: Image.Image, instruction: str, strength: float = 0.75) -> Image.Image:
        img_b64 = self._image_to_base64(image)
        if self._is_qwen_image_edit_model(self.model):
            return self._edit_qwen_image(img_b64, instruction)
        task_id = self._submit_edit_task(img_b64, instruction, strength)
        img_url = self._poll_until_done(task_id)
        return self._download_image(img_url)

    def _edit_qwen_image(self, img_b64: str, instruction: str) -> Image.Image:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": [{"image": img_b64}, {"text": instruction}]}]},
            "parameters": {"watermark": False},
        }
        _log.info("调用 Qwen 图像编辑同步接口 (%s)", self.model)
        resp = _post(_MULTIMODAL_GENERATION_URL, op="qwen-edit",
                     json=payload, headers=headers, timeout=API_TIMEOUT_SYNC)
        resp.raise_for_status()
        data = resp.json()
        img_url = BailianImageAPI._extract_image_url(data)
        if not img_url:
            raise RuntimeError(f"Qwen 编辑未返回图片 URL: {data}")
        return self._download_image(img_url)

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _is_qwen_image_edit_model(model: str) -> bool:
        return (model or "").lower().startswith("qwen-image-edit")

    def _submit_edit_task(self, img_b64: str, instruction: str, strength: float) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                   "X-DashScope-Async": "enable"}
        payload = {
            "model": self.model,
            "input": {"function": "description_edit", "base_image_url": img_b64, "prompt": instruction},
            "parameters": {"strength": round(strength, 2), "n": 1},
        }
        _log.debug("提交编辑任务 (%s)，指令: %s", self.model, instruction[:60])
        resp = _post(_EDIT_URL, op="submit-edit",
                     json=payload, headers=headers, timeout=API_TIMEOUT_SUBMIT)
        if resp.status_code == 400:
            legacy_payload = {
                "model": self.model,
                "input": {"image": img_b64, "prompt": instruction},
                "parameters": {"strength": round(strength, 2), "n": 1},
            }
            _log.warning("[编辑] modern payload 被拒绝（400），尝试 legacy image 字段")
            resp = _post(_EDIT_URL, op="submit-edit-legacy",
                         json=legacy_payload, headers=headers, timeout=API_TIMEOUT_SUBMIT)
        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"未获取到 task_id，返回: {data}")
        _log.debug("任务已提交: %s", task_id)
        return task_id

    def _poll_until_done(self, task_id: str) -> str:
        url = _EDIT_TASK_URL.format(task_id=task_id)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        waited = 0
        while waited < _EDIT_POLL_MAX_WAIT:
            time.sleep(_EDIT_POLL_INTERVAL)
            waited += _EDIT_POLL_INTERVAL
            try:
                resp = _get(url, op="poll-edit", headers=headers, timeout=API_TIMEOUT_POLL)
                data = resp.json()
            except _RETRIABLE_EXC as e:
                _log.warning("[轮询·编辑] task=%s 单次失败 (%s)，继续等待", task_id, type(e).__name__)
                continue
            status = data.get("output", {}).get("task_status", "UNKNOWN")
            if status == "SUCCEEDED":
                results = data["output"].get("results", [])
                if not results or "url" not in results[0]:
                    raise RuntimeError(f"成功但无图片 URL: {data}")
                _log.info("编辑完成（耗时 ~%ss）", waited)
                return results[0]["url"]
            elif status == "FAILED":
                code = data["output"].get("code", "")
                msg = data["output"].get("message", "未知错误")
                raise RuntimeError(f"任务失败 [{code}]: {msg}")
            elif status in ("PENDING", "RUNNING"):
                _log.info("生成中… (%ss)", waited)
        raise TimeoutError(f"超时（>{_EDIT_POLL_MAX_WAIT}s）task_id={task_id}")

    @staticmethod
    def _download_image(url: str) -> Image.Image:
        resp = _get(url, op="download-edit", timeout=API_TIMEOUT_DOWNLOAD)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        _log.debug("下载完成，尺寸=%s", image.size)
        return image
