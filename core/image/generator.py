"""
core.image.generator -- 图像生成器（本地 Z-Image + 阿里百炼 API 双后端）
"""
from PIL import Image
from core.models.manager import ModelManager
from config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_STEPS, IMAGE_GUIDANCE, NEGATIVE_PROMPT,
    DASHSCOPE_API_KEY, IMAGE_API_MODEL, LOCAL_IMAGE_AVAILABLE,
)
from core.logger import get_logger

_log = get_logger(__name__)


class ImageGenerator:
    """统一图像生成接口，内部按 backend 路由。"""

    def __init__(self):
        self.mm = ModelManager()

    def generate(
        self, prompt: str, backend: str = "local",
        negative_prompt: str = None, guidance_scale: float = None,
        api_key: str = None, api_model: str = None,
    ) -> Image.Image:
        if backend == "bailian":
            try:
                return self._generate_bailian(prompt, negative_prompt, api_key, api_model)
            except Exception as e:
                if LOCAL_IMAGE_AVAILABLE:
                    _log.warning("百炼 API 生图失败: %s，降级到本地 Z-Image", e)
                    return self._generate_local(prompt, negative_prompt, guidance_scale)
                _log.error("百炼 API 生图失败且本地 Z-Image 不可用（未配置 ZIMAGE_PATH）")
                raise
        else:
            if not LOCAL_IMAGE_AVAILABLE:
                raise FileNotFoundError(
                    "本地 Z-Image 不可用：未配置 ZIMAGE_PATH 或路径不存在。"
                    "请在 UI「图像生成后端」中改选百炼 API 后端。"
                )
            return self._generate_local(prompt, negative_prompt, guidance_scale)

    def _generate_local(
        self, prompt: str, negative_prompt: str = None, guidance_scale: float = None,
    ) -> Image.Image:
        import torch
        pipe = self.mm.z_pipe
        neg = negative_prompt if negative_prompt is not None else NEGATIVE_PROMPT
        guidance = guidance_scale if guidance_scale is not None else IMAGE_GUIDANCE
        _log.info("开始本地生成…")
        with torch.inference_mode():
            image = pipe(
                prompt=prompt, negative_prompt=neg,
                height=IMAGE_HEIGHT, width=IMAGE_WIDTH,
                num_inference_steps=IMAGE_STEPS, guidance_scale=guidance,
            ).images[0]
        _log.info("本地生成完毕")
        return image

    def _generate_bailian(
        self, prompt: str, negative_prompt: str = None,
        api_key: str = None, api_model: str = None,
    ) -> Image.Image:
        from core.image.api import BailianImageAPI
        key = api_key or DASHSCOPE_API_KEY
        model = api_model or IMAGE_API_MODEL
        _log.info("调用百炼 API: model=%s", model)
        client = BailianImageAPI(api_key=key, model=model)
        return client.generate(prompt=prompt, negative_prompt=negative_prompt or "",
                               width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

    def edit(
        self, image: "Image.Image", instruction: str,
        edit_model: str = "wanx2.1-imageedit", strength: float = 0.75,
        api_key: str = None,
    ) -> "Image.Image":
        from core.image.api import BailianImageEditAPI
        key = api_key or DASHSCOPE_API_KEY
        _log.info("调用百炼编辑 API: model=%s, strength=%s", edit_model, strength)
        client = BailianImageEditAPI(api_key=key, model=edit_model)
        return client.edit(image=image, instruction=instruction, strength=strength)
