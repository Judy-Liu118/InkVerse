"""
core.evaluation.clip -- CLIP 图文一致性评分器
"""
import torch
from PIL import Image
from typing import Optional
from config import CLIP_MODEL_PATH
from core.logger import get_logger

_log = get_logger(__name__)

CLIP_MAX_TOKENS = 77


class CLIPEvaluator:
    """单例 CLIP 评分器，CPU 推理。"""

    _instance: Optional["CLIPEvaluator"] = None
    _model = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            raise ImportError("请先安装 transformers：pip install transformers")
        _log.info("加载 CLIP 模型中…")
        self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH, local_files_only=True)
        self._model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, local_files_only=True)
        self._model.eval()
        _log.info("CLIP 加载完成")

    def score(self, image: Image.Image, text: str) -> float:
        self._load()
        text_clipped = self._clip_text(text)
        inputs = self._processor(
            text=[text_clipped], images=image, return_tensors="pt",
            padding=True, truncation=True, max_length=CLIP_MAX_TOKENS,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            cosine = (img_emb * txt_emb).sum().item()
        score = (cosine + 1.0) / 2.0
        _log.debug("余弦相似度=%.4f → 归一化分数=%.4f", cosine, score)
        return round(score, 4)

    def score_raw_cosine(self, image: Image.Image, text: str) -> float:
        normalized = self.score(image, text)
        return round(normalized * 2.0 - 1.0, 4)

    @staticmethod
    def _clip_text(prompt: str) -> str:
        lines = prompt.strip().split("\n")
        key_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith(("subject:", "environment:", "atmosphere:")):
                key_lines.append(stripped)
        extracted = " ".join(key_lines) if key_lines else prompt
        return extracted[:300]

    def score_text_text(self, text_a: str, text_b: str) -> float:
        """计算两段文本的 CLIP 余弦相似度（用于诗-提示词匹配检查）。

        使用 CLIP text encoder 分别编码两段文本，归一化后计算余弦相似度。
        返回值范围与 score_raw_cosine 一致（原始余弦值，约 0.10~0.40）。
        """
        self._load()
        ta = self._clip_text(text_a)
        tb = self._clip_text(text_b)
        with torch.no_grad():
            emb_a_raw = self._model.get_text_features(
                **self._processor(text=[ta], return_tensors="pt", padding=True, truncation=True)
            )
            emb_b_raw = self._model.get_text_features(
                **self._processor(text=[tb], return_tensors="pt", padding=True, truncation=True)
            )
            # transformers 4.38+ 返回 BaseModelOutputWithPooling（需取 .pooler_output），
            # 较早版本直接返回 Tensor。实测日志中曾出现 AttributeError: 'norm'，故此分支非死代码。
            emb_a = emb_a_raw.pooler_output if hasattr(emb_a_raw, 'pooler_output') else emb_a_raw
            emb_b = emb_b_raw.pooler_output if hasattr(emb_b_raw, 'pooler_output') else emb_b_raw
            emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
            emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)
            cosine = (emb_a * emb_b).sum().item()
        _log.debug("文本-文本余弦相似度=%.4f", cosine)
        return round(cosine, 4)

    def release(self):
        if self._model is not None:
            import gc
            del self._model, self._processor
            self._model = self._processor = None
            gc.collect()
            _log.info("CLIP 模型已从内存释放")
