"""
core.models.manager -- 单例模型管理器，负责加载、缓存和释放所有模型

重型依赖（unsloth / peft / diffusers）和路径校验都延迟到实际加载方法内，
保证未安装这些包或未配置本地模型路径的环境也能 import 本模块，
仅在用户主动选择"本地"后端时才报错。
"""
import gc
import os
import torch
from config import BASE_MODEL_PATH, LORA_PATH, ZIMAGE_PATH
from core.logger import get_logger

_log = get_logger(__name__)


def _require_local_path(path: str, kind: str) -> None:
    """本地模型路径不存在时抛出清晰错误，引导用户配置环境变量。"""
    if not path or not os.path.isdir(path):
        raise FileNotFoundError(
            f"[{kind}] 本地模型路径不存在: {path!r}\n"
            f"如需使用本地后端，请在 .env 中配置对应环境变量"
            f"（参考 .env.example）；或在 UI 中改选 API 后端。"
        )


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._fine_model = None
        self._fine_tokenizer = None
        self._base_model = None
        self._base_tokenizer = None
        self._z_pipe = None

    # ── 微调模型 (诗歌生成) ───────────────────────────────────────────────────
    @property
    def fine_model(self):
        if self._fine_model is None:
            self._load_fine_tuned()
        return self._fine_model

    @property
    def fine_tokenizer(self):
        if self._fine_tokenizer is None:
            self._load_fine_tuned()
        return self._fine_tokenizer

    def _load_fine_tuned(self):
        _require_local_path(BASE_MODEL_PATH, "基础模型")
        _require_local_path(LORA_PATH, "LoRA Adapter")
        self._release_base()
        self._release_z_pipe()
        _log.info(">>> 开始从本地加载微调模型 (诗歌生成)...")
        from unsloth import FastLanguageModel
        from peft import PeftModel
        from config import LORA_MAX_SEQ_LEN
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_PATH,
            max_seq_length=LORA_MAX_SEQ_LEN,
            load_in_4bit=True,
        )
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model = FastLanguageModel.for_inference(model)
        self._fine_model = model
        self._fine_tokenizer = tokenizer
        _log.info("<<< 微调模型加载成功，已进入显存。")

    # ── 基础模型 (起名与提示词) ───────────────────────────────────────────────
    @property
    def base_model(self):
        if self._base_model is None:
            self._load_base()
        return self._base_model

    @property
    def base_tokenizer(self):
        if self._base_tokenizer is None:
            self._load_base()
        return self._base_tokenizer

    def _load_base(self):
        _require_local_path(BASE_MODEL_PATH, "基础模型")
        self._release_fine()
        self._release_z_pipe()
        _log.info(">>> 开始从本地加载 Qwen 基础大模型...")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_PATH,
            max_seq_length=400,
            load_in_4bit=True,
        )
        model = FastLanguageModel.for_inference(model)
        self._base_model = model
        self._base_tokenizer = tokenizer
        _log.info("<<< 基础大模型加载成功，已进入显存。")

    # ── Z-Image (绘图管线) ───────────────────────────────────────────────────
    @property
    def z_pipe(self):
        if self._z_pipe is None:
            self._load_z_pipe()
        return self._z_pipe

    def _load_z_pipe(self):
        _require_local_path(ZIMAGE_PATH, "Z-Image 绘图管线")
        self._release_fine()
        self._release_base()
        _log.info(">>> 开始加载 Z-Image 绘图管线...")
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(
            ZIMAGE_PATH,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        from config import GPU_BEAST_MODE
        if GPU_BEAST_MODE:
            pipe = pipe.to("cuda")
            _log.info("  显存狂暴模式：全量权重直压显存（生图加速）")
        else:
            pipe.enable_model_cpu_offload()
            _log.info("  CPU Offload 模式：省显存，每步搬运权重")
        pipe.vae.enable_tiling()
        self._z_pipe = pipe
        _log.info("<<< Z-Image 绘图管线加载成功。")

    # ── 释放显存 ─────────────────────────────────────────────────────────────
    def _release_fine(self):
        if self._fine_model is not None:
            _log.debug("释放内存：正在卸载微调模型...")
            del self._fine_model, self._fine_tokenizer
            self._fine_model = self._fine_tokenizer = None
            self._flush_gpu()

    def _release_base(self):
        if self._base_model is not None:
            _log.debug("释放内存：正在卸载基础大模型...")
            del self._base_model, self._base_tokenizer
            self._base_model = self._base_tokenizer = None
            self._flush_gpu()

    def _release_z_pipe(self):
        if self._z_pipe is not None:
            _log.debug("释放内存：正在卸载 Z-Image 绘图管线...")
            del self._z_pipe
            self._z_pipe = None
            self._flush_gpu()

    def _flush_gpu(self):
        before_mb = self._cuda_memory_mb()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        after_mb = self._cuda_memory_mb()
        if before_mb is None:
            return
        freed = before_mb - after_mb
        # unsloth 的 4bit 量化模型在 del 后仍可能被内部 patch / 全局缓存持有，
        # 导致 memory_allocated 不下降（已知行为，不是泄漏）。
        # 因此把 0 释放降级为 INFO；只在大模型场景（>2GB）才告警。
        if freed < 50 and before_mb > 2048:
            _log.warning("[显存] flush 后释放量异常低: %.1f → %.1f MB（差 %.1f MB），"
                         "可能有未释放的引用（Gradio 缓存 / 循环引用）",
                         before_mb, after_mb, freed)
        elif freed < 50 and before_mb > 100:
            _log.info("[显存] flush: %.1f MB 保持不变（unsloth 4bit 模型常见，权重仍被内部缓存）",
                      before_mb)
        else:
            _log.info("[显存] flush: %.1f → %.1f MB（释放 %.1f MB）",
                      before_mb, after_mb, freed)

    @staticmethod
    def _cuda_memory_mb():
        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_allocated() / (1024 ** 2)

    def release_all(self):
        _log.info("收到强制清空指令，正在释放所有模型...")
        self._release_fine()
        self._release_base()
        self._release_z_pipe()
        _log.info("显存全线清空。")
