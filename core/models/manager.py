"""
core.models.manager -- 单例模型管理器，负责加载、缓存和释放所有模型
"""
import gc
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from diffusers import ZImagePipeline
from config import BASE_MODEL_PATH, LORA_PATH, ZIMAGE_PATH
from core.logger import get_logger

_log = get_logger(__name__)


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
        self._release_base()
        self._release_z_pipe()
        _log.info(">>> 开始从本地加载微调模型 (诗歌生成)...")
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
        self._release_fine()
        self._release_z_pipe()
        _log.info(">>> 开始从本地加载 Qwen 基础大模型...")
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
        self._release_fine()
        self._release_base()
        _log.info(">>> 开始加载 Z-Image 绘图管线...")
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
        gc.collect()
        torch.cuda.empty_cache()

    def release_all(self):
        _log.info("收到强制清空指令，正在释放所有模型...")
        self._release_fine()
        self._release_base()
        self._release_z_pipe()
        _log.info("显存全线清空。")
