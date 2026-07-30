# 安装与配置

README 的[快速开始](../README.md#快速开始)只列了最短路径。本文补齐模型下载、`.env` 全字段与依赖清单。

## 环境要求

- Python 3.10+
- CUDA 12.4+
- 显存 ≥ 8GB（LoRA 与 Z-Image 分时加载，不共存）

纯 API 模式无需 GPU：跳过下面全部模型下载，只配 `DASHSCOPE_API_KEY` 即可运行。

## 模型下载

**基座模型 Qwen2.5-1.5B-Instruct**

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir D:\AI_Models\Qwen2.5-1.5B-Instruct
```

**古诗 LoRA 权重**

权重发布于 [Judy-Liu118/poetry-lora](https://huggingface.co/Judy-Liu118/poetry-lora)，下载后放入 `models/poetry_lora/`。基于开源古诗语料 [CanvaChen/llm-dataset-chinese-poetry](https://github.com/CanvaChen/llm-dataset-chinese-poetry) 经五阶段格律清洗管线（34.6 万首 → 18.1 万首合律 → 3 万条均衡 SFT 样本，含名诗白名单与出律配额 ≤5% 等质量控制）微调 Qwen2.5-1.5B 得到——数据清洗与训练全记录见 [LORA_TRAINING.md](LORA_TRAINING.md)。古诗生成默认使用 LoRA，本地微调模型在格律规范性上优于通用 API。

```bash
hf download Judy-Liu118/poetry-lora --local-dir models/poetry_lora
```

**Z-Image Turbo FP8**

[ykarout/Z-Image-Turbo-FP8-Full](https://huggingface.co/ykarout/Z-Image-Turbo-FP8-Full)，基于 Tongyi-MAI/Z-Image-Turbo 的 FP8 量化版。

```bash
hf download ykarout/Z-Image-Turbo-FP8-Full --local-dir D:\AI_Models\z_image_fp8_full
```

**CLIP ViT-B/32**

```bash
hf download openai/clip-vit-base-patch32 --local-dir D:\AI_Models\clip-vit-base-patch32
```

## 安装

```bash
git clone https://github.com/Judy-Liu118/InkVerse.git
cd InkVerse
pip install -r requirements.txt
cp .env.example .env          # 仅 Windows PowerShell：Copy-Item .env.example .env
```

## `.env` 配置

```env
# 必填
DASHSCOPE_API_KEY=sk-xxxxxxxx     # 阿里百炼（评分/提示词/图像 API）

# 可选 —— 不填则启动时自动隐藏对应「本地」选项，仅保留 API 后端
# BASE_MODEL_PATH=D:\AI_Models\Qwen2.5-1.5B-Instruct
# LORA_PATH=./models/poetry_lora
# ZIMAGE_PATH=D:\AI_Models\z_image_fp8_full
```

**纯 API 模式**：跳过模型下载、只配 `DASHSCOPE_API_KEY` 即可运行；UI 会自动只显示百炼 API 后端选项。

**本地 + API 混合模式**：填入三条本地路径，UI 同时显示本地 LoRA / Z-Image 与 API 后端，按需切换。

## 运行

```bash
python app.py
```

浏览器打开 `http://localhost:7860`。启动 banner 会打印每项资源的可用性，方便确认当前在哪种模式下运行：

```
本地 LLM 基座:     可用 / 未启用（API 模式）
本地 LoRA Adapter: 可用 / 未启用
本地 Z-Image:     可用 / 未启用（百炼 API 模式）
```

## 依赖

**核心（API 模式必装）**

- `gradio` — Web UI
- `openai` — 阿里百炼 / DeepSeek API（OpenAI 兼容）
- `pypinyin` + `pingshui_rhyme` — 平仄标注与平水韵部
- `Pillow`、`requests`

**CLIP 评分（推荐安装）**

- `torch` + `transformers` — CLIP 图文一致性评分

**本地后端（可选）**

- `unsloth` + `peft` + `bitsandbytes` — Qwen2.5 4-bit LoRA 加载
- `diffusers` + `xformers` — Z-Image Turbo FP8 扩散模型

未安装可选依赖时，相关本地选项会在 UI 中自动隐藏，应用照常运行。
