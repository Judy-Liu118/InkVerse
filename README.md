# InkVerse · 诗画墨语

[![tests](https://github.com/Judy-Liu118/InkVerse/actions/workflows/tests.yml/badge.svg)](https://github.com/Judy-Liu118/InkVerse/actions/workflows/tests.yml)

**AI 古诗创作与水墨画生成系统** —— 本地 LoRA 生成格律诗 + Z-Image Turbo 文生图 + Pairwise 进化择优。消费级显卡可运行。

输入"写一首描写夏天的七言绝句，要有意向荷花"，系统从生成五首候选、硬门控筛选、擂台进化打磨到最终配图出稿，全程无需人工介入。

古诗生成使用本地 LoRA 微调模型（Qwen2.5-1.5B + LoRA，古典诗词数据集训练），图像生成使用本地 FP8 量化 Z-Image Turbo。两者分时加载，消费级 8GB 显存可运行。古诗评审、切题判断、诗名与提示词生成等语言任务调用 API（推荐阿里百炼 qwen 系列）——本地小模型在鉴赏类环节与大模型存在显著差距。

## 作品示例

以下六组均为评估 run 的真实产物（终稿诗 + CLIP 择优终图），未经人工挑改字句。带 † 的两首出自消融实验的"无擂台"对照臂——一次生成直达此质量，选入正好展示基座 LoRA 的底子；其余四首为带擂台的全流程产物。六首诗均由本地 LoRA（Qwen2.5-1.5B + LoRA）生成；配图模型为百炼 `qwen-image-2.0-pro`（《客愁》《春雨江钓》用 2026-03-03 快照，其余四首用 2026-04-22 快照）。

第一排三首是 sparse 抽象主题（只给主题词，不点名意象），看意境经营：

| 「写一首七言律诗，<br>主题是客愁」 | 「写一首五言绝句，<br>主题是溪声」 | 「写一首七言绝句，<br>主题是春雨」 |
|---|---|---|
| ![客愁](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260630_202046/delta_0.20/04_%E9%A3%8E%E7%BF%BB%E5%A2%A8%E6%B5%AA%E5%AE%A2%E8%A1%A3%E5%8D%95_gen1_clip0.347.jpg) | ![溪声](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/07_%E5%BE%AE%E9%A3%94%E7%A9%BF%E7%AB%B9%E5%BE%84_gen2_clip0.308.jpg) | ![春雨江钓](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260630_205747/delta_0.14/10_%E6%B1%9F%E5%A4%B4%E6%98%A5%E9%9B%A8%E6%BF%AF%E4%BA%91%E5%B1%8F_gen1_clip0.322.jpg) |
| **《客愁》**<br>风翻墨浪客衣单<br>霜凝石径暮烟寒<br>雁衔夕照千峰瘦<br>雨织灯痕一水残<br>孤棹摇波星欲堕<br>半窗移竹影初攒<br>归期暗数芦花雪<br>故国遥看月浸滩 | **《溪声》**<br>微飔穿竹径<br>清响落云间<br>松影摇溪浅<br>苔痕浸水闲 | **《春雨江钓》**<br>江头春雨濯云屏<br>风飐垂杨绿欲停<br>偶倚沙痕听鹭语<br>一蓑烟水钓波青 |

左例诗中意象在画面逐一可辨（霜径、雁阵、竹影、月浸滩）；中例竹径夹溪、苔痕浸水，画面即是诗境；右例"濯云屏"（雨洗云如洗屏风）是 sweep 报告赏析认定的全篇诗眼，画中鹭、垂杨、烟水俱在。

第二排看**点名意象的兑现**：三首均为 rich 题，用户点名的意象须在诗与画中同时出现，均经 VLM 硬约束核查 2/2 命中：

| 「写一首送别的七言律诗，<br>要有长亭和折柳」† | 「写一首禅意的七言绝句，<br>要有古刹和钟鼓」† | 「写一首田园的七言绝句，<br>要有耕牛和炊烟」 |
|---|---|---|
| ![离亭折柳](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/16_%E6%B1%9F%E5%A4%B4%E9%85%92%E5%8A%9B%E5%B7%B2%E5%BE%AE%E9%86%BA_gen1_clip0.322.jpg) | ![古刹钟声](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/21_%E5%8F%A4%E5%88%B9%E9%92%9F%E5%A3%B0%E5%A4%9C%E6%9C%AA%E4%BC%91_gen1_clip0.332.jpg) | ![春村耕趣](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/09_%E6%9D%91%E8%B7%AF%E9%A3%8E%E6%B8%85%E9%9B%A8%E8%BF%87%E7%A8%80_gen3_clip0.322.jpg) |
| **《离亭折柳》**<br>江头酒力已微醺<br>一别相逢几许春<br>风递梅花知远近<br>水含明月见相亲<br>青门路断无多雨<br>白发人归少有尘<br>欲向离亭折杨柳<br>可怜无力更伤神 | **《古刹钟声》**<br>古刹钟声夜未休<br>松间野鹤伴云游<br>人间岁月如流水<br>更向何时得少留 | **《春村耕趣》**<br>村路风清雨过稀<br>春来无事亦忘机<br>耕牛未肯还田去<br>犹把新泥护柳枝 |

左例"欲向离亭折杨柳"——画面水畔长廊连着离亭，垂柳正拂过画前；中例首句"古刹钟声夜未休"锚定两个约束，殿中铜钟与松间野鹤俱在画上；右例"耕牛未肯还田去，犹把新泥护柳枝"是消融报告里"擂台把丢失的用户约束补回来"的直接铁证主题（无擂台臂两个约束全丢，擂台臂 2/2）。更多逐题三方对比见 [sweep 报告](eval/REPORT_pairwise_win_delta_sweep_2026-06-30.md) §3、擂台有无的双图并排见[消融报告](eval/REPORT_arena_ablation_20260701.md) §4.1。

## 流程

```mermaid
flowchart TD
    A(["用户一句话要求"]) --> B["任务规划<br>体裁 · 意象 · 风格解析"]

    subgraph P ["诗 —— 本地 LoRA + 两级擂台"]
        C["Arena 海选<br>LoRA 生成 5 首候选"] --> D["硬门控<br>押韵 · 平仄 · 堆砌词 · 重复"]
        D --> E{"合格 ≥ 3 首?"}
        E -- "否 · 重新生成 ≤3 轮" --> C
        E -- "是" --> F["本地五维评分 + 切题评估<br>Top3 轮循 pairwise → 冠军"]
        F --> G["守擂进化 ×2 轮<br>每轮 2 挑战者 1v1 对决<br>A/B 位随机化判定"]
    end
    B --> C

    subgraph I ["画 —— 生图 + CLIP 门控"]
        H["关键词提取 → 诗名<br>→ 英文提示词 + 自检"] --> J["文生图<br>Z-Image Turbo / 百炼 API"]
        J --> K["CLIP 双锚点评分<br>诗-图 0.6 + 提示词-图 0.4"]
        K --> L{"CLIP raw ≥ 0.30?"}
        L -- "否 · ≤2 轮 · 自适应停止" --> M["改图循环<br>从历史最优图出发 · 编辑强度衰减"]
        M --> K
    end
    G --> H
    L -- "是 / 预算耗尽返回历史最优" --> Z(["成品：诗 + 画 + 创作报告"])

    classDef poem fill:#e9f3ec,stroke:#41805e,color:#173324
    classDef img  fill:#e9eef8,stroke:#4a63a8,color:#16213f
    classDef gate fill:#fbf1dd,stroke:#c2913a,color:#43350f
    classDef ends fill:#f7e9e9,stroke:#a25454,color:#3c1c1c
    class B,C,D,F,G poem
    class H,J,K,M img
    class E,L gate
    class A,Z ends
```

### Arena 海选

LoRA 生成 5 首候选。不使用绝对评分排序，而是先过硬门控再做 pairwise 对决。

1. 硬门控：押韵、平仄、堆砌词黑名单、重度重复——纯本地规则，零成本拦截不合格的诗
2. 切题评估：一次 LLM 调用评判五首主题契合度，配合本地季节/昼夜/天气矛盾扫描
3. 本地评分：平仄、押韵、意象丰富度、主题连贯性、切题度，五项加权
4. 轮循 pairwise：Top3 两两对决，LLM 做比较判断
5. 综合：本地分 × 0.75 + pairwise 胜率 × 0.25

单轮合格不足 3 首时重新生成（最多 3 轮），累计合格池直到满足要求。

### 守擂进化

冠军成为擂主，每轮从不同维度生成 2 个挑战者。挑战者先过硬门控，再与擂主进行 1v1 pairwise 对决。综合本地客观分与 pairwise 微调决定是否易位。每轮基于上一轮最优版本继续打磨。

### 图像流水线

冠军诗定稿后：关键词提取 → 诗名生成 → 英文提示词 → 生图 → CLIP 双锚点评分（诗-图 + 提示词-图）→ 改图循环。改图时每轮从历史最优图出发，避免在改坏的图上继续改。自适应停止：连续两轮无显著提升则提前退出。

### 为什么是 pairwise

LLM 给单首诗打绝对分数波动极大——同一首诗两次调用可差 1 分，五首分数全部簇集在 7.5-9.0 区间，缺乏区分度。"这两首哪首更好"是 LLM 擅长的判断。系统所有需要 LLM 评估质量的环节均使用比较而非打分。

## 快速开始

### 环境

- Python 3.10+
- CUDA 12.4+
- 显存 ≥ 8GB（LoRA 与 Z-Image 分时加载，不共存）

### 模型下载

**基座模型 Qwen2.5-1.5B-Instruct**

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir D:\AI_Models\Qwen2.5-1.5B-Instruct
```

**古诗 LoRA 权重**

基于古典诗词数据集微调，数据集 [Judy-Liu118/poetry-lora](https://huggingface.co/datasets/Judy-Liu118/poetry-lora)。权重放入 `models/poetry_lora/`。古诗生成默认使用 LoRA，本地微调模型在格律规范性上优于通用 API。

**Z-Image Turbo FP8**

[ykarout/Z-Image-Turbo-FP8-Full](https://huggingface.co/ykarout/Z-Image-Turbo-FP8-Full)，基于 Tongyi-MAI/Z-Image-Turbo 的 FP8 量化版。

```bash
hf download ykarout/Z-Image-Turbo-FP8-Full --local-dir D:\AI_Models\z_image_fp8_full
```

**CLIP ViT-B/32**

```bash
hf download openai/clip-vit-base-patch32 --local-dir D:\AI_Models\clip-vit-base-patch32
```

### 安装与配置

```bash
git clone https://github.com/Judy-Liu118/InkVerse.git
cd InkVerse
pip install -r requirements.txt
cp .env.example .env          # 仅 Windows PowerShell：Copy-Item .env.example .env
```

编辑 `.env` 填入需要的 Key 和（可选）本地模型路径：

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

启动 banner 会打印每项资源的可用性，方便确认当前在哪种模式下运行：

```
本地 LLM 基座:     可用 / 未启用（API 模式）
本地 LoRA Adapter: 可用 / 未启用
本地 Z-Image:     可用 / 未启用（百炼 API 模式）
```

### 运行

```bash
python app.py
```

浏览器打开 `http://localhost:7860`。

## 界面说明

### 模型选择

| UI 标签 | 推荐模型 | 说明 |
|---------|---------|------|
| 诗歌生成模型 | Qwen2.5-1.5B + LoRA | 本地，格律规范性优于通用 API |
| 意图评分模型 | qwen-plus | API，覆盖切题评估、擂台 pairwise |
| 诗名生成模型 | qwen-plus | API |
| 提示词生成模型 | qwen-max | API，英文结构化 prompt 需要较强模型 |
| 图像后端 | z-image-turbo（百炼 API）/ 本地 Z-Image | API 更快、分辨率更高；本地无网络依赖 |
| 自主图像编辑模型 | qwen-image-edit-max | API，保留构图仅修改局部 |

无 API 时图像编辑自动降级为"改写重生图"模式——LLM 将意见融入 Prompt 后重新生图。图像 API 调用失败也会自动切本地 Z-Image。

### 图像风格

支持五种风格，通过下拉框选择。不同风格会注入对应的英文 prompt 前缀，影响生图效果：

- 水墨画：`Chinese ink wash painting, sumi-e, monochrome, minimalist, Song Dynasty style`
- 写意画：`xieyi freehand ink painting, expressive spontaneous brushwork, loose poetic strokes`
- 青绿山水：`Chinese blue-green landscape, qinglu style, mineral pigments, Tang Dynasty luminous`
- 油画、卡通插画

推荐使用水墨画或写意画，与中国古典诗主题最为契合。

### 按钮功能

| 按钮 | 说明 |
|------|------|
| 开始创作 | 逐步执行：生成候选 → 用户可中途改诗 → 配图。适合已知要写什么诗、需要逐步控制的场景 |
| 全自主创作 | 一键走完 Arena 海选 → 守擂进化 → 配图全流程。选好所有模型后直接点击 |
| 改诗 | 在"改诗意见"框输入修改方向，选择改诗模型。基于当前版本进行定向修改 |
| 仅重新生成图 | 若对当前 Prompt 不满意，直接修改 Prompt 文本框后点击，基于新 Prompt 重新生图 |
| 图像编辑 | 在"改图意见"框输入修改指令（如"增加月光感"），调用编辑 API 保留构图局部修改 |
| 改写重生图 | 在"改图意见"框输入意见，LLM 将意见融入 Prompt 后丢弃原图重新生成，适合大幅改动 |
| 生成报告 | 将当前诗文、图像、模型使用记录导出为 HTML 报告 |

**快速上手**：选好所有模型和风格 → 输入创作要求 → 点「全自主创作」。

**使用已有古诗**：将诗粘贴到"诗文"文本框 → 清空"创作要求" → 点「全自主创作」或「开始创作」。系统以你的诗为擂主直接进入进化打磨，然后配图。

## 使用示例

### 示例一：全自主生成

创作要求：*写一首描写夏天的七言绝句，要有意向荷花*

模型配置：诗歌生成 LoRA + 意图评分 qwen-plus + 诗名 qwen-plus + 提示词 qwen-max + 图像 z-image-turbo + 编辑 qwen-image-edit-max

| 全自主创作·初始 | 评分详情 |
|---|---|
| <img src="https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example1-auto-init.png" alt="全自主创作·初始" width="380"> | <img src="https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example1-auto-score.png" alt="评分详情" width="380"> |

第一轮改图后 CLIP 从 0.312 提升至 0.333，达标退出循环。改图指令为"增加竹荫下的曲径和池亭的细节描绘"。

| 改图前（0.312） | 改图后（0.333） |
|---|---|
| <img src="https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example1-before.png" alt="改图前" width="330"> | <img src="https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example1-after.png" alt="改图后" width="330"> |

### 示例二：回滚机制

创作要求：*写一首描写秋天的七言律诗，要有意向菊花*

第一轮改图 CLIP 从 0.302 退至 0.269——改动降低了图文一致性。系统回滚到初始图，第二轮从初始图出发继续改，而非在改坏的图上叠加修改。最终两轮未达目标，自动退回历史最优结果（0.302）。

| 原版（0.302） | 第一次修改（0.269 ↓） | 第二次修改（0.291 ↓） |
|---|---|---|
| <img src="https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example2-original.png" alt="原版" width="250"> | <img src="https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example2-edit1.png" alt="第一次修改" width="250"> | <img src="https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example2-edit2.png" alt="第二次修改" width="250"> |

- 第一次修改：增加月光洒在鹤羽毛上的细节，增强幽静感（保留原图构图，仅修改指令涉及内容）——得分 0.269 低于原版
- 第二次修改：由于第一版得分低于原版，系统**回滚到原版图**上重新修改，而非在第一版的残骸上继续。增强霜覆盖山岭的效果，突出秋意浓厚——得分 0.291 仍低于原版

两轮得分均低于原版，回退原版，最终效果：

<img src="https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example2-final.png" alt="最终效果" width="700">

### 示例三：输入已有诗 + 手动改图

创作要求：*以边塞为主题写一首七言绝句*

先用全自主生成一首边塞诗。若对某首诗更满意，将创作要求清空、诗文粘贴到文本框，点击「开始创作」。系统跳过生成环节，直接用这首诗生成图像。图为对原图不满意选原擂主诗点击开始创作生成：

![开始创作·初始](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example3-init.png)

对画面不满意时，在"改图意见"框输入具体修改指令：

- "在画面中加上将军，体现将军白发不胜簪"
![修改一·加将军](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example3-edit1.png)

- 发现给的意见太粗糙将军太大了，在此基础上进一步提出修改意见，"不要这么大的将军，将军小一些，背对着，可以坐在战马上"
![修改二·将军调小](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example3-edit2.png)

每次点击「图像编辑」，系统基于当前图像按指令局部修改。

最终修改后和修改前图片对比：

<img src="https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example3-compare.png" alt="修改前后对比" width="360">

### 示例四：生成诗和图后点击生成报告

<img src="https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/docs/images/example4-report.png" alt="生成报告" width="440">

## 目录结构

```
InkVerse/
├── app.py                  # Gradio UI
├── config.py               # 全局配置 + 本地模型路径可用性探测
├── core/
│   ├── agent/
│   │   ├── agent.py        # 创作引擎（PoetryAgent，_phase_* 主流水线）
│   │   ├── autonomous.py   # 全自主模式调度
│   │   ├── state.py        # 状态与追踪
│   │   └── tools.py        # Tool 抽象 + ToolRegistry（Function Calling 兼容）
│   ├── poem/
│   │   ├── generator.py    # 古诗生成 + Arena 海选
│   │   ├── scorer.py       # 评分 + pairwise + 切题评估
│   │   └── theme.py        # 意象与情感主题词表
│   ├── image/
│   │   ├── generator.py    # 图像生成双后端
│   │   ├── prompt.py       # 提示词生成器
│   │   └── api.py          # 百炼 API 客户端
│   ├── models/
│   │   ├── adapter.py      # 统一模型适配层（local/deepseek/qwen）
│   │   └── manager.py      # 显存管理（重型依赖延迟加载）
│   ├── evaluation/
│   │   └── clip.py         # CLIP 评分器
│   └── logger.py
├── prompts/                # 集中化 prompt YAML（含 README 与版本号）
├── tests/                  # pytest 单测（adapter/state/scorer/tools/clip/prompts）
├── models/                 # LoRA 权重
├── outputs/                # 生成的图像与报告
└── fonts/
```

## Prompt 集中管理

所有 LLM system / user prompt 抽离到 `prompts/` 目录，以 YAML 形式管理，由 `core.prompts` 模块统一加载渲染：

```python
from core.prompts import render_messages

messages = render_messages(
    "agent.refine_poem",
    expected_chars=7, expected_lines=4,
    old_poem="...", feedback="加强意境深度",
)
```

- **可审阅**：git diff 时 prompt 变更不被代码改动淹没
- **可枚举**：`list_prompts()` 一键列出全部 prompt，便于审计与 A/B 测试
- **可追踪**：每个 YAML 自带 `version` + `description`，配合 git 即天然版本管理
- **fail-fast**：缺变量直接抛 `KeyError`，避免静默生成残缺 prompt

详见 [`prompts/README.md`](prompts/README.md)。

## Tool 抽象与可调度性

`core.agent.tools` 提供 OpenAI Function Calling 兼容的工具基类（`AgentTool`）与注册表（`ToolRegistry`）：工具可枚举、可 introspection，`to_function_schemas()` 直接导出 LLM tools 描述。

生产中真实使用它的是 LLM-driven 改图循环（`core.agent.controller`）：`build_loop_registry` 注册 `edit_image` / `refine_poem_and_regen` 两个工具，controller 把工具 schema 注入决策 prompt，LLM 返回的 JSON 决策经 `ToolRegistry.execute` 按名真实调度。

```python
from core.agent import PoetryAgent, build_loop_registry

agent = PoetryAgent(...)
reg = build_loop_registry(agent)
print(reg.names)                      # ['edit_image', 'refine_poem_and_regen']

schemas = reg.to_function_schemas()   # 注入 controller 决策 prompt 的 tools 描述
state = reg.execute("edit_image", state, feedback="强化孤舟主体")   # 按名调度
```

早期版本曾把全部 10 个 `_phase_*` 阶段都包成 Tool，但除上述两个外生产从未调度过，已删除——只保留被真实使用的抽象。

## 离线评估

`eval/` 目录提供 4 个独立可跑的评估脚本，用于量化项目里的核心设计点：

```bash
# 1. 诗歌生成模型质量对比（BWS + 跨家族多评委 pairwise，支持 --repeat 多 run）
python -m eval.eval_poem --models local_lora qwen-plus \
    --scorer qwen-max glm-4-plus --n 10

# 2. 双锚点 CLIP vs 单锚点（项目核心创新；含 VLM ground truth）
python -m eval.eval_clip --n 10

# 3. 自动方向性诗评 + refine_poem 的提升幅度
python -m eval.eval_refine --n 10

# 4. 全自主模式 vs 单轮模式 CLIP 终值 + 耗时对比
python -m eval.eval_autonomous --n 5

# 5. LLM-driven vs 写死改图循环（同基图配对 A/B，预登记 + 配额熔断 + 全程审计记录）
python -m eval.eval_llm_loop_ab --limit 1   # 冒烟；全量去掉 --limit
```

完整方法论（公式 / 系数 / 评委 prompt / 阈值）见 [`eval/METHODOLOGY.md`](eval/METHODOLOGY.md) —— 该文档冻结当前 commit 的实验方法，保证后续代码漂移仍能解释历史报告。

每次跑完会在 `outputs/eval/` 下落两份产物：
- `<name>_<timestamp>.json` —— 原始数据，便于二次分析
- `<name>_<timestamp>.md`   —— markdown 报告，含均值/std/配对差值/胜率/抽样对照，可直接抄进实验章节

详见 [`eval/README.md`](eval/README.md)（含 benchmark 数据集说明、参数约定、结果解读建议）。

### 代表性发现 · n=32 × 3 run 主跑

`eval_poem --models local_base local_lora local_lora_naked qwen-plus --scorer deepseek-v4-pro qwen-max glm-4-plus moonshot-v1-32k --n 32 --candidates 5 --repeat 3`

跨 4 LLM 评委（跨家族抗 self-bias）+ 3 run（暴露 LLM noise，主要指标 std 0.005-0.03 → 结论 reproducible）。

**1. LoRA 把格律内化进权重 —— 移掉 system prompt 反而更好**

| 指标 | LoRA full | **LoRA naked** | Δ |
|---|---|---|---|
| 平仄合格率 (≥0.8) | 95.4% ± 4.3% | **96.4% ± 3.0%** | +1.0pp |
| 押韵合格率 (≥0.8) | 32.9% ± 6.2% | **39.0% ± 1.8%** | +6.1pp |

naked 模式仅传简短 user request，不带任何格式约束。微调的格律合规来自权重本身，而非 in-context 引导。

**2. LoRA 提升地板，不提升天花板**

| 指标 | local_base | local_lora | Δ |
|---|---|---|---|
| pass@0.7 候选合格率 | 36.2% ± 2.6% | **64.0% ± 2.4%** | +27.8pp（≈ ×1.8）|
| best 候选 4 维 total | 0.771 ± 0.012 | 0.771 ± 0.004 | 0 |

候选池合格率几乎翻倍，best 候选评分持平。LoRA 收紧分布、砍坏样本 —— alignment fine-tune 的典型 pattern。

**3. LLM-as-judge 对格律给"低优先级"权重（n 较小，初步推测）**

| 模型 | 平仄合格率 | pairwise 胜率 |
|---|---|---|
| local_base | **25.6%** ± 6.4% | 0.356 ± 0.044 |
| local_lora | 95.4% ± 4.3% | 0.312 ± 0.025 |
| local_lora_naked | 96.4% ± 3.0% | 0.319 ± 0.028 |

间接证据：base 平仄只有 LoRA 的 1/4，pairwise 胜率反而最高。Retrospective controlled pair 验证（[REPORT_F3](eval/REPORT_F3_pingze_sensitivity_20260624.md)）：n=4 个"base 严重出律 (pingze 25-38%) 但意境维度 ≥ lora"的对决里，**base 胜率 87.5%** —— 意境维度只领先 ~0.1 就翻盘了极端格律差距。反向 (lora 格律反差，n=5) → lora 全败，说明评委不是完全不在意格律。

**修正表述（n=4-5 较小，属初步推测）：** 评委对格律给低优先级权重，意境维度有 ≥0.1 优势即可覆盖极端格律差距；生产里做格律保证仍必须保留 rule-based scorer 做硬约束，不能完全替换为 LLM judge。强结论需独立的 64-pair 重跑验证。

**方法论亮点：** 跨家族 4 评委集成（DeepSeek + Qwen + GLM + Moonshot）抗 self-bias · forward+reverse 双向 pairwise 暴露 position bias（摇摆率 23% 公开标注）· BWS 选 best 规避评分饱和 · 3 run mean ± std 区分信号 vs 噪声 · 评委解析失败显式弃权不污染 multi-judge 合成。

**评估反哺生产：** eval 侧的 position-bias 方法论反过来审计了生产擂台——发现 judge 固定"擂主 A 位/挑战者 B 位"布局后做双向重判探针，复盘中识别出探针自身的选择-复现混杂并主动降级结论，最终落地 A/B 位随机化防御；后续两个无混杂探针给出方向相反的信号，恰好验证了"随机化不赌方向"的设计选择（严谨版：[sweep 报告 §6.3](eval/REPORT_pairwise_win_delta_sweep_2026-06-30.md) · 通俗版故事线：[METHODOLOGY §7.5](eval/METHODOLOGY.md)）。

完整报告：`outputs/eval/eval_poem_<timestamp>.md` · 32 道分层 benchmark 见 [`eval/benchmark_themes.json`](eval/benchmark_themes.json)。

### 代表性发现 · LLM-driven 改图循环 vs 写死循环（同基图配对，n=27 × 2 次独立运行）

`eval_llm_loop_ab` —— 27 题（硬约束题集全集，零挑题）共享同一基图后 `_copy_state` 分叉两臂：**写死臂**每轮固定 edit API，**LLM 臂**由 controller 每轮自选 `edit_image(rewrite_regen|edit_api)` / `refine_poem_and_regen` / `stop`（real tool dispatch via `ToolRegistry`），两臂同预算（3 轮、target raw=0.30）。规则跑前预登记进脚本 docstring 并 commit；QuotaMeter 做模型白名单断言 + 预算硬熔断；每题落盘诗小分、逐轮中间图、决策埋点与终端全量镜像。**跑完一次后按预登记做了同条件独立重复（run 2）检验 run-to-run 稳定性。**

| 指标（循环子集） | run 1 | run 2 | 判读 |
|---|---|---|---|
| LLM − 写死 mean Δ CLIP raw | +0.004 | −0.004 | **打平复现**（符号翻转、幅度噪声级） |
| 平均循环耗时（写死 / LLM） | 433s / 263s | 240s / 195s | LLM 更快方向复现，幅度不稳 |
| 图像 API 调用（写死 / LLM） | 36 / 30 | 23 / 22 | LLM 略少，两次一致 |
| controller fallback / 改诗 / 主动 stop | 0 / 0 / 0 | 0 / 0 / 0 | 行为特征稳定 |

**两次独立运行支持的结论：**
1. 在这个窄循环里 LLM-driven 控制**没有带来 CLIP 终值优势（打平复现）**，但以略少调用、更短耗时达到相当结果——效率差异来自"达标即停"等可解释机制
2. run 1 观察到的两个苗头（LLM 臂达标更多、edit_api"选择效应"）**被 run 2 证伪并撤回**——这正是做 replication 的价值
3. 新量化：同题同管线独立重跑，基图 CLIP 波动（mean |Δ|=0.029）≈ **臂间效应的 7 倍**——单次运行中 ≤0.01 的差异一律不可判读
4. VLM 硬约束兑现率臂间差异不可判读：像素相同的两张图在 temperature=0 下仍被 qwen-vl-max 判出不同结果（测量噪声 ≥ 臂间差异）

完整报告：[run 1](eval/REPORT_llm_loop_ab_n27_20260710.md) · [run 2 稳定性对照](eval/REPORT_llm_loop_ab_run2_20260711.md)；直观图库（诗 + 基图 + 逐轮改图 + 两臂终图并排）：[run 1](eval/runs/llm_loop_ab/gallery/GALLERY.md) · [run 2](eval/runs/llm_loop_ab_run2/gallery/GALLERY.md)；原始数据 + 终端全量镜像日志在 `eval/runs/llm_loop_ab*/`。

此前的 n=5 先导实验（[REPORT_autonomous_n5](eval/REPORT_autonomous_n5_20260627.md)）曾给出"LLM 臂更差且贵 3×"的负面信号；本实验用配额隔离（消除 API 失败静默降级本地模型的污染）+ 同基图配对（消除前半段管线噪声）后，方向由负转平——两个混杂因子的识别与消除过程本身即是结论的一部分。

LLM-driven 路径在产品 UI 已可勾选启用（`app.py` "🤖 全自主创作"区域，"LLM 驱动改图循环（实验）"复选框），默认关闭走原 fixed loop 保持向后兼容。

## 测试

```bash
pytest tests/ -v
# 164 passed in ~5s（全部离线，不触发真实 LLM / 图像 API）
```

覆盖：
- `test_adapter.py` —— ModelAdapter 后端选择、env-var 回退、key 优先级
- `test_state.py` —— AgentState 默认值、trace 追踪、序列化往返、Phase 枚举稳定性
- `test_scorer.py` —— 平仄/押韵评分边界、合掌词库、堆砌词黑名单
- `test_clip_weights.py` —— CLIP 双锚点稀疏关键词自适应权重切换
- `test_tools.py` —— ToolRegistry 注册/查找/调度、Function Calling schema 形状
- `test_prompts.py` —— prompt YAML 解析、变量插值、缺变量 fail-fast、loader 缓存
- `test_controller.py` —— LLM-driven 改图循环 controller：JSON 解析兜底、非法工具 fallback、dispatch 分发
- `test_metrics.py` —— eval 指标数值正确性（相关性 / 配对差值 / 通过率）
- `test_vlm_judge.py` —— VLM ground-truth judge 解析、归一化、错误兜底
- `test_build_f3_controlled.py` —— F3 controlled pair 筛选边界条件防回归
- `test_poem_refiner.py` —— 擂台 pairwise A/B 位随机化：位置分配、胜负映射、弃权透传

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

## License

MIT
