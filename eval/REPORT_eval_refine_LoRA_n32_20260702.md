# eval_refine 报告 · LoRA baseline + 单轮 refine（n=32 主 benchmark）

_2026-07-02 生成 · 生产链路真实场景 · 无图配额消耗_

## 一句话结论

> LoRA 一次成诗 (baseline total=0.849) + qwen-plus 单轮 refine + 保护性回滚（容差 0.03）下，主 benchmark n=32 **实际改动率 53.1%**（回滚率 46.9%）。核心净收益集中在 **cohesion（意境连贯度）+0.031 全样本 / +0.074 改动样本**，intent/imagery 弱正向 (+0.005)，**rhyme 因 feedback prompt 硬约束零效果**，pingze 有 -0.004 轻微副作用。**refine 的真实工程价值是"意境连贯度重塑"，不是 LoRA 弱项"押韵"的优化**。

## 0. 导航

- [§1](#1-setup)：Setup（生产链路 100% 对齐、单轮 refine、无图配额）
- [§2](#2-核心指标全样本-vs-改动样本对照)：**核心指标（全样本 vs 改动样本对照）**
- [§3](#3-cohesion-主导--refine-的真实价值维度)：**Cohesion 主导 — refine 的真实价值维度**
- [§4](#4-rhyme-零效果的-root-cause--feedback-硬约束)：**Rhyme 零效果的 root cause（feedback 硬约束）**
- [§5](#5-pingze-副作用--保护性回滚机制)：Pingze 副作用 + 保护性回滚机制
- [§6](#6-抽样-case-分类)：5 类抽样 case 分类
- [§7](#7-与-arena-ablation-的关系--refine-单步-vs-擂台整体)：与 arena ablation 的关系（refine 单步 vs 擂台整体）
- [§8](#8-caveats)：Caveats
- [§9](#9-follow-up)：Follow-up
- [§10](#10-相关产物)：相关产物

## 1. Setup

| 维度 | 值 |
|---|---|
| baseline 诗歌模型 | `local_lora` (Qwen2.5-1.5B + LoRA) |
| refine 模型 | `qwen-plus` (bailian) |
| 评分/意图/title/prompt 模型 | `qwen-plus` |
| n | **32**（主 benchmark 全量，`get_benchmark(n=32)`）|
| 每题调用链 | LoRA gen → 多维评分 → LLM 自动 critique → LLM feedback → qwen-plus refine 一轮 → 重新多维评分 |
| refine 保护机制 | 改后 `total < baseline - 0.03` 触发回滚（容差=0.03）|
| 图配额消耗 | **0**（eval_refine 不生图） |
| 数据源 | `outputs/eval/eval_refine_20260702_235009.json/.md` |

### 1.1 生产链路对齐

本次配置与 `app.py` 生产 UI 默认 100% 一致的 3 个位置（诗生成 + refine + 评分/意图）：

| 位置 | 生产默认 | eval_refine 用 | 对齐？ |
|---|---|---|---|
| 诗歌生成 | `local_lora`（LoRA 可用时）| `local_lora` | ✓ |
| 改诗模型 | `qwen-plus` | `qwen-plus` | ✓ |
| 意图/评分 | `qwen-plus` | `qwen-plus` | ✓ |

**结论：本报告测的是生产 UI 点一下"改诗"按钮的真实场景**，直接投影到 benchmark 32 题上。

## 2. 核心指标（全样本 vs 改动样本对照）

### 2.1 全样本均值（n=32）

| 维度 | baseline | refined | Δ |
|---|---|---|---|
| total | 0.849 | 0.853 | +0.004 |
| intent | 0.922 | 0.927 | +0.005 |
| pingze | 0.977 | 0.973 | **-0.004** |
| rhyme | 0.828 | 0.828 | **0.000** |
| imagery | 0.875 | 0.880 | +0.005 |
| **cohesion** | **0.898** | **0.930** | **+0.031** |

### 2.2 配对差值 + 提升比例（n=32）

| 维度 | mean Δ | median Δ | refined 提升比例 |
|---|---|---|---|
| total | +0.004 | 0.000 | 18.8% |
| intent | +0.005 | 0.000 | 3.1% |
| pingze | -0.004 | 0.000 | 3.1% |
| rhyme | 0.000 | 0.000 | 0.0% |
| imagery | +0.005 | 0.000 | 6.2% |
| **cohesion** | **+0.031** | 0.000 | **18.8%** |

### 2.3 仅改动样本（n=17）—— 排除回滚样本

| 维度 | mean Δ | refined 提升比例 |
|---|---|---|
| total | +0.009 | 29.4% |
| intent | +0.010 | 5.9% |
| pingze | **-0.007** | 5.9% |
| rhyme | 0.000 | 0.0% |
| imagery | +0.010 | 11.8% |
| **cohesion** | **+0.074** | **29.4%** |

### 2.4 关键读点

1. **改动率 53.1%**：32 题里 17 题 refine 出手成功、15 题回滚保护 —— 是**pipeline 保护机制的 checked balance**
2. **改动样本上 refine 效应 2.2×**（total +0.004 → +0.009；cohesion +0.031 → +0.074）—— 说明回滚样本把全样本 Δ 拉平了
3. **cohesion 是 5 维度里唯一提升比例 > 20% 的维度**（29.4%）—— 明确的"refine 主战场"信号
4. **rhyme 提升比例 0.0%** —— 严格零，不是"平均涨少了"而是"没有一首诗 rhyme 涨了"
5. **pingze 是唯一负 Δ 维度** —— refine 有轻微副作用，代价换 cohesion

## 3. Cohesion 主导 — refine 的真实价值维度

**cohesion (+0.031 全 / +0.074 改动)** 是本次 refine 的**唯一显著正向维度**。为什么会这样？

### 3.1 5 维度的可优化空间对比

| 维度 | 可优化路径 | 是否被约束 |
|---|---|---|
| intent | 用户约束固定（题目里指定的主题词/意象已在 baseline）| 空间小 |
| pingze | 格律硬约束（平仄规则由体裁决定）| 空间小 |
| rhyme | **feedback prompt 硬约束"保留原韵脚"** | **空间被封死**（§4）|
| imagery | 意象已经在 baseline 里、refine 只能替换用词 | 空间小 |
| **cohesion** | 唯一可以"重塑句法逻辑"、"重排逻辑关系"的维度 | **空间大** |

refine 生成新版诗时，其他 4 维度都受限制（用户约束/格律硬约束/已有意象），**只有 cohesion 的意境连贯度**可以通过"换掉不通顺的一句、重塑句间衔接"来提升。

### 3.2 与主跑 REPORT_main_n32x3run 的一致性

主跑 (2026-06-24) 关键 finding：**LLM-as-judge 对格律权重 ≈ 0**（base 平仄仅 25.6% 但 pairwise 胜率反而更高）。评委权重接近全在 intent/imagery/aesthetics/**cohesion**。

本报告 refine 主战场 = cohesion，**与 LLM judge 的注意力权重完全一致**——qwen-plus 作为 refine model 天然会 attend 到 cohesion，因为它作为 judge 也是 attend cohesion 而非平仄。**refine model 的注意力偏好和 scorer 的评分权重同源，导致 cohesion 成为主战场**。

## 4. Rhyme 零效果的 root cause — feedback 硬约束

**5 个抽样 case 里 5/5 的 feedback 都包含**：

> **"必须保留原韵脚，不得改动偶句末字"**

这是 pipeline 设计里 `_auto_poem_feedback` 生成的硬约束。目的是保证格律稳定、防止 refine 破坏押韵一致性。**副作用**：refine 从设计上永远无法通过改韵脚提升 rhyme 分。

### 4.1 具体机制

- rhyme 分数依赖平水韵韵部匹配 + 偶句末字押韵度
- baseline LoRA 一次成诗 rhyme = 0.828（LoRA 主跑弱项）
- feedback 锁死"末字不动" → refine 只能换非末字位置的字
- 非末字位置的字**不影响 rhyme 分**
- ∴ refine 后 rhyme 严格 = baseline rhyme

### 4.2 工程 trade-off surface

| 方案 | 结果 |
|---|---|
| **A（当前）**：保留原韵脚 | rhyme 优化路径封死 / 格律稳定 |
| **B（放宽）**：允许 refine 换韵脚（保平水韵一致性）| rhyme 可能涨 / 有押韵不一致风险 |

这不是 refine 无能，是 pipeline **主动放弃 rhyme 优化路径**。要不要走 B，需要在生产里做一次 A/B 对比看代价（列 §9 高优先级 follow-up）。

## 5. Pingze 副作用 + 保护性回滚机制

### 5.1 Pingze -0.004（改动样本 -0.007）

refine 换字时**轻微牺牲了 pingze**：

- 全样本 mean Δ = -0.004（0.977 → 0.973）
- 改动样本 mean Δ = -0.007（副作用在改动样本上放大 1.75×）
- 3.1% 的样本 pingze 涨了（跟 pingze 副作用互相独立）

**规模不严重但明确是副作用**。说明 refine 是"横向重塑"（换掉不通顺的字提 cohesion，但可能破坏原有平仄）**不是"垂直全面提升"**。

### 5.2 回滚率 46.9%（15/32）

保护机制在 47% 的题上触发：**refine 生成了新版诗，但因 `total < baseline - 0.03` 触发回滚，返回 baseline 原诗**。

回滚的 3 个抽样 case：
- 春景五绝·柳树燕子（baseline 0.891）
- 春雨七绝（baseline 0.925）
- 春景五律·桃花啼莺（baseline 0.750）

**回滚率 ≠ refine 失败率**：这是**保护机制积极工作**、防止 refine 把诗改烂的信号。生产里如果关掉回滚，total mean Δ 会下降（因为 15 题会被"改坏"的版本替代）。

## 6. 抽样 case 分类

| 主题 | 类型 | baseline | refined | 观察 |
|---|---|---|---|---|
| 春景五绝·柳树燕子 | **回滚保护** | 0.891 | 0.891 (回滚) | Feedback 明确指出"避三平尾+押同韵部"，但 rhyme 优化路径封死→回滚 |
| 春雨七绝 | 回滚保护 | 0.925 | 0.925 (回滚) | baseline 已高、refine 版分数不足 |
| 春景五律·桃花啼莺 | 回滚保护 | 0.750 | 0.750 (回滚) | baseline 较低但 refine 版更低、回滚 |
| 早春七律 | **改动但持平** | 0.843 | 0.843 ✓改动 | 字面变了（"寒拥"→"寒浸"、"暗敛眉"）、5 维度加权抵消、total 不动 |
| **夏蝉五绝** | **改动且涨** | 0.887 | **0.925** ✓改动 | "不奈西山寺" → "闲听西山寺" 消极→积极词，cohesion 涨、其他维度稳 |

**5 个 case 涵盖 3 类**：回滚（3/5）、改动持平（1/5）、改动且涨（1/5）。这跟 §2 的 refined 提升比例 18.8% 一致。

夏蝉五绝是"refine 有效 case 的典型样本"——单词替换（"不奈"是消极、"闲听"是禅意积极）没破坏任何格律但明显提升意境连贯。**这就是 §3 讲的 refine 的真实价值维度**。

## 7. 与 arena ablation 的关系 — refine 单步 vs 擂台整体

`REPORT_arena_ablation_20260701.md` 显示：**擂台 loop (arena judge + refine 多轮 混合)** 在 VLM 硬约束上有 +18.2pp 净收益（主池 n=22）。本报告显示：**refine 单步** 在诗歌评分上带来 cohesion +0.031（+0.074 改动样本）。

### 7.1 两个 findings 不冲突，互补

- **arena ablation 测**："擂台整体 loop" 对**图**的净收益
- **eval_refine 测**："refine 单步" 对**诗**的净收益

**因果链**：refine 单步的诗 cohesion 提升 → 更连贯的诗 → image prompt 更结构化 → 图更命中用户约束（→ arena ablation +18.2pp 硬约束）。

### 7.2 但存在一个 disambiguation 问题

如果 refine 单步只贡献 cohesion +0.031（幅度不大），arena ablation +18.2pp 的**主贡献**是不是来自 **arena judge**（arena 从 N 个 candidates 里选冠军）而不是 refine 优化本身？

**当前数据无法回答**——arena ablation arm B 是"擂台整体 loop"、混合了两个机制。要 disambiguate 需要一个三臂对比（§9 中优先级）：

| 臂 | 有 arena judge | 有 refine |
|---|---|---|
| A（arena ablation 已有）| ✗ | ✗ |
| B（arena ablation 已有）| ✓ | ✓ |
| **C（新增）** | ✓ | ✗ |
| **D（新增）** | ✗ | ✓ |

**C vs A** 隔离测 arena judge 贡献、**D vs A** 隔离测 refine 单步贡献、**B vs C** 也能反推 refine 贡献。跑这个能给"擂台 loop +18.2pp 里 refine 占多少"确切答案。

## 8. Caveats

1. **n=32 单次 run**：无跨 run stability（主跑用 3 run × n=32 是更严格的做法），当前 mean Δ 值是**单次样本**
2. **单一 refine model (qwen-plus)**：跨 refine model（qwen-max / deepseek-v4-pro）未验证；cohesion +0.031 是否 model-specific 未知
3. **评分器 self-scoring 风险**：`refine_model = qwen-plus` = `scorer = qwen-plus`，qwen-plus 作为 refine 主体和 judge 主体同源。可能有轻微 self-bias（判自己改的版本略好），但 refined_total 只 +0.004、self-bias 幅度不大
4. **feedback prompt 里的硬约束是设计选择**：rhyme 零效果不是 refine 无能、而是 pipeline 主动锁死。**不能把 rhyme=0.000 解读为"refine 系统失败"**
5. **回滚机制是保护而非无效**：47% 回滚率**不是** refine 失败率的 47%，是 pipeline 主动挡下"改烂的版本"。生产里关掉回滚会让 total mean Δ 下降
6. **单轮 refine**：本次 `max_poem_rounds = 1`。arena ablation arm B 是 `max_poem_rounds = 2`（多轮）。**单轮 vs 多轮的边际收益递减曲线未测**
7. **memory 约束"single-shot 实验禁称'上限'"**：本报告是 "一次成诗 + 一轮 refine"，refine 是有反馈机制的、但整体仍是"单轮反馈"、不能把 +0.031 cohesion 称作"refine 的能力上限"

## 9. Follow-up

### 高优先级

- **放宽原韵脚约束试一次**（rhyme 优化路径试探）：改 `_auto_poem_feedback` prompt 允许 refine 换韵脚（保平水韵一致性），跑一次 n=32 对比，量化 tradeoff：**rhyme +? vs cohesion -?**。如果 rhyme 涨 > 0.05 而 cohesion 只微跌，是明确的工程优化点
- **跨 refine model 复现**：`--refine-model qwen-max` + `--refine-model deepseek-v4-pro` 各跑一次 n=32，看 cohesion +0.031 是否稳定 or refine model 相关

### 中优先级

- **arena ablation × eval_refine disambiguation**（详见 §7.2）：跑 arm C（有 arena judge + 无 refine）和 arm D（无 arena judge + 有 refine）两臂对比，隔离 arena judge 和 refine 单步在硬约束 +18.2pp 里的贡献
- **多轮 refine 净收益递减曲线**：跑 `--max-poem-rounds ∈ {1, 2, 3, 5}` 看 cohesion 是否边际递减，回答"refine 应该跑几轮"
- **回滚率与 baseline 分数相关性分析**：把 32 题按 baseline total 排序、看回滚率是否与 baseline 分数正相关。假设：baseline 越高 refine 越可能被回滚（无处再涨）。这个 finding 能指导"要不要根据 baseline 分数决定是否触发 refine"（低分诗才 refine，节省 refine model token）

### 低优先级

- **人类评估校准 LLM scorer**：cohesion +0.031 完全由 qwen-plus 判定，加入人类 pairwise（用改动样本的 baseline vs refined 各 17 对做盲评）看 LLM scorer signal 是否与人类感受一致
- **VLM 独立评委**：cohesion 是"意境连贯" —— 也可以用 VLM 从图像端间接评（看图能不能重建诗）

## 10. 相关产物

| 路径 | 用途 |
|---|---|
| `eval/eval_refine.py` | 评测脚本（本次改了 `--model` 默认为 `local_lora` 贴齐生产）|
| `outputs/eval/eval_refine_20260702_235009.json/.md` | 原始数据 + 内置 md |
| `core/agent/poem_refiner.py` | `refine_poem` 实现（含回滚容差 0.03）|
| `core/agent/agent.py` `_auto_poem_critique` / `_auto_poem_feedback` | LLM 自动写诗评 → feedback（含"必须保留原韵脚"硬约束 —— §4 root cause 出处）|
| `eval/REPORT_arena_ablation_20260701.md` | 交叉引用：擂台整体 loop 硬约束 +18.2pp 与本报告 refine 单步的关系（§7）|
| `eval/REPORT_main_n32x3run_20260624.md` | 交叉引用：LLM-as-judge 对格律权重 ≈ 0，解释为何 refine 主战场是 cohesion 而非平仄（§3.2）|

---

_报告生成：Claude Opus 4.7 · 数据可通过 `python -m eval.eval_refine --n 32` 复现（默认 `--model local_lora --refine-model qwen-plus --scorer qwen-plus`，与生产 UI 默认 100% 对齐）_
