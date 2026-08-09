# eval_clip 报告 · 双锚点 CLIP vs 单锚点对齐分（n=15 千古名诗 baseline）

_2026-06-23 数据落盘 · 2026-07-02 入库_

## 一句话结论

> 生产 dual (α=0.6) 与 VLM oracle 的 Spearman ρ = **+0.365**，点估计高于 prompt_only (+0.301)、稍低于 poem_only (+0.384)——注意 n=15 时 Spearman ρ 的标准误约 0.29，这些差值均**未达统计显著**，只是方向一致的排序信号；α grid search 表明最优 α=0.8 处 ρ = **+0.414**，超过任一单 anchor 上限。**dual 设计在千古名诗语料上被数据支持，但生产权重 α 仍有 +0.049 优化空间**。均值层 mean(dual) 与 mean(single) 差距 ≤ 0.003（配对胜率 53.3%）——**信号在排序相关性维度、不在数值均值维度**。

> ⚠️ **2026-08-09 补充（§4.1）**：在 4 个 backend 的同源 LoRA 生诗（n=10）上复现时，**最优 α 全部落在端点（0.0×3 / 1.0×1），未复现名诗组的 α=0.8**，且两 anchor 同质度明显更高（+0.430~+0.867 vs 名诗 +0.304）。因此上述「dual 设计被支持」的结论**限定于名诗语料**；在更接近生产真实输入的 LoRA 生诗上，dual 融合退化为单 anchor。该结果加固了 §4「不要直接把 α 改到 0.8」的工程决定。

## 0. 导航

- [§1](#1-setup)：Setup（数据集、backend、三种 anchor 策略、生产权重）
- [§2](#2-vlm-oracle-相关性核心指标)：**VLM oracle Spearman ρ**（核心指标）
- [§3](#3-均值与配对差值)：均值与配对差值（dual − prompt_only）
- [§4](#4-α-grid-search-与最优权重建议)：**α grid search + 最优 α=0.8 建议**；§4.1 **跨 backend 复现（2026-08-09 补，结论为负）**
- [§5](#5-anchor-独立性--融合动机验证)：anchor 间独立性（dual 融合的 SNR 提升动机）
- [§6](#6-密度分层-caveat)：密度分层（sparse 数据空）
- [§7](#7-三个代表性样本单张观察)：代表性单张观察（3 个 case）
- [§8](#8-caveats)：Caveats
- [§9](#9-follow-up)：Follow-up
- [§10](#10-相关产物)：相关产物

## 1. Setup

| 维度 | 值 |
|---|---|
| 数据集 | 千古名诗 benchmark（15 首）· S 级 5 首国民 + A 级 9 首常见名诗 + B 级 1 首相对冷门 |
| image backend | `bailian:qwen-image-2.0-pro`（single-shot：无 refine loop、无多 seed、无 human-in-loop）|
| 关键词 / 起名 | `qwen-plus` |
| 英文 prompt | `qwen-max` |
| 风格后缀 | 水墨画（Song Dynasty style）|
| VLM oracle | `qwen-vl-max` |
| 数据源 | `outputs/eval/eval_clip_20260623_000437.json` (10) + `eval_clip_20260623_001810.json` (5) |
| α 分析 | `outputs/eval/analyze_dual_20260623_201449.md` |
| 15 首名诗基线原文 | `outputs/eval/REPORT_classics_15.md`（未入库、backend baseline 定位）|

### 1.1 三种 anchor 策略

对同一张图，同时用三套锚点算 CLIP raw cosine 相似度：

| 策略 | 定义 |
|---|---|
| **prompt_only** | 仅英文 prompt 锚点，`score_raw_cosine(image, prompt)` |
| **poem_only** | 仅诗歌视觉关键词锚点，`score_raw_cosine(image, visual_keywords_en)` |
| **dual (production)** | 加权融合 `α · poem + (1−α) · prompt`；rich 分支 α = `CLIP_POEM_WEIGHT` = **0.6**，sparse 分支 α = `CLIP_SPARSE_POEM_WEIGHT`（关键词词数 < `CLIP_SPARSE_WORD_THRESHOLD` 时切换）|

**dual 是生产 pipeline 实际使用的评分器**（`config.py` 里的权重）。本报告的核心问题：dual 相对 single anchor 到底带来多少对齐提升？

### 1.2 为什么用"千古名诗"而非主 benchmark

InkVerse 项目此前反复怀疑："dual CLIP 与 VLM ρ ≈ 0.10 长期不高，是不是 LoRA 生的诗本身意象残缺、'诗 → prompt → 图' 链条断在第一棒？"

千古名诗 benchmark 的意图是把"诗质量"钉死成上限（李白/杜甫/王维），单变量测 dual anchor 设计本身有没有价值。**本报告不是 backend baseline 报告**（那是 `REPORT_classics_15.md` 的定位），而是 **eval_clip 主报告** —— 关注 dual vs single 提升。

## 2. VLM oracle 相关性（核心指标）

**Spearman ρ = "该 CLIP 策略对图文契合度的排序与 VLM ground truth 一致程度"**。理想情况：dual > single anchor。

| CLIP 策略 | n | Spearman ρ vs VLM | Pearson r vs VLM |
|---|---|---|---|
| prompt_only | 15 | +0.301 | +0.197 |
| poem_only | 15 | **+0.384** | **+0.442** |
| **dual (production α=0.6)** | 15 | **+0.365** | +0.424 |

**关键读点**：

1. **dual (+0.365) 点估计优于 prompt_only (+0.301)**——差 +0.064，是本次 ablation 里方向最一致的信号；但 n=15 下 Spearman ρ 标准误约 0.29，该差值**未达统计显著**，定位为"方向性证据、待 n=30 复现"（§9 follow-up）
2. **dual (+0.365) 稍低于 poem_only (+0.384)**——差 -0.019，提示生产 α=0.6 权重可能未调优（§4 grid search 会验证；同样受 n=15 限制）
3. **对比历史 baseline**：此前 LoRA 生诗上 dual ρ ≈ 0.10。切换到千古名诗后 ρ 从 0.10 → 0.365，**3.6× 提升**——**dual 设计本身没问题，问题在诗质量（LoRA 是评测瓶颈，不是 backend）**

## 3. 均值与配对差值

CLIP raw cosine ∈ [−1, 1]，实践中中文诗歌 + 水墨图基本落在 [0.24, 0.35]。

### 3.1 三策略均值

| 策略 | n | mean | std | median | min | max |
|---|---|---|---|---|---|---|
| prompt_only | 15 | 0.293 | ~0.02 | ~0.29 | ~0.25 | ~0.33 |
| poem_only | 15 | 0.296 | ~0.02 | ~0.29 | ~0.25 | ~0.33 |
| dual | 15 | 0.295 | ~0.02 | ~0.29 | ~0.26 | ~0.33 |

**均值层三策略差 ≤ 0.003**——落在噪声内。仅看 mean 无法判定 dual 是否更优。

### 3.2 配对差值 dual − prompt_only

同一张图上做配对差，消除主题波动噪声：

| 指标 | 值 |
|---|---|
| 样本对数 n | 15 |
| **mean Δ** | **+0.0018** |
| median Δ | +0.0006 |
| **dual 胜率**（Δ > 0 的比例）| **53.3%** |

**读点**：mean Δ 仅 +0.002、胜率 刚过 50%——**均值维度信号非常弱**。此前 LoRA 生诗上胜率 10-40%，本次 53.3% 是项目史上首次 > 50%。但**判定 dual 是否有效不能靠 mean Δ**——**核心信号在 §2 Spearman ρ 维度**。

## 4. α grid search 与最优权重建议

生产权重 α=0.6 是初期直觉设定，从未做过 grid 优化。用 `analyze_clip_dual.py` 在 α ∈ {0.0, 0.1, ..., 1.0} 上扫描：

| α | dual_α ρ vs VLM | 相对 max(单) Δ |
|---|---|---|
| 0.0 (=prompt_only) | +0.301 | -0.083 |
| 0.1 | +0.301 | -0.083 |
| 0.2 | +0.323 | -0.061 |
| 0.3 | +0.331 | -0.053 |
| 0.4 | +0.341 | -0.044 |
| 0.5 | +0.384 | +0.000 |
| **0.6 (production)** | **+0.365** | **-0.019** |
| 0.7 | +0.405 | +0.021 |
| **0.8** ⭐ | **+0.414** | **+0.030** |
| 0.9 | +0.392 | +0.008 |
| 1.0 (=poem_only) | +0.384 | +0.000 |

### 结论判定

- **最优 α = 0.8**（内部最优）、ρ = **+0.414** ⭐ —— 点估计 > max(单) = +0.384；注意这是在 n=15 上对 11 个 α 取 max，存在多重比较放大噪声的风险，须经 §9 的 n=30 复现确认
- α = 0.8 ∈ (0, 1) 且 ρ_dual > ρ_max_single → **dual 融合数学上有效**（不是"两个 anchor 相加"式的伪提升）
- 生产 α = 0.6 处于**次优**（-0.019 vs poem_only、-0.049 vs 最优 α=0.8）

### 工程建议

**⚠ 不建议直接改 `config.py:CLIP_POEM_WEIGHT` 到 0.8**，原因：

1. n=15 全 rich 数据，泛化到 sparse benchmark 未验证
2. 数据源单一 backend（qwen-image-2.0-pro），跨 backend 未验证
3. 千古名诗数据分布特殊（详见 §8 caveat）

**推荐路径**：先跑 §9 提到的 sparse 分层复现 + n=30 稳定性验证，两个 sanity check 通过后再调 config.py 权重。

### 4.1 跨 backend 复现：α 最优点不稳定（2026-08-09 补）

执行 §9 中优先级的「跨 backend 复现」。**零 API 消耗**——全部复用已落盘的 `eval_clip` 原始结果。

**⚠ 先澄清数据集口径。** §9 原文设想的是「三 backend × 同 15 首**名诗**」，**该实验无法执行**：千古名诗 benchmark 只在 `qwen-image-2.0-pro` 上跑过，其他 backend 不存在名诗数据。实际可用的同源组是另一批——**4 个 backend 共享同一批 10 首 LoRA 生诗**（均 `reuse_poems_from: eval_clip_20260622_133658.json`，prompt 模型与 VLM oracle 一致，唯一变量为 image backend）。故：4 个 LoRA 组**相互之间**对照干净；LoRA 组 **vs 名诗组**混杂了诗质量，只能作趋势参照。

| image backend | 语料 | n | prompt_only | poem_only | dual(α=0.6) | 最优 α | 最优 ρ | ρ(poem,prompt) |
|---|---|---|---|---|---|---|---|---|
| `wanx2.1-t2i-turbo` | LoRA 生诗 | 10 | +0.118 | −0.042 | +0.021 | **0.0** | +0.118 | +0.430 |
| `qwen-image-max` | LoRA 生诗 | 10 | +0.254 | −0.110 | +0.103 | **0.0** | +0.254 | +0.624 |
| `qwen-image-2.0-pro` | LoRA 生诗 | 10 | +0.216 | +0.374 | +0.308 | **1.0** | +0.374 | +0.867 |
| `qwen-image-2.0-pro-2026-04-22` | LoRA 生诗 | 10 | +0.418 | +0.075 | +0.418 | **0.0** | +0.418 | +0.442 |
| `qwen-image-2.0-pro`（§4 基线） | 千古名诗 | 15 | +0.301 | +0.384 | +0.365 | **0.8** | +0.414 | +0.304 |

**四个发现：**

1. **4 个 backend 的最优 α 全部落在端点，无一落在开区间内**（3 组 α=0.0、1 组 α=1.0）。按 §4 自身的判定标准（「最优 α 在端点 → dual 退化为单 anchor」），**dual 融合在这批 LoRA 生诗上未获支持**，与名诗组 α=0.8 的结果相反。
2. **端点方向不一致**——若结论是 prompt anchor 系统性更优，四组应一致退化到 α=0，但 `qwen-image-2.0-pro` 反向退化到 α=1.0。`poem_only` 的 ρ 在**语料完全相同、仅换 backend** 的四组间从 −0.110 摆到 +0.374（跨度 0.484），提示诗锚点在 LoRA 生诗上的信号强度接近噪声量级。
3. **两 anchor 的同质度显著高于名诗组**（LoRA 组 +0.430 ~ +0.867 vs 名诗组 +0.304），其中 +0.867 已进入 §5 判定表的「融合数学上无用」区间。这在机制上解释了退化：名诗意象密集、视觉可辨识度高，诗锚点能提供独立于提示词的信息；LoRA 生诗的视觉关键词贫乏，提取出的锚点与提示词高度重叠。
4. **生产 α=0.6 在四组中无一跑赢 max(单 anchor)**（差值 −0.097 / −0.151 / −0.066 / ±0.000）。

**⚠ 统计边界：** n=10 时 Spearman ρ 标准误约 `1/√(n−3) = 0.378`，**本节全部 ρ 值及其组间差异均未达统计显著**。因此可以说的是「在 4 个 backend 的同源 LoRA 生诗上**未能复现**名诗组的开区间最优点，且最优 α 的位置与方向在组间不稳定」；**不能**说「双锚点设计无效」或「prompt_only 优于 dual」——样本量不足以支撑任何一侧的强结论。

**对 §4 工程建议的影响：加固「不要直接把 `CLIP_POEM_WEIGHT` 改到 0.8」的决定。** 现在有了直接证据——α 最优点不跨语料/backend 稳定，且在最接近生产真实输入的 LoRA 生诗上跑到了端点。同时这为 §8 caveat 5「名诗结论不能外推」提供了量化支撑：不只是 ρ 绝对值低，连 α 的最优结构本身都不同。

_数据与完整 α grid：`outputs/eval/analyze_dual_cross_backend_20260809.md` · 复现：`python -m eval.analyze_clip_dual --input outputs/eval/eval_clip_20260622_<ts>.json`（四组分别跑，勿合并——同一批诗在不同 backend 上不独立）_

## 5. anchor 独立性 → 融合动机验证

dual 融合"数学上有意义"的前提：poem anchor 和 prompt anchor 的 noise 相对独立。用 poem_only 和 prompt_only 分数间的相关性度量独立性：

| 度量 | ρ(poem_only, prompt_only) |
|---|---|
| Spearman | +0.304 |
| Pearson | +0.345 |

**判定表**：

| 相关性 | 含义 | 本次 |
|---|---|---|
| > 0.8 | 两 anchor 高度同质，融合数学上无用 | ✗ |
| 0.5 ~ 0.8 | 中等相关，融合可能轻微降 noise | ✗ |
| **< 0.5** | 互补性强、融合最有 motivation（noise 独立 → SNR 提升）| ✓ **+0.304** |

**结论**：poem 与 prompt 相关性 < 0.5 → dual 融合的 SNR 提升动机成立，与 §4 α grid 结果一致（最优 α ∈ (0, 1) 处）。

## 6. 密度分层 caveat

n=15 千古名诗 `keyword_word_count` 分布：

```
20260623_000437 (10 首): wc ∈ [12, 19], 全 rich
20260623_001810 (5 首):  wc ∈ [17, 32], 全 rich
```

**15 首中 0 首 sparse** —— 名诗普遍意象密集（每首都有 12+ 视觉关键词），与主 benchmark 32 题的"sparse 只给抽象主题词"完全不同分布。

**本报告的结论只适用于 rich 题**。sparse 分支的 dual 权重 `CLIP_SPARSE_POEM_WEIGHT` 和 `CLIP_SPARSE_WORD_THRESHOLD` **本次未验证** —— 列为 §9 高优 follow-up。

## 7. 三个代表性样本（单张观察）

抽 3 张图暴露 dual anchor 设计的 signal 与盲区：

| 7.1 江雪 · dual 与 VLM 同向 | 7.2 使至塞上 · CLIP 维度盲区 | 7.3 九月九日 · single-shot 局限 |
|---|---|---|
| ![江雪](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/clip_classics_20260623/%E6%B1%9F%E9%9B%AA_dual033_vlm095.jpg) | ![使至塞上](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/clip_classics_20260623/%E4%BD%BF%E8%87%B3%E5%A1%9E%E4%B8%8A_dual028_vlm085.jpg) | ![九月九日](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/clip_classics_20260623/%E4%B9%9D%E6%9C%88%E4%B9%9D%E6%97%A5_dual029_vlm065.jpg) |
| VLM 0.95 · dual 0.33 | VLM 0.85 · dual **0.28** | VLM **0.65** · dual 0.29 |

### 7.1 江雪（VLM 0.95 · dual 0.33）—— dual 与 VLM 同向

柳宗元《江雪》图：渔翁孤舟 + 雪江远山。VLM 极高分（0.95）、dual 也是全 15 首最高之一（0.33）。**当图片语义命中充分时，dual CLIP 与 VLM 判断一致**。

### 7.2 使至塞上（VLM 0.85 · dual 0.28）—— CLIP 维度盲区

王维《使至塞上》图：单车 ✓ + 孤烟 ✓ + 长河 ✓ + 落日 ✓（4 anchor 全命中，见上图中列——垂直烟柱正是"大漠孤烟直"）。但 dual **仅 0.28**，比其他高 VLM 图（江雪 0.33 / 瀑布 0.32）偏低。

> **CLIP 不识"孤烟直"的诗意，只识表面色彩饱和度；VLM 才识古典意象命中度**。这是 dual anchor 设计的**内在天花板**——CLIP encoder 语义分辨力有限，不是 α 权重能修的。

### 7.3 九月九日忆山东兄弟（VLM 0.65 · dual 0.29）—— single-shot 局限

王维《九月九日》图：仅一支带果枝条（上图右列），缺"兄弟登高插茱萸"里的所有人物/动作元素。**VLM 判 0.65（低）、dual 判 0.29（普通）**——两个都识别出了"这张图不太合适"，但**具体是"缺什么"的诊断力不同**（VLM 能说出缺人物，dual 只是个分数）。

> **single-shot 一次生图对人物/动作题材命中率 0/5**（`REPORT_classics_15.md` §4 A finding）。这**不**是 backend 上限——refine loop / 多 seed / human-in-loop 未启用。

**这 3 个 case 汇总**：

| 图 | VLM 语义 | dual 排序 | dual 值 | 说明 |
|---|---|---|---|---|
| 江雪 | 高 (0.95) | top (0.33) | 高 | 语义 + CLIP 都对上 → dual 有效 |
| 使至塞上 | 高 (0.85) | 低 (0.28) | 反常 | **CLIP 维度盲区**（诗意 vs 饱和度）|
| 九月九日 | 低 (0.65) | 普通 (0.29) | 一致 | **single-shot 局限**（人物题材 miss）|

## 8. Caveats

1. **n=15**：样本量小，α = 0.8 是否稳定需 n=30/50 复现
2. **全 rich**：sparse 分层数据 0，`CLIP_SPARSE_*` 权重完全未验证
3. **single-shot 一次生图**：无 refine loop / 无多 seed / 无 human-in-loop——**§7.3 的"人物题材命中率 0"只是 single-shot 表现，非 backend 上限**（memory 里明确记录："single-shot 实验禁称'上限'"）
4. ~~**单 backend**（qwen-image-2.0-pro）：跨 backend 稳定性未验证~~ **已部分补测（2026-08-09，见 §4.1）**：raw 数据里实际有 **4 组**同源 n=10 数据（wanx2.1-t2i-turbo / qwen-image-max / qwen-image-2.0-pro / 2.0-pro-2026-04-22，共享同一批 LoRA 生诗），已完成 α 分析。**结果为负**：四组最优 α 全部落在端点，未复现名诗组的 α=0.8。但**名诗 benchmark 本身仍只有单 backend 数据**——「同一批名诗跨 backend 是否稳定」这一问题尚未回答，需补跑名诗 × 其他 backend（消耗图像配额）
5. **千古名诗数据分布特殊**：意象密集 + 视觉可辨识度高，可能上偏 VLM ρ；不能直接外推到 LoRA 生诗上。**dual ρ 从 0.10 → 0.365 的 3.6× 提升，是"诗质量上限"带来的**——LoRA 生诗上的 dual ρ 仍是长期未解痛点
6. **VLM 作弊未观察到**：S 级国民诗 mean 0.79 vs A 级常见名诗 mean 0.82 反向，若 VLM 因看过原文虚高 S 级则会 S > A。数据支持 VLM 未作弊
7. **CLIP encoder 内在盲区**（§7.2）：dual anchor 的天花板受 CLIP 语义分辨力限制，α 调优只能优化 rank，无法解决"孤烟直"这类古典意象的语义缺口。修法只有换 CLIP encoder 或额外套一层 VLM gate

## 9. Follow-up

### 高优先级

- **Sparse 分层复现**：用主 benchmark 32 题的 sparse 16 首（`get_benchmark(n=16, density='sparse')`）跑一次 eval_clip，验证 `CLIP_SPARSE_*` 权重是否合理、α 最优点是否与 rich 一致
- **n=30 稳定性验证**：把千古名诗 benchmark 扩到 30 首（新增 15 首常见但非国民级名诗），验证最优 α=0.8 是否稳定
- **α=0.8 vs α=0.6 生产回归测**：如果 §9 前两条稳定，改 `config.py:CLIP_POEM_WEIGHT` 0.6 → 0.8 后在主 benchmark 上重跑一次 eval_clip 验证生产链路未回归

### 中优先级

- ✅ ~~**跨 backend 复现**~~ **已执行（2026-08-09，见 §4.1）**，但**原设想不可行**：「三 backend × 同 15 首名诗」无法做——名诗 benchmark 只在一个 backend 上跑过。实际改用 4 组同源 LoRA 生诗（n=10）完成，结论为负（最优 α 全部落端点、方向不一致）。**遗留**：若仍要回答「名诗上最优 α 是否跨 backend 稳定」，须补跑名诗 benchmark × 其他 backend，需消耗图像配额
- **CLIP 语义盲区独立测**：造对照 pair「语义命中 + CLIP 低」vs「语义 miss + CLIP 高」，定量测 CLIP 分辨力，为"何时用 VLM gate 补 CLIP"提供数据支持
- **人物/动作题材专项**：`REPORT_classics_15.md` §4 A 已 flag 五张 single-shot 失败图（九月九日 / 登高 / 春晓 / 凉州词 / 清明），refine loop / 多 seed 是否能把命中率从 0/5 拉到 2-3/5

### 低优先级

- **VLM oracle 多样化**：目前只有 qwen-vl-max，补 glm-4v / gpt-4v 交叉验证 Spearman ρ 的绝对值可信度

## 10. 相关产物

| 路径 | 用途 |
|---|---|
| `eval/eval_clip.py` | 评测脚本 |
| `eval/analyze_clip_dual.py` | α grid search + 相关性分析工具 |
| `eval/build_classics_benchmark.py` | 千古名诗 benchmark 构建脚本 |
| `outputs/eval/benchmark_classics.json` | 15 首诗 + visual_keywords + prompt（评测输入）|
| `outputs/eval/eval_clip_20260623_000437.json/.md` | 前 10 首 eval 原始结果 |
| `outputs/eval/eval_clip_20260623_001810.json/.md` | 后 5 首 eval 原始结果（同一 config 补跑）|
| `outputs/eval/analyze_dual_20260623_201449.md` | α grid search 原始输出（名诗组，§4）|
| `outputs/eval/analyze_dual_cross_backend_20260809.md` | **跨 backend 复现汇总（§4.1）**：5 组完整 α grid + 结论 + 统计边界 |
| `outputs/eval/eval_clip_20260622_133658.json` | LoRA 生诗 10 首基准组（`wanx2.1-t2i-turbo`），另 3 组均 reuse 此文件的诗 |
| `outputs/eval/eval_clip_20260622_210101.json` | 同批诗 · `qwen-image-max` |
| `outputs/eval/eval_clip_20260622_211500.json` | 同批诗 · `qwen-image-2.0-pro` |
| `outputs/eval/eval_clip_20260622_213250.json` | 同批诗 · `qwen-image-2.0-pro-2026-04-22` |
| `outputs/eval/analyze_dual_20260809_2216*.md` | §4.1 三组的单组原始输出（`analyze_clip_dual.py` 直出；`qwen-image-max` 那次因同秒时间戳被覆盖，完整数据见上方汇总文件）|
| `outputs/eval/clip_img_20260622_235822_bailian-qwenimage2/` | 前 10 张水墨图 |
| `outputs/eval/clip_img_20260623_001405_bailian-qwenimage2/` | 后 5 张水墨图 |
| `outputs/eval/REPORT_classics_15.md` | 15 首名诗 backend baseline 报告（未入库、和本报告互补：本报告 focus dual vs single，那份 focus backend single-shot 基线——注意是一次生图、无反馈机制的基线表现，不是 backend 能力上限）|

---

_报告生成：Claude Opus 4.7 · 数据可通过 `python -m eval.eval_clip --reuse-poems-from outputs/eval/benchmark_classics.json --vlm-judge qwen-vl-max` 复现 · α grid 分析 `python -m eval.analyze_clip_dual outputs/eval/eval_clip_20260623_000437.json outputs/eval/eval_clip_20260623_001810.json`_
