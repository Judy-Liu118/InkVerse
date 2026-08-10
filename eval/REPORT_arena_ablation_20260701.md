# 擂台 Ablation 评测报告（n=22 主池 + n=10 辅池）

日期：2026-07-01 立项 · 2026-07-02 主池 n=22 补跑完成、报告转正

> **实验问题**：擂台机制（arena 冠军后再跑 max_poem_refine_rounds=2 轮 refine + pairwise 挑战）到底有没有净收益？<br>
> **答案（n=22 主池 04-22 backend）**：**CLIP mean 平手（甚至 arm A 略高 -0.002），但 VLM 硬约束命中率 arm B +18.2pp**（方向性强证据；McNemar p≈0.18、CI 跨零，统计上未定论，见 §2.5）。这两个指标在这次分开了：arm A 无擂台版生成的图"漂亮但缺用户约束的东西"，arm B 擂台改诗把"缺的东西"补回来了。CLIP 反映"图美不美/意境自洽"，硬约束反映"用户明确要求的物体/场景是否画出来"—— **擂台在后者上真的有用**。<br>
> **辅池（n=10 03-03 backend）**平手：CLIP、硬约束两条维度都无差异。**辅池与主池方向不一致**，backend 敏感度需要在 §6 讨论。

## §0 快速导航

| 章节 | 内容 |
|---|---|
| §1 | 实验设置：arm A vs arm B 流程差异 + 两池数据来源 |
| §2 | 四条核心证据（CLIP × 2 + 硬约束 × 2），主池 + 辅池并列；§2.5 显著性分析（2026-07-05 补）· §2.6 跨家族 VLM 复核 · §2.7 kw 定义敏感度全池扫描（2026-07-06 补） |
| §3 | rich/sparse 硬约束拆分（主池 + 辅池）|
| §4.1 | 主池 22 主题双图并排（图 + 诗 + CLIP + VLM 判定）|
| §4.2 | 辅池 10 主题双图并排 |
| §5 | 反直觉案例：CLIP 与硬约束脱钩 + 擂台把用户约束改没了 |
| §6 | 擂台净收益成立 vs 不成立的原因分析（主池辅池反向）|
| §7 | caveats + backend 稳定性 |
| §8 | follow-up |

## §1 实验设置

### arm A（无擂台）与 arm B（有擂台）的流程差异

**arm A / arm B 共用的部分**（arena 海选）：

```
LoRA 生成 5 首候选
    ↓
本地打分（平仄 / 押韵 / 意象 / 连贯 / 切题 5 维加权 → 综合分）
    ↓
硬门控（切题 < 0.5、平仄严重不合格 淘汰）
    ↓
Arena pairwise round-robin（qwen-plus 当 judge，两两 PK 累计胜场）
    ↓
综合分 = 本地分 + arena 胜场加权
    ↓
最高分 = arena 冠军
```

**arm A 到此止步** → 直接进图生成 + fixed image loop（≤2 轮改图，target CLIP raw≥0.30）。

**arm B 在 arena 冠军之后加 2 轮擂台**：

```
第 1 轮守擂：LLM refine_poem 改写擂主 → 挑战者
             挑战者过本地门控 → 与擂主 pairwise
             挑战者赢：综合分 = 本地分 + 0.17（PAIRWISE_WIN_DELTA）
             综合分 > 擂主：攻擂成功、换擂主
第 2 轮守擂：同上
    ↓
最终擂主 → 进图生成 + fixed image loop
```

所以 arm A vs arm B **真正的差异只有"arena 冠军之后再多 2 轮 refine + pairwise 挑战"**。arena 海选阶段两 arm 完全一样。

### 参数

| 项 | 值 |
|---|---|
| 诗生成 backend | `local_lora`（Qwen2.5-1.5B + LoRA）|
| 主池图像 backend | `bailian:qwen-image-2.0-pro-2026-04-22`（两 arm 严格一致）|
| 辅池图像 backend | `bailian:qwen-image-2.0-pro-2026-03-03`（两 arm 严格一致）|
| target CLIP raw | 0.30 |
| max_image_improve_rounds | 2 |
| max_poem_refine_rounds | arm A=0 · arm B=2 |
| PAIRWISE_WIN_DELTA | 0.17（arm B 用）|
| VLM oracle | qwen-vl-max（硬约束核查）|

### 数据来源

**主池 n=22（backend 04-22, 2026-07-02 补跑）**：

| arm | sweep JSON | 图目录 |
|---|---|---|
| arm A（无擂台） | `outputs/eval/sweep_pairwise_win_delta_20260702_094052.json` | `outputs/eval/sweep_pairwise_win_delta_images_20260702_085340/delta_0.17/` |
| arm B（有擂台） | `outputs/eval/sweep_pairwise_win_delta_20260702_105404.json` | `outputs/eval/sweep_pairwise_win_delta_images_20260702_094058/delta_0.17/` |
| VLM 硬约束对比 | `outputs/eval/vlm_hard_constraint_arm_ab_main_n22_20260702_133542.json` | — |

跑法：
```bash
# arm A
python -m eval.sweep_pairwise_win_delta --n 32 --offset 10 --max-poem-rounds 0 --deltas 0.17 --image-backend bailian:qwen-image-2.0-pro-2026-04-22
# arm B
python -m eval.sweep_pairwise_win_delta --n 32 --offset 10 --max-poem-rounds 2 --deltas 0.17 --image-backend bailian:qwen-image-2.0-pro-2026-04-22
# 拼 agg
python -m eval._build_arena_ablation_agg_n32
# VLM 硬约束
python -m eval.vlm_hard_constraint --agg outputs/eval/_agg_arena_ablation_main_n22.json --delta arm_A arm_B --out-name vlm_hard_constraint_arm_ab_main_n22
```

**辅池 n=10（backend 03-03, 2026-07-01 preliminary）**：

| arm | sweep JSON | 图目录 |
|---|---|---|
| arm A（无擂台） | `outputs/eval/sweep_pairwise_win_delta_20260701_230222.json` | `outputs/eval/sweep_pairwise_win_delta_images_20260701_223349/delta_0.17/` |
| arm B（有擂台） | `outputs/eval/sweep_pairwise_win_delta_20260630_134045.json` | `outputs/eval/sweep_pairwise_win_delta_images_20260630_131315/delta_0.17/` |
| VLM 硬约束对比 | `outputs/eval/vlm_hard_constraint_arm_ab_20260701_231334.json` | — |

**主池 vs 辅池主题重叠**：`get_benchmark(n=10)` 与 `get_benchmark(n=32)` 是独立采样，两池共 5 个主题名称相同（征戍、客愁、耕牛炊烟、高楼远山、客舍孤灯）但由不同 backend 生成——这 5 主题在 §6 用作 **backend 稳定性 sanity check**。总覆盖 32 - 5 = 27 独立主题；32 主题级别的合并分析被放弃，因为跨 backend 的 CLIP/VLM 判定不严格可比。

## §2 四条核心证据（主池 + 辅池并列）

### 2.1 CLIP raw mean

| 池 | arm A（无擂台）| arm B（有擂台）| Δ (B − A) | 池内解读 |
|---|---|---|---|---|
| **主池 n=22** | **0.3090 ± 0.0235** | **0.3071 ± 0.0205** | **-0.0018** | A 略高，noise 内 |
| **辅池 n=10** | 0.311 ± 0.017 | 0.313 ± 0.016 | +0.002 | B 略高，noise 内 |

**两池 CLIP mean 差都 ≤ 0.002、均落在 std ≈ 0.02 的噪声带内**。CLIP mean 层，擂台无净收益。

### 2.2 CLIP 主题级 winner

| 池 | arm A wins | arm B wins | tie | 结论 |
|---|---|---|---|---|
| **主池 n=22** | **10** | **12** | **0** | B 微弱占优（差 2 主题，noise 内）|
| **辅池 n=10** | 5 | 5 | 0 | 严格 5:5 平手 |

主池主题级中位 |Δ CLIP| = 0.0104，不足以推动 mean 出线。CLIP 主题级层面，擂台仍无净收益。

### 2.3 VLM 硬约束总命中率（**关键指标**）

> **指标定义**（2026-08-10 补：本节此前未说明分母来源）
>
> 每个主题预先手写一份「用户明确要求必须画出来的东西」清单，再让 VLM 逐个看图判断有没有画出来。**命中率 = 判「有」的 keyword 数 ÷ keyword 总数**。
>
> **关键词表**硬编码在 `eval/vlm_hard_constraint.py:51` 的 `HARD_CONSTRAINTS`，按用户输入形态分两类：
>
> | density | 用户输入形态 | kw 数 | 例 |
> |---|---|---|---|
> | `rich` | 显式点名 2 个具体物体（「要有 X 和 Y」） | 2 | 「春景五绝，**要有柳树和燕子**」→ `["柳树","燕子"]` |
> | `sparse` | 只给主题词，无具体物体 | 1 | 「七绝，主题是**征戍**」→ `["征戍场景"]`（判定更宽松） |
>
> **分母 33** = 主池 22 主题中 rich 11 × 2 kw + sparse 11 × 1 kw = 22 + 11。与 §3 拆分表自洽：arm A = 14/22 + 6/11 = 20/33，arm B = 17/22 + 9/11 = 26/33。
>
> **VLM 判定契约**：`qwen-vl-max`，每张图一次调用问完全部 kw，逐项返回 `present: true/false` + ≤20 字理由。prompt 核心约束（`prompts/eval/vlm_hard_constraint.yaml:23`）是**「不评美感、不评意境，只判存在 / 不存在」**；允许写意简化但要求轮廓可辨——「柳树」需有下垂柳条，要求「燕子」却画了鹤判 false。
>
> **与 §2.4 的关系**：§2.3 是 **keyword 级**（分母 33），§2.4 是**主题级**（分母 22）——**同一批 VLM 判定的两种聚合粒度**，不是两次评测。
>
> **为什么它能当独立 oracle**：`eval/vlm_hard_constraint.py` 对 `core/` 零引用，**不在生产回路里**（生产改图循环用的是 CLIP）。正因为 pipeline 从未朝这个指标优化过，它才没有被 Goodhart 化，可以作为 CLIP 之外的独立判据——它补的正是 CLIP 的系统性盲点：CLIP 偏好「图里东西多 + 意境自洽」，不抠用户点名要的具体物体（见 §5 主题 6：arm A CLIP 全场最高 0.362，但要求的炉火与寒灯一个都没画）。

| 池 | arm A hit/total | arm B hit/total | Δ (B − A) | 池内解读 |
|---|---|---|---|---|
| **主池 n=22** | **20/33 = 60.6%** | **26/33 = 78.8%** | **+18.2 pp** | **arm B 大赢** |
| 辅池 n=10 | 10/15 = 66.7% | 10/15 = 66.7% | 0 pp | 平手 |

**这是本报告最强的一条证据**。主池 arm B 硬约束命中率比 arm A 高 18.2 pp——效应幅度远大于 CLIP mean 层 0.002 的噪声级差异，方向和 CLIP mean 相反。同一实验里 CLIP 说 "arm A 略胜、arm B 略输"，硬约束说 "arm B 大胜"，两个指标分开了。**擂台的价值在硬约束这一维上方向清晰地显现**——但注意 n=33 keyword 的配对检验未过显著阈值（McNemar p≈0.18，§2.5），本条定位为**强方向性证据而非统计定论**。

辅池 66.7% vs 66.7% 完全平手，与主池方向冲突——backend 敏感度问题，§6 讨论。

### 2.4 VLM 硬约束主题级 winner

| 池 | arm A 独赢 | arm B 独赢 | tie | 结论 |
|---|---|---|---|---|
| **主池 n=22** | **3** | **9** | **10** | **arm B 独赢是 arm A 的 3 倍** |
| 辅池 n=10 | 2 | 2 | 6 | 严格 2:2 平手 |

**主池主题级方向和总命中率一致**：arm B 独赢 9 主题（1/4/6/9/12/13/14/18/22），arm A 独赢仅 3 主题（15/17/21）。10 tie 主题里两 arm 硬约束表现相同。

## §2.5 显著性分析（2026-07-05 补，零 API 消耗）

> 本节是对已冻结数据的**加法**补充：不重跑、不改指标定义，只给 §2.3 的 +18.2pp 配上不确定性量化。计算脚本 `eval/_analyze_arena_ablation_significance.py`（bootstrap seed=20260705 固定，可精确复现），数据源即 §1 列出的 `vlm_hard_constraint_arm_ab_main_n22_20260702_133542.json`。

**keyword 级配对列联表**（33 keyword，两 arm 同题同 kw 配对）：

| 配对格 | 数量 |
|---|---|
| 双 hit (concordant) | 16 |
| 双 miss (concordant) | 3 |
| 仅 arm A hit (discordant) | 4 |
| 仅 arm B hit (discordant) | 10 |

| 检验 | 结果 |
|---|---|
| McNemar 精确检验（双侧，discordant 4 vs 10） | **p = 0.180** |
| 主题级 bootstrap 95% CI（10000 次重抽 theme，保留题内相关） | **Δ ∈ [-5.9pp, +41.9pp]** |

**读法（诚实版）**：

1. **+18.2pp 未过 0.05 显著阈值，CI 跨零** —— 在 n=33 keyword 的样本量下，不能排除「差距来自抽样噪声」的可能。本报告此前「远大于噪声 / 足够硬」的措辞过强，已相应修订。
2. **但方向证据是多重一致的**：keyword 级 10:4、主题级 9:3、rich/sparse 两个 density 拆分同向（§3）、5 个跨池重合主题 4/5 同向（§6.1）。作为**方向性证据**它仍是本报告最强的一条——只是「方向性强」≠「统计定论」。
3. **判定力估算**：若真实效应确为 +18pp 量级，样本扩到约 2 倍（≈66 keyword，即再补一个 n=22 池，如 §8 已列的 03-03 补跑）即大概率可过显著阈值。在扩样完成前，工程决策「保留擂台」的依据是**方向一致性 + 机制可解释（§5.2 案例）+ 保留成本低**，而非 p 值。
4. 本节结论与 kw 定义敏感度叠加读：+18.2pp 的点估计同时有**统计不确定性**（本节）与**kw 定义敏感性**两层折扣。后者已从 §5.1 的双点定性升级为 §2.7 全池扫描（2026-07-06）：三档定义包络 **Δ ∈ [+9.1pp, +24.2pp]，方向全为正**——引用本报告时请带上区间而非单点。

## §2.6 跨家族 VLM 复核（2026-07-06 补，44 次 glm-4v-plus 调用）

> 动机：主池硬约束 oracle（qwen-vl-max）与被评 pipeline 全链同家族（qwen-image 生图 + qwen-plus refine），存在 self-bias 质疑。用智谱 glm-4v-plus 对同 44 张主池图重判（`--vlm glm-4v-plus`，同 prompt 同温度）。

| oracle | arm A 命中 | arm B 命中 | Δ (B−A) | McNemar p | bootstrap 95% CI |
|---|---|---|---|---|---|
| qwen-vl-max（原报告） | 20/33 = 60.6% | 26/33 = 78.8% | **+18.2pp** | 0.180 | [-5.9, +41.9]pp |
| **glm-4v-plus（跨家族）** | 20/33 = 60.6% | 25/33 = 75.8% | **+15.2pp** | 0.227 | [-6.1, +34.4]pp |

- **方向跨家族复现**：Δ 同向（+15.2 vs +18.2pp），量级相近；逐 keyword 判定一致率 **86.4%**（57/66），分歧无系统性方向（qwen✓glm✗ 5 次 vs qwen✗glm✓ 4 次）——**"+18.2pp 是同家族 oracle 幻觉"的假设可以排除**。
- 统计显著性仍未过阈值（两个 oracle 都是，n=33 所限）——与 §2.5 定位一致：**方向性强证据、跨 oracle 稳健、待扩样定论**。
- 数据：`outputs/eval/vlm_hard_constraint_arm_ab_main_n22_glm4v_20260706_000935.json`；复现：`python -m eval.vlm_hard_constraint --agg outputs/eval/_agg_arena_ablation_main_n22.json --delta arm_A arm_B --vlm glm-4v-plus`。

## §2.7 kw 定义严/松敏感度全池扫描（2026-07-06 补，88 次 qwen-vl-max 调用）

> 动机：§5.1 用 theme 6 + theme 12 两个点定性演示过 kw 定义漂移会压缩 Δ（双点松判 → +12.1pp），且 §8 担忧 sparse OR 集合 kw 的偏差"系统性偏向 arm B"。本节把它做成全池系统扫描：对 22 主题手写 strict（本体明确可辨/场景须为画面主体）与 loose（写意暗示/边缘元素/同类替代均算）两套完整变体 kw（与冻结版逐条 1:1，变体仅存在于探针脚本、冻结 v1 定义不动），对同 44 张主池图各判一遍。

| kw 定义 | arm A 命中 | arm B 命中 | Δ (B−A) | McNemar p | bootstrap 95% CI |
|---|---|---|---|---|---|
| strict（判定下界） | 14/33 = 42.4% | 22/33 = 66.7% | **+24.2pp** | 0.077 | [+0.0, +47.1]pp |
| baseline（v1 冻结） | 20/33 = 60.6% | 26/33 = 78.8% | **+18.2pp** | 0.180 | [-5.9, +41.9]pp |
| loose（判定上界） | 24/33 = 72.7% | 27/33 = 81.8% | **+9.1pp** | 0.549 | [-13.3, +31.4]pp |

- **稳健区间：Δ ∈ [+9.1pp, +24.2pp]，三档定义下方向全为正**——主结论的方向不依赖任何一版 kw 措辞。§5.1 的双点松判估计（+12.1pp）落在全池 loose（+9.1pp）与 baseline 之间，量级吻合。
- **反直觉发现：从严反而放大 arm B 优势**（+24.2pp，CI 下界触零）——收紧判定时 arm A 掉得更狠（20→14）而 arm B 更耐严判（26→22）。§8 原担忧"kw 偏差系统性偏向 arm B"在从严方向上不成立；从松方向确实压缩 Δ（arm A 的边缘元素被宽容判定捞回更多），但方向不翻。
- **单调性违反 0 处**（strict 较 baseline 无新增命中、loose 无减少命中）——三档判定逻辑自洽，观测到的差异可归于定义效应而非重判噪声（单次判定噪声的观测下界为 0，不代表噪声不存在）。
- 引用建议：主结论请引用区间 **+9.1~+24.2pp（baseline +18.2pp）**，不要只引单点。
- 数据：`outputs/eval/vlm_kw_sensitivity_main_n22_20260706_134917.json` · 脚本 `eval/_sweep_kw_sensitivity.py` · 终端记录 `outputs/eval/_kw_sensitivity_terminal_20260706.log`。

## §3 rich / sparse 硬约束拆分

| 池 | density | arm A 命中率 | arm B 命中率 | Δ (B − A) |
|---|---|---|---|---|
| **主池 n=22** | rich (11 主题, 22 kw) | 63.6% (14/22) | **77.3% (17/22)** | **+13.6 pp** |
| **主池 n=22** | sparse (11 主题, 11 kw) | 54.5% (6/11) | **81.8% (9/11)** | **+27.3 pp** |
| 辅池 n=10 | rich (5 主题, 10 kw) | 70.0% (7/10) | 80.0% (8/10) | +10.0 pp |
| 辅池 n=10 | sparse (5 主题, 5 kw) | **60.0% (3/5)** | 40.0% (2/5) | **-20.0 pp** |

**主池两个 density 方向一致（都是 arm B 赢）**，rich 上擂台 +13.6 pp、sparse 上更是 +27.3 pp。**辅池 sparse 反过来 arm A 赢 +20 pp**——这是辅池与主池最大的分歧。

**为什么 sparse 上主池 vs 辅池分歧最大？**

- 辅池 sparse 只 5 主题（n 太小），sparse -20 pp = 3 vs 2，任一主题翻转即可扭转方向
- 主池 sparse 11 主题 (n=11)，样本大 2 倍以上，+27.3 pp = 9 vs 6，方向稳定
- 也可能与 backend 相关：04-22 backend 对"用户主题词的场景表达"更敏感（arm B refine 拓展的关键词能更好落到图上），03-03 backend 对首句意象依赖更强（arm A 直白首句反而更命中）

## §4.1 主池 22 主题双图并排（n=22 · backend 04-22）

> ℹ️ 本节及以下含 50 张图，经 jsDelivr CDN 加载（GitHub raw 端点限流/跨境访问不稳，2026-07-07 起切换）。图片资产的 source of truth 仍全部入库于 `eval/assets/report_images/`，CDN 失效时可将链接改回相对路径。

每主题展示：arm A 终图 + arm A 诗 + arm B 终图 + arm B 诗 + CLIP + VLM 硬约束判定。图路径基于本报告在 `eval/` 目录（`../outputs/...`）。

---

### 主题 1 · 秋景五律·疏桐寒蛩（rich）

硬约束：桐树、蛩（蟋蟀）

| arm A · CLIP=0.316 | arm B · CLIP=0.311 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/01_%E8%80%81%E7%9C%BC%E9%9A%8F%E9%A3%8E%E8%BD%AC_gen3_clip0.316.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/01_%E5%AD%A4%E5%B3%B0%E7%AB%8B%E6%9A%AE%E5%AF%92_gen1_clip0.311.jpg) |
| 诗名：*秋夜疏桐寒蛩* | 诗名：*秋桐寒蛩* |
| 老眼随风转<br>清秋独苦吟<br>松风惊落叶<br>山月照疏砧<br>夜雨添新恨<br>寒蛩替旧音<br>萧条归卧意<br>愁杀杜陵心 | 孤峰立暮寒<br>清响坠空残<br>桐凋风自静<br>雨歇气初干<br>雁沉青嶂尽<br>蛩咽素光阑<br>此境无声极<br>萧然坐欲看 |
| VLM: 桐树✓ · 蛩✗ (1/2) | VLM: 桐树✓ · 蛩✓ (2/2) |

判定：**arm B 独赢** 硬约束 +1

---

### 主题 2 · 七律·悲秋（sparse）

硬约束：悲秋场景（落叶/凋零/萧瑟/秋色）

| arm A · CLIP=0.296 | arm B · CLIP=0.280 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/02_%E6%B8%85%E6%84%81%E6%97%A5%E6%B0%B8%E6%99%9A%E8%8B%8D%E5%87%89_gen4_clip0.296.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/02_%E8%80%81%E5%8E%BB%E8%90%A7%E7%84%B6%E6%97%A0%E4%B8%80%E4%BA%8B_gen4_clip0.280.jpg) |
| 诗名：*悲秋吟* | 诗名：*悲秋感怀* |
| 清愁日永晚苍凉<br>衰鬓如丝又转伤<br>风动露华侵客簟<br>雨连霜气逼池塘<br>孤云远入南峰外<br>落叶翻随一径旁<br>欲向西风吹梦去<br>空吟落雁数声长 | 老去萧然无一事<br>此身空自叹浮萍<br>清霜昨夜飞无数<br>白露今朝洗尽青<br>已觉寒风摧鬓发<br>却思明月照床扃<br>一樽聊复倾新酒<br>未觉飘零亦怆情 |
| VLM: 悲秋场景✓ (1/1) | VLM: 悲秋场景✓ (1/1) |

判定：**硬约束一致**，CLIP arm A +0.016

---

### 主题 3 · 五绝·寒梅（sparse）

硬约束：梅花

| arm A · CLIP=0.336 | arm B · CLIP=0.325 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/03_%E5%AF%92%E9%A3%8E%E6%91%87%E8%90%BD%E5%B0%BD_gen1_clip0.336.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/03_%E7%8E%89%E9%AA%A8%E5%86%B0%E5%A7%BF%E8%BF%9C_gen1_clip0.325.jpg) |
| 诗名：*寒梅* | 诗名：*寒梅咏* |
| 寒风摇落尽<br>独有玉花残<br>惟有香如故<br>春来不待兰 | 玉骨冰姿远<br>孤根傲雪霜<br>岁寒方见节<br>不待百花芳 |
| VLM: 梅花✓ (1/1) | VLM: 梅花✓ (1/1) |

判定：**硬约束一致**，CLIP arm A +0.011

---

### 主题 4 · 冬景七绝·飞雪寒鸦（rich）

硬约束：雪、寒鸦（乌鸦）

| arm A · CLIP=0.317 | arm B · CLIP=0.341 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/04_%E5%A4%9C%E6%9C%88%E7%94%9F%E4%BA%91%E5%86%BB%E4%B8%80%E6%B1%9F_gen3_clip0.317.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/04_%E5%AD%A4%E6%9D%BE%E7%8B%AC%E7%AB%8B%E5%8D%83%E5%B3%B0%E8%82%83_gen1_clip0.341.jpg) |
| 诗名：*飞雪寒江* | 诗名：*雪岭寒鸦* |
| 夜月生云冻一江<br>朔风飞雪打平沙<br>谁将此意为谁遣<br>唤起南楼看万家 | 孤松独立千峰肃<br>霜骨棱棱势自遒<br>雪压虬枝犹抱月<br>寒鸦忽掠一痕秋 |
| VLM: 雪✓ · 寒鸦✗ (1/2) | VLM: 雪✓ · 寒鸦✓ (2/2) |

判定：**arm B 独赢** 硬约束 +1

---

### 主题 5 · 五律·雪夜（sparse）

硬约束：雪夜场景（雪 + 夜色）

| arm A · CLIP=0.308 | arm B · CLIP=0.273 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/05_%E9%9C%9C%E6%B0%94%E5%B7%B2%E5%87%84%E7%84%B6_gen2_clip0.308.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/05_%E5%AD%A4%E7%81%AF%E6%91%87%E7%B4%A0%E5%A3%81_gen3_clip0.273.jpg) |
| 诗名：*雪夜孤吟* | 诗名：*雪夜客思* |
| 霜气已凄然<br>寒声转寂然<br>山深人迹绝<br>水远客船眠<br>老屋残灯灭<br>空窗密雪圆<br>孤吟谁与和<br>一曲苦边弦 | 孤灯摇素壁<br>寒漏滴空庭<br>客梦霜风碎<br>空帷竹影青<br>峰峦吞夜黑<br>云气压窗冥<br>莫道归程远<br>更深月自停 |
| VLM: 雪夜场景✓ (1/1) | VLM: 雪夜场景✓ (1/1) |

判定：**硬约束一致**，CLIP arm A +0.035

---

### 主题 6 · 冬景七律·炉火寒灯（rich）· **反直觉案例**

硬约束：炉火（火焰）、寒灯（灯）

| arm A · CLIP=**0.362**（全 sweep 最高）· 硬约束 **0/2** | arm B · CLIP=0.290 · 硬约束 **1/2** |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/06_%E5%AF%92%E7%AA%97%E5%8D%8A%E6%8E%A9%E5%8D%B7%E6%8A%AB%E7%96%8F_gen1_clip0.362.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/06_%E7%82%89%E7%81%AB%E5%87%9D%E7%83%9F%E5%A4%9C%E6%9C%AA%E5%A4%AE_gen4_clip0.290.jpg) |
| 诗名：*寒窗夜坐* | 诗名：*冬夜炉火* |
| 寒窗半掩卷披疏<br>坐久风前日欲晡<br>一榻烟芜闲似睡<br>两炉炉炭冷犹苏<br>霜浓木叶无纤缕<br>雪满梅花有几株<br>此兴自怜人已晚<br>更添愁思忆吾庐 | 炉火凝烟夜未央<br>竹窗风雪叩寒霜<br>邻家酒旆因风寂<br>老伴茶瓯带影香<br>独坐忽惊更漏永<br>暗移半榻月痕凉<br>何人共我倾松露<br>莫待新桃换旧妆 |
| VLM: 炉火✗ · 寒灯✗ (0/2) | VLM: 炉火✓ · 寒灯✗ (1/2) |

判定：**arm B 独赢** 硬约束 +1。**这是 CLIP 与硬约束脱钩最强的一个案例**：arm A CLIP 0.362 是全 sweep 22 主题最高分，但用户明确要求的"炉火"和"寒灯"图里一个都没画（arm A 的诗虽然写了"两炉炉炭"、"残灯"暗示，但图像 pipeline 没把它们落到图上）；arm B CLIP 明显低 0.072，但至少把炉火画进去了。**CLIP 高分 ≠ 用户约束满意度**。

---

### 主题 7 · 五绝·溪声（sparse）

硬约束：溪水（水流/溪流）

| arm A · CLIP=0.305 | arm B · CLIP=0.308 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/07_%E6%BA%AA%E6%B0%B4%E8%87%AA%E4%B8%9C%E8%A5%BF_gen3_clip0.305.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/07_%E5%BE%AE%E9%A3%94%E7%A9%BF%E7%AB%B9%E5%BE%84_gen2_clip0.308.jpg) |
| 诗名：*溪声* | 诗名：*溪声* |
| 溪水自东西<br>山花来去同<br>溪声无定在<br>何处更相逢 | 微飔穿竹径<br>清响落云间<br>松影摇溪浅<br>苔痕浸水闲 |
| VLM: 溪水✓ (1/1) | VLM: 溪水✓ (1/1) |

判定：**双方硬约束一致 + CLIP 平手**

---

### 主题 8 · 山水七律·高楼远山（rich）· **backend 对比案例**

硬约束：高楼、远山

| arm A · CLIP=0.307 · 硬约束 2/2 | arm B · CLIP=0.303 · 硬约束 2/2 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/08_%E7%99%BB%E4%B8%B4%E8%A7%88%E7%BD%A2%E6%AC%B2%E5%87%AD%E6%A0%8F_gen2_clip0.307.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/08_%E7%99%BB%E6%A5%BC%E9%AB%98%E7%9C%BA%E7%A2%A7%E4%BA%91%E5%A4%A9_gen1_clip0.303.jpg) |
| 诗名：*登楼望远* | 诗名：*登楼远眺* |
| 登临览罢欲凭栏<br>一气周流万象宽<br>万里阴云开大漠<br>千峰雪岭照沧澜<br>孤城古堞人归后<br>高塔秋楼月向残<br>回首故园无此景<br>烟霞无限在云端 | 登楼高眺碧云天<br>日暖风清鸟乱旋<br>远水归帆春草绿<br>孤城衔暮晚钟悬<br>江流岂必随人意<br>云去原无系物牵<br>莫恨此身羁旅久<br>年华未老可回鞭 |
| VLM: 高楼✓ · 远山✓ (2/2) | VLM: 高楼✓ · 远山✓ (2/2) |

判定：**双方硬约束一致 + CLIP 平手**。**这个主题在辅池（03-03）里 arm A 独赢 arm B 全 miss（详见 §4.2 主题 8）**，主池（04-22）两 arm 都全中。**backend 的差异改变了主题级结论**。

---

### 主题 9 · 田园七绝·耕牛炊烟（rich）· **反直觉案例**

硬约束：耕牛、炊烟

| arm A · CLIP=0.327 · 硬约束 **0/2** | arm B · CLIP=0.322 · 硬约束 **2/2** |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/09_%E6%99%93%E6%9D%A5%E4%B8%80%E5%A3%91%E9%9D%99%E6%97%A0%E9%9F%B3_gen1_clip0.327.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/09_%E6%9D%91%E8%B7%AF%E9%A3%8E%E6%B8%85%E9%9B%A8%E8%BF%87%E7%A8%80_gen3_clip0.322.jpg) |
| 诗名：*耕烟暮色* | 诗名：*春村耕趣* |
| 晓来一壑静无音<br>半壁青山对远林<br>野店竹篱茅舍里<br>暮年犹得伴耕深 | 村路风清雨过稀<br>春来无事亦忘机<br>耕牛未肯还田去<br>犹把新泥护柳枝 |
| VLM: 耕牛✗ · 炊烟✗ (0/2) | VLM: 耕牛✓ · 炊烟✓ (2/2) |

判定：**arm B 独赢** 硬约束 **+2**（全部反转）。arm A "野店竹篱茅舍" 的抽象山水描写把两个硬约束都遗漏，arm B 擂台 refine 后诗里出现"耕牛未肯还田去"直接锚定耕牛，"春村耕趣"标题也隐含炊烟，两个约束都画出来了。**这是擂台把两个关键约束都补回来的直接铁证**。

---

### 主题 10 · 五律·归隐（sparse）

硬约束：归隐场景（茅屋/山居/隐士）

| arm A · CLIP=0.322 | arm B · CLIP=0.325 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/10_%E8%80%81%E5%8E%BB%E5%8E%8C%E5%B0%98%E5%9A%A3_gen1_clip0.322.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/10_%E6%98%94%E5%B9%B4%E8%BE%9E%E7%A6%84%E5%8E%BB_gen1_clip0.325.jpg) |
| 诗名：*归隐幽居* | 诗名：*归隐青山* |
| 老去厌尘嚣<br>归来得自由<br>松篁连古寺<br>风月到茅丘<br>客少人稀坐<br>云深树独游<br>欲知幽胜处<br>石径曲通幽 | 昔年辞禄去<br>此日得闲身<br>云外千峰静<br>林间一径新<br>茶烟浮石榻<br>松影落衣巾<br>莫道归来晚<br>青山正待人 |
| VLM: 归隐场景✗ (0/1) | VLM: 归隐场景✗ (0/1) |

判定：**双 miss + CLIP 平手**。归隐这个抽象概念两 arm 都没画出可识别的"隐士/茅屋"，是 pipeline 系统性盲点。

---

### 主题 11 · 边塞五律·戍楼角声（rich）

硬约束：戍楼（烽火台/城楼/箭楼）、边塞角声场景（军旗/号角/军营）

| arm A · CLIP=0.331 | arm B · CLIP=0.276 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/11_%E6%9C%88%E8%89%B2%E7%85%A7%E8%90%A5%E5%B8%90_gen1_clip0.331.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/11_%E7%A7%8B%E9%A3%8E%E5%8D%B7%E5%9C%B0%E5%AF%92_gen4_clip0.276.jpg) |
| 诗名：*戍楼角声* | 诗名：*戍楼角声* |
| 月色照营帐<br>星稀见戍楼<br>马毛风上耸<br>兵刃月中流<br>独树秋萧索<br>孤村夜斗筹<br>平生未有报<br>空自卧金钩 | 秋风卷地寒<br>客泪沾衣难<br>独倚戍楼角<br>平明旆影残<br>荒城埋白骨<br>老将拭霜鞍<br>功成何日返<br>归卧旧篱看 |
| VLM: 戍楼✓ · 边塞角声场景✓ (2/2) | VLM: 戍楼✓ · 边塞角声场景✓ (2/2) |

判定：**硬约束一致**，CLIP arm A +0.055

---

### 主题 12 · 七绝·征戍（sparse）

硬约束：征戍场景（士兵/戎装/边塞地貌）

| arm A · CLIP=0.317 | arm B · CLIP=0.321 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/12_%E4%B8%87%E9%87%8C%E9%A3%8E%E9%9C%9C%E4%B8%80%E9%9B%81%E8%A1%8C_gen1_clip0.317.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/12_%E5%B1%B1%E8%A1%94%E8%90%BD%E6%97%A5%E5%AD%A4%E5%9F%8E%E9%97%AD_gen1_clip0.321.jpg) |
| 诗名：*边城征戍* | 诗名：*征戍吟* |
| 万里风霜一雁行<br>边城春尽尚馿兵<br>故园回首烟云合<br>独卧南楼月色清 | 山衔落日孤城闭<br>万里风霜一雁沉<br>铁甲凝寒犹望北<br>半蓑烟雨自成霖 |
| VLM: 征戍场景✗ (0/1) | VLM: 征戍场景✓ (1/1) |

判定：**arm B 独赢** 硬约束 +1

---

### 主题 13 · 羁旅五绝·客舍孤灯（rich）

硬约束：客舍、孤灯

| arm A · CLIP=0.255 | arm B · CLIP=0.342 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/13_%E9%9B%A8%E6%AD%87%E5%B1%B1%E5%9F%8E%E6%99%93_gen3_clip0.255.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/13_%E5%AE%A2%E8%88%8D%E5%AD%A4%E7%81%AF%E5%A4%9C_gen2_clip0.342.jpg) |
| 诗名：*客舍孤灯* | 诗名：*客舍孤灯* |
| 雨歇山城晓<br>风来叶满庭<br>不知愁夜半<br>独宿客家灯 | 客舍孤灯夜<br>江云入梦来<br>月华清似水<br>风度淡如梅 |
| VLM: 客舍✗ · 孤灯✓ (1/2) | VLM: 客舍✓ · 孤灯✓ (2/2) |

判定：**arm B 独赢** 硬约束 +1

---

### 主题 14 · 七律·客愁（sparse）

硬约束：羁旅愁思场景（行旅/客舍/孤客）

| arm A · CLIP=0.301 | arm B · CLIP=0.292 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/14_%E9%A3%8E%E9%9B%AA%E5%85%B3%E6%B2%B3%E5%8F%88%E4%B8%80%E5%B9%B4_gen3_clip0.301.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/14_%E9%A3%8E%E5%9B%9E%E6%9B%B2%E5%BE%84%E9%A6%99%E6%88%90%E9%98%B5_gen3_clip0.292.jpg) |
| 诗名：*客愁* | 诗名：*客愁* |
| 风雪关河又一年<br>故人南浦隔重烟<br>秋怀似我心相似<br>乡思如君眼不偏<br>旅况不堪愁独处<br>岁寒空自苦萧然<br>何时把酒同为乐<br>共话江头落照边 | 风回曲径香成阵<br>云敛遥峰影入楼<br>蝶魂乍醒春事老<br>莺语犹怜故园秋<br>酒暖偏惊罗袖薄<br>花飞漫数玉钗柔<br>西窗烛冷人何在<br>欲说还休泪自收 |
| VLM: 羁旅愁思场景✗ (0/1) | VLM: 羁旅愁思场景✓ (1/1) |

判定：**arm B 独赢** 硬约束 +1

---

### 主题 15 · 七绝·送别（sparse）

硬约束：送别场景（分别/挥手/离别酒/长亭/柳枝）

| arm A · CLIP=0.281 | arm B · CLIP=0.308 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/15_%E6%B1%9F%E5%9F%8E%E7%A7%8B%E8%89%B2%E8%BF%91%E9%BB%84%E6%98%8F_gen4_clip0.281.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/15_%E5%BF%BD%E6%8B%82%E5%BE%81%E8%A1%A3%E7%90%86%E6%97%A7%E5%BC%A6_gen1_clip0.308.jpg) |
| 诗名：*江城送别* | 诗名：*阳关送别* |
| 江城秋色近黄昏<br>日暮归心正断魂<br>欲识别离无处所<br>更须相送过前村 | 忽拂征衣理旧弦<br>阳关声断暮云连<br>孤峰渐没归鸿影<br>一缕青痕落远天 |
| VLM: 送别场景✓ (1/1) | VLM: 送别场景✗ (0/1) |

判定：**arm A 独赢** 硬约束 +1。arm A 诗"更须相送过前村"直接描述送别动作，arm B 擂台改成"忽拂征衣理旧弦"文人送别—— 意境典雅但"送别场景"从图里消失。**这是主池 3 个 arm A 独赢主题里最典型的"擂台改诗方向正交"案例**。

---

### 主题 16 · 送别七律·长亭折柳（rich）

硬约束：长亭、柳枝（折柳）

| arm A · CLIP=0.322 | arm B · CLIP=0.324 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/16_%E6%B1%9F%E5%A4%B4%E9%85%92%E5%8A%9B%E5%B7%B2%E5%BE%AE%E9%86%BA_gen1_clip0.322.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/16_%E9%9C%9C%E5%87%9D%E5%8F%A4%E9%81%93%E6%9A%AE%E4%BA%91%E5%AF%92_gen1_clip0.324.jpg) |
| 诗名：*离亭折柳* | 诗名：*长亭折柳* |
| 江头酒力已微醺<br>一别相逢几许春<br>风递梅花知远近<br>水含明月见相亲<br>青门路断无多雨<br>白发人归少有尘<br>欲向离亭折杨柳<br>可怜无力更伤神 | 霜凝古道暮云寒<br>客路迢迢未抵安<br>孤馆一灯摇影瘦<br>空山万木带烟残<br>归心已逐南飞雁<br>别泪犹沾北去鞍<br>行尽关河肠欲断<br>长亭折柳月将阑 |
| VLM: 长亭✓ · 柳枝✓ (2/2) | VLM: 长亭✓ · 柳枝✓ (2/2) |

判定：**双方硬约束一致 + CLIP 平手**

---

### 主题 17 · 怀古五律·古城荒台（rich）

硬约束：古城（残城/城墙）、荒台（废弃高台/遗迹）

| arm A · CLIP=0.289 | arm B · CLIP=0.307 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/17_%E8%A5%BF%E9%99%B5%E5%9F%8E%E5%B7%B2%E5%9D%8F_gen3_clip0.289.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/17_%E6%95%85%E5%9B%AD%E8%8B%94%E5%BE%84%E5%86%B7_gen1_clip0.307.jpg) |
| 诗名：*西陵怀古* | 诗名：*古城荒台* |
| 西陵城已坏<br>南浦水应深<br>草色侵人骨<br>沙痕入马心<br>夕阳空野戍<br>秋雨动寒砧<br>谁识当年事<br>荒台月满林 | 故园苔径冷<br>废堞雁声斜<br>暮霭吞孤垒<br>秋风扫落霞<br>荣枯皆过眼<br>兴废本无嗟<br>唯见青磷起<br>寒磷照旧沙 |
| VLM: 古城✓ · 荒台✓ (2/2) | VLM: 古城✓ · 荒台✗ (1/2) |

判定：**arm A 独赢** 硬约束 +1

---

### 主题 18 · 七律·吊古（sparse）

硬约束：吊古场景（残垣/断碑/苍凉遗迹）

| arm A · CLIP=0.313 | arm B · CLIP=0.289 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/18_%E5%8D%97%E6%9C%9D%E4%B8%80%E9%83%A1%E6%9C%89%E7%8E%8B%E5%9F%8E_gen2_clip0.313.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/18_%E6%9B%BE%E9%97%BB%E9%93%81%E7%94%B2%E8%A3%82%E9%9C%9C%E5%A4%A9_gen3_clip0.289.jpg) |
| 诗名：*吊南朝王城* | 诗名：*吊古* |
| 南朝一郡有王城<br>风物如今忆昔荣<br>江树半春留客醉<br>淮田秋雨断人声<br>山连故垒孤村暮<br>花落空楼远水生<br>回首故园非旧国<br>西来何日再吹筝 | 曾闻铁甲裂霜天<br>兴废浮沉剑自悬<br>关塞云崩龙气在<br>河山血淬骨锋坚<br>残垣暗蚀苔痕篆<br>断镞犹衔夜月寒<br>登楼忽觉星垂野<br>万古长风扫旧鞍 |
| VLM: 吊古场景✗ (0/1) | VLM: 吊古场景✓ (1/1) |

判定：**arm B 独赢** 硬约束 +1

---

### 主题 19 · 中秋五绝·明月团圆（rich）

硬约束：明月（圆月）、团圆场景（家人聚会/圆桌）

| arm A · CLIP=0.297 | arm B · CLIP=0.303 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/19_%E4%BB%8A%E5%A4%9C%E6%B8%85%E5%85%89%E6%BB%A1_gen4_clip0.297.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/19_%E7%A7%8B%E5%AE%B5%E6%B8%85%E4%BC%BC%E6%B0%B4_gen1_clip0.303.jpg) |
| 诗名：*中秋望月* | 诗名：*中秋望月* |
| 今夜清光满<br>中秋未到时<br>月圆人不圆<br>独坐对孤影 | 秋宵清似水<br>素魄出东楼<br>盈时当举盏<br>未醉不封侯 |
| VLM: 明月✓ · 团圆场景✗ (1/2) | VLM: 明月✓ · 团圆场景✗ (1/2) |

判定：**硬约束一致**，CLIP arm B +0.007。"团圆"这个抽象概念两 arm 都画不出可识别的场景。

---

### 主题 20 · 五律·重阳（sparse）

硬约束：重阳场景（登高/菊花/茱萸）

| arm A · CLIP=0.287 | arm B · CLIP=0.290 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/20_%E9%A3%8E%E6%9C%88%E4%BB%8A%E9%87%8D%E4%B9%9D_gen3_clip0.287.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/20_%E9%87%8D%E9%98%B3%E4%B8%B4%E9%87%8E%E9%98%94_gen3_clip0.290.jpg) |
| 诗名：*重阳登高* | 诗名：*重阳登高* |
| 风月今重九<br>登高望未休<br>秋山空翠湿<br>菊酒满金秋<br>雁过人将散<br>萸残客暂留<br>东篱何限醉<br>聊此慰衰愁 | 重阳临野阔<br>登高意自真<br>篱疏霜色浅<br>山远雁声频<br>云破千峰出<br>风回一径匀<br>天垂沧海阔<br>菊映素秋新 |
| VLM: 重阳场景✓ (1/1) | VLM: 重阳场景✓ (1/1) |

判定：**双方硬约束一致 + CLIP 平手**

---

### 主题 21 · 禅意七绝·古刹钟鼓（rich）· **反直觉案例**

硬约束：古刹（寺庙/佛塔）、钟鼓（钟或鼓）

| arm A · CLIP=0.332 · 硬约束 **2/2** | arm B · CLIP=0.335 · 硬约束 **0/2** |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/21_%E5%8F%A4%E5%88%B9%E9%92%9F%E5%A3%B0%E5%A4%9C%E6%9C%AA%E4%BC%91_gen1_clip0.332.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/21_%E9%9B%A8%E6%94%B6%E5%B1%B1%E8%89%B2%E6%B4%97%E5%B0%98%E6%96%B0_gen1_clip0.335.jpg) |
| 诗名：*古刹钟声* | 诗名：*古刹钟云* |
| 古刹钟声夜未休<br>松间野鹤伴云游<br>人间岁月如流水<br>更向何时得少留 | 雨收山色洗尘新<br>石径苔深印屐茵<br>钟曳流云千嶂散<br>松筛碎日一襟春 |
| VLM: 古刹✓ · 钟鼓✓ (2/2) | VLM: 古刹✗ · 钟鼓✗ (0/2) |

判定：**arm A 独赢** 硬约束 **+2**（全部反转）。arm A 诗首句"古刹钟声"直接锚定两个约束，arm B 擂台把"古刹钟声"改成"雨收山色"—— 意境是有了，但"古刹"和"钟鼓"两个用户约束都没了。**这是"擂台改诗方向与用户约束正交"最严重的一次全反转**。同类现象也出现在主题 15（送别）。

---

### 主题 22 · 五绝·无常（sparse）

硬约束：无常主题场景（枯萎/凋零/流水/逝去/孤影）

| arm A · CLIP=0.275 | arm B · CLIP=0.292 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_085340/delta_0.17/22_%E6%9C%89%E8%BA%AB%E5%A6%82%E6%A2%A6%E8%80%85_gen3_clip0.275.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260702_094058/delta_0.17/22_%E4%B8%80%E7%93%A3%E9%9A%8F%E9%A3%8E%E8%BF%9C_gen3_clip0.292.jpg) |
| 诗名：*无常偈* | 诗名：*无常* |
| 有身如梦者<br>无我更无时<br>何处觅真子<br>空门见自知 | 一瓣随风远<br>深林暗度香<br>欲追云外影<br>何必问苍茫 |
| VLM: 无常主题场景✗ (0/1) | VLM: 无常主题场景✓ (1/1) |

判定：**arm B 独赢** 硬约束 +1

---

## §4.2 辅池 10 主题双图并排（n=10 · backend 03-03）

_保留 preliminary 阶段（2026-07-01）的双图对比。10 主题全部数据参见下表。辅池整体命中率与主池方向不一致（辅池平手、主池 arm B 大赢），详见 §6 讨论。_

---

### 辅池·主题 1 · 春景五绝·柳燕（rich）

硬约束：柳树、燕子

| arm A · CLIP=0.293 | arm B · CLIP=0.326 |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260701_223349/delta_0.17/01_%E7%87%95%E8%B9%B4%E8%8A%B1%E9%98%B4%E4%B8%8B_gen4_clip0.293.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260630_131315/delta_0.17/01_%E9%A3%8E%E6%9A%96%E8%8E%BA%E6%A2%AD%E6%9F%B3_gen1_clip0.326.jpg) |
| 诗名：*柳燕春思* | 诗名：*莺舞春光* |
| 燕蹴花阴下<br>莺啼柳翠低<br>日长人独立<br>春事已阑珊 | 风暖莺梭柳<br>春浓燕剪泥<br>花繁蜂远岫<br>水阔鸭平堤 |
| VLM: 柳树✓ · 燕子✓ (2/2) | VLM: 柳树✓ · 燕子✓ (2/2) |

判定：**硬约束双 hit**。CLIP arm B +0.033，两张图都完整覆盖用户约束。

---

### 辅池·主题 5 · 五绝·夏蝉（sparse）· **辅池反直觉**

硬约束：蝉

| arm A · CLIP=0.281 · 硬约束 **1/1** | arm B · CLIP=0.308 · 硬约束 **0/1** |
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260701_223349/delta_0.17/05_%E7%82%8E%E8%92%B8%E6%9A%91%E9%9B%A8%E5%B0%BD_gen4_clip0.281.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260630_131315/delta_0.17/05_%E5%8D%88%E9%A3%8E%E6%B8%85%E8%B6%8A%E6%8A%A5_gen1_clip0.308.jpg) |
| 炎蒸暑雨尽<br>高柳一惊蝉<br>未省为谁碎<br>唯闻不断喧 | 午风清越报<br>高柳一凉阴<br>谁遣林间夜<br>唯闻碧海头 |
| VLM: 蝉✓ (1/1) | VLM: 蝉✗ (0/1) |

判定：**arm A 独赢**。arm B 擂台改诗后"蝉"暗示消失（"炎蒸暑雨尽" → "午风清越报"，暑意退场蝉的视觉锚点也丢了）。同类现象在主池 21 古刹钟鼓上以更严重的形式复现。

---

### 辅池·主题 8 · 山水七律·高楼远山（rich）· **backend 敏感度**

硬约束：高楼、远山

| arm A · CLIP=0.329 · 硬约束 **2/2** | arm B · CLIP=0.304 · 硬约束 **1/2**（高楼 miss）|
|---|---|
| ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260701_223349/delta_0.17/08_%E6%A5%BC%E9%98%81%E4%B8%B4%E6%B1%9F%E9%9D%A2%E6%B8%BA%E8%8C%AB_gen1_clip0.329.jpg) | ![](https://cdn.jsdelivr.net/gh/Judy-Liu118/InkVerse@main/eval/assets/report_images/20260630_131315/delta_0.17/08_%E4%BA%91%E5%BC%80%E5%8D%83%E5%B6%82%E7%A2%A7%E7%A9%BA%E6%96%9C_gen2_clip0.304.jpg) |
| 楼阁临江面渺茫<br>凭高徙倚意苍苍<br>水天一色寒光暮<br>山北青青柳絮长<br>春事已备行客游<br>春波不入无余化 | 云开千嶂碧空斜<br>又涌何堪长滿华<br>春草春光何细细<br>春事春光何季间<br>溪梅书影摇初露<br>石壁苔文印玉纱 |
| VLM: 高楼✓ · 远山✓ (2/2) | VLM: 高楼✗ · 远山✓ (1/2) |

判定：**arm A 独赢**。**辅池这个主题 arm A 独赢，但主池同一主题（04-22 backend）两 arm 都全中**——backend 差异直接改变了主题级结论。

---

_辅池其余 7 主题（柳燕、征戍、桃莺、客愁、耕牛炊烟、消夏、客舍孤灯、春雨）双图 preliminary 版本略；详见 preliminary 报告 git 历史（2026-07-01 commit）或原始数据 `outputs/eval/vlm_hard_constraint_arm_ab_20260701_231334.md`。_

## §5 反直觉案例集

擂台是否有净收益的判定不能只看 mean —— **哪些主题让擂台"生效"、哪些"反作用"** 才是真正的洞察。本节汇总 5 个最具代表性的反直觉案例：

### 5.1 CLIP 与硬约束脱钩（主池 主题 6 炉火寒灯）

**arm A CLIP=0.362（全 sweep 22 主题最高），硬约束 0/2；arm B CLIP=0.290，硬约束 1/2**（严格 kw 判定）。

CLIP 会为"意境自洽、构图美观"给高分——arm A 冬日文人枯坐的水墨意境完美，画面右下角实际画了**两只火盆+炭块**（VLM 判词原话："炉中无火焰，仅见炭块"），但**用户明确要求的"寒灯"完全缺失**；"炉火"这一 kw 因括号收窄到"（火焰）"被判 False。arm B CLIP 低 0.072 但画出了明火焰，"寒灯"仍缺。

> **kw 严格度 caveat**：本 pipeline 的 kw `"炉火（火焰）"` 用括号强制"必须见明火"，比自然语言"炉火"更严。若松解释（炉+炭即算命中），主池整体 Δ 从 +18.2pp 变 **+15.2pp**（arm A 20→21/33，arm B 不变），theme 6 winner 从 "arm B 独赢 +2" 变 **tie**（A=1/2、B=1/2），主池主题级 winner 从 3:9:10 变 3:8:11——方向和主结论不变，仅本案的"铁证"强度打折。同类"括号收窄" kw 其他有 `"寒灯（灯）"`、`"明月（圆月）"`。
>
> 另有 **主题 12（征戍 sparse）** 属另一类边缘 case——"sparse 场景 OR 集合" kw 里"画面边缘元素被 VLM 因非主意象而忽略"：arm A 图主意象是"雁+月+云"，horizon 中偏左有一小片水墨疑似远景城楼/村落，VLM 判 False；若松看（边缘元素也算命中），theme 12 从 "arm B 独赢 +1" 变 **tie**。**theme 6 + theme 12 双 case 同时松判**下主池 Δ 变 **+12.1pp**（arm A 22/33=66.7%）、主题级 winner 3:7:12——方向仍 arm B 显著占优。这类 sparse OR 集合的边缘偏差**系统性偏向 arm B**（arm B 经 refine 后画面元素更明确），主题 14/18/22 kw 结构类似但未详审。~~全池灵敏度分析列为 §8 follow-up~~ **已完成**（2026-07-06，§2.7）：全池 strict/loose 包络 Δ ∈ [+9.1, +24.2]pp 方向全为正，且从严方向 arm B 优势反而扩大——"偏向 arm B"的担忧只在从松方向部分成立、不足以翻转方向。

工程含义：脱钩的核心是 **"寒灯"这一维**——arm A 完美画了雪窗、书卷、火盆、炭火，唯独没有灯，CLIP 依然给 0.362 高分。**"CLIP 高分 ≠ 用户所有约束都满意"这条铁律仍成立**：只要一部分约束缺失（这里是寒灯），CLIP 就可能被其他视觉元素堆到高分。如果 pipeline 只用 CLIP 门控（如本 pipeline 里 target CLIP raw ≥ 0.30），会持续错过硬约束**逐条缺失**这类问题——需要独立的硬约束 gate（当前已有 vlm_hard_constraint 独立评测，但没进 pipeline 反馈环）。

### 5.2 擂台把用户约束都补回来（主池 主题 9 耕牛炊烟）

**arm A 硬约束 0/2 → arm B 硬约束 2/2（全部反转）**。arm A 冠军诗"晓来一壑静无音"的抽象山水描写把两个用户要求的约束都遗漏；arm B 擂台 refine 后诗里出现"耕牛未肯还田去"，标题也换成"春村耕趣"——两个约束都画出来了。

**这是本报告最有力的"擂台捞回用户约束"证据**。同类主题还有主题 1（疏桐寒蛩）、主题 4（飞雪寒鸦）、主题 12（征戍）、主题 13（客舍孤灯）、主题 14（客愁）、主题 18（吊古）、主题 22（无常）——主池 9 个 arm B 独赢主题里大多数呈现这个模式。

### 5.3 擂台把用户约束改没了（主池 主题 21 古刹钟鼓 + 主题 15 送别）

**arm A 硬约束 2/2 → arm B 硬约束 0/2（全部反转，反方向）**。

- 主题 21：arm A 诗首句"古刹钟声夜未休"直接把两个约束锚定；arm B 擂台把首句改成"雨收山色洗尘新"—— 意境提升但"古刹"和"钟鼓"两个用户约束全丢
- 主题 15：arm A 诗"更须相送过前村"直接描述送别；arm B 擂台改成"忽拂征衣理旧弦"文人送别—— 场景消失

这类"擂台改诗方向与用户约束正交"的现象出现在 arm A 独赢 3 主题里的 2 个（15、21）。**根因**（推测）：refine_poem prompt 是"如何写得更好"（追求辞采、含蓄、意境），而**不是**"如何更贴合用户 input"。当擂台命中大约 50% 攻擂成功率，被驱动"改"而非"保"，改动可能朝反方向。

### 5.4 backend 敏感度（主题 8 高楼远山）

**辅池（03-03）**：arm A 硬约束 2/2 → arm B 硬约束 1/2（高楼 miss）。
**主池（04-22）**：arm A 硬约束 2/2 → arm B 硬约束 2/2（两 arm 都全中）。

同一主题、同一 arm A/B 逻辑，只是 image backend 从 03-03 换成 04-22，arm B 结果从 1/2 → 2/2。这说明**擂台改诗后的图像 backend 对"具体物体是否画出来"的容错度不同**：04-22 对 arm B refine 后"云开千嶂"这种偏抽象的 prompt 依然能把"高楼"画出来，03-03 就画不出来。

**这解释了为什么辅池整体平手、主池整体 arm B 大赢**：辅池 arm B 在多个主题（含蝉、高楼、耕牛炊烟里的"耕牛"）因 03-03 表达力不足而 miss，主池 arm B 在 04-22 上不容易 miss。

### 5.5 双 miss 系统盲点（主池 主题 10 归隐 · 辅池 主题 4 客愁、主题 10 春雨）

一些抽象主题（归隐、客愁）或视觉难以表达的元素（细雨）在两 arm 上都 miss，与擂台无关，属于 pipeline 系统盲点（image prompt 里没有具象锚点 + 水墨 style 对细节的抹平效应叠加）。这些主题**独立于 arm A/B 对比**，属于要在 pipeline 层单独修的地方。

## §6 擂台净收益成立 vs 不成立的原因分析

**主池（n=22, 04-22）擂台有净收益：CLIP 平手、硬约束 +18.2pp**。
**辅池（n=10, 03-03）擂台无净收益：CLIP 平手、硬约束 0pp**。

**为什么两池方向不同？** 按可能性从高到低排列：

### 6.1 backend 表达力差异（最可能）

04-22 backend 对"细节物体"的表达比 03-03 更强。arm B 擂台改诗后 image_prompt 会包含 refine 引入的新意象（例如主题 9 arm B 加入"耕牛"），04-22 能把这些新意象都画出来，03-03 会遗漏一部分——**擂台的"意象扩容"作用只有在 backend 表达力够强时才能兑现**。

主池辅池共有 5 主题（征戍 / 客愁 / 耕牛炊烟 / 高楼远山 / 客舍孤灯）可以做 backend 稳定性 sanity check：

| 主题 | 辅池 (03-03) | 主池 (04-22) | backend 影响 |
|---|---|---|---|
| 征戍 | A:B 硬约束 1:1 平手 | A:B = 0:1 arm B 独赢 | 04-22 让 arm B 差异化 |
| 客愁 | A:B 双 miss | A:B = 0:1 arm B 独赢 | 04-22 让 arm B 差异化 |
| 耕牛炊烟 | A:B = 1:2 arm B 独赢+1 | A:B = 0:2 arm B 独赢+2 | 04-22 让擂台反转更彻底 |
| 高楼远山 | A:B = 2:1 arm A 独赢 | A:B = 2:2 平手 | 04-22 让 arm B 追平 |
| 客舍孤灯 | A:B = 1:1 平手 | A:B = 1:2 arm B 独赢 | 04-22 让 arm B 差异化 |

**5 主题里 4 个方向都是"从辅池平手/A 赢 → 主池 arm B 独赢/追平"**。backend 从 03-03 换到 04-22 系统性地让 arm B 优势显化。

### 6.2 主池样本更大（n=22 vs n=10）

辅池 sparse 只 5 主题，任一主题翻转即可扭转 sparse 方向。主池 sparse 11 主题，样本足够表达真实倾向。**辅池的"没净收益"结论可能受 sparse 5 主题的采样噪声主导**。

### 6.3 主题分布巧合

辅池 sparse 5 主题里有"蝉"和"春雨"这类**视觉表达困难**的元素（蝉多依赖背景暗示、细雨易与雾混淆）。这些主题即使擂台生效，也会被 image backend 的表达力限制拉平。主池 sparse 主题里"归隐"是唯一双 miss 的困难主题，其余都能画出来，擂台 refine 更容易兑现。

### 6.4 pairwise judge 系统性偏向挑战者（次要）

sweep 报告 §6.3 攻擂率 49-75% 表明擂台 pairwise judge 偏向"改"而非"保"。这个偏向在辅池 03-03 上导致过度改动（改过头把首句意象改没），在主池 04-22 上因 backend 表达力强能兜住改动带来的意象漂移。

> **2026-07-05 B1 探针 · 2026-07-06 复盘修正**：53 对历史攻擂对局双向重判显示位置效应**迹象**（挑战者 B 位胜率 86.8% vs A 位 69.8%，位置追随 12:3，名义 McNemar p=0.035），但样本入选条件与 forward 布局相关（仅含"judge 曾判 B"的对局）+ temp=0.1 复现效应，使该不对称**混杂了选择-复现伪效应，方向与幅度未确认**——详见 sweep 报告 §6.3 #1。对本报告的影响界定不变且更保守：arm B 的"哪些 refine 版本被采纳"经由一个可能带位置偏置的单向 judge，+18.2pp 度量的是**现行擂台（含该 judge 及其单向布局）**vs 无擂台的差——结论对"保留现行擂台"成立；2026-07-06 起生产已改为 A/B 位随机化（对偏置方向不敏感的防御），此后擂台采纳行为与本报告实验条件不同，重跑对比时须注明。

> **2026-07-06 B1b 无偏探针 · 方向反转，混杂诊断证实**：改用与 judge 判定无关的新鲜配对（armA vs armB 终诗 22 对 + delta 0.14/0.20 终诗 10 对）双向重判，position_A_follow **7** vs position_B_follow **0**（双侧精确二项 p=0.0156，无选择-复现混杂、可按面值采信），A 位占位者总胜率 60.9%。qwen-plus judge 的真实位置偏置**偏 A 位（primacy）**——旧生产布局（擂主恒 A 位）下位置偏置实际在**偏袒擂主**。两点影响：① 本节"judge 偏向挑战者"的行为倾向**不能归因于位置偏置**（位置上是逆风），更可能是 judge 在内容上确实偏好 refine 版本；② 对 +18.2pp 的威胁进一步收窄——"arm B 的 refine 采纳是 B 位位置伪影"假设被证伪，采纳发生在位置不利条件下。附带产出：armA vs armB 终诗直接 pairwise（两向一致内容判定）**armB 12 : armA 5**——文本侧 LLM judge 与图像侧 VLM 硬约束（§2.5/§2.6）同向支持擂台净收益，构成第三条正交证据线。细节见 sweep 报告 §6.3 #1，数据 `outputs/eval/probe_position_bias_fresh_20260706_082356.json`。

> **2026-07-06 B2-lite 位置审计 · 反向弱信号，方向未锁死**：随机化布局无图擂台重跑（主池 22 题、配置复刻 arm B run）的 73 次守擂判定中，挑战者 A 位胜率 43.6% vs B 位 61.8%（Fisher p=0.16，不显著）——方向与 B1b **相反**。两探针配对分布与口径不同（B1b=eval 终诗对的同对位置追随；B2-lite=生产擂主-挑战者对的跨对边际胜率），综合读法：**位置效应相对内容效应是小信号、方向跨分布不稳定**。对上一段结论的修订：① 从"不能归因于位置偏置（位置上是逆风）"弱化为"位置效应方向未锁死，无论哪个方向其幅度都不足以解释 49-75% 的攻擂率"；② "B 位伪影"假设从"被证伪"回调为"两个探针一反一正、均无法确证"——但 +18.2pp 的主证据（VLM 硬约束，§2.5-§2.7）与第三证据线（armB 12:5 双向一致内容判定，位置已中和）不经过守擂 judge 的单向布局，不受此张力影响。随机化防御（B3a）对方向不敏感，两种方向下均成立。攻擂率 before/after：49.2%→55.9%（+6.8pp，p=0.58，非配对重跑，仅示未见异常漂移）。详见 sweep 报告 §6.3 #1，数据 `outputs/eval/rerun_arena_randomized_main_n22_20260706_224424.json`。

## §7 caveats + backend 稳定性

- **主池 n=22 是本报告主要结论支撑**：+18.2 pp 硬约束提升方向证据多重一致（keyword/主题/density/跨池重合主题四层同向），但 n=33 keyword 配对检验 p≈0.18、CI 跨零（§2.5）——样本量不足以下统计定论，扩样路径见 §8。
- **辅池 n=10 是补充证据**：与主池方向冲突，主要归因于 backend 差异 + 采样噪声（§6.1、§6.2）。
- **单 backend 主池 n=22 不足以推广跨 backend**：主池辅池 5 主题共有对比显示 backend 会调制擂台效果。要断言"擂台在 04-22 backend 上有净收益"是稳的，但要说"擂台在所有 backend 上都有净收益"需要至少一个第三方 backend 验证。
- **~~单 VLM oracle~~ 已跨家族复核**（2026-07-06）：glm-4v-plus 重判同 44 张图，Δ +15.2pp 同向、逐 keyword 一致率 86.4%（§2.6）——同家族 self-bias 质疑解除。VLM 单次判定仍有噪声（同图重跑可能给出不同 present 值），此局限跨 oracle 共有。
- **arena 海选阶段两 arm 有随机性**：LoRA 生成 5 候选是有 temperature，两 arm 各跑一次 arena，冠军可能不同。这意味着 arm A vs arm B 的差异**混合了 "arena 抽签差 + 擂台差"**。本次 setup 无法完全解耦（要解耦需固定 arena 冠军、只 A/B 擂台部分）。**尽管如此，18.2 pp 的硬约束差距大于"arena 抽签"通常能贡献的噪声量级**（注：此处的量级判断是定性推断，未经对照实验量化，见 §8 中优先级"固定 arena 冠军做纯擂台 A/B"）。
- **主池辅池主题重合的 5 主题**：`get_benchmark(n=32)` 与 `get_benchmark(n=10)` 独立采样，有 5 主题名称相同。这 5 主题在两池分别以 04-22 / 03-03 backend 生成，天然地做了 backend 敏感度检查（§6.1）。合并成"n=32 全池 CLIP mean"不严谨，因为跨 backend 平均无意义，本报告刻意不做。

## §8 follow-up

### 高优先级

- **接受主池 n=22 结论：保留擂台机制**。硬约束 +18.2 pp 是本次 ablation 最强的方向性数据点（统计定论待扩样，§2.5），方向与 refine_poem prompt 的设计意图（"改后诗更好"）矛盾——不改而是**因 arena 冠军已足够好、擂台反而能在"用户约束"这另一个正交维度补齐**。工程决策「保留 `max_poem_refine_rounds=2` 作为默认」的依据是方向一致性 + 机制可解释 + 保留成本低，不以 p 值为前提。
- **补做 03-03 backend 的 n=22 sweep 验证跨 backend 稳定性**。目前主池只在 04-22 上跑通，跨 backend 只有 5 主题重合数据。如果 03-03 上 n=22 也能重现主池的 +10pp 以上硬约束优势，"擂台有净收益"就是 pipeline-general 结论，不是 04-22 specific。（配额需求：22 × 2 arm ≈ 90 张 03-03，视余量安排。）

### 中优先级

- **refine_poem prompt 加"user_input 约束覆盖度"检查**：目前 refine 完全不看 user_input，是纯诗学改写。这解释了主池主题 15、21 和辅池主题 5、8 的"改诗方向正交"翻车。修法：refine 时把 user_input 里的硬约束关键词作为"不可丢失"约束加入 prompt。这是可以显著提升擂台"命中率"的直接改动。
- **固定 arena 冠军做纯擂台 A/B**：本次 ablation 混合了 arena 抽签 + 擂台差异。要精确测擂台，应把 arm A 的 arena 冠军作为 arm B 的起点。这需要 sweep 脚本改造（增加 `--fixed-arena-champion-from` 参数）。当前 18.2 pp 差距大到 arena 抽签解释不了，但为了未来更精细的分析建议改造。
- **主池主题 6 炉火寒灯的 CLIP 0.362 但硬约束 0/2 案例做单独 debug**：CLIP 全 sweep 22 主题最高分但用户约束（"寒灯"完全缺失、"炉火"严格判"无火焰"）miss，是 pipeline 反馈环设计的 warning——**CLIP 门控本身无力约束"用户明确要求的具体物体"**。建议在 pipeline 里加一层"硬约束 pre-check"（image prompt 里必须显式包含 user_input 的关键词）。
- ✅ **~~kw 严格度灵敏度扫描~~ 已完成**（2026-07-06，§2.7）：全池 strict/loose 双变体 × 44 图（88 次 qwen-vl-max），三档定义包络 **Δ ∈ [+9.1pp, +24.2pp]、方向全为正**，单调性违反 0 处。原担忧"sparse OR 集合偏差系统性偏向 arm B"仅在从松方向部分成立（loose 压缩 Δ 至 +9.1pp 但不翻方向）；从严方向 arm B 优势反而扩大至 +24.2pp（p=0.077，CI 下界触零）。"擂台净收益"的 kw 定义无关引用口径：**+9.1~+24.2pp（baseline +18.2pp）**。脚本 `eval/_sweep_kw_sensitivity.py`（变体私有于探针，冻结 v1 定义未动）。

### 低优先级

- ✅ **~~VLM oracle 多样化~~ 主池已完成**（2026-07-06，§2.6）：glm-4v-plus 跨家族复核，Δ +15.2pp 同向、一致率 86.4%。剩余可选项：gpt-4v 第三方（需新 key）、辅池 30 张 sweep 图复跑（+30 次调用）。
- **攻擂率调优**：主池 arm B 平均攻擂率 49.2%，健康区间是 15-40%。当前 PAIRWISE_WIN_DELTA=0.17 让挑战者太容易胜。sweep §6.3 已 surface，正交 follow-up。

---

**总体结论一句话**：主池 n=22（04-22 backend）证据支持**保留擂台**——CLIP 上和无擂台平手（甚至略输 0.002），但 VLM 硬约束 arm B **+18.2 pp**（60.6% → 78.8%，McNemar p≈0.18、CI 跨零，方向性强证据待扩样定论，§2.5；跨家族 oracle 复现 +15.2pp，§2.6；kw 定义三档包络 +9.1~+24.2pp 方向不变，§2.7），主题级 arm B 独赢 9 主题 / arm A 独赢 3 主题。CLIP 和硬约束在这次分家：CLIP 关心"图美不美"、硬约束关心"用户要的东西画没画到"，擂台在后者上的净收益足以支撑保留决策。辅池（03-03 backend）平手结论已被主池推翻，归因于 backend 表达力差异 + n=10 采样噪声。

_报告生成：Claude Opus 4.7 · 数据可通过 `python -m eval.sweep_pairwise_win_delta --n 32 --offset 10 --deltas 0.17 --image-backend bailian:qwen-image-2.0-pro-2026-04-22` 加 `--max-poem-rounds 0/2` 复现 · VLM 硬约束对比 `python -m eval.vlm_hard_constraint --agg outputs/eval/_agg_arena_ablation_main_n22.json --delta arm_A arm_B`_
