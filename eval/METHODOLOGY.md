# eval_poem 方法论 · frozen reference

> **本文档锁定 commit `5c13530` (代码) + `fb19f43` (scorer fix) 时的诗歌评估方法论。**
> 用作 `REPORT_main_n32x3run_20260624.md` 的解释手册。后续如果 prompt / 公式 / 系数被改动，**先更新此文档再改代码**，并 bump 顶部锚定的 commit hash。

---

## 1. 实验设计

```
┌── 用户题目 (n=32) ────────────────────────────────────────┐
│                                                            │
│  ┌─[每模型生成 5 候选]───┐                                  │
│  │ local_base           │                                   │
│  │ local_lora (full)    │                                   │
│  │ local_lora (naked)   │ → 本地规则 scorer (确定性)         │
│  │ qwen-plus            │   ↓                              │
│  └──────────────────────┘   ↓                              │
│                              ↓                              │
│  ┌─[BWS 选 best · 4 评委独立投票]────┐                       │
│  │ deepseek-v4-pro                  │                       │
│  │ qwen-max                         │ → 多数决 + 平票兜底     │
│  │ glm-4-plus                       │   ↓                   │
│  │ moonshot-v1-32k                  │   ↓                   │
│  └──────────────────────────────────┘   ↓                   │
│                                          ↓                   │
│  ┌─[best multi-judge 4 维分]────────────┐                    │
│  │ intent / imagery / cohesion /        │ → 中位数合成        │
│  │ aesthetics × 4 评委                  │                    │
│  └──────────────────────────────────────┘                    │
│                                                              │
│  ┌─[跨模型 pairwise 双向 · 4 评委独立判]─┐                    │
│  │ 6 对 × 4 评委 × forward+reverse      │ → 一致性票多数决    │
│  └──────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
         × 3 个独立 run → 跨 run mean ± std
```

**理由 / 设计选择：**

- BWS（Best-Worst-Scaling 简化版：N 选 1）替代绝对评分 → 规避 LLM 评分饱和（5 首诗全打 7.5-9.0）。
- 跨家族 4 评委（DeepSeek + Qwen + GLM + Moonshot）→ 避免 self-bias（防止 Qwen-Max 当评委时偏向 Qwen-Plus）。
- forward+reverse pairwise → 暴露 position bias（评委是否永远偏 A 位 / B 位）。
- 3 run mean ± std → 暴露 LLM noise，证明信号 vs 噪声可分离。
- 评委弃权语义（解析失败 → None，不污染合成）→ 避免 silent 0.5 / 0.3 污染（详见 scorer fix 提交 `fb19f43`）。

---

## 2. 数据集 (n=32)

文件：`eval/dataset.py` · 导出 JSON：`eval/benchmark_themes.json`

**分层结构（互斥 12 主题）：**

| 类别 | 数量 | 主题 × 体裁 |
|---|---|---|
| 写景 | 16 | 春/夏/秋/冬 各 4 道（× 4 体裁），每季 rich:sparse = 2:2 |
| 非写景 | 16 | 山水 / 田园 / 边塞 / 羁旅 / 送别 / 怀古 / 节令 / 哲理 各 2 道（× 2 体裁），每主题 rich:sparse = 1:1 |

**rich vs sparse：**
- `rich`（16 道）：明确给出 ≥2 个具体意象（如"要有柳树和燕子"）→ 必须意象检查会扣分，CLIP 双锚点常规权重
- `sparse`（16 道）：只给抽象主题词（如"主题是春雨"/"主题是无常"）→ CLIP 双锚点切到稀疏权重

**体裁分布（每体裁 8 道）：** 五绝 / 七绝 / 五律 / 七律 各 8。

**n<32 切片规则（stratified round-robin）：**
- 按 (genre, density) 12 个桶分组
- 桶内 scenic / non-scenic 交替，相邻桶起点反相 → n=8 切片仍保 4 体裁 × 2 density × 2 scenic 三层均衡

---

## 3. 候选生成参数

| 项 | 值 | 出处 |
|---|---|---|
| `POEM_CANDIDATE_COUNT` | 5 | `config.py` |
| `POEM_MAX_TOKENS` | 160 | `config.py` |
| `POEM_TEMPERATURE` | 0.8 | `config.py` |
| `LORA_MAX_SEQ_LEN` | 320（prompt + output 总窗口）| `config.py` |

**Prompt 模式（`local_*` 模型适用）：**
- `full`：本地模型接收与 API 相同的 system + 格式约束 + 用户要求（**controlled experiment**，公平对比）
- `naked`：本地模型只接收简短 user request（**LoRA ablation**，验证微调是否内化格式引导）

报告中：`local_base` / `local_lora` 用 full · `local_lora_naked` 用 naked · `qwen-plus` 用 full。

---

## 4. 本地规则评分器（确定性，每候选都跑）

文件：`core/poem/scorer.py`

### 4.1 各维度归一化到 [0, 1]

| 维度 | 算法 | 阈值/系数 |
|---|---|---|
| `pingze`（平仄）| 对比该体裁的全部格律模板（`PATTERN_5_4`/`PATTERN_5_8`/`PATTERN_7_4`/`PATTERN_7_8`），取最佳匹配模板下的 **合规句数 / 总句数**。单句合规为宽松判定：只查偶数位（五言第 2、4 字，七言加第 6 字，即「一三五不论，二四六分明」），声调无法识别的字跳过算通过 | 取值离散，见 §10 |
| `rhyme`（押韵）| 偶数行末字归入平水韵部，按**韵部种类数**三档取值：1 种 → 1.0，2 种 → 0.6，≥3 种 → 0.3 | 取值仅 {0.3, 0.6, 1.0}，见 §10 |
| `imagery`（意象库）| 命中 `ALL_IMAGERY_WORDS` 词表的数量 → log 归一 | — |
| `cohesion`（主题连贯）| 句间意象重叠 + 情感主题归类 | — |
| `intent`（意图）| **LLM 评分**（见 §5）| — |

### 4.2 加权公式（multi-judge 版，`score_single_multi_judge`）

```
raw_total = intent     × 0.30
          + pingze     × 0.25
          + rhyme      × 0.15
          + imagery    × 0.10    ← multi-judge 把 imagery 从 0.15 匀 0.05 给 aesthetics
          + cohesion   × 0.10    ← 同上
          + aesthetics × 0.10    ← multi-judge 新增维度（来自 LLM 评委）
                     (∑ = 1.00)

penalty_c = max(repetition_penalty,  0.7)    ← 重复字符惩罚
clash_c   = max(synonym_clash_penalty, 0.7)  ← 合掌防御（同义词同联）
req_c     = max(required_keywords,    0.7)   ← 必须意象覆盖率

final_total = raw_total × penalty_c × clash_c × req_c
```

**常量** (`config.py`)：

| 常量 | 值 | 说明 |
|---|---|---|
| `WEIGHT_INTENT` | 0.30 | |
| `WEIGHT_PINGZE` | 0.25 | |
| `WEIGHT_RHYME` | 0.15 | |
| `WEIGHT_IMAGERY` | 0.15 | multi-judge 用 0.15 - 0.05 = **0.10** |
| `WEIGHT_COHESION` | 0.15 | multi-judge 用 0.15 - 0.05 = **0.10** |
| `w_aesthetics` | 0.10 | multi-judge 专用，hardcoded in `score_single_multi_judge` |
| `SCORE_PENALTY_FLOOR` | 0.7 | 单惩罚因子下限，防三因子叠乘溶解 (0.7³=0.343) |
| `CLASH_PENALTY_PER_HIT` | 0.75 | 每发现 1 处合掌（同联同义词），品质分 ×0.75 |
| `POEM_QUALITY_THRESHOLD` | 0.70 | `pass@0.7` 候选合格阈值 |

### 4.3 重复惩罚 / 合掌词典

- 重复字符超过阈值 → `penalty ∈ [REPETITION_PENALTY_MAX=0.15, 1.0]`
- 合掌词典（同联出现同义词组 ≥ 2 个 → 算合掌）：
  - 月（玉轮/兔钩/明月/素月/冰轮/蟾宫/玉盘/婵娟/皓月/新月/月色/月光）
  - 银河（银汉/星河/银河/天河/星汉/云汉/玉绳/北斗）
  - 荷（红蕖/荷花/菡萏/芙蓉/荷叶/碧莲/水芙蓉）
  - 雁（归雁/鸿雁/征雁/孤鸿/飞鸿/雁阵/雁字）
  - 落日（残阳/落日/夕阳/斜阳/暮日/夕照）
- 必须意象：从 user_request 抽取"要有X和Y"模式，过 `get_imagery_synonyms` 查同义词表，每缺 1 个 → `coeff ×= 0.75`

### 4.4 required_coeff 设计意图（与 LLM intent 分工）

> 编号勘误：本节与下节曾误标为 §4.1/§4.2（与上文维度表/加权公式重号），2026-07-05 重编号为 §4.4/§4.5。旧提交信息 / 讨论中引用的「§4.1 required_coeff 设计意图」「§4.2 THEME_SYNONYMS 版本快照」即指本两节，内容未变。

- **LLM 意图评分（`WEIGHT_INTENT`=0.30，为加权占比最高维度）是主判**：负责"意思到没到"的语义判断，抽象主题（思乡/壮志/无常）能覆盖，是主要打分来源。
- **`required_coeff` 是硬约束保险**：只对具象要求（"要有柳树和燕子"这类）字面/同义词命中扣分，兜底 LLM 判"意到但字面漏"的情况。
- **false negative（同义词表漏收）会误扣**，但对所有模型/所有 arm 一视同仁 → 是 **fair penalty**，不影响相对排名，只影响绝对分值。
- **有针对性地补表、不追求覆盖全古诗意象**（词表版本见 §4.5）：报告全用相对提升（pass@0.7 / winner Δ / arena Δ），绝对分值偏低不影响任何结论；版本管理成本由 §4.5 版本快照承担。
- **`SCORE_PENALTY_FLOOR = 0.7`**（`config.py`）给多重扣分下限，防止一首诗被 penalty × clash × required 三重压死。

### 4.5 THEME_SYNONYMS 版本快照

`core.poem.theme.THEME_SYNONYMS` 是 `required_coeff` 判"用户明说的意象在诗里是否命中"的同义词底表。改表会让老报告的绝对分数漂移，需要冻结版本以便复现。

**版本 v2 · 2026-07-04**（当前生产，commit 追 `A1 THEME_SYNONYMS 补充`）：
- 37 主槽。相对 v1 新增 12 槽：`雨 / 云 / 烟 / 酒 / 灯 / 笛 / 渔 / 舟 / 钟 / 夕阳 / 故乡 / 愁`；`月` 槽内补 `圆月`（对齐 `vlm_hard_constraint.py` 的 `明月（圆月）` 判定 kw）。

**版本 v1 · 2026-07-04 之前**（本 repo 内所有 `outputs/eval/*` 与 `eval/REPORT_*.md` 老报告依赖此版）：
- 24 主槽：`雪 / 春 / 夏 / 秋 / 冬 / 梅 / 荷 / 菊 / 月 / 山 / 水 / 桃 / 柳 / 燕 / 蛙声 / 归雁 / 竹 / 松 / 兰 / 风筝 / 纸鸢 / 萧瑟 / 凄凉 / 星 / 夜`。
- 若需按此版本复现某份老报告的绝对 `required_coeff` 值，`git show <commit-before-v2>:core/poem/theme.py` 可拿到完整词表。

**已冻结报告不重跑的理由**：所有已发布 `REPORT_*.md`（含 `REPORT_arena_ablation_20260701.md`、`REPORT_main_n32x3run_20260624.md`、`REPORT_pairwise_win_delta_sweep_2026-06-30.md`、`REPORT_eval_refine_LoRA_n32_20260702.md`、`REPORT_vlm_hard_constraint_20260701.md`、`REPORT_eval_clip_dual_anchor_20260623.md`、`REPORT_autonomous_n5_20260627.md`）主结论都基于**相对指标**（pass@0.7 / winner Δ / arena Δ / hit-rate Δ），词表扩容仅平移绝对分数、不改变相对方向。v2 首次生效在本次 commit 之后的所有新 eval。

---

## 5. LLM 评委 4 维 rubric（best 候选用）

调用：`score_4dim_via_llm()` · 模型温度 0.1 · `max_tokens=120`

### 5.1 评委 system + user prompt 全文

```
system: 你是严格的文学评委，只输出一行逗号分隔的标签值对。
```

```
user:
请根据以下用户要求，对古诗进行评分（以鼓励为主，宽容评价，允许0.5分精度）。
用户要求：{user_request}
古诗：{poem}

评分标准（满分10分，每位维度允许0.5分步进，如1.5、2.5）：
1. 主题匹配度（0-3分，0.5精度）：
   - 2.5-3分：完全围绕用户指定主题，核心意象贯穿全诗。
   - 2-2.5分：通过意象、氛围、季节等有效传达了主题，不必苛求主题词字面出现。
   - 1-1.5分：主题基本可辨，表达较隐晦或部分偏离。
   - 0-0.5分：主题完全不符，全诗与用户要求无关联。
       注意：若用户指定了季节（如"夏夜""春日"）而诗中出现了矛盾季节意象（如夏夜写"秋吟""秋堂"），主题匹配度应酌情扣1分。
2. 意象完整性（0-3分，0.5精度）：
   - 2.5-3分：用户要求的必须意象全部鲜明出现且自然融入。
   - 2分：意象出现但较平淡，或有等价意象替代。
   - 1-1.5分：意象勉强涉及但不够明确。
   - 0-0.5分：用户要求的意象完全未出现，也无可替代的等价意象。
3. 意境连贯度（0-2分，0.5精度）：
   - 2分：诗句流畅，意境统一，起承转合自然。
   - 1-1.5分：整体可读，偶有断裂或跳跃。
   - 0-0.5分：逻辑混乱，意象堆砌无关联。
4. 语言优美度（0-2分，0.5精度）：
   - 2分：用词典雅生动，画面感强，遣词精当。
   - 1-1.5分：用词尚可，有一定诗意。
   - 0-0.5分：用词直白、俗套或生硬。

输出格式：主题匹配度:数字,意象完整性:数字,意境连贯度:数字,语言优美度:数字,总分:数字
例如：主题匹配度:2.5,意象完整性:3,意境连贯度:1.5,语言优美度:1.5,总分:8.5
只输出这一行，不要额外解释。
```

### 5.2 解析与归一化

正则：`(主题匹配度|意象完整性|意境连贯度|语言优美度|总分)\s*[:：]\s*(\d+(?:\.\d+)?)`

| 维度 | 原始分制 | 归一化 |
|---|---|---|
| `intent`（主题匹配度）| 0-3 | `min(v, 3) / 3` |
| `imagery`（意象完整性）| 0-3 | `min(v, 3) / 3` |
| `cohesion`（意境连贯度）| 0-2 | `min(v, 2) / 2` |
| `aesthetics`（语言优美度）| 0-2 | `min(v, 2) / 2` |
| `total` | 0-10 | LLM 给的 `min(v, 10) / 10`；缺失则用 `0.3·intent + 0.3·imagery + 0.2·cohesion + 0.2·aesthetics` |

### 5.3 解析失败语义（关键 · `fb19f43`）

```
adapter 异常 / 4 维 + 总分都未匹配 → 返 {intent: None, imagery: None, cohesion: None, aesthetics: None, total: None}
                                  ↓
                          上层 multi-judge 合成时该评委整体弃权，不污染中位数
```

**旧版（已废弃）：** 返 `{all: 0.5}` 或从 reasoning model 长 CoT 里"找最后一个 0-10 数字"做兜底总分 —— 会把 deepseek-v4-pro 偶尔返回的 1480 字 reasoning 解析成总分 0.3，silent 拉低评分。

### 5.4 评委合成（≥3 中位数 / <3 均值 / None 跳过）

```python
def _aggregate(dim_key, rule_fallback):
    vals = [j[dim_key] for j in scores_by_judge.values() if j[dim_key] is not None]
    if not vals:          return rule_fallback           # 全员弃权 → 规则版兜底
    if len(vals) >= 3:    return statistics.median(vals) # 抗异常
    return sum(vals) / len(vals)                         # < 3 用均值
```

---

## 6. BWS 选 best（每模型 5 候选 → 1）

调用：`pick_best_via_bws()` · 温度 **0.0**（确定性）· `max_tokens=10`

### 6.1 Prompt 全文

```
system: 你是古典诗词评委。仅输出阿拉伯数字，禁止任何文字、解释、推理。
```

```
user:
请从下列 {n} 首古诗中选出最佳的一首。
【输出格式】严格仅返回 1-{n} 之间的一个阿拉伯数字，不允许任何其他字符、解释、推理过程、标点。
【输出示例】3

用户要求：{user_request}

{poems_block}    ← 候选诗块（每首前缀 "诗 i:"）

评判维度（同等重要）：主题契合、意象意境、语言典雅、格律合规。
{imagery_grounding}   ← 古典意象字典 in-context grounding（见下）

你的回答（仅一个数字）：
```

### 6.2 In-context grounding（古典意象字典）

`eval/assets/classical_imagery.json` 词条按"植物 / 动物 / 天象时令 / 情感典故 / 地理场景 / 人物典故 / 器物 / 复合意象"分类拼接成附录文本，注入 prompt 末尾。**目的：** 让评委识别"含蓄用典"（诗里用"折柳"而非字面"送别"），不因字面没命中扣分。

### 6.3 候选打散（position bias 防御）

每个评委调用前对 5 候选做 `random.shuffle(perm)`，评委选 shuffled_idx 后映射回 original_idx。

### 6.4 投票聚合 + 平票兜底

```
1. 4 评委独立投票 → vote_count[idx]
2. 弃权（解析失败 / 异常）= -1 → 不计票
3. 若全员弃权 → 按本地 total 兜底 best_idx
4. 否则 → max_votes 多数决
5. 若多个 idx 同票 → 按本地 total 兜底
```

### 6.5 解析规则（reasoning model 适配）

```python
1) 整段回复就是单一数字 → 用之（理想）
2) 否则从回复末尾向前找最后一个独立 1..n 数字
   regex: (?<![0-9])([1-9][0-9]?)(?![0-9])
   理由: reasoning 模型推理之后才给答案，且要避开 prompt 复述里的 "5 首""第 1 首"
3) 都没有 → -1 弃权
```

---

## 7. 跨模型 pairwise（每对 best · 双向）

调用：`compare_poems()` × 2 (forward + reverse) · 温度 0.1 · `max_tokens=10`

> ⚠ **BREAKING · 口径变化（2026-07-05）**：`compare_poems` 引入弃权语义——adapter 异常或回复不含唯一 A/B 时返回 `None`（此前静默返回 `"A"`）。影响：
> ① eval 侧新增「弃权」桶（任一方向 None → 该评委该对决弃权，不计入摇摆/胜负；全员弃权 → `all_abstain`，与 `all_swing` 同样排除出胜率分母）；
> ② **历史报告（含 `REPORT_main_n32x3run_20260624.md` 的摇摆率 23%）把 API 失败与 position bias 混在同一个"摇摆"桶里**，与新口径的摇摆率不严格可比——旧数据不追溯重算，引用跨期对比时须注明；
> ③ 生产擂台侧：判定失败从「隐式判 A（擂主）胜」改为「该挑战者作废、显式记日志守擂」，方向一致但计分路径不同（旧路径挑战者会吃 `PAIRWISE_LOSE_DELTA` 惩罚后仍参与比较，新路径直接作废）。
> 动机：对齐 §5.3 BWS / 4 维评分自 `fb19f43` 起的弃权设计，消除"评委失败被当成 A 胜"的计票污染。

> 📌 **摇摆率旁证（2026-07-06 B1b 无偏探针）**：新口径（弃权已分桶，本次 abstain=0）下，32 对新鲜配对（armA/armB + delta 两档终诗，与 judge 判定无关）双向重判实测摇摆率 **21.9%**，与历史混杂口径的 23% 数值接近——提示历史"摇摆"桶的主要成分是真实 position bias 而非 API 失败。方向上 7 例位置追随全偏 A 位（p=0.0156，primacy；同日 B2-lite 生产分布位置审计给出反向弱信号，方向未锁死，通俗版故事线见 §7.5）。注意可比性是提示性的：B1b 为单 judge（qwen-plus）、诗对分布也不同于 n=32 主跑的跨模型对局；追随样本仅 7 例，作旁证不作定论。数据：`outputs/eval/probe_position_bias_fresh_20260706_082356.json` · 脚本 `eval/_probe_position_bias_fresh_pairs.py` · 详见 sweep 报告 §6.3 #1。

### 7.1 Prompt 全文

```
system: 你是一位严苛的古典诗词评委。只输出 A 或 B。
```

```
user:
请比较以下两首古诗，判断哪一首更符合用户的要求。

用户要求：{user_request}

诗歌 A：
{poem_a}

诗歌 B：
{poem_b}

比较维度（同等重要）：
1. 主题契合度：哪一首更准确、更完整地表达了用户要求的主题？
2. 意象与意境：哪一首的画面感和意境更鲜明、统一？
3. 语言质量：哪一首的用词更典雅、精炼、富有诗意？
4. 格律合规：哪一首更符合平仄和押韵规范？

只输出一个字母：A 或 B。不要输出任何其他内容。
```

### 7.2 双向一致性判定

| forward | reverse | 含义 | 计票 |
|---|---|---|---|
| A | B | 两次都偏向 `model_a` | `ma_votes += 1` ✅ |
| B | A | 两次都偏向 `model_b` | `mb_votes += 1` ✅ |
| A | A | 永远选 A 位 → position bias | `swing += 1` ❌ |
| B | B | 永远选 B 位 → position bias | `swing += 1` ❌ |

### 7.3 单对模型 winner

```
if ma_votes > mb_votes:           winner = model_a
elif mb_votes > ma_votes:         winner = model_b
elif ma_votes == 0 == mb_votes:   winner = "all_swing"   # 4 评委全摇摆
else:                             winner = "tie"          # 同票
```

### 7.4 单模型胜率公式

聚合所有 `n × C(4,2) = 32 × 6 = 192` 对决（每 run）：

```
wins   = 该模型作为 winner 的次数
losses = 该模型作为 loser 的次数
ties   = 该模型参与的 tie 次数
all_swing = 该模型参与的 all_swing 次数
plays  = wins + losses + ties + all_swing

胜率 = (wins + 0.5 × ties) / (wins + losses + ties)    ← 决断场次内
评委有效胜票占比 = consistent_votes_for_model / (consistent + swing)
```

`all_swing` 单独列出但不进胜率分母（无信息量）。

### 7.5 位置偏置故事线（通俗版 · 2026-07-06 收档）

> 本节是给非本项目读者的无术语版本；每一步的严谨表述与数据见 sweep 报告 §6.3 #1。

系统里有个 AI 评委，每轮拿两首诗问它"A 和 B 哪首好"。生产擂台的旧规则是**擂主永远
坐 A 位、挑战者永远坐 B 位**——如果评委对"座位"本身有偏心，比赛就系统性不公平。
围绕这个疑点先后做了四步：

1. **第一次测量（B1，2026-07-05）**：拿历史攻擂对局换座位重判，表面结果"评委偏 B 位"
   且统计显著。**但复盘发现测量本身有坑**：这批对局全是"当年评委判了 B"才入选的，
   低温重判又高度复现原判——零偏置也能测出同样形状的"偏 B"。显著性作废，自我降级。
2. **修正后的测量（B1b，2026-07-06）**：改用与评委判定无关的新鲜诗对，方向**反转**——
   7 例"跟着座位改口"的案例全部偏 **A** 位（p=0.0156）。即旧布局其实在**偏袒擂主**。
3. **生产分布上的复核（B2-lite，同日）**：把擂台整个重跑一遍、座位改抛硬币分配，
   顺手统计挑战者抽到 A 位 vs B 位的胜率：43.6% vs 61.8%——方向又和 B1b **相反**
   （不显著，p=0.16）。
4. **最终读法**：两次无混杂的测量一个说偏 A、一个说偏 B，都不够铁。诚实的结论是
   ——评委的座位偏心是个**小信号**，比"诗本身好不好"的影响小得多，小到换一批诗、
   换一种测法，方向都会翻。就像用体重秤称羽毛：今天 +2 克明天 −1 克，只能确定
   羽毛很轻，确定不了它到底几克。

**为什么这反而是好消息**：生产侧的防御从一开始选的就是**随机化座位**（B3a，抛硬币
定位），而不是"按测出的方向做修正"。修正方案赌方向——方向翻了就把偏差加倍；随机化
不需要知道评委偏哪边、偏多少，长期两边各被偏一半自动抵消，**两个探针方向再怎么打架
它都成立**。附带验证：随机化前后攻擂率 49.2% → 55.9%（不显著），未见异常漂移。

这段故事的方法论价值：发现可疑偏差 → 测出"显著" → 自己找出测量的坑并作废 →
重新设计实验方向反转 → 再验证发现方向锁不死 → 而工程决策因为不依赖结论方向，
每一步都不用返工。**结论一直在变，防御一次都没改。**

数据：`outputs/eval/probe_position_bias_20260705_235340.json`（B1）·
`probe_position_bias_fresh_20260706_082356.json`（B1b）·
`rerun_arena_randomized_main_n22_20260706_224424.json`（B2-lite）

---

## 8. 多 run 聚合（`--repeat N`）

文件：`eval/eval_poem.py` · 函数 `_aggregate_across_runs`

每 run 独立跑完整 pipeline（不同 LLM noise → 不同候选 → 不同评委判断）。

### 8.1 指标分类与聚合方式

| 类别 | 例子 | 聚合方式 |
|---|---|---|
| **标量**（指标）| `pingze_pass@0.8`, `rhyme_pass@0.8`, `consistency_rate` | 跨 run mean ± std |
| **summary**（已含 mean/std/...）| `mean_total`, `intent`, `imagery`, `cohesion`, `aesthetics`, `pingze`, `rhyme` | 取每 run 的 `mean` → 再算跨 run mean ± std |
| **rate**（胜率）| `pairwise_win_rate`, `judge_consistent_pick_rate` | 跨 run mean ± std + 列每 run 原值 |
| **int 累计**（场次）| `pairwise_wins`, `losses`, `ties`, `all_swing`, `plays` | 跨 run **加总**（用 `_total` 后缀字段） |

### 8.2 渲染 `± 0` 折叠

`_fmt_pm`：当 `repeat=1` 时所有指标 std=0，自动折叠为单数字，避免 "0.771 ± 0.000" 噪声。

---

## 9. 关键命令 & 复现

```bash
python -m eval.eval_poem \
    --models local_base local_lora local_lora_naked qwen-plus \
    --scorer deepseek-v4-pro qwen-max glm-4-plus moonshot-v1-32k \
    --n 32 --candidates 5 --repeat 3
```

**复现性边界（重要声明）**：本 pipeline 未固定任何随机 seed——LoRA 候选生成（temperature=0.8）、API 评委采样、BWS 候选打散（`random.shuffle`）均无种子。因此**同命令重跑不产生逐字节相同的结果**；本文档所称"复现"指**统计层复现**：3 run mean ± std 框架下，主要指标应落在报告标注的 std 带内（主报告 std 0.002-0.06）。若未来需要更强的复现保证，应在新 run 中记录并固定 seed，并在报告 setup 表中新增 seed 字段（不追溯已冻结报告）。

预估调用量（每 run）：
- BWS：`n × len(models) × len(judges) = 32 × 4 × 4 = 512`
- best 4 维：`n × len(models) × len(judges) = 512`
- 跨模型 pairwise：`n × C(M,2) × len(judges) × 2 (双向) = 32 × 6 × 4 × 2 = 1536`
- 单 run 总：≈ **2560 评委调用** · 3 run ≈ 7680 次
- 实测 wallclock：~12 小时（瓶颈 deepseek-v4-pro reasoning 模型，5-30s 单次）

---

## 10. 阈值清单速查

| 阈值 | 值 | 用途 | 统计样本（分母） |
|---|---|---|---|
| `POEM_QUALITY_THRESHOLD` | 0.70 | `pass@0.7` 候选合格率 | **全候选**（n_inputs × candidates，如 32×5=160/run），见 `eval_poem.py` `dist["pass_rate_07"]` |
| 平仄合格阈值 | 0.8 | 平仄合格率 (≥0.8) | **best 代表作**（BWS 选出，n_inputs=32/run），见 `_aggregate()` `pingze_pass@0.8` ← `best_scores` |
| 押韵合格阈值 | 0.8 | 押韵合格率 (≥0.8) | **best 代表作**（同上） |
| `SCORE_PENALTY_FLOOR` | 0.7 | 单惩罚因子下限 | — |
| `CLASH_PENALTY_PER_HIT` | 0.75 | 合掌惩罚 | — |
| `REPETITION_PENALTY_MAX` | 0.15 | 重复惩罚下限 | — |
| `PAIRWISE_WIN_DELTA` | +0.17 | （生产 Arena 用，eval 不用）挑战者胜加分 | — |
| `PAIRWISE_LOSE_DELTA` | -0.05 | （同上）挑战者败扣分 | — |

> ⚠ **`pass@0.7` 与平仄/押韵合格率分母不同**，尽管报告里常并排出现：前者统计全部候选，后者只统计每题选出的那 1 首代表作。两者不可互推，跨表引用须注明口径。`pingze` / `rhyme` 本身是 rule-based 确定性分（`core/poem/scorer.py` `_score_pingze` / `_score_rhyme`），不经 LLM；代表作则由评委 BWS 选出，因此合格率的 run 间波动来自候选生成与代表作选择两处，而非评分本身。
> 主跑 artifact 中 `candidates[].local_scores` 保留了每个候选的 rule 分，全候选口径可事后重算——已对 2026-06-24 主跑做过对照，两口径偏移 1.1–3.7pp 且全在 std 内（见 [`REPORT_main_n32x3run_20260624.md`](REPORT_main_n32x3run_20260624.md) §2.4）。

### 10.1 两个 0.8 的实际来源（易误读，2026-08-11 订正）

**平仄/押韵的 0.8 是 `eval/eval_poem.py` 内的字面量**，写在 `_aggregate()` 里：

```python
out["pingze_pass@0.8"] = pass_rate([... best_scores ...], 0.8)   # 字面量，非 config
```

`config.py:182-183` 虽然定义了 `THRESHOLD_PINGZE = 0.8` / `THRESHOLD_RHYME = 0.8`，但**全库没有任何代码读取它们**（`core/poem/theme.py:6` 仅 import 而未使用）。两处 0.8 是分别写的：config 常量来自 2026-06-06 建项目时（`836bae7`），eval 字面量来自 2026-06-20 搭 BWS 体系时（`9676c28`），数值巧合相同，从无引用关系。**改 `config.py` 不会改变 eval 的任何输出**；要改口径须直接改 `eval_poem.py`（且 `pingze_pass@0.8` 这个字段名把 0.8 也写死了，见 `_RATE_METRICS`）。

本文档此前把 `THRESHOLD_PINGZE=0.8` 记作合格阈值来源，属误记，已订正。

### 10.2 阈值落在离散刻度的空隙上

`pingze` = 合规句数/总句数，`rhyme` 只有三档，两者取值都是**离散**的。2026-06-24 主跑全部 1757 个候选的实测取值集合：

| 指标 | 可取值 | ≥0.8 的实际含义 |
|---|---|---|
| `pingze`（4 句） | 0, 0.25, 0.5, 0.75, 1.0 | **⟺ = 1.0**，绝句四句零出律 |
| `pingze`（8 句） | k/8 | **⟺ ≥ 0.875**，律诗最多错一句 |
| `rhyme`（全体裁） | 0.3, 0.6, 1.0（行数不足则 0.0） | **⟺ = 1.0**，偶数行全部同一韵部 |

即 0.8 并非「达到八成」，而是一道几乎等同于满分的门槛。它也不落在任何可取值上，因此**合格率对阈值的微小改动高度敏感**——尤其当某模型的分布众数恰好卡在 0.75 时（`local_base` 即如此，其 best 代表作 40% 落在 0.75）。敏感性实测见 [`REPORT_main_n32x3run_20260624.md`](REPORT_main_n32x3run_20260624.md) §2.4。引用合格率时应连同阈值一起给出，避免「LoRA 是 base 的 N 倍」这类对阈值敏感的表述脱离口径传播。

---

## 11. 与生产路径的区别

| 阶段 | 生产 `score_single` | eval `score_single_multi_judge` |
|---|---|---|
| 维度数 | 5（不含 aesthetics）| 6（含 aesthetics）|
| imagery / cohesion 权重 | 0.15 / 0.15 | 0.10 / 0.10（匀 0.05 给 aesthetics）|
| LLM 评委数 | 1 | 4 |
| 合成 | 直接用 LLM 单值 | ≥3 中位数 / <3 均值 / None 跳过 |

生产仍走 `score_single`（单评委足够、低延迟、低成本）。`score_single_multi_judge` 仅离线评估用。

### 11.1 格律阈值：生产 0.6 与 eval 0.8 互不影响

两边共用同一个打分函数 `scorer._score_pingze` / `_score_rhyme`，但阈值各自独立硬编码，用途也完全不同：

| | 生产 | eval |
|---|---|---|
| 阈值 | **0.6** | **0.8** |
| 位置 | `core/agent/poem_refiner.py:259,261`（`hard_gate_check`） | `eval/eval_poem.py:371,374`（`_aggregate`） |
| 何时执行 | 运行时，每个挑战者生成后立即跑 | 离线评估收尾，全部 pipeline 跑完后聚合报表时 |
| 作用 | **拦截**：不过则该挑战者作废、擂主守擂（`_try_challenger` 返回 `None`） | **计数**：只决定报表百分比，不改变任何诗 / best / pairwise 结果 |
| 0.6 / 0.8 的含义 | 4 句诗 ≥3/4 句合律；8 句诗 ≥5/8 | 4 句诗须零出律；8 句诗最多错一句（见 §10.2）|

生产调用链：`autonomous.py:111` →`poem_refiner._evolve_champion()` → `_try_challenger()` → `hard_gate_check()`。该链仅在 `config.allow_poem_refine` 且已配置 API 改诗模型时触发；纯 LoRA/本地模型会跳过整个擂台阶段。

**`eval/` 在生产运行时不被导入**，因此 eval 的 0.8 永远不会在生产路径上执行；反之调整生产门控的 0.6 也不会改变任何历史评估报告的数字。两个数值不应被视为同一口径的两种取值。
