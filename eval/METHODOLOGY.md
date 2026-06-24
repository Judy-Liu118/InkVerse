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
| `pingze`（平仄）| 对比平水韵 8 种格律模板（`PATTERN_5_4`/`PATTERN_5_8`/`PATTERN_7_4`/`PATTERN_7_8`），逐字打分 → 命中率 | 合格阈值 `THRESHOLD_PINGZE=0.8` |
| `rhyme`（押韵）| 平水韵韵部归类，偶数行末字同部数 / 应押韵字数 | 合格阈值 `THRESHOLD_RHYME=0.8` |
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

预估调用量（每 run）：
- BWS：`n × len(models) × len(judges) = 32 × 4 × 4 = 512`
- best 4 维：`n × len(models) × len(judges) = 512`
- 跨模型 pairwise：`n × C(M,2) × len(judges) × 2 (双向) = 32 × 6 × 4 × 2 = 1536`
- 单 run 总：≈ **2560 评委调用** · 3 run ≈ 7680 次
- 实测 wallclock：~12 小时（瓶颈 deepseek-v4-pro reasoning 模型，5-30s 单次）

---

## 10. 阈值清单速查

| 阈值 | 值 | 用途 |
|---|---|---|
| `POEM_QUALITY_THRESHOLD` | 0.70 | `pass@0.7` 候选合格率 |
| `THRESHOLD_PINGZE` | 0.8 | 平仄合格率 (≥0.8) |
| `THRESHOLD_RHYME` | 0.8 | 押韵合格率 (≥0.8) |
| `SCORE_PENALTY_FLOOR` | 0.7 | 单惩罚因子下限 |
| `CLASH_PENALTY_PER_HIT` | 0.75 | 合掌惩罚 |
| `REPETITION_PENALTY_MAX` | 0.15 | 重复惩罚下限 |
| `PAIRWISE_WIN_DELTA` | +0.17 | （生产 Arena 用，eval 不用）挑战者胜加分 |
| `PAIRWISE_LOSE_DELTA` | -0.05 | （同上）挑战者败扣分 |

---

## 11. 与生产路径的区别

| 阶段 | 生产 `score_single` | eval `score_single_multi_judge` |
|---|---|---|
| 维度数 | 5（不含 aesthetics）| 6（含 aesthetics）|
| imagery / cohesion 权重 | 0.15 / 0.15 | 0.10 / 0.10（匀 0.05 给 aesthetics）|
| LLM 评委数 | 1 | 4 |
| 合成 | 直接用 LLM 单值 | ≥3 中位数 / <3 均值 / None 跳过 |

生产仍走 `score_single`（单评委足够、低延迟、低成本）。`score_single_multi_judge` 仅离线评估用。
