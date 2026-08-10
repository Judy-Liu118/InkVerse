# eval/ —— 离线评估脚本

五个主评估（下表）验证项目最核心的设计点；此外还有一组分析/探针脚本（`_` 前缀，
见"辅助工具与探针"节），承载显著性分析与 position-bias 审计的完整故事线。

> ⚠ 这些脚本会真实调用 API、生图、跑 CLIP，**会烧钱、占显存、花时间**。
> 推荐初次用小 `--n`（如 4）冒烟，确认输出正常后再扩到 32。

## 评估清单

| 脚本 | 验证什么 | 推荐 `--n` | 大致耗时 |
|---|---|---|---|
| `eval_poem.py` | 不同诗歌模型在格律/意境上的真实差距（BWS + 多评委 pairwise）| 4 (冒烟) / 32 (主跑) | 5min / ~10h |
| `eval_clip.py` | 双锚点 CLIP 相对单锚点的对齐提升（含 VLM ground truth）| 10 | 10–20 min |
| `eval_refine.py` | 自动方向性诗评 + refine_poem 的提升幅度 | 10 | 5–10 min |
| `eval_autonomous.py` | 全自主模式相对单轮模式的 CLIP 终值提升 | 5 | 15–30 min |
| `eval_llm_loop_ab.py` | LLM-driven vs 写死改图循环（同基图配对 A/B；预登记 + 配额白名单熔断 + 逐轮中间图/决策埋点/终端镜像全审计）| `--limit 1` (冒烟) / 27 (全量) | 15 min / 3.5–4 h |

辅助工具与探针：
- `analyze_clip_dual.py` —— 对 eval_clip JSON 做 dual 锚点合理性分析（α grid search + 锚点互补性）
- `analyze_judge_pingze_sensitivity.py` —— 对 eval_poem JSON 做 F3 retrospective 分析（按 pingze 差分桶 + controlled pair + winner 切片）
- `build_classics_benchmark.py` —— 构造 15 首唐诗名作 benchmark 供 eval_clip 复用
- `build_f3_controlled.py` —— **构造 F3 controlled pair 池**（base 严重出律 + 意境 ≥ lora），把 anecdote (n=4) 推向 n=64+ 强结论；脚本支持 `--dry-run` 验证拓扑，真跑约 1-2hr API
- `sweep_pairwise_win_delta.py` —— **PAIRWISE_WIN_DELTA 擂台阈值 sweep**（在 [0.10, 0.15, 0.17, 0.20] 上各跑 autonomous fixed loop，看哪个值 CLIP std 最低 + 攻擂率落在 15-40% 健康区间）；monkey-patch config + 自动恢复，`--dry-run` 验证拓扑
- `vlm_hard_constraint.py` —— **VLM 硬约束命中率评测**（读 sweep 聚合 JSON，30 张图 × qwen-vl-max 逐 keyword 判 present）；独立打补丁 CLIP "图里东西多"盲点，不烧文生图配额，约 780 秒跑完 30 张
- `dataset.py` —— 主 benchmark 题源 + `--dump` 导出 `benchmark_themes.json`
- `_analyze_arena_ablation_significance.py` —— 对消融 +18.2pp 差值补不确定性量化：keyword 级 McNemar 精确检验 + 主题级 bootstrap 95% CI（零 API，读已有 JSON）
- `_sweep_kw_sensitivity.py` —— **kw 定义严/松敏感度全池扫描（A2）**：对冻结 HARD_CONSTRAINTS 的 22 题 kw 各写 strict/loose 变体重判 44 图，给 +18.2pp 配上定义无关的稳健区间 [+9.1, +24.2]pp（消融报告 §2.7）
- `_probe_pairwise_position_bias.py` —— **B1 位置偏置探针（含自我证伪复盘）**：历史攻擂对局双向重判，表面显著后诊断出选择-复现混杂并主动降级——docstring 是完整的混杂机制说明
- `_probe_position_bias_fresh_pairs.py` —— **B1b 无偏探针**：新鲜配对（选择与 judge 无关）双向重判，方向反转为偏 A 位（p=0.0156）
- `_rerun_arena_randomized_layout.py` —— **B2-lite 随机化布局无图擂台重跑**：量化 B3a 随机化 before/after 攻擂率（49.2%→55.9%，ns）+ 位置审计给出与 B1b 反向的弱信号——方向锁不死恰好验证"随机化不赌方向"

> 上面四个 `_probe/_sweep/_rerun` 脚本合起来是 position-bias 审计的完整故事线（发现 → 显著 → 自我证伪 → 方向反转 → 再反转 → 随机化防御落地），通俗版见 [`METHODOLOGY.md`](METHODOLOGY.md) §7.5、严谨版见 [sweep 报告 §6.3 #1](REPORT_pairwise_win_delta_sweep_2026-06-30.md)。

## 主跑命令（n=32 × 3 run · 12h overnight）

```bash
python -m eval.eval_poem \
    --models local_base local_lora local_lora_naked qwen-plus \
    --scorer deepseek-v4-pro qwen-max glm-4-plus moonshot-v1-32k \
    --n 32 --candidates 5 --repeat 3
```

冒烟测试（4 道 × 2 run，~10min）：

```bash
python -m eval.eval_poem \
    --models local_lora qwen-plus \
    --scorer qwen-max glm-4-plus \
    --n 4 --candidates 5 --repeat 2
```

CLIP 双锚点（含 VLM ground truth）：

```bash
python -m eval.eval_clip --n 10 --image-backend bailian:z-image-turbo
python -m eval.analyze_clip_dual outputs/eval/eval_clip_<timestamp>.json
```

## 通用命令行约定

```
--n          跑多少条 benchmark（主数据集共 32 条）
--genres     按体裁筛选（如 --genres "五言绝句" "七言绝句"）
--density    按关键词密度筛选（rich / sparse）
--models     被对比的诗歌生成模型列表（eval_poem 专用，支持 ≥ 2 个）
--scorer     LLM 评委列表（eval_poem 专用，支持 ≥ 2 个，跨家族抗 self-bias）
--candidates 每模型生成几个候选（eval_poem，默认 5）
--repeat     完整 pipeline 跑几次 → 跨 run mean ± std
```

每个脚本另有自己的对照轴参数，详见 `python -m eval.<name> --help`。

## 输出位置

每次跑完落两份产物在 `outputs/eval/` （已 gitignore，不入库）：

- `<name>_<timestamp>.json` —— 原始数据，可二次分析、二次画图
- `<name>_<timestamp>.md`   —— markdown 报告，控制台同步 print

**值得长期留存的报告**手动改名复制到 `eval/REPORT_*.md`（入库版本），见下一节。

## 入库的主跑报告

| 文件 | 实验 | 报告日期 |
|---|---|---|
| [`REPORT_eval_clip_dual_anchor_20260623.md`](REPORT_eval_clip_dual_anchor_20260623.md) | **eval_clip 主报告**：双锚点 CLIP vs 单锚点对齐分（n=15 千古名诗 baseline）—— 生产 dual (α=0.6) 与 VLM Spearman ρ=+0.365，点估计高于 prompt_only (+0.301)（n=15 标准误 ≈0.29，未达显著）、α grid search 最优 α=0.8 处 ρ=+0.414；dual 设计在名诗语料上被数据支持，但 §4.1 跨 backend 复现结论为负、α 最优点不稳定，故不调 `CLIP_POEM_WEIGHT` | 2026-06-23 |
| [`REPORT_main_n32x3run_20260624.md`](REPORT_main_n32x3run_20260624.md) | 4 模型 × 4 评委 × n=32 × 3 run 主跑 | 2026-06-24 |
| [`REPORT_F3_pingze_sensitivity_20260624.md`](REPORT_F3_pingze_sensitivity_20260624.md) | F3 retrospective 验证：评委对格律敏感度（n=4-5 controlled pair，初步推测）| 2026-06-24 |
| [`REPORT_autonomous_n5_20260627.md`](REPORT_autonomous_n5_20260627.md) | LLM-driven 改图循环点亮 + eval 三臂对比（single_pass / autonomous(fixed) / autonomous(llm)）+ 诚实性指标 + VLM 独立裁判（n=5，负面 + caveat 充足）| 2026-06-27 |
| [`REPORT_pairwise_win_delta_sweep_2026-06-30.md`](REPORT_pairwise_win_delta_sweep_2026-06-30.md) | 擂台 `PAIRWISE_WIN_DELTA` 阈值 sweep（3 delta × n=10）+ 主题×delta 全景对比 + 攻擂率异常 surface | 2026-06-30 |
| [`REPORT_vlm_hard_constraint_20260701.md`](REPORT_vlm_hard_constraint_20260701.md) | VLM 硬约束命中率（30 张图逐 keyword yes/no）—— 独立打补丁 CLIP "图里东西多"盲点；rich 题 δ=0.17 命中率 80% 佐证 sweep 结论 | 2026-07-01 |
| [`REPORT_arena_ablation_20260701.md`](REPORT_arena_ablation_20260701.md) | **擂台机制净收益消融**：arm A (`max_poem_rounds=0`, 无擂台) vs arm B (`max_poem_rounds=2`, 有擂台)，主池 n=22（04-22 backend）+ 辅池 n=10（03-03 backend）—— CLIP mean 平手（Δ -0.002）但 VLM 硬约束 arm B 主池 +18.2pp（60.6% → 78.8%），结论保留擂台。**后续补强（07-05/07-06）**：§2.5 McNemar+bootstrap 显著性、§2.6 跨家族 VLM 复核（glm-4v-plus 同向 +15.2pp）、§2.7 kw 定义严/松包络 [+9.1, +24.2]pp、§6.4 position-bias 探针系列注记 | 2026-07-01（持续更新） |
| [`REPORT_eval_refine_LoRA_n32_20260702.md`](REPORT_eval_refine_LoRA_n32_20260702.md) | **eval_refine 主报告**：LoRA baseline + qwen-plus 单轮 refine（n=32 主 benchmark，生产链路真实场景）—— 改动率 53.1%、cohesion +0.031 全样本 / +0.074 改动样本主导、rhyme 因 feedback 硬约束零效果、pingze -0.004 副作用；refine 真实价值维度是"意境连贯度重塑"而非 LoRA 弱项押韵优化 | 2026-07-02 |
| [`REPORT_llm_loop_ab_n27_20260710.md`](REPORT_llm_loop_ab_n27_20260710.md) | **实验 B run 1**：LLM-driven vs 写死改图循环，同基图配对 n=27（硬约束题集全集，预登记跑前 commit）—— CLIP 终值打平（循环子集 Δ+0.004）、LLM 臂耗时 -39%/调用更少、target=0.30 区分度不足教训、VLM 同图重判不一致发现；配图库 [`runs/llm_loop_ab/gallery/GALLERY.md`](runs/llm_loop_ab/gallery/GALLERY.md) | 2026-07-10 |
| [`REPORT_llm_loop_ab_run2_20260711.md`](REPORT_llm_loop_ab_run2_20260711.md) | **实验 B run 2（同条件独立重复）**：打平复现（Δ 符号翻转、幅度均 ≤0.005）、效率方向复现；run 1 两个次级苗头（达标更多、edit 选择效应）证伪并撤回；管线噪声 mean 0.029 ≈ 臂间效应 7 倍——单次运行 ≤0.01 差异不可判读的量化证据；配图库 [`runs/llm_loop_ab_run2/gallery/GALLERY.md`](runs/llm_loop_ab_run2/gallery/GALLERY.md) | 2026-07-11 |

完整方法论（公式 / 系数 / 评委 prompt 全文 / 阈值清单）冻结在 [`METHODOLOGY.md`](METHODOLOGY.md) —— 后续代码漂移仍能解释这份报告。

## 主跑代表性发现速览

详见 [`REPORT_main_n32x3run_20260624.md`](REPORT_main_n32x3run_20260624.md) 第 1-5 节，三个最值得复述的 finding：

1. **LoRA naked > LoRA full**（平仄 96.4% vs 95.4%，押韵 39% vs 33%）→ LoRA 把格律内化进权重，prompt 反而成噪声
2. **LoRA 提升地板不提升天花板**：pass@0.7 候选合格率 36.2% → 64.0%（×1.8），但 best 候选 4 维分持平（0.771 vs 0.771）
3. **LLM-as-judge 对格律不敏感**：base 平仄仅 25.6%（LoRA 的 1/4），pairwise 胜率反而更高 → 评委权重接近全在 intent/imagery/aesthetics，生产里必须保留 rule scorer 做格律硬约束

## 数据集

`eval/dataset.py` 里 32 条 benchmark（**互斥 12 主题分层**）：

- **写景 ×16**：春/夏/秋/冬 各 4 道 ×（五绝/七绝/五律/七律），每季 rich:sparse = 2:2
- **非写景 ×16**：山水/田园/边塞/羁旅/送别/怀古/节令/哲理 各 2 道 × 2 体裁
- **rich**（16 道）：明确给出 ≥2 个具体意象（如"要有柳树和燕子"）
- **sparse**（16 道）：只给抽象主题词（如"主题是春雨"）

`get_benchmark(n<32)` 按 stratified round-robin 切片，保持 3 层均衡（体裁 × density × scenic）。

导出题源：

```bash
python -m eval.dataset --dump   # → eval/benchmark_themes.json
```

需要扩充时在 `BENCHMARK` 列表里加 `BenchInput(...)` 即可，新增写景题保持每季 rich:sparse=2:2。

## 注意事项

- **公平性**：所有模型与所有评委同时跑，差异来自被对比变量本身。评委独立调用，不交叉污染。
- **可重复**：`--repeat N` 强制跑 N 次完整 pipeline，主要指标 std 跨 run 一般 0.005-0.03。
- **失败处理**：单条评委调用失败 → 评委整体弃权（None），不污染 multi-judge 合成；单模型一条 input 失败不阻断整批。
- **生产代码隔离**：`score_single_multi_judge` 仅 eval 用，生产仍走 `score_single`（单评委、低成本）。详见 [`METHODOLOGY.md`](METHODOLOGY.md) §11。

## 如何把数字写进简历

不要直接抄 mean，要带 **n + 数字 + std + 一句解读**。

❌ 反例：「CLIP 提升 0.04」

✅ 正例：「n=32 × 3 run 主跑：LoRA 候选 pass@0.7 合格率从 36.2% ± 2.6% 提升到 64.0% ± 2.4%（×1.8 提升地板），但 best 候选 multi-judge 总分 0.771 持平 —— 证明 LoRA 收紧分布而非提升天花板，符合 alignment fine-tune 的典型 pattern。」

简历空间有限可压成一句：「在 4 模型 × 4 评委 × n=32 × 3 run 跨家族 pairwise 评估中，LoRA 把候选合格率提升 1.8 倍（36.2% ± 2.6% → 64.0% ± 2.4%）、内化平仄规则（移除格式 prompt 后合规率未退化，96.4% ± 3.0%），并观察到 LLM-as-judge 对格律赋予极低权重的盲区。」

> ⚠️ 压缩版的两处措辞守则：① 格律一条只能说「未退化」，不能说「反升 / 更好」——naked vs full 的 Δ 是 +1.0pp 而 std 达 ±4.3pp，落在噪声内，说「更好」当场就会被追问幅度与显著性；② 评委格律盲区一条只能说「观察到」，不能说「权重 ≈ 0」——直接证据仅 n=4 controlled pair 且受 BWS 筛选影响，报告中定性为初步推测，强结论需独立 64-pair 重跑（见 [`REPORT_F3_pingze_sensitivity_20260624.md`](REPORT_F3_pingze_sensitivity_20260624.md) 结尾）。

agent 侧的一句版（实验 B）：「预登记的同基图配对 A/B（n=27 × 2 次独立运行）显示 LLM-driven 改图循环相对写死循环 CLIP 终值打平、但以更少调用和 -19~39% 耗时达到同等结果；主动 replication 撤回了首轮两个不稳观察，并量化出管线噪声 ≈ 臂间效应的 7 倍。」
