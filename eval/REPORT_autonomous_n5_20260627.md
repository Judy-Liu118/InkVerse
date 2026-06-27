# eval_autonomous 报告 · single_pass vs autonomous(fixed) vs autonomous(llm-driven)

_n=5 · 目标 CLIP raw=0.3 · max 改图=2 · max 改诗=1_

> **跑法**: `python -m eval.eval_autonomous --n 5 --max-img-rounds 2 --max-poem-rounds 1 --vlm-judge qwen-vl-max`
> · 诗模型 local_lora · prompt qwen-max · scorer qwen-plus · image backend bailian:qwen-image-2.0-pro · VLM oracle qwen-vl-max
> · 跑完时间 2026-06-27 22:39:45 · 代码 commit `cf229ec`（点亮 LLM-driven 改图循环 + 三臂 eval + 诚实性指标 + VLM 独立裁判）

## TL;DR — 三个值得带走的结论（全部小样本，n=4-5）

1. **agentic 在这个窄循环里没赢、且贵 2×**：autonomous(llm-driven) − autonomous(fixed) mean Δ = **-0.023**（n=4 配对），llm 平均耗时 221s vs fixed 105s。这是负面结果，但对"加 agent 就一定更好"的直觉有反例价值。
2. **改图循环本身可能没用**：autonomous(fixed) − single_pass mean Δ = **-0.0008**（n=4，基本持平）；平均改图轮次 0.5（多数样本首图已达 raw≥0.30 目标，提前终止）。
3. **VLM 独立裁判戳穿了"改图涨 CLIP"叙事**：两条 autonomous 臂 after_better 比例都是 **0%**（n=9 合计），mean Δ ≤ 0。CLIP 在动但 loop 外的 oracle 看不到任何提升 —— 是 reward hacking 嫌疑信号，但 n 太小不能下定论。

## 重要 caveats（先看再读数字）

- **n=5 真的太小**。所有 mean Δ 标在 ±0.02-0.03 量级，跨 5 个样本的 std 可能比信号本身还大。结论只是**初步**，需要 n≥20 才能谈"显著"。
- **autonomous(fixed) 缺一条** —— row 1 触发 `core/poem/scorer.py:1023 champion["final"]` 已知 KeyError（gated 候选边界条件，与本次 LLM-driven 改动无关）。autonomous(fixed) 报表里实际 n=4。该 bug 单独追，本报告不修。
- **autonomous_llm 中一条耗时 492s** —— 百炼图像编辑 API 拿到 429 限流后回退到本地 Z-Image，单条加载 + 推理 ~6min；如果只算 API 路径平均耗时应该更接近 fixed 的 2×。
- **VLM 用 qwen-vl-max 单图 0-10 打分两次**（before/after 各一次），不是 pairwise；解析两侧打到相同分（"8.5 vs 8.5"）时 after_better=False。在 n=9 里 8 个就是相等，对零差异敏感度低 —— 后续应换 pairwise prompt 提升分辨率。
- **本次 fixed loop 实际触发改图的样本只有 1/5**（夏蝉那条，跑了 2 轮）。其余 4/5 都是首图就达标，所以"fixed vs single_pass"差异主要不来自改图循环本身，而来自 autonomous 路径里多跑的 Arena 海选 + 守擂改诗。

## 1. CLIP raw 终值对比（三臂）

| 模式 | n | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| single_pass | 5 | 0.303 | 0.031 | 0.300 | 0.266 | 0.358 |
| autonomous | 4 | 0.307 | 0.020 | 0.308 | 0.278 | 0.334 |
| autonomous_llm | 5 | 0.281 | 0.019 | 0.274 | 0.267 | 0.318 |

## 2. 配对差值

### 2.1 autonomous(fixed) − single_pass

| 指标 | 值 |
| --- | --- |
| 样本对数 n | 4 |
| mean Δ | -0.0008 |
| median Δ | 0.0023 |
| autonomous 提升比例 | 50.0% |

### 2.2 autonomous(llm-driven) − single_pass

| 指标 | 值 |
| --- | --- |
| 样本对数 n | 5 |
| mean Δ | -0.0226 |
| median Δ | -0.0263 |
| autonomous_llm 提升比例 | 20.0% |

### 2.3 autonomous(llm-driven) − autonomous(fixed)  ← agentic 是否值钱

| 指标 | 值 |
| --- | --- |
| 样本对数 n | 4 |
| mean Δ | -0.0231 |
| median Δ | -0.0234 |
| autonomous_llm 提升比例 | 25.0% |

## 3. 成本：耗时 + 自主轮次

| 指标 | single_pass | autonomous(fixed) | autonomous(llm) |
| --- | --- | --- | --- |
| 平均耗时 (s) | 61.8 | 105.5 | 221.1 |
| 中位耗时 (s) | 52.3 | 91.3 | 162.9 |
| 平均改图轮次 | — | 0.5 | 0.0 |
| 平均改诗轮次 | — | 0.8 | 0.4 |

中位耗时倍率更代表典型情况：fixed ≈ 1.7× single_pass，llm ≈ 3.1× single_pass。

## 4. LLM-driven 循环诚实性

| 指标 | 值 |
| --- | --- |
| 总决策数（所有 input 累计） | 8 |
| fallback 总数 | 0 |
| 整体 fallback 率 | 0.0% |
| per-input 平均 fallback 率 | 0.0% |
| per-input 平均 stale-override 次数 | 0.00 |
| stale-override 总次数 | 0 |

**工具选择分布:**

| tool | 次数 | 占比 |
| --- | --- | --- |
| edit_image | 8 | 100.0% |
| refine_poem_and_regen | 0 | 0.0% |
| stop | 0 | 0.0% |

**解读:**

- **fallback = 0%**：LLM 决策没崩，JSON 解析全过、工具白名单全合法。这是 controller 实现层的健康信号，但**不等于 agent 有用** —— 它只说明 "agent 没被兜底覆盖"。
- **stale-override = 0 次**：本次样本里没出现"连续 2 轮 CLIP 无提升"的情况（首图常常已经达标，改图循环根本没启动到第 2 轮），所以启发式护栏没机会触发；后续要让 stale-override 数据有意义，应跑更难的样本（首图 raw < 0.25）或调低 target。
- **工具分布 100% edit_image**：LLM 从未选 `refine_poem_and_regen`（改诗 + 重生图）或 `stop`。这是**最值得记的反例**：当前 prompt 描述 + 上下文不足以让 LLM 区分这两种 action 的适用场景 —— "agentic" 这个词在本次实验里**只体现在 edit_image 的 feedback 文字上**，不体现在路由判断上。

## 5. per-decision 归因（tool × 本轮是否提升 CLIP）

| tool | improved | not_improved | no_signal | 总计 |
| --- | --- | --- | --- | --- |
| edit_image | 3 | 1 | 4 | 8 |

`no_signal` = 首轮（无 prev_score 可比）；`improved` / `not_improved` 在有 prev 时按本轮 CLIP 是否提升判桶。

**解读**：列联表本来设计是想看 LLM 在 `edit_image` vs `refine_poem_and_regen` 之间路由是否有判断价值。**本次样本里 LLM 从未选过 refine_poem_and_regen，所以路由列联表退化成单行 —— 暂时无法证伪"路由近随机"或"路由有判断"任意一边**。需要更难样本 + 提高 max_img_rounds 才能触发第二行。

## 6. VLM 独立裁判：改图前 vs 改图后

> 改图循环的优化目标是 CLIP-final，再拿 CLIP-final 当成功指标是部分同义反复。
> 这一节用 VLM（loop 外裁判）直接评 before-image vs after-image，分数与 loop 优化目标解耦。

| 臂 | n | after_better | after 更优比例 | mean Δ(raw 0-10) | median Δ |
| --- | --- | --- | --- | --- | --- |
| autonomous(fixed) | 4 | 0 | 0.0% | 0.000 | 0.000 |
| autonomous(llm) | 5 | 0 | 0.0% | -0.400 | 0.000 |

**解读**：after 更优比例 = VLM 判定改图后图更契合诗的样本占比。这是**非 CLIP 的成功率数字**：成功指标不再是 loop 自己在爬的那个数。

- 两条 autonomous 臂的 after_better 比例都是 0%（n=9 合计）—— CLIP 在动但 VLM oracle 看不到任何提升。
- llm 臂 mean Δ = -0.4（VLM raw 0-10 量纲）说明 VLM 在 9 条里两条判定 llm 改图后**更差**。
- **但有方法学坑**：VLM 用单图 0-10 评分两次，绝大多数样本 before/after 同分（如 8.5/8.5），after_better 默认计入 False；对零差异敏感度极低。**建议下次换 pairwise prompt**（同时给 VLM 两张图，问哪个更好），分辨率会高得多。
- **n=9 也太小**。如果换 pairwise 重跑 n≥20 后 after_better 仍接近 50%、Δ ≈ 0，那"CLIP 在中文诗 + 水墨域可能过拟合到 reward"才算有 finding；目前只够当**怀疑线**。

## 7. 抽样

### 写一首春景的五言绝句，要有柳树和燕子

- single_pass:     CLIP=0.266, 78.64s
- autonomous:      CLIP=0.310, 91.66s, 改图 0 轮, 改诗 0 轮, VLM Δ=0.00 (before=9.0, after=9.0)
- autonomous_llm:  CLIP=0.267, 123.92s, 改图 0 轮, fallback=0/2, stale_override=0, VLM Δ=0.00

### 写一首春景的五言律诗，要有桃花和啼莺

- single_pass:     CLIP=0.358, 87.48s
- autonomous:      ⚠ ERROR (`KeyError: 'final'` from `core/poem/scorer.py:1023`，已知 scorer 边界 bug，与本次改动无关)
- autonomous_llm:  CLIP=0.318, 94.32s, 改图 0 轮, fallback=0/0, stale_override=0, VLM Δ=-1.00 (before=8.5, after=7.5)

### 写一首七言律诗，主题是客愁

- single_pass:     CLIP=0.307, 52.29s
- autonomous:      CLIP=0.334, 83.25s, 改图 0 轮, 改诗 1 轮, VLM Δ=0.00
- autonomous_llm:  CLIP=0.277, 162.86s, 改图 0 轮, fallback=0/2, stale_override=0, VLM Δ=0.00

### 写一首五言绝句，主题是夏蝉

- single_pass:     CLIP=0.300, 43.41s
- autonomous:      CLIP=0.278, 156.05s, 改图 2 轮, 改诗 1 轮, VLM Δ=0.00 (**fixed loop 唯一一条真触发改图的样本**)
- autonomous_llm:  CLIP=0.274, **492.05s**, 改图 0 轮, fallback=0/2, stale_override=0, VLM Δ=0.00（百炼 429 fallback 本地 Z-Image，单条 ~6min）

## 下一步建议（按性价比排）

1. **修 `core/poem/scorer.py:1023` 的 `champion["final"]` KeyError** —— gated 候选边界，应该用 `.get("final", local_total)` 兜底。fix 后 autonomous(fixed) 不再缺样本。
2. **n 提到 ≥20**，否则结论全是 anecdote。但要先排除上面那条 bug，否则 autonomous 臂样本不齐。
3. **VLM 换 pairwise prompt**（同时给两张图）—— 现在 0-10 单图打分对 before/after 微差几乎不敏感。
4. **跑更难的样本（首图 raw < 0.25）或调低 target**，让改图循环真有机会跑到第 2-3 轮，诚实性指标和 per-decision 归因才有数据。
5. **如果 (1)-(4) 跑完仍是负面结果**：写进文档"在这个窄循环里 LLM-driven 没赢 fixed loop 且贵 2-3×"作为成熟负面结论；如果反转那 fine-tune controller prompt 让 LLM 更倾向在 stale 时切 refine_poem_and_regen。

## 结论一句话

**这套 LLM-driven 改图循环已经在产品里点亮、可在 UI 勾选启用，且 eval 有了三臂对比 + 诚实性指标 + VLM 独立裁判的全套基建。但 n=5 的初步结果是负面的：agentic 没赢 fixed loop、贵 2-3×、工具路由从未选过 refine_poem_and_regen、VLM 看不到改图收益。把这套留在代码里作为可选功能 + 测量底座，等 (1)-(4) 都补齐后再决定是否默认开启。**
