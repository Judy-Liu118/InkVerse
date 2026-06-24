# F3 验证报告 · local_base vs local_lora

**数据源：** `eval_poem_20260624_210002.json` · n=32 × 3 run · 评委=['deepseek-v4-pro', 'qwen-max', 'glm-4-plus', 'moonshot-v1-32k']

**F3 命题（待验证）：** LLM-as-judge 对格律（pingze）权重接近 0，主要看 intent/imagery/cohesion/aesthetics。

**方法：** retrospective 切片主跑 `matchups_per_input`（双向 pairwise + 评委一致票多数决），不重跑 API。对每个 (local_base, local_lora) matchup，按 (pingze_b - pingze_a) 分桶 / controlled pair / winner 切片三种方式分析。

## 分析 1：按 pingze_diff 分桶 (local_lora 相对 local_base)

**命题：** 若评委爱格律 → pingze_diff 越大，model_b 胜率越高（桶间单调）。

**反 F3 信号：** 单调性强。**支持 F3 信号：** 各桶胜率接近。

| 分桶 | n | local_lora 胜 | 平 | local_base 胜 | 全摇摆 | local_lora 胜率（决断场次内） |
| --- | --- | --- | --- | --- | --- | --- |
| 极端 (local_lora 格律完美 vs local_base 严重出律): diff>=0.5 | 16 | 6 | 2 | 7 | 1 | **46.7%** |
| 中等 (0.2 <= diff < 0.5) | 45 | 19 | 6 | 18 | 2 | **51.2%** |
| 微差 (0 < diff < 0.2) | 5 | 4 | 0 | 1 | 0 | **80.0%** |
| 持平 (diff == 0) | 12 | 4 | 0 | 8 | 0 | **33.3%** |
| 反向 (diff < 0, local_lora 反而格律差) | 5 | 0 | 0 | 4 | 1 | **0.0%** |

**关键对比：** 极端格律差距桶 local_lora 胜率 46.7% (n=16) vs 格律持平桶 local_lora 胜率 33.3% (n=12)。
→ 差距 +13.3%，**部分支持 F3**：有一定 pingze 敏感性但不显著。

## 分析 2：controlled pair（local_base 格律差但意境 ≥ local_lora）

**筛选条件：** local_base.pingze < 0.5 且 local_base 4 维 LLM 维度均值 ≥ local_lora。

**命题：** F3 若成立 → 评委选 model_a 的比例应不显著低（>= 40%）。
若评委爱格律 → 此处 model_a 几乎全败给 model_b。


**结果：** n=4 controlled pair · local_base 胜=3 平=1 负=0 全摇摆=0 · **local_base 胜率 87.5%**（决断场次 4 内）
→ **强支持 F3**：尽管 local_base 格律差，评委因意境维度持平/略胜仍选它，证明评委对 pingze 权重 << 意境维度。

**具体 pair（local_base 格律差但意境占优）：**

| run | input | local_base.pingze | local_lora.pingze | local_base 意境均值 | local_lora 意境均值 | winner |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 写一首山水的七言律诗，要有高楼和远山 | 0.38 | 1.00 | 0.969 | 0.875 | **local_base** |
| 2 | 写一首冬景的七言绝句，要有飞雪和寒鸦 | 0.25 | 1.00 | 0.969 | 0.791 | 平 |
| 2 | 写一首七言绝句，主题是征戍 | 0.25 | 1.00 | 0.896 | 0.823 | **local_base** |
| 3 | 写一首送别的七言律诗，要有长亭和折柳 | 0.38 | 1.00 | 0.938 | 0.802 | **local_base** |

## 分析 3：winner 切片下的 pingze_diff 分布

**命题：** 若评委爱格律 → local_lora 胜的 matchup 平均 pingze_diff 应明显 > local_base 胜的 matchup。

| winner | n | mean pingze_diff | std | median |
| --- | --- | --- | --- | --- |
| **local_lora** 胜 | 33 | +0.277 | 0.159 | +0.250 |
| **local_base** 胜 | 38 | +0.197 | 0.265 | +0.250 |

**Δ mean = +0.079**（粗 t ≈ +1.55）
→ **支持 F3**：winner 切片对 pingze_diff 几乎无区分度，评委不靠 pingze 决胜。

---

## 综合结论

三块分析交叉看：
- 分析 1 桶间胜率单调性 → 评委是否按 pingze 加权
- 分析 2 controlled pair → 直接测意境 vs 格律权衡
- 分析 3 winner 切片均值差 → pingze_diff 在决胜中的贡献

三块都支持 F3 = 强证据；不一致 = 评委对 pingze 的态度因评委/情境而异。

**注意：** 本分析受主跑 BWS 筛选影响 —— best 候选已经过 BWS 选出，极端格律样本可能被筛掉，使 controlled pair n 偏小。若 n=0 或不显著，需重跑独立的 64-pair 实验。