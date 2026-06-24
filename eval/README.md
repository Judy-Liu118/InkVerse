# eval/ —— 离线评估脚本

四个独立可跑的评估，分别验证项目里最值得写进简历的四个设计点。

> ⚠ 这些脚本会真实调用 API、生图、跑 CLIP，**会烧钱、占显存、花时间**。
> 推荐初次用小 `--n`（如 4）冒烟，确认输出正常后再扩到 32。

## 评估清单

| 脚本 | 验证什么 | 推荐 `--n` | 大致耗时 |
|---|---|---|---|
| `eval_poem.py` | 不同诗歌模型在格律/意境上的真实差距（BWS + 多评委 pairwise）| 4 (冒烟) / 32 (主跑) | 5min / ~10h |
| `eval_clip.py` | 双锚点 CLIP 相对单锚点的对齐提升（含 VLM ground truth）| 10 | 10–20 min |
| `eval_refine.py` | 自动方向性诗评 + refine_poem 的提升幅度 | 10 | 5–10 min |
| `eval_autonomous.py` | 全自主模式相对单轮模式的 CLIP 终值提升 | 5 | 15–30 min |

辅助工具：
- `analyze_clip_dual.py` —— 对 eval_clip JSON 做 dual 锚点合理性分析（α grid search + 锚点互补性）
- `build_classics_benchmark.py` —— 构造 15 首唐诗名作 benchmark 供 eval_clip 复用
- `dataset.py` —— 主 benchmark 题源 + `--dump` 导出 `benchmark_themes.json`

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
| [`REPORT_main_n32x3run_20260624.md`](REPORT_main_n32x3run_20260624.md) | 4 模型 × 4 评委 × n=32 × 3 run 主跑 | 2026-06-24 |

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

简历空间有限可压成一句：「在 4 模型 × 4 评委 × n=32 × 3 run 跨家族 pairwise 评估中，LoRA 把候选合格率提升 1.8 倍（36% → 64%）、内化平仄规则（移 prompt 后合规率反升至 96%），暴露 LLM-as-judge 对格律权重 ≈ 0 的盲区。」
