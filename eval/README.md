# eval/ —— 离线评估脚本

四个独立可跑的评估，分别验证项目里最值得写进简历的四个设计点。

> ⚠ 这些脚本会真实调用 API、生图、跑 CLIP，**会烧钱、占显存、花时间**。
> 推荐初次用小 `--n`（如 5）验证，确认输出正常后再扩到 20。

## 评估清单

| 脚本 | 验证什么 | 推荐 `--n` | 大致耗时 | 简历能写的话术 |
|---|---|---|---|---|
| `eval_poem.py` | 不同诗歌生成模型在格律/意境上的真实差距 | 10–20 | 5–15 min | "本地 LoRA vs qwen-plus：格律合格率 +X%、意境分 −Y%" |
| `eval_clip.py` | 双锚点 CLIP 相对单锚点的对齐提升 | 10 | 10–20 min | "双锚点设计相对单锚点 mean Δ=+0.0X，稀疏关键词诗上提升更显著" |
| `eval_refine.py` | 自动方向性诗评 + refine_poem 的提升幅度 | 10 | 5–10 min | "守擂改诗一轮总分 mean Δ=+0.0X，N% 样本得到提升" |
| `eval_autonomous.py` | 全自主模式相对单轮模式的 CLIP 终值提升 | 5 | 15–30 min | "全自主 CLIP raw 终值 mean Δ=+0.0X，平均 N.N 轮改图后达标" |

## 通用命令行约定

所有脚本共用一套参数：

```
--n               跑多少条 benchmark（数据集共 20 条）
--genres          按体裁筛选（如 --genres "五言绝句" "七言绝句"）
--density         按关键词密度筛选（rich / sparse）
--scorer          打分用 adapter（默认 qwen-plus，公平起见两边复用）
```

每个脚本另有自己的对照轴参数，详见 `python -m eval.<name> --help`。

## 跑法示例

```bash
# 1. 快速冒烟（5 条数据，验证脚本能跑通）
python -m eval.eval_poem --model-a local_lora --model-b qwen-plus --n 5

# 2. 主对照实验（建议跑这一组写进简历）
python -m eval.eval_poem --model-a local_lora --model-b qwen-plus --n 20
python -m eval.eval_clip --n 10 --image-backend bailian:z-image-turbo
python -m eval.eval_refine --n 10 --model qwen-plus
python -m eval.eval_autonomous --n 5

# 3. 专题：只看稀疏关键词诗，看双锚点稀疏分支是否真的奏效
python -m eval.eval_clip --n 5 --density sparse
```

## 输出位置

每次跑完会在 `outputs/eval/` 下生成两个文件：

- `<name>_<timestamp>.json` —— 原始数据，可二次分析、二次画图
- `<name>_<timestamp>.md`   —— markdown 报告，可直接抄进简历或 README 实验章节

控制台也会同步 print 一份 markdown 摘要。

## 报告结构（每份都长这样）

1. **总体均值表** —— 每条策略 / 模型的各维度均值、std
2. **配对差值表** —— 同一条 input 上 A vs B 的差值统计（mean Δ / median Δ / 胜率）
3. **分组对比表** —— 按体裁 / 关键词密度 / 是否改动等切片
4. **抽样诗作 / 抽样输出** —— 看个例子，避免数字撒谎

## 如何把数字写进简历

不要直接抄 mean，要带 **n + 数字 + 一句解读**。

❌ 反例：「CLIP 提升 0.04」

✅ 正例：「在 20 条 benchmark 上，双锚点相对单提示词锚点 CLIP raw 平均提升 0.04（n=20，p<0.05），稀疏关键词诗上提升尤为显著（n=8，mean Δ=0.06）。」

简历空间有限的话可以压缩成一句：「双锚点 CLIP 设计在 20 条 benchmark 上平均提升 raw 0.04，稀疏诗段提升 0.06。」

## 注意事项

- **公平性**：每个评估的 `--scorer` 默认是 `qwen-plus`，两边复用同一打分器，保证差异来自被对比变量本身。如果你要换 scorer 重跑，记得 A/B 两侧同步换。
- **可重复**：同一条 user_input 多次跑结果会有波动（LLM 随机性 + 图像 seed 未固定）。如果你想要"平均 3 次"这种更稳的数字，可以多跑几次报告然后离线合并。
- **失败处理**：单条失败不会阻断整批，会被记为 `error` 跳过，最后报告里 `n` 是实际成功数。

## 数据集

`eval/dataset.py` 里 20 条 benchmark：
- 4 种体裁：五言绝句 / 七言绝句 / 五言律诗 / 七言律诗
- 多种主题：自然（春夏秋冬） / 抒情 / 边塞 / 哲理 / 节令 / 壮阔
- 关键词密度：rich（具体意象，CLIP 双锚点应受益）/ sparse（抽象哲理，CLIP 锚点稀疏）

需要扩充时在 `BENCHMARK` 列表里加 `BenchInput(...)` 即可。
