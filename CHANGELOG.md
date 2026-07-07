# CHANGELOG · breaking 变更登记

只登记**影响评估口径可比性或生产擂台行为**的变更（breaking）。常规功能演进见 git log；
各指标的冻结定义见 [`eval/METHODOLOGY.md`](eval/METHODOLOGY.md)。

引用历史评估数字做跨期对比前，先对照本页确认两侧口径一致。

---

## 2026-07-07 · 擂台第二挑战方向注入已有方向文本

- **commit**: `42aa5ab` · `core/agent/poem_refiner.py` `_evolve_champion`
- **变更**: 每轮第二个挑战方向的生成 prompt 原先只说"不要重复之前的建议方向"，
  但两次 LLM 调用独立无状态，"之前的方向"对模型无所指；现把已生成的方向文本
  显式注入第二次调用的 critique。
- **breaking**: 挑战者多样性提升会改变攻擂率分布——此后 run 的攻擂率与历史 run
  （delta sweep、arena ablation、B2-lite 等）跨期对比时须注明此差异，与 B3a
  （A/B 位随机化）同理叠加。指标定义与判定口径均未变。
- **动机**: 2026-07-07 人工代码检视发现该指令为对空指令。

## 2026-07-07 · 合掌检测补联内上下句交叉

- **commit**: `aebb3b6` · `core/poem/scorer.py` `_check_synonym_clash`
- **变更**: 原实现只逐句检测句内同义词群复用（docstring 声称"同一联"与实现不符）；
  现补上合掌本义的联内检测——一联上下句各用同一同义词群中的不同词
  （如上句"残阳"、下句"落日"）也触发 `CLASH_PENALTY_PER_HIT`（×0.75）。
  句内检测保留；切句从按行改为按标点/换行（对一句一行的既有格式无差异）。
- **breaking**: `clash_penalty` 从此更严，影响品质分 total（候选废弃线判定）与
  arena `local_score`——此后 run 的品质分与历史 run 不严格可比。注意
  [`eval/METHODOLOGY.md`](eval/METHODOLOGY.md) 中该指标的冻结描述本就是
  "同联同义词"，本次是**实现追平既有定义**，非定义变更。旧数据不追溯重算。
- **动机**: 2026-07-07 人工代码检视发现实现与定义不符。

## 2026-07-06 · 擂台守擂 pairwise A/B 位随机化（B3a）

- **commit**: `4dcb677` · `core/agent/poem_refiner.py` 新增 `pairwise_judge_randomized`
- **变更**: 守擂 1v1 判定从固定布局（擂主恒 A 位 / 挑战者恒 B 位）改为每次随机分位。
- **breaking**: 此后的攻擂率与所有历史 run（delta sweep、arena ablation arm B 等）
  不严格可比——跨期对比攻擂率须注明布局差异。
- **动机与验证**: judge 位置偏置的方向经 B1 → B1b → B2-lite 三个探针仍未锁死
  （两个无混杂探针方向相反），随机化是唯一对方向与幅度均不敏感的防御。
  故事线见 [`eval/METHODOLOGY.md`](eval/METHODOLOGY.md) §7.5、
  严谨版见 [sweep 报告 §6.3 #1](eval/REPORT_pairwise_win_delta_sweep_2026-06-30.md)。
- **before/after 实测**: 攻擂率 49.2% → 55.9%（n=22，Fisher p=0.58，未见异常漂移），
  见 `eval/_rerun_arena_randomized_layout.py`（B2-lite）。

## 2026-07-05 · `compare_poems` 弃权语义

- **commit**: `78ea046` · `core/poem/scorer.py`
- **变更**: pairwise 判定失败（adapter 异常 / 回复不含唯一 A/B）从**静默返回 "A"**
  改为返回 `None`（评委弃权）；生产擂台侧判定失败从"隐式擂主胜"改为
  "该挑战者作废、显式记日志守擂"。
- **breaking**: 摇摆率/胜率口径变化——历史报告（含主跑报告摇摆率 23%）把 API 失败
  与 position bias 混在同一"摇摆"桶，与新口径不严格可比。旧数据不追溯重算。
  详见 [`eval/METHODOLOGY.md`](eval/METHODOLOGY.md) §7 BREAKING 块。
- **旁证**: B1b 新口径实测摇摆率 21.9%，与历史 23% 接近——提示历史摇摆桶主要成分
  是真实 position bias 而非 API 失败（METHODOLOGY §7 旁证块）。

## 2026-06-24 · 4 维评分 / BWS 弃权语义

- **commit**: `fb19f43` · `core/poem/scorer.py`
- **变更**: LLM 评委单次调用失败 / 解析失败从"给默认分"改为返回 `None`（弃权），
  multi-judge 合成跳过 None 评委。
- **breaking**: 此前 run 的 4 维分含"失败默认分"污染；2026-06-24 之后的评分
  与更早的探索性 run 不可直接混合平均。主跑报告（06-24）已在新语义下产出。
  详见 [`eval/METHODOLOGY.md`](eval/METHODOLOGY.md) §5.3。
