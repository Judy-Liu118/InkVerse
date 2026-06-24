"""
eval.analyze_judge_pingze_sensitivity -- F3 验证脚本

目的：retrospective 验证主跑 finding F3 "LLM-as-judge 对格律不敏感"。

方法：复用主跑 JSON 里的 base vs lora pairwise matchup（96 对 = 32 input × 3 run），
不重跑 API。按"双方平仄分差距"+"双方意境维度差距"分桶切片，看 lora 胜率/base
胜率是否真的对 pingze_diff 不敏感。

分析三块：
  1. 按 pingze_diff 分桶看 lora 胜率（命题：若评委爱格律 → 桶间应明显单调）
  2. controlled pair：base.pingze 极低 + base.意境维度 ≥ lora 的对决，看 base 胜率
  3. winner 与 pingze_diff 相关性（base 赢的 matchup 平均 pingze_diff 是否显著小）

输出：markdown 报告 + 关键数字 print。
"""
from __future__ import annotations

import json
import sys
import io
import argparse
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Windows console 中文输出兜底
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass


JUDGE_DIMS = ("intent", "imagery", "cohesion", "aesthetics")


def _gather_matchups(payload: dict, model_a: str, model_b: str) -> list[dict]:
    """从所有 run × 所有 input 抽出 (model_a, model_b) 这一对的 matchup，
    并合并每首 best_scores 的维度分到 matchup 字典里。
    """
    rows = []
    for run_idx, run in enumerate(payload["runs"]):
        res_by_model = run["results_per_model"]
        # input -> {model: scores}（按 input_idx 对齐）
        for input_idx, matchups in enumerate(run["matchups_per_input"]):
            scores_a = res_by_model[model_a][input_idx].get("best_scores") or {}
            scores_b = res_by_model[model_b][input_idx].get("best_scores") or {}
            if not scores_a or not scores_b:
                continue
            # 找 (model_a, model_b) 的 matchup
            m = next(
                (mu for mu in matchups
                 if {mu["model_a"], mu["model_b"]} == {model_a, model_b}),
                None,
            )
            if m is None:
                continue
            # winner 归一化为 "a"/"b"/"tie"/"all_swing"
            if m["winner"] == model_a:
                winner_norm = "a"
            elif m["winner"] == model_b:
                winner_norm = "b"
            else:
                winner_norm = m["winner"]  # "tie" / "all_swing" / "skip"
            rows.append({
                "run": run_idx + 1,
                "input_idx": input_idx,
                "user_input": res_by_model[model_a][input_idx]["user_input"],
                "genre": res_by_model[model_a][input_idx]["genre"],
                "scores_a": scores_a,
                "scores_b": scores_b,
                "matchup_orientation": (m["model_a"], m["model_b"]),
                "winner_raw": m["winner"],
                "winner_norm": winner_norm,
                "ma_votes": m["ma_votes"],
                "mb_votes": m["mb_votes"],
                "swing_count": m["swing_count"],
            })
    return rows


def _avg_judge_dims(scores: dict) -> float:
    """4 个 judge 维度的均值（intent/imagery/cohesion/aesthetics）"""
    vals = [scores.get(d) for d in JUDGE_DIMS if isinstance(scores.get(d), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def analysis_1_pingze_diff_buckets(rows: list[dict], model_a: str, model_b: str) -> list[str]:
    """按 (pingze_b - pingze_a) 分桶看 model_b 胜率。

    F3 命题：若评委不在意 pingze → 桶间胜率应大体接近。
    若评委爱 pingze → 桶间应明显单调（pingze_b 越高于 pingze_a，model_b 胜率越高）。
    """
    md = []
    md.append(f"\n## 分析 1：按 pingze_diff 分桶 ({model_b} 相对 {model_a})\n")
    md.append("**命题：** 若评委爱格律 → pingze_diff 越大，model_b 胜率越高（桶间单调）。\n")
    md.append("**反 F3 信号：** 单调性强。**支持 F3 信号：** 各桶胜率接近。\n")

    buckets = {
        "极端 ({m_b} 格律完美 vs {m_a} 严重出律): diff>=0.5": [],
        "中等 (0.2 <= diff < 0.5)": [],
        "微差 (0 < diff < 0.2)": [],
        "持平 (diff == 0)": [],
        "反向 (diff < 0, {m_b} 反而格律差)": [],
    }
    # 替换占位
    buckets = {k.replace("{m_a}", model_a).replace("{m_b}", model_b): v
               for k, v in buckets.items()}
    bucket_keys = list(buckets.keys())

    for r in rows:
        # 注意 matchup 方向：scores_a/scores_b 是固定 model_a/model_b 维度的，
        # 但 matchup 的 model_a/model_b 可能是反过来的。winner_norm 已归一为 a/b（按 scores 的 a/b）
        pz_a = r["scores_a"].get("pingze", 0.0)
        pz_b = r["scores_b"].get("pingze", 0.0)
        diff = pz_b - pz_a
        if diff >= 0.5:
            buckets[bucket_keys[0]].append(r)
        elif 0.2 <= diff < 0.5:
            buckets[bucket_keys[1]].append(r)
        elif 0 < diff < 0.2:
            buckets[bucket_keys[2]].append(r)
        elif diff == 0:
            buckets[bucket_keys[3]].append(r)
        else:
            buckets[bucket_keys[4]].append(r)

    md.append(f"| 分桶 | n | {model_b} 胜 | 平 | {model_a} 胜 | 全摇摆 | {model_b} 胜率（决断场次内） |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for k in bucket_keys:
        items = buckets[k]
        n = len(items)
        wins_b = sum(1 for r in items if r["winner_norm"] == "b")
        wins_a = sum(1 for r in items if r["winner_norm"] == "a")
        ties   = sum(1 for r in items if r["winner_norm"] == "tie")
        swings = sum(1 for r in items if r["winner_norm"] == "all_swing")
        decisive = wins_b + wins_a + ties
        win_rate = (wins_b + 0.5 * ties) / decisive if decisive else 0.0
        rate_str = f"{win_rate:.1%}" if decisive else "—"
        md.append(f"| {k} | {n} | {wins_b} | {ties} | {wins_a} | {swings} | **{rate_str}** |")

    # 单调性诊断：最强格律差距 vs 持平
    extreme = buckets[bucket_keys[0]]
    parity  = buckets[bucket_keys[3]]
    if extreme and parity:
        def _rate(items):
            wins_b = sum(1 for r in items if r["winner_norm"] == "b")
            wins_a = sum(1 for r in items if r["winner_norm"] == "a")
            ties   = sum(1 for r in items if r["winner_norm"] == "tie")
            d = wins_b + wins_a + ties
            return (wins_b + 0.5 * ties) / d if d else None
        r_ext = _rate(extreme)
        r_par = _rate(parity)
        if r_ext is not None and r_par is not None:
            md.append(f"\n**关键对比：** "
                      f"极端格律差距桶 {model_b} 胜率 {r_ext:.1%} (n={len(extreme)}) "
                      f"vs 格律持平桶 {model_b} 胜率 {r_par:.1%} (n={len(parity)})。")
            diff = r_ext - r_par
            if abs(diff) < 0.05:
                md.append(f"→ 差距 {diff:+.1%}，**支持 F3**：评委对 pingze 不敏感。")
            elif diff >= 0.15:
                md.append(f"→ 差距 {diff:+.1%}，**弱化 F3**：极端格律优势确实带来更高胜率。")
            else:
                md.append(f"→ 差距 {diff:+.1%}，**部分支持 F3**：有一定 pingze 敏感性但不显著。")
    return md


def analysis_2_controlled_pair(rows: list[dict], model_a: str, model_b: str) -> list[str]:
    """controlled pair：model_a 格律明显差（pingze<0.5）但意境维度 ≥ model_b 的对决。

    F3 强证据：若评委不爱格律 → 这种 pair 里 model_a 应该不显著输（甚至胜过）model_b。
    """
    md = []
    md.append(f"\n## 分析 2：controlled pair（{model_a} 格律差但意境 ≥ {model_b}）\n")
    md.append(f"**筛选条件：** {model_a}.pingze < 0.5 且 {model_a} 4 维 LLM 维度均值 ≥ {model_b}。\n")
    md.append("**命题：** F3 若成立 → 评委选 model_a 的比例应不显著低（>= 40%）。")
    md.append("若评委爱格律 → 此处 model_a 几乎全败给 model_b。\n")

    controlled = []
    for r in rows:
        pz_a = r["scores_a"].get("pingze", 0.0)
        pz_b = r["scores_b"].get("pingze", 0.0)
        avg_a = _avg_judge_dims(r["scores_a"])
        avg_b = _avg_judge_dims(r["scores_b"])
        if pz_a < 0.5 and avg_a >= avg_b:
            controlled.append({**r,
                               "pz_a": pz_a, "pz_b": pz_b,
                               "judge_avg_a": avg_a, "judge_avg_b": avg_b})

    if not controlled:
        md.append("⚠ 无符合条件的 controlled pair（可能是 BWS 已经把极端样本筛掉了）。")
        return md

    wins_a = sum(1 for r in controlled if r["winner_norm"] == "a")
    wins_b = sum(1 for r in controlled if r["winner_norm"] == "b")
    ties   = sum(1 for r in controlled if r["winner_norm"] == "tie")
    swings = sum(1 for r in controlled if r["winner_norm"] == "all_swing")
    decisive = wins_a + wins_b + ties
    a_rate = (wins_a + 0.5 * ties) / decisive if decisive else 0.0

    md.append(f"\n**结果：** n={len(controlled)} controlled pair · "
              f"{model_a} 胜={wins_a} 平={ties} 负={wins_b} 全摇摆={swings} · "
              f"**{model_a} 胜率 {a_rate:.1%}**（决断场次 {decisive} 内）")

    if a_rate >= 0.40:
        md.append(f"→ **强支持 F3**：尽管 {model_a} 格律差，评委因意境维度持平/略胜仍选它，"
                  f"证明评委对 pingze 权重 << 意境维度。")
    elif a_rate >= 0.25:
        md.append(f"→ **部分支持 F3**：{model_a} 胜率略低于无格律差距时的水平，"
                  f"但格律影响远小于意境维度。")
    else:
        md.append(f"→ **F3 弱化**：即便意境维度持平，{model_a} 也几乎全败 → 评委确实在意格律。")

    # 列出具体 pair 给读者审查
    md.append(f"\n**具体 pair（{model_a} 格律差但意境占优）：**\n")
    md.append(f"| run | input | {model_a}.pingze | {model_b}.pingze | "
              f"{model_a} 意境均值 | {model_b} 意境均值 | winner |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in controlled[:20]:  # 最多列 20 个避免报告过长
        ui = r["user_input"][:30] + "…" if len(r["user_input"]) > 30 else r["user_input"]
        winner_label = {
            "a": f"**{model_a}**", "b": model_b,
            "tie": "平", "all_swing": "全摇摆"
        }.get(r["winner_norm"], r["winner_norm"])
        md.append(f"| {r['run']} | {ui} | {r['pz_a']:.2f} | {r['pz_b']:.2f} | "
                  f"{r['judge_avg_a']:.3f} | {r['judge_avg_b']:.3f} | {winner_label} |")
    if len(controlled) > 20:
        md.append(f"\n_…后续 {len(controlled)-20} 条省略_")
    return md


def analysis_3_winner_vs_pingze(rows: list[dict], model_a: str, model_b: str) -> list[str]:
    """winner 与 pingze_diff 的关系：model_b 胜的 matchup 平均 pingze_diff 是否显著大于 model_a 胜的？"""
    md = []
    md.append(f"\n## 分析 3：winner 切片下的 pingze_diff 分布\n")
    md.append(f"**命题：** 若评委爱格律 → {model_b} 胜的 matchup 平均 pingze_diff 应明显 > "
              f"{model_a} 胜的 matchup。\n")

    b_wins_diffs = [r["scores_b"]["pingze"] - r["scores_a"]["pingze"]
                    for r in rows if r["winner_norm"] == "b"]
    a_wins_diffs = [r["scores_b"]["pingze"] - r["scores_a"]["pingze"]
                    for r in rows if r["winner_norm"] == "a"]

    def _summ(xs):
        if not xs: return None
        return {"n": len(xs), "mean": statistics.fmean(xs),
                "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
                "median": statistics.median(xs)}

    sb = _summ(b_wins_diffs)
    sa = _summ(a_wins_diffs)
    md.append(f"| winner | n | mean pingze_diff | std | median |")
    md.append(f"| --- | --- | --- | --- | --- |")
    if sb:
        md.append(f"| **{model_b}** 胜 | {sb['n']} | {sb['mean']:+.3f} | {sb['std']:.3f} | {sb['median']:+.3f} |")
    if sa:
        md.append(f"| **{model_a}** 胜 | {sa['n']} | {sa['mean']:+.3f} | {sa['std']:.3f} | {sa['median']:+.3f} |")

    if sb and sa:
        delta = sb["mean"] - sa["mean"]
        # 粗 Welch t-stat（不精确但够判断量级）
        pooled_std = ((sb["std"]**2 / sb["n"] + sa["std"]**2 / sa["n"]) ** 0.5
                      if sb["n"] > 1 and sa["n"] > 1 else None)
        md.append(f"\n**Δ mean = {delta:+.3f}**" +
                  (f"（粗 t ≈ {delta / pooled_std:+.2f}）" if pooled_std else ""))
        if abs(delta) < 0.1:
            md.append(f"→ **支持 F3**：winner 切片对 pingze_diff 几乎无区分度，评委不靠 pingze 决胜。")
        elif delta >= 0.2:
            md.append(f"→ **弱化 F3**：{model_b} 胜的 pair 平均 pingze_diff 高出 {delta:.2f}，"
                      f"格律确实是决胜因素之一。")
        else:
            md.append(f"→ **部分支持 F3**：{model_b} 胜的 pair pingze_diff 略高但差距不大。")
    return md


def main():
    ap = argparse.ArgumentParser(description="F3 验证：retrospective 分析主跑 pairwise")
    ap.add_argument("json_path", help="主跑 JSON 路径")
    ap.add_argument("--model-a", default="local_base",
                    help="对比对 A（默认 local_base，格律差侧）")
    ap.add_argument("--model-b", default="local_lora",
                    help="对比对 B（默认 local_lora，格律好侧）")
    ap.add_argument("--output-md", default=None,
                    help="markdown 报告输出路径（默认 stdout 同步落 eval/）")
    args = ap.parse_args()

    src = Path(args.json_path)
    if not src.exists():
        print(f"✗ 找不到 JSON：{src}", file=sys.stderr)
        sys.exit(1)

    print(f"加载主跑 JSON：{src}")
    with src.open(encoding="utf-8") as f:
        payload = json.load(f)
    print(f"  config: models={payload['config']['models']}  n={payload['config']['n']}  repeat={payload['config']['repeat']}")
    print(f"  对比对：{args.model_a} vs {args.model_b}")

    rows = _gather_matchups(payload, args.model_a, args.model_b)
    print(f"  收集到 {len(rows)} 个 matchup（{payload['config']['n']} input × {payload['config']['repeat']} run = 期望 {payload['config']['n'] * payload['config']['repeat']}）")

    md_lines = [
        f"# F3 验证报告 · {args.model_a} vs {args.model_b}",
        "",
        f"**数据源：** `{src.name}` · n={payload['config']['n']} × {payload['config']['repeat']} run · 评委={payload['config']['scorer']}",
        "",
        f"**F3 命题（待验证）：** LLM-as-judge 对格律（pingze）权重接近 0，主要看 intent/imagery/cohesion/aesthetics。",
        "",
        f"**方法：** retrospective 切片主跑 `matchups_per_input`（双向 pairwise + 评委一致票多数决），"
        f"不重跑 API。对每个 ({args.model_a}, {args.model_b}) matchup，按 (pingze_b - pingze_a) "
        f"分桶 / controlled pair / winner 切片三种方式分析。",
    ]

    md_lines += analysis_1_pingze_diff_buckets(rows, args.model_a, args.model_b)
    md_lines += analysis_2_controlled_pair(rows, args.model_a, args.model_b)
    md_lines += analysis_3_winner_vs_pingze(rows, args.model_a, args.model_b)

    md_lines += [
        "",
        "---",
        "",
        "## 综合结论",
        "",
        "三块分析交叉看：",
        "- 分析 1 桶间胜率单调性 → 评委是否按 pingze 加权",
        "- 分析 2 controlled pair → 直接测意境 vs 格律权衡",
        "- 分析 3 winner 切片均值差 → pingze_diff 在决胜中的贡献",
        "",
        "三块都支持 F3 = 强证据；不一致 = 评委对 pingze 的态度因评委/情境而异。",
        "",
        f"**注意：** 本分析受主跑 BWS 筛选影响 —— best 候选已经过 BWS 选出，极端格律样本"
        f"可能被筛掉，使 controlled pair n 偏小。若 n=0 或不显著，需重跑独立的 64-pair 实验。",
    ]

    md = "\n".join(md_lines)
    print()
    print(md)

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n报告已写入：{out}")


if __name__ == "__main__":
    main()
