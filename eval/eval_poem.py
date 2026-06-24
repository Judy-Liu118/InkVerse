"""
eval.eval_poem -- 诗歌生成质量同台对比（BWS 选 best → 跨模型 pairwise）

跑法：
    # 主跑：3 路 × 3 评委（跨家族 self-bias 抗性）
    python -m eval.eval_poem \
        --models local_base local_lora qwen-plus \
        --scorer deepseek-v4-pro qwen-max glm-4-plus --n 5

流程：
    每条 user_input × 每个 model:
      1. 生成 N 候选（默认 5）
      2. 每个候选跑 local_score_poem（rule-based 平仄/押韵/意象/连贯/切题）
      3. 每个评委独立做 BWS 5 选 1 → 多数决得 best；平票按本地 total 兜底
      4. best 跑 multi-judge 4 维评分（intent/imagery/cohesion/aesthetics）
    所有模型跑完一条 user_input 后:
      5. 跨模型 round-robin pairwise：每对 model × 每评委独立判，多数决定胜方
      6. 累计每模型的胜场 → 胜率

输出（outputs/eval/）:
    · markdown 报告，5 张表：
       §1 跨模型 pairwise 胜率（主表 / 上限对决）
       §2 候选分布（avg/std/min/selection_gain，基于本地分；稳定性）
       §3 best 4 维分（multi-judge；强项分析）
       §4 格律合规（pingze/rhyme 通过率）
       §5 全部候选诗 dump（含 BWS 投票详情 + best 维度分）
    · HTML 热图：胜率 + 维度 + 评委分歧
    · JSON：完整原始数据

约束:
    · 评委不应与参赛模型同名（self-bias 警告）
    · 多评委建议跨家族（DeepSeek / Qwen / 智谱）→ ≥3 评委 多数决
"""
from __future__ import annotations

import argparse
import statistics
import time
from itertools import combinations
from typing import Any, Dict, List, Tuple

from core.models.adapter import ModelAdapter
from core.poem.generator import PoemGenerator
from config import (
    DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, ZHIPU_API_KEY, MOONSHOT_API_KEY,
    POEM_QUALITY_THRESHOLD,
)

from eval.dataset import get_benchmark, BenchInput
from eval.metrics import summarize, pass_rate
from eval.report import (
    save_artifacts, table, fmt_num, print_and_return,
    rank_emoji, bold_max,
)

DIMS = ("total", "intent", "imagery", "cohesion", "aesthetics", "pingze", "rhyme")


# ── adapter ────────────────────────────────────────────────────────────────

def _make_adapter(model_choice: str) -> ModelAdapter:
    # alias: local_*_naked = 同后端 + prompt_mode='naked' (LoRA ablation 用)
    if model_choice in ("local_base", "local_base_naked"):
        return ModelAdapter(backend="local", allow_lora_fallback=False)
    if model_choice in ("local_lora", "local_lora_naked"):
        return ModelAdapter(backend="local_lora", allow_lora_fallback=False)
    if model_choice.startswith("deepseek"):
        return ModelAdapter(backend="deepseek", api_key=DEEPSEEK_API_KEY, api_model=model_choice)
    if model_choice.startswith("glm"):
        return ModelAdapter(backend="zhipu", api_key=ZHIPU_API_KEY, api_model=model_choice)
    if model_choice.startswith("kimi") or model_choice.startswith("moonshot"):
        return ModelAdapter(backend="moonshot", api_key=MOONSHOT_API_KEY, api_model=model_choice)
    return ModelAdapter(backend="qwen", api_key=DASHSCOPE_API_KEY, api_model=model_choice)


def _resolve_prompt_mode(model_choice: str) -> str:
    """eval 评测：默认 'full'（与 API 同样的 system+格式约束，公平对照）。
    带 _naked 后缀 = 'naked'（保留生产行为，验证 LoRA 微调是否内化引导）。
    """
    return "naked" if model_choice.endswith("_naked") else "full"


# ── 单模型流程：生成 → 本地分 → BWS 选 best → best multi-judge ────────────

def _run_one(gen: PoemGenerator, item: BenchInput, model_choice: str,
             judges: List[Tuple[str, ModelAdapter]],
             candidate_count: int) -> Dict[str, Any]:
    adapter = _make_adapter(model_choice)
    prompt_mode = _resolve_prompt_mode(model_choice)

    # 把"模型加载"从"纯生成耗时"里剥离，§4 表的对比才公平。
    # API 后端 preload_model 是 no-op，load_elapsed 接近 0。
    t_load_start = time.time()
    gen.preload_model(adapter)
    load_elapsed = time.time() - t_load_start

    t0 = time.time()
    genre_name, poems = gen.generate_candidates_only(
        item.user_input, generation_adapter=adapter, count=candidate_count,
        prompt_mode=prompt_mode,
    )
    gen_elapsed = time.time() - t0

    # 体裁信息（num_lines/chars_per_line）用于本地分
    _, num_lines, chars_per_line = gen.scorer.detect_genre(item.user_input)

    if not poems:
        print(f"      ⚠ [{model_choice}] 0 候选（生成全失败）")
        return {
            "user_input": item.user_input, "genre": item.genre, "theme": item.theme,
            "model": model_choice, "candidates": [], "best_idx": None,
            "best_scores": {}, "best_poem": "", "bws_votes": {},
            "error": "no candidates", "elapsed_sec": round(gen_elapsed, 2),
        }

    print(f"      [{model_choice}] 生成 {len(poems)}/{candidate_count} 候选 ({gen_elapsed:.1f}s)")
    for i, p in enumerate(poems):
        print(f"        c{i+1}: {p.replace(chr(10), ' / ')}")

    # 1. 每个候选跑 local_score_poem（rule-based，无 LLM 调用）
    #    供 avg/std/min/selection_gain 计算 + BWS 平票兜底
    local_scores = []
    for i, p in enumerate(poems):
        try:
            ls = gen.scorer.local_score_poem(p, num_lines, chars_per_line, topic_score=0.7)
        except Exception as e:
            ls = {"total": 0.0, "pingze": 0.0, "rhyme": 0.0,
                  "imagery": 0.0, "cohesion": 0.0, "topic": 0.0}
            print(f"        ⚠ c{i+1} 本地评分异常：{e}")
        local_scores.append(ls)

    # 2. 每个评委独立 BWS 5 选 1（best_idx=-1 表示评委弃权，回复不可解析）
    print(f"      [{model_choice}] BWS 投票（{len(judges)} 评委 × {len(poems)} 候选）")
    bws_votes: Dict[str, int] = {}  # judge_label -> picked idx (-1 = 弃权)
    bws_raw: Dict[str, str] = {}
    for label, jadapter in judges:
        try:
            pick = gen.scorer.pick_best_via_bws(poems, item.user_input, jadapter)
            bws_votes[label] = pick["best_idx"]
            bws_raw[label] = pick["raw_reply"]
            raw_short = pick["raw_reply"].replace("\n", " ⏎ ")[:40]
            if pick["best_idx"] == -1:
                print(f"        {label} ⚠ 弃权（回复不可解析 raw={raw_short!r}）")
            else:
                print(f"        {label} 选 c{pick['best_idx']+1}  (raw={raw_short!r})")
        except Exception as e:
            bws_votes[label] = -1
            bws_raw[label] = f"ERR: {e}"
            print(f"        {label} ⚠ BWS 异常 → 弃权：{e}")

    # 3. 多数决：只统计有效票（>= 0）
    vote_count: Dict[int, int] = {}
    valid_votes = [v for v in bws_votes.values() if v >= 0]
    for idx in valid_votes:
        vote_count[idx] = vote_count.get(idx, 0) + 1
    abstain_count = len(bws_votes) - len(valid_votes)

    if not valid_votes:
        # 全员弃权 → 按本地 total 兜底
        best_idx = max(range(len(poems)), key=lambda i: local_scores[i].get("total", 0.0))
        tiebreak = "all_abstain_local_total"
        print(f"      [{model_choice}] ⚠ 所有评委弃权 → 按本地分兜底 best = c{best_idx+1}")
    else:
        max_votes = max(vote_count.values())
        tied = [idx for idx, cnt in vote_count.items() if cnt == max_votes]
        if len(tied) == 1:
            best_idx = tied[0]
            tiebreak = None
        else:
            best_idx = max(tied, key=lambda i: local_scores[i].get("total", 0.0))
            tiebreak = "local_total"
        tb_note = ""
        if tiebreak:
            tb_note = f", tiebreak=local"
        if abstain_count:
            tb_note += f", 弃权={abstain_count}"
        print(f"      [{model_choice}] best = c{best_idx+1} "
              f"(votes={vote_count.get(best_idx, 0)}/{len(valid_votes)} 有效{tb_note})")

    best_poem = poems[best_idx]

    # 4. best 单独跑 multi-judge 4 维评分（辅助指标 + 评委分歧分析）
    try:
        best_scores = gen.scorer.score_single_multi_judge(
            best_poem, item.user_input, judges, candidate_index=best_idx + 1,
        )
        sbj = best_scores.get("scores_by_judge", {})
        def _javg(jd):
            vals = [v for v in jd.values() if isinstance(v, (int, float))]
            return sum(vals) / len(vals) if vals else 0.0
        judge_brief = " ".join(
            f"{lbl}={fmt_num(_javg(sbj.get(lbl, {})))}" for lbl, _ in judges
        ) if sbj else ""
        print(f"      [{model_choice}] best 4 维：total={fmt_num(best_scores.get('total', 0.0))}"
              f"  intent={fmt_num(best_scores.get('intent', 0))}"
              f"  imagery={fmt_num(best_scores.get('imagery', 0))}"
              f"  cohesion={fmt_num(best_scores.get('cohesion', 0))}"
              f"  aesthetics={fmt_num(best_scores.get('aesthetics', 0))}"
              f"  | {judge_brief}")
    except Exception as e:
        best_scores = {}
        print(f"      [{model_choice}] best 4 维评分异常：{e}")

    # 5. 候选分布统计（基于本地 total，作"模型平均水平 + 稳定性"指标）
    local_totals = [ls.get("total", 0.0) for ls in local_scores]
    dist = {}
    if local_totals:
        dist["mean_total"] = sum(local_totals) / len(local_totals)
        dist["max_total"] = max(local_totals)
        dist["min_total"] = min(local_totals)
        dist["std_total"] = statistics.stdev(local_totals) if len(local_totals) > 1 else 0.0
        dist["selection_gain"] = dist["max_total"] - dist["mean_total"]
        dist["pass_rate_07"] = sum(1 for t in local_totals if t >= POEM_QUALITY_THRESHOLD) / len(local_totals)
        dist["n_candidates"] = len(local_totals)
        # 多样性：唯一候选数 / 总候选数（暴露 mode collapse）
        # 比较时统一去掉空白与换行差异，仅看实际字符序列
        normalized = {"".join(p.split()) for p in poems}
        dist["diversity"] = len(normalized) / len(poems)
        dist["unique_count"] = len(normalized)

    # candidates 列表保留 5 候选 + 本地分（兼容旧报告 dump 章节）
    cand_results = [
        {"poem": p, "local_scores": ls, "is_best": (i == best_idx), "error": None}
        for i, (p, ls) in enumerate(zip(poems, local_scores))
    ]

    elapsed = time.time() - t0
    return {
        "user_input": item.user_input, "genre": item.genre, "theme": item.theme,
        "model": model_choice, "genre_name": genre_name,
        "num_lines": num_lines, "chars_per_line": chars_per_line,
        "candidates": cand_results,
        "best_idx": best_idx, "best_poem": best_poem, "best_scores": best_scores,
        "bws_votes": bws_votes, "bws_raw": bws_raw,
        "vote_count": vote_count, "tiebreak": tiebreak,
        "distribution": dist,
        "elapsed_sec": round(elapsed, 2),
        "gen_elapsed_sec": round(gen_elapsed, 2),
        "load_elapsed_sec": round(load_elapsed, 2),
    }


# ── 跨模型 pairwise round-robin（双向：AB + BA 验证一致性） ────────────────

def _cross_model_pairwise(
    item: BenchInput, model_to_result: Dict[str, Dict[str, Any]],
    judges: List[Tuple[str, ModelAdapter]], scorer,
) -> List[Dict[str, Any]]:
    """对每对模型 best 跑双向 pairwise：每评委独立判 forward(A=ma, B=mb) +
    reverse(A=mb, B=ma) 各 1 次。

    评委一致性：
      · forward=A, reverse=B → 两次都说 ma 更好 → 该评委有效票投 ma
      · forward=B, reverse=A → 两次都说 mb 更好 → 该评委有效票投 mb
      · forward=A, reverse=A → 评委摇摆（永远偏 A 位）→ 无效票
      · forward=B, reverse=B → 评委摇摆（永远偏 B 位）→ 无效票

    胜方由 consistent 票多数决（摇摆票不算）。摇摆率高 → 暴露 position bias。
    """
    models = list(model_to_result.keys())
    pairs = list(combinations(models, 2))
    matchups = []
    print(f"      跨模型 pairwise：{len(pairs)} 对 × {len(judges)} 评委 × 双向")
    for ma, mb in pairs:
        ra = model_to_result.get(ma, {})
        rb = model_to_result.get(mb, {})
        poem_a = ra.get("best_poem", "")
        poem_b = rb.get("best_poem", "")
        if not poem_a or not poem_b:
            print(f"        {ma} vs {mb}: 一方无 best，跳过")
            matchups.append({"model_a": ma, "model_b": mb,
                             "poem_a": poem_a, "poem_b": poem_b,
                             "judge_votes": {}, "winner": "skip",
                             "ma_votes": 0, "mb_votes": 0, "swing_count": 0})
            continue

        judge_votes: Dict[str, Dict[str, Any]] = {}
        ma_votes, mb_votes, swing = 0, 0, 0
        for label, jadapter in judges:
            try:
                v_fwd = scorer.compare_poems(poem_a, poem_b, item.user_input, jadapter)
                v_rev = scorer.compare_poems(poem_b, poem_a, item.user_input, jadapter)
            except Exception as e:
                v_fwd, v_rev = "A", "A"
                print(f"          {label} ⚠ compare 异常：{e}")
            # 一致性判定
            if v_fwd == "A" and v_rev == "B":
                # 两次都说 ma 更好（fwd 选 A=ma；rev 选 B=ma）
                judge_pick = ma
                consistent = True
                ma_votes += 1
            elif v_fwd == "B" and v_rev == "A":
                # 两次都说 mb 更好（fwd 选 B=mb；rev 选 A=mb）
                judge_pick = mb
                consistent = True
                mb_votes += 1
            else:
                # 摇摆：fwd=A rev=A（一直偏 A 位）或 fwd=B rev=B（一直偏 B 位）
                judge_pick = "swing"
                consistent = False
                swing += 1
            judge_votes[label] = {
                "forward": v_fwd, "reverse": v_rev,
                "consistent": consistent, "pick": judge_pick,
            }
        if ma_votes > mb_votes:
            winner = ma
        elif mb_votes > ma_votes:
            winner = mb
        elif ma_votes == 0 and mb_votes == 0:
            winner = "all_swing"
        else:
            winner = "tie"
        consistent_total = ma_votes + mb_votes
        consistency_rate = consistent_total / len(judges) if judges else 0.0
        print(f"        {ma} vs {mb}: {ma}={ma_votes} {mb}={mb_votes} 摇摆={swing} → {winner} "
              f"(一致率 {consistency_rate:.0%})")
        matchups.append({
            "model_a": ma, "model_b": mb,
            "poem_a": poem_a, "poem_b": poem_b,
            "judge_votes": judge_votes,
            "winner": winner,
            "ma_votes": ma_votes, "mb_votes": mb_votes, "swing_count": swing,
            "consistency_rate": consistency_rate,
        })
    return matchups


# ── 聚合 ──────────────────────────────────────────────────────────────────

def _aggregate(per_input_results: List[Dict[str, Any]],
               matchups_per_input: List[List[Dict[str, Any]]],
               label: str) -> Dict[str, Any]:
    """跨 user_input 聚合一个 model 的成绩。"""
    valid = [r for r in per_input_results if r.get("candidates")]
    out = {"model": label, "n_inputs": len(valid)}

    # 候选分布：avg / std / min / max / selection_gain（基于本地 total）
    mean_totals = [r["distribution"].get("mean_total", 0.0) for r in valid]
    max_totals  = [r["distribution"].get("max_total",  0.0) for r in valid]
    min_totals  = [r["distribution"].get("min_total",  0.0) for r in valid]
    std_totals  = [r["distribution"].get("std_total",  0.0) for r in valid]
    sel_gains   = [r["distribution"].get("selection_gain", 0.0) for r in valid]
    pass_07s    = [r["distribution"].get("pass_rate_07", 0.0) for r in valid]
    out["mean_total"]     = summarize(mean_totals)
    out["max_total"]      = summarize(max_totals)
    out["min_total"]      = summarize(min_totals)
    out["std_total"]      = summarize(std_totals)
    out["selection_gain"] = summarize(sel_gains)
    out["pass_rate_07"]   = summarize(pass_07s)
    diversities = [r["distribution"].get("diversity", 1.0) for r in valid]
    out["diversity"] = summarize(diversities)

    # best 4 维分（multi-judge）
    for d in DIMS:
        vals = [r.get("best_scores", {}).get(d, 0.0) for r in valid
                if r.get("best_scores")]
        out[d] = summarize(vals)

    # 格律通过率（基于 best）
    out["pingze_pass@0.8"] = pass_rate(
        [r.get("best_scores", {}).get("pingze", 0.0) for r in valid if r.get("best_scores")], 0.8,
    )
    out["rhyme_pass@0.8"] = pass_rate(
        [r.get("best_scores", {}).get("rhyme", 0.0) for r in valid if r.get("best_scores")], 0.8,
    )
    out["mean_gen_elapsed_sec"] = summarize(
        [r.get("gen_elapsed_sec", 0.0) for r in valid]
    )["mean"]
    out["mean_load_elapsed_sec"] = summarize(
        [r.get("load_elapsed_sec", 0.0) for r in valid]
    )["mean"]

    # 跨模型胜率（基于双向一致票）+ 评委一致率
    wins, losses, ties, all_swing, plays = 0, 0, 0, 0, 0
    label_consistent_votes = 0   # 评委判 label 胜的有效票数（双向一致）
    label_total_judge_calls = 0  # 评委对该模型参与的总判定数（一致+摇摆）
    for matchups in matchups_per_input:
        for m in matchups:
            if label not in (m["model_a"], m["model_b"]):
                continue
            if m["winner"] == "skip":
                continue
            plays += 1
            if m["winner"] == label:
                wins += 1
            elif m["winner"] == "tie":
                ties += 1
            elif m["winner"] == "all_swing":
                all_swing += 1
            else:
                losses += 1
            # 累计该模型在所有对决中的有效票
            if m["model_a"] == label:
                label_consistent_votes += m.get("ma_votes", 0)
            else:
                label_consistent_votes += m.get("mb_votes", 0)
            label_total_judge_calls += (m.get("ma_votes", 0) + m.get("mb_votes", 0)
                                         + m.get("swing_count", 0))

    out["pairwise_wins"] = wins
    out["pairwise_losses"] = losses
    out["pairwise_ties"] = ties
    out["pairwise_all_swing"] = all_swing
    out["pairwise_plays"] = plays
    # win_rate 算法不变（tie 半分；all_swing 视为无有效结论 → 不计入分母）
    decided_plays = plays - all_swing
    out["win_rate"] = (wins + 0.5 * ties) / decided_plays if decided_plays else 0.0
    # 评委一致率：判 label 胜的有效票 / 总判定数（含摇摆）
    out["judge_consistency_rate"] = (
        label_consistent_votes / label_total_judge_calls
        if label_total_judge_calls else 0.0
    )
    return out


def _rank_row(values: List[float]) -> List[str]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: -values[i])
    rank_of = [0] * len(values)
    prev_val = None
    prev_rank = 0
    for r, idx in enumerate(order):
        if prev_val is None or values[idx] < prev_val - 1e-9:
            prev_rank = r
            prev_val = values[idx]
        rank_of[idx] = prev_rank
    return [rank_emoji(r) for r in rank_of]


# ── markdown 报告 ─────────────────────────────────────────────────────────

def _render_markdown(args, models, aggs, results_per_model, matchups_per_input,
                     input_order):
    md = []
    md.append(f"# eval_poem 报告 · {' vs '.join(models)}")
    judge_str = " + ".join(args.scorer)
    bias_note = ""
    if len(args.scorer) >= 3:
        bias_note = "（≥3 评委 · BWS 选 best + 跨模型 pairwise 多数决 · 跨家族抗 self-bias）"
    elif len(args.scorer) == 2:
        bias_note = "（2 评委 · BWS + pairwise · 平票按本地分兜底）"
    md.append(
        f"_n={args.n}（{args.genres or '全部体裁'} / {args.density or '全部密度'}）"
        f" · 评委={judge_str} · 候选数={args.candidates}_"
        f"{bias_note}"
    )
    md.append("")
    md.append("**方法论：** 每模型生成 N 候选 → 评委独立 BWS N 选 1 多数决得 best → "
              "三模型 best 跨模型 round-robin **双向** pairwise"
              "（每对 × 每评委 forward+reverse 各 1 次，两次一致才计有效票，"
              "摇摆=position bias 暴露）。BWS 与 pairwise 都不依赖绝对评分，规避评分饱和。")
    md.append("")
    # Prompt 模式说明：列出每个模型的 prompt_mode（full = 公平对照；naked = ablation）
    prompt_modes = {m: _resolve_prompt_mode(m) for m in models}
    has_naked = any(v == "naked" for v in prompt_modes.values())
    if has_naked:
        modes_brief = "、".join(f"`{m}`={mode}" for m, mode in prompt_modes.items())
        md.append(f"**Prompt 模式：** {modes_brief}")
        md.append("- `full`：本地模型接收与 API 相同的 system + 格式约束 + 用户要求（controlled experiment）")
        md.append("- `naked`：本地模型仅接收简短 user request（LoRA ablation，验证微调是否内化格式引导）")
    else:
        md.append("**Prompt 模式：** 所有本地模型与 API 一律接收同样的 system + 格式约束 + 用户要求，"
                  "为公平对照。生产中 LoRA 接简短 user request 即可（见 `_naked` ablation）。")
    md.append("")

    # ── §1 跨模型 pairwise 胜率（主表：上限对决）
    md.append("## 1. 跨模型 pairwise 胜率（主表 · 上限对决，双向一致票）")
    md.append("_每对 best × 每评委 forward+reverse 各 1 次；两次都判同一首胜 = 有效票，"
              "否则记为摇摆（评委永远偏 A 或永远偏 B 位）。胜率 = (胜+0.5×平) / 决断场次。_")
    md.append("")

    # 计算全局摇摆率
    total_judge_calls, total_swing = 0, 0
    for matchups in matchups_per_input:
        for m in matchups:
            if m["winner"] == "skip":
                continue
            total_swing += m.get("swing_count", 0)
            total_judge_calls += (m.get("ma_votes", 0) + m.get("mb_votes", 0)
                                  + m.get("swing_count", 0))
    global_swing_rate = total_swing / total_judge_calls if total_judge_calls else 0.0
    md.append(f"**全局评委一致率：{1 - global_swing_rate:.0%}**"
              f"（摇摆率 {global_swing_rate:.0%}；摇摆率高 → 评委存在 position bias，胜率结论需打折）")
    md.append("")

    headers = ["指标"] + models + ["排名"]
    win_rates = [aggs[m]["win_rate"] for m in models]
    rows = [["胜率（决断场次内）"] + bold_max(win_rates)
            + [" ".join(f"{e}{m}" for e, m in zip(_rank_row(win_rates), models))]]
    rows.append(["胜 / 平 / 负 / 全摇摆"]
                + [f"{aggs[m]['pairwise_wins']} / {aggs[m]['pairwise_ties']} / "
                   f"{aggs[m]['pairwise_losses']} / {aggs[m]['pairwise_all_swing']}"
                   for m in models] + [""])
    rows.append(["总场次"] + [str(aggs[m]["pairwise_plays"]) for m in models] + [""])
    rows.append(["对该模型评委有效胜票占比"]
                + [f"{aggs[m]['judge_consistency_rate']:.1%}" for m in models]
                + [""])
    md.append(table(headers, rows))
    md.append("")

    # 单独把每对 head-to-head 列出来
    md.append("**Head-to-head（每对模型的累计比分）：**")
    md.append("")
    pair_rows = []
    pair_index: Dict[Tuple[str, str], Dict[str, int]] = {}
    for matchups in matchups_per_input:
        for m in matchups:
            if m["winner"] == "skip":
                continue
            key = tuple(sorted([m["model_a"], m["model_b"]]))
            pair_index.setdefault(key, {"a_wins": 0, "b_wins": 0, "ties": 0,
                                       "all_swing": 0})
            entry = pair_index[key]
            if m["winner"] == "tie":
                entry["ties"] += 1
            elif m["winner"] == "all_swing":
                entry["all_swing"] += 1
            elif m["winner"] == key[0]:
                entry["a_wins"] += 1
            else:
                entry["b_wins"] += 1
    for (ma, mb), entry in pair_index.items():
        pair_rows.append([
            f"`{ma}` vs `{mb}`",
            f"{entry['a_wins']}",
            f"{entry['b_wins']}",
            f"{entry['ties']}",
            f"{entry['all_swing']}",
        ])
    if pair_rows:
        md.append(table(["对决", "前者胜", "后者胜", "平", "全摇摆"], pair_rows))
    md.append("")

    # ── §2 候选分布（平均水平 + 稳定性）
    md.append("## 2. 候选分布（辅表 · 平均水平 + 稳定性）")
    md.append("_基于每模型 N 候选的本地分（rule-based 平仄/押韵/意象/连贯）。"
              "看的是\"典型一次采样多好 + 稳不稳\"，区别于 §1 的\"最强能写多好\"。_")
    md.append("")
    metrics_to_show = [
        ("avg 候选本地总分（期望产出质量）",      "mean_total"),
        ("std 候选内方差（稳定性，越低越稳）",   "std_total"),
        ("min 候选本地总分（worst-case）",       "min_total"),
        ("selection_gain（max − avg，候选选择价值）", "selection_gain"),
        (f"pass@{POEM_QUALITY_THRESHOLD} 候选合格率", "pass_rate_07"),
        ("候选多样性（唯一候选/N，低=mode collapse）", "diversity"),
    ]
    rows = []
    for human, key in metrics_to_show:
        means = [aggs[m][key]["mean"] for m in models]
        if key == "std_total":
            cells = [fmt_num(v) for v in means]
            mn = min(means)
            cells = [f"**{c}**" if abs(v - mn) < 1e-9 else c for c, v in zip(cells, means)]
            rank_emojis = _rank_row([-v for v in means])
        else:
            cells = bold_max(means)
            rank_emojis = _rank_row(means)
        rank_summary = " ".join(f"{e}{m}" for e, m in zip(rank_emojis, models))
        rows.append([human] + cells + [rank_summary])
    md.append(table(["指标"] + models + ["排名"], rows))
    md.append("")

    # ── §3 best 4 维分（强项分析）
    md.append("## 3. best 4 维分（辅表 · 强项分析）")
    md.append("_仅对每模型的 best 候选跑 multi-judge；intent/imagery/cohesion/aesthetics "
              "由 ≥3 评委独立打分取中位数。_")
    md.append("")
    rows = []
    for d in DIMS:
        means = [aggs[m][d]["mean"] for m in models]
        rank_emojis = _rank_row(means)
        rank_summary = " ".join(f"{e}{m}" for e, m in zip(rank_emojis, models))
        rows.append([d] + bold_max(means) + [rank_summary])
    md.append(table(["维度"] + models + ["排名"], rows))
    md.append("")

    # ── §4 格律合规与速度
    md.append("## 4. 格律合规与速度（辅表 · 硬指标）")
    md.append("")
    rate_rows = [
        ["平仄合格率 (≥0.8)"] + [f"{aggs[m]['pingze_pass@0.8']:.1%}" for m in models],
        ["押韵合格率 (≥0.8)"] + [f"{aggs[m]['rhyme_pass@0.8']:.1%}" for m in models],
        ["纯生成耗时均值 (s, 不含 load)"]
        + [fmt_num(aggs[m]["mean_gen_elapsed_sec"], 2) for m in models],
        ["模型 load 耗时均值 (s)"]
        + [fmt_num(aggs[m]["mean_load_elapsed_sec"], 2) for m in models],
    ]
    md.append(table(["指标"] + models, rate_rows))
    md.append("")

    # ── §5 全部候选诗 dump（含 BWS 投票详情）
    md.append("## 5. 全部候选诗 + BWS 投票详情")
    md.append("_格式：每模型 N 候选，标★为 best，列本地分；并展示 N 评委的 BWS 投票分布。_")
    md.append("")
    judge_labels = args.scorer
    for input_idx, item_input in enumerate(input_order):
        md.append(f"### {input_idx + 1}. {item_input}")
        md.append("")
        for m in models:
            for r in results_per_model[m]:
                if r["user_input"] != item_input:
                    continue
                if not r.get("candidates"):
                    md.append(f"- `{m}`：无可用候选")
                    continue
                best_idx = r.get("best_idx", -1)
                votes = r.get("bws_votes", {})
                vote_count = r.get("vote_count", {})
                tiebreak = r.get("tiebreak")

                # BWS 投票汇总（-1 = 弃权）
                vote_parts = []
                for lbl in judge_labels:
                    if lbl not in votes:
                        continue
                    v = votes[lbl]
                    vote_parts.append(f"{lbl}→弃权" if v < 0 else f"{lbl}→c{v+1}")
                vote_summary = ", ".join(vote_parts)
                tb_msg = {
                    None: "",
                    "local_total": "  ⚠ 平票，按本地分兜底",
                    "all_abstain_local_total": "  ⚠ 全员弃权，按本地分兜底",
                }.get(tiebreak, f"  ⚠ {tiebreak}")
                md.append(f"**`{m}`** — best=**c{best_idx+1}**（票数：{vote_count}）  "
                          f"BWS 投票：{vote_summary}{tb_msg}")
                md.append("")

                rows = []
                for ci, cand in enumerate(r["candidates"]):
                    ls = cand.get("local_scores", {})
                    star = "★" if ci == best_idx else ""
                    rows.append([
                        f"{star}c{ci+1}",
                        fmt_num(ls.get("total", 0.0)),
                        fmt_num(ls.get("pingze", 0.0)),
                        fmt_num(ls.get("rhyme", 0.0)),
                        fmt_num(ls.get("imagery", 0.0)),
                        fmt_num(ls.get("cohesion", 0.0)),
                        cand["poem"].replace("\n", " / "),
                    ])
                md.append(table(
                    ["候选", "local_total", "pingze", "rhyme", "imagery", "cohesion", "诗"],
                    rows,
                ))

                # 如果有 best 的 multi-judge 分，单独列
                bs = r.get("best_scores", {})
                if bs:
                    sbj = bs.get("scores_by_judge", {})
                    judge_lines = []
                    for j in judge_labels:
                        jd = sbj.get(j, {})
                        if not jd:
                            continue
                        judge_lines.append(
                            f"`{j}` intent={fmt_num(jd.get('intent', 0))} "
                            f"imagery={fmt_num(jd.get('imagery', 0))} "
                            f"cohesion={fmt_num(jd.get('cohesion', 0))} "
                            f"aesthetics={fmt_num(jd.get('aesthetics', 0))}"
                        )
                    md.append(
                        f"  · best 多评委 4 维分（合成后）: "
                        f"total=**{fmt_num(bs.get('total', 0))}** "
                        f"intent={fmt_num(bs.get('intent', 0))} "
                        f"imagery={fmt_num(bs.get('imagery', 0))} "
                        f"cohesion={fmt_num(bs.get('cohesion', 0))} "
                        f"aesthetics={fmt_num(bs.get('aesthetics', 0))}"
                    )
                    if judge_lines:
                        md.append("")
                        for jl in judge_lines:
                            md.append(f"    - {jl}")
                md.append("")

        # 跨模型 pairwise 对决详情（双向：每评委显示 forward/reverse + 一致性）
        matchups = next(
            (ms for ms, ui in zip(matchups_per_input, input_order) if ui == item_input),
            [],
        )
        if matchups:
            md.append(f"**跨模型对决（双向）：**")
            md.append("")
            for mu in matchups:
                if mu["winner"] == "skip":
                    continue
                jv = mu.get("judge_votes", {})
                parts = []
                for lbl, info in jv.items():
                    if not isinstance(info, dict):
                        continue
                    f_, r_ = info.get("forward", "?"), info.get("reverse", "?")
                    if info.get("consistent"):
                        parts.append(f"{lbl}→{info.get('pick')}({f_}/{r_})")
                    else:
                        parts.append(f"{lbl}→**摇摆**({f_}/{r_})")
                vstr = ", ".join(parts)
                winner_lbl = mu["winner"]
                if winner_lbl == "all_swing":
                    winner_lbl = "**全员摇摆 / 无结论**"
                elif winner_lbl == "tie":
                    winner_lbl = "平"
                md.append(f"- `{mu['model_a']}` vs `{mu['model_b']}`: "
                          f"{vstr}  → **{winner_lbl}** "
                          f"({mu['model_a']}={mu.get('ma_votes', 0)} "
                          f"{mu['model_b']}={mu.get('mb_votes', 0)} "
                          f"摇摆={mu.get('swing_count', 0)})")
            md.append("")

    # ── §6 best 抽样
    md.append("## 6. 抽样诗作（每模型 best 候选）")
    for input_idx, item_input in enumerate(input_order[: args.samples]):
        md.append(f"### {input_idx + 1}. {item_input}")
        bests = {}
        for m in models:
            for r in results_per_model[m]:
                if r["user_input"] != item_input:
                    continue
                if r.get("best_poem"):
                    bests[m] = r
        # 排序：按跨模型胜率排或按 best total
        for m in models:
            if m not in bests:
                continue
            r = bests[m]
            bt = r.get("best_scores", {}).get("total", 0.0)
            md.append(f"- **`{m}`** (best total={fmt_num(bt)})  ")
            md.append("  ```\n  " + r["best_poem"].replace("\n", "\n  ") + "\n  ```")
        md.append("")
    return "\n".join(md)


def _unique_user_inputs(results_per_model, models):
    seen, out = set(), []
    for m in models:
        for r in results_per_model[m]:
            if r["user_input"] not in seen:
                seen.add(r["user_input"])
                out.append(r["user_input"])
    return out


# ── 多 run 支持（Task #61）────────────────────────────────────────────────

def _run_pipeline(args, models: List[str],
                  judges: List[Tuple[str, ModelAdapter]],
                  inputs: List[BenchInput], gen: "PoemGenerator",
                  run_idx: int = 0,
                  ) -> Tuple[Dict[str, Dict[str, Any]],
                              Dict[str, List[Dict[str, Any]]],
                              List[List[Dict[str, Any]]],
                              List[str]]:
    """跑一次完整 pipeline（generate → BWS → multi-judge → pairwise → aggregate）。

    抽出来是为支持 --repeat：多 run 时同一组 inputs 在 LLM temperature noise
    下产生不同候选 / 不同评委判断，跨 run 算 mean ± std。
    """
    results_per_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}
    matchups_per_input: List[List[Dict[str, Any]]] = []
    input_order: List[str] = []
    run_tag = f"[run {run_idx + 1}]" if run_idx is not None else ""

    for i, item in enumerate(inputs):
        print(f"  {run_tag}[{i+1}/{len(inputs)}] {item.user_input[:30]}…")
        input_order.append(item.user_input)
        model_to_result: Dict[str, Dict[str, Any]] = {}
        for m in models:
            try:
                r = _run_one(gen, item, m, judges, args.candidates)
                results_per_model[m].append(r)
                model_to_result[m] = r
                if r.get("distribution"):
                    d = r["distribution"]
                    bs_total = r.get("best_scores", {}).get("total", 0.0)
                    div = d.get("diversity", 1.0)
                    div_flag = " ⚠ mode_collapse" if div <= 0.4 else ""
                    print(f"      {m} best_total={fmt_num(bs_total)} "
                          f"avg={fmt_num(d['mean_total'])} std={fmt_num(d['std_total'])} "
                          f"gain={fmt_num(d['selection_gain'])} "
                          f"diversity={div:.0%}{div_flag}")
            except Exception as e:
                print(f"      ⚠ {m} 异常：{e}")
                err_r = {"user_input": item.user_input, "model": m,
                         "candidates": [], "best_idx": None, "best_poem": "",
                         "best_scores": {}, "bws_votes": {}, "error": str(e)}
                results_per_model[m].append(err_r)
                model_to_result[m] = err_r

        try:
            matchups = _cross_model_pairwise(
                item, model_to_result, judges, gen.scorer,
            )
        except Exception as e:
            print(f"      ⚠ 跨模型 pairwise 异常：{e}")
            matchups = []
        matchups_per_input.append(matchups)

    aggs = {
        m: _aggregate(results_per_model[m], matchups_per_input, m)
        for m in models
    }
    return aggs, results_per_model, matchups_per_input, input_order


# 跨 run mean ± std 聚合的 metric 分类
_SCALAR_METRICS    = ("win_rate", "judge_consistency_rate")
_SUMMARY_METRICS   = ("mean_total", "max_total", "min_total", "std_total",
                       "selection_gain", "pass_rate_07", "diversity") + DIMS
_RATE_METRICS      = ("pingze_pass@0.8", "rhyme_pass@0.8")
_INT_METRICS       = ("pairwise_wins", "pairwise_losses", "pairwise_ties",
                       "pairwise_all_swing", "pairwise_plays")


def _aggregate_across_runs(aggs_per_run: List[Dict[str, Dict[str, Any]]],
                           models: List[str]) -> Dict[str, Dict[str, Any]]:
    """跨 N 次 run，把每个 metric 算 mean ± std。

    每模型输出 dict 含：
      · {metric}_runs : List[float]  —— 每 run 的值
      · {metric}_mean : float        —— 跨 run 均值
      · {metric}_std  : float        —— 跨 run population std
      · 整数 metric 额外 {metric}_total : int  —— 跨 run 累计
    """
    out: Dict[str, Dict[str, Any]] = {}
    for m in models:
        runs = [agg[m] for agg in aggs_per_run]
        merged: Dict[str, Any] = {"model": m, "n_runs": len(runs)}

        for key in _SCALAR_METRICS + _RATE_METRICS:
            vals = [r.get(key, 0.0) for r in runs]
            merged[key + "_runs"] = vals
            merged[key + "_mean"] = statistics.fmean(vals) if vals else 0.0
            merged[key + "_std"]  = statistics.pstdev(vals) if len(vals) > 1 else 0.0

        for key in _SUMMARY_METRICS:
            vals = [
                r.get(key, {}).get("mean", 0.0) if isinstance(r.get(key), dict)
                else r.get(key, 0.0)
                for r in runs
            ]
            merged[key + "_runs"] = vals
            merged[key + "_mean"] = statistics.fmean(vals) if vals else 0.0
            merged[key + "_std"]  = statistics.pstdev(vals) if len(vals) > 1 else 0.0

        for key in _INT_METRICS:
            vals = [r.get(key, 0) for r in runs]
            merged[key + "_runs"]  = vals
            merged[key + "_mean"]  = statistics.fmean(vals) if vals else 0.0
            merged[key + "_total"] = sum(vals)

        out[m] = merged
    return out


def _render_markdown_multirun(args, models: List[str],
                              repeated_aggs: Dict[str, Dict[str, Any]],
                              runs: List[Dict[str, Any]]) -> str:
    """跨 N 次 run 的报告（repeat=1 也走这条路径，± std 自动收起）。"""
    is_repeated = args.repeat > 1

    md: List[str] = []
    title_tag = " · 多 run mean ± std" if is_repeated else ""
    md.append(f"# eval_poem 报告 · {' vs '.join(models)}{title_tag}")
    judge_str = " + ".join(args.scorer)
    repeat_tag = f" × **{args.repeat} runs**" if is_repeated else ""
    md.append(
        f"_n={args.n}（{args.genres or '全部体裁'} / {args.density or '全部密度'}）"
        f"{repeat_tag} · 评委={judge_str} · 候选数={args.candidates}_"
    )
    md.append("")
    if is_repeated:
        md.append(
            "**方法论：** 每 run 完整跑一次 generate → BWS → multi-judge → pairwise pipeline；"
            f"{args.repeat} 个独立 run 在 LLM temperature noise 下产生不同候选 / 不同评委判断。"
            "跨 run mean ± std 同时暴露：(1) 模型生成的固有方差 (2) 评委判断的固有方差。"
        )
        md.append("")
        md.append("> 单 run 的候选 dump / pairwise 详情见报告后半（基于 Run 1）。")
        md.append("")

    headers = ["指标"] + [f"`{m}`" for m in models]

    def _fmt_pm(mean: float, std: float, pct: bool = False, d: int = 3) -> str:
        if not is_repeated:
            return f"**{mean:.1%}**" if pct else f"**{mean:.{d}f}**"
        if pct:
            return f"**{mean:.1%}** ± {std:.1%}"
        return f"**{mean:.{d}f}** ± {std:.{d}f}"

    # ── §1 胜率
    md.append("## 1. 跨模型 pairwise 胜率（mean ± std across runs）")
    md.append("")
    rows = []
    rows.append(["胜率"] + [
        _fmt_pm(repeated_aggs[m]["win_rate_mean"],
                repeated_aggs[m]["win_rate_std"])
        for m in models
    ])
    rows.append(["每 run 胜率"] + [
        ", ".join(f"{v:.3f}" for v in repeated_aggs[m]["win_rate_runs"])
        for m in models
    ])
    rows.append(["累计 胜/平/负/全摇摆"] + [
        f"{repeated_aggs[m]['pairwise_wins_total']}/"
        f"{repeated_aggs[m]['pairwise_ties_total']}/"
        f"{repeated_aggs[m]['pairwise_losses_total']}/"
        f"{repeated_aggs[m]['pairwise_all_swing_total']}"
        for m in models
    ])
    rows.append(["评委有效胜票占比"] + [
        _fmt_pm(repeated_aggs[m]["judge_consistency_rate_mean"],
                repeated_aggs[m]["judge_consistency_rate_std"], pct=True)
        for m in models
    ])
    md.append(table(headers, rows))
    md.append("")

    # ── §2 候选分布
    md.append("## 2. 候选分布（mean ± std across runs）")
    md.append("")
    rows = []
    for human, key in (
        ("avg 候选本地总分",                            "mean_total"),
        ("std 候选内方差（越低越稳）",                  "std_total"),
        ("min 候选本地总分（worst-case）",              "min_total"),
        (f"pass@{POEM_QUALITY_THRESHOLD} 候选合格率",   "pass_rate_07"),
        ("候选多样性（低=mode collapse）",              "diversity"),
    ):
        rows.append([human] + [
            _fmt_pm(repeated_aggs[m][key + "_mean"],
                    repeated_aggs[m][key + "_std"])
            for m in models
        ])
    md.append(table(headers, rows))
    md.append("")

    # ── §3 best 4 维分
    md.append("## 3. best 4 维分（mean ± std across runs）")
    md.append("")
    rows = []
    for d in DIMS:
        rows.append([d] + [
            _fmt_pm(repeated_aggs[m][d + "_mean"],
                    repeated_aggs[m][d + "_std"])
            for m in models
        ])
    md.append(table(headers, rows))
    md.append("")

    # ── §4 格律合规
    md.append("## 4. 格律合规（mean ± std across runs）")
    md.append("")
    rows = []
    for human, key in (("平仄合格率 (≥0.8)", "pingze_pass@0.8"),
                       ("押韵合格率 (≥0.8)", "rhyme_pass@0.8")):
        rows.append([human] + [
            _fmt_pm(repeated_aggs[m][key + "_mean"],
                    repeated_aggs[m][key + "_std"], pct=True)
            for m in models
        ])
    md.append(table(headers, rows))
    md.append("")

    # ── §5 各 run 关键指标矩阵（debug 用，看哪 run 偏离）；repeat=1 时无意义
    if is_repeated:
        md.append("## 5. 每 run 关键指标")
        md.append("")
        rows = []
        for k in range(args.repeat):
            rows.append([f"Run {k+1}"] + [
                (f"胜率 {repeated_aggs[m]['win_rate_runs'][k]:.3f} · "
                 f"avg {repeated_aggs[m]['mean_total_runs'][k]:.3f} · "
                 f"pingze {repeated_aggs[m]['pingze_pass@0.8_runs'][k]:.0%}")
                for m in models
            ])
        md.append(table(headers, rows))
        md.append("")

    return "\n".join(md)


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="诗歌生成质量同台对比（BWS + 跨模型 pairwise）")
    p.add_argument("--models", nargs="+", default=None,
                   help="参赛模型列表，如 --models local_base local_lora qwen-plus")
    p.add_argument("--model-a", default=None, help="（兼容）2 路对比的模型 A")
    p.add_argument("--model-b", default=None, help="（兼容）2 路对比的模型 B")
    p.add_argument("--scorer", nargs="+", default=["qwen-plus"],
                   help="评委 adapter 列表，建议 ≥3 评委跨家族")
    p.add_argument("--n", type=int, default=20, help="user_input 条数")
    p.add_argument("--candidates", type=int, default=5, help="每条 user_input 每模型生成多少候选")
    p.add_argument("--genres", nargs="*", default=None)
    p.add_argument("--density", choices=["rich", "sparse"], default=None)
    p.add_argument("--samples", type=int, default=3, help="抽样诗作展示条数")
    p.add_argument("--repeat", type=int, default=1,
                   help="重复整 pipeline 跑 N 次，报跨 run mean ± std；"
                        ">1 时切换多 run 报告格式（Task #61）")
    args = p.parse_args()

    # 解析参赛模型
    if args.models:
        models = list(args.models)
    elif args.model_a and args.model_b:
        models = [args.model_a, args.model_b]
    else:
        models = ["local_lora", "qwen-plus"]
    assert len(models) >= 2, "至少给两个模型才能对比"
    assert args.repeat >= 1, "--repeat 至少为 1"

    inputs = get_benchmark(n=args.n, genres=args.genres, density=args.density)
    n_pairs = len(models) * (len(models) - 1) // 2
    # 调用预估（乘以 repeat）
    bws_calls = len(inputs) * len(models) * len(args.scorer)
    multi_judge_calls = len(inputs) * len(models)  # best 1 次 multi-judge ≈ 1 套 LLM 调用
    pairwise_calls = len(inputs) * n_pairs * len(args.scorer)
    per_run_est = bws_calls + multi_judge_calls * len(args.scorer) + pairwise_calls
    total_est = per_run_est * args.repeat
    print(f"[eval_poem] {len(inputs)} 条 × {args.repeat} run · {' vs '.join(models)} "
          f"· 评委={'+'.join(args.scorer)}")
    print(f"  候选={args.candidates} · 模型对={n_pairs} 对/题")
    print(f"  调用预估（单 run）：BWS={bws_calls} + best 4 维≈{multi_judge_calls * len(args.scorer)} "
          f"+ 跨模型 pairwise={pairwise_calls} ≈ {per_run_est}")
    if args.repeat > 1:
        print(f"  调用预估（{args.repeat} run 合计）：≈ {total_est}")

    judge_clash = set(args.scorer) & set(models)
    if judge_clash:
        print(f"  ⚠ 警告：评委 {judge_clash} 同时在参赛队伍里，存在 self-bias 风险")

    gen = PoemGenerator()
    judges = [(label, _make_adapter(label)) for label in args.scorer]

    # 跑 args.repeat 次完整 pipeline，跨 run 聚合（repeat=1 是合法特例）
    runs: List[Dict[str, Any]] = []
    for k in range(args.repeat):
        if args.repeat > 1:
            print(f"\n========== Run {k+1}/{args.repeat} ==========")
        aggs_k, rpm_k, mpi_k, io_k = _run_pipeline(
            args, models, judges, inputs, gen, run_idx=k,
        )
        runs.append({
            "aggs":               aggs_k,
            "results_per_model":  rpm_k,
            "matchups_per_input": mpi_k,
            "input_order":        io_k,
        })

    repeated_aggs = _aggregate_across_runs([r["aggs"] for r in runs], models)

    # 跨 run 摘要 + Run 1 详细报告（候选 dump / pairwise 等单 run 视角信息）
    summary_md = _render_markdown_multirun(args, models, repeated_aggs, runs)
    run1_md = _render_markdown(
        args, models, runs[0]["aggs"], runs[0]["results_per_model"],
        runs[0]["matchups_per_input"], runs[0]["input_order"],
    )
    detail_title = (
        f"# Run 1 详细报告（候选 dump / pairwise 详情）" if args.repeat > 1
        else "# 详细报告（候选 dump / pairwise 详情）"
    )
    md = summary_md + f"\n\n---\n\n{detail_title}\n\n" + run1_md

    payload = {
        "config":        vars(args),
        "models":        models,
        "repeated_aggs": repeated_aggs,
        "runs":          runs,
    }
    print_and_return(summary_md)  # 控制台只 print 跨 run 摘要
    paths = save_artifacts("eval_poem", payload, md)
    print(f"原始数据: {paths['json']}")
    print(f"Markdown: {paths['md']}")


if __name__ == "__main__":
    main()
