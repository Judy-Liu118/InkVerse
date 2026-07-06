"""eval._rerun_arena_randomized_layout -- B2-lite: A/B 位随机化下的无图擂台重跑

目的：量化 B3a（守擂 pairwise A/B 位随机化，2026-07-06 落地，见
core/agent/poem_refiner.py pairwise_judge_randomized）对生产擂台攻擂成功率的
before/after 影响。

设计要点（为什么"无图重跑"是干净的）：
  · autonomous_full_run 的擂台进化（第 2 步）发生在任何生图之前（第 4 步），
    该段 state.clip_score_final 恒为空 → critique 提示里的 CLIP 值恒 0.0。
    因此 plan → arena 海选 → 擂台进化 的无图重跑与历史全量 run 的攻擂段
    完全同构，不引入"缺图"混杂，且零图像配额消耗。
  · 历史基线（固定布局：擂主恒 A 位 / 挑战者恒 B 位）：
    outputs/eval/sweep_pairwise_win_delta_20260702_105404.json
    （arena ablation arm B 官方 run · delta=0.17 · poem_rounds=2 · 主池 22 题）
    sweep 口径攻擂 29/59 = 49.2%。
  · 本 run 配置与其完全一致（get_benchmark n=32 offset=10 → 主池 22 题、
    delta=0.17、poem_rounds=2、qwen-plus judge），唯一系统性差异 = 现行代码
    的守擂位随机化。B1b 已定向 judge 偏 A 位（7:0, p=0.0156）→ 预期方向：
    随机化后挑战者半数轮到 A 位顺风，攻擂率相对固定布局应上升。

已知不可比噪声（如实列注，不掩盖）：
  · 非配对重跑：擂主/挑战者诗全部重新生成（LLM 非确定），是两组独立主题
    样本的率比较，不是同对重判 —— 差值里混有生成抽样噪声。
  · 弃权语义（2026-07-05 引入）：历史 run 判定失败静默算 "A"（隐式守擂），
    现行代码显式作废挑战者。两者对"该轮不攻擂"同向，但分母构成略异。
  · sweep 口径 evo_rounds 把 门控拦截/弃权 等 trace 行也计入分母（历史 JSON
    只存该口径），本脚本同口径复算以对齐，另附"有效对决轮"清洁口径（历史
    侧无法回溯复算，标 N/A）。
  · 海选轮循 pairwise 仍是固定布局（本地分高者恒 A 位），不在 B3a 范围内，
    两期一致，不构成本对比的差异项。

位置随机化审计：monkey-patch core.agent.poem_refiner.pairwise_judge_randomized
加记录壳（注入专用 RNG seed=20260706，位置序列可审计），逐挑战者落
位置/胜负/弃权 → 顺带得到随机化布局下"挑战者在 A 位 vs B 位"的胜率对比
—— 生产分布新鲜对上对 B1b 方向（judge 偏 A 位）的直接内部复核
（B1b 用的是 eval 历史对，非生产挑战者分布）。

跑法：
    python -X utf8 -m eval._rerun_arena_randomized_layout --dry-run   # 零 API 拓扑验证
    python -X utf8 -m eval._rerun_arena_randomized_layout             # 真跑（主池 22 题）

成本：~400-450 次 qwen-plus/qwen-max 文本调用 + 本地 LoRA 生诗；零图像调用。
本脚本不修改任何生产代码与冻结契约；monkey-patch 仅在本进程内生效且注入的
是语义等价的记录壳（真实判定仍走原函数）。
"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import math
import os
import random
import time
import traceback
from typing import Any, Dict, List, Optional

import config
from eval.dataset import get_benchmark, BenchInput
from eval.report import save_artifacts, table, fmt_num, OUTPUT_DIR
from eval._analyze_arena_ablation_significance import BOOTSTRAP_ITERS, BOOTSTRAP_SEED

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

POSITION_RNG_SEED = 20260706   # 守擂位随机化专用 RNG；序列可审计复现（给定相同调用次序）
DELTA = 0.17                   # 与历史基线一致 = config.PAIRWISE_WIN_DELTA 生产默认
HIST_BASELINE_JSON = os.path.join(
    ROOT, "outputs", "eval", "sweep_pairwise_win_delta_20260702_105404.json")


# ── sweep 同口径计数（逐字复刻 eval/sweep_pairwise_win_delta.py，保证可比） ──

def _count_attack_succeeded(state) -> int:
    """从 trace 数 "攻擂成功" 次数（擂台进化里被采纳的轮次）。"""
    return sum(1 for s in state.trace if "攻擂成功" in s.action)


def _count_evo_rounds(state) -> int:
    """擂台进化的总轮次（含失败守擂）。⚠ sweep 口径：门控拦截/弃权 等含
    "第N轮" 的擂台 trace 行也计入，分母偏大；保留此口径仅为对齐历史 JSON。"""
    return sum(1 for s in state.trace if "擂台" in s.phase and "第" in s.action and "轮" in s.action)


def _count_decided_rounds(state) -> int:
    """清洁口径：真正走到 1v1 对决判定的轮次（攻擂成功 + 守擂成功）。"""
    return sum(1 for s in state.trace
               if "擂台" in s.phase and ("攻擂成功" in s.action or "守擂成功" in s.action))


# ── 统计工具 ────────────────────────────────────────────────────────────────

def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """2×2 表 [[a,b],[c,d]] 的 Fisher 精确检验（双侧，点概率法）。
    行 = 两臂，列 = 成功/失败。"""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    if n == 0 or r1 == 0 or r1 == n or c1 == 0 or c1 == n:
        return 1.0
    denom = math.comb(n, r1)

    def pmf(x: int) -> float:
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / denom

    p_obs = pmf(a)
    lo = max(0, r1 + c1 - n)
    hi = min(r1, c1)
    p = sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9))
    return min(1.0, p)


def bootstrap_rate_diff_ci(themes_new: List[tuple], themes_hist: List[tuple],
                           iters: int, seed: int) -> tuple:
    """两臂各自主题级重抽，返回 (rate_new − rate_hist) 百分点差的 95% CI。
    themes_* = [(attack, rounds), ...]，rounds 用 sweep 口径。"""
    rng = random.Random(seed)

    def _one(themes: List[tuple]) -> float:
        sample = [themes[rng.randrange(len(themes))] for _ in range(len(themes))]
        att = sum(a for a, _ in sample)
        rnd = sum(r for _, r in sample)
        return att / rnd * 100 if rnd else 0.0

    diffs = sorted(_one(themes_new) - _one(themes_hist) for _ in range(iters))
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


# ── 守擂位判定记录壳 ─────────────────────────────────────────────────────────

class JudgeRecorder:
    """monkey-patch pairwise_judge_randomized：真实判定走原函数，仅注入
    专用 seeded RNG + 逐次记录 (theme, position, chal_won)。"""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.records: List[Dict[str, Any]] = []
        self.current_theme: Optional[str] = None
        self._orig = None

    def install(self) -> None:
        import core.agent.poem_refiner as pr
        self._orig = pr.pairwise_judge_randomized
        rec = self

        @functools.wraps(self._orig)
        def _wrapper(scorer, champion, challenger, user_request, adapter,
                     rng: random.Random = random):
            won, pos = rec._orig(scorer, champion, challenger, user_request,
                                 adapter, rng=rec.rng)
            rec.records.append({
                "theme":     rec.current_theme,
                "position":  pos,
                "chal_won":  won,   # True / False / None(弃权)
                "champion":  champion,
                "challenger": challenger,
            })
            return won, pos

        pr.pairwise_judge_randomized = _wrapper

    def uninstall(self) -> None:
        if self._orig is not None:
            import core.agent.poem_refiner as pr
            pr.pairwise_judge_randomized = self._orig
            self._orig = None


# ── 主流程 ──────────────────────────────────────────────────────────────────

def _cuda_cleanup() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _dump_recovery(args, rows: List[Dict[str, Any]], judge_records: List[Dict]) -> None:
    path = OUTPUT_DIR / "rerun_arena_randomized_RECOVERY.json"
    try:
        path.write_text(
            json.dumps({"config": vars(args), "rows": rows,
                        "judge_records": judge_records,
                        "note": "partial recovery snapshot, overwritten per theme"},
                       ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"      ⚠ recovery dump 失败：{e}")


def _run_one_theme(agent, item: BenchInput, args, recorder: JudgeRecorder) -> Dict[str, Any]:
    from core.agent.state import Phase
    from eval.eval_autonomous import _new_state

    state = _new_state(item, args)
    recorder.current_theme = item.user_input
    t0 = time.time()
    try:
        state = agent._phase_plan(state)
        if state.phase == Phase.ERROR:
            return {"theme": item.user_input, "error": f"plan: {state.error}"}
        state = agent._phase_poem_arena(state)
        if state.phase == Phase.ERROR:
            return {"theme": item.user_input, "error": f"arena: {state.error}"}
        champion_v0 = state.poem

        # 与 autonomous_full_run 第 2 步一致的 refine_adapter 选择
        refine_adapter = agent.score_adapter or agent.prompt_adapter
        poem_versions = [{"version": 0, "poem": champion_v0, "action": "arena 海选冠军"}]
        prev_poem = champion_v0
        for s in agent._evolve_champion(state, refine_adapter=refine_adapter,
                                        evolution_rounds=args.max_poem_rounds):
            if s.poem and s.poem != prev_poem:
                act = s.trace[-1].action if s.trace else ""
                poem_versions.append({"version": len(poem_versions),
                                      "poem": s.poem, "action": act})
                prev_poem = s.poem

        attacks = _count_attack_succeeded(state)
        evo_rounds = _count_evo_rounds(state)
        decided = _count_decided_rounds(state)
        return {
            "theme":            item.user_input,
            "attack_succeed":   attacks,
            "evo_rounds":       evo_rounds,          # sweep 口径（对齐历史）
            "decided_rounds":   decided,             # 清洁口径
            "attack_rate":      (attacks / evo_rounds) if evo_rounds else 0.0,
            "elapsed_sec":      round(time.time() - t0, 2),
            "champion_v0":      champion_v0,
            "final_poem":       state.poem,
            "poem_evolution":   poem_versions,
            "evo_trace":        [{"phase": s.phase, "action": s.action,
                                  "result": (s.result or "")[:200]}
                                 for s in state.trace if "擂台" in s.phase],
        }
    except Exception as e:
        print(f"      ⚠ 异常 [{type(e).__name__}]：{str(e)[:200]}")
        traceback.print_exc()
        return {"theme": item.user_input,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "traceback": traceback.format_exc()[:2000]}
    finally:
        _cuda_cleanup()


def _load_hist_baseline() -> Dict[str, Any]:
    with open(HIST_BASELINE_JSON, encoding="utf-8") as f:
        data = json.load(f)
    res = data["results"][0]
    assert abs(res["delta"] - DELTA) < 1e-9, "历史基线 delta 不为 0.17"
    rows = [r for r in res["rows"] if "error" not in r]
    return {
        "path":   os.path.relpath(HIST_BASELINE_JSON, ROOT),
        "rows":   rows,
        "attack": sum(r["attack_succeed"] for r in rows),
        "rounds": sum(r["evo_rounds"] for r in rows),
        "themes": [(r["attack_succeed"], r["evo_rounds"]) for r in rows],
    }


# ── 报告 ────────────────────────────────────────────────────────────────────

def _position_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    judged = [r for r in records if r["chal_won"] is not None]
    abstain = len(records) - len(judged)
    at_a = [r for r in judged if r["position"] == "A"]
    at_b = [r for r in judged if r["position"] == "B"]
    win_a = sum(1 for r in at_a if r["chal_won"])
    win_b = sum(1 for r in at_b if r["chal_won"])
    p = fisher_exact_two_sided(win_a, len(at_a) - win_a, win_b, len(at_b) - win_b)
    return {"n_judged": len(judged), "n_abstain": abstain,
            "n_at_A": len(at_a), "win_at_A": win_a,
            "n_at_B": len(at_b), "win_at_B": win_b,
            "fisher_p_A_vs_B": p}


def _render_report(args, rows: List[Dict[str, Any]], hist: Dict[str, Any],
                   pos: Dict[str, Any], stats: Dict[str, Any]) -> str:
    md: List[str] = []
    md.append("# B2-lite · A/B 位随机化下的无图擂台重跑（攻擂率 before/after）")
    md.append(f"_主池 {stats['n_valid']} 题 · delta={DELTA} · poem_rounds={args.max_poem_rounds} · "
              f"judge={args.scorer} · 位随机化 RNG seed={POSITION_RNG_SEED} · 零图像调用_")
    md.append("")
    md.append("## 1. 主对比：固定布局（历史）vs 随机化布局（本 run）")
    md.append(table(
        ["臂", "布局", "攻擂成功/轮次（sweep 口径）", "攻擂率"],
        [
            ["历史 arm B（2026-07-02）", "擂主恒 A 位 / 挑战者恒 B 位",
             f"{hist['attack']}/{hist['rounds']}",
             f"{hist['attack'] / hist['rounds'] * 100:.1f}%"],
            ["本 run（B3a 后）", "守擂位随机化",
             f"{stats['attack']}/{stats['rounds']}",
             f"{stats['rate'] * 100:.1f}%" if stats['rounds'] else "—"],
        ],
    ))
    md.append("")
    md.append(f"- 攻擂率差（随机化 − 固定）：**{stats['rate_diff_pp']:+.1f}pp**")
    md.append(f"- Fisher 精确检验（双侧）：p = {stats['fisher_p']:.4f}")
    md.append(f"- 主题级 bootstrap 95% CI（两臂各自重抽 {BOOTSTRAP_ITERS} 次, "
              f"seed={BOOTSTRAP_SEED}）：差 ∈ [{stats['ci_lo']:+.1f}pp, {stats['ci_hi']:+.1f}pp]")
    md.append(f"- 本 run 清洁口径（仅计真正走到 1v1 判定的轮次）：{stats['attack']}/"
              f"{stats['decided']} = "
              f"{(stats['attack'] / stats['decided'] * 100) if stats['decided'] else 0:.1f}%"
              f"（历史 JSON 无法回溯此口径，仅列本 run）")
    md.append("")
    md.append("## 2. 位置审计：随机化下挑战者在 A 位 vs B 位")
    md.append("_对 B1b「judge 偏 A 位」方向的生产分布内部复核（新鲜对、选择独立于 judge）_")
    md.append("")
    rate_a = pos["win_at_A"] / pos["n_at_A"] * 100 if pos["n_at_A"] else 0.0
    rate_b = pos["win_at_B"] / pos["n_at_B"] * 100 if pos["n_at_B"] else 0.0
    md.append(table(
        ["挑战者位置", "判定次数", "挑战者胜", "胜率"],
        [
            ["A 位（顺风，若 B1b 方向成立）", pos["n_at_A"], pos["win_at_A"], f"{rate_a:.1f}%"],
            ["B 位（逆风）", pos["n_at_B"], pos["win_at_B"], f"{rate_b:.1f}%"],
        ],
    ))
    md.append("")
    md.append(f"- A 位 − B 位胜率差：{rate_a - rate_b:+.1f}pp · "
              f"Fisher 双侧 p = {pos['fisher_p_A_vs_B']:.4f}")
    md.append(f"- 评委弃权：{pos['n_abstain']} 次 / 共 {pos['n_judged'] + pos['n_abstain']} 次判定")
    md.append("")
    md.append("## 3. 逐题明细（本 run）")
    md.append(table(
        ["主题", "攻擂/轮次(sweep)", "对决轮(清洁)", "耗时(s)"],
        [[r["theme"], f"{r['attack_succeed']}/{r['evo_rounds']}",
          r["decided_rounds"], r["elapsed_sec"]]
         for r in rows if "error" not in r]
        + [[r["theme"], f"⚠ {r['error']}", "—", "—"] for r in rows if "error" in r],
    ))
    md.append("")
    md.append("## 4. 口径与不可比噪声（引用本报告时必读）")
    md.append("- **非配对重跑**：两臂的擂主/挑战者诗均为各自 run 现场生成（LLM 非确定），"
              "是独立样本率比较，差值含生成抽样噪声，不是同对重判的纯位置效应。")
    md.append("- **弃权语义 BREAKING（2026-07-05）**：历史 run 判定失败静默算 A（隐式守擂），"
              "本 run 显式作废挑战者；两者对攻擂率同向压低，但分母构成略异。")
    md.append("- **sweep 口径分母偏大**：evo_rounds 含门控拦截/弃权 trace 行；"
              "跨臂对比用同口径抵消，§1 末行附本 run 清洁口径。")
    md.append("- **海选轮循不在对比范围**：其 pairwise 两期都是固定布局（本地分高者恒 A 位），"
              "不构成差异项。")
    md.append(f"- 历史基线：`{hist['path']}`（arena ablation arm B 官方 run）。")
    return "\n".join(md)


def main() -> None:
    p = argparse.ArgumentParser(description="B2-lite: 随机化布局无图擂台重跑")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--offset", type=int, default=10,
                   help="与历史 arm B run 相同：n=32 offset=10 → 主池 22 题")
    p.add_argument("--poem-model",   default="local_lora")
    p.add_argument("--prompt-model", default="qwen-max")
    p.add_argument("--scorer",       default="qwen-plus")
    p.add_argument("--image-backend", default="bailian:qwen-image-2.0-pro",
                   help="仅用于构造 state（本脚本不进入生图段，不会实际调用）")
    p.add_argument("--max-poem-rounds", type=int, default=2)
    p.add_argument("--dry-run", action="store_true",
                   help="零 API：验证 patch 安装/还原、基线加载、统计与报告渲染")
    args = p.parse_args()

    hist = _load_hist_baseline()
    print(f"[B2-lite] 历史基线: {hist['path']} · 攻擂 {hist['attack']}/{hist['rounds']} "
          f"= {hist['attack'] / hist['rounds'] * 100:.1f}%（固定布局）")

    if args.dry_run:
        import core.agent.poem_refiner as pr
        orig_fn = pr.pairwise_judge_randomized
        rec = JudgeRecorder(POSITION_RNG_SEED)
        rec.install()
        assert pr.pairwise_judge_randomized is not orig_fn, "patch 未生效"
        rec.uninstall()
        assert pr.pairwise_judge_randomized is orig_fn, "patch 未还原"
        # 合成数据走完整统计 + 渲染路径
        fake_rows = [{"theme": f"t{i}", "attack_succeed": 1, "evo_rounds": 2,
                      "decided_rounds": 2, "attack_rate": 0.5, "elapsed_sec": 0.0}
                     for i in range(22)]
        fake_records = [{"theme": "t0", "position": ("A" if i % 2 else "B"),
                         "chal_won": (i % 3 == 0)} for i in range(88)]
        pos = _position_stats(fake_records)
        n_att = sum(r["attack_succeed"] for r in fake_rows)
        n_rnd = sum(r["evo_rounds"] for r in fake_rows)
        ci = bootstrap_rate_diff_ci([(r["attack_succeed"], r["evo_rounds"]) for r in fake_rows],
                                    hist["themes"], 1000, BOOTSTRAP_SEED)
        stats = {"n_valid": 22, "attack": n_att, "rounds": n_rnd, "decided": n_rnd,
                 "rate": n_att / n_rnd,
                 "rate_diff_pp": n_att / n_rnd * 100 - hist["attack"] / hist["rounds"] * 100,
                 "fisher_p": fisher_exact_two_sided(
                     n_att, n_rnd - n_att, hist["attack"], hist["rounds"] - hist["attack"]),
                 "ci_lo": ci[0], "ci_hi": ci[1]}
        md = _render_report(args, fake_rows, hist, pos, stats)
        print(md[:1500])
        # fisher 自检：对称表 p=1；极端表 p 小
        assert fisher_exact_two_sided(10, 10, 10, 10) > 0.99
        assert fisher_exact_two_sided(19, 1, 1, 19) < 1e-4
        print("\n[B2-lite] dry-run OK：patch 安装/还原、基线加载、Fisher 自检、报告渲染均通过")
        return

    from eval.eval_autonomous import _build_agent
    inputs = get_benchmark(n=args.n)
    if args.offset > 0:
        inputs = inputs[args.offset:]
    print(f"[B2-lite] 实跑 {len(inputs)} 题（原始编号 {args.offset + 1}..{args.n}） · "
          f"delta={DELTA}（生产默认，不 patch） · 布局=随机化（现行代码）")
    assert abs(config.PAIRWISE_WIN_DELTA - DELTA) < 1e-9, \
        "config.PAIRWISE_WIN_DELTA != 0.17，与历史基线不可比，中止"
    hist_themes = {r["theme"] for r in hist["rows"]}
    new_themes = {it.user_input for it in inputs}
    if hist_themes != new_themes:
        print(f"⚠ 主题集与历史基线不一致：仅历史有 {sorted(hist_themes - new_themes)} · "
              f"仅本 run 有 {sorted(new_themes - hist_themes)}")

    agent = _build_agent(args)
    rec = JudgeRecorder(POSITION_RNG_SEED)
    rec.install()
    rows: List[Dict[str, Any]] = []
    t_all = time.time()
    try:
        for i, item in enumerate(inputs):
            print(f"  [{i + 1}/{len(inputs)}] {item.user_input[:30]}…")
            r = _run_one_theme(agent, item, args, rec)
            rows.append(r)
            if "error" not in r:
                print(f"      攻擂 {r['attack_succeed']}/{r['evo_rounds']}"
                      f"（对决轮 {r['decided_rounds']}） {r['elapsed_sec']}s")
            _dump_recovery(args, rows, rec.records)
    finally:
        rec.uninstall()

    valid = [r for r in rows if "error" not in r]
    n_att = sum(r["attack_succeed"] for r in valid)
    n_rnd = sum(r["evo_rounds"] for r in valid)
    n_dec = sum(r["decided_rounds"] for r in valid)
    ci = bootstrap_rate_diff_ci([(r["attack_succeed"], r["evo_rounds"]) for r in valid],
                                hist["themes"], BOOTSTRAP_ITERS, BOOTSTRAP_SEED)
    stats = {
        "n_valid": len(valid), "attack": n_att, "rounds": n_rnd, "decided": n_dec,
        "rate": (n_att / n_rnd) if n_rnd else 0.0,
        "rate_diff_pp": (n_att / n_rnd * 100 - hist["attack"] / hist["rounds"] * 100) if n_rnd else 0.0,
        "fisher_p": fisher_exact_two_sided(
            n_att, n_rnd - n_att, hist["attack"], hist["rounds"] - hist["attack"]),
        "ci_lo": ci[0], "ci_hi": ci[1],
    }
    pos = _position_stats(rec.records)

    md = _render_report(args, rows, hist, pos, stats)
    summary = {
        "config": vars(args),
        "delta": DELTA,
        "position_rng_seed": POSITION_RNG_SEED,
        "hist_baseline": {"path": hist["path"], "attack": hist["attack"],
                          "rounds": hist["rounds"]},
        "stats": stats,
        "position_audit": pos,
        "judge_records": rec.records,
        "rows": rows,
        "total_elapsed_sec": round(time.time() - t_all, 1),
    }
    paths = save_artifacts("rerun_arena_randomized_main_n22", summary, md)
    print(f"\n[B2-lite] 总耗时 {(time.time() - t_all) / 60:.1f} min")
    print(f"原始数据: {paths['json']}")
    print(f"Markdown: {paths['md']}")
    print()
    print(md)


if __name__ == "__main__":
    main()
