"""对 arena ablation 主池 (n=22) 的 VLM 硬约束命中率差做配对显著性分析。

输入：outputs/eval/vlm_hard_constraint_arm_ab_main_n22_*.json（已有数据，零 API 消耗）
输出（stdout，markdown 片段，供 REPORT_arena_ablation_20260701.md §2.5 引用）：
  · keyword 级配对列联表（concordant / A-only / B-only）
  · McNemar 精确检验（二项精确，双侧）——判 +18.2pp 是否过显著阈值
  · 主题级 bootstrap 95% CI（重抽 theme 而非 keyword，避免同题 keyword 相关性
    低估方差）

方法学定位：这是对已冻结报告的"加法"补充——不重跑、不改指标定义，
只给已有差值配上不确定性量化。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260705  # 固定 seed：本脚本输出需可精确复现（METHODOLOGY §9 复现性边界）


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pair_keywords(theme_entry: dict) -> list:
    """返回 [(keyword, present_A, present_B), ...]；两 arm 按 keyword 文本对齐。"""
    res_a = {r["keyword"]: bool(r["present"]) for r in theme_entry["arm_A"]["results"]}
    res_b = {r["keyword"]: bool(r["present"]) for r in theme_entry["arm_B"]["results"]}
    if set(res_a) != set(res_b):
        raise ValueError(f"两 arm keyword 集不一致: {sorted(res_a)} vs {sorted(res_b)}")
    return [(kw, res_a[kw], res_b[kw]) for kw in res_a]


def mcnemar_exact_p(b: int, c: int) -> float:
    """McNemar 精确检验（双侧）：discordant 对里 A-only=b、B-only=c，
    H0 下 b ~ Binomial(b+c, 0.5)。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci_theme_level(themes: list, iters: int, seed: int) -> tuple:
    """重抽 theme（保留题内 keyword 相关结构），返回 Δpp 的 95% percentile CI。"""
    rng = random.Random(seed)
    deltas = []
    for _ in range(iters):
        sample = [themes[rng.randrange(len(themes))] for _ in range(len(themes))]
        hit_a = sum(a for _, pairs in sample for _, a, _ in pairs)
        hit_b = sum(b for _, pairs in sample for _, _, b in pairs)
        total = sum(len(pairs) for _, pairs in sample)
        deltas.append((hit_b - hit_a) / total * 100)
    deltas.sort()
    lo = deltas[int(0.025 * iters)]
    hi = deltas[int(0.975 * iters)]
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json", default=None,
        help="vlm_hard_constraint_arm_ab_main_n22_*.json 路径（默认取最新一份）",
    )
    args = ap.parse_args()

    path = args.json
    if path is None:
        cands = sorted(glob.glob(os.path.join(
            ROOT, "outputs", "eval", "vlm_hard_constraint_arm_ab_main_n22_*.json")))
        if not cands:
            sys.exit("未找到 vlm_hard_constraint_arm_ab_main_n22_*.json")
        path = cands[-1]

    data = _load(path)
    themes = [(name, _pair_keywords(entry)) for name, entry in data["by_theme"].items()]

    n_kw = sum(len(pairs) for _, pairs in themes)
    hit_a = sum(a for _, pairs in themes for _, a, _ in pairs)
    hit_b = sum(b for _, pairs in themes for _, _, b in pairs)
    both = sum(1 for _, pairs in themes for _, a, b in pairs if a and b)
    neither = sum(1 for _, pairs in themes for _, a, b in pairs if not a and not b)
    a_only = sum(1 for _, pairs in themes for _, a, b in pairs if a and not b)
    b_only = sum(1 for _, pairs in themes for _, a, b in pairs if b and not a)

    p = mcnemar_exact_p(a_only, b_only)
    ci_lo, ci_hi = bootstrap_ci_theme_level(themes, BOOTSTRAP_ITERS, BOOTSTRAP_SEED)
    delta_pp = (hit_b - hit_a) / n_kw * 100

    print(f"数据源: {os.path.relpath(path, ROOT)}")
    print(f"主题数: {len(themes)} · keyword 总数: {n_kw}")
    print()
    print("| 配对格 | 数量 |")
    print("|---|---|")
    print(f"| 双 hit (concordant) | {both} |")
    print(f"| 双 miss (concordant) | {neither} |")
    print(f"| 仅 arm A hit (discordant) | {a_only} |")
    print(f"| 仅 arm B hit (discordant) | {b_only} |")
    print()
    print(f"arm A 命中 {hit_a}/{n_kw} = {hit_a / n_kw * 100:.1f}% · "
          f"arm B 命中 {hit_b}/{n_kw} = {hit_b / n_kw * 100:.1f}% · "
          f"Δ = +{delta_pp:.1f}pp")
    print(f"McNemar 精确检验（双侧, discordant {a_only} vs {b_only}）: p = {p:.4f}")
    print(f"主题级 bootstrap 95% CI（{BOOTSTRAP_ITERS} 次重抽, seed={BOOTSTRAP_SEED}）: "
          f"Δ ∈ [{ci_lo:+.1f}pp, {ci_hi:+.1f}pp]")


if __name__ == "__main__":
    main()
