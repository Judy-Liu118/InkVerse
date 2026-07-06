"""生产擂台 pairwise judge 的 position bias 对照探针（B1）。

背景：生产擂台（core/agent/poem_refiner.py `_try_challenger`）是单向单次
pairwise——擂主恒在 A 位、挑战者恒在 B 位；而 eval 侧（METHODOLOGY §7）已
证明评委存在 position bias（摇摆率 23%）并用 forward+reverse 防御。攻擂率
49-75% 超出健康区间（sweep 报告 §6.3），"judge 偏 B 位"是未被排除的假设。

方法：从已有 sweep JSON 的 `poem_evolution` 重建历史（擂主, 攻擂成功挑战者）
对局，用生产同款 judge（qwen-plus）对每对做双向重判：
  · forward：擂主=A、挑战者=B（复刻生产布局）
  · reverse：挑战者=A、擂主=B（交换位置）
分类：
  · content_challenger：两向都选挑战者（fwd=B, rev=A）→ 内容驱动，原判可信
  · content_champion  ：两向都选擂主（fwd=A, rev=B）→ 内容上擂主更好，
                        原"攻擂成功"可能是位置/噪声驱动
  · position_A_follow ：fwd=A, rev=A → 永远选 A 位
  · position_B_follow ：fwd=B, rev=B → 永远选 B 位（生产布局下会推高攻擂率）
  · abstain           ：任一向弃权（compare_poems 返 None）

⚠ 选择-复现混杂声明（2026-07-06 复盘补强，比初版"选择偏置"更根本）：
JSON 只存了**攻擂成功**的对局——入选条件 = "当年 forward 布局下 judge 说了 B"。
temp=0.1 下重判 forward 高度复现原判（首跑 46/53 仍=B），因此：
  · fwd=A 的两类（content_champion / position_A_follow）被机制性压低，
    B_follow : A_follow 的不对称**混杂了选择-复现伪效应**——零位置偏置下
    该设计也会产出同形状的不对称，McNemar p 值不能按面值采信；
  · "B 位胜率 vs A 位胜率"之差同理（前者含复现成分，后者是新鲜判定）。
本探针的产出只能定位为"位置效应迹象"。无偏确认需改用与 judge 选择无关的
新鲜对局（B1b：如 armA vs armB 终诗配对，双向判定），或 B2 全新擂台重跑。

【结局】B1b 已执行（2026-07-06，`_probe_position_bias_fresh_pairs.py`）：
新鲜配对 32 对上 position_A_follow 7 : position_B_follow 0（p=0.0156），
真实位置偏置**偏 A 位**——本探针表面的"偏 B 位"确系选择-复现伪效应。

API 消耗：len(pairs) × 2 次 qwen-plus chat（本仓库三份 JSON 共 53 对 → 106 次）。
不消耗任何图像配额。
"""
from __future__ import annotations

import argparse
import math
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DASHSCOPE_API_KEY  # noqa: E402
from core.models.adapter import ModelAdapter  # noqa: E402
from core.poem.scorer import PoemScorer  # noqa: E402
from eval.report import save_artifacts  # noqa: E402

# (标签, sweep JSON 路径) —— 只含 poem_evolution 字段非空的 run
DEFAULT_SOURCES = [
    ("delta0.14_n10_0630", "outputs/eval/sweep_pairwise_win_delta_20260630_212041.json"),
    ("delta0.20_n10_0630", "outputs/eval/sweep_pairwise_win_delta_20260630_204647.json"),
    ("armB_n22_0702",      "outputs/eval/sweep_pairwise_win_delta_20260702_105404.json"),
]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def extract_pairs(label: str, path: str) -> list:
    """从 poem_evolution 抽 (擂主 v_i, 胜出挑战者 v_{i+1}) 对。"""
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for result in data["results"]:
        for row in result.get("rows", []):
            evo = sorted(row.get("poem_evolution") or [],
                         key=lambda e: e.get("version", 0))
            for i in range(len(evo) - 1):
                pairs.append({
                    "source": label,
                    "theme": row.get("theme", ""),
                    "round": evo[i + 1].get("action", ""),
                    "champion": evo[i]["poem"],
                    "challenger": evo[i + 1]["poem"],
                })
    return pairs


def binom_two_sided_p(k: int, n: int) -> float:
    """H0: p=0.5 的双侧精确二项检验。"""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def classify(fwd: str | None, rev: str | None) -> str:
    if fwd is None or rev is None:
        return "abstain"
    if fwd == "B" and rev == "A":
        return "content_challenger"
    if fwd == "A" and rev == "B":
        return "content_champion"
    if fwd == "A" and rev == "A":
        return "position_A_follow"
    return "position_B_follow"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge-model", default="qwen-plus",
                    help="judge 模型（默认 qwen-plus，与生产擂台一致）")
    ap.add_argument("--limit", type=int, default=None,
                    help="只跑前 N 对（调试用）")
    args = ap.parse_args()

    if not DASHSCOPE_API_KEY:
        sys.exit("缺 DASHSCOPE_API_KEY")

    pairs = []
    for label, path in DEFAULT_SOURCES:
        got = extract_pairs(label, path)
        print(f"{label}: {len(got)} 对")
        pairs.extend(got)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"合计 {len(pairs)} 对 → {len(pairs) * 2} 次 {args.judge_model} 调用\n")

    adapter = ModelAdapter(backend="qwen", api_key=DASHSCOPE_API_KEY,
                           api_model=args.judge_model)
    scorer = PoemScorer()

    t0 = time.time()
    for i, p in enumerate(pairs):
        fwd = scorer.compare_poems(p["champion"], p["challenger"], p["theme"], adapter)
        rev = scorer.compare_poems(p["challenger"], p["champion"], p["theme"], adapter)
        p["forward"] = fwd          # 擂主=A（生产布局）
        p["reverse"] = rev          # 挑战者=A
        p["category"] = classify(fwd, rev)
        print(f"[{i + 1}/{len(pairs)}] {p['source']} · {p['theme'][:18]}… "
              f"fwd={fwd} rev={rev} → {p['category']}")

    cats = ["content_challenger", "content_champion",
            "position_A_follow", "position_B_follow", "abstain"]
    counts = {c: sum(1 for p in pairs if p["category"] == c) for c in cats}
    n_valid = len(pairs) - counts["abstain"]

    # 位置追随方向检验：position-follow 案例里 B-follow vs A-follow
    pf_a, pf_b = counts["position_A_follow"], counts["position_B_follow"]
    p_pos_dir = binom_two_sided_p(pf_b, pf_a + pf_b)

    # 同一对局集上：挑战者在 B 位 vs A 位的胜率（配对，位置效应直读）
    chal_win_at_b = sum(1 for p in pairs if p["forward"] == "B")   # 生产布局
    chal_win_at_a = sum(1 for p in pairs if p["reverse"] == "A")   # 交换后
    decided_f = sum(1 for p in pairs if p["forward"] is not None)
    decided_r = sum(1 for p in pairs if p["reverse"] is not None)
    # McNemar：只在 B 位赢 vs 只在 A 位赢
    only_b = counts["position_B_follow"]          # fwd=B rev=B → 只在"自己是B位"那次…
    # 注意：challenger 视角下 fwd=B 是"B位赢"、rev=A 是"A位赢"。
    # 只 B 位赢 = fwd=B 且 rev=B(即A位输) = position_B_follow
    # 只 A 位赢 = fwd=A 且 rev=A(即A位赢、B位输) = position_A_follow… 反直觉但对：
    #   position_A_follow: fwd=A(挑战者B位输) rev=A(挑战者A位赢)
    only_a = counts["position_A_follow"]
    p_mcnemar = binom_two_sided_p(only_b, only_a + only_b)

    summary = {
        "judge_model": args.judge_model,
        "n_pairs": len(pairs),
        "counts": counts,
        "n_valid": n_valid,
        "challenger_win_rate_at_B(prod_layout)":
            round(chal_win_at_b / decided_f, 3) if decided_f else None,
        "challenger_win_rate_at_A(swapped)":
            round(chal_win_at_a / decided_r, 3) if decided_r else None,
        "position_effect_pp": round(
            (chal_win_at_b / decided_f - chal_win_at_a / decided_r) * 100, 1)
            if decided_f and decided_r else None,
        "mcnemar_p_positional": round(p_mcnemar, 4),
        "binom_p_follow_direction": round(p_pos_dir, 4),
        "elapsed_sec": round(time.time() - t0, 1),
        "selection_bias_caveat":
            "样本仅含历史攻擂成功对局（落败挑战者未落盘），非无偏估计",
    }

    md = ["# 生产擂台 pairwise position bias 探针（B1）", ""]
    md.append(f"judge={args.judge_model} · {len(pairs)} 对 × 双向 · "
              f"耗时 {summary['elapsed_sec']}s")
    md.append("")
    md.append("| 类别 | 数量 | 含义 |")
    md.append("|---|---|---|")
    md.append(f"| content_challenger | {counts['content_challenger']} | 两向都选挑战者（原判内容可信）|")
    md.append(f"| content_champion | {counts['content_champion']} | 两向都选擂主（原'攻擂成功'存疑）|")
    md.append(f"| position_A_follow | {counts['position_A_follow']} | 永远选 A 位 |")
    md.append(f"| position_B_follow | {counts['position_B_follow']} | 永远选 B 位（生产布局下推高攻擂率）|")
    md.append(f"| abstain | {counts['abstain']} | 任一向弃权 |")
    md.append("")
    md.append(f"- 挑战者 **B 位（生产布局）胜率 {summary['challenger_win_rate_at_B(prod_layout)']}** vs "
              f"**A 位（交换后）胜率 {summary['challenger_win_rate_at_A(swapped)']}** · "
              f"位置效应 {summary['position_effect_pp']} pp")
    md.append(f"- McNemar（只 B 位赢 {only_b} vs 只 A 位赢 {only_a}）: p = {p_mcnemar:.4f}")
    md.append("")
    md.append("> ⚠ 选择偏置：样本仅含历史攻擂成功对局，无偏估计需 B2（无图擂台重跑 + 位置随机化）。")

    paths = save_artifacts("probe_position_bias", {"summary": summary, "pairs": pairs},
                           "\n".join(md))
    print("\n" + "\n".join(md))
    print(f"\n原始数据: {paths['json']}\nMarkdown: {paths['md']}")


if __name__ == "__main__":
    main()
