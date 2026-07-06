"""生产擂台 pairwise judge 的无偏 position bias 探针（B1b · 新鲜对局版）。

背景：B1 探针（`_probe_pairwise_position_bias.py`）受**选择-复现混杂**制约——
样本全部来自"当年 forward 布局下 judge 说了 B"的攻擂成功对局，temp=0.1 重判
高度复现原判，机制性压低 fwd=A 的两类，B_follow:A_follow 的不对称不能按面值
采信（详见该脚本 docstring 的 2026-07-06 复盘声明）。

B1b 修正设计：改用**与 judge 选择无关的新鲜配对**——两首诗来自不同 run 的
同题终诗，历史上从未被同一次 pairwise 共同评过，入选条件与任何 judge 判定
无关，因此不存在复现混杂，B_follow vs A_follow 的不对称可直接做无偏检验。

配对来源（同 benchmark 同题对齐）：
  · armA_vs_armB_n22：擂台消融 armA（无擂台）vs armB（有擂台）终诗，22 对
  · delta0.14_vs_0.20_n10：sweep 两档 delta 的终诗，10 对
合计 32 对 × 双向 = 64 次 qwen-plus 调用。不消耗任何图像配额。

判定与分类（X=左源终诗，Y=右源终诗）：
  · forward：X 在 A 位、Y 在 B 位；reverse：交换。
  · content_X（fwd=A, rev=B）/ content_Y（fwd=B, rev=A）→ 两向同人，内容驱动
  · position_A_follow（fwd=A, rev=A）→ 永远选 A 位
  · position_B_follow（fwd=B, rev=B）→ 永远选 B 位
  · abstain → 任一向弃权（compare_poems 返 None）
主检验：position-follow 案例内 B_follow vs A_follow 的双侧精确二项（H0 p=0.5），
等价于 McNemar 不一致对检验；本设计下 p 值可按面值采信。

注意：X/Y 的内容差异（如 armB 经过擂台进化）只影响 content_* 的分布，
不影响 position_* 两类的对称性——位置检验对内容不平衡不敏感。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DASHSCOPE_API_KEY  # noqa: E402
from core.models.adapter import ModelAdapter  # noqa: E402
from core.poem.scorer import PoemScorer  # noqa: E402
from eval.report import save_artifacts  # noqa: E402

# (标签, X 源 JSON, Y 源 JSON, X 名, Y 名) —— rows 按同一 benchmark 顺序同题对齐
PAIR_SPECS = [
    ("armA_vs_armB_n22",
     "outputs/eval/sweep_pairwise_win_delta_20260702_094052.json",
     "outputs/eval/sweep_pairwise_win_delta_20260702_105404.json",
     "armA", "armB"),
    ("delta0.14_vs_0.20_n10",
     "outputs/eval/sweep_pairwise_win_delta_20260630_212041.json",
     "outputs/eval/sweep_pairwise_win_delta_20260630_204647.json",
     "delta0.14", "delta0.20"),
]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_rows(path: str) -> list:
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for result in data["results"]:
        rows.extend(result["rows"])
    return rows


def build_pairs() -> list:
    pairs = []
    for label, path_x, path_y, name_x, name_y in PAIR_SPECS:
        rows_x, rows_y = load_rows(path_x), load_rows(path_y)
        assert [r["theme"] for r in rows_x] == [r["theme"] for r in rows_y], \
            f"{label}: 两源主题不对齐"
        for rx, ry in zip(rows_x, rows_y):
            if rx["poem"] == ry["poem"]:
                continue  # 同诗对局无判别意义
            pairs.append({
                "source": label, "theme": rx["theme"],
                "x_name": name_x, "y_name": name_y,
                "poem_x": rx["poem"], "poem_y": ry["poem"],
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
    if fwd == "A" and rev == "B":
        return "content_X"
    if fwd == "B" and rev == "A":
        return "content_Y"
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

    pairs = build_pairs()
    for label, *_ in PAIR_SPECS:
        print(f"{label}: {sum(1 for p in pairs if p['source'] == label)} 对")
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"合计 {len(pairs)} 对 → {len(pairs) * 2} 次 {args.judge_model} 调用\n")

    adapter = ModelAdapter(backend="qwen", api_key=DASHSCOPE_API_KEY,
                           api_model=args.judge_model)
    scorer = PoemScorer()

    t0 = time.time()
    for i, p in enumerate(pairs):
        fwd = scorer.compare_poems(p["poem_x"], p["poem_y"], p["theme"], adapter)
        rev = scorer.compare_poems(p["poem_y"], p["poem_x"], p["theme"], adapter)
        p["forward"] = fwd          # X=A / Y=B
        p["reverse"] = rev          # Y=A / X=B
        p["category"] = classify(fwd, rev)
        print(f"[{i + 1}/{len(pairs)}] {p['source']} · {p['theme'][:18]}… "
              f"fwd={fwd} rev={rev} → {p['category']}")

    cats = ["content_X", "content_Y",
            "position_A_follow", "position_B_follow", "abstain"]
    counts = {c: sum(1 for p in pairs if p["category"] == c) for c in cats}
    n_valid = len(pairs) - counts["abstain"]

    pf_a, pf_b = counts["position_A_follow"], counts["position_B_follow"]
    p_pos = binom_two_sided_p(pf_b, pf_a + pf_b)
    swing_rate = (pf_a + pf_b) / n_valid if n_valid else None

    # 描述性：全部 64 次判定里 A 位占位者的胜率（同对两判非独立，只作描述）
    a_wins = sum(1 for p in pairs if p["forward"] == "A") \
        + sum(1 for p in pairs if p["reverse"] == "A")
    decided = sum(1 for p in pairs for v in (p["forward"], p["reverse"])
                  if v is not None)

    # 内容侧顺带产出：armB（擂台进化终诗）vs armA 的两向一致胜负
    arm_pairs = [p for p in pairs if p["source"] == "armA_vs_armB_n22"]
    armb_content_wins = sum(1 for p in arm_pairs if p["category"] == "content_Y")
    arma_content_wins = sum(1 for p in arm_pairs if p["category"] == "content_X")

    summary = {
        "design": "B1b 新鲜配对（入选与 judge 判定无关，无选择-复现混杂）",
        "judge_model": args.judge_model,
        "n_pairs": len(pairs),
        "counts": counts,
        "n_valid": n_valid,
        "swing_rate": round(swing_rate, 3) if swing_rate is not None else None,
        "position_follow_B_vs_A": [pf_b, pf_a],
        "binom_p_position": round(p_pos, 4),
        "A_position_win_share_descriptive":
            round(a_wins / decided, 3) if decided else None,
        "armB_vs_armA_content_wins": [armb_content_wins, arma_content_wins],
        "elapsed_sec": round(time.time() - t0, 1),
    }

    md = ["# 生产擂台 pairwise position bias 无偏探针（B1b · 新鲜对局）", ""]
    md.append(f"judge={args.judge_model} · {len(pairs)} 对 × 双向 · "
              f"耗时 {summary['elapsed_sec']}s")
    md.append("")
    md.append("| 类别 | 数量 | 含义 |")
    md.append("|---|---|---|")
    md.append(f"| content_X | {counts['content_X']} | 两向都选 X（内容驱动）|")
    md.append(f"| content_Y | {counts['content_Y']} | 两向都选 Y（内容驱动）|")
    md.append(f"| position_A_follow | {counts['position_A_follow']} | 永远选 A 位 |")
    md.append(f"| position_B_follow | {counts['position_B_follow']} | 永远选 B 位 |")
    md.append(f"| abstain | {counts['abstain']} | 任一向弃权 |")
    md.append("")
    md.append(f"- 摇摆率（位置追随占比）: {summary['swing_rate']}")
    md.append(f"- **位置方向检验** B_follow {pf_b} vs A_follow {pf_a}: "
              f"双侧精确二项 p = {p_pos:.4f}（本设计下可按面值采信）")
    md.append(f"- A 位占位者总胜率（描述性，64 判定）: "
              f"{summary['A_position_win_share_descriptive']}")
    md.append(f"- 顺带（内容侧）armB vs armA 两向一致胜负: "
              f"{armb_content_wins} : {arma_content_wins}")
    paths = save_artifacts("probe_position_bias_fresh",
                           {"summary": summary, "pairs": pairs}, "\n".join(md))
    print("\n" + "\n".join(md))
    print(f"\n原始数据: {paths['json']}\nMarkdown: {paths['md']}")


if __name__ == "__main__":
    main()
