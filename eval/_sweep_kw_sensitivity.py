"""HARD_CONSTRAINTS kw 定义严/松敏感度全池扫描（A2）。

背景：擂台消融主结论 Δ+18.2pp（REPORT_arena_ablation_20260701.md）依赖
HARD_CONSTRAINTS 的 kw 定义，而定义存在两类严格度偏差（该报告 §8）：
"括号收窄型"（炉火（火焰））与"括号扩容型"（钟鼓（钟或鼓）），以及 sparse
场景 OR 集合的"主体 vs 边缘元素"主观取舍。§5.1 已用 theme 6 + theme 12
两个点定性演示（双点松判 → Δ 从 +18.2pp 变 +12.1pp），本脚本把它做成
**全池系统扫描**：给主结论一个 kw 定义无关的稳健区间。

设计：
  · 对 22 主池主题手写两套完整变体 kw（与冻结版逐条 1:1 对应）：
      - strict：物体须本体明确可辨 / 场景须为画面主体（判定下界）
      - loose ：写意暗示、边缘元素、同类替代均算命中（判定上界）
  · 两套变体 × 44 张主池图（22 主题 × arm_A/arm_B）= 88 次 qwen-vl-max。
  · 冻结版（baseline）不重跑，直接读 2026-07-02 已有 JSON。
  · 每档输出 armA/armB 命中率、Δpp、McNemar 精确 p、主题级 bootstrap CI
    （复用 `_analyze_arena_ablation_significance.py` 的函数与 seed）。

⚠ 冻结契约：本脚本**不修改** `eval/vlm_hard_constraint.py` 的 HARD_CONSTRAINTS
（v1 定义仍是唯一正式口径），strict/loose 变体只存在于本探针内，用于
敏感度包络，不用于任何正式命中率口径。

噪声辨析：strict 只收紧判定，逻辑上单 kw 命中不应比 baseline 新增
（loose 反之不应减少）。违反单调性的翻转 = VLM 重判噪声的直接观测，
脚本单独计数上报，用于区分"定义效应"与"重判噪声"。

API 消耗：88 次 qwen-vl-max（纯文本+已有图像输入），零图像生成配额。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.vlm_hard_constraint import (  # noqa: E402
    HARD_CONSTRAINTS, _build_client, _check_one_image, _fmt_hit,
    _fmt_verdict_cell, _load_agg, _load_image, _resolve_image_path,
)
from eval._analyze_arena_ablation_significance import (  # noqa: E402
    BOOTSTRAP_ITERS, BOOTSTRAP_SEED, bootstrap_ci_theme_level, mcnemar_exact_p,
)
from eval.vlm_judge import _load_prompt_yaml  # noqa: E402
from eval.report import save_artifacts, table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARMS = ["arm_A", "arm_B"]
DEFAULT_AGG = "outputs/eval/_agg_arena_ablation_main_n22.json"
DEFAULT_BASELINE = "outputs/eval/vlm_hard_constraint_arm_ab_main_n22_20260702_133542.json"

# ── 严/松变体（22 主池主题，与冻结版 kw 逐条 1:1 对应）────────────────────
# strict = 判定下界：本体明确可辨 / 场景为画面主体
# loose  = 判定上界：写意暗示、边缘元素、同类替代均算
KW_VARIANTS: Dict[str, Dict[str, List[str]]] = {
    "strict": {
        "写一首秋景的五言律诗，要有疏桐和寒蛩": ["桐树（须为可辨认的梧桐树本体）", "蟋蟀（须见可辨认虫体）"],
        "写一首七言律诗，主题是悲秋": ["悲秋场景（落叶或凋零萧瑟秋景须为画面主体）"],
        "写一首五言绝句，主题是寒梅": ["梅花（花朵清晰可辨的梅花本体）"],
        "写一首冬景的七言绝句，要有飞雪和寒鸦": ["雪（明确的降雪或积雪）", "乌鸦（可辨认的鸦鸟本体）"],
        "写一首五言律诗，主题是雪夜": ["雪夜场景（雪与夜色须同时明确呈现）"],
        "写一首冬景的七言律诗，要有炉火和寒灯": ["炉火（须见明火火焰）", "寒灯（灯具本体清晰可见）"],
        "写一首五言绝句，主题是溪声": ["溪流（明确的流水形态）"],
        "写一首山水的七言律诗，要有高楼和远山": ["高楼（多层楼阁建筑本体明确）", "远山（远景山峦轮廓明确）"],
        "写一首田园的七言绝句，要有耕牛和炊烟": ["耕牛（牛本体明确可辨且具劳作语境）", "炊烟（自屋舍升起的烟柱明确可见）"],
        "写一首五言律诗，主题是归隐": ["归隐场景（茅屋或隐士人物须为画面主体）"],
        "写一首边塞的五言律诗，要有戍楼和角声": ["戍楼（明确的军事瞭望建筑本体）", "边塞角声场景（号角或军旗等军事器物明确可见）"],
        "写一首七言绝句，主题是征戍": ["征戍场景（士兵或戎装人物明确可见）"],
        "写一首羁旅的五言绝句，要有客舍和孤灯": ["客舍（旅居语境的屋舍建筑明确）", "孤灯（灯具或灯光明确可见）"],
        "写一首七言律诗，主题是客愁": ["羁旅愁思场景（行旅人物或客舍须为画面主体）"],
        "写一首七言绝句，主题是送别": ["送别场景（两人及以上的离别互动须明确可见）"],
        "写一首送别的七言律诗，要有长亭和折柳": ["长亭（亭子建筑本体明确）", "柳枝（柳树或柳条明确可辨）"],
        "写一首怀古的五言律诗，要有古城和荒台": ["古城（城墙或城郭本体明确）", "荒台（废弃高台本体明确）"],
        "写一首七言律诗，主题是吊古": ["吊古场景（残垣断碑等遗迹须为画面主体）"],
        "写一首中秋的五言绝句，要有明月和团圆": ["明月（完整圆月明确可见）", "团圆场景（多人聚会明确可见）"],
        "写一首五言律诗，主题是重阳": ["重阳场景（登高人物或菊花须为画面主体）"],
        "写一首禅意的七言绝句，要有古刹和钟鼓": ["古刹（寺庙建筑或佛塔本体明确）", "钟鼓（钟或鼓器物本体明确可见）"],
        "写一首五言绝句，主题是无常": ["无常主题场景（枯萎凋零或流逝意象须为画面主体）"],
    },
    "loose": {
        "写一首秋景的五言律诗，要有疏桐和寒蛩": ["疏桐（秋日稀疏树木即可，不苛求树种可辨）", "寒蛩（蟋蟀虫体或秋虫鸣意象暗示均可）"],
        "写一首七言律诗，主题是悲秋": ["悲秋场景（落叶/枯枝/秋色/萧瑟氛围任一元素出现即可，含边缘元素）"],
        "写一首五言绝句，主题是寒梅": ["梅（梅枝/梅花/寒中开花树木均可）"],
        "写一首冬景的七言绝句，要有飞雪和寒鸦": ["雪（雪/积雪/留白雪意均可）", "寒鸦（任何深色鸟均可）"],
        "写一首五言律诗，主题是雪夜": ["雪夜场景（雪或夜色任一明确、另一有暗示即可）"],
        "写一首冬景的七言律诗，要有炉火和寒灯": ["炉火（炉具/火盆/炭火/暖光均可，不苛求明火）", "寒灯（任何灯光/烛光/窗内光晕均可）"],
        "写一首五言绝句，主题是溪声": ["水（溪/泉/瀑/水面任一水体即可）"],
        "写一首山水的七言律诗，要有高楼和远山": ["楼阁（楼/阁/亭台任一较高建筑即可）", "远山（任何山景均可）"],
        "写一首田园的七言绝句，要有耕牛和炊烟": ["牛（任何牛均可）", "炊烟（屋舍上方烟霭/雾气即可）"],
        "写一首五言律诗，主题是归隐": ["归隐场景（茅屋/山居/隐士/空山幽居氛围任一即可）"],
        "写一首边塞的五言律诗，要有戍楼和角声": ["戍楼（烽火台/城楼/箭楼/任何边塞高耸建筑）", "边塞角声场景（军旗/号角/军营/戍卒/关隘任一暗示）"],
        "写一首七言绝句，主题是征戍": ["征戍场景（士兵/戎装/边塞地貌/军旗/关隘任一元素，含边缘元素）"],
        "写一首羁旅的五言绝句，要有客舍和孤灯": ["客舍（任何屋舍建筑即可）", "孤灯（任何灯光/烛光/窗内光即可）"],
        "写一首七言律诗，主题是客愁": ["羁旅愁思场景（行旅/客舍/孤客/孤舟/远路任一即可）"],
        "写一首七言绝句，主题是送别": ["送别场景（长亭/柳枝/孤舟远行/回望/把酒任一离别意象即可）"],
        "写一首送别的七言律诗，要有长亭和折柳": ["亭（任何亭/驿建筑即可）", "柳（任何柳树/垂枝均可）"],
        "写一首怀古的五言律诗，要有古城和荒台": ["古城（残垣/城郭/古建筑群任一）", "荒台（任何台基/残迹/荒丘均可）"],
        "写一首七言律诗，主题是吊古": ["吊古场景（残垣/断碑/古迹/苍凉氛围任一元素即可）"],
        "写一首中秋的五言绝句，要有明月和团圆": ["明月（任何形态的月亮均可）", "团圆场景（人物相聚/庭院家宴/两人及以上同框即可）"],
        "写一首五言律诗，主题是重阳": ["重阳场景（登高/菊花/茱萸/秋日山行任一暗示即可）"],
        "写一首禅意的七言绝句，要有古刹和钟鼓": ["古刹（寺庙/佛塔/山门/檐角任一）", "钟鼓（钟/鼓/钟楼/鼓楼任一暗示即可）"],
        "写一首五言绝句，主题是无常": ["无常主题场景（枯萎/凋零/流水/落花/孤影任一元素即可）"],
    },
}


def _theme_pairs(by_theme: Dict[str, Any], themes: List[str]) -> list:
    """[(theme, [(kw_idx, present_A, present_B), ...]), ...]，按 kw 顺序配对。"""
    out = []
    for t in themes:
        entry = by_theme[t]
        res_a = entry["arm_A"].get("results", [])
        res_b = entry["arm_B"].get("results", [])
        pairs = [(i, bool(res_a[i].get("present")), bool(res_b[i].get("present")))
                 for i in range(min(len(res_a), len(res_b)))]
        out.append((t, pairs))
    return out


def _variant_stats(by_theme: Dict[str, Any], themes: List[str]) -> Dict[str, Any]:
    tp = _theme_pairs(by_theme, themes)
    n_kw = sum(len(p) for _, p in tp)
    hit_a = sum(a for _, p in tp for _, a, _ in p)
    hit_b = sum(b for _, p in tp for _, _, b in p)
    a_only = sum(1 for _, p in tp for _, a, b in p if a and not b)
    b_only = sum(1 for _, p in tp for _, a, b in p if b and not a)
    ci_lo, ci_hi = bootstrap_ci_theme_level(tp, BOOTSTRAP_ITERS, BOOTSTRAP_SEED)
    return {
        "n_kw": n_kw, "hit_a": hit_a, "hit_b": hit_b,
        "rate_a": hit_a / n_kw, "rate_b": hit_b / n_kw,
        "delta_pp": (hit_b - hit_a) / n_kw * 100,
        "a_only": a_only, "b_only": b_only,
        "mcnemar_p": mcnemar_exact_p(a_only, b_only),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
    }


def _presents(by_theme, theme, arm) -> List[bool]:
    return [bool(r.get("present"))
            for r in by_theme[theme][arm].get("results", [])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vlm", default="qwen-vl-max")
    ap.add_argument("--agg", default=DEFAULT_AGG)
    ap.add_argument("--baseline", default=DEFAULT_BASELINE,
                    help="冻结版 kw 的已有判定 JSON（不重跑）")
    ap.add_argument("--themes", nargs="*", type=int, default=None,
                    help="只跑指定主题 index 1..22（调试用）")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    agg = _load_agg(ROOT / args.agg)
    baseline = json.loads((ROOT / args.baseline).read_text(encoding="utf-8"))

    themes_all = [t for t in agg if t in KW_VARIANTS["strict"]]
    assert len(themes_all) == len(agg), "agg 含变体未覆盖的主题"
    themes = ([themes_all[i - 1] for i in args.themes]
              if args.themes else themes_all)

    # 变体与冻结版逐条 1:1 校验
    for variant, mapping in KW_VARIANTS.items():
        for t in themes:
            assert len(mapping[t]) == len(HARD_CONSTRAINTS[t]["kw"]), \
                f"{variant}/{t}: kw 数与冻结版不一致"

    n_calls = len(themes) * len(ARMS) * len(KW_VARIANTS)
    print(f"[kw_sensitivity] {len(themes)} 主题 × {len(ARMS)} arm × "
          f"{len(KW_VARIANTS)} 变体 = {n_calls} 次 {args.vlm} 调用")

    prompt = _load_prompt_yaml("vlm_hard_constraint.yaml")
    client = _build_client(args.vlm)
    t0 = time.time()

    by_variant: Dict[str, Dict[str, Any]] = {}
    done = 0
    for variant, mapping in KW_VARIANTS.items():
        by_theme: Dict[str, Any] = {}
        for theme in themes:
            by_theme[theme] = {}
            for arm in ARMS:
                cell = agg[theme][arm]
                img = _load_image(_resolve_image_path(ROOT, cell["image_path"]))
                if img is None:
                    by_theme[theme][arm] = {"error": "image load failed",
                                            "results": []}
                    continue
                done += 1
                result = _check_one_image(
                    client, args.vlm, prompt,
                    image=img, poem=cell.get("poem", ""),
                    keywords=mapping[theme],
                    max_tokens=args.max_tokens, temperature=args.temperature)
                by_theme[theme][arm] = result
                hit, total = _fmt_hit(result.get("results", []))
                print(f"[{done}/{n_calls}] {variant} · {arm} · {theme[:16]}… "
                      f"{hit}/{total}  {_fmt_verdict_cell(result.get('results', []))}",
                      flush=True)
        by_variant[variant] = by_theme

    # ── 聚合：baseline + 两变体 ───────────────────────────────────────────
    stats = {"baseline(v1 冻结)": _variant_stats(baseline["by_theme"], themes)}
    for variant in KW_VARIANTS:
        stats[variant] = _variant_stats(by_variant[variant], themes)

    # 单调性违反计数（噪声观测）：strict 不应新增命中、loose 不应减少命中
    mono = {"strict_gain_vs_base": 0, "loose_loss_vs_base": 0}
    flips: List[List[str]] = []
    for theme in themes:
        base_kw = HARD_CONSTRAINTS[theme]["kw"]
        for arm in ARMS:
            pb = _presents(baseline["by_theme"], theme, arm)
            ps = _presents(by_variant["strict"], theme, arm)
            pl = _presents(by_variant["loose"], theme, arm)
            for i in range(min(len(pb), len(ps), len(pl))):
                if ps[i] and not pb[i]:
                    mono["strict_gain_vs_base"] += 1
                if pb[i] and not pl[i]:
                    mono["loose_loss_vs_base"] += 1
                if not (pb[i] == ps[i] == pl[i]):
                    flips.append([theme[:14] + "…", arm, base_kw[i],
                                  "✓" if ps[i] else "✗",
                                  "✓" if pb[i] else "✗",
                                  "✓" if pl[i] else "✗"])

    dt = time.time() - t0
    deltas = [s["delta_pp"] for s in stats.values()]

    md = ["# HARD_CONSTRAINTS kw 严/松敏感度全池扫描（A2）", ""]
    md.append(f"VLM oracle: `{args.vlm}` · {len(themes)} 主题 × 2 arm × 2 变体 "
              f"= {n_calls} 次调用 · 耗时 {dt:.0f}s · baseline 读自 `{args.baseline}`（不重跑）")
    md.append("")
    md.append("## §1 三档 kw 定义下的主结论")
    md.append("")
    rows = []
    for name, s in stats.items():
        rows.append([
            f"**{name}**",
            f"{s['hit_a']}/{s['n_kw']} = {s['rate_a']:.1%}",
            f"{s['hit_b']}/{s['n_kw']} = {s['rate_b']:.1%}",
            f"**{s['delta_pp']:+.1f}pp**",
            f"{s['a_only']}:{s['b_only']}",
            f"{s['mcnemar_p']:.3f}",
            f"[{s['ci_lo']:+.1f}, {s['ci_hi']:+.1f}]pp",
        ])
    md.append(table(["kw 定义", "arm A 命中", "arm B 命中", "Δ(B−A)",
                     "A-only:B-only", "McNemar p", "bootstrap 95% CI"], rows))
    md.append("")
    md.append(f"**稳健区间：Δ ∈ [{min(deltas):+.1f}pp, {max(deltas):+.1f}pp]**"
              f"（三档定义包络）")
    md.append("")
    md.append("## §2 判定翻转明细（任一档与另两档不一致的 kw）")
    md.append("")
    md.append(table(["主题", "arm", "kw（冻结版文本）", "strict", "base", "loose"],
                    flips) if flips else "（无翻转）")
    md.append("")
    md.append("## §3 单调性违反（重判噪声观测）")
    md.append("")
    md.append(f"- strict 较 baseline **新增**命中（逻辑上不应发生）: "
              f"{mono['strict_gain_vs_base']} 处")
    md.append(f"- loose 较 baseline **减少**命中（逻辑上不应发生）: "
              f"{mono['loose_loss_vs_base']} 处")
    md.append("- 违反数 = 'VLM 重判噪声' 的直接观测下界；strict/loose 与 baseline "
              "的差异中有同量级成分应归于噪声而非定义效应。")
    md.append("")
    md.append("## §4 caveats")
    md.append("")
    md.append("- 单 VLM oracle（qwen-vl-max）单次判定；baseline 判定于 2026-07-02、"
              "变体判定于本次运行，跨时间重判噪声混入变体差异（§3 给出观测下界）。")
    md.append("- strict/loose 变体是**本探针私有**的敏感度包络，不构成正式口径；"
              "冻结版 v1 定义（`eval/vlm_hard_constraint.py`）不变。")
    md.append("- 变体措辞由人工撰写，'严/松'程度本身带主观性；结论应读区间而非单点。")

    payload = {
        "meta": {"vlm_model": args.vlm, "agg": args.agg,
                 "baseline": args.baseline, "themes": themes,
                 "n_calls": n_calls, "elapsed_sec": round(dt, 1),
                 "bootstrap": {"iters": BOOTSTRAP_ITERS, "seed": BOOTSTRAP_SEED},
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "kw_variants": KW_VARIANTS,
        "by_variant": by_variant,
        "stats": {k: v for k, v in stats.items()},
        "monotonicity_violations": mono,
    }
    md_text = "\n".join(md)
    paths = save_artifacts("vlm_kw_sensitivity_main_n22", payload, md_text)
    print("\n" + md_text)
    print(f"\njson: {paths['json']}\nmd: {paths['md']}")


if __name__ == "__main__":
    main()
