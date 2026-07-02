"""dump §4.1 主池 22 主题双图并排 markdown（一次性生成后可直接嵌报告）。"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "outputs" / "eval" / "_agg_arena_ablation_main_n22.json"
VLM = ROOT / "outputs" / "eval" / "vlm_hard_constraint_arm_ab_main_n22_20260702_133542.json"

# 主题短名映射（用于 §4.1 标题）
SHORT_NAMES = {
    "写一首秋景的五言律诗，要有疏桐和寒蛩": "秋景五律·疏桐寒蛩",
    "写一首七言律诗，主题是悲秋": "七律·悲秋",
    "写一首五言绝句，主题是寒梅": "五绝·寒梅",
    "写一首冬景的七言绝句，要有飞雪和寒鸦": "冬景七绝·飞雪寒鸦",
    "写一首五言律诗，主题是雪夜": "五律·雪夜",
    "写一首冬景的七言律诗，要有炉火和寒灯": "冬景七律·炉火寒灯",
    "写一首五言绝句，主题是溪声": "五绝·溪声",
    "写一首山水的七言律诗，要有高楼和远山": "山水七律·高楼远山",
    "写一首田园的七言绝句，要有耕牛和炊烟": "田园七绝·耕牛炊烟",
    "写一首五言律诗，主题是归隐": "五律·归隐",
    "写一首边塞的五言律诗，要有戍楼和角声": "边塞五律·戍楼角声",
    "写一首七言绝句，主题是征戍": "七绝·征戍",
    "写一首羁旅的五言绝句，要有客舍和孤灯": "羁旅五绝·客舍孤灯",
    "写一首七言律诗，主题是客愁": "七律·客愁",
    "写一首七言绝句，主题是送别": "七绝·送别",
    "写一首送别的七言律诗，要有长亭和折柳": "送别七律·长亭折柳",
    "写一首怀古的五言律诗，要有古城和荒台": "怀古五律·古城荒台",
    "写一首七言律诗，主题是吊古": "七律·吊古",
    "写一首中秋的五言绝句，要有明月和团圆": "中秋五绝·明月团圆",
    "写一首五言律诗，主题是重阳": "五律·重阳",
    "写一首禅意的七言绝句，要有古刹和钟鼓": "禅意七绝·古刹钟鼓",
    "写一首五言绝句，主题是无常": "五绝·无常",
}

DENSITY = {
    "写一首秋景的五言律诗，要有疏桐和寒蛩": "rich",
    "写一首七言律诗，主题是悲秋": "sparse",
    "写一首五言绝句，主题是寒梅": "sparse",
    "写一首冬景的七言绝句，要有飞雪和寒鸦": "rich",
    "写一首五言律诗，主题是雪夜": "sparse",
    "写一首冬景的七言律诗，要有炉火和寒灯": "rich",
    "写一首五言绝句，主题是溪声": "sparse",
    "写一首山水的七言律诗，要有高楼和远山": "rich",
    "写一首田园的七言绝句，要有耕牛和炊烟": "rich",
    "写一首五言律诗，主题是归隐": "sparse",
    "写一首边塞的五言律诗，要有戍楼和角声": "rich",
    "写一首七言绝句，主题是征戍": "sparse",
    "写一首羁旅的五言绝句，要有客舍和孤灯": "rich",
    "写一首七言律诗，主题是客愁": "sparse",
    "写一首七言绝句，主题是送别": "sparse",
    "写一首送别的七言律诗，要有长亭和折柳": "rich",
    "写一首怀古的五言律诗，要有古城和荒台": "rich",
    "写一首七言律诗，主题是吊古": "sparse",
    "写一首中秋的五言绝句，要有明月和团圆": "rich",
    "写一首五言律诗，主题是重阳": "sparse",
    "写一首禅意的七言绝句，要有古刹和钟鼓": "rich",
    "写一首五言绝句，主题是无常": "sparse",
}


def _fmt_vlm(results):
    if not results:
        return "—"
    parts = []
    hit = 0
    for r in results:
        kw = r.get("keyword", "?")
        p = r.get("present")
        mark = "✓" if p is True else ("✗" if p is False else "?")
        if p is True:
            hit += 1
        # 简化 keyword 展示（去掉长括号说明）
        kw_short = kw.split("（")[0]
        parts.append(f"{kw_short}{mark}")
    return " · ".join(parts) + f" ({hit}/{len(results)})"


def _hit_count(results):
    if not results:
        return 0, 0
    hit = sum(1 for r in results if r.get("present") is True)
    return hit, len(results)


def _verdict(a_results, b_results, a_clip, b_clip):
    ah, at = _hit_count(a_results)
    bh, bt = _hit_count(b_results)
    if ah == bh:
        cd = b_clip - a_clip
        if abs(cd) < 0.005:
            return "**双方硬约束一致 + CLIP 平手**"
        winner_c = "arm B" if cd > 0 else "arm A"
        return f"**硬约束一致**，CLIP {winner_c} +{abs(cd):.3f}"
    if bh > ah:
        return f"**arm B 独赢** 硬约束 +{bh - ah}"
    return f"**arm A 独赢** 硬约束 +{ah - bh}"


def _fmt_poem(poem):
    if not poem:
        return "(空)"
    return "<br>".join(l.strip() for l in poem.strip().split("\n") if l.strip())


def _line_for_theme(idx, theme, agg_cell, vlm_cell):
    short = SHORT_NAMES.get(theme, theme[:20])
    density = DENSITY.get(theme, "?")
    a = agg_cell["arm_A"]
    b = agg_cell["arm_B"]
    a_vlm = vlm_cell.get("arm_A", {}).get("results", [])
    b_vlm = vlm_cell.get("arm_B", {}).get("results", [])
    a_clip = a.get("clip_raw")
    b_clip = b.get("clip_raw")
    # 硬约束 kw list（从 VLM cell 里 first result 的 keyword 反查更麻烦，直接从 results 提）
    kws = [r.get("keyword", "?") for r in a_vlm]

    md = []
    md.append(f"### 主题 {idx} · {short}（{density}）")
    md.append("")
    md.append(f"硬约束：{'、'.join(kws)}")
    md.append("")
    md.append(f"| arm A · CLIP={a_clip:.3f} | arm B · CLIP={b_clip:.3f} |")
    md.append("|---|---|")
    md.append(f"| ![](../{a['image_path']}) | ![](../{b['image_path']}) |")
    md.append(f"| 诗名：*{a.get('title', '') or '(无)'}* | 诗名：*{b.get('title', '') or '(无)'}* |")
    md.append(f"| {_fmt_poem(a.get('poem', ''))} | {_fmt_poem(b.get('poem', ''))} |")
    md.append(f"| VLM: {_fmt_vlm(a_vlm)} | VLM: {_fmt_vlm(b_vlm)} |")
    md.append("")
    md.append(f"判定：{_verdict(a_vlm, b_vlm, a_clip, b_clip)}")
    md.append("")
    md.append("---")
    md.append("")
    return "\n".join(md)


def main():
    agg = json.loads(AGG.read_text(encoding="utf-8"))
    vlm = json.loads(VLM.read_text(encoding="utf-8"))["by_theme"]

    lines = []
    for i, (theme, cell) in enumerate(agg.items(), start=1):
        vlm_cell = vlm.get(theme, {})
        lines.append(_line_for_theme(i, theme, cell, vlm_cell))

    out = "\n".join(lines)
    out_path = ROOT / "outputs" / "eval" / "_report_section_main22.md"
    out_path.write_text(out, encoding="utf-8")
    print(f"写入 {out_path}")
    print(f"总行数: {out.count(chr(10))}")


if __name__ == "__main__":
    main()
