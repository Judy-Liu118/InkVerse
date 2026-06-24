"""
eval.analyze_clip_dual -- 验证 dual anchor 加权设计的合理性（B + D 对照）

背景：
  生产 dual = α·poem_only + (1-α)·prompt_only（密集 / 稀疏两套权重）。
  但 eval_clip 报告的"dual - prompt_only 配对差值"在数学上恒等于
  α·(poem_only - prompt_only)，本质只在测「poem 比 prompt 高多少」，
  不是在测「加权融合」本身的价值。

  本脚本跑两类正确对照：
    · B 对照：ρ(poem_only, prompt_only) —— 两 anchor 同质度
      → 决定 dual 加权是否有 motivation（noise 越独立，融合越有效）
    · D 对照：α grid search → 各 α 下 dual_α vs VLM 的 Spearman ρ
      → 若最优 α ∈ (0, 1) 且 ρ > max(单 anchor)，dual 设计才被支持
      → 若最优 α 在端点，dual 加权退化为单 anchor，应 deprecate

  顺便分层（rich / sparse）跑 D，验证生产环境「稀疏自适应权重切换」
  是否真有 motivation（两分层最优 α 应显著不同）。

跑法：
    python -m eval.analyze_clip_dual --input outputs/eval/eval_clip_xxx.json
    python -m eval.analyze_clip_dual --input out1.json out2.json   # 合并多份
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from eval.metrics import pearson_corr, spearman_corr


def load_rows(paths: List[str]) -> List[Dict[str, Any]]:
    """从 eval_clip JSON（payload 顶层 = {"config", "rows"}）加载有效行。
    要求 raw_scores 含 poem_only / prompt_only / vlm_oracle（dual 可选）。
    """
    out = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        rows = data.get("rows", data) if isinstance(data, dict) else data
        for r in rows:
            if "error" in r:
                continue
            scores = r.get("raw_scores") or {}
            poem   = scores.get("poem_only")
            prompt = scores.get("prompt_only")
            dual   = scores.get("dual")
            vlm    = scores.get("vlm_oracle")
            if poem is None or prompt is None or vlm is None:
                continue
            out.append({
                "user_input":         r.get("user_input", ""),
                "genre":              r.get("genre", ""),
                "theme":              r.get("theme", ""),
                "keyword_density":    r.get("keyword_density", ""),
                "keyword_word_count": r.get("keyword_word_count"),
                "poem":      float(poem),
                "prompt":    float(prompt),
                "dual_prod": float(dual) if dual is not None else None,
                "vlm":       float(vlm),
            })
    return out


def _grid_search(poem: List[float], prompt: List[float],
                 vlm: List[float]) -> List[Dict[str, Any]]:
    """α ∈ {0, 0.1, ..., 1.0} 共 11 个权重，各算 dual_α vs VLM 的 Spearman ρ。"""
    grid = []
    for i in range(0, 11):
        a = i / 10
        synth = [a * p + (1 - a) * q for p, q in zip(poem, prompt)]
        grid.append({"alpha": a, "rho": spearman_corr(synth, vlm)})
    return grid


def _best(grid: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(grid, key=lambda x: (x["rho"] if x["rho"] is not None else -2))


def analyze(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    poem   = [r["poem"]   for r in rows]
    prompt = [r["prompt"] for r in rows]
    vlm    = [r["vlm"]    for r in rows]

    # A：基线
    rho_poem      = spearman_corr(poem,   vlm)
    rho_prompt    = spearman_corr(prompt, vlm)
    rho_dual_prod = None
    if all(r["dual_prod"] is not None for r in rows):
        rho_dual_prod = spearman_corr([r["dual_prod"] for r in rows], vlm)

    # B：anchor 间相关性
    rho_anchors_spearman = spearman_corr(poem, prompt)
    rho_anchors_pearson  = pearson_corr(poem, prompt)

    # D：全局 α grid
    grid = _grid_search(poem, prompt, vlm)
    best = _best(grid)

    # D 分层：rich / sparse 各跑一次
    grids_by_density: Dict[str, Dict[str, Any]] = {}
    for d in ("rich", "sparse"):
        sub = [r for r in rows if r["keyword_density"] == d]
        if len(sub) < 3:
            continue
        sub_grid = _grid_search(
            [r["poem"]   for r in sub],
            [r["prompt"] for r in sub],
            [r["vlm"]    for r in sub],
        )
        grids_by_density[d] = {
            "n": len(sub), "grid": sub_grid, "best": _best(sub_grid),
        }

    return {
        "n":                    len(rows),
        "rho_poem":             rho_poem,
        "rho_prompt":           rho_prompt,
        "rho_dual_prod":        rho_dual_prod,
        "rho_anchors_spearman": rho_anchors_spearman,
        "rho_anchors_pearson":  rho_anchors_pearson,
        "grid":                 grid,
        "best":                 best,
        "grids_by_density":     grids_by_density,
    }


def _fmt(x, digits=3):
    if x is None:
        return "—"
    return f"{x:+.{digits}f}"


def render(out: Dict[str, Any], input_paths: List[str]) -> str:
    md: List[str] = []
    md.append("# dual anchor 合理性分析（B + D 对照）")
    md.append(f"_n = {out['n']} · 输入：{', '.join(f'`{p}`' for p in input_paths)}_")
    md.append("")

    # ── A
    md.append("## A. 单 anchor vs VLM Spearman ρ（基线）")
    md.append("")
    md.append("| anchor | ρ vs VLM |")
    md.append("|---|---|")
    md.append(f"| prompt_only      | {_fmt(out['rho_prompt'])} |")
    md.append(f"| poem_only        | {_fmt(out['rho_poem'])} |")
    md.append(f"| dual（生产权重） | {_fmt(out['rho_dual_prod'])} |")
    max_single = max(
        out['rho_poem']   if out['rho_poem']   is not None else -2,
        out['rho_prompt'] if out['rho_prompt'] is not None else -2,
    )
    md.append("")
    md.append(f"`max(单 anchor) = {_fmt(max_single)}`")
    md.append("")
    if out['rho_dual_prod'] is not None:
        diff = out['rho_dual_prod'] - max_single
        if diff > 1e-6:
            md.append(f"→ 生产权重 dual 已 > max(单)，差 {_fmt(diff)} ✓")
        else:
            md.append(f"→ 生产权重 dual ≤ max(单)（差 {_fmt(diff)}），生产 α 可能未调优，看 D")
    md.append("")

    # ── B
    md.append("## B. anchor 间相关性（dual 加权 motivation 检验）")
    md.append("")
    md.append("| 度量 | ρ(poem, prompt) |")
    md.append("|---|---|")
    md.append(f"| Spearman | {_fmt(out['rho_anchors_spearman'])} |")
    md.append(f"| Pearson  | {_fmt(out['rho_anchors_pearson'])} |")
    md.append("")
    md.append("解释（noise 独立性近似）：")
    md.append("- **> 0.8**：两 anchor 高度同质 → 融合数学上没用（dual ≈ 单 anchor）")
    md.append("- **0.5 ~ 0.8**：中等相关 → 融合可能轻微降 noise")
    md.append("- **< 0.5**：互补强 → 融合最有 motivation（noise 独立 → SNR 提升）")
    md.append("")

    # ── D 全局
    md.append("## D. α grid search · dual_α = α·poem + (1-α)·prompt vs VLM")
    md.append("")
    md.append("| α | dual_α ρ | vs max(单) |")
    md.append("|---|---|---|")
    for g in out['grid']:
        marker = " ⭐" if abs(g['alpha'] - out['best']['alpha']) < 1e-9 else ""
        delta = ""
        if g['rho'] is not None:
            delta = f"{g['rho'] - max_single:+.3f}"
        md.append(f"| {g['alpha']:.1f} | {_fmt(g['rho'])}{marker} | {delta} |")
    md.append("")
    md.append(f"**最优 α = {out['best']['alpha']:.1f}**, ρ = {_fmt(out['best']['rho'])}")
    md.append("")

    md.append("### 结论判定")
    best_rho = out['best']['rho']
    best_a   = out['best']['alpha']
    if best_rho is None:
        md.append("- 数据不足，无法判定")
    elif best_a >= 0.999:
        md.append("- **最优 α=1.0 → dual 退化为纯 poem_only，加权融合在该样本上无价值**")
    elif best_a <= 0.001:
        md.append("- **最优 α=0.0 → dual 退化为纯 prompt_only，加权融合在该样本上无价值**")
    elif best_rho > max_single + 0.01:
        md.append(f"- 最优 α={best_a:.1f} ∈ (0, 1) 且 ρ={_fmt(best_rho)} 显著 > max(单)={_fmt(max_single)}")
        md.append("- **→ dual 设计被支持。建议把生产权重 α 调到最优值**")
    else:
        md.append(f"- 最优 α={best_a:.1f} ∈ (0, 1) 但 ρ={_fmt(best_rho)} 仅比 max(单)={_fmt(max_single)} 高 < 0.01")
        md.append("- **→ 提升不显著（可能 noise）。扩样本再判**")
    md.append("")

    # ── D 分层
    if out['grids_by_density']:
        md.append("## D-密度分层（验证生产「稀疏自适应权重切换」的 motivation）")
        md.append("")
        for d, info in out['grids_by_density'].items():
            md.append(f"### {d}（n={info['n']}）")
            md.append("")
            md.append("| α | dual_α ρ |")
            md.append("|---|---|")
            for g in info['grid']:
                marker = " ⭐" if abs(g['alpha'] - info['best']['alpha']) < 1e-9 else ""
                md.append(f"| {g['alpha']:.1f} | {_fmt(g['rho'])}{marker} |")
            md.append("")
            md.append(f"最优 α（{d}）= **{info['best']['alpha']:.1f}**, ρ = {_fmt(info['best']['rho'])}")
            md.append("")
        if "rich" in out['grids_by_density'] and "sparse" in out['grids_by_density']:
            rich_a   = out['grids_by_density']['rich']['best']['alpha']
            sparse_a = out['grids_by_density']['sparse']['best']['alpha']
            md.append(f"**rich 最优 α = {rich_a:.1f}    vs    sparse 最优 α = {sparse_a:.1f}**")
            if abs(rich_a - sparse_a) >= 0.2:
                md.append("- 两分层最优 α 显著不同 → **自适应权重切换有 motivation**")
            else:
                md.append("- 两分层最优 α 接近 → 自适应切换在该样本上没显著 motivation")
            md.append("")

    return "\n".join(md)


def main():
    # Windows 控制台默认 GBK，编不出 ρ / ⭐ 等字符 —— 强制 stdout UTF-8
    import io
    if hasattr(sys.stdout, "buffer") and (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    ap = argparse.ArgumentParser(
        description="验证 eval_clip dual anchor 加权设计的合理性（B + D 对照）",
    )
    ap.add_argument("--input", nargs="+", required=True,
                    help="一个或多个 eval_clip JSON 路径（多份会合并）")
    ap.add_argument("--save", default=None,
                    help="可选：报告 markdown 输出路径（默认 outputs/eval/analyze_dual_<ts>.md）")
    args = ap.parse_args()

    rows = load_rows(args.input)
    if not rows:
        print("ERROR: 没读到任何有效行（要求 raw_scores 含 prompt_only/poem_only/vlm_oracle）",
              file=sys.stderr)
        sys.exit(1)
    print(f"载入 {len(rows)} 行")

    out = analyze(rows)
    md  = render(out, args.input)
    print(md)

    save_path = args.save
    if save_path is None:
        out_dir = Path("outputs/eval")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = str(out_dir / f"analyze_dual_{ts}.md")
    Path(save_path).write_text(md, encoding="utf-8")
    print(f"\n报告: {save_path}")


if __name__ == "__main__":
    main()
