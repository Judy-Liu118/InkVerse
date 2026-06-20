"""
eval.report -- 评估报告输出

每次评估都会同时落两份：
  · `outputs/eval/<name>_<timestamp>.json` —— 原始数据，可二次分析
  · `outputs/eval/<name>_<timestamp>.md`   —— markdown 报告，可直接抄进简历

控制台也会 print 一份 markdown 摘要。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _ROOT / "outputs" / "eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_artifacts(name: str, payload: Dict[str, Any], markdown: str,
                    html: str = None) -> Dict[str, Path]:
    """保存 JSON 原始数据 + markdown 报告，可选 HTML 热图。返回各文件路径。"""
    ts = _ts()
    paths = {}
    paths["json"] = OUTPUT_DIR / f"{name}_{ts}.json"
    paths["md"]   = OUTPUT_DIR / f"{name}_{ts}.md"
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    paths["md"].write_text(markdown, encoding="utf-8")
    if html is not None:
        paths["html"] = OUTPUT_DIR / f"{name}_{ts}.html"
        paths["html"].write_text(html, encoding="utf-8")
    return paths


def fmt_num(x: float, digits: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def table(headers: List[str], rows: List[List[Any]]) -> str:
    """生成一个 GitHub-flavored markdown 表。"""
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(sep) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def summary_block(title: str, stats: Dict[str, float], digits: int = 3) -> str:
    """统一渲染一个 summary：n / mean ± std / median / min~max。"""
    n = stats.get("n", 0)
    if n == 0:
        return f"**{title}** —— 无数据"
    return (
        f"**{title}** · n={n} · "
        f"均值 {fmt_num(stats['mean'], digits)} ± {fmt_num(stats['std'], digits)} · "
        f"中位 {fmt_num(stats['median'], digits)} · "
        f"区间 [{fmt_num(stats['min'], digits)}, {fmt_num(stats['max'], digits)}]"
    )


def print_and_return(markdown: str) -> str:
    """控制台 print 一份，方便交互式查看；同时返回 markdown 文本。"""
    print("\n" + "=" * 70)
    print(markdown)
    print("=" * 70 + "\n")
    return markdown


# ── 排名 / 热图 工具 ───────────────────────────────────────────────────────
_RANK_EMOJI = ["🥇", "🥈", "🥉", "🏅", "🏅"]

def rank_emoji(rank: int) -> str:
    """rank 从 0 开始；0=🥇, 1=🥈, 2=🥉, 其后=🏅。"""
    if rank < 0:
        return ""
    return _RANK_EMOJI[min(rank, len(_RANK_EMOJI) - 1)]


def bold_max(values: List[float], digits: int = 3) -> List[str]:
    """把列表里的最大值用 markdown 加粗，其他正常格式。"""
    if not values:
        return []
    nums = [v if isinstance(v, (int, float)) else float("-inf") for v in values]
    mx = max(nums)
    out = []
    for v in nums:
        s = fmt_num(v, digits)
        out.append(f"**{s}**" if v == mx and mx > float("-inf") else s)
    return out


def _hsl_for(value: float, vmin: float, vmax: float) -> str:
    """0 → 红 (hsl 0)，1 → 绿 (hsl 120)。线性插值，越高越绿。"""
    if vmax <= vmin:
        return "hsl(60, 60%, 88%)"
    t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    hue = int(t * 120)  # 0=red, 120=green
    return f"hsl({hue}, 65%, 82%)"


def heatmap_html(title: str, headers: List[str], rows: List[List[Any]],
                 cell_min: float = 0.0, cell_max: float = 1.0,
                 row_labels_first_col: bool = True) -> str:
    """生成一个带 hsl 渐变的 HTML 热图表（独立 <table>）。

    每个数值 cell 按值在 [cell_min, cell_max] 内的位置上色：绿=高，红=低。
    row_labels_first_col=True 表示第一列是行标签（不上色）。
    """
    th_style = ("padding:6px 12px; border:1px solid #ccc; "
                "background:#f0f0f0; font-weight:600;")
    label_style = ("padding:6px 12px; border:1px solid #ccc; "
                    "font-weight:600; background:#fafafa;")
    parts = [f"<h3 style='margin:18px 0 8px;'>{title}</h3>"]
    parts.append("<table style='border-collapse:collapse; font-family:monospace;'>")
    parts.append("<tr>" + "".join(f"<th style='{th_style}'>{h}</th>" for h in headers) + "</tr>")
    for row in rows:
        cells = []
        for i, v in enumerate(row):
            if row_labels_first_col and i == 0:
                cells.append(f"<td style='{label_style}'>{v}</td>")
                continue
            if isinstance(v, (int, float)):
                bg = _hsl_for(float(v), cell_min, cell_max)
                cells.append(
                    f"<td style='padding:6px 12px; border:1px solid #ccc; "
                    f"background:{bg}; text-align:right;'>{fmt_num(v)}</td>"
                )
            else:
                cells.append(
                    f"<td style='padding:6px 12px; border:1px solid #ccc; "
                    f"text-align:right;'>{v}</td>"
                )
        parts.append("<tr>" + "".join(cells) + "</tr>")
    parts.append("</table>")
    return "\n".join(parts)


def wrap_html_doc(title: str, body: str) -> str:
    """把若干 HTML 片段包成一个完整的可打开 .html 文档。"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
          margin: 24px; color:#222; max-width: 1200px; }}
  h1 {{ font-size: 22px; border-bottom: 2px solid #444; padding-bottom: 6px; }}
  h2 {{ font-size: 18px; margin-top: 28px; }}
  h3 {{ font-size: 16px; }}
  code, pre {{ font-family: "Cascadia Code", Consolas, monospace; background:#f6f6f6;
              padding: 2px 6px; border-radius: 3px; }}
  pre {{ padding: 12px; }}
  .legend {{ font-size: 13px; color:#555; margin-top:4px; }}
  .legend .swatch {{ display:inline-block; width:14px; height:14px; vertical-align:middle;
                     border:1px solid #aaa; margin:0 4px; }}
</style></head><body>
<h1>{title}</h1>
<p class="legend">单元格底色：
  <span class="swatch" style="background:hsl(0,65%,82%);"></span>低分
  <span class="swatch" style="background:hsl(60,65%,82%);"></span>中
  <span class="swatch" style="background:hsl(120,65%,82%);"></span>高分
</p>
{body}
</body></html>"""


__all__ = [
    "OUTPUT_DIR",
    "save_artifacts",
    "fmt_num", "table",
    "summary_block",
    "print_and_return",
    "rank_emoji", "bold_max",
    "heatmap_html", "wrap_html_doc",
]
