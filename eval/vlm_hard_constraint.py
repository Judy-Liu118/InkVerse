"""
eval.vlm_hard_constraint -- VLM 硬约束命中率评测

**独立打补丁**主 sweep 里 CLIP 的系统性盲点：CLIP 偏好"图里东西多 + 意境自洽"，
不抠用户 input 里明确要求的具体物体/场景（如"要有柳树和燕子"里的柳树、燕子）。

用法：
    python -m eval.vlm_hard_constraint \
        --agg outputs/eval/_agg_3deltas.json \
        --vlm qwen-vl-max

    # 冒烟：只跑 delta=0.17 的 3 主题
    python -m eval.vlm_hard_constraint --delta 0.17 --themes 1 2 3

输入：
  · 一份 sweep 聚合 JSON（如 _agg_3deltas.json），结构 {theme_user_input:
    {delta: {image_path, image_paths_all, poem, ...}}}
  · 每个主题对应的硬约束关键词列表（HARD_CONSTRAINTS，硬编码在本文件顶部）

输出：
  · outputs/eval/vlm_hard_constraint_<ts>.json —— 每张图 × 每关键词 yes/no + reason
  · outputs/eval/vlm_hard_constraint_<ts>.md   —— 跨 delta / 跨 density / 主题级 hit rate

VLM oracle 默认 qwen-vl-max（与 REPORT_autonomous_n5_20260627 同源），单张图 1 call
（一次问所有 keyword），30 张图 = 30 calls。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from eval.report import save_artifacts, table, fmt_num, print_and_return
from eval.vlm_judge import (
    _resolve_backend, _resolve_api_key, _resolve_base_url,
    _image_to_data_url, _load_prompt_yaml,
)
from core.logger import get_logger

_log = get_logger(__name__)


# ── 硬约束关键词映射（10 主题，与 dataset.get_benchmark(n=10) 对齐）────
# key = BenchInput.user_input，value = {"kw": [...], "density": "rich"/"sparse"}
HARD_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "写一首春景的五言绝句，要有柳树和燕子": {
        "kw": ["柳树", "燕子"], "density": "rich",
    },
    "写一首七言绝句，主题是征戍": {
        "kw": ["征戍场景（士兵/戎装/边塞地貌）"], "density": "sparse",
    },
    "写一首春景的五言律诗，要有桃花和啼莺": {
        "kw": ["桃花", "莺"], "density": "rich",
    },
    "写一首七言律诗，主题是客愁": {
        "kw": ["羁旅愁思场景（行旅/客舍/孤客）"], "density": "sparse",
    },
    "写一首五言绝句，主题是夏蝉": {
        "kw": ["蝉"], "density": "sparse",
    },
    "写一首田园的七言绝句，要有耕牛和炊烟": {
        "kw": ["耕牛", "炊烟"], "density": "rich",
    },
    "写一首五言律诗，主题是消夏": {
        "kw": ["消夏场景（扇/竹榻/凉席/水塘/浓荫）"], "density": "sparse",
    },
    "写一首山水的七言律诗，要有高楼和远山": {
        "kw": ["高楼", "远山"], "density": "rich",
    },
    "写一首羁旅的五言绝句，要有客舍和孤灯": {
        "kw": ["客舍", "孤灯"], "density": "rich",
    },
    "写一首七言绝句，主题是春雨": {
        "kw": ["雨"], "density": "sparse",
    },
}


# ── VLM 调用 ─────────────────────────────────────────────────────────────
def _parse_constraint_response(raw: str, model: str) -> Dict[str, Any]:
    """从 VLM 回复解析 {results: [{keyword, present, reason}, ...]}。
    解析失败时返回 error 字段，caller 决定是否重试。
    """
    if not raw:
        return {"error": "empty response", "raw": raw}
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(l for l in lines if not l.strip().startswith("```"))
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return {"error": "no JSON braces", "raw": s[:200]}
    try:
        obj = json.loads(s[i:j+1])
    except json.JSONDecodeError as e:
        return {"error": f"json decode: {e}", "raw": s[:200]}
    if not isinstance(obj, dict) or "results" not in obj:
        return {"error": "missing 'results'", "raw": s[:200]}
    if not isinstance(obj["results"], list):
        return {"error": "'results' not a list", "raw": s[:200]}
    return {"results": obj["results"]}


def _check_one_image(
    client, model: str, prompt: Dict[str, str], *,
    image: Image.Image, poem: str, keywords: List[str],
    max_tokens: int, temperature: float,
) -> Dict[str, Any]:
    """一张图 × 一组 keyword，一次 VLM call。返回 {results, error?, raw?}。"""
    keyword_json = json.dumps(keywords, ensure_ascii=False)
    user_text = prompt["user"].format(
        poem=poem.strip() or "（空）",
        keyword_list=keyword_json,
    )
    messages: List[Dict[str, Any]] = []
    if prompt["system"]:
        messages.append({"role": "system", "content": prompt["system"]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": _image_to_data_url(image)}},
            {"type": "text", "text": user_text},
        ],
    })
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
    except Exception as e:
        _log.warning("[vlm_hard_constraint] %s API 失败: %s", model, e)
        return {"error": f"api: {e}", "results": []}
    try:
        raw = resp.choices[0].message.content or ""
    except (AttributeError, IndexError) as e:
        return {"error": f"resp parse: {e}", "results": []}
    parsed = _parse_constraint_response(raw, model)
    parsed["raw"] = raw
    return parsed


def _build_client(model: str):
    backend = _resolve_backend(model)
    api_key: Optional[str]
    try:
        api_key = _resolve_api_key(backend, None)
    except RuntimeError:
        from config import DASHSCOPE_API_KEY, ZHIPU_API_KEY
        api_key = {"qwen": DASHSCOPE_API_KEY, "zhipu": ZHIPU_API_KEY}.get(backend, "")
        if not api_key:
            raise RuntimeError(
                f"{backend} 的 API key 既不在环境变量也不在 config 中"
            )
    base_url = _resolve_base_url(backend)
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("VLM 需要 openai 包") from e
    return OpenAI(api_key=api_key, base_url=base_url)


# ── 主入口 ───────────────────────────────────────────────────────────────
def _load_agg(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_image_path(root: Path, rel: str) -> Path:
    """agg JSON 里的 image_path 是 Windows 反斜杠相对路径，转成 posix 绝对路径。"""
    p = Path(rel.replace("\\", "/"))
    if not p.is_absolute():
        p = root / p
    return p


def _load_image(path: Path) -> Optional[Image.Image]:
    if not path.is_file():
        _log.error("[vlm_hard_constraint] 图片不存在: %s", path)
        return None
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception as e:
        _log.error("[vlm_hard_constraint] 图片读取失败 %s: %s", path, e)
        return None


def _fmt_hit(results: List[Dict[str, Any]]) -> Tuple[int, int]:
    """(hit, total)——present=true 的数量 / 总关键词数。"""
    total = len(results)
    hit = sum(1 for r in results if r.get("present") is True)
    return hit, total


def _fmt_verdict_cell(results: List[Dict[str, Any]]) -> str:
    """把 [{kw, present, reason}] 渲染成 markdown 单元：kw✓/kw✗ 列表。"""
    if not results:
        return "—"
    parts = []
    for r in results:
        kw = r.get("keyword", "?")
        p = r.get("present")
        mark = "✓" if p is True else ("✗" if p is False else "?")
        parts.append(f"{kw}{mark}")
    return " · ".join(parts)


def _render_markdown(payload: Dict[str, Any]) -> str:
    md: List[str] = []
    meta = payload["meta"]
    md.append("# VLM 硬约束命中率评测报告")
    md.append("")
    md.append(f"- VLM oracle: `{meta['vlm_model']}`")
    md.append(f"- 数据源: `{meta['agg_json']}`")
    md.append(f"- 主题数: {meta['n_themes']} · Delta 数: {meta['n_deltas']} · 图像数: {meta['n_images']}")
    md.append(f"- API 成功 call: {meta['n_success_calls']} / {meta['n_total_calls']}")
    md.append(f"- 生成时间: {meta['timestamp']}")
    md.append("")

    md.append("## §1 跨 delta 总命中率")
    md.append("")
    md.append("**hit** = present=true 的 keyword 数；**total** = 应命中的 keyword 数（不含 API 失败样本）。")
    md.append("")
    rows = []
    for delta in meta["deltas"]:
        agg = payload["by_delta"][delta]
        rate = agg["hit"] / agg["total"] if agg["total"] > 0 else 0.0
        rows.append([f"**{delta}**", agg["hit"], agg["total"], f"{rate:.1%}"])
    md.append(table(["delta", "hit", "total", "命中率"], rows))
    md.append("")

    md.append("## §2 rich vs sparse 命中率拆分")
    md.append("")
    md.append("rich 题的硬约束是**具体物体**（柳/燕/桃花/莺/耕牛/炊烟/高楼/远山/客舍/孤灯），");
    md.append("sparse 题只有**场景/主题词**（征戍/客愁/夏蝉/消夏/春雨），后者判定更宽松。");
    md.append("")
    rows = []
    for density in ["rich", "sparse"]:
        for delta in meta["deltas"]:
            agg = payload["by_density_delta"][density][delta]
            rate = agg["hit"] / agg["total"] if agg["total"] > 0 else 0.0
            rows.append([density, delta, agg["hit"], agg["total"], f"{rate:.1%}"])
    md.append(table(["density", "delta", "hit", "total", "命中率"], rows))
    md.append("")

    md.append("## §3 主题 × Delta 明细")
    md.append("")
    for theme in meta["themes"]:
        density = HARD_CONSTRAINTS[theme]["density"]
        kw_list = HARD_CONSTRAINTS[theme]["kw"]
        md.append(f"### {theme}")
        md.append(f"- 硬约束: {kw_list} · density: `{density}`")
        md.append("")
        rows = []
        per_theme = payload["by_theme"][theme]
        for delta in meta["deltas"]:
            entry = per_theme.get(delta)
            if entry is None or entry.get("error"):
                err = (entry or {}).get("error", "无数据")
                rows.append([f"**{delta}**", "—", "—", "—", f"⚠️ {err}"])
                continue
            results = entry["results"]
            hit, total = _fmt_hit(results)
            rows.append([
                f"**{delta}**",
                hit, total,
                f"{hit/total:.0%}" if total > 0 else "—",
                _fmt_verdict_cell(results),
            ])
        md.append(table(["delta", "hit", "total", "%", "逐项判定"], rows))
        md.append("")

    md.append("## §4 与 CLIP 分对照")
    md.append("")
    md.append("检查 CLIP 高的图是否 hit rate 也高（如果无相关，说明 CLIP 与硬约束脱钩）。")
    md.append("")
    rows = []
    for theme in meta["themes"]:
        per_theme = payload["by_theme"][theme]
        for delta in meta["deltas"]:
            entry = per_theme.get(delta)
            if entry is None or entry.get("error"):
                continue
            results = entry["results"]
            hit, total = _fmt_hit(results)
            clip_raw = entry.get("clip_raw")
            rows.append([
                theme[:20] + "..." if len(theme) > 22 else theme,
                delta,
                fmt_num(clip_raw) if clip_raw is not None else "—",
                f"{hit}/{total}" if total > 0 else "—",
                f"{hit/total:.0%}" if total > 0 else "—",
            ])
    md.append(table(["主题（前 20 字）", "delta", "CLIP raw", "hit/total", "命中率"], rows))
    md.append("")

    md.append("## §5 caveats")
    md.append("")
    md.append("- **n=10 主题、单 VLM oracle**：结论强度受 VLM 判断本身分布制约。")
    md.append("- **sparse 题判定宽松**：场景类关键词（如\"消夏场景（扇/竹榻/凉席/水塘/浓荫）\"）只要出现任一元素即算 hit，命中率上限天然偏高。")
    md.append("- **每张图只判一次**：VLM 单次判定有噪声，同图重跑 3 次可能给出不同 present 值。若关键结论押在这份数据上，建议对争议样本 (theme, delta) 二次核查。")
    md.append("- **单 delta 内 n=1**：每 (theme, delta) 只对应 1 张图；rich/sparse 各 5 张 × 3 delta = 15 张。命中率是**逐图逐 keyword 拉平**的，不是主题级平均。")
    md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser(description="VLM 硬约束命中率评测")
    ap.add_argument("--agg", default="outputs/eval/_agg_3deltas.json",
                    help="sweep 聚合 JSON 路径")
    ap.add_argument("--vlm", default="qwen-vl-max",
                    help="VLM oracle 模型（qwen-vl-max / qwen-vl-plus / glm-4v）")
    ap.add_argument("--delta", nargs="*", default=None,
                    help="只跑指定 delta（如 --delta 0.17 0.20）；默认全部")
    ap.add_argument("--themes", nargs="*", type=int, default=None,
                    help="只跑指定主题 index 1..N；默认全部")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="判定应保守，温度尽量低")
    ap.add_argument("--out-name", default="vlm_hard_constraint",
                    help="输出文件名前缀")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    agg_path = (root / args.agg) if not Path(args.agg).is_absolute() else Path(args.agg)
    if not agg_path.is_file():
        raise SystemExit(f"agg JSON 不存在: {agg_path}")

    agg = _load_agg(agg_path)
    themes_all = list(agg.keys())
    if args.themes is not None:
        themes = [themes_all[i - 1] for i in args.themes if 1 <= i <= len(themes_all)]
    else:
        themes = themes_all

    # 校验 HARD_CONSTRAINTS 是否覆盖所有 theme
    missing = [t for t in themes if t not in HARD_CONSTRAINTS]
    if missing:
        raise SystemExit(f"HARD_CONSTRAINTS 未覆盖 {len(missing)} 主题: {missing}")

    deltas_all = None
    for t in themes:
        deltas_all = list(agg[t].keys())
        break
    if args.delta is not None:
        deltas = [d for d in args.delta if d in (deltas_all or [])]
    else:
        deltas = deltas_all or []

    print(f"[vlm_hard_constraint] 目标: {len(themes)} 主题 × {len(deltas)} delta = "
          f"{len(themes) * len(deltas)} 张图")

    prompt = _load_prompt_yaml("vlm_hard_constraint.yaml")
    client = _build_client(args.vlm)

    by_theme: Dict[str, Dict[str, Any]] = {}
    n_total = 0
    n_success = 0
    t0 = time.time()

    for i, theme in enumerate(themes):
        by_theme[theme] = {}
        kw = HARD_CONSTRAINTS[theme]["kw"]
        for delta in deltas:
            cell = agg[theme].get(delta)
            if not cell or not cell.get("image_path"):
                by_theme[theme][delta] = {"error": "no image_path in agg"}
                continue
            img_path = _resolve_image_path(root, cell["image_path"])
            image = _load_image(img_path)
            if image is None:
                by_theme[theme][delta] = {"error": f"image load failed: {img_path}"}
                continue
            n_total += 1
            print(f"  [{i+1}/{len(themes)}] delta={delta}  "
                  f"kw={kw}  img={img_path.name}", flush=True)
            result = _check_one_image(
                client, args.vlm, prompt,
                image=image, poem=cell.get("poem", ""),
                keywords=kw,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
            result["image_path"] = str(img_path.relative_to(root)).replace("\\", "/")
            result["poem"] = cell.get("poem", "")
            result["clip_raw"] = cell.get("clip_raw")
            by_theme[theme][delta] = result
            if not result.get("error"):
                n_success += 1
                results = result.get("results", [])
                hit, total = _fmt_hit(results)
                print(f"      → {hit}/{total} hit  {_fmt_verdict_cell(results)}",
                      flush=True)
            else:
                print(f"      → ERROR: {result['error']}", flush=True)

    dt = time.time() - t0
    print(f"[vlm_hard_constraint] done · {n_success}/{n_total} 成功 · 耗时 {dt:.1f}s")

    # 聚合
    by_delta = {d: {"hit": 0, "total": 0} for d in deltas}
    by_density_delta = {
        density: {d: {"hit": 0, "total": 0} for d in deltas}
        for density in ["rich", "sparse"]
    }
    for theme in themes:
        density = HARD_CONSTRAINTS[theme]["density"]
        for delta in deltas:
            entry = by_theme[theme].get(delta)
            if entry is None or entry.get("error"):
                continue
            hit, total = _fmt_hit(entry.get("results", []))
            by_delta[delta]["hit"] += hit
            by_delta[delta]["total"] += total
            by_density_delta[density][delta]["hit"] += hit
            by_density_delta[density][delta]["total"] += total

    payload = {
        "meta": {
            "vlm_model": args.vlm,
            "agg_json": str(agg_path.relative_to(root)).replace("\\", "/"),
            "themes": themes,
            "deltas": deltas,
            "n_themes": len(themes),
            "n_deltas": len(deltas),
            "n_images": n_total,
            "n_total_calls": n_total,
            "n_success_calls": n_success,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(dt, 1),
        },
        "by_theme": by_theme,
        "by_delta": by_delta,
        "by_density_delta": by_density_delta,
    }

    md = _render_markdown(payload)
    paths = save_artifacts(args.out_name, payload, md)
    print_and_return(md)
    print(f"[vlm_hard_constraint] 保存到:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
