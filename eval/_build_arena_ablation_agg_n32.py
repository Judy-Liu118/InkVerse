"""
eval._build_arena_ablation_agg_n32 -- 主池 n=22 arena ablation agg（04-22 backend）

结构（与 _agg_3deltas.json 兼容，可喂给 eval.vlm_hard_constraint）：
{
    theme: {
        arm_A: {clip_raw, poem, title, image_prompt, image_path, ..., backend, pool},
        arm_B: {clip_raw, poem, title, image_prompt, image_path, ..., backend, pool},
    }
}

数据来源（主对比池 pool="main", n=22, backend 04-22）:
  arm A: outputs/eval/sweep_pairwise_win_delta_20260702_094052.json
         图 outputs/eval/sweep_pairwise_win_delta_images_20260702_085340/delta_0.17/
  arm B: outputs/eval/sweep_pairwise_win_delta_20260702_105404.json
         图 outputs/eval/sweep_pairwise_win_delta_images_20260702_094058/delta_0.17/

设计说明：
  用户跑的是 `--n 32 --offset 10`，但 get_benchmark(n=10) 和 get_benchmark(n=32)
  是独立采样（不是嵌套），所以 n=32 前 10 主题和 n=10 主题池不同。--offset 10
  跳过 n=32 前 10 主题后，剩下 22 主题里有 5 个和 n=10 主题池 accidentally 重合
  （但由不同 backend 生成，其实是"同题跨 backend 各跑一次"）。

  为避免混淆，本 agg 只产主池 n=22（04-22 严格可比）。辅池 n=10（03-03）继续用
  旧 _agg_arena_ablation.json，报告里作为 consistency check 并列。
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "eval"

# 主池（n=22, backend 04-22）
MAIN_ARM_A_JSON = OUTPUT / "sweep_pairwise_win_delta_20260702_094052.json"
MAIN_ARM_A_IMG_DIR = OUTPUT / "sweep_pairwise_win_delta_images_20260702_085340" / "delta_0.17"
MAIN_ARM_B_JSON = OUTPUT / "sweep_pairwise_win_delta_20260702_105404.json"
MAIN_ARM_B_IMG_DIR = OUTPUT / "sweep_pairwise_win_delta_images_20260702_094058" / "delta_0.17"

OUT_JSON = OUTPUT / "_agg_arena_ablation_main_n22.json"


def _pick_terminal_image(theme_idx_1based: int, img_dir: Path) -> Tuple[Optional[Path], List[Path]]:
    """从 sweep 图目录挑该主题 clip 分最高的一张作为"终图"。
    返回 (终图, 所有图 sorted by name)。
    """
    prefix = f"{theme_idx_1based:02d}_"
    candidates: List[Tuple[float, Path]] = []
    all_imgs: List[Path] = []
    for p in sorted(img_dir.glob(f"{prefix}*.png")):
        all_imgs.append(p)
        m = re.search(r"_clip([0-9.]+)\.png$", p.name)
        if not m:
            continue
        try:
            clip = float(m.group(1))
        except ValueError:
            continue
        candidates.append((clip, p))
    if not candidates:
        return None, all_imgs
    candidates.sort(key=lambda x: (x[0], x[1].name))
    return candidates[-1][1], all_imgs


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def _row_to_cell(row: Dict[str, Any], theme_idx_1based: int,
                 img_dir: Path, backend: str, pool: str,
                 max_poem_rounds: int) -> Dict[str, Any]:
    """把 sweep JSON 的 row 转成 agg cell 结构。"""
    terminal_img, all_imgs = _pick_terminal_image(theme_idx_1based, img_dir)
    return {
        "n": 1,
        "clip_raw": row.get("clip_raw"),
        "attack_succeed": row.get("attack_succeed", 0),
        "evo_rounds": row.get("evo_rounds", 0),
        "attack_rate": row.get("attack_rate", 0.0),
        "poem": row.get("poem", ""),
        "title": row.get("title", ""),
        "image_prompt": row.get("image_prompt", ""),
        "image_path": _rel(terminal_img) if terminal_img else None,
        "image_paths_all": [_rel(p) for p in all_imgs],
        "image_count": len(all_imgs),
        "poem_evolution": row.get("poem_evolution", []),
        "backend": backend,
        "pool": pool,
        "max_poem_rounds": max_poem_rounds,
    }


def _load_sweep(path: Path) -> Tuple[List[Dict[str, Any]], str, int]:
    """返回 (rows, backend, max_poem_rounds)。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    return (
        data["results"][0]["rows"],
        cfg.get("image_backend", ""),
        cfg.get("max_poem_rounds", -1),
    )


def main() -> None:
    for p in (MAIN_ARM_A_JSON, MAIN_ARM_B_JSON):
        if not p.is_file():
            raise SystemExit(f"缺文件: {p}")

    main_a_rows, main_a_backend, main_a_mpr = _load_sweep(MAIN_ARM_A_JSON)
    main_b_rows, main_b_backend, main_b_mpr = _load_sweep(MAIN_ARM_B_JSON)
    assert main_a_mpr == 0, f"主池 arm A 应 max_poem_rounds=0，实际 {main_a_mpr}"
    assert main_b_mpr == 2, f"主池 arm B 应 max_poem_rounds=2，实际 {main_b_mpr}"
    assert main_a_backend == main_b_backend, "主池两 arm backend 不一致"

    out: Dict[str, Dict[str, Any]] = {}
    stats = {"n": 0, "missing_img_a": [], "missing_img_b": []}

    themes_a = [r["theme"] for r in main_a_rows]
    themes_b = [r["theme"] for r in main_b_rows]
    if themes_a != themes_b:
        print("⚠ 两 arm 主题顺序不一致，用 arm A 顺序，arm B lookup by theme")
    b_by_theme = {r["theme"]: r for r in main_b_rows}

    for i, ra in enumerate(main_a_rows, start=1):
        theme = ra["theme"]
        rb = b_by_theme.get(theme)
        if rb is None:
            print(f"⚠ arm B 找不到主题: {theme[:30]}")
            continue
        arm_A = _row_to_cell(ra, i, MAIN_ARM_A_IMG_DIR, main_a_backend, "main", 0)
        arm_B = _row_to_cell(rb, i, MAIN_ARM_B_IMG_DIR, main_b_backend, "main", 2)
        if arm_A["image_path"] is None:
            stats["missing_img_a"].append((i, theme))
        if arm_B["image_path"] is None:
            stats["missing_img_b"].append((i, theme))
        out[theme] = {"arm_A": arm_A, "arm_B": arm_B}
        stats["n"] += 1

    OUT_JSON.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"写入 {OUT_JSON} · 主池 n={stats['n']} · backend={main_a_backend}")
    if stats["missing_img_a"]:
        print(f"⚠ arm A 缺图: {stats['missing_img_a']}")
    if stats["missing_img_b"]:
        print(f"⚠ arm B 缺图: {stats['missing_img_b']}")


if __name__ == "__main__":
    main()
