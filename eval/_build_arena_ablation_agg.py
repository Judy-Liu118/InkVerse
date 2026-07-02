"""
eval._build_arena_ablation_agg -- 拼 arm A / arm B 对比用的 agg JSON

结构与 _agg_3deltas.json 兼容（theme -> {arm: {image_path, poem, ...}}），
所以可直接喂给 eval.vlm_hard_constraint 出双 arm 硬约束对比。

arm_A = 无擂台（2026-07-01 新跑的 sweep_pairwise_win_delta_20260701_230222.json）
arm_B = 有擂台，δ=0.17 baseline（复用 sweep_pairwise_win_delta_20260630_134045.json 的图目录 + 已聚合的 _agg_3deltas.json 里 "0.17" 那份）
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "eval"

ARM_A_JSON = OUTPUT / "sweep_pairwise_win_delta_20260701_230222.json"
ARM_A_IMG_DIR = OUTPUT / "sweep_pairwise_win_delta_images_20260701_223349" / "delta_0.17"
AGG_3DELTAS = OUTPUT / "_agg_3deltas.json"          # 已存在，含 arm B 0.17 数据
OUT_JSON = OUTPUT / "_agg_arena_ablation.json"


def _pick_terminal_image(theme_idx_1based: int, arm_a_row: Dict[str, Any]) -> Optional[Path]:
    """按 sweep 图目录里 '{theme_idx:02d}_...' 前缀过滤出该主题所有 gen，
    再挑 clip 分最高的一张作为"终图"（与 autonomous_full_run 的
    "返回历史最优" 语义一致）。"""
    prefix = f"{theme_idx_1based:02d}_"
    candidates: List[tuple] = []
    for p in ARM_A_IMG_DIR.glob(f"{prefix}*.png"):
        m = re.search(r"_clip([0-9.]+)\.png$", p.name)
        if not m:
            continue
        try:
            clip = float(m.group(1))
        except ValueError:
            continue
        candidates.append((clip, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1].name))  # 分高优先，同分取名字后
    return candidates[-1][1]


def main() -> None:
    if not ARM_A_JSON.is_file():
        raise SystemExit(f"arm A JSON 不存在: {ARM_A_JSON}")
    if not AGG_3DELTAS.is_file():
        raise SystemExit(f"arm B agg 不存在: {AGG_3DELTAS}")

    arm_a_data = json.loads(ARM_A_JSON.read_text(encoding="utf-8"))
    arm_b_agg = json.loads(AGG_3DELTAS.read_text(encoding="utf-8"))

    arm_a_rows = arm_a_data["results"][0]["rows"]
    themes_a: List[str] = [r["theme"] for r in arm_a_rows]

    # arm B 主题顺序（应与 arm A 一致，因为都是 get_benchmark(n=10)）
    themes_b = list(arm_b_agg.keys())
    if themes_a != themes_b:
        print(f"⚠ 主题顺序不一致，逐一 lookup by user_input")

    out: Dict[str, Dict[str, Any]] = {}
    missing_image = []
    for i, r in enumerate(arm_a_rows, start=1):
        theme = r["theme"]
        arm_b_cell = arm_b_agg.get(theme, {}).get("0.17")
        if arm_b_cell is None:
            print(f"⚠ 主题 {theme} 在 arm B (0.17) 里找不到")

        img = _pick_terminal_image(i, r)
        if img is None:
            missing_image.append((i, theme))
            arm_a_image_path = None
            arm_a_image_paths_all = []
        else:
            arm_a_image_path = str(img.relative_to(ROOT)).replace("\\", "/")
            arm_a_image_paths_all = [
                str(p.relative_to(ROOT)).replace("\\", "/")
                for p in sorted(ARM_A_IMG_DIR.glob(f"{i:02d}_*.png"))
            ]

        out[theme] = {
            "arm_A": {
                "n": 1,
                "clip_raw": r.get("clip_raw"),
                "attack_succeed": r.get("attack_succeed", 0),
                "evo_rounds": r.get("evo_rounds", 0),
                "poem": r.get("poem", ""),
                "title": r.get("title", ""),
                "image_prompt": r.get("image_prompt", ""),
                "image_path": arm_a_image_path,
                "image_paths_all": arm_a_image_paths_all,
                "image_count": len(arm_a_image_paths_all),
                "poem_evolution": r.get("poem_evolution", []),
            },
            "arm_B": arm_b_cell or {},
        }

    OUT_JSON.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"写入 {OUT_JSON} · {len(out)} 主题")
    if missing_image:
        print(f"⚠ 缺图主题: {missing_image}")


if __name__ == "__main__":
    main()
