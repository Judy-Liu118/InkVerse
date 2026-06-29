"""
eval.sweep_pairwise_win_delta -- PAIRWISE_WIN_DELTA 阈值 sweep

`PAIRWISE_WIN_DELTA` (config.py:199) 控制擂台进化里"挑战者 pairwise 赢"的加分。
默认 0.17，是拍脑袋值。本脚本在多个候选值上各跑一遍 eval_autonomous（fixed
loop 路径走擂台进化），出对比表看哪个值最稳定。

判定指标:
  · 攻擂成功率：擂台进化里挑战者最终被采纳的轮次占比（高 = 更激进改诗）
  · CLIP 终值：最终 raw CLIP（mean ± std）
  · 改诗轮次：实际触发改诗的次数（不一定攻擂成功）
  · 改图轮次：autonomous fixed loop 的改图次数

跑法：
    # 主跑（推荐：[0.10, 0.15, 0.17, 0.20] × n=10，~2hr）
    python -m eval.sweep_pairwise_win_delta --n 10

    # 自定义 deltas
    python -m eval.sweep_pairwise_win_delta --deltas 0.10 0.15 0.20 --n 5

    # 烟测拓扑不调 API（用 monkey-patch + fake run）
    python -m eval.sweep_pairwise_win_delta --dry-run

⚠️ 真跑成本：4 × n autonomous 全跑，每次 ~2-3 min API + 显存。
    n=10 × 4 阈值 ≈ 80-120min。
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from eval.dataset import get_benchmark, BenchInput
from eval.metrics import summarize
from eval.report import save_artifacts, table, fmt_num, OUTPUT_DIR


_INVALID_FNAME_CHARS = re.compile(r'[\\/:*?"<>|\s]+')


def _sanitize_for_filename(s: str, max_len: int = 24) -> str:
    """Windows 文件名安全化：去掉非法字符 + 空白，截短。"""
    s = _INVALID_FNAME_CHARS.sub("_", (s or "").strip())
    s = s.strip("._") or "untitled"
    return s[:max_len]


def _save_sweep_image(image, save_dir: Path, theme_idx: int,
                      first_line: str, gen_idx: int, clip_raw: float) -> None:
    """保存一张生成图到 sweep 图片目录。

    文件名规则（按用户要求）：诗第一句 + 第几次生成 + CLIP 评分
    实际格式：{theme_idx:02d}_{第一句}_gen{gen_idx}_clip{clip:.3f}.png
    theme_idx 前缀只是为了文件管理器里按主题聚合排序，不影响信息。
    """
    if image is None:
        return
    fname = (
        f"{theme_idx:02d}_"
        f"{_sanitize_for_filename(first_line)}_"
        f"gen{gen_idx}_clip{clip_raw:.3f}.png"
    )
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        image.save(save_dir / fname, format="PNG")
    except Exception as e:
        print(f"      ⚠ 图像保存失败 {fname}: {e}")


def _cuda_cleanup() -> None:
    """每条 autonomous 之间强制 GC + CUDA 显存碎片整理。

    8GB GPU 上 LoRA + Z-Image diffusers + CLIP 反复加载/卸载会让显存碎片化，
    跑长 sweep 必撞 OOM（2026-06-29 在 delta=0.10 第 9 条触发，后续 22 条
    全级联失败）。每条之间 best-effort 清一次：不能解所有问题（unsloth 4bit
    权重的内部缓存 empty_cache 也释放不掉），但能拖延 OOM、避免 1 次 OOM
    污染整个 sweep。
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _dump_recovery(args, results: List[Dict[str, Any]]) -> None:
    """每个 delta 跑完后写一份 recovery JSON 到固定路径。

    sweep 单次跑 6hr+ 时任何崩溃都会全丢数据；这个文件让你至少能拿到已完成
    delta 的结果。固定路径覆盖写，每次跑一遍 sweep 都会被新数据覆盖。
    """
    path = OUTPUT_DIR / "sweep_pairwise_win_delta_RECOVERY.json"
    try:
        path.write_text(
            json.dumps(
                {"config": vars(args), "results": results,
                 "note": "partial recovery snapshot, overwritten after each delta"},
                ensure_ascii=False, indent=2, default=str,
            ),
            encoding="utf-8",
        )
        print(f"      [recovery] 已写 {path}")
    except Exception as e:
        print(f"      ⚠ recovery dump 失败：{e}")


def _count_attack_succeeded(state) -> int:
    """从 trace 数 "攻擂成功" 次数（擂台进化里被采纳的轮次）。"""
    return sum(1 for s in state.trace if "攻擂成功" in s.action)


def _count_evo_rounds(state) -> int:
    """擂台进化的总轮次（含失败守擂）。"""
    return sum(1 for s in state.trace if "擂台" in s.phase and "第" in s.action and "轮" in s.action)


def _run_one_delta(delta: float, inputs: List[BenchInput], args,
                   images_dir: Optional[Path] = None) -> Dict[str, Any]:
    """在固定 delta 下跑 inputs 个 autonomous fixed loop，返回聚合统计。

    如果 images_dir 不为 None，每条 autonomous 跑出来的所有图（初次 + 每轮
    改图）都按"诗第一句_gen{N}_clip{score}.png"命名落到该目录。
    """
    # 关键：monkey-patch config.PAIRWISE_WIN_DELTA，让生产代码读到当前 sweep 值
    original = config.PAIRWISE_WIN_DELTA
    config.PAIRWISE_WIN_DELTA = delta
    try:
        # delayed import：让 autonomous.py 顶层 from config import ... 不影响（实际它
        # 在函数内 import，每次都拿当前值）
        from eval.eval_autonomous import (
            _build_agent, _new_state, _raw_clip,
        )
        from core.agent.autonomous import autonomous_full_run, AutonomousConfig

        agent = _build_agent(args)
        rows = []
        t0 = time.time()
        for i, item in enumerate(inputs):
            print(f"  [delta={delta:.2f}] [{i+1}/{len(inputs)}] {item.user_input[:30]}…")
            state = _new_state(item, args)
            ac = AutonomousConfig(
                target_clip_score=args.target,
                max_image_improve_rounds=args.max_img_rounds,
                allow_poem_refine=True,
                max_poem_refine_rounds=args.max_poem_rounds,
                image_improve_mode="rewrite_regen",
                image_loop_llm_driven=False,   # sweep 只测 fixed loop
            )
            final = None
            gen_idx = 0          # 该 autonomous 内已存图序号
            prev_image_id = None  # 跟踪 state.image 对象 id，变化才视作"新图"
            try:
                try:
                    for s in autonomous_full_run(agent, state, config=ac):
                        final = s
                        # 检测到新图：id 变化 = 这次 yield 来自一次新的生图
                        if images_dir is not None and s.image is not None:
                            cur_id = id(s.image)
                            if cur_id != prev_image_id:
                                gen_idx += 1
                                first_line = ""
                                if s.poem:
                                    first_line = s.poem.split("\n")[0].strip()
                                # 用 delta 子目录组织，3 个 sweep run 不会冲突
                                delta_dir = images_dir / f"delta_{delta:.2f}"
                                _save_sweep_image(
                                    s.image, delta_dir, i + 1,
                                    first_line, gen_idx, _raw_clip(s),
                                )
                                prev_image_id = cur_id
                except Exception as e:
                    # 关键：完整 traceback 入库，OOM 这类问题靠 str(e) 看不清
                    err_msg = str(e)[:200]
                    err_type = type(e).__name__
                    print(f"      ⚠ 异常 [{err_type}]：{err_msg}")
                    traceback.print_exc()
                    rows.append({
                        "theme":     item.user_input,
                        "error":     f"{err_type}: {err_msg}",
                        "traceback": traceback.format_exc()[:2000],
                    })
                    continue
                if final is None:
                    rows.append({"theme": item.user_input, "error": "no final"})
                    continue
                attacks = _count_attack_succeeded(final)
                evo_rounds = _count_evo_rounds(final)
                rows.append({
                    "theme":          item.user_input,
                    "clip_raw":       _raw_clip(final),
                    "attack_succeed": attacks,
                    "evo_rounds":     evo_rounds,
                    "attack_rate":    (attacks / evo_rounds) if evo_rounds else 0.0,
                    "elapsed_sec":    round(time.time() - t0, 2),
                })
                print(f"      CLIP={fmt_num(rows[-1]['clip_raw'])}, "
                      f"攻擂 {attacks}/{evo_rounds}")
            finally:
                # 每条之间无条件 cleanup（成功或失败都清），避免单次 OOM 级联污染
                # 后续所有 condition。释放 state 引用 + GC + empty_cache。
                final = None
                state = None
                _cuda_cleanup()
    finally:
        config.PAIRWISE_WIN_DELTA = original

    valid = [r for r in rows if "error" not in r]
    clip_stat = summarize([r["clip_raw"] for r in valid]) if valid else {"n": 0}
    attack_stat = summarize([r["attack_succeed"] for r in valid]) if valid else {"n": 0}
    rate_stat = summarize([r["attack_rate"] for r in valid]) if valid else {"n": 0}
    return {
        "delta":         delta,
        "n_run":         len(rows),
        "n_valid":       len(valid),
        "clip_stat":     clip_stat,
        "attack_stat":   attack_stat,
        "rate_stat":     rate_stat,
        "rows":          rows,
    }


def _render_report(args, results: List[Dict[str, Any]]) -> str:
    md = []
    md.append("# PAIRWISE_WIN_DELTA sweep 报告")
    md.append(f"_n={args.n} 每阈值 · deltas={args.deltas} · "
              f"max_poem_rounds={args.max_poem_rounds}_")
    md.append("")
    md.append("## 1. 跨 delta 主表")
    rows_md = []
    for r in results:
        if r["n_valid"] == 0:
            rows_md.append([f"{r['delta']:.2f}", 0, "—", "—", "—", "—"])
            continue
        cs = r["clip_stat"]
        ats = r["attack_stat"]
        rs = r["rate_stat"]
        rows_md.append([
            f"{r['delta']:.2f}",
            r["n_valid"],
            f"{fmt_num(cs['mean'])} ± {fmt_num(cs['std'])}",
            f"{fmt_num(ats['mean'], 2)}",
            f"{fmt_num(rs['mean'] * 100, 1)}%",
            f"{fmt_num(cs['std'])}",
        ])
    md.append(table(
        ["delta", "n", "CLIP raw (mean ± std)", "平均攻擂成功次数",
         "平均攻擂率", "CLIP std (稳定性)"],
        rows_md,
    ))
    md.append("")
    md.append(
        "**解读规则**：CLIP std 低 = 该 delta 下 fixed loop 结果跨主题更稳；"
        "攻擂率太高（>50%）说明 delta 过宽容（挑战者随便就攻擂），"
        "太低（<10%）说明 delta 过严格（永远守擂、改诗白做）。"
        "推荐：在 CLIP std 最低的同时攻擂率落在 15-40% 区间的 delta。"
    )
    md.append("")
    md.append("## 2. 各 delta 详情")
    for r in results:
        md.append(f"### delta = {r['delta']:.2f}")
        for row in r["rows"]:
            if "error" in row:
                md.append(f"- ⚠ {row['theme']}: {row['error']}")
                continue
            md.append(f"- {row['theme']}: CLIP={fmt_num(row['clip_raw'])}, "
                      f"攻擂 {row['attack_succeed']}/{row['evo_rounds']}")
        md.append("")
    return "\n".join(md)


def main():
    p = argparse.ArgumentParser(
        description="PAIRWISE_WIN_DELTA sweep（擂台进化阈值调参）",
    )
    p.add_argument("--n", type=int, default=10,
                   help="每个 delta 跑多少主题")
    p.add_argument("--deltas", type=float, nargs="*",
                   default=[0.10, 0.15, 0.17, 0.20])
    p.add_argument("--poem-model",   default="local_lora")
    p.add_argument("--prompt-model", default="qwen-max")
    p.add_argument("--scorer",       default="qwen-plus")
    p.add_argument("--image-backend", default="bailian:qwen-image-2.0-pro")
    p.add_argument("--target",       type=float, default=0.30)
    p.add_argument("--max-img-rounds",  type=int, default=2)
    p.add_argument("--max-poem-rounds", type=int, default=2,
                   help="擂台进化轮次（要 >0 才有攻擂数据）")
    p.add_argument("--dry-run", action="store_true",
                   help="不调真 API；monkey-patch 验证 + 出空报告")
    p.add_argument("--no-save-images", action="store_true",
                   help="不保存生成的图（默认会按 主题_第几次_clip 命名落到 "
                        "outputs/eval/sweep_pairwise_win_delta_images_<ts>/）")
    args = p.parse_args()

    if args.dry_run:
        # 仅验证 monkey-patch 逻辑 + 报告渲染拓扑
        print("[sweep] dry-run：验证 monkey-patch 不留状态、报告渲染不抛错")
        from core.agent import autonomous as _auton_mod  # noqa
        orig = config.PAIRWISE_WIN_DELTA
        for d in args.deltas:
            config.PAIRWISE_WIN_DELTA = d
            assert config.PAIRWISE_WIN_DELTA == d
        config.PAIRWISE_WIN_DELTA = orig
        # 合成空 results 触发报告渲染
        synthetic = [{"delta": d, "n_run": 0, "n_valid": 0,
                      "clip_stat": {}, "attack_stat": {}, "rate_stat": {},
                      "rows": []} for d in args.deltas]
        md = _render_report(args, synthetic)
        print(md[:400])
        print(f"\n[sweep] dry-run OK，恢复 PAIRWISE_WIN_DELTA={config.PAIRWISE_WIN_DELTA}")
        return

    inputs = get_benchmark(n=args.n)
    print(f"[sweep] 在 deltas={args.deltas} 上各跑 n={len(inputs)} · "
          f"image={args.image_backend}")
    print(f"[sweep] sweep 期间 PAIRWISE_WIN_DELTA 会被 monkey-patch，结束后恢复")

    # 图像落盘目录（每次 sweep run 一个独立时间戳目录，分 delta 子目录组织）
    images_dir: Optional[Path] = None
    if not args.no_save_images:
        ts_dir = time.strftime("%Y%m%d_%H%M%S")
        images_dir = OUTPUT_DIR / f"sweep_pairwise_win_delta_images_{ts_dir}"
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"[sweep] 图像落盘到: {images_dir}")

    t_all = time.time()
    results = []
    for delta in args.deltas:
        print(f"\n=== sweep delta = {delta:.2f} ===")
        r = _run_one_delta(delta, inputs, args, images_dir=images_dir)
        results.append(r)
        # 每个 delta 跑完落一份 recovery snapshot，崩了至少能拿到已完成 delta
        _dump_recovery(args, results)

    elapsed_all = time.time() - t_all
    print(f"\n[sweep] 总耗时 {elapsed_all/60:.1f} min")

    md = _render_report(args, results)
    summary = {"config": vars(args), "results": results,
               "total_elapsed_sec": round(elapsed_all, 1)}
    paths = save_artifacts("sweep_pairwise_win_delta", summary, md)
    print(f"原始数据: {paths['json']}")
    print(f"Markdown: {paths['md']}")
    print(md[:2000])


if __name__ == "__main__":
    main()
