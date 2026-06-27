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
import json
import time
from typing import Any, Dict, List

import config
from eval.dataset import get_benchmark, BenchInput
from eval.metrics import summarize
from eval.report import save_artifacts, table, fmt_num


def _count_attack_succeeded(state) -> int:
    """从 trace 数 "攻擂成功" 次数（擂台进化里被采纳的轮次）。"""
    return sum(1 for s in state.trace if "攻擂成功" in s.action)


def _count_evo_rounds(state) -> int:
    """擂台进化的总轮次（含失败守擂）。"""
    return sum(1 for s in state.trace if "擂台" in s.phase and "第" in s.action and "轮" in s.action)


def _run_one_delta(delta: float, inputs: List[BenchInput], args) -> Dict[str, Any]:
    """在固定 delta 下跑 inputs 个 autonomous fixed loop，返回聚合统计。"""
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
            try:
                for s in autonomous_full_run(agent, state, config=ac):
                    final = s
            except Exception as e:
                print(f"      ⚠ 异常：{e}")
                rows.append({"theme": item.user_input, "error": str(e)})
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

    t_all = time.time()
    results = []
    for delta in args.deltas:
        print(f"\n=== sweep delta = {delta:.2f} ===")
        r = _run_one_delta(delta, inputs, args)
        results.append(r)

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
