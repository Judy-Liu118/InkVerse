"""
eval.eval_autonomous -- 全自主模式 vs 单轮模式（CLIP 终值 + 耗时对比）

跑法：
    python -m eval.eval_autonomous --n 5
    python -m eval.eval_autonomous --n 5 --max-img-rounds 2 --max-poem-rounds 1

输出：
    每条 user_input 跑两次：
      · single_pass：执行一次完整流水线 plan→poem→prompt→image→CLIP，无改图无改诗
      · autonomous ：autonomous_full_run（Arena 海选 + 守擂进化 + CLIP 改图循环）
    比较：
      · 最终 CLIP raw 分（核心）
      · 端到端耗时（成本）
      · 改图 / 改诗轮次（说明自主策略的开销）

⚠️ 跑这个评估开销最大：每条 input 跑两次完整流水线含生图。建议 --n 5 起步。
   需要 DASHSCOPE_API_KEY + 图像后端 + CLIP 权重。
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List

from core.models.adapter import ModelAdapter
from core.agent.agent import PoetryAgent
from core.agent.autonomous import autonomous_full_run, AutonomousConfig
from core.agent.state import AgentState, Phase
from config import DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, STYLE_MAP

from eval.dataset import get_benchmark, BenchInput
from eval.metrics import summarize, paired_delta
from eval.report import save_artifacts, table, fmt_num, print_and_return


def _make_adapter(model_choice: str, allow_lora_fallback: bool = False) -> ModelAdapter:
    if model_choice == "local_lora":
        return ModelAdapter(backend="local_lora", allow_lora_fallback=allow_lora_fallback)
    if model_choice == "local_base":
        return ModelAdapter(backend="local", allow_lora_fallback=allow_lora_fallback)
    if model_choice.startswith("deepseek"):
        return ModelAdapter(backend="deepseek", api_key=DEEPSEEK_API_KEY, api_model=model_choice)
    return ModelAdapter(backend="qwen", api_key=DASHSCOPE_API_KEY, api_model=model_choice)


def _parse_backend(val: str):
    if val.startswith("bailian:"):
        return "bailian", val.split(":", 1)[1]
    return "local", None


def _build_agent(args) -> PoetryAgent:
    return PoetryAgent(
        generation_adapter=_make_adapter(args.poem_model, allow_lora_fallback=True),
        score_adapter=_make_adapter(args.scorer),
        title_adapter=_make_adapter(args.scorer),
        prompt_adapter=_make_adapter(args.prompt_model),
    )


def _new_state(item: BenchInput, args) -> AgentState:
    img_backend, img_api_model = _parse_backend(args.image_backend)
    return AgentState(
        user_input=item.user_input,
        lang="英文",
        style_suffix=STYLE_MAP["水墨画"],
        image_backend=img_backend,
        image_api_key=DASHSCOPE_API_KEY if img_backend == "bailian" else None,
        image_api_model=img_api_model,
    )


def _raw_clip(state: AgentState) -> float:
    """norm → raw（state.clip_score_final 是归一化分）。"""
    return state.clip_score_final * 2 - 1 if state.clip_score_final else 0.0


def _count_rounds(state: AgentState) -> Dict[str, int]:
    img_rounds = sum(1 for s in state.trace
                     if ("改图循环：第" in s.action and "完成" in s.action)
                     or ("图像优化：第" in s.action and "完成" in s.action))
    poem_rounds = sum(1 for s in state.trace
                      if ("品质改诗第" in s.action and "完成" in s.action)
                      or ("自主改诗：第" in s.action and "完成" in s.action)
                      or ("第" in s.action and "攻擂成功" in s.action))
    return {"image_rounds": img_rounds, "poem_rounds": poem_rounds}


def _run_single_pass(agent: PoetryAgent, item: BenchInput, args) -> Dict[str, Any]:
    state = _new_state(item, args)
    t0 = time.time()
    state = agent._phase_plan(state)
    state = agent._phase_poem(state)
    if state.phase == Phase.ERROR:
        return {"mode": "single_pass", "error": state.error}
    state = agent._phase_keyword_extract(state)
    state = agent._phase_title(state)
    state = agent._phase_prompt(state)
    state = agent._phase_prompt_review(state)
    state = agent._phase_image_clip(state)
    elapsed = time.time() - t0
    return {
        "mode":         "single_pass",
        "clip_raw":     _raw_clip(state),
        "poem":         state.poem,
        "elapsed_sec":  round(elapsed, 2),
        "rounds":       {"image_rounds": 0, "poem_rounds": 0},
    }


def _run_autonomous(agent: PoetryAgent, item: BenchInput, args) -> Dict[str, Any]:
    state = _new_state(item, args)
    config = AutonomousConfig(
        target_clip_score=args.target,
        max_image_improve_rounds=args.max_img_rounds,
        allow_poem_refine=args.max_poem_rounds > 0,
        max_poem_refine_rounds=args.max_poem_rounds,
        image_improve_mode="rewrite_regen",
    )
    t0 = time.time()
    final = None
    try:
        for s in autonomous_full_run(agent, state, config=config):
            final = s
    except Exception as e:
        return {"mode": "autonomous", "error": str(e)}
    if final is None:
        return {"mode": "autonomous", "error": "no final state"}
    elapsed = time.time() - t0
    rounds = _count_rounds(final)
    return {
        "mode":         "autonomous",
        "clip_raw":     _raw_clip(final),
        "poem":         final.poem,
        "elapsed_sec":  round(elapsed, 2),
        "rounds":       rounds,
    }


def _run_one(agent: PoetryAgent, item: BenchInput, args) -> Dict[str, Any]:
    sp = _run_single_pass(agent, item, args)
    au = _run_autonomous(agent, item, args)
    return {
        "user_input": item.user_input,
        "genre": item.genre, "theme": item.theme,
        "single_pass": sp,
        "autonomous":  au,
    }


def _render_report(args, rows):
    md = []
    md.append("# eval_autonomous 报告 · 全自主 vs 单轮")
    md.append(f"_n={len(rows)} · 目标 CLIP raw={args.target} · "
              f"max 改图={args.max_img_rounds} · max 改诗={args.max_poem_rounds}_")
    md.append("")

    sp_scores = [r["single_pass"].get("clip_raw") for r in rows if "error" not in r["single_pass"]]
    au_scores = [r["autonomous"].get("clip_raw")  for r in rows if "error" not in r["autonomous"]]
    sp_times  = [r["single_pass"].get("elapsed_sec") for r in rows if "error" not in r["single_pass"]]
    au_times  = [r["autonomous"].get("elapsed_sec")  for r in rows if "error" not in r["autonomous"]]
    img_rounds = [r["autonomous"]["rounds"]["image_rounds"]
                  for r in rows if "error" not in r["autonomous"]]
    poem_rounds = [r["autonomous"]["rounds"]["poem_rounds"]
                   for r in rows if "error" not in r["autonomous"]]

    if not sp_scores or not au_scores:
        return "\n".join(md + ["无可用数据。"])

    sp_stat = summarize(sp_scores)
    au_stat = summarize(au_scores)

    md.append("## 1. CLIP raw 终值对比（核心）")
    md.append(table(
        ["模式", "n", "mean", "std", "median", "min", "max"],
        [
            ["single_pass", sp_stat["n"], fmt_num(sp_stat["mean"]), fmt_num(sp_stat["std"]),
             fmt_num(sp_stat["median"]), fmt_num(sp_stat["min"]), fmt_num(sp_stat["max"])],
            ["autonomous",  au_stat["n"], fmt_num(au_stat["mean"]), fmt_num(au_stat["std"]),
             fmt_num(au_stat["median"]), fmt_num(au_stat["min"]), fmt_num(au_stat["max"])],
        ],
    ))
    md.append("")

    md.append("## 2. 配对差值：autonomous − single_pass")
    paired_pairs = [
        (r["single_pass"]["clip_raw"], r["autonomous"]["clip_raw"])
        for r in rows
        if "error" not in r["single_pass"] and "error" not in r["autonomous"]
    ]
    if paired_pairs:
        a = [p[0] for p in paired_pairs]
        b = [p[1] for p in paired_pairs]
        delta = paired_delta(a, b)
        md.append(table(
            ["指标", "值"],
            [
                ["样本对数 n", delta["n"]],
                ["mean Δ",     fmt_num(delta["mean_delta"], 4)],
                ["median Δ",   fmt_num(delta["median_delta"], 4)],
                ["autonomous 提升比例", f"{delta['positive_rate']:.1%}"],
            ],
        ))
    md.append("")

    md.append("## 3. 成本：耗时 + 自主轮次")
    sp_t = summarize(sp_times)
    au_t = summarize(au_times)
    md.append(table(
        ["指标", "single_pass", "autonomous", "倍率"],
        [
            ["平均耗时 (s)", fmt_num(sp_t["mean"], 1), fmt_num(au_t["mean"], 1),
             fmt_num(au_t["mean"] / max(sp_t["mean"], 1e-6), 2) + "×"],
            ["中位耗时 (s)", fmt_num(sp_t["median"], 1), fmt_num(au_t["median"], 1), "—"],
            ["平均改图轮次", "—", fmt_num(summarize(img_rounds)["mean"], 1), "—"],
            ["平均改诗轮次", "—", fmt_num(summarize(poem_rounds)["mean"], 1), "—"],
        ],
    ))
    md.append("")

    md.append("## 4. 抽样")
    for r in rows[:5]:
        sp = r["single_pass"]; au = r["autonomous"]
        if "error" in sp or "error" in au:
            continue
        md.append(f"### {r['user_input']}")
        md.append(f"- single_pass: CLIP raw={fmt_num(sp['clip_raw'])}, "
                  f"{sp['elapsed_sec']}s")
        md.append(f"  ```\n  {sp['poem'].replace(chr(10), chr(10) + '  ')}\n  ```")
        md.append(f"- autonomous: CLIP raw={fmt_num(au['clip_raw'])}, "
                  f"{au['elapsed_sec']}s, 改图 {au['rounds']['image_rounds']} 轮, "
                  f"改诗 {au['rounds']['poem_rounds']} 轮")
        md.append(f"  ```\n  {au['poem'].replace(chr(10), chr(10) + '  ')}\n  ```")
        md.append("")
    return "\n".join(md)


def main():
    p = argparse.ArgumentParser(description="全自主 vs 单轮模式 CLIP 终值 + 耗时对比")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--genres", nargs="*", default=None)
    p.add_argument("--density", choices=["rich", "sparse"], default=None)
    p.add_argument("--poem-model",   default="local_lora")
    p.add_argument("--prompt-model", default="qwen-max")
    p.add_argument("--scorer",       default="qwen-plus")
    p.add_argument("--image-backend", default="bailian:qwen-image-2.0-pro")
    p.add_argument("--target",       type=float, default=0.30,
                   help="autonomous 模式的 CLIP 目标分（提前停止阈值）")
    p.add_argument("--max-img-rounds",  type=int, default=2)
    p.add_argument("--max-poem-rounds", type=int, default=1)
    args = p.parse_args()

    inputs = get_benchmark(n=args.n, genres=args.genres, density=args.density)
    print(f"[eval_autonomous] 跑 {len(inputs)} 条 × 2 模式 · image={args.image_backend}")
    agent = _build_agent(args)

    rows = []
    for i, item in enumerate(inputs):
        print(f"  [{i+1}/{len(inputs)}] {item.user_input[:30]}…")
        try:
            r = _run_one(agent, item, args)
            rows.append(r)
            sp = r["single_pass"]; au = r["autonomous"]
            if "error" in sp:
                print(f"      single_pass ⚠ {sp['error']}")
            else:
                print(f"      single_pass  CLIP={fmt_num(sp['clip_raw'])}, {sp['elapsed_sec']}s")
            if "error" in au:
                print(f"      autonomous  ⚠ {au['error']}")
            else:
                print(f"      autonomous   CLIP={fmt_num(au['clip_raw'])}, {au['elapsed_sec']}s, "
                      f"改图 {au['rounds']['image_rounds']} 轮, 改诗 {au['rounds']['poem_rounds']} 轮")
        except Exception as e:
            print(f"      ⚠ 异常：{e}")
            rows.append({"user_input": item.user_input, "error": str(e)})

    md = _render_report(args, [r for r in rows if "user_input" in r and "single_pass" in r])
    print_and_return(md)
    paths = save_artifacts("eval_autonomous", {"config": vars(args), "rows": rows}, md)
    print(f"原始数据: {paths['json']}")
    print(f"Markdown: {paths['md']}")


if __name__ == "__main__":
    main()
