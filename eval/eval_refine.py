"""
eval.eval_refine -- 改诗前后总分对比（守擂进化的核心价值）

跑法：
    python -m eval.eval_refine --n 10
    python -m eval.eval_refine --n 10 --model qwen-plus

输出：
    每条 user_input：
      1. 用 API 模型生成 baseline 诗 + 打分
      2. 自动生成方向性诗评（_auto_poem_critique → _auto_poem_feedback）
      3. 跑 refine_poem 一轮 → 重新打分 refined
    比较 baseline vs refined 的各维度分数提升、提升率。

⚠️ 跑这个评估需要 DASHSCOPE_API_KEY；refine_poem 不支持本地 LoRA。
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List

from core.models.adapter import ModelAdapter
from core.agent.agent import PoetryAgent
from core.agent.state import AgentState, Phase
from config import DASHSCOPE_API_KEY, DEEPSEEK_API_KEY

from eval.dataset import get_benchmark, BenchInput
from eval.metrics import summarize, paired_delta
from eval.report import save_artifacts, table, fmt_num, print_and_return


def _make_adapter(model_choice: str) -> ModelAdapter:
    if model_choice == "local_lora":
        return ModelAdapter(backend="local_lora")
    if model_choice == "local_base":
        return ModelAdapter(backend="local")
    if model_choice.startswith("deepseek"):
        return ModelAdapter(backend="deepseek", api_key=DEEPSEEK_API_KEY, api_model=model_choice)
    return ModelAdapter(backend="qwen", api_key=DASHSCOPE_API_KEY, api_model=model_choice)


def _run_one(agent: PoetryAgent, item: BenchInput, args) -> Dict[str, Any]:
    state = AgentState(user_input=item.user_input)
    t0 = time.time()

    state = agent._phase_plan(state)
    state = agent._phase_poem(state)
    if state.phase == Phase.ERROR:
        return {"user_input": item.user_input, "error": state.error}

    baseline_poem = state.poem
    baseline_scores = agent.poem_gen.score_single_poem(
        baseline_poem, item.user_input, agent.score_adapter,
    )

    # 自动生成方向性诗评
    refine_adapter = _make_adapter(args.refine_model)
    critique = agent._auto_poem_critique(state)
    feedback = agent._auto_poem_feedback(state, critique=critique)

    state = agent.refine_poem(state, feedback, refine_adapter=refine_adapter)
    refined_poem = state.poem
    refined_scores = agent.poem_gen.score_single_poem(
        refined_poem, item.user_input, agent.score_adapter,
    )

    elapsed = time.time() - t0
    return {
        "user_input":      item.user_input,
        "genre":           item.genre,
        "theme":           item.theme,
        "baseline_poem":   baseline_poem,
        "refined_poem":    refined_poem,
        "feedback":        feedback,
        "critique":        critique[:200],
        "baseline_scores": baseline_scores,
        "refined_scores":  refined_scores,
        "changed":         baseline_poem != refined_poem,
        "elapsed_sec":     round(elapsed, 2),
    }


def _render_report(args, rows):
    md = []
    md.append("# eval_refine 报告 · baseline vs refined")
    md.append(f"_n={len(rows)} · poem_model={args.model} · refine_model={args.refine_model}_")
    md.append("")

    ok_rows = [r for r in rows if "error" not in r]
    changed_rows = [r for r in ok_rows if r.get("changed")]
    md.append(f"成功 {len(ok_rows)}/{len(rows)}，其中实际改动 {len(changed_rows)} 条 "
              f"（{len(changed_rows) / max(len(ok_rows), 1):.1%}）。")
    md.append("")

    if not ok_rows:
        return "\n".join(md + ["无可用数据。"])

    md.append("## 1. 全样本均值对比")
    dims = ("total", "intent", "pingze", "rhyme", "imagery", "cohesion")
    rows_md = []
    for d in dims:
        a = summarize([r["baseline_scores"].get(d, 0.0) for r in ok_rows])
        b = summarize([r["refined_scores"].get(d, 0.0)  for r in ok_rows])
        rows_md.append([d, fmt_num(a["mean"]), fmt_num(b["mean"]),
                        fmt_num(b["mean"] - a["mean"])])
    md.append(table(["维度", "baseline", "refined", "Δ"], rows_md))
    md.append("")

    md.append("## 2. 配对差值与提升率")
    rows_md = []
    for d in dims:
        delta = paired_delta(
            [r["baseline_scores"].get(d, 0.0) for r in ok_rows],
            [r["refined_scores"].get(d, 0.0)  for r in ok_rows],
        )
        rows_md.append([d, fmt_num(delta["mean_delta"]),
                        fmt_num(delta["median_delta"]),
                        f"{delta['positive_rate']:.1%}"])
    md.append(table(["维度", "mean Δ", "median Δ", "refined 提升比例"], rows_md))
    md.append("")

    md.append("## 3. 仅看实际改动样本（更能反映改诗本身的效果）")
    if changed_rows:
        rows_md = []
        for d in dims:
            delta = paired_delta(
                [r["baseline_scores"].get(d, 0.0) for r in changed_rows],
                [r["refined_scores"].get(d, 0.0)  for r in changed_rows],
            )
            rows_md.append([d, fmt_num(delta["mean_delta"]),
                            f"{delta['positive_rate']:.1%}"])
        md.append(table(["维度", "mean Δ", "refined 提升比例"], rows_md))
    else:
        md.append("所有样本均未被改动。可能原因：改后字数不符 / 评分回滚 / 模型保守。")
    md.append("")

    md.append("## 4. 抽样改诗对比")
    for r in ok_rows[:5]:
        md.append(f"### {r['user_input']}")
        md.append(f"- 方向：{r['feedback']}")
        md.append(f"- baseline (total={fmt_num(r['baseline_scores'].get('total', 0))})")
        md.append("  ```\n  " + r["baseline_poem"].replace("\n", "\n  ") + "\n  ```")
        md.append(f"- refined (total={fmt_num(r['refined_scores'].get('total', 0))})  "
                  + ("✓ 改动" if r.get("changed") else "— 未变"))
        md.append("  ```\n  " + r["refined_poem"].replace("\n", "\n  ") + "\n  ```")
        md.append("")

    return "\n".join(md)


def main():
    p = argparse.ArgumentParser(description="改诗前后总分对比")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--genres", nargs="*", default=None)
    p.add_argument("--density", choices=["rich", "sparse"], default=None)
    p.add_argument("--model", default="qwen-plus",
                   help="baseline 诗歌生成模型；建议用 API 模型保证 baseline 可被改诗")
    p.add_argument("--refine-model", default="qwen-plus",
                   help="改诗模型；LoRA 不支持改诗，必须为 API")
    p.add_argument("--scorer", default="qwen-plus")
    args = p.parse_args()

    inputs = get_benchmark(n=args.n, genres=args.genres, density=args.density)
    print(f"[eval_refine] 跑 {len(inputs)} 条 · baseline={args.model} · refine={args.refine_model}")
    agent = PoetryAgent(
        generation_adapter=_make_adapter(args.model),
        score_adapter=_make_adapter(args.scorer),
        title_adapter=_make_adapter(args.scorer),
        prompt_adapter=_make_adapter(args.scorer),
    )

    rows = []
    for i, item in enumerate(inputs):
        print(f"  [{i+1}/{len(inputs)}] {item.user_input[:30]}…")
        try:
            r = _run_one(agent, item, args)
            rows.append(r)
            if "error" in r:
                print(f"      ⚠ {r['error']}")
            else:
                a = r["baseline_scores"].get("total", 0)
                b = r["refined_scores"].get("total", 0)
                tag = "✓" if r.get("changed") else "—"
                print(f"      baseline={fmt_num(a)} → refined={fmt_num(b)} ({tag})")
        except Exception as e:
            print(f"      ⚠ 异常：{e}")
            rows.append({"user_input": item.user_input, "error": str(e)})

    md = _render_report(args, rows)
    print_and_return(md)
    paths = save_artifacts("eval_refine", {"config": vars(args), "rows": rows}, md)
    print(f"原始数据: {paths['json']}")
    print(f"Markdown: {paths['md']}")


if __name__ == "__main__":
    main()
