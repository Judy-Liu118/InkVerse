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


def _count_llm_metrics(state: AgentState) -> Dict[str, Any]:
    """从 state.llm_loop_decisions 抽 LLM-driven 路径的诚实性指标。"""
    decisions = list(getattr(state, "llm_loop_decisions", []) or [])
    total = len(decisions)
    fallback_count = sum(1 for d in decisions if d.get("is_fallback"))
    stale_override_count = sum(1 for d in decisions if d.get("stale_override"))
    tool_hist: Dict[str, int] = {}
    for d in decisions:
        t = d.get("tool") or "unknown"
        tool_hist[t] = tool_hist.get(t, 0) + 1
    return {
        "total_decisions":      total,
        "fallback_count":       fallback_count,
        "fallback_rate":        (fallback_count / total) if total else 0.0,
        "stale_override_count": stale_override_count,
        "tool_hist":            tool_hist,
        "decisions":            decisions,   # per-decision 列联表用
    }


def _run_autonomous_full(
    agent: PoetryAgent, item: BenchInput, args,
    *, llm_driven: bool, mode_name: str, vlm_judge=None,
) -> Dict[str, Any]:
    state = _new_state(item, args)
    config = AutonomousConfig(
        target_clip_score=args.target,
        max_image_improve_rounds=args.max_img_rounds,
        allow_poem_refine=args.max_poem_rounds > 0,
        max_poem_refine_rounds=args.max_poem_rounds,
        image_improve_mode="rewrite_regen",
        image_loop_llm_driven=llm_driven,
    )
    t0 = time.time()
    final = None
    first_image = None    # 改图前首图（VLM 独立裁判用）
    try:
        for s in autonomous_full_run(agent, state, config=config):
            final = s
            # 第一次 trace 最后一条含 "生图完成" 时，state.image 是改图循环开始前的首图
            if first_image is None and s.image is not None and s.trace and "生图完成" in s.trace[-1].action:
                try:
                    first_image = s.image.copy()
                except Exception:
                    first_image = None
    except Exception as e:
        return {"mode": mode_name, "error": str(e)}
    if final is None:
        return {"mode": mode_name, "error": "no final state"}
    elapsed = time.time() - t0
    rounds = _count_rounds(final)
    out: Dict[str, Any] = {
        "mode":         mode_name,
        "clip_raw":     _raw_clip(final),
        "poem":         final.poem,
        "elapsed_sec":  round(elapsed, 2),
        "rounds":       rounds,
    }
    if llm_driven:
        out["llm_metrics"] = _count_llm_metrics(final)
    if vlm_judge is not None and first_image is not None and final.image is not None:
        # 改图前 vs 改图后：跑两次 VLM 单图打分，外部 ground truth（不是 loop 的优化目标）
        try:
            v_before = vlm_judge.score(
                image=first_image, poem=final.poem,
                visual_keywords_en=final.visual_keywords_en or "",
            )
            v_after = vlm_judge.score(
                image=final.image, poem=final.poem,
                visual_keywords_en=final.visual_keywords_en or "",
            )
            out["vlm"] = {
                "model":           vlm_judge.model,
                "before_raw":      v_before.raw_score,
                "after_raw":       v_after.raw_score,
                "before_error":    v_before.error,
                "after_error":     v_after.error,
                "after_better":    (
                    v_after.raw_score > v_before.raw_score
                    if v_before.raw_score is not None and v_after.raw_score is not None
                    else None
                ),
                "delta_raw":       (
                    v_after.raw_score - v_before.raw_score
                    if v_before.raw_score is not None and v_after.raw_score is not None
                    else None
                ),
            }
        except Exception as e:
            out["vlm"] = {"error": f"vlm pipeline: {e}"}
    return out


def _run_autonomous(agent: PoetryAgent, item: BenchInput, args, vlm_judge=None) -> Dict[str, Any]:
    return _run_autonomous_full(agent, item, args, llm_driven=False,
                                mode_name="autonomous", vlm_judge=vlm_judge)


def _run_autonomous_llm_driven(agent: PoetryAgent, item: BenchInput, args, vlm_judge=None) -> Dict[str, Any]:
    return _run_autonomous_full(agent, item, args, llm_driven=True,
                                mode_name="autonomous_llm", vlm_judge=vlm_judge)


def _run_one(agent: PoetryAgent, item: BenchInput, args, vlm_judge=None) -> Dict[str, Any]:
    sp = _run_single_pass(agent, item, args)
    au = _run_autonomous(agent, item, args, vlm_judge=vlm_judge)
    au_llm = _run_autonomous_llm_driven(agent, item, args, vlm_judge=vlm_judge)
    return {
        "user_input": item.user_input,
        "genre": item.genre, "theme": item.theme,
        "single_pass":    sp,
        "autonomous":     au,
        "autonomous_llm": au_llm,
    }


def _arm_scores(rows, key):
    return [r[key].get("clip_raw") for r in rows
            if key in r and "error" not in r[key]]


def _arm_times(rows, key):
    return [r[key].get("elapsed_sec") for r in rows
            if key in r and "error" not in r[key]]


def _arm_rounds(rows, key, field):
    return [r[key]["rounds"][field] for r in rows
            if key in r and "error" not in r[key]]


def _arm_stat_row(name, scores):
    if not scores:
        return [name, 0, "—", "—", "—", "—", "—"]
    s = summarize(scores)
    return [name, s["n"], fmt_num(s["mean"]), fmt_num(s["std"]),
            fmt_num(s["median"]), fmt_num(s["min"]), fmt_num(s["max"])]


def _paired_delta_section(rows, key_a, key_b, label_a, label_b):
    pairs = [(r[key_a]["clip_raw"], r[key_b]["clip_raw"]) for r in rows
             if key_a in r and key_b in r
             and "error" not in r[key_a] and "error" not in r[key_b]]
    if not pairs:
        return f"_{label_b} − {label_a}：无可用配对样本_\n"
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    delta = paired_delta(a, b)
    return table(
        ["指标", "值"],
        [
            ["样本对数 n", delta["n"]],
            ["mean Δ", fmt_num(delta["mean_delta"], 4)],
            ["median Δ", fmt_num(delta["median_delta"], 4)],
            [f"{label_b} 提升比例", f"{delta['positive_rate']:.1%}"],
        ],
    )


def _render_report(args, rows):
    md = []
    md.append("# eval_autonomous 报告 · single_pass vs autonomous(fixed) vs autonomous(llm-driven)")
    md.append(f"_n={len(rows)} · 目标 CLIP raw={args.target} · "
              f"max 改图={args.max_img_rounds} · max 改诗={args.max_poem_rounds}_")
    md.append("")

    sp_scores = _arm_scores(rows, "single_pass")
    au_scores = _arm_scores(rows, "autonomous")
    al_scores = _arm_scores(rows, "autonomous_llm")

    if not sp_scores or not au_scores:
        return "\n".join(md + ["无可用数据。"])

    md.append("## 1. CLIP raw 终值对比（三臂）")
    md.append(table(
        ["模式", "n", "mean", "std", "median", "min", "max"],
        [
            _arm_stat_row("single_pass",     sp_scores),
            _arm_stat_row("autonomous",      au_scores),
            _arm_stat_row("autonomous_llm",  al_scores),
        ],
    ))
    md.append("")

    md.append("## 2. 配对差值")
    md.append("### 2.1 autonomous(fixed) − single_pass")
    md.append(_paired_delta_section(rows, "single_pass", "autonomous",
                                    "single_pass", "autonomous"))
    md.append("")
    md.append("### 2.2 autonomous(llm-driven) − single_pass")
    md.append(_paired_delta_section(rows, "single_pass", "autonomous_llm",
                                    "single_pass", "autonomous_llm"))
    md.append("")
    md.append("### 2.3 autonomous(llm-driven) − autonomous(fixed)  ← agentic 是否值钱")
    md.append(_paired_delta_section(rows, "autonomous", "autonomous_llm",
                                    "autonomous", "autonomous_llm"))
    md.append("")

    md.append("## 3. 成本：耗时 + 自主轮次")
    sp_t = summarize(_arm_times(rows, "single_pass") or [0])
    au_t = summarize(_arm_times(rows, "autonomous")  or [0])
    al_t = summarize(_arm_times(rows, "autonomous_llm") or [0])
    au_img = summarize(_arm_rounds(rows, "autonomous",     "image_rounds") or [0])
    al_img = summarize(_arm_rounds(rows, "autonomous_llm", "image_rounds") or [0])
    au_poem = summarize(_arm_rounds(rows, "autonomous",     "poem_rounds") or [0])
    al_poem = summarize(_arm_rounds(rows, "autonomous_llm", "poem_rounds") or [0])
    md.append(table(
        ["指标", "single_pass", "autonomous(fixed)", "autonomous(llm)"],
        [
            ["平均耗时 (s)", fmt_num(sp_t["mean"], 1),
             fmt_num(au_t["mean"], 1), fmt_num(al_t["mean"], 1)],
            ["中位耗时 (s)", fmt_num(sp_t["median"], 1),
             fmt_num(au_t["median"], 1), fmt_num(al_t["median"], 1)],
            ["平均改图轮次", "—", fmt_num(au_img["mean"], 1), fmt_num(al_img["mean"], 1)],
            ["平均改诗轮次", "—", fmt_num(au_poem["mean"], 1), fmt_num(al_poem["mean"], 1)],
        ],
    ))
    md.append("")

    # ─── 第 4 节：诚实性指标 ───────────────────────────────────────────────
    md.append("## 4. LLM-driven 循环诚实性")
    llm_arm = [r["autonomous_llm"] for r in rows
               if "autonomous_llm" in r and "error" not in r["autonomous_llm"]
               and "llm_metrics" in r["autonomous_llm"]]
    if not llm_arm:
        md.append("_无可用 LLM-driven 数据_\n")
    else:
        per_input_fallback_rate = [a["llm_metrics"]["fallback_rate"] for a in llm_arm]
        per_input_stale = [a["llm_metrics"]["stale_override_count"] for a in llm_arm]
        per_input_total = [a["llm_metrics"]["total_decisions"] for a in llm_arm]
        total_decisions = sum(per_input_total)
        total_fallback = sum(a["llm_metrics"]["fallback_count"] for a in llm_arm)
        total_stale = sum(per_input_stale)
        overall_fallback_rate = (total_fallback / total_decisions) if total_decisions else 0.0
        mean_fallback_rate = summarize(per_input_fallback_rate)["mean"] if per_input_fallback_rate else 0.0
        mean_stale = summarize(per_input_stale)["mean"] if per_input_stale else 0.0

        # 合并 tool_hist
        merged_hist: Dict[str, int] = {}
        for a in llm_arm:
            for k, v in a["llm_metrics"]["tool_hist"].items():
                merged_hist[k] = merged_hist.get(k, 0) + v

        md.append(table(
            ["指标", "值"],
            [
                ["总决策数（所有 input 累计）", total_decisions],
                ["fallback 总数",                  total_fallback],
                ["整体 fallback 率",               f"{overall_fallback_rate:.1%}"],
                ["per-input 平均 fallback 率",     f"{mean_fallback_rate:.1%}"],
                ["per-input 平均 stale-override 次数", fmt_num(mean_stale, 2)],
                ["stale-override 总次数",         total_stale],
            ],
        ))
        md.append("")
        md.append("**工具选择分布:**")
        md.append(table(
            ["tool", "次数", "占比"],
            [[k, v, f"{v/max(total_decisions,1):.1%}"]
             for k, v in sorted(merged_hist.items(), key=lambda x: -x[1])],
        ))
        md.append("")
        # 解读（写进报告，面试官能直接读）
        md.append("**解读:**")
        md.append(
            f"- fallback 率 = {overall_fallback_rate:.1%}：fallback 越高，说明 LLM 决策实际上大多被"
            f"确定性兜底覆盖，"
            f"\"agentic\" 这个词的实质含量越低。"
        )
        md.append(
            f"- stale-override 平均 {mean_stale:.2f} 次/条：>0 说明 LLM 给出的 stop 工具是装饰品，"
            f"启发式护栏在替它喊停。"
        )
        md.append("")

        # ─── P1-1：per-decision 归因列联表 ────────────────────────────────
        md.append("## 5. per-decision 归因（tool × 本轮是否提升 CLIP）")
        contingency: Dict[str, Dict[str, int]] = {}
        for a in llm_arm:
            for d in a["llm_metrics"]["decisions"]:
                tool = d.get("tool") or "unknown"
                sb, sa = d.get("score_before"), d.get("score_after")
                # 首轮无 prev / 终止决策无 after → 归 "no_signal"
                if sb is None or sa is None:
                    bucket = "no_signal"
                elif sa - sb > 0:
                    bucket = "improved"
                else:
                    bucket = "not_improved"
                contingency.setdefault(tool, {"improved": 0, "not_improved": 0, "no_signal": 0})
                contingency[tool][bucket] += 1
        if contingency:
            md.append(table(
                ["tool", "improved", "not_improved", "no_signal", "总计"],
                [[t, c["improved"], c["not_improved"], c["no_signal"],
                  c["improved"] + c["not_improved"] + c["no_signal"]]
                 for t, c in sorted(contingency.items(), key=lambda x: -sum(x[1].values()))],
            ))
            md.append("")
            md.append(
                "**解读**：横向看 `refine_poem_and_regen` 行 —— 如果其 improved 占比明显高于 "
                "`edit_image`，则 LLM 在\"反复改图无果\"时切换工具的判断有正向价值；否则路由近随机。"
            )
            md.append("")
        else:
            md.append("_无决策数据可统计_\n")

    # ─── 第 6 节：VLM 独立裁判（破 CLIP 循环论证）────────────────────────
    md.append("## 6. VLM 独立裁判：改图前 vs 改图后")
    md.append("> 改图循环的优化目标是 CLIP-final，再拿 CLIP-final 当成功指标是部分同义反复。")
    md.append("> 这一节用 VLM（loop 外裁判）直接评 before-image vs after-image，分数与 loop 优化目标解耦。")
    md.append("")

    def _vlm_arm_stats(rows, arm_key):
        vlm_pairs = []
        for r in rows:
            arm = r.get(arm_key, {})
            if "error" in arm or "vlm" not in arm:
                continue
            v = arm["vlm"]
            if "error" in v:
                continue
            if v.get("before_raw") is None or v.get("after_raw") is None:
                continue
            vlm_pairs.append(v)
        if not vlm_pairs:
            return None
        deltas = [v["delta_raw"] for v in vlm_pairs]
        better_count = sum(1 for v in vlm_pairs if v["after_better"])
        tie_count = sum(1 for v in vlm_pairs if v["delta_raw"] == 0)
        d_stat = summarize(deltas)
        return {
            "n":            len(vlm_pairs),
            "after_better": better_count,
            "tie":          tie_count,
            "better_rate":  better_count / len(vlm_pairs),
            "mean_delta":   d_stat["mean"],
            "median_delta": d_stat["median"],
        }

    vlm_au = _vlm_arm_stats(rows, "autonomous")
    vlm_al = _vlm_arm_stats(rows, "autonomous_llm")
    if vlm_au is None and vlm_al is None:
        md.append("_未启用 VLM judge（--vlm-judge 未指定），跳过_\n")
    else:
        def _row(arm, stats):
            if stats is None:
                return [arm, "—", "—", "—", "—", "—"]
            return [arm, stats["n"], stats["after_better"],
                    f"{stats['better_rate']:.1%}",
                    fmt_num(stats["mean_delta"], 3),
                    fmt_num(stats["median_delta"], 3)]
        md.append(table(
            ["臂", "n", "after_better", "after 更优比例", "mean Δ(raw 0-10)", "median Δ"],
            [_row("autonomous(fixed)", vlm_au), _row("autonomous(llm)", vlm_al)],
        ))
        md.append("")
        md.append(
            "**解读**：after 更优比例 = VLM 判定改图后图更契合诗的样本占比。"
            "这是**非 CLIP 的成功率数字**：成功指标不再是 loop 自己在爬的那个数。"
            "若该比例远低于 50% 或 mean Δ ≤ 0，说明改图循环按 CLIP 在涨，但外部 oracle 看不到收益 —— "
            "提示 CLIP 在中文诗 + 水墨域可能存在过拟合到 reward 的失败模式。"
        )
        md.append("")

    md.append("## 7. 抽样")
    for r in rows[:5]:
        sp = r.get("single_pass", {}); au = r.get("autonomous", {})
        al = r.get("autonomous_llm", {})
        if "error" in sp or "error" in au or "error" in al:
            continue
        md.append(f"### {r['user_input']}")
        md.append(f"- single_pass:     CLIP={fmt_num(sp['clip_raw'])}, {sp['elapsed_sec']}s")
        au_extra = ""
        if "vlm" in au and "error" not in au["vlm"] and au["vlm"].get("delta_raw") is not None:
            au_extra = f", VLM Δ={fmt_num(au['vlm']['delta_raw'], 2)}"
        md.append(f"- autonomous:      CLIP={fmt_num(au['clip_raw'])}, {au['elapsed_sec']}s, "
                  f"改图 {au['rounds']['image_rounds']} 轮, 改诗 {au['rounds']['poem_rounds']} 轮{au_extra}")
        al_extra = ""
        if "vlm" in al and "error" not in al["vlm"] and al["vlm"].get("delta_raw") is not None:
            al_extra = f", VLM Δ={fmt_num(al['vlm']['delta_raw'], 2)}"
        md.append(f"- autonomous_llm:  CLIP={fmt_num(al['clip_raw'])}, {al['elapsed_sec']}s, "
                  f"改图 {al['rounds']['image_rounds']} 轮, "
                  f"fallback={al.get('llm_metrics',{}).get('fallback_count',0)}/"
                  f"{al.get('llm_metrics',{}).get('total_decisions',0)}, "
                  f"stale_override={al.get('llm_metrics',{}).get('stale_override_count',0)}{al_extra}")
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
    p.add_argument("--vlm-judge",       default=None,
                   help="启用 VLM 独立裁判（外部 oracle，破 CLIP 循环论证），"
                        "例如 qwen-vl-max / glm-4v-plus；留空跳过")
    args = p.parse_args()

    inputs = get_benchmark(n=args.n, genres=args.genres, density=args.density)
    vlm_judge = None
    if args.vlm_judge:
        from eval.vlm_judge import VLMJudge
        vlm_judge = VLMJudge(model=args.vlm_judge)
        print(f"[eval_autonomous] VLM 独立裁判启用: {args.vlm_judge}")
    print(f"[eval_autonomous] 跑 {len(inputs)} 条 × 3 模式（single_pass / autonomous / "
          f"autonomous_llm_driven） · image={args.image_backend}")
    agent = _build_agent(args)

    rows = []
    for i, item in enumerate(inputs):
        print(f"  [{i+1}/{len(inputs)}] {item.user_input[:30]}…")
        try:
            r = _run_one(agent, item, args, vlm_judge=vlm_judge)
            rows.append(r)
            sp = r["single_pass"]; au = r["autonomous"]; al = r["autonomous_llm"]
            if "error" in sp:
                print(f"      single_pass    ⚠ {sp['error']}")
            else:
                print(f"      single_pass    CLIP={fmt_num(sp['clip_raw'])}, {sp['elapsed_sec']}s")
            if "error" in au:
                print(f"      autonomous     ⚠ {au['error']}")
            else:
                print(f"      autonomous     CLIP={fmt_num(au['clip_raw'])}, {au['elapsed_sec']}s, "
                      f"改图 {au['rounds']['image_rounds']} 轮, 改诗 {au['rounds']['poem_rounds']} 轮")
            if "error" in al:
                print(f"      autonomous_llm ⚠ {al['error']}")
            else:
                metrics = al.get("llm_metrics", {}) or {}
                print(f"      autonomous_llm CLIP={fmt_num(al['clip_raw'])}, {al['elapsed_sec']}s, "
                      f"改图 {al['rounds']['image_rounds']} 轮, "
                      f"fallback={metrics.get('fallback_count', 0)}/"
                      f"{metrics.get('total_decisions', 0)}, "
                      f"stale_override={metrics.get('stale_override_count', 0)}")
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
