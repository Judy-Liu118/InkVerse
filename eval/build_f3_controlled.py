"""
eval.build_f3_controlled -- 构造 F3 controlled pair 池

F3 命题：LLM-as-judge 对格律给低优先级权重，意境维度有 ≥margin 优势即可
        覆盖极端格律差距（base 严重出律但意境 ≥ lora → base 仍胜）。

#58 P0 收尾报告里 n=4 anecdote-level controlled pair 验证支持该命题，但
样本量太小（87.5% 胜率但 CI 巨大）。本脚本搭建专门构造 controlled pair
的工具，输出 JSON 池供后续 pairwise judge（n≥64 → 显著结论）。

本脚本只做"构造池"这一步，pairwise judge 留给 eval_poem / 单独脚本
（compare_poems / score_single_multi_judge 已存在，直接复用）。

阶段:
  1. 选 themes（dataset.get_benchmark 默认 32 道）
  2. 对每 theme:
       · base 模型生成 N 候选（默认 5）
       · lora 模型生成 N 候选
  3. 每候选跑 local_score_poem 拿 pingze + score_single_multi_judge 拿
     imagery/cohesion/aesthetics
  4. 笛卡尔积 base × lora = N² pairs per theme，筛 controlled pair：
       base.pingze < pingze_threshold（默认 0.5，严重出律）
       AND mean(base.imagery, base.cohesion, base.aesthetics)
           >= mean(lora.imagery, lora.cohesion, lora.aesthetics) + intent_margin
  5. 输出 JSON: [{theme, base_poem, base_scores, lora_poem, lora_scores}, ...]

跑法（API 费一会儿，n_pairs 是目标量，达到即停）：
    python -m eval.build_f3_controlled \
        --n 64 --candidates 5 \
        --judges qwen-plus glm-4-plus deepseek-v4-pro \
        --base-model local_base --lora-model local_lora

不跑实 API、只验证 pipeline 拓扑：
    python -m eval.build_f3_controlled --n 4 --dry-run

⚠️ 真跑会很慢：(themes × N × 2 模型 × multi_judge) 量级的 API 调用，
    n=64 通常需要 themes=32 全跑一遍，估算 1-2hr。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.models.adapter import ModelAdapter
from core.poem.generator import PoemGenerator
from config import (
    DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, ZHIPU_API_KEY, MOONSHOT_API_KEY,
)
from eval.dataset import get_benchmark, BenchInput
from eval.report import save_artifacts


# ── adapter（与 eval_poem.py 同源；不抽公共模块避免循环依赖）─────────────
def _make_adapter(model_choice: str) -> ModelAdapter:
    if model_choice in ("local_base", "local_base_naked"):
        return ModelAdapter(backend="local", allow_lora_fallback=False)
    if model_choice in ("local_lora", "local_lora_naked"):
        return ModelAdapter(backend="local_lora", allow_lora_fallback=False)
    if model_choice.startswith("deepseek"):
        return ModelAdapter(backend="deepseek", api_key=DEEPSEEK_API_KEY,
                            api_model=model_choice)
    if model_choice.startswith("glm"):
        return ModelAdapter(backend="zhipu", api_key=ZHIPU_API_KEY,
                            api_model=model_choice)
    if model_choice.startswith("kimi") or model_choice.startswith("moonshot"):
        return ModelAdapter(backend="moonshot", api_key=MOONSHOT_API_KEY,
                            api_model=model_choice)
    return ModelAdapter(backend="qwen", api_key=DASHSCOPE_API_KEY,
                        api_model=model_choice)


# ── 核心筛逻辑（纯函数，可单测）──────────────────────────────────────────
def _mean_intent_dims(scores: Dict[str, Any]) -> float:
    """取 imagery / cohesion / aesthetics 的均值作为"意境维度"代表。

    pingze / rhyme 排除（它们才是格律本身，要看的就是这两类的拮抗）。
    intent 也排除：它衡量切题度，与"意境"概念正交。
    None 维度被跳过；全 None 时返回 0.0 让后续筛逻辑过滤掉。
    """
    keys = ("imagery", "cohesion", "aesthetics")
    vals = [scores.get(k) for k in keys]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def find_controlled_pairs(
    base_candidates: List[Dict[str, Any]],
    lora_candidates: List[Dict[str, Any]],
    pingze_threshold: float = 0.5,
    intent_margin: float = 0.0,
) -> List[Dict[str, Any]]:
    """对 base × lora 候选笛卡尔积筛 controlled pair。

    Controlled pair 定义:
      · base.pingze < pingze_threshold（严重出律）
      · mean(base 意境维度) >= mean(lora 意境维度) + intent_margin

    返回每条 pair: {base_poem, base_scores, lora_poem, lora_scores,
                    base_intent_mean, lora_intent_mean, pingze_diff}
    """
    pairs = []
    for b in base_candidates:
        b_pz = b["scores"].get("pingze")
        if not isinstance(b_pz, (int, float)) or b_pz >= pingze_threshold:
            continue
        b_intent = _mean_intent_dims(b["scores"])
        for l in lora_candidates:
            l_intent = _mean_intent_dims(l["scores"])
            if b_intent < l_intent + intent_margin:
                continue
            l_pz = l["scores"].get("pingze", 0.0) or 0.0
            pairs.append({
                "base_poem":       b["poem"],
                "base_scores":     b["scores"],
                "lora_poem":       l["poem"],
                "lora_scores":     l["scores"],
                "base_intent_mean": b_intent,
                "lora_intent_mean": l_intent,
                "pingze_diff":      b_pz - l_pz,   # 负数：base 比 lora 平仄差
            })
    return pairs


# ── 生成 + 评分一首主题的所有候选 ────────────────────────────────────────
def _score_candidates(
    gen: PoemGenerator, poems: List[str], item: BenchInput,
    judges: List[Tuple[str, ModelAdapter]],
) -> List[Dict[str, Any]]:
    """对每个候选诗算 local（pingze/rhyme/imagery/cohesion）+ multi-judge 4 维。"""
    _, num_lines, chars_per_line = gen.scorer.detect_genre(item.user_input)
    out = []
    for i, p in enumerate(poems):
        try:
            local = gen.scorer.local_score_poem(
                p, num_lines, chars_per_line, topic_score=0.7,
            )
        except Exception as e:
            print(f"        ⚠ local_score 异常 c{i+1}: {e}")
            local = {"pingze": 0.0, "rhyme": 0.0, "imagery": 0.0, "cohesion": 0.0}
        try:
            mj = gen.scorer.score_single_multi_judge(
                p, item.user_input, judges, candidate_index=i + 1,
            )
        except Exception as e:
            print(f"        ⚠ multi_judge 异常 c{i+1}: {e}")
            mj = {"intent": None, "imagery": None,
                  "cohesion": None, "aesthetics": None}
        merged = {
            "pingze":     local.get("pingze"),
            "rhyme":      local.get("rhyme"),
            "intent":     mj.get("intent"),
            "imagery":    mj.get("imagery"),
            "cohesion":   mj.get("cohesion"),
            "aesthetics": mj.get("aesthetics"),
        }
        out.append({"poem": p, "scores": merged})
    return out


def _generate_for_theme(
    gen: PoemGenerator, item: BenchInput, adapter: ModelAdapter,
    candidate_count: int,
) -> List[str]:
    _, poems = gen.generate_candidates_only(
        item.user_input, generation_adapter=adapter, count=candidate_count,
        prompt_mode="full",
    )
    return poems or []


# ── 主入口 ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="构造 F3 controlled pair 池（base 严重出律 + 意境 ≥ lora）",
    )
    p.add_argument("--n", type=int, default=64,
                   help="目标 controlled pair 数（达到即停）")
    p.add_argument("--candidates", type=int, default=5,
                   help="每模型每主题生成的候选数")
    p.add_argument("--themes", type=int, default=32,
                   help="最多从 benchmark 取多少主题")
    p.add_argument("--base-model", default="local_base")
    p.add_argument("--lora-model", default="local_lora")
    p.add_argument("--judges", nargs="*",
                   default=["qwen-plus", "glm-4-plus", "deepseek-v4-pro"],
                   help="跨家族多评委（≥3 → 多数决，跨家族 抗 self-bias）")
    p.add_argument("--pingze-threshold", type=float, default=0.5,
                   help="base.pingze < 此值 视为严重出律")
    p.add_argument("--intent-margin", type=float, default=0.0,
                   help="base 意境均值 ≥ lora 意境均值 + 此 margin（默认 0 = 持平算）")
    p.add_argument("--dry-run", action="store_true",
                   help="不调真 API，用 dummy 数据验证 pipeline 拓扑")
    args = p.parse_args()

    if args.dry_run:
        print("[build_f3_controlled] dry-run 模式：用合成数据验证拓扑，不调 API")
        # 合成 8 条假候选对，触发 1-2 条 controlled pair
        base_cands = [
            {"poem": f"假 base 诗 {i}", "scores": {
                "pingze": 0.25 if i < 2 else 0.85, "rhyme": 0.8,
                "intent": 0.7, "imagery": 0.85, "cohesion": 0.8, "aesthetics": 0.75,
            }} for i in range(3)
        ]
        lora_cands = [
            {"poem": f"假 lora 诗 {i}", "scores": {
                "pingze": 0.95, "rhyme": 0.9,
                "intent": 0.7, "imagery": 0.70, "cohesion": 0.75, "aesthetics": 0.70,
            }} for i in range(3)
        ]
        pairs = find_controlled_pairs(
            base_cands, lora_cands,
            pingze_threshold=args.pingze_threshold,
            intent_margin=args.intent_margin,
        )
        print(f"[dry-run] 合成 controlled pair 数: {len(pairs)}")
        for i, pr in enumerate(pairs[:3]):
            print(f"  pair {i+1}: base.pingze={pr['base_scores']['pingze']}, "
                  f"base.intent_mean={pr['base_intent_mean']:.3f}, "
                  f"lora.intent_mean={pr['lora_intent_mean']:.3f}")
        return

    # ── 真跑：构 judges + 加载 base/lora ─────────────────────────────────
    judges = [(label, _make_adapter(label)) for label in args.judges]
    base_adapter = _make_adapter(args.base_model)
    lora_adapter = _make_adapter(args.lora_model)
    gen = PoemGenerator()

    inputs = get_benchmark(n=args.themes)
    print(f"[build_f3_controlled] 目标 {args.n} 对 controlled pair · "
          f"从 {len(inputs)} 主题里跑 base/lora 各 {args.candidates} 候选")

    collected: List[Dict[str, Any]] = []
    per_theme_stats: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, item in enumerate(inputs):
        if len(collected) >= args.n:
            print(f"  达到目标 {args.n} 对，提前停（处理 {i} 主题）")
            break
        print(f"  [{i+1}/{len(inputs)}] {item.user_input[:30]}…")
        try:
            base_poems = _generate_for_theme(gen, item, base_adapter, args.candidates)
            lora_poems = _generate_for_theme(gen, item, lora_adapter, args.candidates)
            base_cands = _score_candidates(gen, base_poems, item, judges)
            lora_cands = _score_candidates(gen, lora_poems, item, judges)
        except Exception as e:
            print(f"      ⚠ 主题处理异常：{e}")
            per_theme_stats.append({
                "theme": item.user_input, "error": str(e),
                "n_pairs": 0,
            })
            continue
        pairs = find_controlled_pairs(
            base_cands, lora_cands,
            pingze_threshold=args.pingze_threshold,
            intent_margin=args.intent_margin,
        )
        # 把主题信息塞进每对，便于后续 pairwise 评测追溯
        for pr in pairs:
            pr["theme"] = item.user_input
            pr["genre"] = item.genre
        collected.extend(pairs)
        per_theme_stats.append({
            "theme":         item.user_input,
            "n_base":        len(base_cands),
            "n_lora":        len(lora_cands),
            "n_pairs":       len(pairs),
        })
        print(f"      controlled pair: +{len(pairs)} (累计 {len(collected)}/{args.n})")

    elapsed = time.time() - t0
    print(f"\n[build_f3_controlled] 完成 · 收集 {len(collected)} 对 · 耗时 {elapsed:.1f}s")

    # ── 落盘 ─────────────────────────────────────────────────────────────
    summary = {
        "config":        vars(args),
        "elapsed_sec":   round(elapsed, 1),
        "n_pairs":       len(collected),
        "per_theme":     per_theme_stats,
        "pairs":         collected,
    }
    md_lines = [
        "# F3 controlled pair 构造报告",
        f"_目标 n={args.n} · 实收 {len(collected)} · pingze_threshold={args.pingze_threshold}"
        f" · intent_margin={args.intent_margin}_",
        "",
        f"- 主题处理: {len(per_theme_stats)} / {len(inputs)}",
        f"- 总耗时: {elapsed:.1f}s",
        "",
        "## 下一步",
        "把 `pairs` 字段读进来跑 forward+reverse pairwise（compare_poems × 2 + 多评委），"
        "统计 base 胜率。胜率 >50% 即支持 F3（评委对格律给低优先级权重）。",
    ]
    paths = save_artifacts("f3_controlled_pairs", summary, "\n".join(md_lines))
    print(f"原始数据: {paths['json']}")
    print(f"Markdown: {paths['md']}")


if __name__ == "__main__":
    main()
