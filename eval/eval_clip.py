"""
eval.eval_clip -- CLIP 双锚点 vs 单锚点对齐分对比（项目核心创新点）

跑法：
    python -m eval.eval_clip --n 10
    python -m eval.eval_clip --n 10 --image-backend bailian:qwen-image-2.0-pro
    python -m eval.eval_clip --n 5 --density sparse   # 只跑稀疏关键词诗

输出：
    对每条 user_input 跑一遍完整流水线（生成诗 → 提取意象 → 提示词 → 生图），
    然后用同一张图同时按 3 套锚点策略评分：
      · single_anchor (prompt-only)：仅提示词锚点
      · dual_anchor   (production)：双锚点 + 稀疏自适应权重（当前生产配置）
      · poem_only_anchor：仅诗歌锚点（对照）
    报告：
      · 双锚点相对单锚点的平均提升 Δ + 显著性比例
      · 关键词密度分层（rich vs sparse）下的提升差异
      · 各模式的均值/方差/区间

⚠️ 跑这个评估需要：DASHSCOPE_API_KEY + 本地或 API 图像后端 + CLIP 权重。
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models.adapter import ModelAdapter
from core.agent.agent import PoetryAgent
from core.agent.state import AgentState, Phase
from config import (
    DASHSCOPE_API_KEY, DEEPSEEK_API_KEY,
    CLIP_POEM_WEIGHT, CLIP_PROMPT_WEIGHT,
    CLIP_SPARSE_POEM_WEIGHT, CLIP_SPARSE_PROMPT_WEIGHT,
    CLIP_SPARSE_WORD_THRESHOLD,
    STYLE_MAP,
)

from eval.dataset import get_benchmark, BenchInput
from eval.metrics import summarize, paired_delta, spearman_corr, pearson_corr
from eval.report import save_artifacts, table, fmt_num, print_and_return
from eval.vlm_judge import VLMJudge


def _make_adapter(model_choice: str, allow_lora_fallback: bool = False) -> ModelAdapter:
    if model_choice == "local_lora":
        return ModelAdapter(backend="local_lora", allow_lora_fallback=allow_lora_fallback)
    if model_choice == "local_base":
        return ModelAdapter(backend="local", allow_lora_fallback=allow_lora_fallback)
    if model_choice.startswith("deepseek"):
        return ModelAdapter(backend="deepseek", api_key=DEEPSEEK_API_KEY, api_model=model_choice)
    return ModelAdapter(backend="qwen", api_key=DASHSCOPE_API_KEY, api_model=model_choice)


def _build_agent(args) -> PoetryAgent:
    return PoetryAgent(
        generation_adapter=_make_adapter(args.poem_model, allow_lora_fallback=True),
        score_adapter=_make_adapter(args.scorer),
        title_adapter=_make_adapter(args.scorer),
        prompt_adapter=_make_adapter(args.prompt_model),
    )


def _parse_backend(val: str):
    if val.startswith("bailian:"):
        return "bailian", val.split(":", 1)[1]
    return "local", None


def _backend_short_tag(backend_str: str) -> str:
    """给图像存档目录用的短 tag，便于多次 run 不撞名。"""
    if backend_str.startswith("bailian:"):
        model = backend_str.split(":", 1)[1].replace(".", "").replace("-", "")
        return f"bailian-{model[:10]}"
    if backend_str == "local":
        return "local-zimg"
    return re.sub(r"[^a-zA-Z0-9]+", "-", backend_str)[:20]


def _load_reused_rows(path: str) -> List[Dict[str, Any]]:
    """从上次 eval_clip 的 JSON 加载成功 row（用于复用诗/锚点/prompt）。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", data) if isinstance(data, dict) else data
    required = ("user_input", "poem", "prompt", "genre", "theme", "keyword_density")
    ok: List[Dict[str, Any]] = []
    for r in rows:
        if "error" in r:
            continue
        missing = [k for k in required if k not in r]
        if missing:
            raise ValueError(
                f"JSON row 缺少字段 {missing}：{str(r.get('user_input',''))[:30]}"
            )
        ok.append(r)
    if not ok:
        raise ValueError(f"{path} 没有可复用的成功 row（rows 全为 error 或缺字段）")
    return ok


_INVALID_FNAME_CHARS = re.compile(r'[\\/:*?"<>|\s]+')


def _sanitize_tag(s: str, max_len: int = 24) -> str:
    s = _INVALID_FNAME_CHARS.sub("_", (s or "").strip())
    s = s.strip("._") or "untitled"
    return s[:max_len]


def _score_tag(prefix: str, val: Optional[float]) -> str:
    if val is None:
        return f"{prefix}xxx"
    pct = max(0, min(100, int(round(val * 100))))
    return f"{prefix}{pct:03d}"


def _save_eval_image(image, save_dir: Path, idx: int, item: "BenchInput",
                     scores: Dict[str, Optional[float]]) -> Optional[str]:
    """落盘生成的图；返回相对 outputs/eval/ 的链接路径（供 md 用）。失败返回 None。"""
    fname = (
        f"{idx:02d}_"
        f"{_score_tag('d', scores.get('dual'))}_"
        f"{_score_tag('v', scores.get('vlm_oracle'))}_"
        f"{_sanitize_tag(item.theme)}.png"
    )
    target = save_dir / fname
    try:
        image.save(target, format="PNG")
    except Exception as e:
        print(f"      ⚠ 图片落盘失败：{e}")
        return None
    return f"{save_dir.name}/{fname}"


def _score_with_anchors(clip_eval, image, *, keywords_en: str, prompt: str,
                        mode: str) -> float:
    """根据 mode 返回该锚点策略下的 CLIP raw 分。

    · prompt_only  → 仅提示词锚点
    · poem_only    → 仅诗歌锚点（关键词为空时返回 None）
    · dual         → 当前生产配置（含稀疏自适应权重）
    """
    raw_prompt = clip_eval.score_raw_cosine(image, prompt)
    if mode == "prompt_only":
        return raw_prompt
    if not keywords_en:
        return None if mode == "poem_only" else raw_prompt
    raw_poem = clip_eval.score_raw_cosine(image, keywords_en)
    if mode == "poem_only":
        return raw_poem
    # dual
    word_count = len([w for w in keywords_en.replace(",", " ").split() if len(w) > 1])
    if word_count < CLIP_SPARSE_WORD_THRESHOLD:
        wa, wb = CLIP_SPARSE_POEM_WEIGHT, CLIP_SPARSE_PROMPT_WEIGHT
    else:
        wa, wb = CLIP_POEM_WEIGHT, CLIP_PROMPT_WEIGHT
    return wa * raw_poem + wb * raw_prompt


def _run_one(agent: PoetryAgent, item: BenchInput, args,
             vlm_judge: "VLMJudge" = None,
             image_save_dir: Optional[Path] = None,
             index: int = 0,
             reused: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    img_backend, img_api_model = _parse_backend(args.image_backend)
    state = AgentState(
        user_input=item.user_input,
        lang="英文",
        style_suffix=STYLE_MAP["水墨画"],
        image_backend=img_backend,
        image_api_key=DASHSCOPE_API_KEY if img_backend == "bailian" else None,
        image_api_model=img_api_model,
    )

    t0 = time.time()
    if reused is not None:
        # 严谨对照模式：复用上次的诗 + 视觉锚点 + 英文 prompt，只重新出图 + 评分
        # → 两次 run 的唯一变量就是 image backend 本身
        state.poem = reused["poem"]
        state.visual_keywords_en = reused.get("visual_keywords_en", "") or ""
        state.prompt = reused["prompt"]
        state = agent._phase_image_clip(state)
    else:
        state = agent._phase_plan(state)
        state = agent._phase_poem(state)
        if state.phase == Phase.ERROR:
            return {"user_input": item.user_input, "error": state.error}
        state = agent._phase_keyword_extract(state)
        state = agent._phase_title(state)
        state = agent._phase_prompt(state)
        if state.phase == Phase.ERROR:
            return {"user_input": item.user_input, "error": state.error}
        state = agent._phase_prompt_review(state)
        # 注意：_phase_image_clip 内部已含 CLIP 重试逻辑；我们需要拿到 image
        state = agent._phase_image_clip(state)
    elapsed = time.time() - t0

    if state.image is None:
        return {"user_input": item.user_input, "error": "无图像产出"}

    clip_eval = agent._get_clip_eval()
    if clip_eval is None:
        return {"user_input": item.user_input, "error": "CLIP 不可用"}

    keywords = state.visual_keywords_en or ""
    word_count = len([w for w in keywords.replace(",", " ").split() if len(w) > 1])

    scores = {
        "prompt_only": _score_with_anchors(
            clip_eval, state.image, keywords_en=keywords, prompt=state.prompt,
            mode="prompt_only"),
        "poem_only":   _score_with_anchors(
            clip_eval, state.image, keywords_en=keywords, prompt=state.prompt,
            mode="poem_only"),
        "dual":        _score_with_anchors(
            clip_eval, state.image, keywords_en=keywords, prompt=state.prompt,
            mode="dual"),
    }

    # VLM oracle（开启时多调一次多模态 API；失败不影响 CLIP 行）
    vlm_block = None
    if vlm_judge is not None:
        verdict = vlm_judge.score(
            image=state.image, poem=state.poem, visual_keywords_en=keywords,
        )
        scores["vlm_oracle"] = verdict.score   # [0, 1]，可能为 None
        vlm_block = verdict.as_dict()

    # 图片落盘（在拿到 dual + vlm 分数后命名，便于按撕裂程度肉眼扫文件夹）
    image_path = None
    if image_save_dir is not None:
        image_path = _save_eval_image(state.image, image_save_dir, index, item, scores)

    return {
        "user_input":      item.user_input,
        "genre":           item.genre,
        "theme":           item.theme,
        "keyword_density": item.keyword_density,
        "poem":            state.poem,
        "visual_keywords_en": keywords,
        "keyword_word_count": word_count,
        "prompt":          state.prompt,
        "elapsed_sec":     round(elapsed, 2),
        "raw_scores":      scores,   # 都是 raw cosine（[-1, 1]）；vlm_oracle ∈ [0,1]
        "vlm":             vlm_block,
        "image_path":      image_path,
    }


def _render_report(args, ok_rows, all_rows):
    md = []
    md.append("# eval_clip 报告 · 双锚点 vs 单锚点 CLIP 对齐分")
    reuse_suffix = ""
    if getattr(args, "reuse_poems_from", None):
        reuse_suffix = f" · **复用诗自** `{args.reuse_poems_from}`（仅 backend 不同）"
        if getattr(args, "reuse_indices", None):
            reuse_suffix += f"，indices={args.reuse_indices}"
    md.append(f"_n={len(ok_rows)}/{len(all_rows)} · image={args.image_backend} · "
              f"权重: 标准={CLIP_POEM_WEIGHT}/{CLIP_PROMPT_WEIGHT} · "
              f"稀疏={CLIP_SPARSE_POEM_WEIGHT}/{CLIP_SPARSE_PROMPT_WEIGHT} (阈值={CLIP_SPARSE_WORD_THRESHOLD}){reuse_suffix}_")
    md.append("")

    if not ok_rows:
        md.append("无可用数据。")
        return "\n".join(md)

    def col(rows, key):
        return [r["raw_scores"][key] for r in rows if r["raw_scores"][key] is not None]

    md.append("## 1. 总体均值（CLIP raw cosine ∈ [−1, 1]）")
    md.append(table(
        ["策略", "n", "mean", "std", "median", "min", "max"],
        [
            ["prompt_only", *[fmt_num(v) for v in (
                summarize(col(ok_rows, "prompt_only"))["n"],
                summarize(col(ok_rows, "prompt_only"))["mean"],
                summarize(col(ok_rows, "prompt_only"))["std"],
                summarize(col(ok_rows, "prompt_only"))["median"],
                summarize(col(ok_rows, "prompt_only"))["min"],
                summarize(col(ok_rows, "prompt_only"))["max"],
            )]],
            ["poem_only",   *[fmt_num(v) for v in (
                summarize(col(ok_rows, "poem_only"))["n"],
                summarize(col(ok_rows, "poem_only"))["mean"],
                summarize(col(ok_rows, "poem_only"))["std"],
                summarize(col(ok_rows, "poem_only"))["median"],
                summarize(col(ok_rows, "poem_only"))["min"],
                summarize(col(ok_rows, "poem_only"))["max"],
            )]],
            ["dual",        *[fmt_num(v) for v in (
                summarize(col(ok_rows, "dual"))["n"],
                summarize(col(ok_rows, "dual"))["mean"],
                summarize(col(ok_rows, "dual"))["std"],
                summarize(col(ok_rows, "dual"))["median"],
                summarize(col(ok_rows, "dual"))["min"],
                summarize(col(ok_rows, "dual"))["max"],
            )]],
        ],
    ))
    md.append("")

    md.append("## 2. 配对差值：dual − prompt_only（核心结论）")
    paired = paired_delta(col(ok_rows, "prompt_only"), col(ok_rows, "dual"))
    md.append(table(
        ["指标", "值"],
        [
            ["样本对数 n", paired["n"]],
            ["mean Δ",     fmt_num(paired["mean_delta"], 4)],
            ["median Δ",   fmt_num(paired["median_delta"], 4)],
            ["dual 提升比例", f"{paired['positive_rate']:.1%}"],
            ["std Δ",      fmt_num(paired["std_delta"], 4)],
        ],
    ))
    md.append("")

    md.append("## 3. 按关键词密度分层（rich vs sparse）")
    rich = [r for r in ok_rows if r["keyword_density"] == "rich"]
    sparse = [r for r in ok_rows if r["keyword_density"] == "sparse"]
    rows = []
    for label, group in [("rich (≥4 词)", rich), ("sparse (<4 词)", sparse)]:
        if not group:
            rows.append([label, 0, "—", "—", "—"])
            continue
        d = paired_delta(col(group, "prompt_only"), col(group, "dual"))
        rows.append([label, d["n"],
                     fmt_num(d["mean_delta"], 4),
                     fmt_num(d["median_delta"], 4),
                     f"{d['positive_rate']:.1%}"])
    md.append(table(
        ["分组", "n", "mean Δ", "median Δ", "dual 提升比例"], rows,
    ))
    md.append("")

    md.append("## 4. 抽样诗作 + 分数")
    for r in ok_rows:
        md.append(f"### {r['user_input']}（{r['keyword_density']}, 关键词 {r['keyword_word_count']} 词）")
        if r.get("image_path"):
            md.append(f"![]({r['image_path']})")
        md.append(f"- 诗：`{r['poem'].replace(chr(10), ' / ')}`")
        md.append(f"- 视觉锚点：{r['visual_keywords_en']}")
        line = (f"- 分数：prompt_only={fmt_num(r['raw_scores']['prompt_only'])} | "
                f"poem_only={fmt_num(r['raw_scores']['poem_only'])} | "
                f"**dual={fmt_num(r['raw_scores']['dual'])}**")
        if r["raw_scores"].get("vlm_oracle") is not None:
            line += f" | _VLM oracle={fmt_num(r['raw_scores']['vlm_oracle'])}_"
        md.append(line)
        if r.get("vlm") and r["vlm"].get("reasoning"):
            md.append(f"- VLM 理由：{r['vlm']['reasoning']}")
        md.append("")

    # ── §5. VLM oracle 相关性（外部 ground truth 锚定）─────────────────────
    has_vlm = any(r["raw_scores"].get("vlm_oracle") is not None for r in ok_rows)
    if has_vlm:
        md.append("## 5. CLIP 策略 vs VLM oracle 相关性（核心结论·外部锚定）")
        vlm_model = next(
            (r["vlm"]["model"] for r in ok_rows if r.get("vlm") and r["vlm"].get("model")),
            "—",
        )
        n_vlm_ok = sum(1 for r in ok_rows if r["raw_scores"].get("vlm_oracle") is not None)
        n_vlm_err = sum(1 for r in ok_rows
                        if r.get("vlm") and r["vlm"].get("error"))
        md.append(f"_VLM judge: **{vlm_model}** · 成功 {n_vlm_ok}/{len(ok_rows)}"
                  f"（失败 {n_vlm_err}）_")
        md.append("")

        vlm_vals = col(ok_rows, "vlm_oracle")
        rows = []
        for strat in ("prompt_only", "poem_only", "dual"):
            paired_rows = [
                r for r in ok_rows
                if r["raw_scores"].get(strat) is not None
                and r["raw_scores"].get("vlm_oracle") is not None
            ]
            xs = [r["raw_scores"][strat] for r in paired_rows]
            ys = [r["raw_scores"]["vlm_oracle"] for r in paired_rows]
            sp = spearman_corr(xs, ys)
            pe = pearson_corr(xs, ys)
            rows.append([
                strat, len(paired_rows),
                fmt_num(sp, 3) if sp is not None else "—",
                fmt_num(pe, 3) if pe is not None else "—",
            ])
        md.append(table(
            ["CLIP 策略", "n", "Spearman ρ", "Pearson r"], rows,
        ))
        md.append("")
        md.append("> **解读**：Spearman 越高 → 该 CLIP 策略对图文契合度的排序与 VLM "
                  "ground truth 越一致。理想情况下 dual > prompt_only / poem_only，"
                  "证明双锚点设计不只是「自我感觉良好」，而是真的更接近人类判图。")
        md.append("")

    return "\n".join(md)


def main():
    p = argparse.ArgumentParser(description="CLIP 双锚点 vs 单锚点对齐分对比")
    p.add_argument("--n", type=int, default=None,
                   help="跑多少条 benchmark；非 reuse 模式默认 10，reuse 模式默认全跑")
    p.add_argument("--genres", nargs="*", default=None)
    p.add_argument("--density", choices=["rich", "sparse"], default=None)
    p.add_argument("--poem-model",   default="local_lora",
                   help="诗歌生成模型；LoRA 优先")
    p.add_argument("--prompt-model", default="qwen-max",
                   help="英文提示词生成模型；推荐 qwen-max")
    p.add_argument("--scorer",       default="qwen-plus",
                   help="评分/关键词抽取/起名模型；qwen-plus 是性价比之选")
    p.add_argument("--image-backend", default="bailian:qwen-image-2.0-pro",
                   help="图像后端，如 local 或 bailian:qwen-image-2.0-pro")
    p.add_argument("--vlm-judge", default="none",
                   help="VLM ground-truth judge: none / qwen-vl-max / qwen-vl-plus / glm-4v-plus")
    p.add_argument("--no-save-images", action="store_true",
                   help="不落盘生成的图（默认会存到 outputs/eval/clip_img_<ts>/）")
    p.add_argument("--reuse-poems-from", default=None,
                   help="复用指定 JSON 文件里的 poem/visual_keywords_en/prompt，"
                        "跳过生诗+生 prompt；用于严谨的 image-backend 对比"
                        "（唯一变量 = 后端）")
    p.add_argument("--reuse-indices", default=None,
                   help="复用模式下挑特定行（1-based，逗号分隔，如 \"2,4,10\"），"
                        "用于跨 backend 的撕裂样本 spot check；"
                        "需配合 --reuse-poems-from；与 --n 同时给时 indices 优先")
    args = p.parse_args()

    reuse_rows: Optional[List[Dict[str, Any]]] = None
    if args.reuse_poems_from:
        reuse_rows = _load_reused_rows(args.reuse_poems_from)

        # --reuse-indices 优先：1-based，跟 baseline 报告 §4 的编号对齐
        if args.reuse_indices:
            try:
                idx_list = [int(x.strip()) for x in args.reuse_indices.split(",")
                            if x.strip()]
            except ValueError as e:
                raise ValueError(
                    f"--reuse-indices 格式错误（应为 \"2,4,10\"）: {e}"
                ) from e
            if not idx_list:
                raise ValueError("--reuse-indices 为空")
            picked: List[Dict[str, Any]] = []
            for i in idx_list:
                if not (1 <= i <= len(reuse_rows)):
                    raise ValueError(
                        f"--reuse-indices 中的 {i} 越界（JSON 共 {len(reuse_rows)} 行）"
                    )
                picked.append(reuse_rows[i - 1])
            reuse_rows = picked
            print(f"  [挑选 indices] {idx_list} → {len(reuse_rows)} 条")
        elif args.n and args.n < len(reuse_rows):
            reuse_rows = reuse_rows[: args.n]

        inputs = [
            BenchInput(
                user_input=r["user_input"],
                genre=r["genre"],
                theme=r["theme"],
                keyword_density=r["keyword_density"],
            )
            for r in reuse_rows
        ]
        print(f"[eval_clip] 复用模式 · 从 {args.reuse_poems_from} 读 {len(inputs)} 首诗 → 只换 backend 重出图")
    elif args.reuse_indices:
        raise ValueError("--reuse-indices 必须配合 --reuse-poems-from 使用")
    else:
        inputs = get_benchmark(n=args.n if args.n is not None else 10,
                               genres=args.genres, density=args.density)

    print(f"[eval_clip] 跑 {len(inputs)} 条 · image={args.image_backend}"
          + (f" · vlm-judge={args.vlm_judge}" if args.vlm_judge != "none" else ""))
    agent = _build_agent(args)

    vlm_judge = None
    if args.vlm_judge and args.vlm_judge.lower() != "none":
        try:
            vlm_judge = VLMJudge(model=args.vlm_judge)
            print(f"  [VLM oracle] 启用 {args.vlm_judge}（每条样本多调 1 次 API）")
        except Exception as e:
            print(f"  [VLM oracle] ⚠ 初始化失败，将不跑 oracle: {e}")
            vlm_judge = None

    image_save_dir: Optional[Path] = None
    if not args.no_save_images:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = _backend_short_tag(args.image_backend)
        image_save_dir = Path("outputs/eval") / f"clip_img_{ts}_{tag}"
        image_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [图像存档] {image_save_dir}（文件名 = 序号_dXXX_vXXX_主题）")

    all_rows = []
    for i, item in enumerate(inputs):
        print(f"  [{i+1}/{len(inputs)}] {item.user_input[:30]}…")
        try:
            r = _run_one(agent, item, args, vlm_judge=vlm_judge,
                         image_save_dir=image_save_dir, index=i + 1,
                         reused=(reuse_rows[i] if reuse_rows else None))
            all_rows.append(r)
            if "error" in r:
                print(f"      ⚠ {r['error']}")
            else:
                s = r["raw_scores"]
                msg = (f"      prompt_only={fmt_num(s['prompt_only'])} | "
                       f"poem_only={fmt_num(s['poem_only'])} | dual={fmt_num(s['dual'])}")
                if s.get("vlm_oracle") is not None:
                    msg += f" | vlm={fmt_num(s['vlm_oracle'])}"
                elif r.get("vlm") and r["vlm"].get("error"):
                    msg += f" | vlm ⚠ {r['vlm']['error'][:30]}"
                print(msg)
        except Exception as e:
            print(f"      ⚠ 异常：{e}")
            all_rows.append({"user_input": item.user_input, "error": str(e)})

    ok_rows = [r for r in all_rows if "error" not in r]
    md = _render_report(args, ok_rows, all_rows)
    print_and_return(md)
    paths = save_artifacts("eval_clip", {"config": vars(args), "rows": all_rows}, md)
    print(f"原始数据: {paths['json']}")
    print(f"Markdown: {paths['md']}")


if __name__ == "__main__":
    main()
