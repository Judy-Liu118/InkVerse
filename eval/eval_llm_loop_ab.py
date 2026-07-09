"""
eval.eval_llm_loop_ab -- 实验 B：LLM-driven vs 写死改图循环（同基图配对 A/B）

跑法（必须 poetry_env）：
    python -m eval.eval_llm_loop_ab --limit 1    # 冒烟：只跑执行序第 1 题
    python -m eval.eval_llm_loop_ab --resume     # 续跑全部剩余题（含首次全量）

════════════════════ 预登记（2026-07-09，跑前 commit，不得事后更改）════════════════════
问题：LLM-driven 改图循环（controller 每轮自选 edit_image(mode=rewrite_regen|edit_api)
/ refine_poem_and_regen / stop）相对写死 edit_api 循环，在同一基图、同轮数预算下，
CLIP 终值与 VLM 硬约束兑现率是否更优。

1. 题集与顺序：HARD_CONSTRAINTS 27 独立主题全集（kw v1 冻结，辅池∪主池），零挑题。
   执行顺序 = EXECUTION_ORDER（12 题材大类轮转交错，跑前写死）；截断只砍尾部，
   且尾部对题材均衡的伤害最小。
2. 双臂共用同一基图：前半段（plan→诗→关键词→标题→prompt→审校→生图→CLIP→反思）
   只跑一次后 _copy_state 分叉。前半段不开擂台、不开守擂改诗——与被测对象（改图
   循环）无关，且两臂共享，不影响配对公平。诗模型 local_lora（生产诗模型）。
3. 全程 config.CLIP_MAX_RETRIES=0：基图与循环内重生图均一次成型，两臂同规则，
   目的是让配额消耗可预算。此为实验条件与生产默认（=2）的已声明差异。
4. 预算与截断：gen=GEN_MODEL ≤100 张、edit=EDIT_MODEL ≤100 张（CLI 可给更小值）。
   每题开跑前：gen 剩余 <4 或 edit 剩余 <6（单题最坏消耗）→ 题目边界截断，
   计划 27 / 实际完成 X 如实入报。QuotaMeter 白名单断言：任何非白名单模型的图像
   API 调用立即抛异常（保护旧配额 z-image-turbo / 旧 qwen-image-edit-max）。
5. 循环参数（两臂一致）：max_image_improve_rounds=3、target_clip_score=0.30、
   adaptive_stop=True(delta=0.01)。写死臂 image_improve_mode="edit_api"；
   LLM 臂 mode 由 controller 自选。controller LLM = scorer adapter（qwen-plus）。
   两臂固定先写死后 LLM 的执行顺序。
6. 指标（全部预定，不加不减）：
   ① 每轮 CLIP raw 轨迹 + 终值配对差（mean/median/正向率）；
   ② VLM 硬约束兑现率（qwen-vl-max，kw v1）：基图 / 写死臂终图 / LLM 臂终图；
   ③ llm_loop_decisions 诚实性埋点首采：fallback 率、工具与 mode 分布、
      stale_override、score_before/after 列联；
   ④ 耗时与配额消耗。
7. 判读口径预定死：n≤27、single-shot、不称 ceiling/上限；不做显著性宣称
   （只报配对描述统计与方向）；结果无论正负如实入报告。
═══════════════════════════════════════════════════════════════════════════════════

记录增强（2026-07-09 冒烟 t01 之后追加）：仅新增审计字段与日志镜像，不改任何
实验条件 / 流程 / 指标口径：
  · shared 增记诗总分 / 艺术分 / selection_mode 与候选池全量明细（每首含小分）；
  · 两臂增记每轮中间图（images/tNN_{fixed|llm}_rK.png）与每轮双锚点 raw 分；
  · 终端输出（print + logging + 三方库）全量镜像到 run 目录 logs/run_*.log。
  t01 先于本增强完成，缺上述新字段（其余字段口径一致）。
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config as _cfg
from config import DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, STYLE_MAP
from core.agent.agent import PoetryAgent
from core.agent.autonomous import AutonomousConfig, run_image_improve_loop
from core.agent.state import AgentState, Phase
from core.models.adapter import ModelAdapter
from core.logger import get_logger

from eval.dataset import BENCHMARK
from eval.metrics import summarize, paired_delta
from eval.report import table, fmt_num
from eval.vlm_hard_constraint import HARD_CONSTRAINTS, _build_client, _check_one_image
from eval.vlm_judge import _load_prompt_yaml

_log = get_logger(__name__)

# ── 冻结实验设置（预登记条款 4/5）─────────────────────────────────────────
GEN_MODEL  = "qwen-image-2.0-pro-2026-06-22"
EDIT_MODEL = "qwen-image-edit-max-2026-01-16"
WORST_GEN_PER_THEME  = 4   # 基图 1 + LLM 臂 3 轮全选重生图
WORST_EDIT_PER_THEME = 6   # 两臂各 3 轮全选编辑

# ── 执行顺序（预登记条款 1：12 题材大类轮转交错，写死）────────────────────
EXECUTION_ORDER: List[str] = [
    # Round 1：每类第 1 题
    "写一首春景的五言绝句，要有柳树和燕子",
    "写一首五言绝句，主题是夏蝉",
    "写一首秋景的五言律诗，要有疏桐和寒蛩",
    "写一首五言绝句，主题是寒梅",
    "写一首五言绝句，主题是溪声",
    "写一首田园的七言绝句，要有耕牛和炊烟",
    "写一首边塞的五言律诗，要有戍楼和角声",
    "写一首羁旅的五言绝句，要有客舍和孤灯",
    "写一首七言绝句，主题是送别",
    "写一首怀古的五言律诗，要有古城和荒台",
    "写一首中秋的五言绝句，要有明月和团圆",
    "写一首禅意的七言绝句，要有古刹和钟鼓",
    # Round 2：每类第 2 题
    "写一首七言绝句，主题是春雨",
    "写一首五言律诗，主题是消夏",
    "写一首七言律诗，主题是悲秋",
    "写一首冬景的七言绝句，要有飞雪和寒鸦",
    "写一首山水的七言律诗，要有高楼和远山",
    "写一首五言律诗，主题是归隐",
    "写一首七言绝句，主题是征戍",
    "写一首七言律诗，主题是客愁",
    "写一首送别的七言律诗，要有长亭和折柳",
    "写一首七言律诗，主题是吊古",
    "写一首五言律诗，主题是重阳",
    "写一首五言绝句，主题是无常",
    # Round 3：余量（春/冬各自的第 3、4 题）
    "写一首春景的五言律诗，要有桃花和啼莺",
    "写一首五言律诗，主题是雪夜",
    "写一首冬景的七言律诗，要有炉火和寒灯",
]

_BENCH_BY_INPUT = {b.user_input: b for b in BENCHMARK}


class QuotaExhausted(RuntimeError):
    """预算硬熔断（题边界检查失守时的最后保险）。"""


class QuotaMeter:
    """包 ImageGenerator.generate/.edit：计数 + 模型白名单断言 + 预算硬熔断。

    计数按"发起的 API 调用"（含失败调用，保守口径）。
    白名单断言保护旧配额：本实验任何图像调用只允许 GEN_MODEL / EDIT_MODEL。
    """

    def __init__(self, image_gen, *, gen_budget: int, edit_budget: int,
                 gen_used: int = 0, edit_used: int = 0):
        self.image_gen = image_gen
        self.gen_budget, self.edit_budget = gen_budget, edit_budget
        self.gen_used, self.edit_used = gen_used, edit_used

    def install(self) -> None:
        orig_edit = self.image_gen.edit
        meter = self

        def generate(prompt, backend="local", api_key=None, api_model=None, **kw):
            if backend != "bailian" or api_model != GEN_MODEL:
                raise RuntimeError(
                    f"配额保护：非白名单生图调用 backend={backend!r} "
                    f"api_model={api_model!r}（仅允许 bailian:{GEN_MODEL}）"
                )
            if meter.gen_used >= meter.gen_budget:
                raise QuotaExhausted(f"gen 预算 {meter.gen_budget} 已耗尽")
            meter.gen_used += 1
            # 直连 bailian 路径，绕过生产的"API 失败降级本地 Z-Image"兜底：
            # 实验要求图像来源单一，API 失败就让异常冒到题级记录 error，
            # 绝不静默混入本地图。
            return meter.image_gen._generate_bailian(
                prompt, kw.get("negative_prompt"), api_key, api_model)

        def edit(image, instruction, edit_model=None, **kw):
            if edit_model != EDIT_MODEL:
                raise RuntimeError(
                    f"配额保护：非白名单编辑模型 {edit_model!r}（仅允许 {EDIT_MODEL}）"
                )
            if meter.edit_used >= meter.edit_budget:
                raise QuotaExhausted(f"edit 预算 {meter.edit_budget} 已耗尽")
            meter.edit_used += 1
            return orig_edit(image=image, instruction=instruction,
                             edit_model=edit_model, **kw)

        self.image_gen.generate = generate
        self.image_gen.edit = edit

    def theme_budget_ok(self) -> bool:
        """题边界检查（预登记条款 4）：下一题的最坏消耗是否装得下。"""
        return (self.gen_budget - self.gen_used >= WORST_GEN_PER_THEME
                and self.edit_budget - self.edit_used >= WORST_EDIT_PER_THEME)


class _Tee:
    """stdout/stderr 双写：终端照常显示，同时镜像到 run 日志（审计：终端可见即落盘）。"""

    def __init__(self, primary, mirror):
        self._primary, self._mirror = primary, mirror

    def write(self, data):
        try:
            self._primary.write(data)
        except UnicodeEncodeError:
            # 终端编码非 UTF-8（如 GBK console）时降级替换，不让日志镜像拖垮主流程
            enc = getattr(self._primary, "encoding", None) or "utf-8"
            self._primary.write(data.encode(enc, errors="replace").decode(enc))
        try:
            self._mirror.write(data)
        except Exception:
            pass

    def flush(self):
        self._primary.flush()
        try:
            self._mirror.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._primary, "isatty", lambda: False)()

    def fileno(self):
        return self._primary.fileno()

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", "utf-8")


def _attach_run_log(log_path: Path):
    """终端输出全量镜像到 run 目录日志。

    print / 三方库输出走 sys.stdout|stderr 的 Tee；inkverse logging 的控制台
    handler 绑的是替换前的原始 stdout（不经 Tee），故另挂一个 DEBUG 级
    StreamHandler 写同一文件，时间序自然交错。
    """
    f = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    fh = logging.StreamHandler(f)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s | %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger("inkverse").addHandler(fh)


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


def _run_shared_stage(agent: PoetryAgent, user_input: str) -> AgentState:
    """前半段（两臂共享）：plan→诗→关键词→标题→prompt→审校→生图→CLIP→反思。"""
    state = AgentState(
        user_input=user_input, lang="英文",
        style_suffix=STYLE_MAP["水墨画"],
        image_backend="bailian",
        image_api_key=DASHSCOPE_API_KEY,
        image_api_model=GEN_MODEL,
    )
    for ph in ("_phase_plan", "_phase_poem", "_phase_keyword_extract",
               "_phase_title", "_phase_prompt", "_phase_prompt_review"):
        state = getattr(agent, ph)(state)
        if state.phase == Phase.ERROR:
            return state
    state = agent._phase_image_clip(state)
    if state.phase == Phase.ERROR or state.image is None:
        return state
    return agent._phase_reflect(state)


_ROUND_SCORE_RE = re.compile(r"CLIP raw=(-?\d+(?:\.\d+)?)")


def _extract_round_scores(final_state: AgentState, base_trace_len: int) -> List[Optional[float]]:
    """从循环期间新增的 trace 提取每轮完成时的 CLIP raw。"""
    scores: List[Optional[float]] = []
    for s in final_state.trace[base_trace_len:]:
        action = s.action or ""
        if "轮完成" in action and ("改图循环：第" in action or "LLM 改图：第" in action):
            m = _ROUND_SCORE_RE.search(s.result or "")
            scores.append(float(m.group(1)) if m else None)
    return scores


def _raw_anchor(norm: float) -> Optional[float]:
    """归一化锚点分（0~1）还原为 CLIP raw 余弦（与 agent._raw_clip 同口径）。"""
    return round(norm * 2 - 1, 3) if norm else None


def _run_arm(agent: PoetryAgent, base_state: AgentState, *,
             llm_driven: bool, args, meter: QuotaMeter,
             img_dir: Path, idx: int, arm_name: str) -> Dict[str, Any]:
    state = agent._copy_state(base_state)
    state.llm_loop_decisions = []   # _copy_state 浅拷贝不含此字段，显式断开共享
    base_trace_len = len(state.trace)
    config = AutonomousConfig(
        target_clip_score=args.target,
        max_image_improve_rounds=args.max_rounds,
        allow_poem_refine=False,
        image_improve_mode="edit_api",     # 写死臂模式；LLM 臂 mode 由 controller 自选
        edit_model=EDIT_MODEL,
        image_loop_llm_driven=llm_driven,
    )
    g0, e0 = meter.gen_used, meter.edit_used
    t0 = time.time()
    final = state
    rounds: List[Dict[str, Any]] = []   # 每轮审计：中间图 + 双锚点 raw（记录增强）
    for s in run_image_improve_loop(agent, state, config, target=args.target):
        final = s
        n = len(_extract_round_scores(s, base_trace_len))
        # 只在"新一轮完成"的 yield 上存图；LLM stop 的 yield 与收尾 yield
        # （历史最优回写，同 final_image）都不新增轮次，自然跳过。
        if n > len(rounds) and s.image is not None:
            rounds.append({
                "round":           n,
                "clip_raw":        agent._raw_clip(s),
                "clip_raw_poem":   _raw_anchor(s.clip_score_poem),
                "clip_raw_prompt": _raw_anchor(s.clip_score_prompt),
                "image": _save_image(
                    s.image, img_dir / f"t{idx:02d}_{arm_name}_r{n}.png"),
            })
    out: Dict[str, Any] = {
        "clip_raw_base":  agent._raw_clip(base_state),
        "clip_raw_final": agent._raw_clip(final),
        "round_scores":   _extract_round_scores(final, base_trace_len),
        "rounds":         rounds,
        "gen_used":       meter.gen_used - g0,
        "edit_used":      meter.edit_used - e0,
        "elapsed_sec":    round(time.time() - t0, 1),
        "poem_final":     final.poem,
        "poem_score_final": final.best_poem_score,  # LLM 臂改诗后会变，写死臂恒等于基线
    }
    if llm_driven:
        out["decisions"] = list(final.llm_loop_decisions)
    out["_image"] = final.image   # 落盘后从记录里剔除
    return out


def _vlm_check(client, prompt, model: str, image, poem: str, kw: List[str]) -> Dict[str, Any]:
    r = _check_one_image(client, model, prompt, image=image, poem=poem,
                         keywords=kw, max_tokens=600, temperature=0.0)
    results = r.get("results") or []
    return {
        "hit":     sum(1 for x in results if x.get("present") is True),
        "total":   len(results) if not r.get("error") else None,
        "detail":  results,
        "error":   r.get("error"),
    }


def _load_previous(jsonl_path: Path):
    """resume：已完成 user_input 集合 + 上次结束时的配额累计。"""
    done, gen_used, edit_used = set(), 0, 0
    if jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done.add(rec["user_input"])
            q = rec.get("quota_after") or {}
            gen_used = q.get("gen_used", gen_used)
            edit_used = q.get("edit_used", edit_used)
    return done, gen_used, edit_used


def _save_image(image, path: Path) -> Optional[str]:
    if image is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path.as_posix()


# ── 汇总（描述统计，预登记条款 6/7 口径）─────────────────────────────────
def _render_summary(records: List[Dict[str, Any]], meter: QuotaMeter, args) -> str:
    ok = [r for r in records if "error" not in r]
    md: List[str] = []
    md.append("# 实验 B 过程汇总 · LLM-driven vs 写死改图循环（同基图配对）")
    md.append(f"_计划 {len(EXECUTION_ORDER)} 题 / 完成 {len(ok)} 题（含历史续跑）· "
              f"single-shot · gen={meter.gen_used}/{meter.gen_budget} · "
              f"edit={meter.edit_used}/{meter.edit_budget} · "
              f"max_rounds={args.max_rounds} · target={args.target}_")
    md.append("")
    if not ok:
        return "\n".join(md + ["无可用数据。"])

    fixed = [r["arm_fixed"]["clip_raw_final"] for r in ok]
    llm   = [r["arm_llm"]["clip_raw_final"] for r in ok]
    base  = [r["arm_fixed"]["clip_raw_base"] for r in ok]
    md.append("## 1. CLIP raw 终值（配对描述统计，不做显著性宣称）")
    rows = []
    for name, xs in (("基图", base), ("写死臂", fixed), ("LLM 臂", llm)):
        s = summarize(xs)
        rows.append([name, s["n"], fmt_num(s["mean"]), fmt_num(s["std"]),
                     fmt_num(s["median"]), fmt_num(s["min"]), fmt_num(s["max"])])
    md.append(table(["臂", "n", "mean", "std", "median", "min", "max"], rows))
    d = paired_delta(fixed, llm)
    md.append("")
    md.append(f"**LLM 臂 − 写死臂**：mean Δ={fmt_num(d['mean_delta'], 4)} · "
              f"median Δ={fmt_num(d['median_delta'], 4)} · "
              f"LLM 臂更高比例={d['positive_rate']:.1%}（n={d['n']}）")
    md.append("")

    md.append("## 2. VLM 硬约束兑现率（kw v1 · qwen-vl-max）")
    def _hit(key):
        h = t = 0
        for r in ok:
            v = (r.get("vlm") or {}).get(key) or {}
            if v.get("error") or v.get("total") is None:
                continue
            h += v["hit"]; t += v["total"]
        return h, t
    rows = []
    for name, key in (("基图", "base"), ("写死臂终图", "fixed"), ("LLM 臂终图", "llm")):
        h, t = _hit(key)
        rows.append([name, h, t, f"{h/t:.1%}" if t else "—"])
    md.append(table(["图", "hit", "total", "兑现率"], rows))
    md.append("")

    md.append("## 3. LLM 臂决策埋点（llm_loop_decisions 首采）")
    decisions = [d_ for r in ok for d_ in (r["arm_llm"].get("decisions") or [])]
    total = len(decisions)
    if total:
        fb = sum(1 for d_ in decisions if d_.get("is_fallback"))
        so = sum(1 for d_ in decisions if d_.get("stale_override"))
        hist: Dict[str, int] = {}
        for d_ in decisions:
            key = d_.get("tool") or "unknown"
            if key == "edit_image" and d_.get("mode"):
                key += f"({d_['mode']})"
            hist[key] = hist.get(key, 0) + 1
        md.append(f"总决策 {total} · fallback {fb}（{fb/total:.1%}）· stale_override {so}")
        md.append("")
        md.append(table(["tool", "次数", "占比"],
                        [[k, v, f"{v/total:.1%}"]
                         for k, v in sorted(hist.items(), key=lambda x: -x[1])]))
    else:
        md.append("_无决策数据_")
    md.append("")

    md.append("## 4. 成本")
    ef = summarize([r["arm_fixed"]["elapsed_sec"] for r in ok])
    el = summarize([r["arm_llm"]["elapsed_sec"] for r in ok])
    md.append(table(
        ["指标", "写死臂", "LLM 臂"],
        [["平均循环耗时 (s)", fmt_num(ef["mean"], 1), fmt_num(el["mean"], 1)],
         ["gen 消耗合计", sum(r["arm_fixed"]["gen_used"] for r in ok),
          sum(r["arm_llm"]["gen_used"] for r in ok)],
         ["edit 消耗合计", sum(r["arm_fixed"]["edit_used"] for r in ok),
          sum(r["arm_llm"]["edit_used"] for r in ok)]],
    ))
    md.append("")
    md.append("> 判读口径（预登记）：n≤27、single-shot、CLIP_MAX_RETRIES=0 为实验条件；"
              "不称 ceiling/上限，不做显著性宣称。正式报告另行撰写。")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser(description="实验 B：LLM-driven vs 写死改图循环（同基图配对 A/B）")
    ap.add_argument("--gen-budget",  type=int, default=100)
    ap.add_argument("--edit-budget", type=int, default=100)
    ap.add_argument("--target",      type=float, default=0.30)
    ap.add_argument("--max-rounds",  type=int, default=3)
    ap.add_argument("--poem-model",   default="local_lora")
    ap.add_argument("--prompt-model", default="qwen-max")
    ap.add_argument("--scorer",       default="qwen-plus")
    ap.add_argument("--vlm",          default="qwen-vl-max")
    ap.add_argument("--skip-vlm",     action="store_true")
    ap.add_argument("--limit",        type=int, default=None,
                    help="本次最多跑几题（冒烟用），按执行序取剩余题的前 N")
    ap.add_argument("--resume",       action="store_true",
                    help="跳过 results.jsonl 中已完成的题并继承配额累计")
    ap.add_argument("--run-dir",      default="eval/runs/llm_loop_ab")
    args = ap.parse_args()

    # 题集完整性断言（预登记条款 1）
    assert set(EXECUTION_ORDER) == set(HARD_CONSTRAINTS), "EXECUTION_ORDER 必须恰为 kw v1 全集"
    assert all(u in _BENCH_BY_INPUT for u in EXECUTION_ORDER), "存在 benchmark 之外的题"

    # 预登记条款 3：全程关 CLIP 生图重试（_phase_image_clip 每次调用时读该模块属性）
    _cfg.CLIP_MAX_RETRIES = 0

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / "logs" / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    _attach_run_log(run_log)
    print(f"[llm_loop_ab] 运行日志（终端全量镜像）: {run_log}")
    jsonl_path = run_dir / "results.jsonl"
    done, gen_used0, edit_used0 = (_load_previous(jsonl_path) if args.resume
                                   else (set(), 0, 0))
    if not args.resume and jsonl_path.is_file():
        raise SystemExit(f"{jsonl_path} 已存在：续跑请加 --resume，重开请换 --run-dir")

    todo = [u for u in EXECUTION_ORDER if u not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"[llm_loop_ab] 计划 {len(EXECUTION_ORDER)} 题 · 已完成 {len(done)} · 本次 {len(todo)}")
    print(f"[llm_loop_ab] gen={GEN_MODEL}（预算 {args.gen_budget}，已用 {gen_used0}） · "
          f"edit={EDIT_MODEL}（预算 {args.edit_budget}，已用 {edit_used0}） · "
          f"CLIP_MAX_RETRIES=0")

    agent = _build_agent(args)
    meter = QuotaMeter(agent.image_gen, gen_budget=args.gen_budget,
                       edit_budget=args.edit_budget,
                       gen_used=gen_used0, edit_used=edit_used0)
    meter.install()

    vlm_client = vlm_prompt = None
    if not args.skip_vlm:
        vlm_prompt = _load_prompt_yaml("vlm_hard_constraint.yaml")
        vlm_client = _build_client(args.vlm)

    records: List[Dict[str, Any]] = []
    for user_input in todo:
        idx = EXECUTION_ORDER.index(user_input) + 1
        if not meter.theme_budget_ok():
            print(f"[llm_loop_ab] ✋ 题边界截断（预登记规则）：gen 剩余 "
                  f"{meter.gen_budget - meter.gen_used} < {WORST_GEN_PER_THEME} 或 "
                  f"edit 剩余 {meter.edit_budget - meter.edit_used} < {WORST_EDIT_PER_THEME}")
            break
        item = _BENCH_BY_INPUT[user_input]
        kw = HARD_CONSTRAINTS[user_input]["kw"]
        print(f"  [{idx:02d}/27] {user_input}")
        rec: Dict[str, Any] = {
            "idx": idx, "user_input": user_input,
            "genre": item.genre, "theme": item.theme,
            "density": item.keyword_density, "kw": kw,
        }
        try:
            base = _run_shared_stage(agent, user_input)
            if base.phase == Phase.ERROR or base.image is None:
                rec["error"] = f"shared stage: {base.error or 'no image'}"
            else:
                img_dir = run_dir / "images"
                rec["shared"] = {
                    "poem": base.poem, "title": base.title,
                    "poem_score": {
                        "total":          base.best_poem_score,
                        "art_quality":    base.best_poem_art_quality,
                        "selection_mode": base.poem_selection_mode,
                    },
                    # 候选池全量明细（每首含小分 dict），供审计生成-筛选过程
                    "poem_candidates": {
                        "qualified": base.qualified_candidates,
                        "rejected":  base.rejected_candidates,
                    },
                    "prompt": base.prompt,
                    "visual_keywords_en": base.visual_keywords_en,
                    "clip_raw_base": agent._raw_clip(base),
                    "clip_raw_poem":   _raw_anchor(base.clip_score_poem),
                    "clip_raw_prompt": _raw_anchor(base.clip_score_prompt),
                    "base_image": _save_image(base.image, img_dir / f"t{idx:02d}_base.png"),
                }
                # 两臂固定顺序：先写死后 LLM（预登记条款 5）
                arm_fixed = _run_arm(agent, base, llm_driven=False, args=args,
                                     meter=meter, img_dir=img_dir, idx=idx,
                                     arm_name="fixed")
                arm_llm   = _run_arm(agent, base, llm_driven=True,  args=args,
                                     meter=meter, img_dir=img_dir, idx=idx,
                                     arm_name="llm")
                arm_fixed["final_image"] = _save_image(
                    arm_fixed.pop("_image"), img_dir / f"t{idx:02d}_fixed.png")
                arm_llm["final_image"] = _save_image(
                    arm_llm.pop("_image"), img_dir / f"t{idx:02d}_llm.png")
                rec["arm_fixed"], rec["arm_llm"] = arm_fixed, arm_llm

                if vlm_client is not None:
                    from PIL import Image as _PILImage
                    # poem 仅作判定上下文，kw 才是判定对象；LLM 臂可能改过诗，
                    # 用各臂自己的最终诗作上下文才如实。
                    def _vlm_on(path, poem):
                        if not path:
                            return {"error": "no image"}
                        return _vlm_check(vlm_client, vlm_prompt, args.vlm,
                                          _PILImage.open(path), poem, kw)
                    rec["vlm"] = {
                        "model": args.vlm,
                        "base":  _vlm_on(rec["shared"]["base_image"], base.poem),
                        "fixed": _vlm_on(arm_fixed["final_image"], arm_fixed["poem_final"]),
                        "llm":   _vlm_on(arm_llm["final_image"], arm_llm["poem_final"]),
                    }
                print(f"      base={fmt_num(rec['shared']['clip_raw_base'])} → "
                      f"fixed={fmt_num(arm_fixed['clip_raw_final'])}"
                      f"（edit {arm_fixed['edit_used']}/gen {arm_fixed['gen_used']}） · "
                      f"llm={fmt_num(arm_llm['clip_raw_final'])}"
                      f"（edit {arm_llm['edit_used']}/gen {arm_llm['gen_used']}，"
                      f"决策 {len(arm_llm.get('decisions') or [])}）")
        except QuotaExhausted as e:
            rec["error"] = f"quota: {e}"
            print(f"      ⚠ {e}（题边界检查失守，硬熔断兜底）")
        except Exception as e:
            rec["error"] = str(e)
            print(f"      ⚠ 异常：{e}")
        rec["quota_after"] = {"gen_used": meter.gen_used, "edit_used": meter.edit_used}
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        records.append(rec)
        if "error" in rec and "quota" in str(rec.get("error", "")):
            break

    all_records = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            all_records.append(json.loads(line))
    md = _render_summary(all_records, meter, args)
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"\n原始数据: {jsonl_path}\n过程汇总: {run_dir / 'summary.md'}")
    print(f"配额消耗: gen={meter.gen_used}/{meter.gen_budget} · "
          f"edit={meter.edit_used}/{meter.edit_budget}")


if __name__ == "__main__":
    main()
