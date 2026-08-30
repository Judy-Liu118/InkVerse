"""诗侧闸门分解：把擂台消融的 +18.2pp 拆成「诗侧覆盖」×「诗→图传导」两段。

背景：`REPORT_arena_ablation_20260701.md` 的干预打在诗生成阶段（擂台），
测量却取在图像末端（VLM 硬约束命中率），中间隔着「诗→提示词→图」两级
转换。因果链的中间节点从未被测过，导致两类性质相反的失败被混在同一个
命中率里：
  · 诗里就没写 → 该修诗侧
  · 诗里写了但画没画出来 → 该修图侧

本脚本补上中间节点。核心指标是「用户点名的物体在不在诗里」(`in_poem`)，
它对擂台是**完全外生**的，三条依据：
  1. 切题分在擂台内是场内公共常数，擂主与挑战者共用、相减即消
     （`core/agent/poem_refiner.py:213-224` 注释明写）
  2. 切题 rubric 里一个字没提具名物体（`core/poem/scorer.py:_TOPIC_FIT_PROMPT`）
  3. 写点评 / 提修改方向 / 写挑战诗三个环节全都拿不到 user_request
     （`poem_refiner.py` 的 `_auto_poem_critique` / `_auto_poem_feedback` /
      `_CHALLENGER_PROMPT`）
擂台从未优化过这个量 → 拿它当尺子不循环。对比之下，报告 §8 那条
「第三条正交证据线」(armB 12:5) 用的是 `PoemScorer.compare_poems` +
qwen-plus，与擂台守擂判定同函数同提示词同模型，是拿训练集当测试集。

⚠ 冻结契约：本脚本**不修改** `eval/vlm_hard_constraint.py` 的 HARD_CONSTRAINTS
（v1 定义仍是唯一正式口径，`eval/METHODOLOGY.md` §冻结条款）。下方
POEM_TOKENS 是本探针私有的诗侧判定口径。

零 API 调用、零生图配额——诗与判定全部读自已落盘 JSON。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.vlm_hard_constraint import HARD_CONSTRAINTS  # noqa: E402
from eval._analyze_arena_ablation_significance import (  # noqa: E402
    BOOTSTRAP_ITERS, BOOTSTRAP_SEED, bootstrap_ci_theme_level, mcnemar_exact_p,
)
from eval.report import save_artifacts, table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARMS = ["arm_A", "arm_B"]
DEFAULT_MAIN = "outputs/eval/vlm_hard_constraint_arm_ab_main_n22_20260702_133542.json"
DEFAULT_CROSS = "outputs/eval/vlm_hard_constraint_arm_ab_main_n22_glm4v_20260706_000935.json"

# ── 诗侧判定口径（本探针私有，与 HARD_CONSTRAINTS 的 kw 逐条 1:1）──────────
#
# pattern = 该 kw 在诗中可能出现的核心字（正则 alternation，直接在诗文上搜）。
# 口径原则：
#   · 只降到「可画的本体字」，剥掉不可视觉判定的修饰语（疏桐→桐、寒蛩→蛩）
#   · 收录常见代称（明月→魄/蟾/婵娟；古刹→寺/塔/禅）——否则会把「素魄出东楼」
#     误判成诗里没写月
#   · 不收上位词（耕牛只认「牛」，不认「耕」——arm A 的「暮年犹得伴耕深」
#     有耕无牛，正是该判 False 的情形）
#
# abstract=True 的 kw 是**声音或抽象概念**，物理上无法作为物体写进诗、也无法
# 字面判定「在不在诗里」。单列上报，不并入主表分母。
POEM_TOKENS: Dict[str, List[Dict[str, Any]]] = {
    "写一首秋景的五言律诗，要有疏桐和寒蛩": [
        {"kw": "桐树", "pattern": "桐"},
        {"kw": "蛩（蟋蟀）", "pattern": "蛩|蟋蟀"},
    ],
    "写一首冬景的七言绝句，要有飞雪和寒鸦": [
        {"kw": "雪", "pattern": "雪"},
        {"kw": "寒鸦（乌鸦）", "pattern": "鸦"},
    ],
    "写一首冬景的七言律诗，要有炉火和寒灯": [
        {"kw": "炉火（火焰）", "pattern": "炉|火"},
        {"kw": "寒灯（灯）", "pattern": "灯|烛"},
    ],
    "写一首山水的七言律诗，要有高楼和远山": [
        {"kw": "高楼", "pattern": "楼|阁|塔"},
        {"kw": "远山", "pattern": "山|峰|岭|嶂|峦"},
    ],
    "写一首田园的七言绝句，要有耕牛和炊烟": [
        {"kw": "耕牛", "pattern": "牛"},
        {"kw": "炊烟", "pattern": "炊|烟"},
    ],
    "写一首边塞的五言律诗，要有戍楼和角声": [
        {"kw": "戍楼（烽火台/城楼/箭楼）", "pattern": "戍楼|烽|城楼|谯楼|楼"},
        {"kw": "边塞角声场景（军旗/号角/军营）", "pattern": "角|旗|旆|营|军",
         "abstract": True},
    ],
    "写一首羁旅的五言绝句，要有客舍和孤灯": [
        {"kw": "客舍", "pattern": "客舍|客家|旅舍|逆旅|客店|驿"},
        {"kw": "孤灯", "pattern": "灯|烛"},
    ],
    "写一首送别的七言律诗，要有长亭和折柳": [
        {"kw": "长亭", "pattern": "亭"},
        {"kw": "柳枝（折柳）", "pattern": "柳"},
    ],
    "写一首怀古的五言律诗，要有古城和荒台": [
        {"kw": "古城（残城/城墙）", "pattern": "城|堞|垒|郭|雉"},
        {"kw": "荒台（废弃高台/遗迹）", "pattern": "台"},
    ],
    "写一首中秋的五言绝句，要有明月和团圆": [
        {"kw": "明月（圆月）", "pattern": "月|魄|蟾|婵娟"},
        {"kw": "团圆场景（家人聚会/圆桌）", "pattern": "团圆|圆|聚|宴",
         "abstract": True},
    ],
    "写一首禅意的七言绝句，要有古刹和钟鼓": [
        {"kw": "古刹（寺庙/佛塔）", "pattern": "刹|寺|塔|禅|僧|庵|伽蓝"},
        {"kw": "钟鼓（钟或鼓）", "pattern": "钟|鼓"},
    ],
}


class Cell:
    """一个 (主题, arm, kw) 格子的诗侧/图侧双判定。"""

    __slots__ = ("theme", "arm", "kw_idx", "kw", "in_poem", "in_image",
                 "abstract", "poem")

    def __init__(self, theme, arm, kw_idx, kw, in_poem, in_image, abstract, poem):
        self.theme, self.arm, self.kw_idx, self.kw = theme, arm, kw_idx, kw
        self.in_poem, self.in_image = in_poem, in_image
        self.abstract, self.poem = abstract, poem


def _validate_tokens() -> List[str]:
    """POEM_TOKENS 与冻结版 HARD_CONSTRAINTS 逐条 1:1 校验。"""
    themes = []
    for theme, specs in POEM_TOKENS.items():
        if theme not in HARD_CONSTRAINTS:
            raise SystemExit(f"POEM_TOKENS 含 HARD_CONSTRAINTS 未定义的主题：{theme}")
        hc = HARD_CONSTRAINTS[theme]
        if hc["density"] != "rich":
            raise SystemExit(f"{theme}: 非 rich 主题不应进入本分析")
        if len(specs) != len(hc["kw"]):
            raise SystemExit(f"{theme}: kw 数与冻结版不一致")
        for spec, kw in zip(specs, hc["kw"]):
            if spec["kw"] != kw:
                raise SystemExit(f"{theme}: kw 文本不一致 {spec['kw']!r} != {kw!r}")
        themes.append(theme)
    return themes


def _build_cells(by_theme: Dict[str, Any], themes: List[str]) -> List[Cell]:
    cells = []
    for theme in themes:
        specs = POEM_TOKENS[theme]
        for arm in ARMS:
            entry = by_theme[theme][arm]
            poem = (entry.get("poem") or "").strip()
            results = entry.get("results", [])
            for i, spec in enumerate(specs):
                if i >= len(results):
                    raise SystemExit(f"{theme}/{arm}: 判定结果缺 kw#{i}")
                cells.append(Cell(
                    theme=theme, arm=arm, kw_idx=i, kw=spec["kw"],
                    in_poem=bool(re.search(spec["pattern"], poem)),
                    in_image=bool(results[i].get("present") is True),
                    abstract=bool(spec.get("abstract")),
                    poem=poem,
                ))
    return cells


def _matrix(cells: List[Cell]) -> Dict[str, int]:
    """2×2 传导矩阵。"""
    m = {"pp": 0, "pn": 0, "np": 0, "nn": 0}
    for c in cells:
        key = ("p" if c.in_poem else "n") + ("p" if c.in_image else "n")
        m[key] += 1
    return m


def _rate(num: int, den: int) -> str:
    return f"{num}/{den} = {num / den:.1%}" if den else f"{num}/0 = —"


def _theme_pairs(cells: List[Cell], field: str) -> list:
    """[(theme, [(kw_idx, val_A, val_B), ...]), ...]，供 bootstrap 复用。"""
    by = {}
    for c in cells:
        by.setdefault(c.theme, {}).setdefault(c.kw_idx, {})[c.arm] = getattr(c, field)
    out = []
    for theme, kws in by.items():
        pairs = [(i, bool(v["arm_A"]), bool(v["arm_B"]))
                 for i, v in sorted(kws.items()) if len(v) == 2]
        out.append((theme, pairs))
    return out


def _arm_stats(cells: List[Cell], field: str) -> Dict[str, Any]:
    tp = _theme_pairs(cells, field)
    n = sum(len(p) for _, p in tp)
    a = sum(1 for _, p in tp for _, x, _ in p if x)
    b = sum(1 for _, p in tp for _, _, y in p if y)
    a_only = sum(1 for _, p in tp for _, x, y in p if x and not y)
    b_only = sum(1 for _, p in tp for _, x, y in p if y and not x)
    lo, hi = bootstrap_ci_theme_level(tp, BOOTSTRAP_ITERS, BOOTSTRAP_SEED)
    return {"n": n, "hit_a": a, "hit_b": b, "delta_pp": (b - a) / n * 100 if n else 0.0,
            "a_only": a_only, "b_only": b_only,
            "mcnemar_p": mcnemar_exact_p(a_only, b_only), "ci_lo": lo, "ci_hi": hi}


def _decompose(cells: List[Cell]) -> Dict[str, Dict[str, int]]:
    """表 C：图侧净增益按「诗侧覆盖是否变化」分层。"""
    by = {}
    for c in cells:
        by.setdefault((c.theme, c.kw_idx), {})[c.arm] = c
    strata = {
        "诗侧覆盖上升（A无→B有）": {"n": 0, "img_a": 0, "img_b": 0},
        "诗侧覆盖下降（A有→B无）": {"n": 0, "img_a": 0, "img_b": 0},
        "诗侧两臂都有":            {"n": 0, "img_a": 0, "img_b": 0},
        "诗侧两臂都无":            {"n": 0, "img_a": 0, "img_b": 0},
    }
    for pair in by.values():
        if len(pair) != 2:
            continue
        ca, cb = pair["arm_A"], pair["arm_B"]
        if cb.in_poem and not ca.in_poem:
            k = "诗侧覆盖上升（A无→B有）"
        elif ca.in_poem and not cb.in_poem:
            k = "诗侧覆盖下降（A有→B无）"
        elif ca.in_poem:
            k = "诗侧两臂都有"
        else:
            k = "诗侧两臂都无"
        strata[k]["n"] += 1
        strata[k]["img_a"] += int(ca.in_image)
        strata[k]["img_b"] += int(cb.in_image)
    return strata


def _render(payload: Dict[str, Any]) -> str:
    meta = payload["meta"]
    md: List[str] = []
    md.append("# 诗侧闸门分解（擂台消融 §2.8 备料）")
    md.append("")
    md.append(f"数据源：`{meta['main_json']}`（主 oracle `{meta['main_model']}`）"
              + (f" · 交叉核验 `{meta['cross_json']}`（`{meta['cross_model']}`）"
                 if meta.get("cross_json") else ""))
    md.append(f"零 API 调用 · 生成时间 {meta['timestamp']}")
    md.append("")
    md.append("## §0 口径")
    md.append("")
    md.append(f"- **纳入**：主池 **rich** 主题（用户显式说「要有 X 和 Y」）"
              f"{meta['n_themes']} 题 × 2 kw × 2 arm = **{meta['n_cells_all']} 格**")
    md.append(f"- **主表分母**：剔除 {meta['n_abstract_kw']} 个抽象/声音类 kw"
              f"（{'、'.join(meta['abstract_kw'])}）后 = **{meta['n_cells_main']} 格**"
              f"（{meta['n_cells_main'] // 2} kw × 2 arm）")
    md.append(f"- **排除**：11 个 sparse 场景类 OR 集合 kw——「在不在诗里」无法字面判定")
    md.append("- ⚠ **本节分母与 §2.3 的 33 不是同一个口径**，勿混用")
    md.append("- `in_poem` 判定表见脚本 `eval/_analyze_poem_gate_coverage.py` 的 "
              "`POEM_TOKENS`（本探针私有，冻结版 `HARD_CONSTRAINTS` 未改动）")
    md.append("")

    md.append("## 表 A · 诗侧覆盖率（擂台的直接效应）")
    md.append("")
    md.append("> 这是对擂台**外生**的指标：擂台的目标函数里不含它（切题分在擂台内是"
              "常数、rubric 不提具名物体、三个生成环节都拿不到 user_request）。")
    md.append("")
    s = payload["table_a"]
    md.append(table(
        ["指标", "arm A", "arm B", "Δ(B−A)", "A-only:B-only", "McNemar p",
         "bootstrap 95% CI"],
        [["**诗侧覆盖**", _rate(s["hit_a"], s["n"]), _rate(s["hit_b"], s["n"]),
          f"**{s['delta_pp']:+.1f}pp**", f"{s['a_only']}:{s['b_only']}",
          f"{s['mcnemar_p']:.3f}", f"[{s['ci_lo']:+.1f}, {s['ci_hi']:+.1f}]pp"],
         ["图侧命中（同口径对照）",
          _rate(payload["table_a_img"]["hit_a"], payload["table_a_img"]["n"]),
          _rate(payload["table_a_img"]["hit_b"], payload["table_a_img"]["n"]),
          f"{payload['table_a_img']['delta_pp']:+.1f}pp",
          f"{payload['table_a_img']['a_only']}:{payload['table_a_img']['b_only']}",
          f"{payload['table_a_img']['mcnemar_p']:.3f}",
          f"[{payload['table_a_img']['ci_lo']:+.1f}, "
          f"{payload['table_a_img']['ci_hi']:+.1f}]pp"]],
    ))
    md.append("")
    md.append(f"_bootstrap {BOOTSTRAP_ITERS} 次重抽 · seed={BOOTSTRAP_SEED}"
              f"（与 §2.5 同口径，主题级重抽）_")
    md.append("")

    md.append("## 表 B · 2×2 传导矩阵")
    md.append("")
    for label, m in payload["table_b"].items():
        tot = sum(m.values())
        md.append(f"### {label}（n={tot}）")
        md.append("")
        md.append(table(
            ["", "画里有", "画里没有", "小计"],
            [["**诗里写了**", m["pp"], f"{m['pn']} ← 图像侧失败", m["pp"] + m["pn"]],
             ["**诗里没写**", f"{m['np']} ← 画图模型自行添加",
              f"{m['nn']} ← 诗侧失败", m["np"] + m["nn"]],
             ["小计", m["pp"] + m["np"], m["pn"] + m["nn"], tot]],
        ))
        md.append("")
        md.append(f"- **传导率 P(画中有 | 诗中有)** = {_rate(m['pp'], m['pp'] + m['pn'])}")
        md.append(f"- **P(画中有 | 诗中无)** = {_rate(m['np'], m['np'] + m['nn'])}")
        md.append("")

    md.append("## 表 C · 图侧净增益按诗侧覆盖变化分层")
    md.append("")
    rows = []
    for k, v in payload["table_c"].items():
        rows.append([k, v["n"], f"{v['img_a']}/{v['n']}", f"{v['img_b']}/{v['n']}",
                     f"{v['img_b'] - v['img_a']:+d}"])
    tot_n = sum(v["n"] for v in payload["table_c"].values())
    tot_d = sum(v["img_b"] - v["img_a"] for v in payload["table_c"].values())
    rows.append(["**合计**", tot_n, "—", "—", f"**{tot_d:+d}**"])
    md.append(table(["诗侧分层", "kw 对数", "arm A 图侧命中", "arm B 图侧命中",
                     "图侧 Δ"], rows))
    md.append("")

    md.append("## 附 · 单列的抽象/声音类 kw")
    md.append("")
    md.append("这两个 kw 物理上无法作为物体写进诗，`in_poem` 判定不可靠，"
              "故不并入主表分母：")
    md.append("")
    md.append(table(["主题", "kw", "arm", "诗里", "画里"],
                    payload["abstract_rows"]))
    md.append("")

    md.append("## 附 · 逐格明细（主表 {} 格）".format(meta["n_cells_main"]))
    md.append("")
    md.append(table(["主题", "kw", "arm", "诗里", "画里", "格"],
                    payload["detail_rows"]))
    return "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", default=DEFAULT_MAIN)
    ap.add_argument("--cross", default=DEFAULT_CROSS,
                    help="交叉核验用的第二 oracle JSON；传 none 跳过")
    args = ap.parse_args()

    themes = _validate_tokens()
    main_doc = json.loads((ROOT / args.main).read_text(encoding="utf-8"))
    cells_all = _build_cells(main_doc["by_theme"], themes)
    cells = [c for c in cells_all if not c.abstract]
    abstract_cells = [c for c in cells_all if c.abstract]

    payload: Dict[str, Any] = {
        "table_a": _arm_stats(cells, "in_poem"),
        "table_a_img": _arm_stats(cells, "in_image"),
        "table_b": {f"{args.main.split('/')[-1].split('_')[-2]} · 两臂合并":
                    _matrix(cells)},
        "table_c": _decompose(cells),
    }
    main_model = main_doc["meta"].get("vlm_model", "?")
    payload["table_b"] = {f"{main_model} · 两臂合并": _matrix(cells)}
    for arm in ARMS:
        payload["table_b"][f"{main_model} · {arm}"] = _matrix(
            [c for c in cells if c.arm == arm])

    cross_model = None
    if args.cross and args.cross.lower() != "none":
        cross_doc = json.loads((ROOT / args.cross).read_text(encoding="utf-8"))
        cross_cells = [c for c in _build_cells(cross_doc["by_theme"], themes)
                       if not c.abstract]
        cross_model = cross_doc["meta"].get("vlm_model", "?")
        payload["table_b"][f"{cross_model} · 两臂合并（交叉核验）"] = _matrix(cross_cells)

    payload["detail_rows"] = [
        [c.theme[6:16] + "…", c.kw[:14], c.arm[-1],
         "✓" if c.in_poem else "✗", "✓" if c.in_image else "✗",
         ("诗有画有" if c.in_poem and c.in_image else
          "诗有画无 ←图侧失败" if c.in_poem else
          "诗无画有 ←自行添加" if c.in_image else "诗无画无 ←诗侧失败")]
        for c in cells
    ]
    payload["abstract_rows"] = [
        [c.theme[6:16] + "…", c.kw[:16], c.arm[-1],
         "✓" if c.in_poem else "✗", "✓" if c.in_image else "✗"]
        for c in abstract_cells
    ]
    payload["meta"] = {
        "main_json": args.main, "main_model": main_model,
        "cross_json": (args.cross if cross_model else None),
        "cross_model": cross_model,
        "n_themes": len(themes),
        "n_cells_all": len(cells_all),
        "n_cells_main": len(cells),
        "n_abstract_kw": len(abstract_cells) // 2,
        "abstract_kw": sorted({c.kw for c in abstract_cells}),
        "bootstrap": {"iters": BOOTSTRAP_ITERS, "seed": BOOTSTRAP_SEED},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload["cells"] = [
        {"theme": c.theme, "arm": c.arm, "kw": c.kw, "in_poem": c.in_poem,
         "in_image": c.in_image, "abstract": c.abstract, "poem": c.poem}
        for c in cells_all
    ]
    payload["poem_tokens"] = POEM_TOKENS

    md = _render(payload)
    paths = save_artifacts("poem_gate_coverage_main_n22", payload, md)
    print(md)
    print(f"\njson: {paths['json']}\nmd: {paths['md']}")


if __name__ == "__main__":
    main()
