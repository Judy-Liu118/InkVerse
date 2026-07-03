"""静态回归：用历史 baseline+refined 诗集，对比新老 _get_stable_rhyme 的 rhyme 分。

用途：验证 `startswith → endswith` + `full[-2:] → full[-1]` 修复
      对真实 n=32 主 benchmark 诗集的影响面。

跑法：
    python -m eval._regression_check_rhyme_endswith outputs/eval/eval_refine_20260702_235009.json

结论解读：
- ↑ 上升条目：iou/uei 韵字被从「误分」正确归入同韵组，是 bug 修复的正向影响
- ↓ 下降条目：不应出现；若有则说明修复引入新回归
- = 持平条目：诗里未含 iou/uei 尾韵字，不受修复影响
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pypinyin
from pingshui_rhyme import RhymeChecker

from core.poem.theme import PINYIN_TO_PINGSHUI

_rc = RhymeChecker()

_REGEX = re.compile(
    r'[東冬江支微魚虞齊佳灰真文元寒刪先蕭肴豪歌麻陽庚青蒸尤侵覃鹽咸]'
)


def new_get_stable_rhyme(char: str) -> str:
    """当前生产代码：endswith + full[-1]。"""
    try:
        result = _rc.get_rhyme_group(char)
        if result and isinstance(result, list):
            ping_entry = next((e for e in result if e[0] == 'ping'), result[0])
            full = ping_entry[2]
            m = _REGEX.search(full)
            if m: return m.group()
            if full: return full[-1]
    except Exception:
        pass
    try:
        fi = pypinyin.pinyin(char, style=pypinyin.Style.FINALS)
        if fi and fi[0][0]:
            final = fi[0][0]
            if final in PINYIN_TO_PINGSHUI:
                return PINYIN_TO_PINGSHUI[final]
            for k in PINYIN_TO_PINGSHUI:
                if final.endswith(k):
                    return PINYIN_TO_PINGSHUI[k]
            return final
    except Exception:
        pass
    return char


def old_get_stable_rhyme(char: str) -> str:
    """老 bug 版：startswith + full[-2:]。"""
    try:
        result = _rc.get_rhyme_group(char)
        if result and isinstance(result, list):
            ping_entry = next((e for e in result if e[0] == 'ping'), result[0])
            full = ping_entry[2]
            m = _REGEX.search(full)
            if m: return m.group()
            if len(full) >= 2: return full[-2:]
    except Exception:
        pass
    try:
        fi = pypinyin.pinyin(char, style=pypinyin.Style.FINALS)
        if fi and fi[0][0]:
            final = fi[0][0]
            if final in PINYIN_TO_PINGSHUI:
                return PINYIN_TO_PINGSHUI[final]
            for k in PINYIN_TO_PINGSHUI:
                if final.startswith(k):
                    return PINYIN_TO_PINGSHUI[k]
            return final
    except Exception:
        pass
    return char


def score_rhyme(poem: str, num_lines: int, rhyme_fn) -> float:
    """复刻 _score_rhyme（跟 scorer.py 一致）。"""
    lines = [l.strip() for l in poem.split('\n') if l.strip()][:num_lines]
    if len(lines) < 4:
        return 0.0
    groups = [rhyme_fn(lines[i][-1]) for i in range(1, num_lines, 2)]
    unique = len(set(groups))
    return 1.0 if unique == 1 else 0.6 if unique == 2 else 0.3


def _detect_num_lines(genre: str) -> int:
    return 8 if "律" in genre else 4


def _detect_last_chars(poem: str, num_lines: int) -> list[str]:
    lines = [l.strip() for l in poem.split('\n') if l.strip()][:num_lines]
    return [lines[i][-1] for i in range(1, num_lines, 2) if lines[i]]


def _iter_poems(data: dict):
    """按不同 json 结构统一提取 (label, poem, genre)。"""
    if "rows" in data:  # eval_refine 结构
        for r in data["rows"]:
            if "error" in r:
                continue
            yield ("base", r["baseline_poem"], r["genre"], r["user_input"][:24])
            yield ("refi", r["refined_poem"],  r["genre"], r["user_input"][:24])
    elif "runs" in data:  # eval_poem 主跑结构
        for run_i, run in enumerate(data["runs"]):
            for model, items in run["results_per_model"].items():
                for it in items:
                    for ci, cand in enumerate(it.get("candidates", [])):
                        if cand.get("error") or not cand.get("poem"):
                            continue
                        tag = f"r{run_i}·{model}·c{ci}"
                        yield (tag, cand["poem"], it["genre"], it["user_input"][:24])


def analyze(json_path: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    print(f"\n═══ 静态回归：{json_path.name} ═══")

    up, down, same = [], [], []
    n = 0
    for label, poem, genre, user_input in _iter_poems(data):
        n += 1
        nl = _detect_num_lines(genre)
        new = score_rhyme(poem, nl, new_get_stable_rhyme)
        old = score_rhyme(poem, nl, old_get_stable_rhyme)
        item = {
            "user_input": user_input,
            "label": label,
            "old": old,
            "new": new,
            "delta": round(new - old, 3),
            "rhyme_chars": _detect_last_chars(poem, nl),
        }
        if new > old:
            up.append(item)
        elif new < old:
            down.append(item)
        else:
            same.append(item)

    print(f"样本总数：{n} 首诗\n")

    n_total = len(up) + len(down) + len(same)
    print("┌─────── 汇总 ───────┐")
    print(f"│ ↑ 上升   {len(up):4d}  ({len(up)/n_total:.1%})")
    print(f"│ = 持平   {len(same):4d}  ({len(same)/n_total:.1%})")
    print(f"│ ↓ 下降   {len(down):4d}  ({len(down)/n_total:.1%})")
    print("└─────────────────────┘\n")

    def _show(items, n_show=15):
        for it in items[:n_show]:
            print(f"  {it['label']:24s}  {it['old']:.1f} → {it['new']:.1f}  "
                  f"Δ{it['delta']:+.1f}  韵脚={it['rhyme_chars']}  {it['user_input']}")
        if len(items) > n_show:
            print(f"  ... 共 {len(items)} 条")

    if up:
        print(f"↑ 上升详情（{len(up)} 条）—— 修复正向影响：")
        _show(up)
    if down:
        print(f"\n↓ 下降详情（{len(down)} 条）—— 需人工核查：")
        _show(down)

    if n_total:
        old_all = sum(it['old'] for it in up + same + down) / n_total
        new_all = sum(it['new'] for it in up + same + down) / n_total
        print(f"\n全样本 mean：old={old_all:.4f}  new={new_all:.4f}  Δ{new_all-old_all:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("json_path", type=Path)
    args = p.parse_args()
    analyze(args.json_path)


if __name__ == "__main__":
    main()
