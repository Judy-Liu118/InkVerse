"""
eval.build_classics_benchmark -- 生成 15 首名诗 benchmark JSON

跑法：
    python -m eval.build_classics_benchmark

输出：
    outputs/eval/benchmark_classics.json，兼容 eval_clip 的 --reuse-poems-from。

意图：
    把"诗本身的质量"这个混淆变量从评测里钉死成"千古名诗"。
    后续 eval_clip 跑这份 JSON 时：
      · 诗端不动（剥离 LoRA 生诗的瑕疵）
      · 锚点 / image prompt 由 qwen-plus + qwen-max 生成（跟现有 baseline
        同链路，保证 apples-to-apples）
      · 唯一变量留给 image backend，专门验证生图能力上限。

依赖：DASHSCOPE_API_KEY；30 次 LLM 调用，约 2 分钟。
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from core.models.adapter import ModelAdapter
from core.agent.agent import PoetryAgent
from core.agent.state import AgentState, Phase
from config import DASHSCOPE_API_KEY, STYLE_MAP


# ── 15 首名诗 benchmark（按体裁分组）────────────────────────────────────
# vlm_level：S = VLM 几乎肯定见过文本/图（国民级名诗，可能记忆 leak）
#            A = 常见名诗
#            B = 教科书但相对冷门，可作 VLM 作弊对照
CLASSICS: List[Dict[str, Any]] = [
    # ── 五言绝句 4 首 ─────────────────────────────────────────────────
    {"author": "李白",   "title": "静夜思",
     "genre": "五言绝句", "theme": "抒情·思乡", "vlm_level": "S",
     "poem": "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡"},
    {"author": "孟浩然", "title": "春晓",
     "genre": "五言绝句", "theme": "自然·春",   "vlm_level": "S",
     "poem": "春眠不觉晓\n处处闻啼鸟\n夜来风雨声\n花落知多少"},
    {"author": "王维",   "title": "鹿柴",
     "genre": "五言绝句", "theme": "自然·山林", "vlm_level": "A",
     "poem": "空山不见人\n但闻人语响\n返景入深林\n复照青苔上"},
    {"author": "柳宗元", "title": "江雪",
     "genre": "五言绝句", "theme": "自然·冬",   "vlm_level": "S",
     "poem": "千山鸟飞绝\n万径人踪灭\n孤舟蓑笠翁\n独钓寒江雪"},

    # ── 七言绝句 6 首 ─────────────────────────────────────────────────
    {"author": "王翰",   "title": "凉州词",
     "genre": "七言绝句", "theme": "边塞",       "vlm_level": "A",
     "poem": "葡萄美酒夜光杯\n欲饮琵琶马上催\n醉卧沙场君莫笑\n古来征战几人回"},
    {"author": "王维",   "title": "九月九日忆山东兄弟",
     "genre": "七言绝句", "theme": "抒情·思乡", "vlm_level": "S",
     "poem": "独在异乡为异客\n每逢佳节倍思亲\n遥知兄弟登高处\n遍插茱萸少一人"},
    {"author": "杜牧",   "title": "清明",
     "genre": "七言绝句", "theme": "节令",       "vlm_level": "S",
     "poem": "清明时节雨纷纷\n路上行人欲断魂\n借问酒家何处有\n牧童遥指杏花村"},
    {"author": "贺知章", "title": "咏柳",
     "genre": "七言绝句", "theme": "咏物",       "vlm_level": "A",
     "poem": "碧玉妆成一树高\n万条垂下绿丝绦\n不知细叶谁裁出\n二月春风似剪刀"},
    {"author": "杜牧",   "title": "山行",
     "genre": "七言绝句", "theme": "自然·秋",   "vlm_level": "A",
     "poem": "远上寒山石径斜\n白云生处有人家\n停车坐爱枫林晚\n霜叶红于二月花"},
    {"author": "李白",   "title": "望庐山瀑布",
     "genre": "七言绝句", "theme": "山水·壮阔", "vlm_level": "A",
     "poem": "日照香炉生紫烟\n遥看瀑布挂前川\n飞流直下三千尺\n疑是银河落九天"},

    # ── 五言律诗 3 首 ─────────────────────────────────────────────────
    {"author": "王维",   "title": "使至塞上",
     "genre": "五言律诗", "theme": "边塞",       "vlm_level": "A",
     "poem": "单车欲问边\n属国过居延\n征蓬出汉塞\n归雁入胡天\n"
             "大漠孤烟直\n长河落日圆\n萧关逢候骑\n都护在燕然"},
    {"author": "王维",   "title": "山居秋暝",
     "genre": "五言律诗", "theme": "山居",       "vlm_level": "A",
     "poem": "空山新雨后\n天气晚来秋\n明月松间照\n清泉石上流\n"
             "竹喧归浣女\n莲动下渔舟\n随意春芳歇\n王孙自可留"},
    {"author": "孟浩然", "title": "过故人庄",
     "genre": "五言律诗", "theme": "田园",       "vlm_level": "B",
     "poem": "故人具鸡黍\n邀我至田家\n绿树村边合\n青山郭外斜\n"
             "开轩面场圃\n把酒话桑麻\n待到重阳日\n还来就菊花"},

    # ── 七言律诗 2 首 ─────────────────────────────────────────────────
    {"author": "杜甫",   "title": "登高",
     "genre": "七言律诗", "theme": "悲秋",       "vlm_level": "A",
     "poem": "风急天高猿啸哀\n渚清沙白鸟飞回\n无边落木萧萧下\n不尽长江滚滚来\n"
             "万里悲秋常作客\n百年多病独登台\n艰难苦恨繁霜鬓\n潦倒新停浊酒杯"},
    {"author": "崔颢",   "title": "黄鹤楼",
     "genre": "七言律诗", "theme": "怀古",       "vlm_level": "A",
     "poem": "昔人已乘黄鹤去\n此地空余黄鹤楼\n黄鹤一去不复返\n白云千载空悠悠\n"
             "晴川历历汉阳树\n芳草萋萋鹦鹉洲\n日暮乡关何处是\n烟波江上使人愁"},
]


def _build_agent() -> PoetryAgent:
    """qwen-plus 抽关键词 / qwen-max 写英文 prompt，跟现有 baseline 同链路。"""
    return PoetryAgent(
        generation_adapter=ModelAdapter(backend="qwen", api_key=DASHSCOPE_API_KEY,
                                        api_model="qwen-plus"),
        score_adapter=ModelAdapter(backend="qwen", api_key=DASHSCOPE_API_KEY,
                                   api_model="qwen-plus"),
        title_adapter=ModelAdapter(backend="qwen", api_key=DASHSCOPE_API_KEY,
                                   api_model="qwen-plus"),
        prompt_adapter=ModelAdapter(backend="qwen", api_key=DASHSCOPE_API_KEY,
                                    api_model="qwen-max"),
    )


def _build_one(agent: PoetryAgent, item: Dict[str, Any]) -> Dict[str, Any]:
    """对单首名诗：跑 keyword 抽取 + image prompt 生成 + prompt 自检。"""
    user_input = f"{item['author']}《{item['title']}》"
    state = AgentState(
        user_input=user_input,
        lang="英文",
        style_suffix=STYLE_MAP["水墨画"],
    )
    # 名诗直接灌入，跳过 _phase_plan / _phase_poem / _phase_title（title 已知）
    state.poem  = item["poem"]
    state.title = item["title"]

    state = agent._phase_keyword_extract(state)
    state = agent._phase_prompt(state)
    if state.phase == Phase.ERROR:
        return {"user_input": user_input, "error": state.error}
    state = agent._phase_prompt_review(state)

    return {
        "user_input":         user_input,
        "author":             item["author"],
        "classic_title":      item["title"],
        "genre":              item["genre"],
        "theme":              item["theme"],
        "keyword_density":    "rich",   # 名诗意象皆丰富，统一 rich（保留字段以兼容 reuse）
        "vlm_level":          item["vlm_level"],
        "is_classic":         True,
        "poem":               item["poem"],
        "visual_keywords_en": state.visual_keywords_en or "",
        "prompt":             state.prompt or "",
    }


def main():
    out_path = Path("outputs/eval/benchmark_classics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build_classics] 生成 {len(CLASSICS)} 首名诗 benchmark")
    print(f"  目标输出: {out_path}")
    print(f"  锚点抽取: qwen-plus | 提示词: qwen-max | 风格: 水墨画")

    agent = _build_agent()

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, item in enumerate(CLASSICS):
        print(f"  [{i+1}/{len(CLASSICS)}] {item['author']}《{item['title']}》"
              f" ({item['genre']} · {item['theme']} · {item['vlm_level']}级) …")
        try:
            row = _build_one(agent, item)
            if "error" in row:
                print(f"      ⚠ 失败：{row['error']}")
            else:
                kw_preview = (row['visual_keywords_en'] or '')[:60]
                pp_preview = (row['prompt'] or '')[:60].replace("\n", " ")
                print(f"      ✓ keywords : {kw_preview}…")
                print(f"      ✓ prompt   : {pp_preview}…")
            rows.append(row)
        except Exception as e:
            print(f"      ⚠ 异常：{e}")
            rows.append({
                "user_input": f"{item['author']}《{item['title']}》",
                "error": str(e),
                **{k: item[k] for k in ("author", "title", "genre", "theme",
                                        "vlm_level", "poem") if k in item},
            })

    elapsed = time.time() - t0
    n_success = sum(1 for r in rows if "error" not in r)
    output = {
        "config": {
            "source":        "build_classics_benchmark",
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "n_classics":    len(CLASSICS),
            "n_success":     n_success,
            "elapsed_sec":   round(elapsed, 1),
            "kw_model":      "qwen-plus",
            "prompt_model":  "qwen-max",
            "style_suffix":  STYLE_MAP["水墨画"],
        },
        "rows": rows,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 完成：成功 {n_success}/{len(rows)} · 耗时 {elapsed:.1f}s")
    print(f"  写出 {out_path}")
    print(f"\n下一步跑 eval：")
    print(f"  python -m eval.eval_clip --reuse-poems-from {out_path} --vlm-judge qwen-vl-max")


if __name__ == "__main__":
    main()
