"""
eval.dataset -- benchmark 数据集

设计覆盖：
  · 4 种近体诗体裁（五绝/七绝/五律/七律）
  · 多种主题（自然 / 抒情 / 边塞 / 哲理 / 节令）
  · 不同关键词密度（具体意象 vs 抽象情感），用于 CLIP 双锚点稀疏分支评估
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BenchInput:
    user_input: str
    genre: str            # 体裁
    theme: str            # 主题大类
    keyword_density: str  # rich | sparse —— 用于 CLIP 稀疏锚点对比


BENCHMARK: List[BenchInput] = [
    # ── 五言绝句 ─────────────────────────────────────────────────────────────
    BenchInput("写一首春天的五言绝句，要有柳树和燕子", "五言绝句", "自然·春", "rich"),
    BenchInput("写一首夏夜的五言绝句，要有荷花和蛙声", "五言绝句", "自然·夏", "rich"),
    BenchInput("写一首秋天的五言绝句，要有菊花和归雁",   "五言绝句", "自然·秋", "rich"),
    BenchInput("写一首冬日的五言绝句，要有寒梅和飞雪",   "五言绝句", "自然·冬", "rich"),
    BenchInput("写一首关于乡愁的五言绝句",               "五言绝句", "抒情",   "sparse"),

    # ── 七言绝句 ─────────────────────────────────────────────────────────────
    BenchInput("写一首七言绝句，描写江南春雨中的杏花村",   "七言绝句", "自然·春", "rich"),
    BenchInput("写一首七言绝句，描写夏日山间的松风泉响",   "七言绝句", "自然·夏", "rich"),
    BenchInput("写一首边塞主题的七言绝句，要有大漠和孤烟", "七言绝句", "边塞",   "rich"),
    BenchInput("写一首七言绝句，主题是友人离别的渡口暮色", "七言绝句", "抒情",   "rich"),
    BenchInput("写一首七言绝句，主题是禅意悟道",           "七言绝句", "哲理",   "sparse"),

    # ── 五言律诗 ─────────────────────────────────────────────────────────────
    BenchInput("写一首五言律诗，秋日山行所见",             "五言律诗", "自然·秋", "rich"),
    BenchInput("写一首五言律诗，描写月夜独酌",             "五言律诗", "抒情",   "rich"),
    BenchInput("写一首五言律诗，归隐田园的耕读生活",       "五言律诗", "哲理",   "rich"),

    # ── 七言律诗 ─────────────────────────────────────────────────────────────
    BenchInput("写一首七言律诗，描写登临高楼的壮阔河山",   "七言律诗", "壮阔",   "rich"),
    BenchInput("写一首七言律诗，主题是除夕守岁",           "七言律诗", "节令",   "rich"),
    BenchInput("写一首七言律诗，怀念故人",                 "七言律诗", "抒情",   "sparse"),

    # ── 抽象/稀疏主题（专门考 CLIP 稀疏锚点分支）─────────────────────────
    BenchInput("写一首关于时间流逝的七言绝句",             "七言绝句", "哲理",   "sparse"),
    BenchInput("写一首关于无常的五言绝句",                 "五言绝句", "哲理",   "sparse"),
    BenchInput("写一首抒发孤独的七言绝句",                 "七言绝句", "抒情",   "sparse"),
    BenchInput("写一首关于离别的五言绝句",                 "五言绝句", "抒情",   "sparse"),
]


def get_benchmark(n: int = None, genres: List[str] = None,
                  density: str = None) -> List[BenchInput]:
    """按需切片 benchmark：可限制条数、体裁、关键词密度。"""
    pool = BENCHMARK
    if genres:
        pool = [b for b in pool if b.genre in genres]
    if density:
        pool = [b for b in pool if b.keyword_density == density]
    if n is not None:
        pool = pool[:n]
    return pool


__all__ = ["BenchInput", "BENCHMARK", "get_benchmark"]
