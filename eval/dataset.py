"""
eval.dataset -- benchmark 数据集

设计覆盖：
  · 4 种近体诗体裁（五绝/七绝/五律/七律），每体裁 8 道
  · 12 个互斥主题大类：春/夏/秋/冬 各 4 道（× 4 体裁），
    山水/田园/边塞/羁旅/送别/怀古/节令/哲理 各 2 道（× 2 体裁）
  · 写景主题跨 4 体裁均匀分布，不再集中在绝句
  · rich / sparse 关键词密度各 16 道，每体裁内 4:4 平衡
  · 总 32 道，n<32 时用 stratified round-robin 切片保持分层均衡
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
    # ── 写景 ×16（春/夏/秋/冬 × 4 体裁，每季 rich:sparse=2:2）────────────
    # 春
    BenchInput("写一首春景的五言绝句，要有柳树和燕子",   "五言绝句", "自然·春", "rich"),
    BenchInput("写一首七言绝句，主题是春雨",             "七言绝句", "自然·春", "sparse"),
    BenchInput("写一首春景的五言律诗，要有桃花和啼莺",   "五言律诗", "自然·春", "rich"),
    BenchInput("写一首七言律诗，主题是早春",             "七言律诗", "自然·春", "sparse"),

    # 夏
    BenchInput("写一首五言绝句，主题是夏蝉",             "五言绝句", "自然·夏", "sparse"),
    BenchInput("写一首夏景的七言绝句，要有荷池和蛙鸣",   "七言绝句", "自然·夏", "rich"),
    BenchInput("写一首五言律诗，主题是消夏",             "五言律诗", "自然·夏", "sparse"),
    BenchInput("写一首夏景的七言律诗，要有竹榻和午梦",   "七言律诗", "自然·夏", "rich"),

    # 秋
    BenchInput("写一首秋景的五言绝句，要有菊花和归雁",   "五言绝句", "自然·秋", "rich"),
    BenchInput("写一首七言绝句，主题是秋怀",             "七言绝句", "自然·秋", "sparse"),
    BenchInput("写一首秋景的五言律诗，要有疏桐和寒蛩",   "五言律诗", "自然·秋", "rich"),
    BenchInput("写一首七言律诗，主题是悲秋",             "七言律诗", "自然·秋", "sparse"),

    # 冬
    BenchInput("写一首五言绝句，主题是寒梅",             "五言绝句", "自然·冬", "sparse"),
    BenchInput("写一首冬景的七言绝句，要有飞雪和寒鸦",   "七言绝句", "自然·冬", "rich"),
    BenchInput("写一首五言律诗，主题是雪夜",             "五言律诗", "自然·冬", "sparse"),
    BenchInput("写一首冬景的七言律诗，要有炉火和寒灯",   "七言律诗", "自然·冬", "rich"),

    # ── 非写景 ×16（8 主题 × 2 体裁，每主题 rich:sparse=1:1）─────────────
    # 山水
    BenchInput("写一首五言绝句，主题是溪声",             "五言绝句", "山水",   "sparse"),
    BenchInput("写一首山水的七言律诗，要有高楼和远山",   "七言律诗", "山水",   "rich"),

    # 田园
    BenchInput("写一首田园的七言绝句，要有耕牛和炊烟",   "七言绝句", "田园",   "rich"),
    BenchInput("写一首五言律诗，主题是归隐",             "五言律诗", "田园",   "sparse"),

    # 边塞
    BenchInput("写一首边塞的五言律诗，要有戍楼和角声",   "五言律诗", "边塞",   "rich"),
    BenchInput("写一首七言绝句，主题是征戍",             "七言绝句", "边塞",   "sparse"),

    # 羁旅·思乡
    BenchInput("写一首羁旅的五言绝句，要有客舍和孤灯",   "五言绝句", "羁旅",   "rich"),
    BenchInput("写一首七言律诗，主题是客愁",             "七言律诗", "羁旅",   "sparse"),

    # 送别
    BenchInput("写一首七言绝句，主题是送别",             "七言绝句", "送别",   "sparse"),
    BenchInput("写一首送别的七言律诗，要有长亭和折柳",   "七言律诗", "送别",   "rich"),

    # 怀古
    BenchInput("写一首怀古的五言律诗，要有古城和荒台",   "五言律诗", "怀古",   "rich"),
    BenchInput("写一首七言律诗，主题是吊古",             "七言律诗", "怀古",   "sparse"),

    # 节令
    BenchInput("写一首中秋的五言绝句，要有明月和团圆",   "五言绝句", "节令",   "rich"),
    BenchInput("写一首五言律诗，主题是重阳",             "五言律诗", "节令",   "sparse"),

    # 哲理·禅意
    BenchInput("写一首禅意的七言绝句，要有古刹和钟鼓",   "七言绝句", "哲理",   "rich"),
    BenchInput("写一首五言绝句，主题是无常",             "五言绝句", "哲理",   "sparse"),
]


def _is_scenic(b: BenchInput) -> bool:
    """是否写景题（主题前缀 "自然·"，即春/夏/秋/冬）。"""
    return b.theme.startswith("自然·")


def get_benchmark(n: int = None, genres: List[str] = None,
                  density: str = None) -> List[BenchInput]:
    """按需切片 benchmark。

    n < len(pool) 时按 (genre, density) 分桶 + 桶内 scenic/非 scenic 交替 +
    相邻桶起点错开，做 round-robin 抽样。三层分层均衡：
      · 体裁  ——  桶外 round-robin
      · density ——  桶外 round-robin
      · 写景/非写景 —— 桶内交替 + 相邻桶反相
    """
    pool = BENCHMARK
    if genres:
        pool = [b for b in pool if b.genre in genres]
    if density:
        pool = [b for b in pool if b.keyword_density == density]
    if n is None or n >= len(pool):
        return list(pool)

    buckets: dict = {}
    order: list = []
    for b in pool:
        key = (b.genre, b.keyword_density)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(b)

    # 桶内按 scenic 交替；相邻桶起点反相 —— 让前 8 道 round 正好 4 写 4 非
    for i, k in enumerate(order):
        scenic = [b for b in buckets[k] if _is_scenic(b)]
        non_scenic = [b for b in buckets[k] if not _is_scenic(b)]
        first, second = (scenic, non_scenic) if i % 2 == 0 else (non_scenic, scenic)
        interleaved = []
        for j in range(max(len(first), len(second))):
            if j < len(first):
                interleaved.append(first[j])
            if j < len(second):
                interleaved.append(second[j])
        buckets[k] = interleaved

    sampled: List[BenchInput] = []
    idx = {k: 0 for k in order}
    while len(sampled) < n:
        progressed = False
        for k in order:
            if idx[k] < len(buckets[k]):
                sampled.append(buckets[k][idx[k]])
                idx[k] += 1
                progressed = True
                if len(sampled) >= n:
                    break
        if not progressed:
            break
    return sampled


def dump_to_json(path: str) -> int:
    """把 BENCHMARK 导出成 JSON，供外部 demo / notebook 读取统一题源。"""
    import json
    from pathlib import Path
    payload = {
        "schema_version": 1,
        "n": len(BENCHMARK),
        "items": [
            {
                "user_input":      b.user_input,
                "genre":           b.genre,
                "theme":           b.theme,
                "keyword_density": b.keyword_density,
            }
            for b in BENCHMARK
        ],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(BENCHMARK)


__all__ = ["BenchInput", "BENCHMARK", "get_benchmark", "dump_to_json"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="dump benchmark BENCHMARK 到 JSON 文件")
    ap.add_argument("--dump", default="eval/benchmark_themes.json",
                    help="JSON 输出路径（默认 eval/benchmark_themes.json，version controlled）")
    args = ap.parse_args()
    n = dump_to_json(args.dump)
    print(f"Dumped {n} 道 → {args.dump}")
