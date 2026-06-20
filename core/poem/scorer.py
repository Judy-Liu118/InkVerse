"""
core.poem.scorer -- 诗歌评分系统

多维度评分：意图匹配(LLM)、平仄、押韵、意象库、主题聚合、重复惩罚、必须意象检查。
"""
import random
import re
import statistics
from typing import Dict, List, Tuple
from collections import Counter
import pypinyin
from pingshui_rhyme import PingZeClassifier, RhymeChecker
from config import (
    GENRE_CONFIG,
    WEIGHT_INTENT, WEIGHT_PINGZE, WEIGHT_RHYME, WEIGHT_IMAGERY, WEIGHT_COHESION,
    REPETITION_PENALTY_MAX, SCORE_PENALTY_FLOOR,
)
from core.poem.theme import (
    resolve_theme_synonyms, strip_meta_prefix, get_imagery_synonyms,
    EMOTION_THEMES, ALL_IMAGERY_WORDS, PINYIN_TO_PINGSHUI,
)
from core.logger import get_logger

_log = get_logger(__name__)

_SCORING_PROMPT_TEMPLATE = """请根据以下用户要求，对古诗进行评分（以鼓励为主，宽容评价，允许0.5分精度）。
用户要求：{user_request}
古诗：{poem}

评分标准（满分10分，每位维度允许0.5分步进，如1.5、2.5）：
1. 主题匹配度（0-3分，0.5精度）：
   - 2.5-3分：完全围绕用户指定主题，核心意象贯穿全诗。
   - 2-2.5分：通过意象、氛围、季节等有效传达了主题，不必苛求主题词字面出现。
   - 1-1.5分：主题基本可辨，表达较隐晦或部分偏离。
   - 0-0.5分：主题完全不符，全诗与用户要求无关联。
	   注意：若用户指定了季节（如"夏夜""春日"）而诗中出现了矛盾季节意象（如夏夜写"秋吟""秋堂"），主题匹配度应酌情扣1分。
2. 意象完整性（0-3分，0.5精度）：
   - 2.5-3分：用户要求的必须意象全部鲜明出现且自然融入。
   - 2分：意象出现但较平淡，或有等价意象替代。
   - 1-1.5分：意象勉强涉及但不够明确。
   - 0-0.5分：用户要求的意象完全未出现，也无可替代的等价意象。
3. 意境连贯度（0-2分，0.5精度）：
   - 2分：诗句流畅，意境统一，起承转合自然。
   - 1-1.5分：整体可读，偶有断裂或跳跃。
   - 0-0.5分：逻辑混乱，意象堆砌无关联。
4. 语言优美度（0-2分，0.5精度）：
   - 2分：用词典雅生动，画面感强，遣词精当。
   - 1-1.5分：用词尚可，有一定诗意。
   - 0-0.5分：用词直白、俗套或生硬。

输出格式：主题匹配度:数字,意象完整性:数字,意境连贯度:数字,语言优美度:数字,总分:数字
例如：主题匹配度:2.5,意象完整性:3,意境连贯度:1.5,语言优美度:1.5,总分:8.5
只输出这一行，不要额外解释。"""

# 同义词合掌防御：同一联中使用同义词群的多个词 → 意象重复
_SYNONYM_CLASH_GROUPS = [
    {"玉轮", "兔钩", "明月", "素月", "冰轮", "蟾宫", "玉盘", "婵娟", "皓月", "新月", "月色", "月光"},
    {"银汉", "星河", "银河", "天河", "星汉", "云汉", "玉绳", "北斗"},
    {"红蕖", "荷花", "菡萏", "芙蓉", "荷叶", "碧莲", "水芙蓉"},
    {"归雁", "鸿雁", "征雁", "孤鸿", "飞鸿", "雁阵", "雁字"},
    {"残阳", "落日", "夕阳", "斜阳", "暮日", "夕照"},
]
CLASH_PENALTY_PER_HIT = 0.75  # 发现一处合掌，品质分打 75 折


class PoemScorer:
    """诗歌多维度评分器。"""

    def __init__(self):
        self._classifier = PingZeClassifier()
        self._rhyme_checker = RhymeChecker()

    # ── 平仄 ───────────────────────────────────────────────────────────────
    PATTERN_5_8 = {
        "仄起首句不入韵": [
            "仄仄平平仄", "平平仄仄平", "平平平仄仄", "仄仄仄平平",
            "仄仄平平仄", "平平仄仄平", "平平平仄仄", "仄仄仄平平"
        ],
        "仄起首句入韵": [
            "仄仄仄平平", "平平仄仄平", "平平平仄仄", "仄仄仄平平",
            "仄仄平平仄", "平平仄仄平", "平平平仄仄", "仄仄仄平平"
        ],
        "平起首句不入韵": [
            "平平平仄仄", "仄仄仄平平", "仄仄平平仄", "平平仄仄平",
            "平平平仄仄", "仄仄仄平平", "仄仄平平仄", "平平仄仄平"
        ],
        "平起首句入韵": [
            "平平仄仄平", "仄仄仄平平", "仄仄平平仄", "平平仄仄平",
            "平平平仄仄", "仄仄仄平平", "仄仄平平仄", "平平仄仄平"
        ],
    }
    PATTERN_5_4 = {
        "仄起首句不入韵": ["仄仄平平仄", "平平仄仄平", "平平平仄仄", "仄仄仄平平"],
        "仄起首句入韵": ["仄仄仄平平", "平平仄仄平", "平平平仄仄", "仄仄仄平平"],
        "平起首句不入韵": ["平平平仄仄", "仄仄仄平平", "仄仄平平仄", "平平仄仄平"],
        "平起首句入韵": ["平平仄仄平", "仄仄仄平平", "仄仄平平仄", "平平仄仄平"],
    }
    PATTERN_7_8 = {
        "仄起首句不入韵": [
            "仄仄平平平仄仄", "平平仄仄仄平平", "平平仄仄平平仄", "仄仄平平仄仄平",
            "仄仄平平平仄仄", "平平仄仄仄平平", "平平仄仄平平仄", "仄仄平平仄仄平"
        ],
        "仄起首句入韵": [
            "仄仄平平仄仄平", "平平仄仄仄平平", "平平仄仄平平仄", "仄仄平平仄仄平",
            "仄仄平平平仄仄", "平平仄仄仄平平", "平平仄仄平平仄", "仄仄平平仄仄平"
        ],
        "平起首句不入韵": [
            "平平仄仄平平仄", "仄仄平平仄仄平", "仄仄平平平仄仄", "平平仄仄仄平平",
            "平平仄仄平平仄", "仄仄平平仄仄平", "仄仄平平平仄仄", "平平仄仄仄平平"
        ],
        "平起首句入韵": [
            "平平仄仄仄平平", "仄仄平平仄仄平", "仄仄平平平仄仄", "平平仄仄仄平平",
            "平平仄仄平平仄", "仄仄平平仄仄平", "仄仄平平平仄仄", "平平仄仄仄平平"
        ],
    }
    PATTERN_7_4 = {
        "仄起首句不入韵": ["仄仄平平平仄仄", "平平仄仄仄平平", "平平仄仄平平仄", "仄仄平平仄仄平"],
        "仄起首句入韵": ["仄仄平平仄仄平", "平平仄仄仄平平", "平平仄仄平平仄", "仄仄平平仄仄平"],
        "平起首句不入韵": ["平平仄仄平平仄", "仄仄平平仄仄平", "仄仄平平平仄仄", "平平仄仄仄平平"],
        "平起首句入韵": ["平平仄仄仄平平", "仄仄平平仄仄平", "仄仄平平平仄仄", "平平仄仄仄平平"],
    }

    @staticmethod
    def detect_genre(user_topic: str) -> Tuple[str, int, int]:
        for name, (nl, cpl) in GENRE_CONFIG.items():
            if name in user_topic:
                return name, nl, cpl
        return "五言绝句", 4, 5

    # ── 完整评分 ───────────────────────────────────────────────────────────
    def evaluate_full(self, poem: str, num_lines: int, chars_per_line: int,
                      user_request: str, adapter, candidate_index: int = 0) -> Dict:
        lines = [l.strip() for l in poem.split('\n') if l.strip()]
        if len(lines) < num_lines or any(len(l) != chars_per_line for l in lines[:num_lines]):
            return {
                "total": 0, "raw_total": 0, "pingze": 0, "rhyme": 0,
                "imagery": 0, "cohesion": 0, "intent": 0,
                "intent_llm": 0, "penalty": 1.0,
            }
        pz = self._score_pingze(poem, num_lines, chars_per_line)
        rh = self._score_rhyme(poem, num_lines)
        # 规则版 imagery/cohesion 作为兜底
        im_rule = self._score_imagery(poem)
        co_rule = self._score_theme_cohesion(poem)
        # LLM 评委 4 维（intent/imagery/cohesion/aesthetics）
        llm_dims = self.score_4dim_via_llm(poem, user_request, adapter)
        intent_score = llm_dims.get("intent", 0.5)
        # 4 维 LLM 评分作为主路；规则版仅在 LLM 失败/本地后端时兜底
        im_score = llm_dims["imagery"]   if llm_dims.get("imagery")    is not None else im_rule
        co_score = llm_dims["cohesion"]  if llm_dims.get("cohesion")   is not None else co_rule
        aesthetics = llm_dims["aesthetics"] if llm_dims.get("aesthetics") is not None else 0.5

        penalty = self._repetition_penalty(poem)
        required_coeff = self._check_required_keywords(poem, user_request, candidate_index)
        clash_penalty = self._check_synonym_clash(poem)

        raw_total = (intent_score * WEIGHT_INTENT + pz * WEIGHT_PINGZE +
                     rh * WEIGHT_RHYME + im_score * WEIGHT_IMAGERY +
                     co_score * WEIGHT_COHESION)
        # 给每个惩罚因子加下限，避免多重惩罚把好诗压成 raw_total × 0.4 以下
        penalty_c  = max(penalty,        SCORE_PENALTY_FLOOR)
        clash_c    = max(clash_penalty,  SCORE_PENALTY_FLOOR)
        required_c = max(required_coeff, SCORE_PENALTY_FLOOR)
        art_quality = raw_total * penalty_c * clash_c
        final_total = art_quality * required_c
        # 原始未夹版（不受 SCORE_PENALTY_FLOOR 影响），供调试与历史对比
        final_total_uncapped = raw_total * penalty * clash_penalty * required_coeff

        return {
            "total":          round(final_total, 3),
            "art_quality":    round(art_quality, 3),
            "raw_total":      round(raw_total, 3),
            "uncapped":       round(final_total_uncapped, 3),
            "pingze":         round(pz, 3),
            "rhyme":          round(rh, 3),
            "imagery":        round(im_score, 3),
            "cohesion":       round(co_score, 3),
            "imagery_rule":   round(im_rule, 3),   # 规则版备查
            "cohesion_rule":  round(co_rule, 3),   # 规则版备查
            "intent":         round(intent_score, 3),
            "intent_llm":     round(intent_score, 3),
            "aesthetics":     round(aesthetics, 3),
            "penalty":        round(penalty, 3),
            "required_coeff": round(required_coeff, 3),
            "llm_source":     "rule_fallback" if llm_dims.get("_local_fallback") else "llm_rubric",
        }

    # ── 多评委 4 维评分（B 方案：LLM rubric + 跨家族多评委）─────────────────
    def score_single_multi_judge(
        self, poem: str, user_request: str,
        judges: list, candidate_index: int = 0,
    ) -> dict:
        """多评委对 4 个主观维度（intent/imagery/cohesion/aesthetics）各自打分，
        然后按"评委数 >= 3 取中位数（对异常更健壮）/ <3 取均值"合成。

        参数 judges: List[Tuple[str, adapter]] —— (label, adapter) 对。
        返回字典含：
          · 标准 scores（用合成后的 4 维 + 规则维度计算 total）
          · scores_by_judge: {judge_label: {intent, imagery, cohesion, aesthetics}}
          · 各维度评委间分歧 (max-min)，便于分析

        ⚠️ 离线评估（eval/）专用，生产仍走 score_single。
        """
        if not judges:
            raise ValueError("judges 不能为空")

        # 1. 平仄/押韵/必须意象/重复 这些规则维度只算一次（确定性的）
        lines = [l.strip() for l in poem.split("\n") if l.strip()]
        if not lines:
            return {"total": 0.0, "intent": 0.0, "imagery": 0.0, "cohesion": 0.0,
                    "aesthetics": 0.0, "pingze": 0.0, "rhyme": 0.0,
                    "scores_by_judge": {}, "judge_disagreement": {}}
        num_lines = len(lines)
        chars_per_line = len(lines[0])
        pz = self._score_pingze(poem, num_lines, chars_per_line)
        rh = self._score_rhyme(poem, num_lines)
        im_rule = self._score_imagery(poem)
        co_rule = self._score_theme_cohesion(poem)
        penalty       = self._repetition_penalty(poem)
        req_coeff     = self._check_required_keywords(poem, user_request, candidate_index)
        clash_penalty = self._check_synonym_clash(poem)

        # 2. 每个评委独立给 4 维分
        scores_by_judge = {}
        for label, adp in judges:
            try:
                dims = self.score_4dim_via_llm(poem, user_request, adp)
                scores_by_judge[label] = {
                    "intent":     dims.get("intent", 0.5),
                    "imagery":    dims.get("imagery"),
                    "cohesion":   dims.get("cohesion"),
                    "aesthetics": dims.get("aesthetics"),
                }
            except Exception as e:
                _log.warning("评委 '%s' 评分失败: %s", label, e)
                scores_by_judge[label] = {
                    "intent": 0.5, "imagery": 0.5,
                    "cohesion": 0.5, "aesthetics": 0.5,
                }

        # 3. 合成各维度（3+ 评委取中位数；< 3 取均值；遇 None 用规则版兜底）
        def _aggregate(dim_key, rule_fallback):
            vals = []
            for j_scores in scores_by_judge.values():
                v = j_scores.get(dim_key)
                vals.append(v if v is not None else rule_fallback)
            if not vals:
                return rule_fallback
            if len(vals) >= 3:
                return statistics.median(vals)
            return sum(vals) / len(vals)

        intent_final     = _aggregate("intent",     0.5)
        imagery_final    = _aggregate("imagery",    im_rule)
        cohesion_final   = _aggregate("cohesion",   co_rule)
        aesthetics_final = _aggregate("aesthetics", 0.5)

        # 4. 各维度分歧 (max - min)，用于分析"评委争议大"的诗
        def _spread(dim_key):
            vals = [j[dim_key] for j in scores_by_judge.values()
                    if j.get(dim_key) is not None]
            return round(max(vals) - min(vals), 3) if len(vals) >= 2 else 0.0
        disagreement = {
            "intent":     _spread("intent"),
            "imagery":    _spread("imagery"),
            "cohesion":   _spread("cohesion"),
            "aesthetics": _spread("aesthetics"),
        }

        # 5. 算 total（aesthetics 进入加权：把 imagery+cohesion 各匀出 5% 权重给 aesthetics）
        from config import WEIGHT_IMAGERY, WEIGHT_COHESION
        w_aesthetics = 0.10
        w_imagery    = max(0.0, WEIGHT_IMAGERY - 0.05)
        w_cohesion   = max(0.0, WEIGHT_COHESION - 0.05)
        raw_total = (intent_final     * WEIGHT_INTENT +
                     pz               * WEIGHT_PINGZE +
                     rh               * WEIGHT_RHYME +
                     imagery_final    * w_imagery +
                     cohesion_final   * w_cohesion +
                     aesthetics_final * w_aesthetics)
        penalty_c = max(penalty,       SCORE_PENALTY_FLOOR)
        clash_c   = max(clash_penalty, SCORE_PENALTY_FLOOR)
        req_c     = max(req_coeff,     SCORE_PENALTY_FLOOR)
        art_quality = raw_total * penalty_c * clash_c
        final_total = art_quality * req_c

        # 6. 兼容旧字段 intent_by_judge：只放 intent 一维（向后兼容 eval_poem 渲染）
        intent_by_judge = {label: round(s["intent"], 3)
                           for label, s in scores_by_judge.items()}

        return {
            "total":          round(final_total, 3),
            "art_quality":    round(art_quality, 3),
            "raw_total":      round(raw_total, 3),
            "intent":         round(intent_final, 3),
            "intent_llm":     round(intent_final, 3),
            "imagery":        round(imagery_final, 3),
            "cohesion":       round(cohesion_final, 3),
            "aesthetics":     round(aesthetics_final, 3),
            "imagery_rule":   round(im_rule, 3),
            "cohesion_rule":  round(co_rule, 3),
            "pingze":         round(pz, 3),
            "rhyme":          round(rh, 3),
            "penalty":        round(penalty, 3),
            "required_coeff": round(req_coeff, 3),
            "intent_by_judge":   intent_by_judge,
            "scores_by_judge":   {k: {kk: round(vv, 3) if vv is not None else None
                                       for kk, vv in v.items()}
                                  for k, v in scores_by_judge.items()},
            "judge_disagreement": disagreement,
            "aggregation_method": "median" if len(judges) >= 3 else "mean",
        }

    # ── 单首评分（供 refine_poem 用）─────────────────────────────────────────
    def score_single(self, poem: str, user_request: str, score_adapter,
                      candidate_index: int = 0) -> dict:
        lines = [l.strip() for l in poem.split("\n") if l.strip()]
        if not lines:
            return {"total": 0.0, "intent": 0.0, "intent_llm": 0.0,
                    "pingze": 0.0, "rhyme": 0.0, "imagery": 0.0,
                    "cohesion": 0.0, "penalty": 1.0, "raw_total": 0.0,
                    "required_coeff": 1.0}
        num_lines = len(lines)
        chars_per_line = len(lines[0])
        try:
            return self.evaluate_full(poem, num_lines, chars_per_line, user_request,
                                      score_adapter, candidate_index)
        except Exception as e:
            _log.error("score_single 评分异常: %s", e)
            return {"total": 0.0, "intent": 0.0, "intent_llm": 0.0,
                    "pingze": 0.0, "rhyme": 0.0, "imagery": 0.0,
                    "cohesion": 0.0, "penalty": 1.0, "raw_total": 0.0,
                    "required_coeff": 1.0}

    # ── 必须意象检查 ───────────────────────────────────────────────────────
    def _check_required_keywords(self, poem: str, user_request: str,
                                  candidate_index: int = 0) -> float:
        poem_text = "".join(poem.split("\n"))
        coeff = 1.0
        META_WORDS = {"意向", "意象", "意境", "主题", "内容", "元素",
                      "风格", "色调", "感觉", "氛围", "情感", "情调"}
        idx_tag = f"[候选{candidate_index}] " if candidate_index else ""

        # "以X为主题" 不再用正则做字面匹配——LLM 意图评分（权重 40%）已负责判断
        # 主题是否命中。抽象主题如"思乡""壮志"等无法靠字面匹配覆盖，正则反而误伤。
        theme_m = re.search(r"以(.{1,4})为主题", user_request)
        if theme_m:
            theme = theme_m.group(1).strip()
            _log.debug("%s[主题] 用户要求以'%s'为主题，交由 LLM 意图评分判断", idx_tag, theme)

        require_pattern = r"(?:要|需要|必须)\s*(?:包含|含有|含|有|加入|体现|出现)\s*([^，。；！？\n]{1,10})"
        # 情绪/氛围后缀，如"萧瑟之感"→剥离"之感"→"萧瑟"
        _MOOD_SUFFIXES = ["之感", "之情", "之意", "之境", "之趣", "之韵", "之色", "之声", "之气", "之态"]
        # 列表连接词：先按这些切分，再用 1-4 字提取，避免 {1,4} 贪婪把"柳树和燕子"切成["柳树和燕","子"]
        _LIST_CONNECTORS = re.compile(r"[和与及、，,]")
        for seg in re.findall(require_pattern, user_request):
            sub_segs = _LIST_CONNECTORS.split(seg) if seg else [seg]
            items = []
            for sub in sub_segs:
                items.extend(re.findall(r"[一-鿿]{1,4}", sub.strip()))
            for raw_item in items:
                item = strip_meta_prefix(raw_item)
                if not item or item in META_WORDS:
                    continue
                # 剥离情绪/氛围后缀
                for suffix in _MOOD_SUFFIXES:
                    if item.endswith(suffix) and len(item) > len(suffix):
                        item = item[:-len(suffix)]
                        break
                synonyms = get_imagery_synonyms(item)
                if not any(s in poem_text for s in synonyms):
                    coeff *= 0.75
                    _log.info("%s[必须意象] ⚠ 要求含'%s'及同义词(%s)均未出现 → ×0.75",
                             idx_tag, item, ','.join(synonyms[:6]))
                else:
                    hit = next(s for s in synonyms if s in poem_text)
                    suffix = f"（同义词命中'{hit}'）" if hit != item else ""
                    _log.info("%s[必须意象] ✓ 要求含'%s'%s 已满足", idx_tag, item, suffix)

        return round(max(0.1, coeff), 4)

    # ── 意图评分（兼容旧接口：返回 intent 单值；新接口 score_4dim_via_llm 返 4 维）─────
    def _score_intent_combined(self, poem, user_request, adapter) -> Dict:
        """旧接口：只回 intent 一维（向后兼容）。"""
        if self._is_api_backend(adapter):
            dims = self._score_llm_api_4dim(poem, user_request, adapter)
            llm_score = dims["intent"]
        else:
            llm_score = self._score_intent_llm_local(poem, user_request, adapter)
        return {"combined": llm_score, "llm": llm_score}

    def score_4dim_via_llm(self, poem, user_request, adapter) -> dict:
        """新接口：让 LLM 评委按 rubric 给 4 维（intent/imagery/cohesion/aesthetics）。

        本地 backend 不支持 4 维评分（提示词太复杂、本地模型不可靠），
        会回落到：intent 走 local CoT、其余 3 维返 None 让 evaluate_full 用规则版兜底。
        """
        if self._is_api_backend(adapter):
            return self._score_llm_api_4dim(poem, user_request, adapter)
        # 本地评分只支持 intent；其余维度返 None，调用方应使用规则版
        intent_score = self._score_intent_llm_local(poem, user_request, adapter)
        return {"intent": intent_score, "imagery": None, "cohesion": None,
                "aesthetics": None, "total": intent_score, "_local_fallback": True}

    @staticmethod
    def _is_api_backend(adapter) -> bool:
        if adapter is None:
            return False
        backend = getattr(adapter, 'backend', '')
        return backend not in ('local', 'local_lora')

    def _score_llm_api_4dim(self, poem, user_request, adapter) -> dict:
        """LLM 评委按 rubric 给 4 维分（intent/imagery/cohesion/aesthetics），归一化到 [0,1]。

        返回 dict 见 _parse_llm_score_reply。失败时返回全 0.5 的 fallback dict。
        """
        prompt = _SCORING_PROMPT_TEMPLATE.format(user_request=user_request, poem=poem)
        messages = [
            {"role": "system", "content": "你是严格的文学评委，只输出一行逗号分隔的标签值对。"},
            {"role": "user", "content": prompt},
        ]
        try:
            reply = adapter.generate(messages, max_tokens=120, temperature=0.1)
            _log.debug("LLM-API 原始返回: %s", reply)
            parsed = self._parse_llm_score_reply(reply)
            if parsed is not None:
                _log.debug("LLM-API 4维: intent=%.2f imagery=%.2f cohesion=%.2f aesthetics=%.2f",
                          parsed["intent"], parsed["imagery"],
                          parsed["cohesion"], parsed["aesthetics"])
                return parsed
        except Exception as e:
            _log.error("LLM-API 评分失败: %s", e)
        return {"intent": 0.5, "imagery": 0.5, "cohesion": 0.5,
                "aesthetics": 0.5, "total": 0.5}

    @staticmethod
    def _parse_llm_score_reply(reply: str) -> dict | None:
        """解析 LLM 评委的 4 维评分回复，返回归一化到 [0,1] 的 dict。

        返回结构（None 表示解析失败）：
        {
            "intent":     0.0-1.0,  # 主题匹配度（3 分制 → /3）
            "imagery":    0.0-1.0,  # 意象完整性（3 分制 → /3）
            "cohesion":   0.0-1.0,  # 意境连贯度（2 分制 → /2）
            "aesthetics": 0.0-1.0,  # 语言优美度（2 分制 → /2）
            "total":      0.0-1.0,  # 总分（10 分制 → /10），LLM 给的总分优先；否则用 4 维加权
        }
        """
        text = (reply or "").strip()
        pairs = {
            key: float(val)
            for key, val in re.findall(
                r'(主题匹配度|意象完整性|意境连贯度|语言优美度|总分)\s*[:：]\s*(\d+(?:\.\d+)?)',
                text,
            )
        }

        # 4 维分独立归一化（即使部分缺失也尽力解析）
        component_norms = {
            "intent":     min(pairs["主题匹配度"], 3.0) / 3.0 if "主题匹配度" in pairs else None,
            "imagery":    min(pairs["意象完整性"], 3.0) / 3.0 if "意象完整性" in pairs else None,
            "cohesion":   min(pairs["意境连贯度"], 2.0) / 2.0 if "意境连贯度" in pairs else None,
            "aesthetics": min(pairs["语言优美度"], 2.0) / 2.0 if "语言优美度" in pairs else None,
        }

        # 总分：优先用 LLM 直接给的；否则用 4 维加权（3+3+2+2=10）
        if "总分" in pairs:
            total = max(0.0, min(10.0, pairs["总分"])) / 10.0
        elif all(v is not None for v in component_norms.values()):
            total = (
                component_norms["intent"]     * 0.3 +
                component_norms["imagery"]    * 0.3 +
                component_norms["cohesion"]   * 0.2 +
                component_norms["aesthetics"] * 0.2
            )
        else:
            # 兜底：随便找最后一个 0-10 的数
            nums = [float(n) for n in re.findall(r'\b\d+(?:\.\d+)?\b', text)]
            candidates = [n for n in nums if 0 <= n <= 10]
            if not candidates:
                return None
            total = candidates[-1] / 10.0

        # 缺失维度用 total 兜底（不让单维度查询失败）
        for k in component_norms:
            if component_norms[k] is None:
                component_norms[k] = total

        component_norms["total"] = round(max(0.0, min(1.0, total)), 4)
        for k in ("intent", "imagery", "cohesion", "aesthetics"):
            component_norms[k] = round(max(0.0, min(1.0, component_norms[k])), 4)
        return component_norms

    def _score_intent_llm_local(self, poem, user_request, adapter) -> float:
        step1_prompt = f"""用户创作要求：{user_request}

生成的古诗：
{poem}

请逐条回答（简短即可）：
A. 用户明确要求的主题/事物：
B. 诗中出现的对应意象（含同义词，如柳树=垂杨）：
C. 明显缺失的要素（若无请写"无"）："""
        messages_s1 = [
            {"role": "system", "content": "你是严格的古诗评审，用简短语言逐条分析。"},
            {"role": "user", "content": step1_prompt},
        ]
        try:
            analysis = adapter.generate(messages_s1, max_tokens=150, temperature=0.3)
        except Exception as e:
            _log.error("CoT-LLM 第一步分析失败: %s", e)
            analysis = "A. 未提取到 B. 未提取到 C. 无"

        step2_prompt = f"""根据以下评审分析，给古诗打分（只输出0到10之间的整数，不输出任何其他内容）：

{analysis}

评分标准：10=完全满足；7-9=基本满足；4-6=部分满足；0-3=明显缺失核心要素

分数："""
        messages_s2 = [
            {"role": "system", "content": "你是打分机器人，只输出一个整数。"},
            {"role": "user", "content": step2_prompt},
        ]
        try:
            reply = adapter.generate(messages_s2, max_tokens=10, temperature=0.1)
            m = re.search(r'(\d+)', reply)
            if m:
                score = min(int(m.group(1)), 10)
                _log.debug("CoT-LLM 得分: %d/10", score)
                return score / 10.0
        except Exception as e:
            _log.error("CoT-LLM 第二步打分失败: %s", e)
        return 0.5

    # ── 平仄评分 ───────────────────────────────────────────────────────────
    def _score_pingze(self, poem, num_lines, chars_per_line) -> float:
        lines = [l.strip() for l in poem.split('\n') if l.strip()][:num_lines]
        if len(lines) != num_lines:
            return 0.0
        tone_seqs = [tuple(self._classifier.classify(l)) for l in lines]
        patterns = self._get_patterns(chars_per_line, num_lines)
        if not patterns:
            return 0.5
        best = 0
        for pat in patterns:
            match = sum(
                1 for i in range(num_lines)
                if self._matches_pattern(tone_seqs[i], pat[i])
            )
            best = max(best, match / num_lines)
        return best

    def _get_patterns(self, word_len, line_count):
        if word_len == 5:
            return list(self.PATTERN_5_4.values()) if line_count == 4 else list(self.PATTERN_5_8.values())
        return list(self.PATTERN_7_4.values()) if line_count == 4 else list(self.PATTERN_7_8.values())

    @staticmethod
    def _matches_pattern(tone_seq, pattern_seq):
        ch_to_en = {'平': 'ping', '仄': 'ze'}
        pat_en = [ch_to_en.get(c, c) for c in pattern_seq]
        word_len = len(tone_seq)
        positions = [1, 3] if word_len == 5 else [1, 3, 5]
        for pos in positions:
            if pos >= len(tone_seq) or pos >= len(pat_en): return False
            if tone_seq[pos] not in ('ping', 'ze'): continue
            if tone_seq[pos] != pat_en[pos]: return False
        return True

    # ── 押韵评分 ───────────────────────────────────────────────────────────
    def _score_rhyme(self, poem, num_lines) -> float:
        lines = [l.strip() for l in poem.split('\n') if l.strip()][:num_lines]
        if len(lines) < 4:
            return 0.0
        groups = [self._get_stable_rhyme(lines[i][-1]) for i in range(1, num_lines, 2)]
        unique = len(set(groups))
        return 1.0 if unique == 1 else 0.6 if unique == 2 else 0.3

    def _get_stable_rhyme(self, char: str) -> str:
        try:
            result = self._rhyme_checker.get_rhyme_group(char)
            if result and isinstance(result, list):
                ping_entry = next((e for e in result if e[0] == 'ping'), result[0])
                full = ping_entry[2]
                m = re.search(
                    r'[東冬江支微魚虞齊佳灰真文元寒刪先蕭肴豪歌麻陽庚青蒸尤侵覃鹽咸]',
                    full,
                )
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

    # ── 意象评分 ───────────────────────────────────────────────────────────
    @staticmethod
    def _score_imagery(poem: str) -> float:
        lines = [l.strip() for l in poem.split('\n') if l.strip()]
        text = ''.join(lines)
        matched = list(set(w for w in ALL_IMAGERY_WORDS if w in text))
        cnt = len(matched)
        diversity = len(set(text)) / max(len(text), 1)
        return min((cnt / max(len(lines), 1)) * 0.5 + diversity * 0.5, 1.0)

    # ── 主题聚合 ──────────────────────────────────────────────────────────
    @staticmethod
    def _score_theme_cohesion(poem: str) -> float:
        lines = [l.strip() for l in poem.split('\n') if l.strip()]
        text = ''.join(lines)
        theme_counts = {
            t: sum(1 for w in words if w in text)
            for t, words in EMOTION_THEMES.items()
        }
        total = sum(theme_counts.values())
        if total == 0:
            return 0.4
        return min(max(theme_counts.values()) / total * 1.2, 1.0)

    # ── 重复惩罚 ───────────────────────────────────────────────────────────
    @staticmethod
    def _repetition_penalty(poem: str) -> float:
        poem_text = ''.join(poem.split())
        length = len(poem_text)
        if length == 0:
            return 1.0
        exempt = set()
        i = 0
        while i < length - 1:
            if poem_text[i] == poem_text[i + 1]:
                exempt.add(i + 1)
                i += 2
            else:
                i += 1
        freq = Counter(ch for idx, ch in enumerate(poem_text) if idx not in exempt)
        max_repeat = max(freq.values()) if freq else 1
        if max_repeat <= 1: return 1.0
        if max_repeat == 2: return 0.95
        if max_repeat == 3: return 0.90
        if max_repeat == 4: return 0.85
        return 0.80

    @staticmethod
    def _check_synonym_clash(poem: str) -> float:
        """检测同一联中是否出现意象合掌（同义词群复用），触发折扣。

        例如同一句同时出现"银汉"和"银河"→判定合掌。
        """
        lines = [l.strip() for l in poem.split('\n') if l.strip()]
        penalty = 1.0
        hits_found = []
        for line in lines:
            for group in _SYNONYM_CLASH_GROUPS:
                matched = [word for word in group if word in line]
                if len(matched) >= 2:
                    penalty *= CLASH_PENALTY_PER_HIT
                    hits_found.append(f"{matched}")
                    _log.warning("[格律合掌] 行内意象重复: %s → ×%.2f", matched, CLASH_PENALTY_PER_HIT)
        if hits_found:
            _log.info("[格律合掌] 共发现 %d 处合掌，合掌系数=%.2f", len(hits_found), penalty)
        return round(max(0.1, penalty), 4)

    # ── Pairwise 比较（锦标赛模式）─────────────────────────────────────────

    _PAIRWISE_PROMPT = (
        "请比较以下两首古诗，判断哪一首更符合用户的要求。\n\n"
        "用户要求：{user_request}\n\n"
        "诗歌 A：\n{poem_a}\n\n"
        "诗歌 B：\n{poem_b}\n\n"
        "比较维度（同等重要）：\n"
        "1. 主题契合度：哪一首更准确、更完整地表达了用户要求的主题？\n"
        "2. 意象与意境：哪一首的画面感和意境更鲜明、统一？\n"
        "3. 语言质量：哪一首的用词更典雅、精炼、富有诗意？\n"
        "4. 格律合规：哪一首更符合平仄和押韵规范？\n\n"
        "只输出一个字母：A 或 B。不要输出任何其他内容。"
    )

    _TOPIC_FIT_PROMPT = (
        "你是一位严格的诗歌选题审核员。评判每首诗是否切合用户要求。\n\n"
        "用户要求：{user_request}\n\n"
        "{poem_texts}\n\n"
        "{anomaly_hints}"
        "按 0-10 分评判每首诗的主题契合度：\n"
        "  · 10分：完全切题，核心意象/情感/场景紧密围绕用户要求\n"
        "  · 7-9分：基本切题，主题明确但个别意象略有游离\n"
        "  · 4-6分：部分相关但整体偏离，或主要写其他主题\n"
        "  · 1-3分：明显离题，核心内容与要求存在矛盾\n"
        "     （如用户要求写春却出现秋景、要求写思乡却写隐逸说理、"
        "要求写夜景却写白昼）\n"
        "  · 0分：完全无关，全文与用户要求无任何关联\n\n"
        "【判分铁律】如果上面出现了【风险提示】中的时令冲突意象，"
        "且该意象在诗中并非虚写或合理联想，而是导致整首诗发生了时序矛盾、"
        "景物错乱（如前半写夏后半写秋），必须判为明显离题，打 1-3 分。\n\n"
        "只输出：诗1=X,诗2=Y,...（X,Y为0-10的数字，一行即可）"
    )

    @staticmethod
    def _extract_seasonal_clues(poems: list, user_request: str) -> str:
        """本地扫描嫌疑词，不判罪，只给 LLM 划重点。
        覆盖：季节矛盾、昼夜矛盾、天气矛盾。
        """
        SEASON_CONFLICTS = {
            "夏": ["秋风", "秋月", "秋声", "秋色", "秋景", "秋雨", "秋意",
                    "落叶", "霜飞", "霜叶", "残菊", "重阳", "寒露",
                    "白露", "萧瑟", "枯叶", "枯草", "雪飞", "朔风",
                    "春风", "春雨", "春花", "桃李", "杏花", "柳絮",
                    "春草", "春色", "春意", "寒梅", "残雪",
                    "秋", "霜", "雪", "冬", "春"],
            "春": ["秋风", "秋月", "秋声", "落叶", "霜飞", "霜叶",
                    "残菊", "重阳", "枯枝", "萧瑟", "雪飞", "朔风",
                    "盛夏", "蝉鸣", "暑气", "炎天", "蛙声", "荷花开",
                    "寒梅", "残雪", "枯草", "寒露", "白露",
                    "秋", "霜", "雪", "冬", "夏", "暑", "蝉", "荷"],
            "秋": ["春风", "春雨", "春花", "桃李", "杏花", "柳絮",
                    "春草", "春色", "春意", "桃花开",
                    "盛夏", "蝉鸣", "暑气", "炎天", "蛙声", "荷花开",
                    "寒梅", "残雪", "朔风", "雪飞",
                    "春", "夏", "暑", "蝉", "荷", "雪", "冬"],
            "冬": ["春风", "春雨", "春花", "桃李", "杏花", "柳絮",
                    "春草", "春色", "春意", "桃花开",
                    "蝉鸣", "盛夏", "蛙声", "暑气", "荷花开",
                    "秋风", "秋月", "秋雨", "落叶", "残菊", "重阳",
                    "春", "夏", "暑", "蝉", "荷", "秋"],
        }
        TIME_CONFLICTS = {
            "夜":  ["日午", "正午", "晌午", "白昼", "日中", "日高",
                     "晴午", "午时", "朝日", "烈日"],
            "晚":  ["日午", "正午", "晌午", "烈日", "朝日"],
            "白昼": ["夜半", "夜深", "月明", "星辉", "灯火", "残灯"],
            "晨":  ["夜半", "夜深", "黄昏", "暮色", "夕阳", "残阳"],
            "暮":  ["朝日", "晨光", "旭日", "晓色", "曙色"],
        }
        WEATHER_CONFLICTS = {
            "雨":  ["晴空", "烈日", "骄阳", "旱", "晴光"],
            "晴":  ["雨声", "烟雨", "霖", "濛濛", "淅沥"],
            "雪":  ["暑气", "炎天", "烈日", "骄阳", "荷花开"],
            "风":  [],  # 风几乎可以和任何天气共存，不检查
        }
        hints = []
        # 季节矛盾
        matched_season = None
        for season in SEASON_CONFLICTS:
            if season in user_request:
                matched_season = season
                break
        if matched_season:
            for i, p in enumerate(poems):
                conflicts = [w for w in SEASON_CONFLICTS[matched_season] if w in p]
                if conflicts:
                    hints.append(
                        f"  ⚠ 诗{i+1} 季节风险：要求'{matched_season}'但出现 {', '.join(conflicts[:5])}"
                        f"{'...' if len(conflicts) > 5 else ''}"
                    )
        # 昼夜矛盾
        for time_key, time_words in TIME_CONFLICTS.items():
            if time_key in user_request and time_words:
                for i, p in enumerate(poems):
                    conflicts = [w for w in time_words if w in p]
                    if conflicts:
                        hints.append(
                            f"  ⚠ 诗{i+1} 昼夜风险：要求'{time_key}'但出现 {', '.join(conflicts[:4])}"
                        )
        # 天气矛盾
        for weather_key, weather_words in WEATHER_CONFLICTS.items():
            if weather_key in user_request and weather_words:
                for i, p in enumerate(poems):
                    conflicts = [w for w in weather_words if w in p]
                    if conflicts:
                        hints.append(
                            f"  ⚠ 诗{i+1} 天气风险：要求'{weather_key}'但出现 {', '.join(conflicts[:4])}"
                        )
        if hints:
            return ("【风险提示】以下诗作包含疑似矛盾意象，"
                    "请甄别是真实矛盾还是虚写联想：\n" + "\n".join(hints))
        return ""

    def score_topic_fit(self, poems: list, user_request: str,
                        adapter) -> dict:
        """一次 LLM 调用评估所有诗的主题契合度，返回 {idx: 0.0-1.0}。"""
        poem_texts = '\n\n'.join(
            f"诗{i+1}：\n{p.strip()}" for i, p in enumerate(poems)
        )
        anomaly_hints = self._extract_seasonal_clues(poems, user_request)
        if anomaly_hints:
            _log.info("本地嫌疑扫描:\n%s", anomaly_hints)
        prompt = self._TOPIC_FIT_PROMPT.format(
            user_request=user_request, poem_texts=poem_texts,
            anomaly_hints=anomaly_hints + "\n\n" if anomaly_hints else "",
        )
        messages = [
            {"role": "system",
             "content": "你是一位严格的诗歌选题审核员。只输出 诗X=Y 格式。"},
            {"role": "user", "content": prompt},
        ]
        try:
            reply = adapter.generate(messages, max_tokens=50, temperature=0.1)
            import re
            scores = {}
            for m in re.finditer(r"诗(\d+)\s*[=：:]\s*(\d+)", reply):
                idx = int(m.group(1)) - 1
                score = min(10, max(0, int(m.group(2)))) / 10.0
                scores[idx] = score
            # 缺失项用本地启发式回落（而非盲目 0.5），保留 Arena 区分度
            fallback = None
            for i in range(len(poems)):
                if i not in scores:
                    if fallback is None:
                        fallback = self._heuristic_topic_scores(poems, user_request)
                    scores[i] = fallback[i]
            _log.info("主题契合度: %s",
                      ' | '.join(f"诗{i+1}={scores[i]:.1f}" for i in range(len(poems))))
            return scores
        except Exception as e:
            _log.warning("主题契合评估 LLM 失败: %s，回落本地启发式", e)
            return self._heuristic_topic_scores(poems, user_request)

    def _heuristic_topic_scores(self, poems: list, user_request: str) -> dict:
        """LLM 切题评分失败时的本地启发式回落。

        从 user_request 中扫描已知主题词（THEME_SYNONYMS keys），统计每首诗对
        各主题词的同义词命中比例，映射到 [0.4, 0.9] 区间，保持 Arena 排序区分度。
        无可识别主题词时仍回 0.5。
        """
        from core.poem.theme import THEME_SYNONYMS, get_imagery_synonyms, strip_meta_prefix
        req = strip_meta_prefix(user_request or "")
        topic_keys = [k for k in THEME_SYNONYMS if k in req]
        if not topic_keys:
            return {i: 0.5 for i in range(len(poems))}
        scores = {}
        for i, p in enumerate(poems):
            hits = 0
            for k in topic_keys:
                syns = set(get_imagery_synonyms(k)) | {k}
                if any(s in p for s in syns):
                    hits += 1
            ratio = hits / len(topic_keys)
            scores[i] = round(0.4 + ratio * 0.5, 2)
        _log.info("[启发式切题] 关键词=%s, 分数=%s",
                  topic_keys, {f"诗{i+1}": scores[i] for i in range(len(poems))})
        return scores

    def local_score_poem(self, poem: str, num_lines: int,
                         chars_per_line: int,
                         topic_score: float = 0.5) -> dict:
        """本地评分：平仄 + 押韵 + 意象 + 连贯 + 切题。切题分由 LLM 评估传入。"""
        from config import (LOCAL_PINGZE_WT, LOCAL_RHYME_WT,
                            LOCAL_IMAGERY_WT, LOCAL_COHESION_WT,
                            LOCAL_TOPIC_WT)
        pz = self._score_pingze(poem, num_lines, chars_per_line)
        rh = self._score_rhyme(poem, num_lines)
        im = self._score_imagery(poem)
        co = self._score_theme_cohesion(poem)
        total = (pz * LOCAL_PINGZE_WT + rh * LOCAL_RHYME_WT +
                 im * LOCAL_IMAGERY_WT + co * LOCAL_COHESION_WT +
                 topic_score * LOCAL_TOPIC_WT)
        return {"pingze": round(pz, 3), "rhyme": round(rh, 3),
                "imagery": round(im, 3), "cohesion": round(co, 3),
                "topic": round(topic_score, 2),
                "total": round(total, 3)}

    def generation_hard_gate(self, poem: str, num_lines: int,
                             chars_per_line: int) -> dict:
        """生成阶段的硬门控（比进化阶段更严，不调 LLM）。"""
        from config import BAD_PATTERNS
        reasons = []
        lines = [l.strip() for l in poem.split('\n') if l.strip()]
        if len(lines) < num_lines:
            reasons.append(f"行数不足（需{num_lines}行）")
        if any(len(l) != chars_per_line for l in lines[:num_lines]):
            reasons.append("字数不符")
        if reasons:
            return {"passed": False, "reasons": reasons}

        pz = self._score_pingze(poem, num_lines, chars_per_line)
        rh = self._score_rhyme(poem, num_lines)
        if rh < 0.5:
            reasons.append(f"押韵严重不合格({rh:.2f})")
        if pz < 0.6:
            reasons.append(f"平仄严重不合格({pz:.2f})")
        poem_text = ''.join(lines)
        hits = [w for w in BAD_PATTERNS if w in poem_text]
        if hits:
            reasons.append(f"AI堆砌词: {', '.join(hits)}")
        # 重度重复检查
        pen = self._repetition_penalty(poem)
        if pen < 0.90:
            reasons.append(f"重复严重(penalty={pen:.2f})")

        return {"passed": len(reasons) == 0, "reasons": reasons,
                "rhyme": rh, "pingze": pz}

    def hard_gate_and_score(self, poems: list, user_request: str,
                            adapter, num_lines: int,
                            chars_per_line: int) -> dict:
        """硬门控 + 切题评估 + 本地评分。不做 pairwise，供上游攒够合格诗后统一 arena。

        返回: {gated: [{idx, poem, local}], rejected: [{idx, poem, reasons}]}
        """
        # 1. 硬门控
        gated = []
        rejected = []
        for i, p in enumerate(poems):
            gate = self.generation_hard_gate(p, num_lines, chars_per_line)
            if gate["passed"]:
                gated.append({"idx": i, "poem": p})
            else:
                rejected.append({"idx": i, "poem": p, "reasons": gate["reasons"]})

        # 2. 切题评估
        raw_topic = {}
        if gated:
            all_poems = [g["poem"] for g in gated]
            raw_topic = self.score_topic_fit(all_poems, user_request, adapter)
        # 映射回原始索引
        topic_scores = {}
        for local_i, g in enumerate(gated):
            topic_scores[g["idx"]] = raw_topic.get(local_i, 0.5)

        # 3. 本地评分
        for g in gated:
            ts = topic_scores.get(g["idx"], 0.5)
            g["local"] = self.local_score_poem(
                g["poem"], num_lines, chars_per_line, topic_score=ts,
            )

        # 打印日志
        _log.info("=" * 60)
        _log.info("【海选·硬门控】%d/%d 首通过", len(gated), len(poems))
        for r in rejected:
            _log.info("  ✗ 诗%d 淘汰: %s", r["idx"] + 1, "; ".join(r["reasons"]))
            for line in r["poem"].strip().split('\n'):
                _log.info("    %s", line.strip())
        for g in gated:
            s = g["local"]
            _log.info("  ✓ 诗%d 本地分=%.3f (平仄=%.2f 押韵=%.2f 意象=%.2f 连贯=%.2f 切题=%.2f)",
                      g["idx"] + 1, s["total"], s["pingze"], s["rhyme"],
                      s["imagery"], s["cohesion"], s["topic"])
            for line in g["poem"].strip().split('\n'):
                _log.info("    %s", line.strip())

        return {"gated": gated, "rejected": rejected}

    def arena_from_gated(self, gated: list, user_request: str,
                         adapter) -> dict:
        """对已评分的门控通过诗做 arena pairwise 选冠军。不重复评分。"""
        from config import ARENA_LOCAL_WT, ARENA_PAIRWISE_WT

        if not gated:
            return {"champion": "", "champion_idx": -1, "backup": "",
                    "backup_idx": -1, "champion_topic": 0.5,
                    "champion_local_total": 0.0, "champion_final": 0.0,
                    "gated_count": 0, "top3_indices": [], "arena_results": [],
                    "all_local_scores": []}

        gated.sort(key=lambda x: x["local"]["total"], reverse=True)
        top_n = min(len(gated), 3)

        _log.info("-" * 60)
        _log.info("【Arena 决选·%d 首候选】", len(gated))
        for g in gated[:top_n]:
            s = g["local"]
            _log.info("  诗%d 本地分=%.3f (平仄=%.2f 押韵=%.2f 意象=%.2f 连贯=%.2f 切题=%.2f)",
                      g["idx"] + 1, s["total"], s["pingze"], s["rhyme"],
                      s["imagery"], s["cohesion"], s["topic"])

        arena_results = []
        arena_wins = {g["idx"]: 0 for g in gated}
        if top_n >= 3:
            pairs = [(0, 1), (0, 2), (1, 2)]
        elif top_n == 2:
            pairs = [(0, 1)]
        else:
            pairs = []
        max_wins = len(pairs) if top_n >= 2 else 0

        for ai, bi in pairs:
            a, b = gated[ai], gated[bi]
            winner = self.compare_poems(a["poem"], b["poem"],
                                        user_request, adapter)
            win_idx = a["idx"] if winner == "A" else b["idx"]
            arena_wins[win_idx] += 1
            arena_results.append({"a_idx": a["idx"], "b_idx": b["idx"],
                                  "winner": winner})
            _log.info("  诗%d vs 诗%d → %s 胜", a["idx"] + 1, b["idx"] + 1,
                      "A" if winner == "A" else "B")

        for t in gated[:top_n]:
            wins = arena_wins[t["idx"]]
            arena_score = wins / max(max_wins, 1)
            t["final"] = (t["local"]["total"] * ARENA_LOCAL_WT +
                          arena_score * ARENA_PAIRWISE_WT)
            t["arena_wins"] = wins
            t["arena_score"] = arena_score
            _log.info("  诗%d 本地=%.3f arena=%d胜 综合=%.3f",
                      t["idx"] + 1, t["local"]["total"], wins, t["final"])

        gated.sort(key=lambda x: x.get("final", x["local"]["total"]),
                   reverse=True)
        champion = gated[0]
        backup = gated[1] if len(gated) > 1 else gated[0]

        _log.info("=" * 60)
        _log.info("【🏆 冠军】诗%d 综合=%.3f", champion["idx"] + 1, champion["final"])
        for line in champion["poem"].strip().split('\n'):
            _log.info("  %s", line.strip())
        _log.info("=" * 60)

        return {
            "champion": champion["poem"],
            "champion_idx": champion["idx"],
            "champion_topic": champion["local"]["topic"],
            "champion_local_total": champion["local"]["total"],
            "champion_final": champion["final"],
            "gated_count": len(gated),
            "backup": backup["poem"],
            "backup_idx": backup["idx"],
            "top3_indices": [g["idx"] for g in gated[:top_n]],
            "arena_results": arena_results,
            "all_local_scores": [{"idx": g["idx"],
                                  "local": g["local"]["total"],
                                  "pingze": g["local"]["pingze"],
                                  "rhyme": g["local"]["rhyme"],
                                  "imagery": g["local"]["imagery"],
                                  "cohesion": g["local"]["cohesion"],
                                  "topic": g["local"]["topic"]}
                                 for g in gated],
        }

    def arena_select_champion(self, poems: list, user_request: str,
                              adapter, num_lines: int,
                              chars_per_line: int) -> dict:
        """Arena 海选：硬门控 → 本地分 Top3 → 轮循 pairwise → 综合定冠军。

        返回: {champion, champion_idx, backup, backup_idx,
                top3_indices, arena_results, all_local_scores, rejected}
        """
        from config import ARENA_LOCAL_WT, ARENA_PAIRWISE_WT

        # 1. 硬门控过滤
        gated = []
        rejected = []
        for i, p in enumerate(poems):
            gate = self.generation_hard_gate(p, num_lines, chars_per_line)
            if gate["passed"]:
                gated.append({"idx": i, "poem": p})
            else:
                rejected.append({"idx": i, "poem": p, "reasons": gate["reasons"]})

        # 2. 评估主题契合度（一次 LLM 调用，所有诗一起判）
        all_poems = [g["poem"] for g in gated]
        raw_topic_scores = self.score_topic_fit(all_poems, user_request, adapter) if gated else {}
        # score_topic_fit 返回 {局部位置: 分数}，需要映射回原始索引
        gated_indices = [g["idx"] for g in gated]
        topic_scores = {}
        for local_i, g in enumerate(gated):
            topic_scores[g["idx"]] = raw_topic_scores.get(local_i, 0.5)

        # 3. 本地评分（含切题维度）
        for g in gated:
            ts = topic_scores.get(g["idx"], 0.5)
            g["local"] = self.local_score_poem(
                g["poem"], num_lines, chars_per_line, topic_score=ts,
            )

        _log.info("=" * 60)
        _log.info("【海选·硬门控】%d/%d 首通过", len(gated), len(poems))
        for r in rejected:
            _log.info("  ✗ 诗%d 淘汰: %s", r["idx"] + 1, "; ".join(r["reasons"]))
            for line in r["poem"].strip().split('\n'):
                _log.info("    %s", line.strip())
        for g in gated:
            s = g["local"]
            _log.info("  ✓ 诗%d 本地分=%.3f (平仄=%.2f 押韵=%.2f 意象=%.2f 连贯=%.2f 切题=%.2f)",
                      g["idx"] + 1, s["total"], s["pingze"], s["rhyme"],
                      s["imagery"], s["cohesion"], s["topic"])
            for line in g["poem"].strip().split('\n'):
                _log.info("    %s", line.strip())

        if not gated:
            _log.warning("Arena: 无候选通过门控")
            return {
                "champion": poems[0], "champion_idx": 0,
                "champion_topic": 0.5,
                "champion_local_total": 0.0,
                "champion_final": 0.0,
                "gated_count": 0,
                "backup": poems[1] if len(poems) > 1 else poems[0],
                "backup_idx": 1 if len(poems) > 1 else 0,
                "top3_indices": [0], "arena_results": [],
                "all_local_scores": [],
                "rejected": rejected,
            }

        # 4. 按本地分排序
        gated.sort(key=lambda x: x["local"]["total"], reverse=True)
        top_n = min(len(gated), 3)

        _log.info("-" * 60)
        _log.info("【海选·Top%d 晋级 Arena】", top_n)

        # 5. 轮循 pairwise arena
        arena_results = []
        arena_wins = {g["idx"]: 0 for g in gated}
        if top_n >= 3:
            pairs = [(0, 1), (0, 2), (1, 2)]
        elif top_n == 2:
            pairs = [(0, 1)]
        else:
            pairs = []
        max_wins = len(pairs) if top_n >= 2 else 0
        for ai, bi in pairs:
            a, b = gated[ai], gated[bi]
            winner = self.compare_poems(a["poem"], b["poem"],
                                        user_request, adapter)
            win_idx = a["idx"] if winner == "A" else b["idx"]
            arena_wins[win_idx] += 1
            arena_results.append({"a_idx": a["idx"], "b_idx": b["idx"],
                                  "winner": winner})
            _log.info("  诗%d vs 诗%d → %s 胜", a["idx"] + 1, b["idx"] + 1,
                      "A" if winner == "A" else "B")

        # 6. 综合分 = 本地 * 0.85 + arena * 0.15
        for t in gated[:top_n]:
            wins = arena_wins[t["idx"]]
            arena_score = wins / max(max_wins, 1)
            t["final"] = (t["local"]["total"] * ARENA_LOCAL_WT +
                          arena_score * ARENA_PAIRWISE_WT)
            t["arena_wins"] = wins
            t["arena_score"] = arena_score
            _log.info("  诗%d 本地=%.3f arena=%d胜 综合=%.3f",
                      t["idx"] + 1, t["local"]["total"], wins, t["final"])

        # 7. 冠军 = 最高综合分
        gated.sort(key=lambda x: x.get("final", x["local"]["total"]), reverse=True)
        champion = gated[0]
        backup = gated[1] if len(gated) > 1 else gated[0]

        _log.info("=" * 60)
        _log.info("【🏆 冠军】诗%d 综合=%.3f", champion["idx"] + 1, champion["final"])
        for line in champion["poem"].strip().split('\n'):
            _log.info("  %s", line.strip())
        _log.info("=" * 60)

        return {
            "champion": champion["poem"],
            "champion_idx": champion["idx"],
            "champion_topic": champion["local"]["topic"],
            "champion_local_total": champion["local"]["total"],
            "champion_final": champion["final"],
            "gated_count": len(gated),
            "backup": backup["poem"],
            "backup_idx": backup["idx"],
            "top3_indices": [g["idx"] for g in gated[:top_n]],
            "arena_results": arena_results,
            "all_local_scores": [{"idx": g["idx"],
                                  "local": g["local"]["total"],
                                  "pingze": g["local"]["pingze"],
                                  "rhyme": g["local"]["rhyme"],
                                  "imagery": g["local"]["imagery"],
                                  "cohesion": g["local"]["cohesion"],
                                  "topic": g["local"]["topic"]}
                                 for g in gated],
            "rejected": rejected,
        }

    def compare_poems(self, poem_a: str, poem_b: str, user_request: str,
                      adapter) -> str:
        """成对决斗：返回 "A" 或 "B"。"""
        prompt = self._PAIRWISE_PROMPT.format(
            user_request=user_request, poem_a=poem_a, poem_b=poem_b,
        )
        messages = [
            {"role": "system", "content": "你是一位严苛的古典诗词评委。只输出 A 或 B。"},
            {"role": "user", "content": prompt},
        ]
        try:
            reply = adapter.generate(messages, max_tokens=10, temperature=0.1)
            reply = reply.strip().upper()
            if "B" in reply and "A" not in reply:
                return "B"
            return "A"
        except Exception as e:
            _log.warning("Pairwise 比较异常: %s，默认判 A 胜", e)
            return "A"

    # ── BWS（Best-Worst-Scaling 简化版：N 候选盲选 1 个 best）─────────────

    # 强约束：开头先讲格式（reasoning 模型容易把"评判维度"理解成要推理），
    # 显式禁止任何额外文字、给出格式示例，把数字范围放在最显眼的位置。
    _BWS_PROMPT = (
        "请从下列 {n} 首古诗中选出最佳的一首。\n"
        "【输出格式】严格仅返回 1-{n} 之间的一个阿拉伯数字，不允许任何其他字符、"
        "解释、推理过程、标点。\n"
        "【输出示例】3\n\n"
        "用户要求：{user_request}\n\n"
        "{poems_block}\n"
        "评判维度（同等重要）：主题契合、意象意境、语言典雅、格律合规。\n"
        "{imagery_grounding}"
        "\n你的回答（仅一个数字）："
    )

    # 古典意象字典 in-context grounding（懒加载 + 类级缓存）
    _imagery_grounding_cache: str = ""

    @classmethod
    def _load_imagery_grounding(cls) -> str:
        """加载 eval/assets/classical_imagery.json 并格式化为 BWS prompt 附录文本。

        缓存到类变量，进程内只加载一次。字典缺失时返回空串（BWS 退化为原行为）。
        """
        if cls._imagery_grounding_cache:
            return "" if cls._imagery_grounding_cache == "_NO_DICT_" else cls._imagery_grounding_cache
        try:
            import json as _json
            import os as _os
            here = _os.path.dirname(_os.path.abspath(__file__))
            root = _os.path.abspath(_os.path.join(here, "..", ".."))
            path = _os.path.join(root, "eval", "assets", "classical_imagery.json")
            if not _os.path.exists(path):
                cls._imagery_grounding_cache = "_NO_DICT_"
                return ""
            with open(path, encoding="utf-8") as f:
                data = _json.load(f)
            from collections import defaultdict
            grouped = defaultdict(list)
            for word, info in data.get("entries", {}).items():
                cat = info.get("category", "其他")
                meaning = info.get("meaning", "")
                if len(meaning) > 30:
                    meaning = meaning[:30] + "…"
                grouped[cat].append(f"{word}={meaning}")
            lines = [
                "\n【古典意象典故参考】（认识下列词，识别诗中含蓄用典；不要因为字面没出现用户要求词就扣分）",
            ]
            for cat in ("植物", "动物", "天象时令", "情感典故", "地理场景",
                        "人物典故", "器物", "复合意象"):
                if cat in grouped:
                    lines.append(f"· {cat}：" + "；".join(grouped[cat]))
            cls._imagery_grounding_cache = "\n".join(lines) + "\n"
            _log.info("BWS imagery grounding 已加载 %d 词条",
                      len(data.get("entries", {})))
            return cls._imagery_grounding_cache
        except Exception as e:
            _log.warning("加载 classical_imagery.json 失败 → BWS 不注入 grounding: %s", e)
            cls._imagery_grounding_cache = "_NO_DICT_"
            return ""

    @staticmethod
    def _parse_bws_reply(reply: str, n: int) -> int:
        """从评委回复里抽出 1..n 范围内的有效数字，返回 0-based idx；找不到返回 -1。

        策略：先看整段回复是不是单一数字（最理想）；否则从**末尾**向前扫，
        取最后一个独立的 1..n 范围数字（reasoning 模型推理之后才给出答案，
        且能避开 prompt 复述里的"5 首""第 1 首"等干扰词）。
        """
        if not reply:
            return -1
        s = reply.strip()
        # 1) 整段就是一个数字（理想情况）
        if s.isdigit():
            v = int(s)
            return v - 1 if 1 <= v <= n else -1
        # 2) 末尾向前找最后一个独立数字（要求左侧不是数字/汉字数字，避免"15"→1）
        matches = list(re.finditer(r"(?<![0-9])([1-9][0-9]?)(?![0-9])", s))
        for m in reversed(matches):
            v = int(m.group(1))
            if 1 <= v <= n:
                return v - 1
        return -1

    def pick_best_via_bws(self, poems: list, user_request: str,
                          adapter) -> dict:
        """单评委 N 选 1：返回 {'best_idx': int, 'raw_reply': str}。

        best_idx 含义：
            >= 0  : 评委明确选了该索引（0-based）
            -1    : 评委弃权（回复无法解析为 1..N 范围内的合法数字）
                    —— 外层 _run_one 看到 -1 不计票，全员弃权按本地分兜底

        BWS 不要求评委给具体分数，只挑最好的一首，规避绝对评分饱和。
        外层应跨多个评委独立调用本方法，再多数决聚合得到该模型的 best。
        """
        if not poems:
            return {"best_idx": -1, "raw_reply": "", "shuffle_perm": []}
        if len(poems) == 1:
            return {"best_idx": 0, "raw_reply": "1", "shuffle_perm": [0]}

        # 打散候选顺序，规避评委的 position bias（习惯性选第 1/最后一首）。
        # perm[shuffled_idx] = original_idx；评委选 shuffled_idx 后映射回原 idx。
        perm = list(range(len(poems)))
        random.shuffle(perm)
        shuffled_poems = [poems[i] for i in perm]

        poems_block = "\n".join(
            f"诗 {i + 1}：\n{p.strip()}\n" for i, p in enumerate(shuffled_poems)
        )
        prompt = self._BWS_PROMPT.format(
            n=len(poems), user_request=user_request, poems_block=poems_block,
            imagery_grounding=self._load_imagery_grounding(),
        )
        messages = [
            {"role": "system",
             "content": "你是古典诗词评委。仅输出阿拉伯数字，禁止任何文字、解释、推理。"},
            {"role": "user", "content": prompt},
        ]
        try:
            reply = adapter.generate(messages, max_tokens=10, temperature=0.0)
            reply_clean = (reply or "").strip()
            shuffled_idx = self._parse_bws_reply(reply_clean, len(poems))
            if shuffled_idx == -1:
                _log.warning("BWS 回复无法解析: %r → 评委弃权", reply_clean[:80])
                return {"best_idx": -1, "raw_reply": reply_clean,
                        "shuffle_perm": perm}
            original_idx = perm[shuffled_idx]
            return {"best_idx": original_idx, "raw_reply": reply_clean,
                    "shuffle_perm": perm, "shuffled_idx": shuffled_idx}
        except Exception as e:
            _log.warning("BWS 选 best 异常: %s → 评委弃权", e)
            return {"best_idx": -1, "raw_reply": f"ERR: {e}",
                    "shuffle_perm": perm}
