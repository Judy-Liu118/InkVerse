"""
core.poem.generator -- 诗歌生成器

两阶段分离：先全部生成候选，再统一评分，避免 fine/base 模型同时占用显存。
"""
import re
import warnings
import torch
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import POEM_CANDIDATE_COUNT, POEM_MAX_TOKENS, POEM_TEMPERATURE, GENRE_CONFIG
from core.models.manager import ModelManager
from core.poem.scorer import PoemScorer
from core.logger import get_logger

_log = get_logger(__name__)


class PoemGenerator:
    def __init__(self):
        self.mm = ModelManager()
        self.scorer = PoemScorer()

    # ── 主入口 ─────────────────────────────────────────────────────────────
    def generate(self, user_request: str, score_adapter,
                 generation_adapter=None, creative_brief: str = "") -> Tuple:
        """传统单轮生成（向后兼容），不做品质筛选。"""
        genre_name, num_lines, chars_per_line = self.scorer.detect_genre(user_request)

        _backend = getattr(generation_adapter, 'backend', '') if generation_adapter else ''
        use_api, model, tokenizer = self._resolve_model(generation_adapter, _backend)

        candidates = self._generate_candidates(
            user_request, generation_adapter, model, tokenizer,
            num_lines, chars_per_line, use_api, creative_brief, POEM_CANDIDATE_COUNT,
            start_index=0,
        )

        if not use_api and model is not None:
            del model, tokenizer
            self.mm._flush_gpu()

        if not candidates:
            _log.error("所有候选诗生成失败")
            return genre_name, "生成失败，请重试", 0.0, 0.0

        scored = self._score_candidates(candidates, num_lines, chars_per_line,
                                        user_request, score_adapter, use_api)
        self._log_candidates_summary(scored, "全部候选诗评分明细")

        scored.sort(key=lambda x: x[0]["total"], reverse=True)
        best_scores, best_poem = scored[0]
        self._log_best_poem(best_scores, best_poem)

        final_lines = [l.strip() for l in best_poem.split('\n') if l.strip()][:num_lines]
        art_quality = best_scores.get("art_quality", best_scores.get("raw_total", best_scores["total"]))
        return genre_name, '\n'.join(final_lines), best_scores['total'], art_quality

    # ── 品质控制入口 ───────────────────────────────────────────────────────
    def generate_with_quality_control(
        self, user_request: str, score_adapter,
        generation_adapter=None, creative_brief: str = "",
        quality_threshold: float = 0.70,
        max_discard_per_batch: int = 2,
        max_rounds: int = 3,
        min_qualified: int = 3,
    ) -> dict:
        """带品质筛选的诗歌生成。

        每轮生成一批候选 → 评分 → 按 total 分流（合格/废弃），
        合格池达标后停止补充生成，从合格池中选总分最高的作为最优结果。
        若全部候选都不合格，fallback 选最高分并发出警告。
        """
        genre_name, num_lines, chars_per_line = self.scorer.detect_genre(user_request)

        _backend = getattr(generation_adapter, 'backend', '') if generation_adapter else ''
        use_api, model, tokenizer = self._resolve_model(generation_adapter, _backend)

        qualified_pool = []   # [(scores_dict, poem_text), ...]
        rejected_pool  = []   # [(scores_dict, poem_text), ...]
        total_generated = 0

        for round_idx in range(max_rounds):
            batch_size = max(1, POEM_CANDIDATE_COUNT - len(qualified_pool))
            _log.info("=" * 70)
            _log.info("第 %d/%d 轮生成 (%d 首候选)", round_idx + 1, max_rounds, batch_size)

            candidates = self._generate_candidates(
                user_request, generation_adapter, model, tokenizer,
                num_lines, chars_per_line, use_api, creative_brief, batch_size,
                start_index=total_generated,
            )
            total_generated += len(candidates)

            if not candidates:
                _log.warning("第 %d 轮无候选生成成功，跳过", round_idx + 1)
                continue

            scored = self._score_candidates(candidates, num_lines, chars_per_line,
                                            user_request, score_adapter, use_api)

            # ── 分流 ────────────────────────────────────────────────────────
            round_qualified = []
            round_rejected  = []
            for sc, pm in scored:
                if sc['total'] >= quality_threshold:
                    round_qualified.append((sc, pm))
                else:
                    round_rejected.append((sc, pm))

            qualified_pool.extend(round_qualified)
            rejected_pool.extend(round_rejected)

            self._log_round_summary(round_idx + 1, round_qualified, round_rejected,
                                    len(qualified_pool), min_qualified)

            # ── 停止条件 ────────────────────────────────────────────────────
            if len(qualified_pool) >= min_qualified:
                _log.info("✓ 合格池已达 %d 首（要求 ≥ %d），停止补充生成",
                         len(qualified_pool), min_qualified)
                break

            if len(round_rejected) <= max_discard_per_batch and len(qualified_pool) >= 1:
                # 本轮废弃不多，合格池虽不足但有候选可用
                if round_idx >= 1:  # 至少跑了一轮补充
                    _log.info("✓ 合格池 %d 首可用，停止补充生成", len(qualified_pool))
                    break

        if not use_api and model is not None:
            del model, tokenizer
            self.mm._flush_gpu()

        # ── 选最优 ────────────────────────────────────────────────────────
        if qualified_pool:
            best_scores, best_poem = max(qualified_pool, key=lambda x: x[0]["total"])
            selection_mode = "qualified_only"
        else:
            _log.warning("⚠ 所有候选均未达到品质线 %.2f，fallback 选最高分", quality_threshold)
            best_scores, best_poem = max(rejected_pool, key=lambda x: x[0]["total"])
            selection_mode = "fallback"

        self._log_best_poem(best_scores, best_poem)

        final_lines = [l.strip() for l in best_poem.split('\n') if l.strip()][:num_lines]
        art_quality = best_scores.get("art_quality", best_scores.get("raw_total", best_scores["total"]))

        return {
            "genre_name":       genre_name,
            "best_poem":        '\n'.join(final_lines),
            "best_score":       best_scores['total'],
            "best_art_quality": art_quality,
            "qualified": [{"poem": pm, "scores": sc} for sc, pm in qualified_pool],
            "rejected":  [{"poem": pm, "scores": sc} for sc, pm in rejected_pool],
            "total_rounds":     round_idx + 1,
            "total_generated":  total_generated,
            "selection_mode":   selection_mode,
        }

    # ── 内部辅助方法 ───────────────────────────────────────────────────────
    def _resolve_model(self, generation_adapter, backend: str):
        """解析模型类型，返回 (use_api, model, tokenizer)。"""
        if generation_adapter is not None and backend not in ('local', 'local_lora'):
            _log.info("使用 API 模型: %s %s", backend, generation_adapter.api_model)
            return True, None, None
        if backend == 'local_lora':
            _log.info("使用本地微调模型 (LoRA)")
        else:
            _log.info("使用本地基础模型")
        model = self.mm.fine_model if backend == 'local_lora' else self.mm.base_model
        tokenizer = self.mm.fine_tokenizer if backend == 'local_lora' else self.mm.base_tokenizer
        return False, model, tokenizer

    def _generate_candidates(self, user_request, generation_adapter,
                             model, tokenizer, num_lines, chars_per_line,
                             use_api: bool, creative_brief: str,
                             count: int, start_index: int = 0) -> list:
        """生成 count 首候选诗，返回 poem 文本列表。"""
        candidates = []
        for i in range(count):
            idx = start_index + i + 1
            if use_api:
                prompt = self._build_api_prompt(user_request, chars_per_line, num_lines,
                                                brief=creative_brief)
                messages = [
                    {"role": "system", "content": "你是一位精通中国古典诗词的创作专家。请严格按照格律生成。"},
                    {"role": "user", "content": prompt},
                ]
                _log.debug("API prompt（候选 %d/%d）:\n%s", idx, count + start_index, prompt)
                raw = generation_adapter.generate(
                    messages, max_tokens=POEM_MAX_TOKENS,
                    temperature=POEM_TEMPERATURE + i * 0.08,
                )
            else:
                temp = POEM_TEMPERATURE + i * 0.08
                if not any(g in user_request for g in ["五言", "七言", "绝句", "律诗"]):
                    enhanced_request = f"{user_request}。请以五言绝句创作，每句5字，共4句。"
                else:
                    enhanced_request = user_request
                _log.debug("LoRA prompt（候选 %d）:\n%s", idx, enhanced_request)
                raw = self._call_model(model, tokenizer, enhanced_request, POEM_MAX_TOKENS, temp)

            poem = self._normalize(raw, chars_per_line)
            lines = [l for l in poem.split('\n') if l.strip()][:num_lines]
            if len(lines) >= num_lines:
                candidates.append('\n'.join(lines))
                _log.info("候选诗 %d 生成成功", idx)
            else:
                _log.warning("候选诗 %d 生成失败，跳过", idx)
        return candidates

    def _score_candidates(self, candidates, num_lines, chars_per_line,
                          user_request, score_adapter, _use_api: bool = True) -> list:
        """对候选诗列表评分，返回 [(scores_dict, poem_text), ...] 列表。

        评分模型始终是 API（qwen/deepseek），因此始终并行调用。
        """
        scored = []
        _log.info("开始评分... (共 %d 首候选)", len(candidates))
        from concurrent.futures import ThreadPoolExecutor, as_completed
        futures = {}
        with ThreadPoolExecutor(max_workers=min(len(candidates), 5)) as pool:
            for idx, poem in enumerate(candidates):
                fut = pool.submit(
                    self.scorer.evaluate_full, poem, num_lines, chars_per_line,
                    user_request, score_adapter, idx + 1,
                )
                futures[fut] = (idx, poem)
        results = {}
        for fut in as_completed(futures):
            idx, poem = futures[fut]
            try:
                results[idx] = (fut.result(), poem)
            except Exception as e:
                _log.error("候选诗 %d 评分异常: %s", idx + 1, e)
                results[idx] = ({"total": 0, "intent": 0, "intent_llm": 0,
                                 "pingze": 0, "rhyme": 0, "imagery": 0, "cohesion": 0,
                                 "penalty": 0, "raw_total": 0}, poem)
        scored = [results[i] for i in sorted(results)]
        return scored

    @staticmethod
    def _log_candidates_summary(scored: list, title: str = "全部候选诗评分明细") -> None:
        """打印候选诗评分摘要（不含合格/废弃标记）。"""
        _log.info("=" * 70)
        _log.info("%s (%d 首):", title, len(scored))
        for rank_idx, (sc, pm) in enumerate(scored):
            poem_one_line = " | ".join(l.strip() for l in pm.split('\n') if l.strip())
            _log.info("【候选 %d】%s", rank_idx + 1, poem_one_line)
            _log.info("  → 总分=%.3f | 意图=%.3f[LLM=%.3f] | 平仄=%.3f | 押韵=%.3f | 意象=%.3f | 聚合=%.3f | 重复=%.3f | 必须意象=%.3f | 艺术品质=%.3f | 原始=%.3f",
                      sc['total'], sc['intent'], sc['intent_llm'],
                      sc['pingze'], sc['rhyme'], sc['imagery'], sc['cohesion'],
                      sc['penalty'], sc.get('required_coeff', 1.0),
                      sc.get('art_quality', sc['raw_total']), sc['raw_total'])

    @staticmethod
    def _log_round_summary(round_num: int, qualified: list, rejected: list,
                           total_qualified: int, min_qualified: int) -> None:
        """打印单轮品质筛选摘要。"""
        _log.info("-" * 70)
        # 打印合格
        for sc, pm in qualified:
            poem_one_line = " | ".join(l.strip() for l in pm.split('\n') if l.strip())[:80]
            _log.info("【合格 ✓ 总分=%.3f】%s", sc['total'], poem_one_line)
        # 打印废弃
        for sc, pm in rejected:
            poem_one_line = " | ".join(l.strip() for l in pm.split('\n') if l.strip())[:80]
            _log.info("【废弃 ✗ 总分=%.3f】%s", sc['total'], poem_one_line)
        _log.info("本轮: %d 首合格, %d 首废弃 | 累计合格池: %d 首（需 ≥ %d）",
                  len(qualified), len(rejected), total_qualified, min_qualified)

    @staticmethod
    def _log_best_poem(best_scores: dict, best_poem: str) -> None:
        """打印最优候选详情。"""
        _log.info("#" * 70)
        _log.info("最优候选 总分: %.3f", best_scores['total'])
        _log.info("  意图=%.3f [LLM=%.3f] | 平仄=%.3f | 押韵=%.3f | 意象=%.3f | 聚合=%.3f",
                  best_scores['intent'], best_scores['intent_llm'],
                  best_scores['pingze'], best_scores['rhyme'],
                  best_scores['imagery'], best_scores['cohesion'])
        _log.info("  重复惩罚=%.3f | 必须意象系数=%.3f | 艺术品质=%.3f | 原始=%.3f",
                  best_scores['penalty'], best_scores.get('required_coeff', 1.0),
                  best_scores.get('art_quality', best_scores['raw_total']),
                  best_scores['raw_total'])
        for line in best_poem.split('\n'):
            _log.info("    %s", line)
        _log.info("#" * 70)

    def generate_and_score(self, user_request: str, score_adapter,
                           generation_adapter=None, creative_brief: str = ""):
        """生成 5 首 → 硬门控 → 本地评分（含切题）。不做 arena 选冠军。

        返回: {genre_name, num_lines, chars_per_line, gated: [{idx, poem, local}],
                rejected, total_generated}
        """
        genre_name, num_lines, chars_per_line = self.scorer.detect_genre(user_request)
        _backend = getattr(generation_adapter, 'backend', 'local') if generation_adapter else 'local'
        use_api, model, tokenizer = self._resolve_model(generation_adapter, _backend)

        _log.info("使用 %s 模型", "API" if use_api else "本地微调模型 (LoRA)")
        candidates = self._generate_candidates(
            user_request, generation_adapter, model, tokenizer,
            num_lines, chars_per_line, use_api,
            creative_brief=creative_brief, count=5, start_index=0,
        )
        if not use_api:
            from core.models.manager import ModelManager
            mm = ModelManager()
            mm._release_fine()
            _log.info("[显存] 候选生成完毕，释放微调模型。")

        if not candidates:
            return {"genre_name": genre_name, "gated": [], "rejected": [],
                    "total_generated": 0, "num_lines": num_lines,
                    "chars_per_line": chars_per_line}

        result = self.scorer.hard_gate_and_score(
            candidates, user_request, score_adapter,
            num_lines, chars_per_line,
        )
        result["genre_name"] = genre_name
        result["num_lines"] = num_lines
        result["chars_per_line"] = chars_per_line
        result["total_generated"] = len(candidates)
        return result

    def generate_arena(self, user_request: str, score_adapter,
                       generation_adapter=None, creative_brief: str = ""):
        """全自主模式：生成 5 首 → 硬门控 → 本地分 Top3 → arena pairwise → 冠军。

        返回 dict: {genre_name, champion, backup, total_generated, arena_result}
        """
        genre_name, num_lines, chars_per_line = self.scorer.detect_genre(user_request)
        _backend = getattr(generation_adapter, 'backend', 'local') if generation_adapter else 'local'
        use_api, model, tokenizer = self._resolve_model(generation_adapter, _backend)

        _log.info("使用 %s 模型", "API" if use_api else "本地微调模型 (LoRA)")
        candidates = self._generate_candidates(
            user_request, generation_adapter, model, tokenizer,
            num_lines, chars_per_line, use_api,
            creative_brief=creative_brief, count=5, start_index=0,
        )
        if not use_api:
            from core.models.manager import ModelManager
            mm = ModelManager()
            mm._release_fine()
            _log.info("[显存] 候选生成完毕，释放微调模型。")

        if not candidates or len(candidates) < 2:
            return {
                "genre_name": genre_name,
                "champion": candidates[0] if candidates else "（生成失败）",
                "backup": candidates[1] if len(candidates) > 1 else "",
                "total_generated": len(candidates),
            }

        # 硬门控 → 本地分 Top3 → arena pairwise → 冠军
        result = self.scorer.arena_select_champion(
            candidates, user_request, score_adapter,
            num_lines, chars_per_line,
        )
        return {
            "genre_name": genre_name,
            "champion": result["champion"],
            "backup": result["backup"],
            "champion_topic": result.get("champion_topic", 0.5),
            "champion_local_total": result.get("champion_local_total", 0.0),
            "champion_final": result.get("champion_final", 0.0),
            "gated_count": result.get("gated_count", 0),
            "total_generated": len(candidates),
            "arena_result": result,
        }

    def score_single_poem(self, poem: str, user_request: str, score_adapter) -> dict:
        return self.scorer.score_single(poem, user_request, score_adapter)

    # ── 辅助方法 ───────────────────────────────────────────────────────────
    @staticmethod
    def _build_api_prompt(user_request: str, chars_per_line: int, num_lines: int,
                          brief: str = "") -> str:
        brief_line = f"\n创作方向（参考）：{brief[:60]}" if brief and brief.strip() else ""
        return (
            f"请根据以下要求创作一首中国古典诗词，"
            f"严格按照格律，每句{chars_per_line}字，共{num_lines}句。"
            f"不要输出任何解释，只输出诗句，每句换行。"
            f"{brief_line}\n\n要求：{user_request}\n"
        )

    def _call_model(self, model, tokenizer, prompt, max_tokens, temperature):
        messages = [
            {"role": "system", "content": "你是一位精通中国古典诗词的创作专家。"},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        if hasattr(model, "generation_config") and \
                getattr(model.generation_config, "max_length", None) is not None:
            model.generation_config.max_length = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning,
                                    message=".*attention mask API.*")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=temperature, top_p=0.9, do_sample=True,
                )
        return tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        ).strip()

    @staticmethod
    def _normalize(raw: str, target_len: int) -> str:
        segs = re.split(r'[，。！？；\n]+', raw)
        lines = []
        for seg in segs:
            seg = seg.strip()
            pure = ''.join(ch for ch in seg if '一' <= ch <= '鿿')
            if len(pure) == target_len:
                lines.append(pure)
        unique = []
        last = None
        for line in lines:
            if line != last:
                unique.append(line)
                last = line
        return '\n'.join(unique[:8])
