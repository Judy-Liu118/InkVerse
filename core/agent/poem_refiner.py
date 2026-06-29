"""诗歌修改与守擂进化 Mixin。"""
from __future__ import annotations

from typing import Any, Dict, List

from core.agent.state import AgentState
from core.logger import get_logger

_log = get_logger(__name__)


class _PoemRefineMixin:
    """诗歌修改和守擂进化方法集。作为 Mixin 挂载在 PoetryAgent 上使用。"""

    _CHALLENGER_PROMPT = (
        "你是一位精通中国古典诗词的创作专家。\n"
        "当前冠军诗作：\n{champion}\n\n"
        "创作方向：\n{feedback}\n\n"
        "请根据上述方向，创作一首全新的诗作为挑战者。\n"
        "格律铁则：每句必须恰好 {chars_per_line} 个汉字，共 {num_lines} 句，"
        "每句换行，不加任何标点或解释。\n"
        "不要重复冠军诗的措辞，尝试用不同的意象、不同的视角表达相近的意境。\n"
        "保持押韵和平仄规范。\n"
        "输出修改后的完整诗句（每行一句，{chars_per_line} 个汉字，纯汉字）："
    )

    def refine_poem(
        self, state: AgentState, feedback: str,
        refine_adapter=None, score_tolerance: float = 0.03,
    ) -> AgentState:
        adapter = refine_adapter or self.generation_adapter
        model_desc = self._adapter_desc(adapter)

        if getattr(adapter, 'backend', '') in ('local', 'local_lora'):
            state.log("诗歌修改", "⚠ 已跳过",
                      "LoRA 模型不具备改诗能力，请在「改诗模型」下拉框中选择 API 模型再试。",
                      model=model_desc)
            return state

        old_poem = state.poem
        orig_score = state.best_poem_score
        orig_lines = [l for l in old_poem.split("\n") if l.strip()]
        expected_lines = len(orig_lines)
        expected_chars = len(orig_lines[0]) if orig_lines else 5

        from core.prompts import render_messages
        msg = render_messages(
            "agent.refine_poem",
            expected_chars=expected_chars,
            expected_lines=expected_lines,
            old_poem=old_poem,
            feedback=feedback,
        )
        try:
            raw = adapter.generate(msg, max_tokens=120, temperature=0.75)
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            clean = ["".join(ch for ch in l if "一" <= ch <= "鿿") for l in lines]
            char_ok = [l for l in clean if len(l) == expected_chars]
            if len(char_ok) < expected_lines:
                state.log("诗歌修改", "⚠ 字数不符（已回滚）",
                          f"改后各行字数: {[len(l) for l in clean if l]}，"
                          f"应为 {expected_chars} 字×{expected_lines} 行。",
                          model=model_desc)
                return state
            candidate = "\n".join(char_ok[:expected_lines])
            new_score_dict = self.poem_gen.score_single_poem(candidate, state.user_input, self.score_adapter)
            if isinstance(new_score_dict, dict):
                new_score = new_score_dict.get("total", 0.0)
                detail = (
                    f"意图={new_score_dict.get('intent',0):.3f}"
                    f"[LLM={new_score_dict.get('intent_llm',0):.3f}] | "
                    f"平仄={new_score_dict.get('pingze',0):.3f} | "
                    f"押韵={new_score_dict.get('rhyme',0):.3f} | "
                    f"意象={new_score_dict.get('imagery',0):.3f} | "
                    f"聚合={new_score_dict.get('cohesion',0):.3f} | "
                    f"重复={new_score_dict.get('penalty',1):.3f} | "
                    f"必须意象={new_score_dict.get('required_coeff',1):.3f} | "
                    f"原始={new_score_dict.get('raw_total',0):.3f}"
                )
                _log.info("改后分数明细: %s", detail)
            else:
                new_score = float(new_score_dict)
                detail = f"总分={new_score:.3f}"

            threshold = max(0.0, orig_score - score_tolerance)
            if orig_score > 0 and new_score < threshold:
                state.log("诗歌修改", "⚠ 改后得分不足（已回滚）",
                          f"改后得分 {new_score:.3f} < 原始得分 {orig_score:.3f} - "
                          f"容差 {score_tolerance} = {threshold:.3f}\n"
                          f"改后各维度：{detail}",
                          model=model_desc)
                return state

            state.poem = candidate
            state.best_poem_score = new_score
            state.model_usage.poem_gen += f" → 修改({model_desc})"

            old_one_line = " | ".join(l.strip() for l in old_poem.split('\n') if l.strip())
            new_one_line = " | ".join(l.strip() for l in candidate.split('\n') if l.strip())
            _log.info("=" * 60)
            _log.info("【改诗对比】")
            _log.info("-" * 60)
            _log.info("[改前 总分=%.3f] %s", orig_score, old_one_line)
            _log.info("[改后 总分=%.3f] %s", new_score, new_one_line)
            _log.info("  各维度: %s", detail)
            _log.info("=" * 60)

            if new_score > orig_score:
                verdict = f"✓ 提升 | 得分 {orig_score:.3f}→{new_score:.3f}"
            elif new_score == orig_score:
                verdict = f"⚠ 持平 | 得分 {orig_score:.3f}→{new_score:.3f}"
            else:
                verdict = f"⚠ 略降（容差内）| 得分 {orig_score:.3f}→{new_score:.3f}"
            state.log("诗歌修改", verdict,
                      f"修改方向：{feedback[:80]}\n改后各维度：{detail}",
                      model=model_desc, score=new_score)
        except Exception as e:
            state.log("诗歌修改", "修改异常", str(e))
        return state

    def refine_multiple_poems(
        self, state: AgentState, candidates: List[Dict],
        refine_adapter=None, score_tolerance: float = 0.03,
    ) -> List[Dict]:
        """对多首候选诗逐一修改，返回所有改后结果 [{poem, scores, original_poem, refined}]。"""
        adapter = refine_adapter or self.generation_adapter

        if getattr(adapter, 'backend', '') in ('local', 'local_lora'):
            _log.warning("批量改诗跳过：LoRA 模型不具备改诗能力")
            return []

        results = []
        for i, cand in enumerate(candidates):
            poem_text = cand["poem"]
            orig_total = cand.get("scores", {}).get("total", state.best_poem_score)

            saved_poem  = state.poem
            saved_score = state.best_poem_score
            state.poem  = poem_text
            state.best_poem_score = orig_total

            _log.info("批量改诗 [%d/%d] 原分=%.3f，开始修改...",
                     i + 1, len(candidates), orig_total)
            try:
                critique = self._auto_poem_critique(state)
                _log.info("批量改诗 [%d/%d] 点评: %s", i + 1, len(candidates), critique[:120])
                auto_fb = self._auto_poem_feedback(state, critique=critique)
                _log.info("批量改诗 [%d/%d] 方向: %s", i + 1, len(candidates), auto_fb)

                state = self.refine_poem(state, auto_fb, refine_adapter=adapter,
                                         score_tolerance=score_tolerance)
                refined_poem = state.poem

                if refined_poem != poem_text:
                    new_score_dict = self.poem_gen.score_single_poem(
                        refined_poem, state.user_input, self.score_adapter)
                    new_score = new_score_dict.get("total", 0.0) if isinstance(new_score_dict, dict) else float(new_score_dict)
                    results.append({
                        "poem": refined_poem,
                        "scores": new_score_dict if isinstance(new_score_dict, dict) else {"total": new_score},
                        "original_poem": poem_text,
                        "refined": True,
                    })
                    _log.info("批量改诗 [%d/%d] ✓ 成功 %.3f→%.3f",
                             i + 1, len(candidates), orig_total, new_score)
                else:
                    _log.info("批量改诗 [%d/%d] - 未变化，保留原诗", i + 1, len(candidates))
            except Exception as e:
                _log.exception("批量改诗 [%d/%d] ✗ 异常", i + 1, len(candidates))

            state.poem            = saved_poem
            state.best_poem_score = saved_score

        return results

    def _local_score_champion(self, poem: str, num_lines: int,
                              chars_per_line: int,
                              state: AgentState = None) -> dict:
        """本地评分。进化阶段沿用 arena 的切题分，不虚高。"""
        topic = getattr(state, 'champion_topic', 1.0) if state else 1.0
        return self.poem_gen.scorer.local_score_poem(poem, num_lines,
                                                      chars_per_line,
                                                      topic_score=topic)

    def _pairwise_delta(self, won: bool) -> float:
        """pairwise 审美 delta：赢 +0.17，输 −0.05（见 config PAIRWISE_*_DELTA）。"""
        from config import PAIRWISE_WIN_DELTA, PAIRWISE_LOSE_DELTA
        return PAIRWISE_WIN_DELTA if won else PAIRWISE_LOSE_DELTA

    def _format_score_log(self, local: dict, pairwise_delta: float) -> str:
        topic_part = f"切题={local.get('topic', 1.0):.2f} " if local.get('topic') is not None else ""
        return (f"本地={local['total']:.3f}("
                f"平仄={local['pingze']:.2f} 押韵={local['rhyme']:.2f} "
                f"意象={local['imagery']:.2f} 连贯={local['cohesion']:.2f} "
                f"{topic_part})"
                f" pairwise={'✓+' if pairwise_delta > 0 else '✗'}"
                f"{pairwise_delta:+.2f}"
                f" 综合={local['total'] + pairwise_delta:.3f}")

    def hard_gate_check(self, poem: str, num_lines: int,
                        chars_per_line: int) -> dict:
        """硬门控：本地规则检查，不消耗 LLM 调用。"""
        from config import BAD_PATTERNS
        reasons = []
        lines = [l.strip() for l in poem.split('\n') if l.strip()]
        if len(lines) < num_lines:
            reasons.append(f"行数不足（需{num_lines}行）")
        char_ok = [l for l in lines[:num_lines] if len(l) == chars_per_line]
        if len(char_ok) < num_lines:
            bad_info = [f"第{i+1}行{len(l)}字" for i, l in enumerate(lines[:num_lines])
                        if len(l) != chars_per_line]
            reasons.append(f"字数不符: {', '.join(bad_info[:3])}")
        if reasons:
            return {"passed": False, "reasons": reasons, "rhyme": 0.0, "pingze": 0.0}

        rhyme_score = self.poem_gen.scorer._score_rhyme(poem, num_lines)
        pingze_score = self.poem_gen.scorer._score_pingze(poem, num_lines, chars_per_line)
        if rhyme_score < 0.6:
            reasons.append(f"押韵不合格（{rhyme_score:.2f} < 0.60）")
        if pingze_score < 0.6:
            reasons.append(f"平仄不合格（{pingze_score:.2f} < 0.60）")
        poem_text = ''.join(lines)
        hits = [w for w in BAD_PATTERNS if w in poem_text]
        if hits:
            reasons.append(f"AI堆砌词汇: {', '.join(hits)}")

        return {"passed": len(reasons) == 0, "reasons": reasons,
                "rhyme": rhyme_score, "pingze": pingze_score}

    def _generate_challenger(self, champion: str, feedback: str,
                             num_lines: int, chars_per_line: int,
                             adapter) -> str | None:
        """生成一首挑战者诗，失败返回 None。"""
        prompt = self._CHALLENGER_PROMPT.format(
            champion=champion, feedback=feedback,
            chars_per_line=chars_per_line, num_lines=num_lines,
        )
        messages = [
            {"role": "system",
             "content": "你是一位精通中国古典诗词的创作专家。只输出诗句，不含任何解释。"},
            {"role": "user", "content": prompt},
        ]
        from config import POEM_TEMPERATURE
        raw = adapter.generate(messages, max_tokens=120,
                               temperature=POEM_TEMPERATURE + 0.05)
        clines = [l.strip() for l in raw.split('\n') if l.strip()]
        challenger_lines = [
            "".join(ch for ch in l if '一' <= ch <= '鿿')
            for l in clines
        ]
        challenger_lines = [l for l in challenger_lines
                            if len(l) == chars_per_line][:num_lines]
        if len(challenger_lines) < num_lines:
            return None
        return '\n'.join(challenger_lines)

    def _try_challenger(self, state: AgentState, champion: str,
                        feedback: str, direction_label: str,
                        num_lines: int, chars_per_line: int,
                        adapter, round_num: int) -> dict | None:
        """试一个挑战者：生成 → 门控 → 本地评分 → pairwise → 返回结果或 None。"""
        challenger = self._generate_challenger(
            champion, feedback, num_lines, chars_per_line, adapter,
        )
        if challenger is None:
            _log.info("  挑战者%s 格式不符，跳过", direction_label)
            return None

        gate = self.hard_gate_check(challenger, num_lines, chars_per_line)
        if not gate["passed"]:
            _log.info("  挑战者%s 硬门控拦截: %s",
                      direction_label, "; ".join(gate["reasons"]))
            state.log("擂台", f"第{round_num}轮{direction_label}·门控拦截",
                      "; ".join(gate["reasons"]))
            return None

        chal_local = self._local_score_champion(challenger, num_lines, chars_per_line, state=state)
        winner = self.poem_gen.scorer.compare_poems(
            champion, challenger, state.user_input, self.score_adapter,
        )
        chal_won = (winner == "B")
        delta = self._pairwise_delta(chal_won)
        combined = chal_local["total"] + delta
        return {"poem": challenger, "local": chal_local,
                "pairwise_won": chal_won, "delta": delta, "combined": combined,
                "direction": direction_label}

    def _evolve_champion(self, state: AgentState, refine_adapter=None,
                         evolution_rounds: int = 3):
        """单轨守擂进化：每轮 2 个不同方向挑战者，硬门控 + 本地评分 + pairwise。

        是一个生成器，每轮 yield state 供 UI 刷新。
        """
        from config import CHALLENGERS_PER_ROUND
        adapter = refine_adapter or self.score_adapter
        champion = state.poem
        lines = [l for l in champion.split('\n') if l.strip()]
        num_lines = len(lines)
        chars_per_line = len(lines[0]) if lines else 5

        champ_local = self._local_score_champion(champion, num_lines, chars_per_line, state=state)
        _log.info("擂台·当前擂主 %s", self._format_score_log(champ_local, 0.0))

        topic_score = getattr(state, 'champion_topic', 1.0)
        topic_hint = ""
        if topic_score < 0.7:
            topic_hint = (f"\n\n【重要】当前诗作与用户要求的主题契合度偏低（切题分仅{topic_score:.1f}），"
                          "请优先考虑增强主题相关性，使诗中意象、场景、情感紧紧围绕用户要求展开。")

        for evo_round in range(evolution_rounds):
            critique = self._auto_poem_critique(state)
            if topic_hint:
                critique = topic_hint + "\n" + critique
            directions = []
            for ci in range(CHALLENGERS_PER_ROUND):
                fb = self._auto_poem_feedback(state, critique=(
                    critique if ci == 0 else
                    critique + "\n请从另一个完全不同的维度给出修改建议，"
                    "不要重复之前的建议方向。"
                ))
                if fb not in [d[0] for d in directions]:
                    directions.append((fb, f"方向{ci+1}"))
            _log.info("擂台第%d轮·%d个方向:", evo_round + 1, len(directions))
            for fb, label in directions:
                _log.info("  %s: %s", label, fb)

            best_result = None
            for fb, label in directions:
                result = self._try_challenger(
                    state, champion, fb, label,
                    num_lines, chars_per_line, adapter, evo_round + 1,
                )
                if result:
                    _log.info("  %s 综合=%.3f (本地=%.3f pairwise=%s)",
                              label, result["combined"], result["local"]["total"],
                              "胜" if result["pairwise_won"] else "败")
                    if best_result is None or result["combined"] > best_result["combined"]:
                        best_result = result

            if best_result is None:
                state.log("擂台", f"第{evo_round+1}轮·无有效挑战者",
                          "所有挑战者均被门控拦截或格式不符")
                yield state
                continue

            champ_combined = champ_local["total"]
            chal = best_result

            _log.info("-" * 60)
            _log.info("【擂台第%d轮·%s】", evo_round + 1, chal["direction"])
            _log.info("  擂主: %s", self._format_score_log(champ_local, 0.0))
            _log.info("  挑战: %s",
                      self._format_score_log(chal["local"], chal["delta"]))

            if chal["combined"] > champ_combined:
                _log.info("  → 攻擂成功 ✓ 综合 %.3f > %.3f",
                          chal["combined"], champ_combined)
                _log.info("  新擂主:")
                for line in chal["poem"].strip().split('\n'):
                    _log.info("    %s", line.strip())
                champion = chal["poem"]
                champ_local = chal["local"]
                state.poem = champion
                state.log("擂台", f"第{evo_round+1}轮·攻擂成功 ✓",
                          f"{chal['direction']} 综合{chal['combined']:.3f} > "
                          f"擂主{champ_combined:.3f}"
                          f"{' | pairwise审美胜出' if chal['pairwise_won'] else ''}")
            else:
                _log.info("  → 守擂成功 擂主 %.3f ≥ 挑战 %.3f",
                          champ_combined, chal["combined"])
                _log.info("  挑战者（被拒）:")
                for line in chal["poem"].strip().split('\n'):
                    _log.info("    %s", line.strip())
                state.log("擂台", f"第{evo_round+1}轮·守擂成功",
                          f"{chal['direction']} 综合{chal['combined']:.3f} ≤ "
                          f"擂主{champ_combined:.3f}"
                          f"{' | pairwise审美胜出但本地拖累' if chal['pairwise_won'] else ''}")
            _log.info("-" * 60)

            yield state

        state.poem = champion

    def _auto_poem_critique(self, state: AgentState) -> str:
        adapter = self.score_adapter or self.generation_adapter
        raw = self._raw_clip(state)
        lines = [l.strip() for l in state.poem.split("\n") if l.strip()]
        chars_per = len(lines[0]) if lines else 7
        msg = [
            {
                "role": "system",
                "content": (
                    "你是一位精通古典诗词格律的文学评论家。"
                    "请从意境深度、画面美感、情感力度、语言锤炼、押韵合规、平仄规范六个维度，"
                    "对这首诗写一段简短的点评（150字以内）：\n"
                    "  • 先用1-2句肯定其可取之处\n"
                    "  • 再指出1-2处最值得打磨的不足（如有押韵或平仄硬伤，优先指出）\n"
                    "注意：不要直接给出修改方案，只需分析不足在哪里；"
                    "不要凭空建议新增诗中没有的人物、器物、动物或情节。"
                    "语言简练，直接输出点评，无需标题或前缀。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"诗歌（{chars_per}言，共{len(lines)}句）：\n{state.poem}\n\n"
                    f"当前图文一致性得分 {raw:.3f}（>0.28 为优秀），"
                    f"说明部分意象在画面转化时仍有空间。\n请点评："
                ),
            },
        ]
        try:
            critique = adapter.generate(msg, max_tokens=200, temperature=0.5).strip()
            return critique[:300] if len(critique) > 300 else critique or "诗意尚佳，但部分意象仍可更具体鲜明。"
        except Exception as e:
            _log.warning("自主改诗点评生成失败: %s", e)
            return "整体意境可取，但仍有句子意象不够鲜明，建议深化视觉细节。"

    def _auto_poem_feedback(self, state: AgentState, critique: str = "") -> str:
        adapter = self.score_adapter or self.generation_adapter
        lines = [l.strip() for l in state.poem.split("\n") if l.strip()]
        chars_per = len(lines[0]) if lines else 7
        msg = [
            {
                "role": "system",
                "content": (
                    "你是古典诗词改诗规划专家。"
                    "根据以下诗评，提炼出一条修改方向（60字以内），"
                    "告诉改诗模型应该在哪个维度、哪一句、如何提升。\n"
                    "要求：\n"
                    "  ① 方向要具体（指出哪句/哪联有问题），但不要给出完整替换句\n"
                    f"  ② 改后每行必须仍是 {chars_per} 个汉字，不得增减\n"
                    "  ③ 不要引入原诗和用户要求之外的具体人物、器物、动物或情节\n"
                    "  ④ 不要说'将X改为Y'这种直接替换格式，而是说'某句可以……，使意境……'\n"
                    "  ⑤ 如果原诗押韵合规，强调'必须保留原韵脚，不得改动偶句末字'\n"
                    "  ⑥ 如果修改涉及任何偶句末字，必须注明'新韵脚需与全诗协调'\n"
                    "直接输出修改方向，不要前缀或解释。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"当前诗歌：\n{state.poem}\n\n"
                    f"诗评：\n{critique}\n\n"
                    "修改方向："
                ),
            },
        ]
        try:
            fb = adapter.generate(msg, max_tokens=80, temperature=0.4).strip()
            fb = fb.splitlines()[0].strip()
            return fb[:80] or "深化诗中视觉意象，使画面感更鲜明，意境更有深度"
        except Exception as e:
            _log.warning("自主改诗修改方向生成失败: %s", e)
            return "深化诗中视觉意象，使画面感更鲜明，意境更有深度"
