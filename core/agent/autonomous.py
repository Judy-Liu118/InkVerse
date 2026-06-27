"""
core.agent.autonomous -- 全自主创作模式

诗是诗，图是图：
  · Arena 海选：生成 5 首 → 硬门控过滤 → 本地分排序 Top3 → 轮循 pairwise → 冠军
  · 单轨守擂进化：硬门控（押韵/平仄/堆砌词）拦截 + 1v1 pairwise 对决，
    挑战者胜出则攻擂成功，否则擂主守擂成功
  · CLIP 门控（改图）：CLIP < target → 纯改图循环，不改诗
  · 自适应停止：连续 2 轮 CLIP 无显著提升（delta < 0.01）→ 提前退出
"""
from __future__ import annotations

from dataclasses import dataclass
from core.agent.state import AgentState, Phase
from core.logger import get_logger
from config import PAIRWISE_EVOLUTION_ROUNDS

_log = get_logger(__name__)


@dataclass
class AutonomousConfig:
    """
    全自主模式配置。

    CLIP 阈值参考：
      基于 CLIPScore 论文（Hessel et al., EMNLP 2021），原始余弦相似度在
      文本-图像生成任务中自然聚类在 0.15~0.35 的窄区间内，0.25 为"合理"水平，
      0.30 为"明显好"的水平。考虑到中国水墨画（低饱和度）和中文诗歌锚点（CLIP
      用英文训练）的额外难度，推荐阈值如下：
        · 基础生成阈值（CLIP_THRESHOLD）：0.22 — 低于此建议重试
        · 自主模式目标（target_clip_score）：0.30 — 有挑战但可达
        · 最高天花板（ViT-B/32 + 中文诗歌 + 水墨）：≈0.32
      升级到 ViT-L/14 可平均提分 0.02~0.04。
    """
    target_clip_score:        float = 0.30   # CLIP 门控目标
    max_image_improve_rounds: int   = 2      # 改图循环硬上限
    allow_poem_refine:        bool  = True   # 是否启用全自主改诗
    max_poem_refine_rounds:   int   = 1      # 改诗轮次（每轮改 top-N 首）
    refine_top_n:             int   = 2      # 每次改几首合格诗（1~5）
    image_improve_mode:       str   = "rewrite_regen"  # "rewrite_regen" | "edit_api"
    edit_model:               str   = "wanx2.1-imageedit"
    # 自适应停止参数
    adaptive_stop:            bool  = True   # 是否启用自适应停止
    adaptive_stop_delta:      float = 0.01   # 连续无提升的 delta 阈值
    # LLM-driven 改图循环：把 state + tool schema 喂 LLM，由 LLM 决定调用
    # edit_image / refine_poem_and_regen / stop（默认关，向后兼容写死流程）
    image_loop_llm_driven:    bool  = False


def autonomous_full_run(agent, state: AgentState, config: AutonomousConfig = None):
    """
    全自主创作模式（生成器）。

    每轮图像完成后 yield state，供流式 UI 刷新。
    """
    if config is None:
        config = AutonomousConfig()

    target = config.target_clip_score
    from config import CLIP_THRESHOLD as _BASE_THRESH
    state.log(
        "自主模式", "启动",
        (
            f"自主目标 CLIP raw≥{target}（仅影响改图循环）| "
            f"基础生成阈值={_BASE_THRESH}（超过即停止重新生图）| "
            f"改图上限={config.max_image_improve_rounds}轮 | "
            f"自适应停止={'启用' if config.adaptive_stop else '禁用'}"
            f"{'（delta<' + str(config.adaptive_stop_delta) + '）' if config.adaptive_stop else ''} | "
            f"擂台进化={'启用，' + str(config.max_poem_refine_rounds) + '轮守擂挑战' if config.allow_poem_refine else '禁用'}"
        ),
    )

    _refine_adapter = agent.score_adapter or agent.prompt_adapter
    _can_refine = (
        _refine_adapter is not None
        and getattr(_refine_adapter, "backend", "") not in ("local", "local_lora")
    )

    # ═════════════════════════════════════════════════════════════════════
    # 第 1 步：规划 + 诗歌生成（pairwise 五选二）
    # ═════════════════════════════════════════════════════════════════════
    state = agent._phase_plan(state)
    if state.phase == Phase.ERROR:
        state.log("自主模式", "规划失败，终止", state.error)
        yield state
        return

    state = agent._phase_poem_arena(state)
    if state.phase == Phase.ERROR:
        state.log("自主模式", "诗歌生成失败，终止", state.error)
        yield state
        return

    state.log("自主模式", "Arena 海选完成",
              f"冠军候选已选定，备份候选: {state.backup_poem[:40] if state.backup_poem else '无'}...")
    yield state

    # ═════════════════════════════════════════════════════════════════════
    # 第 2 步：单轨守擂进化（硬门控 + 1v1 pairwise 对决）
    # ═════════════════════════════════════════════════════════════════════
    if config.allow_poem_refine and _can_refine:
        evo_rounds = getattr(config, 'max_poem_refine_rounds', PAIRWISE_EVOLUTION_ROUNDS)
        state.log("自主模式", "擂台进化开始",
                  f"当前擂主就位，共 {evo_rounds} 轮守擂挑战")
        for _ in agent._evolve_champion(
            state, refine_adapter=_refine_adapter, evolution_rounds=evo_rounds,
        ):
            yield state  # 每轮对决后刷新 UI
    elif config.allow_poem_refine and not _can_refine:
        state.log("自主模式", "⚠ 擂台进化跳过",
                  "未配置 API 改诗模型（LoRA/本地模型不支持改诗），跳过。")

    # ═════════════════════════════════════════════════════════════════════
    # 第 3 步：关键词提取 → 诗名 → 提示词（先改完诗再配图）
    # ═════════════════════════════════════════════════════════════════════
    state = agent._phase_keyword_extract(state)
    state = agent._phase_title(state)
    state = agent._phase_prompt(state)
    if state.phase == Phase.ERROR:
        state.log("自主模式", "提示词生成失败，终止", state.error)
        yield state
        return
    state = agent._phase_prompt_review(state)
    if state.phase == Phase.ERROR:
        state.log("自主模式", "提示词自检失败，终止", state.error)
        yield state
        return

    # ═════════════════════════════════════════════════════════════════════
    # 第 4 步：生图 + CLIP 双锚点评分 + 反思
    # ═════════════════════════════════════════════════════════════════════
    state = agent._phase_image_clip(state)
    if state.image is None:
        state.log("自主模式", "⚠ 提前终止",
                  "未能生成图像，请检查 API Key 配置后重试。")
        state.phase = Phase.DONE
        yield state
        return
    state = agent._phase_reflect(state)

    best_score = agent._raw_clip(state)
    best_state = agent._copy_state(state)
    state.log("自主模式", "生图完成",
              f"CLIP raw={best_score:.3f}（目标≥{target}），开始改图循环（如需要）")
    yield state

    # ═════════════════════════════════════════════════════════════════════
    # CLIP 门控：改图循环（写死流程 vs LLM-driven 控制器）
    # ═════════════════════════════════════════════════════════════════════
    stale_count = 0       # 连续无提升计数器（自适应停止）
    prev_round_score = None  # 上一轮分数，None 表示首轮（不计入 stale）

    if config.image_loop_llm_driven:
        # LLM-driven：每轮把 state + tool schema 喂 LLM，由它决定调
        # edit_image / refine_poem_and_regen / stop（real ToolRegistry dispatch）
        from core.agent.controller import ImageLoopController, build_loop_registry
        _controller_adapter = _refine_adapter or agent.prompt_adapter or agent.score_adapter
        if _controller_adapter is None:
            state.log("自主模式", "⚠ LLM-driven 循环跳过",
                      "未配置 LLM 评分/规划 adapter，无法运行 controller，回退到原写死流程")
            config.image_loop_llm_driven = False  # 本次运行降级
        else:
            loop_registry = build_loop_registry(agent)
            controller = ImageLoopController(
                adapter=_controller_adapter, registry=loop_registry,
            )
            state.log("自主模式", "LLM-driven 改图循环启动",
                      f"工具={sorted(controller.allowed_tools)} | "
                      f"目标 raw={target:.3f} | 预算={config.max_image_improve_rounds}")
            history: list = []
            for img_round in range(config.max_image_improve_rounds):
                if best_score >= target:
                    state.log("自主模式", "LLM 循环·提前达标",
                              f"CLIP raw={best_score:.3f} ≥ {target}")
                    break
                if best_state is not state:
                    state.image = best_state.image
                    state.clip_score_final = best_state.clip_score_final

                decision = controller.decide(
                    state=state, best_score=best_score, target=target,
                    round_used=img_round, max_rounds=config.max_image_improve_rounds,
                    stale_count=stale_count, prev_score=prev_round_score,
                    history=history,
                )
                tool_name = decision.get("tool")
                state.log(
                    "自主模式",
                    f"LLM 决策 ({img_round + 1}/{config.max_image_improve_rounds})",
                    f"tool={tool_name} reasoning={(decision.get('reasoning') or '')[:80]}",
                )
                if decision.get("_fallback"):
                    state.log("自主模式", "⚠ controller fallback",
                              decision.get("reasoning", ""))

                # 诚实性指标埋点：每轮决策结构化落盘
                decision_record = {
                    "round": img_round + 1,
                    "tool": tool_name,
                    "is_fallback": bool(decision.get("_fallback")),
                    "score_before": prev_round_score,
                    "score_after": None,
                    "stale_override": False,
                }
                state.llm_loop_decisions.append(decision_record)

                state, should_stop = controller.dispatch(decision, state)
                history.append(
                    f"{tool_name}: {decision.get('feedback') or decision.get('reason') or ''}"
                )
                if should_stop:
                    state.log("自主模式", "LLM 决定终止改图",
                              decision.get("reason", ""))
                    yield state
                    break
                if state.image is None:
                    state.log("自主模式", "改图：图像生成失败，终止", "")
                    break

                round_score = agent._raw_clip(state)
                decision_record["score_after"] = round_score
                state.log("自主模式",
                          f"LLM 改图：第 {img_round + 1} 轮完成",
                          f"CLIP raw={round_score:.3f}")
                if round_score > best_score:
                    best_score = round_score
                    best_state = agent._copy_state(state)
                if prev_round_score is not None:
                    if round_score - prev_round_score <= config.adaptive_stop_delta:
                        stale_count += 1
                    else:
                        stale_count = 0
                prev_round_score = round_score
                yield state

                # 启发式护栏：即使 LLM 不喊 stop，也尊重 stale 阈值
                if config.adaptive_stop and stale_count >= 2:
                    decision_record["stale_override"] = True
                    state.log("自主模式", "自适应停止（覆盖 LLM 决策）",
                              f"连续 {stale_count} 轮 CLIP 无显著提升，强制退出")
                    break

    if not config.image_loop_llm_driven:
        # 原写死流程（向后兼容默认行为）
        for img_round in range(config.max_image_improve_rounds):
            if best_score >= target:
                state.log("自主模式", "改图循环·提前达标",
                          f"CLIP raw={best_score:.3f} ≥ {target}，跳过剩余改图轮次")
                break

            # 编辑强度衰减：越往后越微调（0.75 → 0.50），防止灾难性遗忘前期特征
            decay_strength = max(0.50, 0.75 - img_round * 0.12)
            # 每轮从最优图出发改，而不是从上一轮可能改砸的图出发
            if best_state is not state:
                state.image = best_state.image
                state.clip_score_final = best_state.clip_score_final
            state.log("自主模式", f"改图循环：第 {img_round + 1} 轮",
                      f"当前 raw={best_score:.3f}，调用 autonomous_improve_image…（强度={decay_strength:.2f}）")
            state = agent.autonomous_improve_image(
                state,
                image_mode=config.image_improve_mode,
                edit_model=getattr(config, "edit_model", "wanx2.1-imageedit"),
                edit_strength=decay_strength,
            )
            round_score = agent._raw_clip(state)
            state.log("自主模式", f"改图循环：第 {img_round + 1} 轮完成",
                      f"CLIP raw={round_score:.3f}")

            if state.image is None:
                state.log("自主模式", "改图：图像生成失败，终止", "")
                break

            if round_score > best_score:
                best_score = round_score
                best_state = agent._copy_state(state)

            # 自适应停止：仅在有"上一轮"可比较时累计 stale
            # delta = 本轮相对上一轮的提升幅度 ≤ 阈值 → stale++
            if prev_round_score is not None:
                if round_score - prev_round_score <= config.adaptive_stop_delta:
                    stale_count += 1
                else:
                    stale_count = 0
            prev_round_score = round_score

            # 每轮改图结果都 yield 给前端（不管分数是否提升，用户都要看到图）
            yield state

            # 自适应停止：连续 2 轮无显著提升
            if config.adaptive_stop and stale_count >= 2:
                state.log("自主模式", "自适应停止",
                          f"连续 {stale_count} 轮 CLIP 无显著提升（delta ≤ {config.adaptive_stop_delta}），提前退出改图循环")
                break

    # ═════════════════════════════════════════════════════════════════════
    # 收尾
    # ═════════════════════════════════════════════════════════════════════
    if best_score < target:
        state.log(
            "自主模式", "预算耗尽，未达到目标分",
            (
                f"当前最优 CLIP raw={best_score:.3f} < 目标 {target}；"
                f"已用完改图上限 {config.max_image_improve_rounds} 轮"
                f"{'，全自主改诗已完成' if config.allow_poem_refine else ''}。"
                "返回历史最优结果，而不是伪装达标。"
            ),
        )
    verdict = ("✓ 达标" if best_score >= target
               else f"⚠ 未达标，返回最优结果（raw={best_score:.3f}）")
    # 将最优结果的关键属性复制到 state，避免 yield best_state
    # （best_state 是早期快照，其 trace/image_history 不完整，单独 yield 会导致重复保存）
    if best_state is not state:
        state.image = best_state.image
        state.clip_score_poem = best_state.clip_score_poem
        state.clip_score_prompt = best_state.clip_score_prompt
        state.clip_score_final = best_state.clip_score_final
        state.clip_msg = best_state.clip_msg
        state.final_reflection = best_state.final_reflection
    state.phase = Phase.DONE
    state.log("自主模式", "结束", f"{verdict} | 目标 raw≥{target}")
    yield state
