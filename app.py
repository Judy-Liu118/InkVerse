"""
app.py -- UI 层（CSS + Gradio Blocks 布局 + main 入口）

业务逻辑见 app_handlers.py，工具函数与常量见 app_utils.py。
"""
import os

import gradio as gr

from core.logger import setup_logging
from config import (
    POEM_CANDIDATE_COUNT, STYLE_MAP,
    DEEPSEEK_API_KEY, DASHSCOPE_API_KEY,
    LOCAL_LLM_AVAILABLE, LOCAL_LORA_AVAILABLE, LOCAL_IMAGE_AVAILABLE,
)
from app_utils import (
    MODEL_CHOICES, POEM_MODEL_CHOICES, REFINE_POEM_MODEL_CHOICES,
    IMAGE_EDIT_MODEL_CHOICES, IMAGE_BACKEND_CHOICES, IMAGE_EDIT_DEFAULT_MODEL,
    _DEFAULT_POEM_MODEL, _DEFAULT_IMAGE_BACKEND, _poem_html,
)
from app_handlers import (
    on_create, on_refine_poem, on_regen_image, on_rewrite_regen,
    on_edit_image_api, on_sync_display, on_autonomous_create, on_report,
)

setup_logging()

# ── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=ZCOOL+XiaoWei&family=Noto+Serif+SC:wght@400;500;600&display=swap');

:root {
    --bg:        #f4ecda;
    --bg2:       #ede0c4;
    --ink:       #28190a;
    --ink-mid:   #5a4430;
    --ink-faint: #9e8870;
    --red:       #a83030;
    --red-h:     #c04040;
    --gold:      #b8892e;
    --border:    #cbb98a;
    --panel:     #faf3e2;
    --shadow:    rgba(40,25,10,0.13);
}

html, body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Noto Serif SC', '宋体', serif !important;
    color: var(--ink) !important;
}

#app-header { text-align:center; padding:30px 0 18px; border-bottom:1px solid var(--border); margin-bottom:26px; }
#app-header h1 { font-family:'ZCOOL XiaoWei','楷体',serif !important; font-size:2.9rem !important; letter-spacing:0.5em !important; color:var(--ink) !important; margin:0 !important; font-weight:400 !important; text-shadow:1px 2px 8px var(--shadow); }
#app-header .sub { font-size:0.8rem; letter-spacing:0.38em; color:var(--ink-faint); margin-top:7px; }

.sec { font-size:0.7rem; letter-spacing:0.45em; color:var(--gold); text-align:center; margin:18px 0 7px; display:flex; align-items:center; gap:10px; }
.sec::before, .sec::after { content:''; flex:1; height:1px; background:var(--border); opacity:0.65; }

label > span.svelte-1gfkn6j, .label-wrap > span { display:none !important; }

textarea, input[type="text"] { background:var(--panel) !important; border:1px solid var(--border) !important; border-radius:3px !important; color:var(--ink) !important; font-family:'Noto Serif SC','宋体',serif !important; font-size:0.96rem !important; line-height:1.75 !important; padding:10px 14px !important; resize:vertical !important; transition:border-color 0.2s,box-shadow 0.2s; }
textarea:focus, input[type="text"]:focus { border-color:var(--gold) !important; box-shadow:0 0 0 3px rgba(184,137,46,0.12) !important; outline:none !important; }
textarea::placeholder { color:var(--ink-faint) !important; }

#title-out textarea, #title-out input { font-family:'ZCOOL XiaoWei','楷体',serif !important; font-size:2rem !important; text-align:center !important; color:var(--red) !important; letter-spacing:0.38em !important; background:transparent !important; border:none !important; border-bottom:1px solid var(--border) !important; border-radius:0 !important; padding:6px 0 10px !important; font-weight:500 !important; }

#poem-edit textarea { font-size:1.04rem !important; line-height:1.9 !important; }
#prompt-out textarea { font-family:'Courier New',monospace !important; font-size:0.82rem !important; color:var(--ink-mid) !important; line-height:1.65 !important; border-style:dashed !important; }
#report-status textarea { font-size:0.83rem !important; color:var(--ink-mid) !important; background:transparent !important; border:none !important; border-top:1px dashed var(--border) !important; border-radius:0 !important; padding:5px 0 !important; }

#image-out { border:1px solid var(--border) !important; border-radius:3px !important; background:var(--panel) !important; box-shadow:3px 5px 18px var(--shadow) !important; overflow:hidden; }

button.primary { background:var(--red) !important; color:#fff !important; border:none !important; font-family:'Noto Serif SC',serif !important; font-size:1.02rem !important; letter-spacing:0.22em !important; padding:11px 0 !important; border-radius:2px !important; box-shadow:0 2px 8px rgba(168,48,48,0.28) !important; transition:background 0.18s,transform 0.12s !important; }
button.primary:hover { background:var(--red-h) !important; transform:translateY(-1px) !important; }
button.secondary { background:transparent !important; border:1px solid var(--border) !important; color:var(--ink-mid) !important; font-family:'Noto Serif SC',serif !important; font-size:0.9rem !important; letter-spacing:0.15em !important; border-radius:2px !important; transition:border-color 0.18s,color 0.18s !important; }
button.secondary:hover { border-color:var(--gold) !important; color:var(--ink) !important; }

#seal { width:54px; height:54px; border:2px solid var(--red); border-radius:4px; color:var(--red); font-family:'ZCOOL XiaoWei','STKaiti','楷体',serif; font-size:0.66rem; display:flex; flex-direction:column; align-items:center; justify-content:center; opacity:0.7; transform:rotate(-8deg); margin:18px auto 6px; line-height:1.5; letter-spacing:0.06em; user-select:none; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg2); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
footer { display:none !important; }

/* 全自主创作按钮：深青色，区别于普通红色 primary */
#auto-btn { background:#1a5c52 !important; box-shadow:0 2px 8px rgba(26,92,82,0.32) !important; letter-spacing:0.28em !important; font-size:1.05rem !important; }
#auto-btn:hover { background:#236e62 !important; }
#auto-desc p { font-size:0.78rem !important; color:var(--ink-faint) !important; line-height:1.6 !important; margin:4px 0 10px !important; }
"""

# ── Gradio Blocks ─────────────────────────────────────────────────────────────
with gr.Blocks(
    title="诗画墨语",
    theme=gr.themes.Base(
        primary_hue="orange", neutral_hue="stone",
        font=[gr.themes.GoogleFont("Noto Serif SC"), "serif"],
    ),
    css=CSS,
) as demo:

    agent_state = gr.State("{}")

    gr.HTML("""
    <div id="app-header">
        <h1>诗　画　墨　语</h1>
        <div class="sub">以诗入画 &middot; 以意生境 &middot; InkVerse</div>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ── 左侧控制面板 ──────────────────────────────────────────────────────
        with gr.Column(scale=4, min_width=280):
            gr.HTML('<div class="sec">创 作 要 求</div>')
            user_req = gr.Textbox(
                show_label=False,
                placeholder="例：写一首以春天为主题的七言绝句……",
                lines=4,
            )

            gr.HTML('<div class="sec">诗 文 输 入 / 编 辑</div>')
            poem_edit = gr.Textbox(
                show_label=False,
                placeholder="粘贴已有诗作可直接配图；\nAI 生成后诗文也将显示于此，可自由修改。",
                lines=5, elem_id="poem-edit", interactive=True,
            )

            gr.HTML('<div class="sec">改 诗（可选不同模型）</div>')
            with gr.Row():
                refine_feedback = gr.Textbox(
                    show_label=False,
                    placeholder="例：意境太浅，改得更深沉、更有禅意",
                    lines=2, scale=3,
                )
                refine_poem_model = gr.Dropdown(
                    choices=REFINE_POEM_MODEL_CHOICES,
                    value="qwen-plus",
                    label="改诗模型（仅 API）", show_label=True, scale=2,
                )
            refine_poem_btn = gr.Button("✦ 改诗", variant="secondary")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="sec">语言</div>')
                    lang_radio = gr.Radio(
                        choices=["英文", "中文"], value="英文",
                        show_label=False, interactive=True,
                    )
                with gr.Column(scale=2):
                    gr.HTML('<div class="sec">图 像 风 格</div>')
                    style_drop = gr.Dropdown(
                        choices=list(STYLE_MAP.keys()), value="水墨画",
                        show_label=False, interactive=True,
                    )

            gr.HTML('<div class="sec">模 型 配 置</div>')
            with gr.Row():
                intent_model = gr.Dropdown(
                    choices=MODEL_CHOICES, value="qwen-plus",
                    label="意图评分模型", show_label=True,
                )
                poem_model = gr.Dropdown(
                    choices=POEM_MODEL_CHOICES, value=_DEFAULT_POEM_MODEL,
                    label="诗歌生成模型", show_label=True,
                )
                title_model = gr.Dropdown(
                    choices=MODEL_CHOICES, value="qwen-plus",
                    label="诗名生成模型", show_label=True,
                )
                prompt_model = gr.Dropdown(
                    choices=MODEL_CHOICES, value="qwen-max",
                    label="提示词生成模型", show_label=True,
                )
            gr.Markdown("💡 各步骤可独立选择模型；改诗时可再换模型。")

            gr.HTML('<div class="sec">图 像 生 成 后 端</div>')
            image_backend = gr.Dropdown(
                choices=IMAGE_BACKEND_CHOICES,
                value=_DEFAULT_IMAGE_BACKEND,
                show_label=False,
                interactive=True,
            )

            gr.HTML('<div class="sec">操　　作</div>')
            with gr.Row():
                submit_btn = gr.Button("✦ 开始创作", variant="primary", scale=3)
                report_btn = gr.Button("生成报告", variant="secondary", scale=2)
            report_out = gr.Textbox(
                show_label=False, interactive=False,
                elem_id="report-status", lines=1,
            )

            gr.HTML('<div class="sec">🤖 全 自 主 创 作</div>')
            gr.Markdown(
                "Agent 自主完成：生成诗 → 品质筛选 → 改诗 → 提取意象 → 生图 → CLIP 评分 → 改图，"
                "无需人工介入，最终返回最优结果。",
                elem_id="auto-desc",
            )
            with gr.Row():
                auto_target_score = gr.Slider(
                    minimum=0.10, maximum=0.40, value=0.30, step=0.01,
                    label="目标 CLIP 分（raw，达到提前停止）",
                    interactive=True, scale=3,
                )
                auto_max_img_rounds = gr.Slider(
                    minimum=0, maximum=4, value=2, step=1,
                    label="最大改图轮次",
                    interactive=True, scale=1,
                )
            with gr.Row():
                auto_allow_poem_refine = gr.Checkbox(
                    value=True, label="允许 Agent 自主改诗",
                    interactive=True, scale=1,
                )
                auto_max_poem_rounds = gr.Slider(
                    minimum=0, maximum=3, value=1, step=1,
                    label="最大改诗轮次",
                    interactive=True, scale=2,
                )
            auto_image_mode = gr.Radio(
                choices=[
                    "改写重生图（LLM 改写 Prompt 后重新生图）",
                    "图像编辑（百炼 API，保留原图构图）",
                ],
                value="改写重生图（LLM 改写 Prompt 后重新生图）",
                label="自主改图模式",
                interactive=True,
            )
            auto_edit_model = gr.Dropdown(
                choices=IMAGE_EDIT_MODEL_CHOICES[:-1],
                value=f"edit:{IMAGE_EDIT_DEFAULT_MODEL}",
                label="自主图像编辑模型",
                interactive=True,
            )
            auto_llm_driven_loop = gr.Checkbox(
                value=False,
                label="LLM 驱动改图循环（实验）",
                info="勾选后改图循环由 LLM 决定调 edit_image / refine_poem_and_regen / stop；默认走写死流程。",
                interactive=True,
            )
            auto_btn = gr.Button("🤖 全自主创作", variant="primary", elem_id="auto-btn")

            gr.HTML('<div id="seal">詩<br>畫<br>工坊</div>')

        # ── 右侧展示面板 ──────────────────────────────────────────────────────
        with gr.Column(scale=8, min_width=500):
            title_out = gr.Textbox(
                show_label=False, interactive=False,
                elem_id="title-out", lines=1, visible=False,
            )
            gr.HTML('<div class="sec">诗 文 展 示</div>')
            poem_display = gr.HTML(value=_poem_html("", placeholder=True))

            with gr.Row(equal_height=True):
                with gr.Column(scale=5):
                    gr.HTML('<div class="sec">绘画提示词（可编辑）</div>')
                    prompt_out = gr.Textbox(
                        show_label=False, lines=12, interactive=True,
                        elem_id="prompt-out",
                        placeholder="Prompt will appear here.\nYou may edit before regenerating.",
                    )
                    regen_btn = gr.Button("↺ 仅重新生图", variant="secondary")
                    gr.HTML('<div class="sec">Agent 改 图</div>')
                    image_edit_feedback = gr.Textbox(
                        show_label=False,
                        lines=3,
                        placeholder="例：把画面改成雨后傍晚，增加远山和一盏孤灯，人物更小一些。",
                    )
                    with gr.Row():
                        edit_image_model = gr.Dropdown(
                            choices=IMAGE_EDIT_MODEL_CHOICES,
                            value=f"edit:{IMAGE_EDIT_DEFAULT_MODEL}",
                            label="编辑模型（仅「图像编辑」按钮使用）",
                            show_label=True,
                            scale=3,
                        )
                    gr.Markdown(
                        "**图像编辑**：百炼编辑 API，保留原图构图，按指令微调（需配置 DASHSCOPE_API_KEY）。  \n"
                        "**改写重生图**：LLM 将意见融入 Prompt 后重新生图，不保留构图（适合大幅改动）。",
                        elem_id="edit-help",
                    )
                    with gr.Row():
                        edit_image_api_btn  = gr.Button("✦ 图像编辑", variant="secondary", scale=1)
                        rewrite_regen_btn   = gr.Button("✦ 改写重生图", variant="secondary", scale=1)
                with gr.Column(scale=5):
                    gr.HTML('<div class="sec">生 成 画 作</div>')
                    image_out = gr.Image(
                        show_label=False, interactive=False,
                        elem_id="image-out", value=None,
                    )
                    image_history_gallery = gr.Gallery(
                        label="图像历史",
                        show_label=True,
                        elem_id="image-history-gallery",
                        columns=3,
                        rows=2,
                        object_fit="contain",
                    )

            clip_score_out = gr.Textbox(
                show_label=False, interactive=False,
                elem_id="clip-score", lines=1,
                placeholder="CLIP 双锚点图文一致性分数将在生成后显示…",
            )
            gr.HTML('<div class="sec">Agent 思 考 轨 迹</div>')
            agent_trace_out = gr.Markdown(
                value="Agent 的规划、执行、自检和反思会显示在这里。",
            )

    # ── 事件绑定 ──────────────────────────────────────────────────────────────
    submit_btn.click(
        fn=lambda v: "" if (v and v.strip()) else gr.update(),
        inputs=[user_req], outputs=[poem_edit],
    )
    submit_btn.click(
        fn=on_create,
        inputs=[
            user_req, poem_edit, lang_radio, style_drop,
            poem_model, intent_model, title_model, prompt_model,
            image_backend,
        ],
        outputs=[
            title_out, poem_edit, poem_display, prompt_out,
            image_out, clip_score_out, image_history_gallery, agent_trace_out, agent_state,
        ],
        concurrency_limit=1,
        show_progress="minimal",
    )

    poem_edit.change(fn=on_sync_display, inputs=[poem_edit], outputs=[poem_display])

    refine_poem_btn.click(
        fn=on_refine_poem,
        inputs=[
            refine_feedback, agent_state,
            poem_model, intent_model, refine_poem_model,
            prompt_model, title_model, lang_radio, style_drop, image_backend,
        ],
        outputs=[
            title_out, poem_edit, poem_display,
            prompt_out, image_out, clip_score_out,
            image_history_gallery, agent_trace_out, agent_state,
        ],
    )

    regen_btn.click(
        fn=on_regen_image,
        inputs=[prompt_out, image_backend, agent_state],
        outputs=[image_out, clip_score_out, image_history_gallery, agent_trace_out, agent_state],
    )

    edit_image_api_btn.click(
        fn=on_edit_image_api,
        inputs=[image_edit_feedback, agent_state, edit_image_model],
        outputs=[image_out, clip_score_out, image_history_gallery, agent_trace_out, agent_state],
    )

    rewrite_regen_btn.click(
        fn=on_rewrite_regen,
        inputs=[image_edit_feedback, prompt_out, image_backend, agent_state],
        outputs=[image_out, clip_score_out, prompt_out, image_history_gallery, agent_trace_out, agent_state],
    )

    report_btn.click(
        fn=on_report,
        inputs=[user_req, title_out, poem_edit, prompt_out, image_out, agent_state],
        outputs=[report_out],
    )

    _auto_shared_outputs = [
        title_out, poem_edit, poem_display, prompt_out,
        image_out, clip_score_out, image_history_gallery, agent_trace_out, agent_state,
    ]
    auto_btn.click(
        fn=lambda v: "" if (v and v.strip()) else gr.update(),
        inputs=[user_req], outputs=[poem_edit],
    )
    auto_btn.click(
        fn=on_autonomous_create,
        inputs=[
            user_req, poem_edit, lang_radio, style_drop,
            poem_model, intent_model, title_model, prompt_model,
            image_backend,
            auto_target_score, auto_max_img_rounds,
            auto_allow_poem_refine, auto_max_poem_rounds,
            auto_image_mode, auto_edit_model, auto_llm_driven_loop,
        ],
        outputs=_auto_shared_outputs,
        concurrency_limit=1,
        show_progress="full",
    )


def main():
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        if "127.0.0.1" not in existing or "localhost" not in existing:
            parts = [p for p in (existing, "localhost", "127.0.0.1") if p]
            os.environ[key] = ",".join(dict.fromkeys(parts))

    print("\n" + "=" * 60)
    print("诗画墨语 · Agent 模式启动")
    print(f"  候选诗数量: {POEM_CANDIDATE_COUNT}")
    print(f"  图像风格: 支持 {len(STYLE_MAP)} 种")
    print(f"  DeepSeek API Key: {'已设置' if DEEPSEEK_API_KEY else '未设置'}")
    print(f"  通义千问 API Key: {'已设置' if DASHSCOPE_API_KEY else '未设置'}")
    print(f"  本地 LLM 基座:    {'可用' if LOCAL_LLM_AVAILABLE else '未启用（API 模式）'}")
    print(f"  本地 LoRA Adapter: {'可用' if LOCAL_LORA_AVAILABLE else '未启用'}")
    print(f"  本地 Z-Image:     {'可用' if LOCAL_IMAGE_AVAILABLE else '未启用（百炼 API 模式）'}")
    print("  Agent 特性:")
    print("    · 双锚点 CLIP（诗-图 × 0.6 + 提示词-图 × 0.4）")
    print("    · 改诗注入 / Agent 改图规划 / 模型追踪 / 报告含模型配置")
    print("    · 🤖 全自主创作：自动改图循环 + 自主改诗 + best-state 快照")
    print("=" * 60 + "\n")
    demo.launch()


if __name__ == "__main__":
    main()
