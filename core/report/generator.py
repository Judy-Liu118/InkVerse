"""
core.report.generator -- 创作报告生成器
"""
import os
import io
import time
import base64
import traceback
import re
from typing import Dict, Optional
from PIL import Image
from config import OUTPUT_DIR
from core.logger import get_logger

_log = get_logger(__name__)


def _poem_to_html_lines(poem_text: str) -> str:
    lines = [l.strip() for l in poem_text.strip().split("\n") if l.strip()]
    clean_lines = []
    for line in lines:
        clean = re.sub(r"[，。！？；、,.;!?]", "", line).strip()
        if clean:
            clean_lines.append(clean)
    punct_style = "font-family:'Noto Serif SC','SimSun','宋体',serif;font-size:inherit;"
    rows_html = ""
    for i in range(0, len(clean_lines), 2):
        first = clean_lines[i] + f'<span style="{punct_style}">，</span>'
        if i + 1 < len(clean_lines):
            second = clean_lines[i + 1] + f'<span style="{punct_style}">。</span>'
            rows_html += f'<div style="margin:0.45em 0;">{first}{second}</div>'
        else:
            rows_html += f'<div style="margin:0.45em 0;">{first}</div>'
    return rows_html


class ReportGenerator:
    @staticmethod
    def generate(
        user_input: str, poem_title: str, poem_text: str,
        prompt_text: str, image,
        model_usage=None,
        clip_info: Optional[Dict] = None,
    ) -> str:
        try:
            if image is None:
                return "错误：图像为空，无法生成报告"
            if not isinstance(image, Image.Image):
                try:
                    import numpy as np
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(image.astype("uint8"), "RGB")
                    else:
                        image = Image.fromarray(image)
                except Exception:
                    return f"不支持的图像类型: {type(image)}"

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = int(time.time())
            # 诗名加入文件名（过滤非法字符）
            safe_title = re.sub(r'[\\/:*?"<>|]', '', (poem_title or '无题').strip())
            safe_title = safe_title[:12] if len(safe_title) > 12 else safe_title
            report_path = os.path.join(OUTPUT_DIR, f"report_{safe_title}_{timestamp}.html")

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            poem_html_lines = _poem_to_html_lines(poem_text)
            prompt_escaped = prompt_text.replace("<", "&lt;").replace(">", "&gt;")
            model_html = ReportGenerator._build_model_html(model_usage)
            clip_html = ReportGenerator._build_clip_html(clip_info)

            html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>《{poem_title}》 · 诗画墨语</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=ZCOOL+XiaoWei&family=Noto+Serif+SC:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:     #f4ecda; --ink: #2a1a08; --red: #a83030;
    --gold:   #b8892e; --border: #cbb98a; --panel: #fdf5e6;
  }}
  body {{
    font-family: 'Noto Serif SC', '楷体', serif;
    background: var(--bg); color: var(--ink);
    max-width: 860px; margin: 0 auto; padding: 40px 32px 60px;
  }}
  h1 {{
    font-family: 'ZCOOL XiaoWei', '楷体', serif;
    font-size: 2.4rem; text-align: center; color: var(--red);
    letter-spacing: 0.4em; font-weight: 400;
    border-bottom: 1px solid var(--border); padding-bottom: 16px;
    margin-bottom: 36px;
  }}
  .poem {{
    font-family: 'Noto Serif SC', '楷体', 'STKaiti', '宋体', serif;
    font-size: 1.5rem; line-height: 2.3; text-align: center;
    padding: 32px; background: var(--panel);
    border: 1px solid var(--border); border-radius: 3px;
    letter-spacing: 0.2em; margin: 0 auto 36px;
    max-width: 520px;
  }}
  .section {{ margin: 28px 0; }}
  .section h3 {{
    font-size: 0.78rem; letter-spacing: 0.38em; color: var(--gold);
    border-bottom: 1px solid var(--border); padding-bottom: 5px;
    margin-bottom: 12px; font-weight: 400;
  }}
  .section p {{ font-size: 0.97rem; line-height: 1.85; }}
  pre {{
    font-family: 'Courier New', monospace;
    font-size: 0.85rem; line-height: 1.65;
    background: #faf4eb; border: 1px dashed var(--border);
    border-radius: 3px; padding: 16px; white-space: pre-wrap;
    color: #5a4430;
  }}
  .model-table {{
    width: 100%; border-collapse: collapse;
    font-size: 0.88rem;
  }}
  .model-table td {{
    padding: 6px 12px; border-bottom: 1px solid #e8dcc4;
    vertical-align: top;
  }}
  .model-table td:first-child {{
    color: var(--gold); white-space: nowrap;
    width: 110px; font-size: 0.8rem; letter-spacing: 0.1em;
  }}
  .clip-row {{ display: flex; gap: 24px; flex-wrap: wrap; margin-top: 8px; }}
  .clip-badge {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 3px; padding: 8px 16px; text-align: center;
    flex: 1; min-width: 120px;
  }}
  .clip-badge .label {{
    font-size: 0.72rem; letter-spacing: 0.15em; color: var(--gold);
    margin-bottom: 4px;
  }}
  .clip-badge .value {{
    font-size: 1.2rem; font-family: 'Courier New', monospace;
    color: var(--ink);
  }}
  .clip-badge .sub {{
    font-size: 0.68rem; color: #9e8870; margin-top: 3px;
  }}
  .painting {{ text-align: center; margin-top: 8px; }}
  .painting img {{
    max-width: 100%; border: 1px solid var(--border);
    border-radius: 3px; box-shadow: 3px 5px 16px rgba(40,25,10,0.12);
  }}
  .footer {{
    text-align: center; margin-top: 48px;
    font-size: 0.75rem; color: #b0956a; letter-spacing: 0.3em;
  }}
</style>
</head>
<body>
  <h1>《{poem_title}》</h1>
  <div class="poem">{poem_html_lines}</div>

  <div class="section">
    <h3>创 作 要 求</h3>
    <p>{user_input if user_input.strip() else "（用户直接提供诗作）"}</p>
  </div>

  {model_html}

  {clip_html}

  <div class="section">
    <h3>绘 画 提 示 词</h3>
    <pre>{prompt_escaped}</pre>
  </div>

  <div class="section">
    <h3>生 成 画 作</h3>
    <div class="painting">
      <img src="data:image/png;base64,{img_b64}" alt="{poem_title}">
    </div>
  </div>

  <div class="footer">诗画墨语 · InkVerse &nbsp;·&nbsp; AI 创作</div>
</body>
</html>"""

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html)

            _log.info("报告已生成 → %s", report_path)
            return report_path

        except Exception as e:
            error_msg = f"生成报告失败：{str(e)}\n{traceback.format_exc()}"
            _log.error(error_msg)
            return error_msg

    @staticmethod
    def _build_model_html(model_usage) -> str:
        if model_usage is None:
            return ""
        if hasattr(model_usage, "as_dict"):
            usage_dict = model_usage.as_dict()
        elif isinstance(model_usage, dict):
            usage_dict = model_usage
        else:
            return ""
        rows = ""
        for step, model in usage_dict.items():
            if model and model != "—":
                rows += f"<tr><td>{step}</td><td>{model}</td></tr>"
        if not rows:
            return ""
        return f"""
  <div class="section">
    <h3>模 型 配 置</h3>
    <table class="model-table">
      {rows}
    </table>
  </div>"""

    @staticmethod
    def _build_clip_html(clip_info: Optional[Dict]) -> str:
        if not clip_info:
            return ""

        def badge(label: str, raw: float, sub: str) -> str:
            norm = (raw + 1.0) / 2.0
            level_color = (
                "#4a8a4a" if raw >= 0.28 else
                "#8a7a4a" if raw >= 0.22 else "#8a4a4a"
            )
            return f"""
      <div class="clip-badge">
        <div class="label">{label}</div>
        <div class="value" style="color:{level_color};">{raw:.3f}</div>
        <div class="sub">归一化 {norm:.3f} · {sub}</div>
      </div>"""

        poem_raw = clip_info.get("poem", 0.0)
        prompt_raw = clip_info.get("prompt", 0.0)
        final_raw = clip_info.get("final", 0.0)
        has_poem = poem_raw > 0.0

        badges = ""
        if has_poem:
            badges += badge("诗-图一致性", poem_raw, "诗歌关键词锚点 × 0.6")
        badges += badge("词-图一致性", prompt_raw, "英文提示词锚点 × 0.4")
        if has_poem:
            badges += badge("综合得分", final_raw, "加权平均")

        anchor_note = (
            "双锚点评估：诗歌直接提取的视觉关键词（主） + 英文提示词（辅）"
            if has_poem else
            "单锚点评估：英文提示词（诗歌关键词提取失败或未启用）"
        )

        return f"""
  <div class="section">
    <h3>C L I P &nbsp; 图 文 一 致 性</h3>
    <p style="font-size:0.78rem;color:#9e8870;margin-bottom:12px;">{anchor_note}</p>
    <div class="clip-row">{badges}</div>
  </div>"""
