"""
core.image.prompt -- 绘画提示词生成器
"""
import re
import time
from config import PROMPT_MAX_TOKENS, PROMPT_TEMPERATURE
from core.logger import get_logger

_log = get_logger(__name__)


class PromptGenerator:
    def __init__(self):
        pass

    def generate(self, poem: str, lang: str, adapter,
                 user_request: str = "") -> str | None:
        _log.info("使用的后端: %s, 语言: %s", adapter.backend, lang)
        req_hint = f"\n\n[User's original request for reference — must be faithfully depicted]:\n{user_request}" if user_request else ""
        if lang == "英文":
            system_msg = (
                "You are a strict poetry-to-image prompt writer for Chinese classical aesthetics. "
                "Reply only in English using the specified format. Never invent concrete visual objects."
            )
            user_prompt = self._build_structured_en_prompt(poem) + req_hint
        else:
            system_msg = (
                "你是一位严格的诗生图提示词撰写者，精通中国古典美学。"
                "请用中文按照指定格式输出描述，不能凭空添加具体物象。"
            )
            user_prompt = self._build_structured_cn_prompt(poem) + req_hint

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                reply = adapter.generate(messages, max_tokens=PROMPT_MAX_TOKENS, temperature=PROMPT_TEMPERATURE)
                reply = reply.replace("```", "").strip()
                reply = self._clean_prompt_output(reply)
                if reply and len(reply) >= 20:
                    _log.info("生成成功，长度=%d 字符", len(reply))
                    return reply
                else:
                    _log.warning("第%d次生成内容过短或为空，%s", attempt + 1,
                                 "重试" if attempt < max_attempts - 1 else "判定失败")
            except Exception as e:
                if attempt < max_attempts - 1:
                    sleep_time = (attempt + 1) * 2  # 指数退避: 2s, 4s
                    _log.warning("第%d次生成异常: %s，%d秒后重试...", attempt + 1, e, sleep_time)
                    time.sleep(sleep_time)
                else:
                    _log.error("第%d次生成异常: %s，重试耗尽", attempt + 1, e)
                    return None
        return None

    @staticmethod
    def _clean_prompt_output(text: str) -> str:
        kept = []
        for line in (text or "").splitlines():
            if re.match(r"\s*(forbidden|negative prompt|禁加元素|负面提示词)\s*[:：]", line, flags=re.I):
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    def _build_structured_en_prompt(self, poem: str) -> str:
        return f"""Analyze the following Chinese classical poem and output ONLY an English visual description in the exact format below. Do not include any explanation or intro text.

Hard constraints:
- Only depict concrete visual elements that are explicitly present in the poem.
- Do NOT add humans, animals, boats, kites, buildings, tools, lanterns, bridges, mountains, birds, moon, sun, flowers, or narrative props unless they are named in the poem.
- If a line implies emotion or thought (for example "asking", "longing", "not knowing"), render it as mood, composition, light, wind, or negative space, NOT as a literal extra person.
- If the poem is mainly about season, wind, light, water, plants, or atmosphere, make those natural elements the subject.
- The Subject field must list only poem-supported visible elements. If no person appears in the poem, do not mention people at all.
- Avoid invented story details. Preserve Chinese ink-wash aesthetics.
- Do not include a negative prompt or a forbidden-object list in the final output.

【Poem】
{poem}

Format to copy exactly:
Subject: [only poem-supported visible subjects; omit people entirely if absent]
Environment: [only poem-supported scenery, weather, season, time, and spatial setting]
Atmosphere: [overall emotional tone, ethereal mood]
Color Palette: [dominant tones, e.g., ink monochrome, soft charcoal gray, washed jade]
Art Style: [traditional Chinese ink wash painting style, artistic minimalism, poetic imagery]
Composition: [viewpoint, negative space layout, rule of thirds]

Now output language description:"""

    def _build_structured_cn_prompt(self, poem: str) -> str:
        return f"""分析以下古诗，严格按照给定的格式输出中文视觉意象描述，不要有任何多余的解释。

硬性约束：
- 只能描绘诗中明确出现的具体视觉元素。
- 诗中没有写到的人物、动物、船、风筝、建筑、器物、灯笼、桥、山、鸟、月、日、花等，不得加入。
- "相问、思念、不知"等情绪或心理句，只能转化为氛围、构图、光影、风感或留白，不得画成额外人物。
- 如果诗主要写季节、风、光、水、草木或气氛，就让这些自然元素成为主体。
- "主体"一栏必须只列诗中支持的可见元素；诗中没有人物时不要提人物。
- 避免添加故事情节，保持中国水墨审美。
- 最终输出中不要写负面提示词或"禁加元素"清单。

【诗作】
{poem}

输出格式：
主体: [仅列诗中支持的可见主体；无人则不要提人物]
环境: [仅列诗中支持的景物、天气、季节、时辰和空间]
氛围: [整体意境与情感基调]
色调: [主导色彩，如水墨留白、淡墨微晕、素雅翠绿]
艺术风格: [中国传统水墨画风格，写意大开大合，留白美学]
构图: [远近视点，留白比例]

现在输出："""
