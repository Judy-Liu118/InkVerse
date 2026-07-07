"""
test_prompt_clean -- 负面提示词剥离（strip_negative_prompt_lines）

T2I 模型按正向语义理解 prompt，"负面清单"里点名的物体反而更容易被画出来，
所以 LLM 输出中的负面内容必须整块剥离，不能只删标题行。
"""
from core.image.prompt import strip_negative_prompt_lines


def test_single_line_negative_header_dropped():
    text = "Subject: lone boat on river\nNegative prompt: no people, no text\nAtmosphere: misty dusk"
    out = strip_negative_prompt_lines(text)
    assert "no people" not in out
    assert "Subject: lone boat on river" in out
    assert "Atmosphere: misty dusk" in out


def test_multiline_negative_block_fully_dropped():
    """多行负面块：标题行之后的条目行必须一并删除，不能泄漏进正向 prompt。"""
    text = (
        "Subject: winter mountain\n"
        "Negative prompt:\n"
        "- no people\n"
        "- no modern buildings\n"
        "\n"
        "Atmosphere: silent snow"
    )
    out = strip_negative_prompt_lines(text)
    assert "no people" not in out
    assert "no modern buildings" not in out
    assert "Subject: winter mountain" in out
    assert "Atmosphere: silent snow" in out


def test_negative_block_ends_at_next_section_header():
    """负面块后紧跟正向段头（无空行分隔）时，段头行应恢复保留。"""
    text = (
        "Subject: plum blossom\n"
        "Forbidden: \n"
        "- no birds\n"
        "Composition: rule of thirds, negative space"
    )
    out = strip_negative_prompt_lines(text)
    assert "no birds" not in out
    assert "Composition: rule of thirds" in out


def test_chinese_negative_block_dropped():
    text = (
        "主体: 孤舟蓑笠\n"
        "负面提示词：\n"
        "- 不要出现现代元素\n"
        "氛围: 江雪空寂"
    )
    out = strip_negative_prompt_lines(text)
    assert "现代元素" not in out
    assert "主体: 孤舟蓑笠" in out
    assert "氛围: 江雪空寂" in out


def test_standalone_negative_bullet_dropped():
    """不带块标题的独立负面条目行也应删除。"""
    text = "Subject: autumn geese\n- no people\n避免出现文字水印\nAtmosphere: vast sky"
    out = strip_negative_prompt_lines(text)
    assert "no people" not in out
    assert "文字水印" not in out
    assert "Atmosphere: vast sky" in out


def test_plain_prompt_untouched():
    text = "Subject: bamboo grove\nEnvironment: light rain\nAtmosphere: serene"
    assert strip_negative_prompt_lines(text) == text
