# prompts/

集中化的 prompt 管理目录。所有 LLM system / user prompt 以 YAML 形式存放于此，由 `core.prompts` 模块统一加载和渲染。

## 设计目标

1. **可审阅**：git diff 时 prompt 变更一目了然，不被 Python 代码改动淹没
2. **可枚举**：`list_prompts()` 一键列出全部可用 prompt，便于审计与 A/B 测试
3. **可追踪**：每个文件都带 `version` 和 `description`，配合 git 即天然版本管理
4. **零运行时成本**：`@lru_cache` 单次加载、常驻内存

## YAML 结构

```yaml
name: keyword_extract           # 与文件名一致
version: 1                       # 修改 prompt 时递增
description: 一句话说明用途
locale: en                       # zh | en | mixed
system: |
  System message here.
  支持 {var} 风格的变量插值。
user: |
  User message here, must reference {variables}.
```

`system` 可省略；`user` 必填。变量用 Python `str.format()` 渲染——缺变量会直接 `KeyError`，避免静默生成残缺 prompt。

## 调用示例

```python
from core.prompts import render_messages

messages = render_messages(
    "agent.refine_poem",
    expected_chars=7, expected_lines=4,
    old_poem="...", feedback="加强意境深度",
)
response = adapter.generate(messages, max_tokens=120, temperature=0.75)
```

点号路径会自动映射成相对路径：`agent.refine_poem` → `prompts/agent/refine_poem.yaml`。

## 当前迁移状态

| Prompt | YAML | Caller |
|---|---|---|
| 视觉关键词抽取 | `agent/keyword_extract.yaml` | `PoetryAgent._phase_keyword_extract` |
| 诗名生成 | `agent/title_generation.yaml` | `PoetryAgent._phase_title` |
| 改诗 | `agent/refine_poem.yaml` | `PoetryAgent.refine_poem` |
| 提示词自检 | `agent/prompt_review.yaml` | `PoetryAgent._phase_prompt_review` |

剩余仍内联在代码中、待后续迁移：
- `prompts/image/structured_prompt_{en,cn}.yaml` ← `core.image.prompt`
- `prompts/scorer/scoring_rubric.yaml` ← `core.poem.scorer._SCORING_PROMPT_TEMPLATE`
- `prompts/agent/{edit_image,challenger}.yaml` ← `PoetryAgent.edit_image_by_feedback` / `_CHALLENGER_PROMPT`
