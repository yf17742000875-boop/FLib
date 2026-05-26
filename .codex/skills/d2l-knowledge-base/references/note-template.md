# D2L Section Note Template

Use this template for one D2L page or one small section.

```markdown
# {section-id} {中文标题}

Source: {D2L URL}
Chapter: {chapter number and name}
Article type: {math / model-theory / data-pipeline / from-scratch / concise-framework / optimization / architecture / engineering / project-bridge}

## 一句话总结

用 1-2 句话说明本节最核心的学习收获。

## 本节解决的问题

- 问题 1
- 问题 2
- 问题 3

## 核心概念 / 公式 / API

### Concept 1

解释概念、公式或 API。优先写“为什么需要它”和“它解决什么问题”。

```python
# Optional minimal API example
```

## 输入输出形状

| Object | Shape / dtype / device | Meaning |
| --- | --- | --- |
| `X` | `(batch_size, ...)`, `float32`, CPU/GPU | input features |
| `y` | `(batch_size,)`, `int64` | labels |
| `y_hat` | `(batch_size, num_classes)` | predictions/logits/probabilities |

## 最小可运行代码

```python
"""
Minimal code that verifies the section's main idea.
Keep it small enough to debug by inspection.
"""
```

Expected output:

```text
...
```

## 课后练习

### Exercise 1

Answer with reasoning:

1. Restate the question.
2. Give the key formula or idea.
3. Work through the answer.
4. Note any assumptions.

## 易错点

- Shape mistake:
- dtype/device mistake:
- API misuse:
- Conceptual trap:

## 和前后章节或真实项目的连接

- Previous section:
- Next section:
- Practical project connection:

## 复习卡片

Q: ...
A: ...

Q: ...
A: ...
```

## Minimal Standards

- Include the D2L source URL.
- Include at least one concrete shape table for tensor-heavy sections.
- Include at least three active-recall cards.
- If code is included, it should be runnable independently or clearly marked as a snippet.
- If exercises are answered, include reasoning and not only final results.
