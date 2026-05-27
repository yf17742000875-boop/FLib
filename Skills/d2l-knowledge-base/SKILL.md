---
name: d2l-knowledge-base
description: Build and update the repository's Dive into Deep Learning (D2L) knowledge base. Use when the user asks to organize D2L study notes, summarize D2L sections or chapters, answer D2L exercises, create review cards, build chapter indexes, connect D2L concepts to practical PyTorch/vision projects, or maintain reusable D2L learning templates.
---

# D2L Knowledge Base

## Core Workflow

Use a section-first structure. Treat one D2L web page as one focused note unless the user asks for a chapter-level review.

1. Identify the D2L source section, chapter number, and article type.
2. Create or update the section note under `notes/d2l/chXX/{section-id}-{slug}.md`.
3. Keep direct-follow classroom code in `d2l/code/`; put extended experiments in `experiments/d2l/chXX/`.
4. Include active-recall review cards and concrete shape/API checks.
5. Add chapter-level synthesis to `reports/d2l/chXX-review.md` when summarizing multiple sections.

Use ASCII filenames for new notes, experiments, and reports. Chinese prose is fine inside files.

## Note Requirements

Every section note should include:

- `一句话总结`
- `本节解决的问题`
- `核心概念 / 公式 / API`
- `输入输出形状`
- `最小可运行代码`
- `课后练习`
- `易错点`
- `和前后章节或真实项目的连接`
- `复习卡片`

Do not force every block to be long. If a section is mostly API usage or data loading, emphasize shapes, dtype, loader parameters, and failure modes. If it is theory-heavy, emphasize formulas, gradients, assumptions, and the connection to implementation.

## References

- For the section-note scaffold, read `references/note-template.md`.
- For choosing the correct note emphasis by article type, read `references/article-taxonomy.md`.
- For chapter reviews, spaced review, and error logs, read `references/review-template.md`.

## Quality Rules

- Explain the concept before code when the section introduces a new idea.
- Prefer minimal runnable code over notebook-only logic.
- Record tensor shapes, dtypes, device assumptions, and loss/metric definitions.
- Answer exercises with reasoning, not only final answers.
- Connect textbook concepts to practical projects such as DINOv3, FoundationPose, or PyTorch debugging when the connection is meaningful.
