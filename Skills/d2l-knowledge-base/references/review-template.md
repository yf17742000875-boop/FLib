# D2L Review Templates

Use this file for chapter reviews, spaced review, and error logs.

## Chapter Review

```markdown
# Chapter {XX} Review: {chapter title}

## Chapter Map

| Section | Main idea | Code artifact | Review priority |
| --- | --- | --- | --- |
| {section-id} | ... | ... | High/Medium/Low |

## Core Thread

Explain how the sections connect. Prefer cause-and-effect:

1. Concept introduced.
2. Minimal implementation.
3. Framework implementation.
4. Practical limitation or extension.

## Must-Know Shapes / APIs

| Topic | Shape / API | Why it matters |
| --- | --- | --- |

## Exercise Patterns

- Pattern:
- Key formula:
- Typical mistake:

## Project Connections

- DINOv3:
- FoundationPose:
- Other PyTorch repos:

## Next Review Actions

- Re-run:
- Re-derive:
- Re-read:
- Build:
```

## Spaced Review Cards

Use short cards that can be answered without rereading the whole section.

```markdown
Q: What shape does `{name}` have after `{operation}`?
A: ...

Q: Why does `{concept}` matter?
A: ...

Q: What PyTorch API corresponds to the manual implementation of `{idea}`?
A: ...

Q: What is the most likely bug when `{symptom}` happens?
A: ...
```

## Error Log

Use this when an implementation or experiment fails.

```markdown
# Error Log: {short title}

## Symptom

What failed? Include command, traceback summary, or wrong output.

## Scope

- Environment:
- Data:
- Model:
- Loss/optimization:
- Evaluation/visualization:

## Root Cause

Explain the smallest confirmed cause.

## Fix

What changed and why?

## Validation

How did you confirm the fix?

## D2L Connection

Which D2L concept explains this failure?
```
