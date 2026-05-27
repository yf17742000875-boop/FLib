"""validate_template.py — 输入合规性检查 (read-only, 不修改任何文件).

把这份模板拷贝到目标项目，改文件顶部的 `# TODO:` 后即可独立运行：

    python validate.py --task-root path/to/inputs/material

输出
----
- stdout: 人类可读报告
- exit code 0 表示完全合规 / 仅有 auto-fix 可修复的情况
- exit code 1 表示有 fatal 问题，prepare 会失败

设计原则
--------
本脚本只读不写。它先按 `references/input-spec.md` 检查输入布局，
再按 `references/auto-fix-rules.md` 把会触发的修复条目列出来给用户预览，
最后给出"建议下一步"。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_DIR_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


@dataclass
class ValidationReport:
    task_root: Path
    layout: str = "unknown"               # "imagefolder" | "pre_split" | "manifest" | "invalid"
    classes: list[str] = field(default_factory=list)
    per_class_count: dict[str, int] = field(default_factory=dict)
    pre_split_detected: bool = False
    fatal: list[str] = field(default_factory=list)
    autofix: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_fatal(self) -> bool:
        return bool(self.fatal)

    def print_human(self) -> None:
        print(f"[validate] task_root = {self.task_root}")
        print(f"[validate] layout    = {self.layout}")
        print(f"[validate] classes   = {self.classes}")
        if self.per_class_count:
            print("[validate] per-class counts:")
            for c, n in self.per_class_count.items():
                print(f"   - {c:<20s} {n:>6d}")
        if self.pre_split_detected:
            print("[validate] pre-existing train/val/test split detected (will be preserved).")
        if self.autofix:
            print("[validate] auto-fix actions that WOULD be taken in fix mode:")
            for line in self.autofix:
                print(f"   ~ {line}")
        if self.warnings:
            print("[validate] warnings:")
            for line in self.warnings:
                print(f"   ! {line}")
        if self.fatal:
            print("[validate] FATAL (prepare will refuse):")
            for line in self.fatal:
                print(f"   X {line}")
        print("")
        if self.fatal:
            print("[validate] result: FATAL. Fix the issues above and re-run.")
        elif self.autofix:
            print("[validate] result: OK with auto-fix. Run `python prepare.py --task-root ...`")
        else:
            print("[validate] result: OK. Run `python prepare.py --task-root ...`")


def _scan_files(root: Path) -> dict[str, list[Path]]:
    """Return {immediate_subdir_name: [image_paths]} for the top level of root.

    Skips files at the root level (they are not classifiable into classes).
    """
    out: dict[str, list[Path]] = defaultdict(list)
    for child in sorted(root.iterdir()):
        if child.is_dir():
            for p in sorted(child.rglob("*")):
                if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                    out[child.name].append(p)
    return out


def _detect_layout(root: Path, report: ValidationReport) -> str:
    if not root.is_dir():
        report.fatal.append(f"task_root is not a directory: {root}")
        return "invalid"

    top_names = {p.name.lower() for p in root.iterdir() if p.is_dir()}
    if top_names and top_names <= set(SPLIT_DIR_ALIASES.keys()):
        report.pre_split_detected = True
        return "pre_split"

    if (root / "images").is_dir() and any(
        (root / f"labels.{ext}").is_file() for ext in ("csv", "tsv", "json", "jsonl")
    ):
        return "manifest"

    if any(p.is_dir() for p in root.iterdir()):
        return "imagefolder"

    report.fatal.append(
        f"No class sub-folders under {root}. "
        f"Expected `<task_root>/<class_name>/*.png` (see input-spec.md §2)."
    )
    return "invalid"


def _check_class_names(files: dict[str, list[Path]], report: ValidationReport) -> None:
    by_lower: dict[str, list[str]] = defaultdict(list)
    for name in files:
        by_lower[name.lower()].append(name)
    for lower, group in by_lower.items():
        if len(group) > 1:
            report.autofix.append(
                f"R1.1 merge case-variant class names {sorted(group)!r} -> "
                f"{sorted(group, key=lambda s: s.lower())[0]!r}"
            )
        for n in group:
            if n != n.strip():
                report.autofix.append(f"R1.2 strip whitespace in class name {n!r}")
            try:
                n.encode("ascii")
            except UnicodeEncodeError:
                report.autofix.append(f"R1.3 non-ascii class name {n!r}, will hash-rename")


def _check_images(files: dict[str, list[Path]], report: ValidationReport, sample_cap: int = 50) -> None:
    seen_hashes: dict[str, tuple[str, Path]] = {}
    label_conflicts: list[tuple[str, str, str]] = []

    total = 0
    corrupt = 0
    for cls, paths in files.items():
        kept = 0
        for path in paths[:sample_cap]:
            total += 1
            try:
                with Image.open(path) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError):
                corrupt += 1
                report.autofix.append(f"R2.1 drop corrupt image {path}")
                continue
            try:
                digest = hashlib.sha1(path.read_bytes()).hexdigest()
            except OSError:
                corrupt += 1
                continue
            if digest in seen_hashes and seen_hashes[digest][0] != cls:
                prev_cls, prev_path = seen_hashes[digest]
                label_conflicts.append((digest, prev_cls, cls))
            else:
                seen_hashes.setdefault(digest, (cls, path))
            kept += 1
        report.per_class_count[cls] = len(paths)
        if len(paths) <= 1:
            report.autofix.append(
                f"R3.2 class {cls!r} has only {len(paths)} sample(s); will be dropped"
            )

    if label_conflicts:
        # Sample-cap means this is a lower bound; we still raise fatal proportionally.
        rate = len(label_conflicts) / max(total, 1)
        msg = f"R2.3 label conflicts detected (same bytes in 2 classes), example: {label_conflicts[:3]}"
        if rate > 0.01:
            report.fatal.append(msg)
        else:
            report.warnings.append(msg)

    if total > 0 and corrupt / total > 0.05:
        report.fatal.append(
            f"R2.1 too many corrupt images: {corrupt}/{total} = {corrupt / total:.1%} "
            f"(threshold 5%). Aborting."
        )


def validate(task_root: Path) -> ValidationReport:
    report = ValidationReport(task_root=task_root.resolve())
    report.layout = _detect_layout(task_root, report)
    if report.layout == "invalid":
        return report
    if report.layout == "pre_split":
        # Validate inside the train/ slice (val/test will be inherited as-is).
        train_dir = next(
            (task_root / p.name for p in task_root.iterdir()
             if p.is_dir() and SPLIT_DIR_ALIASES.get(p.name.lower()) == "train"),
            None,
        )
        if train_dir is None:
            report.fatal.append("pre_split layout missing a `train/` directory.")
            return report
        files = _scan_files(train_dir)
    elif report.layout == "manifest":
        # Light-touch check: trust the csv parser in prepare.py to do the real work.
        report.warnings.append("manifest layout detected; deep validation happens in prepare.py")
        return report
    else:
        files = _scan_files(task_root)

    if not files:
        report.fatal.append(f"No class folders with images found under {task_root}.")
        return report

    report.classes = sorted(files.keys())
    _check_class_names(files, report)
    _check_images(files, report)

    if not report.classes:
        report.fatal.append("All classes were dropped after auto-fix preview; nothing to train on.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate raw input layout (read-only).")
    parser.add_argument("--task-root", required=True, type=Path)
    args = parser.parse_args()

    report = validate(args.task_root)
    report.print_human()
    return 1 if report.is_fatal else 0


if __name__ == "__main__":
    sys.exit(main())
