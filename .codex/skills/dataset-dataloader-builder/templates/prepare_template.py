"""prepare_template.py — 一次性把原始输入归一化为 prepared/ 目录.

把这份模板拷贝到目标项目并改 `# TODO:` 标记后即可独立运行：

    python prepare.py --task-root path/to/inputs/material \
                      --prepared-root outputs/material/prepared

下游 train/infer 只需要 `--prepared-root`，**不**再接触 `--task-root`。

实现要点（与 references/ 的对应关系）
-----------------------------------
- 目录布局 / 命名 / json 文件格式 ............ prepared-layout.md
- auto-fix 顺序与触发条件 ................... auto-fix-rules.md
- 按类分层、可复现切分 ..................... split-policy.md
- 输出 batch 字段（本脚本不直接产 batch，但 manifest 里固化了 image_size / normalize）
  .......................................... output-spec.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_DIR_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}

# Fixed downstream normalization parameters (DINOv3 / torchvision default).
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------

@dataclass
class CollectedSample:
    """An image after we know its final (class_name, target_filename)."""
    src: Path
    cls: str           # post-fix class name (after R1 normalization)
    sha1: str
    split: str | None = None  # set when pre_split_detected, else assigned later


@dataclass
class PrepareReport:
    renamed: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    merged_classes: list[dict] = field(default_factory=list)
    label_conflicts: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    applied: bool = True

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "renamed": self.renamed,
            "dropped": self.dropped,
            "merged_classes": self.merged_classes,
            "label_conflicts": self.label_conflicts,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Discovery: turn arbitrary input into a flat list[CollectedSample]
# ---------------------------------------------------------------------------

def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXTS


def _normalize_class_name(name: str, report: PrepareReport) -> str:
    stripped = name.strip()
    if stripped != name:
        report.warnings.append(f"R1.2 stripped whitespace from class name {name!r}")
    try:
        stripped.encode("ascii")
        return stripped
    except UnicodeEncodeError:
        digest = hashlib.sha1(stripped.encode("utf-8")).hexdigest()[:8]
        new = f"class_{digest}"
        report.warnings.append(f"R1.3 non-ascii class name {stripped!r} -> {new!r}")
        return new


def _merge_case_variants(samples: list[CollectedSample], report: PrepareReport) -> None:
    by_lower: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        by_lower[s.cls.lower()].append(s.cls)
    for lower, group in by_lower.items():
        unique = sorted(set(group))
        if len(unique) > 1:
            canonical = sorted(unique, key=lambda x: (x.lower(), x))[0]
            report.merged_classes.append({"into": canonical, "from": [g for g in unique if g != canonical]})
            for s in samples:
                if s.cls.lower() == lower:
                    s.cls = canonical


def _discover_imagefolder(
    task_root: Path, report: PrepareReport, pre_split: bool = False
) -> list[CollectedSample]:
    out: list[CollectedSample] = []
    if pre_split:
        for top in sorted(task_root.iterdir()):
            if not top.is_dir():
                continue
            split_name = SPLIT_DIR_ALIASES.get(top.name.lower())
            if split_name is None:
                report.warnings.append(f"Ignoring top-level directory not in split aliases: {top.name}")
                continue
            for cls_dir in sorted(top.iterdir()):
                if not cls_dir.is_dir():
                    continue
                cls = _normalize_class_name(cls_dir.name, report)
                for img in sorted(cls_dir.rglob("*")):
                    if _is_image(img):
                        try:
                            digest = _sha1_bytes(img.read_bytes())
                        except OSError:
                            report.dropped.append({"path": str(img), "reason": "io_error"})
                            continue
                        out.append(CollectedSample(src=img, cls=cls, sha1=digest, split=split_name))
    else:
        for cls_dir in sorted(task_root.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = _normalize_class_name(cls_dir.name, report)
            for img in sorted(cls_dir.rglob("*")):
                if _is_image(img):
                    try:
                        digest = _sha1_bytes(img.read_bytes())
                    except OSError:
                        report.dropped.append({"path": str(img), "reason": "io_error"})
                        continue
                    out.append(CollectedSample(src=img, cls=cls, sha1=digest))
    _merge_case_variants(out, report)
    return out


def _discover_manifest(task_root: Path, report: PrepareReport) -> list[CollectedSample]:
    images_dir = task_root / "images"
    if not images_dir.is_dir():
        raise SystemExit(f"manifest layout requires {images_dir} to exist")
    labels: dict[str, str] = {}
    for ext, parser in (("csv", _parse_csv), ("tsv", _parse_tsv), ("json", _parse_json), ("jsonl", _parse_jsonl)):
        f = task_root / f"labels.{ext}"
        if f.is_file():
            labels = parser(f)
            break
    if not labels:
        raise SystemExit(f"no usable labels.{{csv,tsv,json,jsonl}} under {task_root}")
    out: list[CollectedSample] = []
    seen_files = set()
    for fname, raw_cls in labels.items():
        img = images_dir / fname
        if not _is_image(img):
            report.dropped.append({"path": str(img), "reason": "manifest_orphan"})
            continue
        seen_files.add(img.resolve())
        cls = _normalize_class_name(str(raw_cls), report)
        try:
            digest = _sha1_bytes(img.read_bytes())
        except OSError:
            report.dropped.append({"path": str(img), "reason": "io_error"})
            continue
        out.append(CollectedSample(src=img, cls=cls, sha1=digest))
    # Stray files under images/ not referenced by manifest
    for img in images_dir.rglob("*"):
        if _is_image(img) and img.resolve() not in seen_files:
            report.warnings.append(f"R5.3 image under images/ not in manifest, ignored: {img}")
    _merge_case_variants(out, report)
    return out


def _parse_csv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if "filename" in row and "label" in row:
                out[row["filename"].strip()] = row["label"].strip()
    return out


def _parse_tsv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if "filename" in row and "label" in row:
                out[row["filename"].strip()] = row["label"].strip()
    return out


def _parse_json(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def _parse_jsonl(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[str(d["filename"])] = str(d["label"])
    return out


# ---------------------------------------------------------------------------
# Cleaning: corrupt / RGBA / EXIF / hash conflicts / single-sample classes
# ---------------------------------------------------------------------------

def _decode_and_save(
    src: Path,
    dst: Path,
    report: PrepareReport,
    linkonly: bool,
    decode_stats: dict,
) -> bool:
    """Return True on success, False if image is unusable.

    ``decode_stats`` accumulates counts for systematic per-file events
    (mode conversion, EXIF rotation) so that we can emit ONE summary warning
    instead of N per-file warnings. The first 3 examples are kept for
    debugging.
    """
    try:
        with Image.open(src) as im:
            im.load()
            exif_rotated = im.getexif().get(0x0112, 1) != 1
            im = ImageOps.exif_transpose(im)
            if exif_rotated:
                decode_stats["exif_rotated"] = decode_stats.get("exif_rotated", 0) + 1
                ex = decode_stats.setdefault("exif_rotated_examples", [])
                if len(ex) < 3:
                    ex.append(str(src))
            if im.mode != "RGB":
                key = f"mode_{im.mode}"
                decode_stats[key] = decode_stats.get(key, 0) + 1
                ex = decode_stats.setdefault(f"{key}_examples", [])
                if len(ex) < 3:
                    ex.append(str(src))
                im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < 8:
                report.dropped.append({"path": str(src), "reason": "too_small"})
                return False
            dst.parent.mkdir(parents=True, exist_ok=True)
            if linkonly and im.mode == "RGB" and not exif_rotated:
                # Safe to symlink only when no re-encoding was needed; otherwise
                # the on-disk bytes would not match what we just verified.
                try:
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    os.symlink(os.path.relpath(src, dst.parent), dst)
                    return True
                except OSError:
                    pass  # fall through to physical write
            im.save(dst, format="PNG")
            return True
    except (UnidentifiedImageError, OSError) as exc:
        report.dropped.append({"path": str(src), "reason": f"corrupt_image:{type(exc).__name__}"})
        return False


def _summarize_decode_stats(stats: dict, total: int, report: PrepareReport) -> None:
    """Turn per-event counters into a few human-friendly warning lines."""
    for key, count in sorted(stats.items()):
        if not key.startswith("mode_") or key.endswith("_examples"):
            continue
        examples = stats.get(f"{key}_examples", [])
        report.warnings.append(
            f"R2.4 converted {count}/{total} image(s) from {key.removeprefix('mode_')} to RGB "
            f"(examples: {examples})"
        )
    rotated = stats.get("exif_rotated", 0)
    if rotated:
        examples = stats.get("exif_rotated_examples", [])
        report.warnings.append(
            f"R2.5 applied EXIF rotation to {rotated}/{total} image(s) (examples: {examples})"
        )


def _resolve_hash_conflicts(samples: list[CollectedSample], report: PrepareReport) -> list[CollectedSample]:
    by_hash: dict[str, list[CollectedSample]] = defaultdict(list)
    for s in samples:
        by_hash[s.sha1].append(s)
    survivors: list[CollectedSample] = []
    n_conflicts = 0
    for digest, group in by_hash.items():
        classes = {s.cls for s in group}
        if len(classes) > 1:
            n_conflicts += 1
            report.label_conflicts.append({"sha1": digest, "appears_in": sorted(classes)})
            # drop all copies (R2.3) — better lose data than mislabel
            for s in group:
                report.dropped.append({"path": str(s.src), "reason": "label_conflict"})
        else:
            survivors.append(group[0])  # dedupe within-class
            if len(group) > 1:
                report.warnings.append(
                    f"R1.5 {len(group)} duplicate copies of {group[0].src.name}, kept one"
                )
    if n_conflicts and n_conflicts / max(len(by_hash), 1) > 0.01:
        raise SystemExit(
            f"label conflicts {n_conflicts}/{len(by_hash)} > 1% — refusing to prepare."
        )
    return survivors


def _drop_tiny_classes(
    samples: list[CollectedSample], report: PrepareReport, min_per_class: int = 2
) -> list[CollectedSample]:
    by_cls: dict[str, list[CollectedSample]] = defaultdict(list)
    for s in samples:
        by_cls[s.cls].append(s)
    survivors: list[CollectedSample] = []
    for cls, group in by_cls.items():
        if len(group) < min_per_class:
            for s in group:
                report.dropped.append({"path": str(s.src), "reason": "class_too_small"})
        else:
            survivors.extend(group)
    return survivors


# ---------------------------------------------------------------------------
# Split (mirror of references/split-policy.md §2)
# ---------------------------------------------------------------------------

def _quota(n: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    if fraction >= 1:
        return n
    return int(round(n * fraction))


def _split_stratified(
    by_class: dict[str, list[str]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    splits = {"train": [], "val": [], "test": []}
    g = torch.Generator().manual_seed(seed)
    for cls in sorted(by_class):
        items = sorted(by_class[cls])
        n = len(items)
        perm = torch.randperm(n, generator=g).tolist()
        shuffled = [items[i] for i in perm]

        n_test = _quota(n, test_fraction)
        n_val = _quota(n, val_fraction)
        n_train = n - n_val - n_test
        if n >= 2:
            if n_val == 0:
                n_val = 1
            if n_train == 0:
                n_train = 1
            while n_train + n_val + n_test > n:
                if n_test > 0:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    n_train -= 1

        splits["test"].extend(shuffled[:n_test])
        splits["val"].extend(shuffled[n_test:n_test + n_val])
        splits["train"].extend(shuffled[n_test + n_val:])
    return splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(
    task_root: Path,
    prepared_root: Path,
    task_name: str | None = None,
    image_size: int = 224,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
    mode: str = "fix",
    linkonly: bool = False,
    log=print,
) -> dict:
    task_root = Path(task_root).resolve()
    prepared_root = Path(prepared_root).resolve()
    task_name = task_name or task_root.name
    report = PrepareReport(applied=(mode == "fix"))

    if not task_root.is_dir():
        raise SystemExit(f"task_root not a directory: {task_root}")
    if not (0 <= val_fraction + test_fraction < 1):
        raise SystemExit(f"val_fraction + test_fraction must be in [0, 1), got {val_fraction + test_fraction}")

    log(f"[prep] task_root      = {task_root}")
    log(f"[prep] prepared_root  = {prepared_root}")
    log(f"[prep] mode           = {mode}")

    # ---- 1. Discover ------------------------------------------------------
    top_names = {p.name.lower() for p in task_root.iterdir() if p.is_dir()}
    if top_names and top_names <= set(SPLIT_DIR_ALIASES.keys()):
        samples = _discover_imagefolder(task_root, report, pre_split=True)
        pre_split = True
    elif (task_root / "images").is_dir() and any(
        (task_root / f"labels.{ext}").is_file() for ext in ("csv", "tsv", "json", "jsonl")
    ):
        samples = _discover_manifest(task_root, report)
        pre_split = False
    else:
        samples = _discover_imagefolder(task_root, report, pre_split=False)
        pre_split = False
    log(f"[prep] discovered     = {len(samples)} sample(s) from {len({s.cls for s in samples})} class(es)")

    # ---- 2. Resolve hash conflicts (R2.3) ---------------------------------
    samples = _resolve_hash_conflicts(samples, report)

    # ---- 3. Drop tiny classes (R3.2) --------------------------------------
    samples = _drop_tiny_classes(samples, report, min_per_class=2)
    if not samples:
        raise SystemExit("Nothing left after cleaning. Check conversion_report.json.")

    # ---- 4. Assign final filenames & copy/encode --------------------------
    images_root = prepared_root / "images"
    if images_root.exists():
        shutil.rmtree(images_root)
    images_root.mkdir(parents=True, exist_ok=True)

    samples.sort(key=lambda s: (s.cls, s.src.name))
    written_per_class: dict[str, list[str]] = defaultdict(list)
    written_paths: list[CollectedSample] = []
    decode_stats: dict = {}
    n_renamed = 0
    rename_examples: list[dict] = []
    for idx, s in enumerate(samples):
        dst = images_root / s.cls / f"img_{idx:05d}.png"
        if not _decode_and_save(s.src, dst, report, linkonly=linkonly, decode_stats=decode_stats):
            continue
        rel = f"{s.cls}/img_{idx:05d}.png"
        written_per_class[s.cls].append(rel)
        written_paths.append(s)
        if s.src.name != f"img_{idx:05d}.png":
            n_renamed += 1
            if len(rename_examples) < 5:
                rename_examples.append({"from": str(s.src), "to": rel})

    _summarize_decode_stats(decode_stats, total=len(samples), report=report)
    report.renamed = rename_examples  # keep file tiny; full count goes to manifest

    classes = sorted(written_per_class)
    if not classes:
        raise SystemExit("All samples were dropped during decoding. Aborting.")

    # ---- 5. Split ---------------------------------------------------------
    if pre_split:
        # Re-derive each written rel-path's original split by walking
        # written_paths in write order and advancing a per-class cursor through
        # written_per_class[cls] (which was appended in the same order).
        splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        cursor: dict[str, int] = defaultdict(int)
        for s in written_paths:
            rel = written_per_class[s.cls][cursor[s.cls]]
            cursor[s.cls] += 1
            splits[s.split or "train"].append(rel)
        for k in splits:
            splits[k] = sorted(splits[k])

        splits_meta = {
            "schema_version": 1,
            "seed": seed,
            "val_fraction": 0.0,
            "test_fraction": 0.0,
            "pre_split_detected": True,
            **splits,
        }
        if not splits["val"]:
            # R4.2: carve val from existing train slice while keeping test intact.
            train_by_class: dict[str, list[str]] = defaultdict(list)
            for rel in splits["train"]:
                cls = rel.split("/", 1)[0]
                train_by_class[cls].append(rel)
            carved = _split_stratified(
                train_by_class, val_fraction=val_fraction, test_fraction=0.0, seed=seed,
            )
            splits_meta["train"] = sorted(carved["train"])
            splits_meta["val"] = sorted(carved["val"])
            splits_meta["val_fraction"] = val_fraction
            splits_meta["pre_split_detected"] = "partial"
    else:
        carved = _split_stratified(written_per_class, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)
        for k in carved:
            carved[k] = sorted(carved[k])
        splits_meta = {
            "schema_version": 1,
            "seed": seed,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "pre_split_detected": False,
            **carved,
        }

    # ---- 6. Write classes / splits / manifest / report --------------------
    classes_payload = {
        "classes": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "idx_to_class": {str(i): c for i, c in enumerate(classes)},
    }
    _write_json(prepared_root / "classes.json", classes_payload)
    _write_json(prepared_root / "splits.json", splits_meta)

    per_class = {c: len(written_per_class[c]) for c in classes}
    per_split = {k: len(splits_meta[k]) for k in ("train", "val", "test")}

    manifest = {
        "schema_version": 1,
        "task_name": task_name,
        "task_type": "image_classification_single_label",
        "modality": "rgb",
        "source": {
            "root": str(task_root),
            "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "image": {
            "expected_channels": 3,
            "target_size": [image_size, image_size],
            "normalize": {"mean": list(NORMALIZE_MEAN), "std": list(NORMALIZE_STD)},
        },
        "counts": {
            "total": sum(per_class.values()),
            "per_class": per_class,
            "per_split": per_split,
        },
        "auto_fix": {
            "applied": report.applied,
            "n_renamed": n_renamed,
            "n_dropped": len(report.dropped),
            "n_merged_classes": len(report.merged_classes),
            "n_corrupt_skipped": sum(1 for d in report.dropped if "corrupt_image" in d.get("reason", "")),
        },
    }
    _write_json(prepared_root / "manifest.json", manifest)
    _write_json(prepared_root / "conversion_report.json", report.to_dict())

    # ---- 7. Invariant check (output-spec / prepared-layout §2.3) ----------
    _check_invariants(prepared_root, splits_meta, classes)

    log("")
    log(f"[prep] classes        = {classes}")
    log(f"[prep] per_class      = {per_class}")
    log(f"[prep] per_split      = {per_split}")
    log(f"[prep] dropped        = {len(report.dropped)}")
    log(f"[prep] merged_classes = {len(report.merged_classes)}")
    log(f"[prep] warnings       = {len(report.warnings)}")
    log("[prep] DONE")
    return manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _check_invariants(prepared_root: Path, splits_meta: dict, classes: list[str]) -> None:
    train = set(splits_meta["train"])
    val = set(splits_meta["val"])
    test = set(splits_meta["test"])
    if train & val or train & test or val & test:
        raise SystemExit("splits overlap — algorithmic bug, aborting")
    on_disk = set()
    for cls in classes:
        for p in (prepared_root / "images" / cls).glob("*.png"):
            on_disk.add(f"{cls}/{p.name}")
    union = train | val | test
    missing_in_splits = on_disk - union
    missing_on_disk = union - on_disk
    if missing_in_splits:
        raise SystemExit(f"{len(missing_in_splits)} file(s) on disk but not in any split.")
    if missing_on_disk:
        raise SystemExit(f"{len(missing_on_disk)} file(s) in splits.json but missing on disk.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw input -> prepared/ standard layout.")
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("fix", "strict"), default="fix")
    parser.add_argument("--linkonly", action="store_true",
                        help="Symlink instead of copy when no re-encode is needed.")
    args = parser.parse_args()
    prepare(
        task_root=args.task_root,
        prepared_root=args.prepared_root,
        task_name=args.task_name,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        mode=args.mode,
        linkonly=args.linkonly,
    )


if __name__ == "__main__":
    main()
