"""Dataset and dataloader utilities.

The input protocol is a CSV manifest with at least:
image_path, material, process

Recommended optional columns:
sample_id, group_id, split, mask_path, bbox

The bbox column can be written as "x1,y1,x2,y2". If split is missing,
the code assigns groups deterministically using a hash of group_id or
sample_id so that the same physical part never leaks across splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .labels import LabelSchema


NORMALIZATION_PRESETS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "dinov3_lvd": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "dinov3_sat": ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143)),
}


@dataclass
class ImageRecord:
    """A single training sample and its metadata."""

    image_path: Path
    material: str
    process: str
    sample_id: str
    group_id: str
    split: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    mask_path: Path | None = None
    extra: dict[str, object] = field(default_factory=dict)


def _resolve_path(value: object, root: str | Path | None = None) -> Path:
    path = Path(str(value))
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    return path


def _parse_bbox(value: object | None) -> tuple[int, int, int, int] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (tuple, list)) and len(value) == 4:
        return tuple(int(float(v)) for v in value)
    text = str(value).strip()
    if not text:
        return None
    parts = [piece for piece in re.split(r"[,;\s]+", text) if piece]
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox value: {value!r}")
    return tuple(int(float(piece)) for piece in parts)


def _default_group_id(row: pd.Series) -> str:
    for column in ("group_id", "part_id", "capture_id", "sample_id"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return str(row.get("image_path", "")).strip()


def _hash_split(key: str, ratios: tuple[float, float, float], seed: int) -> str:
    train_ratio, val_ratio, test_ratio = ratios
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    if score < train_ratio:
        return "train"
    if score < train_ratio + val_ratio:
        return "val"
    return "test"


def load_records(manifest_path: str | Path, image_root: str | Path | None = None) -> list[ImageRecord]:
    """Load a manifest CSV into strongly typed records."""

    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)
    df.columns = [column.strip().lower() for column in df.columns]

    required = {"image_path", "material", "process"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    records: list[ImageRecord] = []
    for row in df.to_dict(orient="records"):
        image_path = _resolve_path(row["image_path"], root=image_root)
        mask_path = row.get("mask_path")
        bbox = row.get("bbox")

        if bbox is None:
            bbox = row.get("roi_bbox")

        if bbox is None and all(column in row for column in ("x1", "y1", "x2", "y2")):
            bbox = (row["x1"], row["y1"], row["x2"], row["y2"])

        extra = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "image_path",
                "material",
                "process",
                "sample_id",
                "group_id",
                "split",
                "mask_path",
                "bbox",
                "roi_bbox",
                "x1",
                "y1",
                "x2",
                "y2",
            }
        }

        record = ImageRecord(
            image_path=image_path,
            material=str(row["material"]).strip(),
            process=str(row["process"]).strip(),
            sample_id=str(row.get("sample_id", image_path.stem)).strip(),
            group_id=_default_group_id(pd.Series(row)),
            split=str(row["split"]).strip() if pd.notna(row.get("split")) else None,
            bbox=_parse_bbox(bbox),
            mask_path=_resolve_path(mask_path, root=image_root) if pd.notna(mask_path) else None,
            extra=extra,
        )
        records.append(record)

    return records


def build_label_schema(
    records: Iterable[ImageRecord],
    material_labels: Sequence[str] | None = None,
    process_labels: Sequence[str] | None = None,
) -> LabelSchema:
    return LabelSchema.from_records(records, material_labels=material_labels, process_labels=process_labels)


def assign_splits(
    records: Sequence[ImageRecord],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> list[ImageRecord]:
    """Fill missing split values with a deterministic group-based split."""

    updated: list[ImageRecord] = []
    for record in records:
        if record.split:
            updated.append(record)
            continue
        split = _hash_split(record.group_id or record.sample_id or record.image_path.stem, ratios, seed)
        updated.append(
            ImageRecord(
                image_path=record.image_path,
                material=record.material,
                process=record.process,
                sample_id=record.sample_id,
                group_id=record.group_id,
                split=split,
                bbox=record.bbox,
                mask_path=record.mask_path,
                extra=dict(record.extra),
            )
        )
    return updated


def _load_mask_bbox(mask_path: Path) -> tuple[int, int, int, int] | None:
    if not mask_path.exists():
        return None
    mask = Image.open(mask_path).convert("L")
    return mask.getbbox()


def crop_roi(image: Image.Image, record: ImageRecord, roi_mode: str = "bbox") -> Image.Image:
    """Crop the region of interest.

    roi_mode:
      - full: return the full image
      - bbox: use the bbox column when available, else use the mask bbox
      - mask: use the mask bbox when available, else fallback to bbox
    """

    if roi_mode == "full":
        return image

    if roi_mode == "mask" and record.mask_path is not None:
        mask_bbox = _load_mask_bbox(record.mask_path)
        if mask_bbox is not None:
            return image.crop(mask_bbox)

    if record.bbox is not None:
        return image.crop(record.bbox)

    if record.mask_path is not None:
        mask_bbox = _load_mask_bbox(record.mask_path)
        if mask_bbox is not None:
            return image.crop(mask_bbox)

    return image


def build_transforms(
    image_size: int = 224,
    train: bool = True,
    normalization: str = "imagenet",
) -> T.Compose:
    """Build light but stable image transforms for industrial parts."""

    if normalization not in NORMALIZATION_PRESETS:
        raise ValueError(f"Unknown normalization preset: {normalization!r}")

    mean, std = NORMALIZATION_PRESETS[normalization]
    ops: list[object] = []
    if train:
        ops.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.02, hue=0.01),
            ]
        )
    ops.extend(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return T.Compose(ops)


class MaterialProcessDataset(Dataset):
    """Dataset that returns an image tensor and two label targets."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        label_schema: LabelSchema,
        image_size: int = 224,
        roi_mode: str = "bbox",
        train: bool = True,
        normalization: str = "imagenet",
        split: str | None = None,
    ) -> None:
        if split is not None:
            records = [record for record in records if record.split == split]
        self.records = list(records)
        self.label_schema = label_schema
        self.image_size = image_size
        self.roi_mode = roi_mode
        self.train = train
        self.normalization = normalization
        self.transform = build_transforms(image_size=image_size, train=train, normalization=normalization)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, record: ImageRecord) -> Image.Image:
        image = Image.open(record.image_path).convert("RGB")
        return crop_roi(image, record, roi_mode=self.roi_mode)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = self._load_image(record)
        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "material": torch.tensor(self.label_schema.material_index(record.material), dtype=torch.long),
            "process": torch.tensor(self.label_schema.process_index(record.process), dtype=torch.long),
            "sample_id": record.sample_id,
            "group_id": record.group_id,
            "image_path": str(record.image_path),
            "material_name": record.material,
            "process_name": record.process,
            "split": record.split or "",
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_dataloaders(
    manifest_path: str | Path,
    image_root: str | Path | None = None,
    label_schema: LabelSchema | None = None,
    image_size: int = 224,
    roi_mode: str = "bbox",
    normalization: str = "imagenet",
    batch_size: int = 8,
    num_workers: int = 0,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, LabelSchema, list[ImageRecord]]:
    """Create train/val/test dataloaders from a manifest."""

    records = load_records(manifest_path, image_root=image_root)
    records = assign_splits(records, ratios=split_ratios, seed=seed)
    schema = label_schema or build_label_schema(records)

    train_dataset = MaterialProcessDataset(
        records,
        label_schema=schema,
        image_size=image_size,
        roi_mode=roi_mode,
        train=True,
        normalization=normalization,
        split="train",
    )
    val_dataset = MaterialProcessDataset(
        records,
        label_schema=schema,
        image_size=image_size,
        roi_mode=roi_mode,
        train=False,
        normalization=normalization,
        split="val",
    )
    test_dataset = MaterialProcessDataset(
        records,
        label_schema=schema,
        image_size=image_size,
        roi_mode=roi_mode,
        train=False,
        normalization=normalization,
        split="test",
    )

    pin_memory = torch.cuda.is_available()
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = build_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader, schema, records

