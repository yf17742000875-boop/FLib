"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .labels import LabelSchema


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    label_schema: LabelSchema,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "label_schema": label_schema.to_dict(),
        "config": dict(config),
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    if model is not None:
        model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def load_label_schema_from_checkpoint(path: str | Path) -> LabelSchema:
    payload = torch.load(Path(path), map_location="cpu")
    return LabelSchema(
        material_labels=list(payload["label_schema"]["material_labels"]),
        process_labels=list(payload["label_schema"]["process_labels"]),
    )

