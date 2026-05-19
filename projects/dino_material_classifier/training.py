"""Training and evaluation loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .labels import LabelSchema
from .metrics import classification_metrics, prediction_frame, save_confusion_matrix


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["image"] = batch["image"].to(device, non_blocking=True)
    moved["material"] = batch["material"].to(device, non_blocking=True)
    moved["process"] = batch["process"].to(device, non_blocking=True)
    return moved


def compute_loss(
    output: Any,
    batch: dict[str, Any],
    material_criterion: nn.Module,
    process_criterion: nn.Module,
    material_loss_weight: float = 1.0,
    process_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    material_loss = material_criterion(output.material_logits, batch["material"])
    process_loss = process_criterion(output.process_logits, batch["process"])
    total_loss = material_loss_weight * material_loss + process_loss_weight * process_loss
    return total_loss, {
        "loss": float(total_loss.detach().cpu()),
        "material_loss": float(material_loss.detach().cpu()),
        "process_loss": float(process_loss.detach().cpu()),
    }


def _accumulate_average(total: dict[str, float], values: dict[str, float], batch_size: int) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + value * batch_size


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    material_criterion: nn.Module,
    process_criterion: nn.Module,
    device: torch.device,
    material_loss_weight: float = 1.0,
    process_loss_weight: float = 1.0,
    max_grad_norm: float | None = 1.0,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    total_samples = 0
    material_correct = 0
    process_correct = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["image"])
        loss, loss_parts = compute_loss(
            output,
            batch,
            material_criterion,
            process_criterion,
            material_loss_weight=material_loss_weight,
            process_loss_weight=process_loss_weight,
        )
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = int(batch["image"].size(0))
        _accumulate_average(totals, loss_parts, batch_size)
        material_correct += (output.material_logits.argmax(dim=1) == batch["material"]).sum().item()
        process_correct += (output.process_logits.argmax(dim=1) == batch["process"]).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        return {"loss": 0.0, "material_loss": 0.0, "process_loss": 0.0, "material_acc": 0.0, "process_acc": 0.0}

    metrics = {key: value / total_samples for key, value in totals.items()}
    metrics["material_acc"] = material_correct / total_samples
    metrics["process_acc"] = process_correct / total_samples
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    material_criterion: nn.Module,
    process_criterion: nn.Module,
    device: torch.device,
    label_schema: LabelSchema,
    output_dir: str | Path | None = None,
    split_name: str = "val",
    material_loss_weight: float = 1.0,
    process_loss_weight: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {}
    total_samples = 0
    material_true: list[int] = []
    material_pred: list[int] = []
    process_true: list[int] = []
    process_pred: list[int] = []
    rows: list[dict[str, object]] = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        output = model(batch["image"])
        _, loss_parts = compute_loss(
            output,
            batch,
            material_criterion,
            process_criterion,
            material_loss_weight=material_loss_weight,
            process_loss_weight=process_loss_weight,
        )
        batch_size = int(batch["image"].size(0))
        _accumulate_average(totals, loss_parts, batch_size)
        total_samples += batch_size

        material_batch_pred = output.material_logits.argmax(dim=1).detach().cpu()
        process_batch_pred = output.process_logits.argmax(dim=1).detach().cpu()
        material_batch_true = batch["material"].detach().cpu()
        process_batch_true = batch["process"].detach().cpu()
        material_batch_conf = output.material_probs.max(dim=1).values.detach().cpu()
        process_batch_conf = output.process_probs.max(dim=1).values.detach().cpu()

        material_true.extend(material_batch_true.tolist())
        material_pred.extend(material_batch_pred.tolist())
        process_true.extend(process_batch_true.tolist())
        process_pred.extend(process_batch_pred.tolist())

        for idx in range(batch_size):
            material_target = int(material_batch_true[idx])
            material_prediction = int(material_batch_pred[idx])
            process_target = int(process_batch_true[idx])
            process_prediction = int(process_batch_pred[idx])
            rows.append(
                {
                    "sample_id": batch["sample_id"][idx],
                    "group_id": batch["group_id"][idx],
                    "image_path": batch["image_path"][idx],
                    "material_true": label_schema.material_name(material_target),
                    "material_pred": label_schema.material_name(material_prediction),
                    "material_conf": float(material_batch_conf[idx]),
                    "process_true": label_schema.process_name(process_target),
                    "process_pred": label_schema.process_name(process_prediction),
                    "process_conf": float(process_batch_conf[idx]),
                    "material_correct": material_target == material_prediction,
                    "process_correct": process_target == process_prediction,
                }
            )

    material_metrics = classification_metrics(material_true, material_pred, label_schema.material_labels)
    process_metrics = classification_metrics(process_true, process_pred, label_schema.process_labels)
    avg_losses = {key: (value / total_samples if total_samples else 0.0) for key, value in totals.items()}
    result: dict[str, Any] = {
        **avg_losses,
        "material": material_metrics,
        "process": process_metrics,
        "predictions": prediction_frame(rows),
        "num_samples": total_samples,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(material_metrics["confusion_matrix"], output_dir / f"{split_name}_material_confusion.csv")
        save_confusion_matrix(process_metrics["confusion_matrix"], output_dir / f"{split_name}_process_confusion.csv")
        predictions = result["predictions"]
        predictions.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)
        error_rows = predictions[(~predictions["material_correct"]) | (~predictions["process_correct"])]
        error_rows.to_csv(output_dir / f"{split_name}_errors.csv", index=False)

    return result


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Extract scalar metrics for JSON/CSV logs."""

    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, (int, float)):
            flat[name] = float(value)
        elif isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=f"{name}."))
    return flat


def save_history(history: list[dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)

