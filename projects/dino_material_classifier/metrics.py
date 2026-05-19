"""Metrics helpers for the two-head classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def _safe_metric(values: Sequence[int], default: float = 0.0) -> float:
    if len(values) == 0:
        return default
    return float(values)


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]) -> dict[str, object]:
    """Compute the standard metrics for one prediction head."""

    if len(y_true) == 0:
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "confusion_matrix": pd.DataFrame(matrix, index=labels, columns=labels),
            "report": {},
        }

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=list(labels),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": pd.DataFrame(matrix, index=labels, columns=labels),
        "report": report,
    }


def save_confusion_matrix(matrix: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path)


def prediction_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)

