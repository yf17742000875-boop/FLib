"""Label schema utilities.

The classifier predicts two independent label spaces:
material and process. Keeping them separate avoids the class
explosion that happens when every material-process combination is
treated as one label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Sequence


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


@dataclass
class LabelSchema:
    """Bidirectional mapping for the two label spaces."""

    material_labels: list[str] = field(default_factory=list)
    process_labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.material_labels = _ordered_unique(self.material_labels)
        self.process_labels = _ordered_unique(self.process_labels)
        self.material_to_idx = {label: idx for idx, label in enumerate(self.material_labels)}
        self.process_to_idx = {label: idx for idx, label in enumerate(self.process_labels)}

    @property
    def num_materials(self) -> int:
        return len(self.material_labels)

    @property
    def num_processes(self) -> int:
        return len(self.process_labels)

    def material_index(self, label: str) -> int:
        try:
            return self.material_to_idx[str(label).strip()]
        except KeyError as exc:
            raise KeyError(f"Unknown material label: {label!r}") from exc

    def process_index(self, label: str) -> int:
        try:
            return self.process_to_idx[str(label).strip()]
        except KeyError as exc:
            raise KeyError(f"Unknown process label: {label!r}") from exc

    def material_name(self, index: int) -> str:
        return self.material_labels[int(index)]

    def process_name(self, index: int) -> str:
        return self.process_labels[int(index)]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "material_labels": list(self.material_labels),
            "process_labels": list(self.process_labels),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LabelSchema":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            material_labels=list(payload["material_labels"]),
            process_labels=list(payload["process_labels"]),
        )

    @classmethod
    def from_records(
        cls,
        records: Iterable[object],
        material_labels: Sequence[str] | None = None,
        process_labels: Sequence[str] | None = None,
    ) -> "LabelSchema":
        """Build a stable label map from records or explicit label lists."""

        if material_labels is None:
            material_labels = _ordered_unique(getattr(record, "material") for record in records)
        if process_labels is None:
            process_labels = _ordered_unique(getattr(record, "process") for record in records)
        return cls(material_labels=list(material_labels), process_labels=list(process_labels))

