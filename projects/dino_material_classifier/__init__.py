"""DINOv3-based material and process classifier."""

from .backbones import BackboneOutput, DINOv3Backbone, TinyBackbone, build_backbone
from .data import (
    ImageRecord,
    MaterialProcessDataset,
    build_dataloader,
    build_label_schema,
    create_dataloaders,
    load_records,
)
from .labels import LabelSchema
from .model import ClassificationOutput, MaterialProcessClassifier

__all__ = [
    "BackboneOutput",
    "ClassificationOutput",
    "DINOv3Backbone",
    "ImageRecord",
    "LabelSchema",
    "MaterialProcessClassifier",
    "MaterialProcessDataset",
    "TinyBackbone",
    "build_backbone",
    "build_dataloader",
    "build_label_schema",
    "create_dataloaders",
    "load_records",
]

