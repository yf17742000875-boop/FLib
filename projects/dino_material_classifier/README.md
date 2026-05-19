# DINOv3 Material/Process Classifier

This project trains a two-head classifier for industrial part images:

- `material`: aluminum alloy, brass, titanium alloy, etc.
- `process`: polished, sandblasted, anodized, machined, etc.

The v1 assumption is one part per image or one pre-cropped ROI per sample.
Detection/segmentation is intentionally kept outside the first version.

## Data Protocol

Create a CSV manifest with at least:

```csv
image_path,material,process,sample_id,group_id,split,bbox
images/part001.png,aluminum_alloy,polished,part001_view1,part001,train,"40,30,420,380"
```

Required columns:

- `image_path`: absolute path or path relative to `--image-root`
- `material`: one material label
- `process`: one process label

Recommended columns:

- `sample_id`: image-level identifier
- `group_id`: physical part or capture-session identifier
- `split`: `train`, `val`, or `test`
- `bbox`: ROI as `x1,y1,x2,y2`
- `mask_path`: optional binary mask; used when `--roi-mode mask`

If `split` is omitted, samples are split deterministically by `group_id`.
For real experiments, prefer explicit splits by physical part to avoid leakage.

## Smoke Test

This uses a tiny local CNN backbone and synthetic images, so it does not need
DINOv3 weights.

```bat
python -m projects.dino_material_classifier.smoke_test
```

## Train With DINOv3

Clone or install the official DINOv3 repo and pass the repo path plus weights:

```bat
python -m projects.dino_material_classifier.train ^
  --manifest data/material_parts/manifest.csv ^
  --image-root data/material_parts ^
  --output-dir projects/dino_material_classifier/runs/exp001 ^
  --backbone-kind dinov3 ^
  --repo-or-dir D:\path\to\dinov3 ^
  --source local ^
  --model-name dinov3_vitb16 ^
  --weights D:\path\to\dinov3_vitb16_pretrain.pth ^
  --batch-size 8 ^
  --epochs 10 ^
  --head-epochs 3
```

If your weights come from a model variant trained on satellite imagery, use
`--normalization dinov3_sat`; otherwise keep the default ImageNet normalization.

## Inference

Single image:

```bat
python -m projects.dino_material_classifier.infer ^
  --checkpoint projects/dino_material_classifier/runs/exp001/best.pt ^
  --image data/material_parts/images/part001.png
```

Batch:

```bat
python -m projects.dino_material_classifier.infer ^
  --checkpoint projects/dino_material_classifier/runs/exp001/best.pt ^
  --manifest data/material_parts/infer_manifest.csv ^
  --image-root data/material_parts ^
  --output projects/dino_material_classifier/runs/exp001/infer.csv
```

## Outputs

Each run writes:

- `config.json`: training configuration
- `label_schema.json`: material/process label maps
- `history.csv`: epoch metrics
- `best.pt` and `last.pt`: checkpoints
- `eval/*_confusion.csv`: confusion matrices
- `eval/*_predictions.csv`: per-sample predictions
- `eval/*_errors.csv`: error cases for inspection

