# ScrewMetric — Dataset Processing Module

A **production-quality** Python pipeline for validating, splitting, analysing,
and previewing the ScrewMetric annotated screw dataset.

---

## Overview

The module lives in `dataset/scripts/` and processes the COCO-format dataset
exported from CVAT end-to-end in a single command.

```
python dataset_processor.py
```

The pipeline performs five sequential stages:

| # | Stage | Output |
|---|---|---|
| 1 | **Validation** | `dataset/validation_report.json` |
| 2 | **Split** | `dataset/splits/{train,val,test}/{images,annotations}/` |
| 3 | **Statistics** | `dataset/dataset_stats.json` |
| 4 | **Report** | `dataset/DATASET_REPORT.md` |
| 5 | **Previews** | `dataset/previews/preview_{train,val,test}.png` |

---

## Module Structure

```
dataset/scripts/
├── dataset_processor.py       ← CLI entry point & orchestrator
├── dataset_validator.py       ← Part 1: Dataset validation
├── dataset_splitter.py        ← Part 2: 70/20/10 split
├── dataset_statistics.py      ← Part 3: Statistics computation
├── dataset_report_generator.py← Part 4: Markdown report
├── preview_generator.py       ← Part 5: Contact-sheet previews
├── config.py                  ← Centralised configuration (paths, ratios, etc.)
└── utils.py                   ← Shared utilities (JSON I/O, COCO helpers, etc.)
```

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run from inside the `dataset/scripts/` directory:

```bash
cd dataset/scripts
```

### Run the full pipeline

```bash
python dataset_processor.py
```

### Skip individual stages

```bash
# Validate only
python dataset_processor.py --skip-split --skip-stats --skip-report --skip-preview

# Re-generate statistics + report without re-splitting
python dataset_processor.py --skip-validation --skip-split

# Re-generate previews only
python dataset_processor.py --skip-validation --skip-split --skip-stats --skip-report

# Skip validation (useful if already confirmed clean)
python dataset_processor.py --skip-validation
```

### All CLI flags

```
--skip-validation   Skip dataset integrity checks
--skip-split        Skip train/val/test splitting
--skip-stats        Skip statistics computation
--skip-report       Skip Markdown report generation
--skip-preview      Skip contact-sheet preview generation
```

---

## Configuration

All paths, split ratios, and tunable parameters are in `config.py`.
**Never hardcode paths elsewhere** — update `config.py` instead.

Key settings:

| Config class | Key fields |
|---|---|
| `PathConfig` | All filesystem paths (auto-resolved relative to `dataset/`) |
| `SplitConfig` | `train_ratio=0.70`, `val_ratio=0.20`, `test_ratio=0.10`, `random_seed=42` |
| `ValidationConfig` | `supported_extensions`, `verify_image_integrity` |
| `PreviewConfig` | `thumbnail_size`, `max_cols`, `max_images` |

---

## Output Description

### `validation_report.json`

```json
{
  "total_images_in_filesystem": 73,
  "total_images_in_coco": 58,
  "valid_images": 58,
  "missing_images": [],
  "extra_images": ["screw_025.jpg", ...],
  "corrupted_images": [],
  "unannotated_images": [],
  "orphan_annotations": [],
  "warnings": [...],
  "errors": [],
  "is_valid": true
}
```

### `dataset_stats.json`

```json
{
  "total_images": 58,
  "total_annotations": 58,
  "category_distribution": {"screw": 58},
  "resolution": {"avg_width": 3024, "avg_height": 4032, ...},
  "bbox": {"count": 58, "avg_width": 296.3, ...},
  "polygon": {"count": 58, "avg_points_per_polygon": 34.2, ...},
  "splits": {
    "train": {"count": 41, "annotation_count": 41, "ratio": 0.7069},
    "val": {"count": 11, "annotation_count": 11, "ratio": 0.1897},
    "test": {"count": 6, "annotation_count": 6, "ratio": 0.1034}
  }
}
```

### Split structure

```
splits/
├── train/
│   ├── images/         ← 41 images (symlink-free copies)
│   └── annotations/instances_train.json
├── val/
│   ├── images/
│   └── annotations/instances_val.json
└── test/
    ├── images/
    └── annotations/instances_test.json
```

### Previews

Contact sheets (`preview_train.png`, `preview_val.png`, `preview_test.png`)
show a grid of thumbnails where:

* 🟢 **Green border** — image has at least one annotation
* 🟠 **Orange border** — image is unannotated

---

## Design Decisions

| Decision | Rationale |
|---|---|
| SOLID principles | Each class has one responsibility; easy to test, extend, replace |
| Frozen dataclasses for config | Immutable config prevents accidental mutation |
| `pathlib.Path` throughout | OS-agnostic, no string concatenation |
| `seed=42` shuffle | Reproducible splits across machines |
| `copy2` (not symlinks) | Cross-platform; splits are self-contained |
| `tqdm` progress bars | Visibility for large datasets |
| Structured JSON reports | Machine-readable for downstream CI checks |
| Google-style docstrings | Consistent, tooling-friendly documentation |

---

## Running Tests

> Unit tests for this module reside in `tests/test_dataset/`.

```bash
cd <project_root>
python -m pytest tests/test_dataset/ -v
```

---

## Authors

ScrewMetric Team — AI Internship Assessment Project
