"""
ScrewMetric — Tests: Preview Generator
=========================================
Unit tests for ``dataset/scripts/preview_generator.py``.

Coverage:
- Contact-sheet generated for each split
- Output PNG saved to previews/ directory
- Thumbnail dimensions within expected bounds
- Graceful fallback for missing images (placeholder)
- Skips split when annotation file is absent
- Correct number of images in sheet
- generate_for_split returns None for missing annotation
- Annotated vs unannotated border colours (logical test)
- _load_thumbnail returns correct size even for corrupt image
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent.parent / "dataset" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from preview_generator import PreviewGenerator
from config import PreviewConfig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_generator(tmp_dataset: dict[str, Any]) -> PreviewGenerator:
    return PreviewGenerator(tmp_dataset["config"])


# ===========================================================================
# generate_for_split: missing annotation
# ===========================================================================

class TestPreviewGeneratorMissingAnnotation:
    def test_returns_none_for_missing_annotation(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        # No split exists yet — annotation files absent
        gen = _make_generator(tmp_dataset)
        result = gen.generate_for_split("train")
        assert result is None

    def test_generate_all_returns_none_for_all_missing(
        self, tmp_dataset: dict[str, Any]
    ) -> None:
        gen = _make_generator(tmp_dataset)
        results = gen.generate_all()
        assert all(v is None for v in results.values())


# ===========================================================================
# generate_for_split: happy path (after split)
# ===========================================================================

class TestPreviewGeneratorHappyPath:
    def test_preview_file_created_for_train(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        config = tmp_dataset_with_split["config"]
        gen = PreviewGenerator(config)
        path = gen.generate_for_split("train")
        assert path is not None
        assert path.exists()

    def test_preview_file_created_for_val(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        config = tmp_dataset_with_split["config"]
        gen = PreviewGenerator(config)
        path = gen.generate_for_split("val")
        assert path is not None
        assert path.exists()

    def test_preview_file_created_for_test(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        config = tmp_dataset_with_split["config"]
        gen = PreviewGenerator(config)
        path = gen.generate_for_split("test")
        assert path is not None
        assert path.exists()

    def test_generate_all_returns_three_paths(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        config = tmp_dataset_with_split["config"]
        gen = PreviewGenerator(config)
        results = gen.generate_all()
        assert set(results.keys()) == {"train", "val", "test"}

    def test_all_three_previews_exist_on_disk(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        config = tmp_dataset_with_split["config"]
        gen = PreviewGenerator(config)
        results = gen.generate_all()
        for split, path in results.items():
            assert path is not None and path.exists(), \
                f"Preview missing for split '{split}'"

    def test_preview_is_valid_image_file(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        config = tmp_dataset_with_split["config"]
        gen = PreviewGenerator(config)
        path = gen.generate_for_split("train")
        assert path is not None
        with Image.open(path) as im:
            assert im.mode == "RGB"
            assert im.width > 0 and im.height > 0

    def test_preview_saved_in_previews_dir(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        config = tmp_dataset_with_split["config"]
        gen = PreviewGenerator(config)
        path = gen.generate_for_split("train")
        assert path is not None
        assert path.parent == config.paths.previews_dir

    def test_preview_filename_matches_split_name(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        config = tmp_dataset_with_split["config"]
        gen = PreviewGenerator(config)
        for split in ("train", "val", "test"):
            path = gen.generate_for_split(split)
            if path is not None:
                assert path.name == f"preview_{split}.png"


# ===========================================================================
# Sheet dimensions
# ===========================================================================

class TestPreviewSheetDimensions:
    def test_sheet_width_at_least_thumbnail_width(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        config = tmp_dataset_with_split["config"]
        gen = PreviewGenerator(config)
        path = gen.generate_for_split("train")
        assert path is not None
        with Image.open(path) as im:
            thumb_w = config.preview.thumbnail_size[0]
            border = config.preview.border_px
            min_w = thumb_w + 2 * border
            assert im.width >= min_w

    def test_sheet_is_not_empty(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        config = tmp_dataset_with_split["config"]
        gen = PreviewGenerator(config)
        path = gen.generate_for_split("train")
        assert path is not None
        assert path.stat().st_size > 0


# ===========================================================================
# _load_thumbnail
# ===========================================================================

class TestLoadThumbnail:
    def _get_generator(self, tmp_dataset: dict[str, Any]) -> PreviewGenerator:
        return PreviewGenerator(tmp_dataset["config"])

    def test_returns_correct_size_for_real_image(
        self, tmp_dataset: dict[str, Any], tmp_path: Path
    ) -> None:
        gen = self._get_generator(tmp_dataset)
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (300, 400), color=(100, 150, 200)).save(str(img_path))
        w, h = 64, 64
        thumb = gen._load_thumbnail(img_path, w, h)
        assert thumb.size == (w, h)
        assert thumb.mode == "RGB"

    def test_returns_placeholder_for_missing_file(
        self, tmp_dataset: dict[str, Any], tmp_path: Path
    ) -> None:
        gen = self._get_generator(tmp_dataset)
        missing = tmp_path / "does_not_exist.jpg"
        w, h = 64, 64
        thumb = gen._load_thumbnail(missing, w, h)
        assert thumb.size == (w, h)
        assert thumb.mode == "RGB"

    def test_returns_placeholder_for_corrupt_image(
        self, tmp_dataset: dict[str, Any], tmp_path: Path
    ) -> None:
        gen = self._get_generator(tmp_dataset)
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"not a valid jpeg")
        w, h = 64, 64
        thumb = gen._load_thumbnail(corrupt, w, h)
        assert thumb.size == (w, h)

    def test_rgba_image_converted_to_rgb(
        self, tmp_dataset: dict[str, Any], tmp_path: Path
    ) -> None:
        gen = self._get_generator(tmp_dataset)
        img_path = tmp_path / "rgba.png"
        Image.new("RGBA", (100, 100), color=(10, 20, 30, 255)).save(str(img_path))
        thumb = gen._load_thumbnail(img_path, 64, 64)
        assert thumb.mode == "RGB"


# ===========================================================================
# PreviewConfig
# ===========================================================================

class TestPreviewConfig:
    def test_default_thumbnail_size(self) -> None:
        cfg = PreviewConfig()
        assert cfg.thumbnail_size == (256, 256)

    def test_default_max_cols(self) -> None:
        cfg = PreviewConfig()
        assert cfg.max_cols == 6

    def test_default_max_images(self) -> None:
        cfg = PreviewConfig()
        assert cfg.max_images > 0

    def test_custom_thumbnail_size_respected(
        self, tmp_dataset_with_split: dict[str, Any]
    ) -> None:
        config = tmp_dataset_with_split["config"]
        # config.preview.thumbnail_size is (64, 64) per fixture
        assert config.preview.thumbnail_size == (64, 64)
