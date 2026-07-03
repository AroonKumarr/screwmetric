"""
ScrewMetric — Preview Generator
=================================
Generates contact-sheet PNG images for each dataset split so that
researchers can visually audit the split at a glance.

Responsibility (Single Responsibility Principle):
    Only image-grid composition logic lives here.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from tqdm import tqdm

from config import PipelineConfig, PreviewConfig, get_logger
from utils import ensure_dir, load_coco_annotation

logger = get_logger(__name__)

# Attempt to load a nicer font; fall back to the Pillow built-in if unavailable
_FALLBACK_FONT = ImageFont.load_default()


def _try_load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font; return the default if not available.

    Args:
        size: Desired font size in points.

    Returns:
        A Pillow font object.
    """
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",        # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "arial.ttf",                                    # Windows
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return _FALLBACK_FONT


class PreviewGenerator:
    """Generates annotated contact-sheet previews for train / val / test.

    Each contact sheet is a grid of thumbnails with the filename overlaid
    at the bottom of every cell.  Cells that have annotations are outlined
    in green; unannotated cells are outlined in orange.

    Args:
        config: Pipeline configuration.

    Example::

        gen = PreviewGenerator(PipelineConfig.default())
        gen.generate_all()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._paths = config.paths
        self._cfg: PreviewConfig = config.preview

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> dict[str, Path]:
        """Generate contact sheets for all three splits.

        Returns:
            Mapping of split name → output PNG path.
            If a split has no images the path value is ``None``.
        """
        ensure_dir(self._paths.previews_dir)
        results: dict[str, Path] = {}
        for split_name in ("train", "val", "test"):
            path = self.generate_for_split(split_name)
            results[split_name] = path
        return results

    def generate_for_split(self, split_name: str) -> Path | None:
        """Generate a contact sheet for a single split.

        Args:
            split_name: ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            Absolute path to the saved PNG, or ``None`` if the split is
            empty or its annotation file does not exist.
        """
        ann_path = self._paths.split_annotation_path(split_name)
        img_dir = self._paths.split_images_dir(split_name)

        if not ann_path.exists():
            logger.warning(
                "Annotation file not found for split '%s' — skipping preview.", split_name
            )
            return None

        try:
            coco = load_coco_annotation(ann_path)
        except Exception as exc:
            logger.error("Cannot load annotation for split '%s': %s", split_name, exc)
            return None

        images: list[dict[str, Any]] = coco["images"]
        if not images:
            logger.warning("Split '%s' has no images — skipping preview.", split_name)
            return None

        # Limit to max_images
        sample = images[: self._cfg.max_images]
        annotated_ids = {ann["image_id"] for ann in coco.get("annotations", [])}

        output_path = self._paths.previews_dir / f"preview_{split_name}.png"
        ensure_dir(self._paths.previews_dir)
        sheet = self._build_contact_sheet(
            images=sample,
            img_dir=img_dir,
            annotated_ids=annotated_ids,
            split_name=split_name,
            total_count=len(images),
        )
        sheet.save(str(output_path), "PNG")
        logger.info(
            "Preview saved → %s  (%d thumbnails)", output_path, len(sample)
        )
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_contact_sheet(
        self,
        images: list[dict[str, Any]],
        img_dir: Path,
        annotated_ids: set[int],
        split_name: str,
        total_count: int,
    ) -> Image.Image:
        """Compose the contact-sheet grid image.

        Args:
            images: COCO image records to include.
            img_dir: Directory containing the actual image files.
            annotated_ids: Set of image IDs that have at least one annotation.
            split_name: Name of the split (used for the header).
            total_count: True total image count in the split (may exceed sample).

        Returns:
            A composed :class:`PIL.Image.Image` ready to save.
        """
        cfg = self._cfg
        thumb_w, thumb_h = cfg.thumbnail_size
        border = cfg.border_px
        cols = min(cfg.max_cols, len(images))
        rows = math.ceil(len(images) / cols)

        label_h = 22  # pixels reserved for filename label below each thumbnail
        header_h = 48  # header banner height
        cell_w = thumb_w + 2 * border
        cell_h = thumb_h + 2 * border + label_h

        sheet_w = cols * cell_w
        sheet_h = header_h + rows * cell_h

        sheet = Image.new("RGB", (sheet_w, sheet_h), cfg.background_colour)
        draw = ImageDraw.Draw(sheet)
        font = _try_load_font(cfg.font_size)
        small_font = _try_load_font(max(cfg.font_size - 2, 8))

        # Header banner
        header_colour = (45, 125, 210)
        draw.rectangle([0, 0, sheet_w, header_h], fill=header_colour)
        header_text = (
            f"ScrewMetric — {split_name.upper()} split  "
            f"({len(images)} shown / {total_count} total)"
        )
        draw.text((12, 12), header_text, fill=(255, 255, 255), font=font)

        logger.info(
            "Building contact sheet for '%s' (%d × %d grid)…",
            split_name, cols, rows,
        )

        for idx, img_meta in enumerate(
            tqdm(images, desc=f"Thumbnails [{split_name}]", unit="img")
        ):
            row, col = divmod(idx, cols)
            x_off = col * cell_w
            y_off = header_h + row * cell_h

            # Load and resize thumbnail
            thumb = self._load_thumbnail(img_dir / img_meta["file_name"], thumb_w, thumb_h)

            # Paste thumbnail
            sheet.paste(thumb, (x_off + border, y_off + border))

            # Border colour: green if annotated, orange if not
            has_ann = img_meta["id"] in annotated_ids
            border_col = (72, 199, 142) if has_ann else (255, 159, 67)
            draw.rectangle(
                [x_off, y_off, x_off + cell_w - 1, y_off + thumb_h + 2 * border - 1],
                outline=border_col,
                width=border,
            )

            # Filename label
            label = img_meta["file_name"]
            if len(label) > 18:
                label = label[:15] + "…"
            draw.text(
                (x_off + border, y_off + thumb_h + 2 * border + 2),
                label,
                fill=(200, 200, 200),
                font=small_font,
            )

        return sheet

    def _load_thumbnail(self, path: Path, width: int, height: int) -> Image.Image:
        """Load an image and resize it to fit within ``(width, height)``.

        If the file cannot be opened (missing or corrupted), a grey
        placeholder is returned instead.

        Args:
            path: Absolute path to the image file.
            width: Desired thumbnail width.
            height: Desired thumbnail height.

        Returns:
            RGB :class:`PIL.Image.Image` of exactly ``(width, height)``.
        """
        placeholder_colour = (80, 80, 80)
        try:
            with Image.open(path) as im:
                im.thumbnail((width, height), Image.LANCZOS)
                # Centre on a fixed-size canvas
                canvas = Image.new("RGB", (width, height), placeholder_colour)
                paste_x = (width - im.width) // 2
                paste_y = (height - im.height) // 2
                if im.mode != "RGB":
                    im = im.convert("RGB")
                canvas.paste(im, (paste_x, paste_y))
                return canvas
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            logger.debug("Cannot open thumbnail '%s': %s", path, exc)
            canvas = Image.new("RGB", (width, height), placeholder_colour)
            draw = ImageDraw.Draw(canvas)
            draw.text((4, height // 2 - 8), "MISSING", fill=(255, 80, 80))
            return canvas


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate contact-sheet previews for all three splits.

    Reads split annotation files and real images from disk.
    Output PNGs are saved to ``dataset/previews/``.
    """
    print("=" * 60)
    print("  ScrewMetric — Preview Generator Module")
    print("=" * 60)

    try:
        from config import PipelineConfig
        config = PipelineConfig.default()

        print(f"\nSplits dir   : {config.paths.splits_dir}")
        print(f"Previews dir : {config.paths.previews_dir}")
        print(f"Thumbnail    : {config.preview.thumbnail_size}")
        print(f"Max cols     : {config.preview.max_cols}")
        print("\nGenerating previews…\n")

        generator = PreviewGenerator(config)
        results = generator.generate_all()

        print(f"\n{'─' * 40}")
        for split, path in results.items():
            if path and path.exists():
                size_kb = path.stat().st_size // 1024
                print(f"  {split:6s}: {path.name}  ({size_kb} KB)")
            else:
                print(f"  {split:6s}: skipped (no annotation found)")
        print(f"{'─' * 40}")

        print(f"\n✅ preview_generator.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ preview_generator.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

