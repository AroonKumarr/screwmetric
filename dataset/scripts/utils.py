"""
ScrewMetric — Dataset Processing Utilities
==========================================
Shared, stateless helper functions used across the dataset processing pipeline.
No module should duplicate these utilities.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    """Load and deserialise a JSON file.

    Handles a known CVAT export quirk where the file content is prefixed
    with a Markdown code-fence backtick (e.g. `` `{...} ``).
    The backtick and any surrounding whitespace are stripped before parsing.

    Args:
        path: Absolute path to the JSON file.

    Returns:
        The deserialised Python object (dict, list, etc.).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        OSError: For other filesystem-level errors.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    logger.debug("Loading JSON from %s", path)
    with path.open("r", encoding="utf-8") as fh:
        content = fh.read()

    # Strip Markdown code-fence backticks that CVAT sometimes prepends/appends
    content = content.strip()
    if content.startswith("`"):
        content = content.lstrip("`").rstrip("`").strip()

    return json.loads(content)


def save_json(data: Any, path: Path, *, indent: int = 2) -> None:
    """Serialise ``data`` and write it to ``path`` as JSON.

    Parent directories are created automatically.

    Args:
        data: A JSON-serialisable Python object.
        path: Destination file path.
        indent: JSON indentation level (default 2).

    Raises:
        TypeError: If ``data`` contains non-serialisable objects.
        OSError: For filesystem-level errors.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Saving JSON to %s", path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create ``path`` and all parent directories if they do not exist.

    Args:
        path: Directory to create.

    Returns:
        The same ``path`` (for chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: Path, dst: Path) -> None:
    """Copy a file from ``src`` to ``dst``.

    The destination's parent directory is created if necessary.

    Args:
        src: Source file path.
        dst: Destination file path.

    Raises:
        FileNotFoundError: If ``src`` does not exist.
        shutil.Error: For copy-level errors.
    """
    if not src.is_file():
        raise FileNotFoundError(f"Source file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def list_image_files(directory: Path, extensions: frozenset[str]) -> list[Path]:
    """Return a sorted list of image files inside ``directory``.

    Only files whose extension (lower-cased) is in ``extensions`` are included.
    The search is non-recursive (top-level only).

    Args:
        directory: Directory to scan.
        extensions: Accepted file extensions, e.g. ``frozenset({".jpg", ".png"})``.

    Returns:
        Sorted list of absolute :class:`~pathlib.Path` objects.

    Raises:
        NotADirectoryError: If ``directory`` is not a directory.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]
    return sorted(files)


def human_readable_size(num_bytes: int) -> str:
    """Convert a byte count to a human-readable string (e.g. ``"12.4 MB"``).

    Args:
        num_bytes: Size in bytes.

    Returns:
        Formatted string with appropriate unit suffix.
    """
    if num_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    exp = int(math.log(num_bytes, 1024))
    exp = min(exp, len(units) - 1)
    value = num_bytes / (1024 ** exp)
    return f"{value:.1f} {units[exp]}"


def get_directory_size(directory: Path) -> int:
    """Recursively compute the total size of all files in ``directory``.

    Args:
        directory: Root directory to measure.

    Returns:
        Total size in bytes.
    """
    if not directory.exists():
        return 0
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------

def load_coco_annotation(path: Path) -> dict[str, Any]:
    """Load and minimally validate a COCO-format annotation file.

    The function checks that the top-level keys ``"images"``,
    ``"annotations"``, and ``"categories"`` are all present.

    Args:
        path: Path to the COCO JSON file.

    Returns:
        The annotation dict with keys ``images``, ``annotations``,
        ``categories``, ``info``, and ``licenses``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required COCO keys are missing.
        json.JSONDecodeError: If the file is malformed JSON.
    """
    data = load_json(path)
    required_keys = {"images", "annotations", "categories"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(
            f"COCO annotation file is missing required keys: {sorted(missing)}"
        )
    return data


def build_image_id_map(coco_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Build a mapping from image ID to image metadata dict.

    Args:
        coco_data: Loaded COCO annotation dict.

    Returns:
        ``{image_id: image_record, ...}``
    """
    return {img["id"]: img for img in coco_data["images"]}


def build_annotations_by_image(
    coco_data: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    """Group annotations by their ``image_id``.

    Args:
        coco_data: Loaded COCO annotation dict.

    Returns:
        ``{image_id: [annotation, ...], ...}``
    """
    result: dict[int, list[dict[str, Any]]] = {}
    for ann in coco_data["annotations"]:
        result.setdefault(ann["image_id"], []).append(ann)
    return result


def filter_coco_for_image_ids(
    coco_data: dict[str, Any],
    image_ids: list[int],
) -> dict[str, Any]:
    """Return a new COCO dict containing only the specified image IDs.

    All metadata (``info``, ``licenses``, ``categories``) is preserved.
    Images and annotations are filtered to match ``image_ids``.

    Args:
        coco_data: Source COCO annotation dict (not mutated).
        image_ids: List of image IDs to retain.

    Returns:
        Filtered COCO annotation dict.
    """
    id_set = set(image_ids)
    filtered_images = [img for img in coco_data["images"] if img["id"] in id_set]
    filtered_annotations = [
        ann for ann in coco_data["annotations"] if ann["image_id"] in id_set
    ]
    return {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data["categories"],
        "images": filtered_images,
        "annotations": filtered_annotations,
    }


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate utility functions without requiring a real dataset.

    Creates temporary in-memory structures and verifies each helper
    works correctly, printing results to stdout.
    """
    import tempfile
    import os

    print("=" * 60)
    print("  ScrewMetric — Utilities Module")
    print("=" * 60)

    try:
        # ── human_readable_size ──────────────────────────────────────
        print("\n[human_readable_size]")
        for n in [0, 512, 1_024, 1_048_576, 1_073_741_824]:
            print(f"  {n:>15,} bytes → {human_readable_size(n)}")

        # ── JSON round-trip ──────────────────────────────────────────
        print("\n[save_json / load_json]")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "test.json"
            payload = {"key": "value", "numbers": [1, 2, 3]}
            save_json(payload, tmp_path)
            loaded = load_json(tmp_path)
            assert loaded == payload, "JSON round-trip mismatch"
            print(f"  Saved and reloaded: {loaded}")

        # ── Backtick-prefixed JSON (CVAT quirk) ──────────────────────
        print("\n[load_json — CVAT backtick prefix]")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "cvat.json"
            tmp_path.write_text('`{"hello": "world"}`', encoding="utf-8")
            loaded = load_json(tmp_path)
            assert loaded == {"hello": "world"}
            print(f"  Stripped backticks and parsed: {loaded}")

        # ── build_image_id_map ───────────────────────────────────────
        print("\n[build_image_id_map]")
        fake_coco = {
            "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
            "annotations": [
                {"id": 10, "image_id": 1, "category_id": 1},
                {"id": 11, "image_id": 1, "category_id": 1},
                {"id": 12, "image_id": 2, "category_id": 1},
            ],
            "categories": [{"id": 1, "name": "screw"}],
        }
        id_map = build_image_id_map(fake_coco)
        assert id_map[1]["file_name"] == "a.jpg"
        print(f"  id_map keys: {list(id_map.keys())}")

        # ── build_annotations_by_image ───────────────────────────────
        print("\n[build_annotations_by_image]")
        ann_map = build_annotations_by_image(fake_coco)
        assert len(ann_map[1]) == 2
        assert len(ann_map[2]) == 1
        print(f"  image_id=1 has {len(ann_map[1])} annotations")
        print(f"  image_id=2 has {len(ann_map[2])} annotations")

        # ── filter_coco_for_image_ids ────────────────────────────────
        print("\n[filter_coco_for_image_ids]")
        filtered = filter_coco_for_image_ids(fake_coco, [1])
        assert len(filtered["images"]) == 1
        assert len(filtered["annotations"]) == 2
        print(f"  Filtered to image_id=1: {len(filtered['images'])} image, "
              f"{len(filtered['annotations'])} annotations")

        # ── list_image_files ─────────────────────────────────────────
        print("\n[list_image_files]")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for name in ["img1.jpg", "img2.png", "doc.pdf"]:
                (tmp_dir / name).write_bytes(b"x")
            imgs = list_image_files(tmp_dir, frozenset({".jpg", ".png"}))
            assert len(imgs) == 2
            print(f"  Found {len(imgs)} image files (excluded .pdf)")

        print("\n✅ utils.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ utils.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

