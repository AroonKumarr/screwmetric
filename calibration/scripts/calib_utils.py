"""
ScrewMetric — Camera Calibration Utilities
===========================================
Shared, stateless helper functions used across the calibration pipeline.
No module should duplicate these utilities.

Responsibility (Single Responsibility Principle):
    I/O helpers only.  No OpenCV calibration logic lives here.

Authors: ScrewMetric Team
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from calib_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create a directory and all parent directories if they do not exist.

    Args:
        path: Directory path to create.

    Returns:
        The same ``path`` (for chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load an image from disk using OpenCV.

    Args:
        path: Absolute path to the image file.

    Returns:
        Loaded BGR image as a numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If OpenCV cannot decode the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"OpenCV could not decode image: {path}")
    logger.debug("Loaded image: %s  shape=%s", path.name, img.shape)
    return img


def load_image_gray(path: Path) -> np.ndarray:
    """Load an image from disk as grayscale.

    Args:
        path: Absolute path to the image file.

    Returns:
        Loaded single-channel grayscale image as a numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If OpenCV cannot decode the file.
    """
    img = load_image(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def save_image(img: np.ndarray, path: Path) -> None:
    """Save an image to disk using OpenCV.

    Args:
        img: Image array to save.
        path: Destination file path.

    Raises:
        IOError: If OpenCV cannot encode and write the file.
    """
    ensure_dir(path.parent)
    success = cv2.imwrite(str(path), img)
    if not success:
        raise IOError(f"OpenCV could not write image to: {path}")
    logger.debug("Saved image: %s", path.name)


def is_image_readable(path: Path) -> bool:
    """Check whether a file can be decoded as an image by OpenCV.

    Args:
        path: Path to the file to test.

    Returns:
        ``True`` if the file decodes successfully, ``False`` otherwise.
    """
    try:
        img = cv2.imread(str(path))
        return img is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def list_image_files(
    directory: Path,
    extensions: frozenset[str],
) -> list[Path]:
    """Collect image files from a directory (non-recursive).

    Args:
        directory: Directory to search.
        extensions: Set of allowed file extensions (e.g. ``frozenset({".jpg"})``).

    Returns:
        Sorted list of matching :class:`~pathlib.Path` objects.
    """
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def human_readable_size(n_bytes: int) -> str:
    """Convert a byte count to a human-readable string.

    Args:
        n_bytes: Number of bytes.

    Returns:
        Human-readable size string (e.g. ``"3.7 MB"``).
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024.0:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0  # type: ignore[assignment]
    return f"{n_bytes:.1f} PB"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def save_json(data: Any, path: Path, *, indent: int = 2) -> None:
    """Serialise a Python object to a JSON file.

    Args:
        data: JSON-serialisable Python object.
        path: Destination file path.
        indent: JSON indentation level.

    Raises:
        TypeError: If ``data`` is not JSON-serialisable.
    """
    ensure_dir(path.parent)
    logger.debug("Saving JSON to %s", path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load and deserialise a JSON file.

    Args:
        path: Absolute path to the JSON file.

    Returns:
        The deserialised Python object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    logger.debug("Loading JSON from %s", path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def save_yaml(data: dict[str, Any], path: Path) -> None:
    """Serialise a dictionary to a YAML file.

    Args:
        data: Dictionary to serialise.
        path: Destination file path.
    """
    ensure_dir(path.parent)
    logger.debug("Saving YAML to %s", path)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    logger.debug("Loading YAML from %s", path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# NumPy helpers
# ---------------------------------------------------------------------------

def save_numpy(array: np.ndarray, path: Path) -> None:
    """Save a numpy array to a ``.npy`` file.

    Args:
        array: Array to save.
        path: Destination ``.npy`` file path.
    """
    ensure_dir(path.parent)
    logger.debug("Saving numpy array to %s  shape=%s", path.name, array.shape)
    np.save(str(path), array)


def load_numpy(path: Path) -> np.ndarray:
    """Load a numpy array from a ``.npy`` file.

    Args:
        path: Absolute path to the ``.npy`` file.

    Returns:
        Loaded numpy array.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"NumPy file not found: {path}")
    logger.debug("Loading numpy array from %s", path.name)
    return np.load(str(path))


# ---------------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate all utility functions using temporary in-memory data."""
    import tempfile

    print("=" * 62)
    print("  ScrewMetric — Calibration Utilities Module")
    print("=" * 62)

    try:
        # ── ensure_dir ───────────────────────────────────────────────
        print("\n[ensure_dir]")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "a" / "b" / "c"
            ensure_dir(d)
            assert d.exists()
            print(f"  Created nested dirs: {d.relative_to(tmp)}")

        # ── save_json / load_json ─────────────────────────────────────
        print("\n[save_json / load_json]")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.json"
            payload = {"key": "value", "numbers": [1, 2, 3]}
            save_json(payload, p)
            loaded = load_json(p)
            assert loaded == payload
            print(f"  Round-trip OK: {loaded}")

        # ── save_yaml / load_yaml ─────────────────────────────────────
        print("\n[save_yaml / load_yaml]")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.yaml"
            payload = {"camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
            save_yaml(payload, p)
            loaded = load_yaml(p)
            assert loaded == payload
            print(f"  Round-trip OK: {list(loaded.keys())}")

        # ── save_numpy / load_numpy ───────────────────────────────────
        print("\n[save_numpy / load_numpy]")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "matrix.npy"
            arr = np.array([[1.0, 0.0], [0.0, 1.0]])
            save_numpy(arr, p)
            loaded = load_numpy(p)
            assert np.allclose(arr, loaded)
            print(f"  Round-trip OK: shape={loaded.shape}")

        # ── list_image_files ──────────────────────────────────────────
        print("\n[list_image_files]")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ["img1.jpg", "img2.png", "doc.pdf", "data.txt"]:
                (d / name).write_bytes(b"x")
            exts = frozenset({".jpg", ".png"})
            imgs = list_image_files(d, exts)
            assert len(imgs) == 2
            print(f"  Found {len(imgs)} image files (excluded .pdf, .txt)")

        # ── human_readable_size ───────────────────────────────────────
        print("\n[human_readable_size]")
        for n in [0, 1_024, 1_048_576, 1_073_741_824]:
            print(f"  {n:>15,} bytes → {human_readable_size(n)}")

        # ── is_image_readable ─────────────────────────────────────────
        print("\n[is_image_readable]")
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = Path(tmp) / "corrupt.jpg"
            corrupt.write_bytes(b"not a real jpeg file")
            result = is_image_readable(corrupt)
            assert result is False
            print(f"  Corrupt file readable: {result}  (expected False)")

        print("\n✅ utils.py executed successfully.")

    except Exception as exc:
        print(f"\n❌ utils.py failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
