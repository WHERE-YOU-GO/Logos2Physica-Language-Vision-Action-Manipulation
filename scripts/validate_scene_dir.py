from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.path_manager import resolve_path


REQUIRED_META_FIELDS = (
    "prompt",
    "expected_source_color",
    "expected_source_label",
    "expected_target_color",
    "expected_target_label",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a replay scene directory.")
    parser.add_argument("--scene_dir", required=True)
    return parser.parse_args()


def _load_rgb_shape(rgb_path: Path) -> tuple[int, int, int] | None:
    if not rgb_path.exists():
        return None
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to inspect rgb.png in a replay scene directory."
        ) from exc

    with Image.open(rgb_path) as handle:
        array = np.asarray(handle.convert("RGB"))
    return tuple(int(value) for value in array.shape)


def _load_depth_shape(depth_path: Path) -> tuple[int, ...] | None:
    if not depth_path.exists():
        return None
    depth = np.load(depth_path, allow_pickle=False)
    return tuple(int(value) for value in depth.shape)


def _load_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"meta.json must contain a JSON object, got {type(payload).__name__}.")
    return payload


def main() -> None:
    args = _parse_args()
    scene_dir = resolve_path(args.scene_dir)
    rgb_path = scene_dir / "rgb.png"
    depth_path = scene_dir / "depth.npy"
    meta_path = scene_dir / "meta.json"

    print("== Replay Scene Validation ==")
    print(f"scene_dir: {scene_dir}")
    print(f"scene_dir_exists: {scene_dir.exists()}")
    print(f"scene_dir_is_dir: {scene_dir.is_dir()}")

    rgb_exists = rgb_path.exists()
    depth_exists = depth_path.exists()
    meta_exists = meta_path.exists()

    print()
    print("== Required Files ==")
    print(f"rgb.png: {'FOUND' if rgb_exists else 'MISSING'}")
    print(f"depth.npy: {'FOUND' if depth_exists else 'MISSING'}")
    print(f"meta.json: {'FOUND' if meta_exists else 'MISSING'}")

    rgb_shape = None
    depth_shape = None
    meta = None
    rgb_error = None
    depth_error = None
    meta_error = None

    if rgb_exists:
        try:
            rgb_shape = _load_rgb_shape(rgb_path)
        except Exception as exc:
            rgb_error = f"{type(exc).__name__}: {exc}"
    if depth_exists:
        try:
            depth_shape = _load_depth_shape(depth_path)
        except Exception as exc:
            depth_error = f"{type(exc).__name__}: {exc}"
    if meta_exists:
        try:
            meta = _load_meta(meta_path)
        except Exception as exc:
            meta_error = f"{type(exc).__name__}: {exc}"

    print()
    print("== Shapes ==")
    print(f"rgb_size: {rgb_shape if rgb_shape is not None else 'unavailable'}")
    print(f"depth_shape: {depth_shape if depth_shape is not None else 'unavailable'}")
    if rgb_error is not None:
        print(f"rgb_error: {rgb_error}")
    if depth_error is not None:
        print(f"depth_error: {depth_error}")

    aligned = False
    if rgb_shape is not None and depth_shape is not None:
        aligned = tuple(rgb_shape[:2]) == tuple(depth_shape[:2])
    print(f"rgb_depth_aligned: {aligned}")

    print()
    print("== meta.json ==")
    if meta is None:
        print("meta_status: unavailable")
        if meta_error is not None:
            print(f"meta_error: {meta_error}")
        missing_fields = list(REQUIRED_META_FIELDS)
    else:
        missing_fields = [field_name for field_name in REQUIRED_META_FIELDS if field_name not in meta]
        print("meta_status: loaded")
        print(f"meta_keys: {sorted(meta.keys())}")
        print(f"required_fields_present: {not missing_fields}")
        if missing_fields:
            print(f"missing_required_fields: {missing_fields}")
        else:
            print("missing_required_fields: []")

    is_valid = bool(
        scene_dir.exists()
        and scene_dir.is_dir()
        and rgb_exists
        and depth_exists
        and meta_exists
        and rgb_shape is not None
        and depth_shape is not None
        and aligned
        and not missing_fields
    )

    print()
    print("== Summary ==")
    print(f"scene_valid: {is_valid}")


if __name__ == "__main__":
    main()
