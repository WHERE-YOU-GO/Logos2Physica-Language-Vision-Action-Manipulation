from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import argparse
import json
from pathlib import Path

import numpy as np

from common.path_manager import resolve_path
from scripts._demo_support import build_synthetic_frame, default_demo_meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a minimal replay scene asset bundle.")
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file without --overwrite: {path}"
        )


def _scene_meta() -> dict[str, object]:
    meta = default_demo_meta()
    meta.update(
        {
            "intrinsics": {
                "fx": 600.0,
                "fy": 600.0,
                "cx": 320.0,
                "cy": 240.0,
                "width": 640,
                "height": 480,
            },
            "T_base_cam": [
                [1.0, 0.0, 0.0, 0.35],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -0.51],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "timestamp": 0.0,
        }
    )
    return meta


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to save rgb.png.") from exc
    Image.fromarray(rgb, mode="RGB").save(path)


def main() -> None:
    args = _parse_args()
    scene_dir = resolve_path(args.scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = scene_dir / "rgb.png"
    depth_path = scene_dir / "depth.npy"
    meta_path = scene_dir / "meta.json"

    for path in (rgb_path, depth_path, meta_path):
        _ensure_writable(path, overwrite=args.overwrite)

    frame = build_synthetic_frame()
    rgb = np.asarray(frame.rgb, dtype=np.uint8)
    depth = np.asarray(frame.depth, dtype=np.float32)
    meta = _scene_meta()

    _save_rgb(rgb_path, rgb)
    np.save(depth_path, depth)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    print("Generated replay scene assets:")
    print(f"- {rgb_path}")
    print(f"- {depth_path}")
    print(f"- {meta_path}")
    print()
    print("Recommended next step:")
    print(f"python -m scripts.validate_scene_dir --scene_dir {scene_dir}")


if __name__ == "__main__":
    main()
