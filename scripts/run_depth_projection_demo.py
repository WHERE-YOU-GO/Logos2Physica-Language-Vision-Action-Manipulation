from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import json

import numpy as np

from common.datatypes import BBox2D, Detection2D
from perception.depth_project import cam_point_to_base, detection_to_cam_point
from scripts._demo_support import build_synthetic_frame


def main() -> None:
    frame = build_synthetic_frame()
    detection = Detection2D(label="cube", score=0.99, bbox=BBox2D(280, 200, 340, 260))

    point_cam = detection_to_cam_point(detection, frame)
    point_base = cam_point_to_base(point_cam, frame.T_base_cam)

    payload = {
        "camera_point_m": point_cam.tolist(),
        "base_point_m": point_base.tolist(),
        "has_nan": bool(np.isnan(point_cam).any() or np.isnan(point_base).any()),
        "has_inf": bool(np.isinf(point_cam).any() or np.isinf(point_base).any()),
        "units": "meters",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
