from __future__ import annotations

import numpy as np

from common.datatypes import Detection2D, RGBDFrame, SceneObject
from common.exceptions import PerceptionError
from common.geometry import deproject_uv_to_cam, transform_point
from sensing.depth_filter import depth_patch_median


def detection_to_cam_point(det: Detection2D, frame: RGBDFrame) -> np.ndarray:
    depth = np.asarray(frame.depth, dtype=np.float32)
    try:
        depth_m = depth_patch_median(depth, det.bbox, shrink_ratio=0.3)
    except ValueError as exc:
        raise PerceptionError(f"Failed to compute a valid depth patch for detection {det.label!r}.") from exc
    uv = det.bbox.center_uv()
    try:
        return deproject_uv_to_cam(uv, depth_m, frame.intrinsics)
    except Exception as exc:
        raise PerceptionError(f"Failed to deproject detection {det.label!r} into camera coordinates.") from exc


def cam_point_to_base(point_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    try:
        return transform_point(T_base_cam, point_cam)
    except Exception as exc:
        raise PerceptionError("Failed to transform camera-frame point into base frame.") from exc


def detection_to_scene_object(det: Detection2D, frame: RGBDFrame, object_id: str) -> SceneObject:
    center_cam = detection_to_cam_point(det, frame)
    center_base = cam_point_to_base(center_cam, frame.T_base_cam)
    return SceneObject(
        object_id=object_id,
        label=det.label,
        bbox=det.bbox,
        center_cam=center_cam,
        center_base=center_base,
        confidence=det.score,
        color=det.extras.get("color"),
        shape=det.extras.get("shape"),
        extras={
            "phrase": det.phrase,
            "backend": det.extras.get("backend"),
            "category": det.extras.get("category", det.label),
        },
    )
