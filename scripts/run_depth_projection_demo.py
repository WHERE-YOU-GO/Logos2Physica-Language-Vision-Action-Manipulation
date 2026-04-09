from __future__ import annotations

import numpy as np

from common.datatypes import BBox2D, CameraIntrinsics, Detection2D, RGBDFrame
from perception.depth_project import cam_point_to_base, detection_to_cam_point


def main() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32) * 0.6
    intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    T_base_cam = np.eye(4, dtype=np.float64)
    T_base_cam[:3, 3] = np.array([0.2, 0.0, 0.5], dtype=np.float64)
    frame = RGBDFrame(rgb=rgb, depth=depth, intrinsics=intrinsics, T_base_cam=T_base_cam, timestamp=0.0)
    det = Detection2D(label="block", score=0.95, bbox=BBox2D(300, 220, 340, 260))

    point_cam = detection_to_cam_point(det, frame)
    point_base = cam_point_to_base(point_cam, frame.T_base_cam)

    print("camera_point_m:", point_cam.tolist())
    print("base_point_m:", point_base.tolist())


if __name__ == "__main__":
    main()
