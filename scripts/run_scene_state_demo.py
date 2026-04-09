from __future__ import annotations

import numpy as np

from common.datatypes import BBox2D, CameraIntrinsics, Detection2D, RGBDFrame
from common.logger import ProjectLogger
from perception.scene_state import SceneStateBuilder


def main() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[200:260, 280:340] = np.array([255, 0, 0], dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.float32) * 0.55
    intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    frame = RGBDFrame(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        T_base_cam=np.eye(4, dtype=np.float64),
        timestamp=0.0,
    )
    detections = [Detection2D(label="block", score=0.93, bbox=BBox2D(280, 200, 340, 260))]
    builder = SceneStateBuilder(logger=ProjectLogger("logs/scene_state_demo"))
    scene_state = builder.build(frame, detections)
    print(builder.summarize_for_llm(scene_state))


if __name__ == "__main__":
    main()
