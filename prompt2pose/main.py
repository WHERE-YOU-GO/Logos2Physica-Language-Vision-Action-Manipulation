"""End-to-end Prompt2Pose pipeline + CLI entry point.

Usage:
    python -m prompt2pose.main --prompt "Put the red cube on the blue block"

Pipeline:
    1. Parse prompt with LLM API           -> ParsedCommand
    2. Sense scene with ZED                -> RGBA image + XYZ point cloud
    3. Calibrate base->cam with AprilTags  -> 4x4 transform
    4. Detect pick + place objects         -> XYZ in robot base frame
    5. Execute grasp + place on Lite6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from prompt2pose.camera import ZedCamera, compute_T_cam_robot
from prompt2pose.detector import Detection, HSVDetector, OWLViTDetector
from prompt2pose.parser import LLMCommandParser, ParsedCommand
from prompt2pose.robot import Lite6


# ---- Default workspace constants --------------------------------------------
DEFAULT_PLACE_FALLBACK_XY = (0.20, 0.0)   # safe drop pose if place_target is unknown
CUBE_HALF_HEIGHT_M = 0.0125               # ~25 mm cube
STACK_OFFSET_M = 0.025                    # cube edge for stacking


class PromptPipeline:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
        self.cfg = cfg

        llm = cfg["llm"]
        self.parser = LLMCommandParser(
            api_key=llm["api_key"],
            model=llm.get("model", "gpt-4o-mini"),
            base_url=llm.get("base_url"),
            temperature=float(llm.get("temperature", 0.0)),
        )

        det_cfg = cfg.get("detector", {})
        backend = det_cfg.get("backend", "owlvit").lower()
        if backend == "owlvit":
            self.detector = OWLViTDetector(
                model_id=det_cfg.get("model_id", "google/owlv2-base-patch16-ensemble"),
                threshold=float(det_cfg.get("threshold", 0.1)),
            )
        elif backend == "hsv":
            self.detector = HSVDetector(min_area_px=int(det_cfg.get("min_area_px", 200)))
        else:
            raise ValueError(f"Unknown detector backend: {backend}")

        self.camera = ZedCamera(fps=int(cfg.get("camera", {}).get("fps", 15)))
        self.robot = Lite6(robot_ip=cfg["robot"]["ip"])
        self.T_cam_robot: Optional[np.ndarray] = None

    # ---- setup ---------------------------------------------------------------
    def setup(self) -> None:
        self.robot.connect()
        self.robot.home()
        # Step back so the camera has a clean view of the workspace.
        time.sleep(0.5)
        for attempt in range(5):
            T = compute_T_cam_robot(self.camera.image, self.camera.intrinsic)
            if T is not None:
                self.T_cam_robot = T
                print(f"[setup] base->cam calibration ok on attempt {attempt + 1}")
                return
            time.sleep(0.3)
        raise RuntimeError("AprilTag calibration failed; cannot proceed")

    # ---- one command ---------------------------------------------------------
    def execute(self, prompt: str) -> dict:
        cmd = self.parser.parse(prompt)
        print(f"[parser] {cmd.as_dict()}")
        result = {"command": cmd.as_dict(), "status": "fail", "reason": None}

        image = self.camera.image
        cloud = self.camera.point_cloud
        assert self.T_cam_robot is not None

        pick = self.detector.locate(image, cloud, cmd.pick_object, self.T_cam_robot)
        if pick is None:
            result["reason"] = f"could not see pick_object {cmd.pick_object!r}"
            return result
        print(f"[perception] pick {cmd.pick_object!r} at base xyz = {pick.xyz_base.round(3)}")

        place_xyz, stack_offset = self._resolve_place_pose(cmd, pick, image, cloud)
        if place_xyz is None:
            result["reason"] = f"could not resolve place_target {cmd.place_target!r}"
            return result
        print(f"[planner] place at base xyz = {place_xyz.round(3)} (stack_offset={stack_offset:.3f})")

        # Execute on hardware. Comment out the next two lines for a dry run.
        self.robot.grasp_at(pick.xyz_base)
        self.robot.place_at(place_xyz, stack_offset_m=stack_offset)
        self.robot.home()

        result["status"] = "success"
        return result

    # ---- place-pose resolution ----------------------------------------------
    def _resolve_place_pose(
        self,
        cmd: ParsedCommand,
        pick_det: Detection,
        image: np.ndarray,
        cloud: np.ndarray,
    ) -> tuple[Optional[np.ndarray], float]:
        if cmd.place_target is None:
            x, y = DEFAULT_PLACE_FALLBACK_XY
            return np.array([x, y, CUBE_HALF_HEIGHT_M]), 0.0

        assert self.T_cam_robot is not None
        place_det = self.detector.locate(image, cloud, cmd.place_target, self.T_cam_robot)
        if place_det is None:
            return None, 0.0

        target_xyz = place_det.xyz_base.copy()

        if cmd.place_relation == "on":
            stack = STACK_OFFSET_M
            return target_xyz, stack
        if cmd.place_relation == "in":
            return target_xyz, 0.0
        return target_xyz, 0.0

    # ---- shutdown ------------------------------------------------------------
    def shutdown(self) -> None:
        try:
            self.robot.disconnect()
        finally:
            self.camera.close()


def main() -> int:
    cli = argparse.ArgumentParser(description="Prompt2Pose: language-commanded pick & place on Lite6")
    cli.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    cli.add_argument("--prompt", required=True, help="Natural language instruction")
    args = cli.parse_args()

    pipeline = PromptPipeline(args.config)
    try:
        pipeline.setup()
        result = pipeline.execute(args.prompt)
        print(f"[result] {result}")
        return 0 if result["status"] == "success" else 1
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    sys.exit(main())
