from __future__ import annotations

from pathlib import Path
from typing import Any

from common.config_loader import load_robot_config
from common.datatypes import Pose3D
from common.exceptions import ExecutionError, ProjectError
from common.logger import ProjectLogger


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class IKSolver:
    def __init__(self, robot_config_path: str, logger: ProjectLogger) -> None:
        self._logger = logger
        self._robot_config = self._safe_load(robot_config_path)
        self._backend_name = str(self._robot_config.get("ik_backend", "")).strip().lower()

    def _safe_load(self, path: str) -> dict[str, Any]:
        try:
            return load_robot_config(str(Path(path)))
        except ProjectError as exc:
            _log(self._logger, "warn", f"Falling back to default IK config: {exc}")
            return {}

    def solve(self, target_pose: Pose3D, seed_joints=None):
        _ = seed_joints
        if self._backend_name:
            raise RuntimeError(
                f"IK backend {self._backend_name!r} is configured but not implemented in this MVP placeholder."
            )
        raise RuntimeError(
            "No IK backend is configured. Integrate a real IK solver before calling IKSolver.solve()."
        )

    def solve_sequence(self, poses: list[Pose3D], seed_joints=None) -> list:
        solutions = []
        current_seed = seed_joints
        for pose in poses:
            try:
                solution = self.solve(pose, seed_joints=current_seed)
            except Exception as exc:
                raise ExecutionError("Failed to solve IK for a pose sequence.") from exc
            solutions.append(solution)
            current_seed = solution
        return solutions
