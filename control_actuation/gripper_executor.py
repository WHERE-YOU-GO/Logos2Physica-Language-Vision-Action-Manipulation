from __future__ import annotations

from typing import Any

from common.logger import ProjectLogger
from control_actuation.lite6_adapter import Lite6Adapter


class GripperExecutor:
    def __init__(self, robot_adapter: Lite6Adapter, logger: ProjectLogger) -> None:
        self._robot_adapter = robot_adapter
        self._logger = logger

    def open(self) -> None:
        self._robot_adapter.open_gripper()

    def close(self) -> None:
        self._robot_adapter.close_gripper()

    def set_width(self, width_m: float) -> None:
        self._robot_adapter.set_gripper_width(float(width_m))

    def get_state(self) -> dict:
        return self._robot_adapter.get_gripper_state()

    def is_holding_object(self) -> bool:
        state = self.get_state()
        if "is_holding" in state:
            return bool(state["is_holding"])
        width_m = state.get("width_m")
        if width_m is None:
            return False
        width_m = float(width_m)
        return 0.002 < width_m < 0.08
