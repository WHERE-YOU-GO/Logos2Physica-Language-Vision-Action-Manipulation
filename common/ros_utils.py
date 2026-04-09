from __future__ import annotations

import time
from typing import Any

import numpy as np

from common.datatypes import Pose3D

try:
    from builtin_interfaces.msg import Time as RosTime
    from geometry_msgs.msg import PoseStamped
except ImportError:  # pragma: no cover
    RosTime = None  # type: ignore[assignment]
    PoseStamped = None  # type: ignore[assignment]


def ros_time_now() -> float:
    return time.time()


def pose3d_to_pose_stamped(pose: Pose3D):
    timestamp = ros_time_now()
    if PoseStamped is None or RosTime is None:
        return {
            "header": {"stamp": timestamp, "frame_id": pose.frame_id},
            "pose": {
                "position": {
                    "x": float(pose.position[0]),
                    "y": float(pose.position[1]),
                    "z": float(pose.position[2]),
                },
                "orientation": {
                    "x": float(pose.quaternion[0]),
                    "y": float(pose.quaternion[1]),
                    "z": float(pose.quaternion[2]),
                    "w": float(pose.quaternion[3]),
                },
            },
        }

    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1_000_000_000)
    msg = PoseStamped()
    msg.header.stamp = RosTime(sec=sec, nanosec=nanosec)
    msg.header.frame_id = pose.frame_id
    msg.pose.position.x = float(pose.position[0])
    msg.pose.position.y = float(pose.position[1])
    msg.pose.position.z = float(pose.position[2])
    msg.pose.orientation.x = float(pose.quaternion[0])
    msg.pose.orientation.y = float(pose.quaternion[1])
    msg.pose.orientation.z = float(pose.quaternion[2])
    msg.pose.orientation.w = float(pose.quaternion[3])
    return msg


def pose_stamped_to_pose3d(msg: Any) -> Pose3D:
    if isinstance(msg, dict):
        position = np.array(
            [
                msg["pose"]["position"]["x"],
                msg["pose"]["position"]["y"],
                msg["pose"]["position"]["z"],
            ],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                msg["pose"]["orientation"]["x"],
                msg["pose"]["orientation"]["y"],
                msg["pose"]["orientation"]["z"],
                msg["pose"]["orientation"]["w"],
            ],
            dtype=np.float64,
        )
        frame_id = str(msg["header"]["frame_id"])
        return Pose3D(position=position, quaternion=quaternion, frame_id=frame_id)

    return Pose3D(
        position=np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        ),
        quaternion=np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        ),
        frame_id=str(msg.header.frame_id),
    )
