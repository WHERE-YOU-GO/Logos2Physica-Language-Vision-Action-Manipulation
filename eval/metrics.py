from __future__ import annotations

from statistics import mean, median

import numpy as np


def grounding_accuracy(success_flags: list[bool]) -> float:
    if not success_flags:
        return 0.0
    return float(sum(bool(flag) for flag in success_flags) / len(success_flags))


def translation_error_cm(pred_xyz_list, gt_xyz_list) -> float:
    if len(pred_xyz_list) != len(gt_xyz_list):
        raise ValueError("pred_xyz_list and gt_xyz_list must have the same length.")
    if not pred_xyz_list:
        return 0.0

    errors_cm = []
    for pred_xyz, gt_xyz in zip(pred_xyz_list, gt_xyz_list, strict=True):
        pred = np.asarray(pred_xyz, dtype=np.float64).reshape(3)
        gt = np.asarray(gt_xyz, dtype=np.float64).reshape(3)
        errors_cm.append(float(np.linalg.norm(pred - gt) * 100.0))
    return float(mean(errors_cm))


def task_success_rate(success_flags: list[bool]) -> float:
    return grounding_accuracy(success_flags)


def planning_time_stats(times_s: list[float]) -> dict:
    if not times_s:
        return {"count": 0, "mean_s": 0.0, "median_s": 0.0, "max_s": 0.0, "min_s": 0.0}
    values = [float(value) for value in times_s]
    return {
        "count": len(values),
        "mean_s": float(mean(values)),
        "median_s": float(median(values)),
        "max_s": float(max(values)),
        "min_s": float(min(values)),
    }
