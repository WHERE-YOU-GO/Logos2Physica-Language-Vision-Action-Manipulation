from __future__ import annotations

import numpy as np


def _as_matrix(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _as_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _is_rotation_matrix(R: np.ndarray, atol: float = 1e-6) -> bool:
    if R.shape != (3, 3):
        return False
    if not np.all(np.isfinite(R)):
        return False
    should_be_identity = R.T @ R
    if not np.allclose(should_be_identity, np.eye(3, dtype=np.float64), atol=atol):
        return False
    determinant = np.linalg.det(R)
    return bool(np.isclose(determinant, 1.0, atol=atol))


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert an SE(3) homogeneous transform."""

    matrix = _as_matrix(T, (4, 4), "T")
    if not is_valid_transform(matrix):
        raise ValueError("T is not a valid homogeneous transform.")

    R = matrix[:3, :3]
    t = matrix[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ t)
    return T_inv


def compose_transform(T_ab: np.ndarray, T_bc: np.ndarray) -> np.ndarray:
    """Compose two SE(3) homogeneous transforms."""

    left = _as_matrix(T_ab, (4, 4), "T_ab")
    right = _as_matrix(T_bc, (4, 4), "T_bc")
    if not is_valid_transform(left):
        raise ValueError("T_ab is not a valid homogeneous transform.")
    if not is_valid_transform(right):
        raise ValueError("T_bc is not a valid homogeneous transform.")
    return left @ right


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from rotation and translation."""

    rotation = _as_matrix(R, (3, 3), "R")
    translation = _as_vector(t, 3, "t")
    if not _is_rotation_matrix(rotation):
        raise ValueError("R must be a valid rotation matrix.")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def is_valid_transform(T: np.ndarray) -> bool:
    """Return True when T is a numerically valid SE(3) homogeneous transform."""

    try:
        matrix = _as_matrix(T, (4, 4), "T")
    except ValueError:
        return False

    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), atol=1e-6):
        return False

    return bool(_is_rotation_matrix(matrix[:3, :3]))


__all__ = [
    "invert_transform",
    "compose_transform",
    "make_transform",
    "is_valid_transform",
]
