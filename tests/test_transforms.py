from __future__ import annotations

import numpy as np

from common.transforms import compose_transform, invert_transform, is_valid_transform, make_transform


def test_make_and_validate_transform() -> None:
    T = make_transform(np.eye(3, dtype=np.float64), np.array([0.1, 0.2, 0.3], dtype=np.float64))
    assert is_valid_transform(T) is True


def test_invert_transform() -> None:
    T = make_transform(np.eye(3, dtype=np.float64), np.array([0.1, 0.2, 0.3], dtype=np.float64))
    T_inv = invert_transform(T)
    assert np.allclose(T_inv @ T, np.eye(4, dtype=np.float64))


def test_compose_transform() -> None:
    T_ab = make_transform(np.eye(3, dtype=np.float64), np.array([0.1, 0.0, 0.0], dtype=np.float64))
    T_bc = make_transform(np.eye(3, dtype=np.float64), np.array([0.0, 0.2, 0.0], dtype=np.float64))
    T_ac = compose_transform(T_ab, T_bc)
    assert np.allclose(T_ac[:3, 3], np.array([0.1, 0.2, 0.0], dtype=np.float64))


def test_invalid_transform_detection() -> None:
    invalid = np.eye(4, dtype=np.float64)
    invalid[3, 3] = 2.0
    assert is_valid_transform(invalid) is False
