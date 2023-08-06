from typing import Sequence

import numpy as np


def fit_angle_in_range(
    angles: Sequence[float], min_angle: float = -np.pi, max_angle: float = np.pi
) -> np.ndarray:
    """Check angle range and correct the range

    Args:
        angle (numpy.ndarray): in radians
        min_angle (float): maximum of range in radians, default -pi
        max_angle (float): minimum of range in radians, default pi
    Returns:
        fitted_angle (numpy.ndarray): range angle in radians
    """
    if max_angle < min_angle:
        raise ValueError("max angle must be greater than min angle")
    if (max_angle - min_angle) < 2.0 * np.pi:
        raise ValueError(
            "difference between max_angle \
                          and min_angle must be greater than 2.0 * pi"
        )

    output = np.array(angles)
    output_shape = output.shape

    output = output.flatten()
    output -= min_angle
    output %= 2 * np.pi
    output += 2 * np.pi
    output %= 2 * np.pi
    output += min_angle

    output = np.minimum(max_angle, np.maximum(min_angle, output))
    return output.reshape(output_shape)
