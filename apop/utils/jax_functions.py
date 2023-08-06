import jax
from jax import numpy as jnp


@jax.jit
def savgol_filter(
    x: jnp.ndarray,
    window_length: int,
    polyorder: int,
    deriv=0,
    delta=1.0,
    cval=0.0,
):
    pass


@jax.jit
def fit_angle_in_range(
    angles: jnp.ndarray, min_angle: float = -jnp.pi, max_angle: float = jnp.pi
) -> jnp.ndarray:
    """Check angle range and correct the range

    Args:
        angle (numpy.ndarray): in radians
        min_angle (float): maximum of range in radians, default -pi
        max_angle (float): minimum of range in radians, default pi
    Returns:
        fitted_angle (numpy.ndarray): range angle in radians
    """
    angles = angles - min_angle
    angles = angles % (2 * jnp.pi)
    angles = angles + (2 * jnp.pi)
    angles = angles % (2 * jnp.pi)
    angles = angles + min_angle
    angles = jnp.minimum(max_angle, jnp.maximum(min_angle, angles))
    return angles
