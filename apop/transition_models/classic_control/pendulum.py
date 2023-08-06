from functools import partial

import jax
from jax import numpy as jnp

from apop.transition_model import TransitionModel


class PendulumModel(TransitionModel):
    def __init__(
        self,
        max_speed: float = 8.0,
        max_torque: float = 2.0,
        g: float = 10.0,
        m: float = 1.0,
        length: float = 1.0,
        dt: float = 0.05,
    ):
        super().__init__()
        self._max_speed = max_speed
        self._max_torque = max_torque
        self._g = g
        self._m = m
        self._length = length
        self._dt = dt

    @partial(jax.jit, static_argnums=(0,))
    def predict_next_state(
        self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """predict next state

        Args:
            x (jnp.ndarray): states, shape (state_size, )
            u (jnp.ndarray): input, shape (input_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            next_x (jnp.ndarray): next state, shape (state_size, )
        """
        # x = theta, theta_dot
        th = x[0]  # th := theta
        thdot = x[1]
        u = u[0]
        newthdot = (
            thdot
            + (
                3 * self._g / (2 * self._length) * jnp.sin(th)
                + 3.0 / (self._m * self._length**2) * u
            )
            * self._dt
        )

        next_x = jnp.empty_like(x)
        next_x = next_x.at[0].set(x[0] + self._dt * newthdot)
        next_x = next_x.at[1].set(newthdot)
        return next_x
