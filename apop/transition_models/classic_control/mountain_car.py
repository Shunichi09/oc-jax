from functools import partial

import jax
from jax import numpy as jnp

from apop.transition_model import DeterministicTransitionModel


class MountainCarModel(DeterministicTransitionModel):
    def __init__(
        self,
        u_max: float = 1.0,
        u_min: float = -1.0,
        max_speed: float = 0.07,
        min_speed: float = -0.07,
        min_position: float = -1.2,
        max_position: float = 0.6,
        power: float = 0.0015,
    ):
        super().__init__()
        self._u_max = u_max
        self._u_min = u_min
        self._max_speed = max_speed
        self._min_speed = min_speed
        self._max_position = max_position
        self._min_position = min_position
        self._power = power

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
        position = x[0]
        velocity = x[1]
        force = jnp.clip(u[0], a_max=self._u_max, a_min=self._u_min)

        velocity = velocity + force * self._power - 0.0025 * jnp.cos(3 * position)
        velocity = jnp.clip(velocity, a_min=self._min_speed, a_max=self._max_speed)
        position = position + velocity
        position = jnp.clip(
            position, a_min=self._min_position, a_max=self._max_position
        )

        velocity = (
            jnp.float32(
                jnp.logical_and(position > self._min_position, velocity > 0),
            )
            * velocity
        )

        next_x = jnp.empty_like(x)
        next_x = next_x.at[0].set(position)
        next_x = next_x.at[1].set(velocity)
        return next_x
