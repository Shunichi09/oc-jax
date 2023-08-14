from functools import partial

import jax
from jax import numpy as jnp

from apop.transition_model import ProbabilisticTransitionModel
from apop.distributions.gaussian import Gaussian
from apop.random import new_key


class Ball2dTrackingGaussianTransitionModel(ProbabilisticTransitionModel):
    def __init__(
        self,
        dt: float,
        covariance: jnp.ndarray,
        key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ):
        super().__init__(key)
        self._dt = dt
        self._covariance = covariance

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
        distribution = self.predict_next_state_distribution(x, u, t)
        return distribution.sample(1)[0]

    def predict_next_state_distribution(
        self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray
    ) -> Gaussian:
        """predict next state

        Args:
            x (jnp.ndarray): states, shape (state_size, )
            u (jnp.ndarray): input, shape (input_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            Distribution: next state distribution
        """
        pos_x = u[0] * self._dt * jnp.cos(x[2]) + x[0]
        pos_y = u[0] * self._dt * jnp.sin(x[2]) + x[1]
        pos_theta = u[1] * self._dt + x[2]
        mean = jnp.array([pos_x, pos_y, pos_theta])
        self._key = new_key(self._key)
        return Gaussian(self._key, mean, self._covariance)
