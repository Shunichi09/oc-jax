from functools import partial

import jax
from jax import numpy as jnp

from apop.distribution import Distribution
from apop.distributions.gaussian import Gaussian
from apop.transition_model import TransitionModel


class SimpleTwoWheeleDistribution(Distribution):
    def __init__(
        self,
        curr_x: jnp.ndarray,
        curr_u: jnp.ndarray,
        input_covariance: jnp.ndarray,
        dt: float,
    ) -> None:
        super().__init__()
        self._curr_x = curr_x
        self._curr_u = Gaussian(curr_u, input_covariance)
        self._dt = dt

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self, random_key: jax.random.KeyArray, num_samples: int) -> jnp.ndarray:
        curr_u = self._curr_u.sample(random_key, num_samples)
        pos_x = curr_u[:, 0] * self._dt * jnp.cos(self._curr_x[2]) + self._curr_x[0]
        pos_y = curr_u[:, 0] * self._dt * jnp.sin(self._curr_x[2]) + self._curr_x[1]
        pos_theta = curr_u[:, 1] * self._dt + self._curr_x[2]
        sampled = jnp.concatenate(
            [pos_x[:, jnp.newaxis], pos_y[:, jnp.newaxis], pos_theta[:, jnp.newaxis]],
            axis=1,
        )
        return sampled


class Ball2dTrackingGaussianTransitionModel(TransitionModel):
    def __init__(self, dt: float, input_covariance: jnp.ndarray):
        super().__init__()
        self._dt = dt
        self._input_covariance = input_covariance

    @partial(jax.jit, static_argnums=(0,))
    def predict_next_state(
        self,
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: jnp.ndarray,
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """predict next state

        Args:
            x (jnp.ndarray): states, shape (state_size, )
            u (jnp.ndarray): input, shape (input_size, )
            t (jnp.ndarray): time step, shape (1, )
            random_key (jax.random.KeyArray): random key

        Returns:
            next_x (jnp.ndarray): next state, shape (state_size, )
        """
        distribution = self.predict_next_state_distribution(x, u, t)
        return distribution.sample(random_key, 1)[0]

    def predict_next_state_distribution(
        self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray
    ) -> SimpleTwoWheeleDistribution:
        """predict next state

        Args:
            x (jnp.ndarray): states, shape (state_size, )
            u (jnp.ndarray): input, shape (input_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            Distribution: next state distribution
        """
        return SimpleTwoWheeleDistribution(x, u, self._input_covariance, self._dt)
