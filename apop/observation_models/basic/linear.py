from functools import partial

import jax
from jax import numpy as jnp

from apop.distributions.gaussian import Gaussian
from apop.observation_model import ObservationModel


class LinearObservationModel(ObservationModel):
    """linear observation model"""

    _C: jnp.ndarray

    def __init__(self, C: jnp.ndarray):
        super().__init__()
        self._C = C

    @partial(jax.jit, static_argnums=(0,))
    def observe(
        self,
        curr_x: jnp.ndarray,
        observation_mask: jnp.ndarray,
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )

        Returns:
            jnp.ndarray: observation state, shape (observation_size, )
        """
        return jnp.matmul(self._C, curr_x[:, jnp.newaxis]).ravel()


class LinearGaussianObservationModel(ObservationModel):
    """linear observation model with gaussian dist with fixed covariance"""

    def __init__(
        self,
        C: jnp.ndarray,
        covariance: jnp.ndarray,
    ):
        super().__init__()
        self._C = C
        self._covariance = covariance

    @partial(jax.jit, static_argnums=(0,))
    def observe(
        self,
        curr_x: jnp.ndarray,
        observation_mask: jnp.ndarray,
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )

        Returns:
            jnp.ndarray: observation state, shape (observation_size, )
        """
        return self.observe_distribution(curr_x, observation_mask).sample(
            random_key, 1
        )[0]

    def observe_distribution(
        self, curr_x: jnp.ndarray, observation_mask: jnp.ndarray
    ) -> Gaussian:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )

        Returns:
            Gaussian: gaussian distribution
        """
        mean = jnp.matmul(self._C, curr_x[:, jnp.newaxis]).ravel()
        dist = Gaussian(mean=mean, full_covariance=self._covariance)
        return dist
