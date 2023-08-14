from abc import ABCMeta
from functools import partial
from typing import Optional

import jax
from jax import numpy as jnp

from apop.distribution import Distribution


class ObservationModel(metaclass=ABCMeta):
    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def observe_batched(
        self, curr_x: jnp.ndarray, observation_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (batch_size, state_size)
            observation_mask (jnp.ndarray): observation mask, shape (batch_size, )

        Returns:
            jnp.ndarray: observation state, shape (batch_size, observation_size)
        """
        batched_obs_func = jax.vmap(self.observe, in_axes=(0, None), out_axes=0)
        return batched_obs_func(curr_x, observation_mask)

    @partial(jax.jit, static_argnums=(0,))
    def observe(
        self, curr_x: jnp.ndarray, observation_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )
            observation_mask (jnp.ndarray): observation mask

        Returns:
            jnp.ndarray: observation state, shape (observation_size, )
        """
        raise NotImplementedError


class DeterministicObservationModel(ObservationModel, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def is_deterministic(self):
        return True


class ProbabilisticObservationModel(ObservationModel, metaclass=ABCMeta):
    _key: jax.random.KeyArray

    def __init__(self, key: jax.random.KeyArray):
        super().__init__()
        self._key = key

    def is_probabilistic(self):
        return True

    def observe_distribution(
        self, curr_x: jnp.ndarray, observation_mask: Optional[jnp.ndarray] = None
    ) -> Distribution:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )
            observation_mask (jnp.ndarray): observation mask
        Returns:
            Distribution: observation state distribution
        """
        raise NotImplementedError
