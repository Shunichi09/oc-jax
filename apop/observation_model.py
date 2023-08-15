from abc import ABCMeta
from functools import partial

import jax
from jax import numpy as jnp

from apop.distribution import Distribution


class ObservationModel(metaclass=ABCMeta):
    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def observe_batched(
        self,
        curr_x: jnp.ndarray,
        observation_mask: jnp.ndarray,
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (batch_size, state_size)
            observation_mask (jnp.ndarray): observation mask, shape (batch_size, )
            random_key (jax.random.KeyArray): random key

        Returns:
            jnp.ndarray: observation state, shape (batch_size, observation_size)
        """
        batched_obs_func = jax.vmap(self.observe, in_axes=(0, None, 0), out_axes=0)
        return batched_obs_func(
            curr_x, observation_mask, jax.random.split(random_key, curr_x.shape[0])
        )

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
            observation_mask (jnp.ndarray): observation mask
            random_key (jax.random.KeyArray): random key

        Returns:
            jnp.ndarray: observation state, shape (observation_size, )
        """
        raise NotImplementedError

    def observe_distribution(
        self, curr_x: jnp.ndarray, observation_mask: jnp.ndarray
    ) -> Distribution:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )
            observation_mask (jnp.ndarray): observation mask

        Returns:
            Distribution: observation state distribution
        """
        raise NotImplementedError
