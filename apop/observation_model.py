from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Union

import jax
from jax import numpy as jnp

from apop.distribution import Distribution


class ObservationModel(metaclass=ABCMeta):
    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def observe_batched(self, curr_x: jnp.ndarray) -> jnp.ndarray:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (batch_size, state_size)

        Returns:
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def observe(self, curr_x: jnp.ndarray) -> jnp.ndarray:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (batch_size, state_size)

        Returns:
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def observe_distribution(self, curr_x: jnp.ndarray) -> Distribution:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (batch_size, state_size)

        Returns:
        """
        raise NotImplementedError
