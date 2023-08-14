from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Optional

import jax
from jax import numpy as jnp


class Distribution(metaclass=ABCMeta):
    _key: jax.random.KeyArray

    def __init__(self, key: jax.random.KeyArray) -> None:
        self._key = key

    @abstractmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def sample(self, num_samples: int) -> jnp.ndarray:
        """sample variable from the distribution

        Args:
            num_samples (int): num_samples

        Returns:
            jnp:ndarray: sampled variables, shape (num_samples, state_size)
        """
        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def probability(self, x: jnp.ndarray) -> jnp.ndarray:
        """compute probability of the given variable

        Args:
            x (jnp.ndarray): variables, shape (num_samples, state_size)

        Returns:
            jnp:ndarray: probabilities, shape (num_samples, )
        """
        raise NotImplementedError
