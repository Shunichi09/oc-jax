from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, Optional, Union

import jax
from jax import numpy as jnp

from apop.cost_function import CostFunction
from apop.transition_model import TransitionModel


class Filter(metaclass=ABCMeta):
    _transition_model: TransitionModel

    def __init__(self, transition_model: TransitionModel) -> None:
        self._transition_model = transition_model

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def estimate(
        self,
        curr_y: jnp.ndarray,
        curr_u: jnp.ndarray,
    ) -> jnp.ndarray:
        raise NotImplementedError
