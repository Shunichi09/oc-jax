from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, Optional, Union

import jax
from jax import numpy as jnp

from apop.cost_function import CostFunction
from apop.transition_model import TransitionModel


class Controller(metaclass=ABCMeta):
    _transition_model: TransitionModel
    _cost_function: CostFunction

    def __init__(
        self, transition_model: TransitionModel, cost_function: CostFunction
    ) -> None:
        self._transition_model = transition_model
        self._cost_function = cost_function

    @abstractmethod
    def control(
        self,
        curr_x: jnp.ndarray,
        initial_u_sequence: jnp.ndarray,
    ) -> jnp.ndarray:
        raise NotImplementedError
