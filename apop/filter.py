from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, Optional, Union

import jax
from jax import numpy as jnp

from apop.cost_function import CostFunction
from apop.transition_model import TransitionModel
from apop.observation_model import ObservationModel


class Filter(metaclass=ABCMeta):
    _transition_model: TransitionModel
    _observation_model: ObservationModel

    def __init__(
        self, transition_model: TransitionModel, observation_model: ObservationModel
    ) -> None:
        self._transition_model = transition_model
        self._observation_model = observation_model

    @abstractmethod
    def predict(self, curr_u: jnp.ndarray) -> jnp.ndarray:
        """prediction step

        Args:
            curr_u (jnp.ndarray): inputs
        """
        raise NotImplementedError

    @abstractmethod
    def estimate(self, curr_y: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """filtering step

        Args:
            curr_y (jnp.ndarray): observation
            mask (jnp.ndarray): mask, 1 means can observe, 0 means can not observe.

        Returns:
            jnp.ndarray: estimated state
        """
        raise NotImplementedError
