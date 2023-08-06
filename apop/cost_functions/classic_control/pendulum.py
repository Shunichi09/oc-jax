from functools import partial
from typing import Optional

import jax
import numpy as np
from jax import numpy as jnp

from apop.cost_function import CostFunction
from apop.utils.jax_functions import fit_angle_in_range


class PendulumCostFunction(CostFunction):
    def __init__(
        self, R: jnp.ndarray = jnp.eye(1) * 0.001, terminal_weight: float = 10.0
    ):
        super().__init__()
        self._R = R
        self._terminal_weight = terminal_weight

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_stage_cost(
        self, x: jnp.ndarray, u: Optional[jnp.ndarray], t: jnp.ndarray
    ):
        """Evaluate statge cost

        Args:
            x (jnp.ndarray): states, shape (state_size, )
            u (jnp.ndarray): inputs, shape (input_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            jnp:ndarray: stage cost, shape (1, )
        """
        state_cost = fit_angle_in_range(x[0]) ** 2 + 0.1 * (x[1] ** 2)
        input_cost = jnp.matmul(
            u[jnp.newaxis, :], jnp.matmul(self._R, u[:, jnp.newaxis])
        )
        return (state_cost + input_cost).ravel()

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_terminal_cost(self, x: jnp.ndarray, t: jnp.ndarray):
        """Evaluate terminal cost

        Args:
            x (jnp.ndarray): states, (state_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            jnp:ndarray: terminal cost, shape (1, )
        """
        return (
            fit_angle_in_range(x[0]) ** 2 + 0.1 * (x[1] ** 2)
        ) * self._terminal_weight
