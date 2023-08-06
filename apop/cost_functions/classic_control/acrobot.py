from functools import partial
from typing import Optional

import jax
from jax import numpy as jnp

from apop.cost_function import CostFunction
from apop.utils.jax_functions import fit_angle_in_range


class AcrobotCostFunction(CostFunction):
    def __init__(
        self,
        Q: jnp.ndarray,
        Qf: jnp.ndarray,
        R: jnp.ndarray,
        target_state: Optional[jnp.ndarray] = None,
    ):
        super().__init__()
        self._Q = Q
        self._Qf = Qf
        self._R = R
        self._target_state = target_state

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
        diff = self._compute_diff(x)
        state_cost = jnp.matmul(
            diff[jnp.newaxis, :], jnp.matmul(self._Q, diff[:, jnp.newaxis])
        )
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
        diff = self._compute_diff(x)
        state_cost = jnp.matmul(
            diff[jnp.newaxis, :], jnp.matmul(self._Qf, diff[:, jnp.newaxis])
        )
        return state_cost

    @partial(jax.jit, static_argnums=(0,))
    def _compute_diff(self, x: jnp.ndarray) -> jnp.ndarray:
        diff = jnp.empty_like(x)
        diff = diff.at[0].set(fit_angle_in_range(self._target_state[0] - x[0]))
        diff = diff.at[1].set(fit_angle_in_range(self._target_state[1] - x[1]))
        diff = diff.at[2].set(self._target_state[2] - x[2])
        diff = diff.at[3].set(self._target_state[3] - x[3])
        return diff
