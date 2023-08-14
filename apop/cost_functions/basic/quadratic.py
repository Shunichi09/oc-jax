from functools import partial
from typing import Optional

import jax
from jax import numpy as jnp

from apop.cost_function import CostFunction


class QuadraticCostFunction(CostFunction):
    def __init__(
        self,
        Q: jnp.ndarray,
        Qf: jnp.ndarray,
        R: jnp.ndarray,
        F: jnp.ndarray,
        target_state: Optional[jnp.ndarray] = None,
    ):
        super().__init__()
        assert Q.shape == Qf.shape
        assert F.shape[0] == Q.shape[0]
        assert F.shape[1] == R.shape[0]
        self._Q = Q
        self._Qf = Qf
        self._R = R
        self._F = F
        self._target_state = (
            target_state if target_state is not None else jnp.zeros((Q.shape[0]))
        )
        assert Q.shape[0] == self._target_state.shape[0]

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
        state_cost = jnp.matmul(
            (x - self._target_state)[jnp.newaxis, :],
            jnp.matmul(self._Q, (x - self._target_state)[:, jnp.newaxis]),
        )
        cross_cost = 2.0 * jnp.matmul(
            (x - self._target_state)[jnp.newaxis, :],
            jnp.matmul(self._F, u[:, jnp.newaxis]),
        )
        input_cost = jnp.matmul(
            u[jnp.newaxis, :], jnp.matmul(self._R, u[:, jnp.newaxis])
        )
        return (state_cost + cross_cost + input_cost).ravel()

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_terminal_cost(self, x: jnp.ndarray, t: jnp.ndarray):
        """Evaluate terminal cost

        Args:
            x (jnp.ndarray): states, (state_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            jnp:ndarray: terminal cost, shape (1, )
        """
        return jnp.matmul(
            jnp.matmul((x - self._target_state)[jnp.newaxis, :], self._Qf),
            (x - self._target_state)[:, jnp.newaxis],
        ).ravel()
