from functools import partial

import jax
from jax import numpy as jnp

from apop.controller import Controller
from apop.cost_function import CostFunction
from apop.transition_model import TransitionModel


class LinearQuadraticRegulator(Controller):
    _transition_model: TransitionModel
    _cost_function: CostFunction
    _T: int

    def __init__(
        self, transition_model: TransitionModel, cost_function: CostFunction, T: int
    ) -> None:
        super().__init__(transition_model, cost_function)
        self._T = T

    @partial(jax.jit, static_argnums=(0,))
    def control(
        self,
        curr_x: jnp.ndarray,
        initial_u_sequence: jnp.ndarray,
    ) -> jnp.ndarray:
        _, input_size = initial_u_sequence.shape
        # pred_x_sequence.shape = (T+1, state_size), include
        pred_x_sequence = self._transition_model.predict_trajectory(
            curr_x, initial_u_sequence
        )
        # without batch
        Sk = self._cost_function.terminal_cxx(
            pred_x_sequence[self._T][jnp.newaxis, :], self._T
        )[0]

        gains = []
        for t in reversed(range(self._T)):
            # without batch
            A = self._transition_model.fx(
                pred_x_sequence[t][jnp.newaxis, :],
                initial_u_sequence[t][jnp.newaxis, :],
                t,
            )[0]
            B = self._transition_model.fu(
                pred_x_sequence[t][jnp.newaxis, :],
                initial_u_sequence[t][jnp.newaxis, :],
                t,
            )[0]
            Q = self._cost_function.stage_cxx(
                pred_x_sequence[t][jnp.newaxis, :],
                initial_u_sequence[t][jnp.newaxis, :],
                t,
            )[0]
            F = self._cost_function.cxu(
                pred_x_sequence[t][jnp.newaxis, :],
                initial_u_sequence[t][jnp.newaxis, :],
                t,
            )[0]
            R = self._cost_function.cuu(
                pred_x_sequence[t][jnp.newaxis, :],
                initial_u_sequence[t][jnp.newaxis, :],
                t,
            )[0]
            Sk_C = jnp.linalg.inv(R + (B.T.dot(Sk).dot(B)))
            Sk_D = F.T + B.T.dot(Sk).dot(A)
            # compute t's Sk
            Sk = Q + A.T.dot(Sk).dot(A) - Sk_D.T.dot(Sk_C).dot(Sk_D)
            # compute t's gain
            gain_C = jnp.linalg.inv(R + (B.T.dot(Sk).dot(B)))
            gain_D = F.T + B.T.dot(Sk).dot(A)
            K = gain_C.dot(gain_D)
            gains.append(K)

        optimized_u_sequence = jnp.zeros((self._T, input_size))
        x = curr_x
        for t, K in enumerate(reversed(gains)):
            optimized_u = jnp.ravel(-K.dot(x[:, jnp.newaxis]))
            optimized_u_sequence = optimized_u_sequence.at[t].set(optimized_u)
            x = self._transition_model.predict_next_state(x, optimized_u, t)

        return optimized_u_sequence
