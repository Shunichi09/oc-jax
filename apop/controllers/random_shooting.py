from functools import partial

import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from apop.controller import Controller
from apop.cost_function import CostFunction
from apop.transition_model import TransitionModel


class UniformRandomShootingMethod(Controller):
    _transition_model: TransitionModel
    _cost_function: CostFunction
    _T: int

    def __init__(
        self,
        transition_model: TransitionModel,
        cost_function: CostFunction,
        T: int,
        sample_size: int,
        upper_bound: np.ndarray,  # shape (state_size)
        lower_bound: np.ndarray,  # shape (state_size)
        jax_random_key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ) -> None:
        super().__init__(transition_model, cost_function)
        self._T = T
        self._sample_size = sample_size
        self._upper_bound = jnp.array(upper_bound)
        self._lower_bound = jnp.array(lower_bound)
        self._jax_random_key = jax_random_key

    # @partial(jax.jit, static_argnums=(0,))
    def control(
        self,
        curr_x: jnp.ndarray,
        initial_u_sequence: jnp.ndarray,
    ) -> jnp.ndarray:
        _, input_size = initial_u_sequence.shape
        tiled_curr_x = jnp.tile(curr_x, (self._sample_size, 1))

        # sample inputs sequence
        noise = jax.random.uniform(
            self._jax_random_key,
            shape=(self._sample_size, self._T, input_size),
        )
        # shape (batch_size, T, input_size)
        u_sequence_samples = (
            noise
            * (
                self._upper_bound[jnp.newaxis, jnp.newaxis, :]
                - self._lower_bound[jnp.newaxis, jnp.newaxis, :]
            )
            + self._lower_bound[jnp.newaxis, jnp.newaxis, :]
        )
        # pred_x_sequence.shape = (batch_size, T+1, state_size), include init_state
        pred_x_sequence_samples = self._transition_model.predict_batched_trajectory(
            tiled_curr_x, u_sequence_samples
        )
        # costs
        cost_samples = self._cost_function.evaluate_batched_trajectory_cost(
            pred_x_sequence_samples, u_sequence_samples
        ).ravel()
        min_idx = jnp.argmin(cost_samples)
        return u_sequence_samples[min_idx]
