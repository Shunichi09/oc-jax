from functools import partial

import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from apop.controller import Controller
from apop.cost_function import CostFunction
from apop.transition_model import TransitionModel


class TruncatedGaussianCrossEntropyMethod(Controller):
    _transition_model: TransitionModel
    _cost_function: CostFunction
    _T: int

    def __init__(
        self,
        transition_model: TransitionModel,
        cost_function: CostFunction,
        T: int,
        num_iterations: int,
        sample_size: int,
        num_elites: int,
        alpha: float,  # larger than use old mean
        initial_variance: np.ndarray,  # shape (state_size)
        upper_bound: np.ndarray,  # shape (state_size)
        lower_bound: np.ndarray,  # shape (state_size)
        jax_random_key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ) -> None:
        super().__init__(transition_model, cost_function)
        self._T = T
        self._num_iterations = num_iterations
        self._num_elites = num_elites
        self._initial_variance = jnp.array(initial_variance)
        self._sample_size = sample_size
        self._upper_bound = jnp.array(upper_bound)
        self._lower_bound = jnp.array(lower_bound)
        self._alpha = alpha
        self._jax_random_key = jax_random_key

    @partial(jax.jit, static_argnums=(0,))
    def control(
        self,
        curr_x: jnp.ndarray,
        initial_u_sequence: jnp.ndarray,
    ) -> jnp.ndarray:
        _, input_size = initial_u_sequence.shape
        tiled_curr_x = jnp.tile(curr_x, (self._sample_size, 1))
        mean = initial_u_sequence  # shape (T, input_size)
        variance = self._initial_variance  # shape (T, input_size)

        for i in range(self._num_iterations):
            # variance computation
            lower_bound_mean = mean - self._lower_bound  # shape (T, input_size)
            mean_upper_bound = self._upper_bound - mean  # shape (T, input_size)
            # get minimum variance
            constrained_variance = jnp.minimum(
                jnp.minimum(
                    jnp.square(lower_bound_mean * 0.5),
                    jnp.square(mean_upper_bound * 0.5),
                ),
                variance,
            )
            # sample inputs sequence
            noise = jax.random.truncated_normal(
                self._jax_random_key,
                lower=-2.0,
                upper=2.0,
                shape=(self._sample_size, self._T, input_size),
            )
            # shape (batch_size, T, input_size)
            u_sequence_samples = (
                noise * jnp.sqrt(constrained_variance[jnp.newaxis, :, :])
                + mean[jnp.newaxis, :, :]
            )
            # pred_x_sequence.shape = (batch_size, T+1, state_size), include init_state
            pred_x_sequence_samples = self._transition_model.predict_batched_trajectory(
                tiled_curr_x, u_sequence_samples
            )

            # costs
            cost_samples = self._cost_function.evaluate_batched_trajectory_cost(
                pred_x_sequence_samples, u_sequence_samples
            ).ravel()
            # elites, from small
            sorted_idx = jnp.argsort(cost_samples)[: self._num_elites]

            # compute of elites and get
            mean = self._alpha * mean + (1.0 - self._alpha) * jnp.mean(
                u_sequence_samples[sorted_idx], axis=0
            )
            variance = self._alpha * variance + (1.0 - self._alpha) * jnp.var(
                u_sequence_samples[sorted_idx], axis=0
            )

        return mean
