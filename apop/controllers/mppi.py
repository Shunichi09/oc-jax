from functools import partial

import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from apop.controller import Controller
from apop.cost_function import CostFunction
from apop.transition_model import TransitionModel
from apop.random import new_key


class MPPI(Controller):
    _transition_model: TransitionModel
    _cost_function: CostFunction
    _T: int

    def __init__(
        self,
        transition_model: TransitionModel,
        cost_function: CostFunction,
        T: int,
        sample_size: int,
        alpha: float,  # use only noise or not
        gamma: float,
        lmb: float,  # lower is reject a many trajectories
        initial_covariance: np.ndarray,  # shape (input_size, input_size)
        upper_bound: np.ndarray,  # shape (state_size)
        lower_bound: np.ndarray,  # shape (state_size)
        jax_random_key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ) -> None:
        # https://ieeexplore.ieee.org/abstract/document/7487277

        super().__init__(transition_model, cost_function)
        self._T = T
        self._initial_covariance = jnp.array(initial_covariance)
        self._input_size = initial_covariance.shape[0]
        self._initial_precision = jnp.linalg.inv(self._initial_covariance)
        self._sample_size = sample_size
        self._upper_bound = jnp.array(upper_bound)
        self._lower_bound = jnp.array(lower_bound)
        self._alpha = alpha
        self._gamma = gamma
        self._lmb = lmb
        # predict trajectory with noised inputs
        self._num_non_masks_of_u_sequence = int(self._sample_size * self._alpha)

        self._jax_random_key = jax_random_key

    @partial(jax.jit, static_argnums=(0,))
    def control(
        self,
        curr_x: jnp.ndarray,
        initial_u_sequence: jnp.ndarray,
    ) -> jnp.ndarray:
        _, input_size = initial_u_sequence.shape
        assert input_size == self._input_size

        tiled_curr_x = jnp.tile(curr_x, (self._sample_size, 1))
        # tiled_initial_u_sequence.shape = (sample_size, T, input_size)
        tiled_initial_u_sequence = jnp.tile(
            initial_u_sequence, (self._sample_size, 1, 1)
        )
        # sample noise
        self._jax_random_key = new_key(self._jax_random_key)

        # epsilon.shape = (sample_size, T, input_size)
        epsilon = jax.random.multivariate_normal(
            self._jax_random_key,
            mean=jnp.zeros(shape=(self._input_size,)),
            cov=self._initial_covariance,  # shape (input_size, input_size)
            shape=(self._sample_size, self._T),
        )

        initial_u_sequence_with_masked = jnp.concatenate(
            [
                tiled_initial_u_sequence[: self._num_non_masks_of_u_sequence],
                jnp.zeros(
                    (
                        self._sample_size - self._num_non_masks_of_u_sequence,
                        self._T,
                        self._input_size,
                    )
                ),
            ],
            axis=0,
        )

        v_sequence_samples = initial_u_sequence_with_masked + epsilon
        v_sequence_samples = jnp.clip(
            v_sequence_samples, self._lower_bound, self._upper_bound
        )

        # pred_x_sequence.shape = (batch_size, T+1, state_size), include init_state
        pred_x_sequence_samples = self._transition_model.predict_batched_trajectory(
            tiled_curr_x, v_sequence_samples
        )
        # compute costs with zero control
        cost_samples = self._cost_function.evaluate_batched_trajectory_cost(
            pred_x_sequence_samples,
            jnp.zeros((self._sample_size, self._T, self._input_size)),
        ).ravel()

        # computes S
        # tiled_precision.shape = (sample_size*T, input_size, input_size)
        tiled_precision = jnp.tile(
            self._initial_precision, (self._sample_size * self._T, 1, 1)
        )

        # TODO: Implementation is a bit different from the paper because papers typo ?
        delta_S_sequence = self._gamma * jnp.matmul(
            tiled_initial_u_sequence.reshape(
                self._sample_size * self._T, 1, self._input_size
            ),
            jnp.matmul(
                tiled_precision,
                epsilon.reshape(self._sample_size * self._T, self._input_size, 1),
            ),
        ).reshape(self._sample_size, self._T)
        delta_S = jnp.sum(delta_S_sequence, axis=1)

        # S.shape = (batch_size, )
        S = cost_samples + delta_S

        rho = jnp.min(S)
        eta = jnp.sum(jnp.exp(-(S - rho) / self._lmb))
        weights = jnp.exp(-(S - rho) / self._lmb) / eta

        # du.shape = (T, input_size)
        du = jnp.sum(weights[:, jnp.newaxis, jnp.newaxis] * epsilon, axis=0)

        # TODO: apply filter
        # du = filer(du)
        improved_u_sequence = initial_u_sequence + du

        improved_u_sequence = jnp.clip(
            improved_u_sequence, self._lower_bound, self._upper_bound
        )
        return improved_u_sequence
