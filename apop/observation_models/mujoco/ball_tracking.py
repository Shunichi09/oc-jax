from functools import partial

import jax
import jax.numpy as jnp

from apop.distributions.gaussian import IIDGaussian
from apop.observation_model import ObservationModel


class Ball2dTrackingGaussianObservationModel(ObservationModel):
    def __init__(
        self,
        covariance: jnp.ndarray,
        landmark_positions: jnp.ndarray,
    ):
        super().__init__()
        assert len(covariance.shape) == 3  # num_landmarks, state, state
        self._covariance = covariance
        self._landmark_positions = landmark_positions

    @partial(jax.jit, static_argnums=(0,))
    def observe(
        self,
        curr_x: jnp.ndarray,
        observation_mask: jnp.ndarray,
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )

        Returns:
            jnp.ndarray: observation state, shape (observation_size, )
        """
        return self.observe_distribution(curr_x, observation_mask).sample(
            random_key, 1
        )[0]

    def observe_distribution(
        self,
        curr_x: jnp.ndarray,
        observation_mask: jnp.ndarray,
    ) -> IIDGaussian:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )

        Returns:
            Distribution: observation state distribution
        """
        assert observation_mask is not None
        assert observation_mask.shape == (self._landmark_positions.shape[0],)
        assert curr_x.shape == (3,)

        diff = self._landmark_positions[observation_mask] - curr_x[:2]
        mean_position = jnp.sqrt(jnp.sum(diff**2, axis=1))  # shape (num_landmarks, )
        mean_angle = (
            jnp.arctan2(
                self._landmark_positions[observation_mask, 1] - curr_x[1],
                self._landmark_positions[observation_mask, 0] - curr_x[0],
            )
            - curr_x[2]
        )
        dist = IIDGaussian(
            means=jnp.stack([mean_position, mean_angle], axis=1),
            full_covariances=self._covariance,
        )
        return dist
