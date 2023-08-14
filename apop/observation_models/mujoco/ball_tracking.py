import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional

from apop.observation_model import ProbabilisticObservationModel
from apop.distributions.gaussian import IIDGaussian


class Ball2dTrackingGaussianObservationModel(ProbabilisticObservationModel):
    def __init__(
        self,
        covariance: jnp.ndarray,
        landmark_positions: jnp.ndarray,
        key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ):
        super().__init__(key)
        assert len(covariance.shape) == 3
        self._covariance = covariance
        self._landmark_positions = landmark_positions

    @partial(jax.jit, static_argnums=(0,))
    def observe(
        self, curr_x: jnp.ndarray, observation_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )

        Returns:
            jnp.ndarray: observation state, shape (observation_size, )
        """
        return self.observe_distribution(curr_x, observation_mask).sample(1)[0]

    def observe_distribution(
        self, curr_x: jnp.ndarray, observation_mask: Optional[jnp.ndarray] = None
    ) -> IIDGaussian:
        """observe y from x

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )

        Returns:
            Distribution: observation state distribution
        """
        assert observation_mask is not None
        assert observation_mask.shape == (self._landmark_positions.shape[0],)

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
            self._key,
            means=jnp.stack([mean_position, mean_angle], axis=1),
            full_covariances=self._covariance,
        )
        return dist
