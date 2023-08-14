import pytest
from jax import numpy as jnp
import numpy as np

from apop.observation_models.mujoco.ball_tracking import (
    Ball2dTrackingGaussianObservationModel,
)


class TestBall2dTrackingGaussianObservationModel:
    def test_observe_distribution_numeric_assertion(self):
        covariance = np.tile(np.eye(2) * 0.2, (2, 1, 1))
        landmark_positions = np.array([[0.5, 0.6], [-0.7, 0.5], [-0.6, -0.4]])
        observation_model = Ball2dTrackingGaussianObservationModel(
            jnp.array(covariance), jnp.array(landmark_positions)
        )
        curr_x = np.array([0.4, 0.2, np.pi * 0.25])
        distribution = observation_model.observe_distribution(
            jnp.array(curr_x), observation_mask=jnp.array([1, 0, 1], dtype=jnp.bool_)
        )
        curr_distance = np.sqrt(
            np.sum((landmark_positions[[0, 2], :2] - curr_x[:2]) ** 2, axis=1)
        )
        curr_directions = (
            np.arctan2(
                landmark_positions[[0, 2], 1] - curr_x[1],
                landmark_positions[[0, 2], 0] - curr_x[0],
            )
            - curr_x[2]
        )
        curr_y = np.stack([curr_distance, curr_directions], axis=1)
        assert np.allclose(np.array(distribution._means), curr_y)
        probs = distribution.probability(jnp.array(curr_y)[jnp.newaxis, :, :])
        assert probs.shape == (1,)


if __name__ == "__main__":
    pytest.main()
