import pytest
from jax import numpy as jnp
import numpy as np

from apop.observation_models.basic.linear import (
    LinearObservationModel,
    LinearGaussianObservationModel,
)
from apop.random import np_drng
import jax


class TestLinearObservationModel:
    def test_observe_batched_numeric_assertion_with_random_value(self):
        C = np_drng.normal(size=(3, 4))
        obs_model = LinearObservationModel(jnp.array(C))

        curr_x = np_drng.normal(size=(5, 4))
        mask = jnp.ones(5, dtype=jnp.bool_)
        actual = np.array(
            obs_model.observe_batched(
                curr_x, mask, jax.random.PRNGKey(10)
            ).block_until_ready()
        )

        for i in range(curr_x.shape[0]):
            expected = np.matmul(C, curr_x[i][:, np.newaxis]).flatten()
            assert np.allclose(expected, actual[i])


class TestLinearGaussianObservationModel:
    def test_observe_batched_shape_assertion(self):
        C = np_drng.normal(size=(3, 4))
        cov = np.eye(3)
        obs_model = LinearGaussianObservationModel(jnp.array(C), jnp.array(cov))
        curr_x = np_drng.normal(size=(5, 4))
        mask = jnp.ones(5, dtype=jnp.bool_)
        actual = np.array(
            obs_model.observe_batched(
                curr_x, mask, jax.random.PRNGKey(10)
            ).block_until_ready()
        )
        assert actual.shape == (5, 3)

    def test_observe_batched_sample_different_value(self):
        C = np_drng.normal(size=(3, 4))
        cov = np.eye(3)
        obs_model = LinearGaussianObservationModel(jnp.array(C), jnp.array(cov))
        curr_x = np_drng.normal(size=(5, 4))
        mask = jnp.ones(5, dtype=jnp.bool_)
        actual1 = np.array(
            obs_model.observe_batched(
                curr_x, mask, jax.random.PRNGKey(10)
            ).block_until_ready()
        )
        actual2 = np.array(
            obs_model.observe_batched(
                curr_x, mask, jax.random.PRNGKey(100)
            ).block_until_ready()
        )
        assert not np.allclose(actual1, actual2)
