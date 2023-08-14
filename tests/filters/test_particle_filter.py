import pytest
from jax import numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal

from apop.transition_models.basic.linear import LinearGaussianTransitionModel
from apop.observation_models.basic.linear import LinearGaussianObservationModel
from apop.filters.particle_filter import ParticleFilter
from apop.distributions.gaussian import Gaussian

import jax
from apop.random import np_drng

from jax import config

config.update("jax_debug_nans", True)
config.update("jax_disable_jit", True)


class TestParticleFilter:
    def generate_filter(self, num_particles=500):
        # simple velocity model
        dt = 0.01
        A = np.eye(4)
        A[0, 1] = dt
        A[2, 3] = dt
        B = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        covariance = np.eye(2) * (0.01**2)
        init_x = np.array([0.0, 0.0, 0.0, 0.0])
        init_x_cov = np.diag([0.01, 0.01, 0.01, 0.01]) ** 2
        state_x_cov = np.diag([0.002, 0.001, 0.002, 0.001]) ** 2

        transition_model = LinearGaussianTransitionModel(
            jnp.array(A),
            jnp.array(B),
            covariance=state_x_cov,
            key=jax.random.PRNGKey(15),
        )
        observation_model = LinearGaussianObservationModel(
            C=jnp.array(C), covariance=jnp.array(covariance), key=jax.random.PRNGKey(92)
        )
        initial_dist = Gaussian(
            jax.random.PRNGKey(77), jnp.array(init_x), jnp.array(init_x_cov)
        )
        filter = ParticleFilter(
            transition_model,
            observation_model,
            initial_dist,
            num_particles=num_particles,
            resampling_threshold=num_particles / 0.01,
            jax_random_key=jax.random.PRNGKey(100),
        )
        return filter, A, B, C

    def test_initialize_shape_assertions(self):
        num_particles = 500
        filter, *_ = self.generate_filter(num_particles=num_particles)
        particle = filter.particles()
        assert particle.shape == (num_particles, 4)

        dummy_output = filter._particle_weights * jnp.ones(1)
        weight = np.array(dummy_output)
        assert np.allclose(weight, np.ones(num_particles) / float(num_particles))

    def test_predict_numeric_assertions(self):
        num_particles = 100
        filter, A, B, _ = self.generate_filter(num_particles=num_particles)
        u = np.array([0.5, 0.75])
        expected = np.array(filter.predict(u, 0).block_until_ready())
        actual = np.matmul(A, np.zeros((4, 1))) + np.matmul(B, u[:, np.newaxis])
        assert np.allclose(expected[[0, 2]], actual.flatten()[[0, 2]], atol=1e-1)
        assert np.allclose(expected[[1, 3]], actual.flatten()[[1, 3]], atol=1)

    def test_estimate_numeric_assertions(self):
        num_particles = 1000
        filter, A, B, _ = self.generate_filter(num_particles=num_particles)
        u_seq = np_drng.random(size=(10, 2)) * 0.1
        x = np.zeros((4,))

        for i, u in enumerate(u_seq):
            print(i)
            filter.predict(u, 0)
            # gt transition
            gt_state = np.matmul(A, x[:, np.newaxis]) + np.matmul(B, u[:, np.newaxis])
            gt_state = gt_state.flatten()
            # gt obs
            y = np.array([gt_state[0], gt_state[2]])
            # noised obs
            y += np_drng.normal(size=2) * 0.01
            # estimation
            estimated = filter.estimate(
                jnp.array(y), mask=jnp.array([1.0], dtype=jnp.bool_)
            )
            cov_x = jnp.cov(filter.particles()[:, 0])
            cov_xdot = jnp.cov(filter.particles()[:, 1])
            cov_y = jnp.cov(filter.particles()[:, 2])
            cov_ydot = jnp.cov(filter.particles()[:, 3])

            assert np.allclose(np.array(estimated)[[1, 3]], gt_state[[1, 3]], atol=1)
            assert np.allclose(np.array(estimated)[[0, 2]], gt_state[[0, 2]], atol=1e-1)
            print(estimated, gt_state)
            print(cov_x**0.5, cov_xdot**0.5, cov_y**0.5, cov_ydot**0.5)
            # update gt
            x = gt_state


if __name__ == "__main__":
    pytest.main()
