import pytest
from jax import numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal

from apop.distributions.gaussian import Gaussian, IIDGaussian
import jax
from apop.random import np_drng


class TestGaussian:
    def test_sample_numeric_assertion_random_value(self):
        key = jax.random.PRNGKey(0)
        mean = np_drng.random(size=3)
        sample_size = 10000
        covariance = np.eye(3)
        dist = Gaussian(key, jnp.array(mean), jnp.array(covariance))
        samples = dist.sample(sample_size).block_until_ready()
        assert samples.shape == (sample_size, 3)

        assert np.allclose(mean, np.mean(samples, axis=0), atol=1e-1)
        assert np.allclose(covariance[0, 0], np.cov(samples[:, 0]), atol=1e-1)
        assert np.allclose(covariance[1, 1], np.cov(samples[:, 1]), atol=1e-1)
        assert np.allclose(covariance[2, 2], np.cov(samples[:, 2]), atol=1e-1)

    def test_probability_numeric_assertion_random_value(self):
        key = jax.random.PRNGKey(0)
        mean = np_drng.random(size=3)
        covariance = np.eye(3)
        dist = Gaussian(key, jnp.array(mean), jnp.array(covariance))
        samples = dist.sample(500)
        actual = dist.probability(samples).block_until_ready()
        assert actual.shape == (500,)
        expected = multivariate_normal.pdf(np.array(samples), mean, covariance)
        assert np.allclose(expected, actual, atol=1e-6)


class TestIIDGaussian:
    def test_sample_numeric_assertion_with_random_value(self):
        key = jax.random.PRNGKey(0)
        mean1 = np_drng.random(size=(1, 3))
        mean2 = np_drng.random(size=(1, 3)) * 100.0
        covariance = np.tile(np.eye(3), (2, 1, 1))
        dist = IIDGaussian(
            key,
            jnp.array(np.concatenate([mean1, mean2], axis=0)),
            jnp.array(covariance),
        )
        samples = np.array((dist.sample(500)).block_until_ready())
        assert samples.shape == (500, 2, 3)

        assert np.allclose(np.mean(samples[:, 0, :], axis=0), mean1, atol=1e-1)
        assert np.allclose(np.mean(samples[:, 1, :], axis=0), mean2, atol=1e-1)

    def test_probability_numeric_assertion_with_shape(self):
        key = jax.random.PRNGKey(0)
        mean1 = np_drng.random(size=(1, 3))
        mean2 = np_drng.random(size=(1, 3)) * 100.0
        covariance = np.tile(np.eye(3), (2, 1, 1))
        dist = IIDGaussian(
            key,
            jnp.array(np.concatenate([mean1, mean2], axis=0)),
            jnp.array(covariance),
        )
        samples = np.array((dist.sample(500)).block_until_ready())
        actual = dist.probability(samples)
        assert actual.shape == (500,)


if __name__ == "__main__":
    pytest.main()
