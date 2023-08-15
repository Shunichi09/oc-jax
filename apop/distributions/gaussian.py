from functools import partial

import jax
import jax.numpy as jnp

from apop.distribution import Distribution


class Gaussian(Distribution):
    def __init__(self, mean: jnp.ndarray, full_covariance: jnp.ndarray) -> None:
        super().__init__()
        self._mean = mean
        self._full_covariance = full_covariance

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self, random_key: jax.random.KeyArray, num_samples: int) -> jnp.ndarray:
        """sample variable from the distribution

        Args:
            num_samples (int): num_samples

        Returns:
            jnp:ndarray: sampled variables, shape (num_samples, state_size)
        """
        return jax.random.multivariate_normal(
            random_key, self._mean, self._full_covariance, shape=(num_samples,)
        )

    @partial(jax.jit, static_argnums=(0,))
    def probability(self, x: jnp.ndarray) -> jnp.ndarray:
        """compute probability of the given variable

        Args:
            x (jnp.ndarray): variables, shape (num_samples, state_size)

        Returns:
            jnp:ndarray: probabilities, shape (num_samples, )
        """
        return jax.scipy.stats.multivariate_normal.pdf(
            x, self._mean, self._full_covariance
        )


class IIDGaussian(Distribution):
    def __init__(
        self,
        means: jnp.ndarray,
        full_covariances: jnp.ndarray,
    ) -> None:
        super().__init__()
        self._means = means
        self._full_covariances = full_covariances
        assert self._means.shape[0] == self._full_covariances.shape[0]
        self._num_gaussians = self._means.shape[0]

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self, random_key: jax.random.KeyArray, num_samples: int) -> jnp.ndarray:
        """sample variable from the distribution

        Args:
            num_samples (int): num_samples

        Returns:
            jnp:ndarray: sampled variables, shape (num_samples, num_gaussians, state_size)
        """

        def sample_from_one(mean, full_covariance, key):
            return jax.random.multivariate_normal(
                key, mean, full_covariance, shape=(num_samples,)
            )

        batch_sample_func = jax.vmap(sample_from_one, in_axes=(0, 0, 0), out_axes=0)
        samples = batch_sample_func(
            self._means,
            self._full_covariances,
            jax.random.split(random_key, num=self._num_gaussians),
        ).reshape(self._num_gaussians, num_samples, -1)
        # output.shape (num_samples, num_gaussians, state_size)
        return jnp.transpose(samples, (1, 0, 2))

    @partial(jax.jit, static_argnums=(0,))
    def probability(self, x: jnp.ndarray) -> jnp.ndarray:
        """compute probability of the given variable
            This function computes densities and prod them assuming i.i.d.
            density = gaussian1(x[1]) * gaussian2(x[2]) * gaussian3(x[3]) ...

        Args:
            x (jnp.ndarray): variables, shape (num_samples, num_gaussians, state_size)

        Returns:
            jnp:ndarray: probabilities, shape (num_samples, )
        """
        assert len(x.shape) == 3

        def computes_each_pdf(x, mean, full_covariance):
            return jax.scipy.stats.multivariate_normal.pdf(x, mean, full_covariance)

        batch_pdf_func = jax.vmap(computes_each_pdf, in_axes=(0, 0, 0), out_axes=0)
        pdfs = batch_pdf_func(
            jnp.transpose(x, (1, 0, 2)), self._means, self._full_covariances
        ).reshape(self._num_gaussians, x.shape[0])
        mul_pdfs = jnp.cumprod(pdfs, axis=0)[-1, :]
        return mul_pdfs
