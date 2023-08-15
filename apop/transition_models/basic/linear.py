from functools import partial

import jax
from jax import numpy as jnp

from apop.distributions.gaussian import Gaussian
from apop.transition_model import TransitionModel


class LinearTransitionModel(TransitionModel):
    """discrete linear model, x[k+1] = Ax[k] + Bu[k]

    Attributes:
        A (jnp.ndarray): shape(state_size, state_size)
        B (jnp.ndarray): shape(state_size, input_size)
    """

    def __init__(self, A: jnp.ndarray, B: jnp.ndarray):
        """ """
        super().__init__()
        assert A.shape[0] == B.shape[0]
        assert A.shape[0] == A.shape[1]
        self._A = A
        self._B = B

    @partial(jax.jit, static_argnums=(0,))
    def predict_next_state(
        self,
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: jnp.ndarray,
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """predict next state

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )
            u (jnp.ndarray): input, shape (input_size, )
            t (int): time step

        Returns:
            next_x (jnp.ndarray): next state, shape (state_size, )
        """
        next_x = jnp.matmul(self._A, x[:, jnp.newaxis]) + jnp.matmul(
            self._B, u[:, jnp.newaxis]
        )
        return next_x.ravel()


class LinearGaussianTransitionModel(TransitionModel):
    def __init__(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        covariance: jnp.ndarray,
    ):
        super().__init__()
        assert A.shape[0] == B.shape[0]
        assert A.shape[0] == A.shape[1]
        self._A = A
        self._B = B
        self._covariance = covariance

    @partial(jax.jit, static_argnums=(0,))
    def predict_next_state(
        self,
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: jnp.ndarray,
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """predict next state

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )
            u (jnp.ndarray): input, shape (input_size, )
            t (int): time step

        Returns:
            next_x (jnp.ndarray): next state, shape (state_size, )
        """
        distribution = self.predict_next_state_distribution(x, u, t)
        return distribution.sample(random_key, 1)[0]

    def predict_next_state_distribution(
        self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray
    ) -> Gaussian:
        """predict next state

        Args:
            x (jnp.ndarray): states, shape (state_size, )
            u (jnp.ndarray): input, shape (input_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            next_x (jnp.ndarray): next state, shape (state_size, )
        """
        next_x = jnp.matmul(self._A, x[:, jnp.newaxis]) + jnp.matmul(
            self._B, u[:, jnp.newaxis]
        )
        dist = Gaussian(next_x.ravel(), self._covariance)
        return dist
