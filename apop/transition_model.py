from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Union

import jax
from jax import numpy as jnp

from apop.distribution import Distribution


class TransitionModel(metaclass=ABCMeta):
    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def predict_batched_trajectory(
        self,
        curr_x: jnp.ndarray,
        u_sequence: jnp.ndarray,
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """predict trajectories

        Args:
            curr_x (jnp.ndarray): current state, shape (batch_size, state_size)
            u_sequence (jnp.ndarray): inputs, shape (batch_size, pred_len, input_size)
            random_key (jax.random.KeyArray): random key

        Returns:
            pred_x_sequence (jnp.ndarray): predicted state, shape(batch_size, pred_len+1, state_size)
            including current state
        """
        batch_size, _ = curr_x.shape
        assert batch_size == u_sequence.shape[0]
        assert len(u_sequence.shape) == 3

        batched_pred_trajectory_func = jax.vmap(
            self.predict_trajectory, in_axes=0, out_axes=0
        )
        return batched_pred_trajectory_func(
            curr_x, u_sequence, jax.random.split(random_key, num=batch_size)
        )

    @partial(jax.jit, static_argnums=(0,))
    def predict_trajectory(
        self,
        curr_x: jnp.ndarray,
        u_sequence: jnp.ndarray,
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """predict trajectories

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )
            u_sequence (jnp.ndarray): inputs, shape (pred_len, input_size)
            random_key (jax.random.KeyArray): random key

        Returns:
            pred_xs (jnp.ndarray): predicted state,
                shape(pred_len+1, state_size) including current state
        """
        state_size = curr_x.shape[0]
        pred_len = u_sequence.shape[0]

        pred_x_sequence = jnp.zeros((pred_len + 1, state_size))
        x = curr_x
        pred_x_sequence = pred_x_sequence.at[0].set(x)
        random_key_sequence = jax.random.split(random_key, num=pred_len)

        for t in range(pred_len):
            next_x = self.predict_next_state(
                x,
                u_sequence[t],
                jnp.ones(1, dtype=jnp.int32) * t,
                random_key_sequence[t],
            )
            pred_x_sequence = pred_x_sequence.at[t + 1].set(next_x)
            x = next_x

        return pred_x_sequence

    @partial(jax.jit, static_argnums=(0,))
    def predict_batched_next_state(
        self,
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: Union[jnp.ndarray, int],
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """predict batched next state

        Args:
            x (jnp.ndarray): states, shape (batch_size, state_size )
            u (jnp.ndarray): input, shape (batch_size, input_size)
            t (Union[jnp.ndarray, int]): time step, shape (1, )
            random_key (jax.random.KeyArray): random key

        Returns:
            next_x (jnp.ndarray): next state, shape (batch_size, state_size)
        """
        jnp_t = jnp.ones(1, dtype=jnp.int32) * t
        batched_pred_next_state_func = jax.vmap(
            self.predict_next_state, in_axes=(0, 0, None, 0), out_axes=0
        )
        return batched_pred_next_state_func(
            x, u, jnp_t, jax.random.split(random_key, num=x.shape[0])
        )

    @abstractmethod
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
            x (jnp.ndarray): states, shape (state_size, )
            u (jnp.ndarray): input, shape (input_size, )
            t (jnp.ndarray): time step, shape (1, )
            random_key (jax.random.KeyArray): random key

        Returns:
            next_x (jnp.ndarray): next state, shape (state_size, )
        """
        raise NotImplementedError("Implement the model")

    def predict_next_state_distribution(
        self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray
    ) -> Distribution:
        """predict next state

        Args:
            x (jnp.ndarray): states, shape (state_size, )
            u (jnp.ndarray): input, shape (input_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            Distribution: next state distribution
        """
        raise NotImplementedError("Implement the model")

    @partial(jax.jit, static_argnums=(0,))
    def fx(
        self,
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: Union[jnp.ndarray, int],
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """gradient of model with respect to the state in batch form

        Args:
            x (jnp.ndarray): states, (batch_size, state_size)
            u (jnp.ndarray): inputs, (batch_size, input_size)
            t (int): time step
            random_key (jax.random.KeyArray): random key

        Returns:
            jnp:ndarray: gradient of model with respect to the state, shape (batch_size, state_size, state_size)
        """
        assert x.shape[0] == u.shape[0]
        jnp_t = jnp.ones(1, dtype=jnp.int32) * t
        fx = jax.vmap(
            jax.jacfwd(self.predict_next_state, argnums=0),
            in_axes=(0, 0, None, 0),
            out_axes=0,
        )
        return fx(x, u, jnp_t, jax.random.split(random_key, num=x.shape[0]))

    @partial(jax.jit, static_argnums=(0,))
    def fu(
        self,
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: Union[jnp.ndarray, int],
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """gradient of model with respect to the input in batch form

        Args:
            x (jnp.ndarray): states, (batch_size, state_size)
            u (jnp.ndarray): inputs, (batch_size, input_size)
            t (int): time step
            random_key (jax.random.KeyArray): random key

        Returns:
            jnp:ndarray: gradient of model with respect to the state, shape (batch_size, state_size, input_size)
        """
        assert x.shape[0] == u.shape[0]
        jnp_t = jnp.ones(1, dtype=jnp.int32) * t
        fu = jax.vmap(
            jax.jacfwd(self.predict_next_state, argnums=1),
            in_axes=(0, 0, None, 0),
            out_axes=0,
        )
        return fu(x, u, jnp_t, jax.random.split(random_key, num=x.shape[0]))

    @partial(jax.jit, static_argnums=(0,))
    def fxx(
        self,
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: Union[jnp.ndarray, int],
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def fux(
        self,
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: Union[jnp.ndarray, int],
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def fuu(
        self,
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: Union[jnp.ndarray, int],
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        raise NotImplementedError
