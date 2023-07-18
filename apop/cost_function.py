from abc import ABCMeta, abstractmethod
import jax
from jax import numpy as jnp
from functools import partial
from typing import Optional


class CostFunction(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_batched_trajectory_cost(
        self, x_sequence: jnp.ndarray, u_sequence: jnp.ndarray
    ):
        """evaluate trajectory cost

        Args:
            x (jnp.ndarray): states, (batch_size, pred_len + 1, state_size)
            u (jnp.ndarray): inputs, (batch_size, pred_len, input_size)
            t (int): time step

        Returns:
            jnp:ndarray: gradient of model with respect to the state, shape (batch_size, state_size, state_size)
        """
        batch_size, _, state_size = x_sequence.shape
        _, pred_len, input_size = u_sequence.shape
        assert batch_size == u_sequence.shape[0]
        assert pred_len + 1 == x_sequence.shape[1]
        batched_trajectory_cost_func = jax.vmap(
            self.evaluate_trajectory_cost, in_axes=0, out_axes=0
        )
        return batched_trajectory_cost_func(x_sequence, u_sequence)

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_trajectory_cost(
        self, x_sequence: jnp.ndarray, u_sequence: jnp.ndarray
    ):
        """evaluate trajectory cost

        Args:
            x (jnp.ndarray): states, (pred_len + 1, state_size)
            u (jnp.ndarray): inputs, (pred_len, input_size)
            t (int): time step

        Returns:
            jnp:ndarray: gradient of model with respect to the state, shape (batch_size, state_size, state_size)
        """
        _, state_size = x_sequence.shape
        pred_len, input_size = u_sequence.shape
        assert pred_len + 1 == x_sequence.shape[0]
        t_sequence = jnp.arange(0, pred_len + 1)[:, jnp.newaxis]

        # spilt to stage and terminal
        stage_x_sequence, terminal_x_sequence = jnp.split(
            x_sequence, [pred_len], axis=0
        )
        stage_t_sequence, terminal_t_sequence = jnp.split(
            t_sequence, [pred_len], axis=0
        )
        batched_stage_cost_func = jax.vmap(
            self.evaluate_stage_cost, in_axes=0, out_axes=0
        )
        stage_cost = jnp.sum(
            batched_stage_cost_func(stage_x_sequence, u_sequence, stage_t_sequence)
        )
        terminal_cost = self.evaluate_terminal_cost(
            terminal_x_sequence.ravel(), terminal_t_sequence.ravel()
        )
        return stage_cost + terminal_cost

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def evaluate_stage_cost(self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray):
        """evaluate cost

        Args:
            x (jnp.ndarray): states, (state_size, )
            u (jnp.ndarray): inputs, (input_size, )
            t (jnp.ndarray): time step, (1, )

        Returns:
            jnp:ndarray: stage cost (1, )
        """
        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def evaluate_terminal_cost(self, x_sequence: jnp.ndarray, t: jnp.ndarray):
        """evaluate cost

        Args:
            x (jnp.ndarray): states, (state_size, )
            t (jnp.ndarray): time step, (1, )

        Returns:
            jnp:ndarray: stage cost (1, )
        """
        raise NotImplementedError


class QuadraticCostFunction(CostFunction):
    def __init__(self, Q: jnp.ndarray, Qf: jnp.ndarray, R: jnp.ndarray, F: jnp.ndarray):
        super().__init__()
        assert Q.shape == Qf.shape
        assert F.shape[0] == Q.shape[0]
        assert F.shape[1] == R.shape[0]
        self._Q = Q
        self._Qf = Qf
        self._R = R
        self._F = F

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_stage_cost(
        self, x: jnp.ndarray, u: Optional[jnp.ndarray], t: jnp.ndarray
    ):
        """evaluate cost

        Args:
            x (jnp.ndarray): states, (state_size, )
            u (jnp.ndarray): inputs, (input_size, )
            t (jnp.ndarray): time step, (1, )

        Returns:
            jnp:ndarray: stage cost (1, )
        """
        state_cost = jnp.matmul(
            x[jnp.newaxis, :], jnp.matmul(self._Q, x[:, jnp.newaxis])
        )
        cross_cost = 2.0 * jnp.matmul(
            x[jnp.newaxis, :], jnp.matmul(self._F, u[:, jnp.newaxis])
        )
        input_cost = jnp.matmul(
            u[jnp.newaxis, :], jnp.matmul(self._R, u[:, jnp.newaxis])
        )
        return (state_cost + cross_cost + input_cost).ravel()

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_terminal_cost(self, x: jnp.ndarray, t: jnp.ndarray):
        """evaluate cost

        Args:
            x (jnp.ndarray): states, (state_size, )
            t (jnp.ndarray): time step, (1, )

        Returns:
            jnp:ndarray: stage cost (1, )
        """
        return jnp.matmul(
            x[jnp.newaxis, :], jnp.matmul(self._Qf, x[:, jnp.newaxis])
        ).ravel()
