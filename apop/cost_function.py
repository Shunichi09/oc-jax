from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Optional, Union

import jax
from jax import numpy as jnp


class CostFunction(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_batched_trajectory_cost(
        self, x_sequence: jnp.ndarray, u_sequence: jnp.ndarray
    ) -> jnp.ndarray:
        """Evaluate batched trajectory cost

        Args:
            x (jnp.ndarray): states, shape (batch_size, pred_len + 1, state_size)
            u (jnp.ndarray): inputs, shape (batch_size, pred_len, input_size)

        Returns:
            jnp:ndarray: cost of batched trajectory, shape (batch_size, 1)
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
    ) -> jnp.ndarray:
        """Evaluate trajectory cost

        Args:
            x (jnp.ndarray): states, shape (pred_len + 1, state_size)
            u (jnp.ndarray): inputs, shape (pred_len, input_size)

        Returns:
            jnp:ndarray: cost of the given trajectory, shape (1, )
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
    def evaluate_stage_cost(
        self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Evaluate statge cost

        Args:
            x (jnp.ndarray): states, shape (state_size, )
            u (jnp.ndarray): inputs, shape (input_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            jnp:ndarray: stage cost, shape (1, )
        """
        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def evaluate_terminal_cost(
        self, x_sequence: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Evaluate terminal cost

        Args:
            x (jnp.ndarray): states, (state_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            jnp:ndarray: terminal cost, shape (1, )
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def stage_cx(
        self, x: jnp.ndarray, u: jnp.ndarray, t: Union[jnp.ndarray, int]
    ) -> jnp.ndarray:
        """Gradient of stage cost with respect to the given state

        Args:
            x (jnp.ndarray): states, (batch_size, state_size)
            u (jnp.ndarray): inputs, (batch_size, input_size)
            t (Union[jnp.ndarray, int]): time step

        Returns:
            jnp:ndarray: gradient of stage cost with respect to the state, shape (batch_size, state_size, 1)
        """
        assert x.shape[0] == u.shape[0]
        jnp_t = jnp.ones(1, dtype=jnp.int32) * t
        stage_cx_func = jax.vmap(
            jax.jacfwd(self.evaluate_stage_cost, argnums=0),
            in_axes=(0, 0, None),
            out_axes=0,
        )
        return stage_cx_func(x, u, jnp_t)

    @partial(jax.jit, static_argnums=(0,))
    def cu(
        self, x: jnp.ndarray, u: jnp.ndarray, t: Union[jnp.ndarray, int]
    ) -> jnp.ndarray:
        """Gradient of stage cost with respect to the given input

        Args:
            x (jnp.ndarray): states, (batch_size, state_size)
            u (jnp.ndarray): inputs, (batch_size, input_size)
            t (Union[jnp.ndarray, int]): time step

        Returns:
            jnp:ndarray: gradient of cost with respect to the input, shape (batch_size, input_size, 1)
        """
        assert x.shape[0] == u.shape[0]
        jnp_t = jnp.ones(1, dtype=jnp.int32) * t
        cu_func = jax.vmap(
            jax.jacfwd(self.evaluate_stage_cost, argnums=1),
            in_axes=(0, 0, None),
            out_axes=0,
        )
        return cu_func(x, u, jnp_t)

    @partial(jax.jit, static_argnums=(0,))
    def terminal_cx(self, x: jnp.ndarray, t: Union[jnp.ndarray, int]) -> jnp.ndarray:
        """Gradient of terminal cost with respect to the given state

        Args:
            x (jnp.ndarray): states, (batch_size, state_size)
            t (Union[jnp.ndarray, int]): time step

        Returns:
            jnp:ndarray: gradient of terminal cost with respect to the state, shape (batch_size, state_size, 1)
        """
        jnp_t = jnp.ones(1, dtype=jnp.int32) * t
        terminal_cx_func = jax.vmap(
            jax.jacfwd(self.evaluate_terminal_cost, argnums=0),
            in_axes=(0, None),
            out_axes=0,
        )
        return terminal_cx_func(x, jnp_t)

    @partial(jax.jit, static_argnums=(0,))
    def stage_cxx(self, x: jnp.ndarray, u: jnp.ndarray, t: Union[jnp.ndarray, int]):
        """Gradient of cost with respect to the given state

        Args:
            x (jnp.ndarray): states, (batch_size, state_size)
            u (jnp.ndarray): inputs, (batch_size, input_size)
            t (Union[jnp.ndarray, int]): time step

        Returns:
            jnp:ndarray: hessian of cost with respect to the state,
                shape (batch_size, state_size, state_size)
        """
        # See: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
        # jacfwd(jacrev(f)) is typically the most efficient
        jnp_t = jnp.ones(1, dtype=jnp.int32) * t
        stage_cxx_func = jax.vmap(
            jax.jacfwd(jax.jacrev(self.evaluate_stage_cost, argnums=0), argnums=0),
            in_axes=(0, 0, None),
            out_axes=0,
        )
        return stage_cxx_func(x, u, jnp_t)[:, 0, :, :]  # without unused dim

    @partial(jax.jit, static_argnums=(0,))
    def terminal_cxx(self, x: jnp.ndarray, t: Union[jnp.ndarray, int]):
        """Hessian of cost with respect to the given state

        Args:
            x (jnp.ndarray): states, (batch_size, state_size)
            t (Union[jnp.ndarray, int]): time step

        Returns:
            jnp:ndarray: hessian of terminal cost with respect to the state,
                shape (batch_size, state_size, state_size)
        """
        # See: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
        # jacfwd(jacrev(f)) is typically the most efficient
        jnp_t = jnp.ones(1, dtype=jnp.int32) * t
        terminal_cxx_func = jax.vmap(
            jax.jacfwd(jax.jacrev(self.evaluate_terminal_cost, argnums=0), argnums=0),
            in_axes=(0, None),
            out_axes=0,
        )
        return terminal_cxx_func(x, jnp_t)[:, 0, :, :]  # without unused dim

    @partial(jax.jit, static_argnums=(0,))
    def cux(self, x: jnp.ndarray, u: jnp.ndarray, t: Union[jnp.ndarray, int]):
        """Hessian of cost with respect to the given input and state

        Args:
            x (jnp.ndarray): states, (batch_size, state_size)
            u (jnp.ndarray): inputs, (batch_size, input_size)
            t (Union[jnp.ndarray, int]): time step

        Returns:
            jnp:ndarray: hessian of terminal cost cost
                with respect to the input and state,
                shape (batch_size, input_size, state_size)
        """
        # See: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
        # jacfwd(jacrev(f)) is typically the most efficient
        jnp_t = jnp.ones(1, dtype=jnp.int32) * t
        cux_func = jax.vmap(
            jax.jacfwd(jax.jacrev(self.evaluate_stage_cost, argnums=1), argnums=0),
            in_axes=(0, 0, None),
            out_axes=0,
        )
        return cux_func(x, u, jnp_t)[:, 0, :, :]  # without unused dim

    @partial(jax.jit, static_argnums=(0,))
    def cxu(self, x: jnp.ndarray, u: jnp.ndarray, t: Union[jnp.ndarray, int]):
        """Hessian of cost with respect to the given input and state

        Args:
            x (jnp.ndarray): states, (batch_size, state_size)
            u (jnp.ndarray): inputs, (batch_size, input_size)
            t (Union[jnp.ndarray, int]): time step

        Returns:
            jnp:ndarray: hessian of terminal cost
                with respect to the state and input,
                shape (batch_size, state_size, input_size)
        """
        # See: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
        # jacfwd(jacrev(f)) is typically the most efficient
        jnp_t = jnp.ones(1, dtype=jnp.int32) * t
        cux_func = jax.vmap(
            jax.jacfwd(jax.jacrev(self.evaluate_stage_cost, argnums=0), argnums=1),
            in_axes=(0, 0, None),
            out_axes=0,
        )
        return cux_func(x, u, jnp_t)[:, 0, :, :]  # without unused dim

    @partial(jax.jit, static_argnums=(0,))
    def cuu(self, x: jnp.ndarray, u: jnp.ndarray, t: Union[jnp.ndarray, int]):
        """Hessian of cost with respect to the given input

        Args:
            x (jnp.ndarray): states, (batch_size, state_size)
            u (jnp.ndarray): inputs, (batch_size, input_size)
            t (Union[jnp.ndarray, int]): time step

        Returns:
            jnp:ndarray: hessian of terminal cost
                with respect to the state and input,
                shape (batch_size, input_size, input_size)
        """
        # See: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
        # jacfwd(jacrev(f)) is typically the most efficient
        jnp_t = jnp.ones(1, dtype=jnp.int32) * t
        cuu_func = jax.vmap(
            jax.jacfwd(jax.jacrev(self.evaluate_stage_cost, argnums=1), argnums=1),
            in_axes=(0, 0, None),
            out_axes=0,
        )
        return cuu_func(x, u, jnp_t)[:, 0, :, :]  # without unused dim


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
        """Evaluate statge cost

        Args:
            x (jnp.ndarray): states, shape (state_size, )
            u (jnp.ndarray): inputs, shape (input_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            jnp:ndarray: stage cost, shape (1, )
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
        """Evaluate terminal cost

        Args:
            x (jnp.ndarray): states, (state_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            jnp:ndarray: terminal cost, shape (1, )
        """
        return jnp.matmul(
            jnp.matmul(x[jnp.newaxis, :], self._Qf), x[:, jnp.newaxis]
        ).ravel()
