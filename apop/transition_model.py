import numpy as np
import jax
from jax import numpy as jnp
from functools import partial


class TransitionModel:
    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def predict_batched_trajectory(
        self, curr_x: jnp.ndarray, u_sequence: jnp.ndarray
    ) -> jnp.ndarray:
        """predict trajectories

        Args:
            curr_x (jnp.ndarray): current state, shape (batch_size, state_size)
            u_sequence (jnp.ndarray): inputs, shape (batch_size, pred_len, input_size)

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
        return batched_pred_trajectory_func(curr_x, u_sequence)

    @partial(jax.jit, static_argnums=(0,))
    def predict_trajectory(
        self, curr_x: jnp.ndarray, u_sequence: jnp.ndarray
    ) -> jnp.ndarray:
        """predict trajectories

        Args:
            curr_x (jnp.ndarray): current state, shape (state_size, )
            us (jnp.ndarray): inputs, shape (pred_len, input_size)

        Returns:
            pred_xs (jnp.ndarray): predicted state,
                shape(pred_len+1, state_size) including current state
        """
        state_size = curr_x.shape[0]
        pred_len = u_sequence.shape[0]

        pred_x_sequence = jnp.zeros((pred_len + 1, state_size))
        x = curr_x
        pred_x_sequence = pred_x_sequence.at[0].set(x)

        for t in range(pred_len):
            next_x = self.predict_next_state(x, u_sequence[t])
            pred_x_sequence = pred_x_sequence.at[t + 1].set(next_x)
            x = next_x

        return pred_x_sequence

    @partial(jax.jit, static_argnums=(0,))
    def predict_next_state(self, curr_x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """predict next state"""
        raise NotImplementedError("Implement the model")

    @partial(jax.jit, static_argnums=(0,))
    def fx(self, curr_x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """gradient of model with respect to the state in batch form

        Args:
            curr_x (jnp.ndarray): current state, (batch_size, state_size)
            u (jnp.ndarray): inputs, (batch_size, input_size)

        Returns:
            jnp:ndarray: gradient of model with respect to the state, shape (batch_size, state_size, state_size)
        """
        assert curr_x.shape[0] == u.shape[0]
        fx = jax.vmap(
            jax.jacfwd(self.predict_next_state, argnums=0), in_axes=0, out_axes=0
        )
        return fx(curr_x, u)

    @partial(jax.jit, static_argnums=(0,))
    def fu(self, curr_x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """gradient of model with respect to the input in batch form

        Args:
            curr_x (jnp.ndarray): current state, (batch_size, state_size)
            u (jnp.ndarray): inputs, (batch_size, input_size)

        Returns:
            jnp:ndarray: gradient of model with respect to the state, shape (batch_size, state_size, input_size)
        """
        assert curr_x.shape[0] == u.shape[0]
        fu = jax.vmap(
            jax.jacfwd(self.predict_next_state, argnums=1), in_axes=0, out_axes=0
        )
        return fu(curr_x, u)

    @partial(jax.jit, static_argnums=(0,))
    def fxx(self, curr_x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def fux(self, curr_x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def fuu(self, curr_x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class LinearTransitionModel(TransitionModel):
    """discrete linear model, x[k+1] = Ax[k] + Bu[k]

    Attributes:
        A (jnp.ndarray): shape(state_size, state_size)
        B (jnp.ndarray): shape(state_size, input_size)
    """

    def __init__(self, A: jnp.ndarray, B: jnp.ndarray):
        """ """
        super(LinearTransitionModel, self).__init__()
        assert A.shape[0] == B.shape[0]
        assert A.shape[0] == A.shape[1]
        self._A = A
        self._B = B

    @partial(jax.jit, static_argnums=(0,))
    def predict_next_state(self, curr_x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """predict next state

        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            u (numpy.ndarray): input, shape(input_size, )

        Returns:
            next_x (numpy.ndarray): next state, shape(state_size, ) or
                shape(pop_size, state_size)
        """
        next_x = jnp.matmul(self._A, curr_x[:, jnp.newaxis]) + jnp.matmul(
            self._B, u[:, jnp.newaxis]
        )
        return next_x.flatten()
