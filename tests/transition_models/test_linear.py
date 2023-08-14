import pytest
from jax import numpy as jnp
import numpy as np

from apop.transition_models.basic.linear import LinearTransitionModel


class TestLinearTransitionModel:
    def test_predict_batched_trajectory_numeric_assert_fixed_value(self):
        numpy_A = np.eye(3)
        numpy_B = np.ones((3, 2))
        jax_A = jnp.array(numpy_A)
        jax_B = jnp.array(numpy_B)

        linear_model = LinearTransitionModel(jax_A, jax_B)
        batch_size = 5
        pred_len = 6

        curr_x = np.ones(3) * 0.1
        batched_x = np.tile(curr_x, (batch_size, 1))
        batched_u_sequence = np.ones((batch_size, pred_len, 2)) * 0.1
        actual_batched_pred_x_sequence = np.array(
            linear_model.predict_batched_trajectory(
                jnp.array(batched_x), jnp.array(batched_u_sequence)
            ).block_until_ready()
        )

        # Ax + Bu
        for i in range(batch_size):
            x = curr_x
            for t in range(pred_len):
                next_x = np.matmul(numpy_A, x[:, np.newaxis]) + np.matmul(
                    numpy_B, batched_u_sequence[i, t][:, np.newaxis]
                )
                x = next_x.flatten()
                assert np.allclose(
                    next_x.flatten(), actual_batched_pred_x_sequence[i, t + 1]
                )

    def test_predict_batched_trajectory_numeric_assert_random_value(self):
        numpy_A = np.array([[0.5, 0.2, 0.1], [-0.3, 0.25, -0.1], [0.0, 0.9, 0.01]])
        numpy_B = np.array([[0.21, 0.31], [-0.14, 0.015], [0.12, 0.01]])
        jax_A = jnp.array(numpy_A)
        jax_B = jnp.array(numpy_B)

        linear_model = LinearTransitionModel(jax_A, jax_B)
        batch_size = 5
        pred_len = 6

        np_drng = np.random.default_rng()
        curr_x = np_drng.random(size=3) * 0.1
        batched_x = np.tile(curr_x, (batch_size, 1))
        batched_u_sequence = np_drng.random(size=(batch_size, pred_len, 2)) * 0.1
        actual_batched_pred_x_sequence = np.array(
            linear_model.predict_batched_trajectory(
                jnp.array(batched_x), jnp.array(batched_u_sequence)
            ).block_until_ready()
        )

        # Ax + Bu
        for i in range(batch_size):
            x = curr_x
            for t in range(pred_len):
                next_x = np.matmul(numpy_A, x[:, np.newaxis]) + np.matmul(
                    numpy_B, batched_u_sequence[i, t][:, np.newaxis]
                )
                x = next_x.flatten()
                assert np.allclose(
                    next_x.flatten(), actual_batched_pred_x_sequence[i, t + 1]
                )

    def test_batched_next_state_numeric_assert_random_value(self):
        numpy_A = np.array([[0.5, 0.2, 0.1], [-0.3, 0.25, -0.1], [0.0, 0.9, 0.01]])
        numpy_B = np.array([[0.21, 0.31], [-0.14, 0.015], [0.12, 0.01]])
        jax_A = jnp.array(numpy_A)
        jax_B = jnp.array(numpy_B)

        linear_model = LinearTransitionModel(jax_A, jax_B)
        np_drng = np.random.default_rng()
        batch_size = 10
        curr_x = np_drng.random(size=(batch_size, 3)) * 0.1
        u = np_drng.random(size=(batch_size, 2)) * 0.1
        actual_pred_x = np.array(
            linear_model.predict_batched_next_state(
                jnp.array(curr_x), jnp.array(u), 1
            ).block_until_ready()
        )

        for i in range(batch_size):
            next_x = np.matmul(numpy_A, curr_x[i][:, np.newaxis]) + np.matmul(
                numpy_B, u[i][:, np.newaxis]
            )
            assert np.allclose(next_x.flatten(), actual_pred_x[i])

    def test_predict_trajectory_numeric_assert_random_value(self):
        numpy_A = np.array([[0.5, 0.2, 0.1], [-0.3, 0.25, -0.1], [0.0, 0.9, 0.01]])
        numpy_B = np.array([[0.21, 0.31], [-0.14, 0.015], [0.12, 0.01]])
        jax_A = jnp.array(numpy_A)
        jax_B = jnp.array(numpy_B)

        linear_model = LinearTransitionModel(jax_A, jax_B)
        pred_len = 6

        np_drng = np.random.default_rng()
        curr_x = np_drng.random(size=3) * 0.1
        u_sequence = np_drng.random(size=(pred_len, 2)) * 0.1
        actual_pred_x_sequence = np.array(
            linear_model.predict_trajectory(
                jnp.array(curr_x), jnp.array(u_sequence)
            ).block_until_ready()
        )

        x = curr_x
        for t in range(pred_len):
            next_x = np.matmul(numpy_A, x[:, np.newaxis]) + np.matmul(
                numpy_B, u_sequence[t][:, np.newaxis]
            )
            x = next_x.flatten()
            assert np.allclose(next_x.flatten(), actual_pred_x_sequence[t + 1])

    def test_fx_numeric_assert_random_value(self):
        numpy_A = np.array([[0.5, 0.2, 0.1], [-0.3, 0.25, -0.1], [0.0, 0.9, 0.01]])
        numpy_B = np.array([[0.21, 0.31], [-0.14, 0.015], [0.12, 0.01]])
        jax_A = jnp.array(numpy_A)
        jax_B = jnp.array(numpy_B)

        linear_model = LinearTransitionModel(jax_A, jax_B)
        batch_size = 5

        np_drng = np.random.default_rng()
        curr_x = np_drng.random(size=3) * 0.1
        batched_x = np.tile(curr_x, (batch_size, 1))
        batched_u = np_drng.random(size=(batch_size, 2)) * 0.1
        actual_fx = np.array(
            linear_model.fx(
                jnp.array(batched_x), jnp.array(batched_u), 0
            ).block_until_ready()
        )

        for i in range(batch_size):
            assert np.allclose(actual_fx[i], numpy_A, atol=1e-3)

    def test_fu_numeric_assert_random_value(self):
        numpy_A = np.array([[0.5, 0.2, 0.1], [-0.3, 0.25, -0.1], [0.0, 0.9, 0.01]])
        numpy_B = np.array([[0.21, 0.31], [-0.14, 0.015], [0.12, 0.01]])
        jax_A = jnp.array(numpy_A)
        jax_B = jnp.array(numpy_B)

        linear_model = LinearTransitionModel(jax_A, jax_B)
        batch_size = 5

        np_drng = np.random.default_rng()
        curr_x = np_drng.random(size=3) * 0.1
        batched_x = np.tile(curr_x, (batch_size, 1))
        batched_u = np_drng.random(size=(batch_size, 2)) * 0.1
        actual_fu = np.array(
            linear_model.fu(
                jnp.array(batched_x), jnp.array(batched_u), 0
            ).block_until_ready()
        )

        for i in range(batch_size):
            assert np.allclose(actual_fu[i], numpy_B, atol=1e-3)


if __name__ == "__main__":
    pytest.main()