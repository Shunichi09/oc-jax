import pytest
from jax import numpy as jnp
import numpy as np

from apop.cost_functions.basic.quadratic import QuadraticCostFunction
import jax
from apop.random import np_drng


def generate_symmetric_matrix(shape):
    key = jax.random.PRNGKey(np_drng.integers(1000))
    matrix = jax.random.normal(key, shape)
    return (matrix + matrix.T) * 0.5


class TestQuadraticFunc:
    def test_evaluate_state_cost_numeric_assert_with_random_value(self):
        state_size = 3
        input_size = 2

        np_drng = np.random.default_rng()

        numpy_Q = np_drng.random(size=(state_size, state_size)) * 0.1
        Q = jnp.array(numpy_Q)

        numpy_Qf = np_drng.random(size=(state_size, state_size)) * 0.2
        Qf = jnp.array(numpy_Qf)

        numpy_R = np_drng.random(size=(input_size, input_size)) * 0.3
        R = jnp.array(numpy_R)

        numpy_F = np_drng.random(size=(state_size, input_size)) * 0.3
        F = jnp.array(numpy_F)
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        numpy_x = np_drng.random(size=state_size) * 0.1
        numpy_u = np_drng.random(size=input_size) * 0.1
        x = jnp.array(numpy_x)
        u = jnp.array(numpy_u)

        actual_cost = cost_func.evaluate_stage_cost(
            x, u, jnp.ones(1)
        ).block_until_ready()

        # compute numpy costs
        cost = (
            np.matmul(
                numpy_x[np.newaxis, :],
                np.matmul(numpy_Q, numpy_x[:, np.newaxis]),
            )
            + np.matmul(
                numpy_u[np.newaxis, :],
                np.matmul(numpy_R, numpy_u[:, np.newaxis]),
            )
            + 2.0
            * np.matmul(
                numpy_x[np.newaxis, :],
                np.matmul(numpy_F, numpy_u[:, np.newaxis]),
            )
        )
        assert np.allclose(actual_cost, cost)

    def test_evaluate_terminal_cost_numeric_assert_with_random_value(self):
        state_size = 3
        input_size = 2

        np_drng = np.random.default_rng()

        numpy_Q = np_drng.random(size=(state_size, state_size)) * 0.1
        Q = jnp.array(numpy_Q)

        numpy_Qf = np_drng.random(size=(state_size, state_size)) * 0.2
        Qf = jnp.array(numpy_Qf)

        numpy_R = np_drng.random(size=(input_size, input_size)) * 0.3
        R = jnp.array(numpy_R)

        numpy_F = np_drng.random(size=(state_size, input_size)) * 0.3
        F = jnp.array(numpy_F)
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        numpy_x = np_drng.random(size=state_size) * 0.1
        x = jnp.array(numpy_x)

        actual_cost = cost_func.evaluate_terminal_cost(
            x, jnp.ones(1)
        ).block_until_ready()

        # compute numpy costs
        cost = np.matmul(
            numpy_x[np.newaxis, :],
            np.matmul(numpy_Qf, numpy_x[:, np.newaxis]),
        )
        assert np.allclose(actual_cost, cost)

    def test_evaluate_trajectory_cost_numeric_assert_with_fixed_value(self):
        state_size = 3
        batch_size = 2
        pred_len = 3
        input_size = 2

        Q = jnp.eye(state_size) * 0.1
        Qf = jnp.eye(state_size) * 0.2
        R = jnp.eye(input_size) * 0.3
        F = jnp.zeros((state_size, input_size))
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        x_seq = jnp.ones((batch_size, pred_len + 1, state_size))
        u_seq = jnp.ones((batch_size, pred_len, input_size))

        actual_cost = np.array(
            cost_func.evaluate_batched_trajectory_cost(x_seq, u_seq).block_until_ready()
        )

        # stage state 0.1 * 3 * 3
        # stage input 0.3 * 2 * 3
        # terminal 0.2 * 3
        assert np.allclose(actual_cost, np.array([[3.3], [3.3]]))

    def test_evaluate_trajectory_cost_numeric_assert_with_random_value(self):
        state_size = 3
        batch_size = 8
        pred_len = 5
        input_size = 2

        np_drng = np.random.default_rng()

        numpy_Q = np_drng.random(size=(state_size, state_size)) * 0.1
        Q = jnp.array(numpy_Q)

        numpy_Qf = np_drng.random(size=(state_size, state_size)) * 0.2
        Qf = jnp.array(numpy_Qf)

        numpy_R = np_drng.random(size=(input_size, input_size)) * 0.3
        R = jnp.array(numpy_R)

        numpy_F = np_drng.random(size=(state_size, input_size)) * 0.3
        F = jnp.array(numpy_F)
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        numpy_x_seq = np_drng.random(size=(batch_size, pred_len + 1, state_size)) * 0.1
        numpy_u_seq = np_drng.random(size=(batch_size, pred_len, input_size)) * 0.1
        x_seq = jnp.array(numpy_x_seq)
        u_seq = jnp.array(numpy_u_seq)

        actual_cost = np.array(
            cost_func.evaluate_batched_trajectory_cost(x_seq, u_seq).block_until_ready()
        )

        # compute numpy costs
        costs = np.zeros((batch_size, 1))
        for i in range(batch_size):
            total = 0.0
            for t in range(pred_len):
                stage = (
                    np.matmul(
                        numpy_x_seq[i, t][np.newaxis, :],
                        np.matmul(numpy_Q, numpy_x_seq[i, t][:, np.newaxis]),
                    )
                    + np.matmul(
                        numpy_u_seq[i, t][np.newaxis, :],
                        np.matmul(numpy_R, numpy_u_seq[i, t][:, np.newaxis]),
                    )
                    + 2.0
                    * np.matmul(
                        numpy_x_seq[i, t][np.newaxis, :],
                        np.matmul(numpy_F, numpy_u_seq[i, t][:, np.newaxis]),
                    )
                )
                total += float(stage)

            terminal = np.matmul(
                numpy_x_seq[i, -1],
                np.matmul(numpy_Qf, numpy_x_seq[i, -1][:, np.newaxis]),
            )
            costs[i] = total + float(terminal)

        assert np.allclose(actual_cost, costs)

    def test_stage_cx_numeric_assert_with_random_value(self):
        state_size = 3
        input_size = 2
        batch_size = 5

        key = jax.random.PRNGKey(0)
        Q = generate_symmetric_matrix((state_size, state_size))
        Qf = generate_symmetric_matrix((state_size, state_size))
        R = generate_symmetric_matrix((input_size, input_size))
        F = jax.random.normal(key, (state_size, input_size))
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        x = jax.random.uniform(key, (batch_size, state_size)) * 0.1
        u = jax.random.uniform(key, (batch_size, input_size)) * 0.1

        actual_stage_cx = cost_func.stage_cx(x, u, jnp.ones(1)).block_until_ready()

        expected = 2.0 * np.matmul(
            np.array(x)[:, np.newaxis, :], np.tile(np.array(Q), (batch_size, 1, 1))
        ) + 2.0 * np.matmul(
            np.tile(np.array(F), (batch_size, 1, 1)), np.array(u)[:, :, np.newaxis]
        ).reshape(
            batch_size, 1, state_size
        )
        assert np.allclose(expected, np.array(actual_stage_cx), atol=1e-3)

    def test_stage_cu_numeric_assert_with_random_value(self):
        state_size = 3
        input_size = 2
        batch_size = 5

        key = jax.random.PRNGKey(0)
        Q = generate_symmetric_matrix((state_size, state_size))
        Qf = generate_symmetric_matrix((state_size, state_size))
        R = generate_symmetric_matrix((input_size, input_size))
        F = jax.random.normal(key, (state_size, input_size))
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        x = jax.random.uniform(key, (batch_size, state_size)) * 0.1
        u = jax.random.uniform(key, (batch_size, input_size)) * 0.1

        actual_stage_cu = cost_func.cu(x, u, jnp.ones(1)).block_until_ready()

        expected = 2.0 * np.matmul(
            np.array(u)[:, np.newaxis, :], np.tile(np.array(R), (batch_size, 1, 1))
        ) + 2.0 * np.matmul(
            np.array(x)[:, np.newaxis, :], np.tile(np.array(F), (batch_size, 1, 1))
        )
        assert np.allclose(expected, np.array(actual_stage_cu), atol=1e-3)

    def test_terminal_cx_numeric_assert_with_random_value(self):
        state_size = 2
        input_size = 5
        batch_size = 10

        key = jax.random.PRNGKey(0)
        Q = generate_symmetric_matrix((state_size, state_size))
        Qf = generate_symmetric_matrix((state_size, state_size))
        R = generate_symmetric_matrix((input_size, input_size))
        F = jax.random.normal(key, (state_size, input_size))
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        x = jax.random.uniform(key, (batch_size, state_size)) * 0.1
        actual_terminal_cx = cost_func.terminal_cx(x, jnp.ones(1)).block_until_ready()

        expected = 2.0 * np.matmul(
            np.array(x[:, np.newaxis, :]), np.tile(np.array(Qf), (batch_size, 1, 1))
        )
        assert np.allclose(expected, np.array(actual_terminal_cx), atol=1e-3)

    def test_stage_cxx_numeric_assert_with_random_value(self):
        state_size = 2
        input_size = 5
        batch_size = 10

        key = jax.random.PRNGKey(0)
        Q = generate_symmetric_matrix((state_size, state_size))
        Qf = generate_symmetric_matrix((state_size, state_size))
        R = generate_symmetric_matrix((input_size, input_size))
        F = jax.random.normal(key, (state_size, input_size))
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        x = jax.random.uniform(key, (batch_size, state_size)) * 0.1
        u = jax.random.uniform(key, (batch_size, input_size)) * 0.1
        actual_stage_cxx = cost_func.stage_cxx(x, u, jnp.ones(1)).block_until_ready()

        expected = 2.0 * np.tile(Q, (batch_size, 1, 1))
        assert np.allclose(expected, np.array(actual_stage_cxx), atol=1e-2)

    def test_terminal_cxx_numeric_assert_with_random_value(self):
        state_size = 2
        input_size = 5
        batch_size = 10

        key = jax.random.PRNGKey(0)
        Q = generate_symmetric_matrix((state_size, state_size))
        Qf = generate_symmetric_matrix((state_size, state_size))
        R = generate_symmetric_matrix((input_size, input_size))
        F = jax.random.normal(key, (state_size, input_size))
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        x = jax.random.uniform(key, (batch_size, state_size)) * 0.1
        actual_terminal_cxx = cost_func.terminal_cxx(x, jnp.ones(1)).block_until_ready()

        expected = 2.0 * np.tile(Qf, (batch_size, 1, 1))
        assert np.allclose(expected, np.array(actual_terminal_cxx), atol=1e-2)

    def test_cux_numeric_assert_with_random_value(self):
        state_size = 2
        input_size = 5
        batch_size = 10

        key = jax.random.PRNGKey(0)
        Q = generate_symmetric_matrix((state_size, state_size))
        Qf = generate_symmetric_matrix((state_size, state_size))
        R = generate_symmetric_matrix((input_size, input_size))
        F = jax.random.normal(key, (state_size, input_size))
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        x = jax.random.uniform(key, (batch_size, state_size)) * 0.1
        u = jax.random.uniform(key, (batch_size, input_size)) * 0.1
        actual_cux = cost_func.cux(x, u, jnp.ones(1)).block_until_ready()

        expected = 2.0 * np.tile(F.T, (batch_size, 1, 1))
        assert np.allclose(expected, np.array(actual_cux), atol=1e-2)

    def test_cxu_numeric_assert_with_random_value(self):
        state_size = 2
        input_size = 5
        batch_size = 10

        key = jax.random.PRNGKey(0)
        Q = generate_symmetric_matrix((state_size, state_size))
        Qf = generate_symmetric_matrix((state_size, state_size))
        R = generate_symmetric_matrix((input_size, input_size))
        F = jax.random.normal(key, (state_size, input_size))
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        x = jax.random.uniform(key, (batch_size, state_size)) * 0.1
        u = jax.random.uniform(key, (batch_size, input_size)) * 0.1
        actual_cxu = cost_func.cxu(x, u, jnp.ones(1)).block_until_ready()

        expected = 2.0 * np.tile(F, (batch_size, 1, 1))
        assert np.allclose(expected, np.array(actual_cxu), atol=1e-2)

    def test_cuu_numeric_assert_with_random_value(self):
        state_size = 2
        input_size = 5
        batch_size = 10

        key = jax.random.PRNGKey(0)
        Q = generate_symmetric_matrix((state_size, state_size))
        Qf = generate_symmetric_matrix((state_size, state_size))
        R = generate_symmetric_matrix((input_size, input_size))
        F = jax.random.normal(key, (state_size, input_size))
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        x = jax.random.uniform(key, (batch_size, state_size)) * 0.1
        u = jax.random.uniform(key, (batch_size, input_size)) * 0.1
        actual_cuu = cost_func.cuu(x, u, jnp.ones(1)).block_until_ready()

        expected = 2.0 * np.tile(R, (batch_size, 1, 1))
        assert np.allclose(expected, np.array(actual_cuu), atol=1e-2)


if __name__ == "__main__":
    pytest.main()
