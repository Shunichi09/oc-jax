import pytest
from jax import numpy as jnp
import numpy as np

from apop.cost_function import CostFunction, QuadraticCostFunction


class TestCostFunction:
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

        drng = np.random.default_rng()

        numpy_Q = drng.random(size=(state_size, state_size)) * 0.1
        Q = jnp.array(numpy_Q)

        numpy_Qf = drng.random(size=(state_size, state_size)) * 0.2
        Qf = jnp.array(numpy_Qf)

        numpy_R = drng.random(size=(input_size, input_size)) * 0.3
        R = jnp.array(numpy_R)

        numpy_F = drng.random(size=(state_size, input_size)) * 0.3
        F = jnp.array(numpy_F)
        cost_func = QuadraticCostFunction(Q, Qf, R, F)

        numpy_x_seq = drng.random(size=(batch_size, pred_len + 1, state_size)) * 0.1
        numpy_u_seq = drng.random(size=(batch_size, pred_len, input_size)) * 0.1
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

    class TestQuadraticFunc:
        def test_evaluate_state_cost_numeric_assert_with_random_value(self):
            state_size = 3
            input_size = 2

            drng = np.random.default_rng()

            numpy_Q = drng.random(size=(state_size, state_size)) * 0.1
            Q = jnp.array(numpy_Q)

            numpy_Qf = drng.random(size=(state_size, state_size)) * 0.2
            Qf = jnp.array(numpy_Qf)

            numpy_R = drng.random(size=(input_size, input_size)) * 0.3
            R = jnp.array(numpy_R)

            numpy_F = drng.random(size=(state_size, input_size)) * 0.3
            F = jnp.array(numpy_F)
            cost_func = QuadraticCostFunction(Q, Qf, R, F)

            numpy_x = drng.random(size=state_size) * 0.1
            numpy_u = drng.random(size=input_size) * 0.1
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

            drng = np.random.default_rng()

            numpy_Q = drng.random(size=(state_size, state_size)) * 0.1
            Q = jnp.array(numpy_Q)

            numpy_Qf = drng.random(size=(state_size, state_size)) * 0.2
            Qf = jnp.array(numpy_Qf)

            numpy_R = drng.random(size=(input_size, input_size)) * 0.3
            R = jnp.array(numpy_R)

            numpy_F = drng.random(size=(state_size, input_size)) * 0.3
            F = jnp.array(numpy_F)
            cost_func = QuadraticCostFunction(Q, Qf, R, F)

            numpy_x = drng.random(size=state_size) * 0.1
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


if __name__ == "__main__":
    pytest.main()
