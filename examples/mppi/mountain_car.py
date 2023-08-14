import argparse
import os

import gymnasium
import numpy as np
from gymnasium.wrappers.record_video import RecordVideo
from jax import numpy as jnp

import apop
from apop.controllers.mppi import MPPI
from apop.cost_functions.basic.quadratic import QuadraticCostFunction
from apop.transition_models.classic_control.mountain_car import MountainCarModel


def run(args):
    # build env
    if args.record_video:
        env = RecordVideo(
            gymnasium.make(
                "MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.0
            ),
            video_folder=args.video_path,
        )
    else:
        env = gymnasium.make(
            "MountainCarContinuous-v0", render_mode="human", goal_velocity=0.0
        )

    # build controller
    cost_function = QuadraticCostFunction(
        Q=jnp.diag(jnp.array([0.0, 0.0])),
        Qf=jnp.diag(jnp.array([1000.0, 10.0])),
        R=jnp.eye(1) * 0.01,  # NOTE: In mppi, R is ignored.
        F=jnp.zeros((2, 1)),
        target_state=jnp.array([0.5, 0.0]),  # NOTE: reach larger than 0.45 so set 0.5.
    )
    transition_model = MountainCarModel()
    T = 50
    controller = MPPI(
        transition_model,
        cost_function,
        T,
        sample_size=500,
        alpha=1.0,
        gamma=1.0,
        lmb=1.0,
        initial_covariance=np.eye(1) * 1.5,
        upper_bound=np.array([1.0]),
        lower_bound=np.array([-1.0]),
    )

    # run control
    state, info = env.reset()
    terminated = False
    truncated = False

    optimized_u_sequence = jnp.zeros((T, 1))
    total_score = 0.0
    while not (terminated or truncated):
        if args.random_action:
            action = env.action_space.sample()
        else:
            optimized_u_sequence = controller.control(
                jnp.array(state), optimized_u_sequence
            )

        action = np.array(optimized_u_sequence[0])
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        env.render()
        total_score += reward
        # forward inputs
        optimized_u_sequence = jnp.concatenate(
            [optimized_u_sequence[1:], jnp.zeros((1, 1))], axis=0
        )
        print(f"state = {state}")
        print(f"total_score = {total_score}")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--random_action", action="store_true")
    default_path = os.path.join(os.path.dirname(__file__), "video")
    parser.add_argument("--video_path", type=str, default=default_path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
