import argparse
import os

import gymnasium
import numpy as np
from jax import numpy as jnp
from gymnasium.wrappers.record_video import RecordVideo

import apop
from apop.controllers.cem import TruncatedGaussianCrossEntropyMethod
from apop.cost_functions.classic_control.cartpole import CartPoleCostFunction
from apop.transition_models.classic_control.cartpole import SwingUpCartPoleModel

# pip install gymnasium[classic-control]


def run(args):
    # build env
    if args.record_video:
        env = RecordVideo(
            gymnasium.make("ContinuousSwingUpCartPole-v0", render_mode="rgb_array"),
            video_folder=args.video_path,
        )
    else:
        env = gymnasium.make("ContinuousSwingUpCartPole-v0", render_mode="human")

    # build controller
    cost_function = CartPoleCostFunction()
    transition_model = SwingUpCartPoleModel()
    T = 50
    controller = TruncatedGaussianCrossEntropyMethod(
        transition_model,
        cost_function,
        T,
        num_iterations=25,
        sample_size=1000,
        num_elites=50,
        alpha=0.1,
        initial_variance=np.array([100.0]),
        upper_bound=np.array([20]),
        lower_bound=np.array([-20]),
    )

    # run control
    state, info = env.reset()
    terminated = False
    truncated = False

    optimized_u_sequence = jnp.zeros((T, 1))
    total_score = 0.0
    while not (terminated or truncated):
        if args.apply_control:
            optimized_u_sequence = controller.control(
                jnp.array(state), optimized_u_sequence
            )
        else:
            action = env.action_space.sample()

        action = np.array(optimized_u_sequence[0])
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        env.render()
        total_score += reward
        print(f"total_score = {total_score}")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--apply_control", action="store_true")
    default_path = os.path.join(os.path.dirname(__file__), "video")
    parser.add_argument("--video_path", type=str, default=default_path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
