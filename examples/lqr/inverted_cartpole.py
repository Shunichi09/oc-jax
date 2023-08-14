import argparse
import os

import gymnasium
import numpy as np
from gymnasium.wrappers.record_video import RecordVideo
from jax import numpy as jnp

import apop
from apop.controllers.lqr import LinearQuadraticRegulator
from apop.cost_functions.basic.quadratic import QuadraticCostFunction
from apop.transition_models.classic_control.cartpole import LinearInvertedCartPoleModel

# pip install gymnasium[classic-control]


def run(args):
    # build env
    if args.record_video:
        env = RecordVideo(
            gymnasium.make("ContinuousInvertedCartPole-v0", render_mode="rgb_array"),
            video_folder=args.video_path,
        )
    else:
        env = gymnasium.make("ContinuousInvertedCartPole-v0", render_mode="human")

    # build controller
    cost_function = QuadraticCostFunction(
        jnp.eye(4) * 10, jnp.eye(4) * 10, jnp.eye(1) * 0.1, jnp.zeros((4, 1))
    )
    transition_model = LinearInvertedCartPoleModel()
    T = 50
    controller = LinearQuadraticRegulator(transition_model, cost_function, T)

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
