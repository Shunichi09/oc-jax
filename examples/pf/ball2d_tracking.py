import argparse
import os

import matplotlib.pyplot as plt
import gymnasium
import numpy as np
from gymnasium.wrappers.record_video import RecordVideo
from jax import numpy as jnp

import apop
from apop.controllers.lqr import LinearQuadraticRegulator
from apop.cost_functions.basic.quadratic import QuadraticCostFunction

# pip install gymnasium[classic-control]


def run(args):
    # build env
    if args.record_video:
        env = RecordVideo(
            gymnasium.make("Ball2dTracking-v0", render_mode="rgb_array"),
            video_folder=args.video_path,
        )
    else:
        env = gymnasium.make("Ball2dTracking-v0", render_mode="human")

    # build controller
    # cost_function = QuadraticCostFunction(
    #     jnp.eye(4) * 10, jnp.eye(4) * 10, jnp.eye(1) * 0.1, jnp.zeros((4, 1))
    # )
    # transition_model = LinearInvertedCartPoleModel()
    T = 50
    # controller = LinearQuadraticRegulator(transition_model, cost_function, T)

    # run control
    state, info = env.reset()
    terminated = False
    truncated = False

    gt_internal_state = []
    # optimized_u_sequence = jnp.zeros((T, 1))
    total_score = 0.0

    forward_velocity = 0.5
    angular_velocity = 1.0
    while not (terminated or truncated):
        next_state, reward, terminated, truncated, info = env.step(
            np.array([forward_velocity, angular_velocity], dtype=np.float32)
        )
        state = next_state
        env.render()
        gt_internal_state.append(info["internal_state"])

    env.close()

    # visualize
    gt_internal_state = np.array(gt_internal_state)
    position_figure = plt.figure()
    position_axis = position_figure.add_axes(111)
    position_axis.plot(
        gt_internal_state[:, 0], gt_internal_state[:, 1], label="gt", color="r"
    )
    position_axis.set_xlim(-1, 0.5)
    position_axis.set_ylim(-0.75, 0.75)
    position_axis.axis("equal")
    position_axis.set_xlabel("x [m]")
    position_axis.set_ylabel("y [m]")

    angle_figure = plt.figure()
    angle_axis = angle_figure.add_axes(111)
    angle_axis.plot(
        np.arange(len(gt_internal_state)) * env.unwrapped.dt,
        gt_internal_state[:, 2],
        label="gt",
        color="r",
    )
    angle_axis.set_xlabel("time")
    angle_axis.set_ylabel("angle [rad]")
    plt.show()


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
