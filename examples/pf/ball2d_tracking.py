import argparse
import os

import gymnasium
import jax
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers.record_video import RecordVideo
from jax import numpy as jnp

import apop
from apop.filters.particle_filter import ParticleFilter
from apop.observation_models.mujoco.ball_tracking import (
    Ball2dTrackingGaussianObservationModel,
)
from apop.transition_models.mujoco.ball_tracking import (
    Ball2dTrackingGaussianTransitionModel,
)
from apop.distributions.gaussian import Gaussian
from apop.random import new_key
from jax import config

config.update("jax_disable_jit", True)
# config.update("jax_debug_nans", True)


def run(args):
    # build env
    if args.record_video:
        env = RecordVideo(
            gymnasium.make("Ball2dTracking-v0", render_mode="rgb_array"),
            video_folder=args.video_path,
        )
    else:
        env = gymnasium.make("Ball2dTracking-v0", render_mode="human")

    # Build filter
    _, info = env.reset()
    transition_model = Ball2dTrackingGaussianTransitionModel(
        env.unwrapped.dt, jnp.array(np.diag([2.0**2, 0.6**2]))
    )
    observation_model = Ball2dTrackingGaussianObservationModel(
        jnp.array(
            np.stack(
                [
                    np.diag([0.2**2, 0.4**2]),
                    np.diag([0.2**2, 0.4**2]),
                    np.diag([0.2**2, 0.4**2]),
                ],
                axis=0,
            )
        ),
        jnp.array(info["landmark_positions"]),
    )
    filter = ParticleFilter(
        transition_model,
        observation_model,
        Gaussian(jnp.array([-0.3, -0.5, 0.0]), jnp.eye(3) * (0.01**2)),
        num_particles=1000,
        resampling_threshold=1000,
    )

    # run control
    state, info = env.reset()
    terminated = False
    truncated = False

    gt_internal_state = []
    estimated_internal_state = []
    random_key = jax.random.PRNGKey(9)

    forward_velocity = 0.5
    angular_velocity = 1.0
    t = 0
    while not (terminated or truncated):
        input_array = np.array([forward_velocity, angular_velocity], dtype=np.float32)
        next_state, reward, terminated, truncated, info = env.step(input_array)
        print(f"Gt state: {info['internal_state']}")
        print(f"Observation state: \n{next_state}")

        # precition step
        random_key = new_key(random_key)
        filter.predict(jnp.array(input_array), t, random_key)

        # TODO: Support Cannot see landmarks
        random_key = new_key(random_key)
        estimated_state = filter.estimate(
            jnp.array(next_state)[jnp.newaxis, :, :],
            jnp.ones(len(info["landmark_positions"]), dtype=jnp.bool_),
            random_key,
        ).block_until_ready()
        print(f"Estimation state: {np.array(estimated_state)}")

        env.render()
        gt_internal_state.append(info["internal_state"])
        estimated_internal_state.append(np.array(estimated_state))
        t += 1

    env.close()

    # visualize
    gt_internal_state = np.array(gt_internal_state)
    estimated_internal_state = np.array(estimated_internal_state)

    position_figure = plt.figure()
    position_axis = position_figure.add_axes(111)
    position_axis.plot(
        gt_internal_state[:, 0],
        gt_internal_state[:, 1],
        label="gt",
        color="r",
    )
    position_axis.plot(
        estimated_internal_state[:, 0],
        estimated_internal_state[:, 1],
        label="pred",
        color="b",
    )
    position_axis.set_xlim(-1, 0.5)
    position_axis.set_ylim(-0.75, 0.75)
    position_axis.axis("equal")
    position_axis.set_xlabel("x [m]")
    position_axis.set_ylabel("y [m]")
    position_axis.legend()
    position_figure.savefig("position.png")

    pos_x_figure = plt.figure()
    pos_x_axis = pos_x_figure.add_axes(111)
    pos_x_axis.plot(
        np.arange(len(gt_internal_state)) * env.unwrapped.dt,
        gt_internal_state[:, 0],
        label="gt",
        color="r",
    )
    pos_x_axis.plot(
        np.arange(len(gt_internal_state)) * env.unwrapped.dt,
        estimated_internal_state[:, 0],
        label="pred",
        color="b",
    )
    pos_x_axis.set_xlabel("time")
    pos_x_axis.set_ylabel("x [m]")
    pos_x_axis.legend()
    pos_x_figure.savefig("pos_x.png")

    pos_y_figure = plt.figure()
    pos_y_axis = pos_y_figure.add_axes(111)
    pos_y_axis.plot(
        np.arange(len(gt_internal_state)) * env.unwrapped.dt,
        gt_internal_state[:, 1],
        label="gt",
        color="r",
    )
    pos_y_axis.plot(
        np.arange(len(gt_internal_state)) * env.unwrapped.dt,
        estimated_internal_state[:, 1],
        label="pred",
        color="b",
    )
    pos_y_axis.set_xlabel("time")
    pos_y_axis.set_ylabel("y [m]")
    pos_y_axis.legend()
    pos_y_figure.savefig("pos_y.png")

    angle_figure = plt.figure()
    angle_axis = angle_figure.add_axes(111)
    angle_axis.plot(
        np.arange(len(gt_internal_state)) * env.unwrapped.dt,
        gt_internal_state[:, 2],
        label="gt",
        color="r",
    )
    angle_axis.plot(
        np.arange(len(gt_internal_state)) * env.unwrapped.dt,
        estimated_internal_state[:, 2],
        label="pred",
        color="b",
    )
    angle_axis.set_xlabel("time")
    angle_axis.set_ylabel("angle [rad]")
    position_axis.legend()
    angle_figure.savefig("angle.png")


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
