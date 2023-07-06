import argparse
import os
import apop
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import numpy as np


def run(args):
    if args.record_video:
        env = RecordVideo(
            gym.make(args.env, render_mode="rgb_array"), video_folder=args.video_path
        )
    else:
        env = gym.make(args.env, render_mode="human")

    state, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        action = np.ones(6)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(info)

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Arm2dReachingTargetFixed-v0")
    parser.add_argument("--record_video", action="store_true")
    default_path = os.path.join(os.path.dirname(__file__), "video")
    parser.add_argument("--video_path", type=str, default=default_path)
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
