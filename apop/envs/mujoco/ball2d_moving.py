import os
from typing import Any, Dict, Optional

import mujoco
import numpy as np
from gymnasium.spaces import Box

from apop.envs.mujoco.environment import ApopMujocoEnv
from apop.random import np_drng
from apop.utils.maths import fit_angle_in_range


class Ball2dTrackingEnv(ApopMujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "segmentation_array"],
        "render_fps": 250,
    }

    def __init__(
        self,
        forward_env_noise_scale: float = 2.0,
        angle_env_noise_scale: float = 0.5,
        distance_obs_noise_scale: float = 0.2,
        direction_obs_noise_scale: float = 0.3,
        frame_skip: int = 2,
        model_path: str = os.path.join(
            os.path.dirname(__file__), "assets", "ball2d_tracking.xml"
        ),
        render_mode: str = None,
        width: int = 640,
        height: int = 480,
        camera_id: int = None,
        camera_name: str = None,
        default_camera_config: dict = None,
    ):
        # 3 landmarks, (distance and angles)
        observation_space = Box(-np.inf, np.inf, shape=(6,))
        # NOTE: action, forward and angular
        action_space = Box(-np.inf, np.inf, shape=(2,))
        self._angle_env_noise_scale = angle_env_noise_scale
        self._forward_env_noise_scale = forward_env_noise_scale
        self._direction_obs_noise_scale = direction_obs_noise_scale
        self._distance_obs_noise_scale = distance_obs_noise_scale

        super().__init__(
            model_path,
            frame_skip,
            observation_space,
            action_space,
            render_mode,
            width,
            height,
            camera_id,
            camera_name,
            default_camera_config,
        )
        self._position_ctrl_input = None

    def step(self, action: Any):
        action[0] += np_drng.normal() * self._forward_env_noise_scale
        action[1] += np_drng.normal() * self._angle_env_noise_scale

        # get joints
        curr_x = self.data.joint("ball_joint1").qpos[0]
        curr_y = self.data.joint("ball_joint2").qpos[0]
        curr_theta = self.data.joint("ball_joint3").qpos[0]

        # set joints
        x = action[0] * self.dt * np.cos(curr_theta) + curr_x
        y = action[0] * self.dt * np.sin(curr_theta) + curr_y
        theta = action[1] * self.dt + curr_theta

        # set joints
        self.data.joint("ball_joint1").qpos[:] = x
        self.data.joint("ball_joint2").qpos[:] = y
        self.data.joint("ball_joint3").qpos[:] = theta

        # NOTE: Action is empty (No actuators.)
        self.do_simulation([], self.frame_skip, shape_test=False)

        ob, ob_info = self._get_obs()

        if self.render_mode == "human":
            self.render()

        # No rewards
        return ob, None, False, False, ob_info

    def _get_obs(self, apply_noise=True):
        x = self.data.joint("ball_joint1").qpos[0]
        y = self.data.joint("ball_joint2").qpos[0]
        theta = self.data.joint("ball_joint3").qpos[0]
        internal_state = np.array([x, y, theta], dtype=np.float32)

        # computes landmarks
        landmark_1 = self.data.body("landmark1_link").xpos  # r
        landmark_2 = self.data.body("landmark2_link").xpos  # g
        landmark_3 = self.data.body("landmark3_link").xpos  # b

        # TODO: implement occulusions
        distances = np.zeros(3)
        directions = np.zeros(3)
        for i, landmark in enumerate([landmark_1, landmark_2, landmark_3]):
            distances[i] = np.hypot(
                landmark[0] - internal_state[0], landmark[1] - internal_state[1]
            )
            directions[i] = (
                np.arctan2(
                    landmark[1] - internal_state[1], landmark[0] - internal_state[0]
                )
                - internal_state[2]
            )

        # noise
        if apply_noise:
            distances = (
                distances + np_drng.normal(size=3) * self._distance_obs_noise_scale
            )
            directions = (
                directions + np_drng.normal(size=3) * self._direction_obs_noise_scale
            )

        return np.stack([distances, directions], axis=1), {
            "internal_state": np.array(internal_state),
            "landmark_positions": np.array(
                [landmark_1[:2], landmark_2[:2], landmark_3[:2]]
            ),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._reset_simulation()
        # set joints
        self.data.joint("ball_joint1").qpos[:] = -0.3 + np_drng.normal() * 0.01
        self.data.joint("ball_joint2").qpos[:] = -0.5 + np_drng.normal() * 0.01
        self.data.joint("ball_joint3").qpos[:] = 0.0 + np_drng.normal() * 0.01
        mujoco.mj_forward(self.model, self.data)

        ob, ob_info = self._get_obs(apply_noise=False)
        if self.render_mode == "human":
            self.render()
        return ob, ob_info
