from typing import Any, Optional, Dict
from gymnasium.spaces import Box

from apop.envs.mujoco.environment import ApopMujocoEnv
from apop.envs.mujoco.environment_utils import (
    segmentation_object_id_map,
)
from apop.utils.randoms import rand_min_max

import os
import numpy as np
import mujoco


class Arm2dReachingTargetFixedEnv(ApopMujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "segmentation_array"],
        "render_fps": 500,
    }

    def __init__(
        self,
        frame_skip: int = 1,
        model_path: str = os.path.join(
            os.path.dirname(__file__), "assets", "arm2d_reaching.xml"
        ),
        render_mode: str = None,
        width: int = 640,
        height: int = 480,
        camera_id: int = None,
        camera_name: str = None,
        default_camera_config: dict = None,
    ):
        observation_space = Box(-np.inf, np.inf, shape=(4,))
        action_space = Box(-np.inf, np.inf, shape=(6,))

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
        self.do_simulation(action, self.frame_skip)

        ob, ob_info = self._get_obs()

        if self.render_mode == "human":
            self.render()

        # comptue reward
        reward = np.sum(
            (ob_info["internal_state"][:2] - ob_info["internal_state"][2:]) ** 2
        )

        return ob, reward, False, False, ob_info

    def _get_obs(self):
        end_effector_internal_pos_x, end_effector_internal_pos_y = self.data.geom(
            "actor_end_effector_geom"
        ).xpos.copy()[:2]
        target_ball_internal_pos_x, target_ball_internal_pos_y = self.data.geom(
            "target_geom"
        ).xpos.copy()[:2]

        # render segmentation
        segmentation_array = self.mujoco_renderer.render(
            "segmentation_array", None, "observer_camera"
        )
        (object2id_map, id2object_map) = segmentation_object_id_map(
            self.model, segmentation_array, remove_suffixes_func=lambda x: x
        )
        # then if get observation then insert true result else not
        if "actor_end_effector_geom" in object2id_map.keys():
            end_effector_pos_x = end_effector_internal_pos_x
            end_effector_pos_y = end_effector_internal_pos_y
        else:
            end_effector_pos_x = None
            end_effector_pos_y = None

        if "target_geom" in object2id_map.keys():
            target_ball_pos_x = target_ball_internal_pos_x
            target_ball_pos_y = target_ball_internal_pos_y
        else:
            target_ball_pos_x = None
            target_ball_pos_y = None

        return np.array(
            [
                end_effector_pos_x,
                end_effector_pos_y,
                target_ball_pos_x,
                target_ball_pos_y,
            ]
        ), {
            "internal_state": np.array(
                [
                    end_effector_internal_pos_x,
                    end_effector_internal_pos_y,
                    target_ball_internal_pos_x,
                    target_ball_internal_pos_y,
                ]
            ),
            "end_effector_observed": False if end_effector_pos_x is None else True,
            "target_observed": False if target_ball_pos_x is None else True,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._reset_simulation()

        # sample actor angles
        actor_angels = rand_min_max(np.ones(3), -np.ones(3))
        # actor_angels = np.zeros(3)

        # sample observer angles
        observer_angels = rand_min_max(np.ones(3), -np.ones(3))
        # observer_angels = np.zeros(3)

        # sample target positions
        target_position = rand_min_max(np.ones(2) * 0.01, -np.ones(2) * 0.01)

        # set states
        # actor
        self.data.joint("actor_joint1").qpos = actor_angels[0]
        self.data.joint("actor_joint2").qpos = actor_angels[1]
        self.data.joint("actor_joint3").qpos = actor_angels[2]
        # observer
        self.data.joint("observer_joint1").qpos = observer_angels[0]
        self.data.joint("observer_joint2").qpos = observer_angels[1]
        self.data.joint("observer_joint3").qpos = observer_angels[2]

        self._position_ctrl_input = np.concatenate([actor_angels, observer_angels])
        self.data.ctrl[:] = self._position_ctrl_input

        mujoco.mj_forward(self.model, self.data)

        ob, ob_info = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return ob, ob_info
