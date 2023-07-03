from typing import Any, Optional, Dict
from gymnasium.spaces import Space, Box

from apop.envs.environment import ApopMujocoEnv

import os
import numpy as np
import mujoco


class Arm2dReachingTargetFixed(ApopMujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "segmentation_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
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

    def step(self, action: Any):
        # set actions
        self.do_simulation(action, self.frame_skip)

        pass

    def _get_obs(self):
        end_effector_internal_pos = self.data.site().xpos.copy()
        target_ball_internal_pos = self.data.site().xpos.copy()

        # render segmentation
        segmentation_array = self.mujoco_renderer.render(
            "segmentation_array", None, "observer_camera"
        )
        (object2id_map, id2object_map) = segmentation_object_id_map(
            self.model, segmentation_array
        )
        unified_segmentation_arraty = unify_id_of_segmentation_image(
            segmentation_array, object2id_map, id2object_map
        )

        # then if get observation then insert true result else not

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._reset_simulation()

        # sample actor angles

        # sample observer angles

        # sample target positions

        # set states
        # actor
        # self.data.joint().pos =
        # self.data.joint() =
        # self.data.joint() =

        # observer
        # self.data.joint() =
        # self.data.joint() =
        # self.data.joint() =

        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human":
            self.render()


"""
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self._reset_simulation()

        ob = self.reset_model()
        if self.render_mode == "human":
            self.render()
        return ob, {}
"""
