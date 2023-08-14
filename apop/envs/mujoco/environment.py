import os
from typing import Optional

import gymnasium
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer, WindowViewer
from gymnasium.spaces import Space


class ApopMujocoEnv(gymnasium.Env):
    """ """

    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        observation_space: Space,
        action_space: Space,
        render_mode: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        default_camera_config: Optional[dict] = None,
    ):
        """
        Args:
            model_path: str
            frame_skip: int
            observation_space: Space,
            render_mode: Optional[str] = None: this is only fo
            width: int = 640,
            height: int = 480,
            camera_id: Optional[int] = None,
            camera_name: Optional[str] = None,
            default_camera_config: Optional[dict] = None,
        """
        self.width = width
        self.height = height
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise OSError(f"File {self.model_path} does not exist")

        self._initialize_simulation()  # may use width and height

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.frame_skip = frame_skip

        self.observation_space = observation_space
        self.action_space = action_space

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

        self.mujoco_renderer = ApopMujocoRenderer(
            self.model, self.data, default_camera_config
        )

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
            "segmentation_array",
        ], self.metadata["render_modes"]
        assert (
            int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
        ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

    def _initialize_simulation(self):
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        return self.mujoco_renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        )

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames, shape_test=True):
        """
        Step the simulation n number of frames and applying a control action.
        """
        if shape_test:
            # Check control input is contained in the action space
            if np.array(ctrl).shape != self.action_space.shape:
                raise ValueError(
                    f"Action dimension mismatch. Expected {self.action_space.shape}, found {np.array(ctrl).shape}"
                )
        self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)


class ApopMujocoRenderer:
    """This is the MuJoCo renderer manager class for every MuJoCo environment.
    Almost same as the original code, but support segmentaiton.
    """

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        default_cam_config: Optional[dict] = None,
    ):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            model: MjModel data structure of the MuJoCo simulation
            data: MjData data structure of the MuJoCo simulation
            default_cam_config: dictionary with attribute values of the viewer's default camera, https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=camera#visual-global
        """
        self.model = model
        self.data = data
        self._viewers = {}
        self.viewer = None
        self.default_cam_config = default_cam_config

    def render(
        self,
        render_mode: str,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        """Renders a frame of the simulation in a specific format and camera view.

        Args:
            render_mode: The format to render the frame, it can be: "human", "rgb_array", or "depth_array"
            camera_id: The integer camera id from which to render the frame in the MuJoCo simulation
            camera_name: The string name of the camera from which to render the frame in the MuJoCo simulation. This argument should not be passed if using cameara_id instead and vice versa
        Returns:
            If render_mode is "rgb_array" or "depth_arra" it returns a numpy array in the specified format. "human" render mode does not return anything.
        """

        viewer = self._get_viewer(render_mode=render_mode)

        if render_mode in {"rgb_array", "depth_array", "segmentation_array"}:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

            if render_mode == "segmentation_array":
                segmentation = True
            else:
                segmentation = False

            img = viewer.render(
                render_mode=render_mode, camera_id=camera_id, segmentation=segmentation
            )
            return img

        elif render_mode == "human":
            return viewer.render()

    def _get_viewer(self, render_mode: str):
        """Initializes and returns a viewer class depending on the render_mode
        - `WindowViewer` class for "human" render mode
        - `OffScreenViewer` class for "rgb_array" or "depth_array" render mode
        """
        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None:
            if render_mode == "human":
                self.viewer = WindowViewer(self.model, self.data)

            elif render_mode in {"rgb_array", "depth_array", "segmentation_array"}:
                self.viewer = OffScreenViewer(self.model, self.data)
            else:
                raise AttributeError(
                    f"Unexpected mode: {render_mode}, expected modes: human, rgb_array, depth_array, segmentation_array"
                )
            # Add default camera parameters
            self._set_cam_config()
            self._viewers[render_mode] = self.viewer

        if len(self._viewers.keys()) > 1:
            # Only one context can be current at a time
            self.viewer.make_context_current()

        return self.viewer

    def _set_cam_config(self):
        """Set the default camera parameters"""
        assert self.viewer is not None
        if self.default_cam_config is not None:
            for key, value in self.default_cam_config.items():
                if isinstance(value, np.ndarray):
                    getattr(self.viewer.cam, key)[:] = value
                else:
                    setattr(self.viewer.cam, key, value)

    def close(self):
        """Close the OpenGL rendering contexts of all viewer modes"""
        for _, viewer in self._viewers.items():
            viewer.close()
