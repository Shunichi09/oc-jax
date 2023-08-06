from typing import Optional

import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize


class AnglePendulumEnv(PendulumEnv):
    """Pendulum environment where the state is angle and angle speed."""

    def __init__(self, render_mode: Optional[str] = None, g=10):
        super().__init__(render_mode=render_mode, g=g)

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([angle_normalize(theta), thetadot], dtype=np.float32)
