import math
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.envs.classic_control.acrobot import AcrobotEnv, bound, rk4, wrap


class ContinuousAcrobotEnv(AcrobotEnv):
    "Continuous Acrobot env"
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    dt = 0.05  # NOTE: original is 0.2, but it's too large to simulate.

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
        # Overwrite
        # Angle at which to fail the episode
        high = np.array(
            [np.pi, np.pi, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-np.inf]), high=np.array([np.inf]), shape=(1,)
        )
        self.state = None
        self.torque_noise_max = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.1, 0.1  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(
            np.float32
        )

        if self.render_mode == "human":
            self.render()
        return np.array(self.state), {}

    def step(self, action):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        torque = action

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(
                -self.torque_noise_max, self.torque_noise_max
            )

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        # NOTE: Always false
        terminated = False  # self._terminal()

        reward = (
            wrap((ns[0] - np.pi), -np.pi, np.pi) ** 2 * 100
            + ns[1] ** 2 * 100
            + ns[2] ** 2 * 0.01
            + ns[3] ** 2 * 0.01
        )  # -1.0 if not terminated else 0.0

        if self.render_mode == "human":
            self.render()
        return (np.array(self.state, dtype=np.float32), reward, terminated, False, {})
