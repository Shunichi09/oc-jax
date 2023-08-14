from functools import partial

import jax
import numpy as np
from jax import numpy as jnp

from apop.transition_model import DeterministicTransitionModel
from apop.utils.jax_functions import fit_angle_in_range


class AcrobotModel(DeterministicTransitionModel):
    def __init__(
        self,
        link_mass_1: float = 1.0,
        link_mass_2: float = 1.0,
        link_length_1: float = 1.0,
        link_length_2: float = 1.0,
        link_center_of_mass_1: float = 0.5,
        link_center_of_mass_2: float = 0.5,
        g: float = 9.8,
        dt: float = 0.05,
        max_link1_dtheta: float = 4.0 * np.pi,
        max_link2_dtheta: float = 9.0 * np.pi,
    ):
        self._m1 = link_mass_1
        self._m2 = link_mass_2
        self._l1 = link_length_1
        self._lc1 = link_center_of_mass_1
        self._lc2 = link_center_of_mass_2
        self._I1 = link_length_1
        self._I2 = link_length_2
        self._g = g
        self._dt = dt  # seconds between state updates
        self._max_link1_dtheta = max_link1_dtheta
        self._max_link2_dtheta = max_link2_dtheta

    @partial(jax.jit, static_argnums=(0,))
    def predict_next_state(
        self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """predict next state

        Args:
            x (jnp.ndarray): states, shape (state_size, )
            u (jnp.ndarray): input, shape (input_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            next_x (jnp.ndarray): next state, shape (state_size, )
        """
        # x = theta1, theta2, theta1dot, theta2dot
        theta1 = x[0]
        theta2 = x[1]
        dtheta1 = x[2]
        dtheta2 = x[3]

        d1 = (
            self._m1 * self._lc1**2
            + self._m2
            * (
                self._l1**2
                + self._lc2**2
                + 2 * self._l1 * self._lc2 * jnp.cos(theta2)
            )
            + self._I1
            + self._I2
        )
        d2 = (
            self._m2 * (self._lc2**2 + self._l1 * self._lc2 * jnp.cos(theta2))
            + self._I2
        )
        phi2 = self._m2 * self._lc2 * self._g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
        phi1 = (
            -self._m2 * self._l1 * self._lc2 * dtheta2**2 * jnp.sin(theta2)
            - 2 * self._m2 * self._l1 * self._lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
            + (self._m1 * self._lc1 + self._m2 * self._l1)
            * self._g
            * jnp.cos(theta1 - jnp.pi / 2)
            + phi2
        )
        # use book version
        # See: https://github.com/Farama-Foundation/Gymnasium/blob/933d481189322de988214455dd74c868e88cf7a5/gymnasium/envs/classic_control/acrobot.py#L282
        ddtheta2 = (
            u[0]
            + d2 / d1 * phi1
            - self._m2 * self._l1 * self._lc2 * dtheta1**2 * jnp.sin(theta2)
            - phi2
        ) / (self._m2 * self._lc2**2 + self._I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        next_theta1 = fit_angle_in_range(x[0] + self._dt * dtheta1)
        next_theta2 = fit_angle_in_range(x[1] + self._dt * dtheta2)
        next_dtheta1 = jnp.clip(
            x[2] + self._dt * ddtheta1,
            a_min=-self._max_link1_dtheta,
            a_max=self._max_link1_dtheta,
        )
        next_dtheta2 = jnp.clip(
            x[3] + self._dt * ddtheta2,
            a_min=-self._max_link2_dtheta,
            a_max=self._max_link2_dtheta,
        )

        next_x = jnp.empty_like(x)
        next_x = next_x.at[0].set(next_theta1)
        next_x = next_x.at[1].set(next_theta2)
        next_x = next_x.at[2].set(next_dtheta1)
        next_x = next_x.at[3].set(next_dtheta2)
        return next_x
