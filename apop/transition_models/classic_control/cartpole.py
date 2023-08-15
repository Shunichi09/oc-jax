from functools import partial

import jax
from jax import numpy as jnp

from apop.transition_model import TransitionModel
from apop.transition_models.basic.linear import LinearTransitionModel


class LinearInvertedCartPoleModel(LinearTransitionModel):
    def __init__(
        self,
        gravity: float = 9.8,
        mass_cart: float = 1.0,
        mass_pole: float = 0.1,
        length: float = 0.5,  # actually half the pole's length
    ):
        total_mass = mass_pole + mass_cart
        polemass_length = mass_pole * length

        tau = 0.02  # seconds between state updates
        A = jnp.zeros((4, 4))
        A = A.at[0, 1].set(1.0)
        A = A.at[1, 2].set(-polemass_length / total_mass)
        A = A.at[2, 3].set(1.0)
        denom = length * (4.0 / 3.0 - (mass_pole / total_mass))
        A = A.at[3, 2].set(gravity / denom)
        A = A * tau + jnp.eye(4)

        B = jnp.zeros((4, 1))
        B = B.at[1, 0].set(1.0 / total_mass)
        B = B.at[3, 0].set((-1.0 / total_mass) / denom)
        B = B * tau
        super().__init__(A=A, B=B)


class SwingUpCartPoleModel(TransitionModel):
    def __init__(
        self,
        gravity: float = 9.8,
        mass_cart: float = 1.0,
        mass_pole: float = 0.1,
        length: float = 0.5,  # actually half the pole's length
    ):
        self._gravity = gravity
        self._masscart = mass_cart
        self._masspole = mass_pole
        self._total_mass = self._masspole + self._masscart
        self._length = length
        self._polemass_length = self._masspole * self._length
        self._tau = 0.02  # seconds between state updates

    @partial(jax.jit, static_argnums=(0,))
    def predict_next_state(
        self,
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: jnp.ndarray,
        random_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """predict next state

        Args:
            x (jnp.ndarray): states, shape (state_size, )
            u (jnp.ndarray): input, shape (input_size, )
            t (jnp.ndarray): time step, shape (1, )

        Returns:
            next_x (jnp.ndarray): next state, shape (state_size, )
        """
        # x = cart_x, cart_x_dot, pole_theta, pole_theta_dot

        cos_theta = jnp.cos(x[2])
        sin_theta = jnp.sin(x[2])

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (u[0] + self._polemass_length * x[3] ** 2 * sin_theta) / self._total_mass
        thetaacc = (self._gravity * sin_theta - cos_theta * temp) / (
            self._length
            * (4.0 / 3.0 - self._masspole * cos_theta**2 / self._total_mass)
        )
        xacc = temp - self._polemass_length * thetaacc * cos_theta / self._total_mass

        next_x = jnp.empty_like(x)
        next_x = next_x.at[0].set(x[0] + self._tau * x[1])
        next_x = next_x.at[1].set(x[1] + self._tau * xacc)
        next_x = next_x.at[2].set(x[2] + self._tau * x[3])
        next_x = next_x.at[3].set(x[3] + self._tau * thetaacc)

        return next_x
