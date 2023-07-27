from jax import numpy as jnp

from apop.transition_model import LinearTransitionModel


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
