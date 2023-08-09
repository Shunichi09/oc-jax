# https://www.jstage.jst.go.jp/article/jrsj/29/5/29_5_427/_pdf
# https://d-nb.info/1027390056/34

from typing import Optional
import jax
import jax.numpy as jnp
from functools import partial
from apop.filter import Filter
from apop.transition_model import TransitionModel
from apop.distribution import Distribution
from apop.random import new_key
from apop.observation_model import ObservationModel


class ParticleFilter(Filter):
    _x_particles: Optional[jnp.ndarray]
    _particle_weights: Optional[jnp.ndarray]

    def __init__(
        self,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        initial_importance_distribution: Distribution,
        importance_distribution: Optional[Distribution],
        num_particles: int,
        resampling_threshold: float,
        jax_random_key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ) -> None:
        super().__init__(transition_model)
        self._x_particles = None  # shape (num_particles, state_size)
        self._particle_weights = None  # shape (num_particles, )
        self._observation_model = observation_model
        self._initial_importance_distribution = initial_importance_distribution
        self._num_particles = num_particles
        self._resampling_threshold = resampling_threshold
        self._jax_random_key = jax_random_key

        # If none, then use dynamics distribution as importance distribution
        # NOTE: This could update because it unclear for users
        self._importance_distribution = importance_distribution

    @partial(jax.jit, static_argnums=(0, 1))
    def initialize(self, initial_distribution: Optional[Distribution] = None):
        self._x_particles = self._importance_distribution.sample(self._num_particles)

        if initial_distribution is not None:
            x_particle_importance_probs = self._importance_distribution.probability(
                self._x_particles
            )
            x_particle_probs = initial_distribution.probability(self._x_particles)
            weights = x_particle_importance_probs / x_particle_probs
            self._particle_weights = weights / jnp.sum(weights)
        else:
            self._particle_weights = (
                jnp.ones((self._num_particles,)) / self._num_particles
            )

    @partial(jax.jit, static_argnums=(0,))
    def estimate(
        self,
        curr_y: jnp.ndarray,
        curr_u: jnp.ndarray,
    ) -> jnp.ndarray:
        assert self._x_particles is not None, "Call initialize first"
        assert self._particle_weights is not None, "Call initialize first"

        # predict step
        self._x_particles = self._transition_model.predict_batched_next_state(
            self._x_particles, jnp.tile(curr_u, (self._num_particles, 1))
        )

        # filtering step
        # computes Nff
        Nff = self._num_particles / jnp.sum(self._particle_weights**2)

        # resample
        # https://d-nb.info/1027390056/34 page 8.
        self._jax_random_key = new_key(self._jax_random_key)

        if Nff < self._resampling_threshold:
            jax.debug.print("Resampling step")
            I = jnp.cumsum(self._particle_weights)  # shape (num_particles)
            sample_u = jax.random.uniform(
                self._jax_random_key, shape=(self._num_particles,)
            )
            u_l = (
                jnp.arange(1, self._num_particles + 1, dtype=jnp.float32) - 1 + sample_u
            ) / self._num_particles
            inserted_indices = jnp.searchsorted(I, u_l)
            self._x_particles = self._x_particles.at[inserted_indices].set()
            self._particle_weights = (
                jnp.ones((self._num_particles,)) / self._num_particles
            )

        batched_estimate_obs_prob_func = jax.vmap(
            self._estimate_obs_prob, in_axes=(0, None), out_axes=0
        )
        batched_estimated_probability = batched_estimate_obs_prob_func(
            self._x_particles, curr_y
        ).ravel()

        if self._importance_distribution is None:
            new_particle_weights = (
                self._particle_weights * batched_estimated_probability
            )
            self._particle_weights = new_particle_weights / jnp.sum(
                new_particle_weights
            )
        else:
            raise NotImplementedError
            # dynamics_distribution = self._transition_model.distribution()
            # concat and estimate probs
            # self._importance_distribution

        return jnp.sum(
            self._particle_weights[:, jnp.newaxis] * self._x_particles, axis=0
        )

    def particles(self) -> jnp.ndarray:
        return self._x_particles

    @partial(jax.jit, static_argnums=(0,))
    def _estimate_obs_prob(self, curr_x: jnp.ndarray, y: jnp.ndarray):
        observation_distribution = self._observation_model.observe_distribution(curr_x)
        return observation_distribution.probability(y)
