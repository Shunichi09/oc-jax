# https://www.jstage.jst.go.jp/article/jrsj/29/5/29_5_427/_pdf
# https://d-nb.info/1027390056/34

from typing import Optional, Union
import jax
import jax.numpy as jnp
from functools import partial
from apop.filter import Filter
from apop.transition_model import TransitionModel
from apop.distribution import Distribution
from apop.random import new_key, np_drng
from apop.observation_model import ObservationModel


class ParticleFilter(Filter):
    _x_particles: Optional[jnp.ndarray]
    _particle_weights: Optional[jnp.ndarray]

    def __init__(
        self,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        initial_distribution: Distribution,
        num_particles: int,
        resampling_threshold: float,
        initial_importance_distribution: Optional[Distribution] = None,
        importance_distribution: Optional[Distribution] = None,
        jax_random_key: jax.random.KeyArray = jax.random.PRNGKey(0),
    ) -> None:
        super().__init__(transition_model, observation_model)
        self._x_particles = None  # shape (num_particles, state_size)
        self._particle_weights = None  # shape (num_particles, )
        self._initial_distribution = initial_distribution
        self._initial_importance_distribution = initial_importance_distribution
        self._num_particles = num_particles
        self._resampling_threshold = resampling_threshold
        self._jax_random_key = jax_random_key

        # If none, then use dynamics distribution as importance distribution
        # NOTE: This could update because it unclear for users
        self._importance_distribution = importance_distribution

        self._initialize()

    def _initialize(self):
        if self._initial_importance_distribution is not None:
            self._x_particles = self._initial_importance_distribution(
                self._num_particles
            )
            x_particle_importance_probs = (
                self._initial_importance_distribution.probability(self._x_particles)
            )
            x_particle_probs = self._initial_distribution.probability(self._x_particles)
            weights = x_particle_importance_probs / x_particle_probs
            self._particle_weights = weights / jnp.sum(weights)
        else:
            self._x_particles = self._initial_distribution.sample(self._num_particles)
            self._particle_weights = jnp.ones((self._num_particles,)) / float(
                self._num_particles
            )

    # @partial(jax.jit, static_argnums=(0,))
    def predict(self, curr_u: jnp.ndarray, t: Union[jnp.ndarray, int]) -> jnp.ndarray:
        self._x_particles = self._transition_model.predict_batched_next_state(
            self._x_particles, jnp.tile(curr_u, (self._num_particles, 1)), t
        )
        return jnp.sum(
            self._particle_weights[:, jnp.newaxis] * self._x_particles, axis=0
        )

    # @partial(jax.jit, static_argnums=(0,))
    def estimate(self, curr_y: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        assert self._x_particles is not None, "Call initialize first"
        assert self._particle_weights is not None, "Call initialize first"

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
            self._x_particles = self._x_particles[inserted_indices]
            self._particle_weights = (
                jnp.ones((self._num_particles,)) / self._num_particles
            )

        batched_estimate_obs_prob_func = jax.vmap(
            self._estimate_obs_prob, in_axes=(0, None, None), out_axes=0
        )
        batched_estimated_probability = batched_estimate_obs_prob_func(
            self._x_particles, curr_y, mask
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

    # @partial(jax.jit, static_argnums=(0,))
    def _estimate_obs_prob(
        self, curr_x: jnp.ndarray, y: jnp.ndarray, observation_mask: jnp.ndarray
    ):
        observation_distribution = self._observation_model.observe_distribution(
            curr_x, observation_mask
        )
        # NOTE: y.shape = (num_sensors, obs_size) -> output.shape = (1, )
        return observation_distribution.probability(y)
