import jax
import numpy as np

np_drng = np.random.default_rng()


def seed(seed: int):
    global np_drng
    np_drng = np.random.default_rng(seed)


def new_key(key: jax.random.KeyArray):
    _, subkey = jax.random.split(key)
    return subkey
