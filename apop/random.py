import numpy as np

import jax

drng = np.random.default_rng()


def seed(seed: int):
    global drng
    drng = np.random.default_rng(seed)


def new_key(key: jax.random.KeyArray):
    _, subkey = jax.random.split(key)
    return subkey
