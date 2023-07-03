import numpy as np

drng = np.random.default_rng()


def seed(seed: int):
    global drng
    drng = np.random.default_rng(seed)
