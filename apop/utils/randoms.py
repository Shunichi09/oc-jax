from typing import Any, List, Union

import numpy as np

from apop.random import drng


def rand_min_max(max_val, min_val):
    if isinstance(max_val, np.ndarray) and isinstance(min_val, np.ndarray):
        assert max_val.shape == min_val.shape
        assert np.all(max_val >= min_val)
        return np.array(drng.random(*max_val.shape) * (max_val - min_val) + min_val)
    elif isinstance(max_val, float) and isinstance(min_val, float):
        assert max_val >= min_val
        return float(drng.random() * (max_val - min_val) * min_val)
    else:
        raise ValueError


def random_int(low, high=None, size=None, dtype=np.int64, endpoint=False):
    return drng.integers(low, high, size, dtype, endpoint)


def random_choice(a, size=None, replace=True, p=None, axis=0, shuffle=True):
    return drng.choice(a, size, replace, p, axis, shuffle)
