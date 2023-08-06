import pytest
from jax import numpy as jnp
import numpy as np

import jax
from apop.utils.jax_functions import fit_angle_in_range


class TestJaxFunctions:
    def test_fit_angle_in_range_numeric_assertion_upper(self):
        angle = jnp.array([3.56])
        actual = fit_angle_in_range(angle[0]).block_until_ready()
        assert np.allclose(actual, angle - 2.0 * np.pi)

    def test_fit_angle_in_range_numeric_assertion_lower(self):
        angle = jnp.array([-3.56])
        actual = fit_angle_in_range(angle[0]).block_until_ready()
        assert np.allclose(actual, angle + 2.0 * np.pi)


if __name__ == "__main__":
    pytest.main()
