""" Test cases for explora.some_statistics.basic_statistics module """

import pytest
from scipy.stats import hypergeom
import numpy as np
from explora.some_statistics.basic_statistics import hypergeometric_pmf


def test_custom_hypergeometric():
    """ Tests choose_no_overflow for simple cases """

    M = 958
    n = 458
    N = 332
    x = 2
    custom_prob = hypergeometric_pmf(2, 958, 458, 332)
    rv = hypergeom(M, n, N)
    pro_prob = rv.pmf(x)
    np.testing.assert_allclose(custom_prob, pro_prob, rtol=1e-5, atol=0)


if __name__ == "__main__":
    pytest.main([__file__])
