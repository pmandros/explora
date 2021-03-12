""" Test cases for explora.information_theory.mixed_estimator module """

import numpy as np
import pytest
from pytest import approx

from explora.information_theory.estimators import (
    fraction_of_information_permutation)
from explora.information_theory.mixed_estimator import (
    mixed_estimator
)
from explora.utilities.tools import append_two_arrays


def test_mixed_fraction_information_permutation():
    """ Tests choose_no_overflow for simple cases """
    # input
    X = np.arange(8)
    Y = np.array([1, 1, 2, 2, 3, 3, 4, 4])

    error = 1e-10

    # do
    res = mixed_estimator(X, Y, fraction_of_information_permutation, max_number_partitions=4)

    # assert
    assert res[0] == approx(0.4285714285714287, rel=error)
    assert (np.array_equal(res[1], [0, 0, 1, 1, 2, 2, 3, 3]))


def test_mixed_fraction_information_permutation_another():
    """ Tests choose_no_overflow for simple cases """
    # input
    X = np.arange(8)
    Y = np.array([1, 1, 2, 2, 3, 3, 4, 4])

    error = 1e-10

    # do
    res = mixed_estimator(append_two_arrays(X, X), Y, fraction_of_information_permutation, max_number_partitions=4)

    # assert
    assert res[0] == approx(0.4285714285714287, rel=error)
    assert (np.array_equal(res[1][:, 0], [0, 0, 1, 1, 2, 2, 3, 3]))
    assert (np.array_equal(res[1][:, 1], [0, 0, 0, 0, 0, 0, 0, 0]))


if __name__ == "__main__":
    pytest.main([__file__])
