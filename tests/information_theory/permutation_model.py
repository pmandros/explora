""" Test cases for markov_blankets.information_theory.permutation_model module """

import pytest
import numpy as np
from pytest import approx

from markov_blankets.information_theory.permutation_model import (
    expected_mutual_information_permutation_model,
)


def test_expected_mutual_information_permutation_model():
    """ Tests choose_no_overflow for simple cases """
    # input
    X = np.array([1, 1, 2, 2, 3, 3])
    Y = np.array([1, 1, 2, 2, 3, 3])
    expected = 0.7849625007211559
    error = 1e-10

    # do
    res = expected_mutual_information_permutation_model(Y, X)

    # assert
    assert res == approx(expected, rel=error)


def test_expected_mutual_information_permutation_model_parallel():
    """ Tests choose_no_overflow for simple cases """

    # input
    X = np.array([1, 1, 2, 2, 3, 3])
    Y = np.array([1, 1, 2, 2, 3, 3])
    expected = 0.7849625007211559
    error = 1e-10

    # do
    res = expected_mutual_information_permutation_model(Y, X, num_threads=4)

    # assert
    assert res == approx(expected, rel=error)


if __name__ == "__main__":
    pytest.main([__file__])
