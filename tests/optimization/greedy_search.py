""" Test cases for explora.utilities.tools module """
from pathlib import Path

import pandas as pd
import pytest
from pytest import approx

import explora
from explora.information_theory.estimators import (
    mutual_information_permutation_upper_bound,
)
from explora.optimization.greedy_search import greedy_search


def test_greedy_search():
    """ Tests choose_no_overflow for simple cases """

    # mock
    testfile = Path(explora.__file__).parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)

    # do
    _, bestScore = greedy_search(mutual_information_permutation_upper_bound, data)
    error = 1e-10
    expected = 0.28707781833145635

    # assert
    assert bestScore == approx(expected, rel=error)


if __name__ == "__main__":
    pytest.main([__file__])
