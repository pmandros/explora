""" Test cases for explora.optimization.greedy_search module """
from pathlib import Path

import pandas as pd
import pytest
from pytest import approx

import explora
from explora.information_theory.estimators import (
    mutual_information_permutation_upper_bound, fraction_of_information_permutation,
    conditional_mutual_information
)
from explora.optimization.greedy_search import greedy_search


def test_greedy_search_with_upper_bound_mi():
    """ Tests choose_no_overflow for simple cases """

    # mock
    testfile = Path(explora.__file__).parent.parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)

    # do
    selected, best_score = greedy_search(mutual_information_permutation_upper_bound, data)
    error = 1e-10
    expected = 0.28707781833145635

    # assert
    assert best_score == approx(expected, rel=error)
    assert (selected == {1, 5, 9})


def test_greedy_search_with_permutation_fi():
    """ Tests choose_no_overflow for simple cases """

    # mock
    testfile = Path(explora.__file__).parent.parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)

    # do
    selected, best_score = greedy_search(fraction_of_information_permutation, data)
    error = 1e-10
    expected = 0.4447970033469652

    # assert
    assert best_score == approx(expected, rel=error)
    assert (selected == {1, 5, 9, 3, 7})


def test_greedy_search_with_conditional_permutation_fi():
    """ Tests choose_no_overflow for simple cases """

    # mock
    testfile = Path(explora.__file__).parent.parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)

    control_var_set = {5}

    # do
    selected, best_score = greedy_search(conditional_mutual_information, data,
                                         control_variables=control_var_set)
    error = 1e-10
    expected = 0.3561447636856704

    # assert
    assert best_score == approx(expected, rel=error)
    assert (selected == {1, 9, 3, 7})


if __name__ == "__main__":
    pytest.main([__file__])
