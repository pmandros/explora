""" Test cases for explora.information_theory.estimators module """

from pathlib import Path

import pandas as pd
import pytest
from pytest import approx

import explora
from explora.information_theory.estimators import (
    mutual_information_permutation_upper_bound, mutual_information_permutation,
    conditional_fraction_of_information_permutation, fraction_of_information_permutation
)


def test_mutual_information_permutation_upper_bound():
    """ Tests choose_no_overflow for simple cases """
    # input
    testfile = Path(explora.__file__).parent.parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)

    error = 1e-10

    # do
    res_cross_tab = mutual_information_permutation_upper_bound(data.iloc[:, [0, 2, 4, 6, 8]], data.iloc[:, 9],
                                                               with_cross_tab=True)
    res_numpy = mutual_information_permutation_upper_bound(data.iloc[:, [0, 2, 4, 6, 8]], data.iloc[:, 9],
                                                           with_cross_tab=False)

    # assert
    assert res_cross_tab == res_numpy == approx(0.3301286715561232, rel=error)


def test_mutual_information_permutation():
    """ Tests choose_no_overflow for simple cases """
    # input
    testfile = Path(explora.__file__).parent.parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)
    # do
    res_cross_tab = mutual_information_permutation(data.iloc[:, [0, 2, 4, 6, 8]], data.iloc[:, 9],
                                                   with_cross_tab=True)
    res_numpy = mutual_information_permutation(data.iloc[:, [0, 2, 4, 6, 8]], data.iloc[:, 9],
                                               with_cross_tab=False)

    # assert
    assert res_cross_tab == res_numpy


def test_fraction_information_permutation():
    """ Tests choose_no_overflow for simple cases """
    # input
    testfile = Path(explora.__file__).parent.parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)

    error = 1e-10

    # do
    res = fraction_of_information_permutation(data.iloc[:, [0, 2, 4, 6, 8]], data.iloc[:, 9],
                                              with_cross_tab=True)

    # assert
    assert res == approx(0.4447970033469641, rel=error)


def test_conditional_fraction_information_permutation():
    """ Tests choose_no_overflow for simple cases """
    # input
    testfile = Path(explora.__file__).parent.parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)

    error = 1e-10

    # do
    res = conditional_fraction_of_information_permutation(data.iloc[:, [0, 2, 6, 8]], data.iloc[:, 9],
                                                          data.iloc[:, 4])

    # assert
    assert res == approx(0.35614476368567044, rel=error)


if __name__ == "__main__":
    pytest.main([__file__])
