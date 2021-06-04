""" Test cases for explora.information_theory.permutation_model module """

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import approx

import explora
from explora.information_theory.permutation_model import (
    expected_mutual_information_permutation_model, expected_mutual_information_permutation_model_upper_bound
)


def test_expected_mutual_information_permutation_model_synthetic_input():
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


def test_expected_mutual_information_permutation_model_parallel_synthetic_input():
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


def test_expected_mutual_information_permutation_model_tic_tac_toe():
    """ Tests choose_no_overflow for simple cases """

    testfile = Path(explora.__file__).parent.parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)

    error = 1e-10
    emi = expected_mutual_information_permutation_model(data.iloc[:, 4], data.iloc[:, 9])
    assert emi == approx(0.0015104057711462328, rel=error)
    emi = expected_mutual_information_permutation_model(data.iloc[:, [0, 2, 4, 6, 8]], data.iloc[:, 9])
    assert emi == approx(0.158326256563715, rel=error)


def test_expected_mutual_information_permutation_model_upper_bound_tic_tac_toe():
    """ Tests choose_no_overflow for simple cases """

    testfile = Path(explora.__file__).parent.parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)

    error = 1e-10
    emi = expected_mutual_information_permutation_model_upper_bound(data.iloc[:, 4], data.iloc[:, 9])
    assert emi == approx(0.003011890532105174, rel=error)
    emi = expected_mutual_information_permutation_model_upper_bound(data.iloc[:, [0, 2, 4, 6, 8]], data.iloc[:, 9])
    assert emi == approx(0.2422831283458568, rel=error)


def test_expected_mutual_information_permutation_model_convergence():
    """ Tests choose_no_overflow for simple cases """

    testfile = Path(explora.__file__).parent.parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)
    bigger_data = data.append(data).append(data).append(data).append(data).append(data).append(data)
    bigger_bigger_data = bigger_data.append(bigger_data).append(bigger_data).append(bigger_data).append(
        bigger_data).append(
        bigger_data)
    bigger_bigger_data = bigger_bigger_data.append(bigger_bigger_data).append(bigger_bigger_data).append(
        bigger_bigger_data).append(bigger_bigger_data)

    error = 1e-10
    emi = expected_mutual_information_permutation_model(bigger_bigger_data.iloc[:, 4], bigger_bigger_data.iloc[:, 9])
    assert emi == approx(0, rel=error)


if __name__ == "__main__":
    pytest.main([__file__])
