""" Test cases for explora.information_theory.information_theory_basic module """

from pathlib import Path

import pandas as pd
import pytest
from pytest import approx

import explora
from explora.information_theory.information_theory_basic import (
    mutual_information_plugin, fraction_of_information_plugin
)


def test_mutual_information_plugin():
    """ Tests choose_no_overflow for simple cases """
    # input
    testfile = Path(explora.__file__).parent.parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)

    error = 1e-10

    # do
    res = mutual_information_plugin(data.iloc[:, [0, 2, 4, 6, 8]], data.iloc[:, 9], with_cross_tab=False)

    # assert
    assert res == approx(0.57241179990198, rel=error)


def test_fraction_information_plugin():
    """ Tests choose_no_overflow for simple cases """
    # input
    testfile = Path(explora.__file__).parent.parent / "datasets" / "tic_tac_toe.csv"
    data = pd.read_csv(testfile)

    error = 1e-10

    # do
    res = fraction_of_information_plugin(data.iloc[:, [0, 2, 4, 6, 8]], data.iloc[:, 9], with_cross_tab=False)

    # assert
    assert res == approx(0.6148658347844208, rel=error)


if __name__ == "__main__":
    pytest.main([__file__])
