""" Test cases for markov_blankets.utilities.tools module """

import pytest

from markov_blankets.utilities.tools import choose_no_overflow


def test_choose_no_overflow():
    """ Tests choose_no_overflow for simple cases """

    cases = {
        (6, 3): 20,
        (1, 1): 1,
        (10, 1): 10,
        (10, 5): 252,
        (20, 2): 190,
        (50, 7): 99_884_400,
    }
    for given, expected in cases.items():
        assert expected == choose_no_overflow(*given)


if __name__ == "__main__":
    pytest.main([__file__])
