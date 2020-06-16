#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:41:21 2020

@author: Panagiotis Mandros
"""

import math
from functools import partial
from itertools import product
from multiprocessing import Pool
from timeit import timeit

import numba as nb
import numpy as np
import pandas as pd


@nb.jit(nopython=True)
def choose(n, r):
    """
    Computes n! / (r! (n-r)!) exactly. Returns a python int. For some reason it doesnt overflow
    """
    assert 0 <= r <= n

    c = 1
    for num, denom in zip(range(n, n - r, -1), range(1, r + 1, 1)):
        c = (c * num) // denom
    return c


@nb.jit(nopython=True)
def hypergeometric_pmf(k, n, a, b):
    return choose(a, k) * choose(n - a, b - k) / choose(n, b)


def expected_mutual_information_permutation_model(
    X, Y, contingency_table=None, num_threads=1
):
    if contingency_table is None:
        X = concatenate_attributes(X)
        Y = concatenate_attributes(Y)

        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            X_marginal_counts = X.value_counts()
            Y_marginal_counts = Y.value_counts()

            num_samples = len(X.index)
        elif isinstance(X, np.ndarray):
            X_marginal_counts = np.unique(X, return_counts=True, axis=0)[1]
            Y_marginal_counts = np.unique(Y, return_counts=True, axis=0)[1]

            num_samples = np.size(X, 0)
    else:
        contingency_table = contingency_table.to_numpy()
        num_samples = contingency_table[-1, -1]
        Y_marginal_counts = contingency_table[-1, :-1]
        X_marginal_counts = contingency_table[:-1, -1]

    if Y_marginal_counts[0] == num_samples or X_marginal_counts[0] == num_samples:
        return 0

    marginal_counts_cartesian_product = product(X_marginal_counts, Y_marginal_counts)
    if True or num_threads == 1:
        expected_mutual_info = 0
        for cell_marginals in marginal_counts_cartesian_product:
            expected_mutual_info += cell_total_contribution_custom(
                cell_marginals, num_samples
            )
        return expected_mutual_info
    else:
        partial_app = partial(cell_total_contribution_custom, num_samples=num_samples)
        with Pool(num_threads) as processor:
            return sum(processor.map(partial_app, marginal_counts_cartesian_product))


@nb.jit(nopython=True)
def cell_total_contribution_custom(cell_marginals, num_samples):
    """ Calculates cell_contribution for a tuple of marginal counts without external libraries"""
    marginal_count_one, marginal_count_two = cell_marginals

    min_iterator = max(1, marginal_count_one + marginal_count_two - num_samples)
    max_iterator = min(marginal_count_one, marginal_count_two)

    prob = hypergeometric_pmf(
        min_iterator, num_samples, marginal_count_one, marginal_count_two
    )
    cell_contribution = prob * mutual_information_plugin_cell_contribution(
        min_iterator, num_samples, marginal_count_one, marginal_count_two
    )
    possible_values = np.arange(min_iterator + 1, max_iterator + 1)

    for value in possible_values:
        temp_one = prob * (
            (marginal_count_one - (value - 1)) * (marginal_count_two - (value - 1))
        )
        temp_two = value * (
            num_samples - marginal_count_one - marginal_count_two + value
        )
        prob = temp_one / temp_two
        cell_contribution += prob * mutual_information_plugin_cell_contribution(
            value, num_samples, marginal_count_one, marginal_count_two
        )

    return cell_contribution


@nb.jit(nopython=True)
def mutual_information_plugin_cell_contribution(
    cell_count, num_samples, marginal_count_one, marginal_count_two
):
    if cell_count == 0:
        return 0
    else:
        first_part = cell_count / num_samples
        second_part = math.log(
            num_samples * cell_count / (marginal_count_one * marginal_count_two)
        ) / math.log(2)
        return first_part * second_part


def concatenate_attributes(X):
    """ Combines multiple columns into one. The values of the single column are
    the distinct JOINT values from the multiple columns"""

    if isinstance(X, pd.DataFrame):
        num_columns = X.shape[1]
        if num_columns > 1:
            return X[X.columns].astype("str").agg("-".join, axis=1)
        else:
            return X
    elif isinstance(X, np.ndarray):
        num_columns = X.ndim
        if num_columns > 1:
            return np.unique(X, return_inverse=True, axis=0)[1]
        else:
            return X
    elif isinstance(X, pd.Series):
        return X


if __name__ == "__main__":
    X = np.array([1, 1, 2, 2, 3, 3])
    Y = np.array([1, 1, 2, 2, 3, 3])

    assert (
        expected_mutual_information_permutation_model(Y, X)
        == expected_mutual_information_permutation_model(Y, X, num_threads=4)
        == 0.7849625007211563
    )

    X = np.random.randint(100, size=(10000, 1))
    Y = np.random.randint(100, size=(10000, 1))

    num_rep = 5

    single = partial(expected_mutual_information_permutation_model, X, Y)
    print(timeit(single, number=num_rep) / num_rep)

    parallel = partial(
        expected_mutual_information_permutation_model, X, Y, num_threads=4
    )
    print(timeit(parallel, number=num_rep) / num_rep)

    assert single() == parallel()
