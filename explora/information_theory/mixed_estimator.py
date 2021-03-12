#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 03:02:15 2020

@author: Panagiotis Mandros
"""
import math
from operator import itemgetter

import numpy as np
import pandas as pd

from explora.utilities.tools import append_two_arrays, number_of_columns, get_column


def mixed_estimator(C, Y, estimator, G=None, max_number_partitions=None, sorting=False):
    """
    THe mixed estimator for mutual information of Mandros et al. KDD'2020. Discretizes with
    equal-frequency. Estimator is the mutual information estimator of choice, G is the set of already
    discrete variables. Will perfom an initial sorting on marginal mutual informations in two bins
    if sorting=True"""
    if isinstance(C, pd.Series) or isinstance(C, pd.DataFrame):
        C = C.to_numpy()

    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()

    number_of_attributes_in_C = number_of_columns(C)
    num_samples = len(C)

    if max_number_partitions is None:
        max_number_partitions = math.ceil(math.log10(len(C)))

    # sorting in decreasing marginal mutual information
    if sorting and number_of_attributes_in_C > 1:
        generator = sort_generator(
            estimator=estimator,
            G=G,
            Y=Y,
            X=C,
        )
        sorted_attributes = sorted(generator, key=itemgetter(0))
        sorted_column_indices = [row[1] for row in sorted_attributes]
        C = C[:, sorted_column_indices]

    discrete_C = None
    best_score = float("-inf")
    for i in range(number_of_attributes_in_C):
        generator = max_generator(
            estimator=estimator,
            G=G,
            Y=Y,
            X=get_column(C, i),
            max_number_partitions=max_number_partitions
        )

        top_score, top_discretized_attribute = max(generator, key=itemgetter(0))

        if top_score > best_score:
            best_score = top_score
            discrete_C = append_two_arrays(discrete_C, top_discretized_attribute)
        else:
            discrete_C = append_two_arrays(discrete_C, np.zeros((num_samples, 1)))

    return top_score, discrete_C


def max_generator(estimator, G, Y, X, max_number_partitions):
    for num_bins in range(2, max_number_partitions + 1):
        discrete_candidate = pd.qcut(X, num_bins, labels=False, duplicates='drop')
        joint_columns = append_two_arrays(discrete_candidate, G)
        result = estimator(joint_columns, Y)
        yield result, discrete_candidate


def sort_generator(estimator, G, Y, X):
    for i in range(number_of_columns(X)):
        discrete_candidate = pd.qcut(get_column(X, i), 2, labels=False, duplicates='drop')
        joint_columns = append_two_arrays(discrete_candidate, G)
        result = estimator(joint_columns, Y)
        yield result, i

# def main():
#     # test performance
#     X = np.random.uniform(size=(100000,))
#     Y = np.random.randint(15, size=(100000,))
#
#     start_time = time.time()
#     res = mixed_estimator(X, Y, fraction_of_information_permutation, max_number_partitions=15)
#     print("--- %s seconds ---" % (time.time() - start_time))
#     # # num_rep = 1
#     # # single = partial(mixed_estimator, X, Y,fraction_of_information_permutation,                           max_number_partitions=10)
#     # # print(timeit(single, number=num_rep) / num_rep, " seconds")
#
#
# if __name__ == "__main__":
#     main()
