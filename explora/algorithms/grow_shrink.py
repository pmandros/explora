#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 03:34:55 2020

@author: Panagiotis Mandros
"""

import numpy as np
import pandas as pd
import time

from explora.algorithms.shrink import shrink
from explora.information_theory.estimators import fraction_of_information_permutation
from explora.optimization.greedy_search import greedy_search


def grow_shrink(estimator, data, shrink_threshold=0, target=None, limit=None):
    """
    For a dependency measure D(X;Y), it greedily finds a maximizer for D(X;Y), and shrinks
    afterwards with the conditional. Not to be confused with the Grow Shrink for Markov
    blankets (although similar)
    """

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    if target is None:
        target = np.size(data, 1)

    # start_time = time.time()  

    [greedy_result, greedy_score] = greedy_search(estimator, data, target, limit=limit)
    # print("--- %s Time for grow---" % (time.time() - start_time))

    # start_time = time.time()  
    greedy_result = {x - 1 for x in greedy_result}
    shrink_results = shrink(estimator, greedy_result, data, shrink_threshold=shrink_threshold, target=None)
    # print("--- %s Time to shrink---" % (time.time() - start_time))
    return shrink_results

# def main():
#     data = pd.read_csv("../../datasets/tic_tac_toe.csv")
#     biggerData = data.append(data).append(data).append(data).append(data).append(data).append(data)
#     biggerBiggerData = biggerData.append(biggerData).append(biggerData).append(biggerData).append(biggerData).append(
#         biggerData)
#     biggerBiggerData = biggerBiggerData.append(biggerBiggerData).append(biggerBiggerData).append(
#         biggerBiggerData).append(biggerBiggerData)
#
#     # print(data.info())
#     # print(biggerBiggerData.info())
#
#     # # checking with upper-bound FI
#     # selected=grow_shrink(fraction_of_information_permutation_upper_bound,
#     #                      conditional_fraction_of_information_permutation_upper_bound,
#     #                      data.to_numpy(),shrink_threshold=0)
#     # assert(selected=={5,1,9})
#
#     # selected=grow_shrink(fraction_of_information_permutation_upper_bound,
#     #                      conditional_fraction_of_information_permutation_upper_bound,
#     #                      data.to_numpy(),shrink_threshold=0.21)
#     # assert(selected=={5})
#
#     # # checking with upper-bound FI
#     # selected=grow_shrink(fraction_of_information_permutation,
#     #                      conditional_fraction_of_information_permutation,
#     #                      data.to_numpy(),shrink_threshold=0)
#     # assert(selected=={5,1,9,3,7})
#
#     # selected=grow_shrink(fraction_of_information_permutation,
#     #                      conditional_fraction_of_information_permutation,
#     #                      data.to_numpy(),shrink_threshold=0.2)
#     # assert(selected=={5,3,7})
#
#     # # performance with upper-bound corrected FI
#     # start_time=time.time()
#     # selected=grow_shrink(fraction_of_information_permutation_upper_bound,
#     #                      conditional_fraction_of_information_permutation_upper_bound,
#     #                      data.to_numpy(),shrink_threshold=0)
#     # print("--- %s seconds for grow shrink with upper-dound F1 on small data---" % (time.time() - start_time))
#
#     # start_time=time.time()
#     # selected=grow_shrink(fraction_of_information_permutation_upper_bound,
#     #                      conditional_fraction_of_information_permutation_upper_bound,
#     #                      biggerBiggerData.to_numpy(),shrink_threshold=0)
#     # print("--- %s seconds for grow shrink with upper-dound F1 on big data---" % (time.time() - start_time))
#
#     # performance with permutation corrected FI
#     start_time = time.time()
#     selected = grow_shrink(mutual_information_permutation,
#                            data.to_numpy(), shrink_threshold=0)
#     print("--- %s seconds for grow shrink with permutation F1 on small data---" % (time.time() - start_time))
#
#     # start_time=time.time()
#     # selected=grow_shrink(fraction_of_information_permutation,
#     #                      conditional_fraction_of_information_permutation,
#     #                      biggerBiggerData.to_numpy(),shrink_threshold=0)
#     # print("--- %s seconds for grow shrink with permutation F1 on big data---" % (time.time() - start_time))
#
#
# if __name__ == '__main__':
#     main()
