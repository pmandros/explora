#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:45:52 2020

@author: Panagiotis Mandros
"""

import time
from operator import itemgetter

import numpy as np
import pandas as pd

from information_theory.estimators import mutual_information_permutation_upper_bound
from information_theory.information_theory_basic import mutual_information_plugin


def greedy_search(estimator, data, target=None, limit=None, prior_solution_set=None):
    """
    Given data, it greedily maximizes an estimator of a dependency measure D(XY), over
    candidate attribute sets in X in the data. It selects the best candidate to expand 
    in a BFS manner. If no target index is provided (indexed from 1), the target is the last
    attribute. If a limit is provided, the search will stop at at level, e.g., for 2, the search
    wont continue after the second level (pairs of attributes). A prior_solution_set can be used to
    initialize the search.    
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    number_explanatory_variables = np.size(data, 1) - 1

    if target is None:
        target_index = number_explanatory_variables
    else:
        target_index = target - 1
    Y = data[:, target_index]

    if limit is None:
        limit = number_explanatory_variables

    set_of_candidates = set(range(number_explanatory_variables + 1))
    set_of_candidates.remove(target_index)

    if prior_solution_set:
        set_of_candidates.difference_update(prior_solution_set)

    selected_variables = set()
    best_score = -1000

    for i in range(limit):

        generator = sort_generator(
            estimator=estimator,
            data=data,
            Y=Y,
            selected=selected_variables,
            candidates=set_of_candidates,
        )

        top_score, top_candidate_index = max(generator, key=itemgetter(0))

        if top_score <= 0 or top_score <= best_score:
            return selected_variables, best_score

        if top_score > best_score:
            best_score = top_score
            selected_variables.add(top_candidate_index)
            set_of_candidates.remove(top_candidate_index)
    return selected_variables, best_score


def sort_generator(estimator, data, Y, selected, candidates):
    for candidate_idx in candidates:
        result = estimator(data[:, (*selected, candidate_idx)], Y)
        yield result, candidate_idx


def main():
    data = pd.read_csv("../datasets/tic_tac_toe.csv")
    biggerData=data.append(data).append(data).append(data).append(data).append(data).append(data)
    biggerBiggerData=biggerData.append(biggerData).append(biggerData).append(biggerData).append(biggerData).append(biggerData)
    biggerBiggerData=biggerBiggerData.append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData)

    # print(data.info())
    # print(biggerBiggerData.info())

    start_time=time.time()
    [selected,bestScore]=greedy_search(mutual_information_plugin,data)
    print("--- %s seconds for greedy search in small data with plugin MI ---" % (time.time() - start_time))
    print(f' selected by plugin MI on small: {selected} with score {bestScore}')

    start_time=time.time()
    [selected,bestScore]=greedy_search(mutual_information_permutation_upper_bound,data)
    print("--- %s seconds for greedy search in small data with upper-bound MI ---" % (time.time() - start_time))
    print(f' selected by upper-bound MI on small: {selected} with score {bestScore}')
    assert(0.28707781833145635==bestScore)

    start_time=time.time()
    [selected,bestScore]=greedy_search(mutual_information_plugin,data)
    print("--- %s seconds for greedy search in small data with plugin FI ---" % (time.time() - start_time))
    print(f' selected by plugin FI on small: {selected} with score {bestScore}')

    start_time=time.time()
    [selected,bestScore]=greedy_search(mutual_information_permutation_upper_bound,data)
    print("--- %s seconds for greedy search in small data with upper-bound FI ---" % (time.time() - start_time))
    assert(0.3083695032958582==bestScore)
    print(f' selected by upper-bound FI on small: {selected} with score {bestScore}')

    # start_time=time.time()
    # [selected,bestScore]=greedy_search(mutual_information_plugin,biggerBiggerData)
    # print("--- %s seconds for greedy search in big data with plugin FI ---" % (time.time() - start_time))
    # print(f' selected by plugin FI on big: {selected}')

    # start_time=time.time()
    # [selected,bestScore]=greedy_search(mutual_information_permutation_upper_bound,biggerBiggerData)
    # print("--- %s seconds for greedy search in big data with upper-bound FI ---" % (time.time() - start_time))
    # print(f' selected by upper-bound FI on big: {selected}')


if __name__ == "__main__":
    main()
