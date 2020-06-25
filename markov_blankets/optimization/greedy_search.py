#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:45:52 2020

@author: Panagiotis Mandros
"""

import time
from operator import itemgetter
from functools import partial
import numpy as np
import pandas as pd
from timeit import timeit
from  utilities.tools import append_and_merge
import math
import heapq
import random

from information_theory.estimators import mutual_information_permutation_upper_bound, fraction_of_information_permutation_upper_bound, fraction_of_information_permutation
from information_theory.information_theory_basic import mutual_information_plugin, fraction_of_information_plugin


# def greedy_search_old(estimator, data, target=None, limit=None, prior_solution_set=None):
#     """
#     Given data, it greedily maximizes an estimator of a dependency measure D(XY), over
#     candidate attribute sets in X in the data. It selects the best candidate to expand 
#     in a BFS manner. If no target index is provided (indexed from 1), the target is the last
#     attribute. If a limit is provided, the search will stop at at level, e.g., for 2, the search
#     wont continue after the second level (pairs of attributes). A prior_solution_set can be used to
#     initialize the search.    
#     """
#     if isinstance(data, pd.DataFrame):
#         data = data.to_numpy()

#     number_explanatory_variables = np.size(data, 1) - 1

#     if target is None:
#         target_index = number_explanatory_variables
#     else:
#         target_index = target - 1
#     Y = data[:, target_index]

#     if limit is None:
#         limit = number_explanatory_variables

#     set_of_candidates = set(range(number_explanatory_variables + 1))
#     set_of_candidates.remove(target_index)

#     if prior_solution_set:
#         set_of_candidates.difference_update(prior_solution_set)

#     selected_variables = set()
#     best_score = -1000

#     for i in range(limit):

#         generator = sort_generator(
#             estimator=estimator,
#             data=data,
#             Y=Y,
#             selected=selected_variables,
#             candidates=set_of_candidates,
#         )

#         top_score, top_candidate_index = max(generator, key=itemgetter(0))

#         if top_score <= 0 or top_score <= best_score:
#             return selected_variables, best_score

#         if top_score > best_score:
#             best_score = top_score
#             selected_variables.add(top_candidate_index)
#             set_of_candidates.remove(top_candidate_index)
#     return selected_variables, best_score


# def greedy_search_no_random(estimator, data, target=None, limit=None, prior_solution_set=None):
#     """
#     Given data, it greedily maximizes an estimator of a dependency measure D(XY), over
#     candidate attribute sets in X in the data. It selects the best candidate to expand 
#     in a BFS manner. If no target index is provided (indexed from 1), the target is the last
#     attribute. If a limit is provided, the search will stop at at level, e.g., for 2, the search
#     wont continue after the second level (pairs of attributes). A prior_solution_set can be used to
#     initialize the search.    
#     """
#     if isinstance(data, pd.DataFrame):
#         data = data.to_numpy()

#     number_explanatory_variables = np.size(data, 1) - 1

#     if target is None:
#         target_index = number_explanatory_variables
#     else:
#         target_index = target - 1
#     Y = data[:, target_index]

#     if limit is None:
#         limit = number_explanatory_variables

#     set_of_candidates = set(range(number_explanatory_variables + 1))
#     set_of_candidates.remove(target_index)

#     if prior_solution_set:
#         set_of_candidates.difference_update(prior_solution_set)

#     selected_variables = set()
#     best_score = -1000
#     result_single_column=None

#     for i in range(limit):

#         generator = sort_generator(
#             estimator=estimator,
#             data=data,
#             Y=Y,
#             selected=selected_variables,
#             candidates=set_of_candidates,
#             selected_single_column=result_single_column
#         )

#         top_score, top_candidate_index = max(generator, key=itemgetter(0))

#         if top_score <= 0 or top_score <= best_score:
#             print(selected_variables,best_score)
#             return selected_variables, best_score

#         if top_score > best_score:
#             result_single_column=append_and_merge(result_single_column,data[:,top_candidate_index])
#             best_score = top_score
#             selected_variables.add(top_candidate_index)
#             set_of_candidates.remove(top_candidate_index)
    # return selected_variables, best_score

def greedy_search(estimator, data, target=None, limit=None, prior_solution_set=None, randomly_choose_from_top=1):
    """
    Given data, it greedily maximizes an estimator of a dependency measure D(XY), over
    candidate attribute sets in X in the data. It selects the best candidate to expand 
    in a BFS manner. If no target index is provided (indexed from 1), the target is the last
    attribute. If a limit is provided, the search will stop at at level, e.g., for 2, the search
    wont continue after the second level (pairs of attributes). A prior_solution_set can be used to
    initialize the search. Turns into random greedy search when randomly_choose_from_top>1, i.e., select 
    uniformly from randomly_choose_from_top per level and not the maximum.
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
    result_single_column=None

    for i in range(limit):
        generator = sort_generator(
            estimator=estimator,
            data=data,
            Y=Y,
            selected=selected_variables,
            candidates=set_of_candidates,
            selected_single_column=result_single_column
        )
               
        k_largest=heapq.nlargest(randomly_choose_from_top, generator, key=itemgetter(0))
        random_top_score,random_top_candidate_index=random.choice(k_largest)

        if random_top_score <= 0 or random_top_score <= best_score:
            return selected_variables, best_score

        if random_top_score > best_score:
            result_single_column=append_and_merge(result_single_column,data[:,random_top_candidate_index])
            best_score = random_top_score
            selected_variables.add(random_top_candidate_index)
            set_of_candidates.remove(random_top_candidate_index)
    return selected_variables, best_score


def sort_generator(estimator, data, Y, selected, candidates, selected_single_column=None):
    for candidate_idx in candidates:
        if selected_single_column is None:
            result = estimator(data[:, (*selected, candidate_idx)], Y)
        else:
            merged_in_one=append_and_merge(selected_single_column, data[:,candidate_idx])
            result=estimator(merged_in_one,Y)
        yield result, candidate_idx


def main():
    data = pd.read_csv("../datasets/tic_tac_toe.csv")
    biggerData=data.append(data).append(data).append(data).append(data).append(data).append(data)
    biggerBiggerData=biggerData.append(biggerData).append(biggerData).append(biggerData).append(biggerData).append(biggerData)
    biggerBiggerData=biggerBiggerData.append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData)


    # test for upper-bound MI
    [selected,bestScore]=greedy_search(mutual_information_permutation_upper_bound,data)
    assert(math.isclose(0.28707781833145635,bestScore,abs_tol=1e-8))
    assert(selected=={0,4,8})

    # test for permutation upper-bound FI
    [selected,bestScore]=greedy_search(fraction_of_information_permutation_upper_bound,data)
    assert(math.isclose(0.3083695032958573,bestScore,abs_tol=1e-8))
    
    
    # test for permutation FI 
    [selected,bestScore]=greedy_search(fraction_of_information_permutation,data)
    assert(math.isclose(0.4447970033469652,bestScore,abs_tol=1e-8))

    assert(selected=={0,4,8, 2,6})

    

   
    # # # performance tests
    # num_rep=1
 
    # # plugin MI
    # smallDataMI = partial(greedy_search,mutual_information_plugin,data)
    # print(timeit(smallDataMI, number=num_rep)/num_rep, "Small data plugin MI")
    
    # biggerBiggerDataUpMI = partial(greedy_search, mutual_information_plugin,biggerBiggerData)
    # print(timeit(biggerBiggerDataUpMI, number=num_rep)/num_rep,"Big data plugin MI")    
    
    # #  upper bound MI
    # smallDataUpFI = partial(greedy_search, mutual_information_permutation_upper_bound,data)
    # print(timeit(smallDataUpFI, number=num_rep)/num_rep,"Small data upper-bound MI")
    
    # biggerBiggerDataUpFI = partial(greedy_search, mutual_information_permutation_upper_bound,biggerBiggerData)
    # print(timeit(biggerBiggerDataUpFI, number=num_rep)/num_rep,"Big data upper-bound MI")
    
    # # corrected FI
    # smallDataUpFI = partial(greedy_search, fraction_of_information_permutation,data)
    # print(timeit(smallDataUpFI, number=num_rep)/num_rep,"Small data corrected FI")
    
    # biggerBiggerDataUpFI = partial(greedy_search, fraction_of_information_permutation,biggerBiggerData)
    # print(timeit(biggerBiggerDataUpFI, number=num_rep)/num_rep,"Big data corrected FI")
      
    
    #   # performance of random
    # smallDataMI = partial(greedy_search,mutual_information_plugin,data,randomly_choose_from_top=3)
    # print(timeit(smallDataMI, number=num_rep)/num_rep, "Small data random plugin MI")
    
    # biggerBiggerDataUpMI = partial(greedy_search, mutual_information_plugin,biggerBiggerData,randomly_choose_from_top=3)
    # print(timeit(biggerBiggerDataUpMI, number=num_rep)/num_rep,"Big data random plugin MI")    
    
    
    # # permormance with num coluns
    # dfs=[biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData]
    # high_dim_data = pd.concat(dfs, axis=1)
    
    # biggerBiggerDataUpMI = partial(greedy_search, mutual_information_plugin,high_dim_data, randomly_choose_from_top=5)
    # # print(high_dim_data.shape)
    # print(timeit(biggerBiggerDataUpMI, number=num_rep)/num_rep,"Data many column random plugin MI")

    


if __name__ == "__main__":
    main()
