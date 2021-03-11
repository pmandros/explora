#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:45:52 2020

@author: Panagiotis Mandros
"""

import heapq
import random
from operator import itemgetter

import numpy as np
import pandas as pd

from markov_blankets.information_theory.estimators import fraction_of_information_permutation_upper_bound
from markov_blankets.utilities.tools import append_and_merge


def greedy_search(estimator, data, target=None, limit=None, prior_solution_set=None, randomly_choose_from_top=1, control_set=None):
    """
    Given data, it greedily maximizes an estimator of a dependency measure D(X;Y), over
    candidate attribute sets in X in the data. It selects the best candidate to expand 
    in a BFS manner. If no target index is provided (indexed from 1), the target is the last
    attribute. If a limit is provided, the search will stop at at level, e.g., for 2, the search
    wont continue after the second level (pairs of attributes). A prior_solution_set can be used to
    initialize the search. Turns into random greedy search when randomly_choose_from_top>1, i.e., select 
    uniformly from the top randomly_choose_from_top per level and not the maximum. Control variables 
    (their indices, starting from 1) can be passed for conditional maximization. In that case,
    the estimator is a conditional dependency measure D(X;Y|Z)
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
        
    if control_set is None:
        control_set=()
    else:
        control_set=tuple({x-1 for x in control_set})
    
    set_of_candidates = set(range(number_explanatory_variables + 1))
    set_of_candidates.remove(target_index)
    set_of_candidates.difference_update(control_set)

    if prior_solution_set:
        set_of_candidates.difference_update(prior_solution_set)

    selected_variables = set()
    best_score = -float('inf')
    result_single_column=None

    for i in range(limit):
        print(i)
        generator = candidate_generator(
            estimator=estimator,
            data=data,
            Y=Y,
            selected=selected_variables,
            candidates=set_of_candidates,
            selected_single_column=result_single_column,
            control_set=control_set
        )
         
        # random greedy or standard greedy 
        q_best_candidates=heapq.nlargest(randomly_choose_from_top, generator, key=itemgetter(0))
        selected_candidate_score,selected_candidate_index=random.choice(q_best_candidates)

        # terminate early if score does not improve, or when score is <=0
        if selected_candidate_score <= 0 or selected_candidate_score <= best_score:
            return {x+1 for x in selected_variables}, best_score

        if selected_candidate_score > best_score:
            result_single_column=append_and_merge(result_single_column,data[:,selected_candidate_index])
            best_score = selected_candidate_score
            selected_variables.add(selected_candidate_index)
            set_of_candidates.remove(selected_candidate_index)
    return {x+1 for x in selected_variables}, best_score


def candidate_generator(estimator, data, Y, selected, candidates,control_set,selected_single_column=None):
    for candidate_idx in candidates:
        if selected_single_column is None:
            if not control_set:
                result = estimator(data[:, (*selected, candidate_idx)], Y)
            else:
                result = estimator(data[:, (*selected, candidate_idx)], Y,data[:, control_set])
        else:
            merged_in_one=append_and_merge(selected_single_column, data[:,candidate_idx])
            if not control_set:                
                result=estimator(merged_in_one,Y)
            else:
                result = estimator(merged_in_one, Y,data[:, control_set])
        yield result, candidate_idx


def main():
    # data = pd.read_csv("../datasets/tic_tac_toe.csv")
    # biggerData=data.append(data).append(data).append(data).append(data).append(data).append(data)
    # biggerBiggerData=biggerData.append(biggerData).append(biggerData).append(biggerData).append(biggerData).append(biggerData)
    # biggerBiggerData=biggerBiggerData.append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData)


    # # test for upper-bound MI
    # [selected,bestScore]=greedy_search(mutual_information_permutation_upper_bound,data)
    # assert(math.isclose(0.28707781833145635,bestScore,abs_tol=1e-8))
    # assert(selected=={1,5,9})

    # # test for permutation upper-bound FI
    # [selected,bestScore]=greedy_search(fraction_of_information_permutation_upper_bound,data)
    # assert(math.isclose(0.3083695032958573,bestScore,abs_tol=1e-8))
    
    
    # # test for permutation FI 
    # [selected,bestScore]=greedy_search(fraction_of_information_permutation,data)
    # assert(math.isclose(0.4447970033469652,bestScore,abs_tol=1e-8))
    # assert(selected=={1,5,9, 3,7})

    # # conditional test for fraction of information permutation
    # control_var_set={5}
    # [selected,bestScore]=greedy_search(conditional_fraction_of_information_permutation,data,control_set=control_var_set)
    # assert(math.isclose(0.3561447636856704,bestScore,abs_tol=1e-8))
    # assert(selected=={1,9, 3,7})
    
    # # conditional test for fraction of information permutation
    # control_var_set={5}
    # [selected,bestScore]=greedy_search(conditional_fraction_of_information_permutation,data,control_set=control_var_set,randomly_choose_from_top=3)
    # print(f'selected {selected} with score {bestScore}')
    
    
    #  # conditional test for fraction of information permutation
    # [selected,bestScore]=greedy_search(fraction_of_information_permutation,data,randomly_choose_from_top=1,target=1)
    # print(f'selected {selected} with score {bestScore}')
   
    # # # # performance tests
    # num_rep=5
 
    # # # plugin MI
    # # smallDataMI = partial(greedy_search,mutual_information_plugin,data)
    # # print(timeit(smallDataMI, number=num_rep)/num_rep, "Small data plugin MI")
    
    # # biggerBiggerDataUpMI = partial(greedy_search, mutual_information_plugin,biggerBiggerData)
    # # print(timeit(biggerBiggerDataUpMI, number=num_rep)/num_rep,"Big data plugin MI")    
    
    # # #  upper bound MI
    # # smallDataUpFI = partial(greedy_search, mutual_information_permutation_upper_bound,data)
    # # print(timeit(smallDataUpFI, number=num_rep)/num_rep,"Small data upper-bound MI")
    
    # # biggerBiggerDataUpFI = partial(greedy_search, mutual_information_permutation_upper_bound,biggerBiggerData)
    # # print(timeit(biggerBiggerDataUpFI, number=num_rep)/num_rep,"Big data upper-bound MI")
    
    # # corrected FI
    # smallDataUpFI = partial(greedy_search, fraction_of_information_permutation_upper_bound,data)
    # print(timeit(smallDataUpFI, number=num_rep)/num_rep,"Small data corrected FI")
    
    # # biggerBiggerDataUpFI = partial(greedy_search, fraction_of_information_permutation,biggerBiggerData)
    # # print(timeit(biggerBiggerDataUpFI, number=num_rep)/num_rep,"Big data corrected FI")
      
    
    # #   # performance of random
    # # smallDataMI = partial(greedy_search,mutual_information_plugin,data,randomly_choose_from_top=3)
    # # print(timeit(smallDataMI, number=num_rep)/num_rep, "Small data random plugin MI")
    
    # # biggerBiggerDataUpMI = partial(greedy_search, mutual_information_plugin,biggerBiggerData,randomly_choose_from_top=3)
    # # print(timeit(biggerBiggerDataUpMI, number=num_rep)/num_rep,"Big data random plugin MI")    
    
    
    # # # permormance with num coluns
    # # dfs=[biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData]
    # # high_dim_data = pd.concat(dfs, axis=1)
    
    # # biggerBiggerDataUpMI = partial(greedy_search, mutual_information_plugin,high_dim_data, randomly_choose_from_top=5)
    # # # print(high_dim_data.shape)
    # # print(timeit(biggerBiggerDataUpMI, number=num_rep)/num_rep,"Data many column random plugin MI")

    
    data_mnist = pd.read_csv("../datasets/mnist_test_10k_binary.csv")
    [selected,bestScore]=greedy_search(fraction_of_information_permutation_upper_bound,data_mnist,randomly_choose_from_top=1,limit=20)
    print(f'selected for MNIST {selected} with score {bestScore}')
    

if __name__ == "__main__":
    main()
