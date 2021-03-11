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

from timeit import timeit
from  utilities.tools import append_and_merge
from refinement_and_evaluation import refine_evaluate_choose
import math
import heapq
import random

from markov_blankets.information_theory.estimators import fraction_of_information_permutation_upper_bound
from markov_blankets.utilities.tools import append_and_merge


def greedy_search(estimator, data, target_variable=None, limit=None, select_from_top_k=1, is_stochastic=False, control_variables=None):
    """
    Given data and an estimator, it greedily maximizes the estimator of a dependency measure D(X;Y) over
    candidate attribute sets in the data. If no target_variable index is provided (indexed from 1), the target_variable is the last
    attribute. If a limit is provided, the search will stop at the limit level, e.g., for 2, the search
    wont continue after the second level (pairs of attributes). Turns into random greedy search when select_from_top_k>1, i.e., select 
    uniformly from the top select_from_top_k results per level and not the maximum. If stochastic is true,
    then algorithm becomes stochastic greedy by subsampling square root of variables to optimize. Control variables 
    (their indices, starting from 1) can be passed for conditional maximization. In that case,
    the estimator is a conditional dependency measure D(X;Y|Z)
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    #the number of attributes excluding the target_variable
    number_explanatory_variables = np.size(data, 1) - 1

    if target_variable is None:
        target_variable_index = number_explanatory_variables
    else:
        target_variable_index = target_variable - 1
    Y_data_column = data[:, target_variable_index]

    if limit is None:
        limit = number_explanatory_variables
        
    if control_variables is None:
        control_variables_indices=()
    else:
        control_variables_indices=tuple({x-1 for x in control_variables})
    
    candidate_variables_indices = set(range(number_explanatory_variables + 1))
    candidate_variables_indices.remove(target_variable_index)
    candidate_variables_indices.difference_update(control_variables_indices)


    selected_variables_indices = set()
    best_score = -float('inf')
    selected_data_column=None

    for i in range(limit):  
        print(i)
        selected_candidate_score,selected_candidate_index=refine_evaluate_choose(estimator, data, Y_data_column, 
                                                                                 selected_variables_indices, candidate_variables_indices,control_variables_indices,selected_data_column, is_stochastic,select_from_top_k)
        
        # terminate early if score does not improve, or when score is <=0
        if selected_candidate_score <= 0 or selected_candidate_score <= best_score:
            return {x+1 for x in selected_variables_indices}, best_score

        if selected_candidate_score > best_score:
            selected_data_column=append_and_merge(selected_data_column,data[:,selected_candidate_index])
            best_score = selected_candidate_score
            selected_variables_indices.add(selected_candidate_index)
            candidate_variables_indices.remove(selected_candidate_index)
    return {x+1 for x in selected_variables_indices}, best_score


def main():
    # data = pd.read_csv("../datasets/tic_tac_toe.csv")
    # biggerData=data.append(data).append(data).append(data).append(data).append(data).append(data)
    # biggerBiggerData=biggerData.append(biggerData).append(biggerData).append(biggerData).append(biggerData).append(biggerData)
    # biggerBiggerData=biggerBiggerData.append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData)


    # # test for upper-bound MI
    # [selected,bestScore]=greedy_search(mutual_information_permutation_upper_bound,data)
    # assert(math.isclose(0.28707781833145635,bestScore,abs_tol=1e-8))
    # assert(selected=={1,5,9})
    # print(selected)

    # # test for permutation upper-bound FI
    # [selected,bestScore]=greedy_search(fraction_of_information_permutation_upper_bound,data)
    # assert(math.isclose(0.3083695032958573,bestScore,abs_tol=1e-8))
    
    
    # # test for permutation FI 
    # [selected,bestScore]=greedy_search(fraction_of_information_permutation,data)
    # assert(math.isclose(0.4447970033469652,bestScore,abs_tol=1e-8))
    # assert(selected=={1,5,9, 3,7})

    # # conditional test for fraction of information permutation
    # control_var_set={5}
    # [selected,bestScore]=greedy_search(conditional_fraction_of_information_permutation,data,control_variables=control_var_set)
    # assert(math.isclose(0.3561447636856704,bestScore,abs_tol=1e-8))
    # assert(selected=={1,9, 3,7})
    
    # # conditional test for fraction of information permutation
    # control_var_set={5}
    # [selected,bestScore]=greedy_search(conditional_fraction_of_information_permutation,data,control_variables=control_var_set,select_from_top_k=3)
    # print(f'selected {selected} with score {bestScore}')
    
    
    #  # conditional test for fraction of information permutation
    # [selected,bestScore]=greedy_search(fraction_of_information_permutation,data,select_from_top_k=1,target_variable=1)
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
    # # smallDataMI = partial(greedy_search,mutual_information_plugin,data,select_from_top_k=3)
    # # print(timeit(smallDataMI, number=num_rep)/num_rep, "Small data random plugin MI")
    
    # # biggerBiggerDataUpMI = partial(greedy_search, mutual_information_plugin,biggerBiggerData,select_from_top_k=3)
    # # print(timeit(biggerBiggerDataUpMI, number=num_rep)/num_rep,"Big data random plugin MI")    
    
    
    # # # permormance with num coluns
    # # dfs=[biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData,biggerBiggerData]
    # # high_dim_data = pd.concat(dfs, axis=1)
    
    # # biggerBiggerDataUpMI = partial(greedy_search, mutual_information_plugin,high_dim_data, select_from_top_k=5)
    # # # print(high_dim_data.shape)
    # # print(timeit(biggerBiggerDataUpMI, number=num_rep)/num_rep,"Data many column random plugin MI")

    
    data_mnist = pd.read_csv("../datasets/mnist_test_10k_binary.csv",header=None)
    [selected,bestScore]=greedy_search(fraction_of_information_permutation_upper_bound,data_mnist,is_stochastic=True, limit=20)
    print(f'selected for MNIST {selected} with score {bestScore}')
    data_mnist['combined'] = data_mnist[list(selected)].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    cros=pd.crosstab(data_mnist.loc[:,784],data_mnist['combined'])
    cros.to_csv("output.csv", index=False)
    print(cros)

if __name__ == "__main__":
    main()
