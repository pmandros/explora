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

from explora.some_statistics.basic_statistics import hypergeometric_pmf
from explora.utilities.tools import size_and_counts_of_contingency_table


def expected__mutual_information_permutation_model_upper_bound(X,Y,with_cross_tab=False,contingency_table=None):
    """
    Computes an upper-bound (Nguyen et al. 2010) to the expected value of mutual 
    information under the permutation model. Can be computed using Pandas cross_tab, or with a 
    precomputed contingency table"""
     
    size,marginal_counts_X,marginal_counts_Y=size_and_counts_of_contingency_table(X,Y,with_cross_tab,contingency_table)
    domain_size_X=len(marginal_counts_X)
    domain_size_Y=len(marginal_counts_Y)

        
    return np.log2((size+domain_size_X*domain_size_Y-domain_size_X-domain_size_Y)/(size-1))


def expected_mutual_information_permutation_model(X, Y, with_cross_tab=False,contingency_table=None, num_threads=1):
    """
    Computes the expected value of mutual information under the permutation model.
    A precomputed contingency table can be provided. Can be done in parallel"""
   
    num_samples,X_marginal_counts,Y_marginal_counts=size_and_counts_of_contingency_table(X,Y,return_joint_counts=False,
                                                                                                with_cross_tab=with_cross_tab,contingency_table=contingency_table)
    
    if Y_marginal_counts[0] == num_samples or X_marginal_counts[0] == num_samples:
        return 0

    marginal_counts_cartesian_product = product(X_marginal_counts, Y_marginal_counts)
    if num_threads == 1:
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
        cont= 0
    else:
        first_part = cell_count / num_samples
        second_part = math.log(
            num_samples * cell_count / (marginal_count_one * marginal_count_two)
        ) / math.log(2)
        cont= first_part * second_part
    # print("cell cont: ",cont)
    return cont


def main():
   
    # test 1 permutation model
    X = np.array([1, 1, 2, 2, 3, 3])
    Y = np.array([1, 1, 2, 2, 3, 3])

    assert (
        expected_mutual_information_permutation_model(X, Y)
        == expected_mutual_information_permutation_model(X, Y, num_threads=4)
        == 0.7849625007211563
    )

    # test 2 (performance) permutation model "small" input
    X = np.random.randint(100, size=(10000, 1))
    Y = np.random.randint(100, size=(10000, 1))
    
    num_rep = 5
    
    single = partial(expected_mutual_information_permutation_model, X, Y)
    print(timeit(single, number=num_rep) / num_rep, " seconds for single thread on small input")

    parallel = partial(
        expected_mutual_information_permutation_model, X, Y, num_threads=4
    )
    print(timeit(parallel, number=num_rep) / num_rep, " seconds for multi threading (4) on small input")
    
    assert single() == parallel()
    
    # test 3 (performance) permutation model "big" input
    X = np.random.randint(1000, size=(100000, 1))
    Y = np.random.randint(1000, size=(100000, 1))
    
    num_rep = 5
    
    single = partial(expected_mutual_information_permutation_model, X, Y)
    print(timeit(single, number=num_rep) / num_rep, " seconds for single thread on big input")

    parallel = partial(
        expected_mutual_information_permutation_model, X, Y, num_threads=4
    )
    print(timeit(parallel, number=num_rep) / num_rep, " seconds for multi threading (4) on big input")
    
    assert single() == parallel()
    
    
    # test 4 permutation model on real data
    data = pd.read_csv("../datasets/tic_tac_toe.csv");
    emi=expected_mutual_information_permutation_model(data.iloc[:,4],data.iloc[:,9])
    assert(emi==0.0015104057711462328)
    emi=expected_mutual_information_permutation_model(data.iloc[:,[0,2,4,6,8]],data.iloc[:,9])
    assert(emi==0.158326256563715)
    
    # # test 5 permutation model upper-bound on real data
    data = pd.read_csv("../datasets/tic_tac_toe.csv");
    emi=expected__mutual_information_permutation_model_upper_bound(data.iloc[:,4],data.iloc[:,9])
    assert(emi==0.003011890532105174)
    emi=expected__mutual_information_permutation_model_upper_bound(data.iloc[:,[0,2,4,6,8]],data.iloc[:,9])
    assert(emi==0.2422831283458568)
    
    # test 6 permutation model on real big data (should give 0)
    data = pd.read_csv("../datasets/tic_tac_toe.csv");
    biggerData=data.append(data).append(data).append(data).append(data).append(data).append(data)
    biggerBiggerData=biggerData.append(biggerData).append(biggerData).append(biggerData).append(biggerData).append(biggerData)
    biggerBiggerData=biggerBiggerData.append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData) 
    emi=expected_mutual_information_permutation_model(biggerBiggerData.iloc[:,4],biggerBiggerData.iloc[:,9])
    assert(emi==0)
    
    



if __name__ == "__main__":
    main()
