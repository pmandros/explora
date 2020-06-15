#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:41:21 2020

@author: Panagiotis Mandros
"""


from functools import partial
from itertools import product
from multiprocessing import Pool
import pandas as pd
import numpy as np
import math
from scipy.stats import hypergeom
from timeit import timeit
import time
from utilities.tools import hypergeometric_pmf, make_single_column

 

def expected_mutual_information_permutation_model(X,Y,contingency_table=None,num_threads=1): 
    """ Computes the expected value for the mutual information between sets X and Y
    under the permutation model (i.e., a model of independence). A precomputed contingency table
    can speed-up computations if available. Can be done in paraller for num_threads>1"""
    if contingency_table==None:
        X=make_single_column(X);
        Y=make_single_column(Y);

        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
            X_marginal_counts=X.value_counts();
            Y_marginal_counts=Y.value_counts();

            num_samples=len(X.index)
        elif isinstance(X, np.ndarray):
            X_marginal_counts=np.unique(X, return_counts=True,axis=0)[1];
            Y_marginal_counts=np.unique(Y, return_counts=True,axis=0)[1];

            num_samples=np.size(X,0);
    else:
        contingency_table=contingency_table.to_numpy();
        num_samples=contingency_table[-1,-1];
        Y_marginal_counts=contingency_table[-1,:-1];
        X_marginal_counts=contingency_table[:-1,-1];
                     
    if Y_marginal_counts[0]==num_samples or X_marginal_counts[0]==num_samples:
        return 0

    marginal_counts_cartesian_product = product(X_marginal_counts, Y_marginal_counts)
    if num_threads==1:
        expected_mutual_info=0;    
        for cell_marginals in marginal_counts_cartesian_product:
            expected_mutual_info+=cell_total_contibution_custom(cell_marginals, num_samples)
        return expected_mutual_info
    else:
        partial_app = partial(cell_total_contibution_custom, num_samples=num_samples)
        with Pool(num_threads) as processor:
            return sum(processor.map(partial_app, marginal_counts_cartesian_product))
        
# def cell_total_contibution(cell_marginals, num_samples):
#     """ Calculates cell_contribution for a tuple of marginal count"""
#     marginal_count_one, marginal_count_two = cell_marginals

#     min_iterator = max(1, marginal_count_one + marginal_count_two - num_samples)
#     max_iterator = min(marginal_count_one, marginal_count_two)

#     rv = hypergeom(num_samples, marginal_count_one, marginal_count_two)
#     possible_values = np.arange(min_iterator, max_iterator + 1)
#     probabilities = rv.pmf(possible_values)

#     cell_contribution = 0
#     for value, prob in zip(possible_values, probabilities):
#         cell_contribution += prob * mutual_information_plugin_cell_contribution(
#             value, num_samples, marginal_count_one, marginal_count_two)

#     return cell_contribution  

def cell_total_contibution_custom(cell_marginals, num_samples):
    """ Calculates total cell_contribution for a tuple of marginal counts without external libraries"""
    marginal_count_one, marginal_count_two = cell_marginals

    min_iterator = max(1, marginal_count_one + marginal_count_two - num_samples)
    max_iterator = min(marginal_count_one, marginal_count_two)

    prob = hypergeometric_pmf(min_iterator,num_samples, marginal_count_one, marginal_count_two)
    cell_contribution=prob*mutual_information_plugin_cell_contribution(
            min_iterator, num_samples, marginal_count_one, marginal_count_two)
    possible_values = np.arange(min_iterator+1, max_iterator + 1)   
    
    for value in possible_values:
        temp_one=prob*((marginal_count_one-(value-1))*(marginal_count_two-(value-1)))
        temp_two=value*(num_samples-marginal_count_one-marginal_count_two+value)
        prob=temp_one/temp_two
        cell_contribution += prob * mutual_information_plugin_cell_contribution(
            value, num_samples, marginal_count_one, marginal_count_two)

    return cell_contribution
   
     
def mutual_information_plugin_cell_contribution(cell_count,num_samples, marginal_count_one,marginal_count_two):
    """ Computes the mutual information contribution of a cell"""

    if cell_count==0:
        return 0;
    else:
        first_part=cell_count/num_samples;
        second_part= math.log2(num_samples * cell_count / (marginal_count_one * marginal_count_two));
        return first_part*second_part;
    

if __name__ == '__main__':
    X = np.array([1,1,2,2,3,3])
    Y = np.array([1,1,2,2,3,3])
    
    assert(expected_mutual_information_permutation_model(Y,X)==expected_mutual_information_permutation_model(Y,X,num_threads=4)==0.7849625007211559)
    
    X=np.random.randint(100, size=(10000,1))
    Y=np.random.randint(100, size=(10000,1))


    
    num_rep=2

    single = partial(expected_mutual_information_permutation_model, X, Y)
    print(timeit(single, number=num_rep)/num_rep)
    
    parallel = partial(expected_mutual_information_permutation_model, X, Y,num_threads=4)
    print(timeit(parallel, number=num_rep)/num_rep)
    
    assert(single() == parallel())
