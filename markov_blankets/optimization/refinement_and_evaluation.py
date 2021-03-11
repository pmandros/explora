#!/usr/bin/env ptarget_data_columnthon3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:59:26 2021

@author: Panagiotis Mandros
"""

from  markov_blankets.utilities.tools import append_and_merge
import heapq
import random
from operator import itemgetter
import math


def refine_and_evaluate_generator(estimator, data, target_data_column, selected_variables_indices, candidate_variables_indices,control_variable_indices,selected_data_column=None):
    """
    Creates a generator of all refinements and their scores. 
    target_data_column is the data column of the target variable.
    selected_variables_indices is the indices (starting at 0) of variables selected so far during search.
    candidate_variables_indices is the indices of the refinement elements (i.e., the attributes to refine the current best solution).
    control_variable_indices is the indices of variables to control for (if a conditional estimator is being used)
    selected_data_column is the data vector (column) representing the selected variables as one combined variable 
    
    """
    for candidate_index in candidate_variables_indices:
        if selected_data_column is None:
            if not control_variable_indices:
                result = estimator(data[:, (*selected_variables_indices, candidate_index)], target_data_column)
            else:
                result = estimator(data[:, (*selected_variables_indices, candidate_index)], target_data_column,data[:, control_variable_indices])
        else:
            merged_in_one=append_and_merge(selected_data_column, data[:,candidate_index])
            if not control_variable_indices:                
                result=estimator(merged_in_one,target_data_column)
            else:
                result = estimator(merged_in_one, target_data_column,data[:, control_variable_indices])
        yield result, candidate_index
        
        
def refine_evaluate_choose(estimator, data, target_data_column, selected_variables_indices, candidate_variables_indices,control_variable_indices,selected_data_column, stochastic=False,select_from_top_k=1):
    """
    It performs all refinements, computes their score, and greedily selects one in each search level to continue  
    If stochastic greedy is used, then a number of variables is subsampled in each level for greedy search
    If select_from_top_k>1, then random greedy is used, otherwise the standard greedy is used
    
    """
    if stochastic is True:
        candidate_variables_indices=random.sample(candidate_variables_indices,k=math.floor(math.sqrt(len(candidate_variables_indices))))
            
    refinements_and_scores = refine_and_evaluate_generator(
            estimator=estimator,
            data=data,
            target_data_column=target_data_column,
            selected_variables_indices=selected_variables_indices,
            candidate_variables_indices=candidate_variables_indices,
            selected_data_column=selected_data_column,
            control_variable_indices=control_variable_indices
        )

    
    
     # random greedy or standard greedy 
    q_best_candidates=heapq.nlargest(select_from_top_k, refinements_and_scores, key=itemgetter(0))
    selected_candidate_score,selected_candidate_index=random.choice(q_best_candidates)
    return selected_candidate_score,selected_candidate_index
        