#!/usr/bin/env ptarget_data_columnthon3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:59:26 2021

@author: Panagiotis Mandros
"""

import heapq
import math
import random
from operator import itemgetter

from explora.utilities.tools import append_and_merge


def refine_and_evaluate_generator(estimator, data, target_data_column, selected_variables_indices,
                                  candidate_variables_indices, control_variable_indices, selected_data_column=None):
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
                result = estimator(data[:, (*selected_variables_indices, candidate_index)], target_data_column,
                                   data[:, control_variable_indices])
        else:
            merged_in_one = append_and_merge(selected_data_column, data[:, candidate_index])
            if not control_variable_indices:
                result = estimator(merged_in_one, target_data_column)
            else:
                result = estimator(merged_in_one, target_data_column, data[:, control_variable_indices])
        yield result, candidate_index


def refine_evaluate_get_optimistic_value_generator(estimator, data, target_data_column,
                                                   selected_variables_indices,
                                                   candidate_variables_indices,
                                                   selected_data_column=None):
    """
    Creates a generator of all refinements, their scores, and their optimistic value to be used for pruning.
    target_data_column is the data column of the target variable.
    selected_variables_indices is the indices (starting at 0) of variables selected so far during search.
    candidate_variables_indices is the indices of the refinement elements (i.e., the attributes to refine the current best solution).
    selected_data_column is the data vector (column) representing the selected variables as one combined variable

    """
    for candidate_index in candidate_variables_indices:
        if selected_data_column is None:
            result, correction = estimator(data[:, (*selected_variables_indices, candidate_index)],
                                           target_data_column, return_correction_term=True)
        else:
            merged_in_one = append_and_merge(selected_data_column, data[:, candidate_index])
            result, correction = estimator(merged_in_one, target_data_column,
                                           return_correction_term=True)

        estimator_name = estimator.__name__
        if 'fraction' not in estimator_name.lower():
            raise ValueError('Optimistic estimator does not apply to non fraction of information scores')
        else:
            yield result, candidate_index, 1 - correction


def refine_evaluate_choose(estimator, data, target_data_column, selected_variables_indices, candidate_variables_indices,
                           control_variable_indices, selected_data_column, stochastic=False, select_from_top_k=1):
    """
    It performs all refinements, computes their score, and greedily selects one in each search level to continue  
    If stochastic greedy is used, then a number of variables is subsampled in each level for greedy search
    If select_from_top_k>1, then random greedy is used, otherwise the standard greedy is used
    
    """
    if stochastic is True:
        candidate_variables_indices = random.sample(candidate_variables_indices,
                                                    k=math.floor(math.sqrt(len(candidate_variables_indices))))

    scores_and_refinements = refine_and_evaluate_generator(
        estimator=estimator,
        data=data,
        target_data_column=target_data_column,
        selected_variables_indices=selected_variables_indices,
        candidate_variables_indices=candidate_variables_indices,
        selected_data_column=selected_data_column,
        control_variable_indices=control_variable_indices
    )

    # random greedy or standard greedy
    q_best_candidates = heapq.nlargest(select_from_top_k, scores_and_refinements, key=itemgetter(0))
    selected_candidate_score, selected_candidate_index = random.choice(q_best_candidates)
    return selected_candidate_score, selected_candidate_index


def refine_evaluate_get_optimistic_value_sort(estimator, data, target_data_column, selected_variables_indices,
                                              candidate_variables_indices,
                                              selected_data_column=None):
    """
    It performs all refinements, computes their score, gets their optimistic value and sorts

    """
    scores_and_refinement_indices_and_opt_values = refine_evaluate_get_optimistic_value_generator(
        estimator=estimator,
        data=data,
        target_data_column=target_data_column,
        selected_variables_indices=selected_variables_indices,
        candidate_variables_indices=candidate_variables_indices,
        selected_data_column=selected_data_column
    )

    # random greedy or standard greedy
    return sorted(scores_and_refinement_indices_and_opt_values, key=itemgetter(0), reverse=True)
