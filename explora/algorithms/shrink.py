#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 03:10:56 2020

@author: Panagiotis Mandros
"""

from operator import itemgetter

import numpy as np
import pandas as pd


def shrink(conditional_estimator, grow_result, data, shrink_threshold=0, target=None):
    """
    For a dependency measure D(X;Y), it shrinks the grow_result G by calculating
    the conditional D(Z;Y|G-{Z}) for all Z in G and removing Z if bellow threshold
    WARNING: requires grow_result to be 0 indexed (which might not be the case)
    """

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    if target is None:
        target_index = np.size(data, 1) - 1
    else:
        target_index = target - 1
    Y = data[:, target_index]

    while len(grow_result) > 1:
        conditional_scores = [evaluate_candidate(conditional_estimator, data, index, Y, grow_result) for index in
                              grow_result]
        print(conditional_scores)
        conditional_scores.sort(key=itemgetter(0))
        worse = conditional_scores[0]
        worse_score = worse[0]
        worse_candidate_index = worse[1]

        if worse_score <= shrink_threshold + 0.00005:
            grow_result.remove(worse_candidate_index)
        else:
            return {x + 1 for x in grow_result}
    return {x + 1 for x in grow_result}


def evaluate_candidate(estimator, data, candidate_index, Y, selected_variables):
    to_condition = set(selected_variables)
    to_condition.remove(candidate_index)

    return estimator(data[:, candidate_index], Y, data[:, list(to_condition)]), candidate_index
