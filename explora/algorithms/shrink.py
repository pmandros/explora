#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 03:10:56 2020

@author: Panagiotis Mandros
"""

from operator import itemgetter

import numpy as np
import pandas as pd

from explora.information_theory.estimators import conditional_mutual_information
from explora.information_theory.information_theory_basic import entropy_plugin


def shrink(mutual_information_estimator, grow_result, data, shrink_threshold=0, target=None):
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
    entropy_Y=entropy_plugin(Y)

    while len(grow_result) > 1:
        conditional_scores = [evaluate_candidate(mutual_information_estimator, data, index, Y, grow_result, entropy_Y)
                              for index in grow_result]
        print(f'The conditional contributions are: {conditional_scores}')
        conditional_scores.sort(key=itemgetter(0))
        worse = conditional_scores[0]
        worse_score = worse[0]
        worse_candidate_index = worse[1]

        if worse_score <= shrink_threshold + 0.0005:
            grow_result.remove(worse_candidate_index)
        else:
            return {x + 1 for x in grow_result}, mutual_information_estimator(data[:, list(grow_result)], Y)
    return {x + 1 for x in grow_result}, mutual_information_estimator(data[:, list(grow_result)], Y)


def evaluate_candidate(estimator, data, candidate_index, Y, selected_variables, entropy_Y):
    to_condition = set(selected_variables)
    to_condition.remove(candidate_index)
    conditional_score=conditional_mutual_information(estimator, data[:, candidate_index], Y,
                                          data[:, list(to_condition)])/entropy_Y

    return conditional_score, candidate_index
