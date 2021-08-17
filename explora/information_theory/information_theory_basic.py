#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:29:50 2020

@author: Panagiotis Mandros
"""

import numpy as np
import pandas as pd
import scipy.stats as sc

from explora.some_statistics.basic_statistics import empirical_statistics
from explora.utilities.tools import merge_columns, size_and_counts_of_contingency_table, \
    append_two_arrays


def entropy(prob_vector):
    """
    Computes the Shannon entropy of a probability distribution corresponding to
    a random variable"""
    return sc.entropy(prob_vector, base=2);


def entropy_plugin(X, return_statistics=False):
    """
    The plugin estimator for Shannon entropy H(X) of an attribute X. Can optionally 
    return the domain size and length of X"""
    empiricalDistribution, domainSize, length = empirical_statistics(X);
    if return_statistics == True:
        return entropy(empiricalDistribution), domainSize, length
    else:
        return entropy(empiricalDistribution)


def mutual_information(prob_vector_X, prob_vector_Y, prob_vector_XY):
    """
    Computes the mutual information between two sets of random variables X and Y"""
    return entropy(prob_vector_X) + entropy(prob_vector_Y) - entropy(prob_vector_XY);


def mutual_information_plugin(X, Y, with_cross_tab=False, contingency_table=None, return_marginal_entropies=False):
    """
    The plugin estimator for mutual information I(X;Y) between two attribute sets X 
    and Y. It can be computed either using cross_tab from Pandas, or with
    numpy. A precomputed contingency table can be provided if it is available"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.to_numpy();

    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy();

    if with_cross_tab == True or contingency_table != None:
        return mutual_information_from_cross_tab(X, Y, contingency_table=contingency_table,
                                                 return_marginal_entropies=return_marginal_entropies);
    else:
        entropy_X = entropy_plugin(X);
        entropy_Y = entropy_plugin(Y);
        data_XY = append_two_arrays(X, Y);
        entropy_XY = entropy_plugin(data_XY);
        mi = entropy_X + entropy_Y - entropy_XY

        if return_marginal_entropies:
            return mi, entropy_X, entropy_Y
        else:
            return mi


def fraction_of_information_plugin(X, Y, with_cross_tab=False, contingency_table=None):
    """
    The plugin estimator for the fraction of information F(X;Y)=I(X,Y)/H(Y). 
    It can be computed either using cross_tab from Pandas, or with
    numpy. A precomputed contingency table and entropy of Y can be provided if
    it is available"""

    mi, entropy_X, entropy_Y = mutual_information_plugin(X, Y, with_cross_tab=with_cross_tab,
                                                         contingency_table=contingency_table,
                                                         return_marginal_entropies=True)

    return mi / entropy_Y


def conditional_entropy_plugin(Z, Y):
    """
    The plugin estimator for the conditional entropy H(Y|Z) of Y given Z"""
    if isinstance(Z, pd.Series) or isinstance(Z, pd.DataFrame):
        Y = Y.to_numpy();
        Z = Z.to_numpy();

    Z = merge_columns(Z);
    Y = merge_columns(Y);

    length = np.size(Z, 0);
    uniqueValuesZ = np.unique(Z);
    conditional_entropy_plugin = 0;
    for value in uniqueValuesZ:
        indices = np.where(Z == value);
        valueCount = len(indices);

        tempY = Y[indices[0]];
        entropyTempY = entropy_plugin(tempY);

        conditional_entropy_plugin = conditional_entropy_plugin + valueCount / length * entropyTempY;
    return conditional_entropy_plugin;


def mutual_information_from_cross_tab(X, Y, contingency_table=None, return_marginal_entropies=False):
    """
    Computes mutual information using cross_tab from pandas. A precomputed 
    contingency table can be provided if it is available """
    size, marginal_counts_X, marginal_counts_Y, joint_counts = size_and_counts_of_contingency_table(X, Y,
                                                                                                    return_joint_counts=True,
                                                                                                    with_cross_tab=True,
                                                                                                    contingency_table=contingency_table)

    entropy_X = entropy(marginal_counts_X / size)
    entropy_Y = entropy(marginal_counts_Y / size)
    entropy_XY = entropy(joint_counts / size)
    mi = entropy_X + entropy_Y - entropy_XY

    if return_marginal_entropies:
        return mi, entropy_X, entropy_Y
    else:
        return mi

