#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:13:25 2020

@author: Panagiotis Mandros
"""

import time

import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2

from explora.information_theory.information_theory_basic import mutual_information_plugin, entropy_plugin
from explora.information_theory.permutation_model import \
    expected_mutual_information_permutation_model_upper_bound, \
    expected_mutual_information_permutation_model
from explora.utilities.tools import merge_columns


def mutual_information_permutation(X, Y, with_cross_tab=False, contingency_table=None, return_correction_term=False,
                                   return_marginal_entropies=False):
    """
    The corrected estimator for mutual information I(X,Y) between two attribute sets X 
    and Y of Mandros et al. (KDD'2017) (corrects by subtracting the expected value
    of mutual information under the permutation model). It can be computed
    either using cross_tab from Pandas, or with numpy. A precomputed contingency table can be
    provided if it is available"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()

    if return_marginal_entropies:
        mi, entropy_X, entropy_Y = mutual_information_plugin(X, Y, with_cross_tab, contingency_table,
                                                             return_marginal_entropies)
    else:
        mi = mutual_information_plugin(X, Y, with_cross_tab, contingency_table)

    correction = expected_mutual_information_permutation_model(X, Y, contingency_table)

    if return_correction_term is False:
        if return_marginal_entropies is False:
            return mi - correction
        else:
            return mi - correction, entropy_X, entropy_Y
    else:
        if return_marginal_entropies is False:
            return mi - correction, correction
        else:
            return mi - correction, correction, entropy_X, entropy_Y

def fraction_of_information_permutation(X, Y, with_cross_tab=False, contingency_table=None,
                                        return_correction_term=False):
    """
    The corrected estimator for fraction of information F(X,Y) between two attribute sets X 
    and Y of Mandros et al. (KDD'2017) (corrects by subtracting the expected value
    of mutual information under the permutation model). It can be computed
    either using cross_tab from Pandas, or with numpy. A precomputed contingency table, 
    or Y entropy can be provided if it is available"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()

    if return_correction_term is False:
        mi, entropy_X, entropy_Y = mutual_information_permutation(X, Y, with_cross_tab, contingency_table,
                                                                  return_marginal_entropies=True)
        if entropy_Y == 0:
            return 0
        else:
            return mi/entropy_Y
    else:
        mi, correction, entropy_X, entropy_Y = mutual_information_permutation(X, Y, with_cross_tab, contingency_table,
                                                                  return_correction_term = True,
                                                                  return_marginal_entropies=True)
        if entropy_Y == 0:
            return 0,0
        else:
            return mi/entropy_Y, correction

def mutual_information_permutation_upper_bound(X, Y, with_cross_tab=False, contingency_table=None,
                                               return_correction_term=False, return_marginal_entropies=False):
    """
    The plugin estimator for mutual information I(X,Y) between two attribute sets X 
    and Y corrected with an upper bound of the permutation model. It can be computed
    either using cross_tab from Pandas, or with numpy. A precomputed contingency table can be
    provided if it is available"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()

    if return_marginal_entropies:
        mi, entropy_X, entropy_Y = mutual_information_plugin(X, Y, with_cross_tab, contingency_table,
                                                             return_marginal_entropies)
    else:
        mi = mutual_information_plugin(X, Y, with_cross_tab, contingency_table)

    correction = expected_mutual_information_permutation_model_upper_bound(X, Y, contingency_table)
    if return_correction_term is False:
        if return_marginal_entropies is False:
            return mi - correction
        else:
            return mi - correction, entropy_X, entropy_Y
    else:
        if return_marginal_entropies is False:
            return mi - correction, correction
        else:
            return mi - correction, correction, entropy_X, entropy_Y



def fraction_of_information_permutation_upper_bound(X, Y, with_cross_tab=False, contingency_table=None,
                                                    return_correction_term=False):
    """
    The plugin estimator for fraction of information F(X,Y) between two attribute sets X 
    and Y corrected with an upper bound of the permutation nodel. It can be computed
    either using cross_tab from Pandas, or with numpy. A precomputed contingency table, 
    or Y entropy can be provided if it is available"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()

    if return_correction_term is False:
        mi, entropy_X, entropy_Y = mutual_information_permutation_upper_bound(X, Y, with_cross_tab, contingency_table,
                                                                  return_marginal_entropies=True)
        if entropy_Y == 0:
            return 0
        else:
            return mi / entropy_Y
    else:
        mi, correction, entropy_X, entropy_Y = mutual_information_permutation_upper_bound(X, Y, with_cross_tab,
                                                                                          contingency_table,
                                                                              return_correction_term=True,
                                                                              return_marginal_entropies=True)
        if entropy_Y == 0:
            return 0,0
        else:
            return mi / entropy_Y, correction


def mutual_information_chi_square(X, Y, with_cross_tab=False, contingency_table=None,
                                               return_correction_term=False, return_marginal_entropies=False):
    """
    The plugin estimator for mutual information I(X,Y) between two attribute sets X
    and Y corrected with the critical value of the chi square (Nguyen et al., 2014). It can be computed
    either using cross_tab from Pandas, or with numpy. A precomputed contingency table can be
    provided if it is available"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()


    prod_column_X=1
    n_dim_X=X.ndim
    if n_dim_X >1:
        for i in range(np.shape(X)[1]):
            domain=np.unique(X[:, i])
            prod_column_X=prod_column_X*(domain.size-1)
    else:
        domain = np.unique(X)
        prod_column_X = prod_column_X * (domain.size - 1)

    prod_column_Y = 1
    n_dim_Y = Y.ndim
    if n_dim_Y > 1:
        num_samples=np.shape(Y)[0]
        for i in range(np.shape(Y)[1]):
            domain = np.unique(Y[:, i])
            prod_column_Y = prod_column_Y * (domain.size - 1)
    else:
        num_samples = Y.size
        domain = np.unique(Y)
        prod_column_Y = prod_column_Y * (domain.size - 1)

    if return_marginal_entropies:
        mi, entropy_X, entropy_Y = mutual_information_plugin(X, Y, with_cross_tab, contingency_table,
                                                             return_marginal_entropies)
    else:
        mi = mutual_information_plugin(X, Y, with_cross_tab, contingency_table)

    dof = prod_column_X * prod_column_Y
    if dof == 0:
        dof=2
    crit = chi2.ppf(0.99999995, df=dof)
    correction = crit/(2*num_samples)

    if return_correction_term is False:
        if return_marginal_entropies is False:
            return mi - correction
        else:
            return mi - correction, entropy_X, entropy_Y
    else:
        if return_marginal_entropies is False:
            return mi - correction, correction
        else:
            return mi - correction, correction, entropy_X, entropy_Y


def fraction_of_information_chi_square(X, Y, with_cross_tab=False, contingency_table=None,
                                                    return_correction_term=False):
    """
    The plugin estimator for fraction of information F(X,Y) between two attribute sets X
    and Y corrected with the critical value of the chi square (Nguyen et al., 2014). It can be computed
    either using cross_tab from Pandas, or with numpy. A precomputed contingency table,
    or Y entropy can be provided if it is available"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):
        Y = Y.to_numpy()

    if return_correction_term is False:
        mi, entropy_X, entropy_Y = mutual_information_chi_square(X, Y, with_cross_tab, contingency_table,
                                                                  return_marginal_entropies=True)
        if entropy_Y == 0:
            return 0
        else:
            return mi / entropy_Y
    else:
        mi, correction, entropy_X, entropy_Y = mutual_information_chi_square(X, Y, with_cross_tab,
                                                                                          contingency_table,
                                                                              return_correction_term=True,
                                                                              return_marginal_entropies=True)
        if entropy_Y == 0:
            return 0,0
        else:
            return mi / entropy_Y, correction

def conditional_mutual_information(mutual_information_estimator, X, Y, Z):
    """
    The corrected estimator for the conditional fraction
    of information F(XY|Z) between two attribute sets X and Y, given Z. It needs an estimator of mutual information"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X = X.to_numpy()
        Y = Y.to_numpy()
        Z = Z.to_numpy()

    Z = merge_columns(Z)
    Y = merge_columns(Y)
    X = merge_columns(X)

    length = np.size(X, 0)

    uniqueValuesZ = np.unique(Z)
    indices = [np.where(Z == value)[0] for value in uniqueValuesZ]

    probs = [len(valueIndices) / length for valueIndices in indices]
    result = sum(
        [probs[i] * mutual_information_estimator(X[indices[i]], Y[indices[i]]) for i in range(len(uniqueValuesZ))])

    return result


# def conditional_fraction_of_information_permutation_upper_bound(X, Y, Z):
#     """
#     The plugin estimator for the conditional fraction of information F(XY|Z) between two attribute sets X
#     and Y,  given Z, and corrected with an upper bound of the permutation nodel."""
#     if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
#         X = X.to_numpy()
#         Y = Y.to_numpy()
#         Z = Z.to_numpy()
#
#     Z = merge_columns(Z)
#     Y = merge_columns(Y)
#     X = merge_columns(X)
#
#     length = np.size(X, 0)
#     condEntropy = conditional_entropy_permutation_upper_bound(Z, Y)
#
#     uniqueValuesZ = np.unique(Z)
#     indices = [np.where(Z == value)[0] for value in uniqueValuesZ]
#
#     probs = [len(valueIndices) / length for valueIndices in indices]
#     result = sum([probs[i] * mutual_information_permutation_upper_bound(X[indices[i]], Y[indices[i]]) for i in
#                   range(len(uniqueValuesZ))])
#
#     return result / condEntropy
#
#
# def conditional_entropy_permutation_upper_bound(Z, Y):
#     """
#     The plugin estimator for the conditional entropy H(Y|Z), corrected with an
#     upper bound of the permutation nodel."""
#     if isinstance(Z, pd.Series) or isinstance(Z, pd.DataFrame):
#         Y = Y.to_numpy()
#         Z = Z.to_numpy()
#
#     entropy_Y = entropy_plugin(Y)
#     expected_mi = expected_mutual_information_permutation_model_upper_bound(Z, Y)
#     return entropy_Y - expected_mi
#
#
# def conditional_entropy_permutation(Z, Y):
#     """
#     The corrected estimator for the conditional entropy H(Y|Z), corrected with
#     the permutation nodel."""
#     if isinstance(Z, pd.Series) or isinstance(Z, pd.DataFrame):
#         Y = Y.to_numpy()
#         Z = Z.to_numpy()
#
#     entropy_Y = entropy_plugin(Y)
#     expected_mi = expected_mutual_information_permutation_model(Z, Y)
#     return entropy_Y - expected_mi


# def main():
#
# if __name__ == '__main__':
#     main()
