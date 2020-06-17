#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:37:54 2020

@author: Panagiotis Mandros
"""

import pandas as pd
import numpy as np
import numba as nb

from utilities import tools 

def empirical_distribution_from_counts(counts,size=None):
    """
    Computes the empirical distribution of an attribute
    given the counts of its domain values (a.k.a distinct values)
    """
    if size==None:
        size=np.sum(counts);
        
    empirical_distribution=counts/size;
    assert np.isclose(np.sum(empirical_distribution), 1, rtol=1e-05, atol=1e-08, equal_nan=False), "Sum of empirical distibution should be 1";
    return empirical_distribution;

def empirical_statistics(X):     
    """
    Returns the empirical distribution (a.k.a relative frequencies), counts,
    and size of an attribute
    """
    X=tools.make_single_column(X);
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        counts=X.value_counts();
        length=len(X.index)
    elif isinstance(X, np.ndarray):
        counts=np.unique(X, return_counts=True,axis=0)[1];
        length=np.size(X,0);
        
    empirical_distribution=empirical_distribution_from_counts(counts)
    return empirical_distribution,len(counts),length;



@nb.jit(nopython=True)
def choose(n, r):
    """
    Computes n! / (r! (n-r)!) exactly. Returns a python int. For some reason it doesnt overflow
    """
    assert 0 <= r <= n

    c = 1
    for num, denom in zip(range(n, n - r, -1), range(1, r + 1, 1)):
        c = (c * num) // denom
    return c


# def choose(n, k):
#     """
#     A fast way to calculate binomial coefficients by Andrew Dalke (contrib). But it overflows
#     """
#     if 0 <= k <= n:
#         ntok = 1
#         ktok = 1
#         for t in range(1, min(k, n - k) + 1):
#             ntok *= n
#             ktok *= t
#             n -= 1
#         return ntok // ktok
#     else:
#         return 0


@nb.jit(nopython=True)
def hypergeometric_pmf(k, n, a, b):
    return choose(a, k) * choose(n - a, b - k) / choose(n, b)