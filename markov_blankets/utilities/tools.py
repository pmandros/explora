#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:28:55 2020

@author: Panagiotis Mandros
"""

import pandas as pd
import numpy as np

def make_single_column(X):
    """ Combines multiple columns into one with resulting domain the distinct JOINT values of the input columns"""
    if isinstance(X, pd.DataFrame):
        num_columns=X.shape[1];
        if num_columns>1:
            return  X[X.columns].astype('str').agg('-'.join, axis=1);
        else:
            return X;
    elif isinstance(X, np.ndarray):
        num_columns=X.ndim;
        if num_columns>1:
            return np.unique(X,return_inverse=True,axis=0)[1];
        else:
            return X;
    elif isinstance(X,pd.Series):
        return X;
    
    
def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib). But it overflows
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0
    
def choose_no_overflow(n,r):
  """
  Computes n! / (r! (n-r)!) exactly. Returns a python long int. For some reason it doesnt overflow
  """
  assert n >= 0
  assert 0 <= r <= n

  c = 1
  denom = 1
  for (num,denom) in zip(range(n,n-r,-1), range(1,r+1,1)):
    c = (c * num) // denom
  return c


def hypergeometric_pmf(k,n,a,b):
    return choose_no_overflow(a, k) * choose_no_overflow(n - a, b - k) / choose_no_overflow(n, b)