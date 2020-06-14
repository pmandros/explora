#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:37:54 2020

@author: Panagiotis Mandros
"""

import pandas as pd
import numpy as np
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