#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:28:55 2020

@author: Panagiotis Mandros
"""

import pandas as pd
import numpy as np


def merge_columns(X):
    """ Combines multiple columns into one with resulting domain the distinct
    JOINT values of the input columns"""
    if is_pandas_df(X):
        num_columns = X.shape[1]
        if num_columns > 1:
            return X[X.columns].astype("str").agg("-".join, axis=1)
        else:
            return X
    if is_numpy(X):
        num_dim = X.ndim
        if num_dim == 2:
            return count_of_attributes(X)
        elif num_dim == 1:
            return X
    if is_pandas_series(X):
        return X
    raise TypeError("X must be either Numpy array or Pandas DataFrame/Series")


def append_two_arrays(X, Z):
    """ Appends X and Z horizontally """
    if X is None and Z is None:
        raise ValueError("Both arrays cannot be None")

    if Z is None:
        return X

    if X is None:
        return Z

    return np.column_stack((X, Z))


def append_and_merge(X, Y):
    """ Appends X and Y horizontally and then merges"""
    Z = append_two_arrays(X, Y)
    return merge_columns(Z)


def to_numpy_if_not(array):
    """ Returns the numpy representation if dataframe"""
    if is_pandas(array):
        return array.to_numpy()
    if is_numpy(array):
        return array
    raise TypeError("X must be either Numpy array or Pandas DataFrame/Series")


def number_of_columns(X):
    """ Returns the number of columns of X, taking into account different shapes"""
    if is_pandas_df(X):
        return X.shape[1]
    if is_numpy(X):
        num_dim = X.ndim
        if num_dim == 2:
            return np.size(X, 1)
        elif num_dim == 1:
            return 1
    if is_pandas_series(X):
        return 1
    raise TypeError("X must be either Numpy array or Pandas DataFrame/Series")


def get_column(X, i):
    """ Returns the i-th columns of X, taking into account different shapes"""
    if is_pandas_df(X):
        return X.iloc[i]
    if is_numpy(X):
        num_dim = X.ndim
        if num_dim == 2:
            return X[:, i]
        elif num_dim == 1 and i == 0:
            return X
    if is_pandas_series(X):
        return 1
    raise TypeError("X must be either Numpy array or Pandas DataFrame/Series")


def size_from_contingency_table(contingency_table):
    size_ = contingency_table[-1, -1]
    return size_


def marginals_from_contingency_table(contingency_table):
    counts_X = contingency_table[:-1, -1]
    counts_Y = contingency_table[-1, :-1]
    return counts_X, counts_Y


def joint_counts_from_contingency_table(contingency_table):
    return contingency_table[:-1, :-1].flatten()


def size_and_counts_of_contingency_table(
    X, Y, return_joint_counts=False, with_cross_tab=False, contingency_table=None
):
    """
    Returns the size, and the marginal counts of X, Y, and XY (optionally)"""

    if contingency_table is not None or with_cross_tab:
        if with_cross_tab:
            X = merge_columns(to_numpy_if_not(X))
            Y = merge_columns(to_numpy_if_not(Y))
            contingency_table = pd.crosstab(X, Y, margins=True)
        else:
            contingency_table = to_numpy_if_not(contingency_table)
        size_ = size_from_contingency_table(contingency_table)
        marginal_X, marginal_Y = marginals_from_contingency_table(contingency_table)
        if return_joint_counts:
            joint_counts = joint_counts_from_contingency_table(contingency_table)
            return size_, marginal_X, marginal_Y, joint_counts
        return size_, marginal_X, marginal_Y

    else:
        X = merge_columns(to_numpy_if_not(X))
        Y = merge_columns(to_numpy_if_not(Y))

        size_ = size_of(X)
        marginal_X = count_of_attributes(X)
        marginal_Y = count_of_attributes(Y)
        if return_joint_counts:
            XY = append_two_arrays(X, Y)
            joint_counts = count_of_attributes(XY)
            return size_, marginal_X, marginal_Y, joint_counts
        return size_, marginal_X, marginal_Y


def is_numpy(obj) -> bool:
    """ Checks if given object is instance of Numpy Array """
    return isinstance(obj, np.ndarray)


def is_pandas_df(obj) -> bool:
    """ Checks if given object is instance of Pandas DataFrame"""
    return isinstance(obj, pd.DataFrame)


def is_pandas_series(obj) -> bool:
    """ Checks if given object is instance of Pandas Series"""
    return isinstance(obj, pd.DataFrame)


def is_pandas(obj) -> bool:
    """ Checks if given object is instance of Pandas Series or DataFrame"""
    return is_pandas_df(obj) or is_pandas_series(obj)


def size_of(array) -> int:
    """ Returns the size of an array """
    if is_pandas(array):
        return len(array.index)
    if is_numpy(array):
        return np.size(array, 0)
    raise TypeError("X must be either Numpy array or Pandas dataframe")


def count_of_attributes(array) -> int:
    """ Returns the value counts of an array """
    if is_pandas(array):
        return array.value_counts()
    if is_numpy(array):
        return np.unique(array, return_counts=True, axis=0)[1]
    raise TypeError("X must be either Numpy array or Pandas DataFrame")


def size_and_counts_of_attribute(X):
    """ Returns the size, and the value counts of X """
    X = merge_columns(X)
    return size_of(X), count_of_attributes(X)


def concatenate_columns():
    """
    Should concatenate arbitrary number of numpy arrays into one (column-wise)
    like appending one to the other (similar but better to np.column_stack)"""
