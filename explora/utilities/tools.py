#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:28:55 2020

@author: Panagiotis Mandros
"""

import numpy as np
import pandas as pd


def merge_columns(X):
    """ Combines multiple columns into one with resulting domain the distinct JOINT values of the input columns"""
    if isinstance(X, pd.DataFrame):
        num_columns=X.shape[1]
        if num_columns>1:
            return  X[X.columns].astype('str').agg('-'.join, axis=1)
        else:
            return X
    elif isinstance(X, np.ndarray):
        num_dim=X.ndim
        if num_dim==2:
            return np.unique(X,return_inverse=True,axis=0)[1]
        elif num_dim==1:
            return X
    elif isinstance(X,pd.Series):
        return X
  


def append_two_arrays(X,Z):
    """ Appends X and Z horizontally """
    if Z is None:
        return X
    
    if X is None:
        return Z
    
    if X is None and Z is None:
        raise ValueError('Both arrays cannot be None')
     
    return np.column_stack((X,Z))

def append_and_merge(X,Y):
    """ Appends X and Y horizontally and then merges"""
    Z=append_two_arrays(X,Y)
    return merge_columns(Z)
    
    
def to_numpy_if_not(X):
    """ Returns the numpy representation if dataframe"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):  
        X=X.to_numpy()
    return X
    
def number_of_columns(X):
    """ Returns the number of columns of X, taking into account different shapes"""
    if isinstance(X, pd.DataFrame):
        return X.shape[1]
    elif isinstance(X, np.ndarray):
        num_dim=X.ndim
        if num_dim==2:
            return np.size(X,1)
        elif num_dim==1:
            return 1
    elif isinstance(X,pd.Series):
        return 1
    
def get_column(X,i):
    """ Returns the i-th columns of X, taking into account different shapes"""
    if isinstance(X, pd.DataFrame):
        return X.iloc[i]
    elif isinstance(X, np.ndarray):
        num_dim=X.ndim
        if num_dim==2:
            return X[:,i]
        elif num_dim==1 and i==0:
            return X
    elif isinstance(X,pd.Series):
        return 1

    
def size_and_counts_of_contingency_table(X,Y,return_joint_counts=False,with_cross_tab=False,contingency_table=None):
    """
    Returns the size, and the marginal counts of X, Y, and XY (optionally)"""
     
    if contingency_table!=None:        
        contingency_table=to_numpy_if_not(contingency_table)
        size=contingency_table[-1,-1]
        marginal_counts_Y=contingency_table[-1,:-1]
        marginal_counts_X=contingency_table[:-1,-1]     
        if return_joint_counts:
            joint_counts=contingency_table[:-1,:-1].flatten()        
    else:
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):  
            X=X.to_numpy()
        
        if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):  
            Y=Y.to_numpy()
        
        X=merge_columns(X)
        Y=merge_columns(Y)
        
        if with_cross_tab==True:         
            contingency_table=pd.crosstab(X,Y,margins = True)
            contingency_table=to_numpy_if_not(contingency_table)
            size=contingency_table[-1,-1]
            marginal_counts_Y=contingency_table[-1,:-1]
            marginal_counts_X=contingency_table[:-1,-1]
            if return_joint_counts:
                joint_counts=contingency_table[:-1,:-1].flatten()
        else:
            size=np.size(X,0)
            marginal_counts_X=np.unique(X, return_counts=True, axis=0)[1]
            marginal_counts_Y=np.unique(Y, return_counts=True, axis=0)[1] 
            if return_joint_counts:
                XY=append_two_arrays(X,Y)
                joint_counts=np.unique(XY, return_counts=True, axis=0)[1]
                
    if return_joint_counts:
        return size, marginal_counts_X, marginal_counts_Y, joint_counts
    else:
        return size, marginal_counts_X, marginal_counts_Y
    
    
def size_and_counts_of_attribute(X):
    """
    Returns the size, and the value counts of X"""            
    X=merge_columns(X)
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        counts=X.value_counts()
        length=len(X.index)
    elif isinstance(X, np.ndarray):
        counts=np.unique(X, return_counts=True,axis=0)[1]
        length=np.size(X,0)
             
    return length, counts
    


def concatenate_columns():
    """
    Should concatenate arbritrary number of numpy arrays into one (column-wise)
    like appending one to the other (similar but better to np.column_stack)"""


    

