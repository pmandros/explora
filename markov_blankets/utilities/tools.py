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
    
    
def to_numpy_if_not(X):
    """ Returns the numpy representation if dataframe"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):  
        X=X.to_numpy();
    return X
    
    
def size_and_counts_of_contingency_table(X,Y,return_joint_counts=False,with_cross_tab=False,contingency_table=None):
    """
    Returns the size, and the marginal counts of X, Y, and XY (optionally)"""
     
    if contingency_table!=None:        
        contingency_table=to_numpy_if_not(contingency_table);
        size=contingency_table[-1,-1];
        marginal_counts_Y=contingency_table[-1,:-1];
        marginal_counts_X=contingency_table[:-1,-1];     
        if return_joint_counts:
            joint_counts=contingency_table[:-1,:-1].flatten();        
    else:
        if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):  
            X=X.to_numpy();
        
        if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):  
            Y=Y.to_numpy();
        
        X=make_single_column(X);
        Y=make_single_column(Y);
        
        if with_cross_tab==True:         
            contingency_table=pd.crosstab(X,Y,margins = True);

            contingency_table=to_numpy_if_not(contingency_table);
            size=contingency_table[-1,-1];
            marginal_counts_Y=contingency_table[-1,:-1];
            marginal_counts_X=contingency_table[:-1,-1];
            if return_joint_counts:
                joint_counts=contingency_table[:-1,:-1].flatten();
        else:
            size=np.size(X,0);
            marginal_counts_X=np.unique(X, return_counts=True, axis=0)[1];
            marginal_counts_Y=np.unique(Y, return_counts=True, axis=0)[1]; 
            if return_joint_counts:
                XY=np.column_stack((X,Y));
                joint_counts=np.unique(XY, return_counts=True, axis=0)[1];
                
    if return_joint_counts:
        return size, marginal_counts_X, marginal_counts_Y, joint_counts
    else:
        return size, marginal_counts_X, marginal_counts_Y
    
    
def size_and_counts_of_attribute(X):
    """
    Returns the size, and the value counts of X"""            
    X=make_single_column(X);
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        counts=X.value_counts();
        length=len(X.index)
    elif isinstance(X, np.ndarray):
        counts=np.unique(X, return_counts=True,axis=0)[1];
        length=np.size(X,0);
             
    return length, counts
    


def concatenate_columns():
    """
    Should concatenate arbritrary number of numpy arrays into one (column-wise)
    like appending one to the other (similar but better to np.column_stack)"""


    

