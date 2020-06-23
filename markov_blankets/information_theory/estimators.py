#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:13:25 2020

@author: Panagiotis Mandros
"""

import pandas as pd;
import numpy as np;
import time
from information_theory.information_theory_basic import mutual_information_plugin, entropy_plugin
from permutation_model import expected__mutual_information_permutation_model_upper_bound, expected_mutual_information_permutation_model

from utilities.tools import merge_columns



def mutual_information_permutation(X,Y,with_cross_tab=False,contingency_table=None): 
    """
    The corrected estimator for mutual information I(X,Y) between two attribute sets X 
    and Y of Mandros et al. (KDD'2017) (corrects by subtracting the expected value
    of mutual information under the permutation model). It can be computed
    either using cross_tab from Pandas, or with numpy. A precomputed contingency table can be
    provided if it is available"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):  
        X=X.to_numpy();
        
    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):  
        Y=Y.to_numpy();
       
            
    mi=mutual_information_plugin(X,Y,with_cross_tab,contingency_table);
    correction=expected_mutual_information_permutation_model(X,Y,contingency_table)
    return mi-correction;

def fraction_of_information_permutation(X,Y,with_cross_tab=False,contingency_table=None, entropy_Y=None):
    """
    The corrected estimator for fraction of information F(X,Y) between two attribute sets X 
    and Y of Mandros et al. (KDD'2017) (corrects by subtracting the expected value
    of mutual information under the permutation model). It can be computed
    either using cross_tab from Pandas, or with numpy. A precomputed contingency table, 
    or Y entropy can be provided if it is available"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):  
        X=X.to_numpy();
        
    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):  
        Y=Y.to_numpy();        
       
        
    corMi=mutual_information_permutation(X,Y,with_cross_tab,contingency_table);
    if entropy_Y==None:
        entropy_Y=entropy_plugin(Y)          
    return corMi/entropy_Y; 


def mutual_information_permutation_upper_bound(X,Y,with_cross_tab=False,contingency_table=None): 
    """
    The plugin estimator for mutual information I(X,Y) between two attribute sets X 
    and Y corrected with an upper bound of the permutation nodel. It can be computed
    either using cross_tab from Pandas, or with numpy. A precomputed contingency table can be
    provided if it is available"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):  
        X=X.to_numpy();
        
    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):  
        Y=Y.to_numpy();
       
            
    mi=mutual_information_plugin(X,Y,with_cross_tab,contingency_table);
    correction=expected__mutual_information_permutation_model_upper_bound(X,Y,contingency_table)
    return mi-correction;


def fraction_of_information_permutation_upper_bound(X,Y,with_cross_tab=False,contingency_table=None,entropy_Y=None):
    """
    The plugin estimator for fraction of information F(X,Y) between two attribute sets X 
    and Y corrected with an upper bound of the permutation nodel. It can be computed
    either using cross_tab from Pandas, or with numpy. A precomputed contingency table, 
    or Y entropy can be provided if it is available"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):  
        X=X.to_numpy();
        
    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):  
        Y=Y.to_numpy();        
       
        
    corMi=mutual_information_permutation_upper_bound(X,Y,with_cross_tab,contingency_table);
    if entropy_Y==None:
        entropy_Y=entropy_plugin(Y)    
    return corMi/entropy_Y; 


def conditional_fraction_of_information_permutation(X,Y,Z):
    """
    The Mandros et al. (KDD'2017) corrected estimator for the conditional fraction 
    of information F(X;Y|Z) between two attribute sets X and Y, given Z."""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X=X.to_numpy();
        Y=Y.to_numpy();
        Z=Z.to_numpy();
        
    Z=merge_columns(Z);
    Y=merge_columns(Y);
    X=merge_columns(X);
   
    length=np.size(X,0);
    condEntropy=conditional_entropy_permutation(Z,Y);

    uniqueValuesZ=np.unique(Z);
    indices=[np.where(Z == value)[0] for value in uniqueValuesZ]

    probs=[len(valueIndices)/length for valueIndices in indices];
    result=sum([probs[i]*mutual_information_permutation(X[indices[i]],Y[indices[i]] ) for i in range(len(uniqueValuesZ))   ])   
    
    return result/condEntropy;

def conditional_fraction_of_information_permutation_upper_bound(X,Y,Z):
    """
    The plugin estimator for the conditional fraction of information F(X;Y|Z) between two attribute sets X 
    and Y,  given Z, and corrected with an upper bound of the permutation nodel."""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        X=X.to_numpy();
        Y=Y.to_numpy();
        Z=Z.to_numpy();
        
    Z=merge_columns(Z);
    Y=merge_columns(Y);
    X=merge_columns(X);
   
    length=np.size(X,0);
    condEntropy=conditional_entropy_permutation_upper_bound(Z,Y);

    uniqueValuesZ=np.unique(Z);
    indices=[np.where(Z == value)[0] for value in uniqueValuesZ]

    probs=[len(valueIndices)/length for valueIndices in indices];
    result=sum([probs[i]*mutual_information_permutation_upper_bound(X[indices[i]],Y[indices[i]] ) for i in range(len(uniqueValuesZ))   ])   
    
    return result/condEntropy;  

def conditional_entropy_permutation_upper_bound(Z,Y):
    """
    The plugin estimator for the conditional entropy H(Y|Z), corrected with an 
    upper bound of the permutation nodel."""
    if isinstance(Z, pd.Series) or isinstance(Z, pd.DataFrame):
        Y=Y.to_numpy();
        Z=Z.to_numpy();
    
    entropyY=entropy_plugin(Y);
    expectedMutual=expected__mutual_information_permutation_model_upper_bound(Z, Y);      
    return entropyY-expectedMutual;  

def conditional_entropy_permutation(Z,Y):
    """
    The corrected estimator for the conditional entropy H(Y|Z), corrected with
    the permutation nodel."""
    if isinstance(Z, pd.Series) or isinstance(Z, pd.DataFrame):
        Y=Y.to_numpy();
        Z=Z.to_numpy();
    
    entropyY=entropy_plugin(Y);
    expectedMutual=expected_mutual_information_permutation_model(Z, Y);      
    return entropyY-expectedMutual;  


def main():
    data = pd.read_csv("../datasets/tic_tac_toe.csv");
    biggerData=data.append(data).append(data).append(data).append(data).append(data).append(data)
    biggerBiggerData=biggerData.append(biggerData).append(biggerData).append(biggerData).append(biggerData).append(biggerData)
    biggerBiggerData=biggerBiggerData.append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData)
    bigSize=biggerBiggerData.size

    
    print(f'Testing permutation upper-bound')
    # test 1 with permutation upper-bound correction
    start_time_pandas_cross_tab = time.time();  
    mutInfoCrossTab=mutual_information_permutation_upper_bound(data.iloc[:,[0, 2,4,6,8]],data.iloc[:,9],with_cross_tab=True )
    print("--- %s seconds for upper-bound MI calcualtion small data with cross tab pandas ---" % (time.time() - start_time_pandas_cross_tab))


    start_time_numpy = time.time();   
    mutInfoNumpy=mutual_information_permutation_upper_bound(data.iloc[:,[0, 2,4,6,8]],data.iloc[:,9],with_cross_tab=False )
    print("--- %s seconds for upper-bound MI calcualtion small data in numpy ---" % (time.time() - start_time_numpy))
    assert(mutInfoCrossTab==mutInfoNumpy==0.3301286715561232)
    
    
    start_time_numpy = time.time();  
    fracInfo=fraction_of_information_permutation_upper_bound(data.iloc[:,[0, 2,4,6,8]],data.iloc[:,9],with_cross_tab=False )
    print("--- %s seconds for upper bound FI calcualtion small data numpy ---" % (time.time() - start_time_numpy))
    assert(fracInfo==0.3546133068140572)
    
    
    print(f'  Now moving to the big data with mumber of rows: {bigSize}')
    start_time_pandas_cross_tab = time.time();  
    mutInfoCrossTab=mutual_information_permutation_upper_bound(biggerBiggerData.iloc[:,[0, 2,4,6,8]],biggerBiggerData.iloc[:,9],with_cross_tab=True )
    print("--- %s seconds for upper-bound MI calcualtion big data with cross tab pandas ---" % (time.time() - start_time_pandas_cross_tab))


    start_time_numpy = time.time();   
    mutInfoNumpy=mutual_information_permutation_upper_bound(biggerBiggerData.iloc[:,[0, 2,4,6,8]],biggerBiggerData.iloc[:,9],with_cross_tab=False )
    print("--- %s seconds for upper-bound MI calcualtion big data in numpy ---" % (time.time() - start_time_numpy))
    assert(mutInfoCrossTab==mutInfoNumpy)

    
    
    start_time_numpy = time.time();  
    fracInfo=fraction_of_information_permutation_upper_bound(biggerBiggerData.iloc[:,[0, 2,4,6,8]],biggerBiggerData.iloc[:,9],with_cross_tab=False )
    print("--- %s seconds for upper bound FI calcualtion big data numpy ---" % (time.time() - start_time_numpy))
    
    print(f'Testing permutation ')
    # test 2 with permutation correction
    start_time_pandas_cross_tab = time.time();  
    mutInfoCrossTab=mutual_information_permutation(data.iloc[:,[0, 2,4,6,8]],data.iloc[:,9],with_cross_tab=True )
    print("--- %s seconds for permutation MI calcualtion small data with cross tab pandas ---" % (time.time() - start_time_pandas_cross_tab))

    start_time_numpy = time.time();   
    mutInfoNumpy=mutual_information_permutation(data.iloc[:,[0, 2,4,6,8]],data.iloc[:,9],with_cross_tab=False )
    print("--- %s seconds for permutation MI calcualtion small data in numpy ---" % (time.time() - start_time_numpy))
    assert(mutInfoCrossTab==mutInfoNumpy==0.41408554333826497)
    
    
    start_time_numpy = time.time();  
    fracInfo=fraction_of_information_permutation(data.iloc[:,[0, 2,4,6,8]],data.iloc[:,9],with_cross_tab=False )
    print("--- %s seconds for permutation FI calcualtion small data numpy ---" % (time.time() - start_time_numpy))
    assert(fracInfo==0.4447970033469641)

    
    print(f'  Now moving to the big data with mumber of rows: {bigSize}')
    start_time_pandas_cross_tab = time.time();  
    mutInfoCrossTab=mutual_information_permutation(biggerBiggerData.iloc[:,[0, 2,4,6,8]],biggerBiggerData.iloc[:,9],with_cross_tab=True )
    print("--- %s seconds for permutation MI calcualtion big data with cross tab pandas ---" % (time.time() - start_time_pandas_cross_tab))

    start_time_numpy = time.time();   
    mutInfoNumpy=mutual_information_permutation(biggerBiggerData.iloc[:,[0, 2,4,6,8]],biggerBiggerData.iloc[:,9],with_cross_tab=False )
    print("--- %s seconds for permutation MI calcualtion big data in numpy ---" % (time.time() - start_time_numpy))

    assert(mutInfoCrossTab==mutInfoNumpy)
     
    start_time_numpy = time.time();  
    fracInfo=fraction_of_information_permutation(biggerBiggerData.iloc[:,[0, 2,4,6,8]],biggerBiggerData.iloc[:,9],with_cross_tab=False )
    print("--- %s seconds for permutation FI calcualtion big data numpy ---" % (time.time() - start_time_numpy))
    
    print(f'Testing conditional permutation FI')
    # test 3 with conditional permutation correction
    start_time= time.time();  
    condFrac=conditional_fraction_of_information_permutation(data.iloc[:,[0, 2,6,8]],data.iloc[:,9],data.iloc[:,4])
    print("--- %s seconds for conditional corrected calcualtion small data with numpy ---" % (time.time() - start_time))
    assert(condFrac==0.35614476368567044)
    
    print(f'  Now moving to the big data with mumber of rows: {bigSize}')
    start_time= time.time();  
    condFrac=conditional_fraction_of_information_permutation(biggerBiggerData.iloc[:,[0, 2,6,8]],biggerBiggerData.iloc[:,9],biggerBiggerData.iloc[:,4])
    print("--- %s seconds for conditional corrected calcualtion big data with numpy ---" % (time.time() - start_time))
    

if __name__ == '__main__':
    main()   