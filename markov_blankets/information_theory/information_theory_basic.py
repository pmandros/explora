#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:29:50 2020

@author: Panagiotis Mandros
"""

import pandas as pd;
import numpy as np;
import scipy.stats as sc;
import time
from some_statistics.basic_statistics import empirical_statistics
from  utilities.tools import make_single_column, return_size_and_counts_of_contingency_table

def entropy(prob_vector):
    """
    Computes the Shannon entropy of an attribute   """
    return sc.entropy(prob_vector,base=2);


def entropy_plugin(X,return_statistics=False):
    """
    The plugin estimator for Shannon entropy H(X) of an attribute X. Can optionally 
    return the domain size and length of X"""
    empiricalDistribution,domainSize,length=empirical_statistics(X); 
    if return_statistics==True:
        return entropy(empiricalDistribution),domainSize,length
    else:
        return entropy(empiricalDistribution)

def mutual_information(prob_vector_X,prob_vector_Y,prob_vector_XY):
    """
    Computes the mutual information between two sets of random variables X and Y"""
    return entropy(prob_vector_X)+entropy(prob_vector_Y)-entropy(prob_vector_XY);

def mutual_information_plugin(X,Y,with_cross_tab=False,contingency_table=None): 
    """
    The plugin estimator for mutual information I(X;Y) between two attribute sets X 
    and Y. It can be computed either using cross_tab from Pandas, or with
    numpy. A precomputed contingency table can be provided if it is available"""
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):  
        X=X.to_numpy();
        
    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):  
        Y=Y.to_numpy();
          
    if with_cross_tab==True or contingency_table!=None:
        return mutual_information_from_cross_tab(X,Y,contingency_table=contingency_table);
    else:
        entropyX=entropy_plugin(X);
        entropy_Y=entropy_plugin(Y);
        dataXY=np.column_stack((X,Y));      
        entropyXY=entropy_plugin(dataXY);
        return entropyX + entropy_Y- entropyXY   
    

def fraction_of_information_plugin(X,Y,with_cross_tab=False,contingency_table=None,entropy_Y=None):
    """
    The plugin estimator for the fraction of information F(X;Y)=I(X,Y)/H(Y). 
    It can be computed either using cross_tab from Pandas, or with
    numpy. A precomputed contingency table and entropy of Y can be provided if
    it is available"""
    
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):  
        X=X.to_numpy();
        
    if isinstance(Y, pd.Series) or isinstance(Y, pd.DataFrame):  
        Y=Y.to_numpy();
          
    if with_cross_tab==True or contingency_table!=None:
        return fraction_of_information_from_cross_tab(X,Y,contingency_table
                                                         =contingency_table,entropy_Y=entropy_Y);
    else:       
        entropyX=entropy_plugin(X);
        
        if entropy_Y==None:
            entropy_Y=entropy_plugin(Y);
        dataXY=np.column_stack((X,Y));        
        entropyXY=entropy_plugin(dataXY);
        return (entropyX + entropy_Y- entropyXY)/entropy_Y
    
def mutual_information_from_cross_tab(X,Y,contingency_table=None):
    """
    Computes mutual information using cross_tab from pandas. A precomputed 
    contingency table can be provided if it is available """
    size,marginal_counts_X,marginal_counts_Y, joint_counts=return_size_and_counts_of_contingency_table(X,Y,return_joint_counts=True,
                                                                                                       with_cross_tab=True,contingency_table=contingency_table)

    return entropy(marginal_counts_X/size) + entropy(marginal_counts_Y/size)- entropy(joint_counts/size)
                

def conditional_entropy_plugin(Z,Y):
    """
    The plugin estimator for the conditional entropy H(Y|Z) of Y given Z"""
    if isinstance(Z, pd.Series) or isinstance(Z, pd.DataFrame):
        Y=Y.to_numpy();
        Z=Z.to_numpy();
        
    
    Z=make_single_column(Z);
    Y=make_single_column(Y);
    
    length=np.size(Z,0);
    uniqueValuesZ=np.unique(Z); 
    conditional_entropy_plugin=0;
    for value in uniqueValuesZ:
        indices = np.where(Z == value);          
        valueCount=len(indices);

        tempY=Y[indices[0]];
        entropyTempY=entropy_plugin(tempY);
        
        conditional_entropy_plugin=conditional_entropy_plugin+valueCount/length*entropyTempY;      
    return conditional_entropy_plugin;  
    
def fraction_of_information_from_cross_tab(X,Y,contingency_table=[],entropy_Y=None):
    """
    Computes fraction of information using cross_tab from pandas. A precomputed 
    contingency table can be provided if it is available """
    size,marginal_counts_X,marginal_counts_Y, joint_counts=return_size_and_counts_of_contingency_table(X,Y,return_joint_counts=True,
                                                                                                       with_cross_tab=True,contingency_table=contingency_table)
    if entropy_Y==None:
            entropy_Y=entropy(marginal_counts_Y/size);
    return (entropy(marginal_counts_X/size) + entropy_Y- entropy(joint_counts/size))/entropy_Y
    

    
   
def main():
    data = pd.read_csv("../datasets/tic_tac_toe.csv");
    biggerData=data.append(data).append(data).append(data).append(data).append(data).append(data)
    biggerBiggerData=biggerData.append(biggerData).append(biggerData).append(biggerData).append(biggerData).append(biggerData)
    biggerBiggerData=biggerBiggerData.append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData)
      
    start_time_pandas_cross_tab = time.time();  
    mutInfoCrossTab=mutual_information_plugin(data.iloc[:,[0, 2,4,6,8]],data.iloc[:,9],with_cross_tab=True )
    print("--- %s seconds for MI calcualtion small data with cross tab pandas ---" % (time.time() - start_time_pandas_cross_tab))

    start_time_numpy = time.time();   
    mutInfoNumpy=mutual_information_plugin(data.iloc[:,[0, 2,4,6,8]],data.iloc[:,9],with_cross_tab=False )
    print("--- %s seconds for MI calcualtion small data in numpy ---" % (time.time() - start_time_numpy))
    assert(mutInfoCrossTab==mutInfoNumpy==0.57241179990198)
    
    start_time_numpy = time.time();  
    fracInfo=fraction_of_information_plugin(data.iloc[:,[0, 2,4,6,8]],data.iloc[:,9],with_cross_tab=False )
    print("--- %s seconds for FI calcualtion small data numpy ---" % (time.time() - start_time_numpy))
    assert(fracInfo== 0.6148658347844208)

    bigSize=biggerBiggerData.size
    print(f'Number of rows in big data is: {bigSize}')
    start_time_pandas_cross_tab = time.time();  
    mutInfoCrossTab=mutual_information_plugin(biggerBiggerData.iloc[:,[0, 2,4,6,8]],biggerBiggerData.iloc[:,9],with_cross_tab=True )
    print("--- %s seconds for MI calcualtion big data with cross tab pandas ---" % (time.time() - start_time_pandas_cross_tab))

    start_time_numpy = time.time();   
    mutInfoNumpy=mutual_information_plugin(biggerBiggerData.iloc[:,[0, 2,4,6,8]],biggerBiggerData.iloc[:,9],with_cross_tab=False )
    print("--- %s seconds for MI calcualtion big data in numpy ---" % (time.time() - start_time_numpy))
    
    start_time_numpy = time.time();  
    fracInfo=fraction_of_information_plugin(biggerBiggerData.iloc[:,[0, 2,4,6,8]],biggerBiggerData.iloc[:,9],with_cross_tab=False )
    print("--- %s seconds for FI calcualtion big data numpy ---" % (time.time() - start_time_numpy))



if __name__ == '__main__':
    main()   
     

    

        


