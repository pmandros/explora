#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 03:34:55 2020

@author: mbk-97-53
"""

import sys
sys.path.append('..')
import numpy as np;
import pandas as pd
from optimization.greedy_search import greedy_search
from algorithms.shrink import shrink
import time
from information_theory.estimators import fraction_of_information_permutation_upper_bound,conditional_fraction_of_information_permutation_upper_bound, fraction_of_information_permutation, conditional_fraction_of_information_permutation


def grow_shrink(grow_estimator,shrink_estimator, data,  shrink_threshold=0,target=None,limit=None):
    """
    For a dependency measure D(X;Y) and
    conditional D(X;Y|Z), it greedily finds a maximizer for D(X;Y), and shrinks 
    afterwards with the conditional. Not to be confused with the Grow Shrink for Markov
    blankets (although similar)
    """

    if isinstance(data, pd.DataFrame):  
        data=data.to_numpy();
    
    if target==None:
        target=np.size(data,1);

    # start_time = time.time();  

    [greedyResult,greedyScore]=greedy_search(grow_estimator,data,target,limit=limit);
    # print("--- %s Time for grow---" % (time.time() - start_time))

    # start_time = time.time();  

    shrinkResults=shrink(shrink_estimator,greedyResult,data,shrink_threshold=shrink_threshold,target=None)
    # print("--- %s Time to shrink---" % (time.time() - start_time))
    return shrinkResults

    

def main():
    data = pd.read_csv("../datasets/tic_tac_toe.csv");
    biggerData=data.append(data).append(data).append(data).append(data).append(data).append(data)
    biggerBiggerData=biggerData.append(biggerData).append(biggerData).append(biggerData).append(biggerData).append(biggerData)
    biggerBiggerData=biggerBiggerData.append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData)
      
    # print(data.info())
    # print(biggerBiggerData.info())
    
    
    # checking with upper-bound FI
    selected=grow_shrink(fraction_of_information_permutation_upper_bound,
                         conditional_fraction_of_information_permutation_upper_bound,
                         data.to_numpy(),shrink_threshold=0)
    assert(selected=={4,0,8})
    
    selected=grow_shrink(fraction_of_information_permutation_upper_bound,
                         conditional_fraction_of_information_permutation_upper_bound,
                         data.to_numpy(),shrink_threshold=0.21)
    assert(selected=={4})
    
    
    # checking with upper-bound FI
    selected=grow_shrink(fraction_of_information_permutation,
                         conditional_fraction_of_information_permutation,
                         data.to_numpy(),shrink_threshold=0)
    assert(selected=={4,0,8,2,6})
    
    selected=grow_shrink(fraction_of_information_permutation,
                         conditional_fraction_of_information_permutation,
                         data.to_numpy(),shrink_threshold=0.2)
    assert(selected=={4,2,6})
    
    
    
    
    # performance with upper-bound corrected FI
    start_time=time.time()
    selected=grow_shrink(fraction_of_information_permutation_upper_bound,
                         conditional_fraction_of_information_permutation_upper_bound,
                         data.to_numpy(),shrink_threshold=0)
    print("--- %s seconds for grow shrink with upper-dound F1 on small data---" % (time.time() - start_time))
    
    start_time=time.time()
    selected=grow_shrink(fraction_of_information_permutation_upper_bound,
                         conditional_fraction_of_information_permutation_upper_bound,
                         biggerBiggerData.to_numpy(),shrink_threshold=0)
    print("--- %s seconds for grow shrink with upper-dound F1 on big data---" % (time.time() - start_time))

    
    

    # start_time=time.time()
    # [selected,bestScore]=greedy_search(information_theory_basic.fraction_of_information_plugin,data)
    # print("--- %s seconds for greedy search in small data with plugin FI ---" % (time.time() - start_time))
    # print(f' selected by plugin FI on small: {selected} with score {bestScore}')
    
    # start_time=time.time()
    # [selected,bestScore]=greedy_search(estimators.fraction_of_information_permutation_upper_bound,data)
    # print("--- %s seconds for greedy search in small data with upper-bound FI ---" % (time.time() - start_time))
    # assert(0.3083695032958582==bestScore)
    # print(f' selected by upper-bound FI on small: {selected} with score {bestScore}')
    
    
    
    
    # start_time=time.time()
    # [selected,bestScore]=greedy_search(information_theory_basic.fraction_of_information_plugin,biggerBiggerData)
    # print("--- %s seconds for greedy search in big data with plugin FI ---" % (time.time() - start_time))
    # print(f' selected by plugin FI on big: {selected}')
    
    # start_time=time.time()
    # [selected,bestScore]=greedy_search(estimators.fraction_of_information_permutation_upper_bound,biggerBiggerData)
    # print("--- %s seconds for greedy search in big data with upper-bound FI ---" % (time.time() - start_time))
    # print(f' selected by upper-bound FI on big: {selected}')

    


if __name__ == '__main__':
    main()   