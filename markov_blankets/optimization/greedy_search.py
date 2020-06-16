#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:45:52 2020

@author: Panagiotis Mandros
"""

import numpy as np 
import pandas as pd
from operator import itemgetter
from information_theory.estimators import mutual_information_permutation_upper_bound
from information_theory.information_theory_basic import mutual_information_plugin


import time

def greedy_search(estimator,data,target=None,limit=None,prior_solution_set=None):
    """
    Given data, it greedily maximizes an estimator of a dependency measure D(XY), over
    candidate attribute sets in X in the data. It selects the best candidate to expand 
    in a BFS manner. If no target index is provided (indexed from 1), the target is the last
    attribute. If a limit is provided, the search will stop at at level, e.g., for 2, the search
    wont continue after the second level (pairs of attributes). A prior_solution_set can be used to
    initialize the search.    
    """
    if isinstance(data, pd.DataFrame):  
        data=data.to_numpy();
        
    number_explanatory_variables=np.size(data,1)-1

    if target==None:
        target_index=number_explanatory_variables;
    else:
        target_index=target-1;
    Y=data[:,target_index]; 

    if limit==None:
        limit=number_explanatory_variables 
    
    setOfCandidates=set([i for i in range(number_explanatory_variables+1)]) 
    setOfCandidates.remove(target_index) 

    if prior_solution_set==True:
        setOfCandidates.difference_update(prior_solution_set) 
        
    
    selectedVariables=set() 
    bestScore=-1000 
    
    for i in range(limit):

        candidateScores=[evaluateCandidate(estimator,data,index,Y,selectedVariables) for index in setOfCandidates]    
                                   
        candidateScores.sort(key=itemgetter(0),reverse=True) 
        
        top=candidateScores[0] 
        topScore=top[0] 
        topCandidateIndex=top[1] 

        if topScore<=0 or topScore<=bestScore:
            return [selectedVariables,bestScore] 
       
        if topScore>bestScore:
            bestScore=topScore 
            selectedVariables.add(topCandidateIndex) 
            setOfCandidates.remove(topCandidateIndex) 
    return [selectedVariables,bestScore] 
        
    
def evaluateCandidate(estimator,data,candidateIndex,Y,alreadySelectedVariables):
    jointIndices=set(alreadySelectedVariables) 
    jointIndices.add(candidateIndex) 
    result=estimator(data[:,list(jointIndices)],Y),candidateIndex 

    return result
    
    
def main():
    data = pd.read_csv("../datasets/tic_tac_toe.csv");
    biggerData=data.append(data).append(data).append(data).append(data).append(data).append(data)
    biggerBiggerData=biggerData.append(biggerData).append(biggerData).append(biggerData).append(biggerData).append(biggerData)
    biggerBiggerData=biggerBiggerData.append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData).append(biggerBiggerData)
      
    # print(data.info())
    # print(biggerBiggerData.info())
    
    start_time=time.time()
    [selected,bestScore]=greedy_search(mutual_information_plugin,data)
    print("--- %s seconds for greedy search in small data with plugin MI ---" % (time.time() - start_time))
    print(f' selected by plugin MI on small: {selected} with score {bestScore}')
    
    start_time=time.time()
    [selected,bestScore]=greedy_search(mutual_information_permutation_upper_bound,data)
    print("--- %s seconds for greedy search in small data with upper-bound MI ---" % (time.time() - start_time))
    print(f' selected by upper-bound MI on small: {selected} with score {bestScore}')
    assert(0.28707781833145635==bestScore)


    start_time=time.time()
    [selected,bestScore]=greedy_search(mutual_information_plugin,data)
    print("--- %s seconds for greedy search in small data with plugin FI ---" % (time.time() - start_time))
    print(f' selected by plugin FI on small: {selected} with score {bestScore}')
    
    start_time=time.time()
    [selected,bestScore]=greedy_search(mutual_information_permutation_upper_bound,data)
    print("--- %s seconds for greedy search in small data with upper-bound FI ---" % (time.time() - start_time))
    assert(0.3083695032958582==bestScore)
    print(f' selected by upper-bound FI on small: {selected} with score {bestScore}')
    
    
    
    
    # start_time=time.time()
    # [selected,bestScore]=greedy_search(mutual_information_plugin,biggerBiggerData)
    # print("--- %s seconds for greedy search in big data with plugin FI ---" % (time.time() - start_time))
    # print(f' selected by plugin FI on big: {selected}')
    
    # start_time=time.time()
    # [selected,bestScore]=greedy_search(mutual_information_permutation_upper_bound,biggerBiggerData)
    # print("--- %s seconds for greedy search in big data with upper-bound FI ---" % (time.time() - start_time))
    # print(f' selected by upper-bound FI on big: {selected}')

    



if __name__ == '__main__':
    main()   