#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 03:10:56 2020

@author: Panagiotis Mandros
"""

from operator import itemgetter

import numpy as np
import pandas as pd


def shrink(conditional_estimator,grow_result,data,shrink_threshold=0,target=None):
    """
    For a dependency measure D(X;Y), it shrinks the grow_result G by calculating
    the conditional D(Z;Y|G-{Z}) for all Z in G and removing Z if bellow threshold
    """
    
    if isinstance(data, pd.DataFrame):  
        data=data.to_numpy();
    
    if target==None:
        targetIndex=np.size(data,1)-1;
    else:
        targetIndex=target-1;
    Y=data[:,targetIndex];

        
    while len(grow_result)>1:
        conditionalFIs=[evaluateCandidate(conditional_estimator,data,index,Y,grow_result) for index in grow_result]     
        print(conditionalFIs)                                  
        conditionalFIs.sort(key=itemgetter(0));
        worse=conditionalFIs[0];
        worseScore=worse[0];
        worseCandidateIndex=worse[1];
        
        if worseScore<=shrink_threshold+0.00005:
            grow_result.remove(worseCandidateIndex);
        else:
            return {x+1 for x in grow_result};
    # conditional_estimator(data[:,list(grow_result)],Y,[])
    return {x+1 for x in grow_result};  

def evaluateCandidate(estimator,data,candidateIndex,Y,selectedVariables):
    toCondition=set(selectedVariables);
    toCondition.remove(candidateIndex);

    return estimator(data[:,candidateIndex],Y, data[:,list(toCondition)]),candidateIndex;