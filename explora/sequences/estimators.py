#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:19:34 2020

@author: Panagiotis Mandros
"""

import numpy as np
import pandas as pd


def KT_estimator(symbol, history, alphabet_size=None, smoothing_amount=None):
    """
    The Krichevsky–Trofimov estimator. 
    """
    if isinstance(history, pd.Series):
        history = history.to_numpy()
        
    length=np.size(history)
    if length==0:
        return (smoothing_amount)/(alphabet_size*smoothing_amount)
    
    values,counts=np.unique(history, return_counts=True,axis=0)
    
    
    
    almost_symbol_value_index=np.where(values == symbol)[0]

    
    if len(almost_symbol_value_index)!=0:
        # if symbol exists in training sequence
        symbol_count=counts[almost_symbol_value_index[0]]
    else:      
        if not alphabet_size:
            alphabet_size=np.size(values)+1        
        symbol_count=0
    
    if not smoothing_amount:
        smoothing_amount=1/alphabet_size
        
 
    probability=(symbol_count+smoothing_amount)/(length+alphabet_size*smoothing_amount)
    return probability

def KT_estimator_block(block, alphabet_size=None, smoothing_amount=None):
    """
    The Krichevsky–Trofimov block estimate for a sequence . 
    """
    if isinstance(block, pd.Series):
        block = block.to_numpy()
    
    if not alphabet_size:
        alphabet_size=len(np.unique(block, return_counts=True,axis=0)[0])
      
    if not smoothing_amount:
        smoothing_amount=1/alphabet_size
        
    prob_sequences=[KT_estimator(block[i], block[0:i], alphabet_size=alphabet_size, smoothing_amount=smoothing_amount) for i in range(np.size(block)) ]
    
    return np.prod(prob_sequences)  


if __name__ == '__main__':   
        
    seq=[0,1,1,1,0]
    block_kt_estimate=KT_estimator_block(seq)
    assert(block_kt_estimate==3/256)