#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:11:42 2020

@author: Panagiotis Mandros
"""

from collections import defaultdict

import numpy as np
import pandas as pd


def context_tree(sequence, markov_order=1):
    """
    Build the context tree of a sequence up to markov_order
    """
    if isinstance(sequence, pd.Series):
        sequence = sequence.to_numpy()
                
    assert np.size(sequence)!=0, 'Sequence is empty'
        
    markov_order = min(markov_order, np.size(sequence))

    window = tuple(sequence[:markov_order])
    contexts = defaultdict(lambda: defaultdict(int))
    contexts['markov_order'] = markov_order

    for i in range(markov_order,  np.size(sequence)):
        new_value = sequence[i]
        
        contexts[None][new_value] += 1
        contexts[window][new_value] += 1

        for j in range(1, len(window)):
            sub_window = window[j:]
            contexts[sub_window][new_value] += 1

        if len(window) + 1 >= markov_order:
            window = (*window[1:], new_value)
        else:
            window = (*window, new_value)

    return contexts
                              
if __name__ == '__main__':   
    seq=[1,0,1,0,1,1,0,1,0]

    # test 1
    contexts=context_tree(seq,2)
    solutionContectTree = defaultdict(lambda: defaultdict(int))
    solutionContectTree[None]={1:4,0:3}
    solutionContectTree['markov_order']=2
    solutionContectTree[(1,0)]={1:3}
    solutionContectTree[(0,)]={1:3}
    solutionContectTree[(0,1)]={0:2,1:1}
    solutionContectTree[(1,)]={0:3,1:1}
    solutionContectTree[(1,1)]={0:1}
    assert(contexts==solutionContectTree)
    


 
    
    




