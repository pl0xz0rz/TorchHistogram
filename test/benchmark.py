# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:32:34 2020

@author: majoi
"""

import numpy as np
import torch
import timeit

def get_tensors(n,d=3,device=None):
    x = torch.rand((d,n),device=device)
    v = torch.arange(0.0,1.0,.1,device=device)*torch.ones((d,1),device=device)+.05
    return x,v

def get_arrays(n,d=3):
    x = np.random.rand(n,d)
    v = np.arange(0.0,1.0,.1)*np.ones((5,1))+.05
    return x,v
"""
cpu = timeit.repeat(
    stmt="histogramdd(sample,bins,device='cpu')",
    setup="sample,bins = get_tensors(1000000,device='cpu')",
    globals=globals(),
    repeat=20,
    number=1
)
print('CPU: ', min(cpu), sep='\t')
"""