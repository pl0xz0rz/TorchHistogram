# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:45:57 2020

@author: majoi
"""

import numpy as np
import torch
from torchhistogramdd import histogramdd

torch.random.manual_seed(19680801)

sample = torch.arange(0,1,.01)*torch.arange(1,4).reshape([-1,1]) % 1
histnp = np.histogramdd(sample.transpose(0,1))[0]
hist = histogramdd(sample)
print(hist[1:-1,1:-1,1:-1]-histnp)

sample = torch.rand(3,100000)
histnp = np.histogramdd(sample.transpose(0,1),6)[0]
hist = histogramdd(sample,6)
print(hist[1:-1,1:-1,1:-1]-histnp)

sample = torch.rand(3,100000)
histnp = np.histogramdd(sample.transpose(0,1),[4,5,6])[0]
hist = histogramdd(sample,[4,5,6])
print(hist[1:-1,1:-1,1:-1]-histnp)

sample = torch.rand(3,100000)
bins = [torch.rand([8]).sort()[0],torch.rand([7]).sort()[0],torch.rand([6]).sort()[0]]
histnp = np.histogramdd(sample.transpose(0,1),bins)[0]
hist = histogramdd(sample,bins)
print(hist[1:-1,1:-1,1:-1]-histnp)