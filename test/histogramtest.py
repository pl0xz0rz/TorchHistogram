# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:45:57 2020

@author: majoi
"""

import numpy as np
import torch

torch.random.manual_seed(19680801)

sample = torch.arange(0,1,.01)*torch.arange(1,4).reshape([-1,1]) % 1
histnp = np.histogramdd(sample.transpose(0,1))[0]
hist = histogramdd(sample)
print(hist[1:-1,1:-1,1:-1]-histnp)

sample = torch.rand(3,10000)
histnp = np.histogramdd(sample.transpose(0,1),6)[0]
hist = histogramdd(sample,6)
print(hist[1:-1,1:-1,1:-1]-histnp)