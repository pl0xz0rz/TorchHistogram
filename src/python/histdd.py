# -*- coding: utf-8 -*-
"""

hsgh
"""

import torch
from torchsearchsorted import searchsorted
"""
if torch.cuda.is_available():
    device = torch.device("cuda")

x = torch.rand((3,1000),device=device)
v = torch.arange(0.0,1.0,.2,device=device)*torch.ones((3,1),device=device)+.1

k = searchsorted(v,x)
multiindex = torch.tensor([1,6,36],device=device)
l = torch.sum(k.reshape(1000,3)*multiindex,1)
m,index = l.sort()
r = torch.arange(216,device=device)
hist = searchsorted(m.reshape(1,-1),r.reshape(1,-1),side='right')
hist[0,1:]=hist[0,1:]-hist[0,:-1]
hist = hist.reshape(6,6,6)
"""

def histogramdd(sample,bins,device=None):
    D = sample.size(0)
    if bins == None:
        bins = 10
    
    try:
        M = bins.size(0)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except TypeError:
        # bins is an integer
        bins = D*[bins]
    k = searchsorted(bins,sample)
    multiindex = (bins.size(1)+1)**(torch.arange(D,0,-1,device=device)-1)
    l = torch.sum(k.transpose(0,1)*multiindex,1)
    """
    m,index = l.sort()
    r = torch.arange((bins.size(1)+1)**bins.size(0),device=device)
    hist = searchsorted(m.reshape(1,-1),r.reshape(1,-1),side='right')
    hist[0,1:]=hist[0,1:]-hist[0,:-1]
    hist = hist.reshape(tuple(torch.full([bins.size(0)],bins.size(1)+1,dtype=int,device=device)))
    """
    hist = torch.bincount(l,minlength=(bins.size(1)+1)**bins.size(0))
    hist = hist.reshape(tuple(torch.full([D],bins.size(1)+1,dtype=int,device=device)))
    return hist