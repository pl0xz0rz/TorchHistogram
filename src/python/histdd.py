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

def histogramdd(sample,bins=10,device=None,weights=None,ranges=None):
    D = sample.size(0)
    try:
        M = bins.size(0)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except AttributeError:
        # bins is an integer
        bins = torch.full([D],bins,dtype=int,device=device)
    if bins.dim() == 2:
        k = searchsorted(bins,sample)
        multiindex = (bins.size(1)+1)**(torch.arange(D,0,-1,device=device)-1)
        l = torch.sum(k*multiindex.reshape(-1,1),0)
        hist = torch.bincount(l,minlength=(bins.size(1)+1)**bins.size(0),weights=weights)
        hist = hist.reshape(tuple(torch.full([D],bins.size(1)+1,dtype=int,device=device)))
    else:
        if ranges == None:
            ranges = torch.cat((torch.min(sample,1)[0],torch.max(sample,1)[0])).reshape((2,-1))
        tranges = torch.empty_like(ranges)
        tranges[0,:] = -ranges[0,:]
        tranges[1,:] = bins/(ranges[1,:]+tranges[0,:])
        multiindex = torch.flip(torch.cumprod(torch.flip(bins,[0])+1,-1)/(bins[-1]+1),[0]).long()
        k = torch.addcmul(tranges[0,:].reshape(-1,1),sample,tranges[1,:].reshape(-1,1)).long()
        l = torch.sum(k*multiindex.reshape(-1,1),0)
        hist = torch.bincount(l,minlength=(multiindex[0]*(bins[-1]+1)).item(),weights=weights)
        hist = hist.reshape(tuple(bins+1))
    """
    m,index = l.sort()
    r = torch.arange((bins.size(1)+1)**bins.size(0),device=device)
    hist = searchsorted(m.reshape(1,-1),r.reshape(1,-1),side='right')
    hist[0,1:]=hist[0,1:]-hist[0,:-1]
    hist = hist.reshape(tuple(torch.full([bins.size(0)],bins.size(1)+1,dtype=int,device=device)))
    """

    return hist