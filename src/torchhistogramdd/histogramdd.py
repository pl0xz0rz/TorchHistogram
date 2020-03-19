# -*- coding: utf-8 -*-
"""

hsgh
"""

import torch
from torchsearchsorted import searchsorted

def histogramdd(sample,bins=None,ranges=None,weights=None,edges=None,device=None):
    custom_edges = False
    D = sample.size(0)
    if device == None:
        device = sample.device
    if bins == None:
        if edges == None:
            bins = 10
            custom_edges = False
        else:
            try:
                bins = edges.size(1)-1
            except AttributeError:
                bins = torch.empty(D)
                for i in range(len(edges)):
                    bins[i] = edges[i].size(0)-1
                bins = bins.to(device)
            custom_edges = True
    try:
        M = bins.size(0)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except AttributeError:
        # bins is either an integer or a list
        if type(bins) == int:
            bins = torch.full([D],bins,dtype=torch.long,device=device)
        elif torch.is_tensor(bins[0]):
            custom_edges = True
            edges = bins
            bins = torch.empty(D,dtype=torch.long)
            for i in range(len(edges)):
                bins[i] = edges[i].size(0)-1
            bins = bins.to(device)
        else:
            bins = torch.as_tensor(bins)
    if bins.dim() == 2:
        custom_edges = True
        edges = bins
        bins = torch.full([D],bins.size(1)-1,dtype=torch.long,device=device)
    if custom_edges:
        if not torch.is_tensor(edges):
            m = max(i.size(0) for i in edges)
            tmp = torch.empty([D,m])
            for i in range(D):
                s = edges[i].size(0)
                tmp[i,:]=edges[i][-1]
                tmp[i,:s]=edges[i][:]
            edges = tmp.to(device)
        k = searchsorted(edges,sample)
    else:
        if ranges == None:
            ranges = torch.empty(2,D,device=device)
            ranges[0,:]=torch.min(sample,1)[0]
            ranges[1,:]=torch.max(sample,1)[0]
        tranges = torch.empty_like(ranges)
        tranges[1,:] = bins/(ranges[1,:]-ranges[0,:])
        tranges[0,:] = 1-ranges[0,:]*tranges[1,:]
        k = torch.addcmul(tranges[0,:].reshape(-1,1),sample,tranges[1,:].reshape(-1,1)).long() #Get the right index
        k = torch.max(k,torch.tensor(0,device=device)) #Underflow bin
        
    k = torch.min(k,(bins+1).reshape(-1,1))   
    multiindex = torch.ones_like(bins)
    multiindex[1:] = torch.cumprod(torch.flip(bins[1:],[0])+2,-1).long()
    multiindex = torch.flip(multiindex,[0])
    l = torch.sum(k*multiindex.reshape(-1,1),0)
    hist = torch.bincount(l,minlength=(multiindex[0]*(bins[0]+2)).item(),weights=weights)
    hist = hist.reshape(tuple(bins+2))
    """
    m,index = l.sort()
    r = torch.arange((bins.size(1)+1)**bins.size(0),device=device)
    hist = searchsorted(m.reshape(1,-1),r.reshape(1,-1),side='right')
    hist[0,1:]=hist[0,1:]-hist[0,:-1]
    hist = hist.reshape(tuple(torch.full([bins.size(0)],bins.size(1)+1,dtype=int,device=device)))
    """
    return hist