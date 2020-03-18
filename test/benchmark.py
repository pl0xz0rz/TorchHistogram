# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:32:34 2020

@author: majoi
"""

import numpy as np
import torch
import timeit
from matplotlib import pyplot as plt

torch.random.manual_seed(19680801)
torch.cuda.manual_seed_all(19680801)
np.random.seed(19680801)

def get_tensors(n,d=3,device=None):
    x = torch.rand((d,n),device=device)
    torch.cuda.synchronize()
    return x

def get_arrays(n,d=3):
    x = np.random.rand(n,d)
    return x

def get_tensors_edges(n,d=3,device=None):
    x = torch.rand((d,n),device=device)
    v = [torch.arange(0,1,.1,device=device)]*d
    torch.cuda.synchronize()
    return x,v

def get_arrays_edges(n,d=3):
    x = np.random.rand(n,d)
    v = [np.arange(0,1.1,.1)]*d
    return x,v

n = 100
time_cpu = torch.empty((4,6))
time_cuda = torch.empty((4,6))
time_numpy = torch.empty((4,6))

time_cpu_e = torch.empty((4,6))
time_cuda_e = torch.empty((4,6))
time_numpy_e = torch.empty((4,6))
for j in range(6):
    for i in range(3,7):
        print("n= ",n,"d= ",i)
        t = timeit.repeat(
                stmt="histogramdd(sample,device='cpu')",
                setup="sample = get_tensors(n,i,device='cpu')",
                globals=globals(),
                repeat=20,
                number=1
                )
        time_cpu[i-3,j] = min(t)
        print("CPU: ",min(t),sep='\t')
        t = timeit.repeat(
                stmt="histogramdd(sample,device='cuda')",
                setup="sample = get_tensors(n,i,device='cuda')",
                globals=globals(),
                repeat=20,
                number=1
                )
        time_cuda[i-3,j] = min(t)
        print("CUDA: ",min(t),sep='\t')        
        t = timeit.repeat(
                stmt="np.histogramdd(sample)",
                setup="sample = get_arrays(n,i)",
                globals=globals(),
                repeat=20,
                number=1
                )
        time_numpy[i-3,j] = min(t)
        print("Numpy: ",min(t),sep='\t')  
        t = timeit.repeat(
                stmt="histogramdd(sample,bins,device='cpu')",
                setup="sample,bins = get_tensors_edges(n,i,device='cpu')",
                globals=globals(),
                repeat=20,
                number=1
                )
        time_cpu_e[i-3,j] = min(t)
        print("CPU: ",min(t),sep='\t')
        t = timeit.repeat(
                stmt="histogramdd(sample,bins,device='cuda')",
                setup="sample,bins = get_tensors_edges(n,i,device='cuda')",
                globals=globals(),
                repeat=20,
                number=1
                )
        time_cuda_e[i-3,j] = min(t)
        print("CUDA: ",min(t),sep='\t')        
        t = timeit.repeat(
                stmt="np.histogramdd(sample,bins)",
                setup="sample,bins = get_arrays_edges(n,i)",
                globals=globals(),
                repeat=20,
                number=1
                )
        time_numpy_e[i-3,j] = min(t)
        print("Numpy: ",min(t),sep='\t')  
    n *= 10

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
ax1.set_title("d=3")
ax1.loglog([100,1000,10000,100000,1000000,10000000],time_cpu[0,:],label='CPU')
ax1.loglog([100,1000,10000,100000,1000000,10000000],time_cuda[0,:],label='CUDA')
ax1.loglog([100,1000,10000,100000,1000000,10000000],time_numpy[0,:],label='Numpy')
ax2.set_title("d=4")
ax2.loglog([100,1000,10000,100000,1000000,10000000],time_cpu[1,:],label='CPU')
ax2.loglog([100,1000,10000,100000,1000000,10000000],time_cuda[1,:],label='CUDA')
ax2.loglog([100,1000,10000,100000,1000000,10000000],time_numpy[1,:],label='Numpy')
ax3.set_title("d=5")
ax3.loglog([100,1000,10000,100000,1000000,10000000],time_cpu[2,:],label='CPU')
ax3.loglog([100,1000,10000,100000,1000000,10000000],time_cuda[2,:],label='CUDA')
ax3.loglog([100,1000,10000,100000,1000000,10000000],time_numpy[2,:],label='Numpy')
ax4.set_title("d=6")
ax4.loglog([100,1000,10000,100000,1000000,10000000],time_cpu[3,:],label='CPU')
ax4.loglog([100,1000,10000,100000,1000000,10000000],time_cuda[3,:],label='CUDA')
ax4.loglog([100,1000,10000,100000,1000000,10000000],time_numpy[3,:],label='Numpy')
ax4.legend()
fig.show()
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
ax1.set_title("d=3")
ax1.loglog([100,1000,10000,100000,1000000,10000000],time_cpu_e[0,:],label='CPU')
ax1.loglog([100,1000,10000,100000,1000000,10000000],time_cuda_e[0,:],label='CUDA')
ax1.loglog([100,1000,10000,100000,1000000,10000000],time_numpy_e[0,:],label='Numpy')
ax2.set_title("d=4")
ax2.loglog([100,1000,10000,100000,1000000,10000000],time_cpu_e[1,:],label='CPU')
ax2.loglog([100,1000,10000,100000,1000000,10000000],time_cuda_e[1,:],label='CUDA')
ax2.loglog([100,1000,10000,100000,1000000,10000000],time_numpy_e[1,:],label='Numpy')
ax3.set_title("d=5")
ax3.loglog([100,1000,10000,100000,1000000,10000000],time_cpu_e[2,:],label='CPU')
ax3.loglog([100,1000,10000,100000,1000000,10000000],time_cuda_e[2,:],label='CUDA')
ax3.loglog([100,1000,10000,100000,1000000,10000000],time_numpy_e[2,:],label='Numpy')
ax4.set_title("d=6")
ax4.loglog([100,1000,10000,100000,1000000,10000000],time_cpu_e[3,:],label='CPU')
ax4.loglog([100,1000,10000,100000,1000000,10000000],time_cuda_e[3,:],label='CUDA')
ax4.loglog([100,1000,10000,100000,1000000,10000000],time_numpy_e[3,:],label='Numpy')
ax4.legend()
fig.show()