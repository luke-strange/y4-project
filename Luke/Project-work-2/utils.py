#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import time
import pkg_resources
import math
import types
import torch
import torch.optim as optim
from scipy import constants
from scipy import integrate
import scipy.stats as stats
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special as scs


def rescale(x, y, region):
    '''takes normalised parameter regions and 
    rescales them for plotting'''
    xs = region[0][0] + x * (region[0][1] - region[0][0])
    ys = region[1][0] + y * (region[1][1] - region[1][0])
    return xs, ys

def numpy2cuda(array, single=True):
    array = torch.from_numpy(array)
  
    if single:
        array = array.float()
    
    if torch.cuda.is_available():
        array = array.cuda()
    
    return array

def cuda2numpy(tensor):
    return tensor.detach().cpu().numpy()

#Geometry.py
def setgeometry(q):
    global qdim, xmin, xmax, xstops, xmid, xwid

    # bins
    qdim = q

    # prior range for x (will be uniform)
    xmin, xmax = 0, 1

    # definition of quantization bins
    xstops = np.linspace(xmin, xmax, qdim + 1)

    # to plot histograms
    xmid = 0.5 * (xstops[:-1] + xstops[1:])
    xwid = xstops[1] - xstops[0]

setgeometry(64)

#Network.py
def makenet(dims, softmax=True, single=True):
  """Make a fully connected DNN with layer widths described by `dims`.
  CUDA is always enabled, and double precision is set with `single=False`.
  The output layer applies a softmax transformation,
  disabled by setting `softmax=False`."""

  ndims = len(dims)

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # the weights must be set explicitly as attributes in the class
      # (i.e., we can't collect them in a single list)
      for l in range(ndims - 1):
        layer = nn.Linear(dims[l], dims[l+1])
        
        if not single:
          layer = layer.double()
        
        if torch.cuda.is_available():
          layer = layer.cuda()
        
        setattr(self, f'fc{l}', layer)
                
    def forward(self, x):
      # per Alvin's recipe, apply relu everywhere but last layer
      for l in range(ndims - 2):
        x = F.leaky_relu(getattr(self, f'fc{l}')(x), negative_slope=0.2)

      x = getattr(self, f'fc{ndims - 2}')(x)

      if softmax:
        return F.softmax(x, dim=1)
      else:
        return x
  
  return Net


def makenetbn(dims, softmax=True, single=True):
  """A batch-normalizing version of makenet. Experimental."""

  ndims = len(dims)

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # the weights must be set explicitly as attributes in the class
      # (i.e., we can't collect them in a single list)
      for l in range(ndims - 1):
        layer = nn.Linear(dims[l], dims[l+1])
        bn = nn.BatchNorm1d(num_features=dims[l+1])
        
        if not single:
          layer = layer.double()
          bn = bn.double()
        
        if torch.cuda.is_available():
          layer = layer.cuda()
          bn = bn.cuda()
        
        setattr(self, f'fc{l}', layer)
        setattr(self, f'bn{l}', bn)
                
    def forward(self, x):
      # per Alvin's recipe, apply relu everywhere but last layer
      for l in range(ndims - 2):
        x = getattr(self, f'bn{l}')(F.leaky_relu(getattr(self, f'fc{l}')(x), negative_slope=0.2))

      x = getattr(self, f'fc{ndims - 2}')(x)

      if softmax:
        return F.softmax(x, dim=1)
      else:
        return x
  
  return Net

#Loss function - Kullback-Libeler.
def kllossGn2(o, l: 'xtrue'):
  """KL loss for Gaussian-mixture output, 2D, precision-matrix parameters."""

  dx  = o[:,0::6] - l[:,0,np.newaxis]
  dy  = o[:,2::6] - l[:,1,np.newaxis]
  
  # precision matrix is positive definite, so has positive diagonal terms
  Fxx = o[:,1::6]**2
  Fyy = o[:,3::6]**2
  
  # precision matrix is positive definite, so has positive 
  Fxy = torch.atan(o[:,4::6]) / (0.5*math.pi) * o[:,1::6] * o[:,3::6]
  
  weight = torch.softmax(o[:,5::6], dim=1)
   
  # omitting the sqrt(4*math*pi) since it's common to all templates
  return -torch.mean(torch.logsumexp(torch.log(weight) - 0.5*(Fxx*dx*dx + Fyy*dy*dy + 2*Fxy*dx*dy) + 0.5*torch.log(Fxx*Fyy - Fxy*Fxy), dim=1))

