# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:23:57 2021

@author: 13621
"""

import numpy as np 
import os
import sys
import torch
from modelswave import *
def _wave_variable_on_cpu(matr, name, trainable):
    """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
    """
    c0=torch.device('cpu',0)
    var = torch.tensor(matr,dtype=torch.float32, device=c0,requires_grad = True)
    #shape = matr.shape
    #var = tf.get_variable(name, shape)
    return var

def wave_variable_with_l1(matr, name, wd, l1_value, sim_reg=None):
  """Helper to create an wavelt initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    l1_value: add L1Loss to wavelet initialized weight
    sim_reg: regularization terms on forcing trained weight to be similar with the initial weight

  Returns:
    Variable Tensor
  """
  var = _wave_variable_on_cpu(
      matr,
      name = name, trainable=True)
  var_reg = _wave_variable_on_cpu(
      matr,
      name = name, trainable=False)

  if l1_value is not None:
#    l1_loss = tf.multiply(tf.reduce_mean(tf.abs(var)), l1_value, name='l1_loss')
#    tf.add_to_collection('losses', l1_loss)
    l1_loss = torch.mul(torch.mean(torch.abs(var)),l1_value)
    losses.append(l1_loss)
  if wd is not None:
#    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
#    tf.add_to_collection('losses', weight_decay)
    l2_loss = torch.sum(var ** 2, dim=1) / 2
    weight_decay = torch.mul(l2_loss, wd)
    losses.append(weight_decay)
  if sim_reg is not None:
#    similar_reg = tf.multiply(tf.nn.l2_loss(var-var_reg), sim_reg, name='similar_reg_loss')
#    tf.add_to_collection('losses', similar_reg)
    l2_loss1 = torch.sum((var-var_reg) ** 2, dim=1) / 2
    similar_reg = torch.mul(l2_loss1,sim_reg)
    losses.append(similar_reg)
  return var

def torch_concat(axis, values):
    return torch.cat(values, axis)