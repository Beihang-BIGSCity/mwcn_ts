# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:36:46 2021

@author: 13621
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:12:26 2021
@author: 13621 jxh
"""
import numpy as np 
import os
import sys
from base_module import *
import torch
from torch.autograd import Variable # torch 中 Variable 模块
import pdb

losses = []
#lp_filter = np.load('./wave/db4/lp.npy')
#hp_filter = np.load('./wave/db4/hp.npy')

def get_wave_kernel(shape):
    print(shape[0],int(shape[1]))
    shape[1] = int(shape[1])
    mat_hp = np.zeros((shape[0], shape[1]))
    mat_lp = np.zeros((shape[0], shape[1]))

    for i in range(shape[1]):
        for j in range(8):
            mat_lp[2*i-j, i] = lp_filter[j]
            mat_hp[2*i-j, i] = hp_filter[j]

    return mat_lp, mat_hp

def variable_on_cpu(name,shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
#    with tf.device('/cpu:0'):
    dtype = torch.float16 if use_fp16 else torch.float32
#        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    c1 = torch.device('cpu', 0)
    tensor1 = torch.as_tensor(initializer,dtype=dtype, device=c1)
    var = Variable(tensor1)
    return var
"""
def wave_op(input, len_input, scope, is_training, l1_value, weight_decay, sim_reg=0, activation=None):
    lp_mat, hp_mat = get_wave_kernel([len_input, len_input/2])
    lp_weight = wave_variable_with_l1(lp_mat, 'lp_weight', wd=weight_decay, l1_value = l1_value, sim_reg = sim_reg)
    hp_weight = wave_variable_with_l1(hp_mat, 'hp_weight', wd=weight_decay, l1_value = l1_value, sim_reg = sim_reg)
    biases_lp = variable_on_cpu('biases_lp', [len_input/2],
                             torch.Tensor(0.0))
    biases_hp = variable_on_cpu('biases_hp', [len_input/2],
                             torch.Tensor(0.0))
#        lp_out = tf.matmul(input, lp_weight)
#        lp_out = tf.nn.bias_add(lp_out, biases_lp)
    lp_out = torch.matmul(input, lp_weight)
    lp_out = lp_out + biases_lp
#        hp_out = tf.matmul(input, hp_weight)
#        hp_out = tf.nn.bias_add(hp_out, biases_hp)
    hp_out = torch.matmul(input, hp_weight)
    hp_out = hp_out + biases_hp
    if not activation==None:
        hp_out = activation(hp_out)
        lp_out = activation(lp_out)

    all_out = torch_concat(1, [lp_out, hp_out])
    return lp_out, hp_out, all_out


def wave_op_conv(input, len_input, scope, is_training, l1_value, weight_decay, sim_reg=0, activation=None):
    input = tf.squeeze(input)
    lp_mat, hp_mat = get_wave_kernel([len_input, len_input/2])
    lp_weight = wave_variable_with_l1(lp_mat, 'lp_weight', wd=weight_decay, l1_value = l1_value, sim_reg = sim_reg)
    hp_weight = wave_variable_with_l1(hp_mat, 'hp_weight', wd=weight_decay, l1_value = l1_value, sim_reg = sim_reg)
    
#    biases_lp = variable_on_cpu('biases_lp', [len_input/2],
#                             tf.constant_initializer(0.0))
#    biases_hp = variable_on_cpu('biases_hp', [len_input/2],
#                             tf.constant_initializer(0.0))
    biases_lp = variable_on_cpu('biases_lp', [len_input/2],
                             torch.Tensor(0.0))
    biases_hp = variable_on_cpu('biases_hp', [len_input/2],
                             torch.Tensor(0.0))
    lp_out = torch.matmul(input, lp_weight)
    lp_out = lp_out + biases_lp

    hp_out = torch.matmul(input, hp_weight)
    hp_out = hp_out + biases_hp

    if not activation==None:
        hp_out = activation(hp_out)
        lp_out = activation(lp_out)

    hp_out = torch.unsqueeze(hp_out, -1)
    lp_out = torch.unsqueeze(lp_out, -1)

    hp_out = torch.unsqueeze(hp_out, -1)
    lp_out = torch.unsqueeze(lp_out, -1)

    all_out = torch_concat(-1, [lp_out, hp_out])
    return lp_out, hp_out, all_out

def wavelet_conv(timeseries, len_input, num_outputs, is_training, scope, l1_value, dp_kp):

    lp_coe, hp_coe, all_coe = wave_op_conv(timeseries, len_input, scope='wave_func',
            is_training=is_training, l1_value=l1_value, weight_decay=weigth_decay_fc, sim_reg=SIM_REG)

    return lp_coe,hp_coe
"""