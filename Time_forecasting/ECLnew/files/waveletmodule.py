# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:08:28 2022

@author: 13621 wavelet module 
"""

import numpy as np 
import tensorflow as tf 
import os
import sys
from utils import *
import torch
import pdb
from modelswave import *
from base_module import *

lp_filter = np.load('./wave/db4/lp.npy')
hp_filter = np.load('./wave/db4/hp.npy')

def get_wave_kernel(shape):
    mat_hp = np.zeros((int(shape[0]), int(shape[1])))
    mat_lp = np.zeros((int(shape[0]), int(shape[1])))

    for i in range(int(shape[1])):
        for j in range(8):
            mat_lp[2*i-j, i] = lp_filter[j]
            mat_hp[2*i-j, i] = hp_filter[j]

    return mat_lp, mat_hp


class Conv_wave(nn.Module):
#    def __init__(self,  is_training, l1_value, weight_decay,len_input=96, sim_reg=0, activation=None):
#        super(Conv_waveL, self).__init__()
    def __init__(self,  is_training, len_input=96, sim_reg=0, activation=None):
        super(Conv_wave, self).__init__()
        self.len_input = len_input
        lp_mat, hp_mat = get_wave_kernel([len_input, len_input/2])
        lp_w0 = torch.tensor(lp_mat,dtype=torch.float32, requires_grad=True)
        self.lp_w = torch.nn.Parameter(lp_w0,requires_grad = True)#.to(device=0)
        hp_w0 = torch.tensor(hp_mat,dtype=torch.float32, requires_grad=True)
        self.hp_w = torch.nn.Parameter(hp_w0,requires_grad = True)#.to(device=0)
        self.activation = activation
    def forward(self,input):
        biases_lp = variable_on_cpu('biases_lp', [self.len_input/2],
                             torch.zeros([int(self.len_input/2)], dtype=torch.float32))#再品
        biases_hp = variable_on_cpu('biases_hp', [self.len_input/2],
                             torch.zeros([int(self.len_input/2)], dtype=torch.float32))
#        lp_out = torch.matmul(p1,input.float())
#        lp_out = torch.matmul(self.lp_weight,lp_out)
#        hp_out = torch.matmul(p1,input.float())
#        hp_out = torch.matmul(self.hp_weight,hp_out)
        input = input.float()
        lp_out = torch.matmul(input,self.lp_w)
        hp_out = torch.matmul(input,self.hp_w)
        if not self.activation==None:
            hp_out = self.activation(hp_out)
            lp_out = self.activation(lp_out)
        return lp_out,hp_out




def wave_op(input, len_input, scope, is_training, l1_value, weight_decay, sim_reg=0, activation=None):
    lp_mat, hp_mat = get_wave_kernel([len_input, len_input/2])
    lp_weight = wave_variable_with_l1(lp_mat, 'lp_weight', wd=weight_decay, l1_value = l1_value, sim_reg = sim_reg)
    hp_weight = wave_variable_with_l1(hp_mat, 'hp_weight', wd=weight_decay, l1_value = l1_value, sim_reg = sim_reg)
    biases_lp = variable_on_cpu('biases_lp', [len_input/2],
                             tf.constant_initializer(0.0))
    biases_hp = variable_on_cpu('biases_hp', [len_input/2],
                         tf.constant_initializer(0.0))
    lp_out = tf.matmul(input, lp_weight)
    lp_out = tf.nn.bias_add(lp_out, biases_lp)
    
    hp_out = tf.matmul(input, hp_weight)
    hp_out = tf.nn.bias_add(hp_out, biases_hp)
        
    if not activation==None:
        hp_out = activation(hp_out)
        lp_out = activation(lp_out)

    all_out = tf_concat(1, [lp_out, hp_out])
    return lp_out, hp_out, all_out


def wave_op_conv(input, len_input, scope, is_training, l1_value, weight_decay, sim_reg=0, activation=None):
        input = tf.squeeze(input)
        lp_mat, hp_mat = get_wave_kernel([len_input, len_input/2])
        lp_weight = wave_variable_with_l1(lp_mat, 'lp_weight', wd=weight_decay, l1_value = l1_value, sim_reg = sim_reg)
        hp_weight = wave_variable_with_l1(hp_mat, 'hp_weight', wd=weight_decay, l1_value = l1_value, sim_reg = sim_reg)
        biases_lp = variable_on_cpu('biases_lp', [len_input/2],
                             tf.constant_initializer(0.0))
        biases_hp = variable_on_cpu('biases_hp', [len_input/2],
                             tf.constant_initializer(0.0))
        lp_out = tf.matmul(input, lp_weight)
        lp_out = tf.nn.bias_add(lp_out, biases_lp)

        hp_out = tf.matmul(input, hp_weight)
        hp_out = tf.nn.bias_add(hp_out, biases_hp)

        if not activation==None:
            hp_out = activation(hp_out)
            lp_out = activation(lp_out)

        hp_out = tf.expand_dims(hp_out, -1)
        lp_out = tf.expand_dims(lp_out, -1)

        hp_out = tf.expand_dims(hp_out, -1)
        lp_out = tf.expand_dims(lp_out, -1)

        all_out = tf_concat(-1, [lp_out, hp_out])
        return lp_out, hp_out, all_out
