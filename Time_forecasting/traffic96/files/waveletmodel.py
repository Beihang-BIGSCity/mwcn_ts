# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 19:48:10 2021

@author: 13621 waveletmodel
"""
import numpy as np 
import os
import sys
from base_module import *
import torch
import torch.nn as nn
from torch.autograd import Variable # torch 中 Variable 模块
import pdb
from modelswave import *
class Conv_waveL(nn.Module):
    def __init__(self,  is_training, l1_value, weight_decay,len_input=96, sim_reg=0, activation=None):
        super(Conv_waveL, self).__init__()
        self.len_input = len_input
        self.is_training = is_training
        self.l1_value = l1_value
        self.weight_decay = weight_decay
        self.sim_reg = sim_reg
        self.activation = activation
        lp_mat, hp_mat = get_wave_kernel([self.len_input, self.len_input/2])
        self.lp_weight0 = torch.tensor(lp_mat,dtype=torch.float32,requires_grad = True)
        self.lp_weight = torch.nn.Parameter(self.lp_weight0,requires_grad = True)
        self.hp_weight0 = torch.tensor(hp_mat,dtype=torch.float32,requires_grad = True)
        self.hp_weight = torch.nn.Parameter(self.hp_weight0,requires_grad = True)
    def forward(self,input):
        
#        lp_weight = wave_variable_with_l1(lp_mat, 'lp_weight', wd=self.weight_decay, l1_value = self.l1_value, sim_reg = self.sim_reg)
#        hp_weight = wave_variable_with_l1(hp_mat, 'hp_weight', wd=self.weight_decay, l1_value = self.l1_value, sim_reg = self.sim_reg)
        
        
        biases_lp = variable_on_cpu('biases_lp', [self.len_input/2],
                             torch.zeros([48], dtype=torch.float32))
        biases_hp = variable_on_cpu('biases_hp', [self.len_input/2],
                             torch.zeros([48], dtype=torch.float32))
#        lp_out = tf.matmul(input, lp_weight)
#        lp_out = tf.nn.bias_add(lp_out, biases_lp)
        lp_out = torch.matmul(input, self.lp_weight)
        lp_out = lp_out + biases_lp

#        hp_out = tf.matmul(input, hp_weight)
#        hp_out = tf.nn.bias_add(hp_out, biases_hp)
        hp_out = torch.matmul(input, self.hp_weight)
        hp_out = hp_out + biases_hp
        if not self.activation==None:
            hp_out = self.activation(hp_out)
            lp_out = self.activation(lp_out)

        all_out = torch_concat(1, [lp_out, hp_out])
#        print("this")
#        print(self.lp_weight.shape)
#        print(input.shape)
        return lp_out,self.lp_weight

class Conv_waveH(nn.Module):
    def __init__(self,  is_training, l1_value, weight_decay,len_input=96, sim_reg=0, activation=None):
        super(Conv_waveH, self).__init__()
        self.len_input = len_input
        self.is_training = is_training
        self.l1_value = l1_value
        self.weight_decay = weight_decay
        self.sim_reg = sim_reg
        self.activation = activation
        lp_mat, hp_mat = get_wave_kernel([self.len_input, self.len_input/2])
        self.lp_weight0 = torch.tensor(lp_mat,dtype=torch.float32,requires_grad = True)
        self.lp_weight = torch.nn.Parameter(self.lp_weight0,requires_grad = True)
        self.hp_weight0 = torch.tensor(hp_mat,dtype=torch.float32,requires_grad = True)
        self.hp_weight = torch.nn.Parameter(self.hp_weight0,requires_grad = True)
    def forward(self,input):
#        lp_mat, hp_mat = get_wave_kernel([self.len_input, self.len_input/2])
#        lp_weight = wave_variable_with_l1(lp_mat, 'lp_weight', wd=self.weight_decay, l1_value = self.l1_value, sim_reg = self.sim_reg)
#        hp_weight = wave_variable_with_l1(hp_mat, 'hp_weight', wd=self.weight_decay, l1_value = self.l1_value, sim_reg = self.sim_reg)
        biases_lp = variable_on_cpu('biases_lp', [self.len_input/2],
                             torch.zeros([48], dtype=torch.float32))
        biases_hp = variable_on_cpu('biases_hp', [self.len_input/2],
                             torch.zeros([48], dtype=torch.float32))
#        lp_out = tf.matmul(input, lp_weight)
#        lp_out = tf.nn.bias_add(lp_out, biases_lp)
        
        lp_out = torch.matmul(input, self.lp_weight)
        
        lp_out = lp_out + biases_lp
#        hp_out = tf.matmul(input, hp_weight)
#        hp_out = tf.nn.bias_add(hp_out, biases_hp)
        hp_out = torch.matmul(input, self.hp_weight)
        hp_out = hp_out + biases_hp
        if not self.activation==None:
            hp_out = self.activation(hp_out)
            lp_out = self.activation(lp_out)

        all_out = torch_concat(1, [lp_out, hp_out])
        return hp_out