# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:19:40 2021

@author: 13621 multi-wavelet
"""

import numpy as np 
import os
import sys
from base_module import *
import torch
import torch.nn as nn
from torch.autograd import Variable # torch 中 Variable 模块
import numpy as np
import pdb
from modelswave import *
import math as mt
from coef_prep import *
from coef import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def circShift(array,K):
    """
    :param array: input array `numpy.ndarray`
    :param K: int or tuple(int a,int b) or list(int a,int b)
    :return: array `numpy.ndarray`
    """
    height,width = array.shape
    if len(array.shape)>=2 and height*width>=4:
        if type(K) ==int and abs(K)<height:
            updownA = array[:-K, :]
            mainArray = array[-K:,:]
            flip_matrix =  np.concatenate((mainArray,updownA),axis=0)
        elif type(K) ==tuple or type(K) ==list and abs(K[0])<height and abs(K[1])<width:
            updownA = array[:-K[0], :]
            mainArray = array[-K[0]:, :]
            temp = np.concatenate((mainArray, updownA), axis=0)

            leftrightA = temp[:, :-K[1]]
            tempArray = temp[:, -K[1]:]
            flip_matrix =  np.concatenate((tempArray,leftrightA),axis=1)
        else:
            print("`numpy.ndarray`.shape Error")
            flip_matrix = None
    else:
        print('`numpy.ndarray`.shape Error')
        flip_matrix = None
    return flip_matrix

def mdwt_matrix(len1,flt):
#在文件coef.m中，低频滤波器系与高频滤波器系要相同，
#如果不相同，要使用0矩阵，补足到长度相同
    [L,H]=coef(flt)
    L = np.array(L)
    H = np.array(H)    
    print(L.shape)
    [i,j] = L.shape
    r=min(i,j)
    print(f"r in mdwt is:{r}")
    z=np.zeros((r,r))
    k=max(i/r,j/r)
    n=len1/r
    L1=np.hstack((L,np.zeros((r,int(len1-r*k)))))
    H1=np.hstack((H,np.zeros((r,int(len1-r*k)))))
    L_matrix=np.empty((1,int(len1)))
    H_matrix=np.empty((1,int(len1)))
    print("lenn is:%f"%(len1-r*k))
    n = int(n)
    print(n)
    for qq in range(1,int(n/2)+1):
        L_matrix=np.vstack((L_matrix,L1))
        L1=circShift(L1.T,2*r).T
        H_matrix=np.vstack((H_matrix,H1))
        H1=circShift(H1.T,2*r).T
    L_matrix = np.delete(L_matrix,0,axis=0)
    H_matrix = np.delete(H_matrix,0,axis=0)
    return L_matrix,H_matrix

def matrix_random(len1):
    r = 2
    n=int(len1/r)
    L = np.random.rand(2*int(n/2), int(len1))#np.random.rand(0, 2, size=(int(n/2), int(len1)))
    H = np.random.rand(2*int(n/2), int(len1))#np.random.randint(0, 2, size=(int(n/2), int(len1)))
    return L,H

def pmatrix_random(len2):
    r = 2
    n=len2/r
    prmat = np.random.rand(2*int(n), len2)
    pomat = np.random.rand(2*int(n), len2)
    return prmat,pomat


def pmatrix(len2,pflt):
    PR,PO=coef_prep(pflt)
    i,j=PR.shape
    r=min(i,j)
    #print(f"r in pm is:{r}")
    z=np.zeros((r,r))
    k=max(i/r,j/r)
    n=len2/r
    pr1=np.hstack((PR,np.zeros((r,int(len2-r*k)))))
    po1=np.hstack((np.zeros((r,int(len2-r*k))),PO))

    prmat = np.empty((1,len2))
    pomat = np.empty((1,len2))
    for qq in range(1,int(n+1)):
        prmat=np.vstack((prmat,pr1))
        pr1=circShift(pr1.T,r).T
        po1=circShift(po1.T,r).T
        pomat=np.vstack((pomat,po1))
    prmat=np.delete(prmat,0,axis=0)
    pomat=np.delete(pomat,0, axis = 0)
    return prmat,pomat

class Conv_multiwavenew(nn.Module):
#    def __init__(self,  is_training, l1_value, weight_decay,len_input=96, sim_reg=0, activation=None):
#        super(Conv_waveL, self).__init__()
    def __init__(self,  is_training, len_input=96, sim_reg=0, activation=None):
        super(Conv_multiwavenew, self).__init__()
        self.len_input = len_input
        p,q = pmatrix(self.len_input,'ghmap')
        ##p,q = pmatrix_random(self.len_input) random change
        p1 = torch.tensor(p,dtype=torch.float32,requires_grad=True)
        self.p11 = torch.nn.Parameter(p1,requires_grad = True).to(device=0)
        L1,H1 = mdwt_matrix(self.len_input,'ghm') 
        ##L1,H1 = matrix_random(self.len_input) random change
        lp_weight0 = torch.tensor(L1,dtype=torch.float32, requires_grad=True)
        self.lp_weight = torch.nn.Parameter(lp_weight0,requires_grad = True).to(device=0)
        hp_weight0 = torch.tensor(H1,dtype=torch.float32, requires_grad=True)
        self.hp_weight = torch.nn.Parameter(hp_weight0,requires_grad = True).to(device=0)
        self.activation = activation
    def forward(self,input):
#        print(self.lp_weight)
        biases_lp = variable_on_cpu('biases_lp', [self.len_input/2],
                             torch.zeros([int(self.len_input/2)], dtype=torch.float32))#再品
        biases_hp = variable_on_cpu('biases_hp', [self.len_input/2],
                             torch.zeros([int(self.len_input/2)], dtype=torch.float32))
#        lp_out = torch.matmul(p1,input.float())
#        lp_out = torch.matmul(self.lp_weight,lp_out)
#        hp_out = torch.matmul(p1,input.float())
#        hp_out = torch.matmul(self.hp_weight,hp_out)
        #print(f"{self.lp_weight.shape},{self.p11.shape}")
        lp_out = torch.mm(self.lp_weight,self.p11).t()
        input = input.float()
        lp_out = torch.matmul(input,lp_out)
        hp_out = torch.mm(self.hp_weight,self.p11).t()
        hp_out = torch.matmul(input,hp_out)
        #lp_out = lp_out + biases_lp
        #hp_out = hp_out + biases_hp
        if self.activation ==True:
            lp_out = torch.sigmoid(lp_out) 
            hp_out = torch.sigmoid(hp_out) 
        return lp_out,hp_out
#        lp_weight = wave_variable_with_l1(lp_mat, 'lp_weight', wd=self.weight_decay, l1_value = self.l1_value, sim_reg = self.sim_reg)
#        hp_weight = wave_variable_with_l1(hp_mat, 'hp_weight', wd=self.weight_decay, l1_value = self.l1_value, sim_reg = self.sim_reg)


class wavecontinue(nn.Module):
#    def __init__(self,  is_training, l1_value, weight_decay,len_input=96, sim_reg=0, activation=None):
#        super(Conv_waveL, self).__init__()
    def __init__(self,  is_training, len_inputi=96, sim_reg=0, activation=None):
        super(wavecontinue, self).__init__()
        self.len_input = len_inputi
        print(len_inputi)
        Li,Hi = mdwt_matrix(self.len_input,'ghm')
        ##Li,Hi = matrix_random(self.len_input) random change
        lp_weighti = torch.tensor(Li,dtype=torch.float32, requires_grad=True)
        self.lp_weighti = torch.nn.Parameter(lp_weighti,requires_grad = True).to(device=0)
        hp_weighti = torch.tensor(Hi,dtype=torch.float32, requires_grad=True)
        self.hp_weighti = torch.nn.Parameter(hp_weighti,requires_grad = True).to(device=0)
        self.activation = activation
    def forward(self,input):
        biases_lpi = variable_on_cpu('biases_lp', [self.len_input/2],
                             torch.zeros([int(self.len_input/2)], dtype=torch.float32))#再品
        biases_hpi = variable_on_cpu('biases_hp', [self.len_input/2],
                             torch.zeros([int(self.len_input/2)], dtype=torch.float32))
        
        Li = self.lp_weighti.t()
        input = input.float()
        lp_outi = torch.matmul(input,Li)
        #lp_outi = lp_outi + biases_lpi
        Hi = self.hp_weighti.t()
        hp_outi = torch.matmul(input,Hi)
        #hp_outi = hp_outi + biases_hpi
        if self.activation==True:
            hp_outi = torch.sigmoid(hp_outi)
            lp_outi = torch.sigmoid(lp_outi)
        return lp_outi,hp_outi


#class DTW(nn.Module):
#    def __init__(self,v_num,inputv):
#    def forward(self,input):
      
