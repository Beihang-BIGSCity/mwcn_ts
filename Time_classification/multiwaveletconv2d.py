# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:30:47 2023

@author: 13621
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

def mdwt_kernel(len1,flt):#类似于multiwavelet的def mdwt_matrix(len1,flt):
    #[L,H]=coef(flt)
    #自定义LH旋转后或者直接定义四个矩阵
    k_l1 = np.array([[ 3/(5*mt.sqrt(2)), 3/(5*mt.sqrt(2)),0, 0],[4/5,0, 0, 0]])
    k_l2 = np.array([[-1/20,9/20,9/20,  -1/20],[-3/(10*mt.sqrt(2)),1/mt.sqrt(2), -3/(10*mt.sqrt(2)),0]])
    k_l1 = np.expand_dims(k_l1,axis=0)
    k_l1 = np.expand_dims(k_l1,axis=0)
    k_l2 = np.expand_dims(k_l2,axis=0) 
    k_l2 = np.expand_dims(k_l2,axis=0)
    #####print(f"shape of k_l1 is:",k_l1.shape)
    k_h1 = np.array([[-1/20,9/20,9/20,-1/20],[-3/(10*mt.sqrt(2)),-1/mt.sqrt(2),-3/(10*mt.sqrt(2)),0]]);
    k_h2 = np.array([[1/(10*mt.sqrt(2)),-9/(10*mt.sqrt(2)),9/(10*mt.sqrt(2)),-1/(10*mt.sqrt(2))],[3/10,  0,   -3/10, 0]])
    k_h1 = np.expand_dims(k_h1,axis=0)
    k_h1 = np.expand_dims(k_h1,axis=0)
    k_h2 = np.expand_dims(k_h2,axis=0) 
    k_h2 = np.expand_dims(k_h2,axis=0)
    ########print(f"kl1",k_l1.shape)
    return k_l1,k_l2,k_h1,k_h2

def pre_kernel(len2,pflt):#类似于multiwavelet的def pmatrix(len2,pflt):
    PR,PO=coef_prep(pflt)
    i,j=PR.shape
    k_p1 = PR[0,:]
    k_p2 = PR[1,:]
    k_p1 = np.expand_dims(k_p1,axis=0)
    k_p1 = np.expand_dims(k_p1,axis=0)
    k_p2 = np.expand_dims(k_p2,axis=0) 
    k_p2 = np.expand_dims(k_p2,axis=0) 
    return k_p1,k_p2

class Conv_mwavepre1d(nn.Module):
#    def __init__(self,  is_training, l1_value, weight_decay,len_input=96, sim_reg=0, activation=None):
#        super(Conv_waveL, self).__init__()
    def __init__(self,  is_training, len_input=96, sim_reg=0, activation=None):
        super(Conv_mwavepre1d, self).__init__()
        self.len_input = len_input
        kp1,kp2 = pre_kernel(self.len_input,'ghmap')
        #######print("kp1",kp1)
        self.convpre1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4,stride=2,padding=0,dilation=1,bias=False)
        kp_1=torch.Tensor(kp1) # 先创建一个自定义权值的Tensor，这里为了方便将所有权值设为1
        self.convpre1.weight=torch.nn.Parameter(kp_1) # 把Tensor的值作为权值赋值给Conv层，这里需要先转为torch.nn.Parameter类型，否则将报错
        self.convpre2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4,stride=2,padding=0,dilation=1,bias=False)
        kp_2=torch.Tensor(kp2) # 先创建一个自定义权值的Tensor，这里为了方便将所有权值设为1
        self.convpre2.weight=torch.nn.Parameter(kp_2) # 把Tensor的值作为权值赋值给Conv层，这里需要先转为torch.nn.Parameter类型，否则将报错
    def forward(self,input):
        kp1,kp2 = pre_kernel(self.len_input,'ghmap')
        input = input.unsqueeze(1).to(torch.float32)
        ##print(f"input::0",input.shape)
        in00 = input[:,:,0].unsqueeze(2)
        in01 = input[:,:,1].unsqueeze(2)
        input = torch.cat([input,in00,in01],dim=2)
        ##print("inputori",input.shape)
        L01 = self.convpre1(input)#decl1 decl2yiyang 不同
        #print("L01",L01)
        L02 = self.convpre2(input)
        #print("L02",L02)
        if L01.shape[2] != L02.shape[2]:
            exit
        lpre0 = torch.cat([L01,L02],dim=1)
        lpre0 = lpre0.unsqueeze(1)
        ##print("lpre0",lpre0.shape)
        return lpre0
#        lp_weight = wave_variable_with_l1(lp_mat, 'lp_weight', wd=self.weight_decay, l1_value = self.l1_value, sim_reg = self.sim_reg)
#        hp_weight = wave_variable_with_l1(hp_mat, 'hp_weight', wd=self.weight_decay, l1_value = self.l1_value, sim_reg = self.sim_reg)


class Conv_mwavedec2d(nn.Module):
#    def __init__(self,  is_training, l1_value, weight_decay,len_input=96, sim_reg=0, activation=None):
#        super(Conv_waveL, self).__init__()
    def __init__(self,  is_training, len_inputi=96, sim_reg=0, activation=None):
        super(Conv_mwavedec2d, self).__init__()
        self.len_input = len_inputi
        kl1,kl2,kh1,kh2 = mdwt_kernel(self.len_input,'ghm')
        self.convdecl1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[2,4],stride=2,padding=0,dilation=1,bias=False)
        kl_1=torch.Tensor(kl1)
        
        self.convdecl1.weight=torch.nn.Parameter(kl_1) 
        self.convdecl2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[2,4],stride=2,padding=0,dilation=1,bias=False)
        kl_2=torch.Tensor(kl2)
        self.convdecl2.weight=torch.nn.Parameter(kl_2)
        self.convdech1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[2,4],stride=2,padding=0,dilation=1,bias=False)
        kh_1=torch.Tensor(kh1)
        self.convdech1.weight=torch.nn.Parameter(kh_1)
        self.convdech2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[2,4],stride=2,padding=0,dilation=1,bias=False)
        kh_2=torch.Tensor(kh2)
        self.convdech2.weight=torch.nn.Parameter(kh_2)
    def forward(self,input):
        in1 = input[:,:,:,0].unsqueeze(3)
        in2 = input[:,:,:,1].unsqueeze(3)
        input = torch.cat([input,in1,in2],dim=3)
        ##print("conv_input",input.shape)
        ############print(f"final input",input)
        lp_out1 = self.convdecl1(input)
        ##print("lp_out1",lp_out1.shape)
        lp_out2 = self.convdecl2(input)
        ##print("lp_out2",lp_out2.shape)
        hp_out1 =  self.convdech1(input)
        #print("hp_out1",hp_out1.shape)
        hp_out2 =  self.convdech2(input)
        #print("hp_out2",hp_out2.shape)
        lp_out = torch.cat([lp_out1,lp_out2],dim=2)
        hp_out = torch.cat([hp_out1,hp_out2],dim=2)
        ##print("lp_out",lp_out.shape)
        ##print("hp_out",hp_out.shape)
        return lp_out,hp_out


#class DTW(nn.Module):
#    def __init__(self,v_num,inputv):
#    def forward(self,input):
"""
def main():     
#    mdwt_kernel(16,'ghm')
    x = torch.randn(2,1,16)
    model = Conv_mwavepre(is_training=True,len_input=16, sim_reg=0, activation=None)
    y = model(x)
    print("y:",y)
    model1 = Conv_mwavedec(is_training=True,len_inputi=16, sim_reg=0, activation=None)
    y1,y2 = model1(y)
    print("y1:",y1.shape)
    print("y2:",y2.shape)


if __name__ == '__main__':
    main()#反向传播每次初始化？
"""