# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:44:31 2022

@author: 13621 net_multiwaveconv 
"""


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import torch.autograd.variable as Variable
import torch.utils.data as Data
from multiwaveletconv import *
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 16
learning_rate = 0.0001
from vgg1d import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class conv_multiwave(nn.Module):#三层小波分解
    def __init__(self,length_input = 96, number_ouputs=4,mode='dot',factor=5, d_model=512, n_heads=8, d_layers=2,
                dropout=0.0, freq='h',output_attention = False,model='vgg1d',d_prob=0.5):
        super(conv_multiwave, self).__init__()   # 继承__init__功能 pytorch初始化参数
        if length_input % 8 != 0:
            aaa = length_input%8
            length_input = length_input + 8-(length_input%8)
            self.diff = 8-aaa  
        else:
            self.diff = 0
        self.len_input = length_input
        #print("hello")
        #print(length_input)
        #print(self.diff)
        enc_in = 7
        embed = 'fixed'
        self.num_outputs = number_ouputs
        self.factor=factor
        self.d_model=d_model
        self.n_heads=n_heads
        self.d_layers=d_layers
        self.dropout=dropout
        self.freq='h'
        self.output_attention=output_attention
        self.model = model
        kernel_size = 8
        stride = 2
        n_block = 3
        downsample_gap = 6
        increasefilter_gap = 6
        ## 第一层卷积
        self.features1 = Conv_mwavepre(is_training=True,len_input=self.len_input, sim_reg=0, activation=None)
        self.features2 = Conv_mwavedec(is_training=True,len_inputi=self.len_input, sim_reg=0, activation=None)
        self.features3 = Conv_mwavedec(is_training=True,len_inputi=self.len_input/2, sim_reg=0, activation=None)
        self.features4 = Conv_mwavedec(is_training=True,len_inputi=self.len_input/4, sim_reg=0, activation=None)

        self.vgg1d = VGG1d('VGG11',num_class=self.num_outputs)
        self.vgg1d2 = VGG1d('VGG7',num_class=self.num_outputs)
        self.vgg1d3 =  VGG1d('VGG9',num_class=self.num_outputs)
        self.vgg1d4 =  VGG1d('VGGn',num_class=self.num_outputs)
        #self.fc = nn.Linear(in_features=64,out_features=self.num_outputs)
        self.output = nn.Linear(in_features=111, out_features=self.num_outputs)

    def forward(self, x,mode='dot'):
        x= x.to('cuda:0')
        if self.diff !=0:
            if x.shape[0]==16:
                btr = np.zeros((16,1,self.diff))
                ad1 = torch.from_numpy(btr).to(device=0)
                x = torch.cat((x,ad1),2)
            elif x.shape[0]!=16:
                btr = np.zeros((x.shape[0],1,self.diff))
                ad1 = torch.from_numpy(btr).to(device=0)
                x = torch.cat((x,ad1),2)
        if x.shape[0]==1:
            x = x.squeeze(dim=1)
        else:
            x = x.squeeze()
        low0= self.features1(x)#yz为第一层的系数，low2,high2为第二层，low3,high3为第三层
        low0 = low0#.to('cuda:0')
        low1,high1 = self.features2(low0)
       
#print(low2.shape)
        low1 = low1#.to('cuda:0')
        high1 = high1#.to('cuda:0')
        low2,high2 = self.features3(low1)
        low2=low2#.to('cuda:0')
        high2 = high2#.to('cuda:0')
        low3,high3 = self.features4(low2)
        low3=low3#.to('cuda:0')
        high3 = high3#.to('cuda:0')
        if self.model == 'res':
            low1 = low1.to(torch.float32)
            low1=self.resnet(low1)
            high1 = high1.to(torch.float32)
            high1=self.resnet(high1)
            out = torch.cat([low1, high1], dim=1)
            out = out.squeeze()
            out1 = self.fc(out)
            return out1
        elif self.model=='vgg1d':    
            low1 = low1.to(torch.float32)
            #low1=low1.unsqueeze(1)
#            print(low1.shape)
            high1 = high1.to(torch.float32)
            #high1=high1.unsqueeze(1)
            all_out1 = torch.cat([low1, high1],dim=1)#.to('cuda:0')
            all_out1 = self.vgg1d(all_out1)
            low2 = low2.to(torch.float32)
            #low2=low2.unsqueeze(1)
            high2 = high2.to(torch.float32)
            #high2 = high2.unsqueeze(1)
            all_out2 = torch.cat([low2, high2],dim=1)#.to('cuda:0')
            all_out2 = self.vgg1d3(all_out2)
#            low3 = low3.to(torch.float32)
            #low3=low3.unsqueeze(1)
            high3 = high3.to(torch.float32)
            #high3 = high3.unsqueeze(1)
            all_out3 = torch.cat([low3, high3],dim=1)#.to('cuda:0')
            all_out3 = self.vgg1d2(all_out3)
            z=torch.cat([all_out1,all_out2,all_out3],dim=1)       
            output = self.output(z)     # 输出[16,2]
            return output
