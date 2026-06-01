# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:11:32 2023

@author: 13621
"""


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.autograd.variable as Variable
import torch.utils.data as Data
from multiwaveletconv2d import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from SENet import *
#from showdata import *
#from showdataecl import *
batch_size = 16
learning_rate = 0.0001



device = 'cuda' if torch.cuda.is_available() else 'cpu'
class conv2d_multiwave(nn.Module):#三层小波分解
    def __init__(self,length_input, time_step, output_size=96, mode='dot',factor=5, d_model=512, n_heads=8, d_layers=2,
                dropout=0.0, freq='h',model='lstm',d_prob=0.5,hidden_size=6,batch_size1=32):
        super(conv2d_multiwave, self).__init__()   # 继承__init__功能 pytorch初始化参数
        if length_input % 16 != 0:
            aaa = length_input%16
            length_input = length_input + 16-(length_input%16)
            self.diff = 16-aaa  
        else:
            self.diff = 0
        self.len_input = length_input
        enc_in = 7
        embed = 'fixed'
        self.output_size = output_size
        self.factor=factor
        self.d_model=d_model
        self.n_heads=n_heads
        self.hidden_size = hidden_size
        self.d_layers=d_layers
        self.dropout=dropout
        self.freq='h'
        self.batch_size = batch_size1
        self.model = model
        kernel_size = 8
        stride = 2
        n_block = 3
        downsample_gap = 6
        increasefilter_gap = 6
        ## 第一层卷积
        self.features1 = Conv_mwavepre1d(is_training=True,len_input=self.len_input, sim_reg=0, activation=None)
        self.features2 = Conv_mwavedec2d(is_training=True,len_inputi=self.len_input, sim_reg=0, activation=None)
        self.features3 = Conv_mwavedec2d(is_training=True,len_inputi=self.len_input/2, sim_reg=0, activation=None)
        self.features4 = Conv_mwavedec2d(is_training=True,len_inputi=self.len_input/4, sim_reg=0, activation=None)
        self.se_block = se_block(in_channel=6)
        self.embed1 = nn.LSTM(input_size=4, hidden_size=self.hidden_size, batch_first=True,bidirectional=True)#本input 任意的送进的（非，是一个模型） 需要送进的 第一个为本 
        self.embed2 = nn.LSTM(input_size=4, hidden_size=self.hidden_size, batch_first=True,bidirectional=True)#int(time_step/2)
        self.embed3 = nn.LSTM(input_size=4, hidden_size=self.hidden_size, batch_first=True,bidirectional=True)#-2
        self.fc = nn.Linear(in_features=self.hidden_size*2,out_features=1)
        self.output = nn.Linear(in_features=int((self.len_input)/4+(self.len_input)/8+(self.len_input)/16), out_features=self.output_size)

    def forward(self, x,mode='dot'):
        x= x#.to('cuda:0') #[batch_size, 1, len_input]
        ##print("ori size x",x.shape)
        if self.diff !=0:
            if x.shape[0]==self.batch_size:
                btr = np.zeros((self.batch_size,1,self.diff))
                ad1 = torch.from_numpy(btr).to(device=0)
                x = torch.cat((x,ad1),2)
            elif x.shape[0]!=self.batch_size:
                btr = np.zeros((x.shape[0],1,self.diff))
                ad1 = torch.from_numpy(btr).to(device=0)
                x = torch.cat((x,ad1),2)
        ##print("devicein is",torch.cuda.current_device())
        ##print("size x",x.shape)
        #now x is [batch_size, 1, len_input]
        x = x.squeeze(1) #[batch_size, len_input] #x = x.squeeze()
        ##print("squeeze size x",x.shape)
        low0= self.features1(x)#[batch_size, 1, 2, len_input/2]
        ##print("low0",low0.shape)
        low1,high1 = self.features2(low0)
        low1 = low1#.to('cuda:0') [batch_size, 1, 2, len_input/4]
        high1 = high1#.to('cuda:0') [batch_size, 1, 2, len_input/4]
        ##print("low1 high1",low1.shape,high1.shape)
        low2,high2 = self.features3(low1)#[batch_size, 1, 2, len_input/8],[batch_size, 1, 2, len_input/8]
        low2=low2#.to('cuda:0')
        high2 = high2#.to('cuda:0')
        ##print("low2 high2",low2.shape,high2.shape)
        low3,high3 = self.features4(low2)#[batch_size, 1, 2, len_input/16],[batch_size, 1, 2, len_input/16]
        low3=low3#.to('cuda:0')
        high3 = high3#.to('cuda:0')
        ##print("low3 high3",low3.shape,high3.shape)
        if self.model=='lstm':
            low1 = low1.to(torch.float32)
            high1 = high1.to(torch.float32)
            all_out1 = torch.cat([low1, high1],dim=2)#.to('cuda:0') [batch_size, 1, 4, len_input/4]
            #print("all_out1",all_out1.shape)
            all_out1 = all_out1.permute(0,1,3,2)
            all_out1 = all_out1.squeeze(1)
            #print("all_out before embeded",all_out1.shape)
            all_out1,__ = self.embed1(all_out1)#[batch_size,len_input/4,2*hidden_size]
            #print("all_out1 embeded",all_out1.shape)
            low2 = low2.to(torch.float32)
            #low2=low2.unsqueeze(1)
            high2 = high2.to(torch.float32)
            #high2 = high2.unsqueeze(1)
            
            all_out2 = torch.cat([low2, high2],dim=2)#.to('cuda:0') [batch_size, 1, 4, len_input/8]
            all_out2 = all_out2.permute(0,1,3,2)
            all_out2 = all_out2.squeeze(1)
            all_out2,__ = self.embed2(all_out2)#[batch_size,len_input/8,2*hidden_size]
            
            ##print(f"all_out2 vgg3",all_out2.size())
            low3 = low3.to(torch.float32)
            #low3=low3.unsqueeze(1)
            high3 = high3.to(torch.float32)
            #high3 = high3.unsqueeze(1)
            all_out3 = torch.cat([low3, high3],dim=2)#.to('cuda:0') [batch_size, 1, 4, len_input/16]
            all_out3 = all_out3.permute(0,1,3,2)
            all_out3 = all_out3.squeeze(1)
            all_out3,__ = self.embed3(all_out3) #[batch_size,len_input/16,2*hidden_size]
            #print("all_out1,all_out2,all_out3",all_out1.shape,all_out2.shape,all_out3.shape)
            z=torch.cat([all_out1,all_out2,all_out3],dim=1)  # 
            z = self.fc(z)
            z = z.squeeze(2)
            output = self.output(z) #[bacth_size, num_output]    # 输出[16,2]
        return output
