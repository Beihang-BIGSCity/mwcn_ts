# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:11:32 2023

@author: 13621
"""


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import torch.autograd.variable as Variable
import torch.utils.data as Data
from multiwavelet_att3 import *
import torch
import torch.nn as nn
import torch.nn.functional as F
#from showdata import *#品
#from showdataecl import *
batch_size = 16
learning_rate = 0.0001
from vgg_v2 import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class conv2d_multiwave_v2_att22(nn.Module):#三层小波分解
    def __init__(self,length_input = 96, number_ouputs=4,mode='dot',factor=5, d_model=512, n_heads=8, d_layers=2,
                dropout=0.0, freq='h',output_attention = False,model='vgg',d_prob=0.5,batch_size1=32):
        super(conv2d_multiwave_v2_att22, self).__init__()   # 继承__init__功能 pytorch初始化参数
        if length_input % 8 != 0:
            aaa = length_input%8
            length_input = length_input + 8-(length_input%8)
            self.diff = 8-aaa  
        else:
            self.diff = 0
        self.len_input = length_input
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

        self.vgg = VGG('VGG11',num_class=self.num_outputs)
        self.vgg2 = VGG('VGG7',num_class=self.num_outputs)
        self.vgg3 =  VGG('VGG9',num_class=self.num_outputs)
        #self.vgg4 =  VGG('VGGn',num_class=self.num_outputs)
        self.fc = nn.Linear(in_features=64,out_features=self.num_outputs)
        self.output = nn.Linear(in_features=3*self.num_outputs, out_features=self.num_outputs)

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
        if self.model=='vgg':    
            low1 = low1.to(torch.float32)
            #low1=low1.unsqueeze(1)
            #print(low1.shape)
            #print(f"low1 size",low1.size())
            high1 = high1.to(torch.float32)
            #high1=high1.unsqueeze(1)
            #print(f"high1 size",high1.size())
            all_out1 = torch.cat([low1, high1],dim=3)#.to('cuda:0') [batch_size, 1, 2, len_input/2]
            ##print(f"all_out1",all_out1.size())
            all_out1 = self.vgg(all_out1)#[batch_size, num_output]
            ##print(f"all_out1 vgg",all_out1.size())
            low2 = low2.to(torch.float32)
            #low2=low2.unsqueeze(1)
            high2 = high2.to(torch.float32)
            #high2 = high2.unsqueeze(1)
            
            all_out2 = torch.cat([low2, high2],dim=3)#.to('cuda:0') [batch_size, 1, 2, len_input/4]
            ##print(f"all_out2",all_out2.size())
            all_out2 = self.vgg3(all_out2)#vgg1d3 [batch_size, num_output]
            ##print(f"all_out2 vgg3",all_out2.size())
            low3 = low3.to(torch.float32)
            #low3=low3.unsqueeze(1)
            high3 = high3.to(torch.float32)
            #high3 = high3.unsqueeze(1)
            all_out3 = torch.cat([low3, high3],dim=3)#.to('cuda:0') [batch_size, 1, 2, len_input/8]
            ##print(f"all_out3",all_out3.size())
            all_out3 = self.vgg2(all_out3) #[batch_size, num_output]
            ##print(f"all_out3 vgg2",all_out3.size())
            z=torch.cat([all_out1,all_out2,all_out3],dim=1)  #[batch_size, 3*num_output]     
            ##print("z",z.shape)
            #output_fun = nn.Linear(in_features=z.shape[1], out_features=self.num_outputs)
            #output_fun = output_fun.cuda()
            #output = output_fun(z) 
            #exit()
            output = self.output(z) #[bacth_size, num_output]    # 输出[16,2]
            ##print(f"output is",output.shape)
            return output
