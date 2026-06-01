# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:14:22 2022

@author: 13621 net_structure predict
"""


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.autograd.variable as Variable
import torch.utils.data as Data
from waveletmodel import *
from multiwavelet import *
#from selfattn import *
#from embed import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *
#from utils.masking import TriangularCausalMask, ProbMask
#from selfattn import  ProbAttention, AttentionLayer
#from embed import DataEmbedding
#from model.resnet1d import *
batch_size = 16
learning_rate = 0.0001

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class netlstm(nn.Module):#三层小波分解
    def __init__(self, length_input,time_step,output_size=96,mode='dot',factor=5, n_heads=8, d_layers=2,
                dropout=0.0, model='cnn',d_prob=0.5):
        super(netlstm, self).__init__()   # 继承__init__功能
        if length_input % 8 != 0:
            aaa = length_input%8
            length_input = length_input + 8-(length_input%8)
            self.diff = 8-aaa  
        else:
            self.diff = 0
        self.len_input = time_step
        enc_in = 7
        embed = 'fixed'
        self.time_step = time_step
        self.output_size = output_size
        self.factor=factor
        self.n_heads=n_heads
        self.d_layers=d_layers
        self.dropout=dropout
        self.model = model
        kernel_size = 8
        stride = 2
        n_block = 3
        downsample_gap = 6
        increasefilter_gap = 6
        ## 第一层卷积
        self.features1 = Conv_multiwavenew(is_training=True,len_input=self.len_input,sim_reg=0,activation=None)
        ## 输出层
#        self.features2 = Conv_multiwaveH(is_training=True,len_input=self.len_input,sim_reg=0,activation=torch.sigmoid)
        self.features3 = wavecontinue(is_training=True,len_inputi=self.len_input/2,sim_reg=0,activation=None)
        self.features4 = wavecontinue(is_training=True,len_inputi=self.len_input/4,sim_reg=0,activation=None)
        self.embed1 = nn.LSTM(input_size=int(time_step), hidden_size=6, batch_first=True,bidirectional=True)#本input 任意的送进的（非，是一个模型） 需要送进的 第一个为本
        self.embed2 = nn.LSTM(input_size=int(time_step/2), hidden_size=6, batch_first=True,bidirectional=True)
        self.embed3 = nn.LSTM(input_size=int(time_step/4), hidden_size=6, batch_first=True,bidirectional=True)#-2
#        self.freembedding = FixedEmbedding(enc_in, d_model)
#        self.freatten =  AttentionLayer(ProbAttention(False, self.factor, attention_dropout=self.dropout, output_attention=self.output_attention), 
#                                d_model, n_heads)
        self.attn = dot_attention()
#        self.resnet = ResNet1D(in_channels=1, base_filters=32,kernel_size=kernel_size,stride=stride,groups=4,n_block=n_block,n_classes=number_ouputs,downsample_gap=downsample_gap,increasefilter_gap=increasefilter_gap,use_do=True)# 64 for ResNet1D, 352 for ResNeXt1D
#        self.output = nn.Linear(in_features=32*(int(self.len_input/8))*2, out_features=self.num_outputs)
        self.fc = nn.Linear(in_features=64,out_features=self.output_size)
        self.output = nn.Linear(in_features=36, out_features=self.output_size)
#        self.atten = AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
#                                d_model, n_heads)
#        self.embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
    def forward(self, x,mode='dot'):
        x= x#.to('cuda:0')
        if self.diff !=0:
            if x.shape[0]==16:
                btr = np.zeros((16,1,self.diff))
                ad1 = torch.from_numpy(btr).to(device=0)
                x = torch.cat((x,ad1),2)
            elif x.shape[0]!=16:
                btr = np.zeros((x.shape[0],1,self.diff))
                ad1 = torch.from_numpy(btr).to(device=0)
                x = torch.cat((x,ad1),2)
        #x = x.squeeze()
        low1,high1= self.features1(x)#yz为第一层的系数，low2,high2为第二层，low3,high3为第三层
        low1 = low1#.to('cuda:0')
        high1 = high1#.to('cuda:0')
        low2,high2 = self.features3(low1)
       
#print(low2.shape)
        low2 = low2#.to('cuda:0')
        high2 = high2#.to('cuda:0')
        low3,high3 = self.features4(low2)
        low3=low3#.to('cuda:0')
        high3 = high3#.to('cuda:0')
#        print(low3.shape)
#        cat1 = torch.cat([low1, high1], dim=2)
#        cat2 = torch.cat([low2, high2], dim=2)
#        cat3 = torch.cat([low3, high3], dim=2)
#        l0 = torch.cat([Low2,high2, z], dim=2)
#        print(Low2.shape,high2.shape,z.shape)#多尺度形式创
#        if mode == 'self':
#            b1 = l0
#            b1 = self.freembedding(l0)#去掉一个参数,x_mark_enc
#            attn1 = self.atten(ProbAttention(False,factor,attention_dropout = dropout,output_attention=output_attention),d_model,n_heads)
#            enc_self_mask = None
#            attn1 = self.freatten(b1,b1,b1,attn_mask=enc_self_mask)
#            y = attn1
#后段代码本身所在位置

        if self.model == 'res':
            low1 = low1.to(torch.float32)
            low1=self.resnet(low1)
            high1 = high1.to(torch.float32)
            high1=self.resnet(high1)
            out = torch.cat([low1, high1], dim=1)
            out = out.squeeze()
            out1 = self.fc(out)
            return out1
            return output
        elif self.model=='lstm1':    #input与参数不能fujinqu？
            low1 = low1.to(torch.float32)
            low1=low1.unsqueeze(0)
            #print(type(all_out1))
            high1 = high1.to(torch.float32)
            high1=high1.unsqueeze(0)
            all_out1 = torch.cat([low1, high1],dim=2)#.to('cuda:0')
            #all_out1 =  all_out1.permute(2,0,1)
            all_out1,__ = self.embed1(all_out1)
            low2 = low2.to(torch.float32)
            low2=low2.unsqueeze(0)
            high2 = high2.to(torch.float32)
            high2 = high2.unsqueeze(0)
            all_out2 = torch.cat([low2, high2],dim=2)#.to('cuda:0')
            all_out2,__ = self.embed2(all_out2)#vgg3子第三层 这数与embed数同不一定
            low3 = low3.to(torch.float32)
            low3=low3.unsqueeze(0)
            high3 = high3.to(torch.float32)
            high3 = high3.unsqueeze(0)#周代码写的复杂 用它中的预测框架 那不用他的预测框架 自混？
            all_out3 = torch.cat([low3, high3],dim=2)#.to('cuda:0')
            all_out3,__ = self.embed3(all_out3)
#            all_out3 = self.conv2d3(all_out3)
#            all_out3 = self.re(all_out3)
#            all_out3 = self.maxp(all_out3)
            
            z = torch.cat([all_out1,all_out2,all_out3],dim=2)
            #print("z shape:",z.shape)
            z=z.squeeze(0)#有地方等 有地方不等品
            output = self.output(z)     # 输出[16,2]
            #print("output shape:",output.shape)
            return output


