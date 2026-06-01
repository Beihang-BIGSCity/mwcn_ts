# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:35:18 2021

@author: 13621 net_structure
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
class multiCNNatt(nn.Module):#三层小波分解
    def __init__(self,length_input = 96, number_ouputs=4,mode='dot',factor=5, d_model=512, n_heads=8, d_layers=2,
                dropout=0.0, freq='h',output_attention = False,model='cnn'):
        super(multiCNNatt, self).__init__()   # 继承__init__功能
        if length_input % 8 != 0:
            length_input = length_input + 8-(length_input%8)
            self.diff = 8-(length_input%8)   
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
        self.model = model
        kernel_size = 32
        stride = 2
        n_block = 12
        downsample_gap = 6
        increasefilter_gap = 6
        ## 第一层卷积
        self.features1 = Conv_multiwaveL(is_training=True,len_input=self.len_input,sim_reg=0,activation=torch.sigmoid)
        ## 输出层
        self.features2 = Conv_multiwaveH(is_training=True,len_input=self.len_input,sim_reg=0,activation=torch.sigmoid)
        self.features3 = wavecontinue(is_training=True,len_inputi=self.len_input/2,sim_reg=0,activation=torch.sigmoid)
        self.features4 = wavecontinue(is_training=True,len_inputi=self.len_input/4,sim_reg=0,activation=torch.sigmoid)
#        self.freembedding = FixedEmbedding(enc_in, d_model)
#        self.freatten =  AttentionLayer(ProbAttention(False, self.factor, attention_dropout=self.dropout, output_attention=self.output_attention), 
#                                d_model, n_heads)
        self.attn = dot_attention()
        self.resnet = ResNet1D(in_channels=1, base_filters=64,kernel_size=kernel_size,stride=stride,groups=8,n_block=n_block,n_classes=number_ouputs,downsample_gap=downsample_gap,increasefilter_gap=increasefilter_gap,use_do=True)# 64 for ResNet1D, 352 for ResNeXt1D
        self.conv1 = nn.Sequential(
            # 输入[1,len_input]
            nn.Conv1d(
                in_channels=1,    # 输入图片的高度
                out_channels=16,  # 输出图片的高度
                kernel_size=5,    # 1x5的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=2,        # 给图外边补上0
            ),
            # 经过卷积层 输出[16,len_input] 传入池化层
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # 经过池化 输出[16,len_input/2] 传入下一个卷积
        )
        ## 第二层卷积
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,    # 同上
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 经过卷积 输出[32, len_input/2] 传入池化层
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 经过池化 输出[32,len_input/4] 传入输出层
        )
#        print(32*(int(self.len_input/8))*2)
        
#        self.output = nn.Linear(in_features=32*(int(self.len_input/8))*2, out_features=self.num_outputs)
        self.fc = nn.Linear(in_features=256,out_features=self.num_outputs)
        self.output = nn.Linear(in_features=2432, out_features=self.num_outputs)
       
        
#    def _make_layers(self):
#        lp_coe, hp_coe = Conv_wave(len_input=96, is_training=True, l1_value=0, weight_decay=0, sim_reg=0)
#        return lp_coe,hp_coe
#        self.atten = AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
#                                d_model, n_heads)
#        self.embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
    def forward(self, x,mode='dot'):
        if self.diff !=0:
            btr = np.zeros((16,1,self.diff))
            ad1 = torch.from_numpy(btr)
            x = torch.cat((x,ad1),2)
        low1= self.features1(x)#yz为第一层的系数，low2,high2为第二层，low3,high3为第三层
        high1= self.features2(x)
        low2,high2 = self.features3(low1)
        low3,high3 = self.features4(low2)
        cat1 = torch.cat([low1, high1], dim=2)
        cat2 = torch.cat([low2, high2], dim=2)
        cat3 = torch.cat([low3, high3], dim=2)
#        l0 = torch.cat([Low2,high2, z], dim=2)
#        print(Low2.shape,high2.shape,z.shape)#多尺度形式创
#        if mode == 'self':
#            b1 = l0
#            b1 = self.freembedding(l0)#去掉一个参数,x_mark_enc
#            attn1 = self.atten(ProbAttention(False,factor,attention_dropout = dropout,output_attention=output_attention),d_model,n_heads)
#            enc_self_mask = None
#            attn1 = self.freatten(b1,b1,b1,attn_mask=enc_self_mask)
#            y = attn1
        if mode == 'dot':
            cat1 = cat1.permute(2,1,0)#l0改的
            l1,attention = self.attn(cat1,cat1,cat1)
            l1 = l1.permute(2,1,0)
            spl1=torch.split(l1,int(l1.shape[2]/2),dim=2)
#            print(l1.shape)
            low1 = spl1[0]
#            print(low1.shape)
            high1 = spl1[1]
            cat2 = cat2.permute(2,1,0)#l0改的
            l2,attention = self.attn(cat2,cat2,cat2)
            l2 = l2.permute(2,1,0)
            spl2=torch.split(l2,int(l2.shape[2]/2),dim=2)
#            print(l1.shape)
            low2 = spl2[0]
#            print(low1.shape)
            high2 = spl2[1]
            cat3 = cat3.permute(2,1,0)#l0改的
            l3,attention = self.attn(cat3,cat3,cat3)
            l3 = l3.permute(2,1,0)
            spl3=torch.split(l3,int(l3.shape[2]/2),dim=2)
#            print(l1.shape)
            low3 = spl3[0]
#            print(low1.shape)
            high3 = spl3[1]
        if self.model == 'res':
            low1 = low1.to(torch.float32)
            low1=self.resnet(low1)
            high1 = high1.to(torch.float32)
            high1=self.resnet(high1)
            out = torch.cat([low1, high1], dim=1)
            out = out.squeeze()
            out1 = self.fc(out)
            return out1
        elif self.model=='cnn':    
            low1 = low1.to(torch.float32)
            low1=  self.conv1(low1)
            low1 = self.conv2(low1)           # [batch, 32,48]
            low1 = low1.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*24]  
            high1 = high1.to(torch.float32)
            high1 = self.conv1(high1)
            high1 = self.conv2(high1)           # [batch, 32,48]
            high1 = high1.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*24]
            y1 = torch.cat([low1, high1], dim=1)
            low2 = low2.to(torch.float32)
            low2=  self.conv1(low2)
            low2 = self.conv2(low2)           # [batch, 32,48]
            low2 = low2.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*24]  
            high2 = high2.to(torch.float32)
            high2 = self.conv1(high2)
            high2 = self.conv2(high2)           # [batch, 32,48]
            high2 = high2.view(high2.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*24]
            y2 = torch.cat([low2, high2], dim=1)
            low3 = low3.to(torch.float32)
            low3=  self.conv1(low3)
            low3 = self.conv2(low3)           # [batch, 32,48]
            low3 = low3.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*24]  
            high3 = high3.to(torch.float32)
            high3 = self.conv1(high3)
            high3 = self.conv2(high3)           # [batch, 32,48]
            high3 = high3.view(high3.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*24]
            y3 = torch.cat([low3, high3], dim=1)
            z=torch.cat([y1,y2,y3],dim=1)
            output = self.output(z)     # 输出[16,2]
            return output
"""            
        

"""
# 搭建CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()   # 继承__init__功能
        ## 第一层卷积
        self.features1 = Conv_waveL(is_training=True,l1_value=0, weight_decay=0,len_input=96,sim_reg=0,activation=None)
        ## 输出层
        self.features2 = Conv_waveH(is_training=True,l1_value=0, weight_decay=0,len_input=96,sim_reg=0,activation=None)
        self.conv1 = nn.Sequential(
            # 输入[1,96]
            nn.Conv1d(
                in_channels=1,    # 输入图片的高度
                out_channels=16,  # 输出图片的高度
                kernel_size=5,    # 1x5的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=2,        # 给图外边补上0
            ),
            # 经过卷积层 输出[16,96] 传入池化层
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # 经过池化 输出[16,48] 传入下一个卷积
        )
        ## 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,    # 同上
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 经过卷积 输出[32, 48] 传入池化层
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 经过池化 输出[32,24] 传入输出层
        )
       
        self.output = nn.Linear(in_features=32*12, out_features=2)
#    def _make_layers(self):
#        lp_coe, hp_coe = Conv_wave(len_input=96, is_training=True, l1_value=0, weight_decay=0, sim_reg=0)
#        return lp_coe,hp_coe
    def forward(self, x):
        y,p= self.features1(x)
#        print(p)
        z= self.features2(x)
#        print(x)
        y = self.conv1(y)
        y = self.conv2(y)           # [batch, 32,48]
        y = y.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*24]
        z = self.conv1(z)
        z = self.conv2(z)           # [batch, 32,48]
        z = z.view(z.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*24]
        z = z + y
#        print(x.shape)
        output = self.output(z)     # 输出[16,2]
        return output

class multiCNN(nn.Module):
    def __init__(self,):
        super(multiCNN, self).__init__()   # 继承__init__功能
        ## 第一层卷积

        self.features1 = Conv_multiwaveL(is_training=True,len_input=96,sim_reg=0,activation=None)
        ## 输出层
        self.features2 = Conv_multiwaveH(is_training=True,len_input=96,sim_reg=0,activation=None)
        self.conv1 = nn.Sequential(
            # 输入[1,96]
            nn.Conv1d(
                in_channels=1,    # 输入图片的高度
                out_channels=16,  # 输出图片的高度
                kernel_size=5,    # 1x5的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=2,        # 给图外边补上0
            ),
            # 经过卷积层 输出[16,96] 传入池化层
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # 经过池化 输出[16,48] 传入下一个卷积
        )
        ## 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,    # 同上
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 经过卷积 输出[32, 48] 传入池化层
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 经过池化 输出[32,24] 传入输出层
        )
       
        self.output = nn.Linear(in_features=32*12, out_features=2)
#    def _make_layers(self):
#        lp_coe, hp_coe = Conv_wave(len_input=96, is_training=True, l1_value=0, weight_decay=0, sim_reg=0)
#        return lp_coe,hp_coe
    def forward(self, x):
        print(x.size())
        y= self.features1(x)
        print("shape")
        print(y.size())
#        print(p)
        z= self.features2(x)
#        print(x)
        y = self.conv1(y)
        y = self.conv2(y)           # [batch, 32,48]
        y = y.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*24]
        z = self.conv1(z)
        z = self.conv2(z)           # [batch, 32,48]
        z = z.view(z.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*24]
        z = z + y
#        print(x.shape)
        output = self.output(z)     # 输出[16,2]
        return output
    

"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #输入为[1,96]
        self.conv1=nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=4,
            ),
            #经过卷积层输出为[16,96]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)#经过池化层输出为[16,48]
        )
        self.conv2=nn.Sequential(
             nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
            ),#输出为[32,48]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)##经过池化层输出为[32,24]
        )
        self.fc=nn.Linear(32*24,2)

    def forward(self, x):
        x=self.conv1(x),
        x=self.conv2(x),
        x=x.view(x.size(0),-1),
        return self.fc(x)
"""