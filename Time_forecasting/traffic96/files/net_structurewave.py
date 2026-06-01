# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:46:17 2022

@author: 13621 net_structure wave
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:53:54 2021

@author: 13621 选mode concat or add  mode2 loss or add classifier loss
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
from waveletmodule import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *
batch_size = 16
learning_rate = 0.0001
from vgg1d import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class waveCNNatt1(nn.Module):#三层小波分解
    def __init__(self,length_input = 96, number_ouputs=4,mode='dot',factor=5, d_model=512, n_heads=8, d_layers=2,
                dropout=0.0, freq='h',output_attention = False,model='cnn',d_prob=0.5):
        super(waveCNNatt1, self).__init__()   # 继承__init__功能
        if length_input % 8 != 0:
            aaa = length_input%8
            length_input = length_input + 8-(length_input%8)
            self.diff = 8-aaa  
        else:
            self.diff = 0
        self.len_input = length_input
        print("hello")
        print(length_input)
        print(self.diff)
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
        self.features1 = Conv_wave(is_training=True,len_input=self.len_input,sim_reg=0,activation=None)
        self.features2 = Conv_wave(is_training=True,len_input=self.len_input/2,sim_reg=0,activation=None)
        self.features3 = Conv_wave(is_training=True,len_input=self.len_input/4,sim_reg=0,activation=None)
#        self.freembedding = FixedEmbedding(enc_in, d_model) 
#        self.freatten =  AttentionLayer(ProbAttention(False, self.factor, attention_dropout=self.dropout, output_attention=self.output_attention), 
#                                d_model, n_heads)
        self.attn = dot_attention()
#        self.resnet = ResNet1D(in_channels=1, base_filters=32,kernel_size=kernel_size,stride=stride,groups=4,n_block=n_block,n_classes=number_ouputs,downsample_gap=downsample_gap,increasefilter_gap=increasefilter_gap,use_do=True)# 64 for ResNet1D, 352 for ResNeXt1D
        self.conv2d1 = nn.Conv2d(
                in_channels=2,    # 输入图片的高度
                out_channels=32,  # 输出图片的高度
                kernel_size=[5,1],    # 5x5的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=2,        # 给图外边补上0
                )
            # 经过卷积层 输出[16,len_input] 传入池化层
        self.re =nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=2)
        self.maxp1 = nn.MaxPool2d(kernel_size=1)
        ## 第二层卷积
        
        self.conv2d2=nn.Conv2d(
                in_channels=32,    # 同上
                out_channels=64,
                kernel_size=[5,1],
                stride=1,
                padding=2
            )

        self.conv2d3 =nn.Conv2d(
                in_channels=64,    # 同上
                out_channels=128,
                kernel_size=[17,1],
                stride=1,
                padding=2
            )

        self.vgg1d = VGG1d('VGG11',num_class=self.num_outputs)
        self.vgg1d2 = VGG1d('VGG7',num_class=self.num_outputs)
        self.vgg1d3 =  VGG1d('VGG9',num_class=self.num_outputs)
        self.vgg1d4 =  VGG1d('VGGn',num_class=self.num_outputs)
#        self.output = nn.Linear(in_features=32*(int(self.len_input/8))*2, out_features=self.num_outputs)
        self.fc = nn.Linear(in_features=64,out_features=self.num_outputs)
        self.output = nn.Linear(in_features=111, out_features=self.num_outputs)
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
        x = x.squeeze()
        low1,high1= self.features1(x)#yz为第一层的系数，low2,high2为第二层，low3,high3为第三层
        low1 = low1#.to('cuda:0')
        high1 = high1#.to('cuda:0')
        low2,high2 = self.features2(low1)
       
#print(low2.shape)
        low2 = low2#.to('cuda:0')
        high2 = high2#.to('cuda:0')
        low3,high3 = self.features3(low2)
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
        elif self.model=='cnn':    
            low1 = low1.to(torch.float32)
            low1=low1.unsqueeze(-1)
            low1=low1.unsqueeze(-1)

#            print(low1.shape)
            high1 = high1.to(torch.float32)
            high1=high1.unsqueeze(-1)
            high1=high1.unsqueeze(-1)
            all_out1 = torch.cat([low1, high1],dim=-1).to('cuda:0')
            all_out1 = all_out1.permute(0,3,1,2).to('cuda:0')
            all_out1 = self.conv2d1(all_out1)
            all_out1 = self.re(all_out1)
            all_out1 = self.maxp(all_out1)
            all_out1 = self.conv2d2(all_out1)
            all_out1 = self.re(all_out1)
            all_out1 = self.maxp(all_out1)
            all_out1 = self.conv2d3(all_out1)
            all_out1 = self.re(all_out1)
            all_out1 = self.maxp(all_out1)
            all_out1 = all_out1.contiguous().view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*24]
            low2 = low2.to(torch.float32)
            low2=low2.unsqueeze(-1)
            low2=low2.unsqueeze(-1)
            high2 = high2.to(torch.float32)
            high2 = high2.unsqueeze(-1)
            high2 = high2.unsqueeze(-1)
            all_out2 = torch.cat([low2, high2],dim=-1).to('cuda:0')
            all_out2 = all_out2.permute(0,3,1,2).to('cuda:0')
            all_out2 = self.conv2d1(all_out2)
            all_out2 = self.re(all_out2)
            all_out2 = self.maxp(all_out2)
            all_out2 = self.conv2d2(all_out2)           # [batch, 32,48]
            all_out2 = self.re(all_out2)
            all_out2 = self.maxp(all_out2)
            all_out2 = self.conv2d3(all_out2)
            all_out2 = self.re(all_out2)
            all_out2 = self.maxp(all_out2)
            all_out2 = all_out2.contiguous().view(x.size(0), -1)  

#            low3 = low3.to(torch.float32)
            low3=low3.unsqueeze(-1)
            low3=low3.unsqueeze(-1)
            high3 = high3.to(torch.float32)
            high3 = high3.unsqueeze(-1)
            high3 = high3.unsqueeze(-1)
            all_out3 = torch.cat([low3, high3],dim=-1).to('cuda:0')
            all_out3 = all_out3.permute(0,3,1,2).to('cuda:0')
            all_out3 = self.conv2d1(all_out3)
            all_out3 = self.re(all_out3)
            all_out3 = self.maxp(all_out3)
            all_out3 = self.conv2d2(all_out3) # [batch, 32,48]
            all_out3 = self.re(all_out3)
            all_out3 = self.maxp(all_out3)
#            all_out3 = self.conv2d3(all_out3)
#            all_out3 = self.re(all_out3)
#            all_out3 = self.maxp(all_out3)
            all_out3 = all_out3.contiguous().view(x.size(0), -1)
            #print(all_out3.shape)
            z=torch.cat([all_out1,all_out2,all_out3],dim=1)
           
            #print(z.shape)
            output = self.output(z)     # 输出[16,2]
            return output
        elif self.model=='vgg1d':    
            low1 = low1.to(torch.float32)
            low1=low1.unsqueeze(1)

#            print(low1.shape)
            high1 = high1.to(torch.float32)
            high1=high1.unsqueeze(1)
            all_out1 = torch.cat([low1, high1],dim=1)#.to('cuda:0')
            all_out1 = self.vgg1d(all_out1)
            low2 = low2.to(torch.float32)
            low2=low2.unsqueeze(1)
            high2 = high2.to(torch.float32)
            high2 = high2.unsqueeze(1)
            all_out2 = torch.cat([low2, high2],dim=1)#.to('cuda:0')
            all_out2 = self.vgg1d3(all_out2)

#            low3 = low3.to(torch.float32)
            low3=low3.unsqueeze(1)
            high3 = high3.to(torch.float32)
            high3 = high3.unsqueeze(1)
            all_out3 = torch.cat([low3, high3],dim=1)#.to('cuda:0')
            all_out3 = self.vgg1d2(all_out3)
#            all_out3 = self.conv2d3(all_out3)
#            all_out3 = self.re(all_out3)
#            all_out3 = self.maxp(all_out3)
#            print(all_out3.shape)
            z=torch.cat([all_out1,all_out2,all_out3],dim=1)
            
            #print(z.shape)
            output = self.output(z)     # 输出[16,2]
            return output