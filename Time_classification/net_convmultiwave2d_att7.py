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
from multiwaveletconv2d import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace
#from showdata import *#品
#from showdataecl import *
batch_size = 16
learning_rate = 0.0001
from vgg_v2 import *
import torch.nn.functional as F
from SENet import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class conv2d_multiwave_att7(nn.Module):  # 三层小波分解
    def __init__(self, length_input=96, number_ouputs=4, mode='dot', factor=5, d_model=32,
                 n_heads=8, d_layers=2, dropout=0.0, freq='h', output_attention=False,
                 model='vgg', d_prob=0.5, batch_size1=32, ratio=4, num_layers=3):
        super(conv2d_multiwave_att7, self).__init__()

        if length_input % 16 != 0:
            aaa = length_input%16
            length_input = length_input + 16-(length_input%16)
            self.diff = 16-aaa  
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
        self.dropout=nn.Dropout(dropout)
        self.freq='h'
        self.output_attention=output_attention

        self.num_layers = num_layers
        self.len_input = length_input if length_input % 16 == 0 else length_input + 16 - (length_input % 16)
        self.diff = 0 if length_input % 16 == 0 else 16 - (length_input % 16)
        self.num_outputs = number_ouputs
        self.batch_size = batch_size1
        self.model = model
        self.dropout = nn.Dropout(dropout)

        # 权重参数
        W1 = torch.rand(12, 12)
        W2 = torch.rand(12, 12)
        self.W1 = torch.nn.Parameter(W1, requires_grad=True).to(device=0)
        self.W2 = torch.nn.Parameter(W2, requires_grad=True).to(device=0)

        # 特征提取模块：multiwavelet分解
        self.feature_layers = nn.ModuleList()
        len_inputs = [self.len_input // (2 ** i) for i in range(self.num_layers)]
        self.feature_layers.append(Conv_mwavepre1d(is_training=True, len_input=self.len_input, sim_reg=0, activation=None))
        for i in range(self.num_layers):
            self.feature_layers.append(Conv_mwavedec2d(is_training=True, len_inputi=len_inputs[i], sim_reg=0, activation=None))

        # 注意力模块
        self.se_block = se_block(in_channel=2 * self.num_layers, ratio=ratio)

        # VGG分类模块
        self.vgg_layers = nn.ModuleList([
            VGG('VGG11', num_class=self.num_outputs) if i > 0 else VGG('VGG9', num_class=self.num_outputs)
            for i in range(self.num_layers)
        ])

        self.output = nn.Linear(in_features=self.num_outputs * self.num_layers, out_features=self.num_outputs)

    def forward(self, x, mode='dot'):
        if self.diff != 0:
            pad = torch.zeros((x.shape[0], 1, self.diff), device=x.device)
            x = torch.cat((x, pad), dim=2)

        x = x.squeeze(1)  # [B, len_input]

        # 特征提取
        lows = []
        highs = []
        low = self.feature_layers[0](x)
        for i in range(self.num_layers):
            low, high = self.feature_layers[i + 1](low)
            lows.append(low.to(torch.float32))
            highs.append(high.to(torch.float32))
        # set_trace()
        # Padding (对齐尺寸)
        for i in range(1, self.num_layers):
            target_len = lows[0].shape[3]
            pad_len = target_len - lows[i].shape[3]
            pad_size = (0, pad_len)
            lows[i] = F.pad(lows[i], pad_size)
            highs[i] = F.pad(highs[i], pad_size)
        # set_trace()
        # Concatenate and apply SE
        u = torch.cat([lows[i] for i in range(self.num_layers)] + [highs[i] for i in range(self.num_layers)], dim=1)
        u1 = self.se_block(u)
        u1 = self.dropout(u1)

        # 拆分 back
        features = []
        for i in range(self.num_layers):
            low = u1[:, 2 * i, :, :].unsqueeze(1)
            high = u1[:, 2 * i + 1, :, :].unsqueeze(1)
            cat = torch.cat([low, high], dim=2)
            features.append(self.vgg_layers[i](cat))  # 每层对应一个VGG

        # 拼接特征 -> 线性输出
        z = torch.cat(features, dim=1)
        output = self.output(z)
        # print(f"output is",output.shape)
        return output
