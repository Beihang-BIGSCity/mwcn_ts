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
from models.pathformer.PathFormer import PathFormer
from pdb import set_trace
#from showdata import *
#from showdataecl import *
batch_size = 16
learning_rate = 0.0001

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class conv2d_multiwave_pa(nn.Module):
    def __init__(self, configs, length_input, time_step, output_size=96, mode='dot', factor=5, d_model=512, n_heads=8, d_layers=2,
                 dropout=0.0, freq='h', model='attention', d_prob=0.5, hidden_size=6, batch_size1=32):
        super(conv2d_multiwave_pa, self).__init__()
        if length_input % 16 != 0:
            aaa = length_input % 16
            length_input = length_input + 16 - (length_input % 16)
            self.diff = 16 - aaa
        else:
            self.diff = 0
        self.len_input = length_input
        self.output_size = output_size
        self.factor = factor
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.d_layers = d_layers
        self.dropout = nn.Dropout(dropout)
        self.batch_size = batch_size1
        self.model = model

        self.res_layer1 = nn.Linear(self.len_input, self.len_input // 2)
        self.res_layer2 = nn.Linear(self.len_input // 2, self.len_input // 4)
        self.res_layer3 = nn.Linear(self.len_input // 4, self.len_input // 8)
        self.res_layer4 = nn.Linear(self.len_input // 8, self.len_input // 16)

        self.features1 = Conv_mwavepre1d(is_training=True, len_input=self.len_input, sim_reg=0, activation=None)
        self.features2 = Conv_mwavedec2d(is_training=True, len_inputi=self.len_input, sim_reg=0, activation=None)
        self.features3 = Conv_mwavedec2d(is_training=True, len_inputi=self.len_input / 2, sim_reg=0, activation=None)
        self.features4 = Conv_mwavedec2d(is_training=True, len_inputi=self.len_input / 4, sim_reg=0, activation=None)
        self.se_block = se_block(in_channel=6)

        

        # self.attention1 = nn.MultiheadAttention(embed_dim=self.len_input / 4, num_heads=self.n_heads, dropout=dropout, batch_first=True)
        # self.attention2 = nn.MultiheadAttention(embed_dim=self.len_input / 4, num_heads=self.n_heads, dropout=dropout, batch_first=True)
        # self.attention3 = nn.MultiheadAttention(embed_dim=self.len_input / 4, num_heads=self.n_heads, dropout=dropout, batch_first=True)
        # [B, L, F]
        # self.embedding_layer = nn.Linear(4, hidden_size)
        configs.num_nodes = 4
        self.encoder1 = PathFormer(configs, self.len_input // 4, self.len_input // 4, hidden_size)
        self.encoder2 = PathFormer(configs, self.len_input // 4, self.len_input // 4, hidden_size)
        self.encoder3 = PathFormer(configs, self.len_input // 4, self.len_input // 4, hidden_size)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=4, out_features=4 * hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=4 * hidden_size, out_features=1),
        )
        self.output = nn.Linear(in_features=int(self.len_input / 4 + self.len_input / 4 + self.len_input / 4), 
                                out_features=self.output_size)

    def forward(self, x, mode='dot'):
        if self.diff != 0:
            btr = torch.zeros((x.shape[0], 1, self.diff), device=device)
            x = torch.cat((x, btr), 2)
        
        x = x.squeeze(1)  # [batch_size, len_input]
        x = x.float()
        # set_trace()
        low0 = self.features1(x) # [batch_size, 1, 2, len_input//2]
        low0 = low0 + self.res_layer1(x).reshape(x.shape[0], 1, 1, -1)

        low1, high1 = self.features2(low0) # [batch_size, 1, 2, len_input//4]
        low1 = low1 + self.res_layer2(low0).reshape(x.shape[0], 1, 2, -1)

        low2, high2 = self.features3(low1) # [batch_size, 1, 2, len_input//8]
        low2 = low2 + self.res_layer3(low1).reshape(x.shape[0], 1, 2, -1)
        
        low3, high3 = self.features4(low2) # [batch_size, 1, 2, len_input//16]
        low3 = low3 + self.res_layer4(low2).reshape(x.shape[0], 1, 2, -1)
        
        pad_size = (0, low1.shape[3] - low2.shape[3])
        pad_size1 = (0, low1.shape[3] - low3.shape[3])
        low2 = F.pad(low2, pad_size, mode='constant', value=0)
        high2 = F.pad(high2, pad_size, mode='constant', value=0)
        low3 = F.pad(low3, pad_size1, mode='constant', value=0)
        high3 = F.pad(high3, pad_size1, mode='constant', value=0)
        
        u = torch.cat([low1, high1, low2, high2, low3, high3], dim=1)
        u1 = self.se_block(u)
        low1 = u1[:, 0, :, :].unsqueeze(1)
        high1 = u1[:, 1, :, :].unsqueeze(1)
        low2 = u1[:, 2, :, :].unsqueeze(1)
        high2 = u1[:, 3, :, :].unsqueeze(1)
        low3 = u1[:, 4, :, :].unsqueeze(1)
        high3 = u1[:, 5, :, :].unsqueeze(1)
        
        all_out1 = torch.cat([low1, high1], dim=2).reshape(x.size(0), self.len_input // 4, -1)
        all_out2 = torch.cat([low2, high2], dim=2).reshape(x.size(0), self.len_input // 4, -1)
        all_out3 = torch.cat([low3, high3], dim=2).reshape(x.size(0), self.len_input // 4, -1)
        all_out1, loss1 = self.encoder1(all_out1)
        all_out2, loss2 = self.encoder2(all_out2)
        all_out3, loss3 = self.encoder3(all_out3)
        z = torch.cat([all_out1, all_out2, all_out3], dim=1)
        # set_trace()
        # z = z.permute(0, 2, 1)
        z = self.fc(z).squeeze(2)
        z = self.dropout(z)
        output = self.output(z)
        return output, loss1 + loss2 + loss3
