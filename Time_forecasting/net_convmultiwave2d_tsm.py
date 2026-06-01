# # -*- coding: utf-8 -*-
# """
# Created on Sat Aug 26 13:11:32 2023

# @author: 13621
# """
from pdb import set_trace

# import torch
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import torch.autograd.variable as Variable
# import torch.utils.data as Data
# from multiwaveletconv2d import *
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from SENet import *
# from models.TSMixer.TSMixer import Model

# #from showdata import *
# #from showdataecl import *
# batch_size = 16
# learning_rate = 0.0001

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class conv2d_multiwave_tsm(nn.Module):
#     def __init__(self, configs, length_input, time_step, output_size=96, mode='dot', factor=5, d_model=512, n_heads=8, d_layers=2, ratio=4,
#                  dropout=0.0, freq='h', model='attention', d_prob=0.5, hidden_size=6, batch_size1=32):
#         super(conv2d_multiwave_tsm, self).__init__()
#         if length_input % 16 != 0:
#             aaa = length_input % 16
#             length_input = length_input + 16 - (length_input % 16)
#             self.diff = 16 - aaa
#         else:
#             self.diff = 0
#         self.len_input = length_input
#         self.output_size = output_size
#         self.factor = factor
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.hidden_size = hidden_size
#         self.d_layers = d_layers
#         self.dropout = nn.Dropout(dropout)
#         self.batch_size = batch_size1
#         self.model = model

#         self.res_layer1 = nn.Linear(self.len_input, self.len_input // 2)
#         self.res_layer2 = nn.Linear(self.len_input // 2, self.len_input // 4)
#         self.res_layer3 = nn.Linear(self.len_input // 4, self.len_input // 8)
#         self.res_layer4 = nn.Linear(self.len_input // 8, self.len_input // 16)

#         self.features1 = Conv_mwavepre1d(is_training=True, len_input=self.len_input, sim_reg=0, activation=None)
#         self.features2 = Conv_mwavedec2d(is_training=True, len_inputi=self.len_input, sim_reg=0, activation=None)
#         self.features3 = Conv_mwavedec2d(is_training=True, len_inputi=self.len_input / 2, sim_reg=0, activation=None)
#         self.features4 = Conv_mwavedec2d(is_training=True, len_inputi=self.len_input / 4, sim_reg=0, activation=None)
#         self.se_block = se_block(in_channel=6, ratio=ratio)

        

#         # self.attention1 = nn.MultiheadAttention(embed_dim=self.len_input / 4, num_heads=self.n_heads, dropout=dropout, batch_first=True)
#         # self.attention2 = nn.MultiheadAttention(embed_dim=self.len_input / 4, num_heads=self.n_heads, dropout=dropout, batch_first=True)
#         # self.attention3 = nn.MultiheadAttention(embed_dim=self.len_input / 4, num_heads=self.n_heads, dropout=dropout, batch_first=True)
#         # [B, L, F]
#         # self.embedding_layer = nn.Linear(4, hidden_size)
#         self.encoder1 = Model(configs, self.len_input // 4, self.len_input // 4, 4)
#         self.encoder2 = Model(configs, self.len_input // 4, self.len_input // 4, 4)
#         self.encoder3 = Model(configs, self.len_input // 4, self.len_input // 4, 4)
        
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=4, out_features=4 * hidden_size),
#             nn.ReLU(),
#             nn.Linear(in_features=4 * hidden_size, out_features=1),
#         )
#         self.output = nn.Linear(in_features=int(self.len_input / 4 + self.len_input / 4 + self.len_input / 4), 
#                                 out_features=self.output_size)

#     def forward(self, x, mode='dot'):
#         if self.diff != 0:
#             btr = torch.zeros((x.shape[0], 1, self.diff), device=device)
#             x = torch.cat((x, btr), 2)
        
#         x = x.squeeze(1)  # [batch_size, len_input]
#         x = x.float()
#         # set_trace()
#         low0 = self.features1(x) # [batch_size, 1, 2, len_input//2]
#         low0 = low0 + self.res_layer1(x).reshape(x.shape[0], 1, 1, -1)

#         low1, high1 = self.features2(low0) # [batch_size, 1, 2, len_input//4]
#         low1 = low1 + self.res_layer2(low0).reshape(x.shape[0], 1, 2, -1)

#         low2, high2 = self.features3(low1) # [batch_size, 1, 2, len_input//8]
#         low2 = low2 + self.res_layer3(low1).reshape(x.shape[0], 1, 2, -1)
        
#         low3, high3 = self.features4(low2) # [batch_size, 1, 2, len_input//16]
#         low3 = low3 + self.res_layer4(low2).reshape(x.shape[0], 1, 2, -1)
        
#         pad_size = (0, low1.shape[3] - low2.shape[3])
#         pad_size1 = (0, low1.shape[3] - low3.shape[3])
#         low2 = F.pad(low2, pad_size, mode='constant', value=0)
#         high2 = F.pad(high2, pad_size, mode='constant', value=0)
#         low3 = F.pad(low3, pad_size1, mode='constant', value=0)
#         high3 = F.pad(high3, pad_size1, mode='constant', value=0)
        
#         u = torch.cat([low1, high1, low2, high2, low3, high3], dim=1)
#         set_trace()
#         u1 = self.se_block(u)
#         low1 = u1[:, 0, :, :].unsqueeze(1)
#         high1 = u1[:, 1, :, :].unsqueeze(1)
#         low2 = u1[:, 2, :, :].unsqueeze(1)
#         high2 = u1[:, 3, :, :].unsqueeze(1)
#         low3 = u1[:, 4, :, :].unsqueeze(1)
#         high3 = u1[:, 5, :, :].unsqueeze(1)
        
#         all_out1_pre = torch.cat([low1, high1], dim=2).reshape(x.size(0), self.len_input // 4, -1)
#         all_out2_pre = torch.cat([low2, high2], dim=2).reshape(x.size(0), self.len_input // 4, -1)
#         all_out3_pre = torch.cat([low3, high3], dim=2).reshape(x.size(0), self.len_input // 4, -1)
#         all_out1 = self.encoder1(all_out1_pre)
#         all_out2 = self.encoder2(all_out2_pre)
#         all_out3 = self.encoder3(all_out3_pre)
#         z = torch.cat([all_out1, all_out2, all_out3], dim=1)
#         # set_trace()
#         # z = z.permute(0, 2, 1)
#         z = self.fc(z).squeeze(2)
#         z = self.dropout(z)
#         output = self.output(z)
#         return output, all_out1_pre, all_out2_pre, all_out3_pre, all_out1, all_out2, all_out3


import torch
import torch.nn as nn
import torch.nn.functional as F
from multiwaveletconv2d import *
from SENet import *
from models.TSMixer.TSMixer import Model

class conv2d_multiwave_tsm(nn.Module):
    def __init__(self, configs, length_input, time_step, output_size=96, mode='dot', factor=5, d_model=512, 
                 n_heads=8, d_layers=2, ratio=4, dropout=0.0, freq='h', model='attention', 
                 d_prob=0.5, hidden_size=6, batch_size1=32):
        super(conv2d_multiwave_tsm, self).__init__()
        
        # 处理输入长度对齐
        if length_input % 16 != 0:
            self.diff = 16 - (length_input % 16)
            length_input += self.diff
        else:
            self.diff = 0
            
        # 初始化参数
        self.len_input = length_input
        self.d_layers = d_layers
        self.output_size = output_size
        self.ref_len = self.len_input // 4  # 参考长度（第一次分解后的长度）
        self.hidden_size = hidden_size

        # 残差连接层列表
        self.res_layers = nn.ModuleList()
        
        # 预处理层
        self.features = nn.ModuleList()
        self.features.append(Conv_mwavepre1d(is_training=True, len_input=self.len_input, sim_reg=0, activation=None))
        self.res_layers.append(nn.Linear(self.len_input, self.len_input//2))

        # 动态生成小波分解层
        current_len = self.len_input // 2
        for _ in range(d_layers):
            self.features.append(Conv_mwavedec2d(is_training=True, len_inputi=current_len, sim_reg=0, activation=None))
            self.res_layers.append(nn.Linear(current_len, current_len//2))
            current_len = current_len // 2

        # SE模块（通道数=2*d_layers）
        self.se_block = se_block(in_channel=2*d_layers, ratio=ratio)

        # 动态生成TSMixer编码器
        self.encoders = nn.ModuleList()
        for _ in range(d_layers):
            self.encoders.append(Model(configs, self.ref_len, self.ref_len, 4))

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(4, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, 1),
        )
        self.output = nn.Linear(d_layers * self.ref_len, self.output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mode='dot'):
        # 输入预处理
        if self.diff != 0:
            x = F.pad(x, (0, self.diff), "constant", 0)
            
        x = x.squeeze(1).float()
        batch_size = x.size(0)

        # 存储各层输出
        outputs = []
        current = x
        
        # 处理预处理层
        low = self.features[0](current)
        res = self.res_layers[0](current).view(batch_size, 1, 1, -1).expand_as(low)
        low += res
        outputs.append(low)
        # set_trace()
        # 动态处理小波分解层
        old_low = low
        for i in range(1, len(self.features)):
            # 小波分解
            new_low, new_high = self.features[i](old_low)
            
            # 残差连接
            res = self.res_layers[i](old_low).view(new_low.shape)
            new_low += res
            
            # 保存输出
            outputs.extend([new_low, new_high])
            old_low = new_low

        # 填充对齐并拼接特征
        padded_features = []
        for i in range(1, len(outputs), 2):
            # 每层包含low和high
            low = outputs[i]
            high = outputs[i+1] if (i+1) < len(outputs) else 0
            
            # 统一填充到参考长度
            pad_size = self.ref_len - low.shape[-1]
            if pad_size > 0:
                low = F.pad(low, (0, pad_size), "constant", 0)
                high = F.pad(high, (0, pad_size), "constant", 0) if (i+1) < len(outputs) else high
                
            # 收集有效特征
            if i > 0:  # 跳过预处理层的low
                padded_features.extend([low, high])
        # set_trace()
        # 通道拼接和SE处理
        u = torch.cat([f.unsqueeze(1) for f in padded_features], dim=1)
        u = u.squeeze(2)
        u = self.se_block(u)

        # 分割特征并处理编码器
        # set_trace()
        encoder_inputs = []
        for i in range(0, u.size(1), 2):
            # 获取当前层的low和high
            current_low = u[:, i]
            current_high = u[:, i+1] if (i+1) < u.size(1) else 0
            
            # 维度重组并保存
            combined = torch.cat([current_low, current_high], dim=2)
            encoder_inputs.append(combined.view(batch_size, self.ref_len, -1))

        # 动态处理编码器
        encoder_outputs = []
        for i, enc in enumerate(self.encoders):
            if i < len(encoder_inputs):
                encoder_outputs.append(enc(encoder_inputs[i]))

        # 输出处理
        z = torch.cat(encoder_outputs, dim=1)
        z = self.fc(z).squeeze(-1)
        z = self.dropout(z)
        z = self.output(z)
        return z