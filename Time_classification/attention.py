# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:45:15 2021

@author: 13621
"""

import torch
import torch.nn as nn
import numpy as np


class dot_attention(nn.Module):
    """ 点积注意力机制"""

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播
        :param q:
        :param k:
        :param v:
        :param scale:
        :param attn_mask:
        :return: 上下文张量和attention张量。
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        # 是否设置缩放
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)     # 给需要mask的地方设置一个负无穷。
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和v做点积。
        context = torch.bmm(attention, v)
        return context, attention

"""
if __name__ == '__main__':
    q = torch.ones((1, 2, 512))
    k = torch.ones((1, 17, 512))
    v = k
    attention = dot_attention()
    context, attention = attention(q, k, v)
    print("context:", context.size(), context)
    print("attention:", attention)
"""