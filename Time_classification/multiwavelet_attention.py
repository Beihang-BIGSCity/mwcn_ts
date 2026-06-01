"""
multiwavelet attention
"""

import torch
import numpy as np
import math

def multiwavelet_att(matrix_coef,):
    length = matrix_coef.shape[0]
    energy_coef = matrix_coef * matrix_coef
    #multiwavelet_basis
    att = torch.tensor(length,length)#行先？横着的
    c = torch.tensor(2,length)#行先？横着的
    #对于每个index计算attention weight     j,k表示函数的尺度与平移
    for m in range(length):
        for n in range(length):
            j= powertranslate(length)
            #k=n
            if n<=math.floor(length/8):
                k = n
                f1 = phi1(2^(j-3)-k)
                f2 = phi2(2^(j-3)-k)
            elif math.floor(length/8)<n and n<=math.floor(length/4):
                k = n - math.floor(length/8)
                f1 = psi1(2^(j-3)-k)
                f2 = psi2(2^(j-3)-k)
            elif math.floor(length/4)<n and n<=math.floor(length/2):
                k = n - math.floor(length/4)
                f1 = psi1(2^(j-2)-k)
                f2 = psi2(2^(j-2)-k)
            elif math.floor(length/2)<n:
                k = n - math.floor(length/2)
                f1 = psi1(2^(j-1)-k)
                f2 = psi2(2^(j-1)-k)
            att[m][n] = matrix_coef[m]*matrix_coef[m]*(f1+f2)*0.5
        c[:,m]= torch.sum(energy_coef*att[m],axis=0)
    #把c拆开输出
    

def powertranslate():

def phi1():

def phi2():

def psi1():

def psi2():