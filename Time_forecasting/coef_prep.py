# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:56:39 2021

@author: 13621
"""
import numpy as np
import operator as op
import math as mt
def coef_prep(pflt):
    if op.eq(pflt,'ghmap'):        #ghm多小波用双正交插值预滤波
        PR=np.array([[3/(8*mt.sqrt(2)), 10/(8*mt.sqrt(2)), 3/(8*mt.sqrt(2)), 0],[0,             0,              1,             0]])
        PO=np.array([[0,  1,    0,             0],[0, -3/10, 4*mt.sqrt(2)/5, -3/10]])



    elif op.eq(pflt,'sa4ap'):      #sa4多小波用正交预滤波
         PR=np.array([[1,1],[-1,1]])/mt.sqrt(2)
         PO=np.array([[1,-1],[1,1]])/mt.sqrt(2)
    return PR,PO

