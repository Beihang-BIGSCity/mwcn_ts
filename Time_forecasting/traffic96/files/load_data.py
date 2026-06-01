# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 21:21:51 2022

@author: 13621
"""
import numpy as np

#f = open(r"./LD2011_2014.txt",'r')

#s=f.read()

#print(s)
xy = np.loadtxt('./LD2011_2014.txt') # 使用numpy读取数据, delimiter=',', dtype=np.float32
print(xy)