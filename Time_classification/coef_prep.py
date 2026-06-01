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
    elif op.eq(pflt,'OP'):      #OP多小波用正交预滤波
         PR=np.array([[-0.0036,0.00015,0.9928,-0.0849,-0.0036,-0.0844],[0.0844,-0.0036,0.0849,0.9928,0.00015,0.0036]])
         PO=np.array([[-0.0036,0.00015,0.9928,-0.0849,-0.0036,-0.0844],[0.0844,-0.0036,0.0849,0.9928,0.00015,0.0036]])
    elif op.eq(pflt,'ghmorap'):   #ghm多小波用正交逼近预滤波
        PR=np.array([[0.11942337067748,0.99158171438258,0.04967860804828,-0.00598315472909],
           [-0.00598315472909,-0.04967860804828,0.9915817143825,-0.11942337067748]])
        PO=np.array([[0.11942337067748,0.99158171438258,0.04967860804828,-0.00598315472909],
           [-0.00598315472909,-0.04967860804828,0.9915817143825,-0.11942337067748]])
        PO[:,0:1] = PR[:,2:3].T
        PO[:,2:3] = PR[:,0:1].T

    elif op.eq(pflt,'clap'):      #cl多小波用双正交插值预滤波
        PR=np.array([[1/4,            1/4],
            [1/(1+mt.sqrt(7)), -1/(1+mt.sqrt(7))]])
        PO=np.array([[2,  (1+mt.sqrt(7))/2],
            [2, -(1+mt.sqrt(7))/2]])

    elif op.eq(pflt,'bih5ap'):   #双正交h5多小波用双正交插值预滤波
        PR=np.array([[1/4,  1/4],
             [-1, 1]])
        PO=np.array([2, -1/2,2,  1/2])

    elif op.eq(pflt,'bih3ap'):    #双正交h3多小波用双正交插值预滤波
        PR=np.array([[1/4,  1/4],
          [-1/15, 1/15]])
        PO=np.array([2, -15/2,2,  15/2])

    #elif op.eq(pflt,'sa4ap'):      #sa4多小波用正交预滤波
    #    PR=np.array([[1,1],[-1,1]])/mt.sqrt(2)
    #    PO=np.array([[1,-1],[1,1]])/mt.sqrt(2)

    elif op.eq(pflt,'bighm2ap'):    #双正交ghm多小波用双正交插值预滤波
        PR=np.array([[1/4,  1/4],[1/3, -1/3]])
        PO=np.array([2,  3/2,2,  -3/2])

    elif op.eq(pflt,'id'):           #平衡多小波，用于平衡多小波
        PR=np.array([[1, 0],[0, 1]])
        PO=np.array([[1, 0],[0, 1]])
    elif op.eq(pflt,'minial'):      # for GHM multiwavelets
        PR=np.array([[2*mt.sqrt(2) -mt.sqrt(2)],  [1,           0 ]]);
        PO=np.array([[0,           1],[-1/mt.sqrt(2),  2 ]])
    
    elif op.eq(pflt,'orap'):       # for GHM multiwavelets  the result seems bad 
        PR=(mt.sqrt(6)/6)*np.array([[1-mt.sqrt(2), 1+mt.sqrt(2)],[1+mt.sqrt(2), -1+mt.sqrt(2)]])
        PO=np.linalg.inv(PR)
    return PR,PO

