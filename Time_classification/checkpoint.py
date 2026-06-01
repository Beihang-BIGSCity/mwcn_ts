# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:24:01 2022

@author: 13621 check point
"""
import torch
from net_structure1 import *
import argparse

parser = argparse.ArgumentParser(description='PyTorch TimeClassification checking')
args = parser.parse_args()
LOG_DIR = './'
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')
def log_string(out_str, print_out=True):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    if print_out:
        print(out_str)

print("===> Loading model")
savedataroot = ['Adiac/','Adiac1/','Adiac2/','Adiacwave/','Adiacwave1/','CricketX/','CricketX1/','CricketX1multi/','CricketX1wave/','ECG200/','ECG2001/','ECGFiveDaysmulti/','ECGFiveDayswave/','FordAmulti/','FordAmulti0/','FordAwave/','FordAwave0/']
modelname = 'ckpt.t7'
for i in range(len(savedataroot)):
    checkpoint = torch.load(savedataroot[i] + modelname)
    print(savedataroot[i],":",checkpoint['acc'])  # 从字典中依次读取，具体值查看字典更改
    log_string('Dataset: %s | BestAcc: %.3f' %(savedataroot[i],checkpoint['acc']), False)
    print('===> Load last checkpoint data')

