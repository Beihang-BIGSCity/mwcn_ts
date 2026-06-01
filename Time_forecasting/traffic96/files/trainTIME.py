# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:04:51 2021

@author: 13621 other train

"""
import numpy as np 
import os
import sys
from base_module import *
import pickle as pickle
import argparse
import shutil
import pdb
import torch
import torch.nn as nn
import seaborn as sns
import pywt
import matplotlib.pyplot as plt
from math import *
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torchvision import datasets
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from modelswave import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from net_structure import *
from net_structure1 import *
from data import ucr_datasets
from ucr_datasets import *
#from data.uea_dataset import UEA_Dataset
#writer = SummaryWriter(log_dir='ucr_runs')
parser = argparse.ArgumentParser()
torch.manual_seed(1)
# 设置超参数
#learning_rate = 0.001
#dataset_name = 'AtrialFibrillation'
args = parser.parse_args()
def train_model(train_loader, model, optimizer,epoch, step_cnt):
    model.train()
    total = 0
    correct = 0
    losses = []
    for _, batch in enumerate(train_loader):     
        optimizer.zero_grad()
        batch_X = batch['feature']
        batch_y = batch['label'].reshape(-1).long()
        criterion = nn.CrossEntropyLoss()        
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        _, pred = logits.max(1)
        total += batch_y.size(0)
        correct += pred.eq(batch_y).sum().item()
        
        step_cnt['train'] += 1
        batch_acc = pred.eq(batch_y).sum().item() / batch_y.size(0)
        batch_loss = loss.item()
        
#        writer.add_scalar('Loss/train_batch', batch_loss, step_cnt['train'])
#        writer.add_scalar('Accuracy/train_batch', batch_acc, step_cnt['train'])
#        print("train_batch")
#        print(batch_acc)
    
    losses = np.array(losses)
    train_loss = losses.mean()  
    train_acc = correct / total
    
#    writer.add_scalar('Loss/train', train_loss, step_cnt['train'])
#    writer.add_scalar('Accuracy/train', train_acc, step_cnt['train'])
    print("train_acc:")
    print(train_acc)
        
    return train_loss, train_acc

def test_model(test_loader, model,epoch):
    model.eval()
    total = 0
    correct = 0
    losses = []
    for _, batch in enumerate(test_loader):  
        batch_X = batch['feature']
        batch_y = batch['label'].reshape(-1).long()
#        writer.add_graph(model, batch_X)
        criterion = nn.CrossEntropyLoss()        
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        losses.append(loss.item())
        _, pred = logits.max(1)
        total += batch_y.size(0)
        correct += pred.eq(batch_y).sum().item()
        
    losses = np.array(losses)
    test_loss = losses.mean()  
    test_acc = correct / total
    
#    writer.add_scalar('Loss/test', test_loss, epoch)
#    writer.add_scalar('Accuracy/test', test_acc, epoch)
    print("test_acc:") 
    print(test_acc, epoch)
    return test_loss, test_acc
batch_size = 16
epochs = 400
dataset_name = 'Adiac'
train_dataset = UCR_Dataset(train=True, 
                            root_dir='C:/Users/13621/Downloads/研究生学习资料/研究生课题/课题研究/UCR_Dataset/UCRArchive_2018/',
                            dataset_name = dataset_name)

test_dataset = UCR_Dataset(train=False,
                           root_dir='C:/Users/13621/Downloads/研究生学习资料/研究生课题/课题研究/UCR_Dataset/UCRArchive_2018/',
                           dataset_name = dataset_name)
batch_size = min(batch_size, train_dataset.features.shape[0] // 10)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          num_workers=0)
data_len = train_dataset.features.shape[2]
data_channels = train_dataset.features.shape[1]
n_class = len(set(train_dataset.labels.flatten().tolist()))
input_shape = (batch_size, data_channels, data_len)
try:
    os.makedirs(os.path.join('ucr_exp', dataset_name))
except:
    pass
model_name = 'Deep_Multiwavelet_model'
model_save_path = os.path.join('ucr_exp',
                               dataset_name,
                               model_name + '.pth')
for _, batch in enumerate(train_loader):     
    batch_X = batch['feature']
    batch_y = batch['label'].reshape(-1).long()
    print(batch_X.type())

model=multiCNNatt1(data_len,n_class,'dot',factor=5, d_model=data_len/2, n_heads=8, d_layers=2,
                dropout=0.0, freq='h',output_attention = False,model='cnn',d_prob=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
step_cnt = {'train': 0, 'test': 0}
max_test_acc = 0
train_logs = []
for epoch in range(epochs):    
    print("Epoch:", epoch)
    train_loss,train_acc=train_model(train_loader,model,optimizer,epoch,step_cnt)
    print("Train Loss: %.3f, Train Accuracy: %.3f" % (train_loss, train_acc) )
    test_loss, test_acc = test_model(test_loader, 
                                     model, 
                                     epoch)
    print("Test Loss: %.3f, Test Accuracy: %.3f" % (test_loss, test_acc))
    if max_test_acc < test_acc:
        max_test_acc = test_acc
        torch.save(model, model_save_path)
    print("Now max test acc:", max_test_acc)
    
    now_epoch_result = {
                         'epoch':epoch,
                         'train_loss':train_loss,
                         'train_acc':train_acc,
                         'test_loss':test_loss,
                         'test_acc':test_acc,
                         'max_test_acc':max_test_acc
                        }
    train_logs.append(now_epoch_result)



print("Final max test acc:", max_test_acc)
