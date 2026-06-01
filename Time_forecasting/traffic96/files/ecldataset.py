# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:28:02 2022

@author: 13621
"""
"""
def generate(seq,time_step,output_size):
    X = []
    Y = []
    for i in range(len(seq)-time_step-output_size):
        X.append(seq[i:i+time_step])
        Y.append(seq[i+time_step:i+time_step+output_size])
    return np.array(X,dtype =np.float32),np.array(Y,dtype = np.float32)
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader,Dataset
from plotly import graph_objects as go
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('data_converted.csv')

# seq_y = data_day.values.reshape([-1])
#X_train,y_train = generate(seq,TIME_STEPS,OUTPUT_SIZE)
#X_test, y_test = generate(seq[:-24*7],TIME_STEPS,OUTPUT_SIZE)

#print(df.loc[0,  'time' ])
#数据预览
#fig = go.Figure()
#fig.add_trace(go.Scatter(x=[1,2,3], y=[2,1,2], name='MT001'))
#fig.show()

scaler = MinMaxScaler()
predict_field = 'Electricity'
df[predict_field] = scaler.fit_transform(df['MT_261'].values.reshape(-1, 1))
#print(df.head())
train_size = 16000
test_size = 3000
time_step = 96#1000
output_size = 96
#print(df.iloc[0:train_size,376])
print("data length is",len(df[predict_field]))
def create_dataset(data:list, time_step: int,output_size:int):
    arr_x = []
    arr_y = []
    for i in range(len(data) - time_step - output_size):
        x = data[i: i + time_step]
        y = data[i + time_step:i + time_step+output_size]
        arr_x.append(x)
        arr_y.append(y)
#        arr_x = np.vstack((arr_x,x))
#        arr_y = np.vstack((arr_y,y))
#    np.delete(arr_x,0,axis = 0)
#    np.delete(arr_y,0,axis = 0)
#    arr_x = arr_x.unsqueeze(1)
#    arr_y = arr_y.unsqueeze(1)
    return np.array(arr_x), np.array(arr_y)

#长度不一致 object 两个地方均是 加载dataset 用pandas create 中减隔出问题 这话条件？ 确实是不等长的问题
X, Y = create_dataset(df.iloc[0:train_size,376].values, time_step ,96)
print(X)



class TrainData(Dataset):
    def __init__(self):
        X, Y = create_dataset(df.iloc[0:train_size,376].values, time_step ,96)
#        print(Y.shape)
        self.x_data = torch.from_numpy(X)#xy[0:93504-97, 255]li
        self.y_data = torch.from_numpy(Y)#xy[93504-96:93504, 255]
        self.len = self.x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
class TestData(Dataset):
    def __init__(self):
        X_t, Y_t = create_dataset(df.iloc[train_size:train_size+test_size,376].values, time_step ,96) # 使用numpy读取数据
        self.x_data = torch.from_numpy(X_t)#dao差96最后 前后不一致
        self.y_data = torch.from_numpy(Y_t)
        self.len = self.x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


#train_dataset = TrainData()

#test_dataset = TestData()


"""
class TrainData(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data_converted.csv', delimiter=',', dtype=np.float32) # 使用numpy读取数据
        seq = xy[:93504,255]
        TIME_STEPS = 8000
        OUTPUT_SIZE = 96
        self.x_data,__ = torch.from_numpy(generate(seq,TIME_STEPS,OUTPUT_SIZE))#xy[0:93504-97, 255]li
        __,self.y_data = torch.from_numpy(generate(seq,TIME_STEPS,OUTPUT_SIZE))#xy[93504-96:93504, 255]
        self.len = seq.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
class TestData(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data_converted.csv', delimiter=',', dtype=np.float32) # 使用numpy读取数据
        self.x_data = torch.from_numpy(xy[93504:140256-97, 255])#dao差96最后 前后不一致
        self.y_data = torch.from_numpy(xy[140256-96:, 255])
        self.len = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
"""