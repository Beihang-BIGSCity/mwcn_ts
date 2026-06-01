# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:49:10 2024

@author: 13621
"""

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
import os
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader,Dataset
#from plotly import graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from utils1.tools import StandardScaler
from utils1.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ECL.csv', 
                 target='MT_261', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns) # ; cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



def get_data(args,flag = 'train'):
    timeenc = 0 if args.embed!='timeF' else 1
    if flag == 'test':
        shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
    elif flag=='pred':
        shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        #Data = Dataset_Pred
    else:
        shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
    data_set = Dataset_Custom(#哪里的品
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        inverse=args.inverse,
        timeenc=timeenc,
        freq=freq,
        cols=args.cols
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last)

    return data_set, data_loader

# seq_y = data_day.values.reshape([-1])
#X_train,y_train = generate(seq,TIME_STEPS,OUTPUT_SIZE)
#X_test, y_test = generate(seq[:-24*7],TIME_STEPS,OUTPUT_SIZE)

#print(df.loc[0,  'time' ])
#数据预览
#fig = go.Figure()
#fig.add_trace(go.Scatter(x=[1,2,3], y=[2,1,2], name='MT001'))
#fig.show()
"""
df = pd.read_csv('C:/Users/13621/Downloads/predict/Datasets_informer/ECL.csv')
scaler = MinMaxScaler()
predict_field = 'Electricity'
df[predict_field] = scaler.fit_transform(df['MT_261'].values.reshape(-1, 1))
#print(df.head())
#border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
#border2s = [num_train, num_train+num_vali, len(df_raw)]
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