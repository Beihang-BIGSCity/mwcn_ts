# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:11:32 2023

@author: 13621
"""


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import torch.autograd.variable as Variable
import torch.utils.data as Data
from multiwaveletconv2d import *
import torch
import torch.nn as nn
import torch.nn.functional as F
#from showdata import *#品
#from showdataecl import *
batch_size = 16
learning_rate = 0.0001
#from vgg import *
from vgg1d import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
class conv2d_multichannel_att(nn.Module):#三层小波分解
    def __init__(self,length_input = 96, number_ouputs=4,mode='dot',factor=5, d_model=512, n_heads=8, d_layers=2,
                dropout=0.0, freq='h',output_attention = False,model='vgg',d_prob=0.5,batch_size1=32):
        super(conv2d_multichannel_att, self).__init__()   # 继承__init__功能 pytorch初始化参数
        if length_input % 16 != 0:
            aaa = length_input%16
            length_input = length_input + 16-(length_input%16)
            self.diff = 16-aaa  
        else:
            self.diff = 0
        self.len_input = length_input
        enc_in = 7
        embed = 'fixed'
        self.num_outputs = number_ouputs
        self.factor=factor
        self.d_model=d_model
        self.n_heads=n_heads
        self.d_layers=d_layers
        self.dropout=dropout
        W1  =  torch.rand(12,12)
        W2  =  torch.rand(12,12)
        self.W1 = torch.nn.Parameter(W1,requires_grad = True).to(device=0)
        self.W2 = torch.nn.Parameter(W2,requires_grad = True).to(device=0)
        self.output_attention=output_attention
        self.batch_size = batch_size1
        self.model = model
        kernel_size = 8
        stride = 2
        n_block = 3
        downsample_gap = 6
        increasefilter_gap = 6
        ## 第一层卷积
        self.features1 = Conv_mwavepre1d(is_training=True,len_input=self.len_input, sim_reg=0, activation=None)
        self.features2 = Conv_mwavedec2d(is_training=True,len_inputi=self.len_input, sim_reg=0, activation=None)
        self.features3 = Conv_mwavedec2d(is_training=True,len_inputi=self.len_input/2, sim_reg=0, activation=None)
        self.features4 = Conv_mwavedec2d(is_training=True,len_inputi=self.len_input/4, sim_reg=0, activation=None)

        self.vgg = VGG1d('VGG11',num_class=self.num_outputs)
        self.vgg2 = VGG1d('VGG7',num_class=self.num_outputs)
        self.vgg3 =  VGG1d('VGG9',num_class=self.num_outputs)
        self.vgg4 =  VGG1d('VGG7new',num_class=self.num_outputs)
        self.fc = nn.Linear(in_features=64,out_features=self.num_outputs)
        self.fc1 = nn.Linear(in_features=12,out_features=1)
        self.output = nn.Linear(in_features=12*self.num_outputs, out_features=self.num_outputs)
    def channel_att(self,u):
        #z = torch.rand(u.shape[1],12)#[batch_size,12]
        ###print("u",u.shape)#[batch_size,12,num_output]#[batch_size,num_output*12]
        z =  torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32)#赋值为向量测试
        #u_widetilde = torch.rand(u.shape[1],12)#[batch_size,12]
        u_widetilde = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32)
        #for c in range(0,12):
        #    b_u = torch.mul(u[:,c,:],u[:,c,:])#[batch_size,1,num_output]  2                  #[batch_size,num_output]#*12
        #    print("b_u",b_u.shape)
        #    z[c] = torch.sum(b_u,dim=2)##[batch_size,1,1]2
        u1 = torch.mul(u,u)
        z = torch.sum(u1,dim=2)
        z = z.cuda()#  ######[12,batch_size,1,1]2
        print("z",z.shape)
        z = z.unsqueeze(2)
        print("z",z.shape)
        #exit()
        s = torch.sigmoid(torch.matmul(self.W2,torch.relu(torch.matmul(self.W1,z))))
        ###print("s",s.shape)#[16,12,1]
        #s =s.squeeze(2)
        u_widetilde = torch.mul(u,s)#[16,12,num]*[16,12,1]
        print("u",u)
        print("u_w",u_widetilde)
        #for c in range(0,12):
        #    u_widetilde[c] = s[c]*u[c]
        return u_widetilde #[16,12,num]
    def forward(self, x,mode='dot'):
        x= x#.to('cuda:0') #[batch_size, 1, len_input]
        ##print("ori size x",x.shape)
        if self.diff !=0:
            if x.shape[0]==self.batch_size:
                btr = np.zeros((self.batch_size,1,self.diff))
                ad1 = torch.from_numpy(btr).to(device=0)
                x = torch.cat((x,ad1),2)
            elif x.shape[0]!=self.batch_size:
                btr = np.zeros((x.shape[0],1,self.diff))
                ad1 = torch.from_numpy(btr).to(device=0)
                x = torch.cat((x,ad1),2)
        ##print("devicein is",torch.cuda.current_device())
        ##print("size x",x.shape)
        #now x is [batch_size, 1, len_input]
        x = x.squeeze(1) #[batch_size, len_input] #x = x.squeeze()
        ##print("squeeze size x",x.shape)
        low0= self.features1(x)#[batch_size, 1, 2, len_input/2]
        ##print("low0",low0.shape)
        low1,high1 = self.features2(low0)
        low1 = low1#.to('cuda:0') [batch_size, 1, 2, len_input/4]
        high1 = high1#.to('cuda:0') [batch_size, 1, 2, len_input/4]
        ##print("low1 high1",low1.shape,high1.shape)
        low2,high2 = self.features3(low1)#[batch_size, 1, 2, len_input/8],[batch_size, 1, 2, len_input/8]
        low2=low2#.to('cuda:0')
        high2 = high2#.to('cuda:0')
        ##print("low2 high2",low2.shape,high2.shape)
        low3,high3 = self.features4(low2)#[batch_size, 1, 2, len_input/16],[batch_size, 1, 2, len_input/16]
        low3=low3#.to('cuda:0')
        high3 = high3#.to('cuda:0')
        ##print("low3 high3",low3.shape,high3.shape)
        if self.model=='vgg':    
            low1 = low1.to(torch.float32)
            low11 = low1[:,:,0,:]
            low12 = low1[:,:,1,:]
            low11  = self.vgg2(low11)
            low12  = self.vgg2(low12)
            high1 = high1.to(torch.float32)
            high11 = high1[:,:,0,:]
            high12 = high1[:,:,1,:]
            high11  = self.vgg2(high11)
            high12  = self.vgg2(high12)
            ##print(f"all_out1 vgg",all_out1.size())
            low2 = low2.to(torch.float32)
            #low2=low2.unsqueeze(1)
            low21 = low2[:,:,0,:]
            low22 = low2[:,:,1,:]
            low21  = self.vgg4(low21)
            low22  = self.vgg4(low22)
            high2 = high2.to(torch.float32)
            #high2 = high2.unsqueeze(1)
            high21 = high2[:,:,0,:]
            high22 = high2[:,:,1,:]
            high21  = self.vgg4(high21)
            high22  = self.vgg4(high22)
            ##all_out2 = torch.cat([low2, high2],dim=2)#.to('cuda:0') [batch_size, 1, 4, len_input/8]
            ##print(f"all_out2",all_out2.size())
            #all_out2 = self.vgg3(all_out2)#vgg1d3 [batch_size, num_output]
            ##print(f"all_out2 vgg3",all_out2.size())
            low3 = low3.to(torch.float32)
            low31 = low3[:,:,0,:]
            low32 = low3[:,:,1,:]
            low31  = self.vgg2(low31)
            low32  = self.vgg2(low32)
            #low3=low3.unsqueeze(1)
            high3 = high3.to(torch.float32)
            high31 = high3[:,:,0,:]
            high32 = high3[:,:,1,:]
            high31  = self.vgg2(high31)
            high32  = self.vgg2(high32)
            #high3 = high3.unsqueeze(1)
            #all_out3 = torch.cat([low3, high3],dim=2)#.to('cuda:0') [batch_size, 1, 4, len_input/16]
            ##print(f"all_out3",all_out3.size())
            #all_out3 = self.vgg2(all_out3) #[batch_size, num_output]
            ##print(f"all_out3 vgg2",all_out3.size())
            u = torch.stack([low11,low12,high11,high12,low21,low22,high21,high22,low31,low32,high31,high32],dim=1)
            #u = torch.cat([low11,low12,high11,high12,low21,low22,high21,high22,low31,low32,high31,high32],dim=1)#[16,num_output*12]
            #u = [low11,low12,high11,high12,low21,low22,high21,high22,low31,low32,high31,high32]
            #print("u",u.shape,u[0].shape,u[1].shape)
            
            u1 = self.channel_att(u)
            print("u1",u1[0])
            exit()
            ###print(u_widetilde.shape)
            #z = torch.cat(u_widetilde,dim=1)
            #z = torch.sum(u_widetilde,dim=1)#与函数的后几行关系[16,1,37]
            #z = z.squeeze(1)
            batch_size1 = u1.shape[0]
            num_channel = u1.shape[1]
            features1 = u1.shape[2]
            z = u1.reshape(batch_size1,features1,num_channel)
            z = self.fc1(z)
            z = z.squeeze(2)
            #z = u_widetilde.view(batch_size1,num_channel*features1)
            #z=torch.cat([all_out1,all_out2,all_out3],dim=1)  #[batch_size, 3*num_output]     
            ###print("z",z.shape)
            #output_fun = nn.Linear(in_features=z.shape[1], out_features=self.num_outputs)
            #output_fun = output_fun.cuda()
            #output = output_fun(z) 
            ###exit()
            #output = self.output(z) #[bacth_size, num_output]    # 输出[16,2]
            output = z
            ##print(f"output is",output.shape)
            ##exit()
        return output
