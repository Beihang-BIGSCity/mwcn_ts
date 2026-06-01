# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 20:32:02 2024

@author: 13621

"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
import os, shutil
import argparse
import numpy as np


from utils import *
from utils1.metrics import metric
from torch.utils.data import Dataset, DataLoader, TensorDataset
#from ecldataset import *
import matplotlib.pyplot as plt
from net_lstm import netlstm
from net_lstm2 import netlstm2
from net_convmultiwave2d_vgg import conv2d_multiwave_vgg
from net_convmultiwave2d import conv2d_multiwave
from net_convmultiwave2d_att import conv2d_multiwave_att
from target_function import orthogonal_reg
from data_providernew import get_data

parser = argparse.ArgumentParser(description='PyTorch Time prediction Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r',action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='ECLnew', help='Log dir [default: log]')#ECL96 ETT ETT96 weather96
parser.add_argument('--dataset', default='ECL', help='dataset [default: data]')#ECL ETT weather
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--optimizer', default='momentum', help='adam or momentum [default: adam]')
parser.add_argument('--root_path',default='./Datasets_informer/',help='root path')#./Datasets_informer/ETT/
parser.add_argument('--data_path',default='ECL.csv',help='data path')#ETTm2.csv WTH.csv
parser.add_argument('--seq_len',type=int,default=96,help='input length')#36
parser.add_argument('--label_len',type=int,default=0,help='length of label')
parser.add_argument('--pred_len',type=int,default=96,help='predict length')#24
parser.add_argument('--output_size',type=int,default=96,help='output length')
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='MT_320', help='target feature in S or MS task')#'OT'
parser.add_argument('--timeenc',type=int,default=0, help='timeenc')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--hidden_size', type=int, default=32, help='numbers of lstm hidden size')



# pathformer
# model
parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=64)
parser.add_argument('--num_nodes', type=int, default=21)
parser.add_argument('--layer_nums', type=int, default=3)
parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at the every layer ')
parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
parser.add_argument('--patch_size_list', nargs='+', type=int, default=[16,12,8,32,12,8,6,4,8,6,4,2])
parser.add_argument('--residual_connection', type=int, default=0)
parser.add_argument('--metric', type=str, default='mae')
parser.add_argument('--batch_norm', type=int, default=0)
# optimization
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')



args = parser.parse_args()

BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
gpu_index = args.gpu
print(f"gpu_id={gpu_index}")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

LOG_DIR = args.log_dir
name_file = sys.argv[0]
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)
# os.mkdir(LOG_DIR + '/train_img')
# os.mkdir(LOG_DIR + '/test_img')
os.mkdir(LOG_DIR + '/files')
os.system('cp %s %s' % (name_file, LOG_DIR))
os.system('cp %s %s' % ('*.py', os.path.join(LOG_DIR, 'files')))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')

def log_string(out_str, print_out=True):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    if print_out:
	    print(out_str)

st = ' '
log_string(st.join(sys.argv))

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
device = f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

#dataset_name = 'Adiac'
# Data
print('==> Preparing data..')

BATCH_SIZE = args.batch_size

dataset_name = 'ECL'

train_data, train_loader = get_data(args,flag = 'train')
print("train_data[0]",train_data[0][0].shape,train_data[0][1].shape,train_data[0][2].shape,train_data[0][3].shape)
vali_data, vali_loader = get_data(args,flag = 'val')
test_data, test_loader = get_data(args,flag = 'test')
#NUM_CLASS = len(set(train_dataset.labels.flatten().tolist()))
output_size=args.output_size
# Model
print('==> Building predict model..')
#net = ResNet18(num_class=NUM_CLASS)
#net = ResNet_DCF181d(num_classes=NUM_CLASS, bases_grad=False)
#data_len = train_dataset.features.shape[2]

#net = VGG1d('VGG16', num_class=NUM_CLASS)
#net = VGG_DCF1d('VGG11', num_class=NUM_CLASS, bases_grad=False)
#net=multiCNNatt1(data_len,NUM_CLASS,'dot',factor=5, d_model=data_len/2, n_heads=8, d_layers=2,
#                dropout=0.0, freq='h',output_attention = False,model='vgg1d',d_prob=0.5)
time_step = 96
hd = args.hidden_size


# net=conv2d_multiwave_vgg(args.seq_len, args.seq_len,output_size,factor=5, d_model=512, n_heads=2, d_layers=2,
#                 dropout=0.1, freq='h',model='vgg',d_prob=0.5,batch_size1=32)
net=conv2d_multiwave_att(args.seq_len, args.seq_len,output_size,mode='dot',factor=5, d_model=512, n_heads=4, d_layers=6,
                dropout=0.0, freq='h',model='lstm',d_prob=0.5,hidden_size=hd,batch_size1=32)#主模型：多小波二维卷积神经网络+channel attention
#net=conv2d_multiwave(args.seq_len, time_step,output_size,mode='dot',factor=5, d_model=512, n_heads=8, d_layers=2,
#                dropout=0.0, freq='h',model='lstm',d_prob=0.5,hidden_size=hd,batch_size1=32)#消融模型:多小波二维卷积神经网络，无channel attention
#net=netlstm2(args.seq_len,time_step,output_size,'dot',factor=5,  n_heads=8, d_layers=2,
#                dropout=0.0, model='lstm1',d_prob=0.5,hidden_size=hd)#多小波全连接
#net=waveCNNatt1(output_size,'dot',factor=5,  n_heads=8, d_layers=2,
#                dropout=0.0, model='lstm',d_prob=0.5)#小波模型
# net = LeNet()
#net = LeNet_DCF1d()
#net = Net()
# Please try not to train LeNet on CUB, it will make him suffers a lot...

print(net)
net = net.to(device)
#if device == 'cuda':
#    net = torch.nn.DataParallel(net)
#    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
WEIGHT_DECAY = 5e-4
if OPTIMIZER == 'momentum':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == 'nesterov':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)
elif OPTIMIZER == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)    ###__0.0001->0.001
elif OPTIMIZER == 'rmsp':
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
else:
    raise NotImplementedError
    
# Training
def train(epoch):
    log_string('\nEpoch: %d' % epoch)
    net.train()  
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx,batch in enumerate(train_loader):
        inputs,__,__,__ = batch
        __,targets,__,__ = batch
        inputs = inputs.squeeze(2)
        targets = targets.squeeze(2)
#        targets = targets.reshape(-1).long()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        #print("inputs shape",inputs.shape)
        outputs = net(inputs)
        
        #if orthorgonal regularization
        """
        lamda = 0.0001
        Ld1 = net.features2.convdecl1.weight
        Ld2 = net.features2.convdecl2.weight
        Hd1 = net.features2.convdech1.weight
        Hd2 = net.features2.convdech2.weight

        orthogonal_regularization = orthogonal_reg(lamda,Ld1,Ld2,Hd1,Hd2)
        """
        targets = targets.squeeze()
        targets = targets.float()
        #print("targets shape:",targets.shape)
        loss = criterion(outputs, targets)
        loss = loss#+ orthogonal_regularization #if orthorgonal regularization
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        #_, predicted = outputs.max(1)
        total += targets.size(0)
#        correct += predicted.eq(targets).sum().item()
        predicted = outputs.view(-1).data.cpu().numpy()
        progress_bar(batch_idx, len(train_loader), 'MSELoss: %.3f |Total:%d'
            % (train_loss/(batch_idx+1), total))
    log_string('MSELoss: %.3f |Total:%d'
        % (train_loss/(batch_idx+1), total), False)

def test(epoch, best_mse, best_mae):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    preds = []
    targets_total = []
    with torch.no_grad():
        for batch_idx,batch in enumerate(test_loader):
            inputs,__,__,__ = batch
            __,targets,__,__ = batch
            inputs = inputs.squeeze(2)
            targets = targets.squeeze(2)
#            targets = targets.reshape(-1).long()
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze()
            targets = targets.float()
            outputs = net(inputs)
            #print(targets.shape)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            predicted = outputs.data.detach().cpu().numpy()
            #predicted = outputs.view(-1).data.detach().cpu().numpy()
            #_, predicted = outputs.max(1)
#            print(c)
#            print(targets)
            ##print("first predict targets",predicted.shape,targets.shape)
            predicted= np.expand_dims(predicted,axis=2)
            
            #targets= np.expand_dims(targets,axis=0)
            preds.append(predicted)
            targets_total.append(np.expand_dims(targets.detach().cpu().numpy(),axis=2))#
            total += targets.size(0)
#            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(test_loader), 'MSELoss: %.3f |(total:%d)'% (test_loss/(batch_idx+1), total))
        log_string('TEST MSELoss: %.3f | (Total:%d)'% (test_loss/(batch_idx+1), total), False)
        preds = np.array(preds)
        targets_total = np.array(targets_total)
        ##print("preds,targets total",preds.shape,targets_total.shape)
        print('test shape:', preds.shape, targets_total.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        targets_total = targets_total.reshape(-1, targets_total.shape[-2], targets_total.shape[-1])
        print('test shape:', preds.shape, targets_total.shape)
        mae, mse, rmse, mape, mspe = metric(preds, targets_total)
        if best_mae > mae:
            best_mae = mae
        if best_mse > mse:
            best_mse = mse
        print('mse:{}, mae:{}'.format(mse, mae))
        log_string('mse=%.4f,  mae=%.4f'% (mse, mae), False)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(LOG_DIR, 'ckpt.t7'))
        best_acc = acc
    return predicted,targets, best_mse, best_mae 

best_mse = np.inf
best_mae = np.inf
for epoch in range(start_epoch, start_epoch+70):
    if epoch in [20, 70]:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
        log_string('In epoch %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
    train(epoch)
    predicted,targets,best_mse,best_mae = test(epoch, best_mse, best_mae)#ecg resnet品
    targets = targets.view(-1).data.cpu().numpy()
print('best_mse:{}, best_mae:{}'.format(best_mse, best_mae))
log_string('best_mse:{}, best_mae:{}'.format(best_mse, best_mae), False)
#画图
#plt.plot(predicted, 'r', label='prediction')
#plt.plot(targets, 'b', label='real')
#plt.legend(loc='best')