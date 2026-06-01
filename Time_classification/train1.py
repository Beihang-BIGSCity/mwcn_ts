# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:50:53 2020

@author: 13621 main1d

"""
from __future__ import print_function
import sys
import os, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
#from vgg1d import *
#from models import *
from utils import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from ucr_datasets import *
from net_structure1 import *
from net_structurewave import *
from net_convmultiwave import *
from net_convmultiwave2d import *
from net_convmultiwave2d_att import *
from net_convmultiwave2d_att11 import *
from net_convmultiwave2d_att22 import *
from net_convmultiwave_att3 import *
#from net_convmultiwave2d_att4 import conv2d_multiwave_att4
from net_convmultiwave2d_att5 import conv2d_multiwave_att5
from net_convmultiwave2d_att6 import conv2d_multiwave_att6
from net_convmultiwave2d_att7 import conv2d_multiwave_att7
from target_function import *




parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r',action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', type=str, default='2', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='testatt111', help='Log dir [default: log]')#Adiacnewmulti5 45为sa4和random ablmulti1dnew112
parser.add_argument('--dataset', default='CricketX', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--optimizer', default='momentum', help='adam or momentum [default: adam]')
parser.add_argument('--att', default=5, type=int)
parser.add_argument('--d_model', default=32, type=int)
parser.add_argument('--ratio', default=4, type=int)
parser.add_argument('--num_layers', default=4, type=int)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
gpu_index = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
#print("visible device",os.environ["CUDA_VISIBLE_DEVICES"])
LOG_DIR = args.log_dir
name_file = sys.argv[0]
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)
os.mkdir(LOG_DIR + '/files')
os.system('cp %s %s' % (name_file, LOG_DIR))
os.system('cp %s %s' % ('*.py', os.path.join(LOG_DIR, 'files')))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')

dataset_name = args.dataset

def log_string(out_str, print_out=True):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    if print_out:
	    print(out_str)

st = ' '
log_string(st.join(sys.argv))

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

#dataset_name = 'Adiac'
# Data
print('==> Preparing data..')

BATCH_SIZE = args.batch_size

#dataset_name = 'Adiac'#ECG5000
train_dataset = UCR_Dataset(train=True, 
                            root_dir='./data20240812/UCRArchive_2018/',#'/mnt/data/hanxiao1997/MWNNT/newdata/UCRArchive_2018/',#/home/hanxiao1997/MWNNT/newdata/UCRArchive_2018/
                            dataset_name = dataset_name)

test_dataset = UCR_Dataset(train=False,
                           root_dir='./data20240812/UCRArchive_2018/',#'/mnt/data/hanxiao1997/MWNNT/newdata/UCRArchive_2018/'
                           dataset_name = dataset_name)

trainloader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True,
                                           num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=BATCH_SIZE, 
                                          shuffle=False,
                                          num_workers=0)
NUM_CLASS = len(set(train_dataset.labels.flatten().tolist()))
print(train_dataset)
# Model

print("device is",torch.cuda.current_device())

print('==> Building model..')
#net = ResNet18(num_class=NUM_CLASS)
#net = ResNet_DCF181d(num_classes=NUM_CLASS, bases_grad=False)
data_len = train_dataset.features.shape[2]

#net = VGG1d('VGG16', num_class=NUM_CLASS)
#net = VGG_DCF1d('VGG11', num_class=NUM_CLASS, bases_grad=False)

if args.att == 5:
    net = conv2d_multiwave_att5(data_len,NUM_CLASS,'dot',factor=5, d_model=args.d_model, n_heads=8, d_layers=2,
                                ratio=args.ratio,
                    dropout=0.1, freq='h',output_attention = False,model='vgg',d_prob=0.5,batch_size1 = BATCH_SIZE)#主模型卷积二维形式:多小波+channel attention
elif args.att == 6:
    net = conv2d_multiwave_att6(data_len,NUM_CLASS,'dot',factor=5, d_model=args.d_model, n_heads=8, d_layers=2,
                                ratio=args.ratio,
                dropout=0.1, freq='h',output_attention = False,model='vgg',d_prob=0.5,batch_size1 = BATCH_SIZE)#主模型卷积二维形式2:多小波+channel attention
elif args.att == 7:
    net = conv2d_multiwave_att7(data_len,NUM_CLASS,'dot',factor=5, d_model=args.d_model, n_heads=8, d_layers=2,
                                ratio=args.ratio, num_layers=args.num_layers,
                dropout=0.1, freq='h',output_attention = False,model='vgg',d_prob=0.5,batch_size1 = BATCH_SIZE)#主模型卷积二维形式3:多小波+channel attention
#net=conv2d_multiwave(data_len,NUM_CLASS,'dot',factor=5, d_model=data_len/2, n_heads=8, d_layers=2,
#                dropout=0.0, freq='h',output_attention = False,model='vgg',d_prob=0.5,batch_size1 = BATCH_SIZE)#多小波卷积二维形式,无attention
#net = conv2d_multiwave_v2_att22(data_len,NUM_CLASS,'dot',factor=5, d_model=data_len/2, n_heads=8, d_layers=2,
#                dropout=0.0, freq='h',output_attention = False,model='vgg',d_prob=0.5,batch_size1 = BATCH_SIZE)#多小波卷积二维形式+多小波分通道att,前面12通道，类似第一个模型
#net=multiCNNatt1(data_len,NUM_CLASS,'dot',factor=5, d_model=data_len/2, n_heads=8, d_layers=2,
#                dropout=0.0, freq='h',output_attention = False,model='vgg1d',d_prob=0.5)  #多小波全连接
#net=waveCNNatt1(data_len,NUM_CLASS,'dot',factor=5, d_model=data_len/2, n_heads=8, d_layers=2,
#                dropout=0.0, freq='h',output_attention = False,model='vgg1d',d_prob=0.5)  #小波
#net=conv_multiwave(data_len,NUM_CLASS,'dot',factor=5, d_model=data_len/2, n_heads=8, d_layers=2,
#                dropout=0.0, freq='h',output_attention = False,model='vgg1d',d_prob=0.5)  #多小波卷积一维形式
#net = conv2d_multiwave_v2_att33(data_len,NUM_CLASS,'dot',factor=5, d_model=data_len/2, n_heads=8, d_layers=2,
#                dropout=0.0, freq='h',output_attention = False,model='vgg',d_prob=0.5,batch_size1 = BATCH_SIZE)#channel为[12,1,1] ##多小波卷积二维形式+att，类似第一个模型
#net = conv2d_multiwave_att4(data_len,NUM_CLASS,'dot',factor=5, d_model=data_len/2, n_heads=8, d_layers=2,
#                dropout=0.0, freq='h',output_attention = False,model='vgg',d_prob=0.5,batch_size1 = BATCH_SIZE)


print(net)
net = net.to(device)
#if device == 'cuda':
#    net = torch.nn.DataParallel(net)
#    cudnn.benchmark = True
#print("gpu device is",net.device)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
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
    for batch_idx,batch in enumerate(trainloader):
        #print(batch_idx)
        inputs = batch['feature']
        targets = batch['label'].reshape(-1).long() 
        #print(f"input shape:",inputs.shape)
        
#        targets = targets.long()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        lamda = 0.0001
        """if orthorgonal regularization
        Ld1 = net.features2.convdecl1.weight
        Ld2 = net.features2.convdecl2.weight
        Hd1 = net.features2.convdech1.weight
        Hd2 = net.features2.convdech2.weight

        orthogonal_regularization = orthogonal_reg(lamda,Ld1,Ld2,Hd1,Hd2)
        """
#        targets = targets.squeeze()      
        #targets = targets.squeeze()
        loss = criterion(outputs, targets)
        #####loss = loss+ orthogonal_regularization if orthorgonal regularization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    log_string('Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total), False)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    c = []
    i = 0
    q = 0
    qq = 0
    N_CLASSES=NUM_CLASS
    class_correct = list(0. for i in range(N_CLASSES))
    class_total = list(0. for i in range(N_CLASSES))
    with torch.no_grad():
        for batch_idx,batch in enumerate(testloader):
            #print(batch_idx)
            inputs = batch['feature']
            targets = batch['label'].reshape(-1).long() 
            inputs, targets = inputs.to(device), targets.to(device)
            #targets = targets.squeeze()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            c = (predicted == targets).squeeze()
            # rint(c)
            # print(targets)
            for q in range(len(targets)):
                for qq in range(4):
                    if targets[q]==qq:
                        target = qq
                        class_correct[qq] += c[q].item()
                        class_total[qq] += 1
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        log_string('TEST Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total), False)
    for i in range(N_CLASSES):

        if class_total[i]==0:
            # print(classes[i],class_correct[i],class_total[i])
            continue
        else:
            print('Accuracy of %5s : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))    
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
        # log_string('\ntest Epoch: %d' % epoch)
    log_string('Best Acc:%.3f%%'%(best_acc), False)
    print('best acc is: %f' %best_acc)        


for epoch in range(start_epoch, start_epoch+200):# 200
    if epoch in [100,150]:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
        log_string('In epoch %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
    train(epoch)
    test(epoch)
