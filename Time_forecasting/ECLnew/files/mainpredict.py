# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:15:20 2022

@author: 13621 predcitmain
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
from vgg1d import *
#from models import *
from utils import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data import ucr_datasets
from ucr_datasets import *
from net_structure import *
from net_structure1 import *
from net_structurewave import *
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r',action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='Adiac2', help='Log dir [default: log]')
parser.add_argument('--dataset', default='Adiac', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=8, help='Bat2h Size during training [default: 32]')
parser.add_argument('--optimizer', default='momentum', help='adam or momentum [default: adam]')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
gpu_index = args.gpu
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
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

#dataset_name = 'Adiac'
# Data
print('==> Preparing data..')

BATCH_SIZE = args.batch_size

dataset_name = 'Adiac'
train_dataset = UCR_Dataset(train=True, 
                            root_dir='C:/Users/13621/Downloads/MWNNT/data/UCRArchive_2018/',
                            dataset_name = dataset_name)

test_dataset = UCR_Dataset(train=False,
                           root_dir='C:/Users/13621/Downloads/MWNNT/data/UCRArchive_2018/',
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
print('==> Building model..')
path_to_bessel = 'C:/Users/13621/Downloads/DCFNet-Pytorch-master/bessel.npy'
#net = ResNet18(num_class=NUM_CLASS)
#net = ResNet_DCF181d(num_classes=NUM_CLASS, bases_grad=False)
data_len = train_dataset.features.shape[2]

#net = VGG1d('VGG16', num_class=NUM_CLASS)
#net = VGG_DCF1d('VGG11', num_class=NUM_CLASS, bases_grad=False)
#net=multiCNNatt1(data_len,NUM_CLASS,'dot',factor=5, d_model=data_len/2, n_heads=8, d_layers=2,
#                dropout=0.0, freq='h',output_attention = False,model='vgg1d',d_prob=0.5)

net=waveCNNatt1(data_len,NUM_CLASS,'dot',factor=5, d_model=data_len/2, n_heads=8, d_layers=2,
                dropout=0.0, freq='h',output_attention = False,model='vgg1d',d_prob=0.5)# net = LeNet()
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
        inputs = batch['feature']
        targets = batch['label'].reshape(-1).long() 
#        print(inputs.shape)
        
#        targets = targets.long()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        targets = targets.squeeze()
        loss = criterion(outputs, targets)
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
            inputs = batch['feature']
            targets = batch['label'].reshape(-1).long() 
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            c = (predicted == targets).squeeze()
#            print(c)
#            print(targets)
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
            continue
        #print(classes[i],class_correct[i],class_total[i])
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
    print('best acc is: %f' %best_acc)        


for epoch in range(start_epoch, start_epoch+100):
    if epoch in [100, 150]:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
        log_string('In epoch %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
    train(epoch)
    test(epoch)
