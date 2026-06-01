# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 08:49:19 2020

@author: 13621
"""

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGGn':[32,'M',64,64,'M',128,128,'M',128,128],    
    'VGG7':[64,'M',128,128,'M'],
    'VGG9':[64,'M',128,128,'M',256,256,'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG1d(nn.Module):
    def __init__(self, vgg_name, kernel_size=17, num_class=2):
        super(VGG1d, self).__init__()
        self.kernel_size = kernel_size
        self.name = vgg_name
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_class)
        self.classifier1 = nn.Linear(128,num_class)
        self.classifier2 = nn.Linear(256,num_class)
        self.classifier3 = nn.Linear(64,num_class)
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        if self.name=='VGG7':
            out = self.classifier1(out)
        elif self.name=='VGG9': 
            out = self.classifier2(out)
        elif self.name=='VGGn':
            out = self.classifier1(out)
        else:
            out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        padding = int((self.kernel_size - 1) / 2)
        layers = []
        in_channels = 2
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                # layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                layers += [nn.Conv1d(in_channels, x, kernel_size=self.kernel_size, 
                    padding=padding, bias=False),
                           nn.BatchNorm1d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.AdaptiveAvgPool1d(1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
