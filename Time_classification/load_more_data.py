# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:53:05 2022

@author: 13621 load_more_data
"""
import os
from sklearn import preprocessing
from PIL import Image
from sklearn.preprocessing import minmax_scale
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from IPython import embed

def load_dataset(train, root_dir, dataset_name):
    raw_dataset = None
    if train:
        dataset_path = os.path.join(root_dir, dataset_name, 
                                    dataset_name+'_TRAIN.csv')
        raw_dataset = np.loadtxt(dataset_path, delimiter=',', dtype=np.float)
    else:
        dataset_path = os.path.join(root_dir, dataset_name, 
                                    dataset_name+'_TEST.csv')
        raw_dataset = np.loadtxt(dataset_path, delimiter=',', dtype=np.float)
    raw_dataset = raw_dataset.astype(np.float32)
    print(raw_dataset.shape)
    features = raw_dataset[:, 1:]
    features = np.nan_to_num(features)
    labels = raw_dataset[:, 0:1]
    # embed()
    class_list = list(set(labels.flatten().tolist()))
    class_list.sort()
    for i in range(labels.shape[0]):
        labels[i][0] = class_list.index(labels[i][0]) * 1.0
    # embed()
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(labels, axis=1))
    le.transform(np.squeeze(labels, axis=1))
    return features, labels

class Time_more_Dataset(Dataset):
    def __init__(self, train=True, root_dir=None, 
                 dataset_name=None, transform=None):
        if train == True:
            # load all train
            self.features, self.labels = load_dataset(train, root_dir, dataset_name)
            self.features = np.expand_dims(self.features, 1)
            self.labels = self.labels.astype(np.long)
            print("Train shape:", self.features.shape, self.labels.shape)
        else:
            # load all train
            self.features, self.labels = load_dataset(train, root_dir, dataset_name)
            self.features = np.expand_dims(self.features, 1)
            self.labels = self.labels.astype(np.long)
            print("Test shape:", self.features.shape, self.labels.shape)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'feature': self.features[idx], 'label': self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    dataset_name = 'ecgdata'
    batch_size = 16
    train_dataset = Time_more_Dataset(train=True, 
                            root_dir='C:/Users/13621/Downloads/MWNNT/newdata',
                            dataset_name = dataset_name)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            num_workers=4)
    test_dataset = Time_more_Dataset(train=False,
                            root_dir='C:/Users/13621/Downloads/MWNNT/newdata',
                            dataset_name = dataset_name)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            num_workers=4)
