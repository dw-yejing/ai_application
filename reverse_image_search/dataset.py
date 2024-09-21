import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

transform = T.Compose([
        T.ToTensor(),
        # 图像标准化处理
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
cifar10_train = torchvision.datasets.CIFAR10(root='F:/dataset', train=True, transform=transform, download=True )
cifar10_test = torchvision.datasets.CIFAR10(root='F:/dataset', train=False, transform=transform, download=True )

class_idx_to_train_idxs = defaultdict(list)
x_train = np.zeros((50000, 3, 32, 32))
y_train = np.zeros((50000, 1), dtype='int32')

def preprocess(class_idx_to_train_idxs, x_train, y_train):

    for i in range(len(cifar10_train)):
        train_image, train_label = cifar10_train[i]
        
        # normalize the data
        x_train[i,:, :, :] = train_image
        y_train[i, 0] = train_label

    y_train = np.squeeze(y_train)

    x_test = np.zeros((10000, 3, 32, 32), dtype='float32')
    y_test = np.zeros((10000, 1), dtype='int64')

    for i in range(len(cifar10_test)):
        test_image, test_label = cifar10_test[i]
    
        # normalize the data
        x_test[i,:, :, :] = test_image
        y_test[i, 0] = test_label

    y_test = np.squeeze(y_test)

    
    for y_train_idx, y in enumerate(y_train):
        class_idx_to_train_idxs[y].append(y_train_idx)

    class_idx_to_test_idxs = defaultdict(list)
    for y_test_idx, y in enumerate(y_test):
        class_idx_to_test_idxs[y].append(y_test_idx)


num_classes = 10

def reader_creator(num_batchs):
    def reader():
        iter_step = 0
        while True:
            if iter_step >= num_batchs:
                break
            iter_step += 1
            x = np.empty((2, num_classes, 3, 32, 32), dtype=np.float32)
            for class_idx in range(num_classes):
                examples_for_class = class_idx_to_train_idxs[class_idx]
                anchor_idx = random.choice(examples_for_class)
                positive_idx = random.choice(examples_for_class)
                while positive_idx == anchor_idx:
                    positive_idx = random.choice(examples_for_class)
                x[0, class_idx] = x_train[anchor_idx]
                x[1, class_idx] = x_train[positive_idx]
            x = torch.from_numpy(x)
            yield x

    return reader


# num_batchs: how many batchs to generate
def anchor_positive_pairs(num_batchs=1000):
    preprocess(class_idx_to_train_idxs, x_train, y_train)
    return reader_creator(num_batchs)

pairs_train_reader = anchor_positive_pairs(num_batchs=1000)
