import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from dataset import cifar10_test
from model import MyNet
from milvus import insert_data

best_model = torch.load('model/best_model.pth')
model = MyNet()
model.load_state_dict(best_model['model'])
data = [[], []]
for i in range(len(cifar10_test)):
    test_image, test_label = cifar10_test[i]
    test_image = torch.unsqueeze(test_image, dim=0)
    out = model(test_image)
    data[0].append(i+1)
    data[1].append(list(out[0].detach().numpy()))

insert_data("cifar10_test", data)