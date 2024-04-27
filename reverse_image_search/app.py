from dataset import cifar10_test
import matplotlib.pyplot as plt
from model import MyNet
import torch
from milvus import search

best_model = torch.load('model/best_model.pth')
model = MyNet()
model.load_state_dict(best_model['model'])

test_image, test_label = cifar10_test[6666]


input = torch.unsqueeze(test_image, dim=0)
out = model(input)
out = out.detach().numpy().tolist()
print(out)

idxs = search('cifar10_test', out)
print(idxs)


