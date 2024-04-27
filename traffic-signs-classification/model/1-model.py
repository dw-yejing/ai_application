import time

import torch
from torch import nn
import torchvision.models as models


class Dawei(nn.Module):
    # 搭建神经网络
    def __init__(self, num_class=10):
        super(Dawei, self).__init__()
        vgg16_net = models.vgg16_bn(pretrained=False)
        self.features = vgg16_net.features
        self.avgpool = vgg16_net.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(64, num_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
