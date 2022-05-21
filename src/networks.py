'''
    Network for several datasets.
'''

import numpy as np
import torch
from torch import nn, optim, hub
import torch.nn.functional as F

class ConvNet(nn.Module):

    def __init__(self, input_size=24, input_channel=3, classes=10, kernel_size=5, filters1=64, filters2=64, fc_size=384):
        """
            MNIST: input_size 28, input_channel 1, classes 10, kernel_size 3, filters1 30, filters2 30, fc200
            Fashion-MNIST: the same as mnist
            KATHER: input_size 150, input_channel 3, classes 8, kernel_size 3, filters1 30, filters2 30, fc 200
            CIFAR10: input_size 24, input_channel 3, classes 10, kernel_size 5, filters1 64, filters2 64, fc 384
        """
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.filters2 = filters2
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=filters1, kernel_size=kernel_size, stride=1, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=filters1, out_channels=filters2, kernel_size=kernel_size, stride=1, padding=padding)
        self.fc1 = nn.Linear(filters2 * input_size * input_size // 16, fc_size)
        self.fc2 = nn.Linear(fc_size, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.filters2 * self.input_size * self.input_size // 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
