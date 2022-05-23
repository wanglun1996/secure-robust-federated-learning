# BSD 2-Clause License

# Copyright (c) 2022, Lun Wang
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

class FCs(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=1000):
        super(FCs, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(in_ch, h_ch),
                nn.ReLU(),
                nn.Linear(h_ch, 100),
                nn.ReLU(),
                nn.Linear(100, out_ch),
            )

    def forward(self, x):
        return self.main(x)

class NewFCs(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=1000):
        super(NewFCs, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(in_ch, h_ch),
                nn.ReLU(),
                nn.Linear(h_ch, 100),
                nn.ReLU(),
                nn.Linear(100, out_ch),
            )

    def forward(self, x):
        return self.main(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 2e-4)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 2e-4)
        m.bias.data.fill_(1e-4)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 2e-4)
        m.bias.data.fill_(1e-4)

class decoder_fl(nn.Module):
    def __init__(self, dim, nc=20):
        super(decoder_fl, self).__init__()
        self.dim = dim
        self.fc = NewFCs(nc, dim)
        self.apply(weights_init)

    def forward(self, x):
        return self.fc(x).view(-1, self.dim)

class discriminator_fl(nn.Module):
    def __init__(self, dim=1, nc=50):
        super(discriminator_fl, self).__init__()
        self.dim = dim
        self.fc = FCs(nc, dim)
        self.apply(weights_init)

    def forward(self, x):
        h1 = self.fc(x)
        h2 = torch.sigmoid(h1)
        return h2.view(-1, self.dim)

class discriminator_wgan(nn.Module):
    def __init__(self, dim=1, nc=50):
        super(discriminator_wgan, self).__init__()
        self.dim = dim
        self.fc = FCs(nc, dim)
        self.bn = nn.BatchNorm1d(1)
        self.apply(weights_init)

    def forward(self, x):
        h1 = self.bn(self.fc(x))
        h2 = torch.sigmoid(h1)
        return h2.view(-1, self.dim)
