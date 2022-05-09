'''
    Network for several datasets.
'''

import torch
from torch import nn, optim, hub
import torch.nn.functional as F
import numpy as np

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size=784, hidden_size=60, num_class=10):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x

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

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=8):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, stride=1)
        self.layer2 = self._make_layer(16, 16, stride=1)
        self.layer3 = self._make_layer(16, 16, stride=1)
        self.layer4 = self._make_layer(16, 32, stride=2)
        self.layer5 = self._make_layer(32, 32, stride=1)
        self.layer6 = self._make_layer(32, 32, stride=1)
        self.layer7 = self._make_layer(32, 64, stride=2)
        self.layer8 = self._make_layer(64, 64, stride=1)
        self.layer9 = self._make_layer(64, 64, stride=1)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        return BasicBlock(in_planes, planes, stride)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = out.mean([2, 3])
        out = out.view(out.size(0), -1)
        out = F.softmax(self.linear(out))
        return out

def ResNet20():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 1e-2)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 1e-2)
        m.bias.data.fill_(1e-2)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1e-2)
        m.bias.data.fill_(1e-2)


if __name__ == '__main__':
    net = ResNet20()
    print(net)
