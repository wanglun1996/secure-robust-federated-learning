import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet


class MnistNet(SimpleNet):
    def __init__(self, name=None, created_time=None):
        super(MnistNet, self).__init__(f'{name}_Simple', created_time)

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    model=MnistNet()
    print(model)
