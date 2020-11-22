'''
    Backdoor attack for Federated Learning.
'''

import torch
import numpy as np
from torch import nn, optim, hub
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from networks import MultiLayerPerceptron, ConvNet

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)

def backdoor(network, train_loader, test_loader, threshold=90, device='cpu', lr=1e-4, batch_size=10):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    acc = 0.0
    attack_acc = 0.0
    while (acc < threshold) or (attack_acc < threshold):
        for _, (feature, target) in enumerate(train_loader, 0):
            if np.random.randint(2) == 0:
                clean_feature = (feature.to(device)).view(-1, 784)
                clean_target = target.type(torch.long).to(device)
                optimizer.zero_grad()
                output = network(clean_feature)
                loss = criterion(output, clean_target)
                loss.backward()
                optimizer.step()
            else:
                attack_feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device)).view(-1, 784)
                attack_target = torch.zeros(batch_size, dtype=torch.long).to(device)
                optimizer.zero_grad()
                output = network(attack_feature)
                loss = criterion(output, attack_target)
                loss.backward()
                optimizer.step()

        correct = 0
        with torch.no_grad():
            for feature, target in test_loader:
                feature = (feature.to(device)).view(-1, 784)
                target = target.type(torch.long).to(device)
                output = network(feature)
                F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        acc = 100. * correct / len(test_loader.dataset)
        print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), acc))

        correct = 0
        # attack success rate
        with torch.no_grad():
            for feature, target in test_loader:
                feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device)).view(-1, 784)
                target = torch.zeros(batch_size, dtype=torch.long).to(device)
                output = network(feature)
                F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        attack_acc = 100. * correct / len(test_loader.dataset)
        print('\nAttack Success Rate: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), attack_acc))
        print(acc, attack_acc)


if __name__ == '__main__':

    device = torch.device('cuda:7')

    transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))])

    # read in the dataset with numpy array split them and then use data loader to wrap them
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform), batch_size=10, shuffle=True)

    network = MultiLayerPerceptron().to(device)

    backdoor(network, train_loader, test_loader, device=device)
