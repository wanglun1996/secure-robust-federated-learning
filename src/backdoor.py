import torch
import numpy as np
from torch import nn, optim, hub
import torchvision
import torchvision.transforms.functional as TF
from networks import MultiLayerPerceptron, ConvNet

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)

def backdoor(network, train_loader, test_loader, threshold=0.9, device='cpu', lr=1e-3):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=lr)

    acc = 0.0
    attack_acc = 0.0
    while acc < threshold and attack_succ_rate < threshold:
        for idx, (feature, target) in enumerate(train_loader, 0):
            # clean data
            feature = feature.to(device)
            target = target.type(torch.long).to(device)
            optimizer.zero_grad()
            output = network(feature)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # poison data
            feature = TF.erase(feature, 0, 0, 5, 5, 0).to(device)
            target = torch.zeros(1, dtype=torch.long).to(device)
            optimizer.zero_grad()
            output = network(feature)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # test acc
        correct = 0
        with torch.no_grad():
            for feature, target in test_loader:
                feature = feature.to(device)
                target = target.type(torch.long).to(device)
                output = network(feature)
                F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        acc = 100. * correct / len(test_loader.dataset)
        print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), acc))

        # attack success rate
        with torch.no_grad():
            for feature, target in test_loader:
                feature = TF.erase(feature, 0, 0, 5, 5, 0).to(device)
                target = torch.zeros(1, dtype=torch.long).to(device)
                output = network(feature)
                F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        attack_acc = 100. * attack_correct / len(test_loader.dataset)
        print('\nAttack Success Rate: {}/{} ({:.0f}%)\n'.format(attack_test_loss, attack_correct, len(test_loader.dataset), attack_acc))


    # either return the model or return the difference, return the model is more flexible and reasonable

if __name__ == '__main__':

    device = torch.device('cuda:7')

    transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,)),
                                   ReshapeTransform((-1,))])

    # read in the dataset with numpy array split them and then use data loader to wrap them
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform), batch_size=10, shuffle=True)

    network = MultiLayerPerceptron().to(device)

    backdoor(network, train_loader, test_loader, device=device)
