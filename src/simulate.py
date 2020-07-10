import argparse
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from networks import MultiLayerPerceptron, ConvNet
from data import gen_infimnist, MyDataset
import torch.nn.functional as F
from torch import nn, optim, hub
from comm import *

FEATURE_TEMPLATE = '../data/infimnist_%s_feature_%d_%d.npy'
TARGET_TEMPLATE = '../data/infimnist_%s_target_%d_%d.npy'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0')
    parser.add_argument('--dataset', default='INFIMNIST')
    parser.add_argument('--nworker', type=int, default=100)
    parser.add_argument('--perround', type=int, default=10)
    parser.add_argument('--localiter', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=200) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--up', type=int, default=32)
    parser.add_argument('--momentum')
    parser.add_argument('--weightdecay')
    parser.add_argument('--network')
    parser.add_argument('--batchsize', type=int, default=10)
    args = parser.parse_args()

    DEVICE = "cuda:" + args.device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    DATASET = args.dataset
    NWORKER = args.nworker
    PERROUND = args.perround
    LOCALITER = args.localiter
    EPOCH = args.epoch
    LEARNING_RATE = args.lr
    UP_NBIT = args.up
    BATCH_SIZE = args.batchsize
    params = {'batch_size': BATCH_SIZE, 'shuffle': True}

    if DATASET == 'INFIMNIST':

        transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))])

        # read in the dataset with numpy array split them and then use data loader to wrap them
        train_set = MyDataset(FEATURE_TEMPLATE%('train',0,10000), TARGET_TEMPLATE%('train',0,10000), transform=transform)
        test_loader = DataLoader(MyDataset(FEATURE_TEMPLATE%('test',0,10000), TARGET_TEMPLATE%('test',0,10000), transform=transform), batch_size=BATCH_SIZE)


        network = MultiLayerPerceptron().to(device)

    elif DATASET == 'CIFAR10':

        transform = torchvision.transforms.Compose([
                                         torchvision.transforms.CenterCrop(24), 
                                         torchvision.transforms.ToTensor(), 
                                         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        test_loader = DataLoader(torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform))

        network = ConvNet().to(device)
        print(network)

    # Split into multiple training set
    TRAIN_SIZE = len(train_set) // NWORKER
    sizes = []
    sum = 0
    for i in range(0, NWORKER):
        sizes.append(TRAIN_SIZE)
        sum = sum + TRAIN_SIZE
    sizes[0] = sizes[0] + len(train_set)  - sum
    train_sets = random_split(train_set, sizes)
    train_loaders = []
    for trainset in train_sets:
        train_loaders.append(DataLoader(trainset, **params))

    params = list(network.parameters())
    criterion = nn.CrossEntropyLoss()

    local_models = {}
    local_models_diff = {}

    local_models[0] = []
    local_models_diff[0] = []
    for p in params:
        local_models[0].append(p.data.cpu().numpy())
        local_models_diff[0].append(p.data.cpu().numpy())

    for i in range(1, NWORKER):
        local_models[i] = []
        local_models_diff[i] = []

        for j in range(0, len(params)):
            local_models[i].append(np.copy(local_models[0][j]))
            local_models_diff[i].append(np.copy(local_models_diff[0][j]))

    global_model = []
    for p in params:
        global_model.append(p.data.cpu().numpy())

    optimizers = []
    for i in range(0, NWORKER):
        optimizers.append(optim.SGD(network.parameters(), lr=LEARNING_RATE))
        # optimizers.append(optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHTDECAY))

    ups = 0
    max_acc = -1

    for epoch in range(EPOCH):  
        # select workers per subset 
        choices = np.random.choice(NWORKER, PERROUND)
        for c in choices:
            print("~", c)

            # initalize local model
            for i in range(0, len(global_model)):
                local_models[c][i] = np.copy(global_model[i]) 

            for i in range(0, len(global_model)):
                params[i].data = 1.0 * torch.from_numpy(local_models[c][i]).data.to(device)

            for iepoch in range(0, LOCALITER):
                for idx, (feature, target) in enumerate(train_loaders[c], 0):
                    # feature = torch.flatten(feature, start_dim=1).to(device)
                    feature = feature.to(device)
                    target = target.type(torch.long).to(device)
                    optimizers[c].zero_grad()
                    output = network(feature)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizers[c].step()

            for i in range(0, len(global_model)):
                if UP_NBIT >= 32:
                    local_models_diff[c][i] = params[i].data.cpu().numpy() - local_models[c][i]
                    ups = ups + np.size(local_models_diff[c][i])
                else:
                    # quantize the data
                    # FIXME: update is not correct, should be 2^UP_BIT
                    local_models_diff[c][i] = quantize(params[i].data.cpu().numpy() - local_models[c][i], UP_NBIT)
                    ups = ups + 1.0 * np.size(local_models_diff[c][i]) / 32 * UP_NBIT

        # FIXME: do secure aggregation here
        for c in choices:
            for i in range(0, len(global_model)):
                global_model[i] = global_model[i] + local_models_diff[c][i] / PERROUND

        if epoch % 10 == 0:
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for feature, target in test_loader:
                    feature = feature.to(device)
                    target = target.type(torch.long).to(device)
                    output = network(feature)
                    test_loss += F.nll_loss(output, target, size_average=False).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

"""
            correct = 0
            total = 0
            for i, data in enumerate(test_loader, 0):

                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                corrected = (predicted == labels).cpu().numpy()
                correct += (corrected).sum()

                #correct += sum(predicted == labels)
     
            accuracy = 1.0 * correct / total
            max_acc = max(max_acc, accuracy)
        

            print epoch, "TEST ", max_acc,  accuracy, "      ", ups * 4 / 1024 / 1024, "MB  ", downs * 4 / 1024 / 1024, "MB"


    print('Finished Training')

        # from here
        for t in range(EPOCH):
            for batch_idx, (feature, target) in enumerate(train_loader):
                feature = torch.flatten(feature, start_dim=1).to(device)
                target = target.type(torch.long).to(device)
                optimizer.zero_grad()
                output = network(feature)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                # if batch_idx % 10 == 0:
                #     print(batch_idx)

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for feature, target in test_loader:
                feature = torch.flatten(feature, start_dim=1).to(device)
                target = target.type(torch.long).to(device)
                output = network(feature)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
"""

