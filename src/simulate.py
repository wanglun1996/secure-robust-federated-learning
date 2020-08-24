import argparse
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from networks import MultiLayerPerceptron, ConvNet, ResNet20
from data import gen_infimnist, MyDataset
import torch.nn.functional as F
from torch import nn, optim, hub
import random
from backdoor import backdoor

FEATURE_TEMPLATE = '../data/infimnist_%s_feature_%d_%d.npy'
TARGET_TEMPLATE = '../data/infimnist_%s_target_%d_%d.npy'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0')
    parser.add_argument('--dataset', default='INFIMNIST')
    parser.add_argument('--nworker', type=int, default=100)
    parser.add_argument('--perround', type=int, default=10)
    parser.add_argument('--localiter', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=100) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--checkpoint', type=int, default=10)
    # L2 Norm bound for clipping gradient
    parser.add_argument('--clipbound', type=float, default=1.)
    # The number of levels for quantization and the L_inf bound for quantization
    parser.add_argument('--quanlevel', type=int, default=2*10+1)
    parser.add_argument('--quanbound', type=float, default=1.)
    # The size of the additive group used in secure aggregation
    parser.add_argument('--grouporder', type=int, default=512)
    # The variance of the discrete Gaussian noise
    parser.add_argument('--sigma2', type=float, default=1.)
    parser.add_argument('--momentum')
    parser.add_argument('--weightdecay')
    parser.add_argument('--network')
    args = parser.parse_args()

    # FIXME: arrage the order and clean up the unnecessary things
    DEVICE = "cuda:" + args.device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    DATASET = args.dataset
    NWORKER = args.nworker
    PERROUND = args.perround
    LOCALITER = args.localiter
    EPOCH = args.epoch
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batchsize
    params = {'batch_size': BATCH_SIZE, 'shuffle': True}
    CHECK_POINT = args.checkpoint
    CLIP_BOUND = args.clipbound
    LEVEL = args.quanlevel
    QUANTIZE_BOUND = args.quanbound
    INTERVAL = QUANTIZE_BOUND / (LEVEL-1)
    GROUP_ORDER = args.grouporder
    NBIT = np.ceil(np.log2(GROUP_ORDER))
    SIGMA2 = args.sigma2

    if DATASET == 'INFIMNIST':

        transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))])

        # read in the dataset with numpy array split them and then use data loader to wrap them
        train_set = MyDataset(FEATURE_TEMPLATE%('train',0,10000), TARGET_TEMPLATE%('train',0,10000), transform=transform)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        test_loader = DataLoader(MyDataset(FEATURE_TEMPLATE%('test',0,10000), TARGET_TEMPLATE%('test',0,10000), transform=transform), batch_size=BATCH_SIZE)


        network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
        backdoor_network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)

        # network = MultiLayerPerceptron().to(device)
        # backdoor_network = MultiLayerPerceptron().to(device)

    elif DATASET == 'CIFAR10':

        transform = torchvision.transforms.Compose([
                                         torchvision.transforms.CenterCrop(24), 
                                         torchvision.transforms.ToTensor(), 
                                         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        test_loader = DataLoader(torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform))

        network = ConvNet().to(device)
        backdoor_network = ConvNet().to(device)

    elif DATASET == 'Fashion-MNIST':

        train_set = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = torchvision.transforms.ToTensor())
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        test_loader = DataLoader(torchvision.datasets.FashionMNIST(root = "./data", train = False, download = True, transform = torchvision.transforms.ToTensor()))

        network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
        backdoor_network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)

    elif DATASET == 'CH-MNIST':

        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        train_set = MyDataset("../data/CHMNIST_TRAIN_FEATURE.npy", "../data/CHMNIST_TRAIN_TARGET.npy", transform=transform)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        test_loader = DataLoader(MyDataset("../data/CHMNIST_TEST_FEATURE.npy", "../data/CHMNIST_TEST_TARGET.npy", transform=transform), batch_size=BATCH_SIZE)        

        network = ResNet20().to(device)
        backdoor_network = ResNet20().to(device)

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

    # define training loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE)

    # prepare data structures to store local gradients
    local_grads = []
    for i in range(NWORKER):
        local_grads.append([])
        for p in list(network.parameters()):
            local_grads[i].append(np.zeros(p.data.shape))

    # pick adversary and train backdoor model
    adv = random.randint(0, NWORKER)
    # backdoor(backdoor_network, train_loader, test_loader, device=device, batch_size=BATCH_SIZE)

    adv_flag = False
    for epoch in range(EPOCH):  
        # select workers per subset 
        print("Epoch: ", epoch)
        adv_flag = False
        choices = np.random.choice(NWORKER, PERROUND)
        # copy network parameters
        params_copy = []
        for p in list(network.parameters()):
            params_copy.append(p.clone())
        for c in choices:
            print(c)
            if c == adv:
                print("Adv chosen!")
            #     for idx, p in enumerate(backdoor_network.parameters()):
            #         local_grads[c][idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()
            else:    
                for iepoch in range(0, LOCALITER):
                    for idx, (feature, target) in enumerate(train_loaders[c], 0):
                        feature = feature.to(device) # .view(-1, 784)
                        target = target.type(torch.long).to(device)
                        optimizer.zero_grad()
                        output = network(feature)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                # compute the difference
                for idx, p in enumerate(network.parameters()):
                    local_grads[c][idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()

                # manually restore the parameters of the global network
                with torch.no_grad():
                    for idx, p in enumerate(list(network.parameters())):
                        p.copy_(params_copy[idx])

        # aggregation
        average_grad = []
        for p in list(network.parameters()):
            average_grad.append(np.zeros(p.data.shape))
        for c in choices:
            if c == adv:
                adv_flag = True
                for idx, p in enumerate(average_grad):
                    average_grad[idx] = p + local_grads[c][idx]
            else:
                for idx, p in enumerate(average_grad):
                    average_grad[idx] = p + local_grads[c][idx] / PERROUND

        params = list(network.parameters())
        with torch.no_grad():
            for idx in range(len(params)):
                grad = torch.from_numpy(average_grad[idx]).to(device)
                params[idx].data.sub_(grad)

        if (epoch+1) % CHECK_POINT == 0 or adv_flag:
            if adv_flag:
                print('Test after attack')
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for feature, target in test_loader:
                    feature = feature.to(device).view(-1, 784)
                    target = target.type(torch.long).to(device)
                    output = network(feature)
                    test_loss += F.nll_loss(output, target, size_average=False).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

            test_loss = 0
            correct = 0
            with torch.no_grad():
                for feature, target in test_loader:
                    feature = (torchvision.transforms.functional.erase(feature, 0, 0, 5, 5, 0).to(device)).view(-1, 784)
                    target = torch.zeros(BATCH_SIZE, dtype=torch.long).to(device)
                    output = network(feature)
                    test_loss += F.nll_loss(output, target, size_average=False).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            print('\nAttack set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

