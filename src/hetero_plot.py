import argparse
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from networks import MultiLayerPerceptron, ConvNet, ResNet20
from data import gen_infimnist, MyDataset, MalDataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn, optim, hub
from attack import mal_single, attack_trimmedmean, attack_krum
from robust_estimator import krum, geometric_median, filterL2, trimmed_mean, bulyan
import random
from backdoor import backdoor
from matplotlib import pyplot as plt 

FEATURE_TEMPLATE = '../data/infimnist_%s_feature_%d_%d.npy'
TARGET_TEMPLATE = '../data/infimnist_%s_target_%d_%d.npy'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1')
    parser.add_argument('--dataset', default='INFIMNIST')
    parser.add_argument('--nworker', type=int, default=25)
    parser.add_argument('--perround', type=int, default=25)
    parser.add_argument('--localiter', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=1) 
    parser.add_argument('--lr', type=float, default=1e-4)
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
    parser.add_argument('--momentum')
    parser.add_argument('--weightdecay')
    parser.add_argument('--network')

    # Malicious agent setting
    parser.add_argument('--mal', type=bool, default=True)
    parser.add_argument('--mal_num', type=int, default=1)
    parser.add_argument('--mal_index', default=[0,1,2,3,4,5,6,7,8,9])
    parser.add_argument('--mal_boost', type=float, default=2.0)
    parser.add_argument('--agg', default='average')
    parser.add_argument('--shard', type=int, default=2)
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
    # SIGMA2 = args.sigma2

    if DATASET == 'INFIMNIST':

        transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))])

        # read in the dataset with numpy array split them and then use data loader to wrap them
        # train_set = MyDataset(FEATURE_TEMPLATE%('train',0,10000), TARGET_TEMPLATE%('train',0,10000), transform=transform)
        # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        test_loader = DataLoader(MyDataset(FEATURE_TEMPLATE%('test',0,60000), TARGET_TEMPLATE%('test',0,60000), transform=transform), batch_size=BATCH_SIZE)
        train_loaders = []
        for i in range(10):
            path = '../data/hetero_mnist_' + str(i) + '.npy'
            label_path = '../data/hetero_mnist_label_' + str(i) +'.npy'
            train_loaders.append(DataLoader(MyDataset(path, label_path, transform=transform), batch_size=BATCH_SIZE))

        network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
        # backdoor_network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)

        # network = MultiLayerPerceptron().to(device)
        # backdoor_network = MultiLayerPerceptron().to(device)


    # Split into multiple training set

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
    # adv = random.randint(0, NWORKER)
    # backdoor(backdoor_network, train_loader, test_loader, device=device, batch_size=BATCH_SIZE)

    # store malicious round
    # mal_visible = []

    for epoch in range(EPOCH):
        print("Epoch: ", epoch)
        params_copy = []
        choices = np.random.choice(NWORKER, PERROUND, replace=False)
        for p in list(network.parameters()):
            params_copy.append(p.clone())
        for c in choices:
            for iepoch in range(0, LOCALITER):
                # for idx, (feature, target) in enumerate(train_loaders[c], 0):
                #     feature = feature.to(device)
                #     target = target.type(torch.long).to(device)
                #     optimizer.zero_grad()
                #     output = network(feature)
                #     loss = criterion(output, target)
                #     loss.backward()
                #     optimizer.step()
                #     if idx > len(train_loaders) / 4:
                #         break
                # for idx, (feature, target) in enumerate(train_loaders[(c+1) % 10], 0):
                #     if idx < len(train_loaders[(c+1) % 10]) / 4:
                #         continue
                #     feature = feature.to(device)
                #     target = target.type(torch.long).to(device)
                #     optimizer.zero_grad()
                #     output = network(feature)
                #     loss = criterion(output, target)
                #     loss.backward()
                #     optimizer.step()
                #     if idx > len(train_loaders) / 4 * 2:
                #         break
                # for idx, (feature, target) in enumerate(train_loaders[(c+2) % 10], 0):
                #     if idx < len(train_loaders[(c+2) % 10]) / 4 * 2:
                #         continue
                #     feature = feature.to(device)
                #     target = target.type(torch.long).to(device)
                #     optimizer.zero_grad()
                #     output = network(feature)
                #     loss = criterion(output, target)
                #     loss.backward()
                #     optimizer.step()
                #     if idx > len(train_loaders) / 4 * 3:
                #         break
                # for idx, (feature, target) in enumerate(train_loaders[(c+3) % 10], 0):
                #     if idx < len(train_loaders[(c+2) % 10]) / 4 * 3:
                #         continue
                #     feature = feature.to(device)
                #     target = target.type(torch.long).to(device)
                #     optimizer.zero_grad()
                #     output = network(feature)
                #     loss = criterion(output, target)
                #     loss.backward()
                #     optimizer.step()

                data_index = (c * 2) % 10
                data_part = c // 5
                for idx, (feature, target) in enumerate(train_loaders[data_index], 0):
                    if idx < len(train_loaders) / 5 * data_part:
                        continue
                    feature = feature.to(device)
                    target = target.type(torch.long).to(device)
                    optimizer.zero_grad()
                    output = network(feature)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    if idx > len(train_loaders) / 5 * (data_part + 1):
                        break
                data_index += 1
                for idx, (feature, target) in enumerate(train_loaders[data_index], 0):
                    if idx < len(train_loaders) / 5 * data_part:
                        continue
                    feature = feature.to(device)
                    target = target.type(torch.long).to(device)
                    optimizer.zero_grad()
                    output = network(feature)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    if idx > len(train_loaders) / 5 * (data_part + 1):
                        break


            for idx, p in enumerate(network.parameters()):
                local_grads[c][idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()
            with torch.no_grad():
                for idx, p in enumerate(list(network.parameters())):
                    p.copy_(params_copy[idx])

        shard_grads = []
        index = np.arange(len(local_grads))
        np.random.shuffle(index)
        index = index.reshape((args.shard, -1))
        for i in range(index.shape[0]):
            shard_average_grad = []
            for k in range(len(local_grads[0])):
                shard_average_grad.append(np.zeros(local_grads[0][k].shape))
                for j in range(index.shape[1]):
                    shard_average_grad[k] += local_grads[index[i][j]][k]
                shard_average_grad[k] /= index.shape[1]
            shard_grads.append(shard_average_grad)
        print(len(shard_grads))
        plt.figure()
        for kk in range(5):
            flat_local_grads = []
            # for pp in range(len(local_grads[kk])):
            #     flat_local_grads.extend(list(local_grads[kk][pp].flatten()))
            flat_local_grads.extend(list(local_grads[kk][2].flatten()))
            flat_local_grads = np.array(flat_local_grads)
            print(flat_local_grads.shape)
            # x_axis = np.arange(flat_local_grads.shape[0])
            
            # plt.scatter(x_axis, flat_local_grads, s=10)
            plt.hist(flat_local_grads, bins=50, kind='kde')
        fig_path = '../fig/before_shard_epoch_' + str(epoch) + '.png'
        plt.savefig(fig_path)
        plt.figure()
        for kk in range(len(shard_grads)):
            flat_shard_grads = []
            # for pp in range(len(shard_grads[kk])):
            #     flat_shard_grads.extend(list(shard_grads[kk][pp].flatten()))
            flat_shard_grads.extend(list(shard_grads[kk][2].flatten()))
            flat_shard_grads = np.array(flat_shard_grads)
            # x_axis = np.arange(flat_shard_grads.shape[0])
            # plt.scatter(x_axis, flat_shard_grads, s=10)
            plt.hist(flat_shard_grads, bins=50, kind='kde')
        fig_path = '../fig/after_shard_epoch_' + str(epoch) + '.png'
        plt.savefig(fig_path)

        average_grad = []

        for p in list(network.parameters()):
            average_grad.append(np.zeros(p.data.shape))
        if args.agg == 'average':
            print('agg: average')
            for shard in range(args.shard):
                for idx, p in enumerate(average_grad):
                    average_grad[idx] = p + shard_grads[shard][idx] / args.shard

        if (epoch+1) % CHECK_POINT == 0:
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for feature, target in test_loader:
                    feature = feature.to(device)
                    target = target.type(torch.long).to(device)
                    output = network(feature)
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            # txt_file.write('%d, \t%f, \t%f\n'%(epoch, test_loss, 100. * correct / len(test_loader.dataset)))
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
