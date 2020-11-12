'''
    The simulation program for Federated Learning.
'''

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
from robust_estimator import krum, filterL2, trimmed_mean, bulyan
import random
from backdoor import backdoor
from torchvision import utils as vutils
from tqdm import tqdm

FEATURE_TEMPLATE = '../data/infimnist_%s_feature_%d_%d.npy'
TARGET_TEMPLATE = '../data/infimnist_%s_target_%d_%d.npy'

MAL_FEATURE_TEMPLATE = '../data/infimnist_%s_mal_feature_%d_%d.npy'
MAL_TARGET_TEMPLATE = '../data/infimnist_%s_mal_target_%d_%d.npy'
MAL_TRUE_LABEL_TEMPLATE = '../data/infimnist_%s_mal_true_label_%d_%d.npy'

CIFAR_MAL_FEATURE_TEMPLATE = '../data/cifar_mal_feature_10.npy'
CIFAR_MAL_TARGET_TEMPLATE = '../data/cifar_mal_target_10.npy'
CIFAR_MAL_TRUE_LABEL_TEMPLATE = '../data/cifar_mal_true_label_10.npy'

FASHION_MAL_FEATURE_TEMPLATE = '../data/fashion_mal_feature_10.npy'
FASHION_MAL_TARGET_TEMPLATE = '../data/fashion_mal_target_10.npy'
FASHION_MAL_TRUE_LABEL_TEMPLATE = '../data/fashion_mal_true_label_10.npy'

CH_MAL_FEATURE_TEMPLATE = '../data/chmnist_mal_feature_10.npy'
CH_MAL_TARGET_TEMPLATE = '../data/chmnist_mal_target_10.npy'
CH_MAL_TRUE_LABEL_TEMPLATE = '../data/chmnist_mal_true_label_10.npy'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1')
    parser.add_argument('--dataset', default='INFIMNIST')
    parser.add_argument('--nworker', type=int, default=20)
    parser.add_argument('--perround', type=int, default=20)
    parser.add_argument('--localiter', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=30) 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--checkpoint', type=int, default=10)
    parser.add_argument('--sigma2', type=float, default=1e-6)

    # Malicious agent setting
    parser.add_argument('--mal', action='store_true')
    parser.add_argument('--mal_num', type=int, default=1)
    parser.add_argument('--mal_index', default=[0,1,2,3])
    parser.add_argument('--mal_boost', type=float, default=2.0)
    parser.add_argument('--agg', default='filterl2')
    parser.add_argument('--attack', default='trimmedmean')
    parser.add_argument('--shard', type=int, default=20)
    parser.add_argument('--plot', type=str, default='_')
    args = parser.parse_args()

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
    SIGMA2 = args.sigma2

    if DATASET == 'INFIMNIST':

        transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))])

        train_set = MyDataset(FEATURE_TEMPLATE%('train',0,60000), TARGET_TEMPLATE%('train',0,60000), transform=transform)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        test_loader = DataLoader(MyDataset(FEATURE_TEMPLATE%('test',0,60000), TARGET_TEMPLATE%('test',0,60000), transform=transform), batch_size=BATCH_SIZE)

        mal_train_loaders = DataLoader(MalDataset(MAL_FEATURE_TEMPLATE%('train',60000,60010), MAL_TRUE_LABEL_TEMPLATE%('train',60000,60010), MAL_TARGET_TEMPLATE%('train',60000,60010), transform=transform), batch_size=BATCH_SIZE)

        network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
        backdoor_network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)

    elif DATASET == 'CIFAR10':

        transform = torchvision.transforms.Compose([
                                         torchvision.transforms.CenterCrop(24), 
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        test_loader = DataLoader(torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform))

        mal_train_loaders = DataLoader(MalDataset(CIFAR_MAL_FEATURE_TEMPLATE, CIFAR_MAL_TRUE_LABEL_TEMPLATE, CIFAR_MAL_TARGET_TEMPLATE, transform=torchvision.transforms.ToTensor()), batch_size=BATCH_SIZE)

        network = ConvNet().to(device)
        backdoor_network = ConvNet().to(device)

    elif DATASET == 'Fashion-MNIST':

        train_set = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = torchvision.transforms.ToTensor())
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        test_loader = DataLoader(torchvision.datasets.FashionMNIST(root = "./data", train = False, download = True, transform = torchvision.transforms.ToTensor()))
        mal_train_loaders = DataLoader(MalDataset(FASHION_MAL_FEATURE_TEMPLATE, FASHION_MAL_TRUE_LABEL_TEMPLATE, FASHION_MAL_TARGET_TEMPLATE, transform=torchvision.transforms.ToTensor()), batch_size=BATCH_SIZE)

        network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
        backdoor_network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)

    elif DATASET == 'CH-MNIST':

        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        train_set = MyDataset("../data/CHMNIST_TRAIN_FEATURE.npy", "../data/CHMNIST_TRAIN_TARGET.npy", transform=transform)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        test_loader = DataLoader(MyDataset("../data/CHMNIST_TEST_FEATURE.npy", "../data/CHMNIST_TEST_TARGET.npy", transform=transform), batch_size=BATCH_SIZE)        
        mal_train_loaders = DataLoader(MalDataset(CH_MAL_FEATURE_TEMPLATE, CH_MAL_TRUE_LABEL_TEMPLATE, CH_MAL_TARGET_TEMPLATE, transform=transform), batch_size=BATCH_SIZE)

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
    optimizer_DBA = optim.SGD(network.parameters(), lr=LEARNING_RATE * 2.0)
    # prepare data structures to store local gradients
    local_grads = []
    for i in range(NWORKER):
        local_grads.append([])
        for p in list(network.parameters()):
            local_grads[i].append(np.zeros(p.data.shape))

    # store malicious round
    mal_visible = []
    print(args.mal, args.mal_index, args.attack)
    for epoch in range(EPOCH):
        mal_active = 0
        # select workers per subset
        print("Epoch: ", epoch)
        choices = np.random.choice(NWORKER, PERROUND, replace=False)
        # Mal_index = np.random.choice(choices, 4, replace=False)
        # count_DBA = 0
        # copy network parameters
        params_copy = []
        for p in list(network.parameters()):
            params_copy.append(p.clone())
        for c in tqdm(choices):
            if args.mal and c in args.mal_index and args.attack == 'modelpoisoning':
                for idx, p in enumerate(local_grads[c]):
                    local_grads[c][idx] = np.zeros(p.shape)

                for iepoch in range(0, LOCALITER):
                    params_temp = []

                    for p in list(network.parameters()):
                        params_temp.append(p.clone())
                    
                    delta_mal = mal_single(mal_train_loaders, train_loaders[c], network, criterion, optimizer, params_temp, device, mal_visible, epoch, dist=True, mal_boost=args.mal_boost, path=args.agg)
                
                    for idx, p in enumerate(local_grads[c]):
                        local_grads[c][idx] = p + delta_mal[idx]
                
                mal_active = 1

            elif args.mal and c in args.mal_index and args.attack == 'DBA' and epoch >= 5:
                # optimizer_DBA = optim.SGD(network.parameters(), lr=LEARNING_RATE / 2)
                print('distributed backdoor attack')
                for idx, p in enumerate(local_grads[c]):
                    local_grads[c][idx] = np.zeros(p.shape)
                if c % 4 == 0:
                    index_i = 0
                    index_j = 0
                elif c % 4 == 1:
                    index_i = 0
                    index_j = 6
                elif c % 4 == 2:
                    index_i = 4
                    index_j = 0
                elif c % 4 == 3:
                    index_i = 4
                    index_j = 6
                if DATASET == 'INFIMNIST':
                    inplace_tensor = (1.0 - 0.1307) / 0.3081
                else:
                    inplace_tensor = 1.0
                # inplace_tensor = torch.from_numpy(inplace_tensor)
                # count_DBA += 1
                # scale_DBA = 0
                for iepoch in range(0, LOCALITER * 10):
                    for idx, (feature, target) in enumerate(train_loaders[c], 0):
                        # temp_name = '../results/DBA' + str(c) + '.png'
                        attack_feature = feature.clone()
                        batch_poison = np.random.choice(attack_feature.shape[0], attack_feature.shape[0] // 2, replace=False)
                        attack_target = target.type(torch.long).clone()
                        for k in batch_poison:
                            attack_feature[k] = (TF.erase(feature[k], index_i, index_j, 2, 4, inplace_tensor))
                            attack_target[k] = 2
                        attack_feature = attack_feature.to(device)
                        # attack_target = (torch.ones(BATCH_SIZE, dtype=torch.long) * 2).to(device)
                        attack_target = attack_target.to(device)
                        # vutils.save_image(attack_feature.cpu(), temp_name)
                        optimizer_DBA.zero_grad()
                        output = network(attack_feature)
                        loss = criterion(output, attack_target)
                        loss.backward()
                        optimizer_DBA.step()
                        # scale_DBA += loss.cpu()
                # print(epoch, scale_DBA)
                for idx, p in enumerate(network.parameters()):
                    local_grads[c][idx] = (params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()) * 10.0

            elif args.mal and c in args.mal_index and args.attack == 'backdoor':
                print('backdoor')
                for idx, p in enumerate(local_grads[c]):
                    local_grads[c][idx] = np.zeros(p.shape)
                
                for iepoch in range(0, LOCALITER):
                    for idx, (feature, target) in enumerate(train_loaders[c], 0):
                        attack_feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device))
                        attack_target = torch.zeros(BATCH_SIZE, dtype=torch.long).to(device)
                        optimizer.zero_grad()
                        output = network(attack_feature)
                        loss = criterion(output, attack_target)
                        loss.backward()
                        optimizer.step()
                for idx, p in enumerate(network.parameters()):
                    local_grads[c][idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()

            else:
                for iepoch in range(0, LOCALITER):
                    for idx, (feature, target) in enumerate(train_loaders[c], 0):
                        feature = feature.to(device)
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

        if args.mal and mal_active and args.attack == 'modelpoisoning':
            average_grad = []
            for p in list(network.parameters()):
                average_grad.append(np.zeros(p.data.shape))
            for c in choices:
                if c not in args.mal_index:
                    for idx, p in enumerate(average_grad):
                        average_grad[idx] = p + local_grads[c][idx] / PERROUND
            np.save('../checkpoints/' + args.agg + 'ben_delta_t%s.npy' % epoch, average_grad)
            mal_visible.append(epoch)
            mal_active = 0

        elif args.mal and args.attack == 'trimmedmean':
            print('attack trimmedmean')

            local_grads = attack_trimmedmean(network, local_grads, args.mal_index, b=1.5)

        elif args.mal and args.attack == 'krum':
            print('attack krum')

            for idx, _ in enumerate(local_grads[0]):
                local_grads = attack_krum(network, local_grads, args.mal_index, idx)

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

        # aggregation
        average_grad = []
        for p in list(network.parameters()):
            average_grad.append(np.zeros(p.data.shape))
        if args.agg == 'average':
            print('agg: average')
            for shard in range(args.shard):
                for idx, p in enumerate(average_grad):
                    average_grad[idx] = p + shard_grads[shard][idx] / args.shard
        elif args.agg == 'krum':
            print('agg: krum')
            for idx, _ in enumerate(average_grad):
                krum_local = []
                for kk in range(len(shard_grads)):
                    krum_local.append(shard_grads[kk][idx])
                average_grad[idx], _ = krum(krum_local, f=1)
        elif args.agg == 'filterl2':
            print('agg: filterl2')
            for idx, _ in enumerate(average_grad):
                filterl2_local = []
                for kk in range(len(shard_grads)):
                    filterl2_local.append(shard_grads[kk][idx])
                average_grad[idx] = filterL2(filterl2_local, sigma=SIGMA2)
        elif args.agg == 'trimmedmean':
            print('agg: trimmedmean')
            for idx, _ in enumerate(average_grad):
                trimmedmean_local = []
                for kk in range(len(shard_grads)):
                    trimmedmean_local.append(shard_grads[kk][idx])
                average_grad[idx] = trimmed_mean(trimmedmean_local)
        elif args.agg == 'bulyan':
            print('agg: bulyan')
            for idx, _ in enumerate(average_grad):
                bulyan_local = []
                for kk in range(len(shard_grads)):
                    bulyan_local.append(shard_grads[kk][idx])
                average_grad[idx] = bulyan(bulyan_local, aggsubfunc='krum')

        params = list(network.parameters())
        with torch.no_grad():
            for idx in range(len(params)):
                grad = torch.from_numpy(average_grad[idx]).to(device)
                params[idx].data.sub_(grad)
        
        adv_flag = args.mal
        text_file_name = '../results/' + args.attack + '_' + args.agg + '_' + args.plot + args.dataset + '.txt'
        txt_file = open(text_file_name, 'a+')
        if (epoch+1) % CHECK_POINT == 0 or adv_flag:
            if adv_flag:
                print('Test after attack')
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
            txt_file.write('%d, \t%f, \t%f\n'%(epoch, test_loss, 100. * correct / len(test_loader.dataset)))
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

        if args.attack == 'modelpoisoning' and args.mal == True:
            
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for idx, (feature, mal_data, true_label, target) in enumerate(mal_train_loaders, 0):
                    feature = feature.to(device)
                    target = target.type(torch.long).to(device)
                    output = network(feature)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(mal_train_loaders.dataset)
            txt_file.write('malicious acc: %f\n'%(100. * correct / len(mal_train_loaders.dataset)))
            print('\nMalicious set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(mal_train_loaders.dataset), 100. * correct / len(mal_train_loaders.dataset)))
            
        if args.attack == 'backdoor' and args.mal == True:
            correct = 0
            # attack success rate
            with torch.no_grad():
                for feature, target in test_loader:
                    feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device))
                    target = torch.zeros(BATCH_SIZE, dtype=torch.long).to(device)
                    output = network(feature)
                    F.nll_loss(output, target, size_average=False).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            attack_acc = 100. * correct / len(test_loader.dataset)
            txt_file.write('backdoor acc: %f\n'%attack_acc)
            print('\nAttack Success Rate: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), attack_acc))

        if args.attack == 'DBA' and args.mal == True:
            if DATASET == 'INFIMNIST':
                inplace_tensor = (1.0 - 0.1307) / 0.3081
            else:
                inplace_tensor = 1.0
            # inplace_tensor = torch.from_numpy(inplace_tensor)
            correct = 0
            # attack success rate
            with torch.no_grad():
                for feature, target in test_loader:
                    for k in range(feature.shape[0]):
                        feature[k] = (TF.erase(feature[k], 0, 0, 2, 4, inplace_tensor))
                        feature[k] = (TF.erase(feature[k], 0, 6, 2, 4, inplace_tensor))
                        feature[k] = (TF.erase(feature[k], 4, 0, 2, 4, inplace_tensor))
                        feature[k] = (TF.erase(feature[k], 4, 6, 2, 4, inplace_tensor))
                    # vutils.save_image(feature, '../results/DBA.png')
                    feature = feature.to(device)
                    target = (torch.ones(feature.shape[0], dtype=torch.long) * 2).to(device)
                    output = network(feature)
                    F.nll_loss(output, target, size_average=False).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            attack_acc = 100. * correct / len(test_loader.dataset)
            txt_file.write('backdoor acc: %f\n'%attack_acc)
            print('\nAttack Success Rate: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), attack_acc))
