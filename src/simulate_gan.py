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
    The simulation program for Federated Learning.
'''

import argparse
from attack import attack_krum, attack_trimmedmean, backdoor, mal_single
from data import MalDataset
from networks import ConvNet
import numpy as np
import random
from robust_estimator import krum, filterL2, median, trimmed_mean, bulyan, ex_noregret, mom_filterL2, mom_ex_noregret
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

random.seed(2022)
np.random.seed(5)
torch.manual_seed(11)

MAL_FEATURE_FILE = './data/%s_mal_feature_10.npy'
MAL_TARGET_FILE = './data/%s_mal_target_10.npy'
MAL_TRUE_LABEL_FILE = './data/%s_mal_true_label_10.npy'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0')
    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--nworker', type=int, default=1000)
    parser.add_argument('--perround', type=int, default=1000)
    parser.add_argument('--localiter', type=int, default=5)
    parser.add_argument('--round', type=int, default=10)
    parser.add_argument('--current_round', type=int, default=0) 
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--checkpoint', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=1e-5)
    parser.add_argument('--batchsize', type=int, default=10)

    # Malicious agent setting
    parser.add_argument('--malnum', type=int, default=200)
    parser.add_argument('--agg', default='gan', help='average, ex_noregret, filterl2, gan, krum, median, trimmedmean, bulyankrum, bulyantrimmedmean, mom_filterl2, mom_ex_noregret')
    parser.add_argument('--attack', default='noattack', help="noattack, trimmedmean, krum, backdoor, modelpoisoning")
    args = parser.parse_args()

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu") 
    mal_index = list(range(args.malnum))

    if args.dataset == 'MNIST':

        transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))])

        train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=args.batchsize)
        test_loader = DataLoader(torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform))
        mal_train_loaders = DataLoader(MalDataset(MAL_FEATURE_FILE%('mnist'), MAL_TRUE_LABEL_FILE%('mnist'), MAL_TARGET_FILE%('mnist'), transform=transform), batch_size=args.batchsize)
        network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
        backdoor_network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)

    elif args.dataset == 'Fashion-MNIST':

        train_set = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = torchvision.transforms.ToTensor())
        train_loader = DataLoader(train_set, batch_size=args.batchsize)
        test_loader = DataLoader(torchvision.datasets.FashionMNIST(root = "./data", train = False, download = True, transform = torchvision.transforms.ToTensor()))
        mal_train_loaders = DataLoader(MalDataset(MAL_FEATURE_FILE%('fashion'), MAL_TRUE_LABEL_FILE%('fashion'), MAL_TARGET_FILE%('fashion'), transform=torchvision.transforms.ToTensor()), batch_size=args.batchsize)
        network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)
        backdoor_network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).to(device)

    # Split into multiple training set
    local_size = len(train_set) // args.nworker
    sizes = []
    sum = 0
    for i in range(0, args.nworker):
        sizes.append(local_size)
        sum = sum + local_size
    sizes[0] = sizes[0] + len(train_set)  - sum
    train_sets = random_split(train_set, sizes)
    train_loaders = []
    for trainset in train_sets:
        train_loaders.append(DataLoader(trainset, batch_size=args.batchsize, shuffle=True))

    # define training loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=args.lr)
    # prepare data structures to store local gradients
    local_grads = []
    for i in range(args.nworker):
        local_grads.append([])
        for p in list(network.parameters()):
            local_grads[i].append(np.zeros(p.data.shape))

    print('Malicious node indices:', mal_index, 'Attack Type:', args.attack)

    file_name = './results/' + args.attack + '_' + args.agg + '_' + args.dataset + '.txt'
    txt_file = open(file_name, 'w') 

    # for round_idx in range(args.round):
    round_idx = args.current_round
    print("Round: ", round_idx)
    if round_idx != 0:
        with torch.no_grad():
            temp_file_name = './results/ganAgg_' + str(round_idx) + '.npy'
            with open(temp_file_name, 'rb') as f:
                load_GAN_params = np.load(f, allow_pickle=True)
            for idx, p in enumerate(list(network.parameters())):
                p.copy_(torch.from_numpy(load_GAN_params[idx]).cuda())
    choices = np.random.choice(args.nworker, args.perround, replace=False)
    if args.attack == 'modelpoisoning':
        dynamic_mal_index = np.random.choice(choices, args.malnum, replace=False)

    # copy network parameters
    params_copy = []
    for p in list(network.parameters()):
        params_copy.append(p.clone())

    for c in tqdm(choices):

        if args.attack == 'modelpoisoning' and c in dynamic_mal_index:
            for idx, p in enumerate(local_grads[c]):
                local_grads[c][idx] = np.zeros(p.shape)

            for _ in range(0, args.localiter):
                params_temp = []
                for p in list(network.parameters()):
                    params_temp.append(p.clone())
                delta_mal = mal_single(mal_train_loaders, train_loaders[c], network, criterion, optimizer, params_temp, device, [], round_idx, dist=True, mal_boost=2.0, path=args.agg)
                for idx, p in enumerate(local_grads[c]):
                    local_grads[c][idx] = p + delta_mal[idx]

        elif c in mal_index and args.attack == 'backdoor':

            for idx, p in enumerate(local_grads[c]):
                local_grads[c][idx] = np.zeros(p.shape)

            for _ in range(0, args.localiter):
                for idx, (feature, target) in enumerate(train_loaders[c], 0):
                    attack_feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device))
                    attack_target = torch.zeros(args.batchsize, dtype=torch.long).to(device)
                    optimizer.zero_grad()
                    output = network(attack_feature)
                    loss = criterion(output, attack_target)
                    loss.backward()
                    optimizer.step()

            for idx, p in enumerate(network.parameters()):
                local_grads[c][idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()

        else:

            for _ in range(0, args.localiter):
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

    if args.attack == 'modelpoisoning':
        average_grad = []

        for p in list(network.parameters()):
            average_grad.append(np.zeros(p.data.shape))

        for c in choices:
            if c not in mal_index:
                for idx, p in enumerate(average_grad):
                    average_grad[idx] = p + local_grads[c][idx] / args.perround

        np.save('../checkpoints/' + args.agg + 'ben_delta_t%s.npy' % round_idx, average_grad)

    elif args.attack == 'trimmedmean':
        print('attack trimmedmean')
        local_grads = attack_trimmedmean(network, local_grads, mal_index, b=1.5)

    elif args.attack == 'krum':
        print('attack krum')
        for idx, _ in enumerate(local_grads[0]):
            local_grads = attack_krum(network, local_grads, mal_index, idx)

    # aggregation
    average_grad = []
    for p in list(network.parameters()):
        average_grad.append(np.zeros(p.data.shape))
    if args.agg == 'average':
        print('agg: average')
        for idx, p in enumerate(average_grad):
            avg_local = []
            for c in choices:
                avg_local.append(local_grads[c][idx])
            avg_local = np.array(avg_local)
            average_grad[idx] = np.average(avg_local, axis=0)
    elif args.agg == 'krum':
        print('agg: krum')
        for idx, p in enumerate(average_grad):
            krum_local = []
            for c in choices:
                krum_local.append(local_grads[c][idx])
            average_grad[idx], _ = krum(krum_local, f=args.malnum)
    elif args.agg == 'filterl2':
        print('agg: filterl2')
        for idx, _ in enumerate(average_grad):
            filterl2_local = []
            for c in choices:
                filterl2_local.append(local_grads[c][idx])
            average_grad[idx] = filterL2(filterl2_local, eps=args.malnum*1./args.nworker, sigma=args.sigma)
    elif args.agg == 'mom_filterl2':
        print('agg: filterl2')
        for idx, _ in enumerate(average_grad):
            filterl2_local = []
            for c in choices:
                filterl2_local.append(local_grads[c][idx])
            average_grad[idx] = mom_filterL2(filterl2_local, eps=args.malnum*1./args.nworker, sigma=args.sigma)
    elif args.agg == 'median':
        print('agg: median')
        for idx, _ in enumerate(average_grad):
            median_local = []
            for c in choices:
                median_local.append(local_grads[c][idx])
            average_grad[idx] = median(median_local)
    elif args.agg == 'trimmedmean':
        print('agg: trimmedmean')
        for idx, _ in enumerate(average_grad):
            trimmedmean_local = []
            for c in choices:
                trimmedmean_local.append(local_grads[c][idx])
            average_grad[idx] = trimmed_mean(trimmedmean_local)
    elif args.agg == 'bulyankrum':
        print('agg: bulyankrum')
        for idx, _ in enumerate(average_grad):
            bulyan_local = []
            for c in choices:
                bulyan_local.append(local_grads[c][idx])
            average_grad[idx] = bulyan(bulyan_local, args.malnum, aggsubfunc='krum')
    elif args.agg == 'bulyanmedian':
        print('agg: bulyanmedian')
        for idx, _ in enumerate(average_grad):
            bulyan_local = []
            for c in choices:
                bulyan_local.append(local_grads[c][idx])
            average_grad[idx] = bulyan(bulyan_local, args.malnum, aggsubfunc='median')
    elif args.agg == 'bulyantrimmedmean':
        print('agg: bulyantrimmedmean')
        for idx, _ in enumerate(average_grad):
            bulyan_local = []
            for c in choices:
                bulyan_local.append(local_grads[c][idx])
            average_grad[idx] = bulyan(bulyan_local, args.malnum, aggsubfunc='trimmedmean')
    elif args.agg == 'ex_noregret':
        print('agg: explicit non-regret')
        for idx, _ in enumerate(average_grad):
            ex_noregret_local = []
            for c in choices:
                ex_noregret_local.append(local_grads[c][idx])
            average_grad[idx] = ex_noregret(ex_noregret_local, eps=args.malnum*1./args.nworker, sigma=args.sigma)
    elif args.agg == 'mom_ex_noregret':
        print('agg: explicit non-regret')
        for idx, _ in enumerate(average_grad):
            ex_noregret_local = []
            for c in choices:
                ex_noregret_local.append(local_grads[c][idx])
            average_grad[idx] = mom_ex_noregret(ex_noregret_local, eps=args.malnum*1./args.nworker, sigma=args.sigma)
    elif args.agg == 'gan':
        print('agg: GAN')
        for idx, p in enumerate(average_grad):
            gan_local = []
            for c in choices:
                gan_local.append(local_grads[c][idx])
            temp_file_name = './results/gan_' + str(round_idx) +'_' + str(idx) + '.npy'
            with open(temp_file_name, 'wb') as f:
                np.save(f, gan_local)
        for idx, p in enumerate(average_grad):
            avg_local = []
            for c in choices:
                avg_local.append(local_grads[c][idx])
            avg_local = np.array(avg_local)
            average_grad[idx] = np.average(avg_local, axis=0)

    params = list(network.parameters())
    with torch.no_grad():
        for idx in range(len(params)):
            temp_file_name = './results/gan_Global_' + str(round_idx) + '_' + str(idx) + '.npy'
            with open(temp_file_name, 'wb') as f:
                np.save(f, params[idx].data.cpu().numpy())
            grad = torch.from_numpy(average_grad[idx]).to(device)
            params[idx].data.sub_(grad)


    if (round_idx + 1) % args.checkpoint == 0:

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
        accuracy = 100. * correct / len(test_loader.dataset)

        if args.attack == 'modelpoisoning':

            mal_test_loss = 0
            correct = 0
            with torch.no_grad():
                for idx, (feature, mal_data, true_label, target) in enumerate(mal_train_loaders, 0):
                    feature = feature.to(device)
                    target = target.type(torch.long).to(device)
                    output = network(feature)
                    mal_test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            mal_test_loss /= len(mal_train_loaders.dataset)
            txt_file.write('%d, \t%f, \t%f, \t%f, \t%f\n'%(round_idx+1, test_loss, accuracy, mal_test_loss, 100. * correct / len(mal_train_loaders.dataset)))
            print('\nMalicious set: Accuracy: {:.4f}, Attack Success Rate: {}/{} ({:.0f}%)\n'.format(accuracy, correct, len(mal_train_loaders.dataset), 100. * correct / len(mal_train_loaders.dataset)))
        

        elif args.attack == 'backdoor':

            mal_test_loss = 0
            correct = 0
            with torch.no_grad():
                for feature, target in test_loader:
                    feature = (TF.erase(feature, 0, 0, 5, 5, 0).to(device))
                    target = torch.zeros(1, dtype=torch.long).to(device)
                    output = network(feature)
                    mal_test_loss += F.nll_loss(output, target, size_average=False).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            txt_file.write('%d, \t%f, \t%f, \t%f, \t%f\n'%(round_idx+1, test_loss, accuracy, mal_test_loss, 100. * correct / len(test_loader.dataset)))
            print('\nMalicious set: Accuracy: {:.4f}, Attack Success Rate: {}/{} ({:.0f}%)\n'.format(accuracy, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


        else:

            txt_file.write('%d, \t%f, \t%f\n'%(round_idx+1, test_loss, accuracy))
            print('\nTest set, Average aggregator: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
