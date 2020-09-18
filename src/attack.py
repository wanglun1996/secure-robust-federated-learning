import numpy as np
import torch
from torch import nn
import random
from scipy.spatial import distance
from robust_estimator import krum
import copy

# Training benign model at malicious agent
def benign_train(mal_train_loaders, network, criterion, optimizer, params_copy, device):
    local_grads = []
    for p in list(network.parameters()):
        local_grads.append(np.zeros(p.data.shape))
    # count = 0
    for idx, (feature, _, target, true_label) in enumerate(mal_train_loaders, 0):
        feature = feature.to(device)
        target = target.type(torch.long).to(device)
        true_label = true_label.type(torch.long).to(device)
        optimizer.zero_grad()
        output = network(feature)
        loss_val = criterion(output, true_label)
        # loss_mal = criterion(output, target)
        loss_val.backward()
        optimizer.step()
        # if count == 0:
        #     benign_loss = loss_val
        #     mal_loss = loss_mal
        # else:
        #     benign_loss += loss_val
        #     mal_loss += loss_mal
        # count += 1
    
    for idx, p in enumerate(network.parameters()):
        local_grads[idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()

    with torch.no_grad():
        for idx, p in enumerate(list(network.parameters())):
            p.copy_(params_copy[idx])

    return local_grads

# return the grad at previous round
def est_accuracy(mal_visible, t):
    delta_other_prev = None

    if len(mal_visible) >= 1:
        mal_prev_t = mal_visible[-1]
        # print('Loading from previous iteration %s' % mal_prev_t)
        delta_other_prev = np.load('../checkpoints/' + 'ben_delta_t%s.npy' % mal_prev_t, allow_pickle=True)
        delta_other_prev /= (t - mal_prev_t)
    
    return delta_other_prev

def weight_constrain(loss1, network, constrain_weights, t, device):
    params = list(network.parameters())
    loss_fn = nn.MSELoss(size_average=False, reduce=True)
    start_flag = 0

    for idx in range(len(params)):
        grad = torch.from_numpy(constrain_weights[idx]).to(device)
        if start_flag == 0:
            loss2 = loss_fn(grad, params[idx])
        else:
            loss2 += loss_fn(grad, params[idx])
        # print(loss2)
        start_flag = 1
    rho = 1e-3
    loss = loss1 + loss2 * rho
    # mal_loss = mal_loss1

    return loss

def mal_single(mal_train_loaders, train_loaders, network, criterion, optimizer, params_copy, device, mal_visible, t, dist=True, mal_boost=1):
    start_weights = params_copy.copy()
    constrain_weights = []

    for p in list(network.parameters()):
        constrain_weights.append(np.zeros(p.data.shape))

    delta_other_prev = est_accuracy(mal_visible, t)

    # Add benign estimation
    if len(mal_visible) >= 1:
        for idx in range(len(start_weights)):
            delta_other = torch.from_numpy(delta_other_prev[idx]).to(device)
            start_weights[idx].data.sub_(delta_other)
            # start_weights[idx] = p.data.cpu().numpy() - delta_other_prev[idx]
    
    # Load shared weights for malicious agent
    with torch.no_grad():
        for idx, p in enumerate(list(network.parameters())):
            p.copy_(start_weights[idx])

    # if dist:
    final_delta = benign_train(mal_train_loaders, network, criterion, optimizer, start_weights, device)
    for idx, p in enumerate(start_weights):
        constrain_weights[idx] = p.data.cpu().numpy() - final_delta[idx] / 10
    # loss, mal_loss = weight_constrain(loss1, mal_loss1, network, constrain_weights, t)

    delta_mal = []
    delta_local = []
    for p in list(network.parameters()):
        delta_mal.append(np.zeros(p.data.shape))
        delta_local.append(np.zeros(p.data.shape))
    
    for idx, (feature, target) in enumerate(train_loaders, 0):
        feature = feature.to(device)
        target = target.type(torch.long).to(device)
        optimizer.zero_grad()
        output = network(feature)
        loss_val = criterion(output, target)
        loss = weight_constrain(loss_val, network, constrain_weights, t, device)
        loss.backward()
        optimizer.step()

    for idx, p in enumerate(list(network.parameters())):
        delta_local[idx] = params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy()

    for i in range(int(len(train_loaders) / len(mal_train_loaders) + 1)):
        for idx, (feature, mal_data, _, target) in enumerate(mal_train_loaders, 0):
            mal_data = mal_data.to(device)
            target = target.type(torch.long).to(device)
            output = network(mal_data)
            loss_mal = criterion(output, target)

            optimizer.zero_grad()
            loss_mal.backward()
            optimizer.step()

    for idx, p in enumerate(list(network.parameters())):
        delta_mal[idx] = (params_copy[idx].data.cpu().numpy() - p.data.cpu().numpy() - delta_local[idx]) * mal_boost + delta_local[idx]

    # with torch.no_grad():
    #     for idx, p in enumerate(list(network.parameters())):
    #         p.copy_(params_copy[idx])

    return delta_mal

def attack_trimmedmean(network, local_grads, mal_index, b=2):
    benign_max = []
    benign_min = []
    average_sign = []
    mal_param = []
    agent_num = len(local_grads)
    local_param = copy.deepcopy(local_grads)
    for i in sorted(mal_index, reverse=True):
        del local_param[i]
    for p in list(network.parameters()):
        benign_max.append(np.zeros(p.data.shape))
        benign_min.append(np.zeros(p.data.shape))
        average_sign.append(np.zeros(p.data.shape))
        mal_param.append(np.zeros(p.data.shape))
    for idx, p in enumerate(average_sign):
        for c in range(len(local_param)):
            average_sign[idx] += local_param[c][idx]
        average_sign[idx] = np.sign(average_sign[idx])
    for idx, p in enumerate(network.parameters()):
        temp = []
        for c in range(len(local_param)):
            local_param[c][idx] = p.data.cpu().numpy() - local_param[c][idx]
            temp.append(local_param[c][idx])
        temp = np.array(temp)
        benign_max[idx] = np.amax(temp, axis=0)
        benign_min[idx] = np.amin(temp, axis=0)
        #print(temp)
        #print(benign_min[idx])
    
    for idx, p in enumerate(average_sign):
        for aver_sign, b_max, b_min, mal_p in np.nditer([p, benign_max[idx], benign_min[idx], mal_param[idx]], op_flags=['readwrite']):
            # print(b_max, b_min)
            # FIXME: may not correct
            if aver_sign > 0:
                if b_min > 0:
                    mal_p[...] = random.uniform(b_min/b, b_min)
                else:
                    mal_p[...] = random.uniform(b_min*b, b_min)
            else:
                if b_max > 0:
                    mal_p[...] = random.uniform(b_max, b_max*b)
                else:
                    mal_p[...] = random.uniform(b_max, b_max/b)
    for c in mal_index:
        for idx, p in enumerate(network.parameters()):
            #print(mal_param[idx])
            local_grads[c][idx] = -mal_param[idx] + p.data.cpu().numpy()
        # print(local_grads[c])
    #print(local_grads[2])
    return local_grads

def attack_krum(network, local_grads, mal_index, param_index, lower_bound=1e-8, upper_bound=1e-3):

    local_param = copy.deepcopy(local_grads)
    for i in sorted(mal_index, reverse=True):
        del local_param[i]
    m = len(local_grads)
    c = len(mal_index)
    d = local_grads[0][param_index].size

    average_sign = np.zeros(list(network.parameters())[param_index].data.shape)
    benign_max = np.zeros(list(network.parameters())[param_index].data.shape)


    for c in range(len(local_param)):
        average_sign += local_param[c][param_index]
    average_sign  = np.sign(average_sign)
    min_dis = np.inf
    max_dis = -np.inf
    for i in range(m):
        if i in mal_index:
            continue
        else:
            temp_min_dis = 0
            temp_max_dis = 0
            for j in range(m):
                if j in mal_index or j == i:
                    continue
                else:
                    temp_min_dis += distance.euclidean(local_grads[i][param_index].flatten(), local_grads[j][param_index].flatten())
        temp_max_dis += distance.euclidean(local_grads[i][param_index].flatten(), benign_max.flatten())

        if temp_min_dis < min_dis:
            min_dis = temp_min_dis
        if temp_max_dis > max_dis:
            max_dis = temp_max_dis
    
    # upper_bound = 1.0 / (m - 2*c - 1) / np.sqrt(d) * min_dis + 1.0 / np.sqrt(d) * max_dis
    upper_bound = 1.0
    # print(upper_bound, d)
    lambda1 = upper_bound

    if upper_bound < lower_bound:
        print('Wrong lower bound!')

    while True:
        krum_local = []
        for kk in range(len(local_grads)):
            krum_local.append(local_grads[kk][param_index])
        for kk in mal_index:
            krum_local[kk] = -lambda1 * average_sign
        _, choose_index = krum(krum_local, f=1)
        if choose_index in mal_index:
            break
        elif lambda1 < lower_bound:
            print(choose_index, 'Failed to find a proper lambda!')
            break
        else:
            lambda1 /= 2.0
    
    for kk in mal_index:
        local_grads[kk][param_index] = -lambda1 * average_sign
    # print(average_sign[param_index])

    return local_grads

# def model_replacement_attack(original_model, target_model, learning_rate, per_round):
#     return per_round / learning_rate * (target_model - original_model)
