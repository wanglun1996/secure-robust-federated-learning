import numpy as np
import torch
from torch import nn

# Training benign model at malicious agent
def benign_train(mal_train_loaders, network, criterion, optimizer, params_copy, device):
    local_grads = []
    for p in list(network.parameters()):
        local_grads.append(np.zeros(p.data.shape))
    # count = 0
    for idx, (feature, target, true_label) in enumerate(mal_train_loaders, 0):
        feature = feature.to(device)
        target = target.type(torch.long).to(device)
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
        print('Loading from previous iteration %s' % mal_prev_t)
        delta_other_prev = np.load('./checkpoints/' + 'ben_delta_t%s' % mal_prev_t, allow_pickle=True)
        delta_other_prev /= (t - mal_prev_t)
    
    return delta_other_prev

def weight_constrain(loss1, mal_loss1, network, constrain_weights, t):
    params = list(network.parameters())
    loss_fn = nn.MSELoss(size_average=False, reduce=True)
    start_flag = 0

    for idx in range(len(params)):
        grad = torch.from_numpy(constrain_weights[idx]).to(device)
        if start_flag == 0:
            loss2 = loss_fn(grad, params[idx])
        else:
            loss2 += loss_fn(grad, params)
        start_flag = 1
    rho = 1e-4
    loss = loss1 + loss2 * rho
    mal_loss = mal_loss1

    return loss, mal_loss

def mal_single(mal_train_loaders, network, criterion, optimizer, params_copy, device, mal_visible, t, dist=True):
    start_weights = params_copy.cpoy()
    constrain_weights = params_copy.copy()

    delta_other_prev = est_accuracy(mal_visible, t)

    # Add benign estimation
    if len(mal_visible) >= 1:
        for idx, p in enumerate(params_copy):
            start_weights[idx] = p - delta_other_prev[idx]
    
    # Load shared weights for malicious agent
    with torch.no_grad():
        for idx, p in enumerate(list(network.parameters())):
            p.copy_(start_weights[idx])

    # if dist:
    final_delta = benign_train(mal_train_loaders, network, criterion, optimizer, start_weights, device)
    for idx, p in enumerate(start_weights):
        constrain_weights[idx] = p - final_delta[idx]
    # loss, mal_loss = weight_constrain(loss1, mal_loss1, network, constrain_weights, t)

    delta_mal = []
    for p in list(network.parameters()):
        delta_mal.append(np.zeros(p.data.shape))
    
    for idx, (feature, mal_data, true_label, target) in enumerate(mal_train_loaders, 0):
        mal_data = mal_data.to(device)
        target = target.type(torch.long).to(device)
        true_label = true_label.type(torch.long).to(device)
        optimizer.zero_grad()
        output = network(mal_data)
        loss_val = criterion(output, true_label)
        loss_mal = criterion(output, target)

        loss, mal_loss = weight_constrain(loss_val, loss_mal, network, constrain_weights, t)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        mal_loss.backward()
        optimizer.step()

    for idx, p in enumerate(list(network.parameters())):
        delta_mal[idx] = params_copy[idx] - p

    # with torch.no_grad():
    #     for idx, p in enumerate(list(network.parameters())):
    #         p.copy_(params_copy[idx])

    return delta_mal
