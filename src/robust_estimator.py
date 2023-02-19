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
    Robust estimator for Federated Learning.
'''

import argparse
import cvxpy as cvx
import numpy as np
from numpy.random import multivariate_normal
from numpy import linalg as la
from scipy.linalg import eigh
from scipy.special import rel_entr
from sklearn.preprocessing import normalize

MAX_ITER = 100
ITV = 1000

def ex_noregret_(samples, eps=1./12, sigma=1, expansion=20, dis_threshold=0.7):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    size = len(samples)
    f = int(np.ceil(eps*size))
    metric = krum_(list(samples), f)
    indices = np.argpartition(metric, -f)[:-f]
    samples = samples[indices]
    size = samples.shape[0]
    
    dis_list = []
    for i in range(size):
        for j in range(i+1, size):
            dis_list.append(np.linalg.norm(samples[i]-samples[j]))
    step_size = 0.5 / (np.amax(dis_list) ** 2)
    size = samples.shape[0]
    feature_size = samples.shape[1]
    samples_ = samples.reshape(size, 1, feature_size)

    c = np.ones(size)
    for i in range(int(2 * eps * size)):
        avg = np.average(samples, axis=0, weights=c)
        cov = np.average(np.array([np.matmul((sample - avg).T, (sample - avg)) for sample in samples_]), axis=0, weights=c)
        eig_val, eig_vec = eigh(cov, eigvals=(feature_size-1, feature_size-1), eigvals_only=False)
        eig_val = eig_val[0]
        eig_vec = eig_vec.T[0]

        if eig_val * eig_val <= expansion * sigma * sigma:
            return avg

        tau = np.array([np.inner(sample-avg, eig_vec)**2 for sample in samples])
        c = c * (1 - step_size * tau)

        # The projection step
        ordered_c_index = np.flip(np.argsort(c))
        min_KL = None
        projected_c = None
        for i in range(len(c)):
            c_ = np.copy(c)
            for j in range(i+1):   
                c_[ordered_c_index[j]] = 1./(1-eps)/len(c)
            clip_norm = 1 - np.sum(c_[ordered_c_index[:i+1]])
            norm = np.sum(c_[ordered_c_index[i+1:]])
            if clip_norm <= 0:
                break
            scale = clip_norm / norm
            for j in range(i+1, len(c)):
                c_[ordered_c_index[j]] = c_[ordered_c_index[j]] * scale
            if c_[ordered_c_index[i+1]] > 1./(1-eps)/len(c):
                continue
            KL = np.sum(rel_entr(c, c_))
            if min_KL is None or KL < min_KL:
                min_KL = KL
                projected_c = c_

        c = projected_c
        
    avg = np.average(samples, axis=0, weights=c)
    return avg

def ex_noregret(samples, eps=1./12, sigma=1, expansion=20, itv=ITV):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    samples = np.array(samples)
    feature_shape = samples[0].shape
    samples_flatten = []
    for i in range(samples.shape[0]):
        samples_flatten.append(samples[i].flatten())
    samples_flatten = np.array(samples_flatten)
    feature_size = samples_flatten.shape[1]
    if itv is None:
        itv = int(np.floor(np.sqrt(feature_size)))
    cnt = int(feature_size // itv)
    sizes = []
    for i in range(cnt):
        sizes.append(itv)
    if feature_size % itv:
        sizes.append(feature_size - cnt * itv)

    idx = 0
    res = []
    cnt = 0
    for size in sizes:
        cnt += 1
        res.append(ex_noregret_(samples_flatten[:,idx:idx+size], eps, sigma, expansion))
        idx += size

    return np.concatenate(res, axis=0).reshape(feature_shape)

def mom_ex_noregret(samples, eps=0.2, sigma=1, expansion=20, itv=ITV, delta=np.exp(-30)):
    bucket_num = int(np.floor(eps * len(samples)) + np.log(1. / delta))
    bucket_size = int(np.ceil(len(samples) * 1. / bucket_num))

    bucketed_samples = []
    for i in range(bucket_num):
        bucketed_samples.append(np.mean(samples[i*bucket_size:min((i+1)*bucket_size, len(samples))], axis=0))
    return ex_noregret(bucketed_samples, eps, sigma, expansion, itv)

def filterL2_(samples, eps=0.2, sigma=1, expansion=20):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    size = samples.shape[0]
    feature_size = samples.shape[1]

    samples_ = samples.reshape(size, 1, feature_size)

    c = np.ones(size)
    for i in range(2 * int(eps * size)):
        # print(i)
        avg = np.average(samples, axis=0, weights=c)
        cov = np.average(np.array([np.matmul((sample - avg).T, (sample - avg)) for sample in samples_]), axis=0, weights=c)
        eig_val, eig_vec = eigh(cov, eigvals=(feature_size-1, feature_size-1), eigvals_only=False)
        eig_val = eig_val[0]
        eig_vec = eig_vec.T[0]

        if eig_val * eig_val <= expansion * sigma * sigma:
            return avg
        
        tau = np.array([np.inner(sample-avg, eig_vec)**2 for sample in samples])
        tau_max_idx = np.argmax(tau)
        tau_max = tau[tau_max_idx]
        c = c * (1 - tau/tau_max)

        samples = np.concatenate((samples[:tau_max_idx], samples[tau_max_idx+1:]))
        samples_ = samples.reshape(-1, 1, feature_size)
        c = np.concatenate((c[:tau_max_idx], c[tau_max_idx+1:]))
        c = c / np.linalg.norm(c, ord=1)

    avg = np.average(samples, axis=0, weights=c)
    return avg

 
def filterL2(samples, eps=0.2, sigma=1, expansion=20, itv=ITV):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    samples = np.array(samples)
    feature_shape = samples[0].shape
    samples_flatten = []
    for i in range(samples.shape[0]):
        samples_flatten.append(samples[i].flatten())
    samples_flatten = np.array(samples_flatten)
    # print(samples_flatten.shape)
    feature_size = samples_flatten.shape[1]
    if itv is None:
        itv = int(np.floor(np.sqrt(feature_size)))
    cnt = int(feature_size // itv)
    sizes = []
    for i in range(cnt):
        sizes.append(itv)
    if feature_size % itv:
        sizes.append(feature_size - cnt * itv)

    idx = 0
    res = []
    for size in sizes:
        res.append(filterL2_(samples_flatten[:,idx:idx+size], eps, sigma, expansion))
        idx += size

    return np.concatenate(res, axis=0).reshape(feature_shape)

def mom_filterL2(samples, eps=0.2, sigma=1, expansion=20, itv=ITV, delta=np.exp(-30)):
    bucket_num = int(np.floor(eps * len(samples)) + np.log(1. / delta))
    bucket_size = int(np.ceil(len(samples) * 1. / bucket_num))

    bucketed_samples = []
    for i in range(bucket_num):
        bucketed_samples.append(np.mean(samples[i*bucket_size:min((i+1)*bucket_size, len(samples))], axis=0))

    return filterL2(bucketed_samples, eps, sigma, expansion, itv)

def median(samples):
    return np.median(samples, axis=0)

def trimmed_mean(samples, beta=0.1):
    samples = np.array(samples)
    average_grad = np.zeros(samples[0].shape)
    size = samples.shape[0]
    beyond_choose = int(size * beta)
    samples = np.sort(samples, axis=0)
    samples = samples[beyond_choose:size-beyond_choose]
    average_grad = np.average(samples, axis=0)

    return average_grad

def krum_(samples, f):
    size = len(samples)
    size_ = size - f - 2
    metric = []
    for idx in range(size):
        sample = samples[idx]
        samples_ = samples.copy()
        del samples_[idx]
        dis = np.array([np.linalg.norm(sample-sample_) for sample_ in samples_])
        metric.append(np.sum(dis[np.argsort(dis)[:size_]]))
    return metric

def krum(samples, f):
    metric = krum_(samples, f)
    index = np.argmin(metric)
    return samples[index], index

def mom_krum(samples, f, bucket_size=3):
    bucket_num = int(np.ceil(len(samples) * 1. / bucket_size))

    bucketed_samples = []
    for i in range(bucket_num):
        bucketed_samples.append(np.mean(samples[i*bucket_size:min((i+1)*bucket_size, len(samples))], axis=0))
    return krum(bucketed_samples, f=f)[0]

def bulyan_median(arr):
    arr_len = len(arr)
    distances = np.zeros([arr_len, arr_len])
    for i in range(arr_len):
        for j in range(arr_len):
            if i < j:
                distances[i, j] = abs(arr[i] - arr[j])
            elif i > j:
                distances[i, j] = distances[j, i]
    total_dis = np.sum(distances, axis=-1)
    median_index = np.argmin(total_dis)
    return median_index, distances[median_index]

def bulyan_one_coordinate(arr, beta):
    _, distances = bulyan_median(arr)
    median_beta_neighbors = arr[np.argsort(distances)[:beta]]
    return np.mean(median_beta_neighbors)

def bulyan(grads, f, aggsubfunc='trimmedmean'):
    samples = np.array(grads)
    feature_shape = grads[0].shape
    samples_flatten = []
    for i in range(samples.shape[0]):
        samples_flatten.append(samples[i].flatten())

    grads_num = len(samples_flatten)
    theta = grads_num - 2 * f
    # bulyan cannot do the work here when theta <= 0. Actually, it assumes n >= 4 * f + 3
    selected_grads = []
    # here, we use krum as sub algorithm
    if aggsubfunc == 'krum':
        for i in range(theta):
            krum_grads, _ = krum(samples_flatten, f)
            selected_grads.append(krum_grads)
            for j in range(len(samples_flatten)):
                if samples_flatten[j] is krum_grads:
                    del samples_flatten[j]
                    break
    elif aggsubfunc == 'median':
        for i in range(theta):
            median_grads = median(samples_flatten)
            selected_grads.append(median_grads)
            min_dis = np.inf
            min_index = None
            for j in range(len(samples_flatten)):
                temp_dis = np.linalg.norm(median_grads - samples_flatten[j])
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    min_index = j
            assert min_index != None
            del samples_flatten[min_index]
    elif aggsubfunc == 'trimmedmean':
        for i in range(theta):
            trimmedmean_grads = trimmed_mean(samples_flatten)
            selected_grads.append(trimmedmean_grads)
            min_dis = np.inf
            min_index = None
            for j in range(len(samples_flatten)):
                temp_dis = np.linalg.norm(trimmedmean_grads - samples_flatten[j])
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    min_index = j
            assert min_index != None
            del samples_flatten[min_index]

    beta = theta - 2 * f
    np_grads = np.array([g.flatten().tolist() for g in selected_grads])

    grads_dim = len(np_grads[0])
    selected_grads_by_cod = np.zeros([grads_dim, 1])  # shape of torch grads
    for i in range(grads_dim):
        selected_grads_by_cod[i, 0] = bulyan_one_coordinate(np_grads[:, i], beta)

    return selected_grads_by_cod.reshape(feature_shape)

def sever_filter(samples, mask, sigma=1, c=2):
    """SVD, outlier scores and filter.
    Arguments:
        samples: gradients of all the clients from a converged model.
        mask: mask learned from previous rounds.
        sigma: assumed standard deviation of the samples.
        c: constant as in ALgorithm 2 of "Sever: A Robust Meta-Algorithm for Stochastic Optimization".
    Output:
        wether the stopping criteria is met, a subset of clients that will be used for further optimization.
    """
    samples = np.array(samples)
    feature_shape = samples[0].shape
    samples_flatten = []
    for i in range(samples.shape[0]):
        samples_flatten.append(samples[i].flatten())
    samples_flatten = np.array(samples_flatten)
    samples_mean = np.mean([samples_flatten[i] * mask[i] for i in range(len(mask))], axis=0)
    centered_samples = samples_flatten - samples_mean
    _, _, V = la.svd(centered_samples, compute_uv=True)
    v = V[0]
    outlier_scores = [np.dot(sample, v)**2 for sample in centered_samples]
    k = int(0.1 * np.sum(mask))
    T = np.array(outlier_scores) * mask
    T.sort()
    T = T[-k]
    return (np.array(outlier_scores) < T).astype(np.float32) * mask
