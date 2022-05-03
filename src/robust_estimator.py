'''
    Robust estimator for Federated Learning.
'''

import argparse
import numpy as np
from numpy.random import multivariate_normal
from sklearn.preprocessing import normalize
from scipy.linalg import eigh
from scipy.special import rel_entr
import cvxpy as cvx

MAX_ITER = 100
ITV = 1000

def ex_noregret_(samples, eps=1./12, sigma=1, expansion=20, dis_threshold=0.7):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    size = len(samples)
    threshold = 2 * eps * size
    filtered_samples = []
    while len(filtered_samples) == 0:
        for i, sample_i in enumerate(samples):
            far_num = 0
            for j, sample_j in enumerate(samples):
                if i != j:
                    dis_square = sum((sample_i - sample_j)**2)
                    if dis_square > dis_threshold**2:
                        far_num += 1
            if far_num < threshold:
                filtered_samples.append(sample_i)
        dis_threshold *= 2
   
    samples = np.array(filtered_samples)
    size = samples.shape[0]
    feature_size = samples.shape[1]
    samples_ = samples.reshape(size, 1, feature_size)

    c = np.ones(size)
    while True:
        for _ in range(MAX_ITER):
            avg = np.average(samples, axis=0, weights=c)
            cov = np.average(np.array([np.matmul((sample - avg).T, (sample - avg)) for sample in samples_]), axis=0, weights=c)
            eig_val, eig_vec = eigh(cov, eigvals=(feature_size-1, feature_size-1), eigvals_only=False)
            eig_val = eig_val[0]
            eig_vec = eig_vec.T[0]

            if eig_val * eig_val <= expansion * sigma * sigma:
                return avg

            tau = np.array([np.inner(sample-avg, eig_vec)**2 for sample in samples])
            tau_max = np.amax(tau)
            c = c * (1 - tau/tau_max)

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

        expansion *= 2
        
    raise ValueError(f"Cannot suppress the max eigenvalue into given sigma2 value within {MAX_ITER} iterations.") 

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
        # print(cnt, len(sizes))
        cnt += 1
        res.append(ex_noregret_(samples_flatten[:,idx:idx+size], eps, sigma, expansion))
        idx += size

    return np.concatenate(res, axis=0).reshape(feature_shape)

def mom_ex_noregret(samples, eps=0.2, sigma=1, expansion=20, itv=ITV):
    bucket_size = int(np.floor(2 * eps * len(samples)))
    bucket_num = int(np.ceil(.5 / eps))

    bucketed_samples = []
    for i in range(len(samples)):
        bucketed_samples.append(np.sum(samples[i*bucket_size:min((i+1)*bucket_size, len(samples))], axis=0))
    # print(len(bucketed_samples), len(samples), bucketed_samples[0].shape, samples[0].shape)
    return ex_noregret(samples, eps, sigma, expansion, itv)

def filterL2_(samples, sigma=1, expansion=20):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    size = samples.shape[0]
    feature_size = samples.shape[1]


    samples_ = samples.reshape(size, 1, feature_size)

    c = np.ones(size)
    while True:
        for _ in range(MAX_ITER):
            avg = np.average(samples, axis=0, weights=c)
            cov = np.average(np.array([np.matmul((sample - avg).T, (sample - avg)) for sample in samples_]), axis=0, weights=c)
            eig_val, eig_vec = eigh(cov, eigvals=(feature_size-1, feature_size-1), eigvals_only=False)
            eig_val = eig_val[0]
            eig_vec = eig_vec.T[0]

            if eig_val * eig_val <= expansion * sigma * sigma:
                return avg
        
            tau = np.array([np.inner(sample-avg, eig_vec)**2 for sample in samples])
            tau_max = np.amax(tau)
            c = c * (1 - tau/tau_max)

        expansion *= 2
 
def filterL2(samples, sigma=1, expansion=20, itv=ITV):
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
    for size in sizes:
        res.append(filterL2_(samples_flatten[:,idx:idx+size], sigma, expansion))
        idx += size

    return np.concatenate(res, axis=0).reshape(feature_shape)

def mom_filterL2(samples, eps=0.2, sigma=1, expansion=20, itv=ITV):
    bucket_size = int(np.floor(2 * eps * len(samples)))
    bucket_num = int(np.ceil(.5 / eps))

    bucketed_samples = []
    for i in range(len(samples)):
        bucketed_samples.append(np.sum(samples[i*bucket_size:min((i+1)*bucket_size, len(samples))], axis=0))
    # print(len(bucketed_samples), len(samples), bucketed_samples[0].shape, samples[0].shape)
    return filterL2(samples, sigma, expansion, itv)

def trimmed_mean(samples, beta=0.1):
    samples = np.array(samples)
    average_grad = np.zeros(samples[0].shape)
    size = samples.shape[0]
    beyond_choose = int(size * beta)
    samples = np.sort(samples, axis=0)
    samples = samples[beyond_choose:size-beyond_choose]
    average_grad = np.average(samples, axis=0)

    return average_grad

def krum(samples, f=0):
    size = len(samples)
    size_ = size - f - 2
    metric = []
    for idx in range(size):
        sample = samples[idx]
        samples_ = samples.copy()
        del samples_[idx]
        dis = np.array([np.linalg.norm(sample-sample_) for sample_ in samples_])
        metric.append(np.sum(dis[np.argsort(dis)[:size_]]))
    index = np.argmin(metric)
    return samples[index], index

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

def bulyan(grads, aggsubfunc='trimmedmean', f=1):
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
            krum_grad, _ = krum(samples_flatten)
            selected_grads.append(krum_grad)
            for j in range(len(samples_flatten)):
                if samples_flatten[j] is krum_grad:
                    del samples_flatten[j]
                    break
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

if __name__ == '__main__':
    samples = np.concatenate([np.ones((8, 1000)), np.ones((2, 1000)) * 10])
    print(samples.shape, samples)
    # contaminated_samples = [10]
    from time import time
    start = time()
    res = mom_filterL2(np.array(samples))
    end = time()
    print(end-start)
    print(res)
