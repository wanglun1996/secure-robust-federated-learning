import argparse
import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg import eigh

def simulate(size=100, feature_size=10, mean=None, cov=None, malicious=False):
    """simulate the data drawn from multivariate Gaussian distribution
       size: the number of data samples
       feature_size: the dimension of the samples
       mean, cov: distribution parameters
       malicious: contaminated or not
    """
    if mean is None:
        mean = np.zeros(feature_size)
    if cov is None:
        cov = np.identity(feature_size)
    return multivariate_normal(mean, cov, size=size)

def filterL2_(samples, sigma=1):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    size = samples.shape[0]
    feature_size = samples.shape[1]
    samples_ = samples.reshape(size, 1, feature_size)

    c = np.ones(size)

    while True:
        avg = np.average(samples, axis=0, weights=c)
        cov = np.average(np.array([np.matmul((sample - avg).T, (sample - avg)) for sample in samples_]), axis=0, weights=c)
        eig_val, eig_vec = eigh(cov, eigvals=(feature_size-1, feature_size-1), eigvals_only=False)
        eig_val = eig_val[0]
        eig_vec = eig_vec.T[0]

        #FIXME: add argument for 20 here
        if eig_val * eig_val <= 20 * sigma * sigma:
            return avg

        tau = np.array([np.inner(sample-avg, eig_vec) for sample in samples])
        tau_max = np.amax(tau)
        c = c * (1 - tau/tau_max)
 
def filterL2(samples, sigma=1, itv=None):
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
    # feature_size = samples.shape[1]
    if itv is None:
        itv = int(np.floor(np.sqrt(feature_size)))
    cnt = int(feature_size // itv)
    sizes = []
    for i in range(cnt):
        sizes.append(itv)
    sizes[0] = int(feature_size - (cnt - 1) * itv)

    idx = 0
    res = []
    for size in sizes:
        # print(size)
        res.append(filterL2_(samples_flatten[:,idx:idx+size], sigma))
        idx += size

    return np.concatenate(res, axis=0).reshape(feature_shape)
  
#         print(size)
#         res.append(filterL2_(samples[:,idx:idx+size], sigma))
#         idx += size

#     return np.concatenate(res, axis=0)

def geometric_median(samples):
    samples = np.array(samples)
    size = samples.shape[0]
    metric = []

    for idx in range(samples.shape[0]):
        sample = samples[idx]
        samples_ = np.delete(samples, idx, axis=0)
        metric.append(np.sum([np.linalg.norm(sample-sample_) for sample_ in samples_]))

    return samples[np.argmin(metric)] 

def trimmed_mean(samples, beta=0.1):
    samples = np.array(samples)
    # print(samples.shape)
    average_grad = np.zeros(samples[0].shape)
    size = samples.shape[0]
    beyond_choose = int(size * beta)
    # index = np.argsort(samples, axis=0)[beyond_choose:size-beyond_choose]
    # print(index.shape)
    samples = np.sort(samples, axis=0)
    # print(samples.shape)
    samples = samples[beyond_choose:size-beyond_choose]
    average_grad = np.average(samples, axis=0)
    # for i in range(index.shape[0]):
        # average_grad += samples[index[i]] 
    # print(average_grad.shape)
    # average_grad /= float(size - beyond_choose)
    return average_grad

def krum(samples, f=0):
    # samples = np.array(samples)
    size = len(samples)
    # assert this is positive
    # print(size)
    size_ = size - f - 2
    metric = []
    # print(samples[0].shape, samples[1].shape, samples[2].shape)
    for idx in range(size):
        sample = samples[idx]
        # samples_ = np.delete(samples, idx, axis=0)
        samples_ = samples.copy()
        del samples_[idx]
        # print(sample.shape, samples_[0].shape)
        dis = np.array([np.linalg.norm(sample-sample_) for sample_ in samples_])
        metric.append(np.sum(dis[np.argsort(dis)[:size_]]))
    index = np.argmin(metric)
    return samples[index], index

def bulyan(samples, agg=krum, args=None, theta=2):
    samples = np.array(samples)
    feature_size = samples.shape[1]
    # beta = theta - 2*f
    #FIXME: the above is correct
    beta = 2
    S = []

    for _ in range(theta):
       picked_sample = agg(samples)
       S.append(picked_sample)
       samples = np.delete(samples, np.argwhere([picked_sample])[:1], axis=0)

    S = np.array(S)
    res = np.zeros(feature_size)
    # coordinate-wise median
    for idx in range(feature_size):
        samples_ = S[:, idx]
        med = np.median(samples_)
        idxs = np.argsort([np.abs(sample_-med) for sample_ in samples_])[:beta]
        res[idx] = np.average(S[idxs])

    print(res)
    return res
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--feature', type=int, default=1000)
    args = parser.parse_args()

    data = np.random.normal(size=(args.size, args.feature))
    filterL2(data)
