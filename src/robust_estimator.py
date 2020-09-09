import argparse
import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg import eigh
import cvxpy as cvx

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

def filterL2_(samples, sigma=1, expansion=20):
    """
    samples: data samples in numpy array
    sigma: operator norm of covariance matrix assumption
    """
    size = samples.shape[0]
    feature_size = samples.shape[1]
    samples_ = samples.reshape(size, 1, feature_size)

    c = np.ones(size)
    count = 0
    while True:
        count += 1
        if count > 100:
            sigma *= 2
            count = 0
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
 
def filterL2(samples, sigma=1, expansion=20, itv=None):
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
        res.append(filterL2_(samples_flatten[:,idx:idx+size], sigma, expansion))
        idx += size

    return np.concatenate(res, axis=0).reshape(feature_shape)
  
#         print(size)
#         res.append(filterL2_(samples[:,idx:idx+size], sigma))
#         idx += size

#     return np.concatenate(res, axis=0)

def pgd(samples, epsilon=1e-2, lr=1e-3):

    """
    Robust estimator from paper: High-Dimensional Robust Mean Estimation via Gradient Descent
    """

    # FIXME: check that this is correct
    def weighted_cov(weight):
        mean = weight @ samples
        cov = (samples - mean).T @ (weight.reshape(-1, 1) * (samples - mean))
        return cov

    def compute_grad(u, w):
        return (samples @ u.reshape(-1, 1)) * (samples @ u.reshape(-1, 1)) - 2 * (u.reshape(1, -1) @ samples.transpose() @ w.reshape(-1, 1) * samples @ u.reshape(-1, 1))

    # FIXME: make sure the projection is correct
    def project_l2(w):
        x = cvx.Variable(w.shape)
        objective = cvx.Minimize(cvx.norm(w-x, 2))
        constraints = [x >= 0, x <= 1 / (1-epsilon) / w.shape[0], sum(x)==1]
        prob = cvx.Problem(objective, constraints)
        results = prob.solve(verbose=False)

        w = np.array(x.value)
        w[np.abs(w)<1e-9]=0
        return w

    size = samples.shape[0]
    feature_size = samples.shape[1]
    weight = np.ones(size) / size

    # FIXME: no way to really use this
    T = 100 # size * size * feature_size * feature_size * feature_size * feature_size
    for _ in range(T):
        cov = weighted_cov(weight)
        # FIXME: make sure the error is within the allowed range
        eig_val, eig_vec = eigh(cov, eigvals=(feature_size-1, feature_size-1), eigvals_only=False)
        # eig_val = eig_val[0]
        eig_vec = eig_vec.T[0]

        grad = compute_grad(eig_vec, weight)

        weight = project_l2(weight - lr * grad.flatten())

    return weight

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
    # index = np.argsort(samples, axis=0)
    # print(index[0], index[-1])
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

# def bulyan(samples, agg=krum, args=None, theta=2):
#     samples = np.array(samples)
#     feature_size = samples.shape[1]
#     # beta = theta - 2*f
#     #FIXME: the above is correct
#     beta = 2
#     S = []

#     for _ in range(theta):
#        picked_sample = agg(samples)
#        S.append(picked_sample)
#        samples = np.delete(samples, np.argwhere([picked_sample])[:1], axis=0)

#     S = np.array(S)
#     res = np.zeros(feature_size)
#     # coordinate-wise median
#     for idx in range(feature_size):
#         samples_ = S[:, idx]
#         med = np.median(samples_)
#         idxs = np.argsort([np.abs(sample_-med) for sample_ in samples_])[:beta]
#         res[idx] = np.average(S[idxs])

#     print(res)
#     return res
    
# bulyan
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

def bulyan(grads, aggsubfunc='trimmedmean', f=2):
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
        # print(theta)
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

    # print(selected_grads[0].shape)
    beta = theta - 2 * f
    np_grads = np.array([g.flatten().tolist() for g in selected_grads])

    grads_dim = len(np_grads[0])
    selected_grads_by_cod = np.zeros([grads_dim, 1])  # shape of torch grads
    for i in range(grads_dim):
        selected_grads_by_cod[i, 0] = bulyan_one_coordinate(np_grads[:, i], beta)

    return selected_grads_by_cod.reshape(feature_shape)
    # if use_cuda:
    #     cuda_tensor = torch.from_numpy(selected_grads_by_cod.astype(np.float32)).cuda()
    #     return cuda_tensor
    # else:
    #     return torch.from_numpy(selected_grads_by_cod.astype(np.float32))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--feature', type=int, default=1000)
    args = parser.parse_args()

    #data = np.random.normal(size=(args.size, args.feature))
    data = []
    for i in range(20):
        data.append(np.random.normal(size=(20,20)))
    # filterL2(data)
    res = bulyan(data)
    print(res)
    # data = np.random.normal(size=(args.size, args.feature))
    # w = pgd(data)
    # print(w)
