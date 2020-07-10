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
    
def filterL2(samples, sigma=1):
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
        # print(tau)
        tau_max = np.amax(tau)
        c = c * (1 - tau/tau_max)
        # print(c)

        # return

if __name__ == '__main__':
    data = np.array([[0, 2], [1, 1], [2, 0]])#simulate()
    filterL2(data)
