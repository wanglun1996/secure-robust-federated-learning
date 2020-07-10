import numpy as np
from dis_gauss import discretegauss

def dl_gauss(sigma2=1.0, L=1.0, size=None):
    if size is None:
        return L * discretegauss.sample_dgauss(sigma2/L/L)
    else:
        return np.array([dl_gauss(sigma2, L) for _ in range(size)])

def add_gauss(v, sigma2=1.0, L=1.0):
    """ add discrete Gaussian noise to a given vetor
        v: vector
        sigma2: noise parameter
        L: LZ
    """
    noise = dl_gauss(sigma2, L, size=len(v))
    return v + noise

def add_binomial(v, m, p, L):
    """ add binomial noise to a given vector 
        v: vector
        m, p: noise parameters
        L: LZ
    """
    noise = np.random.binomial(m, p, size=len(v))
    return noise + L * noise #TODO: subtract something from this value

if __name__ == '__main__':
    v = [1, 2, 3, 4]
    print(add_binomial(v, 10, 0.5, 1))
