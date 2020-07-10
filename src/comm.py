import numpy as np
from scipy.linalg import hadamard

def quantize(v, k, B):
    """stochastic k-level quantization protocol
       v: the vector to be quantized
       k: the level of quantization
       B: the upper and lower bound
    """
    intervals = np.linspace(-B, B, num=k)
    step_size = 2 * B / (k-1)

    def binarySearch(itv, a):
        l = 0
        u = len(itv) - 1
        while u > l:
            idx = (l+u) // 2
            if itv[idx] == a:
                return idx
            if itv[idx] > a:
                u = idx
                if l == u:
                    return l-1
            if itv[idx] < a:
                l = idx+1
                if l == u:
                    if itv[l] <= a:
                        return l
                    else:
                        return l-1

    idx = np.array([binarySearch(intervals, x) for x in v])
    idx = np.array([[i, i+1] for i in idx])
    probs = np.array([[1-(x-intervals[idx_pair[0]])/step_size, (x-intervals[idx_pair[0]])/step_size] for x, idx_pair in list(zip(v, idx))])

    return np.array([intervals[np.random.choice(idx_pair, p=prob)] for idx_pair, prob in list(zip(idx, probs))])

def randomRotate(v):
    """ random rotate the feature vector
        v: feature vector, the dimension must be a power of 2
    """
    d = len(v)
    diag = np.random.choice([-1, 1], d)

    def fwht(v):
        """In-place Fast Walsh–Hadamard Transform of array a."""
        h = 1
        while h < len(v):
            for i in range(0, len(v), h * 2):
                for j in range(i, i + h):
                    x = v[j]
                    y = v[j + h]
                    v[j] = x + y
                    v[j + h] = x - y
            h *= 2
        return v

    rot_v = fwht(diag * v) / np.sqrt(d)

    return rot_v

def cylicRound(v, step_size, B):
    return np.array([int((x+B)/step_size)%int(2*B/step_size+1) for x in v])

if __name__ == '__main__':
    v = [-4, -3, -2, -1, 1, 2, 3, 4]
    v = quantize(v, 10, 5)
    print(np.linalg.norm(v))
    v = randomRotate(v)
    print(np.linalg.norm(v))
