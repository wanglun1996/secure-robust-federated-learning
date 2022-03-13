import numpy as np
from hashlib import sha256

def count_sketch_encode(gradient, h, w):
    sketch = []
    for prefix in range(h):
        sketch_idx = np.array([int(sha256((str(prefix)+str(idx)).encode()).hexdigest(), 16) % w for idx, value in enumerate(gradient)])
        sketch.append([np.sum(np.take(gradient, np.argwhere(sketch_idx==idx))) for idx in range(w)])
    return sketch

def count_sketch_decode():
    raise NotImplementedError

if __name__ == '__main__':
    gradient = [1., 2., 3., 4.]
    sketch = count_sketch_encode(gradient, 2, 2)
    print(sketch)
