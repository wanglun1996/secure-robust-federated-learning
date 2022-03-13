import numpy as np
from hashlib import sha256

def count_sketch_encode(gradient, h, w):
    sketch = []
    for prefix in range(h):
        flipped_gradient = np.array([(int(sha256((str(idx)+str(prefix)).encode()).hexdigest(), 16) % 2 - 0.5) * 2 * value 
            for idx, value in enumerate(gradient)])
        print(flipped_gradient)
        sketch_idx = np.array([int(sha256((str(prefix)+str(idx)).encode()).hexdigest(), 16) % w 
            for idx in range(len(gradient))])
        sketch.append([np.sum(np.take(flipped_gradient, np.argwhere(sketch_idx==idx))) for idx in range(w)])
    return sketch

def count_sketch_decode(sketch, idx, h, w):
    return np.median([sketch[prefix][int(sha256((str(prefix)+str(idx)).encode()).hexdigest(), 16) % w] 
                        * (int(sha256((str(idx)+str(prefix)).encode()).hexdigest(), 16) % 2 - 0.5) * 2
                        for prefix in range(h)])

if __name__ == '__main__':
    gradient = [1., 2., 3., 4.]
    sketch = count_sketch_encode(gradient, 2, 2)
    print(sketch)
    for i in range(len(gradient)):
        print(count_sketch_decode(sketch, i, 2, 2))
