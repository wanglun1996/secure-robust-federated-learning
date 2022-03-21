import numpy as np
from hashlib import sha256

def count_sketch_encode(gradient, h, w):
    print("there:", np.sum(gradient))
    sketch = []
    for prefix in range(h):
        print('inside')
        flipped_gradient = np.array([(int(sha256((str(idx)+str(prefix)).encode()).hexdigest(), 16) % 2 - 0.5) * 2 * value 
            for idx, value in enumerate(gradient)])
        sketch_idx = np.array([int(sha256((str(prefix)+str(idx)).encode()).hexdigest(), 16) % w 
            for idx in range(len(gradient))])
        sketch.append([np.sum(np.take(flipped_gradient, np.argwhere(sketch_idx==idx))) for idx in range(w)])
    print("dhere:", np.sum(sketch), np.array(sketch).flatten())
    return np.array(sketch).flatten()

def count_sketch_decode(sketch, idx, h, w):
    sketch = np.reshape(sketch, [h, w])
    return np.median([sketch[prefix][int(sha256((str(prefix)+str(idx)).encode()).hexdigest(), 16) % w] 
                        * (int(sha256((str(idx)+str(prefix)).encode()).hexdigest(), 16) % 2 - 0.5) * 2
                        for prefix in range(h)])

def count_sketch_topk(sketch, k, l, h, w):
    assert k <= l, f'top-{k} out of {l} elements'
    topk = sorted([(idx, count_sketch_decode(sketch, idx, h, w)) for idx in range(l)], key=lambda x: x[1])[-k:]
    def one_hot(l, k):
        res = np.zeros(l)
        res[k] = 1
        return res
    return np.sum(np.array([np.zeros(l)+ one_hot(l, k) * v for k, v in topk]), axis=0)

if __name__ == '__main__':
    gradient = [1., 2., 3., 4.]
    sketch = count_sketch_encode(gradient, 2, 2)
    print(sketch)
    for i in range(len(gradient)):
        print(count_sketch_decode(sketch, i, 2, 2))
    print(count_sketch_topk(sketch, 2, 4, 2, 2))
