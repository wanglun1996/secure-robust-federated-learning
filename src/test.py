import argparse
import numpy as np
from comm import quantize, randomRotate, cylicRound
from dis_dist import add_gauss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=4)
    parser.add_argument('--value_bound', type=float, default=1.0)
    parser.add_argument('--tot_bound', type=float, default=10.0)
    parser.add_argument('--level', type=int, default=21)
    parser.add_argument('--sigma2', type=float, default=1.0)
    args = parser.parse_args()

    SIZE = args.size
    V_BOUND = args.value_bound
    BOUND = args.tot_bound
    LEVEL = args.level
    SIGMA2 = args.sigma2
    STEP_SIZE = 2 * V_BOUND / (LEVEL - 1)

    v = np.random.uniform(-1,1,SIZE)
    print(v)
    v = quantize(v, LEVEL, V_BOUND)
    print(v)
    v = add_gauss(v, SIGMA2, STEP_SIZE)
    print(v)
    # round operation
    v = randomRotate(v)
    # although this does not change differential privacy, this will change communication cost by adding a O(\sqrt(d)) factor
    print(v)
    v = cylicRound(v, STEP_SIZE, BOUND)
    print(v)
    
