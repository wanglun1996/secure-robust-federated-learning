import sys
sys.path.append('../infimnist_py')
import _infimnist as infimnist
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader

FEATURE_TEMPLATE = '../data/infimnist_%s_feature_%d_%d.npy'
TARGET_TEMPLATE = '../data/infimnist_%s_target_%d_%d.npy'

# should I include the target in the sample?
class MyDataset(Dataset):
    def __init__(self, feature_path, target_path, transform=None):
        self.feature = np.load(feature_path)
        self.target = np.load(target_path)
        self.transform = transform
    
    def __getitem__(self, idx):
        sample = self.feature[idx]
        if self.transform:
            sample = self.transform(sample).view(28*28)
        return sample, self.target[idx]
    
    def __len__(self):
        return self.target.shape[0]

def gen_infimnist(start=0, end=10000, split=0.8):
    mnist = infimnist.InfimnistGenerator()
    indexes = np.array(np.arange(start, end), dtype=np.int64)
    digits, labels = mnist.gen(indexes)
    digits = digits.astype(np.float32).reshape(-1, 28, 28)
    sidx = int(end * split)
    train_digits = digits[:sidx]
    test_digits = digits[sidx:]
    train_labels = labels[:sidx]
    test_labels = labels[sidx:]
    # print(digits.shape)
    np.save(FEATURE_TEMPLATE%('train', start, end), train_digits)
    np.save(TARGET_TEMPLATE%('train', start, end), train_labels)
    np.save(FEATURE_TEMPLATE%('test', start, end), test_digits)
    np.save(TARGET_TEMPLATE%('test', start, end), test_labels)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=10000)
    args = parser.parse_args()

    gen_infimnist(0, args.size)
    # dataset_loader = DataLoader(MyDataset(FEATURE_TEMPLATE%(0,100), TARGET_TEMPLATE%(0,100)))
    # examples = enumerate(dataset_loader)
    # batch_idx, (feature, target) = next(examples)
    # print(batch_idx, feature, target)
