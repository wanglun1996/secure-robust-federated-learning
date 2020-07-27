import sys
sys.path.append('../infimnist_py')
import _infimnist as infimnist
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader

FEATURE_TEMPLATE = '../data/infimnist_%s_mal_feature_%d_%d.npy'
TARGET_TEMPLATE = '../data/infimnist_%s_mal_target_%d_%d.npy'
TRUE_LABEL_TEMPLATE = '../data/infimnist_%s_mal_true_label_%d_%d.npy'

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

# malicious data loader, return (X, mal_X, Y, mal_Y)
class MalDataset(Dataset):
    def __init__(self, feature_path, true_label_path, target_path, transform=None):
        self.feature = np.load(feature_path)
        self.mal_dada = np.load(feature_path)
        self.true_label = np.load(true_label_path)
        self.target = np.load(target_path)

        self.transform = transform
    
    def __getitem__(self, idx):
        sample = self.feature[idx]
        mal_data = self.mal_dada[idx]
        if self.transform:
            sample = self.transform(sample).view(28*28)
            mal_data = self.transform(mal_data).view(28*28)
        return sample, mal_data, self.true_label[idx], self.target[idx]
    
    def __len__(self):
        return self.target.shape[0]

def gen_mal_data(start=0, end=100, split=0.8):
    mnist = infimnist.InfimnistGenerator()
    indexes = np.array(np.arange(start, end), dtype=np.int64)
    digits, labels = mnist.gen(indexes)
    digits = digits.astype(np.float32).reshape(-1, 28, 28)
    # sidx = int(end * split)
    train_digits = digits
    # test_digits = digits[sidx:]
    train_labels = labels
    # test_labels = labels[sidx:]
    mal_train_labels = train_labels.copy()
    for i in range(train_labels.shape[0]):
        allowed_targets = list(range(10))
        allowed_targets.remove(train_labels[i])
        mal_train_labels[i] = np.random.choice(allowed_targets)

    # print(digits.shape)
    np.save(FEATURE_TEMPLATE%('train', start, end), train_digits)
    np.save(TRUE_LABEL_TEMPLATE%('train', start, end), train_labels)
    np.save(TARGET_TEMPLATE%('train', start, end), mal_train_labels)
    # np.save(FEATURE_TEMPLATE%('test', start, end), test_digits)
    # np.save(TRUE_LABEL_TEMPLATE%('test', start, end), test_labels)
    return None

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
