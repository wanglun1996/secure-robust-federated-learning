import os
import sys
sys.path.append('../infimnist_py')
import _infimnist as infimnist
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

import cv2

INFIMNIST_FEATURE_TEMPLATE = '../data/infimnist_%s_feature_%d_%d.npy'
INFIMNIST_TARGET_TEMPLATE = '../data/infimnist_%s_target_%d_%d.npy'
INFIMNIST_MAL_FEATURE_TEMPLATE = '../data/infimnist_%s_mal_feature_%d_%d.npy'
INFIMNIST_MAL_TARGET_TEMPLATE = '../data/infimnist_%s_mal_target_%d_%d.npy'
INFIMNIST_TRUE_LABEL_TEMPLATE = '../data/infimnist_%s_mal_true_label_%d_%d.npy'

CIFAR_MAL_FEATURE_TEMPLATE = '../data/cifar_mal_feature_10.npy'
CIFAR_MAL_TARGET_TEMPLATE = '../data/cifar_mal_target_10.npy'
CIFAR_MAL_TRUE_LABEL_TEMPLATE = '../data/cifar_mal_true_label_10.npy'

FASHION_MAL_FEATURE_TEMPLATE = '../data/fashion_mal_feature_10.npy'
FASHION_MAL_TARGET_TEMPLATE = '../data/fashion_mal_target_10.npy'
FASHION_MAL_TRUE_LABEL_TEMPLATE = '../data/fashion_mal_true_label_10.npy'

CH_MAL_FEATURE_TEMPLATE = '../data/chmnist_mal_feature_10.npy'
CH_MAL_TARGET_TEMPLATE = '../data/chmnist_mal_target_10.npy'
CH_MAL_TRUE_LABEL_TEMPLATE = '../data/chmnist_mal_true_label_10.npy'

CHMNIST_PATH = "../data/Kather_texture_2016_image_tiles_5000/"

# should I include the target in the sample?
class MyDataset(Dataset):
    def __init__(self, feature_path, target_path, transform=None):
        self.feature = np.load(feature_path)
        self.target = np.load(target_path)
        self.transform = transform
    
    def __getitem__(self, idx):
        sample = self.feature[idx]
        if self.transform:
            sample = self.transform(sample)
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
            sample = self.transform(sample)
            mal_data = self.transform(mal_data)
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
    np.save(INFIMNIST_MAL_FEATURE_TEMPLATE%('train', start, end), train_digits)
    np.save(INFIMNIST_TRUE_LABEL_TEMPLATE%('train', start, end), train_labels)
    np.save(INFIMNIST_MAL_TARGET_TEMPLATE%('train', start, end), mal_train_labels)
    # np.save(FEATURE_TEMPLATE%('test', start, end), test_digits)
    # np.save(TRUE_LABEL_TEMPLATE%('test', start, end), test_labels)
    return None

def gen_mal_cifar(batch_size=10):
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=None)
    sizes = [batch_size] * (len(test_set) // batch_size)
    test_sets = random_split(test_set, sizes)
    for idx, (feature, target) in enumerate(DataLoader(test_sets[0], batch_size=10, shuffle=True), 0):
        print(idx)
        np.save(CIFAR_MAL_FEATURE_TEMPLATE, feature)
        np.save(CIFAR_MAL_TRUE_LABEL_TEMPLATE, target)
        mal_train_labels = target.copy()
        for i in range(target.shape[0]):
            allowed_targets = list(range(10))
            allowed_targets.remove(target[i])
            mal_train_labels[i] = np.random.choice(allowed_targets)
        np.save(CIFAR_MAL_TARGET_TEMPLATE, mal_train_labels)

def gen_mal_fashion(batch_size=10):
    test_set = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=None)
    sizes = [batch_size] * (len(test_set) // batch_size)
    test_sets = random_split(test_set, sizes)
    for idx, (feature, target) in enumerate(DataLoader(test_sets[0], batch_size=10, shuffle=True), 0):
        print(idx)
        np.save(FASHION_MAL_FEATURE_TEMPLATE, feature)
        np.save(FASHION_MAL_TRUE_LABEL_TEMPLATE, target)
        mal_train_labels = target.copy()
        for i in range(target.shape[0]):
            allowed_targets = list(range(10))
            allowed_targets.remove(target[i])
            mal_train_labels[i] = np.random.choice(allowed_targets)
        np.save(FASHION_MAL_TARGET_TEMPLATE, mal_train_labels)

def gen_mal_chmnist(batch_size=10):
    test_set = MyDataset("../data/CHMNIST_TEST_FEATURE.npy", "../data/CHMNIST_TEST_TARGET.npy")
    sizes = [batch_size] * (len(test_set) // batch_size)
    test_sets = random_split(test_set, sizes)
    for idx, (feature, target) in enumerate(DataLoader(test_sets[0], batch_size=10, shuffle=True), 0):
        print(idx)
        np.save(CH_MAL_FEATURE_TEMPLATE, feature)
        np.save(CH_MAL_TRUE_LABEL_TEMPLATE, target)
        mal_train_labels = target.copy()
        for i in range(target.shape[0]):
            allowed_targets = list(range(10))
            allowed_targets.remove(target[i])
            mal_train_labels[i] = np.random.choice(allowed_targets)
        np.save(CH_MAL_TARGET_TEMPLATE, mal_train_labels)


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
    np.save(INFIMNIST_FEATURE_TEMPLATE%('train', start, end), train_digits)
    np.save(INFIMNIST_TARGET_TEMPLATE%('train', start, end), train_labels)
    np.save(INFIMNIST_FEATURE_TEMPLATE%('test', start, end), test_digits)
    np.save(INFIMNIST_TARGET_TEMPLATE%('test', start, end), test_labels)

def gen_chmnist(split=0.8):
    x = []
    y = []
    for category in os.listdir(CHMNIST_PATH):
        for image in os.listdir(CHMNIST_PATH + category):
            if image.endswith(".tif"):
                print("found " + image)
                arr = cv2.imread(CHMNIST_PATH + category + '/' + image)
                x.append(arr)
                y.append(category)

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
 
    x = np.array(x)
    y = np.array(y)
    print(y)

    idx = shuffle(np.arange(x.shape[0]))
    x = x[idx]
    y = y[idx]

    sidx = int(x.shape[0] * split)
    x_train = x[:sidx]
    y_train = y[:sidx]
    x_test = x[sidx:]
    y_test = y[sidx:]

    np.save("../data/CHMNIST_TRAIN_FEATURE.npy", x_train)
    np.save("../data/CHMNIST_TRAIN_TARGET.npy", y_train)
    np.save("../data/CHMNIST_TEST_FEATURE.npy", x_test)
    np.save("../data/CHMNIST_TEST_TARGET.npy", y_test) 

if __name__ == '__main__':

    # gen_chmnist()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=10000)
    args = parser.parse_args()
    gen_infimnist(0, args.size)
    gen_mal_data(0, 10)

    # gen_infimnist(0, args.size)
    # dataset_loader = DataLoader(MyDataset(FEATURE_TEMPLATE%(0,100), TARGET_TEMPLATE%(0,100)))
    # examples = enumerate(dataset_loader)
    # batch_idx, (feature, target) = next(examples)
    # print(batch_idx, feature, target)
