import numpy as np
import gzip
import os
import platform
import pickle

import pandas as pd
import torchvision
from torchvision import transforms as transforms

class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None  # 训练集
        self.train_label = None  # 标签
        self.train_data_size = None  # 训练数据的大小
        self.test_data = None   # 测试数据集
        self.test_label = None  # 测试的标签
        self.test_data_size = None  # 测试集数据Size

        self._index_in_train_epoch = 0

        # 如果数据集是mnist
        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        elif self.name == 'cifar10':
            self.load_data(isIID)
        else:
            pass

    def mnistDataSetConstruct(self, isIID):
        # 加载数据集
        data_dir = r'.\data\MNIST'
        # data_dir = r'./data/MNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        print("-"*5+"train_images"+"-"*5)
        # 输出第一张图片
        #print(train_images[0].reshape(28,28))
        print(train_images.shape) # (60000, 28, 28, 1) 一共60000 张图片，每一张是28*28*1
        print('-'*22+"\n")
        train_labels = extract_labels(train_labels_path)
        print("-" * 5 + "train_labels" + "-" * 5)
        print(train_labels.shape) # (60000, 10)
        print('-'*22+"\n")

        test_images = extract_images(test_images_path)
        print("-" * 5 + "test_images" + "-" * 5)
        print(test_images.shape) # (10000, 28, 28, 1)
        print('-' * 22 + "\n")
        test_labels = extract_labels(test_labels_path)
        print("-" * 5 + "test_labels" + "-" * 5)
        print(test_labels.shape) # (10000, 10) 10000维
        print('-' * 22 + "\n")

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        # 训练数据Size
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        print(train_images.shape)
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])
        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        # 独立同分布
        if isIID:
            #一个参数 默认起点0，步长为1 输出：[0 1 2]
            # a = np.arange(3)
            # 一共60000个
            order = np.arange(self.train_data_size)
            # numpy 中的随机打乱数据方法np.random.shuffle
            np.random.shuffle(order)
            order = order.astype('int')
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:

            labels = np.argmax(train_labels, axis=1)
            # # 对数据标签进行排序
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
            # # 用标签的下标排序的方式
            labels = np.argmax(train_labels, axis=1)
            order = np.arange(labels.size)
            np.random.shuffle(order)
            labels1 = order[0:14285]  # 前5个用户的数据iid后面用户数据noniid
            labels2 = np.argsort(labels[14285:labels.size])+14285

            order1 = np.hstack((labels1, labels2))
            self.train_data = train_images[order1]
            self.train_label = train_labels[order1]
        self.test_data = test_images
        self.test_label = test_labels

    # 加载cifar10 的数据
    def load_data(self, isIID):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                                 transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
        train_data = train_set.data  # (50000, 32, 32, 3)
        train_labels = train_set.targets
        train_labels = np.array(train_labels)  # 将标签转化为
        print(type(train_labels))  # <class 'numpy.ndarray'>
        print(train_labels.shape)  # (50000,)

        test_data = test_set.data  # 测试数据
        test_labels = test_set.targets
        test_labels = np.array(test_labels)
        # print()
        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]

        # 把数据改成(50000,3,32,32)才符合当前神经网络训练规则
        train_images = np.transpose(train_data, (0, 3, 1, 2))
        print("数据集维度", train_images.shape)
        test_images = np.transpose(test_data, (0, 3, 1, 2))
        # ---------------------------归一化处理------------------------------#
        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        # ----------------------------------------------------------------#
        if isIID:
            # 这里将50000 个训练集随机打乱
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            order = order.astype('int')
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # 按照标签的
            # labels = np.argmax(train_labels, axis=1)
            # 对数据标签进行排序
            order = np.argsort(train_labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
            labels1 = np.arange(23800)  # 前10000个数据随机分布
            labels2 = np.argsort(self.train_label[23800:self.train_label.size])+23800 # 后几万个数据按顺序分布
            order1 = np.hstack((labels1, labels2))
            self.train_data = self.train_data[order1]
            self.train_label = self.train_label[order1]
        self.test_data = test_images
        self.test_label = test_labels

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


if __name__=="__main__":
    'test data set'
    mnistDataSet = GetDataSet('cifar10', 1)  # test NON-IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])
    pd.set_option('display.max_columns', None)  # 显示完整的列
    pd.set_option('display.max_rows', None)  # 显示完整的行
    print("前100标签", mnistDataSet.train_label[0:100])
    print("标签", mnistDataSet.train_label[32000:32100])
