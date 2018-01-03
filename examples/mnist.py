#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winton 
@time: 2017-10-25 15:24 
"""
import gzip
import os
import struct
from urllib import request

import numpy as np

from photinia import Dataset


class Data(Dataset):
    def __init__(self,
                 directory):
        self._dir = directory
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        self._source_url = 'http://yann.lecun.com/exdb/mnist/'
        train_images_name = 'train-im_utils-idx3-ubyte.gz'
        train_labels_name = 'train-labels-idx1-ubyte.gz'
        test_images_name = 't10k-im_utils-idx3-ubyte.gz'
        test_labels_name = 't10k-labels-idx1-ubyte.gz'
        self._train_images, self._train_labels = self._get_data(train_images_name, train_labels_name)
        self._test_images, self._test_labels = self._get_data(test_images_name, test_labels_name)
        super().__init__(self._train_images, self._train_labels, dtype=np.float32)

    @property
    def train_images(self):
        return self._train_images

    @property
    def train_labels(self):
        return self._train_labels

    @property
    def test_images(self):
        return self._test_images

    @property
    def test_labels(self):
        return self._test_labels

    def _get_data(self, images_name, labels_name):
        file = self._download(images_name)
        images = self._extract_images(file)
        file = self._download(labels_name)
        labels = self._extract_labels(file)
        return images, labels

    def _download(self, filename):
        """如果数据集不存在的话，下载数据集

        :param filename: 文件名
        :return: 下载文件的路径
        """
        file_path = os.path.join(self._dir, filename)
        if not os.path.exists(file_path):
            print('Downloading {}'.format(filename))
            request.urlretrieve(self._source_url + filename, file_path)
            print('Successfully downloaded {}'.format(filename))
        return file_path

    @staticmethod
    def _extract_images(filename):
        """从图片压缩文件中提取图片的像素矩阵表示

        :param filename: 图片文件名
        :return: 4维的numpy矩阵[index, y, x, depth]， 类型为np.float32
        """
        images = []
        print('Extracting {}'.format(filename))
        with gzip.GzipFile(fileobj=open(filename, 'rb')) as f:
            buf = f.read()
            index = 0
            magic, num_images, rows, cols = struct.unpack_from('>IIII', buf, index)
            if magic != 2051:
                raise ValueError('Invalid magic number {} in MNIST image file: {}'.format(magic, filename))
            index += struct.calcsize('>IIII')
            for i in range(num_images):
                img = struct.unpack_from('>784B', buf, index)
                index += struct.calcsize('>784B')
                img = np.array(img, dtype=np.float32)
                # 将像素从[0,255]转换为[0,1]
                img = np.multiply(img, 1.0 / 255.0)
                img = img.reshape(rows, cols, 1)
                images.append(img)
        return np.array(images, dtype=np.float32)

    @staticmethod
    def _extract_labels(filename, num_classes=10):
        """从类别压缩文件中提取其矩阵表示

        :param filename: 类别文件名
        :param num_classes: 用于one-hot编码的类别数，这里是10类
        :return: 2维的numpy矩阵[index, num_classes]， 类型为np.float32
        """
        labels = []
        print('Extracting {}'.format(filename))
        with gzip.GzipFile(fileobj=open(filename, 'rb')) as f:
            buf = f.read()
            index = 0
            magic, num_labels = struct.unpack_from('>II', buf, index)
            if magic != 2049:
                raise ValueError('Invalid magic number {} in MNIST label file: {}'.format(magic, filename))
            index += struct.calcsize('>II')
            for i in range(num_labels):
                label = struct.unpack_from('>B', buf, index)
                index += struct.calcsize('>B')
                label_one_hot = np.zeros(num_classes, dtype=np.float32)
                label_one_hot[label[0]] = 1
                labels.append(label_one_hot)
        return np.array(labels, dtype=np.float32)
