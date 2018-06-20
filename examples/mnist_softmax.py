#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一个简单的MNIST分类器，一层线性层加softmax实现

@author: winton 
@time: 2017-10-25 11:22 
"""
import os
import sys

import gflags

import photinia
import tensorflow as tf
import numpy as np

from examples import mnist


class Model(photinia.Model):
    """模型定义
    """

    def __init__(self,
                 name,
                 session,
                 input_size,
                 num_classes):
        """模型初始化

        :param name: 模型名
        :param session: 使用的tensorflow会话
        :param input_size: 输入维度
        :param num_classes: 类别数
        """
        self._input_size = input_size
        self._num_classes = num_classes
        super().__init__(name, session)

    def _build(self):
        # 网络模块定义：线性层 --- build
        self._lin = photinia.Linear('LINEAR', self._input_size, self._num_classes)
        # 输入定义
        x = tf.placeholder(dtype=photinia.D_TYPE, shape=[None, self._input_size])
        y_ = tf.placeholder(dtype=photinia.D_TYPE, shape=[None, self._num_classes])
        # 网络结构定义 --- setup
        y = self._lin.setup(x)
        # 损失函数定义， softmax交叉熵函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        # accuracy计算
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, photinia.D_TYPE))
        # 设置训练和预测的slot
        self._add_slot(
            'train',
            outputs=loss,
            inputs=(x, y_),
            updates=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        )
        self._add_slot(
            'predict',
            outputs=accuracy,
            inputs=(x, y_)
        )


class DataSource(photinia.DataSource):
    """数据源定义

    需要把mnist.py返回的图形矩阵(28x28)拉长成一维向量(784)
    """
    def __init__(self, directory):
        self._data = mnist.Data(directory)

    @property
    def test_images(self):
        size = self._data.test_images.shape[0]
        return np.reshape(self._data.test_images, newshape=(size, -1))

    @property
    def test_labels(self):
        return self._data.test_labels

    def next_batch(self, size=0):
        images_batch, labels_batch = self._data.next_batch(size)
        return np.reshape(images_batch, newshape=(size, -1)), labels_batch


def main(flags):
    # 创建数据源对象
    ds = DataSource(flags.directory)
    # tensorflow 配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 开始session
    with tf.Session(config=config) as session:
        # 创建模型对象
        model = Model('Model', session, flags.input_size, flags.num_classes)
        # 获取slot
        train = model.get_slot('train')
        predict = model.get_slot('predict')
        # 参数初始化
        session.run(tf.global_variables_initializer())
        # 开始训练
        for i in range(1, flags.nloop + 1):
            # 获取一个batch的数据
            images_batch, labels_batch = ds.next_batch(flags.bsize)
            # 输出训练交叉熵损失
            loss = train(images_batch, labels_batch)
            print('Loop {}:\tloss= {}'.format(i, loss))
        # 输出在测试集上的accuracy
        accuracy = predict(ds.test_images, ds.test_labels)
        print('Accuracy on test set: {}'.format(accuracy))
    return 0


if __name__ == '__main__':
    global_flags = gflags.FLAGS
    gflags.DEFINE_boolean('help', False, 'Show this help.')
    gflags.DEFINE_string('gpu', '0', 'Which GPU to use.')
    gflags.DEFINE_string('directory', './examples', 'Folder to save the origin data.')
    gflags.DEFINE_integer('input_size', 784, 'Dimension of input data.')
    gflags.DEFINE_integer('num_classes', 10, 'Number of classes.')
    gflags.DEFINE_integer('nloop', 1000, 'Number of loops.')
    gflags.DEFINE_integer('bsize', 100, 'Batch size.')
    global_flags(sys.argv)
    if global_flags.help:
        print(global_flags.main_module_help())
        exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = global_flags.gpu
    exit(main(global_flags))
