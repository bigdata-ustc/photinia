#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用CNN对MNIST数据集进行分类

@author: winton 
@time: 2017-10-26 16:43 
"""
import os
import sys

import gflags

import photinia
import tensorflow as tf

from examples import mnist


class Model(photinia.Trainable):
    """模型定义
    """

    def __init__(self,
                 name,
                 session,
                 height,
                 width,
                 depth,
                 feature_size,
                 num_classes):
        """模型初始化

        :param name: 模型名
        :param session: 使用的tensorflow会话
        :param height: 图片高度
        :param width: 图片宽度
        :param depth: 图片通道数
        :param feature_size: 全连接层输出维度
        :param num_classes: 类别数
        """
        self._height = height
        self._width = width
        self._depth = depth
        self._feature_size = feature_size
        self._num_classes = num_classes
        super().__init__(name, session)

    def _build(self):
        # 网络模块定义 --- build
        self._cnn = photinia.CNN('CNN',
                                 input_height=self._height,
                                 input_width=self._width,
                                 input_depth=1,
                                 layer_shapes=[(5, 5, 32, 2, 2),
                                               (5, 5, 64, 2, 2)],
                                 activation=tf.nn.relu,
                                 with_batch_norm=False
                                 ).build()
        self._lin1 = photinia.Linear('LINEAR1', self._cnn.flat_size, self._feature_size).build()
        self._lin2 = photinia.Linear('LINEAR2', self._feature_size, self._num_classes).build()
        # dropout参数
        keep_prob = tf.placeholder(dtype=photinia.D_TYPE)
        # 输入
        x = tf.placeholder(dtype=photinia.D_TYPE, shape=[None, self._height, self._width, self._depth])
        y_ = tf.placeholder(dtype=photinia.D_TYPE, shape=[None, self._num_classes])
        # 网络结构定义 --- setup
        y = self._cnn.setup(x)
        y = self._lin1.setup(y)
        y = tf.nn.dropout(y, keep_prob)
        y = self._lin2.setup(y)
        # 损失函数定义, softmax交叉熵函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        # accuracy计算
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, photinia.D_TYPE))
        # 设置训练和预测的slot
        self._add_slot(
            'train',
            outputs=(loss, accuracy),
            inputs=(x, y_, keep_prob),
            updates=tf.train.AdamOptimizer(1e-4).minimize(loss)
        )
        self._add_slot(
            'predict',
            outputs=accuracy,
            inputs=(x, y_, keep_prob)
        )


def main(flags):
    # 创建数据源对象
    ds = mnist.Data(flags.directory)
    # tensorflow 配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 开始session
    with tf.Session(config=config) as session:
        # 创建模型对象
        model = Model('Model',
                      session,
                      flags.height,
                      flags.width,
                      flags.depth,
                      flags.feature_size,
                      flags.num_classes).build()
        # 获取slot
        train = model.get_slot('train')
        predict = model.get_slot('predict')
        # 参数初始化
        session.run(tf.global_variables_initializer())
        # 开始训练
        for i in range(1, flags.nloop + 1):
            # 获取一个batch的数据
            images_batch, labels_batch = ds.next_batch(flags.bsize)
            loss, train_accuracy = train(images_batch, labels_batch, 0.5)
            # 每100次迭代输出训练交叉熵损失以batch上的accuracy
            if i % 100 == 0:
                print('Loop {}:\tloss={}\ttrain accuracy={}'.format(i, loss, train_accuracy))
        # 输出在测试集上的accuracy
        accuracy = predict(ds.test_images, ds.test_labels, 1.0)
        print('Accuracy on test set: {}'.format(accuracy))
    return 0


if __name__ == '__main__':
    global_flags = gflags.FLAGS
    gflags.DEFINE_boolean('help', False, 'Show this help.')
    gflags.DEFINE_string('gpu', '0', 'Which GPU to use.')
    gflags.DEFINE_string('directory', './examples', 'Folder to save the origin data.')
    gflags.DEFINE_integer('height', 28, 'Height of image.')
    gflags.DEFINE_integer('width', 28, 'Width of image.')
    gflags.DEFINE_integer('depth', 1, 'Depth of image.')
    gflags.DEFINE_integer('feature_size', 1024, 'Output dimension of fully-connected layer .')
    gflags.DEFINE_integer('num_classes', 10, 'Number of classes.')
    gflags.DEFINE_integer('nloop', 20000, 'Number of loops.')
    gflags.DEFINE_integer('bsize', 50, 'Batch size.')
    global_flags(sys.argv)
    if global_flags.help:
        print(global_flags.main_module_help())
        exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = global_flags.gpu
    exit(main(global_flags))
