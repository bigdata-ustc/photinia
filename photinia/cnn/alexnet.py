#!/usr/bin/env python3

"""
@author: xi
@since: 2018-03-03
"""

import tensorflow as tf

import photinia as ph


class AlexNet(ph.Widget):

    def __init__(self, name='alexnet'):
        self._height = 227
        self._width = 227
        super(AlexNet, self).__init__(name)

    def _build(self):
        #
        # conv1 padding=VALID
        self._conv1 = ph.Conv2D(
            'conv1',
            input_size=[self._height, self._width, 3],
            output_channels=96,
            filter_height=11, filter_width=11, stride_width=4, stride_height=4,
            padding='VALID'
        )
        self._pool1 = ph.Pool2D(
            'pool1',
            input_size=self._conv1.output_size,
            filter_height=3, filter_width=3, stride_height=2, stride_width=2,
            padding='VALID',
            pool_type='max'
        )
        #
        # conv2, 这里是拆分训练的
        self._conv2 = ph.GroupConv2D(
            'conv2',
            input_size=self._pool1.output_size,
            output_channels=256,
            num_groups=2,
            filter_height=5, filter_width=5, stride_height=1, stride_width=1
        )
        self._pool2 = ph.Pool2D(
            'pool2',
            input_size=self._conv2.output_size,
            filter_height=3, filter_width=3, stride_height=2, stride_width=2,
            padding='VALID',
            pool_type='max'
        )
        #
        # conv3
        self._conv3 = ph.Conv2D(
            'conv3',
            input_size=self._pool2.output_size,
            output_channels=384,
            filter_width=3, filter_height=3, stride_width=1, stride_height=1
        )
        #
        # conv4, 这里是拆分训练的
        self._conv4 = ph.GroupConv2D(
            'conv4',
            input_size=self._conv3.output_size,
            output_channels=384,
            num_groups=2,
            filter_width=3, filter_height=3, stride_width=1, stride_height=1
        )
        #
        # conv5, 这里是拆分训练的
        self._conv5 = ph.GroupConv2D(
            'conv5',
            input_size=self._conv4.output_size,
            output_channels=256,
            num_groups=2,
            filter_width=3, filter_height=3, stride_width=1, stride_height=1
        )
        self._pool5 = ph.Pool2D(
            'pool5',
            input_size=self._conv5.output_size,
            filter_height=3, filter_width=3, stride_height=2, stride_width=2,
            padding='VALID', pool_type='max'
        )
        #
        # fc layer
        self._fc6 = ph.Linear('fc6', input_size=self._pool5.flat_size, output_size=4096)
        self._fc7 = ph.Linear('fc7', input_size=self._fc6.output_size, output_size=4096)
        self._fc8 = ph.Linear(
            'fc8',
            input_size=self._fc7.output_size, output_size=1000,
            w_init=ph.init.RandomNormal(stddev=1e-4)
        )

    def _setup(self, x):
        h = ph.setup(
            x,
            [self._conv1, tf.nn.relu, self._lrn, self._pool1,
             self._conv2, tf.nn.relu, self._lrn, self._pool2,
             self._conv3, tf.nn.relu,
             self._conv4, tf.nn.relu,
             self._conv5, tf.nn.relu, self._pool5,
             ph.ops.flatten,
             self._fc6, tf.nn.relu,
             self._fc7, tf.nn.relu]
        )
        y = self._fc8.setup(h)
        y = tf.nn.softmax(y)
        return y, h

    @staticmethod
    def _lrn(x):
        return tf.nn.local_response_normalization(
            x,
            depth_radius=1,
            alpha=1e-5,
            beta=0.75,
            bias=1.0
        )

    def load_pretrain(self, model_file='alexnet.pickle'):
        ph.io.load_model_from_file(self, model_file)
