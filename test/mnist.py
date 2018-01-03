#!/usr/bin/env python3

"""
@author: xi
@since: 2016-11-11
"""

import os
import sys

import gflags
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import photinia as ph


class MNISTModel(ph.Trainer):
    """MNISTModel
    """

    def __init__(self,
                 session,
                 height,
                 width):
        self._height = height
        self._width = width
        super(MNISTModel, self).__init__('MNISTModel', session)

    def _build(self):
        #
        # Layers.
        conv_layers = [
            ph.Convolutional('Conv1', 1, 32, 5, 5),
            ph.Convolutional('Conv2', 32, 64, 5, 5),
            ph.Convolutional('Conv3', 64, 128, 5, 5)
        ]
        hidden_layer = ph.Linear('Hidden', 2048, 4096)
        output_layer = ph.Linear('Output', 4096, 10)
        #
        # Get "y" and "label" from x.
        x = tf.placeholder(
            shape=(None, self._height, self._width, 1),
            dtype=ph.D_TYPE
        )
        h = x
        for conv_layer in conv_layers:
            h = ph.lrelu(conv_layer.setup(h))
        h = tf.reshape(h, (-1, 2048))
        h = ph.lrelu(hidden_layer.setup(h))
        y = ph.lrelu(output_layer.setup(h))
        y = tf.nn.softmax(y)
        label = tf.arg_max(y, 1)
        #
        # Get loss function from "y", "label" and "label_"
        label_ = tf.placeholder(
            shape=(None, 10),
            dtype=ph.D_TYPE
        )
        loss = -tf.reduce_mean(tf.log(tf.reduce_sum(y * label_, 1)))
        correct = tf.equal(label, tf.argmax(label_, 1))
        error = 1 - tf.reduce_mean(tf.cast(correct, ph.D_TYPE))
        #
        # Slots
        self._add_slot(
            name='train',
            inputs=(x, label_),
            updates=tf.train.AdamOptimizer(1e-4, 0.5).minimize(loss, var_list=self.trainable_variables())
        )
        self._add_slot(
            name='evaluate',
            outputs=(error, loss),
            inputs=(x, label_),
        )
        self._add_slot(
            name='predict',
            outputs=label,
            inputs=(x,),
        )
