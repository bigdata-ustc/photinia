#!/usr/bin/env python3

"""
@author: xi
@since: 2017-12-01
"""

import argparse
import os
import pickle

import numpy as np
import tensorflow as tf

import photinia as ph
from photinia import utils


class ImageEmbedding(ph.Widget):

    def __init__(self,
                 name,
                 height,
                 width,
                 channels,
                 emb_size):
        self._height = height
        self._width = width
        self._channels = channels
        self._emb_size = emb_size
        super(ImageEmbedding, self).__init__(name)

    @property
    def emb_size(self):
        return self._emb_size

    @property
    def output_size(self):
        return self._emb_size

    def _build(self):
        self._c1 = ph.Conv2D('c1', (self._height, self._width, 3), 96, 3, 3)
        self._c2 = ph.Conv2D('c2', self._c1.output_size, 96, 3, 3)
        self._c3 = ph.Conv2D('c3', self._c2.output_size, 96, 3, 3, 2, 2)
        #
        self._c4 = ph.Conv2D('c4', self._c3.output_size, 192, 3, 3)
        self._c5 = ph.Conv2D('c5', self._c4.output_size, 192, 3, 3)
        self._c6 = ph.Conv2D('c6', self._c5.output_size, 192, 3, 3, 2, 2)
        #
        self._c7 = ph.Conv2D('c7', self._c6.output_size, 256, 3, 3)
        self._c8 = ph.Conv2D('c8', self._c7.output_size, 256, 3, 3)
        self._c9 = ph.Conv2D('c9', self._c8.output_size, 256, 3, 3, 2, 2)
        #
        self._c10 = ph.Conv2D('c10', self._c9.output_size, 256, 1, 1)
        self._c11 = ph.Conv2D('c11', self._c10.output_size, 256, 1, 1)
        self._c11.flat_output = True
        #
        self._hid = ph.Linear(
            'hid',
            self._c11.flat_size, self._emb_size,
            weight_initializer=ph.RandomNormal(stddev=1e-4)
        )

    def _setup(self, x, activation=ph.swish, dropout=None):
        v = ph.setup(
            x,
            [self._c1, activation,
             self._c2, activation,
             self._c3, activation, dropout,
             self._c4, activation, dropout,
             self._c5, activation, dropout,
             self._c6, activation, dropout,
             self._c7, activation, dropout,
             self._c8, activation, dropout,
             self._c9, activation, dropout,
             self._c10, activation, dropout,
             self._c11, activation, dropout,
             self._hid]
        )
        return v


class Cifar10(ph.Trainer):
    """cifar-10
    """

    def __init__(self, session=None):
        super(Cifar10, self).__init__('Cifar10', session)

    def _build(self):
        emb = ImageEmbedding('emb', 32, 32, 3, 2000)
        out = ph.Linear('out', emb.output_size, 10, weight_initializer=ph.RandomNormal(stddev=1e-4))
        dropout1 = ph.Dropout('dropout')
        #
        # Setup layers.
        x = tf.placeholder(
            dtype=ph.D_TYPE,
            shape=(None, 32, 32, 3)
        )
        y = emb.setup(x, dropout=dropout1)
        y = dropout1.setup(y)
        y = out.setup(y)
        #
        label = tf.argmax(y, 1)
        label_ = tf.placeholder(
            shape=(None, 10),
            dtype=ph.D_TYPE
        )
        loss = tf.reduce_sum((y - label_) ** 2, 1)
        loss_sum = tf.reduce_sum(loss)
        loss_mean = tf.reduce_mean(loss)
        correct = tf.equal(label, tf.argmax(label_, 1))
        correct = tf.reduce_sum(tf.cast(correct, ph.D_TYPE))
        #
        # Slots
        # optimizer = tf.train.AdagradOptimizer(1e-2)
        # optimizer = tf.train.AdadeltaOptimizer(1e-1)
        # optimizer = tf.train.RMSPropOptimizer(1e-5, momentum=0.9)
        optimizer = tf.train.AdamOptimizer(1e-4)
        # optimizer = tf.train.MomentumOptimizer(1e-2, 0.9)
        # optimizer = ph.GradientClipping(optimizer, 1)
        self._add_train_slot(
            inputs=(x, label_),
            outputs=loss_mean,
            updates=optimizer.minimize(loss),
            givens={
                dropout1.keep_prob: 0.5
            }
        )
        self._add_validate_slot(
            inputs=(x, label_),
            outputs=(loss_sum, correct * 100),
            givens={
                dropout1.keep_prob: 1.0
            }
        )
        self._add_predict_slot(
            inputs=(x,),
            outputs=label,
            givens={
                dropout1.keep_prob: 1.0
            }
        )


def main(args):
    train_ds = ph.MongoSource(
        '172.16.46.203', 'admin', 'root', 'SELECT * FROM users;',
        'ayasa', 'cifar_10_train',
        {},
        [('x', utils.pickle_loads, utils.array_to_mat, utils.default_augmentation_filter()),
         ('y', lambda a: utils.one_hot(a, 10))],
        args.batch_size * 100
    )
    valid_ds = ph.MongoSource(
        '172.16.46.203', 'admin', 'root', 'SELECT * FROM users;',
        'ayasa', 'cifar_10_test',
        {},
        [('x', utils.pickle_loads, utils.array_to_mat),
         ('y', lambda a: utils.one_hot(a, 10))],
        args.batch_size * 100
    )
    #
    model = Cifar10()
    model.add_data_trainer(train_ds, args.batch_size)
    model.add_screen_logger(ph.CONTEXT_TRAIN, ('Loss',), interval=10)
    model.add_data_validator(valid_ds, 100, interval=500)
    model.add_screen_logger(ph.CONTEXT_VALID, ('Loss', 'Acc'), interval=500)
    model.initialize_global_variables()
    model.fit(args.max_loops)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    parser.add_argument('-b', '--batch-size', default=32, help='Batch size.', type=int)
    parser.add_argument('-l', '--max-loops', default=1000000, help='Max fit loops.', type=int)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
