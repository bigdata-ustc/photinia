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
            weight_initializer=ph.RandomNormal(stddev=1e-5)
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
        activation = ph.swish
        emb = ImageEmbedding('emb', 32, 32, 3, 500)
        hid = ph.Linear('hid', emb.output_size, emb.output_size, weight_initializer=ph.RandomNormal(stddev=1e-5))
        out = ph.Linear('out', emb.output_size, 10, weight_initializer=ph.RandomNormal(stddev=1e-5))
        dropout1 = ph.Dropout('dropout')
        #
        x1 = tf.placeholder(
            dtype=ph.D_TYPE,
            shape=(None, 32, 32, 3)
        )
        v1 = emb.setup(x1, activation=activation, dropout=dropout1)
        y1 = ph.setup(
            v1,
            [dropout1,
             hid, activation, dropout1,
             out]
        )
        label1 = tf.argmax(y1, 1)
        label1_ = tf.placeholder(
            shape=(None, 10),
            dtype=ph.D_TYPE
        )
        #
        x2 = tf.placeholder(
            dtype=ph.D_TYPE,
            shape=(None, 32, 32, 3)
        )
        v2 = emb.setup(x2, activation=activation, dropout=dropout1)
        y2 = ph.setup(
            v2,
            [dropout1,
             hid, activation, dropout1,
             out]
        )
        label2_ = tf.placeholder(
            shape=(None, 10),
            dtype=ph.D_TYPE
        )
        v1_ = tf.nn.l2_normalize(v1, 1)
        v2_ = tf.nn.l2_normalize(v2, 1)
        loss = (1 - 2 * tf.reduce_sum((label1_ * label2_), 1)) * tf.reduce_sum(v1_ * v2_, 1)
        loss += tf.reduce_sum((y1 - label1_) ** 2, 1)
        loss += tf.reduce_sum((y2 - label2_) ** 2, 1)
        loss_mean = tf.reduce_mean(loss)
        #
        loss_sum = tf.reduce_sum(tf.reduce_sum((y1 - label1_) ** 2, 1))
        correct = tf.equal(label1, tf.argmax(label1_, 1))
        correct = tf.reduce_sum(tf.cast(correct, ph.D_TYPE))
        #
        # Slots
        optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.6, beta2=0.9, epsilon=1e-5)
        self._add_train_slot(
            inputs=(x1, label1_, x2, label2_),
            outputs=loss_mean,
            updates=optimizer.minimize(loss_mean),
            givens={
                dropout1.keep_prob: 0.5
            }
        )
        self._add_validate_slot(
            inputs=(x1, label1_),
            outputs=(loss_sum, correct * 100),
            givens={
                dropout1.keep_prob: 1.0
            }
        )


class TrainSource(ph.DataSource):

    def __init__(self):
        with open('/home/xi/Documents/cifar-10/train_x.pickle', 'rb') as f:
            train_x = pickle.load(f)
        mean = np.mean(train_x)
        var = np.var(train_x)
        train_x = (train_x - mean) / np.sqrt(var)
        with open('/home/xi/Documents/cifar-10/train_y.pickle', 'rb') as f:
            train_y = pickle.load(f)
        self._ds = im_utils.AugmentedImageSource(ph.Dataset(train_x, train_y, dtype=np.float32))

    def next_batch(self, size=0):
        x1, y1 = self._ds.next_batch(size)
        x2, y2 = self._ds.next_batch(size)
        return x1, y1, x2, y2


class ValidSource(ph.DataSource):

    def __init__(self):
        with open('/home/xi/Documents/cifar-10/test_x.pickle', 'rb') as f:
            test_x = pickle.load(f)
        mean = np.mean(test_x)
        var = np.var(test_x)
        test_x = (test_x - mean) / np.sqrt(var)
        with open('/home/xi/Documents/cifar-10/test_y.pickle', 'rb') as f:
            test_y = pickle.load(f)
        self._ds = ph.Dataset(test_x, test_y, dtype=np.float32)

    def next_batch(self, size=0):
        return self._ds.next_batch(size)


class Dumper(ph.Fitter):

    def __init__(self, model_dir, interval):
        super(Dumper, self).__init__(interval, 1)
        self._dumper = ph.TreeDumper(model_dir)

    def fit(self, i, max_loops, context):
        trainer = context[ph.CONTEXT_TRAINER]
        self._dumper.dump(str(context[ph.CONTEXT_LOOP]), trainer)


def main(args):
    train_ds = TrainSource()
    valid_ds = ValidSource()
    #
    trainer = Cifar10()
    trainer.add_data_trainer(train_ds, args.batch_size)
    trainer.add_screen_logger(ph.CONTEXT_TRAIN, ('Loss',), interval=10)
    trainer.add_data_validator(valid_ds, 100, interval=500)
    trainer.add_screen_logger(ph.CONTEXT_VALID, ('Loss', 'Acc'), interval=500)
    trainer.add_fitter(Dumper(args.model_dir, interval=10000))
    trainer.initialize_global_variables()
    trainer.fit(args.max_loops)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    parser.add_argument('-b', '--batch-size', default=32, help='Batch size.', type=int)
    parser.add_argument('-l', '--max-loops', default=1000000, help='Max fit loops.', type=int)
    parser.add_argument('-d', '--model-dir', default='.', help='Dir to save he model.')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
