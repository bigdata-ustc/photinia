#!/usr/bin/env python3


import argparse
import os

import photinia as ph
from photinia import utils
import tensorflow as tf

"""
@author: xi
@since: 2017-12-20
"""


class Wrapper(ph.OptimizerWrapper):

    def __init__(self, opt):
        super(Wrapper, self).__init__(opt)
        self._mask1 = tf.Variable(
            initial_value=np.zeros((2000,), dtype=np.float32),
            dtype=tf.float32,
            trainable=False
        )
        self._mask2 = tf.Variable(
            initial_value=np.zeros((2000,), dtype=np.float32),
            dtype=tf.float32,
            trainable=False
        )

    @property
    def mask1(self):
        return self._mask1

    @property
    def mask2(self):
        return self._mask2

    def _process_gradients(self, pair_list):
        new_list = list()
        for grad, var in pair_list:
            if var.name.find('h0') != -1:
                grad = grad * self._mask1
            elif var.name.find('h1') != -1:
                grad = grad * self._mask2
            new_list.append((grad, var))
        return new_list


class DNN1(ph.Trainer):

    def __init__(self,
                 height,
                 width, session):
        self._height = height
        self._width = width
        super(DNN1, self).__init__('DNN1', session)

    def _build(self):
        layers = list()
        self._layers = layers
        layers.append(ph.Linear('h%d' % len(layers), self._height * self._width, 2000))
        layers.append(ph.Linear('h%d' % len(layers), layers[-1].output_size, 2000))
        layers.append(ph.Linear('h%d' % len(layers), layers[-1].output_size, 5))
        drop = ph.Dropout('drop')
        #
        x = ph.placeholder('x', (None, self._height, self._width, 1))
        y1 = ph.setup(
            x,
            [ph.flatten, layers[0], ph.swish]
        )
        y2 = ph.setup(
            y1,
            [drop, layers[1], ph.swish]
        )
        y = ph.setup(
            y2,
            [drop, layers[2]]
        )
        label = tf.argmax(y, 1)
        label_ = ph.placeholder('label_', (None, 5))
        loss = tf.reduce_sum((y - label_) ** 2, 1)
        loss_sum = tf.reduce_sum(loss)
        loss_mean = tf.reduce_mean(loss)
        error = tf.equal(label, tf.argmax(label_, 1))
        error = tf.reduce_sum(1 - tf.cast(error, ph.D_TYPE))
        #
        self._add_train_slot(
            inputs=(x, label_),
            outputs=(loss_mean, y1, y2),
            updates=tf.train.AdamOptimizer().minimize(loss_mean),
            givens={drop.keep_prob: 1.0}
        )
        self._add_validate_slot(
            inputs=(x, label_),
            outputs=(loss_sum, error * 100),
            givens={drop.keep_prob: 1.0}
        )

    @property
    def layers(self):
        return self._layers


class DNN2(ph.Trainer):

    def __init__(self,
                 dnn1,
                 height,
                 width,
                 session):
        self._dnn1 = dnn1
        self._height = height
        self._width = width
        super(DNN2, self).__init__('DNN2', session)

    def _build(self):
        layers = list()
        layers.append(self._dnn1.layers[0])
        layers.append(self._dnn1.layers[1])
        layers.append(ph.Linear('h%d' % len(layers), layers[-1].output_size, 5))
        drop = ph.Dropout('drop')
        #
        x = ph.placeholder('x', (None, self._height, self._width, 1))
        y1 = ph.setup(
            x,
            [ph.flatten, layers[0], ph.swish]
        )
        y2 = ph.setup(
            y1,
            [drop, layers[1], ph.swish]
        )
        y = ph.setup(
            y2,
            [drop, layers[2]]
        )
        label = tf.argmax(y, 1)
        label_ = ph.placeholder('label_', (None, 5))
        loss = tf.reduce_sum((y - label_) ** 2, 1)
        loss_sum = tf.reduce_sum(loss)
        loss_mean = tf.reduce_mean(loss)
        error = tf.equal(label, tf.argmax(label_, 1))
        error = tf.reduce_sum(1 - tf.cast(error, ph.D_TYPE))
        #
        self.wrapper = Wrapper(tf.train.AdamOptimizer())
        self._add_train_slot(
            inputs=(x, label_),
            outputs=(loss_mean, y1, y2),
            updates=self.wrapper.minimize(loss_mean),
            givens={drop.keep_prob: 1.0}
        )
        self._add_validate_slot(
            inputs=(x, label_),
            outputs=(loss_sum, error * 100),
            givens={drop.keep_prob: 1.0}
        )


import numpy as np
import collections


class Probe(ph.Fitter):

    def __init__(self):
        super(Probe, self).__init__()
        self._y1 = 0
        self._y2 = 0

    def _fit(self, i, max_loop, context):
        v1, v2 = context[ph.CONTEXT_TRAIN][1:]
        value = np.abs(v1)
        value = np.sum(value, axis=0, keepdims=False)
        self._y1 += value
        value = np.abs(v2)
        value = np.sum(value, axis=0, keepdims=False)
        self._y2 += value

    @property
    def y1(self):
        b = np.max(self._y1)
        a = np.min(self._y1)
        c = 1 - (self._y1 - a) / (b - a)
        count = 0
        for i, e in enumerate(c):
            if e < 0.965:
                c[i] = 0.0
                count += 1
        print(count / len(c))
        return c

    @property
    def y2(self):
        b = np.max(self._y2)
        a = np.min(self._y2)
        c = 1 - (self._y2 - a) / (b - a)
        count = 0
        for i, e in enumerate(c):
            if e < 0.7:
                c[i] = 0.0
                count += 1
        print(count / len(c))
        return c


def main(args):
    height, width, channels = 28, 28, 1
    task1_dir = './task1'
    task1_train = im_utils.ImageSourceWithLabels(
        os.path.join(task1_dir, 'train'),
        height, width, channels, 5
    )
    task1_train = im_utils.AugmentedImageSource(task1_train)
    task1_valid = im_utils.ImageSourceWithLabels(
        os.path.join(task1_dir, 'test'),
        height, width, channels, 5
    )
    task2_dir = './task2'
    task2_train = im_utils.ImageSourceWithLabels(
        os.path.join(task2_dir, 'train'),
        height, width, channels, 5
    )
    task2_train = im_utils.AugmentedImageSource(task2_train)
    task2_valid = im_utils.ImageSourceWithLabels(
        os.path.join(task2_dir, 'test'),
        height, width, channels, 5
    )
    #
    with tf.Session() as session:
        p = Probe()
        trainer1 = DNN1(height, width, session)
        trainer2 = DNN2(trainer1, height, width, session)
        trainer1.initialize_global_variables()

        class Fitter(ph.Fitter):

            def _fit(self, i, max_loop, context):
                valid1 = trainer1.get_slot(ph.NAME_VALID_SLOT)
                valid2 = trainer2.get_slot(ph.NAME_VALID_SLOT)
                loss, error = valid1(*task1_valid.next_batch())
                print('task1 Loss=%f\tError=%f' % (loss / 10000, error / 10000))
                loss, error = valid2(*task2_valid.next_batch())
                print('task2 Loss=%f\tError=%f' % (loss / 10000, error / 10000))

        trainer1.add_data_trainer(task1_train, args.batch_size)
        trainer1.add_fitter(p)
        trainer1.add_screen_logger(ph.CONTEXT_TRAIN, ('Loss',), interval=10)
        trainer1.add_fitter(Fitter(200))
        trainer1.fit(args.max_loops)

        mask1 = trainer2.wrapper.mask1
        mask2 = trainer2.wrapper.mask2
        mask1.load(p.y1, session)
        mask2.load(p.y2, session)

        trainer2.add_data_trainer(task2_train, args.batch_size)
        trainer2.add_screen_logger(ph.CONTEXT_TRAIN, ('Loss',), interval=10)
        trainer2.add_fitter(Fitter(200))
        trainer2.fit(args.max_loops)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    parser.add_argument('-b', '--batch-size', default=32, help='Batch size.', type=int)
    parser.add_argument('-l', '--max-loops', default=3000, help='Max fit loops.', type=int)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
