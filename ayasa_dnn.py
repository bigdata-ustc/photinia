#!/usr/bin/env python3


"""
@author: xi
@since: 2017-12-25
"""

import argparse
import os

import tensorflow as tf

import photinia as ph
from photinia import utils


class Probe(object):

    def __init__(self):
        self._record_list = list()

    def __call__(self, x):
        x_value = tf.Variable(
            initial_value=ph.Ones().build(shape=tf.shape(x)),
            dtype=ph.D_TYPE,
            trainable=False
        )
        tf.Session().run()
        update = tf.assign(x_value, x)
        self._record_list.append((x, update))
        return x


class DNN(ph.Trainer):

    def __init__(self,
                 name,
                 height,
                 width,
                 channels):
        self._height = height
        self._width = width
        self._channels = channels
        super(DNN, self).__init__(name)

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def _build(self):
        layers = list()
        layers.append(ph.Linear(
            'h%d' % len(layers),
            self._height * self._width * self._channels,
            5000,
            weight_initializer=ph.RandomNormal(stddev=1e-4)
        ))
        layers.append(ph.Linear(
            'h%d' % len(layers),
            layers[-1].output_size,
            5000,
            weight_initializer=ph.RandomNormal(stddev=1e-4)
        ))
        layers.append(ph.Linear(
            'h%d' % len(layers),
            layers[-1].output_size,
            10,
            weight_initializer=ph.RandomNormal(stddev=1e-4)
        ))
        drop = ph.Dropout('drop')
        #
        x = ph.placeholder('x', (None, None, None, None))
        x_ = tf.image.resize_images(x, (self._height, self._width))
        y1 = ph.setup(
            x_,
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
        label_ = ph.placeholder('label_', (None, 10))
        loss = tf.reduce_sum((y - label_) ** 2, 1)
        loss_sum = tf.reduce_sum(loss)
        loss_mean = tf.reduce_mean(loss)
        error = tf.equal(label, tf.argmax(label_, 1))
        error = tf.reduce_sum(1 - tf.cast(error, ph.D_TYPE))
        #
        self._add_train_slot(
            inputs=(x, label_),
            outputs=(loss_mean, y1, y2),
            updates=tf.train.RMSPropOptimizer(1e-5, 0.9, 0.9).minimize(loss_mean),
            givens={drop.keep_prob: 0.5}
        )
        self._add_validate_slot(
            inputs=(x, label_),
            outputs=(loss_sum, error * 100),
            givens={drop.keep_prob: 1.0}
        )


import pickle


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
    trainer = DNN('mnist_dnn', 48, 48, 3)
    trainer.add_data_trainer(train_ds, args.batch_size)
    trainer.add_screen_logger(ph.CONTEXT_TRAIN, ('Loss',), interval=10)
    trainer.add_data_validator(valid_ds, 100, interval=500)
    trainer.add_screen_logger(ph.CONTEXT_VALID, ('Loss', 'Error'), interval=500)
    trainer.initialize_global_variables()
    trainer.fit(args.max_loops)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    parser.add_argument('-b', '--batch-size', default=32, help='Batch size.', type=int)
    parser.add_argument('-l', '--max-loops', default=5000, help='Max fit loops.', type=int)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
