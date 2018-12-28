#!/usr/bin/env python3

"""
@author: xi
@since: 2018-12-22
"""

import argparse
import os

import tensorflow as tf
from tensorflow.examples.tutorials import mnist

import photinia as ph


class Model(ph.Model):

    def __init__(self, name, hidden_size):
        self._hidden_size = hidden_size
        super(Model, self).__init__(name)

    def _build(self):
        input_image = tf.placeholder(
            shape=(None, 784),
            dtype=tf.float32,
            name='input_image'
        )
        hidden_layer = ph.Linear('hidden_layer', 784, self._hidden_size)
        output_layer = ph.Linear('output_layer', self._hidden_size, 10)
        y = ph.setup(
            input_image, [
                hidden_layer, ph.ops.lrelu,
                output_layer, tf.nn.softmax
            ]
        )
        label = tf.argmax(y, 1)
        input_label = tf.placeholder(
            shape=(None,),
            dtype=tf.int64,
            name='input_label'
        )
        y_ = tf.one_hot(input_label, 10, dtype=tf.float32)
        loss = ph.ops.cross_entropy(y_, y)
        loss = tf.reduce_mean(loss)

        self.train = ph.Step(
            inputs=(input_image, input_label),
            outputs=loss,
            updates=tf.train.RMSPropOptimizer(1e-4, 0.9, 0.9).minimize(loss)
        )
        self.predict = ph.Step(
            inputs=input_image,
            outputs=label
        )


class DataSource(ph.io.MemorySource):

    def __init__(self, mnist_data):
        super(DataSource, self).__init__(
            ['image', 'label'], (
                {'image': image, 'label': label}
                for image, label in zip(mnist_data.images, mnist_data.labels)
            )
        )


class Main(ph.Application):

    def _main(self, args):
        mnist_data = mnist.input_data.read_data_sets('.', one_hot=False)
        ds_train = ph.io.BatchSource(DataSource(mnist_data.train), args.batch_size)
        ds_valid = ph.io.BatchSource(DataSource(mnist_data.validation), args.batch_size)
        ds_test = ph.io.BatchSource(DataSource(mnist_data.test), args.batch_size)

        model = Model('mnist_mlp', 1000)
        ph.initialize_global_variables()

        for i in range(1, args.num_loops + 1):
            self.checkpoint()
            try:
                image, label = ds_train.next()
            except StopIteration:
                image, label = ds_train.next()
            loss, = model.train(image, label)
            if i % 100 == 0:
                print(f'Training [{i}/{args.num_loops}|{i / args.num_loops * 100:.02f}%]... loss={loss:.06f}')

            if i % 500 == 0:
                acc = ph.train.AccCalculator()
                for image, label in ds_valid:
                    label_pred, = model.predict(image)
                    acc.update(label_pred, label)
                print(f'Validation acc={acc.accuracy * 100}%')
                acc = ph.train.AccCalculator()
                for image, label in ds_test:
                    label_pred, = model.predict(image)
                    acc.update(label_pred, label)
                print(f'Test acc={acc.accuracy * 100}%')
        return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    _parser.add_argument('--batch-size', type=int, default=32)
    _parser.add_argument('--num-loops', type=int, default=10000)
    #
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(Main().run(_args))
