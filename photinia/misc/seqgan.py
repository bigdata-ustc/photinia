#!/usr/bin/env python3

"""
@author: xi
@since: 2018-06-07
"""

import argparse

import tensorflow as tf

import photinia as ph


class D(ph.Widget):

    def __init__(self, name, max_len, voc_size, hidden_size):
        self._max_len = max_len
        self._voc_size = voc_size
        self._hidden_size = hidden_size
        super(D, self).__init__(name)

    def _build(self):
        height, width, channels = (self._max_len, self._voc_size, 1)
        output_channels = 8

        self._conv1 = ph.Conv2D(
            'conv1',
            input_size=(height, width, channels),
            output_channels=output_channels,
            filter_height=3, filter_width=width,
            stride_height=1, stride_width=1
        )
        self._pool1 = ph.Pool2D(
            'pool1',
            input_size=(height, width, channels),
            filter_height=3, filter_width=3,
            stride_height=2, stride_width=2,
            pool_type='avg'
        )
        height, width, channels = self._pool1.output_size
        output_channels *= 2

        self._conv2 = ph.Conv2D(
            'conv2',
            input_size=(height, width, channels),
            output_channels=output_channels,
            filter_height=3, filter_width=width,
            stride_height=1, stride_width=1
        )
        self._pool2 = ph.Pool2D(
            'pool2',
            input_size=(height, width, channels),
            filter_height=3, filter_width=3,
            stride_height=2, stride_width=2,
            pool_type='avg'
        )
        height, width, channels = self._pool2.output_size
        output_channels *= 2

        self._conv3 = ph.Conv2D(
            'conv3',
            input_size=(height, width, channels),
            output_channels=output_channels,
            filter_height=3, filter_width=width,
            stride_height=1, stride_width=1
        )
        self._pool3 = ph.Pool2D(
            'pool3',
            input_size=(height, width, channels),
            filter_height=3, filter_width=3,
            stride_height=2, stride_width=2,
            pool_type='avg'
        )

        self._dense1 = ph.Linear(
            'dense1',
            input_size=self._pool3.flat_size,
            output_size=self._hidden_size
        )

        self._dense2 = ph.Linear(
            'dense2',
            input_size=self._hidden_size,
            output_size=1
        )

    def _setup(self, x):
        batch_size, seq_len, voc_size = tf.shape(x)
        y = ph.setup(
            tf.reshape(x, (batch_size, seq_len, voc_size, 1)), [
                self._conv1, self._pool1, ph.ops.lrelu,
                self._conv2, self._pool2, ph.ops.lrelu,
                self._conv3, self._pool3, ph.ops.lrelu,
                ph.ops.flatten,
                self._dense1, ph.ops.lrelu,
                self._dense2, tf.nn.sigmoid,
            ]
        )
        return y


def main(args):
    #
    # TODO: Any code here.
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    #
    # TODO: Define more args here.
    _args = _parser.parse_args()
    exit(main(_args))
