#!/usr/bin/env python3

"""
@author: xi
@since: 2017-12-25
"""

import tensorflow as tf

import photinia as ph


class Probe(object):

    def __init__(self):
        self._record_list = list()

    def __call__(self, x):
        x_value = tf.Variable(
            initial_value=ph.Ones().build(shape=tf.shape(x)),
            dtype=ph.D_TYPE,
            trainable=False
        )
        update = tf.assign(x_value, x)
        self._record_list.append((x, update))
        return x


if __name__ == '__main__':
    exit()
