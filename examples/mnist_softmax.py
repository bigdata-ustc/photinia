#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A very simple MNIST classifier.

@author: winton 
@time: 2017-10-25 11:22 
"""
import os
import sys

import gflags
import tensorflow as tf


#
# TODO: Model definition here.

class Data
from tensorflow.examples.tutorials.mnist import input_data


def main(flags):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        #
        # TODO: Any code here.
        pass
    return 0


if __name__ == '__main__':
    global_flags = gflags.FLAGS
    gflags.DEFINE_boolean('help', False, 'Show this help.')
    gflags.DEFINE_string('gpu', '0', 'Which GPU to use.')
    global_flags(sys.argv)
    if global_flags.help:
        print(global_flags.main_module_help())
        exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = global_flags.gpu
    exit(main(global_flags))