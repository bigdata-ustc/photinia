#!/usr/bin/env python3

"""
@author: xi
"""

import os
import sys

import gflags
import tensorflow as tf


#
# TODO: Model definition here.

#
# TODO: Data source definition here.

def main(flags):
    #
    # TODO: Any code here.
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
    #
    # TODO: Other FLAGS here.
    global_flags(sys.argv)
    if global_flags.help:
        print(global_flags.main_module_help())
        exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = global_flags.gpu
    exit(main(global_flags))
