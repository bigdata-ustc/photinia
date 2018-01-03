#!/usr/bin/env python3

"""
@author: xi
"""

import argparse
import os

import tensorflow as tf


#
# TODO: Model definition here.

#
# TODO: Data source definition here.

def main(args):
    #
    # TODO: Any code here.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        #
        # TODO: Any code here.
        pass
    raise SystemExit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    #
    # TODO: Define more args here.
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
