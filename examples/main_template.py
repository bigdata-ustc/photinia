#!/usr/bin/env python3

"""
@author: xi
"""

import argparse
import os


#
# TODO: Model definition here.

#
# TODO: Data source definition here.

def main(args):
    #
    # TODO: Any code here.
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    #
    # TODO: Define more args here.
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(main(_args))
