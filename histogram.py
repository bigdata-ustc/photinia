#!/usr/bin/env python3

"""
@author: xi
@since: 2017-12-25
"""

import argparse
import os
import re

import numpy as np
from matplotlib import pyplot as plt


def plot_file(file_, pattern, args):
    x = []
    with open(file_, 'rt') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            line = line.strip()
            if line == '':
                continue
            values = pattern.findall(line)
            if values:
                value = float(values[0])
                x.append(value)
    if args.stat:
        mu, sigma = float(np.mean(x)), float(np.sqrt(np.var(x)))
        label = '%s, μ=%f, σ=%f' % (os.path.basename(file_), mu, sigma)
    else:
        label = os.path.basename(file_)
    plt.hist(
        x,
        label=label,
        bins=args.bins,
        histtype=args.histtype,
        rwidth=0.8,
        alpha=args.alpha
    )


def main(args):
    plt.figure(figsize=(args.width, args.height))
    pattern = re.compile(r'%s\s*=\s*([.\d]+)' % args.column)
    for file_ in args.files:
        plot_file(file_, pattern, args)
    plt.legend()
    plt.grid()
    plt.show()
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('files', help='File paths.', nargs='+')
    _parser.add_argument('--height', help='Figure height.', default=4.5, type=float)
    _parser.add_argument('--width', help='Figure width.', default=8, type=float)
    _parser.add_argument('-c', '--column', help='Name of the column to plot.')
    _parser.add_argument('--bins', help='Number of bins.', default=50, type=int)
    _parser.add_argument('--histtype', default='bar')
    _parser.add_argument('--rwidth', default=0.8, type=float)
    _parser.add_argument('--alpha', default=1.0, type=float)
    _parser.add_argument('-s', '--stat', default=False, action='store_true')
    _args = _parser.parse_args()
    exit(main(_args))
