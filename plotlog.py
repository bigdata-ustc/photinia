#!/usr/bin/env python3

import argparse
import os
import re

from matplotlib import pyplot as plt


def main(args):
    plt.figure(figsize=(args.width, args.height))
    pattern_x = re.compile(r'(\d+)/\d+\|')
    pattern_y = re.compile(r'%s\s*=\s*([.\d]+)' % args.column)
    plt.title('Performances on %s' % args.column)
    plt.xlabel('Number of Loops')
    plt.ylabel(args.column)
    for file_ in args.log_files:
        x_keys = set()
        x_list = []
        y_list = []
        i = 1
        with open(file_, 'rt') as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                x = pattern_x.findall(line)
                y = pattern_y.findall(line)
                if len(x) > 0:
                    x = int(x[0])
                else:
                    x = i
                    i += 1
                if len(y) > 0:
                    y = float(y[0])
                    if x in x_keys:
                        continue
                    x_keys.add(x)
                    x_list.append(x)
                    y_list.append(y)
        label = os.path.basename(file_)
        plt.plot(x_list, y_list, label=label, alpha=args.alpha)
    plt.legend()
    plt.grid()
    plt.show()
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('log_files', help='Log file path.', nargs='+')
    _parser.add_argument('--height', help='Figure height.', default=4.5, type=float)
    _parser.add_argument('--width', help='Figure width.', default=8, type=float)
    _parser.add_argument('-c', '--column', help='Name of the column to plot.')
    _parser.add_argument('--alpha', default=1.0, type=float)
    _args = _parser.parse_args()
    exit(main(_args))
