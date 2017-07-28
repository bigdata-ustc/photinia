#!/usr/bin/env python3

"""
@author: xi, anmx
@since: 2017-04-23
"""


def print_values(names, values, fmt=':<15.6f'):
    pairs = []
    for name, value in zip(names, values):
        pairs.append(('{}={' + fmt + '}').format(name, value))
    print('\t'.join(pairs))
