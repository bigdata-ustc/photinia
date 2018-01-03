#!/usr/bin/env python3

"""
@author: xi
@since: 2017-11-11
"""

import collections
import os
import re
import sys

import gflags
from photinia.utils import utils


def main(flags):
    if flags.input is None:
        print('No input file is given.', file=sys.stderr)
        return 1
    print('Reading file...')
    with open(flags.input, 'r') as f:
        lines = f.readlines()
    print('Parsing sentences...')
    pattern = re.compile('|'.join([
        r'([a-z_0-9]+)',
        r'([\,，\;；\.])'
    ]))
    counter = collections.defaultdict(int)
    with open(os.path.join(flags.output, 'sentences.txt'), 'w') as f:
        for i, line in enumerate(lines):
            utils.print_progress(i + 1, len(lines), 'Processing sentence', 10000)
            line = line.lower()
            sentence = [i[0] + i[1] for i in pattern.findall(line)]
            line = ' '.join(sentence)
            f.write(line)
            f.write('\n')
            for word in sentence:
                counter[word] += 1
    print('Writing word table...')
    allowed_words = set()
    with open(os.path.join(flags.output, 'word_table.txt'), 'w') as f:
        for word, count in counter.items():
            if count < flags.count:
                continue
            allowed_words.add(word)
            f.write(word)
            f.write(' ')
            f.write(str(count))
            f.write('\n')
    print('Complete.')
    return 0


if __name__ == '__main__':
    global_flags = gflags.FLAGS
    gflags.DEFINE_boolean('help', False, 'Show this help.')
    gflags.DEFINE_string('input', None, 'Input file. Should be a text file.')
    gflags.DEFINE_string('output', '.', 'Output director')
    gflags.DEFINE_integer('count', 2, 'Min count of word.')
    #
    global_flags(sys.argv)
    if global_flags.help:
        print(global_flags.main_module_help())
        exit(0)
    exit(main(global_flags))
