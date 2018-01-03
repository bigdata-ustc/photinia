#!/usr/bin/env python3

"""
@author: xi
@since: 2017-11-11
"""
import argparse
import os

import numpy as np
import tensorflow as tf

import photinia as ph


class Word2vec(ph.Trainer):
    """"""

    def __init__(self,
                 max_len,
                 voc_size,
                 emb_size,
                 session=None):
        self._max_len = max_len
        self._voc_size = voc_size
        self._emb_size = emb_size
        super(Word2vec, self).__init__('Word2vec', session)

    def _build(self):
        self._emb = ph.Linear(
            'Emb',
            self._voc_size,
            self._emb_size,
            with_bias=False,
            # weight_initializer=ph.TruncatedNormal(stddev=0.0001)
        )
        self._cell = ph.GRUCell('Cell', self._emb_size, self._emb_size)
        self._out = ph.Linear(
            'Out',
            self._emb_size,
            self._voc_size,
            # weight_initializer=ph.TruncatedNormal(stddev=0.0001)
        )
        seq = tf.placeholder(
            shape=(None, self._max_len),
            dtype=tf.int32
        )
        seq1 = tf.one_hot(
            indices=seq,
            depth=self._voc_size,
            dtype=ph.D_TYPE
        )
        seq1 = ph.transpose_sequence(seq1)
        seq2 = seq1[1:]
        seq1 = seq1[:-1]
        states = ph.setup_sequence(
            seq1,
            [self._emb],
            cell=self._cell
        )
        outputs = ph.setup_sequence(
            states,
            [self._out,
             ph.lrelu,
             tf.nn.softmax]
        )
        loss = tf.reduce_mean(-tf.log(1e-5 + tf.reduce_sum(seq2 * outputs, 2)), 0)
        loss = tf.reduce_mean(loss)
        optimizer = tf.train.MomentumOptimizer(0.1, 0.9)
        optimizer = ph.GradientClipping(optimizer, 1.0)
        self._add_slot(
            'train',
            inputs=seq,
            outputs=(loss, tf.norm(self._out.w)),
            updates=optimizer.minimize(loss)
        )
        x = tf.placeholder(
            shape=(None,),
            dtype=tf.int32
        )
        y = tf.map_fn(
            fn=lambda elem: self._emb.w[elem],
            elems=x,
            dtype=ph.D_TYPE
        )
        self._add_slot(
            'predict',
            inputs=x,
            outputs=y
        )


class DataSource(ph.DataSource):
    def __init__(self,
                 word_table_file='word_table.txt',
                 sentences_file='sentences.txt',
                 max_len=20):
        with open(word_table_file, 'r') as f:
            lines = f.readlines()
        word_map = {'': 0}
        word_map1 = {0: ''}
        self._word_map = word_map
        self._word_map1 = word_map1
        for i, line in enumerate(lines):
            word = line.split(' ')[0]
            word_map[word] = i + 1
            word_map1[i + 1] = word
        self._voc_size = len(word_map)
        self._max_len = max_len
        #
        with open(sentences_file, 'r') as f:
            lines = f.readlines()
        sentences = []
        for i, line in enumerate(lines):
            ph.print_progress(i + 1, len(lines), 'Loading sentence', 10000)
            line = line.strip()
            sentence = [word_map[word] for word in line.split(' ') if word in word_map]
            if len(sentence) < max_len:
                sentence += [0 for _ in range(max_len - len(sentence))]
            elif len(sentence) > max_len:
                sentence = sentence[:max_len]
            sentence[-1] = 0
            sentences.append(np.array(sentence, dtype=np.int32))
        self._dataset = ph.Dataset(sentences, dtype=np.int32)

    @property
    def voc_size(self):
        return self._voc_size

    @property
    def max_len(self):
        return self._max_len

    @property
    def word_map(self):
        return self._word_map

    def word_to_index(self, word):
        return self._word_map[word] if word in self._word_map else -1

    def index_to_word(self, index):
        return self._word_map1[index] if index in self._word_map1 else None

    def next_batch(self, size=0):
        return self._dataset.next_batch(size)


def main(args):
    # dumper = ph.TreeDumper('.')
    ds = DataSource()
    model = Word2vec(ds.max_len, ds.voc_size, 100)
    model.add_data_trainer(ds, args.batch_size)
    model.add_screen_logger(ph.CONTEXT_TRAIN, ('Loss', 'WeightNorm'))
    model.fit(100000)
    # with tf.Session() as session:
    #     model = Word2vec(ds.max_len, ds.voc_size, 100, session)
    #     session.run(tf.global_variables_initializer())
    #     train = model.get_slot('train')
    #     predict = model.get_slot('predict')
    #     for i in range(10000):
    #         data_batch = ds.next_batch(32)
    #         loss = train(*data_batch)
    #         print(i, loss)
    #     dumper.dump('model-1000-64', model)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    parser.add_argument('-bs', '--batch-size', default=32, help='Batch size.', type=int)
    parser.add_argument('--input', help='Input.')
    parser.add_argument('--output', default='.', help='Output.')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
