#!/usr/bin/env python3

"""
@author: xi
@since: 2017-03-29
"""

import collections
import os
import sys

import gflags
import numpy as np
import pymongo
import tensorflow as tf

import photinia
import pickle


class CharEmb(photinia.Trainable):
    """Character embedding
    """

    def __init__(self,
                 name,
                 session,
                 voc_size,
                 emb_size,
                 state_size):
        photinia.Trainable.__init__(self, name, session)
        #
        self._voc_size = voc_size
        self._emb_size = emb_size
        self._state_size = state_size

    def _build(self):
        seq = tf.placeholder(
            shape=(None, None, self._voc_size),
            dtype=photinia.D_TYPE
        )
        seq_0 = seq[:, :-1, :]
        seq_1 = seq[:, 1:, :]
        self._emb = photinia.Linear('EMB', self._voc_size, self._emb_size).build()
        self._cell = photinia.GRUCell('CELL', self._emb_size, self._state_size).build()
        self._lin = photinia.Linear('LIN', self._state_size, self._voc_size).build()
        batch_size = tf.shape(seq)[0]
        init_state = tf.zeros(
            shape=(batch_size, self._state_size),
            dtype=photinia.D_TYPE
        )
        states = tf.scan(
            fn=self._rnn_step,
            elems=tf.transpose(seq_0, (1, 0, 2)),
            initializer=init_state
        )
        probs = tf.map_fn(
            fn=self._state_to_prob,
            elems=states
        )
        outputs = tf.map_fn(
            fn=self._prob_to_output,
            elems=probs
        )
        probs = tf.transpose(probs, (1, 0, 2))
        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.concat((seq[:, 0:1, :], outputs), 1)
        loss = tf.reduce_mean(-tf.log(1e-5 + tf.reduce_sum(seq_1 * probs, 2)), 1)
        loss = tf.reduce_mean(loss)
        self._add_slot(
            'train',
            outputs=loss,
            inputs=seq,
            updates=tf.train.AdamOptimizer(1e-3).minimize(loss)
        )
        self._add_slot(
            'evaluate',
            outputs=outputs,
            inputs=seq
        )
        #
        #
        char = tf.placeholder(
            shape=(None, self._voc_size),
            dtype=photinia.D_TYPE
        )
        emb = self._emb.setup(char)
        emb = photinia.lrelu(emb)
        self._add_slot(
            'embedding',
            outputs=emb,
            inputs=char
        )

    def _rnn_step(self, acc, elem):
        emb = self._emb.setup(elem)
        emb = photinia.lrelu(emb)
        state = self._cell.setup(emb, acc)
        return state

    def _state_to_prob(self, state):
        prob = self._lin.setup(state)
        prob = photinia.lrelu(prob)
        prob = tf.nn.softmax(prob)
        return prob

    def _prob_to_output(self, prob):
        return tf.one_hot(tf.arg_max(prob, 1), self._voc_size)


class DanmuDataSource(photinia.DataSource):
    """Danmu data source.
    """

    def __init__(self,
                 host,
                 db,
                 coll,
                 min_len,
                 max_len,
                 max_num):
        text_list = []
        with pymongo.MongoClient(host) as conn:
            db = conn[db]
            coll = db[coll]
            query = coll.find({}, {'text': 1}).limit(max_num)
            for danmu in query:
                text = danmu['text']
                if len(text) < min_len or len(text) > max_len:
                    continue
                text_list.append(text + '\n')
        #
        self._init_groups(text_list)
        self._init_encoder(text_list)
        for key, value in self._groups.items():
            print(key, len(value))
        #
        # Dataset.
        groups = {}
        for key, value in self._groups.items():
            groups[key] = photinia.Dataset(value)
        self._groups = groups

    def _init_groups(self, lines):
        groups = collections.defaultdict(list)
        for text in lines:
            length = len(text)
            groups[length].append(text)
        self._groups = groups

    def _init_encoder(self, lines):
        chars = set()
        for text in lines:
            chars.update(text)
        chars = list(chars)
        self._ctoi = {ch: i for i, ch in enumerate(chars)}
        self._itoc = chars

    @property
    def voc_size(self):
        return len(self._itoc)

    @property
    def voc(self):
        return self._itoc

    def encode(self, text):
        length = len(text)
        mat = np.zeros(shape=(length, self.voc_size), dtype=np.float32)
        for i, ch in enumerate(text):
            mat[i][self._ctoi[ch]] = 1.
        return mat

    def decode(self, mat):
        text = []
        for i in range(len(mat)):
            row = mat[i]
            index = np.argmax(row)
            ch = self._itoc[index]
            if ch == '\n':
                break
            text.append(ch)
        return ''.join(text)

    def next_batch(self, size=0):
        key = np.random.choice(list(self._groups.keys()))
        batch, = self._groups[key].next_batch(size)
        return np.array([self.encode(text) for text in batch], dtype=np.float32),


def main(flags):
    ds = DanmuDataSource(
        flags.host,
        flags.db,
        flags.coll,
        flags.min_len,
        flags.max_len,
        flags.max_num
    )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        model = CharEmb(
            'CharEmb',
            session,
            ds.voc_size,
            flags.emb_size,
            flags.state_size
        ).build()
        train = model.get_slot('train')
        evaluate = model.get_slot('evaluate')
        embedding = model.get_slot('embedding')
        session.run(tf.global_variables_initializer())
        for i in range(1, flags.nloop + 1):
            seq, = ds.next_batch(flags.bsize)
            loss = train(seq)
            print('loop={}\tloss={}'.format(i, loss))
            if i % 50 == 0:
                outputs = evaluate(seq)
                for original, output in zip(seq, outputs):
                    text0 = ds.decode(original)
                    text = ds.decode(output)
                    print('Original: {}'.format(text0))
                    print('  Output: {}'.format(text))
                    print()
        voc = ''.join(ds.voc)
        embs = embedding(ds.encode(voc))
        with pymongo.MongoClient(flags.host) as conn:
            coll = conn[flags.db][flags.out_coll]
            coll.remove()
            for char, emb in zip(voc, embs):
                coll.insert_one({
                    'char': char,
                    'emb': pickle.dumps(emb)
                })
    return 0


if __name__ == '__main__':
    global_flags = gflags.FLAGS
    gflags.DEFINE_boolean('help', False, 'Show this help.')
    gflags.DEFINE_string('gpu', '0', 'Which GPU to use.')
    gflags.DEFINE_integer('bsize', 100, 'Batch size.')
    gflags.DEFINE_integer('nloop', 20000, 'Max number of loop.')
    gflags.DEFINE_string('host', 'localhost', 'MongoDB host.')
    gflags.DEFINE_string('db', 'march', 'Database name.')
    gflags.DEFINE_string('coll', 'danmus', 'Collection name.')
    gflags.DEFINE_string('out_coll', 'embeddings', 'Output collection name.')
    gflags.DEFINE_integer('min_len', 5, 'Min length.')
    gflags.DEFINE_integer('max_len', 15, 'Max length.')
    gflags.DEFINE_integer('max_num', 500000, 'Max number of danmus.')
    gflags.DEFINE_integer('emb_size', 100, 'Embedding size.')
    gflags.DEFINE_integer('state_size', 1000, 'State size.')
    global_flags(sys.argv)
    if global_flags.help:
        print(global_flags.main_module_help())
        exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = global_flags.gpu
    exit(main(global_flags))
