#!/usr/bin/env python3

"""
@author: xi
@since: 2017-03-29
"""

import collections
import os
import sys
from urllib import request

import gflags
import numpy as np
import tensorflow as tf

import photinia
import pickle


class Model(photinia.Trainer):
    """模型定义
    """

    def __init__(self,
                 name,
                 session,
                 voc_size,
                 emb_size,
                 state_size):
        """模型初始化

        :param name: 模型名
        :param session: 使用的tensorflow会话
        :param voc_size: 字典维度
        :param emb_size: 词embedding维度
        :param state_size: GRU单元隐藏单元维度
        """
        self._voc_size = voc_size
        self._emb_size = emb_size
        self._state_size = state_size
        photinia.Trainer.__init__(self, name, session)

    def _build(self):
        # 网络模块定义 --- build
        self._emb = photinia.Linear('EMB', self._voc_size, self._emb_size)
        self._cell = photinia.GRUCell('CELL', self._emb_size, self._state_size)
        self._lin = photinia.Linear('LIN', self._state_size, self._voc_size)
        # 输入定义
        seq = tf.placeholder(
            shape=(None, None, self._voc_size),
            dtype=photinia.D_TYPE
        )
        seq_0 = seq[:, :-1, :]
        seq_1 = seq[:, 1:, :]
        batch_size = tf.shape(seq)[0]
        # RNN结构
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
        word = tf.placeholder(
            shape=(None, self._voc_size),
            dtype=photinia.D_TYPE
        )
        emb = self._emb.setup(word)
        emb = photinia.lrelu(emb)
        self._add_slot(
            'embedding',
            outputs=emb,
            inputs=word
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


class PTBData(photinia.DataSource):
    """数据源定义
    """

    def __init__(self,
                 directory,
                 min_len,
                 max_len):
        self._dir = directory
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        train_name = 'ptb.train.txt'
        # valid_name = 'ptb.valid.txt'
        # test_name = 'ptb.test.txt'
        train_path = os.path.join(self._dir, train_name)
        train_list = self._get_text_list(train_path, min_len=min_len, max_len=max_len)
        # valid_list = self._get_text_list(valid_path)
        # test_list = self._get_text_list(test_path)
        self._init_groups(train_list)
        self._init_encoder(train_list)
        # Dataset.
        groups = {}
        for key, value in self._groups.items():
            groups[key] = photinia.Dataset(value)
        self._groups = groups

    @staticmethod
    def _get_text_list(filename, min_len=0, max_len=1000):
        text_list = []
        with open(filename, 'r') as f:
            temp = f.readlines()
        for text in temp:
            text = text.split()
            if min_len < len(text) < max_len:
                text.append('\n')
                text_list.append(text)
        return text_list

    def _init_groups(self, lines):
        groups = collections.defaultdict(list)
        for text in lines:
            length = len(text)
            groups[length].append(text)
        self._groups = groups

    def _init_encoder(self, lines):
        words = set()
        for text in lines:
            words.update(text)
        words = list(words)
        self._wtoi = {word: i for i, word in enumerate(words)}
        self._itow = words

    @property
    def voc_size(self):
        return len(self._itow)

    @property
    def voc(self):
        return self._itow

    def encode(self, text):
        length = len(text)
        mat = np.zeros(shape=(length, self.voc_size), dtype=np.float32)
        for i, word in enumerate(text):
            if word in self._wtoi:
                mat[i][self._wtoi[word]] = 1.
            else:
                mat[i][self._wtoi['<unk>']] = 1.
        return mat

    def decode(self, mat):
        text = []
        for i in range(len(mat)):
            row = mat[i]
            index = np.argmax(row)
            word = self._itow[index]
            if word == '\n':
                break
            text.append(word)
        return ' '.join(text)

    def next_batch(self, size=0):
        key = np.random.choice(list(self._groups.keys()))
        batch, = self._groups[key].next_batch(size)
        return np.array([self.encode(text) for text in batch], dtype=np.float32)


def main(flags):
    # 创建数据源对象
    ds = PTBData(
        flags.directory,
        flags.min_len,
        flags.max_len
    )
    # 创建存储词向量的文件夹
    if not os.path.exists(flags.save):
        os.makedirs(flags.save)
    # tensorflow 配置
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        # 创建模型对象
        model = Model(
            'Model',
            session,
            ds.voc_size,
            flags.emb_size,
            flags.state_size
        )
        # 获取slot
        train = model.get_slot('train')
        evaluate = model.get_slot('evaluate')
        embedding = model.get_slot('embedding')
        # 参数初始化
        session.run(tf.global_variables_initializer())
        # 开始训练
        for i in range(1, flags.nloop + 1):
            # 获取一个batch的数据
            seq = ds.next_batch(flags.bsize)
            loss = train(seq)
            # 输出损失函数值
            print('loop={}\tloss={}'.format(i, loss))
            # 输出结果对比原始输入与输出
            if i % 50 == 0:
                outputs = evaluate(seq)
                for original, output in zip(seq, outputs):
                    text0 = ds.decode(original)
                    text = ds.decode(output)
                    print('Original: {}'.format(text0))
                    print('  Output: {}'.format(text))
                    print()
            # 存储每个词的embedding到指定文件
            if i % flags.interval == 0:
                emb_dict = {}
                embs = embedding(ds.encode(ds.voc))
                for word, emb in zip(ds.voc, embs):
                    emb_dict[word] = emb
                with open(os.path.join(flags.save, 'embedding_' + str(i) + '.pkl'), 'wb') as f:
                    pickle.dump(emb_dict, f)
    return 0


if __name__ == '__main__':
    global_flags = gflags.FLAGS
    gflags.DEFINE_boolean('help', False, 'Show this help.')
    gflags.DEFINE_string('gpu', '0', 'Which GPU to use.')
    gflags.DEFINE_string('directory', './examples', 'Folder to save the origin data.')
    gflags.DEFINE_string('save', './examples/embedding', 'Folder to save the word\'s embedding.')
    gflags.DEFINE_integer('bsize', 50, 'Batch size.')
    gflags.DEFINE_integer('nloop', 20000, 'Max number of loop.')
    gflags.DEFINE_integer('min_len', 2, 'Min length.')
    gflags.DEFINE_integer('max_len', 40, 'Max length.')
    gflags.DEFINE_integer('emb_size', 100, 'Embedding size.')
    gflags.DEFINE_integer('state_size', 1000, 'State size.')
    gflags.DEFINE_integer('interval', 1000, 'Interval of save the word\'s embedding.')
    global_flags(sys.argv)
    if global_flags.help:
        print(global_flags.main_module_help())
        exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = global_flags.gpu
    exit(main(global_flags))
