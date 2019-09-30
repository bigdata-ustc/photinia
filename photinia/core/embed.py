#!/usr/bin/env python3


"""
@author: xi
@since: 2016-11-11
"""

from . import common
from . import conv
from .. import conf
from .. import init
from ..conf import tf


class Embedding(common.Trainable):

    def __init__(self,
                 name,
                 voc_size,
                 emb_size,
                 trainable=True,
                 w_init=init.GlorotNormal()):
        """Embedding.

        Args:
            name (str): The widget name.
            voc_size (int): The vocabulary size.
            emb_size (int): The embedding size.
            trainable (bool): Is the embedding matrix trainable?
            w_init (init.Initializer): The matrix initializer.

        """
        super(Embedding, self).__init__(name)
        self._voc_size = voc_size
        self._emb_size = emb_size
        self._trainable = trainable
        self._w_init = w_init
        self._w = None

    @property
    def emb_size(self):
        return self._emb_size

    @property
    def output_size(self):
        return self._emb_size

    @property
    def trainable(self):
        return self._trainable

    def setup(self, indexes, name='out'):
        w = common.variable(
            name='w',
            init_value=self._w_init.build(
                name='w_init',
                shape=[self._voc_size, self._emb_size]
            ),
            dtype=conf.dtype,
            trainable=self._trainable
        )
        self._w = w
        return tf.nn.embedding_lookup(w, indexes, name=name)

    def load_embedding(self, emb_matrix):
        self._w.load(emb_matrix, common.get_session())

    def dump_embedding(self):
        return common.get_session().run(self._w)


class CharLevelEmbedding(common.Trainable):

    def __init__(self,
                 name,
                 voc_size,
                 emb_size,
                 channels,
                 pooling='max',
                 kernel_height=3,
                 activation=tf.nn.swish):
        """利用一维卷积来生成字符级别的编码

        Args:
            name (str): The widget name.
            voc_size: Vocabulary size.
            emb_size: Embedding size.
            channels:
            pooling: Polling type.
            kernel_height: Convolutional kernel height.
                Note that the convolutional kernel size is (kernel_height, 1)

        """
        super(CharLevelEmbedding, self).__init__(name)
        self._name = name
        self._voc_size = voc_size
        self._emb_size = emb_size
        self._channels = channels
        self._pooling = pooling
        self._kernel_height = kernel_height
        self._activation = activation

    def _build(self):
        if not isinstance(self._channels, (tuple, list)):
            self._channels = [self._channels]

        self._char_emb = Embedding(
            'char_emb',
            voc_size=self._voc_size,
            emb_size=self._emb_size
        )

        self._conv_layers = []
        current_size = self._emb_size
        for i, state_size in enumerate([*self._channels, self._emb_size]):
            layer = conv.Conv2D(
                f'conv2d_{i}',
                input_channels=current_size,
                output_channels=state_size,
                filter_size=(self._kernel_height, 1),
                stride_size=1
            )
            self._conv_layers.append(layer)
            current_size = state_size

    def setup(self, seq, name=None):
        # (batch_size, seq_len, word_len)
        # => (batch_size, seq_len, word_len, emb_size)
        seq_emb = self._char_emb.setup(seq)

        # (batch_size, seq_len, word_len, emb_size)
        # => (batch_size, seq_len, word_len, kernel_size[-1])
        for layer in self._conv_layers:
            seq_emb = layer.setup(seq_emb)
            if self._activation is not None:
                seq_emb = self._activation(seq_emb)

        if self._pooling == 'max':
            seq_emb = tf.reduce_max(seq_emb, axis=2)
        else:
            seq_emb = tf.reduce_mean(seq_emb, axis=2)
        return seq_emb
