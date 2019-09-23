#!/usr/bin/env python3


"""
@author: xi
@since: 2016-11-11
"""

from . import common
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
