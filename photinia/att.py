#!/usr/bin/env python3

"""
@author: xi
@since: 2019-01-30
"""

import tensorflow as tf

import photinia as ph


class SelfAttention(ph.Widget):

    def __init__(self,
                 name,
                 input_size,
                 attention_size,
                 guide_size=None,
                 w_init=ph.init.TruncatedNormal(),
                 v_init=ph.init.TruncatedNormal()):
        """Additive attention unit.

        Args:
            name (str): The widget name.
            input_size (int): The feature size of the input sequence.
            attention_size: The attention size.
            guide_size: The guide vector size.
            w_init: The w matrix initializer.
            v_init: The v matrix initializer.

        """
        self._name = name
        self._input_size = input_size
        self._attention_size = attention_size
        self._guide_size = guide_size
        self._w_init = w_init
        self._v_init = v_init
        super(SelfAttention, self).__init__(name)

    @property
    def input_size(self):
        return self._input_size

    @property
    def attention_size(self):
        return self._attention_size

    @property
    def guide_size(self):
        return self._guide_size

    def _build(self):
        self._w_attention = self._variable(
            'w_attention',
            initializer=self._w_init,
            shape=(self._input_size, self._attention_size),
            dtype=ph.float
        )
        if self._guide_size is not None:
            self._w_guide = self._variable(
                'w_guide',
                initializer=self._w_init,
                shape=(self._guide_size, self._attention_size),
                dtype=ph.float
            )
        self._v_attention = self._variable(
            'v_attention',
            initializer=self._w_init,
            shape=(self._attention_size, 1),
            dtype=ph.float
        )

    def _setup(self,
               seq,
               seq_mask=None,
               guide=None,
               activation=ph.ops.lrelu,
               scale=True,
               name='self_attention'):
        """Setup for a sequence.

        Args:
            seq: The sequence (batch_size, seq_len, input_size).
            seq_mask: The sequence mask (batch_size, seq_len).
            guide: The guide vector (batch_size, guide_size).
            name (str): The output name.

        Returns:
            The result (batch_size, input_size) and the attention score (batch_size, seq_len).

        """
        # (batch_size, seq_len, input_size) @ (input_size, attention_size)
        # => (batch_size, seq_len, attention_size)
        score = tf.tensordot(seq, self._w_attention, [(-1,), (0,)])
        if activation:
            score = activation(score)

        if guide is not None:
            # (batch_size, guide_size) @ (guide_size, attention_size)
            # => (batch_size, attention_size)
            # => (batch_size, attention_size, 1)
            # => (batch_size, seq_len, attention_size)
            guide_score = tf.tensordot(guide, self._w_guide, [(-1,), (0,)])
            if activation:
                guide_score = activation(guide_score)
            guide_score = tf.expand_dims(guide_score, axis=1)
            score += guide_score

        # (batch_size, seq_len, attention_size) @ (attention_size, 1)
        # => (batch_size, seq_len, 1)
        score = tf.tensordot(score, self._v_attention, [(-1,), (0,)])
        if scale:
            dim_query = tf.shape(seq)[-1]
            score /= tf.sqrt(tf.cast(dim_query, tf.float32))
            score = ph.ops.softmax(score, axis=1, mask=seq_mask)

        # (batch_size, seq_len, 1) * (batch_size, seq_len, input_size)
        # => (batch_size, seq_len, input_size)
        # => (batch_size, input_size)
        result = tf.reduce_sum(score * seq, axis=1, name=f'{name}_result')

        # (batch_size, seq_len, 1)
        #  => (batch_size, seq_len)
        score = tf.squeeze(score, axis=-1, name=f'{name}_score')

        return result, score
