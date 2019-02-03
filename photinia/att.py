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


class DotProductAttention(ph.Widget):

    def __init__(self, name):
        super(DotProductAttention, self).__init__(name)

    def _build(self):
        pass

    def _setup(self,
               query,
               key,
               value=None,
               query_mask=None,
               key_mask=None,
               value_mask=None):
        query_shape = tf.shape(query)
        if len(query_shape) == 3:
            # (batch_size, query_length, feature_size) @ (batch_size, feature_size, key_length)
            # => (batch_size, query_length, key_length)
            score = query @ tf.transpose(key, (0, 2, 1))
        elif len(query_shape) == 2:
            # (batch_size, feature_size) @ (batch_size, key_length, feature_size)
            # => (batch_size, key_length)
            score = tf.tensordot(query, key, [(1,), (-1,)])
        else:
            raise ValueError(f'Invalid query shape {str(query.shape)}')
        query_size = query_shape[-1]
        score /= tf.sqrt(tf.cast(query_size, ph.float))

        # (batch_size, query_length)
        # => (batch_size, query_length, 1)
        query_mask = tf.expand_dims(query_mask, 2)

        # (batch_size, key_length)
        # => (batch_size, 1, key_length)
        key_mask = tf.expand_dims(key_mask, 1)

        # => (batch_size, query_length, key_length)
        att = ph.ops.softmax(score, axis=2, mask=key_mask)
        att *= query_mask

        # (batch_size, query_length, key_length) @ (batch_size, key_length, emb_size)
        # => (batch_size, query_length, emb_size)
        result = att @ value
        return result, att


class MLPAttention(ph.Widget):

    def __init__(self,
                 name,
                 query_size,
                 key_size,
                 attention_size,
                 with_bias=False,
                 w_init=ph.init.GlorotUniform(),
                 b_init=ph.init.Zeros()):
        self._attention_size = attention_size
        self._query_size = query_size
        self._key_size = key_size
        self._with_bias = with_bias
        self._w_init = w_init
        self._b_init = b_init
        super(MLPAttention, self).__init__(name)

    def _build(self):
        if self._query_size is not None:
            self._query_layer = ph.Linear(
                'query_layer',
                input_size=self._query_size,
                output_size=self._attention_size,
                with_bias=self._with_bias
            )
        self._key_layer = ph.Linear(
            'key_layer',
            input_size=self._key_size,
            output_size=self._attention_size,
            with_bias=self._with_bias
        )
        self._att_layer = ph.Linear(
            'att_layer',
            input_size=self._attention_size,
            output_size=1,
            with_bias=self._with_bias
        )

    def _setup(self,
               query,
               key=None,
               value=None,
               query_mask=None,
               key_mask=None,
               activation=ph.ops.lrelu):
        if key is None:
            key = query
            if key_mask is None:
                key_mask = query_mask
        if value is None:
            value = key

        # check order
        # query => (batch_size, query_len, query_size) or (batch_size, query_size)
        # key and value => (batch_size, key_len, key_size)
        query_order = len(query.shape)
        key_order = len(key.shape)
        assert 2 <= query_order <= 3 and key_order == 3

        if query_order == 2:
            # (batch_size, query_size)
            # query_score => (batch_size, attention_size)
            query_score = self._query_layer.setup(query)
            # query_score => (batch_size, 1, attention_size)
            query_score = tf.expand_dims(query_score, axis=1)

            # (batch_size, key_len, key_size)
            # key_score => (batch_size, key_len, attention_size)
            key_score = self._key_layer.setup(key)

            score = query_score + key_score
            if activation is not None:
                score = activation(score)

            # (batch_size, key_len, attention_size)
            # => (batch_size, key_len, 1)
            score = self._att_layer.setup(score)
            score = ph.ops.softmax(score, axis=1, mask=key_mask)

            value = tf.reduce_sum(score * value, axis=1)
            return value, score
        elif query_order == 3:
            query_shape = tf.shape(query)
            key_shape = tf.shape(key)

            # (batch_size, query_len, query_size)
            # query_score => (batch_size, query_len, attention_size)
            query_score = self._query_layer.setup(query)
            # query_score => (batch_size, query_len, 1, attention_size)
            query_score = tf.expand_dims(query_score, axis=2)
            # query_score => (batch_size, query_len, key_len, attention_size)
            query_score = tf.tile(query_score, multiples=(1, 1, key_shape[1], 1))

            # (batch_size, key_len, key_size)
            # key_score => (batch_size, key_len, attention_size)
            key_score = self._key_layer.setup(key)
            # key_score => (batch_size, 1, key_len, attention_size)
            # value => (batch_size, 1, key_len, value_size)
            key_score = tf.expand_dims(key_score, axis=1)
            value = tf.expand_dims(value, axis=1)
            # key_score => (batch_size, query_len, key_len, attention_size)
            # value => (batch_size, query_len, key_len, value_size)
            key_score = tf.tile(key_score, multiples=(1, query_shape[1], 1, 1))
            value = tf.tile(value, multiples=(1, query_shape[1], 1, 1))

            score = query_score + key_score
            if activation is not None:
                score = activation(score)

            # (batch_size, query_len, key_len, attention_size)
            # => (batch_size, query_len, key_len, 1)
            score = self._att_layer.setup(score)
            score = ph.ops.softmax(score, axis=1, mask=key_mask)

            if query_mask is not None:
                query_mask = tf.reshape(query_mask, shape=(query_shape[0], query_shape[1], 1, 1))
                score = query_mask * score

            # value => (batch_size, query_len, value_size)
            value = tf.reduce_sum(score * value, axis=2)
            return value, score
        else:
            raise ValueError()

        # # (batch_size, q,q,..., query_size)
        # # => (batch_size, q,q,..., attention_size)
        # score_query = self._query_layer.setup(query)
        # # (batch_size, k,k,..., key_size)
        # # => (batch_size, k,k,..., attention_size)
        # score_key = self._key_layer.setup(key)
        #
        # # => (batch_size, q,q,..., k,k,..., attention_size)
        # shape_q = tf.unstack(tf.shape(score_query), axis=0)
        # shape_k = tf.unstack(tf.shape(score_key), axis=0)
        # in_shape_q = shape_q[1:-1]
        # in_shape_k = shape_k[1:-1]
        # num_in_axes_q = len(in_shape_q)
        # num_in_axes_k = len(in_shape_k)
        # if num_in_axes_q > 0:
        #     # score_key => (batch_size, q,q,..., k,k,..., attention_size)
        #     expand_shape = (shape_k[0], *(1 for _ in in_shape_q), *in_shape_k, shape_k[-1])
        #     score_key = tf.reshape(score_key, shape=expand_shape)
        #     score_key = tf.tile(score_key, multiples=(1, *in_shape_q, *(1 for _ in in_shape_k), 1))
        #     # mask_key => (batch_size, 1,1,..., k,k,..., 1)
        #     shape_km = tf.unstack(tf.shape(key_mask))
        #     expand_shape = (shape_km[0], *(1 for _ in in_shape_q), *shape_km[1:], 1)
        #     key_mask = tf.reshape(key_mask, shape=expand_shape)
        # if num_in_axes_k > 0:
        #     # score_query => (batch_size, q,q,..., k,k,..., attention_size)
        #     expand_shape = (shape_q[0], *in_shape_q, *(1 for _ in in_shape_k), shape_q[-1])
        #     score_query = tf.reshape(score_query, shape=expand_shape)
        #     score_query = tf.tile(score_query, multiples=(1, *(1 for _ in in_shape_q), *in_shape_k, 1))
        #     # mask_query => (batch_size, q,q,..., 1,1,..., 1)
        #     shape_qm = tf.unstack(tf.shape(query_mask))
        #     expand_shape = (shape_qm[0], *shape_qm[1:], *(1 for _ in in_shape_k), 1)
        #     query_mask = tf.reshape(query_mask, shape=expand_shape)
        # score = score_query + score_key
        # if activation is not None:
        #     score = activation(score)
        #
        # # (batch_size, q,q,..., k,k,..., attention_size)
        # # => (batch_size, q,q,..., k,k,..., 1)
        # score = self._att_layer.setup(score)
        # score = tf.exp(score) * key_mask
        # # z => (batch_size, q,q,..., 1,1,..., 1)
        # reduce_axes = [num_in_axes_q + 1 + i for i in range(num_in_axes_k)]
        # z = tf.reduce_sum(score, axis=reduce_axes, keep_dims=True)
        # # score
        # score = (score / z) * query_mask
        # pass
