#!/usr/bin/env python3

"""
@author: xi
@since: 2019-01-19
"""

import tensorflow as tf

import photinia as ph


class LayerNorm(ph.Widget):

    def __init__(self, name, size, eps=1e-6):
        self._name = name
        self._size = size
        self._eps = eps
        super(LayerNorm, self).__init__(name)

    @property
    def size(self):
        return self._size

    def _build(self):
        self._w = self._variable(
            name='w',
            initializer=ph.init.Zeros(),
            shape=(self._size,),
            dtype=ph.float
        )
        self._b = self._variable(
            name='b',
            initializer=ph.init.Zeros(),
            shape=(self._size,),
            dtype=ph.float
        )

    def _setup(self, x, name='out'):
        """Setup for a tensor.

        Args:
            x: A tensor whose last dimension should be equal to "size".
            name (str): The output name.

        Returns:
            A tensor which has the same shape as "x".

        """
        mean, var = tf.nn.moments(x, axes=-1, keep_dims=True)
        normalized = (x - mean) / tf.sqrt(var + self._eps)
        result = tf.add(self._w * normalized, self._b, name=name)
        return result


class GRU(ph.Widget):

    def __init__(self,
                 name,
                 input_size,
                 state_size):
        """The recurrent neural network with GRU cell.
        All sequence shapes follow (batch_size, seq_len, state_size).

        Args:
            name (str): The widget name.
            input_size: The input size.
            state_size: The state size.

        """
        self._input_size = input_size
        self._state_size = state_size
        super(GRU, self).__init__(name)

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def _build(self):
        self._cell = ph.GRUCell(
            'cell',
            input_size=self._input_size,
            state_size=self._state_size
        )

    def _setup(self,
               seq,
               init_state=None,
               activation=tf.nn.tanh,
               name='states'):
        """Setup a sequence.

        Args:
            seq: The sequences.
                shape = (batch_size, seq_len, input_size)
            init_state: The initial state.
                shape = (batch_size, state_size)
            activation: The activation function for the GRU cell.

        Returns:
            The forward states.
                shape = (batch_size, seq_len, state_size)

        """
        # check forward and backward initial states
        if init_state is None:
            batch_size = tf.shape(seq)[0]
            init_state = tf.zeros(shape=(batch_size, self._state_size), dtype=ph.dtype)

        # connect
        states_forward = tf.scan(
            fn=lambda acc, elem: self._cell.setup(elem, acc, activation=activation),
            elems=tf.transpose(seq, [1, 0, 2]),
            initializer=init_state
        )
        states_forward = tf.transpose(states_forward, [1, 0, 2], name=name)

        return states_forward


class BiGRU(ph.Widget):

    def __init__(self,
                 name,
                 input_size,
                 state_size):
        """A very simple BiGRU structure comes from zhkun.
        All sequence shapes follow (batch_size, seq_len, state_size).

        Args:
            name (str): The widget name.
            input_size: The input size.
            state_size: The state size.

        """
        self._input_size = input_size
        self._state_size = state_size
        super(BiGRU, self).__init__(name)

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def _build(self):
        self._cell_forward = ph.GRUCell(
            'cell_forward',
            input_size=self._input_size,
            state_size=self._state_size
        )
        self._cell_backward = ph.GRUCell(
            'cell_backward',
            input_size=self._input_size,
            state_size=self._state_size
        )

    def _setup(self,
               seq,
               init_state=None,
               activation=tf.nn.tanh,
               name='states'):
        """Setup a sequence.

        Args:
            seq: A sequence or a pair of sequences.
                shape = (batch_size, seq_len, input_size)
            init_state: A tensor or a pair of tensors.
            activation: The activation function for the GRU cell.

        Returns:
            The forward states and the backward states.
                shape = (batch_size, seq_len, state_size)

        """
        # check forward and backward sequences
        if isinstance(seq, (tuple, list)):
            if len(seq) != 2:
                raise ValueError('The seqs should be tuple with 2 elements.')
            seq_forward, seq_backward = seq
        else:
            seq_forward = seq
            seq_backward = tf.reverse(seq, axis=[1])

        # check forward and backward initial states
        if init_state is None:
            batch_size = tf.shape(seq_forward)[0]
            init_state_forward = tf.zeros(shape=(batch_size, self._state_size), dtype=ph.dtype)
            init_state_backward = tf.zeros(shape=(batch_size, self._state_size), dtype=ph.dtype)
        elif isinstance(init_state, (tuple, list)):
            if len(seq) != 2:
                raise ValueError('The init_states should be tuple with 2 elements.')
            init_state_forward, init_state_backward = init_state
        else:
            init_state_forward = init_state
            init_state_backward = init_state

        # connect
        states_forward = tf.scan(
            fn=lambda acc, elem: self._cell_forward.setup(elem, acc, activation=activation),
            elems=tf.transpose(seq_forward, [1, 0, 2]),
            initializer=init_state_forward
        )
        states_forward = tf.transpose(states_forward, [1, 0, 2], name=f'{name}_forward')
        states_backward = tf.scan(
            fn=lambda acc, elem: self._cell_backward.setup(elem, acc, activation=activation),
            elems=tf.transpose(seq_backward, [1, 0, 2]),
            initializer=init_state_backward
        )
        states_backward = tf.transpose(states_backward, [1, 0, 2], name=f'{name}_backward')

        return states_forward, states_backward


class CharLevelEmbedding(ph.Widget):

    def __init__(self,
                 name,
                 voc_size,
                 emb_size,
                 channels,
                 pooling='max',
                 kernel_height=3):
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
        self._name = name
        self._voc_size = voc_size
        self._emb_size = emb_size
        self._channels = channels
        self._pooling = pooling
        self._kernel_height = kernel_height
        super(CharLevelEmbedding, self).__init__(name)

    def _build(self):
        if not isinstance(self._channels, (tuple, list)):
            self._channels = [self._channels]

        self._char_emb = ph.Embedding(
            'char_emb',
            voc_size=self._voc_size,
            emb_size=self._emb_size
        )

        self._conv_layers = []
        current_size = self._emb_size
        for i, state_size in enumerate([*self._channels, self._emb_size]):
            layer = ph.Conv2D(
                f'conv2d_{i}',
                input_size=(None, None, current_size),
                output_channels=state_size,
                filter_height=self._kernel_height,
                filter_width=1,
                stride_height=1,
                stride_width=1,
            )
            self._conv_layers.append(layer)
            current_size = state_size

        self._norm = LayerNorm('norm', size=self._emb_size)

    def _setup(self,
               seq,
               # dropout=None,
               activation=ph.ops.lrelu,
               name='out'):
        # (batch_size, seq_len, word_len)
        # => (batch_size, seq_len, word_len, emb_size)
        seq_emb = self._char_emb.setup(seq)

        # (batch_size, seq_len, word_len, emb_size)
        # => (batch_size, seq_len, word_len, kernel_size[-1])
        for layer in self._conv_layers:
            seq_emb = layer.setup(seq_emb)
            if activation is not None:
                seq_emb = activation(seq_emb)

        if self._pooling == 'max':
            seq_emb = tf.reduce_max(seq_emb, axis=2)
        else:
            seq_emb = tf.reduce_mean(seq_emb, axis=2)

        # if dropout is not None:
        #     seq_emb = dropout.setup(seq_emb)

        # seq_emb = self._norm(seq_emb, name=name)
        return seq_emb


from ..core import common
from .. import init
import math
from .. import conf


class Conv2D(common.Widget):

    def __init__(self,
                 name,
                 input_channels,
                 output_channels,
                 filter_size,
                 stride_size,
                 input_size=None,
                 padding='SAME',
                 use_bias=True,
                 w_init=init.TruncatedNormal(),
                 b_init=init.Zeros()):
        self._input_channels = input_channels
        self._output_channels = output_channels

        if isinstance(filter_size, (tuple, list)):
            assert len(filter_size) == 2
            self._filter_height, self._filter_width = filter_size
        else:
            self._filter_height = self._filter_width = filter_size

        if isinstance(stride_size, (tuple, list)):
            assert len(stride_size) == 2
            self._stride_height, self._stride_width = stride_size
        else:
            self._stride_height = self._stride_width = stride_size

        padding = padding.upper()
        assert padding in {'SAME', 'VALID'}
        self._padding = padding

        self._use_bias = use_bias
        self._w_init = w_init
        self._b_init = b_init

        if input_size is None:
            self._input_size = None
            self._input_height = None
            self._input_width = None
            self._flat_size = None
        else:
            self._input_size = input_size
            if isinstance(input_size, (tuple, list)):
                assert len(input_size) == 2
                self._input_height, self._input_width = input_size
            else:
                self._input_height = input_size
                self._input_width = input_size
            if self._padding == 'SAME':
                self._output_height = math.ceil(self._input_height / self._stride_height)
                self._output_width = math.ceil(self._input_width / self._stride_width)
            else:
                self._output_height = math.ceil((self._input_height - self._filter_height + 1) / self._stride_height)
                self._output_width = math.ceil((self._input_width - self._filter_width + 1) / self._stride_width)
            self._flat_size = self._output_height * self._output_width * output_channels

        super(Conv2D, self).__init__(name)

    @property
    def input_size(self):
        return self._input_size

    @property
    def input_height(self):
        return self._input_height

    @property
    def input_width(self):
        return self._input_width

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def output_size(self):
        return self._output_height, self._output_width

    @property
    def output_height(self):
        return self._output_height

    @property
    def output_width(self):
        return self._output_width

    @property
    def output_channels(self):
        return self._output_channels

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def stride_height(self):
        return self._stride_height

    @property
    def stride_width(self):
        return self._stride_width

    @property
    def flat_size(self):
        return self._flat_size

    def _build(self):
        self._w = self._variable(
            name='w',
            initializer=self._w_init,
            shape=(
                self._filter_height,
                self._filter_width,
                self._input_channels,
                self._output_channels
            ),
            dtype=conf.dtype
        )
        if self._use_bias:
            self._b = self._variable(
                name='b',
                initializer=self._b_init,
                shape=(self._output_channels,),
                dtype=conf.dtype
            )

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

    def _setup(self, x, name='out'):
        if self._use_bias:
            y = tf.nn.conv2d(
                input=x,
                filter=self._w,
                strides=[1, self._stride_height, self._stride_width, 1],
                padding=self._padding,
                data_format='NHWC'
            )
            y = tf.add(y, self._b, name=name)
        else:
            y = tf.nn.conv2d(
                input=x,
                filter=self._w,
                strides=[1, self._stride_height, self._stride_width, 1],
                padding=self._padding,
                data_format='NHWC',
                name=name
            )
        return y


class Deconv2D(common.Widget):

    def __init__(self,
                 name,
                 input_channels,
                 output_channels,
                 filter_size,
                 stride_size,
                 input_size=None,
                 use_bias=True,
                 w_init=init.TruncatedNormal(),
                 b_init=init.Zeros()):
        self._input_channels = input_channels
        self._output_channels = output_channels

        if isinstance(filter_size, (tuple, list)):
            assert len(filter_size) == 2
            self._filter_height, self._filter_width = filter_size
        else:
            self._filter_height = self._filter_width = filter_size

        if isinstance(stride_size, (tuple, list)):
            assert len(stride_size) == 2
            self._stride_height, self._stride_width = stride_size
        else:
            self._stride_height = self._stride_width = stride_size

        # TODO: now it only support the "SAME" padding method

        self._use_bias = use_bias
        self._w_init = w_init
        self._b_init = b_init

        if input_size is None:
            self._input_size = None
            self._input_height = None
            self._input_width = None
            self._flat_size = None
        else:
            self._input_size = input_size
            if isinstance(input_size, (tuple, list)):
                assert len(input_size) == 2
                self._input_height, self._input_width = input_size
            else:
                self._input_height = input_size
                self._input_width = input_size
                self._output_height = self._input_height * self._stride_height
                self._output_width = self._input_width * self._stride_width

        super(Deconv2D, self).__init__(name)

    @property
    def input_size(self):
        return self._input_size

    @property
    def input_height(self):
        return self._input_height

    @property
    def input_width(self):
        return self._input_width

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def output_size(self):
        return self._output_height, self._output_width

    @property
    def output_height(self):
        return self._output_height

    @property
    def output_width(self):
        return self._output_width

    @property
    def output_channels(self):
        return self._output_channels

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def stride_height(self):
        return self._stride_height

    @property
    def stride_width(self):
        return self._stride_width

    def _build(self):
        self._w = self._variable(
            name='w',
            initializer=self._w_init,
            shape=(
                self._filter_height,
                self._filter_width,
                self._output_channels,
                self._input_channels
            ),
            dtype=conf.dtype
        )
        if self._use_bias:
            self._b = self._variable(
                name='b',
                initializer=self._b_init,
                shape=(self._output_channels,),
                dtype=conf.dtype
            )

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

    def _setup(self, x, name='out'):
        input_shape = tf.shape(x)
        batch_size, input_height, input_width = input_shape[0], input_shape[1], input_shape[2]
        output_shape = (
            batch_size,
            input_height * self._stride_height,
            input_width * self._stride_width,
            self._output_channels
        )
        if self._use_bias:
            y = tf.nn.conv2d_transpose(
                value=x,
                filter=self._w,
                output_shape=output_shape,
                strides=[1, self._stride_height, self._stride_width, 1],
                padding='SAME',
                data_format='NHWC'
            )
            y = tf.add(y, self._b, name=name)
        else:
            y = tf.nn.conv2d_transpose(
                value=x,
                filter=self._w,
                output_shape=output_shape,
                strides=[1, self._stride_height, self._stride_width, 1],
                padding='SAME',
                data_format='NHWC',
                name=name
            )
        return y
