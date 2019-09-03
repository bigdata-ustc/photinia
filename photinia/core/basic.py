#!/usr/bin/env python3


"""
@author: xi
@since: 2016-11-11
"""

import tensorflow as tf

from . import common
from .. import conf
from .. import init


class Linear(common.Widget):
    """Linear layer.
    y = wx + b
    """

    def __init__(self,
                 name,
                 input_size,
                 output_size,
                 with_bias=True,
                 w_init=init.GlorotUniform(),
                 b_init=init.Zeros()):
        """Linear layer.

        y = Wx + b

        Args:
            name (str): Widget name.
            input_size (int): Input size.
            output_size (int): Output size.
            with_bias (bool): If the layer contains bias.
            w_init (init.Initializer): Weight initializer.
            b_init (initializers.Initializer): Bias initializer.

        """
        self._input_size = input_size
        self._output_size = output_size
        self._with_bias = with_bias
        self._w_init = w_init
        self._b_init = b_init
        super(Linear, self).__init__(name)

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def with_bias(self):
        return self._with_bias

    def _build(self):
        """Build the linear layer.
        Two parameters: weight and bias.

        """
        self._w = self._variable(
            name='w',
            initializer=self._w_init,
            shape=(self._input_size, self._output_size),
            dtype=conf.dtype,
        )
        if self._with_bias:
            self._b = self._variable(
                name='b',
                initializer=self._b_init,
                shape=(self._output_size,),
                dtype=conf.dtype
            )
        else:
            self._b = None

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

    def _setup(self, x, name='out'):
        """Setup the layer.

        Args:
            x (tf.Tensor): Input tensor.
            name (str): Output name.

        Returns:
            tf.Tensor: Output tensor.

        """
        if self._with_bias:
            if len(x.shape) == 2:
                y = tf.matmul(x, self._w)
            else:
                y = self._matmul(x)
            y = tf.add(y, self._b, name=name)
        else:
            if len(x.shape) == 2:
                y = tf.matmul(x, self._w, name=name)
            else:
                y = self._matmul(x, name=name)
        return y

    def _matmul(self, x, name=None):
        #    [?_1, ?_2, ..., ?_n, input_size]
        # -> [?, input_size]
        x_mat = tf.reshape(x, [-1, self._input_size])
        #    [?, input_size] @ [input_size, output_size]
        # -> [?, output_size]
        y_mat = tf.matmul(x_mat, self._w)
        # -> [?_1, ?_2, ..., ?_{n-1} output_size]
        shape = tf.shape(x)
        shape = tf.unstack(shape)
        shape[-1] = self._output_size
        shape = tf.stack(shape)
        y = tf.reshape(y_mat, shape, name=name)
        return y


class Dropout(common.Widget):

    def __init__(self,
                 name,
                 keep_prob,
                 is_train):
        """Dropout

        Args:
            name (str): Widget name.
            keep_prob (float|tf.Tensor): Keep probability.

        """
        self._keep_prob = keep_prob
        self._rate = 1.0 - keep_prob
        self._is_train = is_train
        super(Dropout, self).__init__(name)

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def rate(self):
        return self._rate

    def _build(self):
        pass

    def _setup(self, x, name='out'):
        """Setup dropout.

        Args:
            x (tf.Tensor): Input tensor.
            name (str): Output name.

        Returns:
            tf.Tensor: Output tensor.

        """
        if isinstance(self._is_train, bool):
            if self._is_train:
                return tf.nn.dropout(x, rate=self._rate, name=name)
            else:
                return x
        else:
            return tf.nn.dropout(
                x,
                rate=self._rate * tf.cast(self._is_train, conf.dtype),
                name=name
            )
