#!/usr/bin/env python3


"""
@author: xi
@since: 2016-11-11
"""

from . import common
from .. import conf
from .. import init
from ..conf import tf


class Linear(common.Module):
    """Linear layer.
    y = wx + b
    """

    def __init__(self,
                 name,
                 input_size,
                 output_size,
                 with_bias=True,
                 w_init=init.GlorotNormal(),
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
        super(Linear, self).__init__(name)
        self._input_size = input_size
        self._output_size = output_size
        self._with_bias = with_bias
        self._w_init = w_init
        self._b_init = b_init

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def with_bias(self):
        return self._with_bias

    def setup(self, x, name=None):
        w = common.variable(
            name='w',
            init_value=self._w_init.build(
                name='w_init',
                shape=[self._input_size, self._output_size]
            ),
            dtype=conf.dtype,
            trainable=True
        )
        if self._with_bias:
            b = common.variable(
                name='b',
                init_value=self._b_init.build(
                    name='b_init',
                    shape=[self._output_size]
                ),
                dtype=conf.dtype,
                trainable=True
            )
            y = tf.matmul(x, w) if len(x.shape) == 2 else self._matmul(x, w)
            y = tf.add(y, b, name=name)
        else:
            y = tf.matmul(x, w, name=name) if len(x.shape) == 2 else self._matmul(x, w, name=name)
        return y

    def _matmul(self, x, w, name=None):
        #    [?_1, ?_2, ..., ?_n, input_size]
        # -> [?, input_size]
        x_mat = tf.reshape(x, [-1, self._input_size])
        #    [?, input_size] @ [input_size, output_size]
        # -> [?, output_size]
        y_mat = tf.matmul(x_mat, w)
        # -> [?_1, ?_2, ..., ?_{n-1} output_size]
        shape = tf.shape(x)
        shape = tf.unstack(shape)
        shape[-1] = self._output_size
        shape = tf.stack(shape)
        y = tf.reshape(y_mat, shape, name=name)
        return y


class Dropout(common.Module):

    def __init__(self,
                 name,
                 keep_prob,
                 is_train):
        """Dropout

        Args:
            name (str): Widget name.
            keep_prob (float|tf.Tensor): Keep probability.

        """
        super(Dropout, self).__init__(name)
        self._keep_prob = keep_prob
        self._rate = 1.0 - keep_prob
        self._is_train = is_train

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def rate(self):
        return self._rate

    def setup(self, x, name='out'):
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
