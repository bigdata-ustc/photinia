#!/usr/bin/env python3


"""
@author: xi
@since: 2016-11-11
"""

from . import basic
from . import common
from .. import conf
from .. import init
from .. import ops
from ..conf import tf


class Gate(common.Trainable):

    def __init__(self,
                 name,
                 input_sizes,
                 output_size,
                 w_init=init.GlorotNormal(),
                 b_init=init.Zeros()):
        super(Gate, self).__init__(name)
        if not isinstance(input_sizes, (tuple, list)):
            input_sizes = (input_sizes,)
        self._input_sizes = input_sizes
        self._output_size = output_size
        self._w_init = w_init
        self._b_init = b_init

    def setup(self, *x_list, name='out'):
        w_list = list()
        for i, input_size in enumerate(self._input_sizes):
            w = common.variable(
                name=f'w_{i}',
                init_value=self._w_init.build(
                    name=f'w_{i}_init',
                    shape=[input_size, self._output_size]
                ),
                dtype=conf.dtype,
                trainable=True
            )
            w_list.append(w)
        b = common.variable(
            name='b',
            init_value=self._b_init.build(
                name='b_init',
                shape=[self._output_size]
            ),
            dtype=conf.dtype,
            trainable=True
        )
        if len(x_list) != len(w_list):
            raise ValueError()
        y = None
        for i, x in enumerate(x_list):
            if y is None:
                y = tf.matmul(x, w_list[i])
            else:
                y += tf.matmul(x, w_list[i])
        y += b
        y = tf.nn.sigmoid(y, name=name)
        return y


class ResidualLayer(common.Trainable):
    """Residual network cell for DNN.

    The original version is contributed by zhkun~(Kun Zhang) in his testing code.
    """

    def __init__(self,
                 name,
                 size,
                 num_layers=1,
                 w_init=init.GlorotNormal(),
                 b_init=init.Zeros(),
                 activation=ops.lrelu):
        """Residual network cell for DNN.

        Args:
            name (str): Widget name.
            size (int): Input and output size.
            num_layers (int): Number of layers.
            w_init (init.Initializer): Initializer for weight.
            b_init (initializers.Initializer): Initializer for bias.

        """
        super(ResidualLayer, self).__init__(name)
        if num_layers < 1:
            raise ValueError(
                'Invalid number of layers. Number that larger than 1 expected, got %d.' % num_layers
            )
        self._size = size
        self._num_layers = num_layers
        self._w_init = w_init
        self._b_init = b_init
        self._activation = activation
        self._layers = list()

    @property
    def size(self):
        return self._size

    @property
    def num_layers(self):
        return self._num_layers

    def setup(self, x, name='out'):
        for i in range(self._num_layers):
            layer = basic.Linear(
                'lin_' + str(i),
                input_size=self._size,
                output_size=self._size,
                w_init=self._w_init,
                b_init=self._b_init
            )
            self._layers.append(layer)
        h = x
        for layer in self._layers[:-1]:
            h = layer.setup(h)
            if self._activation is not None:
                h = self._activation(h)

        h = self._layers[-1].setup(h)
        if self._activation is not None:
            h = tf.add(h, x)
            h = self._activation(h, name=name)
        else:
            h = tf.add(h, x, name=name)
        return h


class HighwayLayer(common.Trainable):
    """Highway network cell for DNN.

    The original version is contributed by zhkun~(Kun Zhang) in his testing code.
    """

    def __init__(self,
                 name,
                 size,
                 w_init=init.GlorotNormal(),
                 b_init=init.Zeros(),
                 activation=ops.lrelu):
        """Highway network cell for DNN.

        Args:
            name (str): Widget name.
            size (int): Input and output size.
            w_init (init.Initializer): Initializer for weight.
            b_init (initializers.Initializer): Initializer for bias.

        """
        super(HighwayLayer, self).__init__(name)
        self._size = size
        self._w_init = w_init
        self._b_init = b_init
        self._activation = activation

    def setup(self, x, name='out'):
        linear = basic.Linear(
            'lin',
            input_size=self._size,
            output_size=self._size,
            w_init=self._w_init,
            b_init=self._b_init
        )
        gate = basic.Linear(
            'gate',
            input_size=self._size,
            output_size=self._size,
            w_init=self._w_init,
            b_init=self._b_init
        )
        h = linear.setup(x)
        if self._activation is not None:
            h = self._activation(h)

        g = gate.setup(x)
        g = tf.nn.sigmoid(g)

        y = tf.add(
            tf.multiply(g, h),
            tf.multiply((1.0 - g), x),
            name=name
        )
        return y
