#!/usr/bin/env python3

import numpy as np

from .. import conf
from .. import core
from .. import init
from ..conf import tf


class EarlyStopping(object):

    def __init__(self, window_size=5, model=None):
        """The early stopping monitor.

        Args:
            window_size (int): The windows size to monitor after the best performance.
            model (photinia.Trainable): The model to tune.

        """
        self.window_size = window_size
        self._model = model

        self._lowest_error = None
        self._best_params = None
        self._counter = 0

    @property
    def lowest_error(self):
        return self._lowest_error

    @property
    def best_parameters(self):
        return self._best_params

    def convergent(self, error):
        if self._lowest_error is None:
            self._lowest_error = error
            if self._model is not None:
                self._best_params = self._model.get_parameters()
            return False
        if error < self._lowest_error:
            self._lowest_error = error
            if self._model is not None:
                self._best_params = self._model.get_parameters()
            self._counter = 0
            return False
        else:
            self._counter += 1
            return self._counter >= self.window_size

    def reset(self):
        self._lowest_error = None
        self._counter = 0


class ExponentialDecayedValue(core.Trainable):
    def __init__(self,
                 name,
                 init_value,
                 shape=None,
                 decay_rate=None,
                 num_loops=None,
                 min_value=None,
                 dtype=conf.float,
                 trainable=False):
        super(ExponentialDecayedValue, self).__init__(name)
        self._init_value = init_value
        self._shape = shape
        self._decay_rate = decay_rate
        self._num_loops = num_loops
        self._min_value = min_value
        self._dtype = dtype
        self._trainable = trainable

        self._variable = None
        self._value = None
        self.reset = None
        self.setup()

    def setup(self):
        if isinstance(self._init_value, init.Initializer):
            if self._shape is None:
                raise ValueError('"shape" must be given when Initializer is used.')
            initializer = self._init_value
        elif isinstance(self._init_value, np.ndarray):
            self._shape = self._init_value.shape
            initializer = init.Constant(self._init_value)
        elif isinstance(self._init_value, (float, int)):
            self._shape = ()
            initializer = init.Constant(self._init_value)
        else:
            raise ValueError('Type of "init_value" should be one of {int, float, np.ndarray, ph.init.Initializer}.')
        self._variable = core.variable(
            name='value',
            init_value=initializer.build(
                name='value_init',
                shape=self._shape
            ),
            dtype=self._dtype,
            trainable=self._trainable
        )

        if self._decay_rate is None:
            if self._num_loops is None or self._min_value is None:
                raise ValueError('"decay_rate" is missing. You should set both "num_loops" and "min_value".')
            self._decay_rate = (self._min_value / self._init_value) ** (1.0 / self._num_loops)
        new_value = tf.multiply(self._variable, self._decay_rate)
        if self._min_value is not None:
            new_value = tf.maximum(new_value, self._min_value)
        with tf.control_dependencies([tf.assign(self._variable, new_value)]):
            self._value = tf.identity(self._variable)

        self.reset = core.Step(
            updates=tf.assign(self._variable, self._init_value)
        )

    @property
    def variable(self):
        return self._variable

    @property
    def value(self):
        return self._value
