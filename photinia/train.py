#!/usr/bin/env python3

"""
@author: xi
@since: 2018-06-17
"""

import math

import numpy as np
import tensorflow as tf

from . import conf
from . import core
from . import init


class AccCalculator(object):

    def __init__(self):
        self._num_hit = 0
        self._num_all = 0

    def update(self, label_pred, label_true):
        hit = np.equal(label_pred, label_true)
        hit = np.float32(hit)
        self._num_hit += float(np.sum(hit))
        self._num_all += len(hit)

    def reset(self):
        self._num_hit = 0
        self._num_all = 0

    @property
    def accuracy(self):
        return self._num_hit / self._num_all if self._num_all > 0 else math.nan


class BiClassCalculator(object):

    def __init__(self):
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0

    def update(self, label_predict, label_true):
        hit = np.equal(label_predict, label_true)
        hit = np.float32(hit)
        miss = 1.0 - hit

        pos = np.float32(label_predict)
        neg = 1.0 - pos

        self._tp += np.sum(hit * pos, keepdims=False)
        self._tn += np.sum(hit * neg, keepdims=False)
        self._fp += np.sum(miss * pos, keepdims=False)
        self._fn += np.sum(miss * neg, keepdims=False)

    @property
    def precision(self):
        num_pos_pred = self._tp + self._fp
        return self._tp / num_pos_pred if num_pos_pred > 0 else math.nan

    @property
    def recall(self):
        num_pos_true = self._tp + self._fn
        return self._tp / num_pos_true if num_pos_true > 0 else math.nan

    @property
    def f1(self):
        pre = self.precision
        rec = self.recall
        return 2 * (pre * rec) / (pre + rec)

    @property
    def accuracy(self):
        num_hit = self._tp + self._tn
        num_all = self._tp + self._tn + self._fp + self._fn
        return num_hit / num_all if num_all > 0 else math.nan


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


class ExponentialDecayedValue(core.Model):
    def __init__(self,
                 name,
                 init_value,
                 shape=None,
                 decay_rate=None,
                 num_loops=None,
                 min_value=None,
                 dtype=conf.float,
                 trainable=False):
        self._init_value = init_value
        self._shape = shape
        self._decay_rate = decay_rate
        self._num_loops = num_loops
        self._min_value = min_value
        self._dtype = dtype
        self._trainable = trainable
        super(ExponentialDecayedValue, self).__init__(name)

    def _build(self):
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
        self._value = self._variable(
            'value',
            initializer=initializer,
            shape=self._shape,
            dtype=self._dtype,
            trainable=self._trainable
        )

        if self._decay_rate is None:
            if self._num_loops is None or self._min_value is None:
                raise ValueError('"decay_rate" is missing. You should set both "num_loops" and "min_value".')
            self._decay_rate = (self._min_value / self._init_value) ** (1.0 / self._num_loops)
        new_value = tf.multiply(self._value, self._decay_rate)
        if self._min_value is not None:
            new_value = tf.maximum(new_value, self._min_value)
        self._update_op = tf.assign(self._value, new_value)
        self.update = core.Step(updates=self._update_op)

    @property
    def value(self):
        return self._value

    @property
    def update_op(self):
        return self._update_op
