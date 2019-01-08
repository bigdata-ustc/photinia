#!/usr/bin/env python3

"""
@author: xi
@since: 2018-06-17
"""

import collections
import datetime as dt
import math

import numpy as np

from . import ops


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


def call_for_batch(context, slot, data_source):
    """

    Args:
        context (dict):
        slot (photinia.Step):
        data_source (photinia.BatchSource):

    Returns:
        dict[str, any]:
        tuple|list:

    """
    data_batch = data_source.next()
    if data_batch is None:
        data_batch = data_source.next()
        if data_batch is None:
            raise RuntimeError('Too many "None" returned by data source.')
    ret = slot(*data_batch)
    if isinstance(ret, (tuple, list)):
        for i, value in enumerate(ret):
            context[i] = value
    elif isinstance(ret, (dict, collections.OrderedDict)):
        context.update(ret)
    else:
        # Should not be reached, since Slot ALWAYS returns tuple or dict.
        raise RuntimeError('Invalid Slot outputs type.')
    return ret


def call_for_all(context, slot, data_source):
    """

    Args:
        context (dict):
        slot (photinia.Step):
        data_source (photinia.BatchSource):

    Returns:
        dict[str, list]:

    """
    ret = collections.defaultdict(list)
    while True:
        data_batch = data_source.next()
        if data_batch is None:
            break
        ret = slot(*data_batch)
        if isinstance(ret, (tuple, list)):
            for i, value in enumerate(ret):
                ret[i].append(value)
        elif isinstance(ret, (dict, collections.OrderedDict)):
            for name, value in ret.items():
                ret[name].append(value)
        else:
            # Should not be reached, since Slot ALWAYS returns tuple or dict.
            raise RuntimeError('Invalid Slot outputs type.')
    context.update(ret)
    return ret


def print_log(context, value_names, i=None, n=None, message=None):
    now = dt.datetime.now()
    print(now.strftime('[%Y-%m-%d %H:%M:%S'), end='')

    if i is not None:
        if n is not None:
            percentage = '%.2f' % (i / n * 100,)
            print(' %s/%s|%s%%]' % (str(i), str(n), percentage), end='')
        else:
            print(' %s]' % str(i), end='')
    else:
        print(']', end='')

    if message is not None:
        print('\t' + str(message), end='')

    values = context[context] if context in context else ()
    if isinstance(values, (tuple, list)):
        for i, name in enumerate(value_names):
            if i < len(values):
                value = values[i]
                print('\t%s=%f' % (name, value), end='')
            else:
                print('\t%s=?' % (name,), end='')
    elif isinstance(values, (dict, collections.OrderedDict)):
        for name in value_names:
            if name in values:
                value = values[name]
                print('\t%s=%f' % (name, value), end='')
            else:
                print('\t%s=?' % (name,), end='')
    print()


class EarlyStopping(object):

    def __init__(self, window_size=5):
        self.window_size = window_size
        self._lowest_error = None
        self._counter = 0

    def convergent(self, error):
        if self._lowest_error is None:
            self._lowest_error = error
            return False
        if error < self._lowest_error:
            self._lowest_error = error
            self._counter = 0
            return False
        else:
            self._counter += 1
            return self._counter >= self.window_size

    def reset(self):
        self._lowest_error = None
        self._counter = 0


class OptimizerWrapper(object):
    """OptimizerWrapper
    """

    def __init__(self,
                 optimizer):
        self._optimizer = optimizer

    @property
    def optimizer(self):
        return self._optimizer

    def minimize(self, loss, var_list=None):
        pair_list = self._optimizer.compute_gradients(loss, var_list=var_list)
        pair_list = self._process_gradients(pair_list)
        return self._optimizer.apply_gradients(pair_list)

    def _process_gradients(self, pair_list):
        raise NotImplementedError


class GradientClipping(OptimizerWrapper):
    """GradientClipping
    """

    def __init__(self, optimizer, max_norm):
        self._max_norm = max_norm
        super(GradientClipping, self).__init__(optimizer)

    @property
    def max_norm(self):
        return self._max_norm

    def _process_gradients(self, pair_list):
        pair_list, raw_grad, grad = ops.clip_gradient(pair_list, self._max_norm)
        self._raw_grad_norm = raw_grad
        self._grad_norm = grad
        return pair_list

    @property
    def raw_grad_norm(self):
        return self._raw_grad_norm

    @property
    def grad_norm(self):
        return self._grad_norm
