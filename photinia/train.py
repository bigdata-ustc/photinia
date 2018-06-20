#!/usr/bin/env python3

"""
@author: xi
@since: 2018-06-17
"""

import collections
import datetime as dt

from . import ops


def call_for_batch(context, slot, data_source):
    """

    Args:
        context (dict):
        slot (photinia.Slot):
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
        slot (photinia.Slot):
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


def calculate_performance(context, tp='tp', tn='tn', fp='fp', fn='fn'):
    """

    Args:
        context (dict):
        tp (str):
        tn (str):
        fp (str):
        fn (str):

    Returns:
        dict:

    """
    tp = sum(context[tp])
    tn = sum(context[tn])
    fp = sum(context[fp])
    fn = sum(context[fn])

    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * (pre * rec) / (pre + rec)
    acc = (tp + tn) / (tp + tn + fp + fn)

    ret = dict(
        pre=pre,
        rec=rec,
        f1=f1,
        acc=acc
    )
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
