#!/usr/bin/env python3

"""
@author: xi
@since: 2018-01-12
"""

import collections

from .conf import tf


class Regularizer(object):

    def __init__(self, weight=None):
        self._weight = weight
        self._items = collections.defaultdict(list)

    def setup(self, tensors):
        raise NotImplementedError()

    def add(self, tensors, reg_op, weight=None):
        if not isinstance(tensors, (list, tuple)):
            tensor = tensors
            item = reg_op(tensor)
            if weight is not None:
                item *= weight
            self._items[tensor].append(item)
        else:
            for tensor in tensors:
                item = reg_op(tensor)
                if weight is not None:
                    item *= weight
                self._items[tensor].append(item)
        return self

    def add_l1(self, tensors, weight=None):
        return self.add(tensors, l1_norm, weight)

    def add_l2(self, tensors, weight=None):
        return self.add(tensors, l2_norm, weight)

    def add_l1_l2(self, tensors, weight=None):
        self.add_l1(tensors, weight)
        self.add_l2(tensors, weight)
        return self

    def remove(self, tensors):
        if not isinstance(tensors, (list, tuple)):
            tensor = tensors
            if tensor in self._items:
                del self._items[tensor]
        else:
            for tensor in tensors:
                if tensor in self._items:
                    del self._items[tensor]
        return self

    def clear(self):
        self._items.clear()
        return self

    def get_loss(self, weight=None):
        items = [
            item
            for partial_items in self._items.values()
            for item in partial_items
        ]
        if len(items) == 0:
            return 0
        loss = items[0]
        for item in items[1:]:
            loss += item
        if weight is None:
            weight = self._weight
        if weight is None:
            raise ValueError('Regularization weight cannot be None.')
        loss *= weight
        return loss


def l1_norm(a, axis=None):
    return tf.reduce_sum(tf.abs(a), axis=axis)


def l2_norm(a, axis=None):
    return tf.reduce_sum(tf.square(a), axis=axis)


class L1Regularizer(Regularizer):

    def __init__(self, weight):
        super(L1Regularizer, self).__init__(weight)

    def setup(self, tensors):
        self.add_l1(tensors)


class L2Regularizer(Regularizer):

    def __init__(self, weight):
        super(L2Regularizer, self).__init__(weight)

    def setup(self, tensors):
        self.add_l2(tensors)


class L1L2Regularizer(Regularizer):

    def __init__(self, weight):
        super(L1L2Regularizer, self).__init__(weight)

    def setup(self, tensors):
        self.add_l1_l2(tensors)
