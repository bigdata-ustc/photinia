#!/usr/bin/env python3

import collections

from ..conf import tf


class Regularizer(object):

    def __init__(self, weight):
        self._weight = weight
        self._items = collections.defaultdict(list)

    def setup(self, tensors, weight=None):
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        for tensor in tensors:
            item = self._setup(tensor)
            if weight is not None:
                item = weight * item
            self._items[tensor].append(item)
        return self

    def _setup(self, tensor):
        raise NotImplementedError()

    def remove(self, tensors):
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        for tensor in tensors:
            if tensor in self._items:
                del self._items[tensor]
        return self

    def clear(self):
        self._items.clear()
        return self

    def get_loss(self):
        items = [
            item
            for partial_items in self._items.values()
            for item in partial_items
        ]
        if len(items) == 0:
            return 0
        loss = self._weight * tf.reduce_sum(items)
        return loss


class L1Regularizer(Regularizer):

    def _setup(self, tensor):
        return tf.reduce_sum(tf.abs(tensor))


class L2Regularizer(Regularizer):

    def _setup(self, tensor):
        return tf.reduce_sum(tf.square(tensor))


class L1L2Regularizer(Regularizer):

    def _setup(self, tensor):
        return tf.reduce_sum(tf.abs(tensor)) + tf.reduce_sum(tf.square(tensor))
