#!/usr/bin/env python3

import math

import numpy as np


class AccCalculator(object):

    def __init__(self):
        self._num_hit = 0
        self._num_all = 0

    def update(self, label_pred, label_true):
        hit = np.equal(label_pred, label_true)
        hit = np.float32(hit)
        self._num_hit += float(np.sum(hit))
        self._num_all += hit.size

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
