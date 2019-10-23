#!/usr/bin/env python3

from . import common
from .. import conf
from .. import init
from ..conf import tf


class BatchNorm(common.Module):

    def __init__(self,
                 name: str,
                 size: int,
                 is_train,
                 decay: float = 0.95,
                 eps: float = 1e-7):
        super(BatchNorm, self).__init__(name)
        self._size = size
        self._is_train = is_train
        self._decay = decay
        self._eps = eps

    def _build(self):
        self._beta = common.variable(
            name='beta',
            init_value=init.Zeros().build(
                name='beta_init',
                shape=(self._size,)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        self._gamma = common.variable(
            name='gamma',
            init_value=init.Ones().build(
                name='gamma_init',
                shape=(self._size,)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        self._mean = common.variable(
            'mean',
            init_value=init.Zeros().build(
                name='mean_init',
                shape=(self._size,)
            ),
            dtype=conf.float,
            trainable=True
        )
        self._var = common.variable(
            'var',
            init_value=init.Ones().build(
                name='var_init',
                shape=(self._size,)
            ),
            dtype=conf.float,
            trainable=True
        )

    def setup(self, x, axis=-1, name=None):
        if isinstance(self._is_train, bool):
            if self._is_train:
                return self._setup_for_train(x, name)
            else:
                return self._setup_for_predict(x, name)
        else:
            return tf.where(
                self._is_train,
                self._setup_for_train(x, None),
                self._setup_for_predict(x, None),
                name=name
            )

    def _setup_for_train(self, x, name):
        axes = tuple(range(len(x.shape) - 1))
        mean, var = tf.nn.moments(x=x, axes=axes)

        d2 = (1. - self._decay) * tf.cast(self._is_train, tf.float32)
        d1 = 1. - d2
        _mean, _var = self._mean, self._var
        with tf.control_dependencies([tf.assign(self._mean, d1 * _mean + d2 * mean)]):
            mean = tf.identity(mean)
        with tf.control_dependencies([tf.assign(self._var, d1 * _var + d2 * var)]):
            var = tf.identity(var)

        x = (x - mean) * tf.rsqrt(var + self._eps)
        x = tf.add(self._gamma * x, self._beta, name=name)
        return x

    def _setup_for_predict(self, x, name):
        mean = tf.stop_gradient(self._mean)
        var = tf.stop_gradient(self._var)
        x = (x - mean) * tf.rsqrt(var + self._eps)
        x = tf.add(self._gamma * x, self._beta, name=name)
        return x


class InstanceNorm(common.Module):

    def __init__(self,
                 name: str,
                 size: int,
                 eps: float = 1e-7):
        super(InstanceNorm, self).__init__(name)
        self._size = size
        self._eps = eps

    def _build(self):
        self._w = common.variable(
            name='w',
            init_value=init.Ones().build(
                name='w_int',
                shape=(self._size,)
            ),
            dtype=conf.float,
            trainable=True
        )
        self._b = common.variable(
            name='b',
            init_value=init.Zeros().build(
                name='b_init',
                shape=(self._size,)
            ),
            dtype=conf.float,
            trainable=True
        )

    def setup(self, x, name=None):
        shape = tf.shape(x)
        batch_size = shape[0]
        feature_size = shape[-1]
        x = tf.reshape(x, (batch_size, -1, feature_size))
        mean, var = tf.nn.moments(x, axes=(1,), keep_dims=True)
        x = (x - mean) * tf.rsqrt(var + self._eps)
        x = tf.reshape(x, shape)
        x = tf.add(self._w * x, self._b, name=name)
        return x


class LayerNorm(common.Module):

    def __init__(self,
                 name: str,
                 size: int,
                 eps: float = 1e-7):
        super(LayerNorm, self).__init__(name)
        self._size = size
        self._eps = eps

    def _build(self):
        self._w = common.variable(
            name='w',
            init_value=init.Ones().build(
                name='w_int',
                shape=(self._size,)
            ),
            dtype=conf.float,
            trainable=True
        )
        self._b = common.variable(
            name='b',
            init_value=init.Zeros().build(
                name='b_init',
                shape=(self._size,)
            ),
            dtype=conf.float,
            trainable=True
        )

    def setup(self, x, name=None):
        axes = tuple(range(1, len(x.shape)))
        mean, var = tf.nn.moments(x, axes=axes, keep_dims=True)
        x = (x - mean) * tf.rsqrt(var + self._eps)
        x = tf.add(self._w * x, self._b, name=name)
        return x


class GroupNorm(common.Module):

    def __init__(self,
                 name: str,
                 size: int,
                 num_groups: int,
                 eps: float = 1e-7):
        super(GroupNorm, self).__init__(name)
        assert size % num_groups == 0
        self._size = size
        self._num_groups = num_groups
        self._eps = eps

    def _build(self):
        self._w = common.variable(
            name='w',
            init_value=init.Ones().build(
                name='w_int',
                shape=(self._size,)
            ),
            dtype=conf.float,
            trainable=True
        )
        self._b = common.variable(
            name='b',
            init_value=init.Zeros().build(
                name='b_init',
                shape=(self._size,)
            ),
            dtype=conf.float,
            trainable=True
        )

    def setup(self, x, name=None):
        shape = tf.shape(x)
        batch_size = shape[0]
        x = tf.reshape(x, (batch_size, -1, self._num_groups, self._size // self._num_groups))
        mean, var = tf.nn.moments(x, axes=(1, 3), keep_dims=True)
        x = (x - mean) * tf.rsqrt(var + self._eps)
        x = tf.reshape(x, shape)
        x = tf.add(self._w * x, self._b, name=name)
        return x
