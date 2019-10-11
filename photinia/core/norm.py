#!/usr/bin/env python3

from . import common
from .. import conf
from .. import init
from ..conf import tf


class BatchNorm(common.Trainable):

    def __init__(self,
                 name: str,
                 size: int,
                 is_train,
                 decay: float = 0.95,
                 epsilon: float = 1e-7):
        super(BatchNorm, self).__init__(name)
        self._size = size
        self._is_train = is_train
        self._decay = decay
        self._epsilon = epsilon

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
        self._variance = common.variable(
            'variance',
            init_value=init.Ones().build(
                name='variance_init',
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
            # Here, "tf.where" cannot be used since it will trigger both train and predict branches.
            # Any Tensors or Operations created outside of true_fn and false_fn will be executed regardless of which
            # branch is selected at runtime.
            return tf.cond(
                self._is_train,
                lambda: self._setup_for_train(x, None),
                lambda: self._setup_for_predict(x, None),
                name=name
            )

    def _setup_for_train(self, x, name):
        axes = tuple(range(len(x.shape) - 1))
        mean, variance = tf.nn.moments(x=x, axes=axes)

        d1 = self._decay
        d2 = 1.0 - d1
        update_mean = tf.assign(
            self._mean,
            tf.multiply(d1, self._mean) + tf.multiply(d2, mean)
        )
        update_variance = tf.assign(
            self._variance,
            tf.multiply(d1, self._variance) + tf.multiply(d2, variance)
        )
        with tf.control_dependencies([update_mean, update_variance]):
            mean = tf.identity(mean)
            variance = tf.identity(variance)

        return tf.nn.batch_normalization(
            x=x,
            mean=mean,
            variance=variance,
            offset=self._beta,
            scale=self._gamma,
            variance_epsilon=self._epsilon,
            name=name
        )

    def _setup_for_predict(self, x, name):
        return tf.nn.batch_normalization(
            x=x,
            mean=tf.stop_gradient(self._mean),
            variance=tf.stop_gradient(self._variance),
            offset=self._beta,
            scale=self._gamma,
            variance_epsilon=self._epsilon,
            name=name
        )


class InstanceNorm(common.Trainable):

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
        x = (x - mean) / tf.sqrt(var + self._eps)
        x = tf.reshape(x, shape)
        x = tf.add(self._w * x, self._b, name=name)
        return x


class LayerNorm(common.Trainable):

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
        x = (x - mean) / tf.sqrt(var + self._eps)
        x = tf.add(self._w * x, self._b, name=name)
        return x


class GroupNorm(common.Trainable):

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
        x = (x - mean) / tf.sqrt(var + self._eps)
        x = tf.reshape(x, shape)
        x = tf.add(self._w * x, self._b, name=name)
        return x
