#!/usr/bin/env python3

"""
@author: xi
@since: 2017-03
"""

import tensorflow as tf

from .. import conf


def log(x, eps=1e-7, name=None):
    """log operation with smooth.

    Args:
        x: Input tensor.
        eps (float): Smooth factor.
        name (str): Operation name.

    Returns:
        Output tensor.

    """
    return tf.log(x + eps, name=name)


def lrelu(x, leak=1e-3, name=None):
    """Leaky ReLU activation function.

    f(x) =        x     , x >= 0,
           leak * x     , x < 0

    Args:
        x: Input tensor.
        leak (float): Leak value. Default is 1e-3.
        name (str): Operation name.

    Returns:
        Output tensor.

    """
    return tf.maximum(x, leak * x, name=name)


def swish(x, name=None):
    """Swish activation function.

    f(x) = x * sigmoid(x)

    Args:
        x: Input tensor.
        name (str): Operation name.

    Returns:
        Output tensor.

    """
    return tf.multiply(tf.nn.sigmoid(x), x, name=name)


def random_gumbel(shape,
                  mu=0.0,
                  beta=1.0,
                  dtype=conf.dtype,
                  seed=None,
                  name=None):
    """Outputs random values from a Gumbel distribution.

    Args:
        shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
        mu (float): mu.
        beta (float): brta.
        dtype: Data type.
        seed (int): Random seed.
        name (str): Operation name.

    Returns:
        Output tensor.

    """
    u = tf.random_uniform(
        shape=shape,
        minval=0,
        maxval=1,
        dtype=dtype,
        seed=seed,
        name=name
    )
    g = -tf.log(-tf.log(u))
    g = mu + g * beta
    return g


def kl_normal(mu0,
              var0,
              mu1=0.0,
              var1=1.0,
              name=None):
    """KL divergence for normal distribution.
    Note that this is a simple version. We don't use covariance matrix (∑) here. Instead,
    var is the vector that indicates the elements in ∑'s main diagonal (diag(∑)).

    Args:
        mu0: mu0.
        var0: diag(∑0).
        mu1: mu1.
        var1: diag(∑1).
        name (str): Operation name.

    Returns:
        Output tensor.

    """
    e = 1e-4
    var0 += e
    if mu1 == 0.0 and var1 == 1.0:
        kl = var0 + mu0 ** 2 - 1 - tf.log(var0)
    else:
        var1 += e
        kl = var0 / var1 + (mu0 - mu1) ** 2 / var1 - 1 - tf.log(var0 / var1)
    kl = tf.multiply(0.5, tf.reduce_sum(kl, 1), name=name)
    return kl


def clip_gradient(pair_list, max_norm):
    """Perform gradient clipping.
    If the gradients' global norm exceed 'max_norm', then shrink it to 'max_norm'.

    Args:
        pair_list (list): (grad, var) pair list.
        max_norm: The max global norm.

    Returns:
        Output tensor.
            (grad, var) pair list, the original gradients' norm, the clipped gradients' norm.

    """
    grad_list = [grad for grad, _ in pair_list]
    grad_list, raw_grad = tf.clip_by_global_norm(grad_list, max_norm)
    grad = tf.global_norm(grad_list)
    pair_list = [(grad, pair[1]) for grad, pair in zip(grad_list, pair_list)]
    return pair_list, raw_grad, grad


def transpose_sequence(seq, seq_axis=1, name=None):
    """Transpose a batch of sequence, i.e., exchange the batch axis and the sequence axis.
    By default, the sequence axis is 1.

    Args:
        seq: Tensor shaped (batch_size, seq_length, ...).
        seq_axis: The sequence axis. Default is 1.
        name (str): Operation name.

    Returns:
        Output tensor. Tensor shaped (seq_length, batch_size, ...).

    """
    perm = [i for i in range(len(seq.shape))]
    perm[0], perm[seq_axis] = seq_axis, 0
    return tf.transpose(seq, perm, name=name)


def flatten(x):
    batch_size = tf.shape(x)[0]
    return tf.reshape(x, (batch_size, -1))


def sequence_length(seq):
    used = tf.sign(tf.reduce_max(tf.abs(seq), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def last_elements(seq, seq_len):
    h, _ = tf.map_fn(
        fn=lambda elem: (elem[0][elem[1] - 1], elem[1]),
        elems=(seq, seq_len)
    )
    return h


def variance(x, axis=-1):
    mu = tf.reduce_mean(x, axis=axis)
    return tf.reduce_mean(x ** 2) - mu ** 2


def skewness(x, axis=-1, epsilon=1e-5):
    mu = tf.reduce_mean(x, axis=axis, keep_dims=True)
    up = tf.reduce_mean((x - mu) ** 3, axis=axis)
    down = tf.reduce_mean((x - mu) ** 2, axis=axis)
    down = tf.sqrt(down) ** 3 + epsilon
    return up / down
