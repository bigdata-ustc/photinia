#!/usr/bin/env python3

"""
@author: xi
@since: 2017-03
"""

import tensorflow as tf


def lrelu(x, leak=1e-2):
    """Leak relu activation.

    :param x: Input tensor.
    :param leak: Leak. Default is 1e-2.
    :return: Output tensor.
    """
    return tf.maximum(x, leak * x)


# def llrelu(x, leak=1e-2, max_value=6):
#     return tf.minimum(tf.maximum(leak * x, x), max_value + leak * x)


def random_gumbel(shape,
                  mu=0.0,
                  beta=1.0,
                  dtype=tf.float32,
                  seed=None,
                  name=None):
    """Outputs random values from a Gumbel distribution.
    
    :param shape: Output shape.
    :param mu: mu.
    :param beta: beta.
    :param dtype: Data type.
    :param seed: Random seed.
    :param name: Op name.
    :return: A tensor of the specified shape filled with random Gumbel values.
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


def kl_normal(mu0, var0,
              mu1=0.0, var1=1.0):
    """KL divergence for normal distribution.
    Note that this is a simple version. We don't use covariance matrix (∑) here. Instead, 
    var is the vector that indicates the elements in ∑'s main diagonal (diag(∑)).

    :param mu0: μ0.
    :param var0: diag(∑0).
    :param mu1: μ1.
    :param var1: diag(∑1).
    :return: The KL divergence.
    """
    e = 1e-4
    var0 += e
    if mu1 == 0.0 and var1 == 1.0:
        kl = var0 + mu0 ** 2 - 1 - tf.log(var0)
    else:
        var1 += e
        kl = var0 / var1 + (mu0 - mu1) ** 2 / var1 - 1 - tf.log(var0 / var1)
    kl = 0.5 * tf.reduce_sum(kl, 1)
    return kl


def clip_gradient(pair_list,
                  max_norm):
    """Perform gradient clipping.
    If the gradients' global norm exceed 'max_norm', then shrink it to 'max_norm'.
    
    :param pair_list: (grad, var) pair list.
    :param max_norm: The max global norm.
    :return: (grad, var) pair list, the original gradients' norm, the clipped gradients' norm
    """
    grad_list = [grad for grad, _ in pair_list]
    grad_list, raw_grad = tf.clip_by_global_norm(grad_list, max_norm)
    grad = tf.global_norm(grad_list)
    pair_list = [(grad, pair[1]) for grad, pair in zip(grad_list, pair_list)]
    return pair_list, raw_grad, grad
