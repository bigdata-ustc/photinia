#!/usr/bin/env python3

"""
@author: xi
@since: 2017-03
"""

import tensorflow as tf

from . import config


def lrelu(x, leak=1e-2):
    """Leak relu activation.

    :param x: Input tensor.
    :param leak: Leak. Default is 1e-2.
    :return: Output tensor.
    """
    return tf.maximum(x, leak * x)


def swish(x):
    return tf.nn.sigmoid(x) * x


def random_gumbel(shape,
                  mu=0.0,
                  beta=1.0,
                  dtype=config.D_TYPE,
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


def setup(x,
          widget_list):
    """Setup a series of widgets/ops with the given input "x".

    :param x: The input tensor.
    :param widget_list: List of widgets/ops.
    :return: The output form the last widget/op.
    """
    if not isinstance(widget_list, (list, tuple)):
        widget_list = [widget_list]
    y = x
    for w in widget_list:
        if callable(w):
            #
            # Note that Widget is also callable.
            y = w(y)
        elif isinstance(w, (tuple, list)):
            if len(w) > 0:
                fn = w[0]
                args = w[1:]
                y = fn(y, *args)
            else:
                continue
        elif w is None:
            continue
        else:
            raise ValueError('%s is not callable.' % str(w))
    return y


def transpose_sequence(seq,
                       seq_axis=1):
    """Transpose a batch of sequence, i.e., exchange the batch axis and the sequence axis.
    By default, the sequence axis is 1.

    :param seq: Tensor shaped (batch_size, seq_length, ...).
    :param seq_axis: The sequence axis. Default is 1.
    :return: Tensor shaped (seq_length, batch_size, ...).
    """
    perm = [i for i in range(len(seq.shape))]
    perm[0], perm[seq_axis] = seq_axis, 0
    return tf.transpose(seq, perm)


def setup_sequence(seq,
                   widget_list,
                   transpose_in=False,
                   transpose_out=False,
                   cell=None,
                   init_state=None):
    """Setup a series of widgets/ops with the given sequence "seq".

    :param seq: Tensor represents a sequence.
    :param widget_list: List of widgets/ops.
    :param transpose_in: Should the sequence tensor be transposed before setup? Default is False.
    :param transpose_out: Should the sequence tensor be transposed after setup? Default is False.
    :param cell: The recurrent cell. If this is not None, the sequence will be setup in a recurrent way.
        Default is None.
    :param init_state: The initial state of the recurrent cell.
    :return: The output sequence.
    """
    if transpose_in:
        seq = transpose_sequence(seq)
    if cell is None:
        y = tf.map_fn(
            fn=lambda elem: setup(elem, widget_list),
            elems=seq
        )
    else:
        if init_state is None:
            batch_size = tf.shape(seq)[1]
            state_size = cell.state_size
            init_state = tf.zeros(
                shape=(batch_size, state_size),
                dtype=config.D_TYPE
            )
        y = tf.scan(
            fn=lambda acc, elem: cell.setup(setup(elem, widget_list), acc),
            elems=seq,
            initializer=init_state
        )
    if transpose_out:
        y = transpose_sequence(y)
    return y


def flatten(x):
    batch_size = tf.shape(x)[0]
    return tf.reshape(x, (batch_size, -1))
