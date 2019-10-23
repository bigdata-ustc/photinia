#!/usr/bin/env python3

"""
@author: xi, anmx
@since: 2017-04-23
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import photinia as ph


def one_hot(index, dims, dtype=np.uint8):
    """Create one hot vector(s) with the given index(indices).

    :param index: int or list(tuple) of int. Indices.
    :param dims: int. Dimension of the one hot vector.
    :param dtype: Numpy data type.
    :return: Numpy array. If index is an int, then return a (1 * dims) vector,
        else return a (len(index), dims) matrix.
    """
    if isinstance(index, int):
        ret = np.zeros((dims,), dtype)
        ret[index] = 1
    elif isinstance(index, (list, tuple)):
        seq_len = len(index)
        ret = np.zeros((seq_len, dims), dtype)
        ret[range(seq_len), index] = 1.0
    else:
        raise ValueError('index should be int or list(tuple) of int.')
    return ret


def get_trainable_variables(include, exclude=None):
    if isinstance(include, ph.Module):
        include = [include]
    if isinstance(exclude, ph.Module):
        exclude = [exclude]
    exclude_prefix = [w.prefix for w in exclude]
    tvars = []
    for w in include:
        for tvar in w.get_trainable_variables():
            add = True
            name = tvar.name
            for prefix in exclude_prefix:
                if name.startswith(prefix):
                    add = False
                    break
            if add:
                tvars.append(tvar)
    return tvars


def read_variables(var_or_list):
    """Get the value from a variable.

    :param var_or_list: tf.Variable.
    :return: numpy.array value.
    """
    session = ph.get_session()
    return session.run(var_or_list)


def write_variables(var_or_list, values):
    """Set the value to a variable.

    :param var_or_list: tf.Variable.
    :param values: numpy.array value.
    """
    session = ph.get_session()
    if isinstance(var_or_list, (tuple, list)):
        for var, value in zip(var_or_list, values):
            var.load(value, session)
    else:
        var_or_list.load(values, session)


def get_operation(name):
    return tf.get_default_graph().get_operation_by_name(name)


def get_tensor(name):
    """Get tensor by name.

    https://stackoverflow.com/questions/37849322/how-to-understand-the-term-tensor-in-tensorflow

    TensorFlow doesn't have first-class Tensor objects, meaning that there are no notion of Tensor in the
    underlying graph that's executed by the runtime.
    Instead the graph consists of op nodes connected to each other, representing operations.
    An operation allocates memory for its outputs, which are available on endpoints :0, :1, etc,
    and you can think of each of these endpoints as a Tensor.
    If you have tensor corresponding to nodename:0 you can fetch its value as sess.run(tensor) or
    sess.run('nodename:0').
    Execution granularity happens at operation level, so the run method will execute op which will compute all of the
    endpoints, not just the :0 endpoint.
    It's possible to have an Op node with no outputs (like tf.group) in which case there are no tensors
    associated with it.
    It is not possible to have tensors without an underlying Op node.

    :param name: Tensor name (must be full name).
    :return: The tensor.
    """
    if name.rfind(':') == -1:
        name += ':0'
    return tf.get_default_graph().get_tensor_by_name(name)


def get_variable(name):
    if name.rfind(':') == -1:
        name += ':0'
    for var in tf.global_variables():
        if name == var.name:
            return var
    return None


def get_basename(name):
    index = name.rfind('/')
    index_ = name.rfind(':')
    if index_ == -1:
        index_ = len(name)
    return name[index + 1: index_]


def dump_graph(names_dest_nodes, name_init_op=None):
    """Dump a sub graph

    Args:
        names_dest_nodes (list[str]|tuple[str]): Name of the destination (operation) nodes.
        name_init_op (str): Name of the init operation of the variables.

    Returns:
        GraphDef.

    """
    if not isinstance(names_dest_nodes, (list, tuple)):
        names_dest_nodes = [names_dest_nodes]

    graph = tf.get_default_graph().as_graph_def()
    if name_init_op is not None:
        sub_graph_def = tf.graph_util.extract_sub_graph(graph, names_dest_nodes)
        names_in_graph = set(_node.name for _node in sub_graph_def.node)
        var_list = [
            _var for _var in tf.global_variables()
            if _var.op.name in names_in_graph
        ]
        value_list = read_variables(var_list)
        assign_list = [
            tf.assign(_var, _value)
            for _var, _value in zip(var_list, value_list)
        ]
        with tf.control_dependencies(assign_list):
            init_op = tf.no_op(name=name_init_op)
        names_dest_nodes.append(init_op.name)

    return tf.graph_util.extract_sub_graph(graph, names_dest_nodes)


def write_graph_def(pb_file, graph_def):
    """Write GraphDef object to pb file.

    Args:
        pb_file (str): Full path of the prorobuf file.
        graph_def: GraphDef object.

    """
    with gfile.GFile(pb_file, 'wb') as f:
        f.write(graph_def.SerializeToString())
