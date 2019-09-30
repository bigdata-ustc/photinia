#!/usr/bin/env python3


"""
@author: xi
@since: 2016-11-11
"""

import collections
import os
import sys

import numpy as np

from .. import conf
from .. import io
from .. import ops
from ..conf import tf


class __GlobalContext(object):

    def __init__(self):
        self._session_config = tf.ConfigProto()
        self._session_config.gpu_options.allow_growth = True
        self._session = None

    # def __del__(self):
    #     if self._session is not None:
    #         self._session.close()

    @property
    def session_config(self):
        return self._session_config

    @property
    def session(self):
        if self._session is None:
            self._session = tf.Session(config=self._session_config)
        return self._session


__GLOBAL = __GlobalContext()

TF_LOG_ALL = '0'
TF_LOG_NO_INFO = '1'
TF_LOG_NO_WARN = '2'
TF_LOG_NONE = '3'


def get_log_level():
    return os.environ['TF_CPP_MIN_LOG_LEVEL']


def set_log_level(level):
    if level not in ('0', '1', '2', '3'):
        raise ValueError(
            'level should be one of {'
            'TF_LOG_ALL, '
            'TF_LOG_NO_INFO, '
            'TF_LOG_NO_WARN, '
            'TF_LOG_NONE}.'
        )
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = level


def get_session_config():
    return __GLOBAL.session_config


def get_session():
    return __GLOBAL.session


def initialize_global_variables():
    __GLOBAL.session.run(tf.global_variables_initializer())


def get_name_scope():
    return tf.get_default_graph().get_name_scope()


def get_full_name(name):
    scope = get_name_scope()
    return f'{scope}/{name}' if scope else name


def deprecated(message):
    def _decorator(fn):
        def _fn(*args, **kwargs):
            print(message, file=sys.stderr)
            return fn(*args, **kwargs)

        return _fn

    return _decorator


_VAR_DICT = {}
_PLACEHOLDER_DICT = {}
_TRAINABLE_DICT = {}


def variable(name,
             init_value,
             dtype,
             trainable=True):
    full_name = get_full_name(name)
    try:
        op = tf.get_default_graph().get_operation_by_name(full_name)
        outputs = op.outputs
        return outputs[0] if len(outputs) == 1 else outputs
    except KeyError:
        pass
    if full_name in _VAR_DICT:
        instance = _VAR_DICT[full_name]
        # if type(instance) is not tf.Variable:
        #     raise TypeError()
    else:
        instance = tf.Variable(
            initial_value=init_value,
            dtype=dtype,
            name=name,
            trainable=trainable
        )
        _VAR_DICT[full_name] = instance
    return instance


def placeholder(name,
                shape,
                dtype=conf.dtype,
                default_value=None):
    """Create a placeholder.
    Shortcut to "tf.placeholder()".

    Args:
        name (str): Name of the placeholder.
        shape (tuple|list): The shape of the tensor to be fed (optional). If the shape is not specified,
            you can feed a tensor of any shape.
        dtype (tf.DType): The type of elements in the tensor to be fed.
        default_value: A `Tensor`. The default value to produce when output is not fed.

    Returns:
        The placeholder tensor.

    """
    full_name = get_full_name(name)
    try:
        op = tf.get_default_graph().get_operation_by_name(full_name)
        outputs = op.outputs
        return outputs[0] if len(outputs) == 1 else outputs
    except KeyError:
        pass
    if full_name in _PLACEHOLDER_DICT:
        instance = _PLACEHOLDER_DICT[full_name]
    else:
        if default_value is None:
            instance = tf.placeholder(name=name, shape=shape, dtype=dtype)
        else:
            instance = tf.placeholder_with_default(input=default_value, shape=shape, name=name)
        _PLACEHOLDER_DICT[full_name] = instance
    return instance


class _TrainableType(type):

    def __call__(cls, name, *args, **kwargs):
        scope = get_name_scope()
        full_name = f'{scope}/{name}' if scope else name
        prefix = full_name + '/'
        if full_name in _TRAINABLE_DICT:
            instance = _TRAINABLE_DICT[full_name]
            if type(instance) is not cls:
                raise TypeError()
        else:
            instance = super(_TrainableType, cls).__call__(
                (name, scope, full_name, prefix),
                *args,
                **kwargs
            )
            setattr(instance, '__init_args__', args)
            setattr(instance, '__init_kwargs__', kwargs)
            _TRAINABLE_DICT[full_name] = instance
            if hasattr(instance, '_build'):
                _build = getattr(instance, '_build')
                with tf.name_scope(prefix):
                    _build()
        return instance


class Trainable(object, metaclass=_TrainableType):

    def __init__(self, name, *args, **kwargs):
        self._name, self._scope, self._full_name, self._prefix = name
        if hasattr(self, '_setup'):
            self._setup_impl = getattr(self, '_setup')
            self.setup = self._setup_wrapper
        else:
            self._setup_impl = self.setup
            self.setup = self._setup_wrapper

    def setup(self, *args, **kwargs):
        pass

    def _setup_wrapper(self, *args, **kwargs):
        with tf.name_scope(self._prefix):
            return self._setup_impl(*args, **kwargs)

    def get_variables(self):
        """Get variables(tensors) of the widget.

        Returns:
            list: List of variables.

        """
        if self._name is None:
            return list()
        prefix = self._prefix
        global_vars = tf.global_variables()
        return [var for var in global_vars if var.name.startswith(prefix)]

    def get_trainable_variables(self):
        """Get variables(tensors that marked as "trainable") of the widget.

        Returns:
            list: List of variables.

        """
        if self._name is None:
            return list()
        trainable_vars = tf.trainable_variables()
        return [var for var in trainable_vars if var.name.startswith(self._prefix)]

    @property
    def full_name(self):
        """Get the full name of the widget.
        E.g., model/layers/layer1
        The full name does not contain "/" character.

        Returns:
            str: Full name of the widget.

        """
        return self._full_name

    @property
    def prefix(self):
        """Get the prefix of the widget.
        E.g., model/layers/layer1/
        The prefix always ends with a "/" character.

        Returns:
            str: Prefix of the widget.

        """
        return self._prefix

    def get_parameters(self):
        """Get parameter values of the widget.

        Returns:
            dict[str, np.ndarray]: Name to value dictionary of the parameters.

        """
        var_list = self.get_trainable_variables()
        param_dict = {var.name: var for var in var_list}
        param_dict = get_session().run(param_dict)
        return param_dict

    def set_parameters(self, param_dict, strict=True):
        """Set values to the parameters.

        Args:
            param_dict (dict[str, np.ndarray]): Name to value dictionary.
            strict (bool): If strict is True, all values in the dictionary must be used to assigned to the
                associated parameter, or an error will be risen.

        Raises:
            ValueError: If strict is True and there are some values in the dictionary unused.

        """
        var_list = self.get_trainable_variables()
        var_dict = {var.name: var for var in var_list}
        session = get_session()
        for name, value in param_dict.items():
            name_replace = name.replace('\\', '/')
            if name_replace not in var_dict:
                if strict:
                    raise ValueError('%s is not in this model.' % name)
            var = var_dict[name_replace]
            var.load(value, session=session)

    def dump(self, name, dumper=None):
        """Dump the model. (Save all trainable variables)

        Args:
            name (str): Model name.
                If the "dumper" argument is None, "name" is the path of the model file.
            dumper (dumpers.ModelDumper): Model dumper.

        """
        if dumper is None:
            io.dump_model_as_file(self, name)
        else:
            dumper.dump(self, name)

    def load(self, name, path=None, strict=True, dumper=None):
        """Load the model.

        Args:
            name (str): Model name.
            path (str): The path would like to be loaded into the target widget.
            strict (bool):  Strict mode.
            dumper (dumpers.ModelDumper): Model dumper.

        """
        if dumper is None:
            io.load_model_from_file(self, name, path, strict)
        else:
            dumper.load(self, name, path, strict)

    def get_operation(self, name):
        name = self._prefix + name
        try:
            return tf.get_default_graph().get_operation_by_name(name)
        except KeyError:
            return None

    def get_tensor(self, name):
        if name.rfind(':') == -1:
            name = '%s%s:0' % (self._prefix, name)
        else:
            name = self._prefix + name
        try:
            return tf.get_default_graph().get_tensor_by_name(name)
        except KeyError:
            return None

    def get_variable(self, name):
        if name.rfind(':') == -1:
            name = '%s%s:0' % (self._prefix, name)
        else:
            name = self._prefix + name
        for var in tf.global_variables():
            if name == var.name:
                return var
        return None

    def __getitem__(self, name):
        name = self._prefix + name
        if name in _TRAINABLE_DICT:
            instance = _TRAINABLE_DICT[name]
            if isinstance(instance, Trainable):
                return instance

        if name.rfind(':') == -1:
            name += ':0'
        try:
            return tf.get_default_graph().get_tensor_by_name(name)
        except KeyError:
            return None


class Model(Trainable):
    pass


class Widget(Trainable):
    pass


def setup(x, widget_list):
    """Setup a series of widgets/ops with the given input "x".

    Args:
        x: The input tensor.
        widget_list (list): List of widgets/ops.

    Returns:
        Output tensor.

    """
    if widget_list is None:
        return x
    if not isinstance(widget_list, (list, tuple)):
        widget_list = [widget_list]
    y = x
    for w in widget_list:
        if callable(w):
            #
            # Note that Widget is also callable.
            y = w(y)
        elif isinstance(w, (tuple, list)):
            if len(w) != 2:
                raise ValueError('The tuple must have two elements.')
            fn = w[0]
            if not callable(fn):
                raise ValueError('%s is not callable.' % str(fn))
            if isinstance(w[1], dict):
                kwargs = w[1]
                y = fn(y, **kwargs)
            elif isinstance(w[1], str):
                y = fn(y, name=w[1])
            elif w[1] is None:
                y = fn(y)
            else:
                raise ValueError('The second term of the tuple must be str or dict.')
        elif isinstance(w, str):
            tf.identity(y, name=w)
        elif w is None:
            continue
        else:
            raise ValueError('%s is not callable.' % str(w))
    return y


def setup_sequence(seq, widget_list):
    """Setup a series of widgets/ops with the given sequence "seq".

    Args:
        seq: Tensor represents a sequence shaped (batch_size, seq_length, ...).
        widget_list (list): List of widgets/ops.

    Returns:
        tf.Tensor: Output tensor.

    """
    seq = ops.transpose_sequence(seq)
    y = tf.map_fn(
        fn=lambda elem: setup(elem, widget_list),
        elems=seq
    )
    y = ops.transpose_sequence(y)
    return y


class Step(object):
    """Train step.
    Trainable is trained and tested step by step~
    """

    def __init__(self,
                 inputs=None,
                 outputs=None,
                 updates=None,
                 givens=None,
                 callbacks=None):
        """A slot object is a callable which accepts multiple tensor inputs
        and gives out multiple outputs.

        Args:
            inputs (list[tf.Tensor]|tuple[tf.Tensor]|tf.Tensor):
                Input tensor(s).
            outputs (dict[str, tf.Tensor]|list[tf.Tensor]|tuple[tf.Tensor]|tf.Tensor):
                Output tensor(s).
            updates (list[tf.Operation]|tuple[tf.Operation]|tf.Operation):
                Operation(s) when invoked. These are usually generated by optimizers.
            givens (dict[tf.Tensor, Any]):
                Preset values for some placeholder, e.g., the keep_prob value for dropout.
            callbacks (list[(Any) -> None]|tuple[(Any) -> None]|(Any) -> None): Callback(s)

        """
        self._session = get_session()
        #
        # Inputs.
        if inputs is None:
            inputs = ()
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        self._inputs = inputs
        #
        # Outputs.
        if outputs is None:
            outputs = ()
        if not isinstance(outputs, (tuple, list)) \
                and not isinstance(outputs, (dict, collections.OrderedDict)):
            outputs = (outputs,)
        self._outputs = outputs
        #
        # Updates.
        if updates is None:
            updates = ()
        if not isinstance(updates, (tuple, list)):
            updates = (updates,)
        self._updates = updates
        #
        # Givens.
        if givens is None:
            givens = {}
        if not isinstance(givens, dict):
            raise ValueError('Givens must be dict.')
        self._givens = givens
        #
        # Callbacks.
        if callbacks is None:
            callbacks = ()
        if not isinstance(callbacks, (tuple, list)):
            callbacks = (callbacks,)
        self._callbacks = callbacks
        #
        self._feed_dict = givens.copy()
        self._fetches = (outputs, updates)
        if len(outputs) == 0 and len(updates) == 0:
            raise ValueError('At least one output or update should be set.')

    @property
    def outputs(self):
        return self._outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def updates(self):
        return self._updates

    @property
    def givens(self):
        return self._givens

    def __call__(self, *args):
        #
        # Check input length.
        if len(args) != len(self._inputs):
            print(len(args), len(self._inputs))
            raise ValueError('The count of parameters is not match the inputs.')
        #
        # Make "feed_dict".
        for index, placeholder_ in enumerate(self._inputs):
            self._feed_dict[placeholder_] = args[index]
        #
        # Run the graph on the session.
        ret = self._session.run(fetches=self._fetches, feed_dict=self._feed_dict)[0]
        for callback in self._callbacks:
            callback(ret)
        return ret
