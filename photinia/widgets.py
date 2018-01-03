#!/usr/bin/env python3

"""
@author: xi
@since: 2016-11-11
"""

import math

import tensorflow as tf

from . import config
from . import initializers
from . import ops


def variable(name, initial_value, trainable=True):
    return tf.Variable(
        name=name,
        initial_value=initial_value,
        trainable=trainable,
        dtype=config.D_TYPE
    )


def placeholder(name, shape, dtype=config.D_TYPE):
    return tf.placeholder(name=name, shape=shape, dtype=dtype)


class Widget(object):
    """Widget
    The basic component to form a model.
    This an abstract class which can only be inherited.
    """

    def __init__(self,
                 name=None,
                 build=True):
        """Construct a widget.

        :param name: Name.
            If the widget has variable that wants to be trained, the name must be given.
        """
        if name is not None:
            if not isinstance(name, str):
                raise ValueError('Widget name must be specified with string.')
            if len(name.strip()) != len(name) or name == '':
                raise ValueError('Widget name cannot be empty or contain space characters.')
        self._scope = ''
        self._name = name
        self._built = False
        if build:
            self.build()

    @property
    def name(self):
        return self._name

    @property
    def built(self):
        return self._built

    def build(self):
        """Build the widget.
        The main purpose of this function is to create the trainable variables (parameters) for the widget.

        :return: None.
        """
        if self._built:
            return self
        else:
            if self._name is None:
                #
                # Build WITHOUT scope.
                self._build()
                self._built = True
                return self
            else:
                #
                # Build WITH scope.
                self._scope = tf.get_variable_scope().name
                with tf.variable_scope(self._name):
                    self._build()
                    self._built = True
                    return self

    def _build(self):
        """Build the widget.
        Abstract method.
        All subclass must implement this method.

        There is one task to be done in this method:
        1) Create the parameters (trainable variables) for the widget.
        """
        raise NotImplementedError()

    def setup(self, *args, **kwargs):
        """Setup the widget.
        "Setup" means to create a new series of operator in the TF graph, which can be called a "path".
        No matter how many paths be created, the number of trainable variables is (and of course cannot) be changed.
        They share the same parameters of the widget.

        :param args:
        :param kwargs:
        :return:
        """
        if not self._built:
            raise RuntimeError('This widget has not been built. Please build first.')
        if self._name is None:
            #
            # Setup only WITHOUT scope.
            return self._setup(*args, **kwargs)
        else:
            #
            # Setup only WITH scope.
            with tf.variable_scope(self._name):
                return self._setup(*args, **kwargs)

    def _setup(self, *args, **kwargs):
        """Setup the widget.
        Abstract method.
        All subclass must implement this method.

        There is one task to be done in this method:
        1) Construct the model's graph structure with TF.

        In this method, you CANNOT create any trainable variables.

        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.setup(*args, **kwargs)

    def variables(self):
        if self._name is None:
            return []
        prefix = self.prefix()
        global_vars = tf.global_variables()
        return [var for var in global_vars if var.name.startswith(prefix)]

    def trainable_variables(self):
        if self._name is None:
            return []
        prefix = self.prefix()
        trainable_vars = tf.trainable_variables()
        return [var for var in trainable_vars if var.name.startswith(prefix)]

    def scope(self):
        return self.prefix()

    def prefix(self):
        if self._scope == '':
            return self._name + '/'
        return self._scope + '/' + self._name + '/'


class Linear(Widget):
    """Linear layer.
    y = wx + b
    """

    def __init__(self,
                 name,
                 input_size,
                 output_size,
                 with_bias=True,
                 weight_initializer=initializers.GlorotUniform(),
                 bias_initializer=initializers.Zeros()):
        """Construct the linear layer.

        :param name: Name.
        :param input_size: Input size.
        :param output_size: Output size.
        :param with_bias: If the widget has a bias variable.
        """
        self._input_size = input_size
        self._output_size = output_size
        self._with_bias = with_bias
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        super(Linear, self).__init__(name)

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def with_bias(self):
        return self._with_bias

    def _build(self):
        """Build the linear layer.
        Two parameters: weight and bias.

        :return: None.
        """
        self._w = tf.Variable(
            self._weight_initializer.build(
                shape=(self._input_size, self._output_size)
            ),
            dtype=config.D_TYPE,
            name='w'
        )
        self._b = tf.Variable(
            self._bias_initializer.build(
                shape=(self._output_size,)
            ),
            dtype=config.D_TYPE,
            name='b'
        ) if self._with_bias else None

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

    def _setup(self, x, axes=None):
        """Setup the linear layer.

        :param x: Input tensor.
        :param axes: Axes. If x is a tensor, the layer will perform tensor dot.
        :return: Output tensor.
        """
        y = tf.matmul(x, self._w) if axes is None else tf.tensordot(x, self._w, axes=axes)
        if self._with_bias:
            y += self._b
        return y


class Dropout(Widget):
    """Dropout
    """

    def __init__(self, name, keep_prob=None):
        self._keep_prob = keep_prob
        super(Dropout, self).__init__(name)

    @property
    def keep_prob(self):
        return self._keep_prob

    def _build(self):
        if self._keep_prob is None:
            self._keep_prob = tf.placeholder(
                shape=(),
                dtype=config.D_TYPE
            )

    def _setup(self, x):
        return tf.nn.dropout(x, self._keep_prob)


class Conv2D(Widget):
    """2D convolutional layer.
    """

    def __init__(self,
                 name,
                 input_size,
                 output_channels,
                 filter_height=3,
                 filter_width=3,
                 stride_height=1,
                 stride_width=1,
                 data_format='NHWC',
                 kernel_initializer=initializers.TruncatedNormal(),
                 bias_initializer=initializers.Zeros(),
                 flat_output=False):
        if not (isinstance(input_size, (tuple, list)) and len(input_size) == 3):
            raise ValueError('input_size should be tuple or list with 3 elements.')
        self._input_height = input_size[0]
        self._input_width = input_size[1]
        self._input_channels = input_size[2]
        self._output_channels = output_channels
        self._filter_height = filter_height
        self._filter_width = filter_width
        self._stride_height = stride_height
        self._stride_width = stride_width
        self._data_format = data_format
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._flat_output = flat_output
        #
        self._output_height = math.ceil(self._input_height / stride_height)
        self._output_width = math.ceil(self._input_width / stride_width)
        self._flat_size = self._output_height * self._output_width * output_channels
        super(Conv2D, self).__init__(name)

    @property
    def input_size(self):
        return self._input_height, self._input_width

    @property
    def input_height(self):
        return self._input_height

    @property
    def input_width(self):
        return self._input_width

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def output_height(self):
        return self._output_height

    @property
    def output_width(self):
        return self._output_width

    @property
    def output_size(self):
        return self._output_height, self._output_width, self._output_channels

    @property
    def output_channels(self):
        return self._output_channels

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def stride_height(self):
        return self._stride_height

    @property
    def stride_width(self):
        return self._stride_width

    @property
    def flat_output(self):
        return self._flat_output

    @flat_output.setter
    def flat_output(self, flat_output):
        self._flat_output = flat_output

    @property
    def flat_size(self):
        return self._flat_size

    def _build(self):
        self._w = tf.Variable(
            self._kernel_initializer.build(
                shape=(
                    self._filter_height,
                    self._filter_width,
                    self._input_channels,
                    self._output_channels
                )
            ),
            dtype=config.D_TYPE,
            name='w'
        )
        self._b = tf.Variable(
            self._bias_initializer.build(
                shape=(self._output_channels,)
            ),
            dtype=config.D_TYPE,
            name='b'
        )

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

    def _setup(self, x):
        y = tf.nn.conv2d(
            input=x,
            filter=self._w,
            strides=[1, self._stride_height, self._stride_width, 1],
            padding='SAME',
            data_format=self._data_format
        ) + self._b
        if self._flat_output:
            y = tf.reshape(y, (-1, self._flat_size))
        return y


class Pool2D(Widget):

    def __init__(self,
                 name,
                 input_size,
                 stride_height,
                 stride_width,
                 pool_type='max'):
        self._input_size = input_size
        self._stride_height = stride_height
        self._stride_width = stride_width
        pool_type = pool_type.lower()
        if pool_type not in {'max', 'avg'}:
            raise ValueError('pool_type should be one of {"max", "avg"}, '
                             'but got %s' % pool_type)
        self._pool_type = pool_type
        #
        self._input_height = input_size[0]
        self._input_width = input_size[1]
        self._input_channels = input_size[2]
        self._output_height = math.ceil(self._input_height / stride_height)
        self._output_width = math.ceil(self._input_width / stride_width)
        self._flat_size = self._output_height * self._output_width * self._input_channels
        super(Pool2D, self).__init__(name)

    @property
    def input_size(self):
        return self._input_size

    @property
    def input_height(self):
        return self._input_height

    @property
    def input_width(self):
        return self._input_width

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def output_size(self):
        return self._output_height, self._output_width, self._input_channels

    @property
    def output_height(self):
        return self._output_height

    @property
    def output_width(self):
        return self._output_width

    @property
    def flat_size(self):
        return self._flat_size

    def _build(self):
        pass

    def _setup(self, x):
        if self._pool_type == 'max':
            y = tf.nn.max_pool(
                value=x,
                ksize=[1, self._stride_height, self._stride_width, 1],
                strides=[1, self._stride_height, self._stride_width, 1],
                padding='SAME',
                data_format='NHWC'
            )
            return y
        elif self._pool_type == 'avg':
            y = tf.nn.avg_pool(
                value=x,
                ksize=[1, self._stride_height, self._stride_width, 1],
                strides=[1, self._stride_height, self._stride_width, 1],
                padding='SAME',
                data_format='NHWC'
            )
            return y


class Conv2DTrans(Widget):
    """ConvTransposeLayer
    """

    def __init__(self,
                 name,
                 output_size,
                 input_channels,
                 filter_height=3,
                 filter_width=3,
                 stride_height=2,
                 stride_width=2,
                 data_format='NHWC',
                 kernel_initializer=initializers.TruncatedNormal(),
                 bias_initializer=initializers.Zeros(),
                 flat_input=False):
        if not (isinstance(output_size, (tuple, list)) and len(output_size) == 3):
            raise ValueError('output_size should be tuple or list with 3 elements.')
        self._output_height = output_size[0]
        self._output_width = output_size[1]
        self._output_channels = output_size[2]
        self._input_channels = input_channels
        self._filter_height = filter_height
        self._filter_width = filter_width
        self._stride_height = stride_height
        self._stride_width = stride_width
        self._data_format = data_format
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._flat_input = flat_input
        #
        self._input_height = math.ceil(self._output_height / stride_height)
        self._input_width = math.ceil(self._output_width / stride_width)
        self._flat_size = self._input_height * self._input_width * input_channels
        super(Conv2DTrans, self).__init__(name)

    @property
    def input_size(self):
        return self._input_height, self._input_width

    @property
    def input_height(self):
        return self._input_height

    @property
    def input_width(self):
        return self._input_width

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def output_size(self):
        return self._output_height, self._output_width, self._output_channels

    @property
    def output_height(self):
        return self._output_height

    @property
    def output_width(self):
        return self._output_width

    @property
    def output_channels(self):
        return self._output_channels

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def stride_height(self):
        return self._stride_height

    @property
    def stride_width(self):
        return self._stride_width

    def _build(self):
        """Build the layer.
        Two parameters: filter (weight) and bias.

        :return: None.
        """
        self._w = tf.Variable(
            self._kernel_initializer.build(
                shape=(
                    self._filter_height,
                    self._filter_width,
                    self._output_channels,
                    self._input_channels
                )
            ),
            dtype=config.D_TYPE,
            name='w'
        )
        self._b = tf.Variable(
            self._bias_initializer.build(
                shape=(self._output_channels,)
            ),
            dtype=config.D_TYPE,
            name='b'
        )

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

    def _setup(self, x):
        """Setup the layer.

        :param x: Input tensor with "NHWC" format.
        :return: Output tensor with "NHWC" format.
        """
        input_shape = tf.shape(x)
        batch_size, input_height, input_width = input_shape[0], input_shape[1], input_shape[2]
        output_shape = (
            batch_size,
            input_height * self._stride_height,
            input_width * self._stride_width,
            self._output_channels
        )
        y = tf.nn.conv2d_transpose(
            value=x,
            filter=self._w,
            output_shape=output_shape,
            strides=[1, self._stride_height, self._stride_width, 1],
            padding='SAME',
            data_format='NHWC'
        ) + self._b
        return y


class GRUCell(Widget):
    """GRUCell
    """

    def __init__(self,
                 name,
                 input_size,
                 state_size,
                 activation=ops.lrelu,
                 weight_initializer=initializers.GlorotUniform(),
                 recurrent_initializer=initializers.GlorotUniform(),
                 bias_initializer=initializers.Zeros()):
        """Construct a cell.
        Does not create the parameters' tensors.

        :param name: Name.
        :param input_size: Input size.
        :param state_size: State size.
        """
        self._input_size = input_size
        self._state_size = state_size
        self._activation = activation
        self._weight_initializer = weight_initializer
        self._recurrent_initializer = recurrent_initializer
        self._bias_initializer = bias_initializer
        super(GRUCell, self).__init__(name)

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def _build(self):
        """Build the cell.
        The GRU cell is consists of 3 kinds of parameters:
        1) Update gate parameters (wz, uz, bz).
        2) Reset gate parameters (wr, ur, br).
        3) Activation parameters (wh, uh, bh).

        :return: None
        """
        self._wz = tf.Variable(
            self._weight_initializer.build(
                shape=(self._input_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='wz'
        )
        self._uz = tf.Variable(
            self._recurrent_initializer.build(
                shape=(self._state_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='uz'
        )
        self._bz = tf.Variable(
            self._bias_initializer.build(
                shape=(self._state_size,)
            ),
            dtype=config.D_TYPE,
            name='bz'
        )
        #
        self._wr = tf.Variable(
            self._weight_initializer.build(
                shape=(self._input_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='wr'
        )
        self._ur = tf.Variable(
            self._recurrent_initializer.build(
                shape=(self._state_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='ur'
        )
        self._br = tf.Variable(
            self._bias_initializer.build(
                shape=(self._state_size,)
            ),
            dtype=config.D_TYPE,
            name='br'
        )
        #
        self._wh = tf.Variable(
            self._weight_initializer.build(
                shape=(self._input_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='wh'
        )
        self._uh = tf.Variable(
            self._recurrent_initializer.build(
                shape=(self._state_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='uh'
        )
        self._bh = tf.Variable(
            self._bias_initializer.build(
                shape=(self._state_size,)
            ),
            dtype=config.D_TYPE,
            name='bh'
        )

    def _setup(self, x, prev_state):
        """Setup the cell.

        :param x: The input tensor.
        :param prev_state: Previous state tensor.
        :return: State tensor.
        """
        z = tf.sigmoid(tf.matmul(x, self._wz) + tf.matmul(prev_state, self._uz) + self._bz)
        r = tf.sigmoid(tf.matmul(x, self._wr) + tf.matmul(prev_state, self._ur) + self._br)
        lin_state = tf.matmul(x, self._wh) + tf.matmul(r * prev_state, self._uh) + self._bh
        state = self._activation(lin_state) if self._activation is not None else lin_state
        state = z * prev_state + (1.0 - z) * state
        return state

    @property
    def wz(self):
        return self._wz

    @property
    def uz(self):
        return self._uz

    @property
    def bz(self):
        return self._bz

    @property
    def wr(self):
        return self._wr

    @property
    def ur(self):
        return self._ur

    @property
    def br(self):
        return self._br

    @property
    def wh(self):
        return self._wh

    @property
    def uh(self):
        return self._uh

    @property
    def bh(self):
        return self._bh


class LSTMCell(Widget):
    """LSTMCell
    """

    def __init__(self,
                 name,
                 input_size,
                 state_size,
                 activation=ops.lrelu,
                 weight_initializer=initializers.GlorotUniform(),
                 recurrent_initializer=initializers.GlorotUniform(),
                 bias_initializer=initializers.Zeros()):
        self._input_size = input_size
        self._state_size = state_size
        self._activation = activation
        self._weight_initializer = weight_initializer
        self._recurrent_initializer = recurrent_initializer
        self._bias_initializer = bias_initializer
        super(LSTMCell, self).__init__(name)

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def _build(self):
        """Build the cell.
        The LSTM cell is consists of 4 kinds of parameters:
        1) Input gate parameters (wi, ui, bi).
        2) Forget gate parameters (wf, uf, bf).
        3) Output gate parameters (wo, uo, bo).
        4) Activation parameters (wc, uc, bc).

        :return: None
        """
        self._wi = tf.Variable(
            self._weight_initializer.build(
                shape=(self._input_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='wi'
        )
        self._ui = tf.Variable(
            self._recurrent_initializer.build(
                shape=(self._state_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='ui'
        )
        self._bi = tf.Variable(
            self._bias_initializer.build(
                shape=(self._state_size,)
            ),
            dtype=config.D_TYPE,
            name='bi'
        )
        #
        self._wf = tf.Variable(
            self._weight_initializer.build(
                shape=(self._input_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='wf'
        )
        self._uf = tf.Variable(
            self._recurrent_initializer.build(
                shape=(self._state_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='uf'
        )
        self._bf = tf.Variable(
            self._bias_initializer.build(
                shape=(self._state_size,)
            ),
            dtype=config.D_TYPE,
            name='bf'
        )
        #
        self._wo = tf.Variable(
            self._weight_initializer.build(
                shape=(self._input_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='wo'
        )
        self._uo = tf.Variable(
            self._recurrent_initializer.build(
                shape=(self._state_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='uo'
        )
        self._bo = tf.Variable(
            self._bias_initializer.build(
                shape=(self._state_size,)
            ),
            dtype=config.D_TYPE,
            name='bo'
        )
        #
        self._wc = tf.Variable(
            self._weight_initializer.build(
                shape=(self._input_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='wc'
        )
        self._uc = tf.Variable(
            self._recurrent_initializer.build(
                shape=(self._state_size, self._state_size)
            ),
            dtype=config.D_TYPE,
            name='uc'
        )
        self._bc = tf.Variable(
            self._bias_initializer.build(
                shape=(self._state_size,)
            ),
            dtype=config.D_TYPE,
            name='bc'
        )

    def _setup(self, x, prev_state, prev_output):
        """Setup the cell.

        :param x: Input tensor.
        :param prev_state: Previous cell state tensor.
        :param prev_output: Previous cell output tensor.
        :return: Tuple of cell state and cell output tensors.
        """
        # Input gate.
        i = tf.nn.sigmoid(tf.matmul(x, self._wi) + tf.matmul(prev_output, self._ui) + self._bi)
        # Forget gate.
        f = tf.nn.sigmoid(tf.matmul(x, self._wf) + tf.matmul(prev_output, self._uf) + self._bf)
        # Output gate.
        o = tf.nn.sigmoid(tf.matmul(x, self._wo) + tf.matmul(prev_output, self._uo) + self._bo)
        # Output and state.
        lin_state = tf.matmul(x, self._wc) + tf.matmul(prev_output, self._uc) + self._bc
        state = self._activation(lin_state) if self._activation is not None else lin_state
        state = f * prev_state + i * state
        output = o * state
        return state, output

    @property
    def wi(self):
        return self._wi

    @property
    def ui(self):
        return self._ui

    @property
    def bi(self):
        return self._bi

    @property
    def wf(self):
        return self._wf

    @property
    def uf(self):
        return self._uf

    @property
    def bf(self):
        return self._bf

    @property
    def wo(self):
        return self._wo

    @property
    def uo(self):
        return self._uo

    @property
    def bo(self):
        return self._bo

    @property
    def wc(self):
        return self._wc

    @property
    def uc(self):
        return self._uc

    @property
    def bc(self):
        return self._bc


class BatchNorm(Widget):
    """BatchNorm
    This class is incomplete. The usage for prediction stage is actually different. Be careful!
    """

    def __init__(self,
                 name,
                 size,
                 epsilon=1e-5):
        self._size = size
        self._epsilon = epsilon
        super(BatchNorm, self).__init__(name)

    @property
    def size(self):
        return self._size

    @property
    def input_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    @property
    def epsilon(self):
        return self._epsilon

    def _build(self):
        beta_init = tf.zeros(
            shape=self._size,
            dtype=config.D_TYPE
        )
        gamma_init = tf.ones(
            shape=self._size,
            dtype=config.D_TYPE
        )
        self._beta = tf.Variable(
            name='beta',
            initial_value=beta_init,
            dtype=config.D_TYPE
        )
        self._gamma = tf.Variable(
            name='gamma',
            initial_value=gamma_init,
            dtype=config.D_TYPE
        )

    def _setup(self, x):
        axes = tuple(range(len(x.get_shape()) - 1))
        mean, variance = tf.nn.moments(x=x, axes=axes)
        y = tf.nn.batch_normalization(
            x=x,
            mean=mean,
            variance=variance,
            offset=self._beta,
            scale=self._gamma,
            variance_epsilon=self._epsilon
        )
        return y

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma


class SoftAttention(Widget):
    """Soft attention.

    The algorithm is described below:

        Sequence: S = {s_1, s_2, ..., s_n'}, in which s_i in R^n.
        Vector: v in R^m.
        Sequence weight: W, a k by n matrix.
        Vector weight: U, a k by m matrix.
        Omega, a k dimension vector.

        Attention sequence: A = {a_1, a_2, ..., a_n'}, in which a_i in R. A is computed as follow:
            a'_i = tanh(W @ c_i + U @ S)
            A = softmax(omega @ A')
        Attention context: AC = sum(A * C)
    """

    def __init__(self,
                 name,
                 seq_elem_size,
                 vec_size,
                 common_size,
                 seq_weight_initializer=initializers.GlorotUniform(),
                 context_weight_initializer=initializers.GlorotUniform(),
                 omega_initializer=initializers.GlorotUniform()):
        self._seq_elem_size = seq_elem_size
        self._vec_size = vec_size
        self._common_size = common_size
        self._seq_weight_initializer = seq_weight_initializer
        self._context_weight_initializer = context_weight_initializer
        self._omega_initializer = omega_initializer
        super(SoftAttention, self).__init__(name)

    @property
    def seq_elem_size(self):
        return self._seq_elem_size

    @property
    def vec_size(self):
        return self._vec_size

    @property
    def common_size(self):
        return self._common_size

    def _build(self):
        self._w = tf.Variable(
            self._seq_weight_initializer.build(
                shape=(self._seq_elem_size, self._common_size)
            ),
            dtype=config.D_TYPE,
            name='w'
        )
        self._u = tf.Variable(
            self._context_weight_initializer.build(
                shape=(self._vec_size, self._common_size)
            ),
            dtype=config.D_TYPE,
            name='u'
        )
        self._omega = tf.Variable(
            self._omega_initializer.build(
                shape=(self._common_size, 1)
            ),
            dtype=config.D_TYPE,
            name='omega'
        )

    @property
    def w(self):
        return self._w

    @property
    def u(self):
        return self._u

    @property
    def omega(self):
        return self._omega

    def _setup(self, seq, vec, activation=tf.nn.tanh):
        """Setup a soft attention mechanism for the given context sequence and state.
        The result is an attention context for the state.

        :param seq: The sequence tensor.
            Its shape is defined as (seq_length, batch_size, seq_elem_size).
        :param vec: The vector tensor.
            Its shape is defined as (batch_size, vec_size).
        :param activation: The activation function.
            Default is tf.nn.tanh.
        :return: An attention context with shape (batch_size, seq_elem_size).
        """
        #
        # (seq_length, batch_size, seq_elem_size) @ (seq_elem_size, common_size)
        # -> (seq_length, batch_size, common_size)
        a = tf.tensordot(seq, self._w, ((2,), (0,)))
        #
        # (batch_size, vec_size) @ (vec_size, common_size)
        # -> (batch_size, common_size)
        # -> (1, batch_size, common_size)
        b = tf.matmul(vec, self._u)
        b = tf.reshape(b, (1, -1, self._common_size))
        #
        # -> (seq_length, batch_size, common_size)
        # (seq_length, batch_size, common_size) @ (common_size, 1)
        # -> (seq_length, batch_size, 1)
        a = activation(a + b) if activation is not None else a + b
        a = tf.tensordot(a, self._omega, ((2,), (0,)))
        a = tf.nn.softmax(a, dim=0)
        #
        # (seq_length, batch_size, 1) * (seq_length, batch_size, seq_elem_size)
        # -> (seq_length, batch_size, seq_elem_size)
        # -> (batch_size, seq_elem_size)
        att_context = tf.reduce_sum(a * seq, 0)
        return att_context
