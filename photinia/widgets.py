#!/usr/bin/env python3

"""
@author: xi
@since: 2016-11-11
"""

import math

import tensorflow as tf

from photinia import ops

D_TYPE = tf.float32


def set_default_dtype(dtype):
    """Set the default data type.
    The default data type is tf.float32.
    
    :param dtype: Data type.
    """
    global D_TYPE
    D_TYPE = dtype


def setup(x=None, widgets=None):
    if widgets is None:
        return x
    if not isinstance(widgets, (tuple, list)):
        widgets = (widgets,)
    y = x
    for widget in widgets:
        if not isinstance(widget, Widget):
            raise ValueError('Only widget can be setup.')
        y = widget.setup(y)
    return y


class Widget(object):
    """Widget
    The basic component to form a model.
    This an abstract class which can only be inherited.
    """

    # 千年之后的你会在哪里 身边有怎样风景
    # 我们的故事并不算美丽 却如此难以忘记
    # 如果当初勇敢的在一起 会不会不同结局
    # 你会不会也有千言万语 埋在沉默的梦里

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

    # 邻家女孩，或是林家女孩，都是想过却不曾有过记忆
    # 是那种被隔壁家大姐姐吊打的感觉
    # 啊～～～变态！！变态！！变态！！巴嘎！变态！！

    def __init__(self,
                 name,
                 input_size,
                 output_size,
                 with_bias=True,
                 with_batch_norm=False):
        """Construct the linear layer.

        :param name: Name.
        :param input_size: Input size.
        :param output_size: Output size.
        :param with_bias: If the widget has a bias variable.
        :param with_batch_norm: If the widget has a batch norm layer.
        """
        self._input_size = input_size
        self._output_size = output_size
        self._with_bias = with_bias
        self._with_batch_norm = with_batch_norm
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

    @property
    def with_batch_norm(self):
        return self._with_batch_norm

    def _build(self):
        """Build the linear layer.
        Two parameters: weight and bias.

        :return: None.
        """
        bound = math.sqrt(6.0 / (self._input_size + self._output_size))
        w_init = tf.random_uniform(
            minval=-bound,
            maxval=bound,
            shape=(self._input_size, self._output_size),
            dtype=D_TYPE,
            name='w_init'
        )
        self._w = tf.Variable(w_init, dtype=D_TYPE, name='w')
        if self._with_bias:
            b_init = tf.zeros(
                shape=(self._output_size,),
                dtype=D_TYPE,
                name='b_init'
            )
            self._b = tf.Variable(b_init, dtype=D_TYPE, name='b')
        else:
            self._b = None
        self._batch_norm = BatchNorm('bn', self._output_size) if self._with_batch_norm else None

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

    @property
    def batch_norm(self):
        return self._batch_norm

    def _setup(self, x, axes=None):
        """Setup the linear layer.

        :param x: Input tensor.
        :param axes: Axes. If x is a tensor, the layer will perform tensor dot.
        :return: Output tensor.
        """
        y = tf.matmul(x, self._w) if axes is None else tf.tensordot(x, self._w, axes=axes)
        if self._with_bias:
            y += self._b
        if self._with_batch_norm:
            y = self._batch_norm.setup(y)
        return y


class Dropout(Widget):
    """Dropout layer.
    """

    def __init__(self, name):
        Widget.__init__(self, name)

    def _build(self):
        self._keep_prob = tf.placeholder(
            shape=(),
            dtype=D_TYPE
        )

    def _setup(self, x):
        return tf.nn.dropout(x, self._keep_prob)

    @property
    def keep_prob(self):
        return self._keep_prob


class Convolutional(Widget):
    """Convolutional layer.
    """

    def __init__(self,
                 name,
                 input_depth,
                 output_depth,
                 filter_height=5,
                 filter_width=5,
                 stride_height=2,
                 stride_width=2):
        self._input_depth = input_depth
        self._output_depth = output_depth
        self._filter_height = filter_height
        self._filter_width = filter_width
        self._stride_height = stride_height
        self._stride_width = stride_width
        super(Convolutional, self).__init__(name)

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def input_depth(self):
        return self._input_depth

    @property
    def output_depth(self):
        return self._output_depth

    @property
    def stride_width(self):
        return self._stride_width

    @property
    def stride_height(self):
        return self._stride_height

    def _build(self):
        w_init = tf.random_normal(
            stddev=0.01,
            shape=(
                self._filter_height,
                self._filter_width,
                self._input_depth,
                self._output_depth
            ),
            dtype=D_TYPE,
            name='w_init'
        )
        b_init = tf.zeros(
            shape=(self._output_depth,),
            dtype=D_TYPE,
            name='b_init'
        )
        self._w = tf.Variable(w_init, dtype=D_TYPE, name='w')
        self._b = tf.Variable(b_init, dtype=D_TYPE, name='b')

    def _setup(self, x):
        y = tf.nn.conv2d(
            input=x,
            filter=self._w,
            strides=[1, self._stride_height, self._stride_width, 1],
            padding='SAME',
            data_format='NHWC'
        ) + self._b
        return y

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b


class ConvPool(Widget):
    """Convolution-Pooling layer
    This layer consists of a convolution layer and a pooling layer.
    """

    # 你是朋友 从今天开始是朋友
    # 不要再说喜欢什麼的了
    # 不要再远离我 即使只能看著你也没关系
    # 不会喊你的名字 不会和你并肩行走
    # 不会没事就打你电话
    # 所以不要再说什麼不能见面了 拜托了

    def __init__(self,
                 name,
                 input_depth,
                 output_depth,
                 filter_height=5,
                 filter_width=5,
                 pool_height=2,
                 pool_width=2,
                 pool_type='max'):
        """Construct a convolutional pooling layer.

        :param name: Name.
        :param input_depth: Input depth (channel).
        :param output_depth: Output depth (channel, number of feature map).
        :param filter_height: Filter height (rows).
        :param filter_width: Filter width (columns).
        :param pool_height: Pooling height (sub-sampling rows).
        :param pool_width: Pooling width (sub-sampling columns).
        :param pool_type: Pooling (sub-sampling) type. Must be one of "max" or "avg".
        """
        self._input_depth = input_depth
        self._output_depth = output_depth
        self._filter_height = filter_height
        self._filter_width = filter_width
        self._pool_height = pool_height
        self._pool_width = pool_width
        if pool_type not in ('max', 'avg'):
            raise ValueError('Pool type must be one of "max" or "avg".')
        self._pool_type = pool_type
        super(ConvPool, self).__init__(name)

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def input_depth(self):
        return self._input_depth

    @property
    def output_depth(self):
        return self._output_depth

    @property
    def pool_width(self):
        return self._pool_width

    @property
    def pool_height(self):
        return self._pool_height

    def _build(self):
        """Build the layer.
        Two parameters: filter (weight) and bias.

        :return: None.
        """
        bound = 0.02
        w_init = tf.random_uniform(
            minval=-bound,
            maxval=bound,
            shape=(
                self._filter_height,
                self._filter_width,
                self._input_depth,
                self._output_depth
            ),
            dtype=D_TYPE,
            name='w_init'
        )
        b_init = tf.zeros(
            shape=(self._output_depth,),
            dtype=D_TYPE,
            name='b_init'
        )
        self._w = tf.Variable(w_init, dtype=D_TYPE, name='w')
        self._b = tf.Variable(b_init, dtype=D_TYPE, name='b')

    def _setup(self, x):
        """Setup the layer.

        :param x: Input tensor with "NHWC" format.
        :return: Output tensor with "NHWC" format.
        """
        y = tf.nn.conv2d(
            input=x,
            filter=self._w,
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        ) + self._b
        if self._pool_type == 'max':
            y = tf.nn.max_pool(
                value=y,
                ksize=[1, self._pool_height, self._pool_width, 1],
                strides=[1, self._pool_height, self._pool_width, 1],
                padding='SAME',
                data_format='NHWC'
            )
        elif self._pool_type == 'avg':
            tf.nn.avg_pool(
                value=y,
                ksize=[1, self._pool_height, self._pool_width, 1],
                strides=[1, self._pool_height, self._pool_width, 1],
                padding='SAME',
                data_format='NHWC'
            )
        return y

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b


class ConvTrans(Widget):
    """ConvTransposeLayer
    """

    # “快点，舔舔看”女孩笑着说
    # 比起反抗，还是顺从不会被欺负的更惨
    # 我知道的！我知道的！
    # 我发誓要成为强者
    # 可是对谁我都不抱希望
    # 如果那些人才是对的话
    # 我会与全世界为敌

    def __init__(self,
                 name,
                 input_depth,
                 output_depth,
                 filter_height=5,
                 filter_width=5,
                 stride_height=2,
                 stride_width=2):
        """Construct a convolutional transpose layer.

        :param name: Name.
        :param input_depth: Input depth (channel).
        :param output_depth: Output depth (channel, number of feature map).
        :param filter_height: Filter height (rows).
        :param filter_width: Filter width (columns).
        :param stride_height: Stride height (up-sampling rows).
        :param stride_width: Stride width (up-sampling columns).
        """
        self._input_depth = input_depth
        self._output_depth = output_depth
        self._filter_height = filter_height
        self._filter_width = filter_width
        self._stride_height = stride_height
        self._stride_width = stride_width
        super(ConvTrans, self).__init__(name)

    @property
    def filter_width(self):
        return self._filter_width

    @property
    def filter_height(self):
        return self._filter_height

    @property
    def input_depth(self):
        return self._input_depth

    @property
    def output_depth(self):
        return self._output_depth

    def _build(self):
        """Build the layer.
        Two parameters: filter (weight) and bias.

        :return: None.
        """
        bound = 0.02
        w_init = tf.random_uniform(
            minval=-bound,
            maxval=bound,
            shape=(
                self._filter_height,
                self._filter_width,
                self._output_depth,
                self._input_depth
            ),
            dtype=D_TYPE,
            name='w_init'
        )
        b_init = tf.zeros(
            shape=(self._output_depth,),
            dtype=D_TYPE,
            name='b_init'
        )
        self._w = tf.Variable(w_init, dtype=D_TYPE, name='w')
        self._b = tf.Variable(b_init, dtype=D_TYPE, name='b')

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
            self._output_depth
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

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b


class GRUCell(Widget):
    """GRUCell
    """

    # "お兄ちゃん~ 朝だよ~ 起きてよ~~~"
    # 一个关西腔的MM在耳边喊道，一边眨巴着大眼睛，还一边扯开我的被子
    # 不知为何，GRU就让我觉得像是这样一个小萝莉
    # 老天啊～你欠我一个妹妹～

    def __init__(self,
                 name,
                 input_size,
                 state_size,
                 activation=ops.lrelu):
        """Construct a cell.
        Does not create the parameters' tensors.

        :param name: Name.
        :param input_size: Input size.
        :param state_size: State size.
        """
        self._input_size = input_size
        self._state_size = state_size
        self._activation = activation
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
        bound = math.sqrt(6.0 / (self._input_size + self._state_size))
        wz_init_value = tf.random_uniform(
            minval=-bound,
            maxval=bound,
            dtype=D_TYPE,
            shape=(self._input_size, self._state_size)
        )
        wr_init_value = tf.random_uniform(
            minval=-bound,
            maxval=bound,
            dtype=D_TYPE,
            shape=(self._input_size, self._state_size)
        )
        wh_init_value = tf.random_uniform(
            minval=-bound,
            maxval=bound,
            dtype=D_TYPE,
            shape=(self._input_size, self._state_size)
        )
        bound = math.sqrt(6.0 / (self._state_size + self._state_size))
        uz_init_value = tf.random_uniform(
            minval=-bound,
            maxval=bound,
            dtype=D_TYPE,
            shape=(self._state_size, self._state_size)
        )
        ur_init_value = tf.random_uniform(
            minval=-bound,
            maxval=bound,
            dtype=D_TYPE,
            shape=(self._state_size, self._state_size)
        )
        uh_init_value = tf.random_uniform(
            minval=-bound,
            maxval=bound,
            dtype=D_TYPE,
            shape=(self._state_size, self._state_size)
        )
        b_init_value = tf.zeros(
            dtype=D_TYPE,
            shape=(self._state_size,)
        )
        self._wz = tf.Variable(name='wz', dtype=D_TYPE, initial_value=wz_init_value)
        self._uz = tf.Variable(name='uz', dtype=D_TYPE, initial_value=uz_init_value)
        self._bz = tf.Variable(name='bz', dtype=D_TYPE, initial_value=b_init_value)
        #
        self._wr = tf.Variable(name='wr', dtype=D_TYPE, initial_value=wr_init_value)
        self._ur = tf.Variable(name='ur', dtype=D_TYPE, initial_value=ur_init_value)
        self._br = tf.Variable(name='br', dtype=D_TYPE, initial_value=b_init_value)
        #
        self._wh = tf.Variable(name='wh', dtype=D_TYPE, initial_value=wh_init_value)
        self._uh = tf.Variable(name='uh', dtype=D_TYPE, initial_value=uh_init_value)
        self._bh = tf.Variable(name='bh', dtype=D_TYPE, initial_value=b_init_value)

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

    # 山外还有山比山高 半山腰 一声惊雷摇晃树梢
    # 人外还有人忘不掉 你怀抱 夜夜都是魂牵梦绕
    # 爱恨情仇都付谈笑 多寂寥 星辰变换诛仙桀骜
    # 引无数英雄竞折腰 江山多娇 封印魂魄于我剑鞘 一声咆哮

    def __init__(self,
                 name,
                 input_size,
                 state_size,
                 activation=ops.lrelu):
        """Construct a cell.
        Does not create the parameters' tensors.

        :param name: Name.
        :param input_size: Input size.
        :param state_size: State size.
        """
        self._input_size = input_size
        self._state_size = state_size
        self._activation = activation
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
        wi_init_value = tf.random_normal(
            dtype=D_TYPE,
            shape=(self._input_size, self._state_size),
            stddev=1.0 / (self._input_size + self._state_size)
        )
        wf_init_value = tf.random_normal(
            dtype=D_TYPE,
            shape=(self._input_size, self._state_size),
            stddev=1.0 / (self._input_size + self._state_size)
        )
        wo_init_value = tf.random_normal(
            dtype=D_TYPE,
            shape=(self._input_size, self._state_size),
            stddev=1.0 / (self._input_size + self._state_size)
        )
        wc_init_value = tf.random_normal(
            dtype=D_TYPE,
            shape=(self._input_size, self._state_size),
            stddev=1.0 / (self._input_size + self._state_size)
        )
        ui_init_value = tf.random_normal(
            dtype=D_TYPE,
            shape=(self._state_size, self._state_size),
            stddev=1.0 / (self._state_size + self._state_size)
        )
        uf_init_value = tf.random_normal(
            dtype=D_TYPE,
            shape=(self._state_size, self._state_size),
            stddev=1.0 / (self._state_size + self._state_size)
        )
        uo_init_value = tf.random_normal(
            dtype=D_TYPE,
            shape=(self._state_size, self._state_size),
            stddev=1.0 / (self._state_size + self._state_size)
        )
        uc_init_value = tf.random_normal(
            dtype=D_TYPE,
            shape=(self._state_size, self._state_size),
            stddev=1.0 / (self._state_size + self._state_size)
        )
        b_init_value = tf.zeros(
            dtype=D_TYPE,
            shape=(self._state_size,)
        )
        self._wi = tf.Variable(name='wi', dtype=D_TYPE, initial_value=wi_init_value)
        self._ui = tf.Variable(name='ui', dtype=D_TYPE, initial_value=ui_init_value)
        self._bi = tf.Variable(name='bi', dtype=D_TYPE, initial_value=b_init_value)
        #
        self._wf = tf.Variable(name='wf', dtype=D_TYPE, initial_value=wf_init_value)
        self._uf = tf.Variable(name='uf', dtype=D_TYPE, initial_value=uf_init_value)
        self._bf = tf.Variable(name='bf', dtype=D_TYPE, initial_value=b_init_value)
        #
        self._wo = tf.Variable(name='wo', dtype=D_TYPE, initial_value=wo_init_value)
        self._uo = tf.Variable(name='uo', dtype=D_TYPE, initial_value=uo_init_value)
        self._bo = tf.Variable(name='bo', dtype=D_TYPE, initial_value=b_init_value)
        #
        self._wc = tf.Variable(name='wc', dtype=D_TYPE, initial_value=wc_init_value)
        self._uc = tf.Variable(name='uc', dtype=D_TYPE, initial_value=uc_init_value)
        self._bc = tf.Variable(name='bc', dtype=D_TYPE, initial_value=b_init_value)

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

    # 我也不想这么样 反反复复 反正最后每个人都孤独
    # 你的甜蜜变成我的痛苦 离开你有没有帮助
    # 我也不想这么样 起起伏伏 反正每段关系都是孤独
    # 你的甜蜜变成我的痛苦 都怪我太渴望 得到你保护

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
            dtype=D_TYPE
        )
        gamma_init = tf.ones(
            shape=self._size,
            dtype=D_TYPE
        )
        self._beta = tf.Variable(
            name='beta',
            initial_value=beta_init,
            dtype=D_TYPE
        )
        self._gamma = tf.Variable(
            name='gamma',
            initial_value=gamma_init,
            dtype=D_TYPE
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


class CNN(Widget):
    """Convolution-Pooling layers
    Stacked Convolution-Pooling layers.
    """

    # 我希望你能长大，长的比玻璃缸还大，比桌子还大，比镜子还大，比床还大，整个屋子都装不下你

    def __init__(self,
                 name,
                 input_height,
                 input_width,
                 input_depth,
                 layer_shapes,
                 activation=ops.lrelu,
                 with_batch_norm=True,
                 flat_output=True):
        """
        Each layer is described as a tuple:
        (filter_height, filter_width,
         output_depth,
         pool_height, pool_width)
        """
        self._input_height = input_height
        self._input_width = input_width
        self._input_depth = input_depth
        assert isinstance(layer_shapes, (tuple, list))
        self._layer_shapes = layer_shapes.copy()
        self._activation = activation
        self._with_batch_norm = with_batch_norm
        self.flat_output = flat_output
        #
        self._layers = []
        #
        # The constructor doesn't do any build operations.
        # It just need to compute the output info.
        last_height, last_width, last_depth = input_height, input_width, input_depth
        for layer_shape in layer_shapes:
            (filter_height, filter_width,
             output_depth,
             pool_height, pool_width) = layer_shape
            last_height = -(-last_height // pool_height)
            last_width = -(-last_width // pool_width)
            last_depth = output_depth
        self._output_height = last_height
        self._output_width = last_width
        self._output_depth = last_depth
        self._flat_size = self._output_height * self._output_width * self._output_depth
        super(CNN, self).__init__(name)

    def _build(self):
        last_depth = self._input_depth
        for index, layer_shape in enumerate(self._layer_shapes):
            #
            # Get layer parameters.
            (filter_height, filter_width,
             output_depth,
             pool_height, pool_width) = layer_shape
            #
            # Create layer.
            layer = Convolutional(
                'C{}'.format(index),
                last_depth, output_depth,
                filter_height, filter_width,
                pool_height, pool_width
            )
            if self._with_batch_norm:
                bn_layer = BatchNorm(
                    'BN{}'.format(index),
                    output_depth
                )
                layer = (layer, bn_layer)
            self._layers.append(layer)
            #
            # Update output.
            last_depth = output_depth

    def _setup(self, x):
        y = x
        for layer in self._layers:
            if isinstance(layer, tuple):
                y = layer[0].setup(y)
                y = layer[1].setup(y)
            else:
                y = layer.setup(y)
            y = self._activation(y) if self._activation is not None else y
        if self.flat_output:
            y = tf.reshape(y, (-1, self._flat_size))
        return y

    @property
    def input_height(self):
        return self._input_height

    @property
    def input_width(self):
        return self._input_width

    @property
    def input_depth(self):
        return self._input_depth

    @property
    def output_height(self):
        return self._output_height

    @property
    def output_width(self):
        return self._output_width

    @property
    def output_depth(self):
        return self._output_depth

    @property
    def flat_size(self):
        return self._flat_size


class TransCNN(Widget):
    """Convolution transpose layers
    Stacked Convolution transpose layers.
    """

    def __init__(self,
                 name,
                 init_height,
                 init_width,
                 init_depth,
                 layer_shapes,
                 activation=ops.lrelu,
                 with_batch_norm=True):
        """
        Each layer is described as a tuple:
        (filter_height, filter_width,
         output_depth,
         stride_height, stride_height)
        """
        self._init_height = init_height
        self._init_width = init_width
        self._init_depth = init_depth
        assert isinstance(layer_shapes, (tuple, list))
        self._layer_shapes = layer_shapes.copy()
        self._activation = activation
        self._with_batch_norm = with_batch_norm
        #
        self._layers = []
        #
        # The constructor doesn't do any build operations.
        # It just need to compute the output info.
        self._init_flat_size = self._init_height * self._init_width * self._init_depth
        super(TransCNN, self).__init__(name)

    def _build(self):
        last_depth = self._init_depth
        for index, layer_shape in enumerate(self._layer_shapes):
            #
            # Get layer parameters.
            (filter_height, filter_width,
             output_depth,
             stride_height, stride_width) = layer_shape
            #
            # Create layer.
            layer = ConvTrans(
                'CT{}'.format(index),
                last_depth, output_depth,
                filter_height, filter_width,
                stride_height, stride_width
            )
            if self._with_batch_norm and index != len(self._layer_shapes) - 1:
                bn_layer = BatchNorm(
                    'BN{}'.format(index),
                    output_depth
                )
                layer = (layer, bn_layer)
            self._layers.append(layer)
            #
            # Update output.
            last_depth = output_depth

    def _setup(self, x):
        maps = tf.reshape(x, (-1, self._init_height, self._init_width, self._init_depth))
        for index, layer in enumerate(self._layers):
            if isinstance(layer, tuple):
                maps = layer[0].setup(maps)
                maps = layer[1].setup(maps)
            else:
                maps = layer.setup(maps)
            if self._activation is not None and index != len(self._layer_shapes) - 1:
                maps = self._activation(maps)
        return maps

    @property
    def init_height(self):
        return self._init_height

    @property
    def init_width(self):
        return self._init_width

    @property
    def init_depth(self):
        return self._init_depth

    @property
    def init_flat_size(self):
        return self._init_flat_size


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
                 common_size):
        self._seq_elem_size = seq_elem_size
        self._vec_size = vec_size
        self._common_size = common_size
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
        bound = math.sqrt(6.0 / (self._seq_elem_size + self._common_size))
        w_init = tf.random_uniform(
            minval=-bound,
            maxval=bound,
            shape=(self._seq_elem_size, self._common_size),
            dtype=D_TYPE,
            name='w_init'
        )
        self._w = tf.Variable(w_init, dtype=D_TYPE, name='w')
        bound = math.sqrt(6.0 / (self._vec_size + self._common_size))
        u_init = tf.random_uniform(
            minval=-bound,
            maxval=bound,
            shape=(self._vec_size, self._common_size),
            dtype=D_TYPE,
            name='u_init'
        )
        self._u = tf.Variable(u_init, dtype=D_TYPE, name='u')
        bound = math.sqrt(6.0 / self._common_size)
        omega_init = tf.random_uniform(
            minval=-bound,
            maxval=bound,
            shape=(self._common_size, 1),
            dtype=D_TYPE,
            name='omega_init'
        )
        self._omega = tf.Variable(omega_init, dtype=D_TYPE, name='omega')

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
