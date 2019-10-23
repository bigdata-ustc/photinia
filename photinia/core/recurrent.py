#!/usr/bin/env python3


"""
@author: xi
@since: 2016-11-11
"""

from . import common
from .. import conf
from .. import init
from ..conf import tf


class GRUCell(common.Module):

    def __init__(self,
                 name,
                 input_size,
                 state_size,
                 with_bias=False,
                 w_init=init.GlorotNormal(),
                 u_init=init.Orthogonal(),
                 b_init=init.Zeros(),
                 activation=tf.nn.tanh):
        """The GRU cell.

        Args:
            name (str): The widget name.
            input_size: The input size.
            state_size: The state size.
            with_bias: If this cell has bias.
            w_init: The input weight initializer.
            u_init: The recurrent weight initializer.
            b_init: The bias initializer.

        """
        super(GRUCell, self).__init__(name)
        self._input_size = input_size
        self._state_size = state_size
        self._with_bias = with_bias
        self._w_init = w_init
        self._u_init = u_init
        self._b_init = b_init
        self._activation = activation

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    @property
    def with_bias(self):
        return self._with_bias

    def _setup(self, x, prev_h, name=None):
        """Setup the cell.

        Args:
            x: The input tensor. shape = (batch_size, input_size)
            prev_h: The previous state tensor. shape = (batch_size, state_size)
            name (str): The output name.

        Returns:
            The state tensor. shape = (batch_size, state_size)

        """
        wz = common.variable(
            name='wz',
            init_value=self._w_init.build(
                name='wz_init',
                shape=(self._input_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        wr = common.variable(
            name='wr',
            init_value=self._w_init.build(
                name='wr_init',
                shape=(self._input_size, self._state_size),
            ),
            dtype=conf.dtype,
            trainable=True
        )
        wh = common.variable(
            name='wh',
            init_value=self._w_init.build(
                name='wh_init',
                shape=(self._input_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        #
        uz = common.variable(
            name='uz',
            init_value=self._u_init.build(
                name='uz_init',
                shape=(self._state_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        ur = common.variable(
            name='ur',
            init_value=self._u_init.build(
                name='ur_init',
                shape=(self._state_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        uh = common.variable(
            name='uh',
            init_value=self._u_init.build(
                name='uh_init',
                shape=(self._state_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        if self._with_bias:
            bz = common.variable(
                name='bz',
                init_value=self._b_init.build(
                    name='bz_init',
                    shape=(self._state_size,)
                ),
                dtype=conf.dtype,
                trainable=True
            )
            br = common.variable(
                name='br',
                init_value=self._b_init.build(
                    name='br_init',
                    shape=(self._state_size,)
                ),
                dtype=conf.dtype,
                trainable=True
            )
            bh = common.variable(
                name='bh',
                init_value=self._b_init.build(
                    name='bh_init',
                    shape=(self._state_size,)
                ),
                dtype=conf.dtype,
                trainable=True
            )
            z = tf.sigmoid(
                tf.matmul(x, wz) + tf.matmul(prev_h, uz) + bz,
                name='update_gate'
            )
            r = tf.sigmoid(
                tf.matmul(x, wr) + tf.matmul(prev_h, ur) + br,
                name='reset_gate'
            )
            h = tf.matmul(x, wh) + tf.matmul(r * prev_h, uh) + bh
        else:
            z = tf.sigmoid(
                tf.matmul(x, wz) + tf.matmul(prev_h, uz),
                name='update_gate'
            )
            r = tf.sigmoid(
                tf.matmul(x, wr) + tf.matmul(prev_h, ur),
                name='reset_gate'
            )
            h = tf.matmul(x, wh) + tf.matmul(r * prev_h, uh)
        if self._activation is not None:
            h = self._activation(h)
        h = tf.add(z * prev_h, (1.0 - z) * h, name=name)
        return h


class LSTMCell(common.Module):

    def __init__(self,
                 name,
                 input_size,
                 state_size,
                 with_bias=True,
                 w_init=init.GlorotNormal(),
                 u_init=init.Orthogonal(),
                 b_init=init.Zeros(),
                 activation=tf.nn.tanh):
        """LSTM cell.

        Args:
            name (str): Widget name.
            input_size (int): Input size.
            state_size (int): State size.
            with_bias (bool): If True, the cell will involve biases.
            w_init (init.Initializer): Input weight initializer.
            u_init (initializers.Initializer): Recurrent weight initializer.
            b_init (initializers.Initializer): Bias initializer.

        """
        super(LSTMCell, self).__init__(name)
        self._input_size = input_size
        self._state_size = state_size
        self._with_bias = with_bias
        self._w_init = w_init
        self._u_init = u_init
        self._b_init = b_init
        self._activation = activation

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def setup(self, x, prev_cell_state, prev_state):
        """Setup the cell.

        Args:
            x (tf.Tensor): Input tensor.
                (batch_size, input_size)
            prev_cell_state (tf.Tensor): Previous cell state.
                (batch_size, state_size)
            prev_state (tf.Tensor): Previous state.
                (batch_size, state_size)

        Returns:
            tuple[tf.Tensor]: Tuple of cell states and states.
                (batch_size, seq_length, state_size)
                (batch_size, seq_length, state_size)

        """
        wi = common.variable(
            name='wi',
            init_value=self._w_init.build(
                name='wi_init',
                shape=(self._input_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        wf = common.variable(
            name='wf',
            init_value=self._w_init.build(
                name='wf_init',
                shape=(self._input_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        wo = common.variable(
            name='wo',
            init_value=self._w_init.build(
                name='wo_init',
                shape=(self._input_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        wc = common.variable(
            name='wc',
            init_value=self._w_init.build(
                name='wc_init',
                shape=(self._input_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        #
        ui = common.variable(
            name='ui',
            init_value=self._u_init.build(
                name='ui_init',
                shape=(self._state_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        uf = common.variable(
            name='uf',
            init_value=self._u_init.build(
                name='uf_init',
                shape=(self._state_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        uo = common.variable(
            name='uo',
            init_value=self._u_init.build(
                name='uo_init',
                shape=(self._state_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        uc = common.variable(
            name='uc',
            init_value=self._u_init.build(
                name='uc_init',
                shape=(self._state_size, self._state_size)
            ),
            dtype=conf.dtype,
            trainable=True
        )
        if self._with_bias:
            bi = common.variable(
                name='bi',
                init_value=self._b_init.build(
                    name='bi_init',
                    shape=(self._state_size,)
                ),
                dtype=conf.dtype,
                trainable=True
            )
            bf = common.variable(
                name='bf',
                init_value=self._b_init.build(
                    name='bf_init',
                    shape=(self._state_size,)
                ),
                dtype=conf.dtype,
                trainable=True
            )
            bo = common.variable(
                name='bo',
                init_value=self._b_init.build(
                    name='bo_init',
                    shape=(self._state_size,)
                ),
                dtype=conf.dtype,
                trainable=True
            )
            bc = common.variable(
                name='bc',
                init_value=self._b_init.build(
                    name='bc_init',
                    shape=(self._state_size,)
                ),
                dtype=conf.dtype,
                trainable=True
            )
            input_gate = tf.nn.sigmoid(
                tf.matmul(x, wi) + tf.matmul(prev_state, ui) + bi,
                name='input_gate'
            )
            forget_gate = tf.nn.sigmoid(
                tf.matmul(x, wf) + tf.matmul(prev_state, uf) + bf,
                name='forget_gate'
            )
            output_gate = tf.nn.sigmoid(
                tf.matmul(x, wo) + tf.matmul(prev_state, uo) + bo,
                name='output_gate'
            )
            cell_state = tf.matmul(x, wc) + tf.matmul(prev_state, uc) + bc
        else:
            input_gate = tf.nn.sigmoid(
                tf.matmul(x, wi) + tf.matmul(prev_state, ui),
                name='input_gate'
            )
            forget_gate = tf.nn.sigmoid(
                tf.matmul(x, wf) + tf.matmul(prev_state, uf),
                name='forget_gate'
            )
            output_gate = tf.nn.sigmoid(
                tf.matmul(x, wo) + tf.matmul(prev_state, uo),
                name='output_gate'
            )
            cell_state = tf.matmul(x, wc) + tf.matmul(prev_state, uc)
        if self._activation is not None:
            cell_state = self._activation(cell_state)
        cell_state = tf.add(forget_gate * prev_cell_state, input_gate * cell_state, name='cell_state')
        if self._activation is not None:
            cell_state = self._activation(cell_state)
        state = tf.multiply(output_gate, cell_state, name='state')
        return cell_state, state


class GRU(common.Module):

    def __init__(self,
                 name,
                 input_size,
                 state_size,
                 activation=tf.nn.tanh):
        """The recurrent neural network with GRU cell.
        All sequence shapes follow (batch_size, seq_len, state_size).

        Args:
            name (str): The widget name.
            input_size: The input size.
            state_size: The state size.
            activation: The activation function for the GRU cell.

        """
        super(GRU, self).__init__(name)
        self._input_size = input_size
        self._state_size = state_size
        self._activation = activation

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def setup(self, seq, init_state=None, name=None):
        """Setup a sequence.

        Args:
            seq: The sequences.
                shape = (batch_size, seq_len, input_size)
            init_state: The initial state.
                shape = (batch_size, state_size)
            name (str): Output tensor name.

        Returns:
            The forward states.
                shape = (batch_size, seq_len, state_size)

        """
        cell = GRUCell(
            'cell',
            input_size=self._input_size,
            state_size=self._state_size
        )
        # check forward and backward initial states
        if init_state is None:
            batch_size = tf.shape(seq)[0]
            init_state = tf.zeros(shape=(batch_size, self._state_size), dtype=conf.dtype)

        # connect
        states_forward = tf.scan(
            fn=lambda acc, elem: cell.setup(elem, acc, activation=self._activation),
            elems=tf.transpose(seq, [1, 0, 2]),
            initializer=init_state
        )
        states_forward = tf.transpose(states_forward, [1, 0, 2], name=name)

        return states_forward


class BiGRU(common.Module):

    def __init__(self,
                 name,
                 input_size,
                 state_size,
                 activation=tf.nn.tanh):
        """A very simple BiGRU structure comes from zhkun.
        All sequence shapes follow (batch_size, seq_len, state_size).

        Args:
            name (str): The widget name.
            input_size: The input size.
            state_size: The state size.
            activation: The activation function for the GRU cell.

        """
        super(BiGRU, self).__init__(name)
        self._input_size = input_size
        self._state_size = state_size
        self._activation = activation

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def setup(self, seq, init_state=None, name=None):
        """Setup a sequence.

        Args:
            seq: A sequence or a pair of sequences.
                shape = (batch_size, seq_len, input_size)
            init_state: A tensor or a pair of tensors.
            name (str): Output tensor name.

        Returns:
            The forward states and the backward states.
                shape = (batch_size, seq_len, state_size)

        """
        cell_forward = GRUCell(
            'cell_forward',
            input_size=self._input_size,
            state_size=self._state_size
        )
        cell_backward = GRUCell(
            'cell_backward',
            input_size=self._input_size,
            state_size=self._state_size
        )
        # check forward and backward sequences
        if isinstance(seq, (tuple, list)):
            if len(seq) != 2:
                raise ValueError('The seqs should be tuple with 2 elements.')
            seq_forward, seq_backward = seq
        else:
            seq_forward = seq
            seq_backward = tf.reverse(seq, axis=[1])

        # check forward and backward initial states
        if init_state is None:
            batch_size = tf.shape(seq_forward)[0]
            init_state_forward = tf.zeros(shape=(batch_size, self._state_size), dtype=conf.dtype)
            init_state_backward = tf.zeros(shape=(batch_size, self._state_size), dtype=conf.dtype)
        elif isinstance(init_state, (tuple, list)):
            if len(seq) != 2:
                raise ValueError('The init_states should be tuple with 2 elements.')
            init_state_forward, init_state_backward = init_state
        else:
            init_state_forward = init_state
            init_state_backward = init_state

        # connect
        states_forward = tf.scan(
            fn=lambda acc, elem: cell_forward.setup(elem, acc, activation=self._activation),
            elems=tf.transpose(seq_forward, [1, 0, 2]),
            initializer=init_state_forward
        )
        states_forward = tf.transpose(states_forward, [1, 0, 2], name=f'{name}_forward')
        states_backward = tf.scan(
            fn=lambda acc, elem: cell_backward.setup(elem, acc, activation=self._activation),
            elems=tf.transpose(seq_backward, [1, 0, 2]),
            initializer=init_state_backward
        )
        states_backward = tf.transpose(states_backward, [1, 0, 2], name=f'{name}_backward')

        return states_forward, states_backward
