#!/usr/bin/env python3

"""
@author: xi
@since: 2018-08-31
"""

import tensorflow as tf

import photinia as ph
from photinia.dnn import mlp
from photinia.dnn import resnet


class MLPActor(ph.Widget):

    def __init__(self,
                 name,
                 state_size,
                 action_size,
                 hidden_size):
        self._state_size = state_size
        self._action_size = action_size
        self._hidden_size = hidden_size
        super(MLPActor, self).__init__(name)

    def _build(self):
        self._mlp = mlp.MLP(
            'mlp',
            self._state_size,
            self._action_size,
            self._hidden_size,
        )

    def _setup(self, input_state, name='action'):
        y = self._mlp.setup(input_state)
        y = tf.nn.tanh(y, name=name)
        return y


class MLPCritic(ph.Widget):

    def __init__(self,
                 name,
                 state_size,
                 action_size,
                 hidden_size):
        self._state_size = state_size
        self._action_size = action_size
        self._hidden_size = hidden_size
        super(MLPCritic, self).__init__(name)

    def _build(self):
        self._mlp = mlp.MLP(
            'mlp',
            self._state_size + self._action_size,
            1,
            self._hidden_size,
        )

    def _setup(self, input_state, input_action, name='reward'):
        h = tf.concat((input_state, input_action), axis=1)
        y = self._mlp.setup(h)
        y = tf.reshape(y, (-1,), name=name)
        return y


class DeepResActor(ph.Widget):

    def __init__(self,
                 name,
                 state_size,
                 action_size,
                 hidden_size,
                 num_layers=1):
        self._state_size = state_size
        self._action_size = action_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        super(DeepResActor, self).__init__(name)

    def _build(self):
        self._resnet = resnet.DeepResNet(
            'resnet',
            self._state_size,
            self._action_size,
            self._hidden_size,
            self._num_layers
        )

    def _setup(self, input_state, name='action'):
        y = self._resnet.setup(input_state)
        y = tf.nn.tanh(y, name=name)
        return y


class DeepResCritic(ph.Widget):

    def __init__(self,
                 name,
                 state_size,
                 action_size,
                 hidden_size,
                 num_layers=1):
        self._state_size = state_size
        self._action_size = action_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        super(DeepResCritic, self).__init__(name)

    def _build(self):
        self._resnet = resnet.DeepResNet(
            'resnet',
            input_size=self._state_size + self._action_size,
            output_size=1,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers
        )

    def _setup(self, input_state, input_action, name='reward'):
        h = tf.concat((input_state, input_action), axis=1)
        y = self._resnet.setup(h)
        y = tf.reshape(y, (-1,), name=name)
        return y
