#!/usr/bin/env python3

"""
@author: xi
@since: 2018-06-09
"""

import tensorflow as tf

import photinia as ph
from . import core


class Actor(ph.Widget):

    def __init__(self, name, state_size, action_size, hidden_size=50):
        self._state_size = state_size
        self._action_size = action_size
        self._hidden_size = hidden_size
        super(Actor, self).__init__(name)

    @property
    def state_size(self):
        return self._state_size

    @property
    def action_size(self):
        return self._action_size

    @property
    def hidden_size(self):
        return self._hidden_size

    def _build(self):
        self._layer1 = ph.Linear(
            'layer1',
            self._state_size, self._hidden_size,
            w_init=ph.init.TruncatedNormal(0.0, 1e-2)
        )
        self._layer2 = ph.Linear(
            'layer2',
            self._hidden_size, self._hidden_size,
            w_init=ph.init.TruncatedNormal(0.0, 1e-2)
        )
        self._layer3 = ph.Linear(
            'layer3',
            self._hidden_size, self._action_size,
            w_init=ph.init.TruncatedNormal(0.0, 1e-2)
        )

    def _setup(self, state):
        return ph.setup(
            state, [
                self._layer1, ph.ops.lrelu,
                self._layer2, ph.ops.lrelu,
                self._layer3,
                tf.nn.tanh
            ]
        )


class Critic(ph.Widget):

    def __init__(self, name, state_size, action_size, hidden_size=50):
        self._state_size = state_size
        self._action_size = action_size
        self._hidden_size = hidden_size
        super(Critic, self).__init__(name)

    @property
    def state_size(self):
        return self._state_size

    @property
    def action_size(self):
        return self._action_size

    @property
    def hidden_size(self):
        return self._hidden_size

    def _build(self):
        self._layer1 = ph.Linear(
            'layer1',
            self._state_size + self._action_size, self._hidden_size,
            w_init=ph.init.TruncatedNormal(0.0, 1e-2)
        )
        self._layer2 = ph.Linear(
            'layer2',
            self._hidden_size, self._hidden_size,
            w_init=ph.init.TruncatedNormal(0.0, 1e-2)
        )
        self._layer3 = ph.Linear(
            'layer3',
            self._hidden_size, 1,
            w_init=ph.init.TruncatedNormal(0.0, 1e-2)
        )

    def _setup(self, state, action):
        return ph.setup(
            tf.concat((state, action), axis=1), [
                self._layer1, ph.ops.lrelu,
                self._layer2, ph.ops.lrelu,
                self._layer3,
                (tf.reshape, {'shape': (-1,)})
            ]
        )


class Agent(ph.Model):

    def __init__(self,
                 name,
                 state_size,
                 action_size,
                 actor_type=Actor,
                 actor_args=None,
                 critic_type=Critic,
                 critic_args=None,
                 gamma=0.9,
                 tao=0.01,
                 replay_size=10000):
        """

        Args:
            name (str): Widget name.
            state_size (int): Dimensions of the state vector.
            action_size (int): Dimensions of the action vector.
            actor_type (type): Type/Class of the actor widget.
            actor_args (dict[str, any]): Arguments used to construct the actor.
            critic_type (type): Type/Class of the critic widget.
            critic_args (dict[str, any]): Arguments used to construct the critic.

        """
        self._state_size = state_size
        self._action_size = action_size
        self._actor_type = actor_type
        self._actor_args = actor_args if actor_args is not None else {}
        self._critic_type = critic_type
        self._critic_args = critic_args if critic_args is not None else {}
        self._gamma = gamma
        self._tao = tao
        self._replay = core.ReplayMemory(replay_size)
        super(Agent, self).__init__(name)

    def _build(self):
        a_source = self._actor_type(
            name='a_source',
            state_size=self._state_size,
            action_size=self._action_size,
            **self._actor_args
        )  # type: ph.Widget
        a_target = self._actor_type(
            name='a_target',
            state_size=self._state_size,
            action_size=self._action_size,
            **self._actor_args
        )  # type: ph.Widget
        q_source = self._critic_type(
            name='q_source',
            state_size=self._state_size,
            action_size=self._action_size,
            **self._critic_args
        )  # type: ph.Widget
        q_target = self._critic_type(
            name='q_target',
            state_size=self._state_size,
            action_size=self._action_size,
            **self._critic_args
        )  # type: ph.Widget

        s_in = ph.placeholder('s_in', (None, self._state_size))
        a_source_pred = a_source.setup(s_in)
        self._add_slot(
            '_predict',
            inputs=s_in,
            outputs=a_source_pred
        )

        r_in = ph.placeholder('r_in', (None,))
        s_in_ = ph.placeholder('s_in_', (None, self._state_size))
        a_target_pred = a_target.setup(s_in_)
        q_target_pred = q_target.setup(s_in_, a_target_pred)
        y = r_in + self._gamma * q_target_pred
        q_source_pred = q_source.setup(s_in, a_source_pred)
        loss = tf.reduce_mean(tf.square(y - q_source_pred))
        var_list = q_source.get_trainable_variables()
        reg = ph.reg.Regularizer().add_l1_l2(var_list)
        self._add_slot(
            '_update_q_source',
            inputs=(s_in, a_source_pred, r_in, s_in_),
            outputs=loss,
            updates=tf.train.RMSPropOptimizer(1e-4, 0.9, 0.9).minimize(
                loss + reg.get_loss(1e-6),
                var_list=var_list
            )
        )

        var_list = a_source.get_trainable_variables()
        # grad_policy = tf.gradients(
        #     ys=a_source_pred,
        #     xs=var_list,
        #     grad_ys=tf.gradients(
        #         tf.reduce_mean(q_source_pred),
        #         a_source_pred
        #     )[0]
        # )
        # updates=tf.train.AdamOptimizer(-1e-3).apply_gradients(
        #     zip(grad_policy, var_list)
        # ),
        loss = tf.reduce_mean(q_source_pred)
        reg = ph.reg.Regularizer().add_l1_l2(var_list)
        self._add_slot(
            '_update_a_source',
            inputs=s_in,
            updates=tf.train.RMSPropOptimizer(1e-4, 0.9, 0.9).minimize(
                -loss + reg.get_loss(1e-6),
                var_list=var_list
            )
        )

        self._add_slot(
            '_update_q_target',
            updates=tf.group(*[
                tf.assign(v_target, self._tao * v_source + (1.0 - self._tao) * v_target)
                for v_source, v_target in zip(
                    q_source.get_trainable_variables(),
                    q_target.get_trainable_variables()
                )
            ])
        )

        self._add_slot(
            '_update_a_target',
            updates=tf.group(*[
                tf.assign(v_target, self._tao * v_source + (1.0 - self._tao) * v_target)
                for v_source, v_target in zip(
                    a_source.get_trainable_variables(),
                    a_target.get_trainable_variables()
                )
            ])
        )

        self._add_slot(
            '_init_a_target',
            updates=tf.group(*[
                tf.assign(v_target, v_source)
                for v_source, v_target in zip(
                    q_source.get_trainable_variables(),
                    q_target.get_trainable_variables()
                )
            ])
        )
        self._add_slot(
            '_init_q_target',
            updates=tf.group(*[
                tf.assign(v_target, v_source)
                for v_source, v_target in zip(
                    a_source.get_trainable_variables(),
                    a_target.get_trainable_variables()
                )
            ])
        )

    def init(self):
        self._init_a_target()
        self._init_q_target()

    def predict(self, s):
        return self._predict([s])[0][0]

    def feedback(self, s, a, r, s_, done=False):
        self._replay.put(s, a, r, s_, done)

    def train(self, batch_size=32):
        s, a, r, s_ = self._replay.get(batch_size)[:-1]
        self._update_q_source(s, a, r, s_)
        self._update_a_source(s)
        self._update_q_target()
        self._update_a_target()
