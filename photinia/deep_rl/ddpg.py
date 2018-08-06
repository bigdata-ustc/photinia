#!/usr/bin/env python3

"""
@author: xi
@since: 2018-06-09
"""

import tensorflow as tf

import photinia as ph
from . import common


class DDPGAgent(ph.Model):

    def __init__(self,
                 name,
                 source_actor,
                 target_actor,
                 source_critic,
                 target_critic,
                 source_state_placeholder,
                 target_state_placeholder,
                 reward_placeholder,
                 gamma=0.9,
                 tao=0.01,
                 replay_size=10000):
        """DDPG agent.

        Args:
            name (str): Model name.
            source_actor (photinia.Widget): Source actor object.
            target_actor (photinia.Widget): Target actor object.
            source_critic (photinia.Widget): Source critic object.
            target_critic (photinia.Widget): Target critic object.
            source_state_placeholder: Placeholder of source state.
            target_state_placeholder: Placeholder of target state.
            reward_placeholder: Placeholder of reward.
            gamma (float): Discount factor of reward.
            tao (float):
            replay_size (int): Size of replay memory.

        """
        self._source_actor = source_actor
        self._target_actor = target_actor
        self._source_critic = source_critic
        self._target_critic = target_critic
        self._source_state_placeholder = source_state_placeholder
        self._target_state_placeholder = target_state_placeholder
        self._reward_placeholder = reward_placeholder
        self._gamma = gamma
        self._tao = tao
        self._replay_size = replay_size
        self._replay = common.ReplayMemory(replay_size)
        super(DDPGAgent, self).__init__(name)

    def _build(self):
        source_actor = self._source_actor
        target_actor = self._target_actor
        source_critic = self._source_critic
        target_critic = self._target_critic

        source_state = self._source_state_placeholder
        target_state = self._target_state_placeholder
        reward = self._reward_placeholder

        source_action = source_actor.setup(source_state)
        self._add_slot(
            '_predict',
            inputs=source_state,
            outputs=source_action
        )

        target_action = target_actor.setup(target_state)
        target_reward = target_critic.setup(target_state, target_action)
        y = reward + self._gamma * target_reward
        source_reward = source_critic.setup(source_state, source_action)
        loss = tf.reduce_mean(tf.square(y - source_reward))
        var_list = source_critic.get_trainable_variables()
        reg = ph.reg.Regularizer().add_l1_l2(var_list)
        self._add_slot(
            '_update_q_source',
            inputs=(source_state, source_action, reward, target_state),
            outputs=loss,
            updates=tf.train.RMSPropOptimizer(1e-4, 0.9, 0.9).minimize(
                loss + reg.get_loss(1e-5),
                var_list=var_list
            )
        )

        var_list = source_actor.get_trainable_variables()
        loss = tf.reduce_mean(source_reward)
        reg = ph.reg.Regularizer().add_l1_l2(var_list)
        self._add_slot(
            '_update_a_source',
            inputs=source_state,
            updates=tf.train.RMSPropOptimizer(1e-4, 0.9, 0.9).minimize(
                -loss + reg.get_loss(1e-5),
                var_list=var_list
            )
        )

        self._add_slot(
            '_update_q_target',
            updates=tf.group(*[
                tf.assign(v_target, self._tao * v_source + (1.0 - self._tao) * v_target)
                for v_source, v_target in zip(
                    source_critic.get_trainable_variables(),
                    target_critic.get_trainable_variables()
                )
            ])
        )

        self._add_slot(
            '_update_a_target',
            updates=tf.group(*[
                tf.assign(v_target, self._tao * v_source + (1.0 - self._tao) * v_target)
                for v_source, v_target in zip(
                    source_actor.get_trainable_variables(),
                    target_actor.get_trainable_variables()
                )
            ])
        )

        self._add_slot(
            '_init_a_target',
            updates=tf.group(*[
                tf.assign(v_target, v_source)
                for v_source, v_target in zip(
                    source_critic.get_trainable_variables(),
                    target_critic.get_trainable_variables()
                )
            ])
        )
        self._add_slot(
            '_init_q_target',
            updates=tf.group(*[
                tf.assign(v_target, v_source)
                for v_source, v_target in zip(
                    source_actor.get_trainable_variables(),
                    target_actor.get_trainable_variables()
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
