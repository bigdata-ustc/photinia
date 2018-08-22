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
        actor_source = self._source_actor
        actor_target = self._target_actor
        critic_source = self._source_critic
        critic_target = self._target_critic

        state_source = self._source_state_placeholder
        action_source = actor_source.setup(state_source)
        reward_source = critic_source.setup(state_source, action_source)

        state_target = self._target_state_placeholder
        action_target = actor_target.setup(state_target)
        reward_target = critic_target.setup(state_target, action_target)

        self._add_slot(
            '_predict',
            inputs=state_source,
            outputs=action_source
        )

        reward = self._reward_placeholder
        y = reward + self._gamma * reward_target
        critic_loss = tf.reduce_mean(tf.square(y - reward_source))
        var_list = critic_source.get_trainable_variables()
        reg = ph.reg.Regularizer().add_l1_l2(var_list)
        self._add_slot(
            '_update_q_source',
            inputs=(state_source, action_source, reward, state_target),
            outputs=critic_loss,
            updates=tf.train.RMSPropOptimizer(1e-4, 0.9, 0.9).minimize(
                critic_loss + reg.get_loss(1e-5),
                var_list=var_list
            )
        )

        var_list = actor_source.get_trainable_variables()
        actor_loss = -tf.reduce_mean(reward_source)
        reg = ph.reg.Regularizer().add_l1_l2(var_list)
        self._add_slot(
            '_update_a_source',
            inputs=state_source,
            updates=tf.train.RMSPropOptimizer(1e-4, 0.9, 0.9).minimize(
                actor_loss + reg.get_loss(1e-5),
                var_list=var_list
            )
        )

        self._add_slot(
            '_update_q_target',
            updates=tf.group(*[
                tf.assign(v_target, self._tao * v_source + (1.0 - self._tao) * v_target)
                for v_source, v_target in zip(
                    critic_source.get_trainable_variables(),
                    critic_target.get_trainable_variables()
                )
            ])
        )

        self._add_slot(
            '_update_a_target',
            updates=tf.group(*[
                tf.assign(v_target, self._tao * v_source + (1.0 - self._tao) * v_target)
                for v_source, v_target in zip(
                    actor_source.get_trainable_variables(),
                    actor_target.get_trainable_variables()
                )
            ])
        )

        self._add_slot(
            '_init_a_target',
            updates=tf.group(*[
                tf.assign(v_target, v_source)
                for v_source, v_target in zip(
                    critic_source.get_trainable_variables(),
                    critic_target.get_trainable_variables()
                )
            ])
        )
        self._add_slot(
            '_init_q_target',
            updates=tf.group(*[
                tf.assign(v_target, v_source)
                for v_source, v_target in zip(
                    actor_source.get_trainable_variables(),
                    actor_target.get_trainable_variables()
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
