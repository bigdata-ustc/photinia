#!/usr/bin/env python3

"""
@author: xi
@since: 2018-06-09
"""

import argparse
import collections
import random

import gym
import numpy as np
import tensorflow as tf

import photinia as ph


class Actor(ph.Widget):

    def __init__(self, name, state_size, action_size, hidden_size=100):
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
            self._hidden_size, self._action_size,
            w_init=ph.init.TruncatedNormal(0.0, 1e-2)
        )

    def _setup(self, state):
        return ph.setup(
            state, [
                self._layer1, ph.ops.lrelu,
                self._layer2,
                tf.nn.tanh
            ]
        ) * 2.0


class Critic(ph.Widget):

    def __init__(self, name, state_size, action_size, hidden_size=100):
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
            self._hidden_size, 1,
            w_init=ph.init.TruncatedNormal(0.0, 1e-2)
        )

    def _setup(self, state, action):
        return ph.setup(
            tf.concat((state, action), axis=1), [
                self._layer1, ph.ops.lrelu,
                self._layer2,
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
                 tao=0.01):
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
            'predict_a',
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
            'update_q_source',
            inputs=(s_in, a_source_pred, r_in, s_in_),
            outputs=loss,
            updates=tf.train.AdamOptimizer(1e-3).minimize(
                loss + reg.get_loss(1e-7),
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
        loss = tf.reduce_mean(q_source_pred)
        reg = ph.reg.Regularizer().add_l1_l2(var_list)
        self._add_slot(
            'update_a_source',
            inputs=s_in,
            # updates=tf.train.AdamOptimizer(-1e-3).apply_gradients(
            #     zip(grad_policy, var_list)
            # ),
            updates=tf.train.AdamOptimizer(1e-3).minimize(
                -loss + reg.get_loss(1e-7),
                var_list=var_list
            )
        )

        self._add_slot(
            'update_q_target',
            updates=tf.group(*[
                tf.assign(v_target, self._tao * v_source + (1.0 - self._tao) * v_target)
                for v_source, v_target in zip(
                    q_source.get_trainable_variables(),
                    q_target.get_trainable_variables()
                )
            ])
        )

        self._add_slot(
            'update_a_target',
            updates=tf.group(*[
                tf.assign(v_target, self._tao * v_source + (1.0 - self._tao) * v_target)
                for v_source, v_target in zip(
                    a_source.get_trainable_variables(),
                    a_target.get_trainable_variables()
                )
            ])
        )

        init_q = tf.group(*[
            tf.assign(v_target, v_source)
            for v_source, v_target in zip(
                q_source.get_trainable_variables(),
                q_target.get_trainable_variables()
            )
        ])
        init_a = tf.group(*[
            tf.assign(v_target, v_source)
            for v_source, v_target in zip(
                a_source.get_trainable_variables(),
                a_target.get_trainable_variables()
            )
        ])
        self._add_slot(
            'init',
            updates=(init_q, init_a)
        )

    def train(self, s, a, r, s_):
        self.update_q_source(s, a, r, s_)
        self.update_a_source(s)

        self.update_q_target()
        self.update_a_target()


class ReplayMemory(object):

    def __init__(self, buffer_size):
        """Replay memory.

        Args:
            buffer_size (int): Max buffer size.

        """
        self._buffer_size = buffer_size
        self._buffer = collections.deque()

    def full(self):
        return len(self._buffer) >= self._buffer_size

    def put(self, s, a, r, s_, done):
        """Put a transition tuple to the replay memory.

        Args:
            s (numpy.ndarray): State s_t.
            a ((numpy.ndarray)): Action a_t.
            r (float): Reward r_{t + 1}.
            s_ (numpy.ndarray): Transition state s_{t + 1}.
            done (bool): Is terminal?

        """
        self._buffer.append((s, a, r, s_, done))
        if len(self._buffer) > self._buffer_size:
            self._buffer.popleft()

    def get(self, batch_size):
        """Get a random batch of transitions from the memory.

        Args:
            batch_size (int): Batch size.

        Returns:
            list[tuple]: List of transition tuples.

        """
        columns = (list(), list(), list(), list(), list())
        rows = random.sample(list(self._buffer), batch_size) if batch_size <= len(self._buffer) else self._buffer
        for row in rows:
            # for col in row:
            #     print(col)
            for i in range(5):
                columns[i].append(row[i])
        return columns


def main(args):
    model = Agent('agent', 3, 1)
    ph.initialize_global_variables()
    model.init()
    replay = ReplayMemory(10000)

    render = False
    var = 3.0
    env = gym.make('Pendulum-v0')
    for i in range(150):
        total_r = 0
        s = env.reset()
        for t in range(1000):
            if render:
                env.render()
            # time.sleep(0.1)
            a = model.predict_a((s,))[0][0]
            a = np.clip(np.random.normal(a, var), -2, 2)
            # print(a)
            s_, r, done, info = env.step(a)
            total_r += r
            # print(s, a, r, s_, done)
            replay.put(s, a, r, s_, done)

            # if replay.full():
            batch = replay.get(32)[:-1]
            model.train(*batch)
            var *= .9995

            s = s_
            if done:
                print('[%d] %f' % (i, total_r))
                if total_r > -300:
                    render = True
                break
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    #
    # TODO: Define more args here.
    _args = _parser.parse_args()
    exit(main(_args))
