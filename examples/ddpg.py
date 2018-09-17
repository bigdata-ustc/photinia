#!/usr/bin/env python3

"""
@author: xi
@since: 2018-07-03
"""

import argparse
import os

import gym

import photinia as ph
from photinia import deep_rl
from photinia.deep_rl import actors_critics
from photinia.deep_rl import ddpg
import tensorflow as tf


class Agent(ddpg.DDPGAgent):

    def __init__(self, name, state_size, action_size, hidden_size=64):
        super(Agent, self).__init__(
            name,
            ph.placeholder('source_state', (None, state_size)),
            ph.placeholder('target_state', (None, state_size)),
            ph.placeholder('reward', (None,)),
            actors_critics.DeepResActor('source_actor', state_size, action_size, hidden_size, 6),
            actors_critics.DeepResActor('target_actor', state_size, action_size, hidden_size, 6),
            actors_critics.DeepResCritic('source_critic', state_size, action_size, hidden_size, 6),
            actors_critics.DeepResCritic('target_critic', state_size, action_size, hidden_size, 6),
            gamma=0.9,
            replay_size=1000,
            optimizer=tf.train.RMSPropOptimizer(1e-4, 0.9, 0.9)
        )


def main(args):
    model = Agent('agent', 3, 1)
    ph.initialize_global_variables()
    model.init()

    render = False
    env = gym.make('Pendulum-v0')
    env_noise = deep_rl.NormalNoise(1.0)
    for i in range(args.num_loops):
        total_r = 0
        s = env.reset()
        for t in range(1000):
            if render:
                env.render()

            a, = model.predict([s])[0]
            a = env_noise.add_noise(a)
            a *= env.action_space.high
            env_noise.discount(0.999)

            s_, r, done, info = env.step(a)

            model.feedback(s, a, r, s_)
            model.train(args.batch_size * 4)

            total_r += r
            s = s_
            if done:
                print('[%d] %f' % (i, total_r))
                if total_r > -300 and i >= 100:
                    render = True
                break
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    _parser.add_argument('-b', '--batch-size', type=int, default=32)
    _parser.add_argument('-n', '--num-loops', type=int, default=200)
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(main(_args))
