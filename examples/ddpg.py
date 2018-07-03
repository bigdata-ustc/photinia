#!/usr/bin/env python3

"""
@author: xi
@since: 2018-07-03
"""

import argparse
import os

import gym
import numpy as np

import photinia as ph
from photinia.deep_rl import ddpg


def main(args):
    model = ddpg.Agent('agent', 3, 1)
    ph.initialize_global_variables()
    model.init()

    render = False
    var = 3.0
    env = gym.make('Pendulum-v0')
    for i in range(args.num_loops):
        total_r = 0
        s = env.reset()
        for t in range(1000):
            if render:
                env.render()

            a = model.predict(s) * env.action_space.high
            a = np.clip(np.random.normal(a, var), -2, 2)

            s_, r, done, info = env.step(a)

            model.feedback(s, a, r, s_)
            model.train(args.batch_size)

            total_r += r
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
    _parser.add_argument('-g', '--gpu', default='0', help='Choose which GPU to use.')
    _parser.add_argument('-b', '--batch-size', type=int, default=32)
    _parser.add_argument('-n', '--num-loops', type=int, default=150)
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(main(_args))
