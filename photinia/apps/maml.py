#!/usr/bin/env python3

"""
@author: xi
@since: 2018-09-20
"""

import tensorflow as tf

import photinia as ph


class MAML(ph.Model):

    def __init__(self,
                 name,
                 input_x,
                 input_y,
                 input_x_prime,
                 input_y_prime,
                 net_fn,
                 loss_fn,
                 train_num_sgd=1,
                 predict_num_sgd=3,
                 alpha=1e-2,
                 optimizer=tf.train.RMSPropOptimizer(1e-4, 0.9, 0.9)):
        self._input_x = input_x
        self._input_y = input_y
        self._input_x_prime = input_x_prime
        self._input_y_prime = input_y_prime
        self._net_fn = net_fn
        self._loss_fn = loss_fn
        self._train_num_sgd = train_num_sgd
        self._predict_num_sgd = predict_num_sgd
        self._alpha = alpha
        self._optimizer = optimizer
        super(MAML, self).__init__(name)

    def _build(self):
        self._net = self._net_fn('net_0')  # type: ph.Widget
        y_primes, losses = tf.map_fn(
            fn=self._per_task,
            elems=(self._input_x, self._input_y, self._input_x_prime, self._input_y_prime),
            dtype=(ph.dtype, ph.dtype)
        )

        self._step_predict = ph.Step(
            inputs=(self._input_x, self._input_y, self._input_x_prime),
            outputs=y_primes
        )

        loss = tf.reduce_mean(losses)
        self._step_train = ph.Step(
            inputs=(self._input_x, self._input_y, self._input_x_prime, self._input_y_prime),
            outputs=loss,
            updates=self._optimizer.minimize(loss)
        )

    def _per_task(self, elem):
        x, y, x_prime, y_prime = elem

        var_dict = {
            var_.name: var_
            for var_ in self._net.get_trainable_variables()
        }

        full_name = self._net.full_name
        prefix = full_name[:full_name.rfind('_')]
        for i in range(1, max(self._train_num_sgd, self._predict_num_sgd) + 1):
            reuse_dict = ph.ReuseContext(var_dict, alias={self._net.full_name: f'{prefix}_{i}'})
            with reuse_dict:
                net = self._net_fn(f'net_{i}')  # type: ph.Widget
            y_ = net.setup(x)
            task_loss = tf.reduce_mean(self._loss_fn(y, y_))
            var_dict = {
                name: var_ - self._alpha * tf.gradients(task_loss, [var_])[0]
                for name, var_ in var_dict.items()
            }
            if i == self._train_num_sgd or i == self._predict_num_sgd:
                reuse_dict = ph.ReuseContext(var_dict, alias={self._net.full_name: f'{prefix}_{i}_prime'})
                with reuse_dict:
                    net_prime = self._net_fn(f'net_{i}_prime')  # type: ph.Widget
                tmp_y_prime_ = net_prime.setup(x_prime)
                if i == self._train_num_sgd:
                    loss = tf.reduce_mean(self._loss_fn(y_prime, tmp_y_prime_))
                if i == self._predict_num_sgd:
                    y_prime_ = tmp_y_prime_

        # net = self._net
        # y_ = net.setup(x)
        # task_loss = tf.reduce_mean(self._loss_fn(y, y_))
        #
        # var_dict = {
        #     var_.name: var_ - self._alpha * tf.gradients(task_loss, [var_])[0]
        #     for var_ in net.get_trainable_variables()
        # }
        # reuse_dict = ph.ReuseContext(var_dict, alias={net.full_name: f'{net.full_name}_prime'})
        # with reuse_dict:
        #     net_prime = self._net_fn('net_prime')  # type: ph.Widget
        # y_prime_ = net_prime.setup(x_prime)
        # loss = tf.reduce_mean(self._loss_fn(y_prime, y_prime_))

        return y_prime_, loss

    def train(self, x, y, x_prime, y_prime):
        return self._step_train(x, y, x_prime, y_prime)

    def predict(self, x, y, x_prime):
        return self._step_predict(x, y, x_prime)
