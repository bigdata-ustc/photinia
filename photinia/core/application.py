#!/usr/bin/env python3

"""
@author: xi
@since: 2018-12-23
"""

import code
import collections
import sys
import threading

import numpy as np
import prettytable
import tensorflow as tf

from . import context


def shell(fn):
    fn.__name__ = f'shell.{fn.__name__}'
    return fn


class Application(object):

    def __init__(self):
        self._app_thread = None
        self._ret_code = -1

        self._local_dict = {}
        self._interrupt_lock = threading.Semaphore(1)
        self._interrupt = False

        self._var_list = []
        self._var_dict = {}

    def _main(self, args):
        self._ret_code = self.main(args)

    def main(self, args):
        raise NotImplementedError()

    def checkpoint(self):
        with self._interrupt_lock:
            if not self._interrupt:
                return
        self._shell()
        with self._interrupt_lock:
            self._interrupt = False

    def run(self, args):
        self._app_thread = threading.Thread(target=self._main, args=(args,))
        self._app_thread.setDaemon(True)
        self._app_thread.start()
        while True:
            try:
                self._app_thread.join()
                break
            except KeyboardInterrupt:
                with self._interrupt_lock:
                    if self._interrupt:
                        break
                    self._interrupt = True
                    print('\nWaiting for the checkpoint...')
        return self._ret_code

    def _shell(self):
        local_dict = self._local_dict = collections.OrderedDict((
            (name, attr)
            for name, attr in ((name, getattr(self, name)) for name in dir(self))
            if hasattr(attr, '__name__') and attr.__name__.find('shell.') != -1
        ))
        code.interact(
            banner='Welcome to the shell!',
            local=local_dict,
            exitmsg='\nYour application will continue to run.\n'
        )

    @shell
    def vars(self, prefix=''):
        """List variables."""
        var_dict = self._var_dict = {
            var_.name: var_
            for var_ in tf.global_variables()
        }
        vars_list = self._var_list = [
            var_
            for name, var_ in var_dict.items()
            if name.startswith(prefix)
        ]
        table = prettytable.PrettyTable([
            '#',
            'Name',
            'Shape',
            'dtype',
            'Trainable'
        ])
        for i, var_ in enumerate(vars_list, 1):
            table.add_row((
                i,
                var_.name,
                var_.shape,
                var_.dtype.name,
                var_.trainable
            ))
        print(table)
        print()

    @shell
    def tvars(self, prefix=''):
        """List trainable variables."""
        self._var_dict = {
            var_.name: var_
            for var_ in tf.global_variables()
        }
        vars_list = self._var_list = [
            var_
            for var_ in tf.trainable_variables()
            if var_.name.startswith(prefix)
        ]
        table = prettytable.PrettyTable([
            '#',
            'Name',
            'Shape',
            'dtype'
        ])
        for i, var_ in enumerate(vars_list, 1):
            table.add_row((
                i,
                var_.name,
                var_.shape,
                var_.dtype.name
            ))
        print(table)
        print()

    @shell
    def echo(self, var_id):
        """Show value of the specific variable."""
        if isinstance(var_id, int):
            try:
                var_ = self._var_list[var_id - 1]
            except IndexError:
                print('No such variable.', file=sys.stderr)
                return
        elif isinstance(var_id, str):
            try:
                var_ = self._var_dict[var_id]
            except KeyError:
                print('No such variable.', file=sys.stderr)
                return
        else:
            print(f'Invalid var_id={var_id}', file=sys.stderr)
            return
        shape = var_.shape
        value = context.get_session().run(var_)
        if len(shape) == 0:
            print(f'{var_.name} = {value}')
        elif len(shape) == 2 and shape[0] <= 50 and shape[1] <= 50:
            table = prettytable.PrettyTable(
                header=False,
                hrules=prettytable.ALL,
                vrules=prettytable.ALL
            )
            for row in value:
                table.add_row(
                    tuple(f'{cell:.04f}' for cell in row)
                )
            print(f'{var_.name} =')
            print(table)
        else:
            value = np.linalg.norm(value, 1)
            print(f'|{var_.name}| = {value}')
        print()

    @shell
    def help(self):
        """Show this help."""
        print('When caught in a desperation, the only person who can save you is yourself.')
        table = prettytable.PrettyTable(['Method', 'Description'])
        table.align['Description'] = 'l'
        for name, obj in self._local_dict.items():
            if name.startswith('_'):
                continue
            doc = obj.__doc__
            if doc is None:
                continue
            table.add_row((name, doc))
        print(table)
        print()
