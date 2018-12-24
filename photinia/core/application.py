#!/usr/bin/env python3

"""
@author: xi
@since: 2018-12-23
"""

import threading
import code
import tensorflow as tf


class Application(object):

    def __init__(self, args):
        self._args = args
        self._app_thread = threading.Thread(target=self._main)
        self._app_thread.setDaemon(True)
        self._ret_code = -1

        self._interrupt_lock = threading.Semaphore(1)
        self._interrupt = False

        self._wait_for_interrupt = threading.Semaphore(0)
        self._wait_for_continue = threading.Semaphore(0)

    def _main(self):
        self._ret_code = self.main()

    def main(self):
        raise NotImplementedError()

    def checkpoint(self):
        with self._interrupt_lock:
            if not self._interrupt:
                return
        self._shell()
        with self._interrupt_lock:
            self._interrupt = False

    def run(self):
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
        code.interact(
            banner='Welcome to the shell!',
            local={
                'help': self.help,
                'vars': self.vars
            },
            exitmsg='\nYour application will continue to run.\n'
        )

    def vars(self):
        print('Variables:')
        vars = tf.global_variables()
        for var_ in vars:
            name = var_.name
            print(f'[{name}]')

    def help(self):
        print('This is help page.')
        print('This is help page.')
        print('This is help page.')
