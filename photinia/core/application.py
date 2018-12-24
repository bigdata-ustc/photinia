#!/usr/bin/env python3

"""
@author: xi
@since: 2018-12-23
"""

import threading
import code


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
            interrupt = self._interrupt
        if interrupt:
            self._wait_for_interrupt.release()
            self._wait_for_continue.acquire()

    def run(self):
        self._app_thread.start()
        while True:
            try:
                self._app_thread.join()
            except KeyboardInterrupt:
                with self._interrupt_lock:
                    self._interrupt = True
                print('\nWaiting for the checkpoint to interrupt...')
                self._wait_for_interrupt.acquire()

                if self._shell():
                    break

                with self._interrupt_lock:
                    self._interrupt = False
                self._wait_for_continue.release()
        return self._ret_code

    def _shell(self):
        code.interact(
            banner='\nYour application is interrupted.',
            local={
                'help': self.help
            },
            exitmsg='\nYour application will continue to run.'
        )

    def help(self):
        print('This is help page.')
        print('This is help page.')
        print('This is help page.')
