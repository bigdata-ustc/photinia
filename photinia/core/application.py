#!/usr/bin/env python3

"""
@author: xi
@since: 2018-12-23
"""

import threading


class Application(object):

    def __init__(self, args):
        self._args = args
        self._app_thread = threading.Thread(target=self._main)
        self._app_thread.setDaemon(True)
        self._ret_code = -1

    def _main(self):
        self._ret_code = self.main()

    def main(self):
        raise NotImplementedError()

    def run(self):
        self._app_thread.start()
        while True:
            try:
                self._app_thread.join()
            except KeyboardInterrupt:
                if self._shell():
                    break
        return self._ret_code

    def _shell(self):
        if input('Exit?').lower() == 'y':
            return True
