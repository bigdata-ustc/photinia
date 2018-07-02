#!/usr/bin/env python3

"""
@author: xi
@since: 2018-07-01
"""

import tensorflow as tf


class __GlobalContext(object):

    def __init__(self):
        self._session_config = tf.ConfigProto()
        self._session_config.gpu_options.allow_growth = True
        self._session = None

    # def __del__(self):
    #     if self._session is not None:
    #         self._session.close()

    @property
    def session_config(self):
        return self._session_config

    @property
    def session(self):
        if self._session is None:
            self._session = tf.Session(config=self._session_config)
        return self._session


__GLOBAL = __GlobalContext()


def get_session_config():
    return __GLOBAL.session_config


def get_session():
    return __GLOBAL.session


def initialize_global_variables():
    __GLOBAL.session.run(tf.global_variables_initializer())
