#!/usr/bin/env python3

"""
@author: xi
@since: 2018-07-01
"""

import os

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

TF_LOG_ALL = '0'
TF_LOG_NO_INFO = '1'
TF_LOG_NO_WARN = '2'
TF_LOG_NONE = '3'


def get_tf_log_level():
    return os.environ['TF_CPP_MIN_LOG_LEVEL']


def set_tf_log_level(level):
    if level not in ('0', '1', '2', '3'):
        raise ValueError(
            'level should be one of {'
            'TF_LOG_ALL, '
            'TF_LOG_NO_INFO, '
            'TF_LOG_NO_WARN, '
            'TF_LOG_NONE}.'
        )
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = level


def get_session_config():
    return __GLOBAL.session_config


def get_session():
    return __GLOBAL.session


def initialize_global_variables():
    __GLOBAL.session.run(tf.global_variables_initializer())
