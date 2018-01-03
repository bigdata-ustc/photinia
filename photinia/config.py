#!/usr/bin/env python3

import tensorflow as tf

D_TYPE = tf.float32

NAME_TRAIN_SLOT = 'train'
NAME_VALID_SLOT = 'validate'
NAME_PREDICT_SLOT = 'predict'

CONTEXT_TRAINER = 'trainable'
CONTEXT_LOOP = 'loop'
CONTEXT_MAX_LOOP = 'max_loop'
CONTEXT_TRAIN = NAME_TRAIN_SLOT
CONTEXT_VALID = NAME_VALID_SLOT
CONTEXT_PREDICT = NAME_PREDICT_SLOT
